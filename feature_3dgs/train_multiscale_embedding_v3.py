"""
Multi-Scale Feature Embedding Training v3 (Channel-Gated)
==========================================================
在 v2 基础上增加 ChannelGate：可学习的通道门控，自动识别并抑制噪声通道。

核心改动:
  1. 在 render_per_scale 后对渲染特征施加 ChannelGate
  2. Loss = reconstruction_loss + λ_sparse * sparsity_loss
  3. 保存 gate 统计信息用于后续分析

用法:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -m feature_3dgs.train_multiscale_embedding_v3 \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --feature_dir output/features_multiscale_compressed/OldHospital_indexed \
        --colmap_dir dataset/OldHospital/sparse/0 \
        --output_dir output/feature_3dgs/oldhospital_v7_gated \
        --num_iters 15000 --grad_accum 4 --precache_gpu \
        --sparsity_weight 0.01 --vis_interval 2000 --vis_frames 0,100,400
"""
import os
import sys
import argparse
import time
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_3dgs.multiscale_gaussian_model import MultiScaleGaussianModel
from feature_3dgs.feature_renderer import FeatureRenderer
from feature_3dgs.multiscale_dataset import MultiScaleFeatureDataset, ColmapMultiScaleFeatureDataset
from feature_3dgs.channel_gate import ChannelGate

# Import from v2
from feature_3dgs.train_multiscale_embedding_v2 import (
    l1_loss, cosine_loss, visualize_features_pca,
    GPUCachedDataset, build_knn_indices, knn_smooth_loss,
)


def render_per_scale(model, pose, scale_intrinsics):
    """Per-scale rendering (same as v2)."""
    S = MultiScaleGaussianModel
    raw_feat = model._loc_feature

    fi = scale_intrinsics['fine']
    fine_colors = F.normalize(raw_feat[:, S.FINE_SD_START:S.FINE_END], p=2, dim=-1)
    r_fine = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=fi['fx'], fy=fi['fy'], cx=fi['cx'], cy=fi['cy'],
        img_height=fi['H'], img_width=fi['W'],
        norm_feat_before_render=False, norm_feat_after_render=False,
        colors_override=fine_colors,
    )
    rendered_fine = F.normalize(r_fine['feature_map'], p=2, dim=0)

    mi = scale_intrinsics['mid']
    mid_colors = F.normalize(raw_feat[:, S.MID_START:S.MID_END], p=2, dim=-1)
    r_mid = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=mi['fx'], fy=mi['fy'], cx=mi['cx'], cy=mi['cy'],
        img_height=mi['H'], img_width=mi['W'],
        norm_feat_before_render=False, norm_feat_after_render=False,
        colors_override=mid_colors,
    )
    rendered_mid = F.normalize(r_mid['feature_map'], p=2, dim=0)

    ci = scale_intrinsics['coarse']
    coarse_colors = F.normalize(raw_feat[:, S.COARSE_START:S.COARSE_END], p=2, dim=-1)
    r_coarse = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=ci['fx'], fy=ci['fy'], cx=ci['cx'], cy=ci['cy'],
        img_height=ci['H'], img_width=ci['W'],
        norm_feat_before_render=False, norm_feat_after_render=False,
        colors_override=coarse_colors,
    )
    rendered_coarse = F.normalize(r_coarse['feature_map'], p=2, dim=0)

    return {'fine': rendered_fine, 'mid': rendered_mid, 'coarse': rendered_coarse}


def compute_gated_loss(rendered, gt_fine, gt_mid, gt_coarse, gate, args):
    """Compute loss with channel gating applied to rendered features only.
    
    Gates are applied only to rendered features (not GT), so that closing
    a gate increases reconstruction error for that channel. This prevents
    the degenerate solution where gates simply zero everything out.
    """
    S = MultiScaleGaussianModel
    
    # Split fine into fine_sd + fine_dino for gate
    rendered_split = {
        'fine_sd': rendered['fine'][:S.FINE_SD_DIM],
        'fine_dino': rendered['fine'][S.FINE_SD_DIM:S.FINE_END],
        'mid': rendered['mid'],
        'coarse': rendered['coarse'],
    }
    
    # Apply gate to rendered only
    gated_rendered = gate(rendered_split)
    
    # Recombine fine
    gated_r_fine = torch.cat([gated_rendered['fine_sd'], gated_rendered['fine_dino']], dim=0)

    loss_fine = l1_loss(gated_r_fine, gt_fine) + args.cos_weight * cosine_loss(gated_r_fine, gt_fine)
    loss_mid = l1_loss(gated_rendered['mid'], gt_mid) + args.cos_weight * cosine_loss(gated_rendered['mid'], gt_mid)
    loss_coarse = l1_loss(gated_rendered['coarse'], gt_coarse) + args.cos_weight * cosine_loss(gated_rendered['coarse'], gt_coarse)

    total = args.w_fine * loss_fine + args.w_mid * loss_mid + args.w_coarse * loss_coarse
    return total, loss_fine.item(), loss_mid.item(), loss_coarse.item()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # === 1. Load 3DGS model ===
    print("\n=== 加载 3DGS 模型 ===")
    model = MultiScaleGaussianModel()
    model.load_ply(args.ply_path)
    model = model.to(device)
    print(f"  Gaussians: {model.num_gaussians:,}  特征维度: {model.TOTAL_DIM}")

    # === 2. Channel Gate ===
    S = MultiScaleGaussianModel
    gate = ChannelGate(
        scale_dims={
            'fine_sd': S.FINE_SD_DIM,
            'fine_dino': S.FINE_DINO_DIM,
            'mid': S.MID_DIM,
            'coarse': S.COARSE_DIM,
        },
        init_bias=args.gate_init_bias,
    ).to(device)
    print(f"  ChannelGate: {sum(p.numel() for p in gate.parameters())} params")

    # === 3. Resume ===
    start_iter = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model._loc_feature.data.copy_(ckpt['loc_feature'])
            if 'gate_state_dict' in ckpt:
                gate.load_state_dict(ckpt['gate_state_dict'])
            start_iter = ckpt.get('iteration', 0) + 1
            print(f"  从 checkpoint 恢复: iter={start_iter-1}")
    elif args.warmstart:
        # Load embedding from existing model, start gate from scratch
        ws_path = Path(args.warmstart)
        if ws_path.exists():
            ckpt = torch.load(ws_path, map_location=device)
            model._loc_feature.data.copy_(ckpt['loc_feature'])
            print(f"  Warmstart embedding from: {ws_path}")

    # === 4. Dataset ===
    print("\n=== 加载数据集 ===")
    intrinsics = {'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy}

    if args.colmap_dir is not None:
        dataset = ColmapMultiScaleFeatureDataset(
            feature_dir=args.feature_dir,
            colmap_dir=args.colmap_dir,
            intrinsics=intrinsics if (args.fx != 320.0 or args.fy != 320.0) else None,
            normalize_features=True,
            max_frames=args.max_frames,
        )
        intrinsics = dataset.intrinsics
    elif args.traj_path is not None:
        dataset = MultiScaleFeatureDataset(
            feature_dir=args.feature_dir,
            traj_path=args.traj_path,
            intrinsics=intrinsics,
            img_size=(args.img_height, args.img_width),
            normalize_features=True,
            max_frames=args.max_frames,
        )
    else:
        raise ValueError("必须提供 --traj_path 或 --colmap_dir 之一")

    fine_H, fine_W = dataset.fine_hw
    mid_H, mid_W = dataset.mid_hw
    coarse_H, coarse_W = dataset.coarse_hw

    if args.precache_gpu:
        cached = GPUCachedDataset(dataset, device)
    else:
        cached = None

    # === 5. Intrinsics per scale ===
    _fx = intrinsics['fx']
    _fy = intrinsics['fy']
    _cx = intrinsics['cx']
    _cy = intrinsics['cy']
    if args.colmap_dir and hasattr(dataset, 'img_w'):
        _ref_w = dataset.img_w
        _ref_h = dataset.img_h
    else:
        _ref_w = args.img_width
        _ref_h = args.img_height

    def _scale_intrinsics(H, W):
        return {
            'fx': _fx * W / _ref_w, 'fy': _fy * H / _ref_h,
            'cx': _cx * W / _ref_w, 'cy': _cy * H / _ref_h,
            'H': H, 'W': W,
        }
    scale_intrinsics = {
        'fine': _scale_intrinsics(fine_H, fine_W),
        'mid': _scale_intrinsics(mid_H, mid_W),
        'coarse': _scale_intrinsics(coarse_H, coarse_W),
    }
    fi = scale_intrinsics['fine']
    print(f"  Fine 内参: fx={fi['fx']:.2f}, fy={fi['fy']:.2f}")
    print(f"  分辨率: fine={fine_W}×{fine_H}, mid={mid_W}×{mid_H}, coarse={coarse_W}×{coarse_H}")

    # === 6. KNN (optional) ===
    knn_indices = None
    if args.knn_smooth_weight > 0:
        knn_indices = build_knn_indices(model.get_xyz, k=args.knn_k)

    # === 7. Optimizer (joint: embedding + gate) ===
    optim_params = [
        {'params': [model._loc_feature], 'lr': args.lr},
        {'params': gate.parameters(), 'lr': args.gate_lr},
    ]
    optimizer = torch.optim.Adam(optim_params, eps=1e-15)
    total_iters = args.num_iters - start_iter + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=args.lr * 0.01)

    # === 8. Output ===
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train_log.txt'
    gate_log = output_dir / 'gate_stats.jsonl'

    vis_frame_ids = [int(x) for x in args.vis_frames.split(',')] if args.vis_frames else []

    # === 9. Training loop ===
    n_frames = len(cached) if cached is not None else len(dataset)
    best_loss = float('inf')
    loss_history = []
    running_loss = 0.0
    running_count = 0
    t_start = time.time()

    print(f"\n=== 开始训练 (v3 Channel-Gated) ===")
    print(f"  迭代: {start_iter}→{args.num_iters}")
    print(f"  Gate sparsity weight: {args.sparsity_weight}")
    print(f"  Gate LR: {args.gate_lr}")

    for iteration in range(start_iter, args.num_iters + 1):
        optimizer.zero_grad()

        iter_loss = 0.0
        iter_fine = 0.0
        iter_mid = 0.0
        iter_coarse = 0.0

        for _ in range(args.grad_accum):
            idx = random.randint(0, n_frames - 1)
            if cached is not None:
                sample = cached[idx]
                gt_fine = sample['fine_feat']
                gt_mid = sample['mid_feat']
                gt_coarse = sample['coarse_feat']
                pose = sample['pose']
            else:
                sample = dataset[idx]
                gt_fine = sample['fine_feat'].to(device)
                gt_mid = sample['mid_feat'].to(device)
                gt_coarse = sample['coarse_feat'].to(device)
                pose = sample['pose'].to(device)

            rendered = render_per_scale(model, pose, scale_intrinsics)
            loss, lf, lm, lc = compute_gated_loss(
                rendered, gt_fine, gt_mid, gt_coarse, gate, args)
            
            # Sparsity loss
            sp_loss = args.sparsity_weight * gate.sparsity_loss()
            total_loss = (loss + sp_loss) / args.grad_accum
            total_loss.backward()

            iter_loss += loss.item() / args.grad_accum
            iter_fine += lf / args.grad_accum
            iter_mid += lm / args.grad_accum
            iter_coarse += lc / args.grad_accum

        # KNN smoothing
        if knn_indices is not None and args.knn_smooth_weight > 0:
            aux = args.knn_smooth_weight * knn_smooth_loss(
                model._loc_feature, knn_indices, num_samples=args.knn_samples)
            aux.backward()

        optimizer.step()
        scheduler.step()

        loss_history.append(iter_loss)
        running_loss += iter_loss
        running_count += 1

        # Log
        if iteration % args.log_interval == 0 or iteration == start_iter:
            elapsed = time.time() - t_start
            avg_loss = running_loss / running_count
            lr = optimizer.param_groups[0]['lr']
            it_per_sec = (iteration - start_iter + 1) / max(elapsed, 1e-6)
            
            gate_stats = gate.get_gate_stats()
            active_str = " ".join(
                f"{k}:{v['active_channels']}/{v['total_channels']}"
                for k, v in gate_stats.items()
            )

            msg = (f"[Iter {iteration:5d}/{args.num_iters}] "
                   f"loss={iter_loss:.6f} (avg={avg_loss:.4f}) "
                   f"(f={iter_fine:.4f} m={iter_mid:.4f} c={iter_coarse:.4f}) "
                   f"gates=[{active_str}] "
                   f"lr={lr:.6f} | {it_per_sec:.1f} it/s")
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

            running_loss = 0.0
            running_count = 0

        # Gate stats log (JSONL)
        if iteration % (args.log_interval * 5) == 0:
            gate_stats = gate.get_gate_stats()
            gate_stats['iteration'] = iteration
            gate_stats['loss'] = iter_loss
            with open(gate_log, 'a') as f:
                f.write(json.dumps(gate_stats) + '\n')

        # Visualization
        if args.vis_interval > 0 and iteration % args.vis_interval == 0 and vis_frame_ids:
            with torch.no_grad():
                for fid in vis_frame_ids:
                    if cached is not None:
                        try:
                            idx = cached.frame_ids.index(fid)
                        except ValueError:
                            continue
                        s = cached[idx]
                    else:
                        try:
                            idx = dataset.frame_ids.index(fid)
                        except ValueError:
                            continue
                        s = dataset[idx]
                        s = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in s.items()}
                    r = render_per_scale(model, s['pose'], scale_intrinsics)
                    gt = {'fine': s['fine_feat'], 'mid': s['mid_feat'], 'coarse': s['coarse_feat']}
                    save_path = vis_dir / f'iter{iteration:05d}_frame{fid}.png'
                    visualize_features_pca(r, gt, str(save_path), f"Iter{iteration} F{fid} ")

        # Save
        if iter_loss < best_loss:
            best_loss = iter_loss
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'gate_state_dict': gate.state_dict(),
                'gate_stats': gate.get_gate_stats(),
                'loss': iter_loss,
                'feature_dim': model.TOTAL_DIM,
            }, output_dir / 'best_model.pth')

        if iteration % args.save_interval == 0:
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'gate_state_dict': gate.state_dict(),
                'gate_stats': gate.get_gate_stats(),
                'loss': iter_loss,
            }, output_dir / f'checkpoint_{iteration}.pth')

    # === Final ===
    elapsed_total = time.time() - t_start
    gate_stats = gate.get_gate_stats()
    print(f"\n=== 训练完成 ===")
    print(f"  最终 loss: {loss_history[-1]:.6f}, 最佳: {best_loss:.6f}")
    print(f"  耗时: {elapsed_total:.1f}s")
    print(f"  Gate 统计:")
    for name, stats in gate_stats.items():
        print(f"    {name}: {stats['active_channels']}/{stats['total_channels']} active "
              f"(mean={stats['mean']:.3f}, min={stats['min']:.3f})")

    # Save per_scale models compatible with renderer
    per_scale_dir = output_dir / 'per_scale'
    per_scale_dir.mkdir(exist_ok=True)
    
    raw = model._loc_feature.data
    for scale_name, (start, end) in [
        ('fine_sd', (S.FINE_SD_START, S.FINE_SD_END)),
        ('fine_dino', (S.FINE_DINO_START, S.FINE_DINO_END)),
        ('mid', (S.MID_START, S.MID_END)),
        ('coarse', (S.COARSE_START, S.COARSE_END)),
    ]:
        torch.save({
            'loc_feature': raw[:, start:end],
            'feature_dim': end - start,
        }, per_scale_dir / f'{scale_name}.pth')
    print(f"  Per-scale models saved to: {per_scale_dir}")

    # Save gate info separately for analysis
    torch.save({
        'gate_state_dict': gate.state_dict(),
        'gate_stats': gate_stats,
    }, output_dir / 'channel_gate.pth')

    torch.save({
        'iteration': args.num_iters,
        'loc_feature': model._loc_feature.data,
        'gate_state_dict': gate.state_dict(),
        'loss': loss_history[-1],
        'loss_history': loss_history,
    }, output_dir / 'final_model.pth')

    print(f"\n结果保存至: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Scale Feature Embedding Training v3 (Channel-Gated)')

    # Data paths
    parser.add_argument('--ply_path', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--traj_path', type=str, default=None)
    parser.add_argument('--colmap_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_3dgs/oldhospital_v7_gated')

    # Camera
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)

    # Training
    parser.add_argument('--num_iters', type=int, default=15000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cos_weight', type=float, default=0.1)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--grad_accum', type=int, default=4)

    # Scale weights
    parser.add_argument('--w_fine', type=float, default=1.0)
    parser.add_argument('--w_mid', type=float, default=0.5)
    parser.add_argument('--w_coarse', type=float, default=0.25)

    # Channel Gate
    parser.add_argument('--sparsity_weight', type=float, default=0.01,
                        help='L1 sparsity on gate activations')
    parser.add_argument('--gate_lr', type=float, default=0.005,
                        help='Learning rate for gate parameters')
    parser.add_argument('--gate_init_bias', type=float, default=2.0,
                        help='Initial gate logit (sigmoid(2.0)≈0.88)')

    # Acceleration
    parser.add_argument('--precache_gpu', action='store_true')

    # Visualization
    parser.add_argument('--vis_interval', type=int, default=2000)
    parser.add_argument('--vis_frames', type=str, default='0,100,400')

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Load embedding weights only (gate starts fresh)')

    # Logging
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=2000)

    # KNN
    parser.add_argument('--knn_smooth_weight', type=float, default=0.0)
    parser.add_argument('--knn_k', type=int, default=8)
    parser.add_argument('--knn_samples', type=int, default=10000)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
