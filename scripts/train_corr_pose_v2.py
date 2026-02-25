#!/usr/bin/env python3
"""
CorrPoseNet 快速训练脚本 — 预渲染 + 双卡流水线

核心思路 (vs train_corr_pose.py):
  原版: 每个sample在训练循环内实时渲染 → 0.89s/sample, 渲染占95%时间
  本版: 每epoch开始时批量预渲染 → 训练时直接读取, batch_size可提高到8+

流程:
  1. 在 render_device (GPU0) 上加载渲染器
  2. 每个 epoch 开始: 
     a. 为810帧生成随机噪声位姿
     b. 批量渲染所有特征图 → 缓存到CPU内存 (约4GB)
  3. 在 train_device (GPU1) 上训练网络:
     a. 用 forward_prerendered (K=1 迭代) 
     b. batch_size=8 → 每epoch ~50个batch, ~30秒
  4. 总计: ~7min渲染 + ~0.5min训练 ≈ 7.5min/epoch (原版12min)
     双卡时: 渲染和上一epoch的验证可重叠

关于 K=1 vs K=3:
  训练时 K=1: 网络学习 "给定当前渲染特征 + query → 预测一步flow修正"
  推理时 K=3: 迭代 render→correct→render→correct→render→correct
  类似 RAFT: 训练12步但推理可用更多步, GRU带记忆自然泛化

Usage:
  # 单卡
  CUDA_VISIBLE_DEVICES=0 python scripts/train_corr_pose_v2.py --epochs 50
  
  # 双卡 (推荐)
  python scripts/train_corr_pose_v2.py --render_device cuda:0 --train_device cuda:1

  # 多迭代训练 (更慢但质量可能更好)
  python scripts/train_corr_pose_v2.py --num_train_iters 3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import gc

from torch.utils.data import DataLoader, TensorDataset
from ic_models.corr_pose_net import CorrPoseNet
from data.dataset_v3 import PoseDatasetV3, collate_v3
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp, se3_log, pose_inverse
from modules.featuremetric import compute_image_jacobian

# ============================================================================
# Constants
# ============================================================================
DEFAULT_PLY = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
DEFAULT_FEATURE_MODEL = 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth'
DEFAULT_FEATURE_DIR = 'output/features_multiscale/room_0'
DEFAULT_TRAJ = 'dataset/room_0/Sequence_1/traj_w_c.txt'
DEFAULT_DEPTH_DIR = 'dataset/room_0/Sequence_1/depth'

INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}
FEAT_DIM = {'fine_dino': 768, 'fine_sd': 640}


# ============================================================================
# Loss Functions (same as train_corr_pose.py)
# ============================================================================

def compute_pose_error(pred_pose, gt_pose):
    R_pred = pred_pose[:, :3, :3]
    R_gt = gt_pose[:, :3, :3]
    t_pred = pred_pose[:, :3, 3]
    t_gt = gt_pose[:, :3, 3]
    R_rel = R_pred @ R_gt.transpose(-1, -2)
    cos_angle = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)
    rot_error_deg = torch.acos(cos_angle) * (180.0 / np.pi)
    trans_error_m = (t_pred - t_gt).norm(dim=-1)
    return rot_error_deg, trans_error_m


def pose_loss_fn(pred_poses, gt_pose, gamma=0.8, trans_weight=10.0):
    K = len(pred_poses) - 1
    total_loss = torch.tensor(0.0, device=gt_pose.device)
    for k in range(K):
        pred = pred_poses[k + 1]
        R_pred = pred[:, :3, :3]
        R_gt = gt_pose[:, :3, :3]
        R_rel = R_pred @ R_gt.transpose(-1, -2)
        cos_angle = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        cos_angle = cos_angle.clamp(-1, 1)
        rot_loss = (1 - cos_angle).mean()
        t_pred = pred[:, :3, 3]
        t_gt = gt_pose[:, :3, 3]
        trans_loss = (t_pred - t_gt).norm(dim=-1).mean()
        weight = gamma ** (K - 1 - k)
        total_loss = total_loss + weight * (rot_loss + trans_weight * trans_loss)
    return total_loss


def linearized_flow_loss_fn(
    pred_flows, pred_confs, poses, pose_gt,
    Ju, Jv, valid, gamma=0.8,
):
    K = len(pred_flows)
    B = pose_gt.shape[0]
    device = pose_gt.device
    total_loss = torch.tensor(0.0, device=device)
    valid_f = valid.float()

    for k in range(K):
        pose_k = poses[k].detach()
        T_rel = pose_gt @ pose_inverse(pose_k)
        xi_gt = se3_log(T_rel)
        gt_flow_u = torch.bmm(Ju, xi_gt.unsqueeze(-1)).squeeze(-1)
        gt_flow_v = torch.bmm(Jv, xi_gt.unsqueeze(-1)).squeeze(-1)
        _, _, pH, pW = pred_flows[k].shape
        gt_flow = torch.stack([
            gt_flow_u.reshape(B, pH, pW),
            gt_flow_v.reshape(B, pH, pW),
        ], dim=1)
        valid_mask = valid_f.reshape(B, 1, pH, pW)
        flow_diff = (pred_flows[k] - gt_flow).abs()
        flow_loss = (flow_diff * valid_mask).sum() / (valid_mask.sum() * 2 + 1e-8)
        conf = pred_confs[k]
        conf_loss = -(conf * valid_mask + 1e-8).log().mean() * 0.01
        weight = gamma ** (K - 1 - k)
        total_loss = total_loss + weight * (flow_loss + conf_loss)
    return total_loss


# ============================================================================
# Prerender
# ============================================================================

@torch.no_grad()
def prerender_epoch(
    dataset, renderer, scale_name, render_device, 
    noise_rot_deg, noise_trans_m, num_iters=1, net=None,
):
    """
    批量预渲染一个epoch的所有特征图.
    
    Args:
        dataset: PoseDatasetV3 (用于获取 query_feats, pose_gt, depth)
        renderer: MultiScaleRenderer (在 render_device 上)
        num_iters: 预渲染迭代数 (1=只渲染初始位姿, >1=需要net做rollout)
        net: 网络 (num_iters>1时用于预测中间位姿)
    
    Returns:
        cache: list of dict, 每个元素包含:
            'query_feats': (D, H, W) CPU tensor
            'rendered_feats_list': list of (D, H, W) CPU tensors (长度=num_iters)
            'initial_pose': (4, 4) CPU tensor
            'pose_gt': (4, 4) CPU tensor
            'depth': (H, W) CPU tensor
    """
    from data.dataset_v3 import perturb_pose
    
    N = len(dataset)
    cache = []
    
    t0 = time.time()
    for i in range(N):
        sample = dataset[i]
        query_feats = sample['query_feats'][scale_name]  # (D, H, W) CPU
        pose_gt = sample['pose_gt']  # (4, 4) CPU
        depth = sample['depth'] if 'depth' in sample else None
        
        # 生成随机噪声位姿 (每epoch每帧不同)
        initial_pose = perturb_pose(pose_gt.clone(), noise_rot_deg, noise_trans_m)
        
        # 渲染第1次迭代的特征
        rendered_list = []
        pose_cur = initial_pose.to(render_device)
        result = renderer.render_scale(scale_name, pose_cur)
        rendered_list.append(result['feature_map'].cpu())
        
        # 多迭代预渲染: 用网络做rollout得到中间位姿
        if num_iters > 1 and net is not None:
            q = query_feats.unsqueeze(0).to(render_device)
            d = depth.unsqueeze(0).to(render_device) if depth is not None else None
            ip = initial_pose.unsqueeze(0).to(render_device)
            
            # 逐迭代: 用已渲染的特征预测位姿 → 渲染下一步
            fmap_q = net.encode(q)
            hidden = torch.tanh(net.context_encoder(q))
            from ic_models.corr_pose_net import local_correlation, diff_pose_solve
            Ju, Jv, valid = compute_image_jacobian(d, INTRINSICS)
            
            pose = ip
            for k in range(num_iters - 1):
                rendered_k = rendered_list[k].unsqueeze(0).to(render_device)
                fmap_r = net.encode(rendered_k)
                corr = local_correlation(fmap_r, fmap_q, net.corr_radius)
                corr_feat = net.corr_encoder(corr)
                hidden = net.gru(hidden, corr_feat)
                flow = net.flow_head(hidden)
                conf = torch.sigmoid(net.conf_head(hidden))
                delta_xi = diff_pose_solve(flow, conf, Ju, Jv, valid, net.damping)
                delta_T = se3_exp(delta_xi)
                pose = delta_T @ pose
                
                # 在预测的新位姿处渲染
                result = renderer.render_scale(scale_name, pose[0])
                rendered_list.append(result['feature_map'].cpu())
        
        cache.append({
            'query_feats': query_feats,  # CPU
            'rendered_feats_list': rendered_list,  # list of CPU tensors
            'initial_pose': initial_pose,  # CPU
            'pose_gt': pose_gt,  # CPU
            'depth': depth,  # CPU
        })
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  Prerender [{i+1}/{N}] {elapsed:.0f}s, ETA {eta:.0f}s")
    
    elapsed = time.time() - t0
    mem_gb = sum(
        s['query_feats'].nelement() * 4 + 
        sum(r.nelement() * 4 for r in s['rendered_feats_list'])
        for s in cache
    ) / 1e9
    print(f"  Prerender done: {elapsed:.0f}s, cache={mem_gb:.1f}GB")
    
    return cache


def cache_collate(batch):
    """Collate cached prerendered samples into batches."""
    keys = ['query_feats', 'initial_pose', 'pose_gt', 'depth']
    result = {}
    for key in keys:
        items = [b[key] for b in batch]
        if items[0] is not None:
            result[key] = torch.stack(items, dim=0)
        else:
            result[key] = None
    
    # rendered_feats_list: list of (B, D, H, W)
    num_iters = len(batch[0]['rendered_feats_list'])
    result['rendered_feats_list'] = []
    for k in range(num_iters):
        result['rendered_feats_list'].append(
            torch.stack([b['rendered_feats_list'][k] for b in batch], dim=0)
        )
    
    return result


# ============================================================================
# Training
# ============================================================================

def train(args):
    render_device = torch.device(args.render_device)
    train_device = torch.device(args.train_device)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # --- Renderer (on render_device) ---
    print(f"Loading renderer on {render_device}...")
    renderer = MultiScaleRenderer(
        ply_path=args.ply_path,
        scale_model_paths={args.scale: args.feature_model},
        device=render_device,
        img_height=480, img_width=640,
        fx=320.0, fy=320.0, cx=319.5, cy=239.5,
    )
    print(f"  Renderer loaded: {args.scale}")
    
    # --- Dataset ---
    feat_dim = FEAT_DIM.get(args.scale, 768)
    noise_trans_m = args.noise_rot_deg / 50.0
    
    train_dataset = PoseDatasetV3(
        feature_base_dir=args.feature_dir,
        traj_path=args.traj_path,
        depth_dir=args.depth_dir,
        frame_indices=list(range(810)),
        scale_names=[args.scale],
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_m=noise_trans_m,
        is_train=True,
        depth_resize=(35, 46),
    )
    
    val_dataset = PoseDatasetV3(
        feature_base_dir=args.feature_dir,
        traj_path=args.traj_path,
        depth_dir=args.depth_dir,
        frame_indices=list(range(810, 900)),
        scale_names=[args.scale],
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_m=noise_trans_m,
        is_train=True,
        depth_resize=(35, 46),
    )
    print(f"  Train: {len(train_dataset)} frames, Val: {len(val_dataset)} frames")
    print(f"  Noise: rot={args.noise_rot_deg}°, trans={noise_trans_m:.3f}m")
    
    # --- Network (on train_device) ---
    net = CorrPoseNet(
        feat_dim=feat_dim,
        enc_dim=args.enc_dim,
        hidden_dim=args.hidden_dim,
        corr_radius=args.corr_radius,
        num_iters=args.num_train_iters,
        damping=args.damping,
    ).to(train_device)
    print(f"\n{net}\n")
    
    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # --- Resume ---
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=train_device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")
    
    # --- Output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'train_log.txt')
    
    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    
    log(f"Training CorrPoseNet (prerender mode) — {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"Args: {vars(args)}")
    log(f"Params: {net.num_parameters():,}")
    log(f"Render: {render_device}, Train: {train_device}")
    log(f"Batch size: {args.batch_size}, Train iters: {args.num_train_iters}")
    log("=" * 70)
    
    best_val_error = float('inf')
    
    # ================================================================
    # Training loop
    # ================================================================
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # --- Phase 1: Prerender all features ---
        log(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        log(f"  Phase 1: Prerendering...")
        
        # For multi-iter: put net on render_device temporarily for rollout
        net_on_render = None
        if args.num_train_iters > 1 and epoch > 0:
            net.eval()
            # Clone net to render_device for rollout (avoid moving the training copy)
            net_on_render = CorrPoseNet(
                feat_dim=feat_dim, enc_dim=args.enc_dim,
                hidden_dim=args.hidden_dim, corr_radius=args.corr_radius,
                num_iters=args.num_train_iters, damping=args.damping,
            ).to(render_device)
            net_on_render.load_state_dict(net.state_dict())
            net_on_render.eval()
        
        cache = prerender_epoch(
            train_dataset, renderer, args.scale, render_device,
            args.noise_rot_deg, noise_trans_m,
            num_iters=args.num_train_iters,
            net=net_on_render,
        )
        
        # Free rollout net
        del net_on_render
        if render_device != train_device:
            torch.cuda.empty_cache()
        
        prerender_time = time.time() - epoch_start
        
        # --- Phase 2: Train with prerendered features ---
        net.train()
        train_start = time.time()
        
        # Shuffle indices
        indices = np.random.permutation(len(cache))
        
        epoch_stats = {
            'loss': [], 'rot_init': [], 'rot_final': [],
            'trans_init': [], 'trans_final': [],
        }
        optimizer.zero_grad()
        
        num_batches = (len(cache) + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(num_batches):
            start_i = batch_idx * args.batch_size
            end_i = min(start_i + args.batch_size, len(cache))
            batch_indices = indices[start_i:end_i]
            
            # Collate batch
            batch_samples = [cache[i] for i in batch_indices]
            batch = cache_collate(batch_samples)
            
            query = batch['query_feats'].to(train_device)
            pose_gt = batch['pose_gt'].to(train_device)
            initial_pose = batch['initial_pose'].to(train_device)
            depth = batch['depth'].to(train_device) if batch['depth'] is not None else None
            rendered_list = [r.to(train_device) for r in batch['rendered_feats_list']]
            
            # Forward with prerendered features
            results = net.forward_prerendered(
                query_feats=query,
                rendered_feats_list=rendered_list,
                initial_pose=initial_pose,
                depth=depth,
                intrinsics=INTRINSICS,
            )
            
            # Pose loss
            loss = pose_loss_fn(
                results['poses'], pose_gt,
                gamma=args.gamma, trans_weight=args.trans_weight,
            )
            
            # Flow supervision
            if args.flow_loss_weight > 0 and depth is not None:
                Ju, Jv, valid = compute_image_jacobian(depth, INTRINSICS)
                flow_loss = linearized_flow_loss_fn(
                    results['flows'], results['confidences'],
                    results['poses'], pose_gt,
                    Ju, Jv, valid, gamma=args.gamma,
                )
                loss = loss + args.flow_loss_weight * flow_loss
            
            # Backward
            (loss / args.grad_accum).backward()
            
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Track metrics
            with torch.no_grad():
                init_rot, init_trans = compute_pose_error(initial_pose, pose_gt)
                final_rot, final_trans = compute_pose_error(
                    results['poses'][-1], pose_gt
                )
                epoch_stats['loss'].append(loss.item())
                epoch_stats['rot_init'].append(init_rot.mean().item())
                epoch_stats['rot_final'].append(final_rot.mean().item())
                epoch_stats['trans_init'].append(init_trans.mean().item())
                epoch_stats['trans_final'].append(final_trans.mean().item())
            
            if (batch_idx + 1) % args.log_every == 0:
                n = min(args.log_every, len(epoch_stats['loss']))
                log(f"  [{batch_idx+1:4d}/{num_batches}] "
                    f"loss={np.mean(epoch_stats['loss'][-n:]):.4f}  "
                    f"rot: {np.mean(epoch_stats['rot_init'][-n:]):.1f}° → "
                    f"{np.mean(epoch_stats['rot_final'][-n:]):.1f}°  "
                    f"trans: {np.mean(epoch_stats['trans_init'][-n:]):.3f} → "
                    f"{np.mean(epoch_stats['trans_final'][-n:]):.3f}m")
        
        # Flush remaining gradients
        if num_batches % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        train_time = time.time() - train_start
        epoch_time = time.time() - epoch_start
        
        # Clear cache to free memory
        del cache
        gc.collect()
        
        # --- Epoch summary ---
        avg = {k: np.mean(v) for k, v in epoch_stats.items()}
        med_final = np.median(epoch_stats['rot_final'])
        improvement = avg['rot_init'] - avg['rot_final']
        
        log(f"  Epoch {epoch+1}/{args.epochs} "
            f"(prerender={prerender_time:.0f}s, train={train_time:.0f}s, "
            f"total={epoch_time:.0f}s)  lr={scheduler.get_last_lr()[0]:.2e}")
        log(f"  Train: loss={avg['loss']:.4f}  "
            f"rot: {avg['rot_init']:.1f}° → {avg['rot_final']:.2f}° "
            f"(median {med_final:.2f}°, Δ={improvement:+.1f}°)  "
            f"trans: {avg['trans_init']:.3f} → {avg['trans_final']:.4f}m")
        
        # --- Validation (with live rendering) ---
        if (epoch + 1) % args.val_every == 0:
            val_results = validate(
                net, val_dataset, renderer, 
                render_device, train_device, args,
            )
            log(f"  Val:   rot: {val_results['rot_init']:.1f}° → "
                f"{val_results['rot_median']:.2f}° "
                f"(mean {val_results['rot_mean']:.2f}°)  "
                f"<1°={val_results['pct_1deg']:.1f}%  "
                f"<5°={val_results['pct_5deg']:.1f}%")
            
            if val_results['rot_median'] < best_val_error:
                best_val_error = val_results['rot_median']
                save_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'val_rot_median': val_results['rot_median'],
                    'val_rot_mean': val_results['rot_mean'],
                    'args': vars(args),
                }, save_path)
                log(f"  → Best model saved (val_rot={val_results['rot_median']:.2f}°)")
        
        # --- Checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
    
    log(f"\nTraining complete. Best val rotation: {best_val_error:.2f}°")


# ============================================================================
# Validation (uses live rendering for accurate multi-iteration eval)
# ============================================================================

@torch.no_grad()
def validate(net, val_dataset, renderer, render_device, train_device, args):
    net.eval()
    rot_inits, rot_finals, trans_finals = [], [], []
    
    # Validation always uses K=3 iterations with live rendering
    val_iters = max(args.num_train_iters, 3)
    
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        query = sample['query_feats'][args.scale].unsqueeze(0).to(train_device)
        pose_gt = sample['pose_gt'].unsqueeze(0).to(train_device)
        initial = sample['initial_pose'].unsqueeze(0).to(train_device)
        depth = sample['depth'].unsqueeze(0).to(train_device) if 'depth' in sample else None
        
        # Live rendering on render_device, network on train_device
        results = forward_cross_device(
            net, query, initial, depth, INTRINSICS,
            renderer, args.scale, render_device, train_device,
            num_iters=val_iters,
        )
        
        init_rot, _ = compute_pose_error(initial, pose_gt)
        final_rot, final_trans = compute_pose_error(results['poses'][-1], pose_gt)
        
        rot_inits.append(init_rot.item())
        rot_finals.append(final_rot.item())
        trans_finals.append(final_trans.item())
    
    net.train()
    
    rot_finals_arr = np.array(rot_finals)
    return {
        'rot_init': np.mean(rot_inits),
        'rot_median': np.median(rot_finals_arr),
        'rot_mean': np.mean(rot_finals_arr),
        'trans_median': np.median(trans_finals),
        'pct_1deg': (rot_finals_arr < 1.0).mean() * 100,
        'pct_5deg': (rot_finals_arr < 5.0).mean() * 100,
    }


@torch.no_grad()
def forward_cross_device(
    net, query_feats, initial_pose, depth, intrinsics,
    renderer, scale_name, render_device, train_device,
    num_iters=3,
):
    """
    双卡 forward: 渲染在 render_device, 网络在 train_device.
    """
    from ic_models.corr_pose_net import local_correlation, diff_pose_solve
    
    B, D, H, W = query_feats.shape
    Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)
    
    fmap_q = net.encode(query_feats)
    hidden = torch.tanh(net.context_encoder(query_feats))
    
    pose = initial_pose
    results = {'poses': [pose], 'flows': [], 'confidences': [], 'delta_xis': []}
    
    for k in range(num_iters):
        # Render on render_device
        feats = []
        for i in range(B):
            result = renderer.render_scale(scale_name, pose[i].to(render_device))
            feats.append(result['feature_map'].to(train_device))
        rendered = torch.stack(feats, dim=0)
        
        fmap_r = net.encode(rendered)
        corr = local_correlation(fmap_r, fmap_q, net.corr_radius)
        corr_feat = net.corr_encoder(corr)
        hidden = net.gru(hidden, corr_feat)
        
        flow = net.flow_head(hidden)
        conf = torch.sigmoid(net.conf_head(hidden))
        delta_xi = diff_pose_solve(flow, conf, Ju, Jv, valid, net.damping)
        
        delta_T = se3_exp(delta_xi)
        pose = delta_T @ pose
        
        results['poses'].append(pose)
        results['flows'].append(flow)
        results['confidences'].append(conf)
        results['delta_xis'].append(delta_xi)
    
    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train CorrPoseNet (fast prerender mode)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    g = parser.add_argument_group('Paths')
    g.add_argument('--ply_path', default=DEFAULT_PLY)
    g.add_argument('--feature_model', default=DEFAULT_FEATURE_MODEL)
    g.add_argument('--feature_dir', default=DEFAULT_FEATURE_DIR)
    g.add_argument('--traj_path', default=DEFAULT_TRAJ)
    g.add_argument('--depth_dir', default=DEFAULT_DEPTH_DIR)
    g.add_argument('--output_dir', default='output/corr_pose_net/room_0_fast')
    g.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    g = parser.add_argument_group('Architecture')
    g.add_argument('--scale', default='fine_dino')
    g.add_argument('--enc_dim', type=int, default=128)
    g.add_argument('--hidden_dim', type=int, default=128)
    g.add_argument('--corr_radius', type=int, default=4)
    g.add_argument('--num_train_iters', type=int, default=1,
                   help='Network iterations during training (1=fast, 3=full)')
    g.add_argument('--damping', type=float, default=1e-3)
    
    g = parser.add_argument_group('Training')
    g.add_argument('--epochs', type=int, default=50)
    g.add_argument('--batch_size', type=int, default=8,
                   help='Batch size (can be large with prerendered features)')
    g.add_argument('--lr', type=float, default=2e-4)
    g.add_argument('--weight_decay', type=float, default=1e-4)
    g.add_argument('--gamma', type=float, default=0.8)
    g.add_argument('--trans_weight', type=float, default=10.0)
    g.add_argument('--flow_loss_weight', type=float, default=1.0)
    g.add_argument('--grad_accum', type=int, default=1,
                   help='Gradient accumulation (1=no accum, larger batch already)')
    g.add_argument('--noise_rot_deg', type=float, default=10.0)
    g.add_argument('--seed', type=int, default=42)
    
    g = parser.add_argument_group('Devices')
    g.add_argument('--render_device', default='cuda:0',
                   help='GPU for rendering')
    g.add_argument('--train_device', default='cuda:0',
                   help='GPU for network training (use cuda:1 for dual-GPU)')
    
    g = parser.add_argument_group('Logging')
    g.add_argument('--log_every', type=int, default=20)
    g.add_argument('--val_every', type=int, default=5)
    g.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    train(args)
