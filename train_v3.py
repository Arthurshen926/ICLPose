#!/usr/bin/env python3
"""
ICPoseNetV3 训练脚本 (v4 - Flow Loss + Curriculum Learning)
=============================================================
核心改进:
  - Flow Loss: dense 2D supervision (35×46=1610 pixel) 提供 ~1600× 更密集的梯度
  - 课程学习: 从小噪声开始 → 逐步增大
  - 缩放后的 intrinsics + 深度图确保 flow GT 分辨率正确
  
设计:
  Pose Loss (全局 6-DOF) + λ_flow × Flow Loss (像素级 2D 对应)
  → Flow Loss 训练特征处理骨架, Pose Loss 训练最终位姿输出
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from ic_models.ic_pose_net_v3 import ICPoseNetV3
from modules.multiscale_renderer import MultiScaleRenderer
from losses.sequence_loss import SequenceLoss, PoseOnlySequenceLoss, rotation_geodesic_loss
from data.dataset_v3 import PoseDatasetV3, collate_v3


# ============ Camera Intrinsics ============
# Replica room_0: 640×480, fx=fy=320, cx=319.5, cy=239.5

IMG_W, IMG_H = 640, 480
FLOW_W, FLOW_H = 46, 35  # Model output resolution

INTRINSICS_FULL = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
INTRINSICS_FLOW = {
    'fx': 320.0 * FLOW_W / IMG_W,   # ≈ 23.0
    'fy': 320.0 * FLOW_H / IMG_H,   # ≈ 23.3
    'cx': 319.5 * FLOW_W / IMG_W,   # ≈ 22.97
    'cy': 239.5 * FLOW_H / IMG_H,   # ≈ 17.47
}


# ============ Curriculum Noise Schedule ============

def get_curriculum_noise(epoch, args):
    """课程噪声调度: warmup → linear ramp-up → final"""
    if not args.curriculum:
        return args.noise_rot, args.noise_trans
    
    warmup = args.warmup_epochs
    rampup_end = args.rampup_epochs
    
    if epoch <= warmup:
        return args.start_noise_rot, args.start_noise_trans
    elif epoch <= rampup_end:
        progress = (epoch - warmup) / max(rampup_end - warmup, 1)
        noise_rot = args.start_noise_rot + (args.noise_rot - args.start_noise_rot) * progress
        noise_trans = args.start_noise_trans + (args.noise_trans - args.start_noise_trans) * progress
        return noise_rot, noise_trans
    else:
        return args.noise_rot, args.noise_trans


def parse_args():
    p = argparse.ArgumentParser(description='ICPoseNetV3 Training')
    
    # 数据路径
    p.add_argument('--scene', default='room_0')
    p.add_argument('--dataset_root', default='dataset')
    p.add_argument('--feature_dir', default='output/features_multiscale/room_0')
    p.add_argument('--ply_path', default='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply')
    p.add_argument('--output_dir', default='output/v3_training')
    
    # 3DGS 模型路径  
    p.add_argument('--model_coarse', default='output/feature_3dgs/room_0_raw/coarse/best_model.pth')
    p.add_argument('--model_fine_sd', default='output/feature_3dgs/room_0_raw/fine_sd/best_model.pth')
    p.add_argument('--model_fine_dino', default='output/feature_3dgs/room_0_raw/fine_dino/best_model.pth')
    
    # 尺度选择
    p.add_argument('--scales', nargs='+', default=['fine_sd', 'fine_dino'])
    
    # 训练参数
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--num_workers', type=int, default=0)
    
    # 网络参数
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_iters', type=int, default=3)
    p.add_argument('--residual_mode', default='concat')
    
    # 损失参数
    p.add_argument('--gamma', type=float, default=0.8)
    p.add_argument('--lambda_trans', type=float, default=0.5)
    p.add_argument('--lambda_flow', type=float, default=0.1,
                   help='Flow loss weight (0 = disable flow loss)')
    p.add_argument('--confidence_penalty', type=float, default=0.1)
    
    # 课程学习
    p.add_argument('--curriculum', action='store_true', default=True)
    p.add_argument('--no_curriculum', dest='curriculum', action='store_false')
    p.add_argument('--start_noise_rot', type=float, default=2.0)
    p.add_argument('--start_noise_trans', type=float, default=0.03)
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--rampup_epochs', type=int, default=30)
    
    # 位姿扰动
    p.add_argument('--noise_rot', type=float, default=15.0)
    p.add_argument('--noise_trans', type=float, default=0.3)
    p.add_argument('--val_noise_rot', type=float, default=10.0)
    p.add_argument('--val_noise_trans', type=float, default=0.2)
    
    # Depth 
    p.add_argument('--depth_scale', type=float, default=1000.0)
    
    # 其他
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--log_interval', type=int, default=20)
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=str, default=None)
    
    return p.parse_args()


def setup_renderer(args, device):
    """初始化 MultiScaleRenderer"""
    scale_model_map = {
        'coarse': args.model_coarse,
        'fine_sd': args.model_fine_sd,
        'fine_dino': args.model_fine_dino,
    }
    
    scale_model_paths = {}
    for name in args.scales:
        path = scale_model_map.get(name, '')
        if os.path.exists(path):
            scale_model_paths[name] = path
        else:
            print(f"[Warning] Model not found: {path}, skipping scale '{name}'")
    
    if not scale_model_paths:
        raise RuntimeError("No 3DGS models available!")
    
    return MultiScaleRenderer(
        ply_path=args.ply_path,
        scale_model_paths=scale_model_paths,
        device=device,
    )


def setup_data(args, scale_names, use_depth=True):
    """创建训练/验证数据集"""
    traj_path = os.path.join(args.dataset_root, args.scene, 'Sequence_1', 'traj_w_c.txt')
    depth_dir = os.path.join(args.dataset_root, args.scene, 'Sequence_1', 'depth') if use_depth else None
    
    all_indices = list(range(900))
    np.random.seed(args.seed)
    np.random.shuffle(all_indices)
    val_size = int(len(all_indices) * args.val_ratio)
    val_indices = sorted(all_indices[:val_size])
    train_indices = sorted(all_indices[val_size:])
    
    print(f"[Data] Train: {len(train_indices)} frames, Val: {len(val_indices)} frames")
    
    init_noise_rot = args.start_noise_rot if args.curriculum else args.noise_rot
    init_noise_trans = args.start_noise_trans if args.curriculum else args.noise_trans
    
    train_dataset = PoseDatasetV3(
        feature_base_dir=args.feature_dir,
        traj_path=traj_path,
        depth_dir=depth_dir,
        frame_indices=train_indices,
        scale_names=scale_names,
        depth_scale=args.depth_scale,
        noise_rot_deg=init_noise_rot,
        noise_trans_m=init_noise_trans,
        is_train=True,
        depth_resize=(FLOW_H, FLOW_W),  # Resize depth to 35×46 for flow
    )
    
    val_dataset = PoseDatasetV3(
        feature_base_dir=args.feature_dir,
        traj_path=traj_path,
        depth_dir=depth_dir,
        frame_indices=val_indices,
        scale_names=scale_names,
        noise_rot_deg=args.val_noise_rot,
        noise_trans_m=args.val_noise_trans,
        is_train=True,
        depth_resize=(FLOW_H, FLOW_W),
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_v3,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_v3,
        pin_memory=True,
    )
    
    return train_loader, val_loader, train_dataset


def train_one_epoch(
    model, renderer, loss_fn, optimizer, 
    train_loader, device, epoch, args,
    intrinsics=None,
):
    model.train()
    
    accum_steps = args.grad_accum
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_rot_init = 0.0
    total_rot_final = 0.0
    total_trans_init = 0.0
    total_trans_final = 0.0
    total_flow_loss = 0.0
    count = 0
    
    t0 = time.time()
    for batch_idx, batch in enumerate(train_loader):
        query_feats = {k: v.to(device) for k, v in batch['query_feats'].items()}
        pose_gt = batch['pose_gt'].to(device)
        initial_pose = batch['initial_pose'].to(device)
        B = pose_gt.shape[0]
        
        # Depth for flow loss
        depth_gt = None
        if 'depth' in batch:
            depth_gt = batch['depth'].to(device)
        
        # 初始误差
        with torch.no_grad():
            init_rot = rotation_geodesic_loss(
                initial_pose[:, :3, :3], pose_gt[:, :3, :3]
            ).mean() * (180.0 / np.pi)
            init_trans = torch.norm(
                initial_pose[:, :3, 3] - pose_gt[:, :3, 3], dim=-1
            ).mean()
        
        # 前向传播
        predictions = model(
            query_feats=query_feats,
            initial_pose=initial_pose,
            renderer=renderer,
            num_iters=args.num_iters,
            depth_gt=depth_gt,
        )
        
        # 计算损失 (包含 flow loss)
        if isinstance(loss_fn, SequenceLoss) and depth_gt is not None:
            loss_dict = loss_fn(predictions, pose_gt, depth_gt=depth_gt, intrinsics=intrinsics)
        else:
            loss_dict = loss_fn(predictions, pose_gt)
        
        loss = loss_dict['total_loss'] / accum_steps
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accum_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计
        cur_rot = loss_dict['final_rotation_error_deg'].item()
        cur_trans = loss_dict['final_translation_error_m'].item()
        total_loss += loss_dict['total_loss'].item() * B
        total_rot_init += init_rot.item() * B
        total_rot_final += cur_rot * B
        total_trans_init += init_trans.item() * B
        total_trans_final += cur_trans * B
        if 'flow_loss' in loss_dict and torch.is_tensor(loss_dict['flow_loss']):
            total_flow_loss += loss_dict['flow_loss'].item() * B
        count += B
        
        if (batch_idx + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            speed = elapsed / (batch_idx + 1)
            xi_max = max(xi.abs().max().item() for xi in predictions['xi_list'])
            avg_rot_init = total_rot_init / count
            avg_rot_final = total_rot_final / count
            improvement = avg_rot_init - avg_rot_final
            flow_str = f" fl={total_flow_loss/count:.4f}" if total_flow_loss > 0 else ""
            mem_str = ""
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / 1e6
                mem_str = f" mem={mem_mb:.0f}MB"
            print(f"  [{epoch}][{batch_idx+1}/{len(train_loader)}] "
                  f"loss={total_loss/count:.3f}{flow_str} "
                  f"rot={init_rot.item():.1f}→{cur_rot:.1f}° "
                  f"[avg:{avg_rot_init:.1f}→{avg_rot_final:.1f}° Δ={improvement:+.2f}°] "
                  f"xi={xi_max:.4f} ({speed:.1f}s/it){mem_str}")
    
    # Final accumulation step
    if len(train_loader) % accum_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    
    n = max(count, 1)
    return {
        'loss': total_loss / n,
        'rot_err': total_rot_final / n,
        'trans_err': total_trans_final / n,
        'init_rot': total_rot_init / n,
        'init_trans': total_trans_init / n,
        'flow_loss': total_flow_loss / n,
    }


@torch.no_grad()
def validate(model, renderer, loss_fn, val_loader, device, args, intrinsics=None):
    model.eval()
    total_loss = 0.0
    total_rot_init = 0.0
    total_rot_final = 0.0
    total_trans_init = 0.0
    total_trans_final = 0.0
    count = 0
    
    eval_iters = max(args.num_iters + 2, 4)
    
    for batch in val_loader:
        query_feats = {k: v.to(device) for k, v in batch['query_feats'].items()}
        pose_gt = batch['pose_gt'].to(device)
        initial_pose = batch['initial_pose'].to(device)
        B = pose_gt.shape[0]
        
        depth_gt = batch['depth'].to(device) if 'depth' in batch else None
        
        init_rot = rotation_geodesic_loss(
            initial_pose[:, :3, :3], pose_gt[:, :3, :3]
        ).mean() * (180.0 / np.pi)
        init_trans = torch.norm(
            initial_pose[:, :3, 3] - pose_gt[:, :3, 3], dim=-1
        ).mean()
        
        predictions = model(
            query_feats=query_feats,
            initial_pose=initial_pose,
            renderer=renderer,
            num_iters=eval_iters,
            depth_gt=depth_gt,
        )
        
        if isinstance(loss_fn, SequenceLoss) and depth_gt is not None:
            loss_dict = loss_fn(predictions, pose_gt, depth_gt=depth_gt, intrinsics=intrinsics)
        else:
            loss_dict = loss_fn(predictions, pose_gt)
        
        total_loss += loss_dict['total_loss'].item() * B
        total_rot_final += loss_dict['final_rotation_error_deg'].item() * B
        total_trans_final += loss_dict['final_translation_error_m'].item() * B
        total_rot_init += init_rot.item() * B
        total_trans_init += init_trans.item() * B
        count += B
    
    n = max(count, 1)
    return {
        'loss': total_loss / n,
        'rot_err': total_rot_final / n,
        'trans_err': total_trans_final / n,
        'init_rot': total_rot_init / n,
        'init_trans': total_trans_init / n,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("ICPoseNetV3 Training (v4 - Flow Loss + Curriculum)")
    print("=" * 70)
    
    # 1. Renderer  
    print(f"\n[1] Loading renderer with scales: {args.scales}")
    renderer = setup_renderer(args, device)
    
    scale_configs = []
    active_scales = []
    for name in args.scales:
        if name not in renderer.scale_info:
            continue
        info = renderer.scale_info[name]
        scale_configs.append({
            'name': name,
            'feat_dim': info['feat_dim'],
            'resolution': info['resolution'],
        })
        active_scales.append(name)
    print(f"  Active: {active_scales}")
    
    # 2. Model
    print("\n[2] Creating model...")
    model = ICPoseNetV3(
        scale_configs=scale_configs,
        hidden_dim=args.hidden_dim,
        output_resolution=(FLOW_H, FLOW_W),
        num_iters=args.num_iters,
        residual_mode=args.residual_mode,
        residual_out_dim=64,
        flow_to_pose_intrinsics=INTRINSICS_FLOW if args.lambda_flow > 0 else None,
        flow_to_pose_damping=1e-3,
    ).to(device)
    
    # 3. Data (with depth for flow loss)
    use_depth = args.lambda_flow > 0
    print(f"\n[3] Loading data... (depth={use_depth})")
    train_loader, val_loader, train_dataset = setup_data(args, active_scales, use_depth=use_depth)
    
    # 4. Loss + Optimizer
    if args.lambda_flow > 0:
        print(f"  Using SequenceLoss with flow_loss (λ_flow={args.lambda_flow})")
        loss_fn = SequenceLoss(
            gamma=args.gamma,
            lambda_translation=args.lambda_trans,
            lambda_flow=args.lambda_flow,
            confidence_penalty=args.confidence_penalty,
            use_flow_loss=True,
        )
        # Intrinsics scaled for flow resolution (35×46)
        flow_intrinsics = INTRINSICS_FLOW
    else:
        print(f"  Using PoseOnlySequenceLoss (no flow)")
        loss_fn = PoseOnlySequenceLoss(
            gamma=args.gamma,
            lambda_translation=args.lambda_trans,
        )
        flow_intrinsics = None
    
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    
    # Resume
    start_epoch = 1
    best_val_rot = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_rot = ckpt.get('val_rot_err', float('inf'))
        print(f"  Resumed from epoch {start_epoch-1}, best_rot={best_val_rot:.2f}°")
    
    eff_bs = args.batch_size * args.grad_accum
    print(f"\n[Config] epochs={args.epochs}, bs={args.batch_size}×{args.grad_accum}={eff_bs}, "
          f"lr={args.lr}, iters={args.num_iters}, γ={args.gamma}, λ_t={args.lambda_trans}, λ_f={args.lambda_flow}")
    if args.curriculum:
        print(f"  Curriculum: noise {args.start_noise_rot}°/{args.start_noise_trans}m "
              f"→ {args.noise_rot}°/{args.noise_trans}m "
              f"(warmup={args.warmup_epochs}, rampup_to={args.rampup_epochs})")
    else:
        print(f"  Fixed noise: rot={args.noise_rot}°, trans={args.noise_trans}m")
    print(f"  Val noise: rot={args.val_noise_rot}°, trans={args.val_noise_trans}m")
    if flow_intrinsics:
        print(f"  Flow intrinsics: fx={flow_intrinsics['fx']:.1f}, fy={flow_intrinsics['fy']:.1f}")
    
    # 5. Train loop
    log_path = os.path.join(args.output_dir, 'train_log.txt')
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        # Curriculum
        cur_noise_rot, cur_noise_trans = get_curriculum_noise(epoch, args)
        train_dataset.noise_rot_deg = cur_noise_rot
        train_dataset.noise_trans_m = cur_noise_trans
        
        train_m = train_one_epoch(
            model, renderer, loss_fn, optimizer,
            train_loader, device, epoch, args,
            intrinsics=flow_intrinsics,
        )
        
        scheduler.step()
        
        val_m = validate(
            model, renderer, loss_fn, val_loader, device, args,
            intrinsics=flow_intrinsics,
        )
        
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        improvement_train = train_m['init_rot'] - train_m['rot_err']
        improvement_val = val_m['init_rot'] - val_m['rot_err']
        
        flow_str = f" fl={train_m['flow_loss']:.4f}" if train_m['flow_loss'] > 0 else ""
        log_line = (
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.0f}s) lr={lr_now:.6f} "
            f"noise={cur_noise_rot:.1f}°/{cur_noise_trans:.2f}m | "
            f"Train: loss={train_m['loss']:.3f}{flow_str} "
            f"rot={train_m['init_rot']:.1f}→{train_m['rot_err']:.1f}° (Δ={improvement_train:+.2f}°) "
            f"t={train_m['init_trans']:.2f}→{train_m['trans_err']:.2f}m | "
            f"Val: rot={val_m['init_rot']:.1f}→{val_m['rot_err']:.1f}° (Δ={improvement_val:+.2f}°) "
            f"t={val_m['init_trans']:.2f}→{val_m['trans_err']:.2f}m"
        )
        print(log_line)
        with open(log_path, 'a') as f:
            f.write(log_line + '\n')
        
        # Save best
        if val_m['rot_err'] < best_val_rot:
            best_val_rot = val_m['rot_err']
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_rot_err': val_m['rot_err'],
                'val_trans_err': val_m['trans_err'],
                'config': vars(args),
                'scale_configs': scale_configs,
            }, save_path)
            print(f"  ★ Best: rot={best_val_rot:.2f}° saved")
        
        # Periodic save
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_ep{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
    
    print(f"\nTraining complete! Best val rotation error: {best_val_rot:.2f}°")


if __name__ == '__main__':
    main()
