#!/usr/bin/env python3
"""
Training script for CorrPoseNet — Correlation-based Pose Refinement.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_corr_pose.py
    CUDA_VISIBLE_DEVICES=0 python scripts/train_corr_pose.py --noise_rot_deg 15 --epochs 80

训练流程 (每个 batch):
1. 加载查询特征 + GT位姿 + 随机扰动的初始位姿 + GT深度
2. 迭代 K 次:
   a. 在当前位姿渲染3DGS特征 (detached, 不反传梯度)
   b. 编码 + 局部 correlation + ConvGRU → flow + confidence
   c. 可微分几何求解器: flow → δξ → 更新位姿
3. 位姿误差 loss: 各迭代加权求和
4. (可选) Flow GT: 从渲染深度 + 位姿差计算

性能预估 (RTX 3090, batch_size=1, 3 iters):
- 渲染: ~500ms/iter × 3 = 1.5s
- 网络 forward+backward: ~50ms
- 总计: ~1.6s/sample → 810 samples × 1.6s ≈ 22 min/epoch
- 50 epochs ≈ 18 hours
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time

from torch.utils.data import DataLoader
from ic_models.corr_pose_net import CorrPoseNet
from data.dataset_v3 import PoseDatasetV3, collate_v3
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_log, pose_inverse, compute_gt_flow


# ============================================================================
# Constants
# ============================================================================

# Replica room_0 defaults
DEFAULT_PLY = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
DEFAULT_FEATURE_MODEL = 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth'
DEFAULT_FEATURE_DIR = 'output/features_multiscale/room_0'
DEFAULT_TRAJ = 'dataset/room_0/Sequence_1/traj_w_c.txt'
DEFAULT_DEPTH_DIR = 'dataset/room_0/Sequence_1/depth'

# Intrinsics at feature resolution 35×46
# fx = 320 * (46/640), fy = 320 * (35/480), etc.
INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

FEAT_DIM = {'fine_dino': 768, 'fine_sd': 640}


# ============================================================================
# Loss Functions
# ============================================================================

def compute_pose_error(pred_pose, gt_pose):
    """
    Compute rotation (degrees) and translation (meters) error.
    
    Args:
        pred_pose: (B, 4, 4) predicted w2c
        gt_pose: (B, 4, 4) ground truth w2c
    Returns:
        rot_error_deg: (B,) 
        trans_error_m: (B,)
    """
    R_pred = pred_pose[:, :3, :3]
    R_gt = gt_pose[:, :3, :3]
    t_pred = pred_pose[:, :3, 3]
    t_gt = gt_pose[:, :3, 3]
    
    # Geodesic rotation error
    R_rel = R_pred @ R_gt.transpose(-1, -2)
    cos_angle = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)
    rot_error_deg = torch.acos(cos_angle) * (180.0 / np.pi)
    
    # Translation error
    trans_error_m = (t_pred - t_gt).norm(dim=-1)
    
    return rot_error_deg, trans_error_m


def pose_loss_fn(pred_poses, gt_pose, gamma=0.8, trans_weight=10.0):
    """
    位姿损失: 各迭代加权求和.
    
    对每个迭代 k:
      rot_loss_k = 1 - cos(θ)   (θ = geodesic angle, ∈ [0, 2])
      trans_loss_k = ||t_pred - t_gt||₂
      
    权重: w_k = γ^(K-1-k), 后迭代权重更高.
    
    Args:
        pred_poses: list of (B, 4, 4), len = K+1 (包含 initial)
        gt_pose: (B, 4, 4)
        gamma: 迭代权重衰减
        trans_weight: 平移loss权重系数
    """
    K = len(pred_poses) - 1  # 迭代次数
    total_loss = torch.tensor(0.0, device=gt_pose.device)
    
    for k in range(K):
        pred = pred_poses[k + 1]  # [0] = initial, [1..K] = predictions
        
        # Rotation loss: 1 - cos(θ) — smooth, gradient = sin(θ)
        R_pred = pred[:, :3, :3]
        R_gt = gt_pose[:, :3, :3]
        R_rel = R_pred @ R_gt.transpose(-1, -2)
        cos_angle = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        cos_angle = cos_angle.clamp(-1, 1)
        rot_loss = (1 - cos_angle).mean()
        
        # Translation loss: L2
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
    """
    线性化 GT flow 监督.
    
    关键原理:
      Image Jacobian 将 se3 运动映射到像素位移:
        Ju · ξ ≈ Δu,  Jv · ξ ≈ Δv
      其中 ξ = se3_log(T_gt · T_k^{-1}) 是从当前位姿到GT位姿的运动.
      
      如果网络预测 flow = J · ξ_gt, 则几何求解器恢复 δξ = ξ_gt ✓
      这直接约束flow为物理正确的位移, 防止退化解.
    
    相比上一版 (compute_gt_flow):
      - 无需渲染深度 / 重投影, 直接用 Image Jacobian
      - 与几何求解器完全一致的线性化
      - 适用于任意位姿偏差 (是linearization, 大偏差有误差但方向正确)
    
    Args:
        pred_flows: list of (B, 2, H, W) 网络预测的flow
        pred_confs: list of (B, 1, H, W) 网络预测的置信度
        poses: list of (B, 4, 4) 各迭代位姿 [initial, after_iter1, ...]
        pose_gt: (B, 4, 4) GT位姿
        Ju: (B, N, 6) Image Jacobian ∂u/∂ξ
        Jv: (B, N, 6) Image Jacobian ∂v/∂ξ
        valid: (B, N) 有效像素mask
        gamma: 迭代权重衰减
    """
    from modules.lie_algebra import se3_log, pose_inverse
    
    K = len(pred_flows)
    B = pose_gt.shape[0]
    H_W = Ju.shape[1]  # N = H * W
    device = pose_gt.device
    total_loss = torch.tensor(0.0, device=device)
    valid_f = valid.float()  # (B, N)
    
    for k in range(K):
        pose_k = poses[k].detach()  # 该迭代开始时的位姿 (detach, 不反传)
        
        # GT relative motion: se3_log(T_gt · T_k^{-1})
        T_rel = pose_gt @ pose_inverse(pose_k)  # (B, 4, 4)
        xi_gt = se3_log(T_rel)  # (B, 6)
        
        # Linearized GT flow: flow = J · ξ
        gt_flow_u = torch.bmm(Ju, xi_gt.unsqueeze(-1)).squeeze(-1)  # (B, N)
        gt_flow_v = torch.bmm(Jv, xi_gt.unsqueeze(-1)).squeeze(-1)  # (B, N)
        
        # Reshape to (B, 2, H, W)
        H = int(H_W ** 0.5 * (46/35) + 0.5) if H_W != 35 * 46 else 35  # handle dynamic
        W = H_W // 35 if H == 35 else H_W // H
        # Safer: infer from pred_flows shape
        _, _, pH, pW = pred_flows[k].shape
        gt_flow = torch.stack([
            gt_flow_u.reshape(B, pH, pW),
            gt_flow_v.reshape(B, pH, pW),
        ], dim=1)  # (B, 2, H, W)
        
        valid_mask = valid_f.reshape(B, 1, pH, pW)  # (B, 1, H, W)
        
        # L1 flow loss (unweighted by confidence — direct supervision)
        flow_diff = (pred_flows[k] - gt_flow).abs()  # (B, 2, H, W)
        flow_loss = (flow_diff * valid_mask).sum() / (valid_mask.sum() * 2 + 1e-8)
        
        # Confidence loss: confidence should be high where flow is accurate
        # Encourage high confidence on valid pixels
        conf = pred_confs[k]  # (B, 1, H, W)
        conf_loss = -(conf * valid_mask + 1e-8).log().mean() * 0.01
        
        weight = gamma ** (K - 1 - k)
        total_loss = total_loss + weight * (flow_loss + conf_loss)
    
    return total_loss


# ============================================================================
# Training
# ============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # --- Curriculum schedule ---
    # 解析课程学习配置: "epoch1:noise1,epoch2:noise2,..."
    # 例如 "0:5,50:10,100:15" → epoch 0-49 用5°, 50-99 用10°, 100+ 用15°
    curriculum_schedule = None
    if args.curriculum:
        curriculum_schedule = []
        for item in args.curriculum.split(','):
            ep, noise = item.strip().split(':')
            curriculum_schedule.append((int(ep), float(noise)))
        curriculum_schedule.sort(key=lambda x: x[0])
        print(f"[Curriculum] Schedule: {curriculum_schedule}")
    
    def get_noise_for_epoch(epoch):
        """根据课程学习schedule返回当前epoch的噪声级别"""
        if curriculum_schedule is None:
            return args.noise_rot_deg
        noise = curriculum_schedule[0][1]  # 默认第一阶段
        for ep, n in curriculum_schedule:
            if epoch >= ep:
                noise = n
            else:
                break
        return noise
    
    # --- Renderer ---
    print("Loading renderer...")
    renderer = MultiScaleRenderer(
        ply_path=args.ply_path,
        scale_model_paths={args.scale: args.feature_model},
        device=device,
        img_height=480, img_width=640,
        fx=320.0, fy=320.0, cx=319.5, cy=239.5,
    )
    print(f"  Renderer loaded: {args.scale}")
    
    # --- Dataset ---
    feat_dim = FEAT_DIM.get(args.scale, 768)
    
    train_dataset = PoseDatasetV3(
        feature_base_dir=args.feature_dir,
        traj_path=args.traj_path,
        depth_dir=args.depth_dir,
        frame_indices=list(range(810)),
        scale_names=[args.scale],
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_m=args.noise_rot_deg / 50.0,
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
        noise_trans_m=args.noise_rot_deg / 50.0,
        is_train=True,
        depth_resize=(35, 46),
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        collate_fn=collate_v3,
        drop_last=True,
        pin_memory=True,
    )
    print(f"  Train: {len(train_dataset)} frames, Val: {len(val_dataset)} frames")
    print(f"  Noise: rot={args.noise_rot_deg}°, trans={args.noise_rot_deg/50:.3f}m")
    
    # --- Network ---
    net = CorrPoseNet(
        feat_dim=feat_dim,
        enc_dim=args.enc_dim,
        hidden_dim=args.hidden_dim,
        corr_radius=args.corr_radius,
        num_iters=args.num_iters,
        damping=args.damping,
        use_multiscale=args.use_multiscale,
        coarse_iters=args.coarse_iters,
        coarse_scale_factor=args.coarse_scale_factor,
    ).to(device)
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
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")
    
    # --- Output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'train_log.txt')
    
    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    
    log(f"Training CorrPoseNet — {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"Args: {vars(args)}")
    log(f"Params: {net.num_parameters():,}")
    log("=" * 70)
    
    best_val_error = float('inf')
    
    # ================================================================
    # Training loop
    # ================================================================
    for epoch in range(start_epoch, args.epochs):
        # --- Curriculum: 动态更新噪声级别 ---
        current_noise = get_noise_for_epoch(epoch)
        current_trans_noise = current_noise / 50.0
        if curriculum_schedule is not None:
            train_dataset.noise_rot_deg = current_noise
            train_dataset.noise_trans_m = current_trans_noise
        
        net.train()
        epoch_stats = {
            'loss': [], 'rot_init': [], 'rot_final': [], 
            'trans_init': [], 'trans_final': [],
        }
        epoch_start = time.time()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            query = batch['query_feats'][args.scale].to(device)
            pose_gt = batch['pose_gt'].to(device)
            initial_pose = batch['initial_pose'].to(device)
            depth = batch['depth'].to(device) if 'depth' in batch else None
            
            # Forward: iterative render-and-compare
            results = net(
                query_feats=query,
                initial_pose=initial_pose,
                depth=depth,
                intrinsics=INTRINSICS,
                renderer=renderer,
                scale_name=args.scale,
            )
            
            # Pose loss
            loss = pose_loss_fn(
                results['poses'], pose_gt, 
                gamma=args.gamma, trans_weight=args.trans_weight,
            )
            
            # Optional: linearized flow supervision
            if args.flow_loss_weight > 0 and depth is not None:
                from modules.featuremetric import compute_image_jacobian
                Ju, Jv, valid = compute_image_jacobian(depth, INTRINSICS)
                
                flow_loss = linearized_flow_loss_fn(
                    results['flows'], results['confidences'],
                    results['poses'], pose_gt,
                    Ju, Jv, valid,
                    gamma=args.gamma,
                )
                loss = loss + args.flow_loss_weight * flow_loss
            
            # Backward with gradient accumulation
            (loss / args.grad_accum).backward()
            
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # --- Track metrics ---
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
            
            # --- Log ---
            if (batch_idx + 1) % args.log_every == 0:
                n = min(args.log_every, len(epoch_stats['loss']))
                log(f"  [{batch_idx+1:4d}/{len(train_loader)}] "
                    f"loss={np.mean(epoch_stats['loss'][-n:]):.4f}  "
                    f"rot: {np.mean(epoch_stats['rot_init'][-n:]):.1f}° → "
                    f"{np.mean(epoch_stats['rot_final'][-n:]):.1f}°  "
                    f"trans: {np.mean(epoch_stats['trans_init'][-n:]):.3f} → "
                    f"{np.mean(epoch_stats['trans_final'][-n:]):.3f}m")
        
        # Flush remaining gradients
        if (batch_idx + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # --- Epoch summary ---
        avg = {k: np.mean(v) for k, v in epoch_stats.items()}
        med_final = np.median(epoch_stats['rot_final'])
        improvement = avg['rot_init'] - avg['rot_final']
        
        log(f"\nEpoch {epoch+1}/{args.epochs} "
            f"({epoch_time:.0f}s, {epoch_time/len(train_dataset):.2f}s/sample)  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
            f"{f'  noise={current_noise:.0f}°' if curriculum_schedule else ''}")
        log(f"  Train: loss={avg['loss']:.4f}  "
            f"rot: {avg['rot_init']:.1f}° → {avg['rot_final']:.2f}° "
            f"(median {med_final:.2f}°, Δ={improvement:+.1f}°)  "
            f"trans: {avg['trans_init']:.3f} → {avg['trans_final']:.4f}m")
        
        # --- Validation ---
        if (epoch + 1) % args.val_every == 0:
            val_results = validate(
                net, val_dataset, renderer, device, args
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
        
        log("")
    
    log(f"Training complete. Best val rotation: {best_val_error:.2f}°")


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate(net, val_dataset, renderer, device, args, val_seed=12345):
    """在验证集上评估，返回统计信息.
    
    Args:
        val_seed: 固定随机种子，确保每次验证使用相同的噪声扰动，
                  消除随机性带来的指标波动。
    """
    net.eval()
    rot_inits, rot_finals, trans_finals = [], [], []
    
    # 保存当前 RNG 状态，验证后恢复
    rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    
    # 固定种子，确保每次验证噪声一致
    torch.manual_seed(val_seed)
    np.random.seed(val_seed)
    
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        query = sample['query_feats'][args.scale].unsqueeze(0).to(device)
        pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
        initial = sample['initial_pose'].unsqueeze(0).to(device)
        depth = sample['depth'].unsqueeze(0).to(device) if 'depth' in sample else None
        
        results = net(
            query_feats=query,
            initial_pose=initial,
            depth=depth,
            intrinsics=INTRINSICS,
            renderer=renderer,
            scale_name=args.scale,
        )
        
        init_rot, _ = compute_pose_error(initial, pose_gt)
        final_rot, final_trans = compute_pose_error(results['poses'][-1], pose_gt)
        
        rot_inits.append(init_rot.item())
        rot_finals.append(final_rot.item())
        trans_finals.append(final_trans.item())
    
    net.train()
    
    # 恢复 RNG 状态，不影响训练随机性
    torch.set_rng_state(rng_state)
    np.random.set_state(np_rng_state)
    
    rot_finals_arr = np.array(rot_finals)
    return {
        'rot_init': np.mean(rot_inits),
        'rot_median': np.median(rot_finals_arr),
        'rot_mean': np.mean(rot_finals_arr),
        'trans_median': np.median(trans_finals),
        'pct_1deg': (rot_finals_arr < 1.0).mean() * 100,
        'pct_5deg': (rot_finals_arr < 5.0).mean() * 100,
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train CorrPoseNet for iterative pose refinement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # --- Paths ---
    g = parser.add_argument_group('Paths')
    g.add_argument('--ply_path', default=DEFAULT_PLY)
    g.add_argument('--feature_model', default=DEFAULT_FEATURE_MODEL)
    g.add_argument('--feature_dir', default=DEFAULT_FEATURE_DIR)
    g.add_argument('--traj_path', default=DEFAULT_TRAJ)
    g.add_argument('--depth_dir', default=DEFAULT_DEPTH_DIR)
    g.add_argument('--output_dir', default='output/corr_pose_net/room_0')
    g.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    # --- Architecture ---
    g = parser.add_argument_group('Architecture')
    g.add_argument('--scale', default='fine_dino', 
                   choices=['fine_dino', 'fine_sd'])
    g.add_argument('--enc_dim', type=int, default=128,
                   help='Encoded feature dimension')
    g.add_argument('--hidden_dim', type=int, default=128,
                   help='ConvGRU hidden dimension')
    g.add_argument('--corr_radius', type=int, default=4,
                   help='Local correlation search radius (4 → 81 channels)')
    g.add_argument('--num_iters', type=int, default=3,
                   help='Number of render-and-compare iterations')
    g.add_argument('--damping', type=float, default=1e-3,
                   help='LM damping for pose solver')
    g.add_argument('--use_multiscale', action='store_true',
                   help='Enable coarse-to-fine multi-scale correlation')
    g.add_argument('--coarse_iters', type=int, default=1,
                   help='Number of coarse-scale iterations (only with --use_multiscale)')
    g.add_argument('--coarse_scale_factor', type=float, default=0.5,
                   help='Coarse downscale factor (0.5 → 18×23 from 35×46)')
    
    # --- Training ---
    g = parser.add_argument_group('Training')
    g.add_argument('--epochs', type=int, default=50)
    g.add_argument('--batch_size', type=int, default=1,
                   help='Batch size (1 recommended due to rendering bottleneck)')
    g.add_argument('--lr', type=float, default=2e-4)
    g.add_argument('--weight_decay', type=float, default=1e-4,
                   help='AdamW weight decay (1e-4 for regularization)')
    g.add_argument('--gamma', type=float, default=0.8,
                   help='Iteration loss weight decay (later iters get higher weight)')
    g.add_argument('--trans_weight', type=float, default=10.0,
                   help='Translation loss weight relative to rotation')
    g.add_argument('--flow_loss_weight', type=float, default=1.0,
                   help='Weight for linearized flow supervision (0 = disabled)')
    g.add_argument('--grad_accum', type=int, default=4,
                   help='Gradient accumulation steps (effective batch = batch_size × grad_accum)')
    g.add_argument('--noise_rot_deg', type=float, default=10.0,
                   help='Per-axis rotation noise σ (actual mean ≈ 1.6×)')
    g.add_argument('--curriculum', type=str, default=None,
                   help='Curriculum noise schedule: "epoch:noise_deg,..." '
                        'e.g. "0:5,50:10,100:15" → epoch 0-49 用5°, 50-99 用10°, 100+ 用15°')
    g.add_argument('--num_workers', type=int, default=2)
    g.add_argument('--seed', type=int, default=42)
    
    # --- Logging ---
    g = parser.add_argument_group('Logging')
    g.add_argument('--log_every', type=int, default=50,
                   help='Log every N batches')
    g.add_argument('--val_every', type=int, default=5,
                   help='Validate every N epochs')
    g.add_argument('--save_every', type=int, default=10,
                   help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train(args)
