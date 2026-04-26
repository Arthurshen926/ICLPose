#!/usr/bin/env python3
"""
评估 CorrPoseNet 在不同推理迭代次数 K 下的性能。

这是一个零成本实验: 不需要重新训练，直接加载已有模型，
用更多迭代次数 (K=3,5,7,10,15) 在验证集上评估。

原理: CorrPoseNet 的 ConvGRU 是循环网络，可以在推理时运行更多次迭代。
更多迭代 → 更多 render-and-compare 机会 → 可能收敛到更小误差。

使用方法:
    # 验证集 (Seq1 frame 810-899)
    python scripts/eval_corrpose_iters.py --checkpoint output/corr_pose/exp007_curriculum_K5/best_model.pth
    
    # 测试集 (Seq2 全部 900 帧)
    python scripts/eval_corrpose_iters.py --checkpoint output/corr_pose/exp007_curriculum_K5/best_model.pth --test_seq2
    
    # 自定义帧范围
    python scripts/eval_corrpose_iters.py --checkpoint ... --frame_start 0 --frame_end 900 --feature_dir ... --traj_path ...
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ic_models.corr_pose_net import CorrPoseNet
from data.dataset_v3 import PoseDatasetV3
from modules.multiscale_renderer import MultiScaleRenderer

# ---- Defaults (Replica room_0) ----
DEFAULT_PLY = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
DEFAULT_FEATURE_MODEL = 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth'
DEFAULT_FEATURE_DIR = 'output/features_multiscale/room_0'
DEFAULT_TRAJ = 'dataset/room_0/Sequence_1/traj_w_c.txt'
DEFAULT_DEPTH_DIR = 'dataset/room_0/Sequence_1/depth'

# Sequence 2 (test set) paths
SEQ2_FEATURE_DIR = 'output/features_multiscale/room_0_seq2'
SEQ2_TRAJ = 'dataset/room_0/Sequence_2/traj_w_c.txt'
SEQ2_DEPTH_DIR = 'dataset/room_0/Sequence_2/depth'

INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}
FEAT_DIM = {'fine_dino': 768, 'fine_sd': 640}


def compute_pose_error(pred, gt):
    """计算位姿误差: 旋转角 (度) + 平移 (米)"""
    R_pred = pred[:, :3, :3]
    R_gt = gt[:, :3, :3]
    R_rel = R_pred @ R_gt.transpose(-1, -2)
    cos_angle = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_angle = cos_angle.clamp(-1, 1)
    rot_error = torch.acos(cos_angle) * 180 / np.pi
    
    t_pred = pred[:, :3, 3]
    t_gt = gt[:, :3, 3]
    trans_error = (t_pred - t_gt).norm(dim=-1)
    return rot_error, trans_error


@torch.no_grad()
def evaluate_iters(net, val_dataset, renderer, device, scale, num_iters, seed=12345):
    """评估给定迭代次数的性能，返回逐帧结果"""
    net.eval()
    
    # 固定随机种子，确保不同 K 使用相同的噪声
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    rot_inits, rot_finals, trans_finals = [], [], []
    iter_rot_errors = {k: [] for k in range(num_iters + 1)}  # 每个迭代的误差
    
    t_start = time.time()
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        query = sample['query_feats'][scale].unsqueeze(0).to(device)
        pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
        initial = sample['initial_pose'].unsqueeze(0).to(device)
        depth = sample['depth'].unsqueeze(0).to(device) if 'depth' in sample else None
        
        results = net(
            query_feats=query,
            initial_pose=initial,
            depth=depth,
            intrinsics=INTRINSICS,
            renderer=renderer,
            scale_name=scale,
            num_iters=num_iters,  # 覆盖默认迭代次数
        )
        
        init_rot, _ = compute_pose_error(initial, pose_gt)
        rot_inits.append(init_rot.item())
        
        # 记录每个迭代的误差
        for k in range(num_iters + 1):
            rot_k, trans_k = compute_pose_error(results['poses'][k], pose_gt)
            iter_rot_errors[k].append(rot_k.item())
        
        final_rot, final_trans = compute_pose_error(results['poses'][-1], pose_gt)
        rot_finals.append(final_rot.item())
        trans_finals.append(final_trans.item())
    
    elapsed = time.time() - t_start
    
    rot_finals_arr = np.array(rot_finals)
    return {
        'num_iters': num_iters,
        'rot_init_mean': np.mean(rot_inits),
        'rot_median': np.median(rot_finals_arr),
        'rot_mean': np.mean(rot_finals_arr),
        'trans_median': np.median(trans_finals),
        'trans_mean': np.mean(trans_finals),
        'pct_1deg': (rot_finals_arr < 1.0).mean() * 100,
        'pct_5deg': (rot_finals_arr < 5.0).mean() * 100,
        'pct_10deg': (rot_finals_arr < 10.0).mean() * 100,
        'time_s': elapsed,
        'per_frame_s': elapsed / len(val_dataset),
        'rot_finals': rot_finals,
        'iter_convergence': {k: np.median(v) for k, v in iter_rot_errors.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CorrPoseNet with different iteration counts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--iters', nargs='+', type=int, default=[3, 5, 7, 10, 15],
                        help='Iteration counts to test')
    parser.add_argument('--ply_path', default=DEFAULT_PLY)
    parser.add_argument('--feature_model', default=DEFAULT_FEATURE_MODEL)
    parser.add_argument('--feature_dir', default=DEFAULT_FEATURE_DIR)
    parser.add_argument('--traj_path', default=DEFAULT_TRAJ)
    parser.add_argument('--depth_dir', default=DEFAULT_DEPTH_DIR)
    parser.add_argument('--scale', default='fine_dino')
    parser.add_argument('--noise_rot_deg', type=float, default=15.0,
                        help='Validation noise level (should match training)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for reproducible noise')
    parser.add_argument('--frame_start', type=int, default=None,
                        help='Start frame index (default: 810 for val, 0 for test_seq2)')
    parser.add_argument('--frame_end', type=int, default=None,
                        help='End frame index (default: 900 for val, 900 for test_seq2)')
    parser.add_argument('--test_seq2', action='store_true',
                        help='Evaluate on Sequence_2 (test set, 900 frames)')
    args = parser.parse_args()
    
    # --test_seq2 覆盖路径和帧范围
    if args.test_seq2:
        args.feature_dir = SEQ2_FEATURE_DIR
        args.traj_path = SEQ2_TRAJ
        args.depth_dir = SEQ2_DEPTH_DIR
        if args.frame_start is None:
            args.frame_start = 0
        if args.frame_end is None:
            args.frame_end = 900
    else:
        if args.frame_start is None:
            args.frame_start = 810
        if args.frame_end is None:
            args.frame_end = 900
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get('args', {})
    
    # 从checkpoint恢复网络参数
    feat_dim = FEAT_DIM.get(args.scale, 768)
    corr_radius = ckpt_args.get('corr_radius', 4)
    enc_dim = ckpt_args.get('enc_dim', 128)
    hidden_dim = ckpt_args.get('hidden_dim', 128)
    damping = ckpt_args.get('damping', 1e-3)
    
    net = CorrPoseNet(
        feat_dim=feat_dim,
        enc_dim=enc_dim,
        hidden_dim=hidden_dim,
        corr_radius=corr_radius,
        num_iters=3,  # 默认，会被覆盖
        damping=damping,
    ).to(device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(f"  Model loaded: {net.num_parameters():,} params, corr_radius={corr_radius}")
    print(f"  Trained val_rot_median={ckpt.get('val_rot_median', 'N/A')}°")
    
    # --- Renderer ---
    print("Loading renderer...")
    renderer = MultiScaleRenderer(
        ply_path=args.ply_path,
        scale_model_paths={args.scale: args.feature_model},
        device=device,
        img_height=480, img_width=640,
        fx=320.0, fy=320.0, cx=319.5, cy=239.5,
    )
    
    # --- Dataset ---
    noise_trans = args.noise_rot_deg / 50.0
    frame_indices = list(range(args.frame_start, args.frame_end))
    split_name = 'TEST (Seq2)' if args.test_seq2 else f'VAL (Seq1 [{args.frame_start}:{args.frame_end}])'
    
    eval_dataset = PoseDatasetV3(
        feature_base_dir=args.feature_dir,
        traj_path=args.traj_path,
        depth_dir=args.depth_dir,
        frame_indices=frame_indices,
        scale_names=[args.scale],
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_m=noise_trans,
        is_train=True,  # True = 使用噪声扰动
        depth_resize=(35, 46),
    )
    
    # --- Evaluate ---
    print(f"\n{'='*80}")
    print(f"[{split_name}] Evaluating K = {args.iters} on {len(eval_dataset)} frames")
    print(f"Noise: rot={args.noise_rot_deg}°, trans={noise_trans:.3f}m, seed={args.seed}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for K in sorted(args.iters):
        print(f"--- K={K} iterations ---")
        result = evaluate_iters(
            net, eval_dataset, renderer, device, args.scale, K, seed=args.seed
        )
        all_results.append(result)
        
        print(f"  Median rot: {result['rot_median']:.2f}°  "
              f"Mean rot: {result['rot_mean']:.2f}°")
        print(f"  <1°: {result['pct_1deg']:.1f}%  "
              f"<5°: {result['pct_5deg']:.1f}%  "
              f"<10°: {result['pct_10deg']:.1f}%")
        print(f"  Trans median: {result['trans_median']:.4f}m")
        print(f"  Time: {result['time_s']:.1f}s ({result['per_frame_s']:.2f}s/frame)")
        
        # 逐迭代收敛曲线
        conv = result['iter_convergence']
        conv_str = " → ".join(f"{conv[k]:.2f}°" for k in sorted(conv.keys()))
        print(f"  Convergence: {conv_str}")
        print()
    
    # --- Summary table ---
    print(f"\n{'='*80}")
    print(f"{'K':>4s} | {'Median°':>8s} | {'Mean°':>8s} | {'<1°%':>6s} | {'<5°%':>6s} | {'<10°%':>6s} | {'Trans(m)':>8s} | {'Time(s)':>8s}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for r in all_results:
        print(f"{r['num_iters']:4d} | {r['rot_median']:8.2f} | {r['rot_mean']:8.2f} | "
              f"{r['pct_1deg']:6.1f} | {r['pct_5deg']:6.1f} | {r['pct_10deg']:6.1f} | "
              f"{r['trans_median']:8.4f} | {r['time_s']:8.1f}")
    print(f"{'='*80}")
    
    # 找到最优K
    best = min(all_results, key=lambda r: r['rot_median'])
    print(f"\n✓ Best: K={best['num_iters']}, median={best['rot_median']:.2f}°, "
          f"<5°={best['pct_5deg']:.1f}%")


if __name__ == '__main__':
    main()
