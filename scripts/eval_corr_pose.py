#!/usr/bin/env python3
"""
Evaluation script for CorrPoseNet (+ optional FDA refinement).

Usage:
    # CorrPoseNet only
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_corr_pose.py --model_path output/corr_pose_net/room_0/best_model.pth
    
    # CorrPoseNet → FDA pipeline
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_corr_pose.py --model_path ... --fda_refine --fda_iters 20

评估流程:
  1. CorrPoseNet: 初始位姿 → 粗对准 (处理大偏差 ~15-30° → ~3-5°)
  2. (可选) FDA: 粗对准结果 → 精修 (3-5° → <1°)

这是完整 pipeline 的设计:
  图像检索 (~30-45°) → CorrPoseNet (~3-5°) → FDA (<1°)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import argparse
import time

from ic_models.corr_pose_net import CorrPoseNet
from data.dataset_v3 import PoseDatasetV3
from modules.multiscale_renderer import MultiScaleRenderer
from modules.featuremetric import FeaturemetricAligner

# Replica room_0 defaults
DEFAULT_PLY = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
DEFAULT_FEATURE_MODEL = 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth'
DEFAULT_FEATURE_DIR = 'output/features_multiscale/room_0'
DEFAULT_TRAJ = 'dataset/room_0/Sequence_1/traj_w_c.txt'
DEFAULT_DEPTH_DIR = 'dataset/room_0/Sequence_1/depth'

INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

FEAT_DIM = {'fine_dino': 768, 'fine_sd': 640}


def compute_rotational_error(pred_pose, gt_pose):
    """Geodesic rotation error in degrees."""
    R_pred = pred_pose[:3, :3]
    R_gt = gt_pose[:3, :3]
    R_rel = R_pred @ R_gt.T
    cos_angle = (R_rel.trace() - 1) / 2
    cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_angle).item() * 180 / np.pi


def compute_translation_error(pred_pose, gt_pose):
    """Translation error in meters."""
    t_pred = pred_pose[:3, 3]
    t_gt = gt_pose[:3, 3]
    return (t_pred - t_gt).norm().item()


@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda')
    
    # Fix seed for reproducible noise
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # --- Load renderer ---
    print("Loading renderer...")
    renderer = MultiScaleRenderer(
        ply_path=args.ply_path,
        scale_model_paths={args.scale: args.feature_model},
        device=device,
        img_height=480, img_width=640,
        fx=320.0, fy=320.0, cx=319.5, cy=239.5,
    )
    
    # --- Load CorrPoseNet ---
    print(f"Loading model from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location=device)
    model_args = ckpt.get('args', {})
    
    feat_dim = FEAT_DIM.get(args.scale, 768)
    net = CorrPoseNet(
        feat_dim=feat_dim,
        enc_dim=model_args.get('enc_dim', 128),
        hidden_dim=model_args.get('hidden_dim', 128),
        corr_radius=model_args.get('corr_radius', 4),
        num_iters=model_args.get('num_iters', 3),
        damping=model_args.get('damping', 1e-3),
    ).to(device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(net)
    
    # --- Optional FDA refiner ---
    fda_aligner = None
    if args.fda_refine:
        fda_aligner = FeaturemetricAligner(
            renderer=renderer,
            scales=[args.scale],
            intrinsics=INTRINSICS,
            max_iters=args.fda_iters,
            damping=1e-2,
            rel_convergence_thresh=args.fda_convergence_thresh,
            rel_convergence_patience=3,
        )
        print(f"FDA refiner enabled: {args.fda_iters} iters")
    
    # --- Evaluate per noise level ---
    for noise_deg in args.noise_levels:
        noise_trans = noise_deg / 50.0
        
        dataset = PoseDatasetV3(
            feature_base_dir=args.feature_dir,
            traj_path=args.traj_path,
            depth_dir=args.depth_dir,
            frame_indices=list(range(args.num_frames)),
            scale_names=[args.scale],
            noise_rot_deg=noise_deg,
            noise_trans_m=noise_trans,
            is_train=True,
            depth_resize=(35, 46),
        )
        
        # Re-seed for consistent noise
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        init_rots, corr_rots, final_rots = [], [], []
        init_trans, corr_trans, final_trans_list = [], [], []
        timings = []
        
        print(f"\n{'='*60}")
        print(f"Evaluating: Noise {noise_deg}° / {noise_trans:.3f}m  |  "
              f"{len(dataset)} frames  |  "
              f"CorrPoseNet {model_args.get('num_iters', 3)} iters"
              + (f" + FDA {args.fda_iters} iters" if args.fda_refine else ""))
        print(f"{'='*60}")
        
        for i in range(len(dataset)):
            sample = dataset[i]
            query = sample['query_feats'][args.scale].unsqueeze(0).to(device)
            pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
            initial = sample['initial_pose'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device) if 'depth' in sample else None
            
            t0 = time.time()
            
            # --- Stage 1: CorrPoseNet ---
            results = net(
                query_feats=query,
                initial_pose=initial,
                depth=depth,
                intrinsics=INTRINSICS,
                renderer=renderer,
                scale_name=args.scale,
            )
            corr_pose = results['poses'][-1]
            
            # --- Stage 2 (optional): FDA refine ---
            final_pose = corr_pose
            if fda_aligner is not None:
                fda_result = fda_aligner.align_fast(
                    query_feats={args.scale: query},
                    initial_pose=corr_pose,
                    depth_for_jac=depth,
                )
                final_pose = fda_result['best_pose']
            
            elapsed = time.time() - t0
            
            # --- Metrics ---
            init_rot = compute_rotational_error(initial[0], pose_gt[0])
            corr_rot = compute_rotational_error(corr_pose[0], pose_gt[0])
            final_rot = compute_rotational_error(final_pose[0], pose_gt[0])
            final_t = compute_translation_error(final_pose[0], pose_gt[0])
            
            init_rots.append(init_rot)
            corr_rots.append(corr_rot)
            final_rots.append(final_rot)
            final_trans_list.append(final_t)
            timings.append(elapsed)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                print(f"  [{i+1:4d}/{len(dataset)}] "
                      f"init={init_rot:.1f}° → corr={corr_rot:.1f}° → "
                      f"final={final_rot:.2f}° "
                      f"({elapsed:.2f}s)")
        
        # --- Summary ---
        init_arr = np.array(init_rots)
        corr_arr = np.array(corr_rots)
        final_arr = np.array(final_rots)
        trans_arr = np.array(final_trans_list)
        time_arr = np.array(timings)
        improved = (final_arr < init_arr).mean() * 100
        
        print(f"\n--- Results: Noise {noise_deg}° ---")
        print(f"  Rotation (°): init={np.median(init_arr):.2f} → "
              f"corr={np.median(corr_arr):.2f} → "
              f"final={np.median(final_arr):.2f}")
        print(f"  Rotation mean: init={np.mean(init_arr):.2f} → "
              f"corr={np.mean(corr_arr):.2f} → "
              f"final={np.mean(final_arr):.2f}")
        print(f"  Translation (m): final={np.median(trans_arr):.4f}")
        print(f"  <1° rate:   {(final_arr < 1).mean()*100:.1f}%  "
              f"(init: {(init_arr < 1).mean()*100:.1f}%)")
        print(f"  <2° rate:   {(final_arr < 2).mean()*100:.1f}%")
        print(f"  <5° rate:   {(final_arr < 5).mean()*100:.1f}%")
        print(f"  Positive rate: {improved:.1f}%")
        print(f"  Time: {np.mean(time_arr):.2f}s/frame")
        
        # Train/Val split
        train_mask = np.arange(len(final_arr)) < 810
        val_mask = ~train_mask
        if val_mask.any():
            print(f"  [Train] rot={np.median(final_arr[train_mask]):.2f}° "
                  f"<1°={((final_arr[train_mask]<1).mean()*100):.1f}%")
            print(f"  [Val  ] rot={np.median(final_arr[val_mask]):.2f}° "
                  f"<1°={((final_arr[val_mask]<1).mean()*100):.1f}%")
    
    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate CorrPoseNet (+ optional FDA)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Paths
    parser.add_argument('--model_path', required=True,
                        help='Path to trained CorrPoseNet checkpoint')
    parser.add_argument('--ply_path', default=DEFAULT_PLY)
    parser.add_argument('--feature_model', default=DEFAULT_FEATURE_MODEL)
    parser.add_argument('--feature_dir', default=DEFAULT_FEATURE_DIR)
    parser.add_argument('--traj_path', default=DEFAULT_TRAJ)
    parser.add_argument('--depth_dir', default=DEFAULT_DEPTH_DIR)
    
    # Eval config
    parser.add_argument('--scale', default='fine_dino')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[5.0, 10.0, 15.0])
    parser.add_argument('--num_frames', type=int, default=900)
    parser.add_argument('--seed', type=int, default=42)
    
    # FDA refinement
    parser.add_argument('--fda_refine', action='store_true',
                        help='Apply FDA refinement after CorrPoseNet')
    parser.add_argument('--fda_iters', type=int, default=20)
    parser.add_argument('--fda_convergence_thresh', type=float, default=0.005)
    
    args = parser.parse_args()
    evaluate(args)
