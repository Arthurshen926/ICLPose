#!/usr/bin/env python3
"""
Analyze convergence of hard frames:
- Why do some frames get stuck at >2° error?
- Is it local minima or insufficient iterations?
"""
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

import torch
import numpy as np
import time
from modules.featuremetric import FeaturemetricAligner
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3
from losses.sequence_loss import rotation_geodesic_loss

device = torch.device('cuda')
INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

print("Loading on GPU (CUDA_VISIBLE_DEVICES should be set to 1)...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

# Same seed as eval script to get same noise
dataset = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    frame_indices=list(range(900)),
    scale_names=['fine_sd', 'fine_dino'],
    noise_rot_deg=5.0,
    noise_trans_m=0.1,
    is_train=True,
    depth_resize=(35, 46),
)

# Test frames known to be "hard" (from eval output)
# frame 3: 2.37°, frame 4: 2.21°, frame 20: 2.67°, frame 30: 2.40°
# Also test an easy frame for comparison: frame 1: 0.73°
test_frames = [0, 2, 3, 19, 29, 39]

# Create aligner with 40 iterations to see if more iters help
aligner = FeaturemetricAligner(
    renderer=renderer,
    intrinsics=INTRINSICS,
    scale_names=['fine_dino'],
    damping=1e-2,
    max_iters=40,
    use_rendered_depth=False,
    rel_convergence_thresh=0.0,  # Disable early stopping to see full convergence
    rel_convergence_patience=999,
)

print("\n" + "="*80)
print("CONVERGENCE ANALYSIS: 5° noise, fine_dino, 40 iterations")
print("="*80)

for idx in test_frames:
    sample = dataset[idx]
    query_feats = {k: v.unsqueeze(0).to(device) for k, v in sample['query_feats'].items()}
    pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
    initial_pose = sample['initial_pose'].unsqueeze(0).to(device)
    depth = sample['depth'].unsqueeze(0).to(device)

    # Check initial error
    init_err = rotation_geodesic_loss(
        initial_pose[:, :3, :3], pose_gt[:, :3, :3]
    ).item() * 180 / np.pi

    print(f"\n--- Frame {idx+1} (0-indexed: {idx}), initial error: {init_err:.2f}° ---")

    result = aligner.align_fast(
        query_feats=query_feats,
        initial_pose=initial_pose,
        depth_for_jac=depth,
        verbose=True,
    )

    final_err = rotation_geodesic_loss(
        result['final_pose'][:, :3, :3], pose_gt[:, :3, :3]
    ).item() * 180 / np.pi
    best_err = rotation_geodesic_loss(
        result['best_pose'][:, :3, :3], pose_gt[:, :3, :3]
    ).item() * 180 / np.pi

    # Also check error at different iteration counts
    # Re-run with fewer iterations to see progression
    milestone_errors = {}
    for n_iters in [8, 12, 15, 20, 30, 40]:
        aligner_tmp = FeaturemetricAligner(
            renderer=renderer, intrinsics=INTRINSICS,
            scale_names=['fine_dino'], damping=1e-2,
            max_iters=n_iters, use_rendered_depth=False,
            rel_convergence_thresh=0.0, rel_convergence_patience=999,
        )
        r = aligner_tmp.align_fast(
            query_feats=query_feats,
            initial_pose=initial_pose,
            depth_for_jac=depth,
        )
        err = rotation_geodesic_loss(
            r['best_pose'][:, :3, :3], pose_gt[:, :3, :3]
        ).item() * 180 / np.pi
        milestone_errors[n_iters] = err

    print(f"  Final: {final_err:.2f}°  Best: {best_err:.2f}°  "
          f"Residual: {result['best_residual']:.4f}  Iters: {result['num_iters']}")
    print(f"  Error by iters: " + "  ".join(
        f"{n}iter={e:.2f}°" for n, e in milestone_errors.items()))

    # Check if the frame is in a "featureless" region
    # by looking at spatial gradient magnitude
    with torch.no_grad():
        rendered = renderer.render_scale('fine_dino', pose_gt[0])
        feat = rendered['feature_map'].unsqueeze(0)
        from modules.featuremetric import compute_spatial_gradient
        gu, gv = compute_spatial_gradient(feat)
        grad_mag = (gu.norm(dim=1) + gv.norm(dim=1)) / 2  # (1, H, W)
        print(f"  Feature gradient: mean={grad_mag.mean().item():.4f} "
              f"median={grad_mag.median().item():.4f} "
              f"max={grad_mag.max().item():.4f} "
              f"std={grad_mag.std().item():.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
