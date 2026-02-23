#!/usr/bin/env python3
"""
测试 Featuremetric Direct Alignment 在单样本上的效果
"""
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

import torch
import numpy as np
from modules.featuremetric import FeaturemetricAligner
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3
from losses.sequence_loss import rotation_geodesic_loss
import torch.nn.functional as F

device = torch.device('cuda')

# 渲染分辨率下的内参
INTRINSICS_FLOW = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

print("Loading renderer...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

print("Loading dataset...")
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

aligner = FeaturemetricAligner(
    renderer=renderer,
    intrinsics=INTRINSICS_FLOW,
    scale_names=['fine_sd', 'fine_dino'],
    damping=1e-2,
    max_iters=30,
    use_rendered_depth=True,
)

# 测试多个样本
print("\n" + "="*70)
print("Featuremetric Direct Alignment Test")
print("="*70)

torch.manual_seed(42)
test_indices = [0, 50, 100, 200, 400, 600, 800]

for noise_deg in [2.0, 5.0, 10.0, 15.0]:
    dataset.noise_rot_deg = noise_deg
    dataset.noise_trans_m = noise_deg / 50.0  # 粗略比例

    improvements = []
    final_errors = []

    print(f"\n--- Noise: {noise_deg}° ---")
    for idx in test_indices:
        sample = dataset[idx]
        query_feats = {k: v.unsqueeze(0).to(device) for k, v in sample['query_feats'].items()}
        pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
        initial_pose = sample['initial_pose'].unsqueeze(0).to(device)
        depth = sample['depth'].unsqueeze(0).to(device)

        init_rot = rotation_geodesic_loss(
            initial_pose[:, :3, :3], pose_gt[:, :3, :3]
        ).item() * 180 / np.pi
        init_trans = torch.norm(initial_pose[:, :3, 3] - pose_gt[:, :3, 3]).item()

        result = aligner.align(
            query_feats=query_feats,
            initial_pose=initial_pose,
            depth_for_jac=depth,
            verbose=(idx == test_indices[0] and noise_deg == 5.0),
        )

        # 使用 best_pose（残差最低的位姿）而非 final_pose
        eval_pose = result.get('best_pose', result['final_pose'])
        final_rot = rotation_geodesic_loss(
            eval_pose[:, :3, :3], pose_gt[:, :3, :3]
        ).item() * 180 / np.pi
        final_trans = torch.norm(eval_pose[:, :3, 3] - pose_gt[:, :3, 3]).item()

        improvement = init_rot - final_rot
        improvements.append(improvement)
        final_errors.append(final_rot)

        if idx == test_indices[0]:
            print(f"  Sample {idx}: {init_rot:.2f}° → {final_rot:.2f}° "
                  f"(Δ={improvement:+.2f}°, {result['num_iters']} iters) "
                  f"t: {init_trans:.3f}→{final_trans:.3f}m")

    avg_imp = np.mean(improvements)
    avg_final = np.mean(final_errors)
    pos_rate = np.mean([1 for i in improvements if i > 0]) * 100
    print(f"  Avg: Δ={avg_imp:+.2f}° final={avg_final:.2f}° positive={pos_rate:.0f}%")

print("\nDone!")
