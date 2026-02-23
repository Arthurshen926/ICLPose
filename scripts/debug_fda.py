#!/usr/bin/env python3
"""Quick debug test for specific frames"""
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

import torch, time, numpy as np
from modules.featuremetric import FeaturemetricAligner
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3
from losses.sequence_loss import rotation_geodesic_loss

device = torch.device('cuda')
INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

print('Loading...', flush=True)
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    }, device=device)

aligner = FeaturemetricAligner(renderer=renderer, intrinsics=INTRINSICS,
    scale_names=['fine_dino'], damping=1e-2, max_iters=6,
    use_rendered_depth=False)  # single scale + GT depth for speed

torch.manual_seed(42)
dataset = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    frame_indices=list(range(900)),
    scale_names=['fine_sd', 'fine_dino'],
    noise_rot_deg=2.0, noise_trans_m=0.04, is_train=True, depth_resize=(35, 46))

print('Testing 20 frames with timing...', flush=True)
for idx in range(20):
    sample = dataset[idx]
    q = {k: v.unsqueeze(0).to(device) for k, v in sample['query_feats'].items()}
    P_gt = sample['pose_gt'].unsqueeze(0).to(device)
    P_init = sample['initial_pose'].unsqueeze(0).to(device)
    depth = sample['depth'].unsqueeze(0).to(device)

    init_rot = rotation_geodesic_loss(P_init[:,:3,:3], P_gt[:,:3,:3]).item()*180/np.pi

    t0 = time.time()
    result = aligner.align_fast(
        query_feats=q, initial_pose=P_init,
        depth_for_jac=depth, verbose=(idx == 4))
    elapsed = time.time() - t0

    bp = result['best_pose']
    final_rot = rotation_geodesic_loss(bp[:,:3,:3], P_gt[:,:3,:3]).item()*180/np.pi
    imp = init_rot - final_rot
    print(f'Frame {idx:3d}: {init_rot:.2f}° -> {final_rot:.2f}° '
          f'(Δ={imp:+.2f}°, {result["num_iters"]} iters, {elapsed:.1f}s)',
          flush=True)

print('Done!', flush=True)
