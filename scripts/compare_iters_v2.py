#!/usr/bin/env python3
"""Quick 100-frame: 40iter+ES and 60iter+ES only"""
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

print("Loading...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

torch.manual_seed(42)
np.random.seed(42)

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

print("Caching 100 samples...")
samples = []
init_errors = []
for i in range(100):
    s = dataset[i]
    sample = {
        'query_feats': {k: v.unsqueeze(0).to(device) for k, v in s['query_feats'].items()},
        'pose_gt': s['pose_gt'].unsqueeze(0).to(device),
        'initial_pose': s['initial_pose'].unsqueeze(0).to(device),
        'depth': s['depth'].unsqueeze(0).to(device),
    }
    samples.append(sample)
    err = rotation_geodesic_loss(
        sample['initial_pose'][:, :3, :3], sample['pose_gt'][:, :3, :3]
    ).item() * 180 / np.pi
    init_errors.append(err)

init_arr = np.array(init_errors)
print(f"Initial: median={np.median(init_arr):.2f} mean={np.mean(init_arr):.2f} "
      f"max={np.max(init_arr):.2f} >10deg={np.sum(init_arr>10)}")

# Warmup
_ = renderer.render_scale('fine_dino', samples[0]['initial_pose'][0])
torch.cuda.synchronize()

configs = [
    {"name": "40iter_noES",  "max_iters": 40, "rel_thresh": 0.0,   "patience": 999},
    {"name": "40iter+ES",    "max_iters": 40, "rel_thresh": 0.005, "patience": 3},
    {"name": "60iter+ES",    "max_iters": 60, "rel_thresh": 0.005, "patience": 3},
]

for cfg in configs:
    aligner = FeaturemetricAligner(
        renderer=renderer, intrinsics=INTRINSICS,
        scale_names=['fine_dino'], damping=1e-2,
        max_iters=cfg['max_iters'],
        use_rendered_depth=False,
        rel_convergence_thresh=cfg['rel_thresh'],
        rel_convergence_patience=cfg['patience'],
    )

    errors = []
    times_list = []
    n_iters_list = []

    for i, s in enumerate(samples):
        torch.cuda.synchronize()
        t0 = time.time()
        result = aligner.align_fast(
            query_feats=s['query_feats'],
            initial_pose=s['initial_pose'],
            depth_for_jac=s['depth'],
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times_list.append(elapsed)
        n_iters_list.append(result['num_iters'])

        eval_pose = result.get('best_pose', result['final_pose'])
        err = rotation_geodesic_loss(
            eval_pose[:, :3, :3], s['pose_gt'][:, :3, :3]
        ).item() * 180 / np.pi
        errors.append(err)
        
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/100] ...", flush=True)

    errors = np.array(errors)
    positive = np.sum(errors < init_arr)
    improved = errors < init_arr

    print(f"\n{'='*60}")
    print(f"Config: {cfg['name']}")
    print(f"{'='*60}")
    print(f"  Median: {np.median(errors):.3f}°  Mean: {np.mean(errors):.3f}°")
    print(f"  <1°: {np.mean(errors<1)*100:.1f}%  <2°: {np.mean(errors<2)*100:.1f}%  "
          f"<5°: {np.mean(errors<5)*100:.1f}%")
    print(f"  Positive: {positive}%  Time: {np.mean(times_list):.2f}s ± {np.std(times_list):.2f}s")
    print(f"  Avg iters: {np.mean(n_iters_list):.1f}  Median iters: {np.median(n_iters_list):.0f}")
    worst5 = np.argsort(errors)[-5:][::-1]
    for idx in worst5:
        print(f"    worst: frame {idx}: {init_arr[idx]:.1f}° → {errors[idx]:.2f}°")

print("\nDone!")
