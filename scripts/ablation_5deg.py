#!/usr/bin/env python3
"""
快速消融实验：在 5° 噪声下对比多种 FDA 配置
测试 7 个样本 × 多种配置
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

print("Loading renderer (both scales)...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

# 5° 噪声数据集
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

torch.manual_seed(42)
test_indices = [0, 50, 100, 200, 400, 600, 800]

# 预加载所有样本
print(f"Loading {len(test_indices)} samples...")
samples = []
for idx in test_indices:
    s = dataset[idx]
    samples.append({
        'query_feats': {k: v.unsqueeze(0).to(device) for k, v in s['query_feats'].items()},
        'pose_gt': s['pose_gt'].unsqueeze(0).to(device),
        'initial_pose': s['initial_pose'].unsqueeze(0).to(device),
        'depth': s['depth'].unsqueeze(0).to(device),
        'idx': idx,
    })

# 计算初始误差
init_errors = []
for s in samples:
    err = rotation_geodesic_loss(
        s['initial_pose'][:, :3, :3], s['pose_gt'][:, :3, :3]
    ).item() * 180 / np.pi
    init_errors.append(err)
print(f"Initial errors: mean={np.mean(init_errors):.2f}°, "
      f"median={np.median(init_errors):.2f}°, "
      f"range=[{np.min(init_errors):.2f}°, {np.max(init_errors):.2f}°]")


def run_config(name, scales, max_iters, mode='fast', damping=1e-2, verbose_idx=None):
    """测试一个配置"""
    aligner = FeaturemetricAligner(
        renderer=renderer,
        intrinsics=INTRINSICS,
        scale_names=scales,
        damping=damping,
        max_iters=max_iters,
        use_rendered_depth=False,
    )

    errors = []
    improvements = []
    timings = []

    for s in samples:
        t0 = time.time()
        align_fn = aligner.align_fast if mode == 'fast' else aligner.align
        result = align_fn(
            query_feats=s['query_feats'],
            initial_pose=s['initial_pose'],
            depth_for_jac=s['depth'],
            verbose=(s['idx'] == verbose_idx),
        )
        elapsed = time.time() - t0

        eval_pose = result.get('best_pose', result['final_pose'])
        final_rot = rotation_geodesic_loss(
            eval_pose[:, :3, :3], s['pose_gt'][:, :3, :3]
        ).item() * 180 / np.pi
        init_rot = rotation_geodesic_loss(
            s['initial_pose'][:, :3, :3], s['pose_gt'][:, :3, :3]
        ).item() * 180 / np.pi

        errors.append(final_rot)
        improvements.append(init_rot - final_rot)
        timings.append(elapsed)

    med_err = np.median(errors)
    mean_err = np.mean(errors)
    avg_imp = np.mean(improvements)
    pos_rate = np.mean([1 for i in improvements if i > 0]) * 100
    lt1 = np.mean([1 for e in errors if e < 1.0]) * 100
    lt5 = np.mean([1 for e in errors if e < 5.0]) * 100
    avg_time = np.mean(timings)

    print(f"  {name:40s} | med={med_err:5.2f}° mean={mean_err:5.2f}° "
          f"<1°={lt1:4.0f}% <5°={lt5:4.0f}% pos={pos_rate:4.0f}% "
          f"t={avg_time:.1f}s")

    return {'name': name, 'median': med_err, 'mean': mean_err,
            'lt1': lt1, 'lt5': lt5, 'pos': pos_rate, 'time': avg_time,
            'errors': errors, 'improvements': improvements}


print(f"\n{'='*90}")
print(f"Ablation Study: 5° noise, {len(test_indices)} samples")
print(f"{'='*90}")

results = []

# --- Baseline: current config ---
print("\n--- Baseline ---")
results.append(run_config(
    "dino_8iter_fast", ['fine_dino'], 8, 'fast'))

# --- More iterations ---
print("\n--- Iteration count ---")
for iters in [12, 15, 20, 30]:
    results.append(run_config(
        f"dino_{iters}iter_fast", ['fine_dino'], iters, 'fast'))

# --- Dual scale ---
print("\n--- Dual scale (fine_sd + fine_dino) ---")
for iters in [8, 12, 15, 20]:
    results.append(run_config(
        f"sd+dino_{iters}iter_fast", ['fine_sd', 'fine_dino'], iters, 'fast'))

# --- LM mode ---
print("\n--- LM mode (with step rejection) ---")
results.append(run_config(
    "dino_20iter_LM", ['fine_dino'], 20, 'lm'))
results.append(run_config(
    "sd+dino_20iter_LM", ['fine_sd', 'fine_dino'], 20, 'lm'))

# --- Damping sensitivity ---
print("\n--- Damping sensitivity (dino, 15iter) ---")
for damp in [1e-3, 1e-2, 1e-1, 1.0]:
    results.append(run_config(
        f"dino_15iter_damp{damp:.0e}", ['fine_dino'], 15, 'fast', damping=damp))

# --- Verbose on one sample for best config ---
print("\n--- Verbose: best dual-scale config (sample 0) ---")
run_config("sd+dino_20iter_fast_verbose", ['fine_sd', 'fine_dino'], 20, 'fast',
           verbose_idx=0)

# --- Summary ---
print(f"\n{'='*90}")
print(f"{'Config':40s} | {'med':>5s}  {'mean':>5s}  {'<1°':>4s}  {'<5°':>4s}  {'pos':>4s}  {'time':>5s}")
print(f"{'-'*90}")
for r in results:
    print(f"  {r['name']:40s} | {r['median']:5.2f}° {r['mean']:5.2f}° "
          f"{r['lt1']:4.0f}% {r['lt5']:4.0f}% {r['pos']:4.0f}% {r['time']:5.1f}s")

print("\nDone!")
