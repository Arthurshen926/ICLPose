#!/usr/bin/env python3
"""
Quick benchmark: measure per-frame time with optimized align_fast
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

print("Loading renderer...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

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
test_indices = [0, 50, 100, 200, 400]

# 预加载
samples = []
for idx in test_indices:
    s = dataset[idx]
    samples.append({
        'query_feats': {k: v.unsqueeze(0).to(device) for k, v in s['query_feats'].items()},
        'pose_gt': s['pose_gt'].unsqueeze(0).to(device),
        'initial_pose': s['initial_pose'].unsqueeze(0).to(device),
        'depth': s['depth'].unsqueeze(0).to(device),
    })

# Warmup
aligner = FeaturemetricAligner(
    renderer=renderer, intrinsics=INTRINSICS,
    scale_names=['fine_dino'], damping=1e-2, max_iters=20,
    use_rendered_depth=False,
)
_ = aligner.align_fast(
    query_feats=samples[0]['query_feats'],
    initial_pose=samples[0]['initial_pose'],
    depth_for_jac=samples[0]['depth'],
)
torch.cuda.synchronize()

# Benchmark: dino 20 iter
print("\n=== Benchmark: dino 20iter ===")
for trial in range(2):
    times = []
    errors = []
    for s in samples:
        torch.cuda.synchronize()
        t0 = time.time()
        result = aligner.align_fast(
            query_feats=s['query_feats'],
            initial_pose=s['initial_pose'],
            depth_for_jac=s['depth'],
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)

        eval_pose = result.get('best_pose', result['final_pose'])
        err = rotation_geodesic_loss(
            eval_pose[:, :3, :3], s['pose_gt'][:, :3, :3]
        ).item() * 180 / np.pi
        errors.append(err)

    print(f"  Trial {trial+1}: {np.mean(times):.2f}s/frame ± {np.std(times):.2f}s  "
          f"median_err={np.median(errors):.2f}° mean_err={np.mean(errors):.2f}°")

# --- Also test verbose on sample 0 to see iteration timing ---
print("\n=== Verbose: sample 0, dino 20iter ===")
torch.cuda.synchronize()
t0 = time.time()
result = aligner.align_fast(
    query_feats=samples[0]['query_feats'],
    initial_pose=samples[0]['initial_pose'],
    depth_for_jac=samples[0]['depth'],
    verbose=True,
)
torch.cuda.synchronize()
print(f"Total: {time.time()-t0:.2f}s, {result['num_iters']} iters")

# --- Measure per-step timing ---
print("\n=== Per-step timing breakdown ===")
from modules.featuremetric import compute_image_jacobian, compute_spatial_gradient
import torch.nn.functional as F

s = samples[0]
pose = s['initial_pose'].clone()
depth = s['depth']
query_normed = {k: F.normalize(v, p=2, dim=1) for k, v in s['query_feats'].items() if k == 'fine_dino'}
Ju, Jv, valid = compute_image_jacobian(depth, INTRINSICS)
N = Ju.shape[1]
valid_f = valid.float()
B = 1

# Render timing
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    with torch.no_grad():
        rendered = renderer.render_scale('fine_dino', pose[0])
torch.cuda.synchronize()
print(f"  Render (5×):     {(time.time()-t0)/5*1000:.1f}ms")

# Normal eq timing (without .item())
feat = rendered['feature_map'].unsqueeze(0)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    D = 768
    residual = query_normed['fine_dino'] - feat
    grad_u, grad_v = compute_spatial_gradient(feat)
    gu = grad_u.reshape(B, D, N)
    gv = grad_v.reshape(B, D, N)
    res = residual.reshape(B, D, N)
    A = (gu * gu).sum(1) * valid_f
    B_ = (gv * gv).sum(1) * valid_f
    C = (gu * gv).sum(1) * valid_f
    Ru = (gu * res).sum(1) * valid_f
    Rv = (gv * res).sum(1) * valid_f
    JtJ = (torch.bmm((Ju * A.unsqueeze(-1)).transpose(1, 2), Ju) +
           torch.bmm((Ju * C.unsqueeze(-1)).transpose(1, 2), Jv) +
           torch.bmm((Jv * C.unsqueeze(-1)).transpose(1, 2), Ju) +
           torch.bmm((Jv * B_.unsqueeze(-1)).transpose(1, 2), Jv))
    JtR = -(torch.bmm(Ju.transpose(1, 2), Ru.unsqueeze(-1)) +
            torch.bmm(Jv.transpose(1, 2), Rv.unsqueeze(-1)))
    total_res = (res ** 2).sum()  # NO .item()!
torch.cuda.synchronize()
print(f"  Normal Eq (5×):  {(time.time()-t0)/5*1000:.1f}ms (no .item() sync)")

# Cholesky solve timing
eye6 = torch.eye(6, device=device).unsqueeze(0)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    A_mat = JtJ + 0.01 * eye6
    L = torch.linalg.cholesky(A_mat)
    delta = torch.cholesky_solve(JtR, L).squeeze(-1)
torch.cuda.synchronize()
print(f"  Cholesky (100×): {(time.time()-t0)/100*1000:.2f}ms")

# linalg.solve for comparison
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    delta2 = torch.linalg.solve(JtJ + 0.01*eye6, JtR).squeeze(-1)
torch.cuda.synchronize()
print(f"  Solve (100×):    {(time.time()-t0)/100*1000:.2f}ms")

print(f"\nGPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print("Done!")
