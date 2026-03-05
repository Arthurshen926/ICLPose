#!/usr/bin/env python3
"""Test 3DGS renderer loading and single batch rendering."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

t0 = time.time()
print("Loading renderer...")
from modules.multiscale_renderer import MultiScaleRenderer

renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'coarse': 'output/feature_3dgs/room_0_raw/coarse/best_model.pth',
        'mid': 'output/feature_3dgs/room_0_raw/mid/best_model.pth',
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device='cuda',
)
print(f"Loaded in {time.time()-t0:.1f}s")
print(f"Scale info: {renderer.get_scale_info()}")

# Check GPU memory
mem = torch.cuda.memory_allocated() / 1024**3
print(f"GPU memory used: {mem:.2f} GB")

# Load poses
traj = np.loadtxt('dataset/room_0/Sequence_1/traj_w_c.txt').reshape(-1, 4, 4)
c2w = torch.from_numpy(traj[0].astype(np.float32))
R, t = c2w[:3, :3], c2w[:3, 3]
w2c = torch.eye(4)
w2c[:3, :3] = R.T
w2c[:3, 3] = -R.T @ t
poses = w2c.unsqueeze(0).cuda()

print(f"\nRendering batch (B=1)...")
t1 = time.time()
result = renderer.render_batch(
    poses, scales=['coarse', 'mid', 'fine_sd', 'fine_dino'],
    return_depth=True)
print(f"Rendered in {time.time()-t1:.2f}s")

for k, v in result.items():
    print(f"  {k}: {v.shape}")

# Render B=4
poses4 = poses.expand(4, -1, -1).contiguous()
t2 = time.time()
result4 = renderer.render_batch(
    poses4, scales=['coarse', 'mid', 'fine_sd', 'fine_dino'],
    return_depth=True)
print(f"\nBatch B=4 rendered in {time.time()-t2:.2f}s")
for k, v in result4.items():
    print(f"  {k}: {v.shape}")

mem2 = torch.cuda.memory_allocated() / 1024**3
print(f"\nGPU memory after rendering: {mem2:.2f} GB")
print("\n✓ Renderer test PASSED!")
