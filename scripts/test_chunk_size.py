#!/usr/bin/env python3
"""Quick test: chunk_size 32 vs 128 rendering speed on GPU 1"""
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

import torch
import time
import numpy as np

device = torch.device('cuda')

# Load one model only (fine_dino, 768d)
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
model = GaussianFeatureModel(feature_dim=768)
model.load_ply('dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply')
ckpt = torch.load('output/feature_3dgs/room_0_raw/fine_dino/best_model.pth', map_location='cpu')
model._loc_feature = torch.nn.Parameter(ckpt['loc_feature'])
model = model.to(device)
print(f"Model loaded: {model.feature_dim}d, {model.get_xyz.shape[0]} gaussians")

from feature_3dgs.feature_renderer import FeatureRenderer
from data.dataset_v3 import PoseDatasetV3

dataset = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    frame_indices=list(range(10)),
    scale_names=['fine_dino'],
    noise_rot_deg=5.0,
    noise_trans_m=0.1,
    is_train=True,
    depth_resize=(35, 46),
)

# Get a sample pose
sample = dataset[0]
pose = sample['initial_pose'].to(device)

# Warmup
for _ in range(3):
    _ = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=23.0, fy=23.3, cx=22.97, cy=17.47,
        img_height=35, img_width=46,
        max_channels_per_chunk=32,
    )
torch.cuda.synchronize()

# Test different chunk sizes
for chunk_size in [32, 64, 128, 256]:
    try:
        times = []
        for trial in range(10):
            torch.cuda.synchronize()
            t0 = time.time()
            result = FeatureRenderer.render_features(
                gaussian_model=model, viewmat=pose,
                fx=23.0, fy=23.3, cx=22.97, cy=17.47,
                img_height=35, img_width=46,
                max_channels_per_chunk=chunk_size,
            )
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        
        n_chunks = (768 + chunk_size - 1) // chunk_size
        print(f"chunk_size={chunk_size:3d}: {n_chunks:2d} chunks, "
              f"{np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")
    except Exception as e:
        print(f"chunk_size={chunk_size:3d}: FAILED - {e}")

print(f"\nGPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
