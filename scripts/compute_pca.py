#!/usr/bin/env python3
"""PCA compression analysis for DINO features"""
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

import torch
import numpy as np
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3

device = torch.device('cuda')
print("Loading renderer (only fine_dino)...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
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
    noise_rot_deg=2.0, noise_trans_m=0.04, is_train=True, depth_resize=(35, 46),
)

# Collect rendered features for PCA from 10 views
print("Collecting rendered features for PCA...")
feats_all = []
for i in range(0, 50, 5):
    s = dataset[i]
    p = s['initial_pose'].to(device)
    with torch.no_grad():
        r = renderer.render_scale('fine_dino', p)
    feats_all.append(r['feature_map'].reshape(768, -1).T)  # (1610, 768)

feats_cat = torch.cat(feats_all, dim=0)  # (K, 768)
print(f'PCA data: {feats_cat.shape}')

mean = feats_cat.mean(0, keepdim=True)
centered = feats_cat - mean
print("Computing SVD...")
U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
total_var = (S**2).sum()
for k in [32, 64, 128, 256, 384]:
    ev = (S[:k]**2).sum() / total_var
    print(f'  PCA-{k:3d}: explained variance = {ev*100:.2f}%')

# Check query features reconstruction
print("\nQuery feature reconstruction error:")
q_feats = []
for i in range(0, 50, 5):
    s = dataset[i]
    q_feats.append(s['query_feats']['fine_dino'].reshape(768, -1).T)
q_cat = torch.cat(q_feats, dim=0).to(device)
q_centered = q_cat - mean
for k in [64, 128, 256]:
    q_proj = q_centered @ Vh[:k].T
    q_recon = q_proj @ Vh[:k] + mean
    recon_error = ((q_cat - q_recon)**2).sum(-1).sqrt().mean()
    orig_norm = q_cat.norm(dim=-1).mean()
    print(f'  PCA-{k:3d}: error={recon_error:.4f} norm={orig_norm:.4f} ratio={recon_error/orig_norm*100:.2f}%')

# Save PCA basis for later use
pca_data = {
    'mean': mean.cpu(),
    'components': Vh.cpu(),  # (768, 768)
    'singular_values': S.cpu(),
}
torch.save(pca_data, 'output/feature_3dgs/room_0_raw/fine_dino/pca_basis.pth')
print("\nSaved PCA basis to output/feature_3dgs/room_0_raw/fine_dino/pca_basis.pth")

print(f"\nGPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
print("Done!")
