#!/usr/bin/env python3
"""Quick diagnostic: verify render-and-compare signal quality"""
import torch
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.insert(0, '/home/yons/Projects/ICLPose')

from data.dataset_v3 import PoseDatasetV3, collate_v3, perturb_pose
from modules.multiscale_renderer import MultiScaleRenderer
from losses.sequence_loss import rotation_geodesic_loss

device = torch.device('cuda:0')

# Load renderer
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

# Load one sample
ds = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir=None,
    frame_indices=[0,1,2,3,4],
    scale_names=['fine_sd', 'fine_dino'],
    noise_rot_deg=5.0,
    noise_trans_m=0.1,
    is_train=True,
)

print("=" * 60)
print("Render-and-Compare Signal Diagnostic")
print("=" * 60)

for sample_idx in range(5):
    sample = ds[sample_idx]
    pose_gt = sample['pose_gt'].to(device)
    query_feats = {k: v.unsqueeze(0).to(device) for k, v in sample['query_feats'].items()}
    
    # L2 normalize query feats
    query_norm = {k: F.normalize(v, p=2, dim=1) for k, v in query_feats.items()}
    
    # Render at GT pose
    rendered_gt = {}
    for name in ['fine_sd', 'fine_dino']:
        r = renderer.render_scale(name, pose_gt)
        rendered_gt[name] = r['feature_map'].unsqueeze(0)
    
    # Compute L2 distances at different perturbation levels
    print(f"\nSample {sample_idx} (frame {ds.frame_indices[sample_idx]}):")
    for noise_deg in [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
        for noise_m in [noise_deg * 0.02]:  # scale trans with rot
            if noise_deg == 0.0:
                perturbed = pose_gt.clone()
            else:
                perturbed = perturb_pose(pose_gt.cpu(), noise_deg, noise_m).to(device)
            
            rot_err = rotation_geodesic_loss(
                perturbed[:3,:3].unsqueeze(0), pose_gt[:3,:3].unsqueeze(0)
            ).item() * 180 / np.pi
            
            # Render at perturbed pose
            dists = {}
            for name in ['fine_sd', 'fine_dino']:
                r = renderer.render_scale(name, perturbed)
                rendered_p = r['feature_map'].unsqueeze(0)
                
                # Distance to query features
                diff_to_query = (query_norm[name] - rendered_p).pow(2).sum(dim=1).sqrt().mean()
                # Distance between GT render and perturbed render
                diff_gt_p = (rendered_gt[name] - rendered_p).pow(2).sum(dim=1).sqrt().mean()
                
                dists[name] = (diff_to_query.item(), diff_gt_p.item())
            
            print(f"  noise={noise_deg:5.1f}° (actual={rot_err:5.1f}°) | "
                  f"fine_sd: q-r={dists['fine_sd'][0]:.4f} gt-r={dists['fine_sd'][1]:.4f} | "
                  f"fine_dino: q-r={dists['fine_dino'][0]:.4f} gt-r={dists['fine_dino'][1]:.4f}")

print("\n" + "=" * 60)
print("Key question: do distances INCREASE with perturbation?")
print("If YES → pipeline is correct, model should learn")
print("If NO → rendering doesn't capture pose change")
print("=" * 60)
