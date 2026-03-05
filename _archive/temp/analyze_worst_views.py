#!/usr/bin/env python3
"""Analyze what's wrong with worst test views — check their spatial location,
nearest training views, and error patterns."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feature_3dgs.train_2dgs_geometry import load_scene

train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene('dataset/OldHospital')

def get_cam_center(cam):
    W2C = np.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    C2W = np.linalg.inv(W2C)
    return C2W[:3, 3]

def get_cam_forward(cam):
    W2C = np.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    C2W = np.linalg.inv(W2C)
    return C2W[:3, 2]  # Z axis in camera frame = forward direction

# Get all camera centers
train_centers = np.array([get_cam_center(c) for c in train_cams])
test_centers = {c.image_name: get_cam_center(c) for c in test_cams}
test_forwards = {c.image_name: get_cam_forward(c) for c in test_cams}

# Per-view PSNR from CSV
import csv
data = list(csv.DictReader(open('output/visualization/retrain31_15k/per_view_psnr.csv')))
# CSV has view names with _ instead of /
view_psnr = {}
for d in data:
    name = d['view'].replace('_frame', '/frame')  # restore original path format
    view_psnr[name] = float(d['psnr'])

# Analyze worst views
print("="*80)
print("ANALYSIS OF WORST TEST VIEWS")
print("="*80)

worst_views = sorted(view_psnr.items(), key=lambda x: x[1])[:15]
best_views = sorted(view_psnr.items(), key=lambda x: x[1])[-10:]

for label, views in [("WORST 15", worst_views), ("BEST 10", best_views)]:
    print(f"\n--- {label} ---")
    for view_name, psnr in views:
        tc = test_centers[view_name]
        tf = test_forwards[view_name]
        
        dists = np.linalg.norm(train_centers - tc, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]
        nearest_name = train_cams[nearest_idx].image_name
        
        # Check viewing direction similarity with nearest train cam
        nearest_fwd = get_cam_forward(train_cams[nearest_idx])
        cos_sim = np.dot(tf, nearest_fwd) / (np.linalg.norm(tf) * np.linalg.norm(nearest_fwd))
        
        # Count training views within various radii
        n_within_1m = np.sum(dists < 1.0)
        n_within_2m = np.sum(dists < 2.0)
        n_within_5m = np.sum(dists < 5.0)
        
        print(f"  {view_name}: PSNR={psnr:.1f}  pos=({tc[0]:.1f},{tc[1]:.1f},{tc[2]:.1f})"
              f"  nearest={nearest_dist:.2f}m ({nearest_name})"
              f"  cos_sim={cos_sim:.3f}"
              f"  train_within: 1m={n_within_1m}, 2m={n_within_2m}, 5m={n_within_5m}")

# Correlation analysis
print("\n\n" + "="*80)
print("CORRELATION: PSNR vs distance to nearest training view")
print("="*80)
psnrs_all = []
dists_all = []
for view_name, psnr in view_psnr.items():
    tc = test_centers[view_name]
    dists = np.linalg.norm(train_centers - tc, axis=1)
    psnrs_all.append(psnr)
    dists_all.append(dists.min())

psnrs_all = np.array(psnrs_all)
dists_all = np.array(dists_all)
corr = np.corrcoef(psnrs_all, dists_all)[0,1]
print(f"Pearson correlation: {corr:.4f}")
print(f"  (negative = farther from training = lower PSNR)")

# Distance bins
for d_lo, d_hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 50)]:
    mask = (dists_all >= d_lo) & (dists_all < d_hi)
    if mask.sum() > 0:
        avg = psnrs_all[mask].mean()
        cnt = mask.sum()
        print(f"  dist [{d_lo:.1f}, {d_hi:.1f})m: {cnt:3d} views, avg PSNR={avg:.2f}")
