#!/usr/bin/env python3
"""Check training view PSNR near worst test views — are these regions poorly reconstructed?"""
import sys, os, math, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor

train_cams, test_cams, *_ = load_scene('dataset/OldHospital')

gaussians = GaussianModel2DGS(3)
gaussians.load_ply('output/2dgs_models/OldHospital/v3_retrain31/point_cloud/iteration_15000/point_cloud.ply')
bg = torch.zeros(3, device='cuda')

def get_center(cam):
    W2C = np.eye(4); W2C[:3,:3] = cam.R.T; W2C[:3,3] = cam.T
    return np.linalg.inv(W2C)[:3,3]

# Focus on region around worst test views (seq4 frame 50-56)
target_pos = np.array([21.1, -0.8, 24.2])  # seq4/frame00054 position

train_centers = np.array([get_center(c) for c in train_cams])
dists = np.linalg.norm(train_centers - target_pos, axis=1)
nearby_idx = np.argsort(dists)[:15]

print("Training views near seq4/frame00054 (PSNR=11.22)")
print("="*70)
print("These are what the model learned from in this region:")
print()

for idx in nearby_idx:
    cam = train_cams[idx]
    with torch.no_grad():
        pkg = render_2dgs(gaussians, cam, bg, longest_edge=0)
        rendered = pkg["render"].clamp(0,1)
        rw, rh = pkg["width"], pkg["height"]
        gt = load_image_tensor(cam)
        gt = F.interpolate(gt.unsqueeze(0), size=(rh,rw), mode="bilinear", align_corners=False).squeeze(0)
        mse = F.mse_loss(rendered, gt).item()
        psnr = -10 * math.log10(mse) if mse > 0 else 0
    print(f"  {cam.image_name}: dist={dists[idx]:.2f}m  train_PSNR={psnr:.2f} dB")

# Now check training views near best test views
print()
target_pos2 = np.array([-11.0, -0.4, 17.1])  # seq8/frame00094 position (20.43 dB)
dists2 = np.linalg.norm(train_centers - target_pos2, axis=1)
nearby_idx2 = np.argsort(dists2)[:15]

print("Training views near seq8/frame00094 (PSNR=20.43)")
print("="*70)
for idx in nearby_idx2:
    cam = train_cams[idx]
    with torch.no_grad():
        pkg = render_2dgs(gaussians, cam, bg, longest_edge=0)
        rendered = pkg["render"].clamp(0,1)
        rw, rh = pkg["width"], pkg["height"]
        gt = load_image_tensor(cam)
        gt = F.interpolate(gt.unsqueeze(0), size=(rh,rw), mode="bilinear", align_corners=False).squeeze(0)
        mse = F.mse_loss(rendered, gt).item()
        psnr = -10 * math.log10(mse) if mse > 0 else 0
    print(f"  {cam.image_name}: dist={dists2[idx]:.2f}m  train_PSNR={psnr:.2f} dB")
