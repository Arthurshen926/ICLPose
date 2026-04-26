"""Quick evaluation: check per-view PSNR distribution and identify bottleneck views."""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import math
import os

# Load the scene
from feature_3dgs.train_2dgs_geometry import (
    GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor
)

print("Loading scene...")
source_dir = "dataset/OldHospital"
train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene(source_dir)
print(f"  cameras_extent = {cameras_extent:.2f}")
print(f"  Train: {len(train_cams)}, Test: {len(test_cams)}")

# Load best checkpoint
ckpt_dir = "output/2dgs_models/OldHospital/v3_retrain17/point_cloud/iteration_15000"
print(f"\nLoading checkpoint from {ckpt_dir}...")
gaussians = GaussianModel2DGS(sh_degree=3)
gaussians.load_ply(os.path.join(ckpt_dir, "point_cloud.ply"))
print(f"  {gaussians.num_points:,} Gaussians")

# Evaluate
bg_color = torch.zeros(3, device="cuda")
psnrs = []
worst = []
best = []

print("\nEvaluating on test views...")
with torch.no_grad():
    for cam in test_cams:
        render_pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=0)
        image = render_pkg["render"].clamp(0, 1)
        rw, rh = render_pkg["width"], render_pkg["height"]
        gt_image = load_image_tensor(cam)
        gt_image = F.interpolate(gt_image.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)
        mse = F.mse_loss(image, gt_image)
        if mse > 0:
            psnr = -10 * math.log10(mse.item())
            psnrs.append((psnr, cam.image_name))

psnrs.sort()
print(f"\nOverall PSNR: {sum(p for p,_ in psnrs)/len(psnrs):.2f} dB ({len(psnrs)} views)")
print(f"\n--- 10 WORST views ---")
for psnr, name in psnrs[:10]:
    print(f"  {psnr:.2f} dB  {name}")
print(f"\n--- 10 BEST views ---")
for psnr, name in psnrs[-10:]:
    print(f"  {psnr:.2f} dB  {name}")

# Statistics
p = np.array([p for p,_ in psnrs])
print(f"\nPSNR statistics:")
print(f"  Min: {p.min():.2f}, Max: {p.max():.2f}")
print(f"  Mean: {p.mean():.2f}, Median: {np.median(p):.2f}")
print(f"  Std: {p.std():.2f}")
print(f"  Q25: {np.percentile(p,25):.2f}, Q75: {np.percentile(p,75):.2f}")

# Check if there's a bimodal distribution (some views much worse)
n_below_15 = np.sum(p < 15)
n_below_14 = np.sum(p < 14)
n_below_13 = np.sum(p < 13)
print(f"\n  Views < 15 dB: {n_below_15}")
print(f"  Views < 14 dB: {n_below_14}")
print(f"  Views < 13 dB: {n_below_13}")
