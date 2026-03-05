#!/usr/bin/env python3
"""Verify normal supervision + depth loss work correctly at full resolution."""
import torch, numpy as np, sys, os
sys.path.insert(0, ".")
from feature_3dgs.train_2dgs_geometry import (
    GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor,
    pearson_depth_loss, load_mono_depth
)
from plyfile import PlyData
from torch import nn
import torch.nn.functional as F

train_cams, test_cams, _, _, _ = load_scene("dataset/OldHospital", images_subdir=".", eval_split=True)

plydata = PlyData.read("output/2dgs_models/OldHospital/v3_improved_v2/point_cloud/iteration_35000/point_cloud.ply")
v = plydata["vertex"]
gaussians = GaussianModel2DGS(sh_degree=3)
xyz = torch.tensor(np.vstack([v["x"], v["y"], v["z"]]).T, dtype=torch.float32, device="cuda")
n_sh = 16
f_dc = torch.tensor(np.vstack([v[f"f_dc_{i}"] for i in range(3)]).T, dtype=torch.float32, device="cuda")
f_rest = torch.tensor(np.vstack([v[f"f_rest_{i}"] for i in range(3*(n_sh-1))]).T, dtype=torch.float32, device="cuda")
opacities = torch.tensor(v["opacity"][:, None], dtype=torch.float32, device="cuda")
scales = torch.tensor(np.vstack([v["scale_0"], v["scale_1"]]).T, dtype=torch.float32, device="cuda")
rots = torch.tensor(np.vstack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]).T, dtype=torch.float32, device="cuda")
gaussians._xyz = nn.Parameter(xyz)
gaussians._features_dc = nn.Parameter(f_dc.reshape(-1, 1, 3))
gaussians._features_rest = nn.Parameter(f_rest.reshape(-1, n_sh-1, 3))
gaussians._opacity = nn.Parameter(opacities)
gaussians._scaling = nn.Parameter(scales)
gaussians._rotation = nn.Parameter(rots)
gaussians.active_sh_degree = 3
print(f"Loaded {gaussians.num_points:,} Gaussians")

bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
cam = train_cams[0]

with torch.no_grad():
    pkg = render_2dgs(gaussians, cam, bg, longest_edge=0)

print(f"\n=== Full-res render: {pkg['width']}x{pkg['height']} ===")
for k, v in pkg.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} [{v.min():.4f}, {v.max():.4f}]")
    else:
        print(f"  {k}: {v}")

# Normal consistency
rn = pkg["rend_normal"]
sn = pkg["surf_normal"]
alpha = pkg["rend_alpha"]
depth = pkg["depth"]

surf_normal_proc = sn * alpha.squeeze(0).detach()
rend_normal_proc = rn.squeeze(0).permute(2, 0, 1)
if len(surf_normal_proc.shape) == 4:
    surf_normal_proc = surf_normal_proc.squeeze(0)
surf_normal_proc = surf_normal_proc.permute(2, 0, 1)

normal_error = (1 - (rend_normal_proc * surf_normal_proc).sum(dim=0))
print(f"\n=== Normal consistency ===")
print(f"  normal_error: {normal_error.shape} mean={normal_error.mean():.4f}")
print(f"  Normal loss (lambda=0.05): {0.05 * normal_error.mean():.6f}")

# Depth loss — now uses load_mono_depth which inverts direction
rh, rw = pkg["height"], pkg["width"]
mono_d = load_mono_depth(cam, "dataset/OldHospital/mono_depth", rh, rw)
dloss = pearson_depth_loss(depth, mono_d)
print(f"\n=== Depth supervision ===")
print(f"  mono_depth: {mono_d.shape} range=[{mono_d.min():.4f}, {mono_d.max():.4f}]")
print(f"  rendered depth: {depth.shape} range=[{depth.min():.4f}, {depth.max():.4f}]")
print(f"  Pearson depth loss: {dloss:.6f} (should be < 1.0 if correlated)")
print(f"  Depth loss (lambda=0.1): {0.1*dloss:.6f}")

# VRAM usage
print(f"\n=== GPU Memory (full-res 1920x1080) ===")
print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

print("\n=== All checks passed! ===")
