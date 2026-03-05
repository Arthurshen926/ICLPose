#!/usr/bin/env python3
"""Diagnose distortion loss values to understand the proper lambda_dist scale."""
import torch, math, os, sys, numpy as np
from PIL import Image
from gsplat import rasterization_2dgs
sys.path.insert(0, ".")
from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor
import torch.nn.functional as F
from plyfile import PlyData
from torch import nn

train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene("dataset/OldHospital", images_subdir=".", eval_split=True)

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
sample_cams = train_cams[:3] + test_cams[:3]

dists = []
l1s = []
for cam in sample_cams:
    with torch.no_grad():
        pkg = render_2dgs(gaussians, cam, bg, longest_edge=1280)
        image = pkg["render"]
        rw, rh = pkg["width"], pkg["height"]
        gt = load_image_tensor(cam)
        gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)
        l1 = F.l1_loss(image, gt).item()
        mse = F.mse_loss(image.clamp(0,1), gt).item()
        psnr = -10*math.log10(mse) if mse > 0 else 0
        dist_val = pkg["rend_dist"].mean().item()
        dist_max = pkg["rend_dist"].max().item()
        l1s.append(l1)
        dists.append(dist_val)
        print(f"  {cam.image_name:<30s} L1={l1:.4f} PSNR={psnr:.1f}dB dist_mean={dist_val:.6f} dist_max={dist_max:.4f}")

print(f"\nAverage L1={np.mean(l1s):.4f}  Average dist_mean={np.mean(dists):.6f}")
print(f"Ratio dist/L1 = {np.mean(dists)/np.mean(l1s):.4f}")
print(f"Recommended lambda_dist = L1/(10*dist) = {np.mean(l1s)/(10*np.mean(dists)):.4f}")
print("(This gives dist_loss ~= 0.1 * rgb_loss)")
