#!/usr/bin/env python3
"""Quick PSNR evaluation for v3 2DGS model."""
import torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor
import torch.nn.functional as F

ckpt = sys.argv[1] if len(sys.argv) > 1 else "output/2dgs_models/OldHospital/v3/point_cloud/iteration_30000/point_cloud.ply"
n_views = int(sys.argv[2]) if len(sys.argv) > 2 else 20

print(f"Evaluating: {ckpt}")
train_cams, test_cams, pcd_xyz, pcd_rgb, extent = load_scene(
    "dataset/OldHospital", images_subdir=".", eval_split=True
)
print(f"Test cameras: {len(test_cams)}")

gaussians = GaussianModel2DGS(sh_degree=3)

# Load PLY manually (no load_ply method)
from plyfile import PlyData
import numpy as np
plydata = PlyData.read(ckpt)
vertex = plydata["vertex"]
xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
n_pts = xyz.shape[0]

# Count SH features
f_dc_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_dc_")])
f_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
scale_names = sorted([p.name for p in vertex.properties if p.name.startswith("scale_")])
rot_names = sorted([p.name for p in vertex.properties if p.name.startswith("rot_")])

f_dc = np.stack([vertex[n] for n in f_dc_names], axis=1).reshape(n_pts, 3, -1).transpose(0, 2, 1)
f_rest = np.stack([vertex[n] for n in f_rest_names], axis=1).reshape(n_pts, 3, -1).transpose(0, 2, 1) if f_rest_names else np.zeros((n_pts, 0, 3))

gaussians._xyz = torch.tensor(xyz, dtype=torch.float32, device="cuda")
gaussians._features_dc = torch.tensor(f_dc, dtype=torch.float32, device="cuda")
gaussians._features_rest = torch.tensor(f_rest, dtype=torch.float32, device="cuda")
gaussians._opacity = torch.tensor(np.array(vertex["opacity"]).reshape(-1, 1), dtype=torch.float32, device="cuda")
gaussians._scaling = torch.tensor(np.stack([vertex[n] for n in scale_names], axis=1), dtype=torch.float32, device="cuda")
gaussians._rotation = torch.tensor(np.stack([vertex[n] for n in rot_names], axis=1), dtype=torch.float32, device="cuda")
print(f"Loaded {gaussians.num_points} Gaussians")

bg = torch.zeros(3, device="cuda")
psnrs = []
for i, cam in enumerate(test_cams[:n_views]):
    pkg = render_2dgs(gaussians, cam, bg, longest_edge=960)
    img = pkg["render"].clamp(0, 1)
    rw, rh = pkg["width"], pkg["height"]
    gt = load_image_tensor(cam)
    gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)
    mse = F.mse_loss(img, gt)
    if mse > 0:
        p = -10 * math.log10(mse.item())
        psnrs.append(p)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_views}: PSNR={p:.2f} dB")

avg = sum(psnrs) / len(psnrs) if psnrs else 0
print(f"\nPSNR ({len(psnrs)} views): {avg:.2f} dB")
