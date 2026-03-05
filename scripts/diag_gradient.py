#!/usr/bin/env python3
"""Quick diagnostic: check if gsplat gradient_2dgs is populated."""
import torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor
import torch.nn.functional as F

train_cams, test_cams, pcd_xyz, pcd_rgb, extent = load_scene(
    "dataset/OldHospital", images_subdir=".", eval_split=True
)
print(f"cameras_extent = {extent:.4f}")

gaussians = GaussianModel2DGS(sh_degree=3)
gaussians.create_from_pcd(pcd_xyz, pcd_rgb, extent)
print(f"N = {gaussians.num_points}")

bg = torch.zeros(3, device="cuda")
cam = train_cams[0]

# Forward
pkg = render_2dgs(gaussians, cam, bg, longest_edge=960)
img = pkg["render"]
gt = load_image_tensor(cam)
rw, rh = pkg["width"], pkg["height"]
gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)

# Loss + backward
loss = F.l1_loss(img, gt)
loss.backward()

# Check gradient_2dgs
grad_2d = pkg["viewspace_points"]
print(f"\ngradient_2dgs shape: {grad_2d.shape}")
print(f"gradient_2dgs dtype: {grad_2d.dtype}")
print(f"gradient_2dgs requires_grad: {grad_2d.requires_grad}")
print(f"gradient_2dgs has grad: {grad_2d.grad is not None}")
print(f"gradient_2dgs abs max: {grad_2d.abs().max().item():.8f}")
print(f"gradient_2dgs abs mean: {grad_2d.abs().mean().item():.8f}")
print(f"gradient_2dgs nonzero: {(grad_2d.abs() > 0).sum().item()} / {grad_2d.numel()}")

# Simulate what add_densification_stats does
grad = grad_2d.squeeze(0)  # [N, 2] or [N, dim]
print(f"\ngrad after squeeze shape: {grad.shape}")
vis = pkg["visibility_filter"]
print(f"visibility_filter true count: {vis.sum().item()} / {vis.shape[0]}")

if grad_2d.grad is not None:
    print(f"\ngradient_2dgs.grad abs max: {grad_2d.grad.abs().max().item():.8f}")
    print(f"gradient_2dgs.grad abs mean: {grad_2d.grad.abs().mean().item():.8f}")
    grad_actual = grad_2d.grad.squeeze(0).clone()
    vis = pkg["visibility_filter"]
    print(f"visibility_filter true count: {vis.sum().item()} / {vis.shape[0]}")
    grad_actual[:, 0] *= rw * 0.5
    grad_actual[:, 1] *= rh * 0.5
    norms = torch.norm(grad_actual[vis, :2], dim=-1)
    print(f"grad norms (visible): max={norms.max().item():.6f}, mean={norms.mean().item():.6f}")
    print(f"grad > 0.0002: {(norms > 0.0002).sum().item()}")
    print(f"grad > 0.0001: {(norms > 0.0001).sum().item()}")
    print(f"grad > 0.00005: {(norms > 0.00005).sum().item()}")
else:
    print("grad_2d.grad is None!")
