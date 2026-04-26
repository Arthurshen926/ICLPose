#!/usr/bin/env python3
"""Detailed debug of pose refinement gradient flow."""
import json, math, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import argparse

def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path")

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))


device = torch.device('cuda')

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder
from gsff.pose_refine import se3_exp, render_features_for_pose
from gsplat import rasterization_2dgs

# Load everything
ckpt = torch.load("output/gsff/OldHospital/checkpoints/best.pth", map_location='cpu')
ckpt_args = argparse.Namespace(**ckpt['args'])

gs_model = GaussianFeatureModel(feature_dim=ckpt_args.feature_dim)
gs_model.load_ply("output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply")
gs_model = gs_model.to(device)
gs_model.eval()

means3d = gs_model.get_xyz.detach()
quats = gs_model.get_rotation.detach()
scales_raw = gs_model.get_scaling
scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1).detach()
opacities = gs_model.get_opacity.squeeze(-1).detach()

scene_extent = ckpt['scene_extent']

triplane = DualScaleTriplane(
    coarse_resolution=ckpt_args.coarse_resolution,
    fine_resolution=ckpt_args.fine_resolution,
    feature_dim=ckpt_args.feature_dim,
    scene_extent=scene_extent,
).to(device)
triplane.load_state_dict(ckpt['triplane'])
triplane.eval()

encoder = DualScaleEncoder(
    feature_dim=ckpt_args.feature_dim,
    freeze_backbone=True,
).to(device)
encoder.load_state_dict(ckpt['encoder'])
encoder.eval()

with torch.no_grad():
    coarse_colors = triplane.extract_coarse(means3d)
    coarse_colors = F.normalize(coarse_colors, p=2, dim=1)

# Load test image
cameras_json = "output/2dgs_models/OldHospital/v7_depth/cameras.json"
with open(cameras_json) as f:
    all_cams = json.load(f)
cam_by_name = {c['img_name']: c for c in all_cams}

source_dir = Path("dataset/OldHospital")
test_file = source_dir / "dataset_test.txt"
with open(test_file) as f:
    lines = [l.strip() for l in f if l.strip() and not l.strip().startswith(('Visual', 'ImageFile'))]
parts = lines[0].split()
test_name = parts[0]

cam = cam_by_name[test_name]
c2w = np.eye(4, dtype=np.float32)
c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
gt_w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)

render_h, render_w = 540, 960
coarse_h, coarse_w = render_h // 14, render_w // 14

orig_w, orig_h = all_cams[0]['width'], all_cams[0]['height']
K_coarse = torch.zeros(3, 3, device=device)
K_coarse[0, 0] = cam['fx'] * coarse_w / orig_w
K_coarse[1, 1] = cam['fy'] * coarse_h / orig_h
K_coarse[0, 2] = coarse_w / 2.0
K_coarse[1, 2] = coarse_h / 2.0
K_coarse[2, 2] = 1.0

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

img = Image.open(source_dir / test_name).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
img_norm = ((img_tensor.unsqueeze(0).to(device)) - MEAN) / STD

with torch.no_grad():
    coarse_feat_2d, _ = encoder(img_norm)
    coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
    coarse_feat_2d = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)

# ---- Test 1: Gradient check from GT pose ----
print("=" * 60)
print("Test 1: Gradient check from GT pose")
print("=" * 60)

delta_xi = torch.zeros(6, device=device, requires_grad=True)
delta_T = se3_exp(delta_xi)
current_viewmat = (delta_T @ gt_w2c).unsqueeze(0)

feat_3d = render_features_for_pose(
    means3d, quats, scales, opacities, coarse_colors,
    current_viewmat, K_coarse.unsqueeze(0), coarse_w, coarse_h, chunk_size=16,
)
feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
loss = F.mse_loss(feat_3d_norm, coarse_feat_2d)
loss.backward()

print(f"  Loss at GT: {loss.item():.6f}")
print(f"  delta_xi grad: {delta_xi.grad}")
print(f"  grad norm: {delta_xi.grad.norm().item():.8f}")

# ---- Test 2: Gradient check from perturbed pose ----
print("\n" + "=" * 60)
print("Test 2: Gradient check from perturbed pose (+10cm, +5°)")
print("=" * 60)

perturb = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.087], device=device)
perturbed_w2c = se3_exp(perturb) @ gt_w2c

delta_xi2 = torch.zeros(6, device=device, requires_grad=True)
delta_T2 = se3_exp(delta_xi2)
current_viewmat2 = (delta_T2 @ perturbed_w2c).unsqueeze(0)

feat_3d2 = render_features_for_pose(
    means3d, quats, scales, opacities, coarse_colors,
    current_viewmat2, K_coarse.unsqueeze(0), coarse_w, coarse_h, chunk_size=16,
)
feat_3d_norm2 = F.normalize(feat_3d2, p=2, dim=1)
loss2 = F.mse_loss(feat_3d_norm2, coarse_feat_2d)
loss2.backward()

print(f"  Loss at perturbed: {loss2.item():.6f}")
print(f"  delta_xi grad: {delta_xi2.grad}")
print(f"  grad norm: {delta_xi2.grad.norm().item():.8f}")

# ---- Test 3: Manual refinement loop with logging ----
print("\n" + "=" * 60)
print("Test 3: Detailed refinement loop from perturbed pose")
print("=" * 60)

delta_xi3 = torch.zeros(6, device=device, requires_grad=True)
optimizer = torch.optim.Adam([delta_xi3], lr=0.01)

for i in range(200):
    optimizer.zero_grad()
    delta_T3 = se3_exp(delta_xi3)
    current_vm = (delta_T3 @ perturbed_w2c).unsqueeze(0)
    
    feat_3d3 = render_features_for_pose(
        means3d, quats, scales, opacities, coarse_colors,
        current_vm, K_coarse.unsqueeze(0), coarse_w, coarse_h, chunk_size=16,
    )
    feat_3d3_norm = F.normalize(feat_3d3, p=2, dim=1)
    loss3 = F.mse_loss(feat_3d3_norm, coarse_feat_2d)
    loss3.backward()
    
    grad_norm = delta_xi3.grad.norm().item()
    optimizer.step()
    
    if i % 20 == 0 or i < 5:
        pred_c2w = torch.inverse(current_vm.squeeze(0).detach())
        gt_c2w = torch.inverse(gt_w2c)
        pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]).item() * 100
        R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
        rot_err = torch.acos(torch.clamp((R_rel[0,0]+R_rel[1,1]+R_rel[2,2]-1)/2, -1, 1)).item() * 180/math.pi
        print(f"  iter {i:3d}: loss={loss3.item():.6f}, grad_norm={grad_norm:.8f}, "
              f"pos_err={pos_err:.1f}cm, rot_err={rot_err:.2f}°, "
              f"xi={delta_xi3.data.cpu().numpy()}")

# ---- Test 4: Try higher resolution rendering ----
print("\n" + "=" * 60)
print("Test 4: Refinement at HIGHER resolution (render_h=270, render_w=480)")
print("=" * 60)

# Use full render resolution with coarse features
higher_h, higher_w = 270, 480
K_higher = torch.zeros(3, 3, device=device)
K_higher[0, 0] = cam['fx'] * higher_w / orig_w
K_higher[1, 1] = cam['fy'] * higher_h / orig_h
K_higher[0, 2] = higher_w / 2.0
K_higher[1, 2] = higher_h / 2.0
K_higher[2, 2] = 1.0

# Need matching 2D features at this resolution
with torch.no_grad():
    coarse_feat_2d_hi = F.interpolate(coarse_feat_2d, (higher_h, higher_w), mode='bilinear', align_corners=False)

delta_xi4 = torch.zeros(6, device=device, requires_grad=True)
optimizer4 = torch.optim.Adam([delta_xi4], lr=0.005)

for i in range(100):
    optimizer4.zero_grad()
    delta_T4 = se3_exp(delta_xi4)
    current_vm4 = (delta_T4 @ perturbed_w2c).unsqueeze(0)
    
    feat_3d4 = render_features_for_pose(
        means3d, quats, scales, opacities, coarse_colors,
        current_vm4, K_higher.unsqueeze(0), higher_w, higher_h, chunk_size=16,
    )
    feat_3d4_norm = F.normalize(feat_3d4, p=2, dim=1)
    loss4 = F.mse_loss(feat_3d4_norm, coarse_feat_2d_hi)
    loss4.backward()
    
    optimizer4.step()
    
    if i % 20 == 0:
        pred_c2w = torch.inverse(current_vm4.squeeze(0).detach())
        gt_c2w = torch.inverse(gt_w2c)
        pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]).item() * 100
        R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
        rot_err = torch.acos(torch.clamp((R_rel[0,0]+R_rel[1,1]+R_rel[2,2]-1)/2, -1, 1)).item() * 180/math.pi
        grad_norm = delta_xi4.grad.norm().item()
        print(f"  iter {i:3d}: loss={loss4.item():.6f}, grad_norm={grad_norm:.8f}, "
              f"pos_err={pos_err:.1f}cm, rot_err={rot_err:.2f}°")

# ---- Test 5: Try raw (unnormalized) loss ----
print("\n" + "=" * 60)
print("Test 5: Refinement with cosine loss instead of MSE")
print("=" * 60)

delta_xi5 = torch.zeros(6, device=device, requires_grad=True)
optimizer5 = torch.optim.Adam([delta_xi5], lr=0.01)

for i in range(200):
    optimizer5.zero_grad()
    delta_T5 = se3_exp(delta_xi5)
    current_vm5 = (delta_T5 @ perturbed_w2c).unsqueeze(0)
    
    feat_3d5 = render_features_for_pose(
        means3d, quats, scales, opacities, coarse_colors,
        current_vm5, K_coarse.unsqueeze(0), coarse_w, coarse_h, chunk_size=16,
    )
    
    # Cosine loss: 1 - cos_sim
    cos_sim = F.cosine_similarity(feat_3d5.flatten(2), coarse_feat_2d.flatten(2), dim=1).mean()
    loss5 = 1.0 - cos_sim
    loss5.backward()
    
    optimizer5.step()
    
    if i % 20 == 0 or i < 5:
        pred_c2w = torch.inverse(current_vm5.squeeze(0).detach())
        gt_c2w = torch.inverse(gt_w2c)
        pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]).item() * 100
        R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
        rot_err = torch.acos(torch.clamp((R_rel[0,0]+R_rel[1,1]+R_rel[2,2]-1)/2, -1, 1)).item() * 180/math.pi
        grad_norm = delta_xi5.grad.norm().item()
        print(f"  iter {i:3d}: loss={loss5.item():.6f}, cos_sim={cos_sim.item():.4f}, grad_norm={grad_norm:.8f}, "
              f"pos_err={pos_err:.1f}cm, rot_err={rot_err:.2f}°")

print("\nDone!")
