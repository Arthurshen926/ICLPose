#!/usr/bin/env python3
"""Test the new differentiable mean/quat transform for pose refinement."""
import torch, sys, argparse, json, math
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
device = torch.device('cuda')

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder
from gsff.pose_refine import se3_exp, render_features_transformed, refine_pose

# Load model
ckpt = torch.load("output/gsff/OldHospital/checkpoints/best.pth", map_location='cpu')
ckpt_args = argparse.Namespace(**ckpt['args'])

gs_model = GaussianFeatureModel(feature_dim=ckpt_args.feature_dim)
gs_model.load_ply("output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply")
gs_model = gs_model.to(device).eval()

means3d = gs_model.get_xyz.detach()
quats = gs_model.get_rotation.detach()
scales_raw = gs_model.get_scaling
scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1).detach()
opacities = gs_model.get_opacity.squeeze(-1).detach()

triplane = DualScaleTriplane(
    coarse_resolution=ckpt_args.coarse_resolution, fine_resolution=ckpt_args.fine_resolution,
    feature_dim=ckpt_args.feature_dim, scene_extent=ckpt['scene_extent'],
).to(device)
triplane.load_state_dict(ckpt['triplane'])
triplane.eval()

encoder = DualScaleEncoder(feature_dim=ckpt_args.feature_dim, freeze_backbone=True).to(device)
encoder.load_state_dict(ckpt['encoder'])
encoder.eval()

with torch.no_grad():
    coarse_colors = F.normalize(triplane.extract_coarse(means3d), p=2, dim=1)

with open("output/2dgs_models/OldHospital/v7_depth/cameras.json") as f:
    all_cams = json.load(f)
cam_by_name = {c['img_name']: c for c in all_cams}
source_dir = Path("dataset/OldHospital")

with open(source_dir / "dataset_test.txt") as f:
    test_names = [l.strip().split()[0] for l in f if l.strip() and not l.strip().startswith(('V','I'))]

test_name = test_names[0]
cam = cam_by_name[test_name]
c2w = np.eye(4, dtype=np.float32)
c2w[:3,:3] = np.array(cam['rotation'], dtype=np.float32)
c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
gt_w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)
gt_c2w = torch.inverse(gt_w2c)

render_h, render_w = 540, 960
coarse_h, coarse_w = render_h // 14, render_w // 14
K_coarse = torch.zeros(3, 3, device=device)
K_coarse[0, 0] = cam['fx'] * coarse_w / cam['width']
K_coarse[1, 1] = cam['fy'] * coarse_h / cam['height']
K_coarse[0, 2] = coarse_w / 2.0
K_coarse[1, 2] = coarse_h / 2.0
K_coarse[2, 2] = 1.0

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
img = Image.open(source_dir / test_name).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
img_t = torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0
img_norm = ((img_t.unsqueeze(0).to(device)) - MEAN) / STD

with torch.no_grad():
    coarse_feat_2d, _ = encoder(img_norm)
    coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
    coarse_feat_2d = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)

def compute_errors(vm):
    p = torch.inverse(vm.detach())
    pos = torch.norm(p[:3,3] - gt_c2w[:3,3]).item() * 100
    R = p[:3,:3] @ gt_c2w[:3,:3].T
    rot = torch.acos(torch.clamp((R[0,0]+R[1,1]+R[2,2]-1)/2, -1, 1)).item() * 180/math.pi
    return pos, rot

# Test 1: Gradient check with new method
print("=" * 60)
print("Test 1: Gradient check with differentiable mean/quat transform")
print("=" * 60)

perturb = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.087], device=device)
perturbed_w2c = se3_exp(perturb) @ gt_w2c
init_pos, init_rot = compute_errors(perturbed_w2c)
print(f"Initial: pos={init_pos:.1f}cm, rot={init_rot:.2f}°")

delta_xi = torch.zeros(6, device=device, requires_grad=True)
delta_T = se3_exp(delta_xi)

feat_3d = render_features_transformed(
    means3d, quats, scales, opacities, coarse_colors,
    delta_T, perturbed_w2c, K_coarse.unsqueeze(0), coarse_w, coarse_h, 16,
)
feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
loss = F.mse_loss(feat_3d_norm, coarse_feat_2d)
loss.backward()

print(f"Loss: {loss.item():.6f}")
print(f"Gradient: {delta_xi.grad}")
print(f"Grad norm: {delta_xi.grad.norm().item():.6f}")

# Test 2: Optimization loop
print("\n" + "=" * 60)
print("Test 2: Optimization loop from small perturbation (10cm, 5°)")
print("=" * 60)

delta_xi = torch.zeros(6, device=device, requires_grad=True)
optimizer = torch.optim.Adam([delta_xi], lr=0.01)

for i in range(200):
    optimizer.zero_grad()
    delta_T = se3_exp(delta_xi)
    
    feat_3d = render_features_transformed(
        means3d, quats, scales, opacities, coarse_colors,
        delta_T, perturbed_w2c, K_coarse.unsqueeze(0), coarse_w, coarse_h, 16,
    )
    feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
    loss = F.mse_loss(feat_3d_norm, coarse_feat_2d)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_([delta_xi], 0.5)
    optimizer.step()
    
    if i % 20 == 0 or i < 5:
        current_vm = se3_exp(delta_xi.detach()) @ perturbed_w2c
        pos_err, rot_err = compute_errors(current_vm)
        grad_norm = delta_xi.grad.norm().item()
        print(f"iter {i:3d}: loss={loss.item():.6f}, grad={grad_norm:.6f}, pos={pos_err:.1f}cm, rot={rot_err:.2f}°")

# Test 3: Larger perturbation (60cm, 18°)
print("\n" + "=" * 60)
print("Test 3: Optimization from larger perturbation (60cm, 18°)")
print("=" * 60)

perturb2 = torch.tensor([0.5, 0.3, -0.2, 0.1, -0.15, 0.26], device=device)
perturbed_w2c2 = se3_exp(perturb2) @ gt_w2c
init_pos2, init_rot2 = compute_errors(perturbed_w2c2)
print(f"Initial: pos={init_pos2:.1f}cm, rot={init_rot2:.2f}°")

delta_xi2 = torch.zeros(6, device=device, requires_grad=True)
optimizer2 = torch.optim.Adam([delta_xi2], lr=0.01)

for i in range(300):
    optimizer2.zero_grad()
    delta_T2 = se3_exp(delta_xi2)
    
    feat_3d = render_features_transformed(
        means3d, quats, scales, opacities, coarse_colors,
        delta_T2, perturbed_w2c2, K_coarse.unsqueeze(0), coarse_w, coarse_h, 16,
    )
    feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
    loss = F.mse_loss(feat_3d_norm, coarse_feat_2d)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_([delta_xi2], 0.5)
    optimizer2.step()
    
    if i % 30 == 0 or i == 299:
        current_vm = se3_exp(delta_xi2.detach()) @ perturbed_w2c2
        pos_err, rot_err = compute_errors(current_vm)
        print(f"iter {i:3d}: loss={loss.item():.6f}, pos={pos_err:.1f}cm, rot={rot_err:.2f}°")

# Test 4: Using the refine_pose function directly
print("\n" + "=" * 60)
print("Test 4: refine_pose() function (small perturbation)")
print("=" * 60)

result = refine_pose(
    coarse_feat_2d, means3d, quats, scales, opacities,
    coarse_colors, perturbed_w2c, K_coarse, coarse_w, coarse_h,
    n_iters=100, lr=0.01, chunk_size=16,
)
pos_err, rot_err = compute_errors(result)
print(f"refine_pose result: pos={pos_err:.1f}cm, rot={rot_err:.2f}° (init: {init_pos:.1f}cm/{init_rot:.2f}°)")

print("\nDone!")
