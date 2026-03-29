#!/usr/bin/env python3
"""Test different pose optimization strategies."""
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
from gsff.pose_refine import se3_exp, render_features_for_pose

# Load everything
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

# Camera setup
with open("output/2dgs_models/OldHospital/v7_depth/cameras.json") as f:
    all_cams = json.load(f)
cam_by_name = {c['img_name']: c for c in all_cams}
source_dir = Path("dataset/OldHospital")

with open(source_dir / "dataset_test.txt") as f:
    test_name = [l.strip().split()[0] for l in f if l.strip() and not l.strip().startswith(('V','I'))][0]

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

# Perturbed pose: 1m offset, 15 degrees
perturb = torch.tensor([0.5, 0.3, -0.2, 0.1, -0.15, 0.26], device=device)  # ~15°, ~0.6m
perturbed_w2c = se3_exp(perturb) @ gt_w2c

def compute_errors(current_vm):
    pred_c2w = torch.inverse(current_vm.squeeze(0).detach() if current_vm.dim() == 3 else current_vm.detach())
    pos_err = torch.norm(pred_c2w[:3,3] - gt_c2w[:3,3]).item() * 100
    R_rel = pred_c2w[:3,:3] @ gt_c2w[:3,:3].T
    rot_err = torch.acos(torch.clamp((R_rel[0,0]+R_rel[1,1]+R_rel[2,2]-1)/2, -1, 1)).item() * 180/math.pi
    return pos_err, rot_err

init_pos, init_rot = compute_errors(perturbed_w2c)
print(f"Initial perturbation: pos={init_pos:.1f}cm, rot={init_rot:.2f}°")

def run_optimization(init_w2c, rot_params, trans_params, rot_lr, trans_lr, n_iters, label):
    """Optimize with separate rotation and translation parameters."""
    delta_rot = torch.zeros(3, device=device, requires_grad=rot_params)
    delta_trans = torch.zeros(3, device=device, requires_grad=trans_params)
    
    params = []
    if rot_params:
        params.append({'params': [delta_rot], 'lr': rot_lr})
    if trans_params:
        params.append({'params': [delta_trans], 'lr': trans_lr})
    
    optimizer = torch.optim.Adam(params)
    
    for i in range(n_iters):
        optimizer.zero_grad()
        
        xi = torch.cat([delta_trans, delta_rot])
        delta_T = se3_exp(xi)
        current_vm = (delta_T @ init_w2c).unsqueeze(0)
        
        feat_3d = render_features_for_pose(
            means3d, quats, scales, opacities, coarse_colors,
            current_vm, K_coarse.unsqueeze(0), coarse_w, coarse_h,
            chunk_size=16, use_3dgs=True,
        )
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        loss = F.mse_loss(feat_3d_norm, coarse_feat_2d)
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0 or i == n_iters-1:
            pos_err, rot_err = compute_errors(current_vm)
            grad_r = delta_rot.grad.norm().item() if delta_rot.grad is not None else 0.0
            grad_t = delta_trans.grad.norm().item() if delta_trans.grad is not None else 0.0
            print(f"  [{label}] iter {i:3d}: loss={loss.item():.6f}, pos={pos_err:.1f}cm, rot={rot_err:.2f}°, "
                  f"grad_t={grad_t:.6f}, grad_r={grad_r:.6f}")
    
    return (delta_T @ init_w2c).detach()

# Strategy 1: Joint optimization (baseline)
print("\n=== Strategy 1: Joint Adam lr=0.01 ===")
run_optimization(perturbed_w2c, True, True, 0.01, 0.01, 200, "joint")

# Strategy 2: Rotation only
print("\n=== Strategy 2: Rotation only ===")
rot_result = run_optimization(perturbed_w2c, True, False, 0.01, 0.0, 200, "rot-only")

# Strategy 3: Rotation then translation
print("\n=== Strategy 3: Rotation first, then both ===")
mid = run_optimization(perturbed_w2c, True, False, 0.01, 0.0, 100, "rot-phase")
run_optimization(mid, True, True, 0.001, 0.001, 200, "both-phase")

# Strategy 4: Different LRs (small trans, bigger rot)
print("\n=== Strategy 4: Joint, trans_lr=0.001, rot_lr=0.01 ===")
run_optimization(perturbed_w2c, True, True, 0.01, 0.001, 200, "diff-lr")

# Strategy 5: SGD instead of Adam
print("\n=== Strategy 5: SGD lr=0.1 joint ===")
delta_xi = torch.zeros(6, device=device, requires_grad=True)
optimizer = torch.optim.SGD([delta_xi], lr=0.1, momentum=0.9)
for i in range(200):
    optimizer.zero_grad()
    delta_T = se3_exp(delta_xi)
    current_vm = (delta_T @ perturbed_w2c).unsqueeze(0)
    feat_3d = render_features_for_pose(
        means3d, quats, scales, opacities, coarse_colors,
        current_vm, K_coarse.unsqueeze(0), coarse_w, coarse_h,
        chunk_size=16, use_3dgs=True,
    )
    loss = F.mse_loss(F.normalize(feat_3d, p=2, dim=1), coarse_feat_2d)
    loss.backward()
    optimizer.step()
    if i % 50 == 0 or i == 199:
        pos_err, rot_err = compute_errors(current_vm)
        print(f"  [sgd] iter {i:3d}: loss={loss.item():.6f}, pos={pos_err:.1f}cm, rot={rot_err:.2f}°")

# Strategy 6: Pixel-wise cosine loss instead of MSE 
print("\n=== Strategy 6: Per-pixel cosine loss, Adam lr=0.01 ===")
delta_xi = torch.zeros(6, device=device, requires_grad=True)
optimizer = torch.optim.Adam([delta_xi], lr=0.01)
for i in range(200):
    optimizer.zero_grad()
    delta_T = se3_exp(delta_xi)
    current_vm = (delta_T @ perturbed_w2c).unsqueeze(0)
    feat_3d = render_features_for_pose(
        means3d, quats, scales, opacities, coarse_colors,
        current_vm, K_coarse.unsqueeze(0), coarse_w, coarse_h,
        chunk_size=16, use_3dgs=True,
    )
    # Per-pixel cosine loss
    loss = (1.0 - F.cosine_similarity(feat_3d, coarse_feat_2d, dim=1)).mean()
    loss.backward()
    optimizer.step()
    if i % 50 == 0 or i == 199:
        pos_err, rot_err = compute_errors(current_vm)
        print(f"  [cos] iter {i:3d}: loss={loss.item():.6f}, pos={pos_err:.1f}cm, rot={rot_err:.2f}°")

print("\nDone!")
