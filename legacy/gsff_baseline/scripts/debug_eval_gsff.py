#!/usr/bin/env python3
"""Debug GSFFs evaluation: check GT poses, initial poses, feature quality."""
import json, math, sys, os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path")

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))


device = torch.device('cuda')

# 1. Load cameras.json and check GT poses
print("=" * 60)
print("1. Checking cameras.json GT poses")
print("=" * 60)

cameras_json = "output/2dgs_models/OldHospital/v7_depth/cameras.json"
with open(cameras_json) as f:
    all_cams = json.load(f)
    
print(f"  Total cameras: {len(all_cams)}")
cam_by_name = {c['img_name']: c for c in all_cams}

# Check first few cameras
for c in all_cams[:3]:
    R = np.array(c['rotation'])
    pos = np.array(c['position'])
    print(f"  {c['img_name']}: pos={pos}, det(R)={np.linalg.det(R):.4f}")

# 2. Load test set and check how many have matching cameras
print("\n" + "=" * 60)
print("2. Checking test set GT availability")
print("=" * 60)

source_dir = Path("dataset/OldHospital")
test_file = source_dir / "dataset_test.txt"
test_samples = []
with open(test_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('Visual') or line.startswith('ImageFile'):
            continue
        parts = line.split()
        test_samples.append({
            'img_name': parts[0],
            'position': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
            'quat': np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]),
        })

n_in_cams = sum(1 for s in test_samples if s['img_name'] in cam_by_name)
print(f"  Test images: {len(test_samples)}")
print(f"  Found in cameras.json: {n_in_cams}")
print(f"  Missing from cameras.json: {len(test_samples) - n_in_cams}")

# Show some test image names
print(f"  First 5 test names: {[s['img_name'] for s in test_samples[:5]]}")
print(f"  First 5 cam  names: {[c['img_name'] for c in all_cams[:5]]}")

# 3. Check GT consistency between cameras.json and dataset_test.txt
print("\n" + "=" * 60)
print("3. Checking GT consistency (cameras.json vs dataset_test.txt)")
print("=" * 60)

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ], dtype=np.float32)

for s in test_samples[:5]:
    name = s['img_name']
    # From dataset_test.txt
    R_txt = quat_to_rotmat(s['quat'])
    t_txt = s['position']
    
    if name in cam_by_name:
        cam = cam_by_name[name]
        R_cam = np.array(cam['rotation'])
        t_cam = np.array(cam['position'])
        
        pos_diff = np.linalg.norm(t_txt - t_cam)
        R_diff = np.arccos(np.clip((np.trace(R_txt @ R_cam.T) - 1) / 2, -1, 1)) * 180 / np.pi
        print(f"  {name}: pos_diff={pos_diff:.4f}m, rot_diff={R_diff:.2f}°")
        print(f"    txt_pos={t_txt[:3]}, cam_pos={t_cam[:3]}")
    else:
        print(f"  {name}: NOT in cameras.json")

# 4. Check train set and nearest-frame retrieval quality
print("\n" + "=" * 60)
print("4. Checking initial pose quality (nearest-frame retrieval)")
print("=" * 60)

train_file = source_dir / "dataset_train.txt"
train_samples = []
with open(train_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('Visual') or line.startswith('ImageFile'):
            continue
        parts = line.split()
        train_samples.append({
            'img_name': parts[0],
            'position': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
        })

train_names = {s['img_name'] for s in train_samples}
train_cams = [c for c in all_cams if c['img_name'] in train_names]

# Check actual nearest pose (by position distance), vs name-based retrieval 
def find_nearest_train_pose(test_name, train_cams):
    """Name-based nearest (as in eval_gsff.py)"""
    test_parts = test_name.replace('/', '_').replace('.png', '').split('_')
    best_cam, best_dist = None, float('inf')
    for cam in train_cams:
        cam_parts = cam['img_name'].replace('/', '_').replace('.png', '').split('_')
        if test_parts[0] == cam_parts[0]:
            try:
                frame_dist = abs(int(test_parts[1].replace('frame', '')) -
                                 int(cam_parts[1].replace('frame', '')))
            except (ValueError, IndexError):
                frame_dist = 1000
        else:
            frame_dist = 10000
        if frame_dist < best_dist:
            best_dist = frame_dist
            best_cam = cam
    return best_cam

init_pos_errors = []
init_rot_errors = []
best_possible_pos_errors = []

for s in test_samples:
    name = s['img_name']
    if name not in cam_by_name:
        continue
    
    gt_cam = cam_by_name[name]
    gt_pos = np.array(gt_cam['position'])
    gt_R = np.array(gt_cam['rotation'])
    
    # Name-based retrieval
    nn_cam = find_nearest_train_pose(name, train_cams)
    nn_pos = np.array(nn_cam['position'])
    nn_R = np.array(nn_cam['rotation'])
    pos_err = np.linalg.norm(gt_pos - nn_pos)
    R_rel = gt_R @ nn_R.T
    rot_err = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)) * 180 / np.pi
    init_pos_errors.append(pos_err * 100)  # in cm
    init_rot_errors.append(rot_err)
    
    # Best possible (nearest by position)
    best_dist = float('inf')
    for tc in train_cams:
        d = np.linalg.norm(gt_pos - np.array(tc['position']))
        if d < best_dist:
            best_dist = d
    best_possible_pos_errors.append(best_dist * 100)

init_pos_errors = np.array(init_pos_errors)
init_rot_errors = np.array(init_rot_errors)
best_possible_pos_errors = np.array(best_possible_pos_errors)

print(f"  Name-based retrieval:")
print(f"    Median init pos error: {np.median(init_pos_errors):.1f} cm")
print(f"    Median init rot error: {np.median(init_rot_errors):.2f}°")
print(f"    Mean init pos error:   {np.mean(init_pos_errors):.1f} cm")
print(f"    Mean init rot error:   {np.mean(init_rot_errors):.2f}°")
print(f"    Max init pos error:    {np.max(init_pos_errors):.1f} cm")
print(f"  Best possible (position-nearest):")
print(f"    Median: {np.median(best_possible_pos_errors):.1f} cm")
print(f"    Mean:   {np.mean(best_possible_pos_errors):.1f} cm")

# 5. Quick feature test: render from GT pose and compare with encoder output
print("\n" + "=" * 60)
print("5. Quick feature matching test (GT pose)")
print("=" * 60)

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder
from gsff.pose_refine import render_features_for_pose
import argparse

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

# Extract triplane features once
with torch.no_grad():
    coarse_colors = triplane.extract_coarse(means3d)
    coarse_colors = F.normalize(coarse_colors, p=2, dim=1)
    fine_colors = triplane.extract_fine(means3d)
    fine_colors = F.normalize(fine_colors, p=2, dim=1)

# Test on first 3 images
render_h, render_w = 540, 960
coarse_h, coarse_w = render_h // 14, render_w // 14
fine_h, fine_w = 270, 480

fx_orig = all_cams[0]['fx']
fy_orig = all_cams[0]['fy']
orig_w, orig_h = all_cams[0]['width'], all_cams[0]['height']

K_full = torch.zeros(3, 3, device=device)
K_full[0, 0] = fx_orig * render_w / orig_w
K_full[1, 1] = fy_orig * render_h / orig_h
K_full[0, 2] = render_w / 2.0
K_full[1, 2] = render_h / 2.0
K_full[2, 2] = 1.0

K_coarse = K_full.clone()
K_coarse[0] *= coarse_w / render_w
K_coarse[1] *= coarse_h / render_h

K_fine = K_full.clone()
K_fine[0] *= fine_w / render_w
K_fine[1] *= fine_h / render_h

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

for idx, s in enumerate(test_samples[:5]):
    name = s['img_name']
    if name not in cam_by_name:
        print(f"  SKIP {name}: not in cameras.json")
        continue
    
    cam = cam_by_name[name]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
    c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    w2c_t = torch.from_numpy(w2c).to(device)
    
    # Load image
    img_path = source_dir / name
    if not img_path.exists():
        img_path = source_dir / 'processed' / name
    img = Image.open(img_path).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_norm = ((img_tensor.unsqueeze(0).to(device)) - MEAN) / STD
    
    with torch.no_grad():
        coarse_feat_2d, fine_feat_2d = encoder(img_norm)
        coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
        fine_feat_2d = F.normalize(fine_feat_2d, p=2, dim=1)
        
        coarse_feat_2d = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
        fine_feat_2d = F.interpolate(fine_feat_2d, (fine_h, fine_w), mode='bilinear', align_corners=False)
    
    # Render from GT pose
    with torch.no_grad():
        coarse_rendered = render_features_for_pose(
            means3d, quats, scales, opacities, coarse_colors,
            w2c_t.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h, chunk_size=16,
        )
        coarse_rendered = F.normalize(coarse_rendered, p=2, dim=1)
        
        fine_rendered = render_features_for_pose(
            means3d, quats, scales, opacities, fine_colors,
            w2c_t.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h, chunk_size=16,
        )
        fine_rendered = F.normalize(fine_rendered, p=2, dim=1)
    
    # Compute similarity
    coarse_mse = F.mse_loss(coarse_rendered, coarse_feat_2d).item()
    fine_mse = F.mse_loss(fine_rendered, fine_feat_2d).item()
    
    # Cosine similarity
    coarse_cos = F.cosine_similarity(
        coarse_rendered.flatten(2), coarse_feat_2d.flatten(2), dim=1
    ).mean().item()
    fine_cos = F.cosine_similarity(
        fine_rendered.flatten(2), fine_feat_2d.flatten(2), dim=1
    ).mean().item()
    
    # Check alpha (are features even rendered?)
    coarse_alpha = coarse_rendered.abs().sum(dim=1).mean().item()
    fine_alpha = fine_rendered.abs().sum(dim=1).mean().item()
    
    print(f"  {name}:")
    print(f"    Coarse: MSE={coarse_mse:.4f}, cos_sim={coarse_cos:.4f}, mean_abs={coarse_alpha:.4f}")
    print(f"    Fine:   MSE={fine_mse:.4f}, cos_sim={fine_cos:.4f}, mean_abs={fine_alpha:.4f}")
    print(f"    2D feat range: coarse=[{coarse_feat_2d.min():.3f}, {coarse_feat_2d.max():.3f}], fine=[{fine_feat_2d.min():.3f}, {fine_feat_2d.max():.3f}]")
    print(f"    3D feat range: coarse=[{coarse_rendered.min():.3f}, {coarse_rendered.max():.3f}], fine=[{fine_rendered.min():.3f}, {fine_rendered.max():.3f}]")

# 6. Quick single-image optimization test
print("\n" + "=" * 60)
print("6. Single-image pose refinement test (from GT → should stay at GT)")
print("=" * 60)

from gsff.pose_refine import refine_pose

s = test_samples[0]
name = s['img_name']
cam = cam_by_name[name]
c2w = np.eye(4, dtype=np.float32)
c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
gt_w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)

img_path = source_dir / name
if not img_path.exists():
    img_path = source_dir / 'processed' / name
img = Image.open(img_path).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
img_norm = ((img_tensor.unsqueeze(0).to(device)) - MEAN) / STD

with torch.no_grad():
    coarse_feat_2d, _ = encoder(img_norm)
    coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
    coarse_feat_2d = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)

refined = refine_pose(
    coarse_feat_2d, means3d, quats, scales, opacities,
    coarse_colors, gt_w2c, K_coarse, coarse_w, coarse_h,
    n_iters=50, lr=0.01, chunk_size=16,
)

pred_c2w = torch.inverse(refined)
gt_c2w_t = torch.inverse(gt_w2c)
pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w_t[:3, 3]).item() * 100
R_rel = pred_c2w[:3, :3] @ gt_c2w_t[:3, :3].T
rot_err = torch.acos(torch.clamp((R_rel[0,0]+R_rel[1,1]+R_rel[2,2]-1)/2, -1, 1)).item() * 180 / math.pi
print(f"  Starting from GT pose (should be ~0):")
print(f"  After 50 iters: pos_err={pos_err:.1f} cm, rot_err={rot_err:.2f}°")

# 7. Test with perturbed GT
print("\n  Starting from perturbed GT (5°, 0.1m offset):")
from gsff.pose_refine import se3_exp
perturb = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.087], device=device)  # ~5° around z, 10cm x
perturbed_w2c = se3_exp(perturb) @ gt_w2c

refined2 = refine_pose(
    coarse_feat_2d, means3d, quats, scales, opacities,
    coarse_colors, perturbed_w2c, K_coarse, coarse_w, coarse_h,
    n_iters=100, lr=0.01, chunk_size=16,
)

pred_c2w2 = torch.inverse(refined2)
pos_err2 = torch.norm(pred_c2w2[:3, 3] - gt_c2w_t[:3, 3]).item() * 100
R_rel2 = pred_c2w2[:3, :3] @ gt_c2w_t[:3, :3].T
rot_err2 = torch.acos(torch.clamp((R_rel2[0,0]+R_rel2[1,1]+R_rel2[2,2]-1)/2, -1, 1)).item() * 180 / math.pi
print(f"  After 100 iters: pos_err={pos_err2:.1f} cm, rot_err={rot_err2:.2f}°")
print(f"  Initial perturbation was ~10 cm, ~5°")

print("\nDone!")
