#!/usr/bin/env python3
"""Test hybrid: feature-based rotation + photometric translation refinement."""
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
from gsff.pose_refine import se3_exp
from gsplat import rasterization

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

# RGB colors from SH DC
C0 = 0.28209479177387814
rgb_colors = torch.clamp(0.5 + C0 * gs_model._features_dc.detach(), 0, 1)

# Feature colors (coarse triplane)
triplane = DualScaleTriplane(
    coarse_resolution=ckpt_args.coarse_resolution, fine_resolution=ckpt_args.fine_resolution,
    feature_dim=ckpt_args.feature_dim, scene_extent=ckpt['scene_extent'],
).to(device)
triplane.load_state_dict(ckpt['triplane'])
triplane.eval()

with torch.no_grad():
    coarse_colors = F.normalize(triplane.extract_coarse(means3d), p=2, dim=1)

# Encoder
encoder = DualScaleEncoder(feature_dim=ckpt_args.feature_dim, freeze_backbone=True).to(device)
encoder.load_state_dict(ckpt['encoder'])
encoder.eval()

# Camera setup
with open("output/2dgs_models/OldHospital/v7_depth/cameras.json") as f:
    all_cams = json.load(f)
cam_by_name = {c['img_name']: c for c in all_cams}
source_dir = Path("dataset/OldHospital")

with open(source_dir / "dataset_test.txt") as f:
    test_names = [l.strip().split()[0] for l in f if l.strip() and not l.strip().startswith(('V','I'))]

# Use first few test images
for test_idx in range(3):
    test_name = test_names[test_idx]
    cam = cam_by_name[test_name]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3,:3] = np.array(cam['rotation'], dtype=np.float32)
    c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
    gt_w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)
    gt_c2w = torch.inverse(gt_w2c)

    render_h, render_w = 540, 960
    coarse_h, coarse_w = render_h // 14, render_w // 14

    orig_w, orig_h = cam['width'], cam['height']
    K_full = torch.zeros(3, 3, device=device)
    K_full[0, 0] = cam['fx'] * render_w / orig_w
    K_full[1, 1] = cam['fy'] * render_h / orig_h
    K_full[0, 2] = render_w / 2.0
    K_full[1, 2] = render_h / 2.0
    K_full[2, 2] = 1.0

    K_coarse = K_full.clone()
    K_coarse[0] *= coarse_w / render_w
    K_coarse[1] *= coarse_h / render_h

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    img = Image.open(source_dir / test_name).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
    img_t = torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0
    target_rgb = img_t.unsqueeze(0).to(device)  # [1, 3, H, W]
    img_norm = (target_rgb - MEAN) / STD

    with torch.no_grad():
        coarse_feat_2d, _ = encoder(img_norm)
        coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
        coarse_feat_2d = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)

    # Perturbed pose: 1.5m offset, 15 degrees
    perturb = torch.tensor([0.5, 0.3, -0.2, 0.1, -0.15, 0.26], device=device)
    perturbed_w2c = se3_exp(perturb) @ gt_w2c

    def compute_errors(vm):
        p = torch.inverse(vm.squeeze(0).detach() if vm.dim() == 3 else vm.detach())
        pos = torch.norm(p[:3,3] - gt_c2w[:3,3]).item() * 100
        R = p[:3,:3] @ gt_c2w[:3,:3].T
        rot = torch.acos(torch.clamp((R[0,0]+R[1,1]+R[2,2]-1)/2, -1, 1)).item() * 180/math.pi
        return pos, rot

    def render_3dgs(means, q, s, o, colors, vm, K, w, h):
        chunks = []
        D = colors.shape[1]
        for c_start in range(0, D, 3):
            c_end = min(c_start+3, D)
            rendered, alphas, *_ = rasterization(
                means=means, quats=q, scales=s, opacities=o,
                colors=colors[:, c_start:c_end],
                viewmats=vm, Ks=K, width=w, height=h,
                packed=False, near_plane=0.01, far_plane=1e5, render_mode='RGB',
            )
            chunks.append(rendered)
        return torch.cat(chunks, dim=-1).permute(0, 3, 1, 2)

    init_pos, init_rot = compute_errors(perturbed_w2c)
    print(f"\n{'='*60}")
    print(f"Image: {test_name}")
    print(f"Initial: pos={init_pos:.1f}cm, rot={init_rot:.2f}°")

    # === Hybrid strategy ===
    # Phase 1: Feature-based rotation refinement (100 iters)
    print("Phase 1: Feature-based rotation refinement")
    delta_rot = torch.zeros(3, device=device, requires_grad=True)
    opt1 = torch.optim.Adam([delta_rot], lr=0.01)
    
    for i in range(100):
        opt1.zero_grad()
        xi = torch.cat([torch.zeros(3, device=device), delta_rot])
        delta_T = se3_exp(xi)
        vm = (delta_T @ perturbed_w2c).unsqueeze(0)
        
        feat_3d = render_3dgs(means3d, quats, scales, opacities, coarse_colors,
                              vm, K_coarse.unsqueeze(0), coarse_w, coarse_h)
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        loss = F.mse_loss(feat_3d_norm, coarse_feat_2d)
        loss.backward()
        opt1.step()
        
        if i % 25 == 0 or i == 99:
            pos, rot = compute_errors(vm)
            print(f"  iter {i:3d}: loss={loss.item():.6f}, pos={pos:.1f}cm, rot={rot:.2f}°")

    rot_refined_w2c = (se3_exp(torch.cat([torch.zeros(3, device=device), delta_rot.detach()])) @ perturbed_w2c).detach()

    # Phase 2: Photometric translation refinement (200 iters at 270x480)
    print("Phase 2: Photometric translation+rotation refinement")
    photo_h, photo_w = 270, 480
    K_photo = K_full.clone()
    K_photo[0] *= photo_w / render_w
    K_photo[1] *= photo_h / render_h

    target_rgb_small = F.interpolate(target_rgb, (photo_h, photo_w), mode='bilinear', align_corners=False)

    delta_xi2 = torch.zeros(6, device=device, requires_grad=True)
    opt2 = torch.optim.Adam([delta_xi2], lr=0.005)

    for i in range(300):
        opt2.zero_grad()
        delta_T2 = se3_exp(delta_xi2)
        vm2 = (delta_T2 @ rot_refined_w2c).unsqueeze(0)
        
        rendered_rgb = render_3dgs(means3d, quats, scales, opacities, rgb_colors,
                                   vm2, K_photo.unsqueeze(0), photo_w, photo_h)
        
        # Multi-scale photometric loss
        loss_photo = F.mse_loss(rendered_rgb, target_rgb_small)
        
        # Also add SSIM-like loss (gradient-based)
        loss = loss_photo
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([delta_xi2], 0.1)
        opt2.step()
        
        if i % 50 == 0 or i == 299:
            pos, rot = compute_errors(vm2)
            gt = delta_xi2.grad
            print(f"  iter {i:3d}: loss={loss.item():.6f}, pos={pos:.1f}cm, rot={rot:.2f}°, "
                  f"grad_t={gt[:3].norm().item():.6f}, grad_r={gt[3:].norm().item():.6f}")

    final_pos, final_rot = compute_errors(vm2)
    print(f"Final: pos={final_pos:.1f}cm, rot={final_rot:.2f}° (from {init_pos:.1f}cm/{init_rot:.2f}°)")

print("\nDone!")
