#!/usr/bin/env python3
"""
Comprehensive gap analysis: why are we at 39.9cm when paper gets 21cm?

Tests:
1. Feature quality at GT pose (coarse & fine) 
2. Refinement landscape: does loss minimum coincide with GT?
3. Convergence analysis: where does optimization stall?
4. Per-sample error distribution: which cases fail?
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder, OldFineEncoder
from gsff.pose_refine import (
    se3_exp, render_features_transformed, render_features_for_pose,
    refine_pose, _transform_gaussians, _render_with_transformed_gaussians,
)
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from torchvision import transforms


def load_model(ckpt_path, ply_path, device):
    """Load checkpoint with backward compatibility."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    scene_extent = ckpt.get('scene_extent', 70.0)
    
    # Detect encoder type from checkpoint keys
    enc_keys = list(ckpt['encoder'].keys())
    use_old = any('fine_encoder.encoder.' in k for k in enc_keys)
    
    triplane = DualScaleTriplane(
        coarse_resolution=256, fine_resolution=1024,
        feature_dim=16, scene_extent=scene_extent
    ).to(device)
    triplane.load_state_dict(ckpt['triplane'])
    triplane.eval()
    
    encoder = DualScaleEncoder(feature_dim=16, freeze_backbone=True,
                               use_old_fine_encoder=use_old).to(device)
    encoder.load_state_dict(ckpt['encoder'], strict=False)
    encoder.eval()
    
    # Load Gaussians
    gs = GaussianFeatureModel(feature_dim=16)
    gs.load_ply(ply_path)
    gs = gs.cuda()
    
    means3d = gs._xyz.detach().to(device)
    quats = gs.get_rotation.detach().to(device)
    scales_raw = gs.get_scaling.detach().to(device)
    # 2DGS has 2 scales; rasterization_2dgs expects 3 — pad with ones
    if scales_raw.shape[1] == 2:
        scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1)
    else:
        scales = scales_raw
    opacities = gs.get_opacity.squeeze(-1).detach().to(device)
    
    return triplane, encoder, means3d, quats, scales, opacities, scene_extent


def load_cameras(cameras_json):
    with open(cameras_json) as f:
        cams = json.load(f)
    return cams


def cam_to_w2c(cam):
    R = torch.tensor(cam['rotation'], dtype=torch.float32)
    pos = torch.tensor(cam['position'], dtype=torch.float32)
    w2c = torch.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ pos
    return w2c


def main():
    device = torch.device('cuda')
    
    ckpt_path = 'output/gsff/OldHospital/checkpoints/final.pth'
    ply_path = 'output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply'
    cameras_json = 'output/2dgs_models/OldHospital/v7_depth/cameras.json'
    source_dir = 'dataset/OldHospital'
    
    print("Loading model...")
    triplane, encoder, means3d, quats, scales, opacities, scene_extent = \
        load_model(ckpt_path, ply_path, device)
    
    cams = load_cameras(cameras_json)
    
    # Separate train/test
    test_file = os.path.join(source_dir, 'dataset_test.txt')
    test_names = set()
    with open(test_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                test_names.add(parts[0])
    
    train_cams = [c for c in cams if c['img_name'] not in test_names]
    test_cams = [c for c in cams if c['img_name'] in test_names]
    train_positions = np.array([c['position'] for c in train_cams])
    
    print(f"Train: {len(train_cams)}, Test: {len(test_cams)}")
    
    # Camera intrinsics
    cam0 = cams[0]
    fx, fy = cam0['fx'], cam0['fy']
    cx, cy = cam0['width'] / 2, cam0['height'] / 2
    
    # Rendering config
    render_h, render_w = 540, 960
    coarse_h, coarse_w = 38, 68
    fine_h, fine_w = 270, 480
    
    scale_x = render_w / cam0['width']
    scale_y = render_h / cam0['height']
    K_render = torch.tensor([
        [fx * scale_x, 0, cx * scale_x],
        [0, fy * scale_y, cy * scale_y],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    K_coarse = K_render.clone()
    K_coarse[0] *= coarse_w / render_w
    K_coarse[1] *= coarse_h / render_h
    K_fine = K_render.clone()
    K_fine[0] *= fine_w / render_w
    K_fine[1] *= fine_h / render_h
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Pre-extract triplane features
    with torch.no_grad():
        coarse_colors = F.normalize(triplane.extract_coarse(means3d), p=2, dim=1)
        fine_colors = F.normalize(triplane.extract_fine(means3d), p=2, dim=1)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: Feature Quality at GT Pose")
    print("="*70)
    
    cos_coarse_list, cos_fine_list = [], []
    init_pos_list, init_rot_list = [], []
    
    # Test on 20 uniformly sampled test images
    sample_indices = np.linspace(0, len(test_cams)-1, 20, dtype=int)
    
    for idx in sample_indices:
        cam = test_cams[idx]
        gt_w2c = cam_to_w2c(cam).to(device)
        
        # Load and encode image
        img_path = os.path.join(source_dir, cam['img_name'])
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        img_norm = normalize(img_tensor)
        
        # 2D features
        with torch.no_grad():
            coarse_2d, fine_2d = encoder(img_norm)
            # Resize to match rendering resolution
            coarse_2d = F.interpolate(coarse_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
            coarse_2d = F.normalize(coarse_2d, p=2, dim=1)
            fine_2d_resized = F.interpolate(fine_2d, (fine_h, fine_w), mode='bilinear', align_corners=False)
            fine_2d_resized = F.normalize(fine_2d_resized, p=2, dim=1)
        
        # Render 3D features at GT pose
        vm = gt_w2c.unsqueeze(0)
        with torch.no_grad():
            coarse_3d = render_features_for_pose(
                means3d, quats, scales, opacities, coarse_colors,
                vm, K_coarse.unsqueeze(0), coarse_w, coarse_h)
            coarse_3d = F.normalize(coarse_3d, p=2, dim=1)
            
            fine_3d = render_features_for_pose(
                means3d, quats, scales, opacities, fine_colors,
                vm, K_fine.unsqueeze(0), fine_w, fine_h)
            fine_3d = F.normalize(fine_3d, p=2, dim=1)
        
        cos_c = (coarse_2d * coarse_3d).sum(dim=1).mean().item()
        cos_f = (fine_2d_resized * fine_3d).sum(dim=1).mean().item()
        cos_coarse_list.append(cos_c)
        cos_fine_list.append(cos_f)
        
        # Init pose error
        dists = np.linalg.norm(train_positions - np.array(cam['position']), axis=1)
        nn_idx = np.argmin(dists)
        nn_w2c = cam_to_w2c(train_cams[nn_idx]).to(device)
        
        R_rel = gt_w2c[:3,:3] @ nn_w2c[:3,:3].T
        trace = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
        rot_err = torch.acos(trace).item() * 180 / np.pi
        pos_err = torch.norm(
            torch.inverse(gt_w2c)[:3,3] - torch.inverse(nn_w2c)[:3,3]
        ).item() * 100
        
        init_pos_list.append(pos_err)
        init_rot_list.append(rot_err)
    
    print(f"\nCoarse cos_sim at GT: mean={np.mean(cos_coarse_list):.3f}, "
          f"std={np.std(cos_coarse_list):.3f}, min={np.min(cos_coarse_list):.3f}")
    print(f"Fine   cos_sim at GT: mean={np.mean(cos_fine_list):.3f}, "
          f"std={np.std(cos_fine_list):.3f}, min={np.min(cos_fine_list):.3f}")
    print(f"Init pos error: mean={np.mean(init_pos_list):.1f}cm, "
          f"median={np.median(init_pos_list):.1f}cm")
    print(f"Init rot error: mean={np.mean(init_rot_list):.1f}°, "
          f"median={np.median(init_rot_list):.1f}°")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: Loss Landscape near GT")
    print("="*70)
    
    # Pick a representative test image
    cam = test_cams[sample_indices[5]]
    gt_w2c = cam_to_w2c(cam).to(device)
    
    img = Image.open(os.path.join(source_dir, cam['img_name'])).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img_norm = normalize(img_tensor)
    
    with torch.no_grad():
        coarse_2d, fine_2d = encoder(img_norm)
        coarse_2d = F.interpolate(coarse_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
        coarse_2d = F.normalize(coarse_2d, p=2, dim=1)
        fine_2d_resized = F.interpolate(fine_2d, (fine_h, fine_w), mode='bilinear', align_corners=False)
        fine_2d_resized = F.normalize(fine_2d_resized, p=2, dim=1)
    
    # Sweep translation offsets and measure loss
    print(f"\nSample: {cam['img_name']}")
    print("Translation offset (cm) vs MSE loss (coarse | fine):")
    for offset_cm in [0, 5, 10, 20, 50, 100, 200]:
        offset_m = offset_cm / 100.0
        perturbed = gt_w2c.clone()
        perturbed[0, 3] += offset_m  # X-axis offset
        
        vm = perturbed.unsqueeze(0)
        with torch.no_grad():
            c3d = render_features_for_pose(
                means3d, quats, scales, opacities, coarse_colors,
                vm, K_coarse.unsqueeze(0), coarse_w, coarse_h)
            c3d = F.normalize(c3d, p=2, dim=1)
            c_loss = F.mse_loss(c3d, coarse_2d).item()
            
            f3d = render_features_for_pose(
                means3d, quats, scales, opacities, fine_colors,
                vm, K_fine.unsqueeze(0), fine_w, fine_h)
            f3d = F.normalize(f3d, p=2, dim=1)
            f_loss = F.mse_loss(f3d, fine_2d_resized).item()
        
        print(f"  {offset_cm:4d}cm: coarse_mse={c_loss:.6f}  fine_mse={f_loss:.6f}")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Per-sample Refinement (2R, 300+300)")
    print("="*70)
    
    # Test on 10 samples with detailed logging
    pos_errors_1r, rot_errors_1r = [], []
    pos_errors_2r, rot_errors_2r = [], []
    
    for idx in sample_indices[:10]:
        cam = test_cams[idx]
        gt_w2c = cam_to_w2c(cam).to(device)
        
        # NN init
        dists = np.linalg.norm(train_positions - np.array(cam['position']), axis=1)
        nn_idx = np.argmin(dists)
        current_w2c = cam_to_w2c(train_cams[nn_idx]).to(device)
        
        # Load image
        img = Image.open(os.path.join(source_dir, cam['img_name'])).convert('RGB')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        img_norm = normalize(img_tensor)
        
        with torch.no_grad():
            coarse_2d, fine_2d = encoder(img_norm)
            coarse_2d = F.interpolate(coarse_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
            coarse_2d = F.normalize(coarse_2d, p=2, dim=1)
            fine_2d_resized = F.interpolate(fine_2d, (fine_h, fine_w), mode='bilinear', align_corners=False)
            fine_2d_resized = F.normalize(fine_2d_resized, p=2, dim=1)
        
        init_gt_diff = torch.inverse(gt_w2c)[:3,3] - torch.inverse(current_w2c)[:3,3]
        init_pos = torch.norm(init_gt_diff).item() * 100
        
        # Round 1
        for rnd in range(2):
            current_w2c = refine_pose(
                coarse_2d, means3d, quats, scales, opacities, coarse_colors,
                current_w2c, K_coarse, coarse_w, coarse_h,
                n_iters=300, lr=0.01)
            
            current_w2c = refine_pose(
                fine_2d_resized, means3d, quats, scales, opacities, fine_colors,
                current_w2c, K_fine, fine_w, fine_h,
                n_iters=300, lr=0.005)
            
            # Measure error
            R_rel = gt_w2c[:3,:3] @ current_w2c[:3,:3].T
            trace = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
            rot = torch.acos(trace).item() * 180 / np.pi
            pos = torch.norm(
                torch.inverse(gt_w2c)[:3,3] - torch.inverse(current_w2c)[:3,3]
            ).item() * 100
            
            if rnd == 0:
                pos_errors_1r.append(pos)
                rot_errors_1r.append(rot)
            else:
                pos_errors_2r.append(pos)
                rot_errors_2r.append(rot)
        
        print(f"  [{idx:3d}] init={init_pos:.1f}cm → "
              f"1R={pos_errors_1r[-1]:.1f}cm/{rot_errors_1r[-1]:.2f}° → "
              f"2R={pos_errors_2r[-1]:.1f}cm/{rot_errors_2r[-1]:.2f}°")
    
    print(f"\n  1R median: {np.median(pos_errors_1r):.1f}cm / {np.median(rot_errors_1r):.2f}°")
    print(f"  2R median: {np.median(pos_errors_2r):.1f}cm / {np.median(rot_errors_2r):.2f}°")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: Optimization Trajectory (detailed)")
    print("="*70)
    
    # Detailed trajectory for one medium-difficulty sample
    cam = test_cams[sample_indices[3]]
    gt_w2c = cam_to_w2c(cam).to(device)
    dists = np.linalg.norm(train_positions - np.array(cam['position']), axis=1)
    nn_idx = np.argmin(dists)
    current_w2c = cam_to_w2c(train_cams[nn_idx]).to(device)
    
    img = Image.open(os.path.join(source_dir, cam['img_name'])).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img_norm = normalize(img_tensor)
    
    with torch.no_grad():
        coarse_2d, fine_2d = encoder(img_norm)
        coarse_2d = F.interpolate(coarse_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
        coarse_2d = F.normalize(coarse_2d, p=2, dim=1)
        fine_2d_resized = F.interpolate(fine_2d, (fine_h, fine_w), mode='bilinear', align_corners=False)
        fine_2d_resized = F.normalize(fine_2d_resized, p=2, dim=1)
    
    print(f"\nSample: {cam['img_name']}")
    
    # Manual refinement with trajectory logging
    delta_xi = torch.zeros(6, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_xi], lr=0.01)
    
    print("Coarse refinement trajectory:")
    for i in range(300):
        optimizer.zero_grad()
        delta_T = se3_exp(delta_xi)
        feat_3d = render_features_transformed(
            means3d, quats, scales, opacities, coarse_colors,
            delta_T, current_w2c, K_coarse.unsqueeze(0), coarse_w, coarse_h)
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        loss = F.mse_loss(feat_3d_norm, coarse_2d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_xi], 0.5)
        optimizer.step()
        
        if i % 50 == 0 or i == 299:
            with torch.no_grad():
                vm = se3_exp(delta_xi) @ current_w2c
                R_rel = gt_w2c[:3,:3] @ vm[:3,:3].T
                trace = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
                rot = torch.acos(trace).item() * 180 / np.pi
                pos = torch.norm(
                    torch.inverse(gt_w2c)[:3,3] - torch.inverse(vm)[:3,3]
                ).item() * 100
                grad_t = delta_xi.grad[:3].norm().item()
                grad_r = delta_xi.grad[3:].norm().item()
            print(f"  iter {i:3d}: loss={loss.item():.6f} pos={pos:.1f}cm "
                  f"rot={rot:.2f}° grad_t={grad_t:.6f} grad_r={grad_r:.6f}")
    
    # Apply coarse result
    with torch.no_grad():
        current_w2c = se3_exp(delta_xi) @ current_w2c
    
    # Fine refinement
    delta_xi = torch.zeros(6, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_xi], lr=0.005)
    
    print("\nFine refinement trajectory:")
    for i in range(300):
        optimizer.zero_grad()
        delta_T = se3_exp(delta_xi)
        feat_3d = render_features_transformed(
            means3d, quats, scales, opacities, fine_colors,
            delta_T, current_w2c, K_fine.unsqueeze(0), fine_w, fine_h)
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        loss = F.mse_loss(feat_3d_norm, fine_2d_resized)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_xi], 0.5)
        optimizer.step()
        
        if i % 50 == 0 or i == 299:
            with torch.no_grad():
                vm = se3_exp(delta_xi) @ current_w2c
                R_rel = gt_w2c[:3,:3] @ vm[:3,:3].T
                trace = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
                rot = torch.acos(trace).item() * 180 / np.pi
                pos = torch.norm(
                    torch.inverse(gt_w2c)[:3,3] - torch.inverse(vm)[:3,3]
                ).item() * 100
                grad_t = delta_xi.grad[:3].norm().item()
                grad_r = delta_xi.grad[3:].norm().item()
            print(f"  iter {i:3d}: loss={loss.item():.6f} pos={pos:.1f}cm "
                  f"rot={rot:.2f}° grad_t={grad_t:.6f} grad_r={grad_r:.6f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Coarse cos_sim at GT: {np.mean(cos_coarse_list):.3f}")
    print(f"Fine cos_sim at GT:   {np.mean(cos_fine_list):.3f}")
    print(f"1R median: {np.median(pos_errors_1r):.1f}cm / {np.median(rot_errors_1r):.2f}°")
    print(f"2R median: {np.median(pos_errors_2r):.1f}cm / {np.median(rot_errors_2r):.2f}°")
    print(f"Paper target: 21cm / 0.41°")
    print(f"Gap: {np.median(pos_errors_2r)/21:.1f}x position, {np.median(rot_errors_2r)/0.41:.1f}x rotation")


if __name__ == '__main__':
    main()
