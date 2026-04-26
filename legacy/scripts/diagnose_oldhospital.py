#!/usr/bin/env python3
"""
Diagnose OldHospital training divergence.

Tests:
1. GT flow through geometry solver → does pose improve?
2. Feature similarity check
3. Per-iteration pose error tracking
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import yaml

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve
from modules.lie_algebra import se3_exp
from modules.multiscale_renderer import MultiScaleRenderer
from ic_models.ms_flow_pose_net import MSFlowPoseNet


def angle_error(R_pred, R_gt):
    """Rotation error in degrees."""
    R_rel = R_pred @ R_gt.transpose(-1, -2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1, 1)
    return torch.acos(cos_angle) * 180 / np.pi


def trans_error(t_pred, t_gt):
    """Translation error in meters."""
    return (t_pred - t_gt).norm(dim=-1)


def add_noise(pose_w2c, rot_deg, trans_m):
    """Add random SE(3) noise to poses."""
    B = pose_w2c.shape[0]
    device = pose_w2c.device
    # Random rotation axis
    axis = F.normalize(torch.randn(B, 3, device=device), dim=1)
    angle = torch.randn(B, 1, device=device) * (rot_deg * np.pi / 180)
    omega = axis * angle
    # Random translation
    v = torch.randn(B, 3, device=device) * trans_m
    xi = torch.cat([v, omega], dim=1)  # (B, 6)
    T_noise = se3_exp(xi)
    return T_noise @ pose_w2c


def main():
    device = torch.device('cuda:0')
    config_path = 'configs/gsff_oldhospital.yaml'

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rcfg = cfg['renderer']
    mcfg = cfg['model']
    
    print("=" * 70)
    print("DIAGNOSTIC: OldHospital Training Divergence")
    print("=" * 70)

    # ── Load renderer ──
    print("\n[1] Loading renderer...")
    renderer = MultiScaleRenderer(
        ply_path=rcfg['ply_path'],
        scale_model_paths=rcfg['scale_model_paths'],
        img_height=rcfg['img_height'], img_width=rcfg['img_width'],
        fx=rcfg['fx'], fy=rcfg['fy'],
        cx=rcfg['cx'], cy=rcfg['cy'],
    ).to(device)

    # Check renderer depth resolution
    scale_info = renderer.get_scale_info()
    print(f"  Renderer scale info:")
    for name, info in scale_info.items():
        print(f"    {name}: resolution={info.get('resolution', 'N/A')}")

    # ── Load dataset ──
    print("\n[2] Loading data...")
    feat_dir = cfg['data']['train_feature_dir']
    traj_path = cfg['data']['train_traj_path']
    
    # Load poses
    poses_c2w = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses_c2w.append(np.array(vals).reshape(4, 4))
    poses_c2w = np.array(poses_c2w)
    
    # Load train indices
    train_indices = np.load(os.path.join(feat_dir, 'train_indices.npy'))
    
    # Pick a few training frames
    sample_idx = train_indices[:4]
    print(f"  Using frames: {sample_idx}")
    
    # Load features for these frames
    scale_dirs = {'coarse': 'sd_s5', 'mid': 'sd_s4', 'fine_sd': 'sd_s3', 'fine_dino': 'dino'}
    alt_scale_dirs = {'coarse': 'coarse', 'mid': 'mid', 'fine_sd': 'fine_sd', 'fine_dino': 'fine_dino'}
    
    query_feats = {}
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        feats_list = []
        for idx in sample_idx:
            # Try both naming conventions
            for sdir in [scale_dirs[scale], alt_scale_dirs[scale]]:
                pattern = f"rgb_{idx}_*.pt"
                scale_dir = os.path.join(feat_dir, sdir)
                if os.path.isdir(scale_dir):
                    files = [f for f in os.listdir(scale_dir) if f.startswith(f"rgb_{idx}_")]
                    if files:
                        feat = torch.load(os.path.join(scale_dir, files[0]), map_location='cpu')
                        feats_list.append(feat)
                        break
        if feats_list:
            query_feats[scale] = torch.stack(feats_list).to(device)
            print(f"  {scale}: {query_feats[scale].shape}")
    
    B = len(sample_idx)
    
    # GT poses (w2c)
    poses_gt_c2w = torch.tensor(poses_c2w[sample_idx], dtype=torch.float32, device=device)
    poses_gt_w2c = torch.linalg.inv(poses_gt_c2w)
    
    # ── Test 1: Geometry solver with GT flow ──
    print("\n" + "=" * 70)
    print("TEST 1: Geometry Solver with GT Flow")
    print("=" * 70)
    
    FINE_HW = tuple(mcfg['fine_hw'])
    fx_base = rcfg['fx']
    fy_base = rcfg['fy']
    cx_base = rcfg['cx']
    cy_base = rcfg['cy']
    img_h = rcfg['img_height']
    img_w = rcfg['img_width']
    
    geo_intrinsics = {
        'fx': fx_base * FINE_HW[1] / img_w,
        'fy': fy_base * FINE_HW[0] / img_h,
        'cx': cx_base * FINE_HW[1] / img_w,
        'cy': cy_base * FINE_HW[0] / img_h,
    }
    print(f"  Fine intrinsics: {geo_intrinsics}")
    
    for noise_deg in [1.0, 3.0, 5.0, 10.0]:
        noise_trans = noise_deg * 0.05  # rough scaling
        poses_noisy = add_noise(poses_gt_w2c, noise_deg, noise_trans)
        
        # Render depth at noisy pose
        with torch.no_grad():
            depth = renderer.render_depth_batch(poses_noisy)
            
        # Resize depth to FINE_HW
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=FINE_HW, mode='bilinear', align_corners=False)
        depth = depth.squeeze(1)
        
        print(f"\n  Noise: {noise_deg}° / {noise_trans:.2f}m")
        print(f"  Depth range: {depth.min().item():.2f} - {depth.max().item():.2f}")
        print(f"  Depth valid: {(depth > 0.05).float().mean().item():.1%}")
        
        # Compute GT flow at noisy pose depth
        H, W = FINE_HW
        fx_s = geo_intrinsics['fx']
        fy_s = geo_intrinsics['fy']
        cx_s = geo_intrinsics['cx']
        cy_s = geo_intrinsics['cy']
        
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij')
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)
        
        X = (u_coords - cx_s) / fx_s * depth
        Y = (v_coords - cy_s) / fy_s * depth
        Z = depth
        
        pts = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
        pts_flat = pts.reshape(B, -1, 4).permute(0, 2, 1)
        
        T_rel = poses_gt_w2c @ torch.linalg.inv(poses_noisy)
        pts_gt_cam = torch.bmm(T_rel[:, :3, :], pts_flat).reshape(B, 3, H, W)
        
        Z_gt = pts_gt_cam[:, 2:3].clamp(min=0.01)
        u_gt = fx_s * pts_gt_cam[:, 0:1] / Z_gt + cx_s
        v_gt = fy_s * pts_gt_cam[:, 1:2] / Z_gt + cy_s
        
        flow_gt = torch.cat([u_gt - u_coords.unsqueeze(1),
                              v_gt - v_coords.unsqueeze(1)], dim=1)
        
        valid_mask = (depth.unsqueeze(1) > 0.05) & (pts_gt_cam[:, 2:3] > 0.1)
        flow_gt = flow_gt * valid_mask.float()
        
        flow_mag = flow_gt.pow(2).sum(1).sqrt()
        print(f"  GT flow magnitude: mean={flow_mag[valid_mask.squeeze(1)].mean().item():.2f}, "
              f"max={flow_mag[valid_mask.squeeze(1)].max().item():.2f}")
        
        # Run geometry solver with GT flow
        Ju, Jv, valid = compute_image_jacobian(depth, geo_intrinsics)
        
        conf = valid_mask.float().reshape(B, 1, H, W)
        
        delta_xi = diff_pose_solve(
            flow_gt, conf, Ju, Jv, valid, damping=0.001)
        
        T_delta = se3_exp(delta_xi)
        pose_updated = T_delta @ poses_noisy
        
        # Measure improvement
        err_before = angle_error(poses_noisy[:, :3, :3], poses_gt_w2c[:, :3, :3])
        err_after = angle_error(pose_updated[:, :3, :3], poses_gt_w2c[:, :3, :3])
        
        t_err_before = trans_error(poses_noisy[:, :3, 3], poses_gt_w2c[:, :3, 3])
        t_err_after = trans_error(pose_updated[:, :3, 3], poses_gt_w2c[:, :3, 3])
        
        print(f"  Rot error: {err_before.mean().item():.3f}° → {err_after.mean().item():.3f}°")
        print(f"  Trans error: {t_err_before.mean().item():.4f}m → {t_err_after.mean().item():.4f}m")
        print(f"  delta_xi: {delta_xi[0].cpu().numpy()}")
        
        # Iterate multiple times
        pose_iter = poses_noisy.clone()
        for i in range(5):
            # Re-render depth
            with torch.no_grad():
                d_iter = renderer.render_depth_batch(pose_iter)
            if d_iter.ndim == 3:
                d_iter = d_iter.unsqueeze(1)
            d_iter = F.interpolate(d_iter, size=FINE_HW, mode='bilinear', align_corners=False).squeeze(1)
            
            # Recompute GT flow at current pose
            X_i = (u_coords - cx_s) / fx_s * d_iter
            Y_i = (v_coords - cy_s) / fy_s * d_iter
            pts_i = torch.stack([X_i, Y_i, d_iter, torch.ones_like(d_iter)], dim=-1)
            pts_i_flat = pts_i.reshape(B, -1, 4).permute(0, 2, 1)
            T_rel_i = poses_gt_w2c @ torch.linalg.inv(pose_iter)
            pts_gt_i = torch.bmm(T_rel_i[:, :3, :], pts_i_flat).reshape(B, 3, H, W)
            Z_gt_i = pts_gt_i[:, 2:3].clamp(min=0.01)
            u_gt_i = fx_s * pts_gt_i[:, 0:1] / Z_gt_i + cx_s
            v_gt_i = fy_s * pts_gt_i[:, 1:2] / Z_gt_i + cy_s
            flow_gt_i = torch.cat([u_gt_i - u_coords.unsqueeze(1),
                                    v_gt_i - v_coords.unsqueeze(1)], dim=1)
            valid_i = (d_iter.unsqueeze(1) > 0.05) & (pts_gt_i[:, 2:3] > 0.1)
            flow_gt_i = flow_gt_i * valid_i.float()
            
            Ju_i, Jv_i, val_i = compute_image_jacobian(d_iter, geo_intrinsics)
            conf_i = valid_i.float().reshape(B, 1, H, W)
            xi_i = diff_pose_solve(flow_gt_i, conf_i, Ju_i, Jv_i, val_i, damping=0.001)
            T_d_i = se3_exp(xi_i)
            pose_iter = T_d_i @ pose_iter
            
            rot_err = angle_error(pose_iter[:, :3, :3], poses_gt_w2c[:, :3, :3])
            t_err = trans_error(pose_iter[:, :3, 3], poses_gt_w2c[:, :3, 3])
            print(f"    iter {i+1}: rot={rot_err.mean().item():.4f}° trans={t_err.mean().item():.6f}m")

    # ── Test 2: Feature quality check ──
    print("\n" + "=" * 70)
    print("TEST 2: Feature Quality (3DGS Rendering vs Stored)")
    print("=" * 70)
    
    with torch.no_grad():
        render_out = renderer.render_batch(poses_gt_w2c)
    
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        rkey = f'{scale}_feat'
        if rkey in render_out and scale in query_feats:
            rendered = render_out[rkey]
            stored = query_feats[scale]
            
            # Resize to match if needed
            if rendered.shape[-2:] != stored.shape[-2:]:
                rendered = F.interpolate(rendered, size=stored.shape[-2:],
                                          mode='bilinear', align_corners=False)
            
            cos_sim = F.cosine_similarity(rendered, stored, dim=1).mean().item()
            l2_dist = (rendered - stored).pow(2).mean().sqrt().item()
            print(f"  {scale}: cos_sim={cos_sim:.4f}, L2={l2_dist:.4f}")
    
    # ── Test 3: Model inference with loaded checkpoint ──
    print("\n" + "=" * 70)
    print("TEST 3: Model Forward Pass Check")
    print("=" * 70)
    
    # Check for existing checkpoint
    ckpt_paths = [
        'output/gsff_oldhospital/checkpoints/latest.pth',
        'output/gsff_oldhospital_render/checkpoints/latest.pth',
    ]
    
    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            print(f"  {ckpt_path}: not found, skipping")
            continue
            
        print(f"\n  Loading: {ckpt_path}")
        model = MSFlowPoseNet(
            coarse_hw=mcfg['coarse_hw'],
            mid_hw=mcfg['mid_hw'],
            fine_hw=mcfg['fine_hw'],
            hidden_dim=mcfg.get('hidden_dim', 128),
            decode_dim=mcfg.get('decode_dim', 64),
            local_radius=mcfg.get('local_radius', 4),
            fine_iters=mcfg.get('fine_iters', 8),
            damping=mcfg.get('damping', 0.001),
            coarse_in_dim=mcfg.get('coarse_in_dim', 32),
            mid_in_dim=mcfg.get('mid_in_dim', 64),
            fine_sd_in_dim=mcfg.get('fine_sd_in_dim', 64),
            fine_dino_in_dim=mcfg.get('fine_dino_in_dim', 64),
            intrinsics={'fx': fx_base, 'fy': fy_base, 'cx': cx_base, 'cy': cy_base},
            img_hw=(img_h, img_w),
        ).to(device)
        
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        
        for noise_deg in [3.0, 10.0]:
            noise_trans = noise_deg * 0.05
            poses_noisy = add_noise(poses_gt_w2c, noise_deg, noise_trans)
            
            # Render at noisy pose
            with torch.no_grad():
                render_out = renderer.render_batch(poses_noisy)
                depth = render_out.get('depth_map', renderer.render_depth_batch(poses_noisy))
                
                if depth.ndim == 3:
                    depth_in = depth.unsqueeze(1)
                else:
                    depth_in = depth
                depth_fine = F.interpolate(depth_in, size=FINE_HW, 
                                           mode='bilinear', align_corners=False).squeeze(1)
                
                # Resize render feats to match model expectations
                r_feats = {}
                for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
                    rkey = f'{scale}_feat'
                    if rkey in render_out:
                        r_feats[scale] = render_out[rkey]
                
                # Resize query feats to match rendered
                q_feats = {}
                for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
                    if scale in query_feats:
                        q_feats[scale] = query_feats[scale]
                        
                pred = model(q_feats, r_feats, depth_fine)
                
                flow_pred = pred['flow_fine']
                
                # Compute GT flow for comparison
                flow_gt_check, valid_check = model.compute_gt_flow(
                    poses_noisy, poses_gt_w2c, depth_fine, FINE_HW)
                
                # Flow EPE
                diff = (flow_pred - flow_gt_check).pow(2).sum(1).sqrt()
                valid_1d = valid_check.squeeze(1) > 0.5
                epe = diff[valid_1d].mean().item() if valid_1d.any() else float('nan')
                
                # Flow direction cosine similarity
                flow_cos = F.cosine_similarity(
                    flow_pred.reshape(B, 2, -1), 
                    flow_gt_check.reshape(B, 2, -1), dim=1)
                flow_dir_sim = flow_cos.mean().item()
                
                # Pose update
                if 'delta_xi' in pred:
                    T_delta = se3_exp(pred['delta_xi'].float())
                    pose_updated = T_delta @ poses_noisy
                    
                    rot_before = angle_error(poses_noisy[:, :3, :3], poses_gt_w2c[:, :3, :3])
                    rot_after = angle_error(pose_updated[:, :3, :3], poses_gt_w2c[:, :3, :3])
                    
                    print(f"  Noise {noise_deg}°: flow_epe={epe:.2f}, flow_dir={flow_dir_sim:.3f}, "
                          f"rot {rot_before.mean().item():.2f}° → {rot_after.mean().item():.2f}°")
                else:
                    print(f"  Noise {noise_deg}°: flow_epe={epe:.2f}, flow_dir={flow_dir_sim:.3f}, "
                          f"no delta_xi")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
