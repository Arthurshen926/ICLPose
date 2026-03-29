#!/usr/bin/env python3
"""Diagnostic: test if the geometry solver can recover pose from GT flow."""
import sys
sys.path.insert(0, '.')

import torch
import yaml
import math
import numpy as np
from modules.geometry_solver import compute_image_jacobian, diff_pose_solve_sequential
from modules.lie_algebra import se3_exp, se3_log

def main():
    device = torch.device('cuda')
    
    # OldHospital intrinsics at solver resolution (68x120)
    sH, sW = 68, 120
    fx = 1663.12 * sW / 1920
    fy = 1663.12 * sH / 1080
    cx = 960.0 * sW / 1920
    cy = 540.0 * sH / 1080
    intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    print(f"Solver resolution: {sH}x{sW}, fx={fx:.2f}, fy={fy:.2f}")
    
    # Simulate depth (OldHospital: 4-44m, mean ~33m)
    depth = torch.ones(1, sH, sW, device=device) * 30.0  # uniform 30m
    
    # Test: apply known rotation and see if solver recovers it
    for test_angle_deg in [0.5, 1.0, 2.0, 4.0, 8.0]:
        # Create ground truth xi: pure rotation around Y axis
        angle_rad = test_angle_deg * math.pi / 180.0
        gt_xi = torch.zeros(1, 6, device=device)
        gt_xi[0, 4] = angle_rad  # wy
        
        # Convert to transformation matrix  
        T_delta = se3_exp(gt_xi)
        
        # Compute GT flow from this pose change
        # Create pixel grid at solver resolution
        gy, gx = torch.meshgrid(
            torch.arange(sH, device=device, dtype=torch.float32),
            torch.arange(sW, device=device, dtype=torch.float32), indexing='ij')
        
        # Unproject to 3D using depth
        Z = depth[0]
        X = (gx - cx) * Z / fx
        Y = (gy - cy) * Z / fy
        pts = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)
        
        # Transform points
        R = T_delta[0, :3, :3]
        t = T_delta[0, :3, 3]
        pts_flat = pts.reshape(-1, 3)
        pts_transformed = (R @ pts_flat.T).T + t
        pts_transformed = pts_transformed.reshape(sH, sW, 3)
        
        # Reproject
        X2 = pts_transformed[:, :, 0]
        Y2 = pts_transformed[:, :, 1]
        Z2 = pts_transformed[:, :, 2]
        u2 = fx * X2 / Z2 + cx
        v2 = fy * Y2 / Z2 + cy
        
        # Flow = reprojected - original
        flow_u = u2 - gx
        flow_v = v2 - gy
        gt_flow = torch.stack([flow_u, flow_v], dim=0).unsqueeze(0)  # (1, 2, H, W)
        
        print(f"\n--- Test: {test_angle_deg}° rotation around Y ---")
        print(f"  GT flow magnitude: {gt_flow.abs().mean():.4f} px, max: {gt_flow.abs().max():.4f} px")
        
        # Now run the solver
        confidence = torch.ones(1, 1, sH, sW, device=device)
        Ju, Jv, valid = compute_image_jacobian(depth[0:1], intrinsics)
        
        delta_xi = diff_pose_solve_sequential(
            gt_flow, confidence, Ju, Jv, valid, damping=0.001)
        
        # Check recovered rotation
        recovered_angle_rad = delta_xi[0, 3:].norm().item()
        recovered_angle_deg = recovered_angle_rad * 180.0 / math.pi
        
        # Apply recovered xi and compute pose error
        T_recovered = se3_exp(delta_xi)
        T_composed = T_recovered @ torch.eye(4, device=device).unsqueeze(0)
        T_error = T_delta @ torch.inverse(T_composed)
        
        R_err = T_error[0, :3, :3]
        trace = R_err[0, 0] + R_err[1, 1] + R_err[2, 2]
        cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        residual_deg = torch.acos(cos_a).item() * 180.0 / math.pi
        
        print(f"  GT xi:        [{gt_xi[0].cpu().numpy()}]")
        print(f"  Recovered xi: [{delta_xi[0].detach().cpu().numpy()}]")
        print(f"  Recovered rot: {recovered_angle_deg:.4f}°, residual: {residual_deg:.4f}°")
        
    # Also test with noisy flow
    print(f"\n\n=== Now test with NOISY flow (adding Gaussian noise) ===")
    gt_xi = torch.zeros(1, 6, device=device)
    gt_xi[0, 4] = 4.0 * math.pi / 180.0  # 4° around Y
    T_delta = se3_exp(gt_xi)
    
    # Recompute GT flow
    Z = depth[0]
    X = (gx - cx) * Z / fx  
    Y = (gy - cy) * Z / fy
    pts = torch.stack([X, Y, Z], dim=-1)
    R = T_delta[0, :3, :3]
    t = T_delta[0, :3, 3]
    pts_flat = pts.reshape(-1, 3)
    pts_transformed = (R @ pts_flat.T).T + t
    pts_transformed = pts_transformed.reshape(sH, sW, 3)
    X2 = pts_transformed[:, :, 0]
    Y2 = pts_transformed[:, :, 1]
    Z2 = pts_transformed[:, :, 2]
    u2 = fx * X2 / Z2 + cx
    v2 = fy * Y2 / Z2 + cy
    flow_u = u2 - gx
    flow_v = v2 - gy
    gt_flow = torch.stack([flow_u, flow_v], dim=0).unsqueeze(0)
    
    Ju, Jv, valid = compute_image_jacobian(depth[0:1], intrinsics)
    
    for noise_std in [0.0, 0.5, 1.0, 2.0, 5.0]:
        noisy_flow = gt_flow + torch.randn_like(gt_flow) * noise_std
        confidence = torch.ones(1, 1, sH, sW, device=device)
        delta_xi = diff_pose_solve_sequential(
            noisy_flow, confidence, Ju, Jv, valid, damping=0.001)
        
        T_recovered = se3_exp(delta_xi)
        T_error = T_delta @ torch.inverse(T_recovered)
        R_err = T_error[0, :3, :3]
        trace = R_err[0, 0] + R_err[1, 1] + R_err[2, 2]
        cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        residual_deg = torch.acos(cos_a).item() * 180.0 / math.pi
        
        print(f"  Noise std={noise_std:.1f}px → residual: {residual_deg:.4f}°, " +
              f"recovered wy: {delta_xi[0, 4].item()*180/math.pi:.4f}°")

if __name__ == '__main__':
    main()
