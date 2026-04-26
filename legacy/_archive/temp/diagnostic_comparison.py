#!/usr/bin/env python3
"""Compare test views with their nearest training views to understand the core quality gap.
Generates side-by-side images: test_GT | nearest_train | rendered"""
import sys, os, math, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_3dgs.train_2dgs_geometry import (
    GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor
)
from torchvision.utils import save_image

def get_cam_center(cam):
    W2C = np.eye(4); W2C[:3, :3] = cam.R.T; W2C[:3, 3] = cam.T
    C2W = np.linalg.inv(W2C)
    return C2W[:3, 3]

def main():
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene('dataset/OldHospital')
    
    # Load Gaussians
    gaussians = GaussianModel2DGS(3)
    gaussians.load_ply('output/2dgs_models/OldHospital/v3_retrain31/point_cloud/iteration_15000/point_cloud.ply')
    bg_color = torch.zeros(3, device='cuda')
    
    train_centers = np.array([get_cam_center(c) for c in train_cams])
    train_names = [c.image_name for c in train_cams]
    
    # Target: worst test views
    worst_test_names = [
        'seq4/frame00054.png',  # 11.22 dB
        'seq4/frame00053.png',  # 11.88 dB
        'seq4/frame00051.png',  # 13.13 dB
        'seq8/frame00011.png',  # 13.34 dB
        'seq4/frame00001.png',  # 19.73 (best for comparison)
        'seq8/frame00094.png',  # 20.43 (best overall)
    ]
    
    out_dir = 'output/visualization/diagnostic'
    os.makedirs(out_dir, exist_ok=True)
    
    for tname in worst_test_names:
        test_cam = next((c for c in test_cams if c.image_name == tname), None)
        if test_cam is None:
            print(f"SKIP: {tname} not found")
            continue
        
        tc = get_cam_center(test_cam)
        dists = np.linalg.norm(train_centers - tc, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_cam = train_cams[nearest_idx]
        
        print(f"\n{'='*60}")
        print(f"Test: {tname}")
        print(f"Nearest train: {nearest_cam.image_name} (dist={dists[nearest_idx]:.2f}m)")
        
        with torch.no_grad():
            # Render from test viewpoint
            test_pkg = render_2dgs(gaussians, test_cam, bg_color, longest_edge=0)
            test_rendered = test_pkg["render"].clamp(0, 1)
            rw, rh = test_pkg["width"], test_pkg["height"]
            
            # Load test GT
            test_gt = load_image_tensor(test_cam)
            test_gt = F.interpolate(test_gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)
            
            # Render from nearest train viewpoint
            train_pkg = render_2dgs(gaussians, nearest_cam, bg_color, longest_edge=0)
            train_rendered = train_pkg["render"].clamp(0, 1)
            trw, trh = train_pkg["width"], train_pkg["height"]
            
            # Load train GT
            train_gt = load_image_tensor(nearest_cam)  
            train_gt = F.interpolate(train_gt.unsqueeze(0), size=(trh, trw), mode="bilinear", align_corners=False).squeeze(0)
            
            # Compute PSNRs
            test_mse = F.mse_loss(test_rendered, test_gt).item()
            test_psnr = -10 * math.log10(test_mse) if test_mse > 0 else 0
            
            train_mse = F.mse_loss(train_rendered, train_gt).item()
            train_psnr = -10 * math.log10(train_mse) if train_mse > 0 else 0
            
            # Also compute cross: how well does the train GT match test GT?
            # Resize to same size
            train_gt_resized = F.interpolate(train_gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)
            cross_mse = F.mse_loss(train_gt_resized, test_gt).item()
            cross_psnr = -10 * math.log10(cross_mse) if cross_mse > 0 else 0
            
            print(f"  Test PSNR (rendered vs GT): {test_psnr:.2f} dB")
            print(f"  Train PSNR (rendered vs GT): {train_psnr:.2f} dB")
            print(f"  Cross PSNR (train GT vs test GT): {cross_psnr:.2f} dB")
            print(f"  → If cross PSNR is also low, it means the scene changed between captures!")
            
            # Error map
            error = ((test_rendered - test_gt) ** 2).mean(dim=0, keepdim=True).sqrt()
            error_vis = error.repeat(3, 1, 1)  # grayscale error
            error_vis = (error_vis / 0.2).clamp(0, 1)  # normalize
            
            # Save comparison: test_GT | rendered_test | error | train_GT | rendered_train
            # Resize train images to match test height
            if trh != rh:
                train_gt_vis = F.interpolate(train_gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)
                train_rendered_vis = F.interpolate(train_rendered.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)
            else:
                train_gt_vis = train_gt
                train_rendered_vis = train_rendered
            
            row = torch.cat([test_gt, test_rendered, error_vis, train_gt_vis, train_rendered_vis], dim=2)
            
            safe_name = tname.replace('/', '_').replace('.png', '')
            save_image(row, os.path.join(out_dir, f'{safe_name}.png'))
            print(f"  Saved: {safe_name}.png")
            print(f"  Layout: [test_GT | test_rendered | error | train_GT | train_rendered]")

if __name__ == "__main__":
    main()
