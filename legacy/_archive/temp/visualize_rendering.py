#!/usr/bin/env python3
"""Visualize 2DGS rendering quality — render test views, compare with GT, save error maps.

Usage:
    python _archive/temp/visualize_rendering.py \
        --model_dir output/2dgs_models/OldHospital/v3_retrain31 \
        --iteration 15000 \
        --source_dir dataset/OldHospital \
        --output_dir output/visualization/retrain31_15k \
        --num_views 20
"""
import argparse
import math
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_3dgs.train_2dgs_geometry import (
    GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor, AppearanceNetwork
)


def make_error_heatmap(error_map, max_val=0.15):
    """Convert single-channel error map [1,H,W] to colored heatmap [3,H,W]."""
    err = error_map.squeeze(0).clamp(0, max_val) / max_val  # [H,W] in [0,1]
    # Blue->Green->Yellow->Red
    r = torch.clamp(2 * err - 0.5, 0, 1)
    g = torch.where(err < 0.5, 2 * err, 2 * (1 - err))
    b = torch.clamp(1 - 2 * err, 0, 1)
    return torch.stack([r, g, b], dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_views", type=int, default=0, help="0=all test views")
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load scene
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene(args.source_dir)
    print(f"  Train: {len(train_cams)}, Test: {len(test_cams)}")

    # Load Gaussians
    print(f"Loading Gaussians from iteration {args.iteration}...")
    gaussians = GaussianModel2DGS(args.sh_degree)
    ply_path = os.path.join(args.model_dir, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    print(f"  {gaussians.get_xyz.shape[0]} Gaussians loaded")

    # Try loading appearance network
    app_path = os.path.join(args.model_dir, "point_cloud", f"iteration_{args.iteration}", "appearance_net.pth")
    appearance_net = None
    if os.path.exists(app_path):
        ckpt = torch.load(app_path, map_location=device)
        appearance_net = AppearanceNetwork(
            n_images=ckpt.get('n_images', len(train_cams)),
            embed_dim=ckpt.get('embed_dim', 32),
            scale_range=ckpt.get('scale_range', 0.4),
            bias_range=ckpt.get('bias_range', 0.05),
        ).to(device)
        appearance_net.load_state_dict(ckpt['state_dict'])
        appearance_net.eval()
        print(f"  Appearance network loaded (scale_range={ckpt.get('scale_range')}, bias_range={ckpt.get('bias_range')})")

    bg_color = torch.zeros(3, device=device)

    # Select views
    views = test_cams
    if args.num_views > 0:
        # Sample evenly across test set
        indices = np.linspace(0, len(views) - 1, args.num_views, dtype=int)
        views = [views[i] for i in indices]

    print(f"\nRendering {len(views)} test views...")
    
    psnrs = []
    per_view_data = []

    from torchvision.utils import save_image
    
    for i, cam in enumerate(views):
        with torch.no_grad():
            render_pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=0)
            rendered = render_pkg["render"].clamp(0, 1)  # [3, H, W]
            rw, rh = render_pkg["width"], render_pkg["height"]

            gt = load_image_tensor(cam)
            gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)

            # Per-pixel error
            error = ((rendered - gt) ** 2).mean(dim=0, keepdim=True)  # [1, H, W]
            rmse = error.sqrt()
            
            mse_val = F.mse_loss(rendered, gt).item()
            psnr = -10 * math.log10(mse_val) if mse_val > 0 else 0
            psnrs.append(psnr)

            # Error heatmap
            heatmap = make_error_heatmap(rmse, max_val=0.2)

            # Side-by-side: GT | Rendered | Error
            # Resize all to same height for concat
            comparison = torch.cat([gt, rendered, heatmap], dim=2)  # [3, H, 3*W]

            name = cam.image_name.replace("/", "_").replace("\\", "_")
            save_image(comparison, os.path.join(args.output_dir, f"{i:03d}_{name}_psnr{psnr:.1f}.png"))
            
            per_view_data.append((name, psnr, mse_val))
            
            if (i + 1) % 10 == 0 or i == len(views) - 1:
                print(f"  [{i+1}/{len(views)}] {name}: PSNR={psnr:.2f} dB")

    # Summary
    avg_psnr = np.mean(psnrs)
    print(f"\n{'='*60}")
    print(f"Average PSNR: {avg_psnr:.2f} dB ({len(views)} views)")
    print(f"Min PSNR: {min(psnrs):.2f} dB")
    print(f"Max PSNR: {max(psnrs):.2f} dB")
    print(f"Std PSNR: {np.std(psnrs):.2f} dB")
    
    # Worst and best views
    sorted_data = sorted(per_view_data, key=lambda x: x[1])
    print(f"\n--- Worst 10 views ---")
    for name, psnr, mse in sorted_data[:10]:
        print(f"  {name}: {psnr:.2f} dB")
    print(f"\n--- Best 10 views ---")
    for name, psnr, mse in sorted_data[-10:]:
        print(f"  {name}: {psnr:.2f} dB")

    # Save per-view PSNR to CSV
    csv_path = os.path.join(args.output_dir, "per_view_psnr.csv")
    with open(csv_path, "w") as f:
        f.write("view,psnr,mse\n")
        for name, psnr, mse in per_view_data:
            f.write(f"{name},{psnr:.4f},{mse:.8f}\n")
    
    print(f"\nSaved {len(views)} visualizations to {args.output_dir}")
    print(f"Per-view PSNR saved to {csv_path}")


if __name__ == "__main__":
    main()
