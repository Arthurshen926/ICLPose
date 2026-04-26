#!/usr/bin/env python3
"""Diagnostic: Evaluate PSNR with and without appearance correction.

Tests the hypothesis that Gaussians have learned good geometry
but the appearance network absorbs exposure differences that
are not applied during eval, dragging down the metric.

Also evaluates per-view PSNR to find patterns.

Usage:
    python scripts/eval_appearance_diagnostic.py \
        --source_dir dataset/OldHospital \
        --model_dir output/2dgs_models/OldHospital/v3_retrain17 \
        --iteration 15000 \
        [--optimize_test_appearance]
"""

import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=15000)
    parser.add_argument("--optimize_test_appearance", action="store_true",
                        help="Optimize per-test-image affine params (scale+bias) to find best-case PSNR")
    args = parser.parse_args()

    # Import training module
    from feature_3dgs.train_2dgs_geometry import (
        GaussianModel2DGS, AppearanceNetwork, load_scene, render_2dgs, load_image_tensor
    )

    # Load scene
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene(args.source_dir)
    print(f"  Test cameras: {len(test_cams)}")

    # Load Gaussians
    ply_path = os.path.join(args.model_dir, "point_cloud",
                            f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"ERROR: Checkpoint not found: {ply_path}")
        return

    gaussians = GaussianModel2DGS(sh_degree=3)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = 3
    print(f"  Loaded {gaussians.num_points:,} Gaussians from iter {args.iteration}")

    # Load appearance network if available
    app_path = os.path.join(args.model_dir, "appearance_network.pth")
    appearance_net = None
    if os.path.exists(app_path):
        state = torch.load(app_path, map_location="cuda")
        n_images = state["embedding.weight"].shape[0]
        appearance_net = AppearanceNetwork(n_images).cuda()
        appearance_net.load_state_dict(state)
        appearance_net.eval()
        print(f"  Loaded appearance network ({n_images} embeddings)")

    bg_color = torch.zeros(3, device="cuda")

    # Build image name → index mapping (for training images)
    cam_name_to_idx = {}
    for i, cam in enumerate(train_cams):
        cam_name_to_idx[cam.image_name] = i

    # Evaluate
    raw_psnrs = []
    optimized_psnrs = []
    results = []

    print("\nEvaluating test views...")
    with torch.no_grad():
        for i, cam in enumerate(test_cams):
            render_pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=0)
            image = render_pkg["render"].clamp(0, 1)
            rw, rh = render_pkg["width"], render_pkg["height"]
            gt_image = load_image_tensor(cam)
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=(rh, rw),
                                     mode="bilinear", align_corners=False).squeeze(0)

            # Raw PSNR
            mse_raw = F.mse_loss(image, gt_image).item()
            psnr_raw = -10 * math.log10(mse_raw) if mse_raw > 0 else 100
            raw_psnrs.append(psnr_raw)

            result = {"name": cam.image_name, "psnr_raw": psnr_raw}

            # Appearance-corrected PSNR (use training image's embedding if available)
            if appearance_net is not None and cam.image_name in cam_name_to_idx:
                idx = cam_name_to_idx[cam.image_name]
                corrected = appearance_net(image, idx).clamp(0, 1)
                mse_app = F.mse_loss(corrected, gt_image).item()
                psnr_app = -10 * math.log10(mse_app) if mse_app > 0 else 100
                result["psnr_appearance"] = psnr_app

            results.append(result)

    # Optimize per-test-image appearance (finds best-case PSNR)
    if args.optimize_test_appearance:
        print("\nOptimizing per-test-image appearance (scale+bias)...")
        for i, cam in enumerate(test_cams):
            render_pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=0)
            image = render_pkg["render"].clamp(0, 1).detach()
            rw, rh = render_pkg["width"], render_pkg["height"]
            gt_image = load_image_tensor(cam)
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=(rh, rw),
                                     mode="bilinear", align_corners=False).squeeze(0)

            # Optimize scale [3] and bias [3] to minimize MSE
            scale = torch.ones(3, 1, 1, device="cuda", requires_grad=True)
            bias = torch.zeros(3, 1, 1, device="cuda", requires_grad=True)
            optimizer = torch.optim.Adam([scale, bias], lr=0.01)

            for step in range(200):
                adjusted = image * scale + bias
                loss = F.mse_loss(adjusted.clamp(0, 1), gt_image)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                adjusted = (image * scale + bias).clamp(0, 1)
                mse_opt = F.mse_loss(adjusted, gt_image).item()
                psnr_opt = -10 * math.log10(mse_opt) if mse_opt > 0 else 100
                optimized_psnrs.append(psnr_opt)
                results[i]["psnr_optimized"] = psnr_opt
                results[i]["opt_scale"] = scale.squeeze().cpu().tolist()
                results[i]["opt_bias"] = bias.squeeze().cpu().tolist()

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(test_cams)} views processed")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)
    avg_raw = np.mean(raw_psnrs)
    print(f"  Raw PSNR:           {avg_raw:.2f} dB ({len(raw_psnrs)} views)")

    if any("psnr_appearance" in r for r in results):
        app_vals = [r["psnr_appearance"] for r in results if "psnr_appearance" in r]
        print(f"  Appearance PSNR:    {np.mean(app_vals):.2f} dB ({len(app_vals)} views)")
        print(f"  Gain from app:      {np.mean(app_vals) - avg_raw:+.2f} dB")

    if optimized_psnrs:
        avg_opt = np.mean(optimized_psnrs)
        print(f"  Optimized PSNR:     {avg_opt:.2f} dB ({len(optimized_psnrs)} views)")
        print(f"  Gain from optimize: {avg_opt - avg_raw:+.2f} dB")
        print(f"  → This is the UPPER BOUND of what appearance correction can achieve")

    # Per-view details for worst/best
    sorted_results = sorted(results, key=lambda x: x["psnr_raw"])
    print(f"\nWorst 10 views (raw):")
    for r in sorted_results[:10]:
        line = f"  {r['name']}: raw={r['psnr_raw']:.2f}"
        if "psnr_optimized" in r:
            line += f" → opt={r['psnr_optimized']:.2f} (+{r['psnr_optimized']-r['psnr_raw']:.2f})"
            line += f" scale={[f'{s:.3f}' for s in r['opt_scale']]}"
        print(line)

    print(f"\nBest 10 views (raw):")
    for r in sorted_results[-10:]:
        line = f"  {r['name']}: raw={r['psnr_raw']:.2f}"
        if "psnr_optimized" in r:
            line += f" → opt={r['psnr_optimized']:.2f} (+{r['psnr_optimized']-r['psnr_raw']:.2f})"
        print(line)

    # Histogram
    bins = [0, 14, 16, 18, 20, 22, 100]
    labels = ["<14", "14-16", "16-18", "18-20", "20-22", "22+"]
    print(f"\nPSNR distribution (raw):")
    for j in range(len(labels)):
        count = sum(1 for p in raw_psnrs if bins[j] <= p < bins[j+1])
        bar = "█" * count
        print(f"  {labels[j]:>6}: {count:3d} {bar}")

    if optimized_psnrs:
        print(f"\nPSNR distribution (optimized):")
        for j in range(len(labels)):
            count = sum(1 for p in optimized_psnrs if bins[j] <= p < bins[j+1])
            bar = "█" * count
            print(f"  {labels[j]:>6}: {count:3d} {bar}")


if __name__ == "__main__":
    main()
