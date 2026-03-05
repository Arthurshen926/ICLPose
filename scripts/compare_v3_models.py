#!/usr/bin/env python3
"""Side-by-side comparison of two 2DGS models on the same views."""
import argparse
import os
import sys
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(ply_path):
    """Load Gaussian model from PLY."""
    from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS
    gaussians = GaussianModel2DGS(sh_degree=3)

    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    names = [p.name for p in vertex.properties]
    n_sh = sum(1 for n in names if n.startswith('f_rest_'))
    sh_degree = {0: 0, 8: 1, 24: 2, 48: 3}.get(n_sh, 3)

    gaussians = GaussianModel2DGS(sh_degree=sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians


def render_view(gaussians, cam, bg_color, longest_edge):
    """Render a single view."""
    from feature_3dgs.train_2dgs_geometry import render_2dgs
    render_pkg = render_2dgs(cam, gaussians, bg_color, longest_edge)
    return render_pkg["render"].clamp(0, 1)


def psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return -10 * np.log10(mse)


def main():
    parser = argparse.ArgumentParser(description="Compare two 2DGS models side by side")
    parser.add_argument("--ply_a", required=True, help="PLY path for model A")
    parser.add_argument("--ply_b", required=True, help="PLY path for model B")
    parser.add_argument("--label_a", default="Model A", help="Label for model A")
    parser.add_argument("--label_b", default="Model B", help="Label for model B")
    parser.add_argument("--output_dir", default="output/comparison", help="Output directory")
    parser.add_argument("--source_dir", default="dataset/OldHospital", help="Dataset source directory")
    parser.add_argument("--views_per_seq", type=int, default=2, help="Views per sequence")
    parser.add_argument("--longest_edge", type=int, default=1280)
    parser.add_argument("--images", default=".", help="Images subfolder")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load cameras from model_a's cameras.json
    model_dir_a = os.path.dirname(os.path.dirname(os.path.dirname(args.ply_a)))
    cam_json = os.path.join(model_dir_a, "cameras.json")
    if not os.path.exists(cam_json):
        # Try model_b
        model_dir_b = os.path.dirname(os.path.dirname(os.path.dirname(args.ply_b)))
        cam_json = os.path.join(model_dir_b, "cameras.json")
    assert os.path.exists(cam_json), f"cameras.json not found in either model dir"

    # Parse split files
    train_txt = os.path.join(args.source_dir, "dataset_train.txt")
    test_txt = os.path.join(args.source_dir, "dataset_test.txt")
    train_names = set()
    test_names = set()
    if os.path.exists(train_txt):
        with open(train_txt) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    train_names.add(line)
    if os.path.exists(test_txt):
        with open(test_txt) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    test_names.add(line)

    with open(cam_json) as f:
        all_cams_json = json.load(f)

    # Organize cameras by sequence
    from collections import defaultdict
    seq_cams = defaultdict(list)
    for cam_data in all_cams_json:
        name = cam_data['img_name']
        seq = name.split('/')[0]
        split = "test" if name in test_names else "train"
        seq_cams[seq].append((cam_data, split))

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Select views
    selected_cams = []
    for seq_name in sorted(seq_cams.keys()):
        cams = seq_cams[seq_name]
        n = min(args.views_per_seq, len(cams))
        step = max(1, len(cams) // n)
        for i in range(0, min(n * step, len(cams)), step):
            if len(selected_cams) < (len(seq_cams) * args.views_per_seq):
                selected_cams.append((seq_name, cams[i]))

    # Load models
    print(f"\nLoading model A: {args.ply_a}")
    model_a = load_model(args.ply_a)
    print(f"  → {model_a.num_points:,} Gaussians")

    print(f"Loading model B: {args.ply_b}")
    model_b = load_model(args.ply_b)
    print(f"  → {model_b.num_points:,} Gaussians")

    # Render and compare
    results = []
    comparison_images = []

    for idx, (seq_name, cam) in enumerate(selected_cams):
        print(f"  [{idx+1}/{len(selected_cams)}] {cam.image_name}")

        # Ground truth
        gt = cam.original_image.cuda()

        # Render both models
        render_a = render_view(model_a, cam, bg_color, args.longest_edge)
        render_b = render_view(model_b, cam, bg_color, args.longest_edge)

        # Compute PSNR
        gt_np = gt.permute(1, 2, 0).cpu().numpy()
        ra_np = render_a.permute(1, 2, 0).cpu().numpy()
        rb_np = render_b.permute(1, 2, 0).cpu().numpy()

        psnr_a = psnr(gt_np, ra_np)
        psnr_b = psnr(gt_np, rb_np)
        diff = psnr_b - psnr_a

        results.append({
            "seq": seq_name,
            "image": cam.image_name,
            "psnr_a": round(psnr_a, 2),
            "psnr_b": round(psnr_b, 2),
            "diff": round(diff, 2),
        })

        # Create comparison row: GT | Model A | Model B | Diff map
        h, w = gt_np.shape[:2]
        gt_pil = Image.fromarray((gt_np * 255).astype(np.uint8))
        ra_pil = Image.fromarray((ra_np * 255).astype(np.uint8))
        rb_pil = Image.fromarray((rb_np * 255).astype(np.uint8))

        # Error maps (amplified)
        err_a = np.abs(gt_np - ra_np).mean(axis=2)
        err_b = np.abs(gt_np - rb_np).mean(axis=2)
        err_a_pil = Image.fromarray((np.clip(err_a * 5, 0, 1) * 255).astype(np.uint8))
        err_b_pil = Image.fromarray((np.clip(err_b * 5, 0, 1) * 255).astype(np.uint8))

        # Compose row: GT | A | B
        row_w = w * 3 + 20  # 10px padding between each
        row = Image.new("RGB", (row_w, h + 30), (255, 255, 255))

        row.paste(gt_pil, (0, 30))
        row.paste(ra_pil, (w + 10, 30))
        row.paste(rb_pil, (w * 2 + 20, 30))

        # Add labels
        draw = ImageDraw.Draw(row)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((5, 5), f"GT ({seq_name})", fill=(0, 0, 0), font=font)
        draw.text((w + 15, 5), f"{args.label_a}: {psnr_a:.2f} dB", fill=(200, 0, 0), font=font)

        color_b = (0, 150, 0) if diff > 0 else (200, 0, 0)
        draw.text((w * 2 + 25, 5), f"{args.label_b}: {psnr_b:.2f} dB ({diff:+.2f})", fill=color_b, font=font)

        comparison_images.append(row)

        # Save individual comparison
        row.save(os.path.join(args.output_dir, f"compare_{seq_name}_{os.path.basename(cam.image_name)}"))

    # Create overview grid
    if comparison_images:
        total_h = sum(img.height for img in comparison_images) + 10 * (len(comparison_images) - 1)
        max_w = max(img.width for img in comparison_images)
        grid = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for img in comparison_images:
            grid.paste(img, (0, y))
            y += img.height + 10
        grid.save(os.path.join(args.output_dir, "comparison_grid.png"))

    # Summary
    psnrs_a = [r["psnr_a"] for r in results]
    psnrs_b = [r["psnr_b"] for r in results]
    mean_a = np.mean(psnrs_a)
    mean_b = np.mean(psnrs_b)

    print(f"\n{'='*60}")
    print(f"  Comparison: {args.label_a} vs {args.label_b}")
    print(f"{'='*60}")
    for r in results:
        marker = "✓" if r["diff"] > 0 else "✗"
        print(f"  {r['seq']:6s} | {args.label_a}={r['psnr_a']:.2f} | {args.label_b}={r['psnr_b']:.2f} | {r['diff']:+.2f} {marker}")
    print(f"  {'─'*50}")
    print(f"  {'Mean':6s} | {args.label_a}={mean_a:.2f} | {args.label_b}={mean_b:.2f} | {mean_b - mean_a:+.2f}")
    print(f"{'='*60}")

    # Save stats
    stats = {
        "model_a": args.ply_a,
        "model_b": args.ply_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "per_view": results,
        "mean_psnr_a": round(mean_a, 2),
        "mean_psnr_b": round(mean_b, 2),
        "delta": round(mean_b - mean_a, 2),
    }
    with open(os.path.join(args.output_dir, "comparison_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: comparison_stats.json")
    print(f"  Grid saved: comparison_grid.png")


if __name__ == "__main__":
    main()
