#!/usr/bin/env python3
"""Quick visualization: render test views from a 2DGS checkpoint and generate
side-by-side comparison images (GT | Rendered | Error map).

Usage:
  python scripts/visualize_render.py \
    --ckpt output/2dgs_models/OldHospital/v3_retrain10/point_cloud/iteration_20000/point_cloud.ply \
    --source dataset/OldHospital \
    --out output/viz_retrain10_20k \
    --n_views 8

Generates:
  - Per-view comparison images: {view_name}_compare.png
  - Summary grid: summary_grid.png
  - Text report: report.txt
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from feature_3dgs.train_2dgs_geometry import (
    GaussianModel2DGS, load_scene, load_image_tensor, render_2dgs
)
from plyfile import PlyData


def load_gaussians_from_ply(ply_path, sh_degree=3):
    """Load a trained Gaussian model from PLY file."""
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    N = len(v)
    
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    
    # DC features
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
    
    # Rest features
    n_rest = (sh_degree + 1) ** 2 - 1
    prop_names = [p.name for p in v.properties]
    f_rest_list = []
    for i in range(n_rest * 3):
        key = f"f_rest_{i}"
        if key in prop_names:
            f_rest_list.append(v[key])
    if f_rest_list:
        f_rest = np.stack(f_rest_list, axis=1).astype(np.float32)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)
    
    opacity = v["opacity"].astype(np.float32).reshape(-1, 1)
    
    # Scales (2D for 2DGS)
    prop_names_all = [p.name for p in v.properties]
    scale_names = sorted([n for n in prop_names_all if n.startswith("scale_")])
    scales = np.stack([v[n] for n in scale_names], axis=1).astype(np.float32)
    
    rot_names = sorted([n for n in prop_names_all if n.startswith("rot_")])
    rot_names = sorted(rot_names)
    rotations = np.stack([v[n] for n in rot_names], axis=1).astype(np.float32)
    
    # Construct model
    model = GaussianModel2DGS(sh_degree=sh_degree)
    model.active_sh_degree = sh_degree
    
    model._xyz = torch.tensor(xyz, device="cuda")
    # PLY stores: f_dc [N, 3] (transposed+flattened from [N, 1, 3])
    #             f_rest [N, 45] (transposed+flattened from [N, 15, 3])
    # Internal format: _features_dc [N, 1, 3], _features_rest [N, n_coeffs, 3]
    model._features_dc = torch.tensor(f_dc, device="cuda").reshape(N, 3, 1).transpose(1, 2).contiguous()  # [N,1,3]
    
    if f_rest.shape[1] > 0:
        n_coeffs = n_rest  # 15 for sh_degree=3
        model._features_rest = torch.tensor(f_rest, device="cuda").reshape(N, 3, n_coeffs).transpose(1, 2).contiguous()  # [N,15,3]
    else:
        model._features_rest = torch.zeros(N, 0, 3, device="cuda")
    
    model._opacity = torch.tensor(opacity, device="cuda")
    model._scaling = torch.tensor(scales, device="cuda")
    model._rotation = torch.tensor(rotations, device="cuda")
    
    return model


def make_error_heatmap(gt, render, scale=5.0):
    """Create error heatmap [3,H,W] from GT and render [3,H,W] tensors.
    Uses red=high error, blue=low error colormap."""
    err = (gt - render).pow(2).mean(dim=0)  # [H,W] MSE per pixel
    err = (err * scale).clamp(0, 1)  # amplify for visibility
    
    # Simple red-blue heatmap: blue(0) -> yellow(0.5) -> red(1)
    r = err.clamp(0, 1)
    g = (1 - (err - 0.5).abs() * 2).clamp(0, 1) * 0.8
    b = (1 - err).clamp(0, 1)
    return torch.stack([r, g, b], dim=0)


def tensor_to_pil(t, max_edge=960):
    """Convert [3,H,W] tensor (0-1) to PIL Image, optionally downsize."""
    t = t.clamp(0, 1)
    _, h, w = t.shape
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        t = F.interpolate(t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(arr)


def create_comparison(gt_t, render_t, error_t, psnr_val, view_name, max_edge=640):
    """Create side-by-side: GT | Rendered | Error, with PSNR label."""
    gt_img = tensor_to_pil(gt_t, max_edge)
    rend_img = tensor_to_pil(render_t, max_edge)
    err_img = tensor_to_pil(error_t, max_edge)
    
    w, h = gt_img.size
    gap = 4
    label_h = 24
    canvas = PILImage.new("RGB", (w * 3 + gap * 2, h + label_h), (40, 40, 40))
    canvas.paste(gt_img, (0, label_h))
    canvas.paste(rend_img, (w + gap, label_h))
    canvas.paste(err_img, (w * 2 + gap * 2, label_h))
    
    # Try to add text labels
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((10, 2), f"GT: {view_name}", fill=(255, 255, 255), font=font)
        draw.text((w + gap + 10, 2), f"Rendered (PSNR: {psnr_val:.1f} dB)", fill=(255, 255, 255), font=font)
        draw.text((w * 2 + gap * 2 + 10, 2), "Error (5x)", fill=(255, 255, 255), font=font)
    except:
        pass
    
    return canvas


def create_summary_grid(comparisons, per_view_psnr, cols=2):
    """Create a grid of comparisons."""
    if not comparisons:
        return None
    
    # Sort by PSNR (worst first for quick inspection)
    items = sorted(zip(per_view_psnr, comparisons), key=lambda x: x[0])
    
    w, h = items[0][1].size
    rows = math.ceil(len(items) / cols)
    gap = 2
    canvas = PILImage.new("RGB", (w * cols + gap * (cols - 1), h * rows + gap * (rows - 1)), (20, 20, 20))
    
    for i, (psnr, img) in enumerate(items):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * (w + gap), r * (h + gap)))
    
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to point_cloud.ply checkpoint")
    parser.add_argument("--source", required=True, help="Dataset source dir (e.g. dataset/OldHospital)")
    parser.add_argument("--out", required=True, help="Output directory for visualizations")
    parser.add_argument("--n_views", type=int, default=12, help="Number of test views to render (0=all)")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--max_edge", type=int, default=640, help="Max edge for comparison images")
    parser.add_argument("--longest_edge", type=int, default=0, help="Render resolution limit (0=full)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model from: {args.ckpt}")
    gaussians = load_gaussians_from_ply(args.ckpt, sh_degree=args.sh_degree)
    print(f"  {gaussians.num_points:,} Gaussians loaded")

    print(f"Loading scene from: {args.source}")
    train_cams, test_cams, _, _, _ = load_scene(args.source, eval_split=True)
    print(f"  {len(test_cams)} test views")

    if args.n_views > 0 and args.n_views < len(test_cams):
        # Sample evenly across test set
        indices = np.linspace(0, len(test_cams) - 1, args.n_views, dtype=int)
        test_cams = [test_cams[i] for i in indices]
        print(f"  Selected {len(test_cams)} views for visualization")

    bg_color = torch.zeros(3, device="cuda")  # black background

    comparisons = []
    per_view_psnr = []
    report_lines = []
    report_lines.append(f"Checkpoint: {args.ckpt}")
    report_lines.append(f"Gaussians: {gaussians.num_points:,}")
    report_lines.append(f"Views rendered: {len(test_cams)}")
    report_lines.append("")

    print(f"\nRendering {len(test_cams)} test views...")
    with torch.no_grad():
        for i, cam in enumerate(test_cams):
            render_pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=args.longest_edge)
            rendered = render_pkg["render"].clamp(0, 1)
            rw, rh = render_pkg["width"], render_pkg["height"]

            gt = load_image_tensor(cam)
            gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)

            mse = F.mse_loss(rendered, gt).item()
            psnr = -10 * math.log10(mse) if mse > 0 else 100.0
            per_view_psnr.append(psnr)

            # Alpha coverage
            alpha = render_pkg["rend_alpha"]
            if alpha is not None:
                alpha_2d = alpha.squeeze()
                if alpha_2d.dim() > 2:
                    alpha_2d = alpha_2d[0]
                low_alpha_pct = (alpha_2d < 0.5).float().mean().item() * 100
            else:
                low_alpha_pct = -1

            error_map = make_error_heatmap(gt, rendered, scale=5.0)
            comp = create_comparison(gt, rendered, error_map, psnr, cam.image_name, max_edge=args.max_edge)
            comparisons.append(comp)

            # Save individual comparison
            safe_name = cam.image_name.replace("/", "_").replace("\\", "_")
            safe_name = os.path.splitext(safe_name)[0]
            comp.save(os.path.join(args.out, f"{safe_name}_compare.png"))

            line = f"  {cam.image_name:40s}  PSNR={psnr:6.2f} dB  low_alpha={low_alpha_pct:5.1f}%"
            report_lines.append(line)
            print(f"  [{i+1}/{len(test_cams)}] {cam.image_name} → PSNR={psnr:.2f} dB")

    avg_psnr = np.mean(per_view_psnr) if per_view_psnr else 0
    min_psnr = np.min(per_view_psnr) if per_view_psnr else 0
    max_psnr = np.max(per_view_psnr) if per_view_psnr else 0

    report_lines.append("")
    report_lines.append(f"Mean PSNR: {avg_psnr:.2f} dB")
    report_lines.append(f"Min  PSNR: {min_psnr:.2f} dB")
    report_lines.append(f"Max  PSNR: {max_psnr:.2f} dB")

    # Save grid
    grid = create_summary_grid(comparisons, per_view_psnr, cols=2)
    if grid:
        grid.save(os.path.join(args.out, "summary_grid.png"))
        print(f"\nSaved summary grid → {args.out}/summary_grid.png")

    # Save report
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'='*50}")
    print(f"  Mean PSNR: {avg_psnr:.2f} dB  (min={min_psnr:.2f}, max={max_psnr:.2f})")
    print(f"  Output: {args.out}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
