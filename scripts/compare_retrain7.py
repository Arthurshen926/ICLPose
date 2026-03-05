#!/usr/bin/env python3
"""
Compare retrain7 (5 improvements + batch) vs previous models.
Renders same test views from all models side-by-side.
"""
import os, sys, json, math, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.visualize_2dgs_recon import load_ply_2dgs, cam_to_viewmat, render_2dgs

def compute_psnr(img, gt):
    mse = ((img - gt) ** 2).mean().item()
    if mse <= 0: return 50.0
    return -10 * math.log10(mse)

def parse_split_file(path):
    names = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Visual') or line.startswith('Image'):
                continue
            parts = line.split()
            if parts and '/' in parts[0]:
                names.add(parts[0])
    return names

def load_gt_image(source_dir, img_name, w, h):
    for candidate in [
        os.path.join(source_dir, img_name),
        os.path.join(source_dir, 'processed', img_name),
    ]:
        if os.path.exists(candidate):
            gt_img = Image.open(candidate).convert('RGB').resize((w, h), Image.LANCZOS)
            return np.array(gt_img)
    return None

def get_font(size=16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

def main():
    source_dir = "dataset/OldHospital"
    output_dir = "output/2dgs_models/OldHospital/comparison_retrain7"
    os.makedirs(output_dir, exist_ok=True)

    # Models to compare
    models = [
        {
            "name": "v3 (retrain6)",
            "short": "retrain6",
            "model_dir": "output/2dgs_models/OldHospital/v3_improved",
            "iteration": 25000,
        },
        {
            "name": "v3_retrain7 (5 fixes)",
            "short": "retrain7",
            "model_dir": "output/2dgs_models/OldHospital/v3_retrain7",
            "iteration": 15000,
        },
    ]

    # Check if earlier v3 baseline exists
    v3_base_ply = "output/2dgs_models/OldHospital/v3/point_cloud/iteration_30000/point_cloud.ply"
    if os.path.exists(v3_base_ply):
        models.insert(0, {
            "name": "v3 baseline",
            "short": "baseline",
            "model_dir": "output/2dgs_models/OldHospital/v3",
            "iteration": 30000,
        })

    # Load test split
    test_names = parse_split_file(os.path.join(source_dir, 'dataset_test.txt'))

    # Load cameras from retrain7 (has all cameras)
    cam_path = os.path.join(models[-1]["model_dir"], "cameras.json")
    with open(cam_path) as f:
        all_cams = json.load(f)
    test_cams = [c for c in all_cams if c['img_name'] in test_names]

    # Group by sequence
    seq_cams = defaultdict(list)
    for cam in test_cams:
        seq = cam['img_name'].split('/')[0]
        seq_cams[seq].append(cam)

    # Select 2 views per sequence (evenly spaced)
    VIEWS_PER_SEQ = 2
    selected_cams = []
    for seq in sorted(seq_cams.keys()):
        cams = seq_cams[seq]
        step = max(1, len(cams) // VIEWS_PER_SEQ)
        for i in range(0, len(cams), step)[:VIEWS_PER_SEQ]:
            selected_cams.append(cams[i])
    print(f"Selected {len(selected_cams)} test views from {len(seq_cams)} sequences")

    # Render target resolution
    LONGEST_EDGE = 960  # for visualization

    # Load all models
    loaded_models = []
    for m in models:
        ply_path = os.path.join(m["model_dir"], "point_cloud",
                                f"iteration_{m['iteration']}", "point_cloud.ply")
        if not os.path.exists(ply_path):
            print(f"  SKIP {m['name']}: {ply_path} not found")
            continue
        print(f"  Loading {m['name']} from {ply_path}...")
        model_data = load_ply_2dgs(ply_path)
        n_gaussians = model_data['xyz'].shape[0]
        print(f"    → {n_gaussians:,} Gaussians")
        loaded_models.append({**m, "model": model_data, "n_gaussians": n_gaussians})

    if not loaded_models:
        print("No models found!")
        return

    # Render all views for all models
    all_psnrs = {m["short"]: [] for m in loaded_models}
    per_view_results = []

    font = get_font(16)
    font_large = get_font(20)

    for vi, cam in enumerate(selected_cams):
        ow, oh = cam['width'], cam['height']
        scale = min(LONGEST_EDGE / max(ow, oh), 1.0)
        w, h = int(ow * scale), int(oh * scale)
        fx, fy = cam['fx'] * scale, cam['fy'] * scale
        viewmat = torch.tensor(cam_to_viewmat(cam), device='cuda')
        K = torch.tensor([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], device='cuda')

        img_name = cam['img_name']
        seq = img_name.split('/')[0]
        safe_name = img_name.replace('/', '_').replace('.png', '')

        print(f"\n  [{vi+1}/{len(selected_cams)}] {img_name} ({w}x{h})")

        # Load GT
        gt_np = load_gt_image(source_dir, img_name, w, h)
        if gt_np is None:
            print(f"    GT not found, skipping")
            continue
        gt_t = torch.tensor(gt_np, device='cuda', dtype=torch.float32) / 255.0

        # Render from each model
        panels = []  # (label, rgb_np, psnr)
        panels.append(("GT", gt_np, None))

        for m in loaded_models:
            with torch.no_grad():
                rgb, depth, alpha, normal = render_2dgs(m["model"], viewmat, K, w, h)
            rgb_clamped = rgb.clamp(0, 1)
            psnr = compute_psnr(rgb_clamped, gt_t)
            rgb_np = (rgb_clamped.cpu().numpy() * 255).astype(np.uint8)
            all_psnrs[m["short"]].append(psnr)
            panels.append((f"{m['short']} ({psnr:.1f}dB)", rgb_np, psnr))
            print(f"    {m['short']}: {psnr:.2f} dB")

        # Create comparison strip: [GT | model1 | model2 | ...]
        label_h = 28
        n_panels = len(panels)
        strip_w = w * n_panels
        strip_h = h + label_h
        strip = Image.new('RGB', (strip_w, strip_h), (30, 30, 30))
        draw = ImageDraw.Draw(strip)

        for pi, (label, img_np, psnr) in enumerate(panels):
            x_off = pi * w
            strip.paste(Image.fromarray(img_np), (x_off, label_h))
            # Draw label
            color = (200, 200, 200) if pi == 0 else (100, 255, 100) if psnr and psnr == max(p for _, _, p in panels[1:] if p) else (200, 200, 200)
            draw.text((x_off + 8, 5), label, fill=color, font=font)

        strip.save(os.path.join(output_dir, f"{safe_name}_compare.png"))

        # Also save error maps (absolute diff from GT, amplified)
        if len(loaded_models) >= 2:
            error_panels = []
            for m in loaded_models:
                with torch.no_grad():
                    rgb, _, _, _ = render_2dgs(m["model"], viewmat, K, w, h)
                diff = (rgb.clamp(0, 1) - gt_t).abs()
                # Amplify 3x for visibility
                diff_np = (diff.clamp(0, 1).cpu().numpy() * 3 * 255).clip(0, 255).astype(np.uint8)
                error_panels.append((f"{m['short']} error (3x)", diff_np))

            err_strip = Image.new('RGB', (w * len(error_panels), h + label_h), (30, 30, 30))
            err_draw = ImageDraw.Draw(err_strip)
            for pi, (label, img_np) in enumerate(error_panels):
                x_off = pi * w
                err_strip.paste(Image.fromarray(img_np), (x_off, label_h))
                err_draw.text((x_off + 8, 5), label, fill=(255, 180, 180), font=font)
            err_strip.save(os.path.join(output_dir, f"{safe_name}_error.png"))

        per_view_results.append({
            'name': img_name, 'seq': seq,
            'psnrs': {m['short']: all_psnrs[m['short']][-1] for m in loaded_models}
        })

    # Print summary
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    for m in loaded_models:
        ps = all_psnrs[m["short"]]
        avg = sum(ps) / len(ps) if ps else 0
        med = sorted(ps)[len(ps)//2] if ps else 0
        mn = min(ps) if ps else 0
        mx = max(ps) if ps else 0
        print(f"  {m['name']:30s}  PSNR avg={avg:.2f}  med={med:.2f}  min={mn:.2f}  max={mx:.2f}  ({m['n_gaussians']:,} Gaussians)")
    print(f"{'='*70}")

    # Per-sequence breakdown
    seq_psnrs = {m["short"]: defaultdict(list) for m in loaded_models}
    for r in per_view_results:
        for m in loaded_models:
            seq_psnrs[m["short"]][r['seq']].append(r['psnrs'][m['short']])

    print(f"\n  Per-sequence PSNR:")
    seqs = sorted(set(r['seq'] for r in per_view_results))
    header = f"  {'Seq':8s}"
    for m in loaded_models:
        header += f"  {m['short']:>12s}"
    print(header)
    for seq in seqs:
        row = f"  {seq:8s}"
        for m in loaded_models:
            ps = seq_psnrs[m["short"]].get(seq, [])
            avg = sum(ps) / len(ps) if ps else 0
            row += f"  {avg:>10.2f}dB"
        print(row)

    # Create overview grid: best & worst views
    print(f"\n  Creating overview grid...")
    compare_files = sorted([f for f in os.listdir(output_dir) if f.endswith('_compare.png')])
    if compare_files:
        imgs = [Image.open(os.path.join(output_dir, f)) for f in compare_files]
        # Vertical stack
        w_grid = max(img.size[0] for img in imgs)
        h_grid = sum(img.size[1] for img in imgs)
        grid = Image.new('RGB', (w_grid, h_grid))
        y = 0
        for img in imgs:
            grid.paste(img, (0, y))
            y += img.size[1]
        grid_path = os.path.join(output_dir, 'overview_all.png')
        grid.save(grid_path)
        print(f"  Saved: {grid_path}")

    # Save stats
    stats = {
        'models': [{k: v for k, v in m.items() if k != 'model'} for m in loaded_models],
        'per_view': per_view_results,
        'summary': {m["short"]: {
            'avg_psnr': sum(all_psnrs[m["short"]]) / len(all_psnrs[m["short"]]) if all_psnrs[m["short"]] else 0,
            'n_gaussians': m['n_gaussians'],
        } for m in loaded_models},
    }
    with open(os.path.join(output_dir, 'comparison_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: comparison_stats.json")


if __name__ == '__main__':
    main()
