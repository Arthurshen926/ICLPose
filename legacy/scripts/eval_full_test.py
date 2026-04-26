#!/usr/bin/env python3
"""Full evaluation: compute PSNR on ALL 182 test views for all models."""
import os, sys, json, math
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

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

def eval_model(model_data, test_cams, source_dir, longest_edge=0):
    """Evaluate model on all test views. Returns list of (name, psnr)."""
    results = []
    for cam in tqdm(test_cams, desc="Eval", leave=False):
        ow, oh = cam['width'], cam['height']
        if longest_edge > 0:
            scale = min(longest_edge / max(ow, oh), 1.0)
        else:
            scale = 1.0
        w, h = int(ow * scale), int(oh * scale)
        fx, fy = cam['fx'] * scale, cam['fy'] * scale
        viewmat = torch.tensor(cam_to_viewmat(cam), device='cuda')
        K = torch.tensor([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], device='cuda')

        with torch.no_grad():
            rgb, _, _, _ = render_2dgs(model_data, viewmat, K, w, h)
        rgb_clamped = rgb.clamp(0, 1)

        # Load GT
        gt_np = None
        for candidate in [
            os.path.join(source_dir, cam['img_name']),
            os.path.join(source_dir, 'processed', cam['img_name']),
        ]:
            if os.path.exists(candidate):
                gt_img = Image.open(candidate).convert('RGB').resize((w, h), Image.LANCZOS)
                gt_np = np.array(gt_img)
                break
        if gt_np is None:
            continue

        gt_t = torch.tensor(gt_np, device='cuda', dtype=torch.float32) / 255.0
        psnr = compute_psnr(rgb_clamped, gt_t)
        results.append((cam['img_name'], psnr))
    return results

def main():
    source_dir = "dataset/OldHospital"
    test_names = parse_split_file(os.path.join(source_dir, 'dataset_test.txt'))

    models = []
    configs = [
        ("v3 baseline (30k)", "output/2dgs_models/OldHospital/v3", 30000),
        ("retrain6 (25k)", "output/2dgs_models/OldHospital/v3_improved", 25000),
        ("retrain7 (15k)", "output/2dgs_models/OldHospital/v3_retrain7", 15000),
    ]

    for name, model_dir, iteration in configs:
        ply = os.path.join(model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        cam_json = os.path.join(model_dir, "cameras.json")
        if not os.path.exists(ply) or not os.path.exists(cam_json):
            print(f"  SKIP {name}: files not found")
            continue
        models.append((name, model_dir, iteration, ply, cam_json))

    # Use full resolution (same as retrain7 training)
    LONGEST_EDGE = 0  # full res

    for name, model_dir, iteration, ply, cam_json in models:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  PLY: {ply}")

        with open(cam_json) as f:
            all_cams = json.load(f)
        test_cams = [c for c in all_cams if c['img_name'] in test_names]

        model_data = load_ply_2dgs(ply)
        n_g = model_data['xyz'].shape[0]
        print(f"  Gaussians: {n_g:,}")
        print(f"  Test views: {len(test_cams)}")

        results = eval_model(model_data, test_cams, source_dir, longest_edge=LONGEST_EDGE)

        psnrs = [p for _, p in results]
        avg = sum(psnrs) / len(psnrs) if psnrs else 0
        med = sorted(psnrs)[len(psnrs)//2] if psnrs else 0
        mn = min(psnrs) if psnrs else 0
        mx = max(psnrs) if psnrs else 0

        # Per-sequence
        seq_p = defaultdict(list)
        for n, p in results:
            seq_p[n.split('/')[0]].append(p)

        print(f"\n  PSNR: avg={avg:.2f}  med={med:.2f}  min={mn:.2f}  max={mx:.2f}  ({len(psnrs)} views)")
        for seq in sorted(seq_p.keys()):
            ps = seq_p[seq]
            print(f"    {seq}: {sum(ps)/len(ps):.2f} dB ({len(ps)} views)")

    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()
