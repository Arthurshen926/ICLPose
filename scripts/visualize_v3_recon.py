#!/usr/bin/env python3
"""
Visualize v3 2DGS reconstruction quality for OldHospital.
Renders test+train views from each sequence, compares with GT, computes PSNR.
Outputs: per-view comparison images, per-sequence grids, summary stats.
"""
import os, sys, json, math, argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.visualize_2dgs_recon import load_ply_2dgs, cam_to_viewmat, render_2dgs
from scripts.visualize_2dgs_recon import colorize_depth, colorize_normal


def parse_split_file(path):
    """Parse dataset_train.txt / dataset_test.txt — returns set of image names."""
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


def compute_psnr(img, gt):
    """Compute PSNR between two [H,W,3] float tensors in [0,1]."""
    mse = ((img - gt) ** 2).mean().item()
    if mse <= 0:
        return 50.0
    return -10 * math.log10(mse)


def main():
    parser = argparse.ArgumentParser(description='Visualize v3 2DGS reconstruction')
    parser.add_argument('--model_dir', default='output/2dgs_models/OldHospital/v3')
    parser.add_argument('--source_dir', default='dataset/OldHospital')
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--output_dir', default='output/2dgs_models/OldHospital/v3/visualization')
    parser.add_argument('--views_per_seq', type=int, default=3,
                        help='Number of views to render per sequence')
    parser.add_argument('--longest_edge', type=int, default=960)
    parser.add_argument('--ply_path', default=None,
                        help='Override PLY path (default: auto from model_dir/iteration)')
    args = parser.parse_args()

    # Paths
    if args.ply_path:
        ply_path = args.ply_path
    else:
        ply_path = os.path.join(args.model_dir, 'point_cloud',
                                f'iteration_{args.iteration}', 'point_cloud.ply')
    cam_path = os.path.join(args.model_dir, 'cameras.json')
    assert os.path.exists(ply_path), f'Not found: {ply_path}'

    os.makedirs(args.output_dir, exist_ok=True)

    # Load train/test split
    train_names = parse_split_file(os.path.join(args.source_dir, 'dataset_train.txt'))
    test_names = parse_split_file(os.path.join(args.source_dir, 'dataset_test.txt'))
    print(f"Split: {len(train_names)} train, {len(test_names)} test")

    # Load cameras, group by sequence and split
    with open(cam_path) as f:
        all_cams = json.load(f)

    seq_train = defaultdict(list)
    seq_test = defaultdict(list)
    for cam in all_cams:
        name = cam['img_name']
        seq = name.split('/')[0]
        if name in test_names:
            seq_test[seq].append(cam)
        elif name in train_names:
            seq_train[seq].append(cam)
        # else: not in either split (shouldn't happen)

    all_seqs = sorted(set(list(seq_train.keys()) + list(seq_test.keys())))
    print(f"Sequences: {all_seqs}")
    for seq in all_seqs:
        print(f"  {seq}: {len(seq_train[seq])} train, {len(seq_test[seq])} test")

    # Select views: for each seq, pick evenly-spaced test + train views
    selected = []  # (cam, split, seq)
    for seq in all_seqs:
        # Prefer test views
        test_cams = seq_test[seq]
        train_cams = seq_train[seq]

        n_test = min(args.views_per_seq, len(test_cams))
        n_train = min(args.views_per_seq - n_test, len(train_cams))

        if n_test > 0:
            step = max(1, len(test_cams) // n_test)
            for i in range(0, len(test_cams), step)[:n_test]:
                selected.append((test_cams[i], 'test', seq))

        if n_train > 0:
            step = max(1, len(train_cams) // n_train)
            for i in range(0, len(train_cams), step)[:n_train]:
                selected.append((train_cams[i], 'train', seq))

    print(f"\nTotal views to render: {len(selected)}")

    # Load model
    print(f"\nLoading model: {ply_path}")
    model = load_ply_2dgs(ply_path)

    # Render each view
    results = []  # (seq, split, name, psnr, paths)
    seq_psnrs = defaultdict(list)

    for idx, (cam, split, seq) in enumerate(selected):
        ow, oh = cam['width'], cam['height']
        scale = min(args.longest_edge / max(ow, oh), 1.0)
        w, h = int(ow * scale), int(oh * scale)
        fx, fy = cam['fx'] * scale, cam['fy'] * scale

        viewmat = torch.tensor(cam_to_viewmat(cam), device='cuda')
        K = torch.tensor([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], device='cuda')

        img_name = cam['img_name']
        safe_name = img_name.replace('/', '_').replace('.png', '')

        print(f"  [{idx+1}/{len(selected)}] {seq}/{split}: {img_name} ({w}x{h})")

        with torch.no_grad():
            rgb, depth, alpha, normal = render_2dgs(model, viewmat, K, w, h)

        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        # Load GT
        gt_path = None
        for candidate in [
            os.path.join(args.source_dir, img_name),
            os.path.join(args.source_dir, 'processed', img_name),
        ]:
            if os.path.exists(candidate):
                gt_path = candidate
                break

        psnr = None
        gt_np = None
        if gt_path:
            gt_img = Image.open(gt_path).convert('RGB').resize((w, h), Image.LANCZOS)
            gt_np = np.array(gt_img)
            gt_t = torch.tensor(gt_np, device='cuda', dtype=torch.float32) / 255.0
            rgb_t = rgb.clamp(0, 1)
            psnr = compute_psnr(rgb_t, gt_t)
            seq_psnrs[seq].append(psnr)

        # Save outputs
        seq_dir = os.path.join(args.output_dir, seq)
        os.makedirs(seq_dir, exist_ok=True)

        # Save rendered RGB
        Image.fromarray(rgb_np).save(os.path.join(seq_dir, f'{safe_name}_render.png'))

        # Save depth
        depth_colored = colorize_depth(depth, alpha)
        Image.fromarray(depth_colored).save(os.path.join(seq_dir, f'{safe_name}_depth.png'))

        # Save comparison (GT | Render)
        if gt_np is not None:
            # Add label bar
            label_h = 30
            label_gt = np.zeros((label_h, w, 3), dtype=np.uint8)
            label_gt[:, :, :] = [40, 40, 40]
            label_rd = np.zeros((label_h, w, 3), dtype=np.uint8)
            label_rd[:, :, :] = [40, 40, 40]

            gt_panel = np.concatenate([label_gt, gt_np], axis=0)
            rd_panel = np.concatenate([label_rd, rgb_np], axis=0)
            compare = np.concatenate([gt_panel, rd_panel], axis=1)

            # Add PSNR text overlay (draw simple)
            psnr_str = f"PSNR: {psnr:.2f} dB" if psnr else "N/A"
            split_str = f"[{split.upper()}]"
            compare_img = Image.fromarray(compare)
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(compare_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 5), f"GT — {img_name}", fill=(200, 200, 200), font=font)
                draw.text((w + 10, 5), f"Render — {psnr_str} {split_str}", fill=(200, 200, 200), font=font)
            except:
                pass

            compare_img.save(os.path.join(seq_dir, f'{safe_name}_compare.png'))

        results.append((seq, split, img_name, psnr))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  OldHospital v3 2DGS — Iter {args.iteration}")
    print(f"{'='*60}")
    all_psnrs = []
    for seq in all_seqs:
        ps = seq_psnrs.get(seq, [])
        if ps:
            avg = sum(ps) / len(ps)
            all_psnrs.extend(ps)
            split_label = "test" if seq in seq_test and seq_test[seq] else "train"
            print(f"  {seq} ({split_label:5s}): PSNR = {avg:.2f} dB  ({len(ps)} views)")
    if all_psnrs:
        print(f"  {'Overall':12s}: PSNR = {sum(all_psnrs)/len(all_psnrs):.2f} dB  ({len(all_psnrs)} views)")
    print(f"{'='*60}")

    # Create per-sequence grid comparisons
    try:
        for seq in all_seqs:
            seq_dir = os.path.join(args.output_dir, seq)
            compare_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('_compare.png')])
            if not compare_files:
                continue
            imgs = [Image.open(os.path.join(seq_dir, f)) for f in compare_files]
            # Vertical stack
            w_grid = max(img.size[0] for img in imgs)
            h_grid = sum(img.size[1] for img in imgs)
            grid = Image.new('RGB', (w_grid, h_grid))
            y = 0
            for img in imgs:
                grid.paste(img, (0, y))
                y += img.size[1]
            grid.save(os.path.join(args.output_dir, f'{seq}_grid.png'))

        # Overall grid: 1 view per sequence
        best_per_seq = []
        for seq in all_seqs:
            seq_dir = os.path.join(args.output_dir, seq)
            compare_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('_compare.png')])
            if compare_files:
                best_per_seq.append(Image.open(os.path.join(seq_dir, compare_files[0])))

        if best_per_seq:
            # 3x3 grid
            cols = 3
            rows = (len(best_per_seq) + cols - 1) // cols
            # Resize to uniform width
            target_w = min(img.size[0] for img in best_per_seq)
            resized = []
            for img in best_per_seq:
                ratio = target_w / img.size[0]
                new_h = int(img.size[1] * ratio)
                resized.append(img.resize((target_w, new_h), Image.LANCZOS))

            max_h = max(img.size[1] for img in resized)
            grid = Image.new('RGB', (target_w * cols, max_h * rows))
            for i, img in enumerate(resized):
                r, c = i // cols, i % cols
                grid.paste(img, (c * target_w, r * max_h))
            grid.save(os.path.join(args.output_dir, 'overview_grid.png'))
            print(f"\n  Overview grid saved: overview_grid.png ({cols}x{rows})")
    except Exception as e:
        print(f"  Grid creation failed: {e}")

    # Save stats JSON
    stats = {
        'iteration': args.iteration,
        'model': args.model_dir,
        'views': [{'seq': s, 'split': sp, 'name': n, 'psnr': p} for s, sp, n, p in results],
        'per_seq': {s: sum(v)/len(v) for s, v in seq_psnrs.items() if v},
        'overall_psnr': sum(all_psnrs)/len(all_psnrs) if all_psnrs else 0,
    }
    with open(os.path.join(args.output_dir, 'eval_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: eval_stats.json\n")


if __name__ == '__main__':
    main()
