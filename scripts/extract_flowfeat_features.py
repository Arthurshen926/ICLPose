#!/usr/bin/env python3
"""
Extract FlowFeat multi-scale features for all frames of a scene.

Produces 3 scales of 128d features from FlowFeat's pretrained DPT decoder:
  coarse: 128d @ layer_4_rn resolution  (e.g. 19×34 for 532×952 input)
  mid:    128d @ path_4 resolution      (e.g. 38×68)
  fine:   128d @ path_3 resolution      (e.g. 76×136)

Output format compatible with dataset_v4:
  {output_dir}/{scale}/rgb_{idx}_{scale}_{C}x{H}x{W}.pt

Usage:
    python scripts/extract_flowfeat_features.py \
        --scene OldHospital \
        --image_dir dataset/OldHospital \
        --output_dir output/features_flowfeat/OldHospital \
        --resize_factor 0.5 \
        --gpu 0
"""

import argparse
import os
import sys
import glob
import re
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from feature_extraction.flowfeat_extractor import FlowFeatExtractor


def parse_cambridge_splits(image_dir: str, split_files: list) -> list:
    """Parse Cambridge Landmarks split files to get ordered (idx, path) pairs.

    Returns frames sorted alphabetically by relative path (matching existing pipeline).
    """
    rel_paths = []
    for split_file in split_files:
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Visual') or line.startswith('Image'):
                    continue
                parts = line.split()
                if len(parts) >= 1 and '/' in parts[0]:
                    rel_paths.append(parts[0])

    rel_paths_sorted = sorted(rel_paths)
    images = []
    for i, rp in enumerate(rel_paths_sorted):
        full_path = os.path.join(image_dir, rp)
        if os.path.exists(full_path):
            images.append((i, full_path))
        else:
            print(f"  WARNING: {full_path} not found, skipping")
    return images


def find_images(image_dir: str) -> list:
    """Find all sequence image directories and return sorted (idx, path) pairs."""
    images = []
    # Support structures: seq*/frame*.png (Cambridge), Sequence_*/rgb/ (room),
    # frame_*.color.png (7scenes), rgb_*.png, or flat directory

    # Try Cambridge-style: seq*/frame*.png (OldHospital, KingsCollege, etc.)
    seq_dirs = sorted(glob.glob(os.path.join(image_dir, "seq*")))
    if seq_dirs and all(os.path.isdir(d) for d in seq_dirs):
        global_idx = 0
        for seq_dir in seq_dirs:
            for fname in sorted(os.listdir(seq_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    images.append((global_idx, os.path.join(seq_dir, fname)))
                    global_idx += 1
        if images:
            return images

    # Try room-style: Sequence_N/rgb/*.png
    seq_dirs = sorted(glob.glob(os.path.join(image_dir, "Sequence_*")))
    if seq_dirs:
        global_idx = 0
        for seq_dir in seq_dirs:
            rgb_dir = os.path.join(seq_dir, "rgb")
            if not os.path.isdir(rgb_dir):
                rgb_dir = seq_dir
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    images.append((global_idx, os.path.join(rgb_dir, fname)))
                    global_idx += 1
        return images

    # Try frame_NNNNN.color.png pattern (7scenes)
    frame_files = sorted(glob.glob(os.path.join(image_dir, "**", "frame_*.color.png"), recursive=True))
    if frame_files:
        for i, f in enumerate(frame_files):
            images.append((i, f))
        return images

    # Try rgb_{idx}.png pattern
    rgb_files = sorted(glob.glob(os.path.join(image_dir, "rgb_*.png")))
    if rgb_files:
        for f in rgb_files:
            m = re.search(r'rgb_(\d+)', os.path.basename(f))
            if m:
                images.append((int(m.group(1)), f))
        return images

    # Fallback: all images in directory
    for i, fname in enumerate(sorted(os.listdir(image_dir))):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            images.append((i, os.path.join(image_dir, fname)))

    return images


def save_feature(feat: torch.Tensor, output_dir: str, scale: str, idx: int):
    """Save feature tensor in dataset_v4 compatible format."""
    scale_dir = os.path.join(output_dir, scale)
    os.makedirs(scale_dir, exist_ok=True)
    C, H, W = feat.shape
    fname = f"rgb_{idx}_{scale}_{C}x{H}x{W}.pt"
    torch.save(feat, os.path.join(scale_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="Extract FlowFeat multi-scale features")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14_yt",
                        help="FlowFeat model name")
    parser.add_argument("--resize_factor", type=float, default=0.5,
                        help="Resize factor relative to original (0.5 = half-res)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--start_idx", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_idx", type=int, default=-1, help="End frame index (-1 = all)")
    parser.add_argument("--split_files", type=str, nargs='+', default=None,
                        help="Cambridge split files (dataset_train.txt dataset_test.txt)")
    parser.add_argument("--traj_path", type=str, default=None,
                        help="Trajectory file to copy into output dir")
    parser.add_argument("--coarse_hw", type=int, nargs=2, default=[20, 35])
    parser.add_argument("--mid_hw", type=int, nargs=2, default=[40, 70])
    parser.add_argument("--fine_hw", type=int, nargs=2, default=[80, 140])
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    target_resolutions = {
        'coarse': tuple(args.coarse_hw),
        'mid': tuple(args.mid_hw),
        'fine': tuple(args.fine_hw),
    }

    # Initialize extractor
    extractor = FlowFeatExtractor(
        model_name=args.model_name,
        device=device,
        target_resolutions=target_resolutions,
    )

    # Find images
    if args.split_files:
        images = parse_cambridge_splits(args.image_dir, args.split_files)
        print(f"[{args.scene}] Loaded {len(images)} frames from split files")
    else:
        images = find_images(args.image_dir)
        print(f"[{args.scene}] Found {len(images)} images")

    # Slice
    if args.end_idx > 0:
        images = images[args.start_idx:args.end_idx]
    elif args.start_idx > 0:
        images = images[args.start_idx:]

    # Extract
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, img_path in tqdm(images, desc=f"Extracting {args.scene}"):
        # Check if already done
        existing = glob.glob(os.path.join(args.output_dir, "coarse", f"rgb_{idx}_coarse_*.pt"))
        if existing:
            continue

        feats = extractor.extract_from_path(img_path, resize_factor=args.resize_factor)

        for scale, feat in feats.items():
            save_feature(feat, args.output_dir, scale, idx)

    # Print summary
    for scale in ["coarse", "mid", "fine"]:
        scale_dir = os.path.join(args.output_dir, scale)
        if os.path.isdir(scale_dir):
            n = len(os.listdir(scale_dir))
            sample = sorted(os.listdir(scale_dir))[0] if n > 0 else "N/A"
            print(f"  {scale}: {n} files, e.g. {sample}")

    # Copy traj if provided
    if args.traj_path and os.path.exists(args.traj_path):
        import shutil
        dest = os.path.join(args.output_dir, "traj_w_c.txt")
        if not os.path.exists(dest):
            shutil.copy2(args.traj_path, dest)
            print(f"Copied trajectory to {dest}")

    print("Done!")


if __name__ == "__main__":
    main()
