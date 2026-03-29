#!/usr/bin/env python3
"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) 图像预处理

对训练图像逐通道做 CLAHE，消除光照/曝光变化，替代复杂的 AppearanceNetwork。
参考 STDLoc 的做法：cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

处理后的图片存储在 {output_dir}/ 中，目录结构与原始图相同（seq*/frame.png）。
后续训练脚本（train_2dgs_joint_v3 等）通过 dataset.images 指向该目录即可生效。

用法:
    python scripts/preprocess_clahe.py \
        --image_dir dataset/OldHospital \
        --output_dir dataset/OldHospital/clahe_images \
        [--clip_limit 2.0] [--tile_size 8]

    # 多场景批处理:
    python scripts/preprocess_clahe.py --image_dir dataset/stairs --output_dir dataset/stairs/clahe_images
    python scripts/preprocess_clahe.py --image_dir dataset/room_0 --output_dir dataset/room_0/clahe_images
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def find_images(image_dir: str) -> list:
    """Find all images, return list of (relative_path, absolute_path)."""
    images = []
    image_dir = Path(image_dir)

    # Cambridge Landmarks: seq*/frame*.png
    seq_dirs = sorted(glob.glob(str(image_dir / "seq*")))
    if seq_dirs and all(os.path.isdir(d) for d in seq_dirs):
        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            for fname in sorted(os.listdir(seq_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel = os.path.join(seq_name, fname)
                    images.append((rel, os.path.join(seq_dir, fname)))
        if images:
            return images

    # Replica: Sequence_*/rgb/*.png
    seq_dirs = sorted(glob.glob(str(image_dir / "Sequence_*")))
    if seq_dirs:
        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            rgb_dir = os.path.join(seq_dir, "rgb")
            if not os.path.isdir(rgb_dir):
                rgb_dir = seq_dir
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel = os.path.join(seq_name, "rgb", fname)
                    images.append((rel, os.path.join(rgb_dir, fname)))
        return images

    # Flat directory
    for fname in sorted(os.listdir(str(image_dir))):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append((fname, os.path.join(str(image_dir), fname)))
    return images


def apply_clahe(image_bgr, clahe):
    """Apply CLAHE to each BGR channel independently."""
    b, g, r = cv2.split(image_bgr)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    return cv2.merge((b, g, r))


def main():
    parser = argparse.ArgumentParser(
        description='CLAHE preprocessing for training images')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Source image directory (e.g. dataset/OldHospital)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for CLAHE-processed images')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                        help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile_size', type=int, default=8,
                        help='CLAHE tile grid size (default: 8)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for output (default: 95, only for .jpg)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clahe = cv2.createCLAHE(
        clipLimit=args.clip_limit,
        tileGridSize=(args.tile_size, args.tile_size)
    )

    images = find_images(args.image_dir)
    if not images:
        print(f"ERROR: No images found in {args.image_dir}")
        sys.exit(1)

    print(f"CLAHE preprocessing: {len(images)} images")
    print(f"  Source:     {args.image_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  clipLimit:  {args.clip_limit}")
    print(f"  tileGrid:   ({args.tile_size}, {args.tile_size})")

    for rel_path, abs_path in tqdm(images, desc="CLAHE"):
        img_bgr = cv2.imread(abs_path)
        if img_bgr is None:
            print(f"  WARNING: cannot read {abs_path}, skipping")
            continue

        processed = apply_clahe(img_bgr, clahe)

        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if rel_path.lower().endswith('.jpg') or rel_path.lower().endswith('.jpeg'):
            cv2.imwrite(str(out_path), processed,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        else:
            cv2.imwrite(str(out_path), processed)

    print(f"\n✓ Done! {len(images)} images saved to {output_dir}/")
    print(f"  To use in training: set dataset.images to point to this directory")
    print(f"  Or re-extract features with --image_dir {output_dir}")


if __name__ == '__main__':
    main()
