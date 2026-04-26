#!/usr/bin/env python3
"""
Precompute monocular depth maps using DPT-Large for all training images.

Saves depth maps as float32 .npy files, normalized to inverse disparity.
These are used for depth supervision in 2DGS training.

Usage:
    python scripts/precompute_mono_depth.py \
        --source_dir dataset/OldHospital \
        --images . \
        --output_dir dataset/OldHospital/mono_depth \
        --device cuda:0
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def read_images_binary(path):
    """Read COLMAP images.bin to get image names."""
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("I", f.read(4))[0]
            f.read(32 + 24)  # skip qvec + tvec
            cam_id = struct.unpack("I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            num_pts = struct.unpack("Q", f.read(8))[0]
            f.read(num_pts * 24)
            images[img_id] = name
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="COLMAP dataset path")
    parser.add_argument("--images", default="", help="Image subdirectory")
    parser.add_argument("--output_dir", default=None, help="Output dir for depth maps (default: source_dir/mono_depth)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", default="Intel/dpt-large", help="HuggingFace model name")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.source_dir, "mono_depth")
    os.makedirs(output_dir, exist_ok=True)

    # Get image list from COLMAP
    images_bin = os.path.join(args.source_dir, "sparse", "0", "images.bin")
    colmap_images = read_images_binary(images_bin)
    image_names = sorted(colmap_images.values())

    # Resolve image directory
    if args.images:
        images_dir = os.path.join(args.source_dir, args.images)
    else:
        images_dir = os.path.join(args.source_dir, "images")

    print(f"Source: {args.source_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(image_names)} images from COLMAP")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    # Load model
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    processor = DPTImageProcessor.from_pretrained(args.model)
    model = DPTForDepthEstimation.from_pretrained(args.model)
    model.eval().to(args.device)
    print(f"Model loaded: {args.model}")

    # Process images
    skipped = 0
    processed = 0
    for name in tqdm(image_names, desc="Computing depth"):
        # Output path preserves directory structure (e.g., seq1/frame00001.npy)
        out_name = os.path.splitext(name)[0] + ".npy"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            skipped += 1
            continue

        img_path = os.path.join(images_dir, name)
        if not os.path.exists(img_path):
            # Fallback
            img_path = os.path.join(args.source_dir, "images", name)

        if not os.path.exists(img_path):
            print(f"  Warning: {name} not found, skipping")
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # DPT inference
        inputs = processor(images=img, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # [1, H', W'] — inverse depth (larger = closer)

        # Resize to original resolution
        depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalize to [0, 1] range (inverse disparity)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        # Save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, depth.astype(np.float32))
        processed += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Depth maps saved to: {output_dir}")


if __name__ == "__main__":
    main()
