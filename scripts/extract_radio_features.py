#!/usr/bin/env python3
"""
RADIO (C-RADIOv4-H) 特征提取脚本

从 RGB 图像通过 RADIO ViT-H/16 提取:
  - Local features:  1280d @ H/16 × W/16 → PCA → target_dim
  - Global summary:  2560d summary vector (用于定位初始化)

用法:
    CUDA_VISIBLE_DEVICES=5 python scripts/extract_radio_features.py \
        --image_dir dataset/OldHospital \
        --output_dir output/features_radio/OldHospital_indexed \
        --traj_source output/features_selected_pca/OldHospital_indexed/traj_w_c.txt \
        --target_hw 68 120 --target_dim 64

    # 提取并保存全局 summary 向量:
    CUDA_VISIBLE_DEVICES=5 python scripts/extract_radio_features.py \
        --image_dir dataset/OldHospital \
        --output_dir output/features_radio/OldHospital_indexed \
        --target_hw 68 120 --target_dim 64 --save_summary
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_images(image_dir: str) -> list:
    """Find all sequence images and return sorted (idx, path) pairs.
    Supports: seq*/frame*.png (Cambridge), Sequence_*/rgb/ (room)"""
    images = []
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

    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            images.append((len(images), os.path.join(image_dir, fname)))
    return images


def fit_pca(features_list, target_dim, desc=""):
    """Fit PCA on sampled features.

    Args:
        features_list: list of (C, H, W) tensors
        target_dim: target PCA dimension

    Returns:
        pca_matrix: (target_dim, C), pca_mean: (C,)
    """
    all_pixels = []
    sample_interval = max(1, len(features_list) // 100)
    for i in range(0, len(features_list), sample_interval):
        feat = features_list[i]  # (C, H, W)
        C, H, W = feat.shape
        pixels = feat.reshape(C, -1).T  # (H*W, C)
        n_sample = min(500, pixels.shape[0])
        indices = torch.randperm(pixels.shape[0])[:n_sample]
        all_pixels.append(pixels[indices])

    all_pixels = torch.cat(all_pixels, dim=0).float()  # (N, C)
    print(f"  PCA {desc}: {all_pixels.shape[0]} samples, {all_pixels.shape[1]}d → {target_dim}d")

    mean = all_pixels.mean(dim=0)
    centered = all_pixels - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:target_dim]  # (target_dim, C)

    var_explained = (S[:target_dim] ** 2).sum() / (S ** 2).sum()
    print(f"  PCA {desc}: explained variance = {var_explained:.3f}")

    return components, mean


def apply_pca(feat, pca_matrix, pca_mean):
    """Apply PCA transform: (C, H, W) → (target_dim, H, W)"""
    C, H, W = feat.shape
    pixels = feat.reshape(C, -1).T.float()  # (H*W, C)
    centered = pixels - pca_mean.to(pixels.device)
    transformed = centered @ pca_matrix.T.to(pixels.device)  # (H*W, d)
    return transformed.T.reshape(-1, H, W)  # (d, H, W)


def main():
    parser = argparse.ArgumentParser(description='Extract RADIO features')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Dataset image directory (e.g., dataset/OldHospital)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output feature directory')
    parser.add_argument('--traj_source', type=str, default=None,
                        help='Copy trajectory file from this path')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--radio_repo', type=str, default='/root/RADIO',
                        help='Path to RADIO repo')
    parser.add_argument('--radio_version', type=str, default='c-radio_v4-h',
                        help='RADIO model version')

    # Target resolution for local features
    # For 1920x1080 input: RADIO native output is 120x68 (W/16, H/16)
    parser.add_argument('--target_hw', type=int, nargs=2, default=None,
                        help='Target (H, W) for local features. '
                             'If None, use native RADIO resolution.')
    # PCA compression
    parser.add_argument('--target_dim', type=int, default=64,
                        help='Target dimension after PCA (from 1280d)')
    parser.add_argument('--no_pca', action='store_true',
                        help='Skip PCA, save raw 1280d features')

    # Summary vectors
    parser.add_argument('--save_summary', action='store_true',
                        help='Also save global summary vectors (2560d)')

    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # Create directories
    (output_dir / 'fine_radio').mkdir(parents=True, exist_ok=True)
    if not args.no_pca:
        (output_dir / 'pca_params').mkdir(parents=True, exist_ok=True)
    if args.save_summary:
        (output_dir / 'summary').mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(args.image_dir)
    print(f"Found {len(images)} images in {args.image_dir}")

    # Limit to trajectory count if provided
    if args.traj_source and os.path.exists(args.traj_source):
        n_traj = sum(1 for _ in open(args.traj_source))
        if n_traj < len(images):
            print(f"Limiting to {n_traj} images (matching trajectory)")
            images = images[:n_traj]

    # Load RADIO model
    from feature_extraction.extractor_radio import RADIOFeatureExtractor
    extractor = RADIOFeatureExtractor(
        version=args.radio_version,
        device=args.device,
        radio_repo=args.radio_repo,
    )

    # ====== Pass 1: Extract raw features ======
    print(f"\n=== Pass 1: Extracting RADIO features ({len(images)} images) ===")
    raw_locals = []
    all_summaries = []

    for idx, img_path in tqdm(images, desc="Extracting"):
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        result = extractor.extract(img_tensor)

        local_feat = result['local'].cpu()  # (1280, Hp, Wp)

        # Resize to target resolution if specified
        if args.target_hw is not None:
            tH, tW = args.target_hw
            local_feat = F.interpolate(
                local_feat.unsqueeze(0), size=(tH, tW),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        raw_locals.append(local_feat)

        if args.save_summary:
            all_summaries.append(result['summary'].cpu())

    # Record native output resolution
    native_H, native_W = raw_locals[0].shape[1], raw_locals[0].shape[2]
    print(f"  Local features: 1280d @ {native_H}×{native_W}")
    if all_summaries:
        print(f"  Summary vectors: {all_summaries[0].shape[0]}d")

    # ====== Pass 2: PCA compression ======
    if args.no_pca:
        print("\n=== Skipping PCA (--no_pca) ===")
        final_locals = raw_locals
        actual_dim = 1280
    else:
        print("\n=== Pass 2: PCA compression ===")
        actual_dim = args.target_dim
        pca_matrix, pca_mean = fit_pca(raw_locals, args.target_dim, "radio_local")

        # Save PCA params
        torch.save({
            'components': pca_matrix,
            'mean': pca_mean,
            'source_dim': 1280,
            'target_dim': args.target_dim,
        }, output_dir / 'pca_params' / 'fine_radio_pca.pt')

        final_locals = [
            apply_pca(f, pca_matrix, pca_mean)
            for f in tqdm(raw_locals, desc="PCA local")
        ]

    # ====== Pass 3: Save features ======
    print("\n=== Saving features ===")
    feat_H, feat_W = final_locals[0].shape[1], final_locals[0].shape[2]

    for i, (idx, img_path) in enumerate(tqdm(images, desc="Saving local")):
        f = final_locals[i].half()
        fname = f"rgb_{idx}_fine_radio_{actual_dim}x{feat_H}x{feat_W}.pt"
        torch.save(f, output_dir / 'fine_radio' / fname)

    # Save summary vectors
    if args.save_summary and all_summaries:
        summary_dim = all_summaries[0].shape[0]
        for i, (idx, img_path) in enumerate(tqdm(images, desc="Saving summary")):
            fname = f"rgb_{idx}_summary_{summary_dim}.pt"
            torch.save(all_summaries[i].half(), output_dir / 'summary' / fname)

        # Also save stacked matrix for retrieval (N, 2560)
        summary_matrix = torch.stack(all_summaries, dim=0)  # (N, 2560)
        torch.save(summary_matrix, output_dir / 'summary_matrix.pt')
        print(f"  Saved summary matrix: {summary_matrix.shape}")

    # Copy trajectory file
    if args.traj_source and os.path.exists(args.traj_source):
        import shutil
        shutil.copy2(args.traj_source, output_dir / 'traj_w_c.txt')
        print(f"Copied trajectory from {args.traj_source}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Frames:  {len(images)}")
    print(f"  Local:   {actual_dim}d @ {feat_H}×{feat_W} (fine_radio)")
    if all_summaries:
        print(f"  Summary: {summary_dim}d × {len(all_summaries)} frames")
    print(f"  Output:  {output_dir}")

    total_bytes = sum(
        f.stat().st_size
        for f in output_dir.rglob('*.pt')
    )
    print(f"  Disk:    {total_bytes / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
