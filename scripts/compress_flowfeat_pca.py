#!/usr/bin/env python3
"""
PCA Compression of FlowFeat Features
=====================================
Compress 128d FlowFeat DPT intermediate features to target dimensions
using PCA-nomean (without mean subtraction for cross-domain compatibility).

Input:  output/features_flowfeat/{scene}/{coarse,mid,fine}/rgb_{idx}_{scale}_128x{H}x{W}.pt
Output: output/features_flowfeat_pca/{scene}/{coarse,mid,fine}/rgb_{idx}_{scale}_{D}x{H}x{W}.pt
        output/features_flowfeat_pca/{scene}/pca_params/{scale}_pca.pth

Usage:
    python scripts/compress_flowfeat_pca.py \
        --input_dir output/features_flowfeat/OldHospital \
        --output_dir output/features_flowfeat_pca/OldHospital \
        --coarse_dim 32 --mid_dim 64 --fine_dim 64 \
        --max_fit_frames 200
"""

import argparse
import os
import re
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fit_pca_nomean(features: np.ndarray, target_dim: int) -> np.ndarray:
    """Fit PCA without mean subtraction (PCA-nomean).

    Args:
        features: (N_pixels, C) feature matrix
        target_dim: output dimension

    Returns:
        components: (target_dim, C) projection matrix
    """
    # SVD on raw data (no mean centering)
    # For large N, use randomized SVD
    from sklearn.utils.extmath import randomized_svd

    U, S, Vt = randomized_svd(features, n_components=target_dim, random_state=42)
    components = Vt[:target_dim]  # (target_dim, C)

    # Explained variance ratio
    total_var = np.sum(features ** 2) / features.shape[0]
    explained = np.sum(S[:target_dim] ** 2) / features.shape[0]
    ratio = explained / total_var * 100
    print(f"    PCA-nomean: {features.shape[1]}d → {target_dim}d, "
          f"explained variance: {ratio:.1f}%")

    return components


def compress_scale(input_dir: str, output_dir: str, scale: str,
                   target_dim: int, max_fit_frames: int = 200):
    """Compress features for one scale."""
    scale_input = os.path.join(input_dir, scale)
    scale_output = os.path.join(output_dir, scale)
    pca_dir = os.path.join(output_dir, "pca_params")
    os.makedirs(scale_output, exist_ok=True)
    os.makedirs(pca_dir, exist_ok=True)

    # List feature files
    files = sorted([f for f in os.listdir(scale_input) if f.endswith('.pt')])
    if not files:
        print(f"  [{scale}] No feature files found")
        return

    print(f"\n  [{scale}] {len(files)} files, target_dim={target_dim}")

    # Step 1: Fit PCA on subset of frames
    fit_files = files[:max_fit_frames]
    pixels = []
    for fname in tqdm(fit_files, desc=f"  [{scale}] Loading for PCA"):
        feat = torch.load(os.path.join(scale_input, fname), map_location='cpu')
        C, H, W = feat.shape
        flat = feat.reshape(C, -1).T.numpy()  # (H*W, C)
        # Subsample pixels for memory
        if flat.shape[0] > 5000:
            idx = np.random.choice(flat.shape[0], 5000, replace=False)
            flat = flat[idx]
        pixels.append(flat)

    all_pixels = np.concatenate(pixels, axis=0)
    print(f"    Fitting PCA on {all_pixels.shape[0]} pixels × {all_pixels.shape[1]}d")

    components = fit_pca_nomean(all_pixels, target_dim)
    components_t = torch.from_numpy(components).float()  # (target_dim, C)

    # Save PCA params
    torch.save({
        'components': components_t,
        'source_dim': all_pixels.shape[1],
        'target_dim': target_dim,
        'scale': scale,
    }, os.path.join(pca_dir, f'{scale}_pca.pth'))

    # Step 2: Project all features
    for fname in tqdm(files, desc=f"  [{scale}] Compressing"):
        feat = torch.load(os.path.join(scale_input, fname), map_location='cpu')
        C, H, W = feat.shape

        # Project: (target_dim, C) @ (C, H*W) → (target_dim, H*W)
        flat = feat.reshape(C, -1)  # (C, H*W)
        compressed = components_t @ flat  # (target_dim, H*W)
        compressed = compressed.reshape(target_dim, H, W)

        # L2 normalize per-pixel
        compressed = torch.nn.functional.normalize(compressed, p=2, dim=0)

        # Save with new filename
        # rgb_0_coarse_256x19x34.pt → rgb_0_coarse_32x19x34.pt
        new_fname = re.sub(r'_\d+x(\d+x\d+)', f'_{target_dim}x\\1', fname)
        torch.save(compressed, os.path.join(scale_output, new_fname))

    print(f"    → Saved to {scale_output}")


def main():
    parser = argparse.ArgumentParser(description="PCA compress FlowFeat features")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--coarse_dim", type=int, default=32)
    parser.add_argument("--mid_dim", type=int, default=64)
    parser.add_argument("--fine_dim", type=int, default=64)
    parser.add_argument("--max_fit_frames", type=int, default=200)
    parser.add_argument("--traj_path", type=str, default=None,
                        help="Trajectory file to copy into output dir")
    args = parser.parse_args()

    print(f"PCA Compression: {args.input_dir} → {args.output_dir}")
    print(f"  Targets: coarse={args.coarse_dim}, mid={args.mid_dim}, fine={args.fine_dim}")

    for scale, dim in [('coarse', args.coarse_dim),
                       ('mid', args.mid_dim),
                       ('fine', args.fine_dim)]:
        compress_scale(args.input_dir, args.output_dir, scale,
                       dim, args.max_fit_frames)

    # Copy traj file
    traj_dest = os.path.join(args.output_dir, 'traj_w_c.txt')
    if not os.path.exists(traj_dest):
        import shutil
        traj_src = None
        if args.traj_path and os.path.exists(args.traj_path):
            traj_src = args.traj_path
        else:
            # Try to find one nearby
            for candidate in [os.path.join(args.input_dir, 'traj_w_c.txt'),
                              os.path.join(os.path.dirname(args.input_dir), 'traj_w_c.txt')]:
                if os.path.exists(candidate):
                    traj_src = candidate
                    break
        if traj_src:
            shutil.copy2(traj_src, traj_dest)
            print(f"\nCopied traj to {traj_dest}")

    print("\nDone!")


if __name__ == "__main__":
    main()
