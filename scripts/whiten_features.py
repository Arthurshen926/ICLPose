#!/usr/bin/env python3
"""
PCA-Whitening for Extracted Multi-Scale Features
=================================================
Offline post-processing step that removes global mean bias and equalizes
channel variance in extracted SD/DINO features.

Transform: f_white = Λ^{-1/2} V^T (f - μ)
  where μ is the global spatial mean, V are PCA eigenvectors,
  Λ are eigenvalues. This produces zero-mean, identity-covariance features.

The whitening transform is computed from a random subset of pixels across all
frames (configurable), then applied to all features. The transform matrices
are saved alongside the whitened features for reproducibility.

Usage:
  python scripts/whiten_features.py \
    --feature_dir output/features_multiscale/room_0 \
    --output_dir output/features_multiscale_whitened/room_0 \
    --n_sample_pixels 100000

  # Or whiten in-place (overwrites original features):
  python scripts/whiten_features.py \
    --feature_dir output/features_multiscale/room_0 \
    --inplace
"""

import argparse
import glob
import os
import shutil
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F


def compute_whitening_transform(
    feature_dir: Path,
    scale: str,
    n_sample_pixels: int = 100000,
    pca_dim: int = 0,
    device: str = 'cpu',
):
    """
    Compute PCA-whitening transform from feature files.

    Args:
        feature_dir: directory containing .pt feature files
        scale: scale name (for logging)
        n_sample_pixels: number of pixels to sample for covariance estimation
        pca_dim: if > 0, reduce dimensionality to this many components
        device: computation device

    Returns:
        mean: (C,) global mean
        whiten_matrix: (C, C) or (pca_dim, C) whitening matrix
        info: dict with eigenvalue spectrum info
    """
    pt_files = sorted(glob.glob(str(feature_dir / '*.pt')))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {feature_dir}")

    # First pass: collect random pixels
    pixels_per_file = max(1, n_sample_pixels // len(pt_files))
    all_pixels = []

    for fpath in tqdm(pt_files, desc=f'[{scale}] Sampling pixels'):
        feat = torch.load(fpath, map_location='cpu')  # (C, H, W)
        C, H, W = feat.shape
        N = H * W
        n_sample = min(pixels_per_file, N)
        idx = torch.randperm(N)[:n_sample]
        flat = feat.reshape(C, N)[:, idx].t()  # (n_sample, C)
        all_pixels.append(flat)

    all_pixels = torch.cat(all_pixels, dim=0).to(device)  # (M, C)
    M, C = all_pixels.shape
    print(f'  [{scale}] Sampled {M} pixels, feature dim={C}')

    # Compute mean
    mean = all_pixels.mean(dim=0)  # (C,)

    # Center
    centered = all_pixels - mean.unsqueeze(0)

    # Covariance: (C, C) = X^T X / (M-1)
    cov = (centered.t() @ centered) / (M - 1)

    # Eigen-decomposition (symmetric → use eigh for stability)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # eigh returns ascending order; we want descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # Compute variance explained
    total_var = eigenvalues.sum()
    cumvar = eigenvalues.cumsum(0) / total_var
    top1_ratio = eigenvalues[0] / total_var
    top10_ratio = cumvar[min(9, len(cumvar) - 1)]

    print(f'  [{scale}] Top-1 eigenvalue explains {top1_ratio:.1%} of variance')
    print(f'  [{scale}] Top-10 explain {top10_ratio:.1%}')
    print(f'  [{scale}] Eigenvalue range: {eigenvalues[0]:.4f} to {eigenvalues[-1]:.6f}')

    # Whitening matrix: Λ^{-1/2} V^T
    # Clamp small eigenvalues for numerical stability
    eps = 1e-5
    inv_sqrt_eigvals = 1.0 / torch.sqrt(eigenvalues.clamp(min=eps))

    if pca_dim > 0 and pca_dim < C:
        # Reduce dimensionality: keep top pca_dim components
        whiten_matrix = torch.diag(inv_sqrt_eigvals[:pca_dim]) @ eigenvectors[:, :pca_dim].t()
        print(f'  [{scale}] PCA reduction: {C} → {pca_dim} dims '
              f'({cumvar[pca_dim-1]:.1%} variance retained)')
    else:
        whiten_matrix = torch.diag(inv_sqrt_eigvals) @ eigenvectors.t()

    info = {
        'eigenvalues': eigenvalues.cpu(),
        'cumulative_variance': cumvar.cpu(),
        'top1_ratio': top1_ratio.item(),
        'n_samples': M,
        'feature_dim': C,
    }

    return mean.cpu(), whiten_matrix.cpu(), info


def apply_whitening(
    feat: torch.Tensor,
    mean: torch.Tensor,
    whiten_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Apply PCA-whitening to a feature tensor.

    Args:
        feat: (C, H, W) raw features
        mean: (C,) global mean
        whiten_matrix: (D, C) whitening matrix

    Returns:
        (D, H, W) whitened and L2-normalized features
    """
    C, H, W = feat.shape
    flat = feat.reshape(C, H * W).t()       # (N, C)
    centered = flat - mean.unsqueeze(0)       # (N, C)
    whitened = centered @ whiten_matrix.t()   # (N, D)
    # L2-normalize per pixel
    whitened = F.normalize(whitened, p=2, dim=1)
    D = whitened.shape[1]
    return whitened.t().reshape(D, H, W)


def main():
    parser = argparse.ArgumentParser(description='PCA-whiten multi-scale features')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Root feature directory (e.g. output/features_multiscale/room_0)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: {feature_dir}_whitened)')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite original features in-place')
    parser.add_argument('--n_sample_pixels', type=int, default=100000,
                        help='Pixels to sample for covariance estimation')
    parser.add_argument('--pca_dim', type=int, default=0,
                        help='Reduce to this many PCA dimensions (0=keep all)')
    parser.add_argument('--scales', nargs='+',
                        default=['coarse', 'mid', 'fine_sd', 'fine_dino'],
                        help='Which scales to whiten')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    if args.inplace:
        output_dir = feature_dir
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = feature_dir.parent / (feature_dir.name + '_whitened')

    print(f'Feature dir: {feature_dir}')
    print(f'Output dir:  {output_dir}')
    print(f'Scales:      {args.scales}')
    print(f'Sample size: {args.n_sample_pixels}')
    if args.pca_dim > 0:
        print(f'PCA dim:     {args.pca_dim}')

    for scale in args.scales:
        scale_dir = feature_dir / scale
        if not scale_dir.exists():
            print(f'\n⚠ Scale directory not found: {scale_dir}, skipping')
            continue

        print(f'\n{"="*60}')
        print(f'Processing scale: {scale}')
        print(f'{"="*60}')

        # Compute whitening transform
        mean, whiten_matrix, info = compute_whitening_transform(
            scale_dir, scale,
            n_sample_pixels=args.n_sample_pixels,
            pca_dim=args.pca_dim,
            device=args.device,
        )

        # Save transform
        out_scale_dir = output_dir / scale
        out_scale_dir.mkdir(parents=True, exist_ok=True)
        transform_path = out_scale_dir / '_whitening_transform.pt'
        torch.save({
            'mean': mean,
            'whiten_matrix': whiten_matrix,
            'eigenvalues': info['eigenvalues'],
            'n_samples': info['n_samples'],
        }, transform_path)
        print(f'  Saved transform → {transform_path}')

        # Apply whitening to all features
        pt_files = sorted(glob.glob(str(scale_dir / '*.pt')))
        # Skip transform file itself
        pt_files = [f for f in pt_files if not os.path.basename(f).startswith('_')]

        for fpath in tqdm(pt_files, desc=f'[{scale}] Whitening'):
            feat = torch.load(fpath, map_location='cpu')
            whitened = apply_whitening(feat, mean, whiten_matrix)

            # Build output filename with new shape
            fname = os.path.basename(fpath)
            # Update shape in filename: rgb_0_coarse_CxHxW.pt → rgb_0_coarse_DxHxW.pt
            D, H, W = whitened.shape
            # Find and replace shape pattern (digits x digits x digits before .pt)
            import re
            new_fname = re.sub(
                r'\d+x\d+x\d+\.pt$',
                f'{D}x{H}x{W}.pt',
                fname,
            )
            out_path = out_scale_dir / new_fname
            torch.save(whitened, out_path)

        print(f'  ✓ Whitened {len(pt_files)} files → {out_scale_dir}')

    # Copy non-scale directories (cls, vis, etc.) if not in-place
    if not args.inplace and output_dir != feature_dir:
        for item in feature_dir.iterdir():
            if item.is_dir() and item.name not in args.scales:
                dest = output_dir / item.name
                if not dest.exists():
                    shutil.copytree(str(item), str(dest))
                    print(f'\nCopied {item.name}/ → {dest}')

    print(f'\n✓ Done! Whitened features saved to {output_dir}')


if __name__ == '__main__':
    main()
