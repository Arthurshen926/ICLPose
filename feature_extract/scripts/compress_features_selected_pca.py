#!/usr/bin/env python3
"""
Channel-selected PCA compression: select top-K channels from raw features
based on importance scores, then apply PCA on the reduced channel set.

Key insight: Standard PCA on all 512d mixes in noisy channels.
Channel selection first removes noise, then PCA finds better projections.

Pipeline:
  raw 512d → select top-K channels → PCA on K channels → target_dim

Usage:
    python scripts/compress_features_selected_pca.py \
        --input_dir output/features_multiscale_stride7/OldHospital \
        --analysis_dir output/channel_analysis/OldHospital \
        --output_dir output/features_selected_pca/OldHospital_indexed \
        --traj_path output/features_pca_nomean/OldHospital_indexed/traj_w_c.txt \
        --top_k 256  \
        --no_mean
"""
import argparse
import os
import sys
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCALES = ['coarse', 'mid', 'fine_sd', 'fine_dino']

TARGET_DIMS = {
    'coarse': 32,
    'mid': 64,
    'fine_sd': 64,
    'fine_dino': 64,
}

TARGET_RES = {
    'coarse': (15, 26),
    'mid': (30, 53),
    'fine_sd': (69, 121),
    'fine_dino': (69, 121),
}


def load_channel_selection(analysis_dir, scale, top_k):
    """Load channel importance scores and return top-K indices."""
    scores_path = os.path.join(analysis_dir, f'{scale}_channel_scores.npz')
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Channel scores not found: {scores_path}")

    data = np.load(scores_path)
    ranked = data['ranked_indices']  # Already sorted by combined score descending
    combined = data['combined']

    n_channels = len(combined)
    k = min(top_k, n_channels)
    selected = sorted(ranked[:k].tolist())  # Sort for consistent ordering

    total_score = combined.sum()
    selected_score = combined[selected].sum()
    print(f"  Selected {k}/{n_channels} channels, "
          f"capturing {selected_score/total_score*100:.1f}% of importance")

    return selected


def load_all_features(feat_dir, scale):
    """Load all feature files for a scale."""
    scale_dir = os.path.join(feat_dir, scale)
    files = sorted([f for f in os.listdir(scale_dir) if f.endswith('.pt')])

    features = {}
    for fn in tqdm(files, desc=f"Loading {scale}", leave=False):
        feat = torch.load(os.path.join(scale_dir, fn), map_location='cpu')
        if isinstance(feat, dict):
            feat = list(feat.values())[0]
        # Extract stem
        parts = fn.rsplit(f'_{scale}_', 1)
        stem = parts[0] if len(parts) > 1 else fn.replace('.pt', '')
        features[stem] = feat.float()

    return features


def fit_pca(features_dict, selected_channels, target_dim, no_mean=True, max_samples=50000):
    """Fit PCA on selected channels only.

    Returns: mean (K,), components (target_dim, K), singular_values
    """
    all_pixels = []
    for feat in features_dict.values():
        # Select channels first
        feat_sel = feat[selected_channels]  # (K, H, W)
        C, H, W = feat_sel.shape
        all_pixels.append(feat_sel.reshape(C, -1).T)  # (HW, K)

    X = torch.cat(all_pixels, dim=0).numpy()  # (N, K)
    print(f"  Total pixels for PCA: {X.shape[0]}, channels: {X.shape[1]}")

    rng = np.random.RandomState(42)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    if no_mean:
        mean = np.zeros(X_sub.shape[1], dtype=np.float32)
        X_centered = X_sub
    else:
        mean = X_sub.mean(axis=0).astype(np.float32)
        X_centered = X_sub - mean

    print(f"  Computing SVD on {X_centered.shape} (no_mean={no_mean})...")
    _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:target_dim]  # (target_dim, K)
    explained = (S[:target_dim] ** 2).sum() / (S ** 2).sum()
    print(f"  Explained variance: {explained * 100:.1f}%")

    return mean, components, S


def project_features(feat, selected_channels, mean, components, target_res):
    """Project a single feature tensor: select channels → PCA → target resolution."""
    C_in, H_in, W_in = feat.shape
    C_out = components.shape[0]
    H_out, W_out = target_res

    # Select channels
    feat_sel = feat[selected_channels]  # (K, H_in, W_in)

    # Resize spatial dimensions if needed
    if (H_in, W_in) != (H_out, W_out):
        feat_sel = F.interpolate(
            feat_sel.unsqueeze(0), size=(H_out, W_out),
            mode='bilinear', align_corners=False
        ).squeeze(0)

    K, H, W = feat_sel.shape
    pixels = feat_sel.reshape(K, -1).T  # (H*W, K)

    mean_t = torch.from_numpy(mean).float()
    comp_t = torch.from_numpy(components).float()

    centered = pixels - mean_t
    projected = centered @ comp_t.T  # (H*W, C_out)

    return projected.T.reshape(C_out, H, W)


def verify_distinctiveness(proj_sample):
    """Compute off-diagonal correlation as quality metric."""
    fn = F.normalize(proj_sample.unsqueeze(0), dim=1)
    C, H, W = proj_sample.shape
    q = fn.reshape(1, C, -1).permute(0, 2, 1)
    r = fn.reshape(1, C, -1)
    corr = torch.bmm(q, r).squeeze(0)
    off_diag = (corr.sum() - torch.diagonal(corr).sum()) / (H * W * (H * W - 1))
    gap = 1.0 - off_diag.item()
    return off_diag.item(), gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--analysis_dir', required=True,
                        help='Dir with channel_scores.npz files from analyze_channel_importance.py')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--traj_path', required=True, help='Trajectory file to copy')
    parser.add_argument('--top_k', type=int, default=256,
                        help='Number of top channels to select per scale')
    parser.add_argument('--no_mean', action='store_true', default=True,
                        help='Skip mean subtraction in PCA (default: True)')
    parser.add_argument('--max_pca_samples', type=int, default=50000)
    parser.add_argument('--scales', nargs='+', default=SCALES)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy trajectory
    traj_out = os.path.join(args.output_dir, 'traj_w_c.txt')
    if not os.path.exists(traj_out) and os.path.exists(args.traj_path):
        shutil.copy2(args.traj_path, traj_out)
        print(f"Copied trajectory to {traj_out}")

    # Build frame index mapping
    all_stems = set()
    for scale in args.scales:
        scale_dir = os.path.join(args.input_dir, scale)
        if not os.path.isdir(scale_dir):
            continue
        for fn in os.listdir(scale_dir):
            if fn.endswith('.pt'):
                parts = fn.rsplit(f'_{scale}_', 1)
                stem = parts[0] if len(parts) > 1 else fn.replace('.pt', '')
                all_stems.add(stem)

    stems_sorted = sorted(all_stems)
    stem_to_idx = {s: i for i, s in enumerate(stems_sorted)}
    print(f"Total frames: {len(stems_sorted)}")

    # Save params
    params_dir = os.path.join(args.output_dir, 'pca_params')
    os.makedirs(params_dir, exist_ok=True)

    for scale in args.scales:
        scale_dir = os.path.join(args.input_dir, scale)
        if not os.path.isdir(scale_dir):
            print(f"Skipping {scale}: not found")
            continue

        target_dim = TARGET_DIMS[scale]
        target_res = TARGET_RES[scale]

        print(f"\n{'=' * 60}")
        print(f"Processing {scale}: top-{args.top_k} channels → {target_dim}d")
        print(f"{'=' * 60}")

        # Load channel selection
        selected = load_channel_selection(args.analysis_dir, scale, args.top_k)

        # Load all features
        features = load_all_features(args.input_dir, scale)
        print(f"  Loaded {len(features)} features")

        # Fit PCA on selected channels
        mean, components, S = fit_pca(
            features, selected, target_dim,
            no_mean=args.no_mean, max_samples=args.max_pca_samples
        )

        # Save params (including selected channels for reproducibility)
        np.savez(
            os.path.join(params_dir, f'{scale}_pca.npz'),
            mean=mean,
            components=components,
            singular_values=S[:target_dim],
            selected_channels=np.array(selected),
            top_k=args.top_k,
        )

        # Project and save
        out_scale_dir = os.path.join(args.output_dir, scale)
        os.makedirs(out_scale_dir, exist_ok=True)

        sample_proj = None
        for stem, feat in tqdm(features.items(), desc=f"Projecting {scale}"):
            projected = project_features(feat, selected, mean, components, target_res)

            idx = stem_to_idx[stem]
            out_name = f'rgb_{idx}_{scale}_{target_dim}x{target_res[0]}x{target_res[1]}.pt'
            torch.save(projected, os.path.join(out_scale_dir, out_name))

            if sample_proj is None:
                sample_proj = projected

        # Verify quality
        off_diag, gap = verify_distinctiveness(sample_proj)
        print(f"  Quality: off_diag={off_diag:.4f}, gap={gap:.4f}")

    print(f"\n✓ Channel-selected PCA compression complete!")
    print(f"  Output: {args.output_dir}")
    print(f"  PCA params: {params_dir}")


if __name__ == '__main__':
    main()
