#!/usr/bin/env python3
"""
Analyze per-channel importance in raw high-dimensional features (512d SD / 768d DINO).

Three metrics per channel:
  1. Spatial Distinctiveness: std over spatial dimensions across all frames.
     High = channel varies spatially → useful for matching.
  2. Cross-View Stability: 1 - relative change between consecutive frames in same sequence.
     High = channel is stable under viewpoint change → reliable for pose estimation.
  3. Uniqueness: 1 - max absolute correlation with any other channel.
     High = channel carries non-redundant information.

Combined score = distinctiveness * stability * uniqueness

Usage:
    python scripts/analyze_channel_importance.py \
        --input_dir output/features_multiscale_stride7/OldHospital \
        --output_dir output/channel_analysis/OldHospital \
        --max_frames 200
"""
import argparse
import os
import sys
import json

import numpy as np
import torch
from tqdm import tqdm

SCALES = ['coarse', 'mid', 'fine_sd', 'fine_dino']


def parse_frame_info(filename, scale):
    """Extract (seq_name, frame_idx) from filename like seq1_frame00001_coarse_512x15x26.pt"""
    stem = filename.rsplit(f'_{scale}_', 1)[0]
    parts = stem.split('_frame')
    seq = parts[0]
    frame_idx = int(parts[1]) if len(parts) > 1 else 0
    return seq, frame_idx, stem


def load_features_by_sequence(feat_dir, scale, max_frames=None):
    """Load features grouped by sequence, sorted by frame index."""
    scale_dir = os.path.join(feat_dir, scale)
    files = sorted([f for f in os.listdir(scale_dir) if f.endswith('.pt')])

    seq_dict = {}
    for fn in files:
        seq, fidx, stem = parse_frame_info(fn, scale)
        if seq not in seq_dict:
            seq_dict[seq] = []
        seq_dict[seq].append((fidx, fn, stem))

    # Sort each sequence by frame index
    for seq in seq_dict:
        seq_dict[seq].sort(key=lambda x: x[0])

    # Subsample if needed
    if max_frames is not None:
        total = sum(len(v) for v in seq_dict.values())
        if total > max_frames:
            keep_ratio = max_frames / total
            for seq in seq_dict:
                n_keep = max(2, int(len(seq_dict[seq]) * keep_ratio))
                indices = np.linspace(0, len(seq_dict[seq]) - 1, n_keep, dtype=int)
                seq_dict[seq] = [seq_dict[seq][i] for i in indices]

    # Load tensors
    result = {}
    total_loaded = 0
    for seq in sorted(seq_dict.keys()):
        frames = []
        for fidx, fn, stem in seq_dict[seq]:
            feat = torch.load(os.path.join(scale_dir, fn), map_location='cpu')
            if isinstance(feat, dict):
                feat = list(feat.values())[0]
            frames.append(feat.float())
            total_loaded += 1
        result[seq] = frames

    print(f"  Loaded {total_loaded} frames across {len(result)} sequences")
    return result


def compute_spatial_distinctiveness(seq_features):
    """Per-channel std over spatial dims, averaged across all frames."""
    all_stds = []
    for seq, frames in seq_features.items():
        for feat in frames:
            C, H, W = feat.shape
            # Std over spatial dimensions for each channel
            channel_std = feat.reshape(C, -1).std(dim=1)  # (C,)
            all_stds.append(channel_std)

    stds = torch.stack(all_stds)  # (N_frames, C)
    return stds.mean(dim=0)  # (C,)


def compute_cross_view_stability(seq_features):
    """Per-channel stability: 1 - mean relative change between consecutive frames."""
    all_changes = []
    for seq, frames in seq_features.items():
        if len(frames) < 2:
            continue
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            C = f1.shape[0]
            # Mean absolute value per channel
            m1 = f1.reshape(C, -1).abs().mean(dim=1).clamp(min=1e-6)
            m2 = f2.reshape(C, -1).abs().mean(dim=1).clamp(min=1e-6)
            # Relative change: |mean1 - mean2| / max(mean1, mean2)
            rel_change = (m1 - m2).abs() / torch.max(m1, m2)
            all_changes.append(rel_change)

    if not all_changes:
        return torch.ones(f1.shape[0])

    changes = torch.stack(all_changes).mean(dim=0)  # (C,)
    return 1.0 - changes


def compute_channel_uniqueness(seq_features, max_pixels=100000):
    """Per-channel uniqueness: 1 - max |correlation| with any other channel."""
    # Collect pixel features
    all_pixels = []
    for seq, frames in seq_features.items():
        for feat in frames:
            C, H, W = feat.shape
            all_pixels.append(feat.reshape(C, -1).T)  # (HW, C)

    X = torch.cat(all_pixels, dim=0)  # (N, C)

    # Subsample
    if X.shape[0] > max_pixels:
        indices = torch.randperm(X.shape[0])[:max_pixels]
        X = X[indices]

    # Compute correlation matrix
    X_centered = X - X.mean(dim=0, keepdim=True)
    norms = X_centered.norm(dim=0, keepdim=True).clamp(min=1e-6)
    X_normed = X_centered / norms
    corr_matrix = X_normed.T @ X_normed / X.shape[0]  # (C, C)

    # For each channel, max absolute correlation with OTHER channels
    C = corr_matrix.shape[0]
    corr_abs = corr_matrix.abs()
    corr_abs.fill_diagonal_(0)  # Exclude self-correlation
    max_corr = corr_abs.max(dim=1).values  # (C,)

    return 1.0 - max_corr  # High uniqueness = low max correlation


def analyze_scale(feat_dir, scale, max_frames=200):
    """Analyze one scale, return per-channel scores."""
    print(f"\n{'='*60}")
    print(f"Analyzing {scale}")
    print(f"{'='*60}")

    seq_features = load_features_by_sequence(feat_dir, scale, max_frames)

    print("  Computing spatial distinctiveness...")
    distinctiveness = compute_spatial_distinctiveness(seq_features)

    print("  Computing cross-view stability...")
    stability = compute_cross_view_stability(seq_features)

    print("  Computing channel uniqueness...")
    uniqueness = compute_channel_uniqueness(seq_features)

    # Normalize each metric to [0, 1] range
    def norm01(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return torch.ones_like(x)
        return (x - xmin) / (xmax - xmin)

    d_norm = norm01(distinctiveness)
    s_norm = norm01(stability)
    u_norm = norm01(uniqueness)

    # Combined score
    combined = d_norm * s_norm * u_norm

    # Rank by combined score
    ranked_indices = combined.argsort(descending=True)

    C = len(combined)
    print(f"\n  Channel statistics ({C} channels):")
    print(f"    Distinctiveness: mean={distinctiveness.mean():.4f}, std={distinctiveness.std():.4f}")
    print(f"    Stability:       mean={stability.mean():.4f}, std={stability.std():.4f}")
    print(f"    Uniqueness:      mean={uniqueness.mean():.4f}, std={uniqueness.std():.4f}")
    print(f"    Combined:        mean={combined.mean():.4f}, std={combined.std():.4f}")
    print(f"\n  Top 10 channels: {ranked_indices[:10].tolist()}")
    print(f"  Bottom 10 channels: {ranked_indices[-10:].tolist()}")

    # Cumulative explained importance for top-K
    sorted_scores = combined[ranked_indices]
    total = sorted_scores.sum()
    cumsum = sorted_scores.cumsum(0) / total
    for frac in [0.5, 0.8, 0.9, 0.95]:
        k = (cumsum >= frac).nonzero(as_tuple=True)[0]
        k = k[0].item() + 1 if len(k) > 0 else C
        print(f"    Top-{k:3d} channels capture {frac*100:.0f}% of combined importance")

    return {
        'distinctiveness': distinctiveness.numpy(),
        'stability': stability.numpy(),
        'uniqueness': uniqueness.numpy(),
        'combined': combined.numpy(),
        'ranked_indices': ranked_indices.numpy(),
        'd_norm': d_norm.numpy(),
        's_norm': s_norm.numpy(),
        'u_norm': u_norm.numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Raw uncompressed features dir (e.g., output/features_multiscale_stride7/OldHospital)')
    parser.add_argument('--output_dir', required=True,
                        help='Output dir for analysis results')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Max frames to load per scale (subsampled uniformly)')
    parser.add_argument('--scales', nargs='+', default=SCALES,
                        help='Scales to analyze')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for scale in args.scales:
        scale_dir = os.path.join(args.input_dir, scale)
        if not os.path.isdir(scale_dir):
            print(f"Skipping {scale}: {scale_dir} not found")
            continue
        results[scale] = analyze_scale(args.input_dir, scale, args.max_frames)

    # Save per-scale results
    for scale, data in results.items():
        out_path = os.path.join(args.output_dir, f'{scale}_channel_scores.npz')
        np.savez(out_path, **data)
        print(f"Saved {scale} analysis to {out_path}")

    # Save summary JSON
    summary = {}
    for scale, data in results.items():
        C = len(data['combined'])
        ranked = data['ranked_indices'].tolist()
        summary[scale] = {
            'n_channels': C,
            'top_20_channels': ranked[:20],
            'bottom_20_channels': ranked[-20:],
            'combined_mean': float(data['combined'].mean()),
            'combined_std': float(data['combined'].std()),
        }

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
