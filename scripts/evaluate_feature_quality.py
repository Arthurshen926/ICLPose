#!/usr/bin/env python3
"""
Feature Quality Evaluation & Visualization
===========================================
Comprehensive diagnostic tool for assessing feature embedding quality across
the ICLPose pipeline. Measures and visualizes:

1. **Cosine Similarity Distribution**: Per-pixel cosine similarity between
   rendered and GT features — healthy features should show a peaked distribution
   with distinct low-similarity regions, not a flat near-1.0 plateau.

2. **Distinctiveness Score**: For each pixel, max cosine similarity to all other
   pixels. Higher distinctiveness = better localizability. Feature collapse shows
   as distinctiveness → 1.0 everywhere.

3. **Channel Statistics**: Per-channel mean and std across spatial positions.
   Healthy features have near-zero mean (after standardization) and varied std.
   Feature collapse shows large shared mean with small per-pixel variance.

4. **PCA Variance Spectrum**: Eigenvalue distribution — how many principal
   components are needed to explain 90/95/99% of variance. Collapsed features
   concentrate variance in very few components.

5. **Correlation Map Visualization**: For selected query pixels, visualize the
   correlation response across the feature map. Sharp peaks = discriminative,
   flat response = collapsed.

Usage:
  # Evaluate extracted features (offline, no GPU rendering needed)
  python scripts/evaluate_feature_quality.py \
    --feature_dir output/features_multiscale/room_0 \
    --output_dir output/feature_quality/room_0 \
    --n_frames 10

  # Compare raw vs whitened
  python scripts/evaluate_feature_quality.py \
    --feature_dir output/features_multiscale/room_0 \
    --compare_dir output/features_multiscale_whitened/room_0 \
    --output_dir output/feature_quality/room_0_comparison
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def compute_distinctiveness(feat: torch.Tensor, n_sample: int = 500) -> dict:
    """
    Compute per-pixel distinctiveness metrics.

    Args:
        feat: (C, H, W) L2-normalized feature map
        n_sample: max pixels to compute full pairwise similarity for

    Returns:
        dict with keys: mean_self_sim, distinctiveness_map, top1_sim_excl_self
    """
    C, H, W = feat.shape
    N = H * W
    flat = feat.reshape(C, N).t()  # (N, C)

    # Subsample for efficiency
    idx = torch.randperm(N)[:min(n_sample, N)]
    sub = flat[idx]  # (M, C)

    # Pairwise cosine similarity
    sim = sub @ sub.t()  # (M, M), values in [-1, 1]

    # Self-similarity stats (excluding diagonal)
    mask = ~torch.eye(len(idx), dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]

    # Per-pixel: max similarity to any other pixel (lower = more distinctive)
    sim_no_self = sim.clone()
    sim_no_self.fill_diagonal_(-float('inf'))
    top1_sim = sim_no_self.max(dim=1)[0]

    return {
        'mean_cosim': off_diag.mean().item(),
        'std_cosim': off_diag.std().item(),
        'median_cosim': off_diag.median().item(),
        'mean_top1_sim': top1_sim.mean().item(),
        'distinctiveness': (1.0 - top1_sim.mean()).item(),
        'sim_histogram': off_diag.cpu().numpy(),
    }


def compute_channel_stats(feat: torch.Tensor) -> dict:
    """
    Per-channel spatial statistics.

    Args:
        feat: (C, H, W) feature map

    Returns:
        dict with per-channel mean, std, and global stats
    """
    C, H, W = feat.shape
    # Per-channel: mean and std over spatial (H, W)
    ch_mean = feat.mean(dim=(1, 2))   # (C,)
    ch_std = feat.std(dim=(1, 2))     # (C,)

    return {
        'channel_means': ch_mean.cpu().numpy(),
        'channel_stds': ch_std.cpu().numpy(),
        'mean_of_means': ch_mean.mean().item(),
        'std_of_means': ch_mean.std().item(),
        'mean_of_stds': ch_std.mean().item(),
        'mean_bias_norm': ch_mean.norm().item(),  # L2 norm of mean vector
    }


def compute_pca_spectrum(feat: torch.Tensor, n_sample: int = 5000) -> dict:
    """
    PCA eigenvalue spectrum analysis.

    Args:
        feat: (C, H, W) feature map
        n_sample: pixels to sample for covariance

    Returns:
        dict with eigenvalues, cumulative variance, etc.
    """
    C, H, W = feat.shape
    N = H * W
    flat = feat.reshape(C, N).t()  # (N, C)

    idx = torch.randperm(N)[:min(n_sample, N)]
    sub = flat[idx].float()  # (M, C)

    mean = sub.mean(dim=0)
    centered = sub - mean
    cov = (centered.t() @ centered) / (len(idx) - 1)

    eigenvalues = torch.linalg.eigvalsh(cov).flip(0)
    total_var = eigenvalues.sum()
    cumvar = eigenvalues.cumsum(0) / total_var

    # Find dimensions for 90/95/99% variance
    dims_90 = (cumvar < 0.90).sum().item() + 1
    dims_95 = (cumvar < 0.95).sum().item() + 1
    dims_99 = (cumvar < 0.99).sum().item() + 1

    return {
        'eigenvalues': eigenvalues.cpu().numpy(),
        'cumulative_variance': cumvar.cpu().numpy(),
        'top1_ratio': (eigenvalues[0] / total_var).item(),
        'top5_ratio': (cumvar[min(4, len(cumvar) - 1)]).item(),
        'dims_90': dims_90,
        'dims_95': dims_95,
        'dims_99': dims_99,
        'effective_dim': (total_var ** 2 / (eigenvalues ** 2).sum()).item(),
    }


def compute_correlation_maps(feat: torch.Tensor, query_points: list) -> list:
    """
    Compute correlation response maps for selected query points.

    Args:
        feat: (C, H, W)
        query_points: list of (y, x) pixel coordinates

    Returns:
        list of (H, W) correlation maps
    """
    C, H, W = feat.shape
    flat = feat.reshape(C, H * W)  # (C, N)

    corr_maps = []
    for (qy, qx) in query_points:
        qy = min(qy, H - 1)
        qx = min(qx, W - 1)
        q_feat = feat[:, qy, qx]  # (C,)
        corr = (q_feat.unsqueeze(1) * flat).sum(dim=0)  # (N,)
        corr_maps.append(corr.reshape(H, W).cpu().numpy())

    return corr_maps


def visualize_all(metrics: dict, output_dir: Path, tag: str = ''):
    """Generate visualization plots from computed metrics."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  ⚠ matplotlib not available, skipping visualization")
        return

    prefix = f'{tag}_' if tag else ''

    # ── 1. Cosine Similarity Histogram ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for scale, m in metrics.items():
        if 'distinctiveness' in m:
            hist_data = m['distinctiveness']['sim_histogram']
            ax.hist(hist_data, bins=100, alpha=0.6, label=scale, density=True)
    ax.set_xlabel('Pairwise Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Feature Pairwise Cosine Similarity Distribution')
    ax.legend()
    ax.axvline(x=0.0, color='gray', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{prefix}cosine_similarity_hist.png', dpi=150)
    plt.close(fig)

    # ── 2. Channel Statistics ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for scale, m in metrics.items():
        if 'channel_stats' in m:
            cs = m['channel_stats']
            x = np.arange(len(cs['channel_means']))
            axes[0].bar(x, cs['channel_means'], alpha=0.5, label=scale)
            axes[1].bar(x, cs['channel_stds'], alpha=0.5, label=scale)
    axes[0].set_ylabel('Channel Mean')
    axes[0].set_title('Per-Channel Spatial Mean (should be ≈0 after standardization)')
    axes[0].legend()
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Channel Std')
    axes[1].set_xlabel('Channel Index')
    axes[1].set_title('Per-Channel Spatial Std (should be roughly equal)')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f'{prefix}channel_stats.png', dpi=150)
    plt.close(fig)

    # ── 3. PCA Variance Spectrum ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for scale, m in metrics.items():
        if 'pca_spectrum' in m:
            ps = m['pca_spectrum']
            n_show = min(50, len(ps['eigenvalues']))
            x = np.arange(1, n_show + 1)
            ax1.plot(x, ps['eigenvalues'][:n_show], 'o-', label=scale, markersize=3)
            ax2.plot(np.arange(1, len(ps['cumulative_variance']) + 1),
                     ps['cumulative_variance'], '-', label=scale)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('PCA Eigenvalue Spectrum (top 50)')
    ax1.set_yscale('log')
    ax1.legend()
    ax2.set_xlabel('# Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('PCA Cumulative Variance')
    ax2.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f'{prefix}pca_spectrum.png', dpi=150)
    plt.close(fig)

    # ── 4. Correlation Maps ──
    for scale, m in metrics.items():
        if 'correlation_maps' not in m:
            continue
        corr_maps = m['correlation_maps']
        n = len(corr_maps)
        if n == 0:
            continue
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for i, cmap in enumerate(corr_maps):
            im = axes[i].imshow(cmap, cmap='hot', vmin=-0.2, vmax=1.0)
            qy, qx = m['query_points'][i]
            axes[i].plot(qx, qy, 'c+', markersize=15, markeredgewidth=2)
            axes[i].set_title(f'{scale} query ({qy},{qx})\nmax={cmap.max():.3f}')
            fig.colorbar(im, ax=axes[i], fraction=0.046)
        fig.suptitle(f'Correlation Maps — {scale}')
        fig.tight_layout()
        fig.savefig(output_dir / f'{prefix}correlation_maps_{scale}.png', dpi=150)
        plt.close(fig)

    # ── 5. Summary Table ──
    print(f'\n{"="*70}')
    print(f'Feature Quality Summary {f"({tag})" if tag else ""}')
    print(f'{"="*70}')
    print(f'{"Scale":<12} {"MeanCosSim":>11} {"Distinct":>9} {"MeanBias":>9} '
          f'{"TopEV%":>7} {"EffDim":>7} {"Dims95":>7}')
    print('-' * 70)
    for scale, m in metrics.items():
        d = m.get('distinctiveness', {})
        cs = m.get('channel_stats', {})
        ps = m.get('pca_spectrum', {})
        print(f'{scale:<12} '
              f'{d.get("mean_cosim", 0):>11.4f} '
              f'{d.get("distinctiveness", 0):>9.4f} '
              f'{cs.get("mean_bias_norm", 0):>9.4f} '
              f'{ps.get("top1_ratio", 0)*100:>6.1f}% '
              f'{ps.get("effective_dim", 0):>7.1f} '
              f'{ps.get("dims_95", 0):>7d}')
    print(f'{"="*70}')


def evaluate_features(feature_dir: Path, scales: list,
                      n_frames: int = 10, device: str = 'cpu') -> dict:
    """
    Evaluate feature quality for all scales.

    Returns:
        dict: {scale_name: {distinctiveness: ..., channel_stats: ..., pca_spectrum: ...}}
    """
    metrics = {}

    for scale in scales:
        scale_dir = feature_dir / scale
        if not scale_dir.exists():
            print(f'  ⚠ {scale_dir} not found, skipping')
            continue

        pt_files = sorted(glob.glob(str(scale_dir / '*.pt')))
        pt_files = [f for f in pt_files if not os.path.basename(f).startswith('_')]
        if not pt_files:
            continue

        # Sample frames
        indices = np.linspace(0, len(pt_files) - 1, min(n_frames, len(pt_files)),
                              dtype=int)

        all_dist = []
        all_ch = []
        all_pca = []

        print(f'\n[{scale}] Evaluating {len(indices)} frames from {len(pt_files)} total...')

        for fi in indices:
            feat = torch.load(pt_files[fi], map_location=device)  # (C, H, W)

            dist = compute_distinctiveness(feat)
            ch = compute_channel_stats(feat)
            pca = compute_pca_spectrum(feat)

            all_dist.append(dist)
            all_ch.append(ch)
            all_pca.append(pca)

        # Aggregate across frames
        agg_dist = {
            'mean_cosim': np.mean([d['mean_cosim'] for d in all_dist]),
            'std_cosim': np.mean([d['std_cosim'] for d in all_dist]),
            'mean_top1_sim': np.mean([d['mean_top1_sim'] for d in all_dist]),
            'distinctiveness': np.mean([d['distinctiveness'] for d in all_dist]),
            'sim_histogram': np.concatenate([d['sim_histogram'] for d in all_dist]),
        }
        agg_ch = {
            'channel_means': np.mean([c['channel_means'] for c in all_ch], axis=0),
            'channel_stds': np.mean([c['channel_stds'] for c in all_ch], axis=0),
            'mean_of_means': np.mean([c['mean_of_means'] for c in all_ch]),
            'std_of_means': np.mean([c['std_of_means'] for c in all_ch]),
            'mean_of_stds': np.mean([c['mean_of_stds'] for c in all_ch]),
            'mean_bias_norm': np.mean([c['mean_bias_norm'] for c in all_ch]),
        }
        agg_pca = {
            'eigenvalues': np.mean([p['eigenvalues'] for p in all_pca], axis=0),
            'cumulative_variance': np.mean([p['cumulative_variance'] for p in all_pca], axis=0),
            'top1_ratio': np.mean([p['top1_ratio'] for p in all_pca]),
            'top5_ratio': np.mean([p['top5_ratio'] for p in all_pca]),
            'dims_90': int(np.mean([p['dims_90'] for p in all_pca])),
            'dims_95': int(np.mean([p['dims_95'] for p in all_pca])),
            'dims_99': int(np.mean([p['dims_99'] for p in all_pca])),
            'effective_dim': np.mean([p['effective_dim'] for p in all_pca]),
        }

        # Compute correlation maps for one representative frame (middle one)
        mid_idx = indices[len(indices) // 2]
        feat = torch.load(pt_files[mid_idx], map_location=device)
        C, H, W = feat.shape
        query_points = [
            (H // 4, W // 4),       # top-left region
            (H // 2, W // 2),       # center
            (3 * H // 4, 3 * W // 4),  # bottom-right region
        ]
        corr_maps = compute_correlation_maps(feat, query_points)

        metrics[scale] = {
            'distinctiveness': agg_dist,
            'channel_stats': agg_ch,
            'pca_spectrum': agg_pca,
            'correlation_maps': corr_maps,
            'query_points': query_points,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate feature quality')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Feature directory (e.g. output/features_multiscale/room_0)')
    parser.add_argument('--compare_dir', type=str, default=None,
                        help='Optional second directory for A/B comparison')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: feature_dir/quality_eval)')
    parser.add_argument('--scales', nargs='+',
                        default=['coarse', 'mid', 'fine_sd', 'fine_dino'],
                        help='Scales to evaluate')
    parser.add_argument('--n_frames', type=int, default=10,
                        help='Number of frames to sample per scale')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir) if args.output_dir else feature_dir / 'quality_eval'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Feature dir: {feature_dir}')
    print(f'Output dir:  {output_dir}')

    # Evaluate primary features
    metrics = evaluate_features(feature_dir, args.scales, args.n_frames, args.device)
    visualize_all(metrics, output_dir, tag='raw')

    # Save raw metrics (without large arrays for JSON serialization)
    summary = {}
    for scale, m in metrics.items():
        summary[scale] = {
            'distinctiveness': {k: v for k, v in m['distinctiveness'].items()
                                if k != 'sim_histogram'},
            'channel_stats': {k: v for k, v in m['channel_stats'].items()
                              if not isinstance(v, np.ndarray)},
            'pca_spectrum': {k: v for k, v in m['pca_spectrum'].items()
                            if not isinstance(v, np.ndarray)},
        }
    torch.save(summary, output_dir / 'metrics_summary.pt')

    # Optional A/B comparison
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        print(f'\n\n{"#"*70}')
        print(f'Comparison: {compare_dir}')
        print(f'{"#"*70}')
        metrics_b = evaluate_features(compare_dir, args.scales, args.n_frames, args.device)
        visualize_all(metrics_b, output_dir, tag='whitened')

        # Side-by-side comparison
        print(f'\n{"="*70}')
        print('A/B Comparison')
        print(f'{"="*70}')
        print(f'  A (raw):      {feature_dir}')
        print(f'  B (whitened):  {compare_dir}')
        print(f'\n{"Scale":<12} {"A CosSim":>9} {"B CosSim":>9} | '
              f'{"A Dist":>7} {"B Dist":>7} | '
              f'{"A Dims95":>8} {"B Dims95":>8}')
        print('-' * 70)
        for scale in args.scales:
            if scale not in metrics or scale not in metrics_b:
                continue
            da = metrics[scale]['distinctiveness']
            db = metrics_b[scale]['distinctiveness']
            pa = metrics[scale]['pca_spectrum']
            pb = metrics_b[scale]['pca_spectrum']
            print(f'{scale:<12} '
                  f'{da["mean_cosim"]:>9.4f} {db["mean_cosim"]:>9.4f} | '
                  f'{da["distinctiveness"]:>7.4f} {db["distinctiveness"]:>7.4f} | '
                  f'{pa["dims_95"]:>8d} {pb["dims_95"]:>8d}')

    print(f'\n✓ Results saved to {output_dir}')


if __name__ == '__main__':
    main()
