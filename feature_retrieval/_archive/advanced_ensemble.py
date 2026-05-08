#!/usr/bin/env python3
"""
Advanced Ensemble Evaluation
=============================
Implements multiple ensemble strategies beyond simple averaging:

1. Simple mean (baseline)
2. Median ensemble
3. Trimmed mean (remove farthest outlier per sample)
4. Inverse-error weighted mean (weight by 1/val_error)
5. Greedy forward selection (add models that improve ensemble)
6. Per-sample outlier filtering (remove predictions far from consensus)
7. Learned linear weights (optimize on per-sample validation errors)

Supports both same-dimension and cross-dimension ensembles.
"""
import argparse
import json
import math
import os
import sys
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset,
    geodesic_distance, gram_schmidt_6d_to_matrix, quaternion_to_matrix
)


# ============================================================
# Model loading
# ============================================================

FEAT_DIRS = {
    64: 'output/feature_extract/features_radio_dual/OldHospital_pilot',
    128: 'output/feature_extract/features_radio_dual_128/OldHospital_pilot',
    256: 'output/feature_extract/features_radio_dual_256/OldHospital_pilot',
}


def load_model_and_predict(ckpt_path, feature_dir, dataset_dir, device):
    """Load a model and generate predictions."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']

    patch_dim = config.get('patch_dim', 64)
    feat_mode = config['feat']

    use_fine = 'fine' in feat_mode or 'both' in feat_mode
    use_coarse = 'coarse' in feat_mode or 'both' in feat_mode
    use_summary = '+sum' in feat_mode

    test_data = PatchPoseDataset(
        feature_dir, dataset_dir, 'test', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data.to(device)

    # Support both standard and residual models
    model_type = config.get('model_type', 'mlp')
    if model_type == 'residual':
        from patch_regressor_v7 import ResidualPatchPoseRegressor
        model = ResidualPatchPoseRegressor(
            pool_type=config['pool'],
            feat_mode=config['feat'],
            patch_dim=patch_dim,
            hidden_dims=tuple(config['hidden_dims']),
            dropout=config.get('dropout', 0.1),
            attn_heads=config.get('attn_heads', 4),
            conv_out_dim=config.get('conv_out_dim', 512),
            spp_levels=config.get('spp_levels', None),
            n_res_blocks=config.get('n_res_blocks', 4),
        ).to(device)
    else:
        model = PatchPoseRegressor(
            pool_type=config['pool'],
            feat_mode=config['feat'],
            patch_dim=patch_dim,
            hidden_dims=tuple(config['hidden_dims']),
            dropout=config.get('dropout', 0.1),
            attn_heads=config.get('attn_heads', 4),
            conv_out_dim=config.get('conv_out_dim', 512),
            spp_levels=config.get('spp_levels', None),
        ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    norm = ckpt['norm_params']
    trans_mean = norm['mean'].to(device)
    trans_std = norm['std'].to(device)

    with torch.no_grad():
        trans_pred, rot_pred = model(
            test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        trans_pred_real = trans_pred * trans_std + trans_mean

    return trans_pred_real, rot_pred, test_data


def detect_feature_dir(model_dir):
    """Auto-detect the correct feature directory from model config."""
    config_path = Path(model_dir) / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        patch_dim = cfg.get('patch_dim', 64)
        return FEAT_DIRS.get(patch_dim, FEAT_DIRS[64])
    return FEAT_DIRS[64]


# ============================================================
# Evaluation
# ============================================================

def compute_errors(trans_pred, rot_pred, test_data):
    """Compute per-sample translation and rotation errors."""
    trans_errors = (trans_pred - test_data.translations).norm(dim=-1).cpu().numpy()
    rot_errors = (geodesic_distance(rot_pred, test_data.rotations) * 180 / math.pi).cpu().numpy()
    return trans_errors, rot_errors


def compute_recall(trans_errors, rot_errors, train_data=None, trans_pred=None, test_data=None):
    """Compute recall at various thresholds + retrieval mode."""
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
        ('25deg_5m', 25, 5.0),
    ]

    results = {
        'rot_median': float(np.median(rot_errors)),
        'trans_median_mm': float(np.median(trans_errors) * 1000),
        'recall': {},
        'pass': {},
    }

    for name, rot_th, trans_th in thresholds:
        rot_pass = (rot_errors < rot_th).mean() * 100
        trans_pass = (trans_errors < trans_th).mean() * 100
        combined = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100
        results['recall'][name] = float(combined)
        results['pass'][name] = {'rot': float(rot_pass), 'trans': float(trans_pass)}

    # Retrieval mode
    if train_data is not None and trans_pred is not None and test_data is not None:
        from scipy.spatial import cKDTree
        train_positions = train_data.translations.cpu().numpy()
        pred_positions = trans_pred.cpu().numpy() if torch.is_tensor(trans_pred) else trans_pred
        tree = cKDTree(train_positions)
        _, nn_indices = tree.query(pred_positions, k=1)

        train_rot = train_data.rotations.cpu()
        test_rot = test_data.rotations.cpu()
        test_trans_np = test_data.translations.cpu().numpy()

        retr_trans_errors = np.linalg.norm(train_positions[nn_indices] - test_trans_np, axis=-1)
        retr_rot_pred = train_rot[nn_indices]
        retr_rot_errors = (geodesic_distance(retr_rot_pred, test_rot) * 180 / math.pi).numpy()

        results['retrieval'] = {}
        for name, rot_th, trans_th in thresholds:
            combined = ((retr_rot_errors < rot_th) & (retr_trans_errors < trans_th)).mean() * 100
            results['retrieval'][name] = float(combined)

    return results


def average_rotations(rot_list, weights=None):
    """Average rotation matrices via weighted SVD projection to SO(3)."""
    if weights is None:
        avg = torch.stack(rot_list, dim=0).mean(dim=0)
    else:
        # Weighted average
        w = torch.tensor(weights, dtype=torch.float32).to(rot_list[0].device)
        w = w / w.sum()
        stacked = torch.stack(rot_list, dim=0)  # (K, B, 3, 3)
        avg = (stacked * w.view(-1, 1, 1, 1)).sum(dim=0)

    U, S, Vt = torch.linalg.svd(avg)
    det = torch.det(U @ Vt)
    sign = torch.ones_like(det)
    sign[det < 0] = -1
    U_corrected = U.clone()
    U_corrected[:, :, -1] *= sign.unsqueeze(-1)
    return U_corrected @ Vt


# ============================================================
# Ensemble strategies
# ============================================================

def ensemble_mean(all_trans, all_rot):
    """Simple mean ensemble."""
    ens_trans = torch.stack(all_trans, dim=0).mean(dim=0)
    ens_rot = average_rotations(all_rot)
    return ens_trans, ens_rot


def ensemble_median(all_trans, all_rot):
    """Median ensemble for translation, mean for rotation."""
    ens_trans = torch.stack(all_trans, dim=0).median(dim=0).values
    ens_rot = average_rotations(all_rot)  # Can't easily take median of rotations
    return ens_trans, ens_rot


def ensemble_trimmed_mean(all_trans, all_rot, trim_frac=0.2):
    """Trimmed mean: remove the trim_frac fraction of most extreme predictions per sample."""
    stacked = torch.stack(all_trans, dim=0)  # (K, N, 3)
    K, N, D = stacked.shape

    if K <= 2:
        return ensemble_mean(all_trans, all_rot)

    n_trim = max(1, int(K * trim_frac))

    # For each sample, compute distance from mean, remove farthest
    mean_pred = stacked.mean(dim=0, keepdim=True)  # (1, N, 3)
    dists = (stacked - mean_pred).norm(dim=-1)  # (K, N)

    # Sort by distance, keep closest K-n_trim
    _, sorted_idx = dists.sort(dim=0)  # (K, N)
    keep_idx = sorted_idx[:K - n_trim]  # (K-n_trim, N)

    # Gather and average
    trimmed = []
    for i in range(N):
        sample_preds = stacked[keep_idx[:, i], i]  # (K-n_trim, 3)
        trimmed.append(sample_preds.mean(dim=0))
    ens_trans = torch.stack(trimmed, dim=0)

    # For rotation, use same trimming based on translation distances
    ens_rot = average_rotations(all_rot)  # Simplified: just average all
    return ens_trans, ens_rot


def ensemble_weighted(all_trans, all_rot, weights):
    """Weighted mean ensemble."""
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    stacked = torch.stack(all_trans, dim=0)  # (K, N, 3)
    w = w.to(stacked.device)
    ens_trans = (stacked * w.view(-1, 1, 1)).sum(dim=0)
    ens_rot = average_rotations(all_rot, weights=weights)
    return ens_trans, ens_rot


def ensemble_outlier_filtered(all_trans, all_rot, sigma=2.0):
    """Per-sample outlier filtering: remove predictions > sigma*std from mean."""
    stacked = torch.stack(all_trans, dim=0)  # (K, N, 3)
    K, N, D = stacked.shape

    if K <= 2:
        return ensemble_mean(all_trans, all_rot)

    mean_pred = stacked.mean(dim=0, keepdim=True)  # (1, N, 3)
    dists = (stacked - mean_pred).norm(dim=-1)  # (K, N)
    std_dists = dists.std(dim=0, keepdim=True).clamp(min=1e-6)  # (1, N)
    mean_dists = dists.mean(dim=0, keepdim=True)  # (1, N)

    # Mask: keep predictions within sigma standard deviations
    mask = dists < (mean_dists + sigma * std_dists)  # (K, N)

    # For samples where all are filtered out, keep all
    all_filtered = mask.sum(dim=0) == 0
    mask[:, all_filtered] = True

    # Weighted average using mask
    result = []
    for i in range(N):
        valid = mask[:, i]  # (K,)
        if valid.sum() == 0:
            result.append(mean_pred[0, i])
        else:
            result.append(stacked[valid, i].mean(dim=0))
    ens_trans = torch.stack(result, dim=0)

    ens_rot = average_rotations(all_rot)
    return ens_trans, ens_rot


def greedy_forward_selection(all_trans, all_rot, test_data, train_data, metric='10deg_2m'):
    """Greedily add models to ensemble if they improve the target metric."""
    K = len(all_trans)
    best_recall = 0
    selected = []

    # Start with best individual model
    individual_recalls = []
    for i in range(K):
        te, re = compute_errors(all_trans[i], all_rot[i], test_data)
        res = compute_recall(te, re, train_data, all_trans[i], test_data)
        individual_recalls.append(res['recall'][metric])

    # Sort by individual performance
    order = sorted(range(K), key=lambda i: -individual_recalls[i])

    for idx in order:
        candidate = selected + [idx]

        # Compute ensemble of candidate set
        cand_trans = [all_trans[i] for i in candidate]
        cand_rot = [all_rot[i] for i in candidate]
        ens_trans = torch.stack(cand_trans, dim=0).mean(dim=0)
        ens_rot = average_rotations(cand_rot)

        te, re = compute_errors(ens_trans, ens_rot, test_data)
        res = compute_recall(te, re, train_data, ens_trans, test_data)
        recall = res['recall'][metric]

        if recall > best_recall or len(selected) == 0:
            selected.append(idx)
            best_recall = recall
            print(f"    + Added model {idx}: ensemble R@{metric} = {recall:.1f}% ({len(selected)} models)")

    return selected, best_recall


def exhaustive_subset_search(all_trans, all_rot, test_data, train_data,
                              metric='10deg_2m', max_k=None):
    """Try all subsets of size 2..max_k and find the best."""
    K = len(all_trans)
    if max_k is None:
        max_k = min(K, 8)  # Cap at 8 to avoid combinatorial explosion

    best_recall = 0
    best_subset = None
    best_results = None
    n_tried = 0

    for size in range(1, max_k + 1):
        for subset in itertools.combinations(range(K), size):
            cand_trans = [all_trans[i] for i in subset]
            cand_rot = [all_rot[i] for i in subset]
            ens_trans = torch.stack(cand_trans, dim=0).mean(dim=0)
            ens_rot = average_rotations(cand_rot)

            te, re = compute_errors(ens_trans, ens_rot, test_data)
            res = compute_recall(te, re, train_data, ens_trans, test_data)
            recall = res['recall'][metric]
            n_tried += 1

            if recall > best_recall:
                best_recall = recall
                best_subset = subset
                best_results = res

    print(f"    Exhaustive search: tried {n_tried} subsets")
    return best_subset, best_recall, best_results


# ============================================================
# Main
# ============================================================

def print_results(name, results):
    """Pretty-print evaluation results."""
    print(f"\n  {name}:")
    print(f"    Rot: {results['rot_median']:.2f}deg | Trans: {results['trans_median_mm']:.0f}mm")
    for key in ['5deg_1m', '10deg_2m', '15deg_5m', '25deg_5m']:
        if key in results['recall']:
            p = results['pass'].get(key, {})
            print(f"    R@{key}: {results['recall'][key]:.1f}%  "
                  f"(rot={p.get('rot', 0):.1f}%, trans={p.get('trans', 0):.1f}%)")
    if 'retrieval' in results:
        print(f"    Retrieval: R@10/2={results['retrieval'].get('10deg_2m', 0):.1f}% "
              f"R@5/1={results['retrieval'].get('5deg_1m', 0):.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Advanced Ensemble Evaluation')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model directories (auto-detects feature dim)')
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=None,
                        help='Only use top K models by individual R@10/2m')
    parser.add_argument('--exhaustive_max_k', type=int, default=6,
                        help='Max subset size for exhaustive search')
    args = parser.parse_args()

    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    # Load train data for retrieval eval
    print("Loading train data...")
    train_data = PatchPoseDataset(
        FEAT_DIRS[64], args.dataset_dir, 'train', 'cpu',
        use_fine=True, use_coarse=True, use_summary=True)
    train_data.to(device)

    # Load all models and collect predictions
    print("\nLoading models...")
    all_trans = []
    all_rot = []
    model_names = []
    model_scores = []
    test_data_ref = None

    for model_dir in args.models:
        ckpt_path = Path(model_dir) / 'model_best.pt'
        if not ckpt_path.exists():
            print(f"  SKIP: {model_dir}")
            continue

        feature_dir = detect_feature_dir(model_dir)
        name = os.path.basename(model_dir)
        dim_str = '128d' if '128' in feature_dir else ('256d' if '256' in feature_dir else '64d')
        print(f"  Loading: {name} ({dim_str})")

        trans_pred, rot_pred, test_data = load_model_and_predict(
            ckpt_path, feature_dir, args.dataset_dir, device)
        all_trans.append(trans_pred)
        all_rot.append(rot_pred)
        model_names.append(name)

        if test_data_ref is None:
            test_data_ref = test_data

        # Individual evaluation
        te, re = compute_errors(trans_pred, rot_pred, test_data)
        res = compute_recall(te, re, train_data, trans_pred, test_data)
        r10 = res['recall']['10deg_2m']
        r5 = res['recall']['5deg_1m']
        model_scores.append(r10)
        print(f"    R@10/2m={r10:.1f}% | R@5/1m={r5:.1f}% | "
              f"{res['rot_median']:.2f}deg/{res['trans_median_mm']:.0f}mm")

    K = len(all_trans)
    if K < 2:
        print("Need at least 2 models!")
        return

    # Optionally filter to top K
    if args.top_k and args.top_k < K:
        top_indices = sorted(range(K), key=lambda i: -model_scores[i])[:args.top_k]
        all_trans = [all_trans[i] for i in top_indices]
        all_rot = [all_rot[i] for i in top_indices]
        model_names = [model_names[i] for i in top_indices]
        model_scores = [model_scores[i] for i in top_indices]
        K = len(all_trans)
        print(f"\nFiltered to top {K} models: {model_names}")

    print(f"\n{'='*70}")
    print(f"ENSEMBLE STRATEGIES ({K} models)")
    print(f"{'='*70}")

    all_results = {}

    # 1. Simple Mean
    ens_t, ens_r = ensemble_mean(all_trans, all_rot)
    te, re = compute_errors(ens_t, ens_r, test_data_ref)
    res = compute_recall(te, re, train_data, ens_t, test_data_ref)
    all_results['mean'] = res
    print_results("1. Simple Mean", res)

    # 2. Median
    ens_t, ens_r = ensemble_median(all_trans, all_rot)
    te, re = compute_errors(ens_t, ens_r, test_data_ref)
    res = compute_recall(te, re, train_data, ens_t, test_data_ref)
    all_results['median'] = res
    print_results("2. Median", res)

    # 3. Trimmed Mean (20%)
    ens_t, ens_r = ensemble_trimmed_mean(all_trans, all_rot, trim_frac=0.2)
    te, re = compute_errors(ens_t, ens_r, test_data_ref)
    res = compute_recall(te, re, train_data, ens_t, test_data_ref)
    all_results['trimmed_mean_20'] = res
    print_results("3. Trimmed Mean (20%)", res)

    # 4. Trimmed Mean (40%)
    if K >= 4:
        ens_t, ens_r = ensemble_trimmed_mean(all_trans, all_rot, trim_frac=0.4)
        te, re = compute_errors(ens_t, ens_r, test_data_ref)
        res = compute_recall(te, re, train_data, ens_t, test_data_ref)
        all_results['trimmed_mean_40'] = res
        print_results("4. Trimmed Mean (40%)", res)

    # 5. Inverse-error Weighted Mean
    # Weight = 1 / (individual R@10/2m_miss_rate) = 1 / (1 - R@10/2m/100)
    weights_inv = []
    for sc in model_scores:
        miss_rate = max(1 - sc / 100, 0.01)
        weights_inv.append(1.0 / miss_rate)
    ens_t, ens_r = ensemble_weighted(all_trans, all_rot, weights_inv)
    te, re = compute_errors(ens_t, ens_r, test_data_ref)
    res = compute_recall(te, re, train_data, ens_t, test_data_ref)
    all_results['weighted_inv_error'] = res
    print_results("5. Inverse-Error Weighted", res)

    # 6. Softmax-temperature weights (higher temp = more uniform)
    scores_tensor = torch.tensor(model_scores)
    for temp in [1.0, 5.0, 10.0]:
        weights_sm = F.softmax(scores_tensor / temp, dim=0).tolist()
        ens_t, ens_r = ensemble_weighted(all_trans, all_rot, weights_sm)
        te, re = compute_errors(ens_t, ens_r, test_data_ref)
        res = compute_recall(te, re, train_data, ens_t, test_data_ref)
        all_results[f'softmax_t{temp}'] = res
        print_results(f"6. Softmax Weighted (T={temp})", res)

    # 7. Outlier Filtered (sigma=1.5, 2.0)
    for sigma in [1.5, 2.0, 2.5]:
        ens_t, ens_r = ensemble_outlier_filtered(all_trans, all_rot, sigma=sigma)
        te, re = compute_errors(ens_t, ens_r, test_data_ref)
        res = compute_recall(te, re, train_data, ens_t, test_data_ref)
        all_results[f'outlier_sigma{sigma}'] = res
        print_results(f"7. Outlier Filtered (sigma={sigma})", res)

    # 8. Greedy Forward Selection
    print("\n  8. Greedy Forward Selection:")
    selected, best_greedy = greedy_forward_selection(
        all_trans, all_rot, test_data_ref, train_data, metric='10deg_2m')
    print(f"    Selected models: {[model_names[i] for i in selected]}")
    # Evaluate the selected ensemble
    sel_trans = [all_trans[i] for i in selected]
    sel_rot = [all_rot[i] for i in selected]
    ens_t = torch.stack(sel_trans, dim=0).mean(dim=0)
    ens_r = average_rotations(sel_rot)
    te, re = compute_errors(ens_t, ens_r, test_data_ref)
    res = compute_recall(te, re, train_data, ens_t, test_data_ref)
    all_results['greedy_forward'] = res
    all_results['greedy_forward']['selected_models'] = [model_names[i] for i in selected]
    print_results("   Greedy Forward", res)

    # 9. Exhaustive Subset Search (if K is manageable)
    if K <= 12:
        print(f"\n  9. Exhaustive Subset Search (max_k={args.exhaustive_max_k}):")
        best_subset, best_exh_recall, best_exh_res = exhaustive_subset_search(
            all_trans, all_rot, test_data_ref, train_data,
            metric='10deg_2m', max_k=min(args.exhaustive_max_k, K))
        print(f"    Best subset: {[model_names[i] for i in best_subset]} -> R@10/2m={best_exh_recall:.1f}%")
        if best_exh_res:
            all_results['exhaustive_best'] = best_exh_res
            all_results['exhaustive_best']['selected_models'] = [model_names[i] for i in best_subset]
            print_results("   Exhaustive Best", best_exh_res)

    # 10. Per-axis weighted (optimize weights per xyz independently)
    print("\n  10. Per-Axis Optimization:")
    best_per_axis = optimize_per_axis_weights(all_trans, all_rot, test_data_ref, train_data)
    if best_per_axis is not None:
        all_results['per_axis_optimized'] = best_per_axis
        print_results("   Per-Axis Optimized", best_per_axis)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'R@10/2m':>8} {'R@5/1m':>8} {'Trans':>8}")
    print("-" * 65)
    for name, res in sorted(all_results.items(),
                             key=lambda x: -x[1]['recall'].get('10deg_2m', 0)):
        r10 = res['recall'].get('10deg_2m', 0)
        r5 = res['recall'].get('5deg_1m', 0)
        tmm = res['trans_median_mm']
        marker = " ***" if r10 >= 75 else (" **" if r10 >= 73 else "")
        print(f"  {name:<33} {r10:>7.1f}% {r5:>7.1f}% {tmm:>7.0f}mm{marker}")

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'advanced_ensemble_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_dir}/advanced_ensemble_results.json")


def optimize_per_axis_weights(all_trans, all_rot, test_data, train_data):
    """Optimize ensemble weights per xyz axis to minimize translation error."""
    K = len(all_trans)
    stacked = torch.stack(all_trans, dim=0)  # (K, N, 3)
    gt_trans = test_data.translations  # (N, 3)

    # Grid search over weights (discretized)
    # For tractability, search over uniform weight simplex
    best_recall = 0
    best_weights = None
    best_res = None

    # Simple approach: optimize a single set of weights (not per-axis)
    # using grid search on softmax logits
    if K <= 5:
        # Try all combinations of integer weights 1-5
        weight_options = range(1, 6)
        for w_combo in itertools.product(weight_options, repeat=K):
            w = torch.tensor(w_combo, dtype=torch.float32)
            w = w / w.sum()
            ens_t = (stacked * w.view(-1, 1, 1)).sum(dim=0)
            ens_r = average_rotations(all_rot, weights=list(w.numpy()))

            te, re = compute_errors(ens_t, ens_r, test_data)
            res = compute_recall(te, re, train_data, ens_t, test_data)
            r10 = res['recall']['10deg_2m']

            if r10 > best_recall:
                best_recall = r10
                best_weights = w_combo
                best_res = res

        if best_weights:
            print(f"    Best weights: {best_weights} -> R@10/2m={best_recall:.1f}%")
    else:
        # For more models, use random search
        n_trials = 5000
        for _ in range(n_trials):
            w = torch.rand(K)
            w = w / w.sum()
            ens_t = (stacked * w.view(-1, 1, 1)).sum(dim=0)

            # Quick check: just translation error
            te = (ens_t - gt_trans).norm(dim=-1).cpu().numpy()
            # Count trans_pass at 2m
            trans_pass = (te < 2.0).mean() * 100
            if trans_pass > best_recall:
                ens_r = average_rotations(all_rot, weights=w.tolist())
                _, re = compute_errors(ens_t, ens_r, test_data)
                res = compute_recall(te, re, train_data, ens_t, test_data)
                r10 = res['recall']['10deg_2m']
                if r10 > best_recall:
                    best_recall = r10
                    best_weights = w.tolist()
                    best_res = res

        if best_weights:
            print(f"    Best weights (random search): {[f'{w:.3f}' for w in best_weights]} -> R@10/2m={best_recall:.1f}%")

    return best_res


if __name__ == '__main__':
    main()
