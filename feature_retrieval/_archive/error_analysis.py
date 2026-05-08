#!/usr/bin/env python3
"""Error analysis for the best pose regression model on OldHospital."""

import sys, os, math, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset, 
    geodesic_distance, quaternion_to_matrix
)

def main():
    device = torch.device('cpu')
    
    # Paths
    exp_dir = Path('output/feature_retrieval/pose_regression/exp14s_128d_wider_seed314')
    feature_dir = 'output/feature_extract/features_radio_dual_128/OldHospital_pilot'
    dataset_dir = 'dataset/OldHospital'
    
    # Load model
    print("Loading model...")
    model = PatchPoseRegressor(
        pool_type='spp', feat_mode='both+sum',
        patch_dim=128, hidden_dims=(2048, 1024, 512),
        dropout=0.15
    )
    ckpt = torch.load(exp_dir / 'model_best.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    trans_mean = ckpt['norm_params']['mean']
    trans_std = ckpt['norm_params']['std']
    
    # Load datasets
    print("Loading data...")
    train_data = PatchPoseDataset(feature_dir, dataset_dir, 'train', 'cpu',
                                   use_fine=True, use_coarse=True, use_summary=True)
    test_data = PatchPoseDataset(feature_dir, dataset_dir, 'test', 'cpu',
                                  use_fine=True, use_coarse=True, use_summary=True)
    
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        trans_pred_norm, rot_pred = model(
            test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        trans_pred = trans_pred_norm * trans_std + trans_mean
    
    # Compute errors
    trans_errors_m = (trans_pred - test_data.translations).norm(dim=-1).numpy()
    trans_errors_mm = trans_errors_m * 1000
    rot_errors_deg = (geodesic_distance(rot_pred, test_data.rotations) * 180 / math.pi).numpy()
    
    # GT positions
    test_pos = test_data.translations.numpy()  # (182, 3)
    train_pos = train_data.translations.numpy()  # (895, 3)
    
    # KDTree for nearest training image
    tree = cKDTree(train_pos)
    nn_dists, nn_idx = tree.query(test_pos, k=1)
    
    # Count training images within various radii
    counts_2m = np.array([len(tree.query_ball_point(p, 2.0)) for p in test_pos])
    counts_5m = np.array([len(tree.query_ball_point(p, 5.0)) for p in test_pos])
    counts_10m = np.array([len(tree.query_ball_point(p, 10.0)) for p in test_pos])
    
    # Failure mask at 10deg/2m
    fail_mask = ~((rot_errors_deg < 10) & (trans_errors_m < 2.0))
    n_fail = fail_mask.sum()
    n_total = len(fail_mask)
    
    # Save CSV
    csv_path = exp_dir / 'per_sample_errors.csv'
    with open(csv_path, 'w') as f:
        f.write('image,trans_error_mm,rot_error_deg,nn_train_dist_m,train_within_2m,train_within_5m,train_within_10m,gt_x,gt_y,gt_z,fail_10deg2m\n')
        for i in range(n_total):
            f.write(f'{test_data.names[i]},{trans_errors_mm[i]:.1f},{rot_errors_deg[i]:.3f},'
                    f'{nn_dists[i]:.3f},{counts_2m[i]},{counts_5m[i]},{counts_10m[i]},'
                    f'{test_pos[i,0]:.3f},{test_pos[i,1]:.3f},{test_pos[i,2]:.3f},'
                    f'{int(fail_mask[i])}\n')
    print(f"Saved per-sample CSV to {csv_path}")
    
    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: R@10deg/2m failures")
    print("=" * 70)
    
    print(f"\nTotal test images: {n_total}")
    print(f"Failures at R@10deg/2m: {n_fail} ({n_fail/n_total*100:.1f}%)")
    print(f"  - Translation > 2m: {(trans_errors_m >= 2.0).sum()}")
    print(f"  - Rotation > 10deg: {(rot_errors_deg >= 10).sum()}")
    print(f"  - Both: {((trans_errors_m >= 2.0) & (rot_errors_deg >= 10)).sum()}")
    
    # Error stats
    print(f"\nError statistics (all):")
    print(f"  Trans: median={np.median(trans_errors_mm):.0f}mm, mean={np.mean(trans_errors_mm):.0f}mm, "
          f"max={np.max(trans_errors_mm):.0f}mm")
    print(f"  Rot:   median={np.median(rot_errors_deg):.2f}deg, mean={np.mean(rot_errors_deg):.2f}deg, "
          f"max={np.max(rot_errors_deg):.2f}deg")
    
    print(f"\nError statistics (failures only):")
    if n_fail > 0:
        print(f"  Trans: median={np.median(trans_errors_mm[fail_mask]):.0f}mm, "
              f"mean={np.mean(trans_errors_mm[fail_mask]):.0f}mm")
        print(f"  Rot:   median={np.median(rot_errors_deg[fail_mask]):.2f}deg, "
              f"mean={np.mean(rot_errors_deg[fail_mask]):.2f}deg")
    
    # Sparsity analysis
    print(f"\n--- Sparsity Analysis ---")
    for radius, counts in [(2, counts_2m), (5, counts_5m), (10, counts_10m)]:
        sparse = counts == 0
        fail_and_sparse = fail_mask & sparse
        if sparse.sum() > 0:
            fail_rate_sparse = fail_and_sparse.sum() / sparse.sum() * 100
        else:
            fail_rate_sparse = 0
        pct_failures_from_sparse = fail_and_sparse.sum() / max(n_fail, 1) * 100
        print(f"  No training image within {radius}m: {sparse.sum()} test images, "
              f"{pct_failures_from_sparse:.0f}% of failures come from these, "
              f"fail rate={fail_rate_sparse:.0f}%")
    
    print(f"\n  NN distance stats (all): median={np.median(nn_dists):.2f}m, "
          f"mean={np.mean(nn_dists):.2f}m, max={np.max(nn_dists):.2f}m")
    print(f"  NN distance stats (fail): median={np.median(nn_dists[fail_mask]):.2f}m, "
          f"mean={np.mean(nn_dists[fail_mask]):.2f}m")
    print(f"  NN distance stats (pass): median={np.median(nn_dists[~fail_mask]):.2f}m, "
          f"mean={np.mean(nn_dists[~fail_mask]):.2f}m")
    
    # Correlation
    print(f"\n--- Correlation: NN distance vs errors ---")
    r_trans, p_trans = pearsonr(nn_dists, trans_errors_m)
    rho_trans, _ = spearmanr(nn_dists, trans_errors_m)
    r_rot, p_rot = pearsonr(nn_dists, rot_errors_deg)
    print(f"  NN dist vs trans error: Pearson r={r_trans:.3f} (p={p_trans:.2e}), Spearman rho={rho_trans:.3f}")
    print(f"  NN dist vs rot error:   Pearson r={r_rot:.3f} (p={p_rot:.2e})")
    
    # Density correlation
    r_density, _ = pearsonr(counts_5m, trans_errors_m)
    print(f"  Train density (5m) vs trans error: Pearson r={r_density:.3f}")
    
    # Spatial clustering of failures
    print(f"\n--- Spatial Clustering of Failures ---")
    fail_positions = test_pos[fail_mask]
    pass_positions = test_pos[~fail_mask]
    
    if n_fail > 1:
        # Pairwise distances among failures
        fail_tree = cKDTree(fail_positions)
        fail_nn_dists, _ = fail_tree.query(fail_positions, k=2)  # k=2 to skip self
        avg_fail_nn = fail_nn_dists[:, 1].mean()
        
        # Compare to random subset of same size
        all_nn_dists, _ = cKDTree(test_pos).query(test_pos, k=2)
        avg_all_nn = all_nn_dists[:, 1].mean()
        
        print(f"  Avg NN distance among failures: {avg_fail_nn:.2f}m")
        print(f"  Avg NN distance among all test:  {avg_all_nn:.2f}m")
        print(f"  Ratio (< 1 means clustered): {avg_fail_nn/avg_all_nn:.2f}")
        
        # Spatial extent
        fail_range = fail_positions.max(axis=0) - fail_positions.min(axis=0)
        all_range = test_pos.max(axis=0) - test_pos.min(axis=0)
        print(f"  Failure spatial range: X={fail_range[0]:.1f}m, Y={fail_range[1]:.1f}m, Z={fail_range[2]:.1f}m")
        print(f"  All test spatial range: X={all_range[0]:.1f}m, Y={all_range[1]:.1f}m, Z={all_range[2]:.1f}m")
        
        # Identify clusters using simple grid binning
        print(f"\n  Failure positions (XZ plane, grouped by 5m grid):")
        grid_size = 5.0
        fail_xz = fail_positions[:, [0, 2]]
        grid_keys = {}
        for i, (x, z) in enumerate(fail_xz):
            key = (int(x // grid_size) * int(grid_size), int(z // grid_size) * int(grid_size))
            if key not in grid_keys:
                grid_keys[key] = []
            grid_keys[key].append(i)
        
        for key in sorted(grid_keys.keys(), key=lambda k: -len(grid_keys[k])):
            indices = grid_keys[key]
            avg_trans = np.mean(trans_errors_mm[fail_mask][indices])
            avg_rot = np.mean(rot_errors_deg[fail_mask][indices])
            avg_nn = np.mean(nn_dists[fail_mask][indices])
            print(f"    Grid ({key[0]:+3d},{key[1]:+3d}): {len(indices)} failures, "
                  f"avg trans={avg_trans:.0f}mm, avg rot={avg_rot:.1f}deg, avg NN dist={avg_nn:.2f}m")
    
    # Top 10 worst predictions
    print(f"\n--- Top 10 Worst Translations ---")
    worst_idx = np.argsort(trans_errors_mm)[::-1][:10]
    for rank, i in enumerate(worst_idx):
        print(f"  {rank+1}. {test_data.names[i]:30s} trans={trans_errors_mm[i]:.0f}mm "
              f"rot={rot_errors_deg[i]:.1f}deg NN_dist={nn_dists[i]:.2f}m train_2m={counts_2m[i]}")
    
    print(f"\n--- Top 10 Worst Rotations ---")
    worst_rot_idx = np.argsort(rot_errors_deg)[::-1][:10]
    for rank, i in enumerate(worst_rot_idx):
        print(f"  {rank+1}. {test_data.names[i]:30s} rot={rot_errors_deg[i]:.1f}deg "
              f"trans={trans_errors_mm[i]:.0f}mm NN_dist={nn_dists[i]:.2f}m")
    
    # Percentile analysis
    print(f"\n--- Translation Error Percentiles ---")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(trans_errors_mm, p):.0f}mm")
    
    print(f"\n--- NN Distance Bins vs Failure Rate ---")
    bins = [0, 0.5, 1, 2, 3, 5, 10, 50]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (nn_dists >= lo) & (nn_dists < hi)
        if mask.sum() > 0:
            rate = fail_mask[mask].sum() / mask.sum() * 100
            avg_t = np.mean(trans_errors_mm[mask])
            print(f"  NN dist [{lo:.1f}, {hi:.1f})m: {mask.sum()} images, "
                  f"fail rate={rate:.0f}%, avg trans err={avg_t:.0f}mm")

if __name__ == '__main__':
    main()
