#!/usr/bin/env python3
"""
Snapshot Ensemble Training with Cosine Warm Restarts
=====================================================
Captures diverse model snapshots from different loss basins during a single
training run using SGDR (cosine annealing with warm restarts).

At the end of each cosine cycle (when LR hits minimum), the model has converged
to a local basin — we save a snapshot. The LR restart pushes the model out of
that basin into a new one, yielding diverse snapshots for ensembling.

Ensemble strategies:
  - Mean: average predictions from all snapshots
  - Greedy: forward selection of best snapshot subset
  - Weighted: inverse-error weighted average

Usage:
  python -m feature_retrieval.snapshot_ensemble_train \
      --gpu 3 --seed 314 \
      --T0 500 --T_mult 1 --n_cycles 10 \
      --lr_max 0.001 --lr_min 1e-6 \
      --exp_name exp15a_snapshot_T500x10
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '/root/ICLPose-loc')
from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset,
    quaternion_to_matrix, gram_schmidt_6d_to_matrix, geodesic_distance,
    evaluate, TRANS_LOSS_REGISTRY, precompute_pooled_features,
)


# ============================================================
# Cycle-end detection for CosineAnnealingWarmRestarts
# ============================================================

def compute_cycle_end_epochs(T_0, T_mult, n_cycles):
    """Compute the epoch numbers at which each cycle ends.
    
    Cycle i has length T_0 * T_mult^i epochs.
    Cycle ends are cumulative sums of cycle lengths.
    """
    ends = []
    t = T_0
    cumulative = 0
    for i in range(n_cycles):
        cumulative += t
        ends.append(cumulative)
        t = int(t * T_mult)
    return ends


def compute_total_epochs(T_0, T_mult, n_cycles):
    """Total epochs for n_cycles."""
    return compute_cycle_end_epochs(T_0, T_mult, n_cycles)[-1]


# ============================================================
# Training with snapshot saving
# ============================================================

def train_with_snapshots(args):
    """Train a model with cosine warm restarts, saving snapshots at cycle ends."""
    
    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = out_dir / 'snapshots'
    snapshot_dir.mkdir(exist_ok=True)
    
    # Compute cycle schedule
    cycle_ends = compute_cycle_end_epochs(args.T0, args.T_mult, args.n_cycles)
    total_epochs = cycle_ends[-1]
    print(f"\nCycle schedule: T0={args.T0}, T_mult={args.T_mult}, n_cycles={args.n_cycles}")
    print(f"Cycle end epochs: {cycle_ends}")
    print(f"Total epochs: {total_epochs}")
    
    # Data
    use_fine = True   # both+sum
    use_coarse = True
    use_summary = True
    
    print(f"\nLoading data...")
    train_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'train', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'test', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    
    # Move poses to device
    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)
    
    # Normalize translations
    trans_mean, trans_std = train_data.compute_normalization()
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    print(f"Translation norm: mean={trans_mean.cpu().numpy()}, std={trans_std.cpu().numpy()}")
    torch.save({'mean': trans_mean, 'std': trans_std}, out_dir / 'norm_params.pt')
    
    # Model — same architecture as best model (exp14s_128d_wider_seed314)
    model = PatchPoseRegressor(
        pool_type='spp',
        feat_mode='both+sum',
        patch_dim=args.patch_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        attn_heads=4,
        conv_out_dim=512,
        spp_levels=None,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Pre-compute pooled features (SPP with 128d patches is large)
    print("\nPre-computing pooled features...")
    train_pooled = precompute_pooled_features(model, train_data, device, batch_size=16)
    test_pooled = precompute_pooled_features(model, test_data, device, batch_size=16)
    # Free raw patch features
    train_data.patch_fine = None
    train_data.patch_coarse = None
    test_data.patch_fine = None
    test_data.patch_coarse = None
    if train_data.summary is not None:
        train_data.summary = train_data.summary.to(device)
    if test_data.summary is not None:
        test_data.summary = test_data.summary.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save config
    config = {
        'pool': 'spp', 'feat': 'both+sum', 'patch_dim': args.patch_dim,
        'hidden_dims': args.hidden_dims, 'dropout': args.dropout,
        'feature_dropout': args.feature_dropout,
        'T0': args.T0, 'T_mult': args.T_mult, 'n_cycles': args.n_cycles,
        'lr_max': args.lr_max, 'lr_min': args.lr_min,
        'total_epochs': total_epochs, 'cycle_ends': cycle_ends,
        'seed': args.seed, 'n_params': n_params,
        'total_feature_dim': model.total_dim,
        'weight_decay': args.weight_decay,
        'trans_loss': args.trans_loss,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Optimizer + CosineAnnealingWarmRestarts
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T0, T_mult=args.T_mult, eta_min=args.lr_min)
    
    # Training state
    n_train = train_data.N
    batch_size = args.batch_size
    cycle_end_set = set(cycle_ends)
    snapshot_info = []
    
    history = {
        'epoch': [], 'loss': [], 'lr': [],
        'val_rot_median': [], 'val_trans_median': [],
    }
    best_val_score = float('inf')
    best_epoch = 0
    
    trans_loss_fn = TRANS_LOSS_REGISTRY[args.trans_loss]
    
    t0 = time.time()
    print(f"\nStarting training for {total_epochs} epochs...")
    print(f"{'='*70}")
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        
        # Full batch or mini-batch
        if batch_size >= n_train:
            indices = torch.arange(n_train, device=device)
        else:
            indices = torch.randperm(n_train, device=device)[:batch_size]
        
        target_trans = train_data.translations_norm[indices]
        target_rot = train_data.rotations[indices]
        
        # Forward (pre-computed pooled features)
        pooled_batch = train_pooled[indices]
        if args.feature_dropout > 0:
            pooled_batch = F.dropout(pooled_batch, p=args.feature_dropout, training=True)
        
        trans_pred, rot_pred = model.forward_from_pooled(pooled_batch)
        
        # Loss
        loss_trans = trans_loss_fn(trans_pred, target_trans)
        loss_rot = geodesic_distance(rot_pred, target_rot).mean()
        beta_rot = torch.exp(model.s_rot)
        loss = loss_trans + beta_rot * loss_rot
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if epoch % args.log_every == 0 or epoch == 1 or epoch in cycle_end_set:
            model.eval()
            with torch.no_grad():
                vt_pred, vr_pred = model.forward_from_pooled(test_pooled)
                vt_pred_real = test_data.denormalize(vt_pred, trans_mean, trans_std)
                trans_errors = (vt_pred_real - test_data.translations).norm(dim=-1)
                rot_errors = geodesic_distance(vr_pred, test_data.rotations) * 180 / math.pi
                
                val_trans_med = trans_errors.median().item()
                val_rot_med = rot_errors.median().item()
                val_score = val_trans_med + val_rot_med * 0.1
            
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['lr'].append(current_lr)
            history['val_rot_median'].append(val_rot_med)
            history['val_trans_median'].append(val_trans_med)
            
            if epoch % (args.log_every * 10) == 0 or epoch == 1 or epoch in cycle_end_set:
                elapsed = time.time() - t0
                cycle_marker = " *** CYCLE END ***" if epoch in cycle_end_set else ""
                print(f"[{epoch:5d}/{total_epochs}] loss={loss.item():.4f} "
                      f"lr={current_lr:.2e} beta={beta_rot.item():.2f} "
                      f"| val: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                      f"| {elapsed:.1f}s{cycle_marker}")
            
            # Track global best
            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_rot_median': val_rot_med,
                    'val_trans_median': val_trans_med,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                    'config': config,
                }, out_dir / 'model_best.pt')
        
        # Save snapshot at cycle end
        if epoch in cycle_end_set:
            cycle_idx = cycle_ends.index(epoch)
            snapshot_path = snapshot_dir / f'snapshot_cycle{cycle_idx:02d}_epoch{epoch}.pt'
            
            # Evaluate snapshot
            model.eval()
            with torch.no_grad():
                vt_pred, vr_pred = model.forward_from_pooled(test_pooled)
                vt_pred_real = test_data.denormalize(vt_pred, trans_mean, trans_std)
                t_err = (vt_pred_real - test_data.translations).norm(dim=-1)
                r_err = geodesic_distance(vr_pred, test_data.rotations) * 180 / math.pi
                
                snap_trans_med = t_err.median().item()
                snap_rot_med = r_err.median().item()
                
                # Recall@10deg/2m
                recall_10_2 = ((r_err < 10) & (t_err < 2)).float().mean().item() * 100
            
            torch.save({
                'epoch': epoch,
                'cycle': cycle_idx,
                'model_state_dict': model.state_dict(),
                'val_rot_median': snap_rot_med,
                'val_trans_median': snap_trans_med,
                'recall_10deg_2m': recall_10_2,
                'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                'config': config,
            }, snapshot_path)
            
            info = {
                'cycle': cycle_idx,
                'epoch': epoch,
                'path': str(snapshot_path),
                'rot_median': snap_rot_med,
                'trans_median_mm': snap_trans_med * 1000,
                'recall_10deg_2m': recall_10_2,
            }
            snapshot_info.append(info)
            print(f"  >> Saved snapshot {cycle_idx}: {snap_rot_med:.2f}deg / "
                  f"{snap_trans_med*1000:.0f}mm, R@10/2={recall_10_2:.1f}%")
    
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Best single epoch: {best_epoch} (score={best_val_score:.4f})")
    print(f"Saved {len(snapshot_info)} snapshots")
    
    # Save training history and snapshot info
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(out_dir / 'snapshot_info.json', 'w') as f:
        json.dump(snapshot_info, f, indent=2)
    
    # Plot LR schedule and validation curves
    plot_training(history, cycle_ends, out_dir)
    
    return model, train_data, test_data, trans_mean, trans_std, snapshot_info, test_pooled


def plot_training(history, cycle_ends, out_dir):
    """Plot training curves with cycle boundaries."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    epochs = history['epoch']
    
    # LR schedule
    axes[0].plot(epochs, history['lr'], 'b-', linewidth=0.5)
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_yscale('log')
    axes[0].set_title('Cosine Warm Restarts Schedule')
    
    # Loss
    axes[1].plot(epochs, history['loss'], 'r-', linewidth=0.5)
    axes[1].set_ylabel('Loss')
    
    # Validation
    axes[2].plot(epochs, history['val_rot_median'], 'g-', linewidth=0.5, label='Rot (deg)')
    ax2b = axes[2].twinx()
    ax2b.plot(epochs, [v * 1000 for v in history['val_trans_median']], 
              'b-', linewidth=0.5, label='Trans (mm)')
    axes[2].set_ylabel('Rot median (deg)')
    ax2b.set_ylabel('Trans median (mm)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend(loc='upper left')
    ax2b.legend(loc='upper right')
    
    # Mark cycle ends
    for ax in axes:
        for ce in cycle_ends:
            ax.axvline(x=ce, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    for ce in cycle_ends:
        ax2b.axvline(x=ce, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150)
    plt.close()


# ============================================================
# Ensemble evaluation
# ============================================================

def load_snapshot_predictions(snapshot_info, test_data, trans_mean, trans_std, device,
                              test_pooled, model_template):
    """Load each snapshot and compute predictions.
    
    Returns:
        all_trans_pred: (n_snapshots, N_test, 3) real-space translations
        all_rot_pred:   (n_snapshots, N_test, 3, 3) rotation matrices
    """
    all_trans = []
    all_rot = []
    
    for info in snapshot_info:
        ckpt = torch.load(info['path'], map_location=device)
        model_template.load_state_dict(ckpt['model_state_dict'])
        model_template.eval()
        
        with torch.no_grad():
            t_pred, r_pred = model_template.forward_from_pooled(test_pooled)
            t_pred_real = test_data.denormalize(t_pred, trans_mean, trans_std)
            all_trans.append(t_pred_real)
            all_rot.append(r_pred)
    
    return torch.stack(all_trans), torch.stack(all_rot)


def evaluate_ensemble(all_trans, all_rot, test_data, snapshot_info, out_dir):
    """Evaluate multiple ensemble strategies.
    
    Args:
        all_trans: (K, N, 3) predicted translations
        all_rot:   (K, N, 3, 3) predicted rotation matrices
        test_data: test dataset with ground truth
    """
    out_dir = Path(out_dir)
    K, N, _ = all_trans.shape
    gt_trans = test_data.translations  # (N, 3)
    gt_rot = test_data.rotations       # (N, 3, 3)
    
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
        ('25deg_5m', 25, 5.0),
    ]
    
    results = {}
    
    # ---- 1. Individual snapshot results ----
    print(f"\n{'='*70}")
    print("INDIVIDUAL SNAPSHOT RESULTS")
    print(f"{'='*70}")
    print(f"{'Snap':>5} {'Epoch':>6} {'Rot(deg)':>9} {'Trans(mm)':>10} {'R@10/2':>7}")
    print(f"{'-'*45}")
    
    individual_scores = []  # (score, idx) for greedy selection
    individual_errors = []  # (trans_err, rot_err) per snapshot
    
    for k in range(K):
        t_err = (all_trans[k] - gt_trans).norm(dim=-1)
        r_err = geodesic_distance(all_rot[k], gt_rot) * 180 / math.pi
        
        t_med = t_err.median().item()
        r_med = r_err.median().item()
        recall = ((r_err < 10) & (t_err < 2)).float().mean().item() * 100
        
        individual_errors.append((t_err, r_err))
        individual_scores.append((t_med + r_med * 0.1, k))
        
        print(f"{k:5d} {snapshot_info[k]['epoch']:6d} {r_med:9.2f} {t_med*1000:10.0f} {recall:7.1f}%")
    
    # ---- 2. Mean ensemble ----
    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*70}")
    
    # Translation: simple mean
    mean_trans = all_trans.mean(dim=0)  # (N, 3)
    
    # Rotation: average rotation matrices then project to nearest rotation via SVD
    mean_rot_raw = all_rot.mean(dim=0)  # (N, 3, 3)
    U, S, Vh = torch.linalg.svd(mean_rot_raw)
    mean_rot = torch.bmm(U, Vh)  # nearest rotation matrix
    # Fix determinant (ensure SO(3) not O(3))
    det = torch.det(mean_rot)
    fix = torch.ones_like(Vh)
    fix[:, -1, :] *= det.sign().unsqueeze(-1)
    mean_rot = torch.bmm(U, fix * Vh)
    
    t_err_mean = (mean_trans - gt_trans).norm(dim=-1)
    r_err_mean = geodesic_distance(mean_rot, gt_rot) * 180 / math.pi
    
    print(f"\n  Mean Ensemble ({K} snapshots):")
    print(f"    Rot median: {r_err_mean.median().item():.2f}deg")
    print(f"    Trans median: {t_err_mean.median().item()*1000:.0f}mm")
    for name, rot_th, trans_th in thresholds:
        recall = ((r_err_mean < rot_th) & (t_err_mean < trans_th)).float().mean().item() * 100
        print(f"    R@{name}: {recall:.1f}%")
    
    results['mean_ensemble'] = {
        'n_snapshots': K,
        'rot_median': r_err_mean.median().item(),
        'trans_median_mm': t_err_mean.median().item() * 1000,
        'recall': {name: float(((r_err_mean < rt) & (t_err_mean < tt)).float().mean().item() * 100)
                   for name, rt, tt in thresholds},
    }
    
    # ---- 3. Weighted ensemble (inverse median error) ----
    weights = []
    for k in range(K):
        score = individual_scores[k][0]
        weights.append(1.0 / (score + 1e-8))
    weights = torch.tensor(weights, device=all_trans.device, dtype=all_trans.dtype)
    weights = weights / weights.sum()
    
    w_trans = (all_trans * weights.view(K, 1, 1)).sum(dim=0)
    w_rot_raw = (all_rot * weights.view(K, 1, 1, 1)).sum(dim=0)
    U, S, Vh = torch.linalg.svd(w_rot_raw)
    w_rot = torch.bmm(U, Vh)
    det = torch.det(w_rot)
    fix = torch.ones_like(Vh)
    fix[:, -1, :] *= det.sign().unsqueeze(-1)
    w_rot = torch.bmm(U, fix * Vh)
    
    t_err_w = (w_trans - gt_trans).norm(dim=-1)
    r_err_w = geodesic_distance(w_rot, gt_rot) * 180 / math.pi
    
    print(f"\n  Weighted Ensemble ({K} snapshots, inverse-error weights):")
    print(f"    Weights: {[f'{w:.3f}' for w in weights.cpu().tolist()]}")
    print(f"    Rot median: {r_err_w.median().item():.2f}deg")
    print(f"    Trans median: {t_err_w.median().item()*1000:.0f}mm")
    for name, rot_th, trans_th in thresholds:
        recall = ((r_err_w < rot_th) & (t_err_w < trans_th)).float().mean().item() * 100
        print(f"    R@{name}: {recall:.1f}%")
    
    results['weighted_ensemble'] = {
        'n_snapshots': K,
        'weights': weights.cpu().tolist(),
        'rot_median': r_err_w.median().item(),
        'trans_median_mm': t_err_w.median().item() * 1000,
        'recall': {name: float(((r_err_w < rt) & (t_err_w < tt)).float().mean().item() * 100)
                   for name, rt, tt in thresholds},
    }
    
    # ---- 4. Greedy forward selection ----
    print(f"\n  Greedy Forward Selection:")
    
    # Sort by individual performance
    individual_scores.sort()
    best_subset = [individual_scores[0][1]]  # start with best individual
    best_recall = -1
    
    remaining = set(range(K)) - {best_subset[0]}
    
    for step in range(K - 1):
        best_candidate = None
        best_candidate_recall = -1
        
        for c in remaining:
            subset = best_subset + [c]
            # Mean ensemble of subset
            sub_trans = all_trans[subset].mean(dim=0)
            sub_rot_raw = all_rot[subset].mean(dim=0)
            U, S, Vh = torch.linalg.svd(sub_rot_raw)
            sub_rot = torch.bmm(U, Vh)
            det = torch.det(sub_rot)
            fix_m = torch.ones_like(Vh)
            fix_m[:, -1, :] *= det.sign().unsqueeze(-1)
            sub_rot = torch.bmm(U, fix_m * Vh)
            
            t_e = (sub_trans - gt_trans).norm(dim=-1)
            r_e = geodesic_distance(sub_rot, gt_rot) * 180 / math.pi
            recall_10_2 = ((r_e < 10) & (t_e < 2)).float().mean().item() * 100
            
            if recall_10_2 > best_candidate_recall:
                best_candidate_recall = recall_10_2
                best_candidate = c
        
        if best_candidate_recall > best_recall:
            best_recall = best_candidate_recall
            best_subset.append(best_candidate)
            remaining.remove(best_candidate)
            print(f"    +snap {best_candidate} (epoch {snapshot_info[best_candidate]['epoch']}) "
                  f"-> {len(best_subset)} snaps, R@10/2={best_recall:.1f}%")
        else:
            # No improvement — stop
            print(f"    (no improvement, stopping at {len(best_subset)} snapshots)")
            break
    
    # Evaluate best greedy subset
    sub_trans = all_trans[best_subset].mean(dim=0)
    sub_rot_raw = all_rot[best_subset].mean(dim=0)
    U, S, Vh = torch.linalg.svd(sub_rot_raw)
    sub_rot = torch.bmm(U, Vh)
    det = torch.det(sub_rot)
    fix_m = torch.ones_like(Vh)
    fix_m[:, -1, :] *= det.sign().unsqueeze(-1)
    sub_rot = torch.bmm(U, fix_m * Vh)
    
    t_err_g = (sub_trans - gt_trans).norm(dim=-1)
    r_err_g = geodesic_distance(sub_rot, gt_rot) * 180 / math.pi
    
    print(f"\n    Best greedy subset: {best_subset} ({len(best_subset)} snapshots)")
    print(f"    Rot median: {r_err_g.median().item():.2f}deg")
    print(f"    Trans median: {t_err_g.median().item()*1000:.0f}mm")
    for name, rot_th, trans_th in thresholds:
        recall = ((r_err_g < rot_th) & (t_err_g < trans_th)).float().mean().item() * 100
        print(f"    R@{name}: {recall:.1f}%")
    
    results['greedy_ensemble'] = {
        'n_snapshots': len(best_subset),
        'snapshot_indices': best_subset,
        'rot_median': r_err_g.median().item(),
        'trans_median_mm': t_err_g.median().item() * 1000,
        'recall': {name: float(((r_err_g < rt) & (t_err_g < tt)).float().mean().item() * 100)
                   for name, rt, tt in thresholds},
    }
    
    # ---- Save results ----
    with open(out_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {out_dir / 'ensemble_results.json'}")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Snapshot Ensemble with Cosine Warm Restarts')
    
    # Data / output
    parser.add_argument('--feature_dir', type=str,
                        default='output/feature_extract/features_radio_dual_128/OldHospital_pilot')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_base', type=str,
                        default='output/feature_retrieval/pose_regression')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=314)
    
    # Architecture (matching best model)
    parser.add_argument('--patch_dim', type=int, default=128)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[2048, 1024, 512])
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--feature_dropout', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=895)
    
    # Cosine warm restarts
    parser.add_argument('--T0', type=int, default=500,
                        help='Base cycle length (epochs)')
    parser.add_argument('--T_mult', type=int, default=1,
                        help='Cycle length multiplier (1=fixed, 2=doubling)')
    parser.add_argument('--n_cycles', type=int, default=10,
                        help='Number of cycles (= number of snapshots)')
    parser.add_argument('--lr_max', type=float, default=0.001,
                        help='Max learning rate (at restart)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Min learning rate (at cycle end)')
    
    # Training
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--trans_loss', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'log_cosh', 'wing', 'mse'])
    parser.add_argument('--log_every', type=int, default=10)
    
    # Eval only mode
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, only evaluate existing snapshots')
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_base, args.exp_name)
    
    print(f"{'='*70}")
    print(f"Snapshot Ensemble: {args.exp_name}")
    print(f"Schedule: T0={args.T0}, T_mult={args.T_mult}, {args.n_cycles} cycles")
    print(f"LR: {args.lr_max} -> {args.lr_min}")
    total = compute_total_epochs(args.T0, args.T_mult, args.n_cycles)
    print(f"Total epochs: {total}")
    print(f"{'='*70}")
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    if not args.eval_only:
        model, train_data, test_data, trans_mean, trans_std, snapshot_info, test_pooled = \
            train_with_snapshots(args)
    else:
        # Load existing snapshot info
        out_dir = Path(args.output_dir)
        with open(out_dir / 'snapshot_info.json') as f:
            snapshot_info = json.load(f)
        with open(out_dir / 'config.json') as f:
            config = json.load(f)
        
        print(f"Loaded {len(snapshot_info)} snapshots from {out_dir}")
        
        # Load data
        use_fine = True
        use_coarse = True
        use_summary = True
        
        train_data = PatchPoseDataset(
            args.feature_dir, args.dataset_dir, 'train', 'cpu',
            use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
        test_data = PatchPoseDataset(
            args.feature_dir, args.dataset_dir, 'test', 'cpu',
            use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
        
        train_data.translations = train_data.translations.to(device)
        train_data.rotations = train_data.rotations.to(device)
        test_data.translations = test_data.translations.to(device)
        test_data.rotations = test_data.rotations.to(device)
        
        norm = torch.load(out_dir / 'norm_params.pt', map_location=device)
        trans_mean, trans_std = norm['mean'], norm['std']
        train_data.normalize_translations(trans_mean, trans_std)
        test_data.normalize_translations(trans_mean, trans_std)
        
        model = PatchPoseRegressor(
            pool_type='spp', feat_mode='both+sum',
            patch_dim=args.patch_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        ).to(device)
        
        # Pre-compute pooled features
        test_pooled = precompute_pooled_features(model, test_data, device, batch_size=16)
        test_data.patch_fine = None
        test_data.patch_coarse = None
        if test_data.summary is not None:
            test_data.summary = test_data.summary.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Evaluate ensemble
    print(f"\nLoading {len(snapshot_info)} snapshots for ensemble evaluation...")
    all_trans, all_rot = load_snapshot_predictions(
        snapshot_info, test_data, trans_mean, trans_std, device, test_pooled, model)
    
    results = evaluate_ensemble(
        all_trans, all_rot, test_data, snapshot_info, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
