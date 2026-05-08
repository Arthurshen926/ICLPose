#!/usr/bin/env python3
"""
Phase 1: Direct Pose Regression from RADIO Summary Tokens
==========================================================
Single-file implementation: model, dataset, training, evaluation, visualization.

Usage:
    python feature_retrieval/pose_regressor.py --mode train_eval
    python feature_retrieval/pose_regressor.py --mode eval --checkpoint path/to/model_best.pt
"""

import argparse
import glob
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


# ============================================================
# Utility: Rotation representations
# ============================================================

def quaternion_to_matrix(q):
    """Convert w-first quaternion [w,x,y,z] to 3x3 rotation matrix.
    Args: q: (*, 4) tensor
    Returns: (*, 3, 3) tensor
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    mat = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*B, 3, 3)
    return mat


def gram_schmidt_6d_to_matrix(v):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Zhou et al., CVPR 2019: On the Continuity of Rotation Representations.
    Args: v: (*, 6) tensor
    Returns: (*, 3, 3) tensor
    """
    a1 = v[..., :3]
    a2 = v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)  # (*, 3, 3), rows are basis vectors


def geodesic_distance(R1, R2):
    """Geodesic distance between rotation matrices.
    Args: R1, R2: (*, 3, 3) tensors
    Returns: (*,) tensor of angles in radians
    """
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).reshape(R1.shape[:-2])


# ============================================================
# Dataset
# ============================================================

class PoseRegressionDataset:
    """In-memory dataset: RADIO summary tokens + GT poses."""

    def __init__(self, summary_matrix_path, dataset_dir, split='train', device='cpu'):
        self.device = device

        # Load summary matrix
        summary_matrix = torch.load(summary_matrix_path, map_location='cpu')
        assert summary_matrix.shape[1] == 2560, f"Expected 2560-d, got {summary_matrix.shape[1]}"

        # Build index mapping: image_name -> summary_matrix row index
        image_paths = sorted(glob.glob(os.path.join(dataset_dir, 'seq*/*.png')))
        name_to_idx = {os.path.relpath(p, dataset_dir): i for i, p in enumerate(image_paths)}

        # Parse split file
        split_file = os.path.join(dataset_dir, f'dataset_{split}.txt')
        with open(split_file) as f:
            lines = f.readlines()

        features_list = []
        translations_list = []
        rotations_list = []
        names_list = []

        for line in lines[3:]:  # Skip 3 header lines
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            img_name = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            idx = name_to_idx.get(img_name)
            if idx is None:
                print(f"WARNING: {img_name} not found in summary_matrix index")
                continue

            features_list.append(summary_matrix[idx])
            translations_list.append(torch.tensor([x, y, z], dtype=torch.float32))
            rotations_list.append(torch.tensor([w, p, q, r], dtype=torch.float32))
            names_list.append(img_name)

        self.features = torch.stack(features_list).to(device)       # (N, 2560)
        self.translations = torch.stack(translations_list).to(device) # (N, 3)
        quat = torch.stack(rotations_list)                           # (N, 4) w-first
        self.rotations = quaternion_to_matrix(quat).to(device)       # (N, 3, 3)
        self.names = names_list
        self.N = len(names_list)

        print(f"[{split}] Loaded {self.N} samples, features {self.features.shape}")

    def compute_normalization(self):
        """Compute translation mean/std for standardization."""
        self.trans_mean = self.translations.mean(dim=0)
        self.trans_std = self.translations.std(dim=0).clamp(min=1e-6)
        return self.trans_mean, self.trans_std

    def normalize_translations(self, mean, std):
        """Apply z-score normalization to translations."""
        self.translations_norm = (self.translations - mean) / std

    def denormalize(self, trans_norm, mean, std):
        """Convert normalized predictions back to original scale."""
        return trans_norm * std + mean


# ============================================================
# Model
# ============================================================

class PoseRegressorMLP(nn.Module):
    """MLP: RADIO summary token (2560) -> pose (translation + rotation)."""

    def __init__(self, input_dim=2560, hidden_dims=(1024, 512, 256), dropout=0.1):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.trans_head = nn.Linear(hidden_dims[-1], 3)
        self.rot_head = nn.Linear(hidden_dims[-1], 6)

        # Learnable loss weight (log scale): L = L_trans + exp(s_rot) * L_rot
        self.s_rot = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args: x: (B, 2560) summary tokens
        Returns:
            trans: (B, 3) predicted translation (normalized scale)
            rot_mat: (B, 3, 3) predicted rotation matrix
        """
        h = self.backbone(x)
        trans = self.trans_head(h)
        rot_6d = self.rot_head(h)
        rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
        return trans, rot_mat


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'train', device)
    test_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'test', device)

    # Normalize translations
    trans_mean, trans_std = train_data.compute_normalization()
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    print(f"Translation normalization: mean={trans_mean.cpu().numpy()}, std={trans_std.cpu().numpy()}")

    # Save normalization params
    torch.save({'mean': trans_mean, 'std': trans_std}, out_dir / 'norm_params.pt')

    # Model
    model = PoseRegressorMLP(
        input_dim=2560,
        hidden_dims=(1024, 512, 256),
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    history = {
        'epoch': [], 'loss': [], 'loss_trans': [], 'loss_rot': [],
        'val_rot_median': [], 'val_trans_median': [], 'beta_rot': [],
    }
    best_val_score = float('inf')
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()

        # Full-batch forward
        feat = train_data.features
        if args.feature_dropout > 0:
            feat = F.dropout(feat, p=args.feature_dropout, training=True)

        trans_pred, rot_pred = model(feat)

        # Translation loss (SmoothL1 on normalized coords)
        loss_trans = F.smooth_l1_loss(trans_pred, train_data.translations_norm)

        # Rotation loss (geodesic distance)
        loss_rot = geodesic_distance(rot_pred, train_data.rotations).mean()

        # Combined loss with learnable weighting
        beta_rot = torch.exp(model.s_rot)
        loss = loss_trans + beta_rot * loss_rot

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vt_pred, vr_pred = model(test_data.features)
                vt_pred_real = test_data.denormalize(vt_pred, trans_mean, trans_std)
                trans_errors = (vt_pred_real - test_data.translations).norm(dim=-1)
                rot_errors = geodesic_distance(vr_pred, test_data.rotations) * 180 / math.pi

                val_trans_med = trans_errors.median().item()
                val_rot_med = rot_errors.median().item()
                val_score = val_trans_med + val_rot_med * 0.1

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_trans'].append(loss_trans.item())
            history['loss_rot'].append(loss_rot.item())
            history['val_rot_median'].append(val_rot_med)
            history['val_trans_median'].append(val_trans_med)
            history['beta_rot'].append(beta_rot.item())

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                      f"(t={loss_trans.item():.4f} r={loss_rot.item():.4f} beta={beta_rot.item():.2f}) "
                      f"| val: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                      f"| {elapsed:.1f}s")

            # Save best model
            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rot_median': val_rot_med,
                    'val_trans_median': val_trans_med,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                }, out_dir / 'model_best.pt')

    print(f"\nTraining done. Best epoch: {best_epoch}, score: {best_val_score:.4f}")

    # Save training history
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
    }, out_dir / 'model_final.pt')

    return model, train_data, test_data, trans_mean, trans_std, history


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, test_data, trans_mean, trans_std, out_dir, device):
    """Full evaluation with metrics and per-sample errors."""
    out_dir = Path(out_dir)
    model.eval()

    with torch.no_grad():
        trans_pred, rot_pred = model(test_data.features)
        trans_pred_real = test_data.denormalize(trans_pred, trans_mean, trans_std)

    # Per-sample errors
    trans_errors = (trans_pred_real - test_data.translations).norm(dim=-1).cpu().numpy()
    rot_errors = (geodesic_distance(rot_pred, test_data.rotations) * 180 / math.pi).cpu().numpy()

    # Metrics
    results = {
        'rotation_deg': {
            'median': float(np.median(rot_errors)),
            'mean': float(np.mean(rot_errors)),
            'std': float(np.std(rot_errors)),
            'max': float(np.max(rot_errors)),
            'min': float(np.min(rot_errors)),
        },
        'translation_m': {
            'median': float(np.median(trans_errors)),
            'mean': float(np.mean(trans_errors)),
            'std': float(np.std(trans_errors)),
            'max': float(np.max(trans_errors)),
            'min': float(np.min(trans_errors)),
        },
        'translation_mm': {
            'median': float(np.median(trans_errors) * 1000),
            'mean': float(np.mean(trans_errors) * 1000),
        },
        'recall': {},
    }

    # Recall thresholds
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
        ('25deg_5m', 25, 5.0),
    ]
    for name, rot_th, trans_th in thresholds:
        mask = (rot_errors < rot_th) & (trans_errors < trans_th)
        results['recall'][name] = float(mask.mean()) * 100

    # CLS cosine baseline comparison
    results['baseline_cls_cosine'] = {
        'rotation_median_deg': 11.5,
        'translation_median_mm': 1360,
    }
    results['improvement'] = {
        'rotation_deg': 11.5 - results['rotation_deg']['median'],
        'translation_mm': 1360 - results['translation_mm']['median'],
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Rotation  (median): {results['rotation_deg']['median']:.2f}deg "
          f"(mean: {results['rotation_deg']['mean']:.2f}deg)")
    print(f"Translation (median): {results['translation_mm']['median']:.0f}mm "
          f"(mean: {results['translation_mm']['mean']:.0f}mm)")
    print()
    for name, rot_th, trans_th in thresholds:
        print(f"Recall@{name}: {results['recall'][name]:.1f}%")
    print()
    print(f"vs CLS cosine baseline (11.5deg / 1360mm):")
    print(f"  Rotation improvement:    {results['improvement']['rotation_deg']:+.2f}deg")
    print(f"  Translation improvement: {results['improvement']['translation_mm']:+.0f}mm")
    print("=" * 60)

    # Save results
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save per-sample errors for analysis
    per_sample = []
    for i in range(test_data.N):
        per_sample.append({
            'name': test_data.names[i],
            'rot_error_deg': float(rot_errors[i]),
            'trans_error_m': float(trans_errors[i]),
            'gt_trans': test_data.translations[i].cpu().tolist(),
            'pred_trans': trans_pred_real[i].cpu().tolist(),
        })
    per_sample.sort(key=lambda x: x['trans_error_m'], reverse=True)
    with open(out_dir / 'per_sample_errors.json', 'w') as f:
        json.dump(per_sample, f, indent=2)

    return results, trans_errors, rot_errors, trans_pred_real.cpu().numpy()


# ============================================================
# Visualization
# ============================================================

def visualize(history, results, trans_errors, rot_errors, trans_pred, test_data, train_data, out_dir):
    """Generate all visualizations."""
    out_dir = Path(out_dir)

    # 1. Training loss curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(history['epoch'], history['loss'], 'b-', alpha=0.8, label='Total')
    ax.plot(history['epoch'], history['loss_trans'], 'g-', alpha=0.8, label='Translation')
    ax.plot(history['epoch'], history['loss_rot'], 'r-', alpha=0.8, label='Rotation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history['epoch'], history['val_rot_median'], 'r-', label='Rotation (deg)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Median Rotation Error (deg)')
    ax.set_title('Validation Rotation Error')
    ax.axhline(y=11.5, color='gray', linestyle='--', label='CLS baseline (11.5deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    val_trans_mm = [x * 1000 for x in history['val_trans_median']]
    ax.plot(history['epoch'], val_trans_mm, 'g-', label='Translation (mm)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Median Translation Error (mm)')
    ax.set_title('Validation Translation Error')
    ax.axhline(y=1360, color='gray', linestyle='--', label='CLS baseline (1360mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'loss_curves.png'}")

    # 2. Error distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(rot_errors, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.median(rot_errors), color='red', linestyle='--',
               label=f'Median: {np.median(rot_errors):.2f}deg')
    ax.axvline(x=11.5, color='gray', linestyle='--', label='CLS baseline: 11.5deg')
    ax.set_xlabel('Rotation Error (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Rotation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(trans_errors * 1000, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.median(trans_errors) * 1000, color='blue', linestyle='--',
               label=f'Median: {np.median(trans_errors)*1000:.0f}mm')
    ax.axvline(x=1360, color='gray', linestyle='--', label='CLS baseline: 1360mm')
    ax.set_xlabel('Translation Error (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Translation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'error_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'error_histogram.png'}")

    # 3. BEV scatter plot (X-Z plane, Y is up)
    gt_trans = test_data.translations.cpu().numpy()
    train_trans = train_data.translations.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot train positions in background
    ax.scatter(train_trans[:, 0], train_trans[:, 2], c='lightgray', s=10, alpha=0.4,
               label=f'Train ({len(train_trans)})', zorder=1)

    # Plot GT test positions
    ax.scatter(gt_trans[:, 0], gt_trans[:, 2], c='black', s=30, alpha=0.6,
               label='GT test', zorder=3)

    # Plot predicted positions, colored by error
    sc = ax.scatter(trans_pred[:, 0], trans_pred[:, 2],
                    c=trans_errors * 1000, cmap='RdYlGn_r', s=30, alpha=0.8,
                    vmin=0, vmax=np.percentile(trans_errors * 1000, 95),
                    edgecolors='gray', linewidths=0.5, zorder=4,
                    label='Predicted test')
    plt.colorbar(sc, ax=ax, label='Translation Error (mm)', shrink=0.8)

    # Draw error lines
    for i in range(len(gt_trans)):
        ax.plot([gt_trans[i, 0], trans_pred[i, 0]],
                [gt_trans[i, 2], trans_pred[i, 2]],
                'r-', alpha=0.15, linewidth=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'BEV: GT vs Predicted (median err: {np.median(trans_errors)*1000:.0f}mm)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'bev_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'bev_scatter.png'}")

    # 4. Beta (loss weight) evolution
    if 'beta_rot' in history and len(history['beta_rot']) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history['epoch'], history['beta_rot'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('beta_rot = exp(s_rot)')
        ax.set_title('Learnable Rotation Loss Weight')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'beta_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_dir / 'beta_evolution.png'}")

    # 5. Worst-case analysis (top 20 failures)
    worst_idx = np.argsort(trans_errors)[-20:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    names = [test_data.names[i].split('/')[-1][:12] for i in worst_idx]
    ax.barh(range(len(worst_idx)), trans_errors[worst_idx] * 1000, color='steelblue')
    ax.set_yticks(range(len(worst_idx)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Translation Error (mm)')
    ax.set_title('Top 20 Worst Translations')
    ax.invert_yaxis()

    worst_rot_idx = np.argsort(rot_errors)[-20:][::-1]
    ax = axes[1]
    names_r = [test_data.names[i].split('/')[-1][:12] for i in worst_rot_idx]
    ax.barh(range(len(worst_rot_idx)), rot_errors[worst_rot_idx], color='coral')
    ax.set_yticks(range(len(worst_rot_idx)))
    ax.set_yticklabels(names_r, fontsize=7)
    ax.set_xlabel('Rotation Error (deg)')
    ax.set_title('Top 20 Worst Rotations')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_dir / 'failure_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'failure_analysis.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Pose Regression from RADIO Summary Tokens')
    parser.add_argument('--mode', type=str, default='train_eval',
                        choices=['train_eval', 'eval'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--summary_matrix', type=str,
                        default='output/feature_extract/features_radio_dual/OldHospital_pilot/summary_matrix.pt')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_retrieval/pose_regression/exp01_direct_mlp')
    parser.add_argument('--checkpoint', type=str, default=None)

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
    parser.add_argument('--log_every', type=int, default=10)

    args = parser.parse_args()

    if args.mode == 'train_eval':
        model, train_data, test_data, trans_mean, trans_std, history = train(args)
        # Load best model for evaluation
        ckpt = torch.load(Path(args.output_dir) / 'model_best.pt',
                          map_location=f'cuda:{args.gpu}')
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nLoaded best model from epoch {ckpt['epoch']} "
              f"(val: {ckpt['val_rot_median']:.2f}deg / {ckpt['val_trans_median']*1000:.0f}mm)")

        results, trans_errors, rot_errors, trans_pred = evaluate(
            model, test_data, trans_mean, trans_std, args.output_dir,
            torch.device(f'cuda:{args.gpu}'))
        visualize(history, results, trans_errors, rot_errors, trans_pred, test_data,
                  train_data, args.output_dir)

    elif args.mode == 'eval':
        assert args.checkpoint, "Must provide --checkpoint for eval mode"
        device = torch.device(f'cuda:{args.gpu}')
        ckpt = torch.load(args.checkpoint, map_location=device)
        trans_mean = ckpt['norm_params']['mean'].to(device)
        trans_std = ckpt['norm_params']['std'].to(device)

        model = PoseRegressorMLP().to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        test_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'test', device)
        test_data.normalize_translations(trans_mean, trans_std)

        # Need train data for visualization
        train_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'train', device)

        results, trans_errors, rot_errors, trans_pred = evaluate(
            model, test_data, trans_mean, trans_std, args.output_dir, device)

    print("\nDone!")


if __name__ == '__main__':
    main()
