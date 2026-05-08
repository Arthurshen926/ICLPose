#!/usr/bin/env python3
"""
Phase 2: Multi-Hypothesis Pose Regression + Hybrid Approaches
==============================================================
Extends Phase 1 with:
- K-hypothesis WTA (Winner-Takes-All) regression
- Hybrid: regression rotation + nearest-neighbor translation
- Configurable MLP width/depth

Usage:
    # Multi-hypothesis WTA
    python feature_retrieval/pose_regressor_v2.py --mode train_eval --num_hypotheses 8 \
        --output_dir output/feature_retrieval/pose_regression/exp02_multi_hyp_k8

    # Wider MLP
    python feature_retrieval/pose_regressor_v2.py --mode train_eval --hidden_dims 2048,1024,512 \
        --output_dir output/feature_retrieval/pose_regression/exp04_wider_mlp

    # Hybrid eval from existing checkpoint
    python feature_retrieval/pose_regressor_v2.py --mode hybrid_eval \
        --checkpoint output/feature_retrieval/pose_regression/exp01_direct_mlp/model_best.pt
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
# Utility: Rotation representations (same as v1)
# ============================================================

def quaternion_to_matrix(q):
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
    a1 = v[..., :3]
    a2 = v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def geodesic_distance(R1, R2):
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).reshape(R1.shape[:-2])


# ============================================================
# Dataset (same as v1)
# ============================================================

class PoseRegressionDataset:
    def __init__(self, summary_matrix_path, dataset_dir, split='train', device='cpu'):
        self.device = device
        summary_matrix = torch.load(summary_matrix_path, map_location='cpu')
        assert summary_matrix.shape[1] == 2560

        image_paths = sorted(glob.glob(os.path.join(dataset_dir, 'seq*/*.png')))
        name_to_idx = {os.path.relpath(p, dataset_dir): i for i, p in enumerate(image_paths)}

        split_file = os.path.join(dataset_dir, f'dataset_{split}.txt')
        with open(split_file) as f:
            lines = f.readlines()

        features_list, translations_list, rotations_list, names_list = [], [], [], []
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            img_name = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            idx = name_to_idx.get(img_name)
            if idx is None:
                continue
            features_list.append(summary_matrix[idx])
            translations_list.append(torch.tensor([x, y, z], dtype=torch.float32))
            rotations_list.append(torch.tensor([w, p, q, r], dtype=torch.float32))
            names_list.append(img_name)

        self.features = torch.stack(features_list).to(device)
        self.translations = torch.stack(translations_list).to(device)
        quat = torch.stack(rotations_list)
        self.rotations = quaternion_to_matrix(quat).to(device)
        self.names = names_list
        self.N = len(names_list)
        print(f"[{split}] Loaded {self.N} samples, features {self.features.shape}")

    def compute_normalization(self):
        self.trans_mean = self.translations.mean(dim=0)
        self.trans_std = self.translations.std(dim=0).clamp(min=1e-6)
        return self.trans_mean, self.trans_std

    def normalize_translations(self, mean, std):
        self.translations_norm = (self.translations - mean) / std

    def denormalize(self, trans_norm, mean, std):
        return trans_norm * std + mean


# ============================================================
# Models
# ============================================================

class MultiHypothesisPoseRegressor(nn.Module):
    """K-hypothesis pose regression with shared backbone + K independent heads."""

    def __init__(self, input_dim=2560, hidden_dims=(1024, 512, 256),
                 dropout=0.1, num_hypotheses=8):
        super().__init__()
        self.K = num_hypotheses

        # Shared backbone
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims[:-1]:  # All but last layer shared
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.shared_backbone = nn.Sequential(*layers)

        # Per-hypothesis final layer + heads
        last_dim = hidden_dims[-1]
        self.hyp_layers = nn.ModuleList()
        self.trans_heads = nn.ModuleList()
        self.rot_heads = nn.ModuleList()

        for _ in range(num_hypotheses):
            self.hyp_layers.append(nn.Sequential(
                nn.Linear(in_dim, last_dim),
                nn.LayerNorm(last_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            self.trans_heads.append(nn.Linear(last_dim, 3))
            self.rot_heads.append(nn.Linear(last_dim, 6))

        # Learnable loss weight
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
        Returns:
            trans_all: (B, K, 3)
            rot_all: (B, K, 3, 3)
        """
        h = self.shared_backbone(x)  # (B, hidden)
        trans_list, rot_list = [], []
        for k in range(self.K):
            hk = self.hyp_layers[k](h)
            trans_list.append(self.trans_heads[k](hk))
            rot_6d = self.rot_heads[k](hk)
            rot_list.append(gram_schmidt_6d_to_matrix(rot_6d))

        return torch.stack(trans_list, dim=1), torch.stack(rot_list, dim=1)


class PoseRegressorMLP(nn.Module):
    """Single-hypothesis MLP (same as v1 but with configurable dims)."""

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
        self.s_rot = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        trans = self.trans_head(h)
        rot_6d = self.rot_head(h)
        rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
        return trans, rot_mat


# ============================================================
# Training: Multi-hypothesis WTA
# ============================================================

def train_multi_hypothesis(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'train', device)
    test_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'test', device)

    trans_mean, trans_std = train_data.compute_normalization()
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    print(f"Translation normalization: mean={trans_mean.cpu().numpy()}, std={trans_std.cpu().numpy()}")

    torch.save({'mean': trans_mean, 'std': trans_std}, out_dir / 'norm_params.pt')

    K = args.num_hypotheses
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
    model = MultiHypothesisPoseRegressor(
        input_dim=2560, hidden_dims=hidden_dims,
        dropout=args.dropout, num_hypotheses=K
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: K={K} hypotheses, hidden={hidden_dims}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        'epoch': [], 'loss': [], 'loss_trans': [], 'loss_rot': [],
        'val_rot_median': [], 'val_trans_median': [], 'beta_rot': [],
        'val_rot_median_oracle': [], 'val_trans_median_oracle': [],
    }
    best_val_score = float('inf')
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()

        feat = train_data.features
        if args.feature_dropout > 0:
            feat = F.dropout(feat, p=args.feature_dropout, training=True)

        # Forward: (N, K, 3) and (N, K, 3, 3)
        trans_all, rot_all = model(feat)
        N = train_data.N

        # WTA loss: compute per-hypothesis errors, pick winner
        gt_trans = train_data.translations_norm.unsqueeze(1).expand(-1, K, -1)  # (N, K, 3)
        gt_rot = train_data.rotations.unsqueeze(1).expand(-1, K, 3, 3)         # (N, K, 3, 3)

        # Per-hypothesis translation loss: (N, K)
        per_hyp_trans = F.smooth_l1_loss(
            trans_all.reshape(-1, 3), gt_trans.reshape(-1, 3), reduction='none'
        ).reshape(N, K, 3).mean(dim=-1)

        # Per-hypothesis rotation loss: (N, K)
        per_hyp_rot = geodesic_distance(
            rot_all.reshape(-1, 3, 3), gt_rot.reshape(-1, 3, 3)
        ).reshape(N, K)

        # Combined per-hypothesis cost
        beta_rot = torch.exp(model.s_rot)
        per_hyp_cost = per_hyp_trans + beta_rot * per_hyp_rot  # (N, K)

        # Winner-Takes-All: only backprop through best hypothesis
        winner_idx = per_hyp_cost.argmin(dim=1)  # (N,)
        winner_trans_loss = per_hyp_trans[torch.arange(N, device=device), winner_idx].mean()
        winner_rot_loss = per_hyp_rot[torch.arange(N, device=device), winner_idx].mean()
        loss = winner_trans_loss + beta_rot * winner_rot_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vt_all, vr_all = model(test_data.features)  # (Ntest, K, 3), (Ntest, K, 3, 3)
                Nt = test_data.N

                # Denormalize all hypotheses
                vt_all_real = test_data.denormalize(vt_all, trans_mean.unsqueeze(0), trans_std.unsqueeze(0))

                # Per-hypothesis errors
                gt_t_exp = test_data.translations.unsqueeze(1).expand(-1, K, -1)
                gt_r_exp = test_data.rotations.unsqueeze(1).expand(-1, K, 3, 3)

                trans_err_all = (vt_all_real - gt_t_exp).norm(dim=-1)  # (Nt, K)
                rot_err_all = geodesic_distance(
                    vr_all.reshape(-1, 3, 3), gt_r_exp.reshape(-1, 3, 3)
                ).reshape(Nt, K) * 180 / math.pi  # (Nt, K)

                # Oracle: pick best hypothesis per sample
                # Use combined metric for selection
                combined_err = trans_err_all + rot_err_all * 0.1
                oracle_idx = combined_err.argmin(dim=1)  # (Nt,)
                oracle_trans = trans_err_all[torch.arange(Nt, device=device), oracle_idx]
                oracle_rot = rot_err_all[torch.arange(Nt, device=device), oracle_idx]

                # Hypothesis 0 (default/best-trained)
                h0_trans = trans_err_all[:, 0]
                h0_rot = rot_err_all[:, 0]

                val_trans_med = oracle_trans.median().item()
                val_rot_med = oracle_rot.median().item()
                val_score = val_trans_med + val_rot_med * 0.1

                h0_trans_med = h0_trans.median().item()
                h0_rot_med = h0_rot.median().item()

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_trans'].append(winner_trans_loss.item())
            history['loss_rot'].append(winner_rot_loss.item())
            history['val_rot_median'].append(h0_rot_med)
            history['val_trans_median'].append(h0_trans_med)
            history['val_rot_median_oracle'].append(val_rot_med)
            history['val_trans_median_oracle'].append(val_trans_med)
            history['beta_rot'].append(beta_rot.item())

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                      f"beta={beta_rot.item():.2f} "
                      f"| h0: {h0_rot_med:.2f}deg/{h0_trans_med*1000:.0f}mm "
                      f"| oracle: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                      f"| {elapsed:.1f}s")

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_rot_median_oracle': val_rot_med,
                    'val_trans_median_oracle': val_trans_med,
                    'num_hypotheses': K,
                    'hidden_dims': hidden_dims,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                }, out_dir / 'model_best.pt')

    print(f"\nTraining done. Best epoch: {best_epoch}")

    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'num_hypotheses': K,
        'hidden_dims': hidden_dims,
        'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
    }, out_dir / 'model_final.pt')

    return model, train_data, test_data, trans_mean, trans_std, history


# ============================================================
# Training: Single hypothesis (same as v1 but configurable)
# ============================================================

def train_single(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'train', device)
    test_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'test', device)

    trans_mean, trans_std = train_data.compute_normalization()
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    print(f"Translation normalization: mean={trans_mean.cpu().numpy()}, std={trans_std.cpu().numpy()}")
    torch.save({'mean': trans_mean, 'std': trans_std}, out_dir / 'norm_params.pt')

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
    model = PoseRegressorMLP(
        input_dim=2560, hidden_dims=hidden_dims, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: single-hypothesis, hidden={hidden_dims}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        'epoch': [], 'loss': [], 'loss_trans': [], 'loss_rot': [],
        'val_rot_median': [], 'val_trans_median': [], 'beta_rot': [],
    }
    best_val_score = float('inf')
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        feat = train_data.features
        if args.feature_dropout > 0:
            feat = F.dropout(feat, p=args.feature_dropout, training=True)

        trans_pred, rot_pred = model(feat)
        loss_trans = F.smooth_l1_loss(trans_pred, train_data.translations_norm)
        loss_rot = geodesic_distance(rot_pred, train_data.rotations).mean()
        beta_rot = torch.exp(model.s_rot)
        loss = loss_trans + beta_rot * loss_rot

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

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

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_rot_median': val_rot_med,
                    'val_trans_median': val_trans_med,
                    'hidden_dims': hidden_dims,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                }, out_dir / 'model_best.pt')

    print(f"\nTraining done. Best epoch: {best_epoch}, score: {best_val_score:.4f}")
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    return model, train_data, test_data, trans_mean, trans_std, history


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, test_data, train_data, trans_mean, trans_std, out_dir, device,
                   is_multi=False, K=1):
    """Unified evaluation for single and multi-hypothesis models."""
    out_dir = Path(out_dir)
    model.eval()

    with torch.no_grad():
        if is_multi:
            trans_all, rot_all = model(test_data.features)  # (N, K, 3), (N, K, 3, 3)
            trans_all_real = test_data.denormalize(
                trans_all, trans_mean.unsqueeze(0), trans_std.unsqueeze(0))

            gt_t = test_data.translations.unsqueeze(1).expand(-1, K, -1)
            gt_r = test_data.rotations.unsqueeze(1).expand(-1, K, 3, 3)
            N = test_data.N

            trans_err_all = (trans_all_real - gt_t).norm(dim=-1).cpu().numpy()  # (N, K)
            rot_err_all = (geodesic_distance(
                rot_all.reshape(-1, 3, 3), gt_r.reshape(-1, 3, 3)
            ).reshape(N, K) * 180 / math.pi).cpu().numpy()

            # Oracle selection
            combined = trans_err_all + rot_err_all * 0.1
            oracle_idx = combined.argmin(axis=1)
            trans_errors = trans_err_all[np.arange(N), oracle_idx]
            rot_errors = rot_err_all[np.arange(N), oracle_idx]
            trans_pred = trans_all_real.cpu().numpy()[np.arange(N), oracle_idx]

            # Also report h0 and per-hypothesis stats
            h0_trans = trans_err_all[:, 0]
            h0_rot = rot_err_all[:, 0]
        else:
            trans_pred_norm, rot_pred = model(test_data.features)
            trans_pred_real = test_data.denormalize(trans_pred_norm, trans_mean, trans_std)
            trans_errors = (trans_pred_real - test_data.translations).norm(dim=-1).cpu().numpy()
            rot_errors = (geodesic_distance(rot_pred, test_data.rotations) * 180 / math.pi).cpu().numpy()
            trans_pred = trans_pred_real.cpu().numpy()

    # --- Hybrid evaluation: regression rotation + NN translation ---
    # For each test sample, find nearest train sample by FEATURE similarity
    with torch.no_grad():
        feat_sim = torch.mm(
            F.normalize(test_data.features, dim=1),
            F.normalize(train_data.features, dim=1).T
        )  # (Ntest, Ntrain)
        nn_idx = feat_sim.argmax(dim=1)  # (Ntest,)
        nn_trans = train_data.translations[nn_idx]  # (Ntest, 3) — use NN translation

    hybrid_trans_errors = (nn_trans - test_data.translations).norm(dim=-1).cpu().numpy()

    # Also try: NN by predicted position (find nearest train position to predicted position)
    pred_trans_tensor = torch.tensor(trans_pred, device=device)
    dist_to_train = torch.cdist(pred_trans_tensor.unsqueeze(0),
                                 train_data.translations.unsqueeze(0)).squeeze(0)  # (Ntest, Ntrain)
    spatial_nn_idx = dist_to_train.argmin(dim=1)
    spatial_nn_trans = train_data.translations[spatial_nn_idx]
    spatial_nn_rot = train_data.rotations[spatial_nn_idx]
    spatial_nn_trans_errors = (spatial_nn_trans - test_data.translations).norm(dim=-1).cpu().numpy()
    spatial_nn_rot_errors = (geodesic_distance(spatial_nn_rot, test_data.rotations) * 180 / math.pi).cpu().numpy()

    # Metrics
    results = {}

    # Primary method results
    method_name = f'multi_hyp_k{K}_oracle' if is_multi else 'single_hyp'
    results[method_name] = _compute_metrics(trans_errors, rot_errors)

    if is_multi:
        results[f'multi_hyp_k{K}_h0'] = _compute_metrics(h0_trans, h0_rot)

    # Hybrid: regression rotation + feature-NN translation
    results['hybrid_regrot_nntrans'] = _compute_metrics(hybrid_trans_errors, rot_errors)

    # Hybrid: spatial-NN (regression position -> nearest train)
    results['hybrid_spatial_nn'] = _compute_metrics(spatial_nn_trans_errors, spatial_nn_rot_errors)

    # Hybrid: regression rotation + spatial-NN translation
    results['hybrid_regrot_spatial_nn_trans'] = _compute_metrics(spatial_nn_trans_errors, rot_errors)

    # Baseline
    results['baseline_cls_cosine'] = {
        'rotation_median_deg': 11.5, 'translation_median_mm': 1360,
    }

    # Print comparison
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Method':<40} {'Rot(deg)':<12} {'Trans(mm)':<12} {'R@10d/2m':<10}")
    print("-" * 80)
    for name, m in results.items():
        if 'rotation_median_deg' in m:
            rot_med = m['rotation_median_deg']
            trans_med = m['translation_median_mm']
            recall = m.get('recall_10deg_2m', 'N/A')
            recall_str = f"{recall:.1f}%" if isinstance(recall, float) else recall
            print(f"{name:<40} {rot_med:<12.2f} {trans_med:<12.0f} {recall_str:<10}")
    print("=" * 80)

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results, trans_errors, rot_errors, trans_pred


def _compute_metrics(trans_errors, rot_errors):
    thresholds = [('5deg_1m', 5, 1.0), ('10deg_2m', 10, 2.0), ('15deg_5m', 15, 5.0), ('25deg_5m', 25, 5.0)]
    m = {
        'rotation_median_deg': float(np.median(rot_errors)),
        'rotation_mean_deg': float(np.mean(rot_errors)),
        'translation_median_mm': float(np.median(trans_errors) * 1000),
        'translation_mean_mm': float(np.mean(trans_errors) * 1000),
    }
    for name, rt, tt in thresholds:
        mask = (rot_errors < rt) & (trans_errors < tt)
        m[f'recall_{name}'] = float(mask.mean()) * 100
    return m


# ============================================================
# Visualization
# ============================================================

def visualize_comparison(results, out_dir):
    """Bar chart comparing all methods."""
    out_dir = Path(out_dir)

    methods = []
    rot_meds = []
    trans_meds = []
    for name, m in results.items():
        if 'rotation_median_deg' in m:
            methods.append(name.replace('_', '\n'))
            rot_meds.append(m['rotation_median_deg'])
            trans_meds.append(m['translation_median_mm'])

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(methods)*2), 6))

    ax = axes[0]
    bars = ax.bar(range(len(methods)), rot_meds, color='coral', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7, rotation=0)
    ax.set_ylabel('Median Rotation Error (deg)')
    ax.set_title('Rotation Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rot_meds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(range(len(methods)), trans_meds, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7, rotation=0)
    ax.set_ylabel('Median Translation Error (mm)')
    ax.set_title('Translation Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, trans_meds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'method_comparison.png'}")


def visualize_training(history, out_dir, is_multi=False):
    """Training curves."""
    out_dir = Path(out_dir)

    if is_multi:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
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
    ax.plot(history['epoch'], history['val_rot_median'], 'r-', label='h0 Rotation')
    if is_multi and 'val_rot_median_oracle' in history:
        ax.plot(history['epoch'], history['val_rot_median_oracle'], 'r--', label='Oracle Rotation')
    ax.axhline(y=11.5, color='gray', linestyle='--', alpha=0.5, label='CLS baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Median Rotation Error (deg)')
    ax.set_title('Validation Rotation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    val_trans_mm = [x * 1000 for x in history['val_trans_median']]
    ax.plot(history['epoch'], val_trans_mm, 'g-', label='h0 Translation')
    if is_multi and 'val_trans_median_oracle' in history:
        oracle_mm = [x * 1000 for x in history['val_trans_median_oracle']]
        ax.plot(history['epoch'], oracle_mm, 'g--', label='Oracle Translation')
    ax.axhline(y=1360, color='gray', linestyle='--', alpha=0.5, label='CLS baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Median Translation Error (mm)')
    ax.set_title('Validation Translation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'loss_curves.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Hypothesis Pose Regression')
    parser.add_argument('--mode', type=str, default='train_eval',
                        choices=['train_eval', 'hybrid_eval'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--summary_matrix', type=str,
                        default='output/feature_extract/features_radio_dual/OldHospital_pilot/summary_matrix.pt')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_retrieval/pose_regression/exp02_multi_hyp')
    parser.add_argument('--checkpoint', type=str, default=None)

    # Architecture
    parser.add_argument('--num_hypotheses', type=int, default=1,
                        help='K hypotheses (1=single, >1=multi-hypothesis WTA)')
    parser.add_argument('--hidden_dims', type=str, default='1024,512,256',
                        help='Comma-separated hidden dimensions')

    # Training
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
    parser.add_argument('--log_every', type=int, default=10)

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train_eval':
        K = args.num_hypotheses
        is_multi = K > 1

        if is_multi:
            model, train_data, test_data, trans_mean, trans_std, history = \
                train_multi_hypothesis(args)
        else:
            model, train_data, test_data, trans_mean, trans_std, history = \
                train_single(args)

        # Load best model
        ckpt = torch.load(Path(args.output_dir) / 'model_best.pt', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nLoaded best model from epoch {ckpt['epoch']}")

        results, trans_errors, rot_errors, trans_pred = evaluate_model(
            model, test_data, train_data, trans_mean, trans_std,
            args.output_dir, device, is_multi=is_multi, K=K)

        visualize_training(history, args.output_dir, is_multi=is_multi)
        visualize_comparison(results, args.output_dir)

    elif args.mode == 'hybrid_eval':
        assert args.checkpoint, "Must provide --checkpoint"
        ckpt = torch.load(args.checkpoint, map_location=device)
        trans_mean = ckpt['norm_params']['mean'].to(device)
        trans_std = ckpt['norm_params']['std'].to(device)

        hidden_dims = ckpt.get('hidden_dims', (1024, 512, 256))
        K = ckpt.get('num_hypotheses', 1)
        is_multi = K > 1

        if is_multi:
            model = MultiHypothesisPoseRegressor(
                hidden_dims=hidden_dims, num_hypotheses=K).to(device)
        else:
            model = PoseRegressorMLP(hidden_dims=hidden_dims).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        train_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'train', device)
        test_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'test', device)
        test_data.normalize_translations(trans_mean, trans_std)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results, _, _, _ = evaluate_model(
            model, test_data, train_data, trans_mean, trans_std,
            args.output_dir, device, is_multi=is_multi, K=K)
        visualize_comparison(results, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
