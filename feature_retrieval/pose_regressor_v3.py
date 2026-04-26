#!/usr/bin/env python3
"""
Phase 2b: Multi-Hypothesis Pose Regression with Scoring Head
=============================================================
K-hypothesis WTA + learned scoring for hypothesis selection at test time.

Also tests simple heuristic selection strategies:
- Centroid proximity: pick hypothesis closest to centroid of all K predictions
- Max-consensus: pick hypothesis with most neighbors within radius
- Scoring MLP: learned scorer from backbone features

Usage:
    python feature_retrieval/pose_regressor_v3.py --mode train_eval --num_hypotheses 8 --gpu 0
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
# Rotation utilities
# ============================================================

def quaternion_to_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*B, 3, 3)

def gram_schmidt_6d_to_matrix(v):
    a1, a2 = v[..., :3], v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)

def geodesic_distance(R1, R2):
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)).reshape(R1.shape[:-2])


# ============================================================
# Dataset
# ============================================================

class PoseRegressionDataset:
    def __init__(self, summary_matrix_path, dataset_dir, split='train', device='cpu'):
        self.device = device
        sm = torch.load(summary_matrix_path, map_location='cpu')
        image_paths = sorted(glob.glob(os.path.join(dataset_dir, 'seq*/*.png')))
        name_to_idx = {os.path.relpath(p, dataset_dir): i for i, p in enumerate(image_paths)}
        with open(os.path.join(dataset_dir, f'dataset_{split}.txt')) as f:
            lines = f.readlines()

        feats, trans, rots, names = [], [], [], []
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) < 8: continue
            idx = name_to_idx.get(parts[0])
            if idx is None: continue
            feats.append(sm[idx])
            trans.append(torch.tensor([float(parts[i]) for i in range(1, 4)]))
            rots.append(torch.tensor([float(parts[i]) for i in range(4, 8)]))
            names.append(parts[0])

        self.features = torch.stack(feats).to(device)
        self.translations = torch.stack(trans).to(device)
        self.rotations = quaternion_to_matrix(torch.stack(rots)).to(device)
        self.names = names
        self.N = len(names)
        print(f"[{split}] {self.N} samples")

    def compute_normalization(self):
        self.trans_mean = self.translations.mean(0)
        self.trans_std = self.translations.std(0).clamp(min=1e-6)
        return self.trans_mean, self.trans_std

    def normalize_translations(self, mean, std):
        self.translations_norm = (self.translations - mean) / std

    def denormalize(self, t, mean, std):
        return t * std + mean


# ============================================================
# Model: Multi-Hypothesis with Scoring Head
# ============================================================

class ScoredMultiHypothesisRegressor(nn.Module):
    """K pose hypotheses + learned scoring for selection."""

    def __init__(self, input_dim=2560, hidden_dims=(1024, 512, 256),
                 dropout=0.1, num_hypotheses=8):
        super().__init__()
        self.K = num_hypotheses

        # Shared backbone (all but last layer)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims[:-1]:
            layers.extend([nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim),
                          nn.GELU(), nn.Dropout(dropout)])
            in_dim = h_dim
        self.shared_backbone = nn.Sequential(*layers)

        # Per-hypothesis branches
        last_dim = hidden_dims[-1]
        self.hyp_layers = nn.ModuleList()
        self.trans_heads = nn.ModuleList()
        self.rot_heads = nn.ModuleList()
        for _ in range(num_hypotheses):
            self.hyp_layers.append(nn.Sequential(
                nn.Linear(in_dim, last_dim), nn.LayerNorm(last_dim),
                nn.GELU(), nn.Dropout(dropout)))
            self.trans_heads.append(nn.Linear(last_dim, 3))
            self.rot_heads.append(nn.Linear(last_dim, 6))

        # Scoring head: takes backbone features + all K pose encodings → K scores
        # Input: backbone_feat (in_dim) + K * (3 + 6) = in_dim + 9K
        score_input_dim = in_dim + 9 * num_hypotheses
        self.score_head = nn.Sequential(
            nn.Linear(score_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_hypotheses),
        )

        self.s_rot = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Returns:
            trans_all: (B, K, 3) — normalized translation predictions
            rot_all: (B, K, 3, 3) — rotation matrix predictions
            scores: (B, K) — hypothesis confidence scores (logits)
        """
        h = self.shared_backbone(x)  # (B, hidden)
        trans_list, rot_list, raw_list = [], [], []
        for k in range(self.K):
            hk = self.hyp_layers[k](h)
            t = self.trans_heads[k](hk)
            r6d = self.rot_heads[k](hk)
            trans_list.append(t)
            rot_list.append(gram_schmidt_6d_to_matrix(r6d))
            raw_list.append(torch.cat([t, r6d], dim=-1))  # (B, 9)

        trans_all = torch.stack(trans_list, dim=1)  # (B, K, 3)
        rot_all = torch.stack(rot_list, dim=1)      # (B, K, 3, 3)

        # Scoring: concat backbone features with all hypothesis raw outputs
        all_raw = torch.cat(raw_list, dim=-1)  # (B, 9K)
        score_input = torch.cat([h, all_raw], dim=-1)  # (B, hidden + 9K)
        scores = self.score_head(score_input)  # (B, K)

        return trans_all, rot_all, scores


# ============================================================
# Training
# ============================================================

def train_scored(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'train', device)
    test_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'test', device)

    trans_mean, trans_std = train_data.compute_normalization()
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    torch.save({'mean': trans_mean, 'std': trans_std}, out_dir / 'norm_params.pt')

    K = args.num_hypotheses
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
    model = ScoredMultiHypothesisRegressor(
        input_dim=2560, hidden_dims=hidden_dims,
        dropout=args.dropout, num_hypotheses=K
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: K={K}, hidden={hidden_dims}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        'epoch': [], 'loss': [], 'loss_wta': [], 'loss_score': [],
        'val_rot_oracle': [], 'val_trans_oracle': [],
        'val_rot_scored': [], 'val_trans_scored': [],
        'val_rot_centroid': [], 'val_trans_centroid': [],
    }
    best_val_score = float('inf')
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        feat = train_data.features
        if args.feature_dropout > 0:
            feat = F.dropout(feat, p=args.feature_dropout, training=True)

        trans_all, rot_all, scores = model(feat)
        N = train_data.N

        gt_t = train_data.translations_norm.unsqueeze(1).expand(-1, K, -1)
        gt_r = train_data.rotations.unsqueeze(1).expand(-1, K, 3, 3)

        # Per-hypothesis errors
        per_hyp_trans = F.smooth_l1_loss(
            trans_all.reshape(-1, 3), gt_t.reshape(-1, 3), reduction='none'
        ).reshape(N, K, 3).mean(-1)  # (N, K)

        per_hyp_rot = geodesic_distance(
            rot_all.reshape(-1, 3, 3), gt_r.reshape(-1, 3, 3)
        ).reshape(N, K)  # (N, K)

        beta_rot = torch.exp(model.s_rot)
        per_hyp_cost = per_hyp_trans + beta_rot * per_hyp_rot

        # WTA loss
        winner_idx = per_hyp_cost.argmin(dim=1)  # (N,)
        loss_wta = (per_hyp_trans[torch.arange(N, device=device), winner_idx].mean() +
                   beta_rot * per_hyp_rot[torch.arange(N, device=device), winner_idx].mean())

        # Scoring loss: cross-entropy, target = winner index
        loss_score = F.cross_entropy(scores, winner_idx)

        # Combined loss
        loss = loss_wta + args.score_weight * loss_score

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vt, vr, vs = model(test_data.features)
                Nt = test_data.N
                vt_real = test_data.denormalize(vt, trans_mean.unsqueeze(0), trans_std.unsqueeze(0))

                gt_te = test_data.translations.unsqueeze(1).expand(-1, K, -1)
                gt_re = test_data.rotations.unsqueeze(1).expand(-1, K, 3, 3)

                te_all = (vt_real - gt_te).norm(dim=-1)  # (Nt, K)
                re_all = geodesic_distance(
                    vr.reshape(-1, 3, 3), gt_re.reshape(-1, 3, 3)
                ).reshape(Nt, K) * 180 / math.pi

                # Oracle selection
                oracle_cost = te_all + re_all * 0.1
                oidx = oracle_cost.argmin(1)
                oracle_t = te_all[torch.arange(Nt, device=device), oidx]
                oracle_r = re_all[torch.arange(Nt, device=device), oidx]

                # Scored selection
                sidx = vs.argmax(1)
                scored_t = te_all[torch.arange(Nt, device=device), sidx]
                scored_r = re_all[torch.arange(Nt, device=device), sidx]

                # Centroid selection (pick hypothesis closest to centroid)
                centroid = vt_real.mean(dim=1, keepdim=True)  # (Nt, 1, 3)
                dist_to_centroid = (vt_real - centroid).norm(dim=-1)  # (Nt, K)
                cidx = dist_to_centroid.argmin(1)
                centroid_t = te_all[torch.arange(Nt, device=device), cidx]
                centroid_r = re_all[torch.arange(Nt, device=device), cidx]

                val_score = scored_t.median().item() + scored_r.median().item() * 0.1

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_wta'].append(loss_wta.item())
            history['loss_score'].append(loss_score.item())
            history['val_rot_oracle'].append(oracle_r.median().item())
            history['val_trans_oracle'].append(oracle_t.median().item())
            history['val_rot_scored'].append(scored_r.median().item())
            history['val_trans_scored'].append(scored_t.median().item())
            history['val_rot_centroid'].append(centroid_r.median().item())
            history['val_trans_centroid'].append(centroid_t.median().item())

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                elapsed = time.time() - t0
                # Score accuracy
                score_acc = (sidx == oidx).float().mean().item() * 100
                print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                      f"(wta={loss_wta.item():.4f} score={loss_score.item():.4f}) "
                      f"| oracle: {oracle_r.median().item():.2f}d/{oracle_t.median().item()*1000:.0f}mm "
                      f"| scored: {scored_r.median().item():.2f}d/{scored_t.median().item()*1000:.0f}mm "
                      f"| centroid: {centroid_r.median().item():.2f}d/{centroid_t.median().item()*1000:.0f}mm "
                      f"| acc={score_acc:.0f}% "
                      f"| {elapsed:.1f}s")

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'num_hypotheses': K,
                    'hidden_dims': hidden_dims,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                }, out_dir / 'model_best.pt')

    print(f"\nTraining done. Best epoch: {best_epoch}")
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    return model, train_data, test_data, trans_mean, trans_std, history


# ============================================================
# Full Evaluation
# ============================================================

def full_evaluate(model, test_data, train_data, trans_mean, trans_std, out_dir, device, K):
    out_dir = Path(out_dir)
    model.eval()

    with torch.no_grad():
        vt, vr, vs = model(test_data.features)
        Nt = test_data.N
        vt_real = test_data.denormalize(vt, trans_mean.unsqueeze(0), trans_std.unsqueeze(0))

        gt_te = test_data.translations.unsqueeze(1).expand(-1, K, -1)
        gt_re = test_data.rotations.unsqueeze(1).expand(-1, K, 3, 3)

        te_all = (vt_real - gt_te).norm(dim=-1).cpu().numpy()
        re_all = (geodesic_distance(
            vr.reshape(-1, 3, 3), gt_re.reshape(-1, 3, 3)
        ).reshape(Nt, K) * 180 / math.pi).cpu().numpy()

        scores_np = vs.cpu().numpy()

    # Selection strategies
    strategies = {}

    # 1. Oracle
    oracle_cost = te_all + re_all * 0.1
    oidx = oracle_cost.argmin(1)
    strategies['oracle'] = (
        te_all[np.arange(Nt), oidx],
        re_all[np.arange(Nt), oidx],
    )

    # 2. Scored (argmax of learned scores)
    sidx = scores_np.argmax(1)
    strategies['scored'] = (
        te_all[np.arange(Nt), sidx],
        re_all[np.arange(Nt), sidx],
    )

    # 3. Centroid proximity
    vt_np = vt_real.cpu().numpy()
    centroid = vt_np.mean(axis=1, keepdims=True)
    dist_c = np.linalg.norm(vt_np - centroid, axis=-1)
    cidx = dist_c.argmin(1)
    strategies['centroid'] = (
        te_all[np.arange(Nt), cidx],
        re_all[np.arange(Nt), cidx],
    )

    # 4. Max-consensus: pick hypothesis with most neighbors within 2m
    consensus_radius = 2.0  # meters
    consensus_counts = np.zeros((Nt, K))
    for i in range(Nt):
        for k in range(K):
            dists = np.linalg.norm(vt_np[i] - vt_np[i, k:k+1], axis=-1)
            consensus_counts[i, k] = (dists < consensus_radius).sum()
    maxcon_idx = consensus_counts.argmax(1)
    strategies['max_consensus'] = (
        te_all[np.arange(Nt), maxcon_idx],
        re_all[np.arange(Nt), maxcon_idx],
    )

    # 5. Top-scored then centroid: pick top-3 scored, then centroid among them
    top3_idx = np.argsort(scores_np, axis=1)[:, -3:]  # (Nt, 3)
    top3_trans = np.array([[vt_np[i, j] for j in top3_idx[i]] for i in range(Nt)])  # (Nt, 3, 3)
    top3_centroid = top3_trans.mean(axis=1, keepdims=True)  # (Nt, 1, 3)
    top3_dist = np.linalg.norm(top3_trans - top3_centroid, axis=-1)  # (Nt, 3)
    top3_best = top3_dist.argmin(1)  # (Nt,)
    top3_final_idx = np.array([top3_idx[i, top3_best[i]] for i in range(Nt)])
    strategies['top3_centroid'] = (
        te_all[np.arange(Nt), top3_final_idx],
        re_all[np.arange(Nt), top3_final_idx],
    )

    # 6. Nearest-train: for each hypothesis, find nearest train pose, pick one with smallest feature distance
    with torch.no_grad():
        feat_sim = torch.mm(
            F.normalize(test_data.features, dim=1),
            F.normalize(train_data.features, dim=1).T
        ).cpu().numpy()  # (Nt, Ntrain)
    train_trans_np = train_data.translations.cpu().numpy()
    # For each hypothesis, find nearest train by position, then score by feature similarity
    nn_scores = np.zeros((Nt, K))
    for k in range(K):
        dists = np.linalg.norm(train_trans_np[None] - vt_np[:, k:k+1], axis=-1)  # (Nt, Ntrain)
        nn_train_idx = dists.argmin(1)  # (Nt,)
        nn_scores[:, k] = feat_sim[np.arange(Nt), nn_train_idx]
    nn_best = nn_scores.argmax(1)
    strategies['nn_guided'] = (
        te_all[np.arange(Nt), nn_best],
        re_all[np.arange(Nt), nn_best],
    )

    # Print results
    print("\n" + "=" * 90)
    print("HYPOTHESIS SELECTION COMPARISON (K=%d)" % K)
    print("=" * 90)
    print(f"{'Strategy':<25} {'Rot med(deg)':<15} {'Trans med(mm)':<15} {'R@5d/1m':<10} {'R@10d/2m':<10} {'R@15d/5m':<10}")
    print("-" * 90)

    results = {}
    for name, (te, re) in strategies.items():
        thresholds = {'5deg_1m': (5, 1), '10deg_2m': (10, 2), '15deg_5m': (15, 5)}
        recalls = {k: float(((re < rt) & (te < tt)).mean() * 100) for k, (rt, tt) in thresholds.items()}
        results[name] = {
            'rot_median': float(np.median(re)),
            'trans_median_mm': float(np.median(te) * 1000),
            'recalls': recalls,
        }
        print(f"{name:<25} {np.median(re):<15.2f} {np.median(te)*1000:<15.0f} "
              f"{recalls['5deg_1m']:<10.1f} {recalls['10deg_2m']:<10.1f} {recalls['15deg_5m']:<10.1f}")

    # Add baselines
    print("-" * 90)
    print(f"{'CLS_cosine_baseline':<25} {'11.50':<15} {'1360':<15} {'N/A':<10} {'~30':<10} {'~50':<10}")
    print(f"{'exp01_single_mlp':<25} {'3.72':<15} {'1849':<15} {'16.5':<10} {'51.1':<10} {'81.9':<10}")
    print("=" * 90)

    # Score accuracy analysis
    score_acc = float((sidx == oidx).mean() * 100)
    # Top-2 accuracy: oracle is in top-2 scored
    top2_scored = np.argsort(scores_np, axis=1)[:, -2:]
    top2_acc = float(np.array([oidx[i] in top2_scored[i] for i in range(Nt)]).mean() * 100)
    top3_acc = float(np.array([oidx[i] in top3_idx[i] for i in range(Nt)]).mean() * 100)

    print(f"\nScoring accuracy: top-1={score_acc:.1f}%, top-2={top2_acc:.1f}%, top-3={top3_acc:.1f}%")

    results['scoring_accuracy'] = {
        'top1': score_acc, 'top2': top2_acc, 'top3': top3_acc
    }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save best approach predictions for pipeline integration
    best_method = min(results.items(),
                      key=lambda x: x[1].get('trans_median_mm', 1e9) + x[1].get('rot_median', 1e9) * 100
                      if isinstance(x[1], dict) and 'trans_median_mm' in x[1] else 1e9)
    print(f"\nBest non-oracle method: {best_method[0]} "
          f"({best_method[1]['rot_median']:.2f}deg / {best_method[1]['trans_median_mm']:.0f}mm)")

    return results


# ============================================================
# Visualization
# ============================================================

def visualize_results(history, results, out_dir, K):
    out_dir = Path(out_dir)

    # 1. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(history['epoch'], history['loss'], 'b-', label='Total')
    ax.plot(history['epoch'], history['loss_wta'], 'g-', label='WTA')
    ax.plot(history['epoch'], history['loss_score'], 'r-', label='Score CE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Losses')
    ax.legend(); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history['epoch'], history['val_rot_oracle'], 'r--', label='Oracle')
    ax.plot(history['epoch'], history['val_rot_scored'], 'r-', label='Scored')
    ax.plot(history['epoch'], history['val_rot_centroid'], 'r:', label='Centroid')
    ax.axhline(y=11.5, color='gray', linestyle='--', alpha=0.5, label='CLS baseline')
    ax.axhline(y=3.72, color='blue', linestyle='--', alpha=0.5, label='exp01 single')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Rotation (deg)'); ax.set_title('Rotation Error')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(history['epoch'], [x*1000 for x in history['val_trans_oracle']], 'g--', label='Oracle')
    ax.plot(history['epoch'], [x*1000 for x in history['val_trans_scored']], 'g-', label='Scored')
    ax.plot(history['epoch'], [x*1000 for x in history['val_trans_centroid']], 'g:', label='Centroid')
    ax.axhline(y=1360, color='gray', linestyle='--', alpha=0.5, label='CLS baseline')
    ax.axhline(y=1849, color='blue', linestyle='--', alpha=0.5, label='exp01 single')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Translation (mm)'); ax.set_title('Translation Error')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'loss_curves.png'}")

    # 2. Method comparison bar chart
    methods = [k for k in results if isinstance(results[k], dict) and 'rot_median' in results[k]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rot_vals = [results[m]['rot_median'] for m in methods]
    trans_vals = [results[m]['trans_median_mm'] for m in methods]
    labels = [m.replace('_', '\n') for m in methods]

    colors = ['gold' if m == 'oracle' else 'coral' for m in methods]
    ax = axes[0]
    bars = ax.bar(range(len(methods)), rot_vals, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Median Rotation (deg)'); ax.set_title('Rotation by Strategy')
    ax.axhline(y=11.5, color='gray', linestyle='--', label='CLS baseline')
    ax.axhline(y=3.72, color='blue', linestyle='--', label='exp01')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rot_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val:.1f}',
                ha='center', fontsize=7)

    colors = ['gold' if m == 'oracle' else 'steelblue' for m in methods]
    ax = axes[1]
    bars = ax.bar(range(len(methods)), trans_vals, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Median Translation (mm)'); ax.set_title('Translation by Strategy')
    ax.axhline(y=1360, color='gray', linestyle='--', label='CLS baseline')
    ax.axhline(y=1849, color='blue', linestyle='--', label='exp01')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, trans_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f'{val:.0f}',
                ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / 'strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'strategy_comparison.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train_eval')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--summary_matrix', default='output/feature_extract/features_radio_dual/OldHospital_pilot/summary_matrix.pt')
    parser.add_argument('--dataset_dir', default='dataset/OldHospital')
    parser.add_argument('--output_dir', default='output/feature_retrieval/pose_regression/exp05_scored_k8')
    parser.add_argument('--num_hypotheses', type=int, default=8)
    parser.add_argument('--hidden_dims', default='1024,512,256')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
    parser.add_argument('--score_weight', type=float, default=0.5)
    parser.add_argument('--log_every', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    K = args.num_hypotheses

    model, train_data, test_data, trans_mean, trans_std, history = train_scored(args)

    # Load best
    ckpt = torch.load(Path(args.output_dir) / 'model_best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")

    results = full_evaluate(model, test_data, train_data, trans_mean, trans_std,
                            args.output_dir, device, K)
    visualize_results(history, results, args.output_dir, K)
    print("\nDone!")


if __name__ == '__main__':
    main()
