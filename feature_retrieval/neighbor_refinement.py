#!/usr/bin/env python3
"""
Experiment 25: Neighbor-Weighted Refinement (Post-Processing)
=============================================================
Uses base model predictions as initialization, then refines using
nearest training images weighted by feature similarity.

Two-pass approach:
  Pass 1: Base model predicts initial pose → p_hat
  Pass 2: Find K nearest training images to p_hat (in pose space),
           weight by feature similarity, refine prediction

Also includes a learned refinement model (two-pass training).
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def geodesic_distance(R1, R2):
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).reshape(R1.shape[:-2])


def gram_schmidt_6d_to_matrix(v):
    a1 = v[..., :3]
    a2 = v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def load_poses(dataset_dir, split):
    """Load poses from dataset file."""
    split_file = os.path.join(dataset_dir, f'dataset_{split}.txt')
    with open(split_file) as f:
        lines = f.readlines()
    
    translations = []
    rotations = []
    names = []
    
    for line in lines[3:]:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        names.append(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        translations.append([x, y, z])
        rotations.append([w, p, q, r])
    
    trans = torch.tensor(translations, dtype=torch.float32)
    quats = torch.tensor(rotations, dtype=torch.float32)
    rot_mats = quaternion_to_matrix(quats)
    return trans, rot_mats, names


def load_base_predictions(exp_dir, dataset_dir, feature_dir, gpu=0):
    """Load base model and generate predictions for test set."""
    from feature_retrieval.patch_regressor_v7 import (
        PatchPoseRegressor, PatchPoseDataset, SPPPooling, precompute_pooled_features
    )
    
    device = torch.device(f'cuda:{gpu}')
    
    # Load config
    config_path = Path(exp_dir) / 'config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    model = PatchPoseRegressor(
        pool_type=config['pool'],
        feat_mode=config['feat'],
        patch_dim=config['patch_dim'],
        hidden_dims=tuple(config['hidden_dims']),
        dropout=config['dropout'],
        attn_heads=config.get('attn_heads', 4),
        spp_levels=config.get('spp_levels', None),
    ).to(device)
    
    ckpt = torch.load(Path(exp_dir) / 'model_best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load normalization params
    norm = ckpt['norm_params']
    trans_mean = norm['mean'].to(device)
    trans_std = norm['std'].to(device)
    
    # Load test data and predict
    use_fine = 'fine' in config['feat'] or 'both' in config['feat']
    use_coarse = 'coarse' in config['feat'] or 'both' in config['feat']
    use_summary = '+sum' in config['feat']
    
    test_data = PatchPoseDataset(
        feature_dir, dataset_dir, 'test', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    
    # Precompute pooled features
    test_pooled = precompute_pooled_features(model, test_data, device, batch_size=16)
    
    with torch.no_grad():
        trans_pred_norm, rot_pred = model.forward_from_pooled(test_pooled)
        trans_pred = trans_pred_norm * trans_std + trans_mean
    
    return trans_pred.cpu(), rot_pred.cpu()


def evaluate_predictions(trans_pred, rot_pred, trans_gt, rot_gt, label=""):
    """Evaluate pose predictions against ground truth."""
    trans_errors = (trans_pred - trans_gt).norm(dim=-1).numpy()
    rot_errors = (geodesic_distance(rot_pred, rot_gt) * 180 / math.pi).numpy()
    
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
    ]
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Rotation median:    {np.median(rot_errors):.2f}°")
    print(f"  Translation median: {np.median(trans_errors)*1000:.0f}mm")
    
    results = {}
    for name, rot_th, trans_th in thresholds:
        rot_pass = (rot_errors < rot_th).mean() * 100
        trans_pass = (trans_errors < trans_th).mean() * 100
        combined = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100
        print(f"  R@{name}: {combined:.1f}%  (rot={rot_pass:.1f}%, trans={trans_pass:.1f}%)")
        results[name] = combined
    
    return results, trans_errors, rot_errors


def evaluate_retrieval(trans_pred, train_trans, train_rot, test_trans, test_rot, label=""):
    """Evaluate retrieval: find nearest training image to predicted position."""
    tree = cKDTree(train_trans.numpy())
    _, nn_idx = tree.query(trans_pred.numpy(), k=1)
    
    retr_trans = train_trans[nn_idx]
    retr_rot = train_rot[nn_idx]
    
    return evaluate_predictions(retr_trans, retr_rot, test_trans, test_rot, label=label)


# ============================================================
# Method 1: Simple Neighbor Voting (no training needed)
# ============================================================

def neighbor_voting_refinement(
    base_trans_pred,    # (N_test, 3) — base model's translation predictions
    base_rot_pred,      # (N_test, 3, 3) — base model's rotation predictions
    train_trans,        # (N_train, 3) — training positions
    train_rot,          # (N_train, 3, 3) — training rotations
    test_summary,       # (N_test, D) — test image features
    train_summary,      # (N_train, D) — training image features
    K=5,                # number of neighbors
    alpha=0.5,          # interpolation weight: alpha * neighbor_avg + (1-alpha) * base
    weight_by_sim=True, # weight neighbors by feature similarity
    weight_by_dist=True,# weight neighbors by inverse pose distance
):
    """Refine base predictions using nearest training neighbors."""
    N_test = base_trans_pred.shape[0]
    
    # Build KD-tree on training positions
    tree = cKDTree(train_trans.numpy())
    
    # Find K nearest training images to each predicted position
    dists, nn_indices = tree.query(base_trans_pred.numpy(), k=K)  # (N_test, K)
    
    refined_trans = torch.zeros_like(base_trans_pred)
    refined_rot = torch.zeros_like(base_rot_pred)
    
    # Normalize features for cosine similarity
    test_norm = F.normalize(test_summary, dim=-1)
    train_norm = F.normalize(train_summary, dim=-1)
    
    for i in range(N_test):
        neighbor_idx = nn_indices[i]  # (K,)
        neighbor_trans = train_trans[neighbor_idx]  # (K, 3)
        neighbor_rot = train_rot[neighbor_idx]  # (K, 3, 3)
        neighbor_dists = dists[i]  # (K,)
        
        # Compute weights
        weights = torch.ones(K, dtype=torch.float32)
        
        if weight_by_sim:
            # Cosine similarity between test image and each neighbor
            sims = (test_norm[i:i+1] @ train_norm[neighbor_idx].T).squeeze(0)  # (K,)
            sims = torch.clamp(sims, min=0.0)  # only positive similarities
            weights = weights * (sims + 1e-6)
        
        if weight_by_dist:
            # Inverse distance weighting
            inv_dist = 1.0 / (torch.tensor(neighbor_dists, dtype=torch.float32) + 0.1)
            weights = weights * inv_dist
        
        weights = weights / weights.sum()  # normalize
        
        # Weighted average of neighbor translations
        neighbor_avg_trans = (weights.unsqueeze(-1) * neighbor_trans).sum(dim=0)
        
        # Interpolate between base prediction and neighbor average
        refined_trans[i] = (1 - alpha) * base_trans_pred[i] + alpha * neighbor_avg_trans
        
        # For rotation: use the closest neighbor's rotation (or the base prediction)
        # Weighted rotation average via SVD projection
        weighted_rot = (weights.reshape(K, 1, 1) * neighbor_rot).sum(dim=0)
        U, S, Vh = torch.linalg.svd(weighted_rot)
        rot_avg = U @ Vh
        # Ensure proper rotation (det = +1)
        if torch.det(rot_avg) < 0:
            Vh[-1] *= -1
            rot_avg = U @ Vh
        
        refined_rot[i] = (1 - alpha) * base_rot_pred[i] + alpha * rot_avg
        # Re-orthogonalize
        U, S, Vh = torch.linalg.svd(refined_rot[i])
        det = torch.det(U @ Vh)
        if det < 0:
            Vh[-1] *= -1
        refined_rot[i] = U @ Vh
    
    return refined_trans, refined_rot


# ============================================================
# Method 2: Learned Residual Refinement (with K-fold CV)
# ============================================================

class RefinementMLP(nn.Module):
    """Small MLP that predicts pose correction given base prediction + neighbor info."""
    
    def __init__(self, input_dim, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_d = h_d
        self.backbone = nn.Sequential(*layers)
        self.trans_head = nn.Linear(hidden_dims[-1], 3)
        self.rot_head = nn.Linear(hidden_dims[-1], 6)
    
    def forward(self, x):
        h = self.backbone(x)
        trans_correction = self.trans_head(h)
        rot_6d = self.rot_head(h)
        return trans_correction, rot_6d


def build_refinement_features(
    base_trans_pred,    # (N, 3) — initial predictions (normalized)
    train_trans_norm,   # (N_train, 3) — training positions (normalized)
    train_rot,          # (N_train, 3, 3) — training rotations
    query_summary,      # (N, D) — query features
    train_summary,      # (N_train, D) — training features
    K=10,
    exclude_self_idx=None,  # If provided, exclude these indices from neighbors
):
    """Build input features for refinement model.
    
    For each query, finds K nearest training images (by predicted position)
    and constructs a feature vector encoding neighbor information.
    
    Output features per query:
        - base prediction (3d position, normalized)
        - for each of K neighbors:
            - position offset from prediction (3d)
            - feature similarity (1d)
            - distance from prediction (1d)
        Total: 3 + K * 5 = 53d (for K=10)
    """
    N = base_trans_pred.shape[0]
    tree = cKDTree(train_trans_norm.numpy())
    
    # Normalize query features
    query_norm = F.normalize(query_summary, dim=-1)
    train_norm = F.normalize(train_summary, dim=-1)
    
    # Find neighbors
    k_query = K + 1 if exclude_self_idx is not None else K
    dists, nn_indices = tree.query(base_trans_pred.numpy(), k=k_query)
    
    features_list = []
    for i in range(N):
        feat_parts = [base_trans_pred[i]]  # 3d
        
        neighbors = nn_indices[i].tolist()
        neighbor_dists = dists[i].tolist()
        
        # Exclude self if needed (for CV training)
        if exclude_self_idx is not None:
            self_idx = exclude_self_idx[i]
            filtered = [(n, d) for n, d in zip(neighbors, neighbor_dists) if n != self_idx]
            neighbors = [n for n, d in filtered[:K]]
            neighbor_dists = [d for n, d in filtered[:K]]
        else:
            neighbors = neighbors[:K]
            neighbor_dists = neighbor_dists[:K]
        
        # Pad if not enough neighbors
        while len(neighbors) < K:
            neighbors.append(neighbors[-1])
            neighbor_dists.append(neighbor_dists[-1])
        
        for j in range(K):
            n_idx = neighbors[j]
            n_dist = neighbor_dists[j]
            
            # Position offset
            offset = train_trans_norm[n_idx] - base_trans_pred[i]
            feat_parts.append(offset)  # 3d
            
            # Feature similarity
            sim = (query_norm[i] @ train_norm[n_idx]).unsqueeze(0)
            feat_parts.append(sim)  # 1d
            
            # Distance
            feat_parts.append(torch.tensor([n_dist], dtype=torch.float32))  # 1d
        
        features_list.append(torch.cat(feat_parts))
    
    return torch.stack(features_list)  # (N, 3 + K*5)


def train_refinement_with_cv(
    train_trans, train_rot, train_summary,
    feature_dir, dataset_dir, gpu=0, K=10, n_folds=5,
    epochs=3000, lr=1e-3, hidden_dims=(256, 128),
):
    """Train refinement model using K-fold cross-validation.
    
    1. Split training set into folds
    2. For each fold, train base model on other folds, predict this fold
    3. Build refinement features from OOF predictions
    4. Train refinement MLP on all training samples
    """
    from feature_retrieval.patch_regressor_v7 import (
        PatchPoseRegressor, PatchPoseDataset, precompute_pooled_features
    )
    
    device = torch.device(f'cuda:{gpu}')
    N_train = train_trans.shape[0]
    
    # Normalize translations
    trans_mean = train_trans.mean(dim=0)
    trans_std = train_trans.std(dim=0).clamp(min=1e-6)
    train_trans_norm = (train_trans - trans_mean) / trans_std
    
    print(f"\n{'='*60}")
    print(f"  Training Refinement Model with {n_folds}-Fold CV")
    print(f"{'='*60}")
    
    # Create fold indices
    indices = torch.randperm(N_train)
    fold_size = N_train // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else N_train
        folds.append(indices[start:end])
    
    # Get OOF predictions for each fold
    oof_trans_pred = torch.zeros(N_train, 3)
    oof_rot_pred = torch.zeros(N_train, 3, 3)
    
    for fold_idx in range(n_folds):
        print(f"\n  Fold {fold_idx+1}/{n_folds}:")
        val_indices = folds[fold_idx]
        train_indices = torch.cat([folds[j] for j in range(n_folds) if j != fold_idx])
        
        # Load full features for base model training
        # (This is simplified — in practice, would need to create subset datasets)
        # For now, use a simpler approach: train on subset, predict on holdout
        print(f"    Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
        
        # Quick base model training on this fold
        model = PatchPoseRegressor(
            pool_type='spp', feat_mode='both+sum', patch_dim=128,
            hidden_dims=(2048, 1024, 512), dropout=0.15,
        ).to(device)
        
        # Load features
        test_data = PatchPoseDataset(
            feature_dir, dataset_dir, 'train', 'cpu',
            use_fine=True, use_coarse=True, use_summary=True)
        
        # Precompute pooled features
        all_pooled = precompute_pooled_features(model, test_data, device, batch_size=16)
        
        # Get subset pooled features
        fold_train_pooled = all_pooled[train_indices]
        fold_val_pooled = all_pooled[val_indices]
        fold_train_trans_norm = train_trans_norm[train_indices].to(device)
        fold_train_rot = train_rot[train_indices].to(device)
        fold_val_trans_norm = train_trans_norm[val_indices].to(device)
        
        # Train base model on this fold
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-6)
        
        model.train()
        for ep in range(1, 3001):
            trans_pred_norm, rot_pred = model.forward_from_pooled(fold_train_pooled)
            loss_t = F.smooth_l1_loss(trans_pred_norm, fold_train_trans_norm)
            loss_r = geodesic_distance(rot_pred, fold_train_rot).mean()
            beta = torch.exp(model.s_rot)
            loss = loss_t + beta * loss_r
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # Predict on holdout
        model.eval()
        with torch.no_grad():
            val_t_pred, val_r_pred = model.forward_from_pooled(fold_val_pooled)
            val_t_real = val_t_pred * trans_std.to(device) + trans_mean.to(device)
            oof_trans_pred[val_indices] = val_t_real.cpu()
            oof_rot_pred[val_indices] = val_r_pred.cpu()
        
        # Evaluate OOF for this fold
        val_gt_trans = train_trans[val_indices]
        val_gt_rot = train_rot[val_indices]
        trans_errs = (oof_trans_pred[val_indices] - val_gt_trans).norm(dim=-1)
        rot_errs = geodesic_distance(oof_rot_pred[val_indices], val_gt_rot) * 180 / math.pi
        r10_2 = ((rot_errs < 10) & (trans_errs < 2.0)).float().mean() * 100
        print(f"    OOF R@10/2m: {r10_2:.1f}%  trans_med: {trans_errs.median()*1000:.0f}mm")
        
        del model, optimizer, all_pooled, fold_train_pooled, fold_val_pooled
        torch.cuda.empty_cache()
    
    print(f"\n  Overall OOF predictions computed for {N_train} training images")
    
    # Now build refinement features
    oof_trans_norm = (oof_trans_pred - trans_mean) / trans_std
    refine_features = build_refinement_features(
        oof_trans_norm, train_trans_norm, train_rot,
        train_summary, train_summary, K=K,
        exclude_self_idx=list(range(N_train))
    )
    
    # Refinement target: GT - base prediction (normalized)
    target_correction = train_trans_norm - oof_trans_norm
    target_rot = train_rot  # full rotation target
    
    print(f"  Refinement features: {refine_features.shape}")
    print(f"  Correction stats: mean={target_correction.mean(dim=0).numpy()}, "
          f"std={target_correction.std(dim=0).numpy()}")
    
    # Train refinement MLP
    input_dim = refine_features.shape[1]
    refine_model = RefinementMLP(input_dim, hidden_dims=hidden_dims).to(device)
    n_params = sum(p.numel() for p in refine_model.parameters())
    print(f"  Refinement model: input={input_dim}d, params={n_params:,}")
    
    refine_features = refine_features.to(device)
    target_correction = target_correction.to(device)
    target_rot = target_rot.to(device)
    
    optimizer = torch.optim.AdamW(refine_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_loss = float('inf')
    for ep in range(1, epochs + 1):
        refine_model.train()
        correction_pred, rot_6d = refine_model(refine_features)
        loss = F.smooth_l1_loss(correction_pred, target_correction)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if ep % 500 == 0 or ep == 1:
            print(f"  [Refinement {ep}/{epochs}] loss={loss.item():.6f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in refine_model.state_dict().items()}
    
    refine_model.load_state_dict(best_state)
    
    return refine_model, trans_mean, trans_std, oof_trans_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_exp_dir', type=str, required=True,
                        help='Directory of base model experiment')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feature_dir', type=str,
                        default='output/feature_extract/features_radio_dual_128/OldHospital_pilot')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='voting',
                        choices=['voting', 'learned', 'both'],
                        help='Refinement mode: voting (no training) or learned (K-fold CV)')
    parser.add_argument('--K', type=int, default=10, help='Number of neighbors')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.base_exp_dir + '_refined'
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load ground truth
    train_trans, train_rot, train_names = load_poses(args.dataset_dir, 'train')
    test_trans, test_rot, test_names = load_poses(args.dataset_dir, 'test')
    
    # Load summaries for feature similarity
    import glob as glob_module
    summary_matrix = torch.load(
        os.path.join(args.feature_dir, 'summary_matrix.pt'), map_location='cpu').float()
    image_paths = sorted(glob_module.glob(os.path.join(args.dataset_dir, 'seq*/*.png')))
    name_to_idx = {os.path.relpath(p, args.dataset_dir): i for i, p in enumerate(image_paths)}
    
    def get_summary(names, all_summary):
        indices = [name_to_idx[n] for n in names]
        return all_summary[indices]
    
    train_summary = get_summary(train_names, summary_matrix)
    test_summary = get_summary(test_names, summary_matrix)
    
    # Load base model predictions
    print("Loading base model predictions...")
    base_trans_pred, base_rot_pred = load_base_predictions(
        args.base_exp_dir, args.dataset_dir, args.feature_dir, args.gpu)
    
    # Evaluate base predictions
    print("\n" + "="*60)
    print("BASE MODEL (before refinement)")
    evaluate_predictions(base_trans_pred, base_rot_pred, test_trans, test_rot,
                        label="Base — Direct")
    evaluate_retrieval(base_trans_pred, train_trans, train_rot, test_trans, test_rot,
                      label="Base — Retrieval")
    
    # ================================================================
    # Method 1: Neighbor Voting (no training)
    # ================================================================
    if args.mode in ('voting', 'both'):
        print("\n" + "="*60)
        print("NEIGHBOR VOTING REFINEMENT")
        print("="*60)
        
        configs = [
            # (K, alpha, weight_by_sim, weight_by_dist, label)
            (3,  0.3, True,  True,  "K=3, α=0.3, sim+dist"),
            (3,  0.5, True,  True,  "K=3, α=0.5, sim+dist"),
            (5,  0.3, True,  True,  "K=5, α=0.3, sim+dist"),
            (5,  0.5, True,  True,  "K=5, α=0.5, sim+dist"),
            (5,  0.7, True,  True,  "K=5, α=0.7, sim+dist"),
            (10, 0.3, True,  True,  "K=10, α=0.3, sim+dist"),
            (10, 0.5, True,  True,  "K=10, α=0.5, sim+dist"),
            (5,  0.5, True,  False, "K=5, α=0.5, sim only"),
            (5,  0.5, False, True,  "K=5, α=0.5, dist only"),
            (5,  0.5, False, False, "K=5, α=0.5, uniform"),
            (1,  1.0, False, False, "K=1, α=1.0 (pure NN)"),
            (1,  0.5, True,  True,  "K=1, α=0.5, sim+dist"),
        ]
        
        best_r10 = 0
        best_label = ""
        
        for K, alpha, w_sim, w_dist, label in configs:
            ref_trans, ref_rot = neighbor_voting_refinement(
                base_trans_pred, base_rot_pred,
                train_trans, train_rot,
                test_summary, train_summary,
                K=K, alpha=alpha,
                weight_by_sim=w_sim, weight_by_dist=w_dist,
            )
            
            results, _, _ = evaluate_predictions(
                ref_trans, ref_rot, test_trans, test_rot,
                label=f"Voting: {label}")
            
            retr_results, _, _ = evaluate_retrieval(
                ref_trans, train_trans, train_rot, test_trans, test_rot,
                label=f"Voting Retrieval: {label}")
            
            r10 = results.get('10deg_2m', 0)
            if r10 > best_r10:
                best_r10 = r10
                best_label = label
        
        print(f"\n{'='*60}")
        print(f"  BEST VOTING CONFIG: {best_label} — R@10/2m = {best_r10:.1f}%")
        print(f"{'='*60}")
    
    # ================================================================
    # Method 2: Learned Refinement
    # ================================================================
    if args.mode in ('learned', 'both'):
        print("\n" + "="*60)
        print("LEARNED REFINEMENT (5-fold CV)")
        print("="*60)
        
        refine_model, trans_mean, trans_std, oof_preds = train_refinement_with_cv(
            train_trans, train_rot, train_summary,
            args.feature_dir, args.dataset_dir,
            gpu=args.gpu, K=args.K, n_folds=5,
        )
        
        # Apply to test set
        trans_mean_dev = trans_mean.to(device)
        trans_std_dev = trans_std.to(device)
        
        base_trans_norm = (base_trans_pred - trans_mean) / trans_std
        train_trans_norm = (train_trans - trans_mean) / trans_std
        
        test_refine_features = build_refinement_features(
            base_trans_norm, train_trans_norm, train_rot,
            test_summary, train_summary, K=args.K,
        ).to(device)
        
        refine_model.eval()
        with torch.no_grad():
            correction, _ = refine_model(test_refine_features)
            corrected_trans_norm = base_trans_norm.to(device) + correction
            corrected_trans = corrected_trans_norm * trans_std_dev + trans_mean_dev
        
        evaluate_predictions(
            corrected_trans.cpu(), base_rot_pred, test_trans, test_rot,
            label="Learned Refinement — Direct")
        evaluate_retrieval(
            corrected_trans.cpu(), train_trans, train_rot, test_trans, test_rot,
            label="Learned Refinement — Retrieval")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
