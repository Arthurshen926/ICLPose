#!/usr/bin/env python3
"""
k-NN and Non-Parametric Pose Regression
========================================
Instead of training an MLP to map features → pose, use k-NN:
  1. Pool features (SPP) → 7936d vector per image
  2. For each test image, find k nearest training images in feature space
  3. Average (or distance-weighted average) their poses

Key advantages:
  - ZERO trainable parameters → no overfitting
  - Can try many k values and distance metrics instantly
  - Can combine with dimensionality reduction (PCA, UMAP)
  
Also includes:
  - Random Forest regression
  - Kernel regression (Nadaraya-Watson)
  - Feature PCA + k-NN
"""

import argparse
import json
import math
import os
import glob as glob_module
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


# ============================================================
# Rotation utilities (from patch_regressor_v7.py)
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


def matrix_to_quaternion(R):
    """Convert rotation matrices to quaternions (batch)."""
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    s = torch.sqrt(trace.clamp(min=0) + 1.0) * 2
    mask = trace > 0
    q[mask, 0] = 0.25 * s[mask]
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]
    
    # Case 2: R[0,0] is max
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-8) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]
    
    # Case 3: R[1,1] is max
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-8) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]
    
    # Case 4: R[2,2] is max
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-8) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]
    
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return q.reshape(*batch_shape, 4)


def geodesic_distance(R1, R2):
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).reshape(R1.shape[:-2])


def average_quaternions(quats, weights=None):
    """Average quaternions using weighted sum (for small rotational differences).
    quats: (k, 4), weights: (k,) or None
    Returns: (4,)
    """
    if weights is None:
        weights = np.ones(len(quats)) / len(quats)
    else:
        weights = weights / weights.sum()
    
    # Ensure consistent quaternion signs (flip if dot product is negative)
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    
    avg = (quats.T @ weights)
    avg = avg / np.linalg.norm(avg)
    return avg


# ============================================================
# Data Loading (simplified from patch_regressor_v7.py)
# ============================================================

class SPPPooling(nn.Module):
    def __init__(self, C, levels=(1, 2, 4)):
        super().__init__()
        self.levels = levels
        self.output_dim = C * sum(l * l for l in self.levels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        pooled = []
        for level in self.levels:
            out = F.adaptive_avg_pool2d(x, (level, level))
            pooled.append(out.reshape(B, -1))
        return torch.cat(pooled, dim=-1)


class GAPPooling(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.output_dim = C
    
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)


def load_data(feature_dir, dataset_dir, split):
    """Load features and poses for a split."""
    # Load patch features
    fine = torch.load(os.path.join(feature_dir, f'fine_geo_{split}.pt'), map_location='cpu').float()
    coarse = torch.load(os.path.join(feature_dir, f'coarse_sem_{split}.pt'), map_location='cpu').float()
    
    # Load summary
    summary_matrix = torch.load(os.path.join(feature_dir, 'summary_matrix.pt'), map_location='cpu').float()
    image_paths = sorted(glob_module.glob(os.path.join(dataset_dir, 'seq*/*.png')))
    name_to_idx = {os.path.relpath(p, dataset_dir): i for i, p in enumerate(image_paths)}
    
    split_file = os.path.join(dataset_dir, f'dataset_{split}.txt')
    with open(split_file) as f:
        lines = f.readlines()
    
    indices = []
    translations_list = []
    rotations_list = []
    names_list = []
    
    for line in lines[3:]:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        img_name = parts[0]
        idx = name_to_idx.get(img_name)
        if idx is not None:
            indices.append(idx)
        
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        
        translations_list.append([x, y, z])
        rotations_list.append([w, p, q, r])
        names_list.append(img_name)
    
    summary = summary_matrix[indices]
    translations = np.array(translations_list, dtype=np.float32)
    quaternions = np.array(rotations_list, dtype=np.float32)
    rotations = quaternion_to_matrix(torch.tensor(quaternions)).numpy()
    
    return {
        'fine': fine,
        'coarse': coarse,
        'summary': summary,
        'translations': translations,
        'quaternions': quaternions,
        'rotations': rotations,
        'names': names_list,
    }


def pool_features(fine, coarse, summary, pool_type='spp', levels=(1, 2, 4)):
    """Pool patch features and concatenate with summary."""
    C = fine.shape[1]  # 128
    
    if pool_type == 'spp':
        pooler = SPPPooling(C, levels)
    elif pool_type == 'gap':
        pooler = GAPPooling(C)
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")
    
    with torch.no_grad():
        fine_pooled = pooler(fine)
        coarse_pooled = pooler(coarse)
    
    # Concatenate: pooled_fine + pooled_coarse + summary
    features = torch.cat([fine_pooled, coarse_pooled, summary], dim=-1)
    return features.numpy()


# ============================================================
# Evaluation
# ============================================================

def evaluate_predictions(pred_trans, pred_rot, gt_trans, gt_rot, train_trans, train_rot, names=None):
    """Evaluate predicted poses against ground truth."""
    
    # Translation errors (meters)
    trans_errors = np.linalg.norm(pred_trans - gt_trans, axis=-1)
    
    # Rotation errors (degrees)
    pred_rot_t = torch.tensor(pred_rot, dtype=torch.float32)
    gt_rot_t = torch.tensor(gt_rot, dtype=torch.float32)
    rot_errors = (geodesic_distance(pred_rot_t, gt_rot_t) * 180 / math.pi).numpy()
    
    results = {
        'rotation_deg': {
            'median': float(np.median(rot_errors)),
            'mean': float(np.mean(rot_errors)),
        },
        'translation_m': {
            'median': float(np.median(trans_errors)),
            'mean': float(np.mean(trans_errors)),
        },
        'translation_mm': {
            'median': float(np.median(trans_errors) * 1000),
            'mean': float(np.mean(trans_errors) * 1000),
        },
        'recall': {},
        'individual_pass': {},
    }
    
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
        ('25deg_5m', 25, 5.0),
    ]
    
    for name, rot_th, trans_th in thresholds:
        rot_pass = (rot_errors < rot_th).mean() * 100
        trans_pass = (trans_errors < trans_th).mean() * 100
        combined = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100
        results['recall'][name] = float(combined)
        results['individual_pass'][name] = {
            'rot_pass': float(rot_pass),
            'trans_pass': float(trans_pass),
            'combined': float(combined),
        }
    
    # Retrieval evaluation
    if train_trans is not None:
        tree = cKDTree(train_trans)
        dists, nn_indices = tree.query(pred_trans, k=1)
        
        retr_trans_errors = np.linalg.norm(train_trans[nn_indices] - gt_trans, axis=-1)
        retr_rot_pred = torch.tensor(train_rot[nn_indices], dtype=torch.float32)
        retr_rot_errors = (geodesic_distance(retr_rot_pred, gt_rot_t) * 180 / math.pi).numpy()
        
        results['retrieval'] = {}
        for name, rot_th, trans_th in thresholds:
            rot_pass = (retr_rot_errors < rot_th).mean() * 100
            trans_pass = (retr_trans_errors < trans_th).mean() * 100
            combined = ((retr_rot_errors < rot_th) & (retr_trans_errors < trans_th)).mean() * 100
            results['retrieval'][name] = {
                'combined': float(combined),
                'rot_pass': float(rot_pass),
                'trans_pass': float(trans_pass),
            }
        results['retrieval']['median_rot_deg'] = float(np.median(retr_rot_errors))
        results['retrieval']['median_trans_mm'] = float(np.median(retr_trans_errors) * 1000)
    
    return results, trans_errors, rot_errors


def print_results(results, method_name):
    """Print results in a standardized format."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {method_name}")
    print(f"{'='*70}")
    print(f"Rotation  (median): {results['rotation_deg']['median']:.2f}deg")
    print(f"Translation (median): {results['translation_mm']['median']:.0f}mm")
    print()
    for name in ['5deg_1m', '10deg_2m', '15deg_5m', '25deg_5m']:
        r = results['individual_pass'][name]
        print(f"  R@{name}: {results['recall'][name]:.1f}%  "
              f"(rot_pass={r['rot_pass']:.1f}%, trans_pass={r['trans_pass']:.1f}%)")
    
    if 'retrieval' in results:
        print(f"\nRetrieval median: {results['retrieval']['median_rot_deg']:.2f}deg / "
              f"{results['retrieval']['median_trans_mm']:.0f}mm")
        for name in ['5deg_1m', '10deg_2m', '15deg_5m', '25deg_5m']:
            r = results['retrieval'][name]
            print(f"  R@{name}: {r['combined']:.1f}%  "
                  f"(rot={r['rot_pass']:.1f}%, trans={r['trans_pass']:.1f}%)")
    print(f"{'='*70}")


# ============================================================
# k-NN Regression
# ============================================================

def knn_regression(train_feats, train_trans, train_quats, train_rots,
                   test_feats, k=1, metric='cosine', sigma=None,
                   weight_mode='uniform'):
    """
    k-NN regression for pose prediction.
    
    Args:
        train_feats: (N_train, D) feature vectors
        train_trans: (N_train, 3) translations
        train_quats: (N_train, 4) quaternions 
        train_rots: (N_train, 3, 3) rotation matrices
        test_feats: (N_test, D) feature vectors
        k: number of neighbors
        metric: distance metric ('cosine', 'euclidean', 'l2_normalized')
        sigma: bandwidth for Gaussian weighting (None = auto)
        weight_mode: 'uniform', 'distance', 'gaussian'
    
    Returns:
        pred_trans: (N_test, 3)
        pred_rots: (N_test, 3, 3)
    """
    N_test = test_feats.shape[0]
    
    # Normalize features
    if metric == 'cosine' or metric == 'l2_normalized':
        train_norm = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-8)
        test_norm = test_feats / (np.linalg.norm(test_feats, axis=1, keepdims=True) + 1e-8)
        
        if metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            # For L2-normalized vectors: ||a-b||^2 = 2 - 2*cos(a,b)
            # So we can use L2 distance on normalized vectors
            dists = cdist(test_norm, train_norm, metric='euclidean')
        else:
            dists = cdist(test_norm, train_norm, metric='euclidean')
    elif metric == 'euclidean':
        dists = cdist(test_feats, train_feats, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Find k nearest neighbors
    nn_indices = np.argpartition(dists, k, axis=1)[:, :k]
    
    # Get actual distances for the k neighbors
    nn_dists = np.take_along_axis(dists, nn_indices, axis=1)
    
    # Compute weights
    if weight_mode == 'uniform':
        weights = np.ones_like(nn_dists) / k
    elif weight_mode == 'distance':
        # Inverse distance weighting
        weights = 1.0 / (nn_dists + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
    elif weight_mode == 'gaussian':
        if sigma is None:
            # Auto sigma: median of k-th nearest neighbor distances
            sigma = np.median(nn_dists[:, -1])
        weights = np.exp(-nn_dists**2 / (2 * sigma**2))
        weights = weights / weights.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown weight mode: {weight_mode}")
    
    # Predict translations (weighted average)
    pred_trans = np.zeros((N_test, 3), dtype=np.float32)
    for i in range(N_test):
        pred_trans[i] = (train_trans[nn_indices[i]] * weights[i, :, None]).sum(axis=0)
    
    # Predict rotations (weighted quaternion average)
    pred_rots = np.zeros((N_test, 3, 3), dtype=np.float32)
    for i in range(N_test):
        q_avg = average_quaternions(train_quats[nn_indices[i]].copy(), weights[i])
        pred_rots[i] = quaternion_to_matrix(torch.tensor(q_avg).unsqueeze(0)).numpy()[0]
    
    return pred_trans, pred_rots


# ============================================================
# Kernel Regression (Nadaraya-Watson)
# ============================================================

def kernel_regression(train_feats, train_trans, train_quats, train_rots,
                      test_feats, sigma='auto', metric='cosine'):
    """
    Nadaraya-Watson kernel regression: use ALL training samples,
    weighted by Gaussian kernel in feature space.
    """
    # Normalize
    if metric == 'cosine':
        train_norm = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-8)
        test_norm = test_feats / (np.linalg.norm(test_feats, axis=1, keepdims=True) + 1e-8)
        dists = cdist(test_norm, train_norm, metric='euclidean')
    else:
        dists = cdist(test_feats, train_feats, metric='euclidean')
    
    if sigma == 'auto':
        # Use median distance
        sigma = np.median(dists)
    
    N_test = test_feats.shape[0]
    
    # Gaussian weights
    weights = np.exp(-dists**2 / (2 * sigma**2))
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Predict translations
    pred_trans = weights @ train_trans
    
    # Predict rotations
    pred_rots = np.zeros((N_test, 3, 3), dtype=np.float32)
    for i in range(N_test):
        q_avg = average_quaternions(train_quats.copy(), weights[i])
        pred_rots[i] = quaternion_to_matrix(torch.tensor(q_avg).unsqueeze(0)).numpy()[0]
    
    return pred_trans, pred_rots


# ============================================================
# Feature PCA
# ============================================================

def apply_pca(train_feats, test_feats, n_components=256):
    """Apply PCA to reduce feature dimensionality."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_feats)
    test_pca = pca.transform(test_feats)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {train_feats.shape[1]}d → {n_components}d (explained variance: {explained:.1%})")
    return train_pca, test_pca


def apply_whitening(train_feats, test_feats, n_components=None):
    """Apply PCA whitening (decorrelation + unit variance)."""
    from sklearn.decomposition import PCA
    if n_components is None:
        n_components = min(train_feats.shape)
    pca = PCA(n_components=n_components, whiten=True)
    train_w = pca.fit_transform(train_feats)
    test_w = pca.transform(test_feats)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Whitening {train_feats.shape[1]}d → {n_components}d (explained variance: {explained:.1%})")
    return train_w, test_w


# ============================================================
# Random Forest / Gradient Boosting
# ============================================================

def random_forest_regression(train_feats, train_trans, train_quats, train_rots,
                             test_feats, n_estimators=500, max_depth=None,
                             n_components_pca=256):
    """Random Forest regression for pose prediction."""
    from sklearn.ensemble import RandomForestRegressor
    
    # PCA first to reduce dimensionality
    if n_components_pca and train_feats.shape[1] > n_components_pca:
        train_feats, test_feats = apply_pca(train_feats, test_feats, n_components_pca)
    
    # Predict translations
    print(f"  Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf_trans = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     n_jobs=-1, random_state=42)
    rf_trans.fit(train_feats, train_trans)
    pred_trans = rf_trans.predict(test_feats)
    
    # Predict quaternions
    rf_rot = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                    n_jobs=-1, random_state=42)
    rf_rot.fit(train_feats, train_quats)
    pred_quats = rf_rot.predict(test_feats)
    
    # Normalize quaternions and convert to rotation matrices
    pred_quats = pred_quats / (np.linalg.norm(pred_quats, axis=1, keepdims=True) + 1e-8)
    pred_rots = quaternion_to_matrix(torch.tensor(pred_quats, dtype=torch.float32)).numpy()
    
    return pred_trans.astype(np.float32), pred_rots


def gradient_boost_regression(train_feats, train_trans, train_quats, train_rots,
                               test_feats, n_estimators=500, max_depth=5, lr=0.1,
                               n_components_pca=256):
    """Gradient Boosting regression for pose prediction."""
    from sklearn.ensemble import GradientBoostingRegressor
    
    if n_components_pca and train_feats.shape[1] > n_components_pca:
        train_feats, test_feats = apply_pca(train_feats, test_feats, n_components_pca)
    
    pred_trans = np.zeros((test_feats.shape[0], 3), dtype=np.float32)
    pred_quats = np.zeros((test_feats.shape[0], 4), dtype=np.float32)
    
    # Predict each dimension separately
    print(f"  Training GBR (n_estimators={n_estimators}, max_depth={max_depth}, lr={lr})...")
    for dim in range(3):
        gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                         learning_rate=lr, random_state=42)
        gbr.fit(train_feats, train_trans[:, dim])
        pred_trans[:, dim] = gbr.predict(test_feats)
    
    for dim in range(4):
        gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                         learning_rate=lr, random_state=42)
        gbr.fit(train_feats, train_quats[:, dim])
        pred_quats[:, dim] = gbr.predict(test_feats)
    
    pred_quats = pred_quats / (np.linalg.norm(pred_quats, axis=1, keepdims=True) + 1e-8)
    pred_rots = quaternion_to_matrix(torch.tensor(pred_quats, dtype=torch.float32)).numpy()
    
    return pred_trans, pred_rots


# ============================================================
# MLP k-NN Hybrid: use MLP features as input to k-NN
# ============================================================

def load_mlp_features(model_dir, feature_dir, dataset_dir, device='cpu'):
    """Load a trained MLP model and extract intermediate features."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from feature_retrieval.patch_regressor_v7 import (
        PatchPoseDataset, PatchPoseRegressor, precompute_pooled_features, POOL_REGISTRY
    )
    
    config = json.load(open(os.path.join(model_dir, 'config.json')))
    
    use_fine = 'fine' in config['feat'] or 'both' in config['feat']
    use_coarse = 'coarse' in config['feat'] or 'both' in config['feat']
    use_summary = '+sum' in config['feat']
    
    train_data = PatchPoseDataset(feature_dir, dataset_dir, 'train', device,
                                   use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(feature_dir, dataset_dir, 'test', device,
                                  use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    
    # Build model
    patch_dim = config.get('patch_dim', 64)
    pool_cls = POOL_REGISTRY[config['pool']]
    spp_levels = config.get('spp_levels', None)
    
    pooler_fine = pool_cls(patch_dim, spp_levels=spp_levels) if use_fine else None
    pooler_coarse = pool_cls(patch_dim, spp_levels=spp_levels) if use_coarse else None
    
    total_dim = 0
    if pooler_fine:
        total_dim += pooler_fine.output_dim
    if pooler_coarse:
        total_dim += pooler_coarse.output_dim
    if use_summary:
        total_dim += 2560
    
    model = PatchPoseRegressor(
        total_dim=total_dim,
        hidden_dims=config['hidden_dims'],
        pool_type=config['pool'],
        patch_dim=patch_dim,
        use_fine=use_fine,
        use_coarse=use_coarse,
        use_summary=use_summary,
        dropout=config.get('dropout', 0.0),
        feature_dropout=config.get('feature_dropout', 0.0),
        attn_heads=config.get('attn_heads', 4),
        conv_out_dim=config.get('conv_out_dim', 512),
        spp_levels=spp_levels,
    )
    
    # Load weights
    state = torch.load(os.path.join(model_dir, 'model_best.pt'), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(torch.device(device))
    
    return model, train_data, test_data, config


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='k-NN and Non-Parametric Pose Regression')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/feature_retrieval/pose_regression/knn_results')
    parser.add_argument('--pool', type=str, default='spp', choices=['spp', 'gap'])
    parser.add_argument('--patch_dim', type=int, default=128)
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['knn', 'kernel', 'rf', 'gbr'],
                        help='Methods to run: knn, kernel, rf, gbr')
    parser.add_argument('--k_values', type=int, nargs='+', 
                        default=[1, 3, 5, 7, 10, 15, 20, 30, 50],
                        help='k values for k-NN')
    parser.add_argument('--pca_dims', type=int, nargs='+',
                        default=[0, 64, 128, 256, 512, 1024],
                        help='PCA dimensions to try (0 = no PCA)')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    train = load_data(args.feature_dir, args.dataset_dir, 'train')
    test = load_data(args.feature_dir, args.dataset_dir, 'test')
    
    print(f"  Train: {len(train['names'])} samples")
    print(f"  Test: {len(test['names'])} samples")
    
    print(f"\nPooling features ({args.pool})...")
    train_feats = pool_features(train['fine'], train['coarse'], train['summary'], 
                                pool_type=args.pool)
    test_feats = pool_features(test['fine'], test['coarse'], test['summary'],
                               pool_type=args.pool)
    print(f"  Feature dim: {train_feats.shape[1]}")
    
    all_results = {}
    best_r10 = 0
    best_method = None
    
    # ============================================================
    # k-NN Experiments
    # ============================================================
    if 'knn' in args.methods:
        print(f"\n{'='*70}")
        print("k-NN REGRESSION EXPERIMENTS")
        print(f"{'='*70}")
        
        for pca_dim in args.pca_dims:
            if pca_dim > 0 and pca_dim < train_feats.shape[1]:
                tf_train, tf_test = apply_pca(train_feats, test_feats, pca_dim)
                pca_label = f"_pca{pca_dim}"
            elif pca_dim == 0:
                tf_train, tf_test = train_feats, test_feats
                pca_label = ""
            else:
                continue
            
            for metric in ['cosine', 'euclidean']:
                for weight_mode in ['uniform', 'distance', 'gaussian']:
                    for k in args.k_values:
                        method_name = f"knn_k{k}_{metric}_{weight_mode}{pca_label}"
                        t0 = time.time()
                        
                        pred_trans, pred_rots = knn_regression(
                            tf_train, train['translations'], train['quaternions'], train['rotations'],
                            tf_test, k=k, metric=metric, weight_mode=weight_mode
                        )
                        
                        results, te, re = evaluate_predictions(
                            pred_trans, pred_rots,
                            test['translations'], test['rotations'],
                            train['translations'], train['rotations']
                        )
                        
                        r10 = results['recall']['10deg_2m']
                        r5 = results['recall']['5deg_1m']
                        elapsed = time.time() - t0
                        
                        print(f"  {method_name}: R@10°/2m={r10:.1f}%, R@5°/1m={r5:.1f}%  ({elapsed:.1f}s)")
                        
                        all_results[method_name] = results
                        
                        if r10 > best_r10:
                            best_r10 = r10
                            best_method = method_name
        
        # Also try whitened features
        print("\n  --- With Feature Whitening ---")
        for whiten_dim in [128, 256, 512]:
            if whiten_dim > min(train_feats.shape):
                continue
            tf_train_w, tf_test_w = apply_whitening(train_feats, test_feats, whiten_dim)
            
            for metric in ['cosine', 'euclidean']:
                for weight_mode in ['uniform', 'distance']:
                    for k in args.k_values:
                        method_name = f"knn_k{k}_{metric}_{weight_mode}_whiten{whiten_dim}"
                        
                        pred_trans, pred_rots = knn_regression(
                            tf_train_w, train['translations'], train['quaternions'], train['rotations'],
                            tf_test_w, k=k, metric=metric, weight_mode=weight_mode
                        )
                        
                        results, _, _ = evaluate_predictions(
                            pred_trans, pred_rots,
                            test['translations'], test['rotations'],
                            train['translations'], train['rotations']
                        )
                        
                        r10 = results['recall']['10deg_2m']
                        r5 = results['recall']['5deg_1m']
                        print(f"  {method_name}: R@10°/2m={r10:.1f}%, R@5°/1m={r5:.1f}%")
                        
                        all_results[method_name] = results
                        if r10 > best_r10:
                            best_r10 = r10
                            best_method = method_name
    
    # ============================================================
    # Kernel Regression
    # ============================================================
    if 'kernel' in args.methods:
        print(f"\n{'='*70}")
        print("KERNEL REGRESSION (Nadaraya-Watson)")
        print(f"{'='*70}")
        
        for pca_dim in [0, 128, 256, 512]:
            if pca_dim > 0 and pca_dim < train_feats.shape[1]:
                tf_train, tf_test = apply_pca(train_feats, test_feats, pca_dim)
                pca_label = f"_pca{pca_dim}"
            elif pca_dim == 0:
                tf_train, tf_test = train_feats, test_feats
                pca_label = ""
            else:
                continue
            
            for metric in ['cosine', 'euclidean']:
                for sigma_mult in [0.1, 0.25, 0.5, 1.0, 2.0]:
                    method_name = f"kernel_{metric}_sigma{sigma_mult}{pca_label}"
                    
                    # Auto sigma * multiplier
                    if metric == 'cosine':
                        tf_n = tf_train / (np.linalg.norm(tf_train, axis=1, keepdims=True) + 1e-8)
                        base_sigma = np.median(cdist(tf_n[:50], tf_n, metric='euclidean'))
                    else:
                        base_sigma = np.median(cdist(tf_train[:50], tf_train, metric='euclidean'))
                    sigma = base_sigma * sigma_mult
                    
                    pred_trans, pred_rots = kernel_regression(
                        tf_train, train['translations'], train['quaternions'], train['rotations'],
                        tf_test, sigma=sigma, metric=metric
                    )
                    
                    results, _, _ = evaluate_predictions(
                        pred_trans, pred_rots,
                        test['translations'], test['rotations'],
                        train['translations'], train['rotations']
                    )
                    
                    r10 = results['recall']['10deg_2m']
                    r5 = results['recall']['5deg_1m']
                    print(f"  {method_name}: R@10°/2m={r10:.1f}%, R@5°/1m={r5:.1f}%")
                    
                    all_results[method_name] = results
                    if r10 > best_r10:
                        best_r10 = r10
                        best_method = method_name
    
    # ============================================================
    # Random Forest
    # ============================================================
    if 'rf' in args.methods:
        print(f"\n{'='*70}")
        print("RANDOM FOREST REGRESSION")
        print(f"{'='*70}")
        
        for pca_dim in [128, 256, 512]:
            for n_est in [200, 500, 1000]:
                for max_depth in [None, 10, 20]:
                    method_name = f"rf_n{n_est}_d{max_depth}_pca{pca_dim}"
                    t0 = time.time()
                    
                    pred_trans, pred_rots = random_forest_regression(
                        train_feats, train['translations'], train['quaternions'], train['rotations'],
                        test_feats, n_estimators=n_est, max_depth=max_depth,
                        n_components_pca=pca_dim
                    )
                    
                    results, _, _ = evaluate_predictions(
                        pred_trans, pred_rots,
                        test['translations'], test['rotations'],
                        train['translations'], train['rotations']
                    )
                    
                    r10 = results['recall']['10deg_2m']
                    r5 = results['recall']['5deg_1m']
                    elapsed = time.time() - t0
                    print(f"  {method_name}: R@10°/2m={r10:.1f}%, R@5°/1m={r5:.1f}%  ({elapsed:.1f}s)")
                    
                    all_results[method_name] = results
                    if r10 > best_r10:
                        best_r10 = r10
                        best_method = method_name
    
    # ============================================================
    # Gradient Boosting
    # ============================================================
    if 'gbr' in args.methods:
        print(f"\n{'='*70}")
        print("GRADIENT BOOSTING REGRESSION")
        print(f"{'='*70}")
        
        for pca_dim in [128, 256]:
            for n_est in [500, 1000]:
                for max_depth in [3, 5, 8]:
                    for lr in [0.05, 0.1]:
                        method_name = f"gbr_n{n_est}_d{max_depth}_lr{lr}_pca{pca_dim}"
                        t0 = time.time()
                        
                        pred_trans, pred_rots = gradient_boost_regression(
                            train_feats, train['translations'], train['quaternions'], train['rotations'],
                            test_feats, n_estimators=n_est, max_depth=max_depth, lr=lr,
                            n_components_pca=pca_dim
                        )
                        
                        results, _, _ = evaluate_predictions(
                            pred_trans, pred_rots,
                            test['translations'], test['rotations'],
                            train['translations'], train['rotations']
                        )
                        
                        r10 = results['recall']['10deg_2m']
                        r5 = results['recall']['5deg_1m']
                        elapsed = time.time() - t0
                        print(f"  {method_name}: R@10°/2m={r10:.1f}%, R@5°/1m={r5:.1f}%  ({elapsed:.1f}s)")
                        
                        all_results[method_name] = results
                        if r10 > best_r10:
                            best_r10 = r10
                            best_method = method_name
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY — All Methods Ranked by R@10°/2m")
    print(f"{'='*70}")
    
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['recall']['10deg_2m'], reverse=True)
    
    for i, (name, res) in enumerate(sorted_results[:30]):
        r10 = res['recall']['10deg_2m']
        r5 = res['recall']['5deg_1m']
        rot_med = res['rotation_deg']['median']
        trans_med = res['translation_mm']['median']
        print(f"  {i+1:2d}. {name:50s}  R@10/2m={r10:.1f}%  R@5/1m={r5:.1f}%  "
              f"rot={rot_med:.1f}°  trans={trans_med:.0f}mm")
    
    if best_method:
        print(f"\n  BEST: {best_method} → R@10°/2m = {best_r10:.1f}%")
        print_results(all_results[best_method], best_method)
    
    # Save all results
    results_file = out_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
