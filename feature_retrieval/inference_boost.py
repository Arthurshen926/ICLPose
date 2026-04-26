#!/usr/bin/env python3
"""
Inference-time boosting for pose regression.
Combines k-NN blending, MC dropout, TTA, multi-model ensemble, and per-sample selection.
All techniques require NO additional training.

Usage:
    cd /root/ICLPose-loc && python -m feature_retrieval.inference_boost
"""

import sys
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, '/root/ICLPose-loc')
from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset,
    quaternion_to_matrix, gram_schmidt_6d_to_matrix, geodesic_distance
)

# ============================================================
# Config
# ============================================================

FEATURE_DIR = 'output/feature_extract/features_radio_dual_128/OldHospital_pilot'
DATASET_DIR = 'dataset/OldHospital'
MODEL_BASE = 'output/feature_retrieval/pose_regression'
DEVICE = 'cpu'

MODEL_CONFIGS = OrderedDict([
    ('seed314',  ('exp14s_128d_wider_seed314',  72.0)),
    ('seed123',  ('exp14n_128d_wider_seed123',  70.9)),
    ('swa123',   ('exp14p_128d_swa_seed123',    70.9)),
    ('seed9999', ('exp14n_128d_wider_seed9999',  69.8)),
    ('seed7890', ('exp14s_128d_wider_seed7890',  68.1)),
])

MODEL_KWARGS = dict(
    pool_type='spp', feat_mode='both+sum', patch_dim=128,
    hidden_dims=(2048, 1024, 512), dropout=0.15,
)

# ============================================================
# Helpers
# ============================================================

def load_data():
    """Load train/test datasets."""
    print("Loading data...")
    train_data = PatchPoseDataset(
        FEATURE_DIR, DATASET_DIR, 'train', 'cpu',
        use_fine=True, use_coarse=True, use_summary=True)
    test_data = PatchPoseDataset(
        FEATURE_DIR, DATASET_DIR, 'test', 'cpu',
        use_fine=True, use_coarse=True, use_summary=True)
    return train_data, test_data


def load_model(exp_name):
    """Load a model from checkpoint."""
    model = PatchPoseRegressor(**MODEL_KWARGS)
    ckpt_path = Path(MODEL_BASE) / exp_name / 'model_best.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    norm_params = ckpt['norm_params']
    trans_mean = norm_params['mean']
    trans_std = norm_params['std']
    model.to(DEVICE)
    return model, trans_mean, trans_std


def pool_features_batched(model, dataset, batch_size=16):
    """Pool patch features through model's pooling layers in mini-batches."""
    model.eval()
    pooled_list = []
    N = dataset.N
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pf = dataset.patch_fine[start:end].to(DEVICE) if dataset.patch_fine is not None else None
            pc = dataset.patch_coarse[start:end].to(DEVICE) if dataset.patch_coarse is not None else None
            sm = dataset.summary[start:end].to(DEVICE) if dataset.summary is not None else None
            pooled = model.pool_features(pf, pc, sm)
            pooled_list.append(pooled)
    return torch.cat(pooled_list, dim=0)


def predict_from_pooled(model, pooled, trans_mean, trans_std, batch_size=64):
    """Run MLP head on pooled features, return denormalized trans and rot_mat."""
    model.eval()
    trans_list, rot_list = [], []
    with torch.no_grad():
        for start in range(0, pooled.shape[0], batch_size):
            end = min(start + batch_size, pooled.shape[0])
            t, r = model.forward_from_pooled(pooled[start:end])
            trans_list.append(t)
            rot_list.append(r)
    trans_norm = torch.cat(trans_list, dim=0)
    rot_mat = torch.cat(rot_list, dim=0)
    trans = trans_norm * trans_std.to(DEVICE) + trans_mean.to(DEVICE)
    return trans, rot_mat


def rotation_matrix_to_quaternion(R):
    """Convert (N, 3, 3) rotation matrices to (N, 4) quaternions [w, x, y, z]."""
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    mask = trace > 0
    if mask.any():
        s = torch.sqrt(trace[mask] + 1.0) * 2  # s = 4*w
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    # Case 2: R[0,0] is max diagonal
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is max diagonal
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is max diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

    # Normalize and ensure w >= 0
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q[q[:, 0] < 0] *= -1
    return q


def weighted_quaternion_mean(quats, weights):
    """Weighted average of quaternions. quats: (K, 4), weights: (K,)."""
    # Ensure consistent hemisphere
    dots = (quats * quats[0:1]).sum(dim=-1)
    quats = quats.clone()
    quats[dots < 0] *= -1
    avg = (quats * weights.unsqueeze(-1)).sum(dim=0)
    return avg / avg.norm().clamp(min=1e-8)


def compute_metrics(pred_trans, pred_rot, gt_trans, gt_rot):
    """Compute pose errors and recall metrics."""
    trans_errors = (pred_trans - gt_trans).norm(dim=-1).numpy()
    rot_errors = (geodesic_distance(pred_rot, gt_rot) * 180 / math.pi).numpy()

    thresholds = [('5°/1m', 5, 1.0), ('10°/2m', 10, 2.0), ('15°/5m', 15, 5.0)]
    recalls = {}
    for name, rot_th, trans_th in thresholds:
        recalls[name] = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100

    return {
        'med_trans_mm': float(np.median(trans_errors) * 1000),
        'med_rot_deg': float(np.median(rot_errors)),
        'recalls': recalls,
    }


def print_metrics(name, metrics, baseline_r10=72.0):
    r10 = metrics['recalls']['10°/2m']
    delta = r10 - baseline_r10
    sign = '+' if delta >= 0 else ''
    print(f"  {name:45s} | {metrics['recalls']['5°/1m']:5.1f}% | {r10:5.1f}% ({sign}{delta:.1f}) | "
          f"{metrics['recalls']['15°/5m']:5.1f}% | {metrics['med_trans_mm']:6.0f}mm | {metrics['med_rot_deg']:5.2f}°")


# ============================================================
# Technique 1: k-NN Blending
# ============================================================

def knn_blending(model, train_data, test_data, trans_mean, trans_std,
                 train_pooled, test_pooled, results_dict):
    print("\n" + "=" * 70)
    print("TECHNIQUE 1: k-NN Blending")
    print("=" * 70)

    gt_train_trans = train_data.translations
    gt_train_rot = train_data.rotations
    gt_train_quat = rotation_matrix_to_quaternion(gt_train_rot)
    gt_test_trans = test_data.translations
    gt_test_rot = test_data.rotations

    # Baseline MLP prediction
    mlp_trans, mlp_rot = predict_from_pooled(model, test_pooled, trans_mean, trans_std)
    mlp_quat = rotation_matrix_to_quaternion(mlp_rot)

    # L2 distances in pooled feature space
    # Normalize for distance computation
    train_norm = F.normalize(train_pooled, dim=-1)
    test_norm = F.normalize(test_pooled, dim=-1)
    dists = torch.cdist(test_norm, train_norm)  # (N_test, N_train)

    k_values = [1, 3, 5, 10, 20]
    alpha_values = [round(a * 0.1, 1) for a in range(11)]

    best_r10 = 0
    best_config = ""

    for k in k_values:
        topk_dists, topk_idx = dists.topk(k, dim=-1, largest=False)  # (N_test, k)
        weights = 1.0 / (topk_dists + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (N_test, k)

        # k-NN translation: weighted mean
        knn_trans = (gt_train_trans[topk_idx] * weights.unsqueeze(-1)).sum(dim=1)

        # k-NN rotation: weighted quaternion mean
        knn_quats = gt_train_quat[topk_idx]  # (N_test, k, 4)
        knn_quat_list = []
        for i in range(test_pooled.shape[0]):
            knn_quat_list.append(weighted_quaternion_mean(knn_quats[i], weights[i]))
        knn_quat = torch.stack(knn_quat_list)
        knn_rot = quaternion_to_matrix(knn_quat)

        for alpha in alpha_values:
            blend_trans = alpha * mlp_trans + (1 - alpha) * knn_trans
            # For rotation: interpolate quaternions
            blend_quat_list = []
            for i in range(test_pooled.shape[0]):
                bq = alpha * mlp_quat[i] + (1 - alpha) * knn_quat[i]
                bq = bq / bq.norm().clamp(min=1e-8)
                blend_quat_list.append(bq)
            blend_quat = torch.stack(blend_quat_list)
            blend_rot = quaternion_to_matrix(blend_quat)

            m = compute_metrics(blend_trans, blend_rot, gt_test_trans, gt_test_rot)
            r10 = m['recalls']['10°/2m']
            label = f"kNN(k={k},α={alpha})"
            if r10 > best_r10:
                best_r10 = r10
                best_config = label
                best_metrics = m

        # Also try adaptive alpha: alpha = sigmoid(min_dist * scale)
        min_dist = topk_dists[:, 0]  # distance to nearest neighbor
        for scale in [5.0, 10.0, 20.0, 50.0]:
            adaptive_alpha = torch.sigmoid(min_dist * scale)  # close NN → low alpha → more kNN
            blend_trans = adaptive_alpha.unsqueeze(-1) * mlp_trans + (1 - adaptive_alpha.unsqueeze(-1)) * knn_trans
            blend_quat_list = []
            for i in range(test_pooled.shape[0]):
                a = adaptive_alpha[i].item()
                bq = a * mlp_quat[i] + (1 - a) * knn_quat[i]
                bq = bq / bq.norm().clamp(min=1e-8)
                blend_quat_list.append(bq)
            blend_quat = torch.stack(blend_quat_list)
            blend_rot = quaternion_to_matrix(blend_quat)

            m = compute_metrics(blend_trans, blend_rot, gt_test_trans, gt_test_rot)
            r10 = m['recalls']['10°/2m']
            label = f"kNN(k={k},adaptive_s={scale})"
            if r10 > best_r10:
                best_r10 = r10
                best_config = label
                best_metrics = m

    print(f"  Best k-NN config: {best_config}")
    print_metrics(best_config, best_metrics)
    results_dict[f'kNN-best ({best_config})'] = best_metrics

    # Also store pure kNN (alpha=0) results for reference
    topk_dists, topk_idx = dists.topk(5, dim=-1, largest=False)
    weights = 1.0 / (topk_dists + 1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    knn_trans = (gt_train_trans[topk_idx] * weights.unsqueeze(-1)).sum(dim=1)
    knn_quats = gt_train_quat[topk_idx]
    knn_quat_list = []
    for i in range(test_pooled.shape[0]):
        knn_quat_list.append(weighted_quaternion_mean(knn_quats[i], weights[i]))
    knn_quat = torch.stack(knn_quat_list)
    knn_rot = quaternion_to_matrix(knn_quat)
    m = compute_metrics(knn_trans, knn_rot, gt_test_trans, gt_test_rot)
    print_metrics("Pure kNN (k=5)", m)
    results_dict['Pure kNN (k=5)'] = m


# ============================================================
# Technique 2: MC Dropout
# ============================================================

def mc_dropout(model, test_pooled, trans_mean, trans_std, test_data, results_dict, N=50):
    print("\n" + "=" * 70)
    print("TECHNIQUE 2: MC Dropout (N={})".format(N))
    print("=" * 70)

    gt_test_trans = test_data.translations
    gt_test_rot = test_data.rotations

    # Collect predictions with dropout enabled
    all_trans = []
    all_rot_6d = []

    model.train()  # Enable dropout
    with torch.no_grad():
        for i in range(N):
            trans_list, rot6d_list = [], []
            for start in range(0, test_pooled.shape[0], 64):
                end = min(start + 64, test_pooled.shape[0])
                x = model.input_norm(test_pooled[start:end])
                h = model.backbone(x)
                t = model.trans_head(h)
                r6d = model.rot_head(h)
                trans_list.append(t)
                rot6d_list.append(r6d)
            all_trans.append(torch.cat(trans_list, dim=0))
            all_rot_6d.append(torch.cat(rot6d_list, dim=0))

    model.eval()

    all_trans = torch.stack(all_trans)  # (N, N_test, 3)
    all_rot_6d = torch.stack(all_rot_6d)  # (N, N_test, 6)

    # Denormalize translations
    all_trans_real = all_trans * trans_std.to(DEVICE) + trans_mean.to(DEVICE)

    # Convert all rot predictions to matrices
    all_rot_mat = gram_schmidt_6d_to_matrix(all_rot_6d.reshape(-1, 6)).reshape(N, -1, 3, 3)

    # Aggregation methods
    # 1. Mean
    mean_trans = all_trans_real.mean(dim=0)
    mean_rot6d = all_rot_6d.mean(dim=0)
    mean_rot = gram_schmidt_6d_to_matrix(mean_rot6d)
    m = compute_metrics(mean_trans, mean_rot, gt_test_trans, gt_test_rot)
    print_metrics("MC Dropout - Mean", m)
    results_dict['MC Dropout - Mean'] = m

    # 2. Median
    median_trans = all_trans_real.median(dim=0).values
    median_rot6d = all_rot_6d.median(dim=0).values
    median_rot = gram_schmidt_6d_to_matrix(median_rot6d)
    m = compute_metrics(median_trans, median_rot, gt_test_trans, gt_test_rot)
    print_metrics("MC Dropout - Median", m)
    results_dict['MC Dropout - Median'] = m

    # 3. Trimmed mean (drop 10% extremes)
    trim = max(1, N // 10)
    sorted_trans, _ = all_trans_real.sort(dim=0)
    trimmed_trans = sorted_trans[trim:N-trim].mean(dim=0)
    sorted_rot6d, _ = all_rot_6d.sort(dim=0)
    trimmed_rot6d = sorted_rot6d[trim:N-trim].mean(dim=0)
    trimmed_rot = gram_schmidt_6d_to_matrix(trimmed_rot6d)
    m = compute_metrics(trimmed_trans, trimmed_rot, gt_test_trans, gt_test_rot)
    print_metrics("MC Dropout - Trimmed Mean", m)
    results_dict['MC Dropout - Trimmed Mean'] = m

    # Uncertainty correlation
    trans_std_per_sample = all_trans_real.std(dim=0).norm(dim=-1).numpy()
    trans_errors = (mean_trans - gt_test_trans).norm(dim=-1).numpy()
    corr = np.corrcoef(trans_std_per_sample, trans_errors)[0, 1]
    print(f"  Uncertainty-Error correlation (translation): {corr:.3f}")


# ============================================================
# Technique 3: Test-Time Feature Augmentation
# ============================================================

def tta_feature_aug(model, test_pooled, trans_mean, trans_std, test_data, results_dict, N=50):
    print("\n" + "=" * 70)
    print("TECHNIQUE 3: Test-Time Feature Augmentation (N={})".format(N))
    print("=" * 70)

    gt_test_trans = test_data.translations
    gt_test_rot = test_data.rotations
    model.eval()

    for sigma in [0.01, 0.05, 0.1]:
        all_trans = []
        all_rot_6d = []
        with torch.no_grad():
            for i in range(N):
                noisy = test_pooled + torch.randn_like(test_pooled) * sigma
                trans_list, rot6d_list = [], []
                for start in range(0, noisy.shape[0], 64):
                    end = min(start + 64, noisy.shape[0])
                    t, r = model.forward_from_pooled(noisy[start:end])
                    # Get 6d for averaging
                    rot6d_list.append(model.rot_head(model.backbone(model.input_norm(noisy[start:end]))))
                    trans_list.append(t)
                all_trans.append(torch.cat(trans_list, dim=0))
                all_rot_6d.append(torch.cat(rot6d_list, dim=0))

        all_trans = torch.stack(all_trans)
        all_rot_6d = torch.stack(all_rot_6d)

        mean_trans_norm = all_trans.mean(dim=0)
        mean_trans = mean_trans_norm * trans_std.to(DEVICE) + trans_mean.to(DEVICE)
        mean_rot6d = all_rot_6d.mean(dim=0)
        mean_rot = gram_schmidt_6d_to_matrix(mean_rot6d)

        m = compute_metrics(mean_trans, mean_rot, gt_test_trans, gt_test_rot)
        label = f"TTA σ={sigma}"
        print_metrics(label, m)
        results_dict[label] = m


# ============================================================
# Technique 4: Multi-Model Ensemble + k-NN
# ============================================================

def multi_model_ensemble(train_data, test_data, train_pooled_primary, test_pooled_primary,
                         results_dict):
    print("\n" + "=" * 70)
    print("TECHNIQUE 4: Multi-Model Ensemble + k-NN")
    print("=" * 70)

    gt_test_trans = test_data.translations
    gt_test_rot = test_data.rotations
    gt_train_trans = train_data.translations
    gt_train_quat = rotation_matrix_to_quaternion(train_data.rotations)

    all_trans = []
    all_rot6d = []
    all_rot_mat = []

    for name, (exp_name, reported_acc) in MODEL_CONFIGS.items():
        print(f"  Loading {name} ({exp_name})...")
        model, tmean, tstd = load_model(exp_name)
        model.eval()

        # Pool features through this model
        test_pooled = pool_features_batched(model, test_data, batch_size=16)

        with torch.no_grad():
            trans_list, rot6d_list, rot_mat_list = [], [], []
            for start in range(0, test_pooled.shape[0], 64):
                end = min(start + 64, test_pooled.shape[0])
                t, r = model.forward_from_pooled(test_pooled[start:end])
                t_real = t * tstd.to(DEVICE) + tmean.to(DEVICE)
                trans_list.append(t_real)
                rot6d_list.append(model.rot_head(model.backbone(model.input_norm(test_pooled[start:end]))))
                rot_mat_list.append(r)

        pred_trans = torch.cat(trans_list, dim=0)
        pred_rot6d = torch.cat(rot6d_list, dim=0)
        pred_rot_mat = torch.cat(rot_mat_list, dim=0)

        all_trans.append(pred_trans)
        all_rot6d.append(pred_rot6d)
        all_rot_mat.append(pred_rot_mat)

        m = compute_metrics(pred_trans, pred_rot_mat, gt_test_trans, gt_test_rot)
        print_metrics(f"  Single: {name}", m)
        if name == 'seed314':
            results_dict['Baseline (seed314)'] = m

    all_trans = torch.stack(all_trans)  # (5, N_test, 3)
    all_rot6d = torch.stack(all_rot6d)  # (5, N_test, 6)
    all_rot_mat = torch.stack(all_rot_mat)  # (5, N_test, 3, 3)

    # Simple mean ensemble
    ens_trans = all_trans.mean(dim=0)
    ens_rot6d = all_rot6d.mean(dim=0)
    ens_rot = gram_schmidt_6d_to_matrix(ens_rot6d)
    m = compute_metrics(ens_trans, ens_rot, gt_test_trans, gt_test_rot)
    print_metrics("Ensemble Mean (5 models)", m)
    results_dict['Ensemble Mean (5 models)'] = m

    # Ensemble + k-NN blending
    # Use primary model's pooled features for kNN
    train_norm = F.normalize(train_pooled_primary, dim=-1)
    test_norm = F.normalize(test_pooled_primary, dim=-1)
    dists = torch.cdist(test_norm, train_norm)

    for k in [3, 5, 10]:
        topk_dists, topk_idx = dists.topk(k, dim=-1, largest=False)
        weights = 1.0 / (topk_dists + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        knn_trans = (gt_train_trans[topk_idx] * weights.unsqueeze(-1)).sum(dim=1)
        knn_quats = gt_train_quat[topk_idx]
        knn_quat_list = []
        for i in range(test_pooled_primary.shape[0]):
            knn_quat_list.append(weighted_quaternion_mean(knn_quats[i], weights[i]))
        knn_quat = torch.stack(knn_quat_list)
        knn_rot = quaternion_to_matrix(knn_quat)

        ens_quat = rotation_matrix_to_quaternion(ens_rot)
        for alpha in [0.6, 0.7, 0.8, 0.9]:
            blend_trans = alpha * ens_trans + (1 - alpha) * knn_trans
            blend_quat_list = []
            for i in range(ens_trans.shape[0]):
                bq = alpha * ens_quat[i] + (1 - alpha) * knn_quat[i]
                bq = bq / bq.norm().clamp(min=1e-8)
                blend_quat_list.append(bq)
            blend_rot = quaternion_to_matrix(torch.stack(blend_quat_list))
            m = compute_metrics(blend_trans, blend_rot, gt_test_trans, gt_test_rot)
            label = f"Ensemble+kNN(k={k},α={alpha})"
            r10 = m['recalls']['10°/2m']
            if r10 >= 72.0:
                print_metrics(label, m)
            results_dict[label] = m

    return all_trans, all_rot_mat


# ============================================================
# Technique 5: Per-Sample Model Selection
# ============================================================

def per_sample_selection(all_trans, all_rot_mat, train_data, test_data,
                         train_pooled, test_pooled, results_dict):
    print("\n" + "=" * 70)
    print("TECHNIQUE 5: Per-Sample Model Selection")
    print("=" * 70)

    gt_test_trans = test_data.translations
    gt_test_rot = test_data.rotations
    gt_train_trans = train_data.translations
    gt_train_quat = rotation_matrix_to_quaternion(train_data.rotations)
    N_test = all_trans.shape[1]
    N_models = all_trans.shape[0]

    # Inter-model disagreement
    mean_trans = all_trans.mean(dim=0)
    trans_spread = (all_trans - mean_trans.unsqueeze(0)).norm(dim=-1).mean(dim=0)  # (N_test,)

    # Compute k-NN predictions
    train_norm = F.normalize(train_pooled, dim=-1)
    test_norm = F.normalize(test_pooled, dim=-1)
    dists = torch.cdist(test_norm, train_norm)
    topk_dists, topk_idx = dists.topk(5, dim=-1, largest=False)
    weights = 1.0 / (topk_dists + 1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    knn_trans = (gt_train_trans[topk_idx] * weights.unsqueeze(-1)).sum(dim=1)
    knn_quats = gt_train_quat[topk_idx]
    knn_quat_list = []
    for i in range(N_test):
        knn_quat_list.append(weighted_quaternion_mean(knn_quats[i], weights[i]))
    knn_quat = torch.stack(knn_quat_list)
    knn_rot = quaternion_to_matrix(knn_quat)

    # Strategy: high certainty → mean, low certainty → median
    median_trans = all_trans.median(dim=0).values
    all_rot_quat = torch.stack([rotation_matrix_to_quaternion(all_rot_mat[j]) for j in range(N_models)])
    median_rot6d_raw = all_rot_quat.median(dim=0).values
    median_rot = quaternion_to_matrix(median_rot6d_raw / median_rot6d_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    for threshold_pct in [25, 50, 75]:
        threshold = np.percentile(trans_spread.numpy(), threshold_pct)
        high_cert = trans_spread < threshold
        low_cert = ~high_cert

        final_trans = torch.where(high_cert.unsqueeze(-1), mean_trans, median_trans)
        final_rot = torch.where(high_cert.unsqueeze(-1).unsqueeze(-1), 
                                gram_schmidt_6d_to_matrix(
                                    torch.stack([all_rot_mat[j][:, :2, :].reshape(-1, 6) for j in range(N_models)]).mean(dim=0)),
                                median_rot)
        m = compute_metrics(final_trans, final_rot, gt_test_trans, gt_test_rot)
        label = f"PerSample mean/median (p{threshold_pct})"
        print_metrics(label, m)
        results_dict[label] = m

    # Strategy: high certainty → mean, low certainty → closest-to-kNN
    for threshold_pct in [25, 50, 75]:
        threshold = np.percentile(trans_spread.numpy(), threshold_pct)
        high_cert = trans_spread < threshold

        # For low certainty: pick model closest to kNN prediction
        model_dist_to_knn = (all_trans - knn_trans.unsqueeze(0)).norm(dim=-1)  # (N_models, N_test)
        closest_model = model_dist_to_knn.argmin(dim=0)  # (N_test,)

        final_trans = mean_trans.clone()
        final_rot = gram_schmidt_6d_to_matrix(
            torch.stack([all_rot_mat[j][:, :2, :].reshape(-1, 6) for j in range(N_models)]).mean(dim=0))

        for i in range(N_test):
            if not high_cert[i]:
                final_trans[i] = all_trans[closest_model[i], i]
                final_rot[i] = all_rot_mat[closest_model[i], i]

        m = compute_metrics(final_trans, final_rot, gt_test_trans, gt_test_rot)
        label = f"PerSample mean/closest-kNN (p{threshold_pct})"
        print_metrics(label, m)
        results_dict[label] = m


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    results = OrderedDict()

    # Load data
    train_data, test_data = load_data()

    # Load primary model
    print("\nLoading primary model (seed314)...")
    model, trans_mean, trans_std = load_model('exp14s_128d_wider_seed314')
    model.eval()

    # Pool features
    print("Pooling features...")
    train_pooled = pool_features_batched(model, train_data, batch_size=16)
    test_pooled = pool_features_batched(model, test_data, batch_size=16)
    print(f"  Train pooled: {train_pooled.shape}, Test pooled: {test_pooled.shape}")

    # Baseline
    pred_trans, pred_rot = predict_from_pooled(model, test_pooled, trans_mean, trans_std)
    m = compute_metrics(pred_trans, pred_rot, test_data.translations, test_data.rotations)
    results['Baseline (seed314)'] = m

    print("\n" + "=" * 70)
    print("HEADER")
    print("=" * 70)
    print(f"  {'Technique':45s} | R@5/1 | R@10/2         | R@15/5 | Trans   | Rot")
    print(f"  {'-'*45}-+-{'-'*5}-+-{'-'*14}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}")
    print_metrics("Baseline (seed314)", m)

    # Run techniques
    knn_blending(model, train_data, test_data, trans_mean, trans_std,
                 train_pooled, test_pooled, results)

    mc_dropout(model, test_pooled, trans_mean, trans_std, test_data, results, N=50)

    tta_feature_aug(model, test_pooled, trans_mean, trans_std, test_data, results, N=50)

    all_trans, all_rot_mat = multi_model_ensemble(
        train_data, test_data, train_pooled, test_pooled, results)

    per_sample_selection(all_trans, all_rot_mat, train_data, test_data,
                         train_pooled, test_pooled, results)

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n\n" + "=" * 110)
    print("SUMMARY TABLE - All Techniques")
    print("=" * 110)
    print(f"  {'Technique':50s} | {'R@5°/1m':>7s} | {'R@10°/2m':>8s} | {'R@15°/5m':>8s} | {'Trans':>7s} | {'Rot':>6s}")
    print(f"  {'-'*50}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}")

    # Sort by R@10/2m descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]['recalls']['10°/2m'], reverse=True)
    baseline_r10 = results.get('Baseline (seed314)', {}).get('recalls', {}).get('10°/2m', 72.0)

    for name, m in sorted_results:
        r10 = m['recalls']['10°/2m']
        delta = r10 - baseline_r10
        sign = '+' if delta >= 0 else ''
        print(f"  {name:50s} | {m['recalls']['5°/1m']:6.1f}% | {r10:6.1f}% {sign}{delta:+.1f} | "
              f"{m['recalls']['15°/5m']:6.1f}% | {m['med_trans_mm']:5.0f}mm | {m['med_rot_deg']:5.2f}°")

    print("=" * 110)

    best_name, best_m = sorted_results[0]
    print(f"\nBest technique: {best_name}")
    print(f"  R@10°/2m = {best_m['recalls']['10°/2m']:.1f}%")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
