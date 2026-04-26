#!/usr/bin/env python3
"""
Pose Regression v4: Decoupled / Frustum / Grid Classification
==============================================================
Three architectures for improving feature retrieval quality, targeting
80% recall at R@10°/2m (from 51.1% in exp01).

Modes:
    decoupled  — Separate R/t branches with explicit loss weighting
    frustum    — Camera center + viewing direction + up vector
    grid_cls   — K-means spatial classification + residual regression

Usage:
    python feature_retrieval/pose_regressor_v4.py --mode decoupled --output_dir output/.../exp08
    python feature_retrieval/pose_regressor_v4.py --mode frustum   --output_dir output/.../exp09
    python feature_retrieval/pose_regressor_v4.py --mode grid_cls  --output_dir output/.../exp10
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
from sklearn.cluster import KMeans


# ============================================================
# Utility: Rotation representations
# ============================================================

def quaternion_to_matrix(q):
    """Convert w-first quaternion [w,x,y,z] to 3x3 rotation matrix."""
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
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1 = v[..., :3]
    a2 = v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)  # (*, 3, 3), rows = basis vectors


def geodesic_distance(R1, R2):
    """Geodesic distance between rotation matrices in radians."""
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).reshape(R1.shape[:-2])


def rotation_to_forward_up(R_w2c):
    """Extract forward and up vectors from w2c rotation matrix.
    
    R_w2c rows are the camera axes in world coordinates.
    Cambridge/COLMAP convention:
        row 0 = right (x), row 1 = down (y), row 2 = forward (z)
    
    We define:
        forward = R_w2c[2, :] (camera z-axis in world coords, i.e., viewing direction)
        up = -R_w2c[1, :] (negative of camera y-axis = world up from camera perspective)
    
    Args: R_w2c (*, 3, 3)
    Returns: forward (*, 3), up (*, 3) — both unit vectors
    """
    forward = R_w2c[..., 2, :]   # (*, 3)
    up = -R_w2c[..., 1, :]       # (*, 3)
    return F.normalize(forward, dim=-1), F.normalize(up, dim=-1)


def forward_up_to_rotation(forward, up):
    """Reconstruct R_w2c from forward and up vectors via Gram-Schmidt.
    
    Args: forward (*, 3), up (*, 3)
    Returns: R_w2c (*, 3, 3)
    """
    z = F.normalize(forward, dim=-1)         # forward = camera z
    # Orthogonalize up w.r.t. forward
    y_neg = up - (z * up).sum(dim=-1, keepdim=True) * z
    y_neg = F.normalize(y_neg, dim=-1)       # up direction
    y = -y_neg                               # camera y = down
    x = torch.cross(y, z, dim=-1)            # right = down × forward
    x = F.normalize(x, dim=-1)
    # R_w2c: rows are [right, down, forward]
    R = torch.stack([x, y, z], dim=-2)       # (*, 3, 3)
    return R


# ============================================================
# Dataset
# ============================================================

class PoseRegressionDataset:
    """In-memory dataset: RADIO summary tokens + GT poses."""

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
        self.translations = torch.stack(translations_list).to(device)  # camera centers
        quat = torch.stack(rotations_list)
        self.rotations = quaternion_to_matrix(quat).to(device)  # R_w2c
        self.names = names_list
        self.N = len(names_list)

        # Pre-compute forward and up vectors
        self.forward_vecs, self.up_vecs = rotation_to_forward_up(self.rotations)

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

def make_mlp_block(in_dim, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class DecoupledPoseRegressor(nn.Module):
    """Separate backbone branches for rotation and translation."""

    def __init__(self, input_dim=2560, dropout=0.1):
        super().__init__()
        # Shared initial layer
        self.shared = make_mlp_block(input_dim, 1024, dropout)
        # Separate branches
        self.rot_branch = nn.Sequential(
            make_mlp_block(1024, 512, dropout),
            make_mlp_block(512, 256, dropout),
        )
        self.trans_branch = nn.Sequential(
            make_mlp_block(1024, 512, dropout),
            make_mlp_block(512, 256, dropout),
        )
        self.rot_head = nn.Linear(256, 6)
        self.trans_head = nn.Linear(256, 3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x)
        h_rot = self.rot_branch(h)
        h_trans = self.trans_branch(h)
        trans = self.trans_head(h_trans)
        rot_6d = self.rot_head(h_rot)
        rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
        return trans, rot_mat


class FrustumPoseRegressor(nn.Module):
    """Predict camera center + forward direction + up direction."""

    def __init__(self, input_dim=2560, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            make_mlp_block(input_dim, 1024, dropout),
            make_mlp_block(1024, 512, dropout),
            make_mlp_block(512, 256, dropout),
        )
        self.center_head = nn.Linear(256, 3)
        self.forward_head = nn.Linear(256, 3)
        self.up_head = nn.Linear(256, 3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        center = self.center_head(h)              # (B, 3) normalized coords
        forward_raw = self.forward_head(h)         # (B, 3)
        up_raw = self.up_head(h)                   # (B, 3)
        forward_vec = F.normalize(forward_raw, dim=-1)
        up_vec = F.normalize(up_raw, dim=-1)
        return center, forward_vec, up_vec


class GridClassPoseRegressor(nn.Module):
    """K-means grid classification + residual regression + rotation."""

    def __init__(self, input_dim=2560, n_classes=50, dropout=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = nn.Sequential(
            make_mlp_block(input_dim, 1024, dropout),
            make_mlp_block(1024, 512, dropout),
            make_mlp_block(512, 256, dropout),
        )
        self.cls_head = nn.Linear(256, n_classes)
        self.residual_head = nn.Linear(256, 3)
        self.rot_head = nn.Linear(256, 6)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        cls_logits = self.cls_head(h)           # (B, K)
        residual = self.residual_head(h)        # (B, 3) in meters
        rot_6d = self.rot_head(h)               # (B, 6)
        rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
        return cls_logits, residual, rot_mat


# ============================================================
# Training
# ============================================================

def compute_kmeans_clusters(train_positions, n_clusters=50):
    """Run K-means on camera XZ positions."""
    pos_np = train_positions.cpu().numpy()
    # Use XYZ for clustering (Y is nearly flat but include it for completeness)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    km.fit(pos_np)
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    labels = torch.tensor(km.labels_, dtype=torch.long)
    print(f"  K-means: {n_clusters} clusters, inertia={km.inertia_:.1f}")
    counts = np.bincount(km.labels_)
    print(f"  Samples per cluster: min={counts.min()}, mean={counts.mean():.1f}, "
          f"median={np.median(counts):.0f}, max={counts.max()}")
    return centroids, labels, km


def assign_test_to_clusters(test_positions, km):
    """Assign test positions to nearest K-means centroid."""
    pos_np = test_positions.cpu().numpy()
    labels = km.predict(pos_np)
    return torch.tensor(labels, dtype=torch.long)


def train_decoupled(args, train_data, test_data, trans_mean, trans_std, device, out_dir):
    """Train decoupled R+t architecture with explicit loss weighting."""
    model = DecoupledPoseRegressor(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Decoupled] Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Fixed loss weights: emphasize translation
    lambda_t = args.lambda_t
    lambda_r = args.lambda_r
    print(f"  Loss weights: lambda_t={lambda_t}, lambda_r={lambda_r}")

    history = {'epoch': [], 'loss': [], 'loss_trans': [], 'loss_rot': [],
               'val_rot_median': [], 'val_trans_median': []}
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
        loss = lambda_t * loss_trans + lambda_r * loss_rot

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vt, vr = model(test_data.features)
                vt_real = test_data.denormalize(vt, trans_mean, trans_std)
                te = (vt_real - test_data.translations).norm(dim=-1)
                re = geodesic_distance(vr, test_data.rotations) * 180 / math.pi
                vtm = te.median().item()
                vrm = re.median().item()
                val_score = vtm + vrm * 0.1

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_trans'].append(loss_trans.item())
            history['loss_rot'].append(loss_rot.item())
            history['val_rot_median'].append(vrm)
            history['val_trans_median'].append(vtm)

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                      f"(t={loss_trans.item():.4f} r={loss_rot.item():.4f}) "
                      f"| val: {vrm:.2f}°/{vtm*1000:.0f}mm | {time.time()-t0:.1f}s")

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_rot_median': vrm, 'val_trans_median': vtm,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                    'mode': 'decoupled', 'lambda_t': lambda_t, 'lambda_r': lambda_r,
                }, out_dir / 'model_best.pt')

    print(f"Training done. Best epoch: {best_epoch}")
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    return model, history


def train_frustum(args, train_data, test_data, trans_mean, trans_std, device, out_dir):
    """Train frustum (center + forward + up) architecture."""
    model = FrustumPoseRegressor(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Frustum] Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    lambda_center = args.lambda_t
    lambda_fwd = args.lambda_fwd
    lambda_up = args.lambda_up
    print(f"  Loss weights: center={lambda_center}, fwd={lambda_fwd}, up={lambda_up}")

    history = {'epoch': [], 'loss': [], 'loss_center': [], 'loss_fwd': [], 'loss_up': [],
               'val_rot_median': [], 'val_trans_median': []}
    best_val_score = float('inf')
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        feat = train_data.features
        if args.feature_dropout > 0:
            feat = F.dropout(feat, p=args.feature_dropout, training=True)

        center_pred, fwd_pred, up_pred = model(feat)

        # Center loss (SmoothL1 on normalized)
        loss_center = F.smooth_l1_loss(center_pred, train_data.translations_norm)

        # Forward direction loss (1 - cosine similarity)
        loss_fwd = (1.0 - (fwd_pred * train_data.forward_vecs).sum(dim=-1)).mean()

        # Up direction loss
        loss_up = (1.0 - (up_pred * train_data.up_vecs).sum(dim=-1)).mean()

        loss = lambda_center * loss_center + lambda_fwd * loss_fwd + lambda_up * loss_up

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vc, vf, vu = model(test_data.features)
                vc_real = test_data.denormalize(vc, trans_mean, trans_std)
                # Reconstruct rotation from forward + up
                vr = forward_up_to_rotation(vf, vu)
                te = (vc_real - test_data.translations).norm(dim=-1)
                re = geodesic_distance(vr, test_data.rotations) * 180 / math.pi
                vtm = te.median().item()
                vrm = re.median().item()
                val_score = vtm + vrm * 0.1

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_center'].append(loss_center.item())
            history['loss_fwd'].append(loss_fwd.item())
            history['loss_up'].append(loss_up.item())
            history['val_rot_median'].append(vrm)
            history['val_trans_median'].append(vtm)

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                      f"(c={loss_center.item():.4f} f={loss_fwd.item():.4f} u={loss_up.item():.4f}) "
                      f"| val: {vrm:.2f}°/{vtm*1000:.0f}mm | {time.time()-t0:.1f}s")

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_rot_median': vrm, 'val_trans_median': vtm,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                    'mode': 'frustum',
                }, out_dir / 'model_best.pt')

    print(f"Training done. Best epoch: {best_epoch}")
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    return model, history


def train_grid_cls(args, train_data, test_data, trans_mean, trans_std, device, out_dir):
    """Train grid classification + residual regression."""
    n_clusters = args.n_clusters
    print(f"\n[Grid Classification] Setting up K-means with K={n_clusters}...")

    # K-means on training camera positions (world coordinates, not normalized)
    centroids, train_labels, km = compute_kmeans_clusters(train_data.translations, n_clusters)
    centroids = centroids.to(device)
    train_labels = train_labels.to(device)

    # Compute residuals: position - assigned centroid
    train_residuals = train_data.translations - centroids[train_labels]
    residual_scale = train_residuals.abs().max().item()
    print(f"  Max residual magnitude: {residual_scale:.2f}m")

    # Assign test samples to clusters (for evaluation/logging only)
    test_labels = assign_test_to_clusters(test_data.translations, km).to(device)

    # Save cluster info
    torch.save({
        'centroids': centroids.cpu(),
        'train_labels': train_labels.cpu(),
        'km_centers': km.cluster_centers_,
    }, out_dir / 'cluster_info.pt')

    model = GridClassPoseRegressor(n_classes=n_clusters, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GridCls] Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    lambda_cls = args.lambda_cls
    lambda_res = args.lambda_t
    lambda_rot = args.lambda_r
    print(f"  Loss weights: cls={lambda_cls}, residual={lambda_res}, rot={lambda_rot}")

    history = {'epoch': [], 'loss': [], 'loss_cls': [], 'loss_res': [], 'loss_rot': [],
               'val_rot_median': [], 'val_trans_median': [], 'val_cls_acc': []}
    best_val_score = float('inf')
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        feat = train_data.features
        if args.feature_dropout > 0:
            feat = F.dropout(feat, p=args.feature_dropout, training=True)

        cls_logits, residual_pred, rot_pred = model(feat)

        # Classification loss
        loss_cls = F.cross_entropy(cls_logits, train_labels)

        # Residual loss (only for correct cell — use GT labels)
        gt_residuals = train_data.translations - centroids[train_labels]
        loss_res = F.smooth_l1_loss(residual_pred, gt_residuals)

        # Rotation loss
        loss_rot = geodesic_distance(rot_pred, train_data.rotations).mean()

        loss = lambda_cls * loss_cls + lambda_res * loss_res + lambda_rot * loss_rot

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vcls, vres, vrot = model(test_data.features)

                # Position = centroid[predicted_class] + residual
                pred_cls = vcls.argmax(dim=-1)
                pred_center = centroids[pred_cls] + vres

                te = (pred_center - test_data.translations).norm(dim=-1)
                re = geodesic_distance(vrot, test_data.rotations) * 180 / math.pi
                vtm = te.median().item()
                vrm = re.median().item()
                val_score = vtm + vrm * 0.1

                cls_acc = (pred_cls == test_labels).float().mean().item() * 100

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_cls'].append(loss_cls.item())
            history['loss_res'].append(loss_res.item())
            history['loss_rot'].append(loss_rot.item())
            history['val_rot_median'].append(vrm)
            history['val_trans_median'].append(vtm)
            history['val_cls_acc'].append(cls_acc)

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                      f"(cls={loss_cls.item():.4f} res={loss_res.item():.4f} rot={loss_rot.item():.4f}) "
                      f"| val: {vrm:.2f}°/{vtm*1000:.0f}mm cls_acc={cls_acc:.1f}% "
                      f"| {time.time()-t0:.1f}s")

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_rot_median': vrm, 'val_trans_median': vtm,
                    'norm_params': {'mean': trans_mean.cpu(), 'std': trans_std.cpu()},
                    'mode': 'grid_cls', 'n_clusters': n_clusters,
                    'centroids': centroids.cpu(),
                }, out_dir / 'model_best.pt')

    print(f"Training done. Best epoch: {best_epoch}")
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    return model, centroids, km, history


# ============================================================
# Evaluation (unified for all modes)
# ============================================================

def predict_poses(model, features, trans_mean, trans_std, mode, centroids=None):
    """Get predicted camera center and R_w2c from any model."""
    model.eval()
    with torch.no_grad():
        if mode == 'decoupled':
            trans_norm, rot_mat = model(features)
            center = trans_norm * trans_std + trans_mean
            return center, rot_mat

        elif mode == 'frustum':
            center_norm, fwd, up = model(features)
            center = center_norm * trans_std + trans_mean
            rot_mat = forward_up_to_rotation(fwd, up)
            return center, rot_mat

        elif mode == 'grid_cls':
            cls_logits, residual, rot_mat = model(features)
            pred_cls = cls_logits.argmax(dim=-1)
            center = centroids[pred_cls] + residual
            return center, rot_mat


def evaluate_all(pred_center, pred_rot, test_data, train_data, out_dir, mode_name):
    """Full evaluation: direct pose + retrieval-mode."""
    out_dir = Path(out_dir)

    # Direct pose errors
    trans_errors = (pred_center - test_data.translations).norm(dim=-1).cpu().numpy()
    rot_errors = (geodesic_distance(pred_rot, test_data.rotations) * 180 / math.pi).cpu().numpy()

    # Retrieval-mode: use predicted center to find nearest training image
    pred_c_np = pred_center.cpu().numpy()
    train_c_np = train_data.translations.cpu().numpy()
    train_R_np = train_data.rotations.cpu().numpy()
    test_c_np = test_data.translations.cpu().numpy()
    test_R_np = test_data.rotations.cpu().numpy()

    # Build KD-tree for fast nearest neighbor
    from scipy.spatial import cKDTree
    tree = cKDTree(train_c_np)

    retrieval_trans_errors = np.zeros(test_data.N)
    retrieval_rot_errors = np.zeros(test_data.N)
    retrieval_top3_trans = np.zeros(test_data.N)
    retrieval_top3_rot = np.zeros(test_data.N)

    for i in range(test_data.N):
        # Top-1 nearest by predicted center
        _, idx = tree.query(pred_c_np[i], k=1)
        ret_c = train_c_np[idx]
        ret_R = train_R_np[idx]
        retrieval_trans_errors[i] = np.linalg.norm(test_c_np[i] - ret_c)
        Rd = test_R_np[i] @ ret_R.T
        tr = np.clip(np.trace(Rd), -1, 3)
        retrieval_rot_errors[i] = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))

        # Top-3 spatial oracle
        _, idxs = tree.query(pred_c_np[i], k=3)
        dists = np.array([np.linalg.norm(test_c_np[i] - train_c_np[j]) for j in idxs])
        best_j = idxs[np.argmin(dists)]
        retrieval_top3_trans[i] = dists.min()
        Rd3 = test_R_np[i] @ train_R_np[best_j].T
        tr3 = np.clip(np.trace(Rd3), -1, 3)
        retrieval_top3_rot[i] = np.degrees(np.arccos(np.clip((tr3 - 1) / 2, -1, 1)))

    results = {
        'mode': mode_name,
        'direct_pose': {
            'rotation_deg': {
                'median': float(np.median(rot_errors)),
                'mean': float(np.mean(rot_errors)),
                'std': float(np.std(rot_errors)),
                'p25': float(np.percentile(rot_errors, 25)),
                'p75': float(np.percentile(rot_errors, 75)),
                'p90': float(np.percentile(rot_errors, 90)),
                'p95': float(np.percentile(rot_errors, 95)),
            },
            'translation_mm': {
                'median': float(np.median(trans_errors) * 1000),
                'mean': float(np.mean(trans_errors) * 1000),
                'p25': float(np.percentile(trans_errors, 25) * 1000),
                'p75': float(np.percentile(trans_errors, 75) * 1000),
                'p90': float(np.percentile(trans_errors, 90) * 1000),
                'p95': float(np.percentile(trans_errors, 95) * 1000),
            },
            'recall': {},
            'individual_pass_rates': {},
        },
        'retrieval_mode': {
            'top1': {
                'rot_median': float(np.median(retrieval_rot_errors)),
                'trans_median_mm': float(np.median(retrieval_trans_errors) * 1000),
            },
            'top3_oracle': {
                'rot_median': float(np.median(retrieval_top3_rot)),
                'trans_median_mm': float(np.median(retrieval_top3_trans) * 1000),
            },
            'recall': {},
        },
    }

    # Recall thresholds
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
        ('25deg_5m', 25, 5.0),
    ]

    for name, rot_th, trans_th in thresholds:
        # Direct pose recall
        rot_pass = rot_errors < rot_th
        trans_pass = trans_errors < trans_th
        both = rot_pass & trans_pass
        results['direct_pose']['recall'][name] = float(both.mean()) * 100
        results['direct_pose']['individual_pass_rates'][name] = {
            'rot_pass': float(rot_pass.mean()) * 100,
            'trans_pass': float(trans_pass.mean()) * 100,
        }

        # Retrieval mode recall
        r_rot_pass = retrieval_rot_errors < rot_th
        r_trans_pass = retrieval_trans_errors < trans_th
        results['retrieval_mode']['recall'][name] = float((r_rot_pass & r_trans_pass).mean()) * 100

    # Print
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS — {mode_name}")
    print("=" * 70)

    dp = results['direct_pose']
    print(f"\n[Direct Pose]")
    print(f"  Rotation  median: {dp['rotation_deg']['median']:.2f}° "
          f"(mean={dp['rotation_deg']['mean']:.2f}°)")
    print(f"  Translation median: {dp['translation_mm']['median']:.0f}mm "
          f"(mean={dp['translation_mm']['mean']:.0f}mm)")

    print(f"\n  Recall (direct):")
    for name, rot_th, trans_th in thresholds:
        ipr = dp['individual_pass_rates'][name]
        print(f"    R@{rot_th}°/{trans_th}m: {dp['recall'][name]:.1f}%  "
              f"(rot={ipr['rot_pass']:.1f}%, trans={ipr['trans_pass']:.1f}%)")

    rm = results['retrieval_mode']
    print(f"\n[Retrieval Mode]")
    print(f"  Top-1 nearest: rot={rm['top1']['rot_median']:.2f}°  "
          f"trans={rm['top1']['trans_median_mm']:.0f}mm")
    print(f"  Top-3 oracle:  rot={rm['top3_oracle']['rot_median']:.2f}°  "
          f"trans={rm['top3_oracle']['trans_median_mm']:.0f}mm")
    print(f"\n  Recall (retrieval top-1):")
    for name, rot_th, trans_th in thresholds:
        print(f"    R@{rot_th}°/{trans_th}m: {rm['recall'][name]:.1f}%")

    print("=" * 70)

    # Save results
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save per-sample errors
    per_sample = []
    for i in range(test_data.N):
        per_sample.append({
            'name': test_data.names[i],
            'rot_error_deg': float(rot_errors[i]),
            'trans_error_m': float(trans_errors[i]),
            'retrieval_rot_deg': float(retrieval_rot_errors[i]),
            'retrieval_trans_m': float(retrieval_trans_errors[i]),
            'gt_center': test_data.translations[i].cpu().tolist(),
            'pred_center': pred_center[i].cpu().tolist(),
        })
    with open(out_dir / 'per_sample_errors.json', 'w') as f:
        json.dump(per_sample, f, indent=2)

    return results, trans_errors, rot_errors, retrieval_trans_errors, retrieval_rot_errors


# ============================================================
# Visualization
# ============================================================

def visualize_all(history, results, trans_errors, rot_errors,
                  retrieval_trans_errors, retrieval_rot_errors,
                  pred_center, test_data, train_data, out_dir, mode_name,
                  centroids=None):
    """Generate visualizations for any mode."""
    out_dir = Path(out_dir)
    pred_c_np = pred_center.cpu().numpy()
    gt_c_np = test_data.translations.cpu().numpy()
    train_c_np = train_data.translations.cpu().numpy()

    # 1. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{mode_name} — Training', fontsize=14)

    ax = axes[0]
    ax.plot(history['epoch'], history['loss'], 'b-', alpha=0.8, label='Total')
    if 'loss_trans' in history:
        ax.plot(history['epoch'], history['loss_trans'], 'g-', alpha=0.6, label='Trans')
        ax.plot(history['epoch'], history['loss_rot'], 'r-', alpha=0.6, label='Rot')
    if 'loss_center' in history:
        ax.plot(history['epoch'], history['loss_center'], 'g-', alpha=0.6, label='Center')
        ax.plot(history['epoch'], history['loss_fwd'], 'r-', alpha=0.6, label='Forward')
        ax.plot(history['epoch'], history['loss_up'], 'm-', alpha=0.6, label='Up')
    if 'loss_cls' in history:
        ax.plot(history['epoch'], history['loss_cls'], 'c-', alpha=0.6, label='Cls')
        ax.plot(history['epoch'], history['loss_res'], 'g-', alpha=0.6, label='Residual')
        ax.plot(history['epoch'], history['loss_rot'], 'r-', alpha=0.6, label='Rot')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Losses')
    ax.legend(fontsize=8); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history['epoch'], history['val_rot_median'], 'r-')
    ax.axhline(y=11.5, color='gray', linestyle='--', label='CLS baseline')
    ax.axhline(y=3.72, color='blue', linestyle='--', alpha=0.5, label='exp01')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Rotation Error (°)'); ax.set_title('Val Rotation')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(history['epoch'], [x*1000 for x in history['val_trans_median']], 'g-')
    ax.axhline(y=1360, color='gray', linestyle='--', label='CLS baseline')
    ax.axhline(y=1849, color='blue', linestyle='--', alpha=0.5, label='exp01')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Translation Error (mm)'); ax.set_title('Val Translation')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Error distributions (4 subplots: direct rot, direct trans, retr rot, retr trans)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{mode_name} — Error Distributions', fontsize=14)

    for ax, data, label, color, baseline in [
        (axes[0, 0], rot_errors, 'Direct Rot Error (°)', 'coral', 11.5),
        (axes[0, 1], trans_errors * 1000, 'Direct Trans Error (mm)', 'steelblue', 1360),
        (axes[1, 0], retrieval_rot_errors, 'Retrieval Rot Error (°)', 'salmon', 11.5),
        (axes[1, 1], retrieval_trans_errors * 1000, 'Retrieval Trans Error (mm)', 'skyblue', 1360),
    ]:
        ax.hist(data, bins=30, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.median(data), color='red', linestyle='--',
                   label=f'Median: {np.median(data):.1f}')
        ax.axvline(x=baseline, color='gray', linestyle='--', label=f'CLS: {baseline}')
        ax.set_xlabel(label); ax.set_ylabel('Count'); ax.set_title(label)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. BEV scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(train_c_np[:, 0], train_c_np[:, 2], c='lightgray', s=10, alpha=0.4,
               label=f'Train ({len(train_c_np)})')

    if centroids is not None:
        c_np = centroids.cpu().numpy()
        ax.scatter(c_np[:, 0], c_np[:, 2], c='orange', s=80, marker='x', linewidths=2,
                   label=f'Centroids ({len(c_np)})', zorder=5)

    ax.scatter(gt_c_np[:, 0], gt_c_np[:, 2], c='black', s=30, alpha=0.6, label='GT test', zorder=3)
    sc = ax.scatter(pred_c_np[:, 0], pred_c_np[:, 2],
                    c=trans_errors * 1000, cmap='RdYlGn_r', s=30, alpha=0.8,
                    vmin=0, vmax=np.percentile(trans_errors * 1000, 95),
                    edgecolors='gray', linewidths=0.5, zorder=4, label='Predicted')
    plt.colorbar(sc, ax=ax, label='Trans Error (mm)', shrink=0.8)

    for i in range(len(gt_c_np)):
        ax.plot([gt_c_np[i, 0], pred_c_np[i, 0]], [gt_c_np[i, 2], pred_c_np[i, 2]],
                'r-', alpha=0.15, linewidth=0.5)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_title(f'{mode_name} BEV (median err: {np.median(trans_errors)*1000:.0f}mm)')
    ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'bev_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Pose Regression v4')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['decoupled', 'frustum', 'grid_cls'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--summary_matrix', type=str,
                        default='output/feature_extract/features_radio_dual/OldHospital_pilot/summary_matrix.pt')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str, required=True)

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
    parser.add_argument('--log_every', type=int, default=10)

    # Loss weights
    parser.add_argument('--lambda_t', type=float, default=1.0, help='Translation/center loss weight')
    parser.add_argument('--lambda_r', type=float, default=1.0, help='Rotation loss weight')
    parser.add_argument('--lambda_fwd', type=float, default=1.0, help='Forward direction loss weight')
    parser.add_argument('--lambda_up', type=float, default=1.0, help='Up direction loss weight')
    parser.add_argument('--lambda_cls', type=float, default=1.0, help='Classification loss weight')

    # Grid classification
    parser.add_argument('--n_clusters', type=int, default=50, help='Number of K-means clusters')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    print(f"Loading data...")
    train_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'train', device)
    test_data = PoseRegressionDataset(args.summary_matrix, args.dataset_dir, 'test', device)

    trans_mean, trans_std = train_data.compute_normalization()
    train_data.normalize_translations(trans_mean, trans_std)
    test_data.normalize_translations(trans_mean, trans_std)
    print(f"Translation norm: mean={trans_mean.cpu().numpy()}, std={trans_std.cpu().numpy()}")

    torch.save({'mean': trans_mean, 'std': trans_std}, out_dir / 'norm_params.pt')

    # Train
    centroids = None
    if args.mode == 'decoupled':
        model, history = train_decoupled(args, train_data, test_data, trans_mean, trans_std, device, out_dir)
    elif args.mode == 'frustum':
        model, history = train_frustum(args, train_data, test_data, trans_mean, trans_std, device, out_dir)
    elif args.mode == 'grid_cls':
        model, centroids, km, history = train_grid_cls(args, train_data, test_data, trans_mean, trans_std, device, out_dir)

    # Load best model
    ckpt = torch.load(out_dir / 'model_best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\nLoaded best model from epoch {ckpt['epoch']} "
          f"(val: {ckpt['val_rot_median']:.2f}°/{ckpt['val_trans_median']*1000:.0f}mm)")

    if args.mode == 'grid_cls' and 'centroids' in ckpt:
        centroids = ckpt['centroids'].to(device)

    # Evaluate
    pred_center, pred_rot = predict_poses(model, test_data.features, trans_mean, trans_std,
                                          args.mode, centroids)

    results, trans_errors, rot_errors, retr_te, retr_re = evaluate_all(
        pred_center, pred_rot, test_data, train_data, out_dir, args.mode)

    # Visualize
    visualize_all(history, results, trans_errors, rot_errors, retr_te, retr_re,
                  pred_center, test_data, train_data, out_dir, args.mode, centroids)

    print("\nDone!")


if __name__ == '__main__':
    main()
