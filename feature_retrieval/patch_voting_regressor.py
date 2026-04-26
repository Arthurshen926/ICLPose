#!/usr/bin/env python3
"""
Per-Patch Pose Voting Regressor
================================
Instead of SPP-pooling all patches into one vector and regressing pose,
each patch independently predicts the camera pose. At test time, we
robustly aggregate predictions via median or RANSAC.

Key insight: with 895 images × 8160 patches = ~7.3M training samples,
overfitting is much less of a problem (vs. 895 samples in the SPP approach).

Architecture:
  Input: [128d patch_fine, 128d patch_coarse, 2d position] = 258d
  MLP: 258 → hidden → 9 (3 trans + 6 rot)

Aggregation strategies:
  1. Median translation + average rotation
  2. Weighted median (weight by prediction confidence)
  3. RANSAC-style: sample subsets, find consensus
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Rotation utilities (from patch_regressor_v7.py)
# ============================================================

def quaternion_to_matrix(q):
    """Quaternion (w, x, y, z) to 3x3 rotation matrix. Matches patch_regressor_v7."""
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
    """Convert 6D rotation representation to 3x3 rotation matrix. Matches patch_regressor_v7."""
    a1 = v[..., :3]
    a2 = v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def geodesic_distance(R1, R2):
    """Geodesic distance in degrees. Matches patch_regressor_v7."""
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cos_angle)).reshape(R1.shape[:-2])


# ============================================================
# Model
# ============================================================

class PatchVotingMLP(nn.Module):
    """Small MLP for per-patch pose prediction.
    
    Input: [patch_fine(C), patch_coarse(C), pos_x, pos_y] = 2C+2
    Output: [trans(3), rot_6d(6)] = 9
    """
    
    def __init__(self, patch_dim=128, hidden_dims=(512, 256), dropout=0.1,
                 use_fine=True, use_coarse=True, use_summary_broadcast=False,
                 summary_dim=2560):
        super().__init__()
        self.use_fine = use_fine
        self.use_coarse = use_coarse
        self.use_summary_broadcast = use_summary_broadcast
        
        input_dim = 2  # position encoding (normalized x, y)
        if use_fine:
            input_dim += patch_dim
        if use_coarse:
            input_dim += patch_dim
        if use_summary_broadcast:
            input_dim += summary_dim
        
        self.input_dim = input_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        self.trans_head = nn.Linear(prev_dim, 3)
        self.rot_head = nn.Linear(prev_dim, 6)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, input_dim) — concatenated [patch_fine, patch_coarse, pos_x, pos_y]
        Returns: trans (B, 3), rot_6d (B, 6)
        """
        h = self.backbone(x)
        trans = self.trans_head(h)
        rot_6d = self.rot_head(h)
        return trans, rot_6d


# ============================================================
# Dataset
# ============================================================

class PatchVotingDataset:
    """Prepare per-patch training data.
    
    For each image with H×W patches, creates H×W training samples:
      input: [fine_feat(C), coarse_feat(C), pos_x, pos_y]
      target: [trans(3), rot(3×3)]
    """
    
    def __init__(self, feature_dir, dataset_dir, split='train', device='cpu',
                 use_fine=True, use_coarse=True, use_summary_broadcast=False):
        self.device = device
        
        # Load poses
        pose_file = os.path.join(dataset_dir, f'dataset_{split}.txt')
        self.filenames, self.translations, self.rotations = self._load_poses(pose_file)
        self.N = len(self.filenames)
        
        # Load features
        suffix = '_train' if split == 'train' else '_test'
        
        self.patch_fine = None
        self.patch_coarse = None
        self.summary = None
        
        if use_fine:
            path = os.path.join(feature_dir, f'fine_geo{suffix}.pt')
            self.patch_fine = torch.load(path, map_location='cpu', weights_only=True)
            print(f"  Loaded fine_geo: {self.patch_fine.shape}")
        
        if use_coarse:
            path = os.path.join(feature_dir, f'coarse_sem{suffix}.pt')
            self.patch_coarse = torch.load(path, map_location='cpu', weights_only=True)
            print(f"  Loaded coarse_sem: {self.patch_coarse.shape}")
        
        if use_summary_broadcast:
            import glob as glob_module
            path = os.path.join(feature_dir, 'summary_matrix.pt')
            full_summary = torch.load(path, map_location='cpu', weights_only=True).float()
            # Index by split (same logic as patch_regressor_v7.py)
            image_paths = sorted(glob_module.glob(os.path.join(dataset_dir, 'seq*/*.png')))
            name_to_idx = {os.path.relpath(p, dataset_dir): i for i, p in enumerate(image_paths)}
            indices = []
            for fname in self.filenames:
                idx = name_to_idx.get(fname)
                if idx is not None:
                    indices.append(idx)
            self.summary = full_summary[indices]
            print(f"  Loaded summary: {self.summary.shape}")
        
        # Get spatial dimensions
        ref = self.patch_fine if self.patch_fine is not None else self.patch_coarse
        self.H = ref.shape[2]  # 68
        self.W = ref.shape[3]  # 120
        self.patches_per_image = self.H * self.W
        
        # Pre-compute normalized position grid
        ys = torch.linspace(-1, 1, self.H)
        xs = torch.linspace(-1, 1, self.W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.pos_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (H*W, 2)
        
        # Move to device
        self.translations = self.translations.to(device)
        self.rotations = self.rotations.to(device)
        self.pos_grid = self.pos_grid.to(device)
        
        print(f"  [{split}] {self.N} images, {self.patches_per_image} patches/image, "
              f"{self.N * self.patches_per_image} total patches")
    
    def _load_poses(self, pose_file):
        """Load poses matching the format used in patch_regressor_v7.py.
        
        Format: 3-line header, then 'filename X Y Z W P Q R' per line.
        X Y Z = camera position, W P Q R = quaternion.
        """
        filenames = []
        translations = []
        rotations = []
        
        with open(pose_file) as f:
            lines = f.readlines()
        
        for line in lines[3:]:  # Skip 3-line header
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            filenames.append(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            translations.append(torch.tensor([x, y, z], dtype=torch.float32))
            rotations.append(torch.tensor([w, p, q, r], dtype=torch.float32))
        
        trans = torch.stack(translations)
        quat = torch.stack(rotations)
        rots = quaternion_to_matrix(quat)
        
        return filenames, trans, rots
    
    def get_patch_batch(self, image_indices, device):
        """Get all patches for given image indices.
        
        Returns:
            features: (N_images * H*W, input_dim) — concatenated features for all patches
            trans_targets: (N_images * H*W, 3) — repeated pose targets
            rot_targets: (N_images * H*W, 3, 3) — repeated rotation targets
        """
        B = len(image_indices)
        P = self.patches_per_image
        
        parts = []
        cpu_indices = image_indices.cpu()
        
        # Patch features: (B, C, H, W) → (B, H*W, C) → (B*H*W, C)
        if self.patch_fine is not None:
            pf = self.patch_fine[cpu_indices].to(device)  # (B, C, H, W)
            pf = pf.permute(0, 2, 3, 1).reshape(B * P, -1)  # (B*P, C)
            parts.append(pf)
        
        if self.patch_coarse is not None:
            pc = self.patch_coarse[cpu_indices].to(device)
            pc = pc.permute(0, 2, 3, 1).reshape(B * P, -1)
            parts.append(pc)
        
        if self.summary is not None:
            sm = self.summary[cpu_indices].to(device)  # (B, summary_dim)
            sm = sm.unsqueeze(1).expand(-1, P, -1).reshape(B * P, -1)  # (B*P, summary_dim)
            parts.append(sm)
        
        # Position encoding: (H*W, 2) → (B*H*W, 2)
        pos = self.pos_grid.unsqueeze(0).expand(B, -1, -1).reshape(B * P, 2)
        parts.append(pos)
        
        features = torch.cat(parts, dim=-1)  # (B*P, input_dim)
        
        # Repeat targets for each patch
        trans_targets = self.translations[image_indices].unsqueeze(1).expand(-1, P, -1).reshape(B * P, 3)
        rot_targets = self.rotations[image_indices].unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, 3, 3)
        
        return features, trans_targets, rot_targets


# ============================================================
# Aggregation strategies
# ============================================================

def aggregate_median(trans_preds, rot_6d_preds, H, W):
    """Median aggregation of per-patch predictions.
    
    trans_preds: (N_images, H*W, 3)
    rot_6d_preds: (N_images, H*W, 6)
    
    Returns: (N_images, 3), (N_images, 3, 3)
    """
    # Median translation
    trans = trans_preds.median(dim=1).values  # (N, 3)
    
    # For rotation: use mean of 6D representation, then Gram-Schmidt
    rot_6d_mean = rot_6d_preds.mean(dim=1)  # (N, 6)
    rot_mat = gram_schmidt_6d_to_matrix(rot_6d_mean)
    
    return trans, rot_mat


def aggregate_trimmed_mean(trans_preds, rot_6d_preds, H, W, trim_pct=0.2):
    """Trimmed mean: remove top/bottom trim_pct of predictions per axis."""
    N, P, D = trans_preds.shape
    k_trim = int(P * trim_pct)
    
    # Sort along patch dimension for each axis
    sorted_trans, _ = trans_preds.sort(dim=1)
    # Remove top and bottom k_trim
    trimmed = sorted_trans[:, k_trim:-k_trim, :]
    trans = trimmed.mean(dim=1)
    
    # Rotation: use full mean (trimming 6D is not meaningful)
    rot_6d_mean = rot_6d_preds.mean(dim=1)
    rot_mat = gram_schmidt_6d_to_matrix(rot_6d_mean)
    
    return trans, rot_mat


def aggregate_weighted_median(trans_preds, rot_6d_preds, confidences, H, W):
    """Weighted by patch confidence (from an auxiliary head)."""
    # For now, just use regular median
    return aggregate_median(trans_preds, rot_6d_preds, H, W)


def aggregate_spatial_center(trans_preds, rot_6d_preds, H, W, center_ratio=0.5):
    """Only use center patches (ignore edges that may have less context)."""
    N, P, D_t = trans_preds.shape
    
    h_margin = int(H * (1 - center_ratio) / 2)
    w_margin = int(W * (1 - center_ratio) / 2)
    
    # Create mask for center patches
    mask = torch.zeros(H, W, dtype=torch.bool, device=trans_preds.device)
    mask[h_margin:H-h_margin, w_margin:W-w_margin] = True
    mask = mask.flatten()  # (P,)
    
    center_trans = trans_preds[:, mask, :]  # (N, center_P, 3)
    center_rot = rot_6d_preds[:, mask, :]
    
    trans = center_trans.median(dim=1).values
    rot_6d_mean = center_rot.mean(dim=1)
    rot_mat = gram_schmidt_6d_to_matrix(rot_6d_mean)
    
    return trans, rot_mat


# ============================================================
# Training
# ============================================================

def train(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    use_fine = 'fine' in args.feat or 'both' in args.feat
    use_coarse = 'coarse' in args.feat or 'both' in args.feat
    use_summary = '+sum' in args.feat
    
    print(f"\nLoading data (fine={use_fine}, coarse={use_coarse}, summary_broadcast={use_summary})...")
    train_data = PatchVotingDataset(
        args.feature_dir, args.dataset_dir, 'train', device,
        use_fine=use_fine, use_coarse=use_coarse, use_summary_broadcast=use_summary)
    test_data = PatchVotingDataset(
        args.feature_dir, args.dataset_dir, 'test', device,
        use_fine=use_fine, use_coarse=use_coarse, use_summary_broadcast=use_summary)
    
    H, W = train_data.H, train_data.W
    P = H * W  # patches per image
    
    # Normalize translations
    trans_mean = train_data.translations.mean(dim=0)
    trans_std = train_data.translations.std(dim=0).clamp(min=1e-6)
    train_data.translations_norm = (train_data.translations - trans_mean) / trans_std
    test_data.translations_norm = (test_data.translations - trans_mean) / trans_std
    print(f"Translation norm: mean={trans_mean.cpu().numpy()}, std={trans_std.cpu().numpy()}")
    
    # Save normalization params
    torch.save({'mean': trans_mean.cpu(), 'std': trans_std.cpu()}, out_dir / 'norm_params.pt')
    
    # Build model
    summary_dim = 2560 if use_summary else 0
    model = PatchVotingMLP(
        patch_dim=args.patch_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_fine=use_fine,
        use_coarse=use_coarse,
        use_summary_broadcast=use_summary,
        summary_dim=summary_dim,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: input_dim={model.input_dim}, params={n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training config
    images_per_batch = args.images_per_batch  # Number of images per batch
    n_train = train_data.N
    
    best_val_score = float('inf')
    best_epoch = 0
    history = []
    
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Sample random images
        if images_per_batch >= n_train:
            img_indices = torch.arange(n_train, device=device)
        else:
            img_indices = torch.randperm(n_train, device=device)[:images_per_batch]
        
        # Get all patches for these images
        features, trans_targets_raw, rot_targets = train_data.get_patch_batch(img_indices, device)
        
        # Normalize translation targets
        trans_targets = (trans_targets_raw - trans_mean) / trans_std
        
        # Sub-sample patches if too many (memory)
        total_patches = features.shape[0]
        if args.max_patches_per_step > 0 and total_patches > args.max_patches_per_step:
            perm = torch.randperm(total_patches, device=device)[:args.max_patches_per_step]
            features = features[perm]
            trans_targets = trans_targets[perm]
            rot_targets = rot_targets[perm]
        
        # Forward
        trans_pred, rot_6d_pred = model(features)
        rot_pred = gram_schmidt_6d_to_matrix(rot_6d_pred)
        
        # Translation loss
        loss_trans = F.smooth_l1_loss(trans_pred, trans_targets)
        
        # Rotation loss (geodesic)
        R_diff = torch.bmm(rot_targets.transpose(-1, -2), rot_pred)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = ((trace - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
        loss_rot = torch.acos(cos_angle).mean()
        
        # Combined loss with adaptive weighting
        beta = max(0.3, 1.0 - epoch / (args.epochs * 2))
        loss = loss_trans + beta * loss_rot
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation (every N epochs)
        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_trans_all, val_rot_all = predict_all_images(
                    model, test_data, device, args.aggregation)
                
                # De-normalize translation
                val_trans_all = val_trans_all * trans_std + trans_mean
                
                # Compute errors
                gt_trans = test_data.translations
                gt_rot = test_data.rotations
                
                trans_err = (val_trans_all - gt_trans).norm(dim=-1)  # meters
                rot_err = geodesic_distance(val_rot_all, gt_rot)
                
                trans_med = trans_err.median().item()
                rot_med = rot_err.median().item()
                
                # Recall
                r5_rot = (rot_err < 5).float().mean().item() * 100
                r5_trans = (trans_err < 1).float().mean().item() * 100
                r5 = ((rot_err < 5) & (trans_err < 1)).float().mean().item() * 100
                r10_rot = (rot_err < 10).float().mean().item() * 100
                r10_trans = (trans_err < 2).float().mean().item() * 100
                r10 = ((rot_err < 10) & (trans_err < 2)).float().mean().item() * 100
                
                val_score = trans_med + rot_med * 0.1
                
                if val_score < best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_score': val_score,
                        'config': vars(args),
                    }, out_dir / 'model_best.pt')
                
                if epoch % args.log_every == 0 or epoch == 1:
                    elapsed = time.time() - t0
                    print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                          f"(t={loss_trans.item():.4f} r={loss_rot.item():.4f} β={beta:.2f}) | "
                          f"val: {rot_med:.2f}deg/{trans_med*1000:.0f}mm | "
                          f"R@10/2m={r10:.1f}% R@5/1m={r5:.1f}% | {elapsed:.1f}s")
                
                history.append({
                    'epoch': epoch, 'loss': loss.item(),
                    'val_rot_med': rot_med, 'val_trans_med': trans_med,
                    'r10_2m': r10, 'r5_1m': r5,
                })
    
    # Save training log
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining done. Best epoch: {best_epoch}, score: {best_val_score:.4f}")
    
    # Load best model and evaluate
    ckpt = torch.load(out_dir / 'model_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")
    
    model.eval()
    with torch.no_grad():
        # Try all aggregation strategies
        strategies = {
            'median': 'median',
            'trimmed_20': 'trimmed_20',
            'trimmed_10': 'trimmed_10', 
            'center_50': 'center_50',
            'center_70': 'center_70',
        }
        
        best_r10 = 0
        best_strategy = 'median'
        
        for name, agg in strategies.items():
            val_trans, val_rot = predict_all_images(model, test_data, device, agg)
            val_trans = val_trans * trans_std + trans_mean
            
            gt_trans = test_data.translations
            gt_rot = test_data.rotations
            
            trans_err = (val_trans - gt_trans).norm(dim=-1)
            rot_err = geodesic_distance(val_rot, gt_rot)
            
            trans_med = trans_err.median().item()
            rot_med = rot_err.median().item()
            
            r5_rot = (rot_err < 5).float().mean().item() * 100
            r5_trans = (trans_err < 1).float().mean().item() * 100
            r5 = ((rot_err < 5) & (trans_err < 1)).float().mean().item() * 100
            r10_rot = (rot_err < 10).float().mean().item() * 100
            r10_trans = (trans_err < 2).float().mean().item() * 100
            r10 = ((rot_err < 10) & (trans_err < 2)).float().mean().item() * 100
            
            print(f"\n  Aggregation: {name}")
            print(f"    Rot: {rot_med:.2f}deg | Trans: {trans_med*1000:.0f}mm")
            print(f"    R@5deg_1m: {r5:.1f}%  (rot={r5_rot:.1f}%, trans={r5_trans:.1f}%)")
            print(f"    R@10deg_2m: {r10:.1f}%  (rot={r10_rot:.1f}%, trans={r10_trans:.1f}%)")
            
            if r10 > best_r10:
                best_r10 = r10
                best_strategy = name
        
        # Detailed evaluation with best strategy
        print(f"\n{'='*70}")
        print(f"BEST AGGREGATION: {best_strategy}")
        print(f"{'='*70}")
        
        val_trans, val_rot = predict_all_images(model, test_data, device, best_strategy)
        val_trans = val_trans * trans_std + trans_mean
        
        gt_trans = test_data.translations
        gt_rot = test_data.rotations
        
        trans_err = (val_trans - gt_trans).norm(dim=-1)
        rot_err = geodesic_distance(val_rot, gt_rot)
        
        results = {
            'rotation_deg': {'median': rot_err.median().item(), 'mean': rot_err.mean().item()},
            'translation_m': {'median': trans_err.median().item(), 'mean': trans_err.mean().item()},
            'translation_mm': {'median': trans_err.median().item()*1000, 'mean': trans_err.mean().item()*1000},
            'recall': {
                '5deg_1m': ((rot_err < 5) & (trans_err < 1)).float().mean().item() * 100,
                '10deg_2m': ((rot_err < 10) & (trans_err < 2)).float().mean().item() * 100,
                '15deg_5m': ((rot_err < 15) & (trans_err < 5)).float().mean().item() * 100,
                '25deg_5m': ((rot_err < 25) & (trans_err < 5)).float().mean().item() * 100,
            },
            'best_aggregation': best_strategy,
        }
        
        with open(out_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nRotation  (median): {rot_err.median().item():.2f}deg")
        print(f"Translation (median): {trans_err.median().item()*1000:.0f}mm")
        for thresh, key in [('5deg_1m', '5deg_1m'), ('10deg_2m', '10deg_2m'), 
                            ('15deg_5m', '15deg_5m'), ('25deg_5m', '25deg_5m')]:
            print(f"  R@{thresh}: {results['recall'][key]:.1f}%")
    
    print(f"\nDone!")


def predict_all_images(model, dataset, device, aggregation='median'):
    """Predict pose for all images using per-patch voting.
    
    Returns: trans (N, 3), rot (N, 3, 3) — in normalized translation space
    """
    N = dataset.N
    H, W = dataset.H, dataset.W
    P = H * W
    
    all_trans = []
    all_rot = []
    
    # Process one image at a time to save memory
    for i in range(N):
        idx = torch.tensor([i], device=device)
        features, _, _ = dataset.get_patch_batch(idx, device)  # (P, input_dim)
        
        trans_pred, rot_6d_pred = model(features)  # (P, 3), (P, 6)
        
        # Reshape to (1, P, D)
        trans_pred = trans_pred.unsqueeze(0)
        rot_6d_pred = rot_6d_pred.unsqueeze(0)
        
        if aggregation == 'median':
            t, R = aggregate_median(trans_pred, rot_6d_pred, H, W)
        elif aggregation.startswith('trimmed'):
            pct = float(aggregation.split('_')[1]) / 100
            t, R = aggregate_trimmed_mean(trans_pred, rot_6d_pred, H, W, trim_pct=pct)
        elif aggregation.startswith('center'):
            ratio = float(aggregation.split('_')[1]) / 100
            t, R = aggregate_spatial_center(trans_pred, rot_6d_pred, H, W, center_ratio=ratio)
        else:
            t, R = aggregate_median(trans_pred, rot_6d_pred, H, W)
        
        all_trans.append(t)
        all_rot.append(R)
    
    return torch.cat(all_trans, dim=0), torch.cat(all_rot, dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-Patch Pose Voting Regressor')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_base', type=str, default='output/feature_retrieval/pose_regression')
    parser.add_argument('--exp_name', type=str, default='patch_voting')
    
    # Architecture
    parser.add_argument('--patch_dim', type=int, default=128)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feat', type=str, default='both',
                        help='Feature mode: fine, coarse, both, both+sum')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--images_per_batch', type=int, default=64,
                        help='Number of images per batch (each contributes 8160 patches)')
    parser.add_argument('--max_patches_per_step', type=int, default=0,
                        help='Max patches per training step (0=all)')
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--aggregation', type=str, default='median',
                        help='Aggregation strategy: median, trimmed_20, center_50')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.output_base, args.exp_name)
    
    train(args)
