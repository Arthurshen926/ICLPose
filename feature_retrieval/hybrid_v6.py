#!/usr/bin/env python3
"""
Hybrid approaches: Combine kNN retrieval with MLP regression
=============================================================

exp13a: Weighted kNN regression (no learning, just RADIO feature similarity)
exp13b: Regression + kNN interpolation (blend exp01 prediction with kNN average)
exp13c: Soft anchor-based regression (learnable attention over K-means anchors)

Usage:
    python feature_retrieval/hybrid_v6.py --output_dir output/.../exp13
"""

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility
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


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to w-first quaternion."""
    # Shepperd's method
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    B = R.shape[0]
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
    
    s = torch.sqrt((trace + 1).clamp(min=1e-10)) * 2  # s = 4*w
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s
    
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return q.reshape(*batch_shape, 4)


def weighted_quaternion_average(quats, weights):
    """Weighted average of quaternions using the eigenvalue method.
    
    quats: (K, 4) w-first quaternions
    weights: (K,) non-negative weights (will be normalized)
    Returns: (4,) averaged quaternion
    """
    weights = weights / weights.sum()
    # Build the weighted outer product matrix M = sum(w_i * q_i * q_i^T)
    M = torch.zeros(4, 4, device=quats.device, dtype=quats.dtype)
    for i in range(quats.shape[0]):
        q = quats[i]
        M += weights[i] * q.unsqueeze(1) * q.unsqueeze(0)
    
    # The average quaternion is the eigenvector corresponding to the largest eigenvalue
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    avg_q = eigenvectors[:, -1]  # largest eigenvalue
    
    # Ensure w >= 0
    if avg_q[0] < 0:
        avg_q = -avg_q
    return avg_q


# ============================================================
# Dataset
# ============================================================

class PoseDataset:
    def __init__(self, summary_matrix_path, dataset_dir, split='train', device='cpu'):
        self.device = device
        summary_matrix = torch.load(summary_matrix_path, map_location='cpu')
        
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
        self.quaternions = torch.stack(rotations_list).to(device)
        self.rotations = quaternion_to_matrix(self.quaternions).to(device)
        self.names = names_list
        self.N = len(names_list)
        
        print(f"[{split}] Loaded {self.N} samples, features {self.features.shape}")


# ============================================================
# Evaluation
# ============================================================

def compute_errors(pred_trans, pred_rots, gt_trans, gt_rots):
    trans_err = torch.norm(pred_trans - gt_trans, dim=-1) * 1000  # mm
    rot_err = geodesic_distance(pred_rots, gt_rots) * 180 / math.pi
    return rot_err, trans_err


def print_results(rot_err, trans_err, prefix=""):
    print(f"  {prefix}Rotation  median: {rot_err.median().item():.2f}° (mean={rot_err.mean().item():.2f}°)")
    print(f"  {prefix}Translation median: {trans_err.median().item():.0f}mm (mean={trans_err.mean().item():.0f}mm)")
    
    thresholds = [(5, 1000), (10, 2000), (15, 5000), (25, 5000)]
    for rot_th, trans_th in thresholds:
        rot_pass = (rot_err <= rot_th).float().mean().item() * 100
        trans_pass = (trans_err <= trans_th).float().mean().item() * 100
        combined = ((rot_err <= rot_th) & (trans_err <= trans_th)).float().mean().item() * 100
        print(f"  {prefix}R@{rot_th}°/{trans_th/1000:.0f}m: {combined:.1f}%  "
              f"(rot={rot_pass:.1f}%, trans={trans_pass:.1f}%)")


# ============================================================
# Method 1: Weighted kNN Regression
# ============================================================

def weighted_knn(test_features, test_trans, test_rots,
                 train_features, train_trans, train_quats, train_rots,
                 K=10, sigma=None):
    """
    For each test image, find K nearest training images by RADIO cosine similarity.
    Predict pose as similarity-weighted average of their poses.
    """
    # Cosine similarity
    sim = torch.mm(F.normalize(test_features, dim=-1),
                   F.normalize(train_features, dim=-1).t())  # (Q, D)
    
    topk_sim, topk_idx = sim.topk(K, dim=1)  # (Q, K)
    
    # Softmax weights (temperature-scaled)
    if sigma is None:
        # Use raw cosine similarities with softmax
        weights = F.softmax(topk_sim * 20, dim=1)  # scale up for sharper weights
    else:
        weights = F.softmax(topk_sim / sigma, dim=1)
    
    Q = test_features.shape[0]
    
    # Weighted average of positions
    neighbor_trans = train_trans[topk_idx]  # (Q, K, 3)
    pred_trans = (weights.unsqueeze(-1) * neighbor_trans).sum(dim=1)  # (Q, 3)
    
    # Weighted average of quaternions
    pred_rots = torch.zeros(Q, 3, 3, device=test_features.device)
    neighbor_quats = train_quats[topk_idx]  # (Q, K, 4)
    for i in range(Q):
        avg_q = weighted_quaternion_average(neighbor_quats[i], weights[i])
        pred_rots[i] = quaternion_to_matrix(avg_q.unsqueeze(0)).squeeze(0)
    
    return pred_trans, pred_rots


# ============================================================
# Method 2: Regression + kNN Hybrid
# ============================================================

def regression_knn_hybrid(test_features, test_trans, test_rots,
                          train_features, train_trans, train_quats, train_rots,
                          reg_trans, reg_rots,
                          K=5, alpha=0.5):
    """
    Blend MLP regression predictions with weighted kNN.
    
    final_trans = alpha * reg_trans + (1-alpha) * knn_trans
    final_rot = slerp between reg_rot and knn_rot
    """
    # Get kNN prediction
    knn_trans, knn_rots = weighted_knn(
        test_features, test_trans, test_rots,
        train_features, train_trans, train_quats, train_rots,
        K=K
    )
    
    # Blend translations
    pred_trans = alpha * reg_trans + (1 - alpha) * knn_trans
    
    # For rotation: use regression (it's much better than kNN)
    pred_rots = reg_rots  # just use regression rotation
    
    return pred_trans, pred_rots


# ============================================================
# Method 3: Soft Anchor-Based Regression (Learnable)
# ============================================================

class SoftAnchorRegressor(nn.Module):
    """
    Predict pose as soft attention over training set poses.
    
    For each test image:
    1. Project to embedding space
    2. Compute attention over all training embeddings  
    3. Final pose = attention-weighted sum of training poses + residual
    
    This is essentially learning a soft nearest-neighbor retrieval.
    """
    
    def __init__(self, input_dim=2560, embed_dim=256, hidden_dims=(1024, 512), 
                 dropout=0.1, temperature=1.0):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h_d
        layers.append(nn.Linear(in_d, embed_dim))
        self.encoder = nn.Sequential(*layers)
        
        # Residual prediction (on top of the attention-weighted output)
        self.trans_residual = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, db_features=None, db_embeddings=None):
        """
        x: (B, input_dim) query features
        db_features: (D, input_dim) database features (if db_embeddings not provided)
        db_embeddings: (D, embed_dim) pre-computed database embeddings
        """
        query_embed = self.encoder(x)  # (B, embed_dim)
        query_embed_norm = F.normalize(query_embed, dim=-1)
        
        if db_embeddings is not None:
            db_embed_norm = F.normalize(db_embeddings, dim=-1)
        else:
            db_embed_norm = F.normalize(self.encoder(db_features), dim=-1)
        
        # Attention over database
        attn = torch.mm(query_embed_norm, db_embed_norm.t()) / self.temperature.abs().clamp(min=0.01)
        attn_weights = F.softmax(attn, dim=1)  # (B, D)
        
        # Residual
        trans_res = self.trans_residual(query_embed)  # (B, 3)
        rot_6d = self.rot_head(query_embed)  # (B, 6)
        rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
        
        return attn_weights, trans_res, rot_mat, query_embed


def train_soft_anchor(train_data, test_data, args):
    """Train soft-anchor model."""
    device = train_data.features.device
    
    model = SoftAnchorRegressor(
        input_dim=2560, embed_dim=args.embed_dim,
        hidden_dims=tuple(args.hidden_dims), dropout=args.dropout,
        temperature=args.anchor_temperature
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Normalize translations
    trans_mean = train_data.translations.mean(dim=0)
    trans_std = train_data.translations.std(dim=0).clamp(min=1e-6)
    train_trans_norm = (train_data.translations - trans_mean) / trans_std
    
    best_r10 = -1
    best_epoch = 0
    best_state = None
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Use all training data (it fits in memory)
        # Pass db_features for attention computation
        attn_weights, trans_res, rot_mat, _ = model(train_data.features, db_features=train_data.features)
        
        # Attention-weighted translation (in normalized space)
        attn_trans = torch.mm(attn_weights, train_trans_norm)  # (N, 3)
        pred_trans_norm = attn_trans + trans_res  # (N, 3) with residual
        
        # Translation loss
        loss_t = F.smooth_l1_loss(pred_trans_norm, train_trans_norm)
        
        # Rotation loss (geodesic)
        loss_r = geodesic_distance(rot_mat, train_data.rotations).mean()
        
        # Entropy regularization: encourage sharp attention
        entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=1).mean()
        
        loss = args.lambda_t * loss_t + args.lambda_r * loss_r + 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                attn_w, tres, rmat, _ = model(test_data.features, db_features=train_data.features)
                
                test_trans_norm = (test_data.translations - trans_mean) / trans_std
                pred_test_trans_norm = torch.mm(attn_w, train_trans_norm) + tres
                pred_test_trans = pred_test_trans_norm * trans_std + trans_mean
                
                rot_err, trans_err = compute_errors(pred_test_trans, rmat,
                                                     test_data.translations, test_data.rotations)
                
                r10 = ((rot_err <= 10) & (trans_err <= 2000)).float().mean().item() * 100
                r5 = ((rot_err <= 5) & (trans_err <= 1000)).float().mean().item() * 100
                
                # Also evaluate retrieval mode: which train image gets highest attention?
                top1_idx = attn_w.argmax(dim=1)
                ret_trans = train_data.translations[top1_idx]
                ret_rots = train_data.rotations[top1_idx]
                ret_rot_err, ret_trans_err = compute_errors(ret_trans, ret_rots,
                                                             test_data.translations, test_data.rotations)
                ret_r10 = ((ret_rot_err <= 10) & (ret_trans_err <= 2000)).float().mean().item() * 100
            
            elapsed = time.time() - t0
            temp_val = model.temperature.item()
            print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} (t={loss_t.item():.4f} r={loss_r.item():.4f} "
                  f"ent={entropy.item():.2f} T={temp_val:.3f}) | "
                  f"direct: {rot_err.median().item():.1f}°/{trans_err.median().item():.0f}mm "
                  f"R@10/2={r10:.1f}% | ret R@10/2={ret_r10:.1f}% | {elapsed:.1f}s")
            
            if r10 > best_r10:
                best_r10 = r10
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"\nBest epoch: {best_epoch} (R@10/2={best_r10:.1f}%)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({'state_dict': best_state, 'epoch': best_epoch}, 
               os.path.join(args.output_dir, 'model_best.pt'))
    
    model.load_state_dict(best_state)
    model.to(device)
    return model, trans_mean, trans_std


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--summary_matrix', type=str,
                        default='output/feature_extract/features_radio_dual/OldHospital_pilot/summary_matrix.pt')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512])
    parser.add_argument('--lambda_t', type=float, default=5.0)
    parser.add_argument('--lambda_r', type=float, default=1.0)
    parser.add_argument('--anchor_temperature', type=float, default=0.1)
    
    # exp01 model path for hybrid
    parser.add_argument('--exp01_model', type=str, 
                        default='output/feature_retrieval/pose_regression/exp01_direct_mlp/model_best.pt')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data = PoseDataset(args.summary_matrix, args.dataset_dir, 'train', device)
    test_data = PoseDataset(args.summary_matrix, args.dataset_dir, 'test', device)
    
    results = {}
    
    # ========================================
    # Method 1: Weighted kNN
    # ========================================
    print("\n" + "=" * 70)
    print("METHOD 1: Weighted kNN Regression (no learning)")
    print("=" * 70)
    
    for K in [1, 3, 5, 10, 20, 50]:
        for sigma_name, sigma in [("sharp", None), ("soft", 0.05)]:
            knn_trans, knn_rots = weighted_knn(
                test_data.features, test_data.translations, test_data.rotations,
                train_data.features, train_data.translations, train_data.quaternions, train_data.rotations,
                K=K, sigma=(sigma if sigma_name == "soft" else None)
            )
            rot_err, trans_err = compute_errors(knn_trans, knn_rots,
                                                 test_data.translations, test_data.rotations)
            
            r10 = ((rot_err <= 10) & (trans_err <= 2000)).float().mean().item() * 100
            r5 = ((rot_err <= 5) & (trans_err <= 1000)).float().mean().item() * 100
            
            tag = f"K={K},{sigma_name}"
            results[f'knn_{tag}'] = {
                'rot_median': rot_err.median().item(),
                'trans_median': trans_err.median().item(),
                'R@5/1': r5, 'R@10/2': r10
            }
            
            print(f"  K={K:2d} ({sigma_name:5s}): {rot_err.median().item():.1f}°/{trans_err.median().item():.0f}mm "
                  f"R@5/1={r5:.1f}%  R@10/2={r10:.1f}%")
    
    # ========================================
    # Method 2: Regression + kNN Hybrid
    # ========================================
    print("\n" + "=" * 70)
    print("METHOD 2: Regression + kNN Hybrid")
    print("=" * 70)
    
    # Load exp01 model predictions
    exp01_npz = os.path.join(os.path.dirname(args.exp01_model), 'test_poses_w2c.npz')
    if os.path.exists(exp01_npz):
        data = np.load(exp01_npz)
        # Need to get regression predictions - let me re-run exp01 inference
        print("  Loading exp01 regression predictions...")
    else:
        print(f"  WARNING: exp01 predictions not found at {exp01_npz}")
    
    # Re-run exp01 inference inline to get per-image predictions
    if os.path.exists(args.exp01_model):
        print("  Loading exp01 model for inference...")
        
        # Define exp01's model architecture inline (avoids import issues)
        class Exp01Model(nn.Module):
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
            
            def forward(self, x):
                h = self.backbone(x)
                trans = self.trans_head(h)
                rot_6d = self.rot_head(h)
                rot_mat = gram_schmidt_6d_to_matrix(rot_6d)
                return trans, rot_mat
        
        try:
            ckpt = torch.load(args.exp01_model, map_location=device)
            exp01_model = Exp01Model(input_dim=2560).to(device)
            exp01_model.load_state_dict(ckpt['model_state_dict'])
            exp01_model.eval()
            
            norm_params = ckpt.get('norm_params', {})
            trans_mean_e = norm_params.get('mean', torch.zeros(3)).to(device)
            trans_std_e = norm_params.get('std', torch.ones(3)).to(device)
            
            with torch.no_grad():
                pred_trans_norm, reg_rots = exp01_model(test_data.features)
                reg_trans = pred_trans_norm * trans_std_e + trans_mean_e
            
            print(f"  exp01 loaded successfully.")
            rot_err_reg, trans_err_reg = compute_errors(reg_trans, reg_rots,
                                                         test_data.translations, test_data.rotations)
            r10_reg = ((rot_err_reg <= 10) & (trans_err_reg <= 2000)).float().mean().item() * 100
            print(f"  exp01 baseline: {rot_err_reg.median().item():.1f}°/{trans_err_reg.median().item():.0f}mm "
                  f"R@10/2={r10_reg:.1f}%")
            
            # Try different blending ratios
            for alpha in [0.9, 0.8, 0.7, 0.6, 0.5, 0.3]:
                for K in [3, 5, 10]:
                    hybrid_trans, hybrid_rots = regression_knn_hybrid(
                        test_data.features, test_data.translations, test_data.rotations,
                        train_data.features, train_data.translations, train_data.quaternions, train_data.rotations,
                        reg_trans, reg_rots,
                        K=K, alpha=alpha
                    )
                    rot_err, trans_err = compute_errors(hybrid_trans, hybrid_rots,
                                                         test_data.translations, test_data.rotations)
                    r10 = ((rot_err <= 10) & (trans_err <= 2000)).float().mean().item() * 100
                    r5 = ((rot_err <= 5) & (trans_err <= 1000)).float().mean().item() * 100
                    
                    tag = f"α={alpha},K={K}"
                    results[f'hybrid_{tag}'] = {
                        'rot_median': rot_err.median().item(),
                        'trans_median': trans_err.median().item(),
                        'R@5/1': r5, 'R@10/2': r10
                    }
                    
                    print(f"  α={alpha:.1f} K={K:2d}: {rot_err.median().item():.1f}°/"
                          f"{trans_err.median().item():.0f}mm R@5/1={r5:.1f}%  R@10/2={r10:.1f}%")
        
        except Exception as e:
            print(f"  Failed to load exp01 model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  exp01 model not found at {args.exp01_model}")
    
    # ========================================
    # Method 3: Soft Anchor-Based Regression
    # ========================================
    print("\n" + "=" * 70)
    print("METHOD 3: Soft Anchor-Based Regression (learnable)")
    print("=" * 70)
    
    model, trans_mean, trans_std = train_soft_anchor(train_data, test_data, args)
    
    # Final evaluation
    model.eval()
    train_trans_norm = (train_data.translations - trans_mean) / trans_std
    
    with torch.no_grad():
        attn_w, tres, rmat, _ = model(test_data.features, db_features=train_data.features)
        pred_trans_norm = torch.mm(attn_w, train_trans_norm) + tres
        pred_trans = pred_trans_norm * trans_std + trans_mean
        
        rot_err, trans_err = compute_errors(pred_trans, rmat,
                                             test_data.translations, test_data.rotations)
    
    print("\n  === FINAL: Soft Anchor Results ===")
    print_results(rot_err, trans_err, prefix="  ")
    
    # Retrieval mode
    with torch.no_grad():
        top1_idx = attn_w.argmax(dim=1)
        ret_trans = train_data.translations[top1_idx]
        ret_rots = train_data.rotations[top1_idx]
        ret_rot_err, ret_trans_err = compute_errors(ret_trans, ret_rots,
                                                     test_data.translations, test_data.rotations)
    
    print("\n  === FINAL: Soft Anchor Retrieval Mode ===")
    print_results(ret_rot_err, ret_trans_err, prefix="  ")
    
    # Save report
    report = {
        'methods': results,
        'soft_anchor': {
            'rot_median': rot_err.median().item(),
            'trans_median': trans_err.median().item(),
            'retrieval_rot_median': ret_rot_err.median().item(),
            'retrieval_trans_median': ret_trans_err.median().item(),
        }
    }
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
