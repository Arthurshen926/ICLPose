#!/usr/bin/env python3
"""
Retrieval-focused approach (v5): Learn embeddings optimized for spatial retrieval
=================================================================================

Instead of regressing poses directly, learn a projection that maps RADIO features
to an embedding space where spatial neighbors are close. At test time, find the
nearest training image and use its pose.

Experiments:
    1. Oracle NN ceiling (best possible single-image retrieval)
    2. Raw RADIO cosine NN (no learning)
    3. Learned projection with NT-Xent contrastive loss
    4. Learned projection with multi-similarity loss
    5. Regression-guided NN (use exp01 predicted position to find nearest train)

Usage:
    python feature_retrieval/retrieval_v5.py --output_dir output/feature_retrieval/pose_regression/exp11_retrieval
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
# Utility: Rotation
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


def geodesic_distance(R1, R2):
    """Geodesic distance between rotation matrices in radians."""
    Rd = torch.bmm(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3).transpose(-1, -2))
    trace = Rd[:, 0, 0] + Rd[:, 1, 1] + Rd[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle).reshape(R1.shape[:-2])


# ============================================================
# Dataset
# ============================================================

class RetrievalDataset:
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

        self.features = torch.stack(features_list).to(device)         # (N, 2560)
        self.translations = torch.stack(translations_list).to(device)  # (N, 3) camera centers
        quat = torch.stack(rotations_list)
        self.rotations = quaternion_to_matrix(quat).to(device)         # (N, 3, 3) R_w2c
        self.names = names_list
        self.N = len(names_list)

        print(f"[{split}] Loaded {self.N} samples, features {self.features.shape}")


# ============================================================
# Models
# ============================================================

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim=2560, embed_dim=256, hidden_dims=(1024, 512), dropout=0.1):
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
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ProjectionHeadWithPose(nn.Module):
    """Projection head that also does pose regression as auxiliary task."""

    def __init__(self, input_dim=2560, embed_dim=256, hidden_dims=(1024, 512), dropout=0.1):
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
        self.backbone = nn.Sequential(*layers)
        self.embed_head = nn.Linear(in_d, embed_dim)
        self.trans_head = nn.Linear(in_d, 3)
        self.rot_head = nn.Linear(in_d, 6)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        embed = F.normalize(self.embed_head(h), dim=-1)
        trans = self.trans_head(h)
        rot_6d = self.rot_head(h)
        return embed, trans, rot_6d


def gram_schmidt_6d_to_matrix(v):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1 = v[..., :3]
    a2 = v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


# ============================================================
# Contrastive Losses
# ============================================================

def build_distance_matrix(positions):
    """Pairwise Euclidean distance matrix for camera centers."""
    # positions: (N, 3)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
    return torch.norm(diff, dim=-1)  # (N, N)


def nt_xent_with_spatial_positives(embeddings, distance_matrix, pos_radius=5.0, temperature=0.07):
    """
    NT-Xent (InfoNCE) loss where positives are images within pos_radius meters.
    
    For each anchor, the positive set is all images within pos_radius.
    Negatives are all other images.
    """
    N = embeddings.shape[0]
    # Cosine similarity matrix (already L2-normalized)
    sim = torch.mm(embeddings, embeddings.t()) / temperature  # (N, N)
    
    # Positive mask: spatial neighbors within radius (excluding self)
    pos_mask = (distance_matrix < pos_radius).float()
    pos_mask.fill_diagonal_(0)
    
    # For numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    
    # Compute log-sum-exp over all non-self entries
    self_mask = torch.eye(N, device=embeddings.device)
    neg_mask = 1.0 - self_mask  # everything except self
    
    exp_sim = torch.exp(sim) * neg_mask
    log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)
    
    # For each anchor, average over its positives
    # Loss = -1/|P(i)| * sum_{p in P(i)} sim(i, p) + log_sum_exp
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    pos_sim_sum = (sim * pos_mask).sum(dim=1)
    
    loss = (-pos_sim_sum / pos_count + log_sum_exp)
    
    # Only average over anchors that have at least one positive
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss[valid].mean()


def multi_similarity_loss(embeddings, distance_matrix, pos_radius=5.0, neg_radius=15.0,
                          alpha=2.0, beta=50.0, lam=1.0):
    """
    Multi-Similarity Loss (Wang et al., CVPR 2019).
    Uses spatial distance to define positive/negative pairs.
    """
    N = embeddings.shape[0]
    sim = torch.mm(embeddings, embeddings.t())  # (N, N)
    
    pos_mask = (distance_matrix < pos_radius).float()
    pos_mask.fill_diagonal_(0)
    neg_mask = (distance_matrix > neg_radius).float()
    
    # For each anchor, find hard positives and hard negatives
    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0
    
    for i in range(N):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        
        pos_sim = sim[i, pos_idx]
        neg_sim = sim[i, neg_idx]
        
        # Hard mining: keep positives harder than easiest negative - margin
        neg_max = neg_sim.max()
        pos_min = pos_sim.min()
        
        hard_pos = pos_sim[pos_sim < neg_max + 0.1]
        hard_neg = neg_sim[neg_sim > pos_min - 0.1]
        
        if len(hard_pos) == 0:
            hard_pos = pos_sim
        if len(hard_neg) == 0:
            hard_neg = neg_sim
        
        # Positive term
        pos_loss = torch.log(1 + torch.exp(-alpha * (hard_pos - lam)).sum()) / alpha
        # Negative term
        neg_loss = torch.log(1 + torch.exp(beta * (hard_neg - lam)).sum()) / beta
        
        loss = loss + pos_loss + neg_loss
        count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss / count


def ap_loss(embeddings, distance_matrix, pos_radius=5.0):
    """
    Average Precision loss: directly optimizes retrieval AP.
    Uses soft ranking approximation.
    """
    N = embeddings.shape[0]
    sim = torch.mm(embeddings, embeddings.t())  # (N, N)
    
    pos_mask = (distance_matrix < pos_radius).float()
    pos_mask.fill_diagonal_(0)
    
    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0
    
    for i in range(N):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            continue
        
        # Similarities to all other images (excluding self)
        sims = sim[i].clone()
        sims[i] = -1e9  # exclude self
        
        # For each positive, count how many negatives rank higher (soft)
        pos_sims = sims[pos_idx]  # (P,)
        all_sims = sims.unsqueeze(0).expand(len(pos_idx), -1)  # (P, N)
        
        # Soft indicator: how many items rank above each positive
        # Using sigmoid approximation
        diff = all_sims - pos_sims.unsqueeze(1)  # (P, N)
        soft_rank = torch.sigmoid(diff * 10.0)  # (P, N)
        # Exclude self (avoid in-place op on sigmoid output for autograd)
        mask = torch.ones(N, device=embeddings.device)
        mask[i] = 0
        soft_rank = soft_rank * mask.unsqueeze(0)
        
        # AP contribution of each positive
        # For each positive p, AP contribution = 1 / rank(p) * (# positives ranked above p + 1)
        ranks = soft_rank.sum(dim=1) + 1  # (P,)
        
        # Count positives ranked above each positive
        pos_ranks = soft_rank[:, pos_idx].sum(dim=1) + 1  # (P,)
        
        precision_at_p = pos_ranks / ranks
        ap = precision_at_p.mean()
        
        loss = loss + (1 - ap)
        count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss / count


# ============================================================
# Evaluation
# ============================================================

def evaluate_retrieval(query_features, query_translations, query_rotations,
                       db_features, db_translations, db_rotations,
                       topk=(1, 3, 5, 10)):
    """
    Evaluate retrieval quality: for each query, find nearest DB image and report pose error.
    
    Returns dict with metrics.
    """
    # Cosine similarity
    sim = torch.mm(F.normalize(query_features, dim=-1),
                   F.normalize(db_features, dim=-1).t())  # (Q, D)
    
    results = {}
    
    for k in topk:
        topk_vals, topk_idx = sim.topk(k, dim=1)  # (Q, k)
        
        # For top-1
        if k == 1:
            ret_idx = topk_idx.squeeze(1)  # (Q,)
            ret_trans = db_translations[ret_idx]  # (Q, 3)
            ret_rots = db_rotations[ret_idx]      # (Q, 3, 3)
            
            trans_err = torch.norm(ret_trans - query_translations, dim=-1) * 1000  # mm
            rot_err = geodesic_distance(ret_rots, query_rotations) * 180 / math.pi  # deg
            
            results['top1_trans_median'] = trans_err.median().item()
            results['top1_trans_mean'] = trans_err.mean().item()
            results['top1_rot_median'] = rot_err.median().item()
            results['top1_rot_mean'] = rot_err.mean().item()
            results['top1_trans_err'] = trans_err
            results['top1_rot_err'] = rot_err
        
        # Oracle top-k: pick the best among top-k
        ret_trans_k = db_translations[topk_idx]  # (Q, k, 3)
        ret_rots_k = db_rotations[topk_idx]      # (Q, k, 3, 3)
        
        trans_err_k = torch.norm(ret_trans_k - query_translations.unsqueeze(1), dim=-1) * 1000  # (Q, k) mm
        
        Q = query_rotations.shape[0]
        rot_err_k = torch.zeros(Q, k, device=query_features.device)
        for j in range(k):
            rot_err_k[:, j] = geodesic_distance(ret_rots_k[:, j], query_rotations) * 180 / math.pi
        
        # Oracle: pick the k-neighbor with smallest translation error
        best_trans_idx = trans_err_k.argmin(dim=1)
        oracle_trans_err = trans_err_k[torch.arange(Q), best_trans_idx]
        oracle_rot_err = rot_err_k[torch.arange(Q), best_trans_idx]
        
        results[f'top{k}_oracle_trans_median'] = oracle_trans_err.median().item()
        results[f'top{k}_oracle_rot_median'] = oracle_rot_err.median().item()
    
    return results


def compute_recall(rot_err, trans_err, thresholds=((5, 1000), (10, 2000), (15, 5000), (25, 5000))):
    """Compute recall at various thresholds."""
    recalls = {}
    for rot_th, trans_th in thresholds:
        rot_pass = rot_err <= rot_th
        trans_pass = trans_err <= trans_th
        combined = rot_pass & trans_pass
        recalls[f'R@{rot_th}°/{trans_th/1000:.0f}m'] = {
            'combined': combined.float().mean().item() * 100,
            'rot': rot_pass.float().mean().item() * 100,
            'trans': trans_pass.float().mean().item() * 100,
        }
    return recalls


def print_recall_table(recalls, prefix=""):
    for key, val in recalls.items():
        print(f"  {prefix}{key}: {val['combined']:.1f}%  (rot={val['rot']:.1f}%, trans={val['trans']:.1f}%)")


# ============================================================
# Training: Contrastive
# ============================================================

def train_contrastive(train_data, test_data, args):
    """Train contrastive projection head."""
    device = train_data.features.device
    
    if args.with_pose_aux:
        model = ProjectionHeadWithPose(
            input_dim=2560, embed_dim=args.embed_dim,
            hidden_dims=tuple(args.hidden_dims), dropout=args.dropout
        ).to(device)
    else:
        model = ProjectionHead(
            input_dim=2560, embed_dim=args.embed_dim,
            hidden_dims=tuple(args.hidden_dims), dropout=args.dropout
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Pre-compute distance matrix for training set
    train_dist = build_distance_matrix(train_data.translations)  # (N, N) in meters
    
    # For pose auxiliary task: normalize translations
    if args.with_pose_aux:
        trans_mean = train_data.translations.mean(dim=0)
        trans_std = train_data.translations.std(dim=0).clamp(min=1e-6)
        train_trans_norm = (train_data.translations - trans_mean) / trans_std
        test_trans_norm = (test_data.translations - trans_mean) / trans_std
    
    # Select loss function
    loss_fn_name = args.loss_fn
    
    best_val_recall = -1
    best_epoch = 0
    best_state = None
    
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Mini-batch sampling: random subset for efficiency
        batch_size = min(args.batch_size, train_data.N)
        perm = torch.randperm(train_data.N, device=device)[:batch_size]
        
        feats_batch = train_data.features[perm]
        dist_batch = train_dist[perm][:, perm]
        
        if args.with_pose_aux:
            out = model(feats_batch)
            embeddings, pred_trans, pred_rot_6d = out
        else:
            embeddings = model(feats_batch)
        
        # Contrastive loss
        if loss_fn_name == 'ntxent':
            loss_contrastive = nt_xent_with_spatial_positives(
                embeddings, dist_batch, pos_radius=args.pos_radius,
                temperature=args.temperature
            )
        elif loss_fn_name == 'multisim':
            loss_contrastive = multi_similarity_loss(
                embeddings, dist_batch, pos_radius=args.pos_radius,
                neg_radius=args.neg_radius
            )
        elif loss_fn_name == 'ap':
            loss_contrastive = ap_loss(
                embeddings, dist_batch, pos_radius=args.pos_radius
            )
        else:
            raise ValueError(f"Unknown loss: {loss_fn_name}")
        
        loss = loss_contrastive
        
        # Pose auxiliary loss
        if args.with_pose_aux:
            trans_target = train_trans_norm[perm]
            pose_loss_t = F.smooth_l1_loss(pred_trans, trans_target)
            
            rot_target = train_data.rotations[perm]  # (B, 3, 3)
            pred_rot_mat = gram_schmidt_6d_to_matrix(pred_rot_6d)
            pose_loss_r = geodesic_distance(pred_rot_mat, rot_target).mean()
            
            loss = loss + args.lambda_pose * (pose_loss_t + pose_loss_r)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Evaluate periodically
        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if args.with_pose_aux:
                    test_embed, _, _ = model(test_data.features)
                    train_embed, _, _ = model(train_data.features)
                else:
                    test_embed = model(test_data.features)
                    train_embed = model(train_data.features)
                
                ret_results = evaluate_retrieval(
                    test_embed, test_data.translations, test_data.rotations,
                    train_embed, train_data.translations, train_data.rotations,
                    topk=(1, 5)
                )
                
                recalls = compute_recall(ret_results['top1_rot_err'], ret_results['top1_trans_err'])
                r10_2 = recalls['R@10°/2m']['combined']
                r5_1 = recalls['R@5°/1m']['combined']
            
            elapsed = time.time() - t0
            
            extra = ""
            if args.with_pose_aux:
                extra = f" pose_t={pose_loss_t.item():.4f} pose_r={pose_loss_r.item():.4f}"
            
            print(f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f}{extra} | "
                  f"ret: {ret_results['top1_rot_median']:.1f}°/{ret_results['top1_trans_median']:.0f}mm "
                  f"R@5/1={r5_1:.1f}% R@10/2={r10_2:.1f}% | {elapsed:.1f}s")
            
            # Track best by R@10°/2m
            if r10_2 > best_val_recall:
                best_val_recall = r10_2
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"\nTraining done. Best epoch: {best_epoch} (R@10/2={best_val_recall:.1f}%)")
    
    # Save best model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'model_best.pt')
    torch.save({'state_dict': best_state, 'epoch': best_epoch, 'recall': best_val_recall}, save_path)
    
    # Load best and return
    model.load_state_dict(best_state)
    model.to(device)
    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Retrieval v5: Contrastive feature learning')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--summary_matrix', type=str,
                        default='output/feature_extract/features_radio_dual/OldHospital_pilot/summary_matrix.pt')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Training
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for contrastive sampling (use all train for best results)')
    parser.add_argument('--log_every', type=int, default=50)
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512])
    
    # Contrastive
    parser.add_argument('--loss_fn', type=str, default='ntxent',
                        choices=['ntxent', 'multisim', 'ap'])
    parser.add_argument('--pos_radius', type=float, default=5.0,
                        help='Positive radius in meters for contrastive pairs')
    parser.add_argument('--neg_radius', type=float, default=15.0,
                        help='Negative radius in meters (for multi-sim loss)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for NT-Xent loss')
    
    # Auxiliary
    parser.add_argument('--with_pose_aux', action='store_true',
                        help='Add pose regression auxiliary loss')
    parser.add_argument('--lambda_pose', type=float, default=0.1,
                        help='Weight for pose auxiliary loss')
    
    # Ablations
    parser.add_argument('--skip_training', action='store_true',
                        help='Only run baselines (oracle, raw NN), skip contrastive training')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data = RetrievalDataset(args.summary_matrix, args.dataset_dir, 'train', device)
    test_data = RetrievalDataset(args.summary_matrix, args.dataset_dir, 'test', device)
    
    # ========================================
    # Baseline 1: Oracle NN ceiling
    # ========================================
    print("\n" + "=" * 70)
    print("BASELINE 1: Oracle NN Ceiling (best possible single-image retrieval)")
    print("=" * 70)
    
    # For each test image, find the training image with smallest position distance
    test_pos = test_data.translations  # (Q, 3)
    train_pos = train_data.translations  # (D, 3)
    
    pos_dist = torch.cdist(test_pos, train_pos)  # (Q, D) in meters
    
    for k in [1, 3, 5, 10]:
        topk_dist, topk_idx = pos_dist.topk(k, dim=1, largest=False)  # smallest distance
        
        Q = test_data.N
        best_rot_err = torch.full((Q,), float('inf'), device=device)
        best_trans_err = torch.full((Q,), float('inf'), device=device)
        
        for j in range(k):
            ret_idx = topk_idx[:, j]
            ret_trans = train_data.translations[ret_idx]
            ret_rots = train_data.rotations[ret_idx]
            
            trans_err = torch.norm(ret_trans - test_data.translations, dim=-1) * 1000  # mm
            rot_err = geodesic_distance(ret_rots, test_data.rotations) * 180 / math.pi
            
            # Keep best (smallest combined score)
            score_old = best_trans_err / 2000 + best_rot_err / 10
            score_new = trans_err / 2000 + rot_err / 10
            better = score_new < score_old
            best_trans_err = torch.where(better, trans_err, best_trans_err)
            best_rot_err = torch.where(better, rot_err, best_rot_err)
        
        recalls = compute_recall(best_rot_err, best_trans_err)
        print(f"\n  Oracle top-{k}: {best_rot_err.median().item():.1f}°/{best_trans_err.median().item():.0f}mm")
        print_recall_table(recalls, prefix=f"  [top-{k}] ")
    
    # ========================================
    # Baseline 2: Raw RADIO NN
    # ========================================
    print("\n" + "=" * 70)
    print("BASELINE 2: Raw RADIO Cosine NN (no learning)")
    print("=" * 70)
    
    with torch.no_grad():
        raw_results = evaluate_retrieval(
            test_data.features, test_data.translations, test_data.rotations,
            train_data.features, train_data.translations, train_data.rotations,
            topk=(1, 3, 5, 10)
        )
    
    print(f"\n  Top-1: {raw_results['top1_rot_median']:.1f}°/{raw_results['top1_trans_median']:.0f}mm")
    recalls_raw = compute_recall(raw_results['top1_rot_err'], raw_results['top1_trans_err'])
    print_recall_table(recalls_raw, prefix="  [top-1] ")
    
    for k in [3, 5, 10]:
        print(f"  Top-{k} oracle: {raw_results[f'top{k}_oracle_rot_median']:.1f}°/"
              f"{raw_results[f'top{k}_oracle_trans_median']:.0f}mm")
    
    if args.skip_training:
        print("\nSkipping training (--skip_training flag).")
        save_report(args.output_dir, raw_results, recalls_raw, None, None, args)
        return
    
    # ========================================
    # Contrastive Training
    # ========================================
    print("\n" + "=" * 70)
    print(f"CONTRASTIVE TRAINING: {args.loss_fn} (pos_r={args.pos_radius}m, "
          f"embed_dim={args.embed_dim}, epochs={args.epochs})")
    print("=" * 70)
    
    model = train_contrastive(train_data, test_data, args)
    
    # ========================================
    # Final Evaluation
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION — Learned Embedding Retrieval")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        if args.with_pose_aux:
            test_embed, _, _ = model(test_data.features)
            train_embed, _, _ = model(train_data.features)
        else:
            test_embed = model(test_data.features)
            train_embed = model(train_data.features)
        
        learned_results = evaluate_retrieval(
            test_embed, test_data.translations, test_data.rotations,
            train_embed, train_data.translations, train_data.rotations,
            topk=(1, 3, 5, 10)
        )
    
    print(f"\n  Top-1: {learned_results['top1_rot_median']:.1f}°/{learned_results['top1_trans_median']:.0f}mm")
    recalls_learned = compute_recall(learned_results['top1_rot_err'], learned_results['top1_trans_err'])
    print_recall_table(recalls_learned, prefix="  [top-1] ")
    
    for k in [3, 5, 10]:
        print(f"  Top-{k} oracle: {learned_results[f'top{k}_oracle_rot_median']:.1f}°/"
              f"{learned_results[f'top{k}_oracle_trans_median']:.0f}mm")
    
    # ========================================
    # Improvement summary
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    raw_r10 = recalls_raw['R@10°/2m']['combined']
    learned_r10 = recalls_learned['R@10°/2m']['combined']
    raw_r5 = recalls_raw['R@5°/1m']['combined']
    learned_r5 = recalls_learned['R@5°/1m']['combined']
    
    print(f"  {'Method':<30} {'Rot Med':>8} {'Trans Med':>10} {'R@5/1':>6} {'R@10/2':>7}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*6} {'-'*7}")
    print(f"  {'Raw RADIO NN':<30} {raw_results['top1_rot_median']:>7.1f}° {raw_results['top1_trans_median']:>9.0f}mm {raw_r5:>5.1f}% {raw_r10:>6.1f}%")
    print(f"  {f'Learned ({args.loss_fn})':<30} {learned_results['top1_rot_median']:>7.1f}° {learned_results['top1_trans_median']:>9.0f}mm {learned_r5:>5.1f}% {learned_r10:>6.1f}%")
    print(f"  {'Improvement':<30} {'':>8} {'':>10} {learned_r5-raw_r5:>+5.1f}% {learned_r10-raw_r10:>+6.1f}%")
    
    # Save
    save_report(args.output_dir, raw_results, recalls_raw, learned_results, recalls_learned, args)
    
    # Visualization
    visualize_results(args.output_dir, test_data, train_data, 
                      raw_results, learned_results, test_embed, train_embed)
    
    print(f"\nResults saved to {args.output_dir}")


def save_report(output_dir, raw_results, recalls_raw, learned_results, recalls_learned, args):
    """Save JSON report."""
    report = {
        'args': vars(args),
        'raw_nn': {
            'rot_median': raw_results['top1_rot_median'],
            'trans_median': raw_results['top1_trans_median'],
            'recalls': {k: v['combined'] for k, v in recalls_raw.items()},
        }
    }
    if learned_results is not None:
        report['learned'] = {
            'rot_median': learned_results['top1_rot_median'],
            'trans_median': learned_results['top1_trans_median'],
            'recalls': {k: v['combined'] for k, v in recalls_learned.items()},
        }
    
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def visualize_results(output_dir, test_data, train_data,
                      raw_results, learned_results, test_embed, train_embed):
    """Create visualization plots."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Translation error CDF comparison
        ax = axes[0]
        raw_t = raw_results['top1_trans_err'].cpu().numpy()
        learned_t = learned_results['top1_trans_err'].cpu().numpy()
        
        for data, label in [(raw_t, 'Raw RADIO NN'), (learned_t, 'Learned')]:
            sorted_d = np.sort(data)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.plot(sorted_d / 1000, cdf, label=label)
        ax.axvline(x=2.0, color='r', linestyle='--', alpha=0.5, label='2m threshold')
        ax.set_xlabel('Translation Error (m)')
        ax.set_ylabel('CDF')
        ax.set_title('Translation Error CDF')
        ax.legend()
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Rotation error CDF
        ax = axes[1]
        raw_r = raw_results['top1_rot_err'].cpu().numpy()
        learned_r = learned_results['top1_rot_err'].cpu().numpy()
        
        for data, label in [(raw_r, 'Raw RADIO NN'), (learned_r, 'Learned')]:
            sorted_d = np.sort(data)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.plot(sorted_d, cdf, label=label)
        ax.axvline(x=10.0, color='r', linestyle='--', alpha=0.5, label='10° threshold')
        ax.set_xlabel('Rotation Error (°)')
        ax.set_ylabel('CDF')
        ax.set_title('Rotation Error CDF')
        ax.legend()
        ax.set_xlim(0, 30)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: 2D embedding visualization (PCA)
        ax = axes[2]
        all_embed = torch.cat([train_embed, test_embed], dim=0).cpu().numpy()
        # Simple 2D PCA
        mean = all_embed.mean(axis=0)
        all_embed_centered = all_embed - mean
        U, S, Vt = np.linalg.svd(all_embed_centered, full_matrices=False)
        proj_2d = all_embed_centered @ Vt[:2].T
        
        n_train = train_data.N
        ax.scatter(proj_2d[:n_train, 0], proj_2d[:n_train, 1], 
                   c='blue', alpha=0.3, s=5, label='Train')
        ax.scatter(proj_2d[n_train:, 0], proj_2d[n_train:, 1],
                   c='red', alpha=0.5, s=10, label='Test')
        ax.set_title('Embedding Space (PCA 2D)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'retrieval_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Visualization saved.")
    except Exception as e:
        print(f"  Visualization failed: {e}")


if __name__ == '__main__':
    main()
