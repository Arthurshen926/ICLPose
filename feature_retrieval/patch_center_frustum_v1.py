#!/usr/bin/env python3
"""Patch-token center-anchor + frustum proposal without summary token.

Design:
  - input features: patch tokens only (`feat=both` by default)
  - translation: anchor classification + per-anchor residual
  - rotation: forward + up vectors (camera frustum)
  - evaluation: direct pose, retrieval-mode, and top-k anchor oracle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from feature_retrieval.patch_regressor_v7 import POOL_REGISTRY, PatchPoseDataset, geodesic_distance


def gram_schmidt_rows(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    b1 = F.normalize(a1, dim=-1)
    proj = (a2 * b1).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def rotation_to_forward_up(R_w2c: torch.Tensor):
    forward = R_w2c[..., 2, :]
    up = -R_w2c[..., 1, :]
    return F.normalize(forward, dim=-1), F.normalize(up, dim=-1)


def forward_up_to_rotation(forward: torch.Tensor, up: torch.Tensor):
    z = F.normalize(forward, dim=-1)
    y_neg = up - (z * up).sum(dim=-1, keepdim=True) * z
    y_neg = F.normalize(y_neg, dim=-1)
    y = -y_neg
    x = F.normalize(torch.cross(y, z, dim=-1), dim=-1)
    return torch.stack([x, y, z], dim=-2)


class PatchFeaturePooler(nn.Module):
    def __init__(
        self,
        pool_type="attn",
        feat_mode="both",
        patch_dim=128,
        attn_heads=4,
        conv_out_dim=512,
        spp_levels=None,
    ):
        super().__init__()
        self.feat_mode = feat_mode
        pool_cls = POOL_REGISTRY[pool_type]
        pool_kwargs = {
            "num_heads": attn_heads,
            "out_dim": conv_out_dim,
            "spp_levels": tuple(spp_levels) if spp_levels else None,
        }
        self.pool_fine = None
        self.pool_coarse = None
        use_fine = "fine" in feat_mode or "both" in feat_mode
        use_coarse = "coarse" in feat_mode or "both" in feat_mode
        if use_fine:
            self.pool_fine = pool_cls(patch_dim, **pool_kwargs)
        if use_coarse:
            self.pool_coarse = pool_cls(patch_dim, **pool_kwargs)
        total_dim = 0
        if use_fine:
            total_dim += self.pool_fine.output_dim
        if use_coarse:
            total_dim += self.pool_coarse.output_dim
        self.total_dim = total_dim
        print(f"  Feature dim after pooling: {total_dim}")

    def forward(self, patch_fine=None, patch_coarse=None, summary=None):
        parts = []
        if self.pool_fine is not None and patch_fine is not None:
            parts.append(self.pool_fine(patch_fine))
        if self.pool_coarse is not None and patch_coarse is not None:
            parts.append(self.pool_coarse(patch_coarse))
        return torch.cat(parts, dim=-1)


def make_mlp(in_dim, hidden_dims, dropout):
    layers = []
    cur = in_dim
    for h in hidden_dims:
        layers.extend([
            nn.Linear(cur, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        cur = h
    return nn.Sequential(*layers), cur


class PatchCenterFrustumNet(nn.Module):
    def __init__(
        self,
        n_anchors,
        pool_type="attn",
        feat_mode="both",
        patch_dim=128,
        hidden_dims=(2048, 1024, 512),
        rot_hidden_dims=(1024, 512),
        dropout=0.15,
        attn_heads=4,
        conv_out_dim=512,
        spp_levels=None,
    ):
        super().__init__()
        self.n_anchors = int(n_anchors)
        self.pooler = PatchFeaturePooler(
            pool_type=pool_type,
            feat_mode=feat_mode,
            patch_dim=patch_dim,
            attn_heads=attn_heads,
            conv_out_dim=conv_out_dim,
            spp_levels=spp_levels,
        )
        self.input_norm = nn.LayerNorm(self.pooler.total_dim)
        self.trans_backbone, trans_dim = make_mlp(self.pooler.total_dim, hidden_dims, dropout)
        self.rot_backbone, rot_dim = make_mlp(self.pooler.total_dim, rot_hidden_dims, dropout)

        self.cls_head = nn.Linear(trans_dim, self.n_anchors)
        self.residual_head = nn.Linear(trans_dim, self.n_anchors * 3)
        self.forward_head = nn.Linear(rot_dim, 3)
        self.up_head = nn.Linear(rot_dim, 3)
        self.s_rot = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_fine=None, patch_coarse=None, summary=None):
        pooled = self.pooler(patch_fine, patch_coarse, summary)
        x = self.input_norm(pooled)
        h_t = self.trans_backbone(x)
        h_r = self.rot_backbone(x)
        logits = self.cls_head(h_t)
        residuals = self.residual_head(h_t).reshape(h_t.shape[0], self.n_anchors, 3)
        forward = F.normalize(self.forward_head(h_r), dim=-1)
        up = F.normalize(self.up_head(h_r), dim=-1)
        rot = forward_up_to_rotation(forward, up)
        return logits, residuals, forward, up, rot


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Random seed: {seed}")


def move_dataset_features_to_device(dataset, device):
    if dataset.patch_fine is not None:
        dataset.patch_fine = dataset.patch_fine.to(device)
    if dataset.patch_coarse is not None:
        dataset.patch_coarse = dataset.patch_coarse.to(device)
    if dataset.summary is not None:
        dataset.summary = dataset.summary.to(device)


def compute_anchor_centers(train_positions, n_anchors, seed):
    km = KMeans(n_clusters=n_anchors, random_state=seed, n_init=20, max_iter=500)
    km.fit(train_positions.cpu().numpy())
    anchors = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    labels = torch.tensor(km.labels_, dtype=torch.long)
    counts = np.bincount(km.labels_, minlength=n_anchors)
    print(f"  Anchor KMeans: K={n_anchors}, inertia={km.inertia_:.1f}")
    print(f"  Samples per anchor: min={counts.min()}, mean={counts.mean():.1f}, median={np.median(counts):.0f}, max={counts.max()}")
    return anchors, labels, km


def get_all_centers(residual_all, anchors, residual_std=None):
    if residual_std is None:
        return anchors.unsqueeze(0) + residual_all
    return anchors.unsqueeze(0) + residual_all * residual_std.view(1, 1, 3)


def predict_from_logits(logits, residual_all, anchors, residual_std=None, mode="argmax", topk=3, temperature=1.0):
    centers_all = get_all_centers(residual_all, anchors, residual_std=residual_std)
    pred_labels = logits.argmax(dim=-1)
    if mode == "mixture":
        k = min(int(topk), logits.shape[-1])
        top_logits, top_idx = logits.topk(k=k, dim=-1)
        top_weights = F.softmax(top_logits / max(float(temperature), 1e-6), dim=-1)
        top_centers = torch.gather(centers_all, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
        pred_center = (top_weights.unsqueeze(-1) * top_centers).sum(dim=1)
    else:
        pred_center = centers_all[torch.arange(pred_labels.shape[0], device=pred_labels.device), pred_labels]
    return pred_center, pred_labels, centers_all


def decode_topk_proposals(logits, residual_all, anchors, residual_std=None, topk=5):
    centers_all = get_all_centers(residual_all, anchors, residual_std=residual_std)
    k = min(int(topk), logits.shape[-1])
    top_logits, top_idx = logits.topk(k=k, dim=-1)
    top_scores = F.softmax(top_logits, dim=-1)
    top_centers = torch.gather(centers_all, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
    return top_idx, top_scores, top_centers


def build_soft_anchor_targets(gt_positions, anchors, topm=3, sigma=2.0):
    dists = torch.cdist(gt_positions, anchors)
    m = min(int(topm), anchors.shape[0])
    top_d, top_idx = dists.topk(k=m, dim=-1, largest=False)
    weights = torch.exp(-(top_d ** 2) / (2.0 * float(sigma) ** 2))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target = torch.zeros_like(dists)
    target.scatter_(1, top_idx, weights)
    return target, top_idx


def soft_cross_entropy(logits, target_probs):
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def weighted_multi_anchor_residual_loss(residual_all, gt_positions, anchors, residual_std, target_probs):
    gt_residual_all = (gt_positions.unsqueeze(1) - anchors.unsqueeze(0)) / residual_std.view(1, 1, 3)
    per_anchor = F.smooth_l1_loss(residual_all, gt_residual_all, reduction="none").mean(dim=-1)
    return (per_anchor * target_probs).sum(dim=-1).mean()


def evaluate_predictions(pred_center, pred_rot, test_data, train_data, out_dir, extra=None):
    out_dir = Path(out_dir)
    trans_errors = (pred_center - test_data.translations).norm(dim=-1).cpu().numpy()
    rot_errors = (geodesic_distance(pred_rot, test_data.rotations) * 180 / math.pi).cpu().numpy()
    thresholds = [
        ("5deg_1m", 5, 1.0),
        ("10deg_2m", 10, 2.0),
        ("15deg_5m", 15, 5.0),
        ("25deg_5m", 25, 5.0),
    ]
    results = {
        "rotation_deg": {"median": float(np.median(rot_errors)), "mean": float(np.mean(rot_errors))},
        "translation_m": {"median": float(np.median(trans_errors)), "mean": float(np.mean(trans_errors))},
        "translation_mm": {"median": float(np.median(trans_errors) * 1000.0), "mean": float(np.mean(trans_errors) * 1000.0)},
        "recall": {},
        "individual_pass": {},
    }
    for name, rot_th, trans_th in thresholds:
        rot_pass = (rot_errors < rot_th).mean() * 100.0
        trans_pass = (trans_errors < trans_th).mean() * 100.0
        combined = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100.0
        results["recall"][name] = float(combined)
        results["individual_pass"][name] = {
            "rot_pass": float(rot_pass),
            "trans_pass": float(trans_pass),
            "combined": float(combined),
        }

    train_positions = train_data.translations.cpu().numpy()
    pred_positions = pred_center.cpu().numpy()
    tree = cKDTree(train_positions)
    _, nn_indices = tree.query(pred_positions, k=1)
    test_trans_np = test_data.translations.cpu().numpy()
    retr_trans_errors = np.linalg.norm(train_positions[nn_indices] - test_trans_np, axis=-1)
    retr_rot_errors = (geodesic_distance(train_data.rotations.cpu()[nn_indices], test_data.rotations.cpu()) * 180 / math.pi).numpy()
    results["retrieval"] = {
        "median_rot_deg": float(np.median(retr_rot_errors)),
        "median_trans_mm": float(np.median(retr_trans_errors) * 1000.0),
    }
    for name, rot_th, trans_th in thresholds:
        results["retrieval"][name] = {
            "combined": float(((retr_rot_errors < rot_th) & (retr_trans_errors < trans_th)).mean() * 100.0),
            "rot_pass": float((retr_rot_errors < rot_th).mean() * 100.0),
            "trans_pass": float((retr_trans_errors < trans_th).mean() * 100.0),
        }

    if extra is not None:
        results["center_frustum"] = extra

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Center + Frustum)")
    print("=" * 70)
    print(f"Rotation  (median): {results['rotation_deg']['median']:.2f}deg")
    print(f"Translation (median): {results['translation_mm']['median']:.0f}mm")
    for name, _, _ in thresholds:
        ip = results["individual_pass"][name]
        print(f"  R@{name}: {results['recall'][name]:.1f}%  (rot_pass={ip['rot_pass']:.1f}%, trans_pass={ip['trans_pass']:.1f}%)")
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Retrieval Mode)")
    print("=" * 70)
    print(f"Retrieval median: {results['retrieval']['median_rot_deg']:.2f}deg / {results['retrieval']['median_trans_mm']:.0f}mm")
    for name, _, _ in thresholds:
        r = results["retrieval"][name]
        print(f"  R@{name}: {r['combined']:.1f}%  (rot={r['rot_pass']:.1f}%, trans={r['trans_pass']:.1f}%)")
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_fine = "fine" in args.feat or "both" in args.feat
    use_coarse = "coarse" in args.feat or "both" in args.feat
    use_summary = "+sum" in args.feat
    print(f"\nLoading data (fine={use_fine}, coarse={use_coarse}, summary={use_summary})...")
    train_data = PatchPoseDataset(args.feature_dir, args.dataset_dir, "train", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(args.feature_dir, args.dataset_dir, "test", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)
    move_dataset_features_to_device(train_data, device)
    move_dataset_features_to_device(test_data, device)
    train_fwd, train_up = rotation_to_forward_up(train_data.rotations)

    print("\nBuilding center anchors...")
    anchors_cpu, train_anchor_labels_cpu, km = compute_anchor_centers(train_data.translations.cpu(), args.n_anchors, args.anchor_seed)
    test_anchor_labels_cpu = torch.tensor(km.predict(test_data.translations.cpu().numpy()), dtype=torch.long)
    anchors = anchors_cpu.to(device)
    train_anchor_labels = train_anchor_labels_cpu.to(device)
    test_anchor_labels = test_anchor_labels_cpu.to(device)
    train_anchor_residuals = train_data.translations - anchors[train_anchor_labels]
    residual_std = train_anchor_residuals.std(dim=0).clamp(min=1e-3)
    print(f"  Residual std: {residual_std.detach().cpu().numpy()}")

    torch.save(
        {
            "anchors": anchors_cpu,
            "train_anchor_labels": train_anchor_labels_cpu,
            "test_anchor_labels": test_anchor_labels_cpu,
            "residual_std": residual_std.detach().cpu(),
        },
        out_dir / "anchor_info.pt",
    )

    model = PatchCenterFrustumNet(
        n_anchors=args.n_anchors,
        pool_type=args.pool,
        feat_mode=args.feat,
        patch_dim=args.patch_dim,
        hidden_dims=tuple(args.hidden_dims),
        rot_hidden_dims=tuple(args.rot_hidden_dims),
        dropout=args.dropout,
        attn_heads=args.attn_heads,
        conv_out_dim=args.conv_out_dim,
        spp_levels=args.spp_levels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    config = vars(args).copy()
    config["output_dir"] = str(out_dir)
    config["n_params"] = n_params
    config["total_feature_dim"] = model.pooler.total_dim
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    history = {
        "epoch": [],
        "loss": [],
        "loss_cls": [],
        "loss_res": [],
        "loss_fwd": [],
        "loss_up": [],
        "val_rot_median": [],
        "val_trans_median": [],
        "val_anchor_top1": [],
        "val_anchor_top3": [],
        "val_anchor_top5": [],
        "val_top5_oracle_2m": [],
    }
    best_val_score = float("inf")
    best_epoch = 0
    t0 = time.time()
    n_train = train_data.N

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.batch_size >= n_train:
            indices = torch.arange(n_train, device=device)
        else:
            indices = torch.randperm(n_train, device=device)[: args.batch_size]

        pf = train_data.patch_fine[indices] if train_data.patch_fine is not None else None
        pc = train_data.patch_coarse[indices] if train_data.patch_coarse is not None else None
        sm = train_data.summary[indices] if train_data.summary is not None else None

        logits, residuals, forward_pred, up_pred, rot_pred = model(pf, pc, sm)
        target_anchor = train_anchor_labels[indices]
        idx = torch.arange(indices.shape[0], device=device)
        pred_residual = residuals[idx, target_anchor]
        gt_residual = (train_data.translations[indices] - anchors[target_anchor]) / residual_std.unsqueeze(0)
        target_anchor_probs, _ = build_soft_anchor_targets(
            train_data.translations[indices], anchors, topm=args.soft_topm, sigma=args.anchor_sigma
        )

        if args.soft_anchor_targets:
            loss_cls = soft_cross_entropy(logits, target_anchor_probs)
            loss_res = weighted_multi_anchor_residual_loss(
                residuals, train_data.translations[indices], anchors, residual_std, target_anchor_probs
            )
        else:
            loss_cls = F.cross_entropy(logits, target_anchor)
            loss_res = F.smooth_l1_loss(pred_residual, gt_residual)
        loss_fwd = (1.0 - (forward_pred * train_fwd[indices]).sum(dim=-1)).mean()
        loss_up = (1.0 - (up_pred * train_up[indices]).sum(dim=-1)).mean()
        loss = args.lambda_cls * loss_cls + args.lambda_res * loss_res + args.lambda_fwd * loss_fwd + args.lambda_up * loss_up

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits_t, residuals_t, _, _, rot_t = model(test_data.patch_fine, test_data.patch_coarse, test_data.summary)
                pred_center, pred_anchor, _ = predict_from_logits(
                    logits_t, residuals_t, anchors, residual_std=residual_std,
                    mode=args.center_mode, topk=args.center_topk, temperature=args.center_temperature,
                )
                trans_errors = (pred_center - test_data.translations).norm(dim=-1)
                rot_errors = geodesic_distance(rot_t, test_data.rotations) * 180 / math.pi
                val_trans_med = trans_errors.median().item()
                val_rot_med = rot_errors.median().item()
                val_score = val_trans_med + 0.1 * val_rot_med
                top1_acc = (pred_anchor == test_anchor_labels).float().mean().item() * 100.0
                top3_idx = logits_t.topk(k=min(3, args.n_anchors), dim=-1).indices
                top3_acc = (top3_idx == test_anchor_labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
                top5_idx = logits_t.topk(k=min(5, args.n_anchors), dim=-1).indices
                top5_acc = (top5_idx == test_anchor_labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
                _, _, top5_centers = decode_topk_proposals(logits_t, residuals_t, anchors, residual_std=residual_std, topk=min(5, args.n_anchors))
                top5_oracle_trans = torch.norm(top5_centers - test_data.translations.unsqueeze(1), dim=-1).min(dim=1).values
                top5_oracle_2m = (top5_oracle_trans < 2.0).float().mean().item() * 100.0

            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["loss_cls"].append(loss_cls.item())
            history["loss_res"].append(loss_res.item())
            history["loss_fwd"].append(loss_fwd.item())
            history["loss_up"].append(loss_up.item())
            history["val_rot_median"].append(val_rot_med)
            history["val_trans_median"].append(val_trans_med)
            history["val_anchor_top1"].append(top1_acc)
            history["val_anchor_top3"].append(top3_acc)
            history["val_anchor_top5"].append(top5_acc)
            history["val_top5_oracle_2m"].append(top5_oracle_2m)

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                r10 = ((rot_errors < 10) & (trans_errors < 2.0)).float().mean().item() * 100.0
                r5 = ((rot_errors < 5) & (trans_errors < 1.0)).float().mean().item() * 100.0
                print(
                    f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                    f"(cls={loss_cls.item():.4f} res={loss_res.item():.4f} f={loss_fwd.item():.4f} u={loss_up.item():.4f}) "
                    f"| val: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                    f"R@10/2={r10:.1f}% R@5/1={r5:.1f}% a@1={top1_acc:.1f}% a@3={top3_acc:.1f}% a@5={top5_acc:.1f}% oracle2m={top5_oracle_2m:.1f}% | {time.time()-t0:.1f}s"
                )

            if args.select_by_topk_oracle:
                val_score = -top5_oracle_2m + 0.01 * val_trans_med + 0.001 * val_rot_med

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_rot_median": val_rot_med,
                        "val_trans_median": val_trans_med,
                        "config": config,
                        "anchors": anchors_cpu,
                    },
                    out_dir / "model_best.pt",
                )

    print(f"\nTraining done. Best epoch: {best_epoch}, score={best_val_score:.4f}")
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(out_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits_t, residuals_t, _, _, rot_t = model(test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        pred_center, pred_anchor, centers_all = predict_from_logits(
            logits_t, residuals_t, anchors, residual_std=residual_std,
            mode=args.center_mode, topk=args.center_topk, temperature=args.center_temperature,
        )
        top3_idx = logits_t.topk(k=min(3, args.n_anchors), dim=-1).indices
        top3_acc = (top3_idx == test_anchor_labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
        top1_acc = (pred_anchor == test_anchor_labels).float().mean().item() * 100.0
        gt_trans = test_data.translations.unsqueeze(1)
        center_dists = torch.norm(centers_all - gt_trans, dim=-1)
        topk_idx = logits_t.topk(k=min(args.eval_topk_anchor, args.n_anchors), dim=-1).indices
        topk_dist = torch.gather(center_dists, 1, topk_idx)
        topk_oracle_trans = topk_dist.min(dim=1).values.cpu().numpy()
        topk_oracle_r10 = float((((geodesic_distance(rot_t, test_data.rotations) * 180 / math.pi).cpu().numpy() < 10.0) & (topk_oracle_trans < 2.0)).mean() * 100.0)

    return evaluate_predictions(
        pred_center,
        rot_t,
        test_data,
        train_data,
        out_dir,
        extra={
            "anchor_top1_acc": float(top1_acc),
            "anchor_top3_acc": float(top3_acc),
            "center_mode": args.center_mode,
            "topk_anchor_oracle_r10": topk_oracle_r10,
            "soft_anchor_targets": bool(args.soft_anchor_targets),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-token center + frustum proposal")
    parser.add_argument("--pool", type=str, default="attn", choices=["gap", "gem", "spp", "attn", "conv"])
    parser.add_argument("--feat", type=str, default="both", choices=["fine", "coarse", "both", "fine+sum", "coarse+sum", "both+sum"])
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--feature_dir", type=str, default="output/feature_extract/features_radio_dual_128/OldHospital_pilot")
    parser.add_argument("--dataset_dir", type=str, default="dataset/OldHospital")
    parser.add_argument("--output_base", type=str, default="output/feature_retrieval/pose_regression")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[2048, 1024, 512])
    parser.add_argument("--rot_hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--conv_out_dim", type=int, default=512)
    parser.add_argument("--spp_levels", type=int, nargs="+", default=None)
    parser.add_argument("--patch_dim", type=int, default=128)
    parser.add_argument("--n_anchors", type=int, default=32)
    parser.add_argument("--anchor_seed", type=int, default=42)
    parser.add_argument("--center_mode", type=str, default="mixture", choices=["argmax", "mixture"])
    parser.add_argument("--center_topk", type=int, default=3)
    parser.add_argument("--center_temperature", type=float, default=1.0)
    parser.add_argument("--eval_topk_anchor", type=int, default=5)
    parser.add_argument("--soft_anchor_targets", action="store_true")
    parser.add_argument("--soft_topm", type=int, default=3)
    parser.add_argument("--anchor_sigma", type=float, default=2.0)
    parser.add_argument("--select_by_topk_oracle", action="store_true")
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--lambda_fwd", type=float, default=0.5)
    parser.add_argument("--lambda_up", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.output_base, args.exp_name)
    train(args)


if __name__ == "__main__":
    main()
