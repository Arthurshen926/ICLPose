#!/usr/bin/env python3
"""Patch-token spatial cell classification with residual translation.

This is the patch-token upgrade of the old summary-token grid classifier.
Instead of regressing translation directly, it predicts:
  1. a coarse spatial cell
  2. a residual translation inside that cell
  3. full rotation

Current label mode:
  - camera_kmeans: K-means over training camera centers
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

from feature_retrieval.patch_regressor_v7 import (
    POOL_REGISTRY,
    PatchPoseDataset,
    geodesic_distance,
    gram_schmidt_6d_to_matrix,
)


class PatchFeaturePooler(nn.Module):
    """Pooling-only front-end shared with patch_regressor_v7."""

    def __init__(
        self,
        pool_type="attn",
        feat_mode="both+sum",
        patch_dim=128,
        attn_heads=4,
        conv_out_dim=512,
        spp_levels=None,
    ):
        super().__init__()
        self.feat_mode = feat_mode
        self.pool_type = pool_type

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
        if "+sum" in feat_mode:
            total_dim += 2560

        self.total_dim = total_dim
        print(f"  Feature dim after pooling: {total_dim}")

    def pool_features(self, patch_fine=None, patch_coarse=None, summary=None):
        parts = []
        if self.pool_fine is not None and patch_fine is not None:
            parts.append(self.pool_fine(patch_fine))
        if self.pool_coarse is not None and patch_coarse is not None:
            parts.append(self.pool_coarse(patch_coarse))
        if "+sum" in self.feat_mode and summary is not None:
            parts.append(summary)
        return torch.cat(parts, dim=-1)


def make_mlp_stack(in_dim, hidden_dims, dropout):
    layers = []
    cur_dim = in_dim
    for h_dim in hidden_dims:
        layers.extend([
            nn.Linear(cur_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        cur_dim = h_dim
    return nn.Sequential(*layers), cur_dim


class PatchGridClassifier(nn.Module):
    """Predict translation by cells and rotation by a separate or shared branch."""

    def __init__(
        self,
        n_cells,
        pool_type="attn",
        feat_mode="both+sum",
        patch_dim=128,
        hidden_dims=(2048, 1024, 512),
        dropout=0.15,
        attn_heads=4,
        conv_out_dim=512,
        spp_levels=None,
        decouple_rot=False,
        rot_hidden_dims=None,
    ):
        super().__init__()
        self.n_cells = int(n_cells)
        self.decouple_rot = bool(decouple_rot)
        self.pooler = PatchFeaturePooler(
            pool_type=pool_type,
            feat_mode=feat_mode,
            patch_dim=patch_dim,
            attn_heads=attn_heads,
            conv_out_dim=conv_out_dim,
            spp_levels=spp_levels,
        )

        self.input_norm = nn.LayerNorm(self.pooler.total_dim)

        self.trans_backbone, trans_dim = make_mlp_stack(self.pooler.total_dim, hidden_dims, dropout)
        if self.decouple_rot:
            if rot_hidden_dims is None:
                rot_hidden_dims = tuple(hidden_dims[-2:] if len(hidden_dims) > 1 else hidden_dims)
            self.rot_backbone, rot_dim = make_mlp_stack(self.pooler.total_dim, rot_hidden_dims, dropout)
        else:
            self.rot_backbone = None
            rot_dim = trans_dim

        self.cls_head = nn.Linear(trans_dim, self.n_cells)
        self.residual_head = nn.Linear(trans_dim, self.n_cells * 3)
        self.rot_head = nn.Linear(rot_dim, 6)
        self.s_rot = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def pool_features(self, patch_fine=None, patch_coarse=None, summary=None):
        return self.pooler.pool_features(patch_fine, patch_coarse, summary)

    def forward_from_pooled(self, pooled):
        x = self.input_norm(pooled)
        h_trans = self.trans_backbone(x)
        h_rot = self.rot_backbone(x) if self.rot_backbone is not None else h_trans
        cls_logits = self.cls_head(h_trans)
        residuals = self.residual_head(h_trans).reshape(h_trans.shape[0], self.n_cells, 3)
        rot_mat = gram_schmidt_6d_to_matrix(self.rot_head(h_rot))
        return cls_logits, residuals, rot_mat

    def forward(self, patch_fine=None, patch_coarse=None, summary=None):
        pooled = self.pool_features(patch_fine, patch_coarse, summary)
        return self.forward_from_pooled(pooled)


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


def compute_camera_kmeans(train_positions, test_positions, n_cells, seed):
    pos_np = train_positions.cpu().numpy()
    km = KMeans(n_clusters=n_cells, random_state=seed, n_init=20, max_iter=500)
    km.fit(pos_np)

    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    train_labels = torch.tensor(km.labels_, dtype=torch.long)
    test_labels = torch.tensor(km.predict(test_positions.cpu().numpy()), dtype=torch.long)

    counts = np.bincount(km.labels_, minlength=n_cells)
    print(f"  Camera K-means: K={n_cells}, inertia={km.inertia_:.1f}")
    print(
        "  Samples per cell: "
        f"min={counts.min()}, mean={counts.mean():.1f}, "
        f"median={np.median(counts):.0f}, max={counts.max()}"
    )
    return centroids, train_labels, test_labels, counts


def gather_residuals(residual_all, labels):
    idx = torch.arange(labels.shape[0], device=labels.device)
    return residual_all[idx, labels]


def get_all_cell_centers(residual_all, centroids, residual_std):
    return centroids.unsqueeze(0) + residual_all * residual_std.view(1, 1, 3)


def expected_translation_from_logits(cls_logits, residual_all, centroids, residual_std, temperature=1.0):
    cell_centers = get_all_cell_centers(residual_all, centroids, residual_std)
    weights = F.softmax(cls_logits / max(float(temperature), 1e-6), dim=-1)
    return (weights.unsqueeze(-1) * cell_centers).sum(dim=1)


def predict_translation_from_logits(
    cls_logits,
    residual_all,
    centroids,
    residual_std,
    mode="argmax",
    topk=3,
    temperature=1.0,
):
    pred_labels = cls_logits.argmax(dim=-1)
    cell_centers = get_all_cell_centers(residual_all, centroids, residual_std)

    if mode == "mixture":
        k = min(int(topk), cls_logits.shape[-1])
        top_logits, top_idx = cls_logits.topk(k=k, dim=-1)
        top_weights = F.softmax(top_logits / max(float(temperature), 1e-6), dim=-1)
        top_centers = torch.gather(cell_centers, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
        pred_centers = (top_weights.unsqueeze(-1) * top_centers).sum(dim=1)
    else:
        pred_centers = gather_residuals(cell_centers, pred_labels)
    return pred_centers, pred_labels


def predict_centers_and_rotation(
    model,
    dataset,
    centroids,
    residual_std,
    translation_mode="argmax",
    translation_topk=3,
    mixture_temperature=1.0,
):
    with torch.no_grad():
        cls_logits, residual_all, rot_pred = model(
            dataset.patch_fine, dataset.patch_coarse, dataset.summary
        )
        pred_centers, pred_labels = predict_translation_from_logits(
            cls_logits,
            residual_all,
            centroids,
            residual_std,
            mode=translation_mode,
            topk=translation_topk,
            temperature=mixture_temperature,
        )
    return pred_centers, rot_pred, pred_labels, cls_logits


def evaluate_predictions(pred_centers, pred_rot, test_data, train_data, out_dir, extra=None):
    out_dir = Path(out_dir)

    trans_errors = (pred_centers - test_data.translations).norm(dim=-1).cpu().numpy()
    rot_errors = (geodesic_distance(pred_rot, test_data.rotations) * 180 / math.pi).cpu().numpy()

    thresholds = [
        ("5deg_1m", 5, 1.0),
        ("10deg_2m", 10, 2.0),
        ("15deg_5m", 15, 5.0),
        ("25deg_5m", 25, 5.0),
    ]

    results = {
        "rotation_deg": {
            "median": float(np.median(rot_errors)),
            "mean": float(np.mean(rot_errors)),
        },
        "translation_m": {
            "median": float(np.median(trans_errors)),
            "mean": float(np.mean(trans_errors)),
        },
        "translation_mm": {
            "median": float(np.median(trans_errors) * 1000.0),
            "mean": float(np.mean(trans_errors) * 1000.0),
        },
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
    pred_positions = pred_centers.cpu().numpy()
    tree = cKDTree(train_positions)
    _, nn_indices = tree.query(pred_positions, k=1)

    test_trans_np = test_data.translations.cpu().numpy()
    train_rot = train_data.rotations.cpu()
    test_rot = test_data.rotations.cpu()
    retr_trans_errors = np.linalg.norm(train_positions[nn_indices] - test_trans_np, axis=-1)
    retr_rot_pred = train_rot[nn_indices]
    retr_rot_errors = (geodesic_distance(retr_rot_pred, test_rot) * 180 / math.pi).numpy()

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
        results["classification"] = extra

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Grid Classification + Residual)")
    print("=" * 70)
    print(f"Rotation  (median): {results['rotation_deg']['median']:.2f}deg")
    print(f"Translation (median): {results['translation_mm']['median']:.0f}mm")
    for name, _, _ in thresholds:
        ip = results["individual_pass"][name]
        print(
            f"  R@{name}: {results['recall'][name]:.1f}%  "
            f"(rot_pass={ip['rot_pass']:.1f}%, trans_pass={ip['trans_pass']:.1f}%)"
        )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Retrieval Mode)")
    print("=" * 70)
    print(
        f"Retrieval median: {results['retrieval']['median_rot_deg']:.2f}deg / "
        f"{results['retrieval']['median_trans_mm']:.0f}mm"
    )
    for name, _, _ in thresholds:
        r = results["retrieval"][name]
        print(
            f"  R@{name}: {r['combined']:.1f}%  "
            f"(rot={r['rot_pass']:.1f}%, trans={r['trans_pass']:.1f}%)"
        )

    if extra is not None:
        print("\n" + "=" * 70)
        print("CLASSIFICATION")
        print("=" * 70)
        print(f"  Top-1 cell acc: {extra['top1_acc']:.1f}%")
        print(f"  Top-3 cell acc: {extra['top3_acc']:.1f}%")

    with open(out_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
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
    train_data = PatchPoseDataset(
        args.feature_dir,
        args.dataset_dir,
        "train",
        "cpu",
        use_fine=use_fine,
        use_coarse=use_coarse,
        use_summary=use_summary,
    )
    test_data = PatchPoseDataset(
        args.feature_dir,
        args.dataset_dir,
        "test",
        "cpu",
        use_fine=use_fine,
        use_coarse=use_coarse,
        use_summary=use_summary,
    )

    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)
    move_dataset_features_to_device(train_data, device)
    move_dataset_features_to_device(test_data, device)

    print("\nBuilding spatial cells...")
    centroids_cpu, train_labels_cpu, test_labels_cpu, cell_counts = compute_camera_kmeans(
        train_data.translations.cpu(), test_data.translations.cpu(), args.n_cells, args.kmeans_seed
    )
    centroids = centroids_cpu.to(device)
    train_labels = train_labels_cpu.to(device)
    test_labels = test_labels_cpu.to(device)

    train_residuals = train_data.translations - centroids[train_labels]
    residual_std = train_residuals.std(dim=0).clamp(min=1e-3)
    print(f"  Residual std: {residual_std.detach().cpu().numpy()}")

    torch.save(
        {
            "centroids": centroids_cpu,
            "train_labels": train_labels_cpu,
            "test_labels": test_labels_cpu,
            "cell_counts": torch.tensor(cell_counts),
            "residual_std": residual_std.detach().cpu(),
        },
        out_dir / "cluster_info.pt",
    )

    model = PatchGridClassifier(
        n_cells=args.n_cells,
        pool_type=args.pool,
        feat_mode=args.feat,
        patch_dim=args.patch_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        attn_heads=args.attn_heads,
        conv_out_dim=args.conv_out_dim,
        spp_levels=args.spp_levels,
        decouple_rot=args.decouple_rot,
        rot_hidden_dims=tuple(args.rot_hidden_dims) if args.rot_hidden_dims is not None else None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    config = vars(args).copy()
    config["output_dir"] = str(out_dir)
    config["n_params"] = n_params
    config["total_feature_dim"] = model.pooler.total_dim
    with open(out_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    batch_size = args.batch_size
    n_train = train_data.N

    history = {
        "epoch": [],
        "loss": [],
        "loss_cls": [],
        "loss_res": [],
        "loss_mix": [],
        "loss_rot": [],
        "val_rot_median": [],
        "val_trans_median": [],
        "val_cls_top1": [],
        "val_cls_top3": [],
        "beta_rot": [],
    }
    best_val_score = float("inf")
    best_epoch = 0
    trans_scale = train_data.translations.std(dim=0).clamp(min=1e-3)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()

        if batch_size >= n_train:
            indices = torch.arange(n_train, device=device)
        else:
            indices = torch.randperm(n_train, device=device)[:batch_size]

        pf = train_data.patch_fine[indices] if train_data.patch_fine is not None else None
        pc = train_data.patch_coarse[indices] if train_data.patch_coarse is not None else None
        sm = train_data.summary[indices] if train_data.summary is not None else None
        if sm is not None and args.feature_dropout > 0:
            sm = F.dropout(sm, p=args.feature_dropout, training=True)

        target_rot = train_data.rotations[indices]
        target_labels = train_labels[indices]
        target_residual_norm = (train_data.translations[indices] - centroids[target_labels]) / residual_std

        cls_logits, residual_all, rot_pred = model(pf, pc, sm)
        residual_pred = gather_residuals(residual_all, target_labels)
        mix_center_pred = expected_translation_from_logits(
            cls_logits,
            residual_all,
            centroids,
            residual_std,
            temperature=args.mixture_temperature,
        )

        loss_cls = F.cross_entropy(cls_logits, target_labels)
        loss_res = F.smooth_l1_loss(residual_pred, target_residual_norm)
        loss_mix = F.smooth_l1_loss(
            mix_center_pred / trans_scale.unsqueeze(0),
            train_data.translations[indices] / trans_scale.unsqueeze(0),
        )
        loss_rot = geodesic_distance(rot_pred, target_rot).mean()
        beta_rot = torch.tensor(args.rot_weight, device=device) if args.rot_weight is not None else torch.exp(model.s_rot)
        loss = (
            args.lambda_cls * loss_cls
            + args.lambda_res * loss_res
            + args.lambda_mix * loss_mix
            + beta_rot * loss_rot
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_centers, pred_rot, pred_labels, pred_logits = predict_centers_and_rotation(
                    model,
                    test_data,
                    centroids,
                    residual_std,
                    translation_mode=args.translation_mode,
                    translation_topk=args.translation_topk,
                    mixture_temperature=args.mixture_temperature,
                )
                trans_errors = (pred_centers - test_data.translations).norm(dim=-1)
                rot_errors = geodesic_distance(pred_rot, test_data.rotations) * 180 / math.pi

                val_trans_med = trans_errors.median().item()
                val_rot_med = rot_errors.median().item()
                val_score = val_trans_med + val_rot_med * 0.1

                top1_acc = (pred_labels == test_labels).float().mean().item() * 100.0
                topk = min(3, args.n_cells)
                top3 = pred_logits.topk(k=topk, dim=-1).indices
                top3_acc = (top3 == test_labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["loss_cls"].append(loss_cls.item())
            history["loss_res"].append(loss_res.item())
            history["loss_mix"].append(loss_mix.item())
            history["loss_rot"].append(loss_rot.item())
            history["val_rot_median"].append(val_rot_med)
            history["val_trans_median"].append(val_trans_med)
            history["val_cls_top1"].append(top1_acc)
            history["val_cls_top3"].append(top3_acc)
            history["beta_rot"].append(beta_rot.item())

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                elapsed = time.time() - t0
                r10 = ((rot_errors < 10) & (trans_errors < 2.0)).float().mean().item() * 100.0
                r5 = ((rot_errors < 5) & (trans_errors < 1.0)).float().mean().item() * 100.0
                print(
                    f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                    f"(cls={loss_cls.item():.4f} res={loss_res.item():.4f} mix={loss_mix.item():.4f} "
                    f"rot={loss_rot.item():.4f} beta={beta_rot.item():.2f}) "
                    f"| val: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                    f"R@10/2={r10:.1f}% R@5/1={r5:.1f}% "
                    f"cell@1={top1_acc:.1f}% cell@3={top3_acc:.1f}% | {elapsed:.1f}s"
                )

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_rot_median": val_rot_med,
                        "val_trans_median": val_trans_med,
                        "val_cls_top1": top1_acc,
                        "val_cls_top3": top3_acc,
                        "centroids": centroids.detach().cpu(),
                        "residual_std": residual_std.detach().cpu(),
                        "config": config,
                    },
                    out_dir / "model_best.pt",
                )

    print(f"\nTraining done. Best epoch: {best_epoch}, score: {best_val_score:.4f}")

    with open(out_dir / "training_log.json", "w") as handle:
        json.dump(history, handle, indent=2)

    best_ckpt = torch.load(out_dir / "model_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    pred_centers, pred_rot, pred_labels, pred_logits = predict_centers_and_rotation(
        model,
        test_data,
        centroids,
        residual_std,
        translation_mode=args.translation_mode,
        translation_topk=args.translation_topk,
        mixture_temperature=args.mixture_temperature,
    )
    top1_acc = (pred_labels == test_labels).float().mean().item() * 100.0
    topk = min(3, args.n_cells)
    top3 = pred_logits.topk(k=topk, dim=-1).indices
    top3_acc = (top3 == test_labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

    results = evaluate_predictions(
        pred_centers,
        pred_rot,
        test_data,
        train_data,
        out_dir,
        extra={
            "top1_acc": float(top1_acc),
            "top3_acc": float(top3_acc),
            "translation_mode": args.translation_mode,
            "translation_topk": int(args.translation_topk),
            "lambda_mix": float(args.lambda_mix),
            "decouple_rot": bool(args.decouple_rot),
        },
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-token grid classifier")
    parser.add_argument("--pool", type=str, default="attn", choices=["gap", "gem", "spp", "attn", "conv"])
    parser.add_argument(
        "--feat",
        type=str,
        default="both+sum",
        choices=["fine", "coarse", "both", "fine+sum", "coarse+sum", "both+sum"],
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="output/feature_extract/features_radio_dual_128/OldHospital_pilot",
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset/OldHospital")
    parser.add_argument("--output_base", type=str, default="output/feature_retrieval/pose_regression")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--feature_dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[2048, 1024, 512])
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--conv_out_dim", type=int, default=512)
    parser.add_argument("--spp_levels", type=int, nargs="+", default=None)
    parser.add_argument("--patch_dim", type=int, default=128)
    parser.add_argument("--n_cells", type=int, default=64)
    parser.add_argument("--kmeans_seed", type=int, default=42)
    parser.add_argument("--translation_mode", type=str, default="argmax", choices=["argmax", "mixture"])
    parser.add_argument("--translation_topk", type=int, default=3)
    parser.add_argument("--mixture_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--lambda_mix", type=float, default=0.0)
    parser.add_argument("--rot_weight", type=float, default=None)
    parser.add_argument("--decouple_rot", action="store_true")
    parser.add_argument("--rot_hidden_dims", type=int, nargs="+", default=None)
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
