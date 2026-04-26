#!/usr/bin/env python3
"""Patch-token anchor-mixture translation + decoupled rotation.

Translation is modeled as a soft mixture over learned anchor embeddings,
while rotation is predicted by a separate regression branch.
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

from feature_retrieval.patch_regressor_v7 import (
    POOL_REGISTRY,
    PatchPoseDataset,
    geodesic_distance,
    gram_schmidt_6d_to_matrix,
)


class PatchFeaturePooler(nn.Module):
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

    def forward(self, patch_fine=None, patch_coarse=None, summary=None):
        parts = []
        if self.pool_fine is not None and patch_fine is not None:
            parts.append(self.pool_fine(patch_fine))
        if self.pool_coarse is not None and patch_coarse is not None:
            parts.append(self.pool_coarse(patch_coarse))
        if "+sum" in self.feat_mode and summary is not None:
            parts.append(summary)
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


class PatchAnchorRegressor(nn.Module):
    def __init__(
        self,
        n_anchors,
        pool_type="attn",
        feat_mode="both+sum",
        patch_dim=128,
        hidden_dims=(2048, 1024, 512),
        rot_hidden_dims=(1024, 512),
        embed_dim=256,
        dropout=0.15,
        attn_heads=4,
        conv_out_dim=512,
        spp_levels=None,
        anchor_temperature=0.07,
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

        self.query_head = nn.Linear(trans_dim, embed_dim)
        self.anchor_proj = nn.Parameter(torch.randn(self.n_anchors, embed_dim) * 0.02)
        self.trans_residual = nn.Sequential(
            nn.Linear(trans_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )
        self.rot_head = nn.Linear(rot_dim, 6)

        self.logit_scale = nn.Parameter(torch.tensor(float(math.log(1.0 / anchor_temperature))))
        self.s_rot = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def pool_features(self, patch_fine=None, patch_coarse=None, summary=None):
        return self.pooler(patch_fine, patch_coarse, summary)

    def forward_from_pooled(self, pooled, anchor_centers):
        x = self.input_norm(pooled)
        h_t = self.trans_backbone(x)
        h_r = self.rot_backbone(x)

        q = F.normalize(self.query_head(h_t), dim=-1)
        k = F.normalize(self.anchor_proj, dim=-1)
        logits = q @ k.t() * self.logit_scale.exp().clamp(max=100.0)
        attn = F.softmax(logits, dim=-1)

        anchor_mix = attn @ anchor_centers
        trans = anchor_mix + self.trans_residual(h_t)
        rot = gram_schmidt_6d_to_matrix(self.rot_head(h_r))
        return trans, rot, logits, attn

    def forward(self, patch_fine=None, patch_coarse=None, summary=None, anchor_centers=None):
        pooled = self.pool_features(patch_fine, patch_coarse, summary)
        return self.forward_from_pooled(pooled, anchor_centers)


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Random seed: {seed}")


def move_features(dataset, device):
    if dataset.patch_fine is not None:
        dataset.patch_fine = dataset.patch_fine.to(device)
    if dataset.patch_coarse is not None:
        dataset.patch_coarse = dataset.patch_coarse.to(device)
    if dataset.summary is not None:
        dataset.summary = dataset.summary.to(device)


def build_anchors(train_positions, n_anchors, seed):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_anchors, random_state=seed, n_init=20, max_iter=500)
    km.fit(train_positions.cpu().numpy())
    anchors = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    labels = torch.tensor(km.labels_, dtype=torch.long)
    counts = np.bincount(km.labels_, minlength=n_anchors)
    print(f"  Anchors: K={n_anchors}, inertia={km.inertia_:.1f}")
    print(
        f"  Samples per anchor: min={counts.min()}, mean={counts.mean():.1f}, "
        f"median={np.median(counts):.0f}, max={counts.max()}"
    )
    return anchors, labels


def evaluate(pred_trans, pred_rot, test_data, train_data, out_dir, extra=None):
    out_dir = Path(out_dir)
    trans_errors = (pred_trans - test_data.translations).norm(dim=-1).cpu().numpy()
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

    train_pos = train_data.translations.cpu().numpy()
    tree = cKDTree(train_pos)
    _, nn_idx = tree.query(pred_trans.cpu().numpy(), k=1)
    test_trans_np = test_data.translations.cpu().numpy()
    retr_trans_errors = np.linalg.norm(train_pos[nn_idx] - test_trans_np, axis=-1)
    retr_rot_errors = (geodesic_distance(train_data.rotations.cpu()[nn_idx], test_data.rotations.cpu()) * 180 / math.pi).numpy()
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
        results["anchors"] = extra

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Anchor Mixture + Rotation Regression)")
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

    if extra is not None:
        print("\n" + "=" * 70)
        print("ANCHORS")
        print("=" * 70)
        print(f"  Top-1 anchor acc: {extra['top1_acc']:.1f}%")
        print(f"  Top-3 anchor acc: {extra['top3_acc']:.1f}%")

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

    train_data = PatchPoseDataset(args.feature_dir, args.dataset_dir, "train", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(args.feature_dir, args.dataset_dir, "test", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)
    move_features(train_data, device)
    move_features(test_data, device)

    print("\nBuilding anchors...")
    anchor_centers_cpu, train_anchor_labels = build_anchors(train_data.translations.cpu(), args.n_anchors, args.anchor_seed)
    anchor_centers = anchor_centers_cpu.to(device)
    train_anchor_labels = train_anchor_labels.to(device)

    model = PatchAnchorRegressor(
        n_anchors=args.n_anchors,
        pool_type=args.pool,
        feat_mode=args.feat,
        patch_dim=args.patch_dim,
        hidden_dims=tuple(args.hidden_dims),
        rot_hidden_dims=tuple(args.rot_hidden_dims),
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        attn_heads=args.attn_heads,
        conv_out_dim=args.conv_out_dim,
        spp_levels=args.spp_levels,
        anchor_temperature=args.anchor_temperature,
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

    n_train = train_data.N
    batch_size = args.batch_size
    trans_scale = train_data.translations.std(dim=0).clamp(min=1e-3)

    history = {
        "epoch": [],
        "loss": [],
        "loss_trans": [],
        "loss_rot": [],
        "loss_cls": [],
        "val_rot_median": [],
        "val_trans_median": [],
        "val_top1": [],
        "val_top3": [],
        "beta_rot": [],
    }
    best_val_score = float("inf")
    best_epoch = 0
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

        target_trans = train_data.translations[indices]
        target_rot = train_data.rotations[indices]
        target_anchor = train_anchor_labels[indices]

        pred_trans, pred_rot, logits, attn = model(pf, pc, sm, anchor_centers=anchor_centers)
        loss_trans = F.smooth_l1_loss(pred_trans / trans_scale.unsqueeze(0), target_trans / trans_scale.unsqueeze(0))
        loss_cls = F.cross_entropy(logits, target_anchor)
        loss_rot = geodesic_distance(pred_rot, target_rot).mean()
        beta_rot = torch.tensor(args.rot_weight, device=device) if args.rot_weight is not None else torch.exp(model.s_rot)
        loss = args.lambda_trans * loss_trans + args.lambda_anchor_cls * loss_cls + beta_rot * loss_rot

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_trans_eval, pred_rot_eval, logits_eval, _ = model(
                    test_data.patch_fine, test_data.patch_coarse, test_data.summary, anchor_centers=anchor_centers
                )
                trans_errors = (pred_trans_eval - test_data.translations).norm(dim=-1)
                rot_errors = geodesic_distance(pred_rot_eval, test_data.rotations) * 180 / math.pi
                val_trans_med = trans_errors.median().item()
                val_rot_med = rot_errors.median().item()
                val_score = val_trans_med + val_rot_med * 0.1

                test_anchor_dists = torch.cdist(test_data.translations, anchor_centers)
                test_anchor_labels = test_anchor_dists.argmin(dim=-1)
                pred_top1 = logits_eval.argmax(dim=-1)
                top1_acc = (pred_top1 == test_anchor_labels).float().mean().item() * 100.0
                top3 = logits_eval.topk(k=min(3, args.n_anchors), dim=-1).indices
                top3_acc = (top3 == test_anchor_labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["loss_trans"].append(loss_trans.item())
            history["loss_rot"].append(loss_rot.item())
            history["loss_cls"].append(loss_cls.item())
            history["val_rot_median"].append(val_rot_med)
            history["val_trans_median"].append(val_trans_med)
            history["val_top1"].append(top1_acc)
            history["val_top3"].append(top3_acc)
            history["beta_rot"].append(beta_rot.item())

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                r10 = ((rot_errors < 10) & (trans_errors < 2.0)).float().mean().item() * 100.0
                r5 = ((rot_errors < 5) & (trans_errors < 1.0)).float().mean().item() * 100.0
                print(
                    f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                    f"(t={loss_trans.item():.4f} cls={loss_cls.item():.4f} rot={loss_rot.item():.4f} beta={beta_rot.item():.2f}) "
                    f"| val: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                    f"R@10/2={r10:.1f}% R@5/1={r5:.1f}% a@1={top1_acc:.1f}% a@3={top3_acc:.1f}% | {time.time()-t0:.1f}s"
                )

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_rot_median": val_rot_med,
                        "val_trans_median": val_trans_med,
                        "anchor_centers": anchor_centers_cpu,
                        "config": config,
                    },
                    out_dir / "model_best.pt",
                )

    print(f"\nTraining done. Best epoch: {best_epoch}, score: {best_val_score:.4f}")
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(out_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_trans, pred_rot, logits, _ = model(
            test_data.patch_fine, test_data.patch_coarse, test_data.summary, anchor_centers=anchor_centers
        )
        test_anchor_dists = torch.cdist(test_data.translations, anchor_centers)
        test_anchor_labels = test_anchor_dists.argmin(dim=-1)
        pred_top1 = logits.argmax(dim=-1)
        top1_acc = (pred_top1 == test_anchor_labels).float().mean().item() * 100.0
        top3 = logits.topk(k=min(3, args.n_anchors), dim=-1).indices
        top3_acc = (top3 == test_anchor_labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

    return evaluate(
        pred_trans,
        pred_rot,
        test_data,
        train_data,
        out_dir,
        extra={"top1_acc": float(top1_acc), "top3_acc": float(top3_acc)},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-token anchor regressor")
    parser.add_argument("--pool", type=str, default="attn", choices=["gap", "gem", "spp", "attn", "conv"])
    parser.add_argument("--feat", type=str, default="both+sum", choices=["fine", "coarse", "both", "fine+sum", "coarse+sum", "both+sum"])
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
    parser.add_argument("--feature_dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[2048, 1024, 512])
    parser.add_argument("--rot_hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--conv_out_dim", type=int, default=512)
    parser.add_argument("--spp_levels", type=int, nargs="+", default=None)
    parser.add_argument("--patch_dim", type=int, default=128)
    parser.add_argument("--n_anchors", type=int, default=64)
    parser.add_argument("--anchor_seed", type=int, default=42)
    parser.add_argument("--anchor_temperature", type=float, default=0.07)
    parser.add_argument("--lambda_trans", type=float, default=1.0)
    parser.add_argument("--lambda_anchor_cls", type=float, default=0.2)
    parser.add_argument("--rot_weight", type=float, default=None)
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
