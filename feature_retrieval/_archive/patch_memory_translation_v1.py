#!/usr/bin/env python3
"""Frozen patch pooling + memory-based translation + decoupled rotation.

Translation is predicted as:
  softmax(query_embed @ train_embed^T / T) @ train_centers + residual

Rotation is predicted by an independent regression branch.
The pooling front-end is frozen from a strong patch-regression checkpoint.
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
    PatchPoseDataset,
    PatchPoseRegressor,
    geodesic_distance,
    gram_schmidt_6d_to_matrix,
    precompute_pooled_features,
)


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Random seed: {seed}")


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


class MemoryTranslationPoseNet(nn.Module):
    def __init__(
        self,
        input_dim,
        trans_hidden_dims=(1024, 512),
        rot_hidden_dims=(1024, 512),
        embed_dim=256,
        dropout=0.1,
        init_temperature=0.07,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.trans_backbone, trans_dim = make_mlp(input_dim, trans_hidden_dims, dropout)
        self.rot_backbone, rot_dim = make_mlp(input_dim, rot_hidden_dims, dropout)
        self.query_head = nn.Linear(trans_dim, embed_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(trans_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )
        self.rot_head = nn.Linear(rot_dim, 6)
        self.logit_scale = nn.Parameter(torch.tensor(float(math.log(1.0 / init_temperature))))
        self.s_rot = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_query(self, pooled):
        x = self.input_norm(pooled)
        h_t = self.trans_backbone(x)
        q = F.normalize(self.query_head(h_t), dim=-1)
        residual = self.residual_head(h_t)
        return q, residual

    def encode_memory(self, pooled):
        x = self.input_norm(pooled)
        h_t = self.trans_backbone(x)
        return F.normalize(self.query_head(h_t), dim=-1)

    def predict_rotation(self, pooled):
        x = self.input_norm(pooled)
        h_r = self.rot_backbone(x)
        return gram_schmidt_6d_to_matrix(self.rot_head(h_r))

    def forward(self, query_pooled, memory_pooled, memory_trans_norm, exclude_indices=None):
        q, residual = self.encode_query(query_pooled)
        k = self.encode_memory(memory_pooled)
        logits = q @ k.t() * self.logit_scale.exp().clamp(max=100.0)
        if exclude_indices is not None:
            logits = logits.clone()
            logits[torch.arange(logits.shape[0], device=logits.device), exclude_indices] = -1e9
        attn = F.softmax(logits, dim=-1)
        trans_norm = attn @ memory_trans_norm + residual
        rot = self.predict_rotation(query_pooled)
        return trans_norm, rot, logits, attn


def load_frozen_pooler(exp_dir, device):
    exp_dir = Path(exp_dir)
    with open(exp_dir / "config.json", "r") as handle:
        config = json.load(handle)
    hidden_dims = config.get("hidden_dims", [1024, 512, 256])
    if isinstance(hidden_dims, str):
        hidden_dims = [int(x) for x in hidden_dims.split(",")]

    model = PatchPoseRegressor(
        pool_type=config["pool"],
        feat_mode=config["feat"],
        patch_dim=config["patch_dim"],
        hidden_dims=tuple(hidden_dims),
        dropout=config.get("dropout", 0.15),
        attn_heads=config.get("attn_heads", 4),
        conv_out_dim=config.get("conv_out_dim", 512),
        spp_levels=config.get("spp_levels", None),
    ).to(device)
    ckpt = torch.load(exp_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def evaluate_predictions(pred_trans, pred_rot, test_data, train_data, out_dir, extra=None):
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

    train_positions = train_data.translations.cpu().numpy()
    pred_positions = pred_trans.cpu().numpy()
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
        results["memory"] = extra

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Memory Translation + Rotation Regression)")
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
        print("MEMORY")
        print("=" * 70)
        for key, value in extra.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")

    with open(out_dir / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    return results


def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooler_model, pooler_cfg = load_frozen_pooler(args.pooler_exp_dir, device)
    use_fine = "fine" in pooler_cfg["feat"] or "both" in pooler_cfg["feat"]
    use_coarse = "coarse" in pooler_cfg["feat"] or "both" in pooler_cfg["feat"]
    use_summary = "+sum" in pooler_cfg["feat"]

    print(f"\nLoading raw data for frozen pooling (fine={use_fine}, coarse={use_coarse}, summary={use_summary})...")
    train_data = PatchPoseDataset(args.feature_dir, args.dataset_dir, "train", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(args.feature_dir, args.dataset_dir, "test", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)

    print("\nPrecomputing frozen pooled features...")
    train_pooled = precompute_pooled_features(pooler_model, train_data, device, batch_size=16)
    test_pooled = precompute_pooled_features(pooler_model, test_data, device, batch_size=16)

    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)

    trans_mean = train_data.translations.mean(dim=0)
    trans_std = train_data.translations.std(dim=0).clamp(min=1e-6)
    train_trans_norm = (train_data.translations - trans_mean) / trans_std
    test_trans_norm = (test_data.translations - trans_mean) / trans_std

    model = MemoryTranslationPoseNet(
        input_dim=train_pooled.shape[1],
        trans_hidden_dims=tuple(args.trans_hidden_dims),
        rot_hidden_dims=tuple(args.rot_hidden_dims),
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        init_temperature=args.memory_temperature,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    config = vars(args).copy()
    config["output_dir"] = str(out_dir)
    config["pooled_dim"] = int(train_pooled.shape[1])
    config["n_params"] = int(n_params)
    with open(out_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        "epoch": [],
        "loss": [],
        "loss_trans": [],
        "loss_rank": [],
        "loss_rot": [],
        "val_rot_median": [],
        "val_trans_median": [],
        "attn_top1_2m": [],
        "beta_rot": [],
    }
    best_val_score = float("inf")
    best_epoch = 0
    n_train = train_pooled.shape[0]
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.batch_size >= n_train:
            indices = torch.arange(n_train, device=device)
        else:
            indices = torch.randperm(n_train, device=device)[: args.batch_size]

        query_pooled = train_pooled[indices]
        target_trans_norm = train_trans_norm[indices]
        target_rot = train_data.rotations[indices]

        pred_trans_norm, pred_rot, logits, attn = model(
            query_pooled,
            train_pooled,
            train_trans_norm,
            exclude_indices=indices,
        )

        dist_matrix = torch.cdist(train_data.translations[indices], train_data.translations)
        dist_matrix[torch.arange(indices.shape[0], device=device), indices] = 1e9
        target_weights = F.softmax(-dist_matrix / max(float(args.label_temperature), 1e-6), dim=-1)

        loss_trans = F.smooth_l1_loss(pred_trans_norm, target_trans_norm)
        loss_rank = -(target_weights * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        loss_rot = geodesic_distance(pred_rot, target_rot).mean()
        beta_rot = torch.tensor(args.rot_weight, device=device) if args.rot_weight is not None else torch.exp(model.s_rot)
        loss = args.lambda_trans * loss_trans + args.lambda_rank * loss_rank + beta_rot * loss_rot

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_test_norm, pred_test_rot, logits_test, attn_test = model(
                    test_pooled,
                    train_pooled,
                    train_trans_norm,
                )
                pred_test_trans = pred_test_norm * trans_std.unsqueeze(0) + trans_mean.unsqueeze(0)
                trans_errors = (pred_test_trans - test_data.translations).norm(dim=-1)
                rot_errors = geodesic_distance(pred_test_rot, test_data.rotations) * 180 / math.pi
                val_trans_med = trans_errors.median().item()
                val_rot_med = rot_errors.median().item()
                val_score = val_trans_med + val_rot_med * 0.1

                top1_idx = attn_test.argmax(dim=-1)
                top1_trans_err = (train_data.translations[top1_idx] - test_data.translations).norm(dim=-1)
                attn_top1_2m = (top1_trans_err < 2.0).float().mean().item() * 100.0

            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["loss_trans"].append(loss_trans.item())
            history["loss_rank"].append(loss_rank.item())
            history["loss_rot"].append(loss_rot.item())
            history["val_rot_median"].append(val_rot_med)
            history["val_trans_median"].append(val_trans_med)
            history["attn_top1_2m"].append(attn_top1_2m)
            history["beta_rot"].append(beta_rot.item())

            if epoch % (args.log_every * 10) == 0 or epoch == 1:
                r10 = ((rot_errors < 10) & (trans_errors < 2.0)).float().mean().item() * 100.0
                r5 = ((rot_errors < 5) & (trans_errors < 1.0)).float().mean().item() * 100.0
                print(
                    f"[{epoch:5d}/{args.epochs}] loss={loss.item():.4f} "
                    f"(t={loss_trans.item():.4f} rank={loss_rank.item():.4f} rot={loss_rot.item():.4f} beta={beta_rot.item():.2f}) "
                    f"| val: {val_rot_med:.2f}deg/{val_trans_med*1000:.0f}mm "
                    f"R@10/2={r10:.1f}% R@5/1={r5:.1f}% attn_top1<2m={attn_top1_2m:.1f}% | {time.time()-t0:.1f}s"
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
                        "config": config,
                    },
                    out_dir / "model_best.pt",
                )

    print(f"\nTraining done. Best epoch: {best_epoch}, score: {best_val_score:.4f}")
    with open(out_dir / "training_log.json", "w") as handle:
        json.dump(history, handle, indent=2)

    ckpt = torch.load(out_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_test_norm, pred_test_rot, logits_test, attn_test = model(test_pooled, train_pooled, train_trans_norm)
        pred_test_trans = pred_test_norm * trans_std.unsqueeze(0) + trans_mean.unsqueeze(0)
        top1_idx = attn_test.argmax(dim=-1)
        top1_trans_err = (train_data.translations[top1_idx] - test_data.translations).norm(dim=-1)
        attn_top1_2m = (top1_trans_err < 2.0).float().mean().item() * 100.0
        attn_top3_idx = logits_test.topk(k=min(3, logits_test.shape[-1]), dim=-1).indices
        top3_trans = train_data.translations[attn_top3_idx]
        top3_err = torch.norm(top3_trans - test_data.translations.unsqueeze(1), dim=-1)
        attn_top3_2m = (top3_err.min(dim=1).values < 2.0).float().mean().item() * 100.0

    return evaluate_predictions(
        pred_test_trans,
        pred_test_rot,
        test_data,
        train_data,
        out_dir,
        extra={
            "attn_top1_lt2m": float(attn_top1_2m),
            "attn_top3_lt2m": float(attn_top3_2m),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Memory translation with frozen patch pooling")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--feature_dir", type=str, default="output/feature_extract/features_radio_dual_128/OldHospital_pilot")
    parser.add_argument("--dataset_dir", type=str, default="dataset/OldHospital")
    parser.add_argument("--pooler_exp_dir", type=str, default="output/feature_retrieval/pose_regression/exp24e_attn_h4_wider_seed123")
    parser.add_argument("--output_base", type=str, default="output/feature_retrieval/pose_regression")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--trans_hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--rot_hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--memory_temperature", type=float, default=0.07)
    parser.add_argument("--label_temperature", type=float, default=2.0)
    parser.add_argument("--lambda_trans", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.2)
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
