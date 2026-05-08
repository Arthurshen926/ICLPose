#!/usr/bin/env python3
"""Analyze top-k cell ceiling for patch grid classifier checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_retrieval.patch_grid_classifier_v1 import (
    PatchGridClassifier,
    PatchPoseDataset,
    compute_camera_kmeans,
    get_all_cell_centers,
    geodesic_distance,
)


def load_model(exp_dir: Path, device: torch.device):
    with open(exp_dir / "config.json", "r") as handle:
        cfg = json.load(handle)

    model = PatchGridClassifier(
        n_cells=cfg["n_cells"],
        pool_type=cfg["pool"],
        feat_mode=cfg["feat"],
        patch_dim=cfg["patch_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        attn_heads=cfg.get("attn_heads", 4),
        conv_out_dim=cfg.get("conv_out_dim", 512),
        spp_levels=cfg.get("spp_levels"),
        decouple_rot=cfg.get("decouple_rot", False),
        rot_hidden_dims=tuple(cfg["rot_hidden_dims"]) if cfg.get("rot_hidden_dims") else None,
    ).to(device)
    ckpt = torch.load(exp_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(exp_dir, device)

    use_fine = "fine" in cfg["feat"] or "both" in cfg["feat"]
    use_coarse = "coarse" in cfg["feat"] or "both" in cfg["feat"]
    use_summary = "+sum" in cfg["feat"]

    train_data = PatchPoseDataset(
        cfg["feature_dir"], cfg["dataset_dir"], "train", "cpu",
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary,
    )
    test_data = PatchPoseDataset(
        cfg["feature_dir"], cfg["dataset_dir"], "test", "cpu",
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary,
    )
    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)
    if train_data.patch_fine is not None:
        train_data.patch_fine = train_data.patch_fine.to(device)
    if train_data.patch_coarse is not None:
        train_data.patch_coarse = train_data.patch_coarse.to(device)
    if train_data.summary is not None:
        train_data.summary = train_data.summary.to(device)
    if test_data.patch_fine is not None:
        test_data.patch_fine = test_data.patch_fine.to(device)
    if test_data.patch_coarse is not None:
        test_data.patch_coarse = test_data.patch_coarse.to(device)
    if test_data.summary is not None:
        test_data.summary = test_data.summary.to(device)

    centroids, train_labels, test_labels, _ = compute_camera_kmeans(
        train_data.translations.cpu(), test_data.translations.cpu(), cfg["n_cells"], cfg["kmeans_seed"]
    )
    centroids = centroids.to(device)
    train_labels = train_labels.to(device)
    residual_std = (train_data.translations - centroids[train_labels]).std(dim=0).clamp(min=1e-3)

    with torch.no_grad():
        cls_logits, residual_all, rot_pred = model(
            test_data.patch_fine, test_data.patch_coarse, test_data.summary
        )
        cell_centers = get_all_cell_centers(residual_all, centroids, residual_std)

    gt_trans = test_data.translations.unsqueeze(1)
    cell_dists = torch.norm(cell_centers - gt_trans, dim=-1)

    rot_errors = (geodesic_distance(rot_pred, test_data.rotations) * 180.0 / np.pi).cpu().numpy()

    print("=" * 70)
    print(exp_dir.name)
    print("=" * 70)
    print(f"Rotation median: {np.median(rot_errors):.2f} deg")

    for k in [1, 2, 3, 5, 8]:
        top_idx = cls_logits.topk(k=min(k, cls_logits.shape[-1]), dim=-1).indices
        top_dists = torch.gather(cell_dists, 1, top_idx)
        best_trans = top_dists.min(dim=1).values.cpu().numpy()
        r10 = ((rot_errors < 10.0) & (best_trans < 2.0)).mean() * 100.0
        r5 = ((rot_errors < 5.0) & (best_trans < 1.0)).mean() * 100.0
        print(
            f"Top-{k} cell oracle: median_trans={np.median(best_trans)*1000:.0f}mm "
            f"R@10/2={r10:.1f}% R@5/1={r5:.1f}%"
        )


if __name__ == "__main__":
    main()
