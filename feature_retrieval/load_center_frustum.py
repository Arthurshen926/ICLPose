#!/usr/bin/env python3
"""Utility to load center/frustum checkpoints and predict on test split."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from feature_retrieval.patch_center_frustum_v1 import PatchCenterFrustumNet, predict_from_logits, decode_topk_proposals
from feature_retrieval.patch_regressor_v7 import PatchPoseDataset


def load_center_frustum_predictions(
    exp_dir: str,
    feature_dir: str,
    dataset_dir: str,
    device: str = "cuda:0",
    split: str = "test",
):
    exp_dir = Path(exp_dir)
    with open(exp_dir / "config.json", "r") as f:
        cfg = json.load(f)

    use_fine = "fine" in cfg["feat"] or "both" in cfg["feat"]
    use_coarse = "coarse" in cfg["feat"] or "both" in cfg["feat"]
    use_summary = "+sum" in cfg["feat"]
    test_data = PatchPoseDataset(
        feature_dir,
        dataset_dir,
        split,
        "cpu",
        use_fine=use_fine,
        use_coarse=use_coarse,
        use_summary=use_summary,
    )
    if test_data.patch_fine is not None:
        test_data.patch_fine = test_data.patch_fine.to(device)
    if test_data.patch_coarse is not None:
        test_data.patch_coarse = test_data.patch_coarse.to(device)
    if test_data.summary is not None:
        test_data.summary = test_data.summary.to(device)

    model = PatchCenterFrustumNet(
        n_anchors=cfg["n_anchors"],
        pool_type=cfg["pool"],
        feat_mode=cfg["feat"],
        patch_dim=cfg["patch_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        rot_hidden_dims=tuple(cfg["rot_hidden_dims"]),
        dropout=cfg["dropout"],
        attn_heads=cfg.get("attn_heads", 4),
        conv_out_dim=cfg.get("conv_out_dim", 512),
        spp_levels=cfg.get("spp_levels"),
    ).to(device)
    ckpt = torch.load(exp_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    anchor_info = torch.load(exp_dir / "anchor_info.pt", map_location="cpu")
    anchors = anchor_info["anchors"].to(device)
    residual_std = anchor_info.get("residual_std")
    if residual_std is not None:
        residual_std = residual_std.to(device)

    with torch.no_grad():
        logits, residuals, _, _, rot = model(test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        center, _, _ = predict_from_logits(
            logits,
            residuals,
            anchors,
            residual_std=residual_std,
            mode=cfg.get("center_mode", "mixture"),
            topk=cfg.get("center_topk", 3),
            temperature=cfg.get("center_temperature", 1.0),
        )
        topk_idx, topk_scores, topk_centers = decode_topk_proposals(
            logits,
            residuals,
            anchors,
            residual_std=residual_std,
            topk=cfg.get("eval_topk_anchor", 5),
        )

    return {
        "trans_pred": center.cpu().numpy(),
        "rot_pred": rot.cpu(),
        "topk_centers": topk_centers.cpu().numpy(),
        "topk_scores": topk_scores.cpu().numpy(),
        "topk_indices": topk_idx.cpu().numpy(),
        "name": exp_dir.name,
        "config": cfg,
    }
