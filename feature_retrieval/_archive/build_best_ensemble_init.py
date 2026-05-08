#!/usr/bin/env python3
"""Export the current best ensemble initializer to init-pose npz."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from data.radio_loc_retrieval_dataset import list_colmap_split_samples, save_retrieval_init_entries
from feature_retrieval.decoupled_init_search import (
    FEATURE_DIR,
    DATASET_DIR,
    BASE,
    load_any_model,
    TRANSLATION_MODELS,
    average_rotations,
)


def center_rot_to_w2c(center: np.ndarray, rot_w2c: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot_w2c.astype(np.float32)
    pose[:3, 3] = (-rot_w2c @ center.reshape(3, 1)).reshape(3).astype(np.float32)
    return pose


def weighted_translation(stacked_trans, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.einsum("i,ijk->jk", w, stacked_trans)


def main():
    out_path = Path(BASE) / "best_ensemble_init_top5.npz"
    device_cycle = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    # Base-5 current strongest translation ensemble + small additives from local refinement.
    trans_cfg = {
        "e123": 6.0 * (2.266666666666667 / (2.266666666666667 + 1.4 + 0.9666666666666667 + 0.1 + 0.5333333333333333)),
        "e38": 6.0 * (1.4 / (2.266666666666667 + 1.4 + 0.9666666666666667 + 0.1 + 0.5333333333333333)),
        "f42": 6.0 * (0.9666666666666667 / (2.266666666666667 + 1.4 + 0.9666666666666667 + 0.1 + 0.5333333333333333)),
        "e40": 6.0 * (0.1 / (2.266666666666667 + 1.4 + 0.9666666666666667 + 0.1 + 0.5333333333333333)),
        "e47": 6.0 * (0.5333333333333333 / (2.266666666666667 + 1.4 + 0.9666666666666667 + 0.1 + 0.5333333333333333)),
        "f15": 0.2,
        "e60": 0.1,
        "s314": 0.05,
    }
    trans_names = list(trans_cfg.keys())
    preds = {}
    for i, name in enumerate(trans_names):
        preds[name] = load_any_model(TRANSLATION_MODELS[name], FEATURE_DIR, DATASET_DIR, device_cycle[i % len(device_cycle)])

    stacked_trans = np.stack([preds[n]["trans_pred"] for n in trans_names], axis=0)
    trans_pred = weighted_translation(stacked_trans, [trans_cfg[n] for n in trans_names])
    rot_pred = preds["f15"]["rot_pred"].cpu().numpy()

    train_samples = list_colmap_split_samples(f"{DATASET_DIR}/sparse/0", f"{DATASET_DIR}/dataset_train.txt")
    query_samples = list_colmap_split_samples(f"{DATASET_DIR}/sparse/0", f"{DATASET_DIR}/dataset_test.txt")
    train_positions = np.stack([(-s["pose_w2c"][:3, :3].T @ s["pose_w2c"][:3, 3]).astype(np.float32) for s in train_samples], axis=0)
    tree = cKDTree(train_positions)
    dists, nn_idx = tree.query(trans_pred, k=5)

    entries = []
    for i, query in enumerate(query_samples):
        candidate_poses = np.stack([train_samples[int(j)]["pose_w2c"].copy() for j in nn_idx[i]], axis=0).astype(np.float32)
        candidate_poses[0] = center_rot_to_w2c(trans_pred[i].astype(np.float32), rot_pred[i].astype(np.float32))
        entries.append(
            {
                "query_img_id": int(query["img_id"]),
                "query_image_name": query["image_name"],
                "query_image_stem": query["image_stem"],
                "pose_init": candidate_poses[0],
                "init_source": "learned_best_ensemble",
                "retrieval_frame_id": int(train_samples[int(nn_idx[i][0])]["img_id"]),
                "retrieval_image_name": train_samples[int(nn_idx[i][0])]["image_name"],
                "retrieval_score": float(-dists[i][0]),
                "pose_init_candidates": candidate_poses,
                "candidate_valid_mask": np.ones((5,), dtype=bool),
                "retrieval_frame_ids_candidates": np.array([int(train_samples[int(j)]["img_id"]) for j in nn_idx[i]], dtype=np.int64),
                "retrieval_image_names_candidates": np.array([train_samples[int(j)]["image_name"] for j in nn_idx[i]]),
                "retrieval_scores_candidates": (-dists[i]).astype(np.float32),
            }
        )

    stats = {
        "method_requested": "learned_ensemble",
        "method_used": "learned_best_ensemble",
        "retrieval_topk_requested": 5,
        "num_train_samples": len(train_samples),
        "num_query_samples": len(query_samples),
        "counts_by_source": {"learned_best_ensemble": len(entries)},
        "translation_weights": trans_cfg,
        "rotation_source": "f15",
    }
    save_retrieval_init_entries(entries, stats, str(out_path))
    print(f"Saved best ensemble init to {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
