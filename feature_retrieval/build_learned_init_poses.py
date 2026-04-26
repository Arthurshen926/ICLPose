#!/usr/bin/env python3
"""Build init-pose cache from learned feature-retrieval models.

Outputs the same .npz schema used by RadioLocRetrievalDataset so that
all learned initializers share one runtime evaluation path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.radio_loc_retrieval_dataset import list_colmap_split_samples, save_retrieval_init_entries
from feature_retrieval.cross_arch_ensemble import load_and_predict
from feature_retrieval.load_center_frustum import load_center_frustum_predictions
from feature_retrieval.patch_regressor_v7 import PatchPoseDataset, geodesic_distance


def center_rot_to_w2c(center: np.ndarray, rot_w2c: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot_w2c.astype(np.float32)
    pose[:3, 3] = (-rot_w2c @ center.reshape(3, 1)).reshape(3).astype(np.float32)
    return pose


def centers_rots_to_pose_batch(centers: np.ndarray, rot_w2c: np.ndarray) -> np.ndarray:
    poses = np.zeros((centers.shape[0], 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1.0
    poses[:, :3, :3] = rot_w2c[None]
    poses[:, :3, 3] = (-np.einsum("ij,nj->ni", rot_w2c, centers)).astype(np.float32)
    return poses


def parse_args():
    parser = argparse.ArgumentParser(description="Build learned init poses cache")
    parser.add_argument("--exp_dir", required=True, help="Model experiment directory")
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_path", required=True)
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "patch_regressor", "center_frustum"],
        help="How to load predictions from exp_dir",
    )
    parser.add_argument(
        "--candidate_policy",
        default="nearest_train_translation",
        choices=["nearest_train_translation", "oracle_gt_translation"],
        help="How to turn predicted continuous centers into top-k train pose candidates",
    )
    return parser.parse_args()


def build_entries_from_prediction(
    exp_dir: str,
    feature_dir: str,
    dataset_dir: str,
    colmap_dir: str,
    train_split: str,
    query_split: str,
    save_path: str,
    topk: int,
    gpu: int,
    candidate_policy: str,
    model_type: str,
):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    split_name = Path(query_split).name
    if split_name.startswith("dataset_") and split_name.endswith(".txt"):
        split_name = split_name[len("dataset_"):-len(".txt")]
    else:
        raise ValueError(
            f"Unable to infer split name from query_split={query_split}; expected dataset_<split>.txt"
        )

    if model_type == "center_frustum":
        pred = load_center_frustum_predictions(
            exp_dir,
            feature_dir,
            dataset_dir,
            device=device,
            split=split_name,
        )
    elif model_type == "patch_regressor":
        pred = load_and_predict(
            exp_dir,
            feature_dir,
            dataset_dir,
            device=device,
            split=split_name,
        )
    else:
        with open(Path(exp_dir) / "config.json", "r") as f:
            cfg = json.load(f)
        if cfg.get("n_anchors") is not None and cfg.get("lambda_fwd") is not None and cfg.get("lambda_up") is not None:
            pred = load_center_frustum_predictions(
                exp_dir,
                feature_dir,
                dataset_dir,
                device=device,
                split=split_name,
            )
        else:
            pred = load_and_predict(
                exp_dir,
                feature_dir,
                dataset_dir,
                device=device,
                split=split_name,
            )

    train_pose_ds = PatchPoseDataset(
        feature_dir,
        dataset_dir,
        "train",
        "cpu",
        use_fine=False,
        use_coarse=False,
        use_summary=True,
    )
    query_pose_ds = PatchPoseDataset(
        feature_dir,
        dataset_dir,
        split_name,
        "cpu",
        use_fine=False,
        use_coarse=False,
        use_summary=True,
    )
    train_positions = train_pose_ds.translations.cpu().numpy()
    train_rot = train_pose_ds.rotations.cpu().numpy()
    query_positions = query_pose_ds.translations.cpu().numpy()
    query_rot = query_pose_ds.rotations.cpu()

    train_samples = list_colmap_split_samples(colmap_dir, train_split)
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    if len(query_samples) != len(pred["trans_pred"]):
        raise RuntimeError(f"Prediction count mismatch: {len(pred['trans_pred'])} vs {len(query_samples)}")
    if len(query_samples) != len(query_positions):
        raise RuntimeError(f"Query split size mismatch: {len(query_samples)} vs {len(query_positions)}")

    # Train sample order in feature_retrieval follows dataset_train.txt order, which should match split enumeration.
    if len(train_samples) != len(train_positions):
        raise RuntimeError(f"Train split size mismatch: {len(train_samples)} vs {len(train_positions)}")

    tree = cKDTree(train_positions)
    if candidate_policy == "oracle_gt_translation":
        dists, nn_idx = tree.query(query_positions, k=min(topk, len(train_positions)))
    else:
        dists, nn_idx = tree.query(pred["trans_pred"], k=min(topk, len(train_positions)))

    if topk == 1:
        nn_idx = nn_idx[:, None]
        dists = dists[:, None]

    entries: List[Dict] = []
    rot_pred = pred["rot_pred"]
    rot_pred_np = rot_pred.cpu().numpy()

    source_name = f"learned_{Path(exp_dir).name}"
    for i, query in enumerate(query_samples):
        cand_indices = nn_idx[i]
        cand_scores = -np.asarray(dists[i], dtype=np.float32)
        cand_valid = np.ones((len(cand_indices),), dtype=bool)

        if model_type == "center_frustum" or (model_type == "auto" and "topk_centers" in pred):
            topk_centers = pred["topk_centers"][i].astype(np.float32)
            topk_scores = pred["topk_scores"][i].astype(np.float32)
            candidate_poses = centers_rots_to_pose_batch(topk_centers, rot_pred_np[i].astype(np.float32))
            cand_scores = topk_scores
            meta_tree_dists, meta_tree_idx = tree.query(topk_centers, k=1)
            cand_indices = np.asarray(meta_tree_idx, dtype=np.int64)
        else:
            candidate_poses = np.stack([train_samples[int(j)]["pose_w2c"].copy() for j in cand_indices], axis=0).astype(np.float32)
            learned_pose = center_rot_to_w2c(pred["trans_pred"][i].astype(np.float32), rot_pred_np[i].astype(np.float32))
            candidate_poses[0] = learned_pose

        entries.append(
            {
                "query_img_id": int(query["img_id"]),
                "query_image_name": query["image_name"],
                "query_image_stem": query["image_stem"],
                "pose_init": candidate_poses[0].astype(np.float32),
                "init_source": source_name,
                "retrieval_frame_id": int(train_samples[int(cand_indices[0])]["img_id"]),
                "retrieval_image_name": train_samples[int(cand_indices[0])]["image_name"],
                "retrieval_score": float(cand_scores[0]),
                "pose_init_candidates": candidate_poses.astype(np.float32),
                "candidate_valid_mask": cand_valid.astype(bool),
                "retrieval_frame_ids_candidates": np.array([int(train_samples[int(j)]["img_id"]) for j in cand_indices], dtype=np.int64),
                "retrieval_image_names_candidates": np.array([train_samples[int(j)]["image_name"] for j in cand_indices]),
                "retrieval_scores_candidates": cand_scores.astype(np.float32),
            }
        )

    # Diagnostics
    top1_pose = np.stack([e["pose_init"] for e in entries], axis=0)
    top1_rot = torch.from_numpy(top1_pose[:, :3, :3]).float()
    top1_center = pred["trans_pred"].astype(np.float32)
    rot_err = (geodesic_distance(top1_rot, query_rot) * 180 / math.pi).numpy()
    trans_err = np.linalg.norm(top1_center - query_positions, axis=-1) * 1000.0

    def pct(mask):
        return float(np.mean(mask) * 100.0)

    stats = {
        "method_requested": "learned",
        "method_used": source_name,
        "retrieval_topk_requested": int(topk),
        "num_train_samples": int(len(train_samples)),
        "num_query_samples": int(len(query_samples)),
        "counts_by_source": {source_name: int(len(entries))},
        "candidate_policy": candidate_policy,
        "query_split_name": split_name,
        "top1_rot_median_deg": float(np.median(rot_err)),
        "top1_trans_median_mm": float(np.median(trans_err)),
        "top1_rot_pass_5deg": pct(rot_err < 5.0),
        "top1_rot_pass_10deg": pct(rot_err < 10.0),
        "top1_trans_pass_1000mm": pct(trans_err < 1000.0),
        "top1_trans_pass_2000mm": pct(trans_err < 2000.0),
        "top1_joint_5deg_1m": pct((rot_err < 5.0) & (trans_err < 1000.0)),
        "top1_joint_10deg_2m": pct((rot_err < 10.0) & (trans_err < 2000.0)),
    }

    save_retrieval_init_entries(entries, stats, save_path)
    print(f"Saved {len(entries)} learned init poses to {save_path}")
    print(json.dumps(stats, indent=2))


def main():
    args = parse_args()
    build_entries_from_prediction(
        exp_dir=args.exp_dir,
        feature_dir=args.feature_dir,
        dataset_dir=args.dataset_dir,
        colmap_dir=args.colmap_dir,
        train_split=args.train_split,
        query_split=args.query_split,
        save_path=args.save_path,
        topk=args.topk,
        gpu=args.gpu,
        candidate_policy=args.candidate_policy,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
