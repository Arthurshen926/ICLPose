"""Fast oracle-only dense context diagnostic without match-object construction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _limit_submap,
    _load_camera_with_source,
    _load_query_feature,
    _load_reference_submaps,
    _load_track_stats,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.dense_context_oracle_metrics import score_metrics_from_matrices
from feature_extract.vfm.dense_patch_context import align_dense_context_to_landmarks, dense_context_scores
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    _flatten_query_features,
    _query_landmark_topk,
    build_patch_positive_sets,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _positive_label_matrix(top_indices: np.ndarray, token_indices: np.ndarray, track_ids: np.ndarray, positives) -> np.ndarray:
    labels = np.zeros(top_indices.shape, dtype=bool)
    for row, token in enumerate(token_indices.tolist()):
        positive = positives.by_token.get(int(token))
        if positive is None:
            continue
        labels[row] = np.asarray([int(track_ids[int(idx)]) in positive.track_ids for idx in top_indices[row]], dtype=bool)
    return labels


def _markdown_summary(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Fast Dense Context Oracle Diagnostic",
        "",
        f"- diagnostic pass: `{summary.get('diagnostic_pass')}`",
        f"- best combined score: `{summary.get('best_combined_score')}`",
        "",
        "| Score | AUROC | AUPRC | Top1 acc | Mean best positive rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in dict(summary.get("score_metrics") or {}).items():
        auroc = "n/a" if metric["auroc"] is None else f"{float(metric['auroc']):.4f}"
        auprc = "n/a" if metric["auprc"] is None else f"{float(metric['auprc']):.4f}"
        rank = "n/a" if metric["mean_best_positive_rank"] is None else f"{float(metric['mean_best_positive_rank']):.3f}"
        lines.append(f"| {name} | {auroc} | {auprc} | {float(metric['top1_accuracy']):.4f} | {rank} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Fast dense context oracle ranking diagnostic")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--dense_context_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--visibility_index", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--top_m_anchor", type=int, default=5)
    parser.add_argument("--context_mode", default="max_proto", choices=("mean", "medoid", "max_proto", "topk_avg"))
    parser.add_argument("--topk_average_k", type=int, default=3)
    parser.add_argument("--context_weights", default="0.05,0.1,0.2")
    parser.add_argument("--apply_reliability", action="store_true")
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--min_anchor_similarity", type=float, default=0.2)
    parser.add_argument("--match_block_size", type=int, default=1024)
    parser.add_argument("--similarity_device", default="cpu")
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--min_auc_delta", type=float, default=0.005)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--summary_md", default="")
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    xyz_by_track, reprojection_by_track = _load_track_stats(Path(args.track_observations))
    sparse = LandmarkMapIndex.from_track_bank(load_selected_track_bank_npz(Path(args.landmark_bank)), xyz_by_track, reprojection_by_track)
    from feature_extract.vfm.dense_patch_context import DensePatchContextBank

    context_bank = DensePatchContextBank.load_npz(Path(args.dense_context_bank))
    visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index))
    reference_submaps = _load_reference_submaps(args.candidate_bank, int(args.submap_top_n))
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    context_weights = _parse_floats(args.context_weights)
    score_names = ["anchor", f"context_{args.context_mode}"] + [
        f"anchor_context_{args.context_mode}_w{str(weight).replace('.', 'p')}" for weight in context_weights
    ]

    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    rows = []
    global_labels = []
    global_valid = []
    global_scores: dict[str, list[np.ndarray]] = {name: [] for name in score_names}
    for record in records:
        pose = poses.get(record.image_id)
        if pose is None:
            continue
        references = reference_submaps.get(record.image_id, [])
        submap, visibility_gate = filter_landmarks_by_visibility(sparse, visibility_index, references)
        submap = _limit_submap(submap, int(args.max_submap_landmarks))
        submap, local_context = align_dense_context_to_landmarks(submap, context_bank)
        if len(submap) == 0:
            continue
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        query_features, token_indices = _flatten_query_features(query_feature, int(args.query_token_step))
        query_features, valid_query = normalize_rows(query_features)
        valid_rows = np.flatnonzero(valid_query)
        if valid_rows.size == 0:
            continue
        query_features = query_features[valid_rows]
        token_indices = token_indices[valid_rows]
        landmark_features, valid_landmarks = normalize_rows(submap.features)
        if not np.all(valid_landmarks):
            submap = submap.subset(valid_landmarks)
            local_context = local_context.subset(valid_landmarks)
            landmark_features = landmark_features[valid_landmarks]
        if len(submap) == 0:
            continue
        top_indices, top_scores = _query_landmark_topk(
            query_features,
            landmark_features,
            top_k=int(args.top_m_anchor),
            block_size=int(args.match_block_size),
            device=args.similarity_device,
        )
        keep = top_scores >= float(args.min_anchor_similarity)
        top_indices = np.where(keep, top_indices, 0)
        top_scores = np.where(keep, top_scores, -np.inf).astype(np.float32)
        positives = build_patch_positive_sets(
            submap,
            pose.pose_w2c,
            camera,
            int(query_feature.shape[2]),
            int(query_feature.shape[1]),
            patch_scale=float(args.patch_scale),
        )
        labels = _positive_label_matrix(top_indices, token_indices, submap.track_ids, positives)
        labels &= keep
        query_rows = np.repeat(np.arange(top_indices.shape[0], dtype=np.int64), top_indices.shape[1])
        context_flat = dense_context_scores(
            query_features[query_rows],
            local_context,
            top_indices.reshape(-1),
            mode=args.context_mode,
            topk_average_k=int(args.topk_average_k),
            apply_reliability=bool(args.apply_reliability),
            require_support=True,
        ).reshape(top_indices.shape)
        context_flat = np.where(keep, context_flat, -np.inf).astype(np.float32)
        score_mats = {
            "anchor": top_scores,
            f"context_{args.context_mode}": context_flat,
        }
        for weight in context_weights:
            name = f"anchor_context_{args.context_mode}_w{str(weight).replace('.', 'p')}"
            score_mats[name] = top_scores + float(weight) * context_flat
        query_metrics = {name: score_metrics_from_matrices(labels, score_mats[name], valid_mask=keep) for name in score_names}
        rows.append(
            {
                "query_id": record.image_id,
                "submap_landmark_count": int(len(submap)),
                "candidate_count": int(np.sum(keep)),
                "positive_count": int(np.sum(labels)),
                "visibility_gate": visibility_gate,
                "metrics": query_metrics,
            }
        )
        global_labels.append(labels)
        global_valid.append(keep)
        for name in score_names:
            global_scores[name].append(score_mats[name])

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    labels_all = np.concatenate(global_labels, axis=0) if global_labels else np.zeros((0, int(args.top_m_anchor)), dtype=bool)
    valid_all = np.concatenate(global_valid, axis=0) if global_valid else np.zeros((0, int(args.top_m_anchor)), dtype=bool)
    global_metrics = {
        name: score_metrics_from_matrices(labels_all, np.concatenate(global_scores[name], axis=0), valid_mask=valid_all)
        for name in score_names
    }
    anchor_auc = global_metrics["anchor"]["auroc"]
    best_name = None
    best_auc = -np.inf
    for name, metric in global_metrics.items():
        if not name.startswith("anchor_context_"):
            continue
        value = metric["auroc"]
        if value is not None and float(value) > best_auc:
            best_name = name
            best_auc = float(value)
    diagnostic_pass = bool(anchor_auc is not None and best_name is not None and best_auc >= float(anchor_auc) + float(args.min_auc_delta))
    summary = {
        "stage": "fast_dense_context_oracle_diagnostic",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": int(len(rows)),
        "camera_source": camera_source,
        "candidate_count": int(np.sum(valid_all)),
        "positive_count": int(np.sum(labels_all)),
        "score_metrics": global_metrics,
        "best_combined_score": best_name,
        "diagnostic_pass": diagnostic_pass,
        "diagnostic_pass_rule": f"best combined AUROC >= anchor AUROC + {float(args.min_auc_delta)}",
        "config": {
            "top_m_anchor": int(args.top_m_anchor),
            "context_mode": args.context_mode,
            "context_weights": context_weights,
            "apply_reliability": bool(args.apply_reliability),
            "submap_top_n": int(args.submap_top_n),
            "max_submap_landmarks": int(args.max_submap_landmarks),
        },
        "inputs": {
            "query_manifest": args.query_manifest,
            "landmark_bank": args.landmark_bank,
            "dense_context_bank": args.dense_context_bank,
            "track_observations": args.track_observations,
            "visibility_index": args.visibility_index,
            "query_pose_file": args.query_pose_file,
            "candidate_bank": args.candidate_bank,
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.summary_md:
        _markdown_summary(summary, Path(args.summary_md))


if __name__ == "__main__":
    main()
