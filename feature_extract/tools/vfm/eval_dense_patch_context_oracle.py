"""Dense Context Oracle Diagnostic and optional sparse-anchor rerank localization."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
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
from feature_extract.vfm.dense_patch_context import (
    DensePatchContextBank,
    align_dense_context_to_landmarks,
    dense_context_topm_candidates,
)
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    build_patch_positive_sets,
    evaluate_patch_matches,
    patch_uncertainty_pnp_threshold,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    pos = int(np.sum(y))
    neg = int(y.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float64)
    return float((np.sum(ranks[y]) - pos * (pos + 1) / 2.0) / max(pos * neg, 1))


def _ap(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    pos = int(np.sum(y))
    if pos == 0:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted.astype(np.float64))
    precision = tp / np.arange(1, y_sorted.size + 1, dtype=np.float64)
    return float(np.sum(precision[y_sorted]) / max(pos, 1))


def _score_metrics(labels: list[bool], scores: list[float], token_ids: list[str]) -> dict[str, object]:
    by_token: dict[str, list[int]] = defaultdict(list)
    for idx, token in enumerate(token_ids):
        by_token[str(token)].append(int(idx))
    top1_correct = []
    positive_best_ranks = []
    for indices in by_token.values():
        order = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        top1_correct.append(bool(labels[order[0]]))
        ranks = [rank + 1 for rank, idx in enumerate(order) if bool(labels[idx])]
        if ranks:
            positive_best_ranks.append(float(min(ranks)))
    return {
        "candidate_count": int(len(labels)),
        "positive_count": int(np.sum(np.asarray(labels, dtype=bool))),
        "positive_prior": float(np.mean(np.asarray(labels, dtype=bool))) if labels else 0.0,
        "auroc": _auc(labels, scores),
        "auprc": _ap(labels, scores),
        "top1_accuracy": float(np.mean(top1_correct)) if top1_correct else 0.0,
        "mean_best_positive_rank": None if not positive_best_ranks else float(np.mean(positive_best_ranks)),
        "median_best_positive_rank": None if not positive_best_ranks else float(np.median(positive_best_ranks)),
    }


def _is_success(translation: float | None, rotation: float | None, t: float, r: float) -> bool:
    return bool(translation is not None and rotation is not None and float(translation) <= t and float(rotation) <= r)


def _select_top1_matches(
    candidate_rows: list[dict[str, object]],
    candidates: list[tuple[QueryTo3DMatch, float, float, int]],
    score_key: str,
    max_matches: int,
) -> list[QueryTo3DMatch]:
    by_token: dict[int, tuple[int, float]] = {}
    for idx, row in enumerate(candidate_rows):
        token = int(row["token_index"])
        score = float(row[score_key])
        current = by_token.get(token)
        if current is None or score > current[1]:
            by_token[token] = (int(idx), score)
    selected = []
    for idx, score in by_token.values():
        match = candidates[idx][0]
        selected.append(
            QueryTo3DMatch(
                token_index=match.token_index,
                xy=match.xy,
                track_id=match.track_id,
                xyz=match.xyz,
                similarity=float(score),
                ratio=match.ratio,
                landmark_variance=match.landmark_variance,
                source=f"dense_context_rerank_{score_key}",
                observation_count=match.observation_count,
                visibility_count=match.visibility_count,
                landmark_reprojection_error=match.landmark_reprojection_error,
                similarity_margin=match.similarity_margin,
                local_consistency_support=match.local_consistency_support,
                local_consistency_score=match.local_consistency_score,
                quality_weighted_similarity=float(score),
                pnp_soft_score=float(score),
            )
        )
    selected.sort(key=lambda item: float(item.similarity), reverse=True)
    return selected[: int(max_matches)]


def main(argv: Optional[Sequence[str]] = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate dense patch context as sparse-anchor match evidence")
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
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=0.75)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_method", default="EPNP", choices=("AP3P", "EPNP", "ITERATIVE", "P3P", "SQPNP"))
    parser.add_argument("--pnp_refine_method", default="LM", choices=("none", "LM", "VVS"))
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--run_pnp", action="store_true")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--min_auc_delta", type=float, default=0.005)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_candidates_jsonl", default="")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--summary_md", default="")
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    xyz_by_track, reprojection_by_track = _load_track_stats(Path(args.track_observations))
    sparse = LandmarkMapIndex.from_track_bank(load_selected_track_bank_npz(Path(args.landmark_bank)), xyz_by_track, reprojection_by_track)
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
    all_labels: list[bool] = []
    all_token_ids: list[str] = []
    all_scores: dict[str, list[float]] = {name: [] for name in score_names}
    rows: list[dict[str, object]] = []
    candidate_output_rows: list[dict[str, object]] = []
    pnp_rows_by_score: dict[str, list[dict[str, object]]] = {name: [] for name in score_names} if args.run_pnp else {}

    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    for record in records:
        pose = poses.get(record.image_id)
        if pose is None:
            continue
        references = reference_submaps.get(record.image_id, [])
        submap, visibility_gate = filter_landmarks_by_visibility(sparse, visibility_index, references)
        submap = _limit_submap(submap, int(args.max_submap_landmarks))
        submap, local_context = align_dense_context_to_landmarks(submap, context_bank)
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_height, token_width = query_feature.shape
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        stride = max(stride_x, stride_y)
        positives = build_patch_positive_sets(submap, pose.pose_w2c, camera, token_width, token_height, patch_scale=float(args.patch_scale))
        candidates = dense_context_topm_candidates(
            query_feature,
            submap,
            local_context,
            top_m=int(args.top_m_anchor),
            query_token_step=int(args.query_token_step),
            context_mode=args.context_mode,
            topk_average_k=int(args.topk_average_k),
            image_width=int(camera.width),
            image_height=int(camera.height),
            block_size=int(args.match_block_size),
            similarity_device=args.similarity_device,
            min_anchor_similarity=float(args.min_anchor_similarity),
            apply_reliability=bool(args.apply_reliability),
        )
        query_labels = []
        query_token_ids = []
        query_scores: dict[str, list[float]] = {name: [] for name in score_names}
        query_candidate_rows: list[dict[str, object]] = []
        for idx, (match, anchor_score, context_score, _anchor_row) in enumerate(candidates):
            positive = positives.by_token.get(int(match.token_index))
            label = bool(positive is not None and int(match.track_id) in positive.track_ids)
            score_values = {
                "anchor": float(anchor_score),
                f"context_{args.context_mode}": float(context_score),
            }
            for weight in context_weights:
                score_values[f"anchor_context_{args.context_mode}_w{str(weight).replace('.', 'p')}"] = float(
                    anchor_score + float(weight) * float(context_score)
                )
            token_key = f"{record.image_id}:{int(match.token_index)}"
            query_labels.append(label)
            query_token_ids.append(token_key)
            all_labels.append(label)
            all_token_ids.append(token_key)
            row = {
                "query_id": record.image_id,
                "candidate_index": int(idx),
                "token_index": int(match.token_index),
                "track_id": int(match.track_id),
                "patch_correct": label,
                "anchor_score": float(anchor_score),
                "context_score": float(context_score),
                "support_count": int(match.local_consistency_support or 0),
            }
            for name, value in score_values.items():
                query_scores[name].append(float(value))
                all_scores[name].append(float(value))
                row[name] = float(value)
            query_candidate_rows.append(row)
            if args.output_candidates_jsonl:
                candidate_output_rows.append(row)
        query_metrics = {
            name: _score_metrics(query_labels, query_scores[name], query_token_ids) for name in score_names
        }
        row = {
            "query_id": record.image_id,
            "submap_landmark_count": int(len(submap)),
            "candidate_count": int(len(query_candidate_rows)),
            "positive_count": int(sum(1 for item in query_candidate_rows if item["patch_correct"])),
            "visibility_gate": visibility_gate,
            "metrics": query_metrics,
        }
        if args.run_pnp:
            pnp_threshold = patch_uncertainty_pnp_threshold(stride, float(args.pnp_threshold_stride_multiplier))
            for name in score_names:
                selected = _select_top1_matches(query_candidate_rows, candidates, name, int(args.max_matches))
                pnp = estimate_pose_pnp_ransac(
                    selected,
                    camera,
                    reprojection_error_px=pnp_threshold,
                    iterations=int(args.pnp_iterations),
                    confidence=float(args.pnp_confidence),
                    pnp_method=args.pnp_method,
                    refine_method=args.pnp_refine_method,
                )
                pose_error = pnp_pose_error(pnp.pose_w2c, pose.pose_w2c) if pnp.pose_w2c is not None else None
                translation = None if pose_error is None else float(pose_error.translation_m)
                rotation = None if pose_error is None else float(pose_error.rotation_deg)
                patch_stats = evaluate_patch_matches(selected, positives, pose.pose_w2c, camera, stride_px=stride, pnp_inlier_mask=pnp.inlier_mask)
                pnp_row = {
                    "query_id": record.image_id,
                    "score_name": name,
                    "match_count": int(len(selected)),
                    "pnp_solve": bool(pnp.success),
                    "pnp_inlier_count": int(pnp.inlier_count),
                    "translation_error_m": translation,
                    "rotation_error_deg": rotation,
                    "success_25cm_10deg": _is_success(translation, rotation, 0.25, 10.0),
                    "success_50cm_10deg": _is_success(translation, rotation, 0.50, 10.0),
                    "pnp_inlier_patch_at_1": patch_stats["pnp_inlier_patch_at_1"],
                }
                pnp_rows_by_score[name].append(pnp_row)
            row["pnp"] = {name: pnp_rows_by_score[name][-1] for name in score_names}
        rows.append(row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    if args.output_candidates_jsonl:
        path = Path(args.output_candidates_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in candidate_output_rows) + ("\n" if candidate_output_rows else ""))

    global_metrics = {name: _score_metrics(all_labels, all_scores[name], all_token_ids) for name in score_names}
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
    localization = {}
    if args.run_pnp:
        for name, score_rows in pnp_rows_by_score.items():
            labeled = [row for row in score_rows if row["translation_error_m"] is not None]
            localization[name] = {
                "query_count": int(len(score_rows)),
                "pnp_solve_rate": float(np.mean([1.0 if row["pnp_solve"] else 0.0 for row in score_rows])) if score_rows else 0.0,
                "success_25cm_10deg": float(np.mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in labeled])) if labeled else 0.0,
                "success_50cm_10deg": float(np.mean([1.0 if row["success_50cm_10deg"] else 0.0 for row in labeled])) if labeled else 0.0,
                "median_translation_error_m": None if not labeled else float(np.median([row["translation_error_m"] for row in labeled])),
                "median_rotation_error_deg": None if not labeled else float(np.median([row["rotation_error_deg"] for row in labeled])),
                "mean_pnp_inlier_count": float(np.mean([row["pnp_inlier_count"] for row in score_rows])) if score_rows else 0.0,
            }
    summary = {
        "stage": "dense_patch_context_oracle_diagnostic",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": int(len(rows)),
        "camera_source": camera_source,
        "candidate_count": int(len(all_labels)),
        "positive_count": int(np.sum(np.asarray(all_labels, dtype=bool))),
        "score_metrics": global_metrics,
        "best_combined_score": best_name,
        "diagnostic_pass": diagnostic_pass,
        "diagnostic_pass_rule": f"best combined AUROC >= anchor AUROC + {float(args.min_auc_delta)}",
        "localization": localization,
        "config": {
            "top_m_anchor": int(args.top_m_anchor),
            "context_mode": args.context_mode,
            "context_weights": context_weights,
            "apply_reliability": bool(args.apply_reliability),
            "submap_top_n": int(args.submap_top_n),
            "max_submap_landmarks": int(args.max_submap_landmarks),
            "run_pnp": bool(args.run_pnp),
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
        lines = [
            "# Dense Patch Context Oracle Diagnostic",
            "",
            f"- diagnostic pass: `{diagnostic_pass}`",
            f"- best combined score: `{best_name}`",
            "",
            "| Score | AUROC | AUPRC | Top1 acc | Mean best positive rank |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, metric in global_metrics.items():
            auroc = "n/a" if metric["auroc"] is None else f"{float(metric['auroc']):.4f}"
            auprc = "n/a" if metric["auprc"] is None else f"{float(metric['auprc']):.4f}"
            rank = "n/a" if metric["mean_best_positive_rank"] is None else f"{float(metric['mean_best_positive_rank']):.3f}"
            lines.append(f"| {name} | {auroc} | {auprc} | {float(metric['top1_accuracy']):.4f} | {rank} |")
        if localization:
            lines.extend(["", "## Localization", "", "| Score | S@25 | S@50 | Median t | Median r |", "|---|---:|---:|---:|---:|"])
            for name, metric in localization.items():
                lines.append(
                    f"| {name} | {float(metric['success_25cm_10deg']):.4f} | {float(metric['success_50cm_10deg']):.4f} | "
                    f"{metric['median_translation_error_m']} | {metric['median_rotation_error_deg']} |"
                )
        Path(args.summary_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
