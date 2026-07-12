"""Evaluate dense-context reranking for patch-to-3D PnP localization."""

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
    _mean,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.dense_context_localization import build_selected_matches, select_top1_candidate_rows
from feature_extract.vfm.dense_context_oracle_metrics import score_metrics_from_matrices
from feature_extract.vfm.dense_patch_context import DensePatchContextBank, align_dense_context_to_landmarks, dense_context_scores
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    _flatten_query_features,
    _query_landmark_topk,
    build_patch_positive_sets,
    evaluate_patch_matches,
    patch_positive_set_stats,
    patch_uncertainty_pnp_threshold,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    estimate_pose_pnp_ransac,
    match_spatial_distribution_stats,
    normalize_rows,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    token_grid_xy,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _is_success(translation: float | None, rotation: float | None, t: float, r: float) -> bool:
    return bool(translation is not None and rotation is not None and float(translation) <= t and float(rotation) <= r)


def _positive_label_matrix(top_indices: np.ndarray, token_indices: np.ndarray, track_ids: np.ndarray, positives) -> np.ndarray:
    labels = np.zeros(top_indices.shape, dtype=bool)
    for row, token in enumerate(token_indices.tolist()):
        positive = positives.by_token.get(int(token))
        if positive is None:
            continue
        labels[row] = np.asarray([int(track_ids[int(idx)]) in positive.track_ids for idx in top_indices[row]], dtype=bool)
    return labels


def _median_present(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not present else float(np.median(present))


def _summarize_score_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    labeled = [row for row in rows if row["translation_error_m"] is not None]
    failures = [row for row in rows if not bool(row["success_25cm_10deg"])]
    return {
        "query_count": int(len(rows)),
        "pnp_solve_rate": _mean([1.0 if row["pnp_solve"] else 0.0 for row in rows]),
        "success_10cm_5deg": _mean([1.0 if row["success_10cm_5deg"] else 0.0 for row in labeled]),
        "success_25cm_10deg": _mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in labeled]),
        "success_50cm_10deg": _mean([1.0 if row["success_50cm_10deg"] else 0.0 for row in labeled]),
        "success_1m_10deg": _mean([1.0 if row["success_1m_10deg"] else 0.0 for row in labeled]),
        "median_translation_error_m": _median_present([row["translation_error_m"] for row in labeled]),
        "median_rotation_error_deg": _median_present([row["rotation_error_deg"] for row in labeled]),
        "mean_match_count": _mean([row["match_count"] for row in rows]),
        "mean_pnp_inlier_count": _mean([row["pnp_inlier_count"] for row in rows]),
        "mean_selected_patch_at_1": _mean([row["selected_patch_at_1"] for row in rows]),
        "mean_selected_gt_precision_stride": _mean([row["selected_gt_precision_stride"] for row in rows]),
        "mean_pnp_inlier_patch_at_1": _mean(
            [row["pnp_inlier_patch_at_1"] for row in rows if row["pnp_inlier_patch_at_1"] is not None]
        ),
        "mean_topm_patch_recall": _mean([row["topm_patch_recall"] for row in rows]),
        "mean_nonempty_patch_fraction": _mean([row["nonempty_patch_fraction"] for row in rows]),
        "failure_low_topm_recall": int(sum(float(row["topm_patch_recall"]) < 0.05 for row in failures)),
        "failure_low_selected_patch": int(sum(float(row["selected_patch_at_1"]) < 0.05 for row in failures)),
        "failure_low_inlier_count": int(sum(int(row["pnp_inlier_count"]) < 8 for row in failures)),
        "failure_wrong_self_consistent_pose": int(
            sum(int(row["pnp_inlier_count"]) >= 12 and float(row["selected_patch_at_1"]) < 0.10 for row in failures)
        ),
    }


def _write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Dense Context Localization Diagnostic",
        "",
        "| Score | S@10cm/5deg | S@25cm/10deg | S@50cm/10deg | Median t | Median r | Inlier Patch@1 | Inliers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in dict(summary.get("localization") or {}).items():
        lines.append(
            f"| {name} | {float(row['success_10cm_5deg']):.4f} | {float(row['success_25cm_10deg']):.4f} | "
            f"{float(row['success_50cm_10deg']):.4f} | {row['median_translation_error_m']} | "
            f"{row['median_rotation_error_deg']} | {float(row['mean_pnp_inlier_patch_at_1']):.4f} | "
            f"{float(row['mean_pnp_inlier_count']):.2f} |"
        )
    lines.extend(["", "## Bottleneck", ""])
    for item in summary.get("bottleneck_findings", []):
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate dense-context reranked patch-to-3D localization")
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
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=0.75)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_method", default="EPNP", choices=("AP3P", "EPNP", "ITERATIVE", "P3P", "SQPNP"))
    parser.add_argument("--pnp_refine_method", default="LM", choices=("none", "LM", "VVS"))
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
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
    score_names = ["anchor"] + [
        f"anchor_context_{args.context_mode}_w{str(weight).replace('.', 'p')}" for weight in context_weights
    ]

    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    all_rows = []
    score_rows: dict[str, list[dict[str, object]]] = {name: [] for name in score_names}
    oracle_labels = []
    oracle_valid = []
    oracle_scores: dict[str, list[np.ndarray]] = {name: [] for name in score_names}
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
        channels, token_height, token_width = query_feature.shape
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
        valid = top_scores >= float(args.min_anchor_similarity)
        top_indices = np.where(valid, top_indices, 0)
        top_scores = np.where(valid, top_scores, -np.inf).astype(np.float32)
        positives = build_patch_positive_sets(submap, pose.pose_w2c, camera, token_width, token_height, patch_scale=float(args.patch_scale))
        labels = _positive_label_matrix(top_indices, token_indices, submap.track_ids, positives) & valid
        query_rows = np.repeat(np.arange(top_indices.shape[0], dtype=np.int64), top_indices.shape[1])
        context_scores = dense_context_scores(
            query_features[query_rows],
            local_context,
            top_indices.reshape(-1),
            mode=args.context_mode,
            topk_average_k=int(args.topk_average_k),
            apply_reliability=bool(args.apply_reliability),
            require_support=True,
        ).reshape(top_indices.shape)
        context_scores = np.where(valid, context_scores, -np.inf).astype(np.float32)
        score_mats = {"anchor": top_scores}
        for weight in context_weights:
            name = f"anchor_context_{args.context_mode}_w{str(weight).replace('.', 'p')}"
            score_mats[name] = top_scores + float(weight) * context_scores
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        stride = max(stride_x, stride_y)
        centers_full = token_grid_xy(token_width, token_height, int(camera.width), int(camera.height), step=1)
        centers = centers_full[token_indices]
        positive_stats = patch_positive_set_stats(positives)
        topm_recall = float(np.mean(np.any(labels, axis=1))) if labels.size else 0.0
        row = {
            "query_id": record.image_id,
            "submap_landmark_count": int(len(submap)),
            "candidate_count": int(np.sum(valid)),
            "positive_count": int(np.sum(labels)),
            "topm_patch_recall": topm_recall,
            "nonempty_patch_fraction": positive_stats["nonempty_patch_fraction"],
            "visibility_gate": visibility_gate,
            "scores": {},
        }
        oracle_labels.append(labels)
        oracle_valid.append(valid)
        for name in score_names:
            oracle_scores[name].append(score_mats[name])
            token_rows, landmark_rows, selected_scores = select_top1_candidate_rows(top_indices, score_mats[name], valid)
            matches = build_selected_matches(
                token_rows,
                token_indices,
                landmark_rows,
                selected_scores,
                submap,
                centers,
                source=f"dense_context_localization_{name}",
                max_matches=int(args.max_matches),
            )
            pnp_threshold = patch_uncertainty_pnp_threshold(stride, float(args.pnp_threshold_stride_multiplier))
            pnp = estimate_pose_pnp_ransac(
                matches,
                camera,
                reprojection_error_px=pnp_threshold,
                iterations=int(args.pnp_iterations),
                confidence=float(args.pnp_confidence),
                pnp_method=args.pnp_method,
                refine_method=args.pnp_refine_method,
            )
            pose_error = pnp_pose_error(pnp.pose_w2c, pose.pose_w2c) if pnp.pose_w2c is not None else None
            translation = None if pose_error is None or not np.isfinite(pose_error.translation_m) else float(pose_error.translation_m)
            rotation = None if pose_error is None or not np.isfinite(pose_error.rotation_deg) else float(pose_error.rotation_deg)
            patch_stats = evaluate_patch_matches(matches, positives, pose.pose_w2c, camera, stride_px=stride, pnp_inlier_mask=pnp.inlier_mask)
            spatial = match_spatial_distribution_stats(
                matches,
                int(camera.width),
                int(camera.height),
                mask=pnp.inlier_mask,
                pose_w2c=pnp.pose_w2c,
            )
            residual = pnp_reprojection_residual_stats(matches, pnp.pose_w2c, camera, inlier_mask=pnp.inlier_mask)
            score_row = {
                "query_id": record.image_id,
                "score_name": name,
                "match_count": int(len(matches)),
                "pnp_solve": bool(pnp.success),
                "pnp_inlier_count": int(pnp.inlier_count),
                "pnp_inlier_ratio": float(pnp.inlier_ratio),
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
                "success_10cm_5deg": _is_success(translation, rotation, 0.10, 5.0),
                "success_25cm_10deg": _is_success(translation, rotation, 0.25, 10.0),
                "success_50cm_10deg": _is_success(translation, rotation, 0.50, 10.0),
                "success_1m_10deg": _is_success(translation, rotation, 1.0, 10.0),
                "selected_patch_at_1": patch_stats["patch_at_1"],
                "selected_gt_precision_stride": patch_stats["gt_precision_stride"],
                "selected_gt_reproj_median_px": patch_stats["gt_reproj_median_px"],
                "pnp_inlier_patch_at_1": patch_stats["pnp_inlier_patch_at_1"],
                "pnp_inlier_gt_precision_stride": patch_stats["pnp_inlier_gt_precision_stride"],
                "topm_patch_recall": topm_recall,
                "nonempty_patch_fraction": positive_stats["nonempty_patch_fraction"],
                "inlier_grid_4x4_occupancy_frac": spatial["grid_4x4_occupancy_frac"],
                "inlier_convex_hull_area_frac": spatial["convex_hull_area_frac"],
                "inlier_depth_range_m": spatial["depth_range_m"],
                "pnp_reproj_inlier_median_px": residual["pnp_reproj_inlier_median_px"],
            }
            score_rows[name].append(score_row)
            row["scores"][name] = score_row
        all_rows.append(row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in all_rows) + ("\n" if all_rows else ""))
    labels_all = np.concatenate(oracle_labels, axis=0) if oracle_labels else np.zeros((0, int(args.top_m_anchor)), dtype=bool)
    valid_all = np.concatenate(oracle_valid, axis=0) if oracle_valid else np.zeros((0, int(args.top_m_anchor)), dtype=bool)
    oracle_metrics = {
        name: score_metrics_from_matrices(labels_all, np.concatenate(oracle_scores[name], axis=0), valid_mask=valid_all)
        for name in score_names
    }
    localization = {name: _summarize_score_rows(rows) for name, rows in score_rows.items()}
    anchor = localization.get("anchor", {})
    best_name = max(
        localization,
        key=lambda name: (
            float(localization[name]["success_25cm_10deg"]),
            -float(localization[name]["median_translation_error_m"] or 1e9),
        ),
    ) if localization else None
    best = localization.get(best_name, {}) if best_name is not None else {}
    findings = []
    if best_name and best_name != "anchor":
        findings.append(
            f"Best localization score is {best_name}: S@25 {best.get('success_25cm_10deg')} vs anchor {anchor.get('success_25cm_10deg')}."
        )
    else:
        findings.append("Dense context did not improve the main S@25 localization score over anchor-only.")
    if anchor:
        nonempty = max(float(anchor.get("mean_nonempty_patch_fraction") or 0.0), 1e-9)
        topm_given_nonempty = float(anchor.get("mean_topm_patch_recall") or 0.0) / nonempty
        findings.append(
            f"Only {anchor.get('mean_nonempty_patch_fraction')} of tokens have any GT-visible positive landmark, and top-M recall over all tokens is {anchor.get('mean_topm_patch_recall')} ({topm_given_nonempty:.3f} normalized by nonempty patches). Candidate coverage is a primary bottleneck."
        )
        findings.append(
            f"Selected Patch@1 is {anchor.get('mean_selected_patch_at_1')}; dense/context reranking can improve this slightly, but many selected matches remain patch-level wrong."
        )
        findings.append(
            f"PnP inlier Patch@1 is {anchor.get('mean_pnp_inlier_patch_at_1')} with mean inliers {anchor.get('mean_pnp_inlier_count')}; failures with many inliers indicate repeated-structure or patch-uncertainty self-consistency, not PnP solve failure."
        )
    summary = {
        "stage": "dense_context_localization_fast",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": int(len(all_rows)),
        "camera_source": camera_source,
        "score_names": score_names,
        "oracle_metrics": oracle_metrics,
        "localization": localization,
        "best_score_name": best_name,
        "bottleneck_findings": findings,
        "config": {
            "top_m_anchor": int(args.top_m_anchor),
            "context_mode": args.context_mode,
            "context_weights": context_weights,
            "submap_top_n": int(args.submap_top_n),
            "max_submap_landmarks": int(args.max_submap_landmarks),
            "max_matches": int(args.max_matches),
            "pnp_threshold_stride_multiplier": float(args.pnp_threshold_stride_multiplier),
            "pnp_method": args.pnp_method,
            "pnp_refine_method": args.pnp_refine_method,
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
        _write_markdown(summary, Path(args.summary_md))


if __name__ == "__main__":
    main()
