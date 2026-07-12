#!/usr/bin/env python3
"""Evaluate confidence-based correspondence selection and hypothesis rescoring."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.correspondence_confidence import (
    CalibratedLogisticConfidence,
    select_confident_matches_coverage_preserving,
    vectorize_match_rows,
)
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    match_reprojection_errors,
    match_spatial_distribution_stats,
    pnp_pose_error,
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _group_by_query(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id", ""))].append(row)
    return dict(grouped)


def _match_from_row(row: dict[str, object]) -> QueryTo3DMatch:
    return QueryTo3DMatch(
        token_index=int(row.get("token_index", 0)),
        xy=np.asarray(row.get("xy", [0.0, 0.0]), dtype=np.float64).reshape(2),
        track_id=int(row.get("track_id", 0)),
        xyz=np.asarray(row.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3),
        similarity=float(row.get("similarity", 0.0)),
        ratio=0.0,
        landmark_variance=float(row.get("landmark_variance", 0.0)),
        source=str(row.get("source", "sparse_patch")),
        observation_count=None if row.get("observation_count") is None else int(row.get("observation_count")),
        visibility_count=None if row.get("visibility_count") is None else int(row.get("visibility_count")),
        landmark_reprojection_error=None
        if row.get("landmark_reprojection_error") is None
        else float(row.get("landmark_reprojection_error")),
        landmark_quality=None if row.get("landmark_quality") is None else float(row.get("landmark_quality")),
        landmark_ambiguity=None if row.get("landmark_ambiguity") is None else float(row.get("landmark_ambiguity")),
        quality_weighted_similarity=None
        if row.get("quality_weighted_similarity") is None
        else float(row.get("quality_weighted_similarity")),
        pairwise_inlier_logit=None
        if row.get("pairwise_inlier_logit") is None
        else float(row.get("pairwise_inlier_logit")),
        pairwise_inlier_logprob=None
        if row.get("pairwise_inlier_logprob") is None
        else float(row.get("pairwise_inlier_logprob")),
        similarity_margin=None if row.get("similarity_margin") is None else float(row.get("similarity_margin")),
        distance_to_boundary_px=None
        if row.get("distance_to_boundary_px") is None
        else float(row.get("distance_to_boundary_px")),
        map_reliability=None if row.get("map_reliability") is None else float(row.get("map_reliability")),
        local_consistency_support=None
        if row.get("local_consistency_support") is None
        else int(row.get("local_consistency_support")),
        local_consistency_score=None
        if row.get("local_consistency_score") is None
        else float(row.get("local_consistency_score")),
        pnp_soft_score=None if row.get("confidence") is None else float(row.get("confidence")),
    )


def _selection_label(spec: str) -> tuple[str, str, int, int]:
    parts = [item.strip() for item in spec.split(":")]
    if len(parts) < 2:
        raise ValueError("selection spec must be label:mode[:max_matches[:per_cell]]")
    label = parts[0]
    mode = parts[1]
    max_matches = int(parts[2]) if len(parts) >= 3 and parts[2] else 1000
    per_cell = int(parts[3]) if len(parts) >= 4 and parts[3] else 64
    if mode not in {"none", "global", "grid2d"}:
        raise ValueError("selection mode must be one of: none, global, grid2d")
    return label, mode, max_matches, per_cell


def _select_rows(
    rows: list[dict[str, object]],
    mode: str,
    max_matches: int,
    per_cell: int,
    image_width: int,
    image_height: int,
) -> list[dict[str, object]]:
    if mode == "none":
        return rows[: int(max_matches)]
    if mode == "global":
        return sorted(rows, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)[: int(max_matches)]
    return list(
        select_confident_matches_coverage_preserving(
            rows,
            image_width=image_width,
            image_height=image_height,
            max_matches=max_matches,
            grid_rows=4,
            grid_cols=4,
            per_cell=per_cell,
            confidence_key="confidence",
        )
    )


def _pose_success(t: float | None, r: float | None, max_t: float, max_r: float) -> bool:
    return bool(t is not None and r is not None and float(t) <= float(max_t) and float(r) <= float(max_r))


def _hypothesis_score(matches: list[QueryTo3DMatch], confidences: np.ndarray, pose_w2c: np.ndarray | None, camera, threshold_px: float) -> float:
    if pose_w2c is None or not matches:
        return float("-inf")
    residuals = match_reprojection_errors(matches, pose_w2c, camera)
    inliers = residuals <= float(threshold_px)
    if not np.any(inliers):
        return float("-inf")
    probs = np.clip(confidences[inliers], 1e-4, 1.0 - 1e-4)
    spatial = match_spatial_distribution_stats(
        matches,
        int(camera.width),
        int(camera.height),
        inliers,
        pose_w2c=pose_w2c,
    )
    coverage = float(spatial.get("grid_4x4_occupancy_frac") or 0.0)
    planarity = float(spatial.get("xyz_planarity_ratio") or 0.0)
    depth_range = float(spatial.get("depth_range_m") or 0.0)
    return float(
        np.mean(np.log(probs))
        - 0.02 * np.mean(residuals[inliers])
        + 0.15 * coverage
        + 0.05 * min(depth_range / 5.0, 1.0)
        + 0.05 * min(planarity / 0.02, 1.0)
        + 0.02 * math.log1p(float(np.sum(inliers)))
    )


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    t_values = [float(row["translation_error_m"]) for row in rows if row["translation_error_m"] is not None]
    r_values = [float(row["rotation_error_deg"]) for row in rows if row["rotation_error_deg"] is not None]
    return {
        "query_count": int(len(rows)),
        "success_10cm_5deg": float(np.mean([_pose_success(row["translation_error_m"], row["rotation_error_deg"], 0.10, 5.0) for row in rows])),
        "success_25cm_10deg": float(np.mean([_pose_success(row["translation_error_m"], row["rotation_error_deg"], 0.25, 10.0) for row in rows])),
        "success_50cm_10deg": float(np.mean([_pose_success(row["translation_error_m"], row["rotation_error_deg"], 0.50, 10.0) for row in rows])),
        "median_translation_error_m": None if not t_values else float(np.median(t_values)),
        "median_rotation_error_deg": None if not r_values else float(np.median(r_values)),
        "mean_pnp_inlier_patch_at_1": float(np.mean([float(row["pnp_inlier_patch_at_1"]) for row in rows if row["pnp_inlier_patch_at_1"] is not None])),
        "mean_pnp_inlier_strong_positive_rate": float(
            np.mean(
                [
                    float(row["pnp_inlier_strong_positive_rate"])
                    for row in rows
                    if row.get("pnp_inlier_strong_positive_rate") is not None
                ]
            )
        ),
        "mean_pnp_inlier_count": float(np.mean([float(row["pnp_inlier_count"]) for row in rows])),
        "mean_pnp_match_count": float(np.mean([float(row["pnp_match_count"]) for row in rows])),
        "pnp_solve_rate": float(np.mean([1.0 if row["pnp_solve"] else 0.0 for row in rows])),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_jsonl", required=True)
    parser.add_argument("--match_jsonl", required=True)
    parser.add_argument("--confidence_model", required=True)
    parser.add_argument("--feature_set", default="descriptor_map")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--selection", action="append", default=["canonical:none:1000", "global800:global:800", "grid800:grid2d:800:64"])
    parser.add_argument("--enable_rescore", action="store_true")
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=0.75)
    parser.add_argument("--pnp_method", default="EPNP")
    parser.add_argument("--pnp_refine_method", default="LM")
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    query_rows = _group_by_query(_load_jsonl(Path(args.rows_jsonl)))
    match_rows = _group_by_query(_load_jsonl(Path(args.match_jsonl)))
    model = CalibratedLogisticConfidence.load_json(args.confidence_model)
    camera, camera_source = _load_camera_with_source(args.camera_model_dir, _parse_default_camera(args.default_camera))
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    selections = [_selection_label(spec) for spec in args.selection]

    output_rows: list[dict[str, object]] = []
    per_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for query_id, rows in sorted(match_rows.items()):
        if query_id not in gt_by_query or not rows:
            continue
        matrix, _labels, _keep, _names = vectorize_match_rows(rows, feature_set=args.feature_set)
        scores = model.predict_proba(matrix)
        scored_rows = []
        for row, score in zip(rows, scores):
            item = dict(row)
            item["confidence"] = float(score)
            scored_rows.append(item)
        audit = dict((query_rows.get(query_id) or [{}])[0].get("coordinate_audit") or {})
        stride = max(float(audit.get("token_grid_scale_x_px", 16.0)), float(audit.get("token_grid_scale_y_px", 16.0)))
        threshold_px = float(stride) * float(args.pnp_threshold_stride_multiplier)
        candidate_results = []
        all_matches = [_match_from_row(row) for row in scored_rows]
        all_confidences = np.asarray([float(row.get("confidence", 0.0)) for row in scored_rows], dtype=np.float64)
        for label, mode, max_matches, per_cell in selections:
            selected_rows = _select_rows(scored_rows, mode, max_matches, per_cell, int(camera.width), int(camera.height))
            selected_matches = [_match_from_row(row) for row in selected_rows]
            pnp = estimate_pose_pnp_ransac(
                selected_matches,
                camera,
                reprojection_error_px=threshold_px,
                confidence=float(args.pnp_confidence),
                iterations=int(args.pnp_iterations),
                pnp_method=args.pnp_method,
                refine_method=args.pnp_refine_method,
            )
            error = pnp_pose_error(pnp.pose_w2c, gt_by_query[query_id].pose_w2c)
            inlier_mask = np.asarray(pnp.inlier_mask, dtype=bool).reshape(-1)
            selected_patch = np.asarray([bool(row.get("patch_positive_label", row.get("patch_correct", False))) for row in selected_rows], dtype=bool)
            selected_strong = np.asarray([bool(row.get("strong_positive_label", row.get("patch_correct", False))) for row in selected_rows], dtype=bool)
            inlier_patch = None if not np.any(inlier_mask) else float(np.mean(selected_patch[inlier_mask]))
            inlier_strong = None if not np.any(inlier_mask) else float(np.mean(selected_strong[inlier_mask]))
            result_row = {
                "query_id": query_id,
                "method": label,
                "selection_mode": mode,
                "pnp_match_count": int(pnp.match_count),
                "pnp_inlier_count": int(pnp.inlier_count),
                "pnp_inlier_patch_at_1": inlier_patch,
                "pnp_inlier_strong_positive_rate": inlier_strong,
                "pnp_solve": bool(pnp.success),
                "translation_error_m": None if not np.isfinite(error.translation_m) else float(error.translation_m),
                "rotation_error_deg": None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg),
                "hypothesis_score": _hypothesis_score(all_matches, all_confidences, pnp.pose_w2c, camera, threshold_px),
            }
            candidate_results.append((result_row, pnp.pose_w2c))
            output_rows.append(result_row)
            per_method[label].append(result_row)
        if args.enable_rescore and candidate_results:
            best_row, _pose = max(candidate_results, key=lambda item: float(item[0]["hypothesis_score"]))
            rescored = dict(best_row)
            rescored["method"] = "rescored"
            rescored["selection_mode"] = "confidence_hypothesis_rescore"
            output_rows.append(rescored)
            per_method["rescored"].append(rescored)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).write_text("\n".join(json.dumps(row, sort_keys=True) for row in output_rows) + ("\n" if output_rows else ""))
    summary = {
        "stage": "stage_c28_confidence_selection",
        "elapsed_sec": float(time.perf_counter() - started),
        "camera": {"source": camera_source, "width": int(camera.width), "height": int(camera.height)},
        "inputs": {
            "rows_jsonl": args.rows_jsonl,
            "match_jsonl": args.match_jsonl,
            "confidence_model": args.confidence_model,
            "query_pose_file": args.query_pose_file,
        },
        "feature_set": args.feature_set,
        "selection_specs": args.selection,
        "methods": {method: _summarize(rows) for method, rows in sorted(per_method.items())},
        "outputs": {"rows": args.output_jsonl},
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
