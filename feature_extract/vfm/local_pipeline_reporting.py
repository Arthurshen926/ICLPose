"""Reporting helpers for the canonical VFM local patch-to-3D pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import json
import math


def _finite_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_jsonl_rows(path: str | Path) -> list[dict[str, object]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def metric_row_from_summary(scene: str, method: str, summary: Mapping[str, object]) -> dict[str, object]:
    query_count = int(summary.get("query_count") or 0)
    elapsed = _finite_float(summary.get("elapsed_sec"), 0.0) or 0.0
    return {
        "scene": str(scene),
        "method": str(method),
        "query_count": query_count,
        "success_10cm_5deg": _finite_float(summary.get("success_10cm_5deg")),
        "success_25cm_10deg": _finite_float(summary.get("success_25cm_10deg")),
        "success_50cm_10deg": _finite_float(summary.get("success_50cm_10deg")),
        "median_translation_error_m": _finite_float(summary.get("median_translation_error_m")),
        "median_rotation_error_deg": _finite_float(summary.get("median_rotation_error_deg")),
        "mean_pnp_inlier_patch_at_1": _finite_float(summary.get("mean_pnp_inlier_patch_at_1")),
        "mean_pnp_inlier_count": _finite_float(summary.get("mean_pnp_inlier_count")),
        "pnp_solve_rate": _finite_float(summary.get("pnp_solve_rate")),
        "runtime_sec_per_query": None if query_count <= 0 else elapsed / float(query_count),
    }


def _bucket() -> dict[str, object]:
    return {"count": 0, "examples": []}


def _append_bucket(buckets: dict[str, dict[str, object]], name: str, query_id: str, max_examples: int) -> None:
    bucket = buckets.setdefault(name, _bucket())
    bucket["count"] = int(bucket["count"]) + 1
    examples = bucket["examples"]
    if isinstance(examples, list) and len(examples) < int(max_examples):
        examples.append(str(query_id))


def failure_bucket_summary(
    rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]] | None = None,
    success_key: str = "success_25cm_10deg",
    max_examples: int = 12,
) -> dict[str, object]:
    baseline_by_query = {str(row.get("query_id")): row for row in (baseline_rows or [])}
    bucket_names = (
        "inlier_count_too_low",
        "inlier_count_high_but_wrong_pose",
        "inliers_spatially_degenerate",
        "matches_concentrated_one_surface",
        "reference_top10_coverage_low",
        "repeated_structure_confusion",
        "lm_refinement_worsens",
        "ransac_succeeds_but_final_error_gt_25cm",
    )
    buckets: dict[str, dict[str, object]] = {name: _bucket() for name in bucket_names}
    failed_count = 0
    for row in rows:
        if bool(row.get(success_key)):
            continue
        failed_count += 1
        query_id = str(row.get("query_id", ""))
        inlier_count = _finite_float(row.get("pnp_inlier_count"), 0.0) or 0.0
        visible_recall = _finite_float(row.get("visible_landmark_recall"))
        mean_similarity = _finite_float(row.get("mean_similarity"), 0.0) or 0.0
        patch = dict(row.get("patch_geometry") or {})
        spatial = dict(row.get("pnp_inlier_spatial") or {})
        patch_at_1 = _finite_float(patch.get("patch_at_1"), 0.0) or 0.0
        occupancy = _finite_float(spatial.get("grid_4x4_occupancy_frac"), 0.0) or 0.0
        depth_range = _finite_float(spatial.get("depth_range_m"), 0.0) or 0.0
        planarity = _finite_float(spatial.get("xyz_planarity_ratio"), 0.0) or 0.0

        if inlier_count < 64:
            _append_bucket(buckets, "inlier_count_too_low", query_id, max_examples)
        else:
            _append_bucket(buckets, "inlier_count_high_but_wrong_pose", query_id, max_examples)
        if occupancy < 0.25 or depth_range < 1.0 or planarity < 0.02:
            _append_bucket(buckets, "inliers_spatially_degenerate", query_id, max_examples)
        if planarity < 0.005 and inlier_count >= 32:
            _append_bucket(buckets, "matches_concentrated_one_surface", query_id, max_examples)
        if visible_recall is not None and visible_recall < 0.5:
            _append_bucket(buckets, "reference_top10_coverage_low", query_id, max_examples)
        if mean_similarity >= 0.5 and patch_at_1 < 0.35:
            _append_bucket(buckets, "repeated_structure_confusion", query_id, max_examples)
        if bool(row.get("pnp_solve")):
            _append_bucket(buckets, "ransac_succeeds_but_final_error_gt_25cm", query_id, max_examples)

        baseline = baseline_by_query.get(query_id)
        if baseline is not None:
            baseline_t = _finite_float(baseline.get("translation_error_m"))
            current_t = _finite_float(row.get("translation_error_m"))
            baseline_success = bool(baseline.get(success_key))
            if baseline_success and current_t is not None and baseline_t is not None and current_t > baseline_t + 0.05:
                _append_bucket(buckets, "lm_refinement_worsens", query_id, max_examples)

    return {
        "query_count": int(len(rows)),
        "failed_query_count": int(failed_count),
        "success_key": str(success_key),
        "buckets": buckets,
    }
