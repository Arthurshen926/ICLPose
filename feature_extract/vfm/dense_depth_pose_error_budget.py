from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.dense_depth_measurement_fusion import (
    GEOMETRY_SOURCE_CHOICES,
    dense_depth_matches_from_rows,
)
from feature_extract.vfm.render_pose_diagnostics import run_pnp_solver_ablation


POSE_BUDGET_FIELDNAMES = [
    "group_id",
    "query_id",
    "candidate_id",
    "variant",
    "solver",
    "success",
    "input_row_count",
    "kept_row_count",
    "match_count",
    "inlier_count",
    "inlier_ratio",
    "translation_error_m",
    "rotation_error_deg",
    "residual_median_px",
    "residual_p90_px",
    "error",
    "center_error_median_px",
    "center_error_p90_px",
    "valid_2px_count",
    "valid_5px_count",
    "grid4_coverage",
    "depth_p05_m",
    "depth_p95_m",
    "depth_span_m",
]

VALIDITY_FIELDNAMES = [
    "query_id",
    "candidate_id",
    "match_index",
    "center_error_px",
    "valid_0p5px",
    "valid_1px",
    "valid_2px",
    "valid_5px",
    "query_center_x",
    "query_center_y",
    "query_gt_x",
    "query_gt_y",
    "render_depth",
    "radio_match_score",
    "measurement_valid_prob",
    "local_cost_entropy",
    "local_cost_peak_prob",
    "local_cost_top2_gap",
]

VALIDITY_SCORE_COLUMNS = (
    "measurement_valid_prob",
    "radio_match_score",
    "local_cost_peak_prob",
    "local_cost_top2_gap",
    "inverse_local_cost_entropy",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return float(number) if np.isfinite(number) else None


def _candidate_id(row: Mapping[str, Any]) -> str:
    for name in ("candidate_id", "render_pose_id", "render_pose_label", "initial_render_pose_label"):
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _group_id(query_id: str, candidate_id: str) -> str:
    return str(query_id) if not str(candidate_id).strip() else f"{query_id}::{candidate_id}"


def _rows_by_pose_group(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        candidate_id = _candidate_id(row)
        grouped.setdefault((query_id, candidate_id), []).append(dict(row))
    return grouped


def _xy(row: Mapping[str, Any], x_name: str, y_name: str) -> np.ndarray | None:
    x = _finite_float(row.get(x_name))
    y = _finite_float(row.get(y_name))
    if x is None or y is None:
        return None
    return np.asarray([x, y], dtype=np.float64)


def _center_xy(row: Mapping[str, Any]) -> np.ndarray | None:
    value = _xy(row, "query_center_x", "query_center_y")
    if value is not None:
        return value
    return _xy(row, "center_x", "center_y")


def _gt_xy(row: Mapping[str, Any]) -> np.ndarray | None:
    return _xy(row, "query_gt_x", "query_gt_y")


def _center_error(row: Mapping[str, Any]) -> float | None:
    center = _center_xy(row)
    gt = _gt_xy(row)
    if center is None or gt is None:
        return None
    return float(np.linalg.norm(center - gt))


def _set_refined_xy(row: Mapping[str, Any], xy: np.ndarray) -> dict[str, Any]:
    item = dict(row)
    item["query_refined_x"] = float(xy[0])
    item["query_refined_y"] = float(xy[1])
    center = _center_xy(item)
    if center is not None:
        item["measurement_dx"] = float(xy[0] - center[0])
        item["measurement_dy"] = float(xy[1] - center[1])
    return item


def _variant_rng(seed: int, variant: str) -> np.random.Generator:
    # Stable, process-independent variant seed.
    value = int(seed)
    for char in str(variant):
        value = (value * 131 + ord(char)) % (2**32 - 1)
    return np.random.default_rng(value)


def _format_px(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def _variant_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = str(variant)
    output: list[dict[str, Any]] = []
    missing_center = 0
    missing_gt = 0
    rng = _variant_rng(int(seed), name)
    for row in rows:
        center = _center_xy(row)
        gt = _gt_xy(row)
        if name == "center":
            if center is None:
                missing_center += 1
                continue
            output.append(_set_refined_xy(row, center))
            continue
        if gt is None:
            missing_gt += 1
            continue
        if name == "oracle":
            output.append(_set_refined_xy(row, gt))
            continue
        if name.startswith("oracle_noise_") and name.endswith("px"):
            sigma_text = name[len("oracle_noise_") : -2].replace("p", ".")
            sigma = float(sigma_text)
            noise = rng.normal(loc=0.0, scale=sigma, size=2)
            output.append(_set_refined_xy(row, gt + noise))
            continue
        if name.startswith("oracle_quantized_stride") and name.endswith("px"):
            stride_text = name[len("oracle_quantized_stride") : -2].replace("p", ".")
            stride = float(stride_text)
            if stride <= 0.0:
                raise ValueError("quantization stride must be positive")
            output.append(_set_refined_xy(row, np.round(gt / stride) * stride))
            continue
        if name.startswith("center_valid_") and name.endswith("px"):
            if center is None:
                missing_center += 1
                continue
            threshold = float(name[len("center_valid_") : -2].replace("p", "."))
            error = _center_error(row)
            if error is not None and error <= threshold:
                output.append(_set_refined_xy(row, center))
            continue
        if name.startswith("oracle_valid_") and name.endswith("px"):
            threshold = float(name[len("oracle_valid_") : -2].replace("p", "."))
            error = _center_error(row)
            if error is not None and error <= threshold:
                output.append(_set_refined_xy(row, gt))
            continue
        raise ValueError(f"unsupported pose budget variant: {variant}")
    return output, {
        "variant": name,
        "input_row_count": int(len(rows)),
        "kept_row_count": int(len(output)),
        "missing_center_count": int(missing_center),
        "missing_gt_count": int(missing_gt),
    }


def default_pose_budget_variants(
    *,
    noise_sigmas_px: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    quantization_strides_px: Sequence[float] = (2.0, 4.0, 8.0),
    validity_thresholds_px: Sequence[float] = (2.0, 5.0),
) -> list[str]:
    variants = ["center", "oracle"]
    variants.extend(f"oracle_noise_{_format_px(value)}px" for value in noise_sigmas_px)
    variants.extend(f"oracle_quantized_stride{_format_px(value)}px" for value in quantization_strides_px)
    for threshold in validity_thresholds_px:
        tag = _format_px(threshold)
        variants.append(f"center_valid_{tag}px")
        variants.append(f"oracle_valid_{tag}px")
    seen: set[str] = set()
    out: list[str] = []
    for variant in variants:
        if variant not in seen:
            out.append(variant)
            seen.add(variant)
    return out


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def _coverage_summary(rows: Sequence[Mapping[str, Any]], *, image_width: int, image_height: int) -> dict[str, Any]:
    center_errors = [value for row in rows if (value := _center_error(row)) is not None]
    depths = [value for row in rows if (value := _finite_float(row.get("render_depth"))) is not None]
    cells: set[tuple[int, int]] = set()
    width = max(int(image_width), 1)
    height = max(int(image_height), 1)
    for row in rows:
        center = _center_xy(row)
        if center is None or not np.isfinite(center).all():
            continue
        ix = min(max(int(math.floor(float(center[0]) / width * 4.0)), 0), 3)
        iy = min(max(int(math.floor(float(center[1]) / height * 4.0)), 0), 3)
        cells.add((ix, iy))
    return {
        "center_error_median_px": _percentile(center_errors, 50.0),
        "center_error_p90_px": _percentile(center_errors, 90.0),
        "valid_2px_count": int(sum(1 for value in center_errors if value <= 2.0)),
        "valid_5px_count": int(sum(1 for value in center_errors if value <= 5.0)),
        "grid4_coverage": int(len(cells)),
        "depth_p05_m": _percentile(depths, 5.0),
        "depth_p95_m": _percentile(depths, 95.0),
        "depth_span_m": None if not depths else float(max(depths) - min(depths)),
    }


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _aggregate_pose_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("variant", "")), str(row.get("solver", ""))), []).append(row)
    out: list[dict[str, Any]] = []
    for (variant, solver), values in sorted(by_key.items()):
        query_count = len(values)
        successes = [_bool_value(row.get("success")) for row in values]
        translations = [_finite_float(row.get("translation_error_m")) for row in values]
        rotations = [_finite_float(row.get("rotation_error_deg")) for row in values]
        residuals = [_finite_float(row.get("residual_median_px")) for row in values]
        inliers = [_finite_float(row.get("inlier_count")) for row in values]
        kept = [_finite_float(row.get("kept_row_count")) for row in values]
        grid = [_finite_float(row.get("grid4_coverage")) for row in values]

        def success_rate(t_threshold: float, r_threshold: float) -> float:
            count = 0
            for ok, t_err, r_err in zip(successes, translations, rotations):
                if ok and t_err is not None and r_err is not None and t_err <= t_threshold and r_err <= r_threshold:
                    count += 1
            return float(count / query_count) if query_count else 0.0

        out.append(
            {
                "variant": variant,
                "solver": solver,
                "query_count": int(query_count),
                "pnp_success_rate": float(sum(1 for item in successes if item) / query_count) if query_count else 0.0,
                "median_translation_error_m": _percentile([item for item in translations if item is not None], 50.0),
                "p90_translation_error_m": _percentile([item for item in translations if item is not None], 90.0),
                "median_rotation_error_deg": _percentile([item for item in rotations if item is not None], 50.0),
                "p90_rotation_error_deg": _percentile([item for item in rotations if item is not None], 90.0),
                "success_3cm_1deg": success_rate(0.03, 1.0),
                "success_5cm_2deg": success_rate(0.05, 2.0),
                "success_10cm_5deg": success_rate(0.10, 5.0),
                "median_residual_median_px": _percentile([item for item in residuals if item is not None], 50.0),
                "median_inlier_count": _percentile([item for item in inliers if item is not None], 50.0),
                "median_kept_row_count": _percentile([item for item in kept if item is not None], 50.0),
                "median_grid4_coverage": _percentile([item for item in grid if item is not None], 50.0),
            }
        )
    return out


def _failed_solver_rows(solvers: Sequence[str], match_count: int, error: str) -> dict[str, dict[str, Any]]:
    return {
        str(solver): {
            "solver": str(solver),
            "success": False,
            "match_count": int(match_count),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "translation_error_m": None,
            "rotation_error_deg": None,
            "residual_median_px": None,
            "residual_p90_px": None,
            "error": str(error),
        }
        for solver in solvers
    }


def _run_pnp_solver_ablation_safe(
    matches: Sequence[Any],
    camera: ColmapCamera,
    *,
    gt_pose_w2c: np.ndarray | None,
    solvers: Sequence[str],
    reprojection_error_px: float,
) -> dict[str, dict[str, Any]]:
    values = list(matches)
    if len(values) < 6:
        return _failed_solver_rows(solvers, len(values), "insufficient_matches_for_budget_pnp")
    try:
        return run_pnp_solver_ablation(
            values,
            camera,
            gt_pose_w2c=gt_pose_w2c,
            solvers=tuple(str(value) for value in solvers),
            reprojection_error_px=float(reprojection_error_px),
        )
    except Exception as exc:
        return _failed_solver_rows(solvers, len(values), f"{type(exc).__name__}: {exc}")


def build_center_validity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        error = _center_error(row)
        center = _center_xy(row)
        gt = _gt_xy(row)
        candidate_id = _candidate_id(row)
        item = {
            "query_id": str(row.get("query_id", "")),
            "candidate_id": candidate_id,
            "match_index": row.get("match_index", index),
            "center_error_px": "" if error is None else float(error),
            "valid_0p5px": "" if error is None else bool(error <= 0.5),
            "valid_1px": "" if error is None else bool(error <= 1.0),
            "valid_2px": "" if error is None else bool(error <= 2.0),
            "valid_5px": "" if error is None else bool(error <= 5.0),
            "query_center_x": "" if center is None else float(center[0]),
            "query_center_y": "" if center is None else float(center[1]),
            "query_gt_x": "" if gt is None else float(gt[0]),
            "query_gt_y": "" if gt is None else float(gt[1]),
        }
        for name in (
            "render_depth",
            "radio_match_score",
            "measurement_valid_prob",
            "local_cost_entropy",
            "local_cost_peak_prob",
            "local_cost_top2_gap",
        ):
            item[name] = row.get(name, "")
        output.append(item)
    return output


def _score_values(rows: Sequence[Mapping[str, Any]], score_column: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if score_column == "inverse_local_cost_entropy":
            value = _finite_float(row.get("local_cost_entropy"))
            values.append(float("nan") if value is None else 1.0 - value)
        else:
            value = _finite_float(row.get(score_column))
            values.append(float("nan") if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return None
    s = scores[valid]
    y = labels[valid].astype(bool)
    positives = int(np.count_nonzero(y))
    if positives == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(y_sorted.size, dtype=np.float64) + 1.0)
    return float(np.sum(precision[y_sorted]) / positives)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return None
    s = scores[valid]
    y = labels[valid].astype(bool)
    pos = int(np.count_nonzero(y))
    neg = int(y.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average tied ranks.
    unique_scores, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    del unique_scores
    if np.any(counts > 1):
        rank_sums = np.bincount(inverse, weights=ranks)
        avg_ranks = rank_sums / counts
        ranks = avg_ranks[inverse]
    pos_rank_sum = float(np.sum(ranks[y]))
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _ece(scores: np.ndarray, labels: np.ndarray, *, bin_count: int = 10) -> float | None:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return None
    s = np.clip(scores[valid], 0.0, 1.0)
    y = labels[valid].astype(np.float64)
    ece = 0.0
    for index in range(int(bin_count)):
        lo = index / float(bin_count)
        hi = (index + 1) / float(bin_count)
        mask = (s >= lo) & (s <= hi if index == int(bin_count) - 1 else s < hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(s[mask])) - float(np.mean(y[mask])))
    return float(ece)


def _precision_at_coverages(scores: np.ndarray, labels: np.ndarray, coverages: Sequence[float]) -> list[dict[str, Any]]:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return []
    s = scores[valid]
    y = labels[valid].astype(bool)
    order = np.argsort(-s, kind="mergesort")
    rows: list[dict[str, Any]] = []
    for coverage in coverages:
        k = max(1, int(math.ceil(float(coverage) * y.size)))
        selected = y[order[:k]]
        rows.append(
            {
                "coverage": float(coverage),
                "selected_count": int(k),
                "precision": float(np.mean(selected)) if selected.size else None,
                "positive_count": int(np.count_nonzero(selected)),
            }
        )
    return rows


def center_validity_score_summary(
    validity_rows: Sequence[Mapping[str, Any]],
    *,
    score_columns: Sequence[str] = VALIDITY_SCORE_COLUMNS,
    thresholds_px: Sequence[float] = (2.0, 5.0),
    coverages: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> dict[str, Any]:
    rows = list(validity_rows)
    summary: dict[str, Any] = {"row_count": int(len(rows)), "thresholds": {}}
    for threshold in thresholds_px:
        label_key = f"valid_{_format_px(threshold)}px"
        labels = np.asarray([1.0 if str(row.get(label_key, "")).lower() == "true" else 0.0 for row in rows], dtype=np.float64)
        threshold_summary: dict[str, Any] = {
            "positive_rate": float(np.mean(labels)) if labels.size else None,
            "positive_count": int(np.count_nonzero(labels)),
            "score_columns": {},
        }
        for score_column in score_columns:
            scores = _score_values(rows, str(score_column))
            finite = np.isfinite(scores)
            if not np.any(finite):
                continue
            clipped = np.clip(scores, 0.0, 1.0)
            threshold_summary["score_columns"][str(score_column)] = {
                "finite_count": int(np.count_nonzero(finite)),
                "score_min": float(np.nanmin(scores)),
                "score_max": float(np.nanmax(scores)),
                "auroc": _auroc(scores, labels),
                "auprc": _average_precision(scores, labels),
                "brier_clipped": float(np.mean((clipped[finite] - labels[finite]) ** 2)),
                "ece_clipped": _ece(scores, labels),
                "precision_at_coverage": _precision_at_coverages(scores, labels, coverages),
            }
        summary["thresholds"][label_key] = threshold_summary
    return summary


def evaluate_pose_error_budget(
    *,
    match_table_csv: Path,
    output_dir: Path,
    camera: ColmapCamera,
    query_pose_w2c_by_id: Mapping[str, np.ndarray] | None = None,
    variants: Sequence[str] | None = None,
    solvers: Sequence[str] = ("plain", "ransac", "weighted"),
    reprojection_error_px: float = 8.0,
    geometry_source: str = "prefer_world_xyz",
    world_xyz_consistency_threshold_m: float = 1e-4,
    image_width: int | None = None,
    image_height: int | None = None,
    seed: int = 20260706,
) -> dict[str, Any]:
    if str(geometry_source) not in GEOMETRY_SOURCE_CHOICES:
        raise ValueError(f"geometry_source must be one of {GEOMETRY_SOURCE_CHOICES}")
    rows = _read_csv(Path(match_table_csv))
    groups = _rows_by_pose_group(rows)
    variant_names = list(variants) if variants is not None else default_pose_budget_variants()
    width = int(image_width if image_width is not None else camera.width)
    height = int(image_height if image_height is not None else camera.height)
    pose_rows: list[dict[str, Any]] = []
    variant_prep: dict[str, Any] = {}
    coverage_rows: list[dict[str, Any]] = []
    for (query_id, candidate_id), group_rows in sorted(groups.items()):
        gt_pose = None if query_pose_w2c_by_id is None else query_pose_w2c_by_id.get(str(query_id))
        gid = _group_id(str(query_id), str(candidate_id))
        for variant in variant_names:
            prepared, prep = _variant_rows(group_rows, variant=str(variant), seed=int(seed))
            variant_prep.setdefault(str(variant), {"input_row_count": 0, "kept_row_count": 0})
            variant_prep[str(variant)]["input_row_count"] += int(prep["input_row_count"])
            variant_prep[str(variant)]["kept_row_count"] += int(prep["kept_row_count"])
            coverage = _coverage_summary(prepared, image_width=width, image_height=height)
            coverage_rows.append(
                {
                    "group_id": gid,
                    "query_id": str(query_id),
                    "candidate_id": str(candidate_id),
                    "variant": str(variant),
                    "input_row_count": int(prep["input_row_count"]),
                    "kept_row_count": int(prep["kept_row_count"]),
                    **coverage,
                }
            )
            matches, _match_summary = dense_depth_matches_from_rows(
                prepared,
                camera=camera,
                render_pose_w2c=None,
                source=f"pose_error_budget:{variant}",
                geometry_source=str(geometry_source),
                world_xyz_consistency_threshold_m=float(world_xyz_consistency_threshold_m),
            )
            report = _run_pnp_solver_ablation_safe(
                matches,
                camera,
                gt_pose_w2c=gt_pose,
                solvers=tuple(str(value) for value in solvers),
                reprojection_error_px=float(reprojection_error_px),
            )
            for solver_name, solver_row in report.items():
                pose_rows.append(
                    {
                        "group_id": gid,
                        "query_id": str(query_id),
                        "candidate_id": str(candidate_id),
                        "variant": str(variant),
                        "solver": str(solver_name),
                        "input_row_count": int(prep["input_row_count"]),
                        "kept_row_count": int(prep["kept_row_count"]),
                        **coverage,
                        **solver_row,
                    }
                )
    validity_rows = build_center_validity_rows(rows)
    validity_summary = center_validity_score_summary(validity_rows)
    aggregate_rows = _aggregate_pose_rows(pose_rows)
    output = Path(output_dir)
    _write_csv(output / "pose_rows.csv", pose_rows, POSE_BUDGET_FIELDNAMES)
    _write_csv(output / "pose_budget_summary.tsv", aggregate_rows, list(aggregate_rows[0].keys()) if aggregate_rows else [], delimiter="\t")
    _write_csv(output / "center_validity_rows.csv", validity_rows, VALIDITY_FIELDNAMES)
    _write_csv(output / "coverage_rows.csv", coverage_rows, list(coverage_rows[0].keys()) if coverage_rows else [])
    summary = {
        "stage": "dense_depth_pose_error_budget",
        "match_table_csv": str(match_table_csv),
        "row_count": int(len(rows)),
        "query_count": int(len({str(row.get("query_id", "")) for row in rows})),
        "pose_group_count": int(len(groups)),
        "variants": variant_names,
        "solvers": [str(value) for value in solvers],
        "geometry_source": str(geometry_source),
        "variant_preparation": variant_prep,
        "pose_summary": aggregate_rows,
        "center_validity_summary": validity_summary,
        "outputs": {
            "pose_rows_csv": str(output / "pose_rows.csv"),
            "pose_budget_summary_tsv": str(output / "pose_budget_summary.tsv"),
            "center_validity_rows_csv": str(output / "center_validity_rows.csv"),
            "coverage_rows_csv": str(output / "coverage_rows.csv"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
