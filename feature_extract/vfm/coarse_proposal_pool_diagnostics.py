"""Diagnostics for coarse proposal-pool recall.

The matcher can increase the number of coarse hypotheses without improving the
usable pool. These helpers measure the quantity that matters for downstream
measurement and PnP: how many rows are already within a small pixel residual of
the GT projection.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _xy(row: Mapping[str, object], x_keys: Sequence[str], y_keys: Sequence[str]) -> tuple[float, float] | None:
    for x_key, y_key in zip(x_keys, y_keys):
        x = _finite_float(row.get(x_key))
        y = _finite_float(row.get(y_key))
        if x is not None and y is not None:
            return (float(x), float(y))
    return None


def row_residual_px(row: Mapping[str, object]) -> tuple[float | None, str]:
    """Return the coarse proposal residual and the source used to compute it."""

    center = _xy(
        row,
        ("query_center_x", "query_x", "center_x", "query_refined_x"),
        ("query_center_y", "query_y", "center_y", "query_refined_y"),
    )
    gt = _xy(row, ("query_gt_x", "gt_query_x"), ("query_gt_y", "gt_query_y"))
    if center is not None and gt is not None:
        return (float(math.hypot(center[0] - gt[0], center[1] - gt[1])), "xy_to_gt")
    residual = _finite_float(row.get("gt_reproj_error_px"))
    if residual is not None:
        return (float(residual), "gt_reproj_error_px")
    residual = _finite_float(row.get("requested_residual_px"))
    if residual is not None:
        return (float(residual), "requested_residual_px")
    return (None, "missing")


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _threshold_key(threshold_px: float) -> str:
    value = float(threshold_px)
    if value.is_integer():
        return f"{int(value)}px"
    return f"{str(value).replace('.', 'p')}px"


def summarize_pool_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    thresholds_px: Sequence[float] = (2.0, 5.0),
) -> dict[str, object]:
    values = list(rows)
    per_query_errors: dict[str, list[float]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    errors: list[float] = []
    missing_count = 0
    for index, row in enumerate(values):
        query_id = str(row.get("query_id", f"__missing_query_{index}"))
        residual, source = row_residual_px(row)
        source_counts[source] += 1
        if residual is None:
            missing_count += 1
            continue
        errors.append(float(residual))
        per_query_errors[query_id].append(float(residual))

    thresholds: dict[str, object] = {}
    for threshold in thresholds_px:
        key = _threshold_key(float(threshold))
        per_query_valid = [sum(1 for error in group if error <= float(threshold)) for group in per_query_errors.values()]
        valid_count = int(sum(per_query_valid))
        thresholds[key] = {
            "threshold_px": float(threshold),
            "valid_count": valid_count,
            "valid_rate": float(valid_count / len(errors)) if errors else None,
            "queries_with_valid_count": int(sum(1 for count in per_query_valid if count > 0)),
            "queries_with_valid_rate": (
                float(sum(1 for count in per_query_valid if count > 0) / len(per_query_valid))
                if per_query_valid
                else None
            ),
            "valid_per_query_mean": float(np.mean(per_query_valid)) if per_query_valid else None,
            "valid_per_query_p50": _percentile([float(v) for v in per_query_valid], 50.0),
            "valid_per_query_p90": _percentile([float(v) for v in per_query_valid], 90.0),
        }

    return {
        "row_count": int(len(values)),
        "usable_error_count": int(len(errors)),
        "missing_error_count": int(missing_count),
        "query_count": int(len(per_query_errors)),
        "rows_per_query_mean": float(len(values) / len(per_query_errors)) if per_query_errors else None,
        "error_px_median": _percentile(errors, 50.0),
        "error_px_p90": _percentile(errors, 90.0),
        "error_source_counts": dict(source_counts),
        "thresholds": thresholds,
    }


def compare_pool_summaries(
    baseline: Mapping[str, object],
    current: Mapping[str, object],
    *,
    growth_threshold: float = 3.0,
) -> dict[str, object]:
    baseline_thresholds = baseline.get("thresholds", {})
    current_thresholds = current.get("thresholds", {})
    comparisons: dict[str, object] = {}
    for key, current_value in current_thresholds.items():
        if not isinstance(current_value, Mapping):
            continue
        base_value = baseline_thresholds.get(key, {}) if isinstance(baseline_thresholds, Mapping) else {}
        if not isinstance(base_value, Mapping):
            base_value = {}
        base_count = int(base_value.get("valid_count", 0) or 0)
        current_count = int(current_value.get("valid_count", 0) or 0)
        base_queries = int(base_value.get("queries_with_valid_count", 0) or 0)
        current_queries = int(current_value.get("queries_with_valid_count", 0) or 0)
        count_growth = None if base_count == 0 else float(current_count / base_count)
        query_growth = None if base_queries == 0 else float(current_queries / base_queries)
        comparisons[key] = {
            "baseline_valid_count": base_count,
            "current_valid_count": current_count,
            "valid_count_delta": int(current_count - base_count),
            "valid_count_growth": count_growth,
            "passes_3x_growth": bool(count_growth is not None and count_growth >= float(growth_threshold)),
            "baseline_queries_with_valid_count": base_queries,
            "current_queries_with_valid_count": current_queries,
            "queries_with_valid_growth": query_growth,
        }
    return {
        "baseline_row_count": int(baseline.get("row_count", 0) or 0),
        "current_row_count": int(current.get("row_count", 0) or 0),
        "row_count_growth": (
            None
            if int(baseline.get("row_count", 0) or 0) == 0
            else float(int(current.get("row_count", 0) or 0) / int(baseline.get("row_count", 0) or 0))
        ),
        "thresholds": comparisons,
    }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary_json(path: str | Path, summary: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
