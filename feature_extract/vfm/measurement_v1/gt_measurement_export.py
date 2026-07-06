from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _read_csv(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = str(row.get(name, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def _str(row: dict[str, str], name: str, default: str = "") -> str:
    value = str(row.get(name, "")).strip()
    return value if value else default


def _metrics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "mean": float(np.mean(arr)),
    }


def export_gt_measurement_from_legacy_eval(
    *,
    eval_dir: Path,
    output_dir: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Export measurement_v1 audit tables from an existing GT-render eval.

    This function does not claim to be a new learned measurement branch. It
    re-expresses existing per-match GT-render diagnostics as fixed-anchor
    measurement rows so first-principles gates can be checked explicitly.
    """

    source = Path(eval_dir)
    match_table = source / "match_table.csv"
    if not match_table.exists():
        raise FileNotFoundError(f"missing match_table.csv: {match_table}")
    rows = _read_csv(match_table, max_rows=max_rows)
    match_query_ids = {str(row.get("query_id", "")) for row in rows if str(row.get("query_id", "")).strip()}
    rows_csv = source / "rows.csv"
    source_query_ids = set(match_query_ids)
    if rows_csv.exists() and max_rows is None:
        source_rows = _read_csv(rows_csv)
        source_query_ids = {str(row.get("query_id", "")) for row in source_rows if str(row.get("query_id", "")).strip()}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    anchor_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    residual_before: list[float] = []
    residual_after: list[float] = []
    paired_residuals: list[tuple[float, float]] = []
    for anchor_id, row in enumerate(rows):
        query_id = _str(row, "query_id")
        candidate_id = _str(row, "candidate_id", "gt")
        before = _float(row, "baseline_reproj_residual_px", default=np.nan)
        after = _float(row, "gt_reproj_error_px", default=np.nan)
        if np.isfinite(before):
            residual_before.append(float(before))
        if np.isfinite(after):
            residual_after.append(float(after))
        if np.isfinite(before) and np.isfinite(after):
            paired_residuals.append((float(before), float(after)))
        alpha = _float(row, "render_alpha", default=1.0)
        confidence = float(np.clip(_float(row, "confidence", default=1.0), 0.0, 1.0))
        anchor_rows.append(
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "anchor_id": anchor_id,
                "token_index": _str(row, "render_index", str(anchor_id)),
                "subanchor_index": 0,
                "render_x": _float(row, "render_x"),
                "render_y": _float(row, "render_y"),
                "X": _float(row, "world_x"),
                "Y": _float(row, "world_y"),
                "Z": _float(row, "world_z"),
                "depth": _float(row, "render_depth"),
                "alpha": alpha,
                "quality": float(np.clip(alpha, 0.0, 1.0)),
                "rejection_reason": "",
            }
        )
        measurement_rows.append(
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "anchor_id": anchor_id,
                "fit_or_verify": "verify",
                "query_prior_x": _float(row, "render_x"),
                "query_prior_y": _float(row, "render_y"),
                "query_gt_x": "",
                "query_gt_y": "",
                "query_pred_x": _float(row, "query_x"),
                "query_pred_y": _float(row, "query_y"),
                "residual_before_px": before,
                "residual_after_px": after,
                "fine_improved": bool(np.isfinite(before) and np.isfinite(after) and after < before),
                "p_visible": float(np.clip(alpha, 0.0, 1.0)),
                "p_assignment": confidence,
                "p_valid": float(np.clip(alpha, 0.0, 1.0) * confidence),
                "cov_xx": "",
                "cov_xy": "",
                "cov_yy": "",
                "mahalanobis2": "",
                "anchor_quality": float(np.clip(alpha, 0.0, 1.0)),
                "surface_type": "legacy_match_table_anchor",
            }
        )
    before_stats = _metrics(residual_before)
    after_stats = _metrics(residual_after)
    improved = [1.0 for before, after in paired_residuals if float(after) < float(before)]
    improve_ratio = None if not paired_residuals else float(sum(improved) / len(paired_residuals))
    after_arr = np.asarray(residual_after, dtype=np.float64)
    after_arr = after_arr[np.isfinite(after_arr)]
    summary = {
        "stage": "measurement_v1_gt_render_from_legacy_eval",
        "source_eval_dir": str(source),
        "metrics": {
            "source_query_count": int(len(source_query_ids)),
            "match_table_query_count": int(len(match_query_ids)),
            "missing_match_table_query_count": int(max(len(source_query_ids - match_query_ids), 0)),
            "query_count": int(len(match_query_ids)),
            "anchor_count": int(len(anchor_rows)),
            "measurement_count": int(len(measurement_rows)),
            "before_median_px": before_stats["median"],
            "before_p90_px": before_stats["p90"],
            "after_median_px": after_stats["median"],
            "after_p90_px": after_stats["p90"],
            "paired_residual_count": int(len(paired_residuals)),
            "fine_after_better_ratio": improve_ratio,
            "after_recall_0p5px": float(np.mean(after_arr <= 0.5)) if after_arr.size else None,
            "after_recall_1px": float(np.mean(after_arr <= 1.0)) if after_arr.size else None,
            "after_recall_2px": float(np.mean(after_arr <= 2.0)) if after_arr.size else None,
            "after_recall_5px": float(np.mean(after_arr <= 5.0)) if after_arr.size else None,
        },
        "gates": {},
        "outputs": {
            "anchor_rows": str(output / "anchor_rows.csv"),
            "measurement_rows": str(output / "measurement_rows.csv"),
            "summary": str(output / "summary.json"),
        },
        "limitations": [
            "This export reuses legacy match_table residuals; it is an audit view, not a new cache-backed measurement branch.",
            "query_gt_x/query_gt_y are unavailable in the legacy match_table and are left blank.",
        ],
    }
    summary["gates"]["G3_measurement_effective"] = bool(
        summary["metrics"]["after_median_px"] is not None
        and summary["metrics"]["after_p90_px"] is not None
        and float(summary["metrics"]["after_median_px"]) < 0.5
        and float(summary["metrics"]["after_p90_px"]) < 1.5
        and improve_ratio is not None
        and float(improve_ratio) > 0.8
    )
    _write_csv(
        output / "anchor_rows.csv",
        anchor_rows,
        [
            "query_id",
            "candidate_id",
            "anchor_id",
            "token_index",
            "subanchor_index",
            "render_x",
            "render_y",
            "X",
            "Y",
            "Z",
            "depth",
            "alpha",
            "quality",
            "rejection_reason",
        ],
    )
    _write_csv(
        output / "measurement_rows.csv",
        measurement_rows,
        [
            "query_id",
            "candidate_id",
            "anchor_id",
            "fit_or_verify",
            "query_prior_x",
            "query_prior_y",
            "query_gt_x",
            "query_gt_y",
            "query_pred_x",
            "query_pred_y",
            "residual_before_px",
            "residual_after_px",
            "fine_improved",
            "p_visible",
            "p_assignment",
            "p_valid",
            "cov_xx",
            "cov_xy",
            "cov_yy",
            "mahalanobis2",
            "anchor_quality",
            "surface_type",
        ],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
