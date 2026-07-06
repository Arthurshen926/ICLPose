from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _float_or_nan(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _median(values: list[float]) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _mean(values: list[float]) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _read_match_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing match table: {path}")
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _summary_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    payload = json.loads(path.read_text())
    return dict(payload.get("metrics", payload))


def summarize_legacy_measurement_eval(eval_dir: str | Path) -> dict[str, Any]:
    """Summarize a legacy eval directory through measurement-first diagnostics.

    This report does not claim to run measurement_v1. It reinterprets legacy
    full-eval artifacts to decide which first-principles gates are currently
    unsupported: fine residual improvement, confidence calibration, oracle fine
    gap, and reference-prior basin gap.
    """

    root = Path(eval_dir)
    metrics = _summary_metrics(root / "summary.json")
    rows = _read_match_rows(root / "match_table.csv")
    before = [_float_or_nan(row.get("baseline_reproj_residual_px")) for row in rows]
    after = [_float_or_nan(row.get("gt_reproj_error_px")) for row in rows]
    confidences = [_float_or_nan(row.get("confidence")) for row in rows]
    before_arr = np.asarray(before, dtype=np.float64)
    after_arr = np.asarray(after, dtype=np.float64)
    valid = np.isfinite(before_arr) & np.isfinite(after_arr)
    if np.any(valid):
        improvements = before_arr[valid] - after_arr[valid]
        fine_after_better_ratio = float(np.mean(improvements > 0.0))
        fine_improvement_mean_px = float(np.mean(improvements))
        after_recall_5px = float(np.mean(after_arr[valid] <= 5.0))
        after_recall_1px = float(np.mean(after_arr[valid] <= 1.0))
        before_recall_5px = float(np.mean(before_arr[valid] <= 5.0))
        before_recall_1px = float(np.mean(before_arr[valid] <= 1.0))
    else:
        fine_after_better_ratio = 0.0
        fine_improvement_mean_px = None
        after_recall_5px = 0.0
        after_recall_1px = 0.0
        before_recall_5px = 0.0
        before_recall_1px = 0.0
    correct_5 = [_bool_value(row.get("gt_correct_5px")) for row in rows]
    pnp_inlier = [_bool_value(row.get("pnp_inlier")) for row in rows]
    confidence_arr = np.asarray(confidences, dtype=np.float64)
    inlier_arr = np.asarray(pnp_inlier, dtype=bool)
    confidence_valid = np.isfinite(confidence_arr)
    if np.any(confidence_valid & inlier_arr) and np.any(confidence_valid & ~inlier_arr):
        confidence_inlier_gap = float(np.mean(confidence_arr[confidence_valid & inlier_arr]) - np.mean(confidence_arr[confidence_valid & ~inlier_arr]))
    else:
        confidence_inlier_gap = None
    normal_median = metrics.get("oracle_pnp_all_ransac_median_translation_error_m", metrics.get("median_translation_error_m"))
    oracle_fine_median = metrics.get("oracle_pnp_oracle_fine_all_ransac_median_translation_error_m")
    if normal_median is not None and oracle_fine_median is not None:
        oracle_fine_gap = float(normal_median) - float(oracle_fine_median)
    else:
        oracle_fine_gap = None
    measurement = {
        "match_count": int(len(rows)),
        "query_count_from_matches": int(len({str(row.get("query_id", "")) for row in rows})),
        "before_median_px": _median(before),
        "after_median_px": _median(after),
        "fine_after_better_ratio": fine_after_better_ratio,
        "fine_improvement_mean_px": fine_improvement_mean_px,
        "before_recall_1px": before_recall_1px,
        "before_recall_5px": before_recall_5px,
        "after_recall_1px": after_recall_1px,
        "after_recall_5px": after_recall_5px,
        "gt_correct_5px_rate": float(np.mean(correct_5)) if correct_5 else 0.0,
    }
    calibration = {
        "confidence_mean": _mean(confidences),
        "confidence_inlier_gap": confidence_inlier_gap,
        "match_validity_ece_5px": metrics.get("match_validity_ece_5px"),
        "match_validity_ece_10px": metrics.get("match_validity_ece_10px"),
        "confidence_ece_16px": metrics.get("confidence_ece_16px"),
    }
    oracle = {
        "normal_median_translation_error_m": normal_median,
        "oracle_fine_median_translation_error_m": oracle_fine_median,
        "oracle_fine_gap_m": oracle_fine_gap,
        "oracle_uncertainty_median_translation_error_m": metrics.get("oracle_pnp_all_oracle_uncertainty_median_translation_error_m"),
        "oracle_match_5px_median_translation_error_m": metrics.get("oracle_pnp_oracle_match_5px_ransac_median_translation_error_m"),
    }
    fine_gate = bool(
        measurement["after_median_px"] is not None
        and measurement["before_median_px"] is not None
        and float(measurement["after_median_px"]) < float(measurement["before_median_px"])
        and fine_after_better_ratio >= 0.8
    )
    confidence_gate = bool(
        calibration["match_validity_ece_5px"] is not None
        and float(calibration["match_validity_ece_5px"]) < 0.05
        and confidence_inlier_gap is not None
        and float(confidence_inlier_gap) > 0.05
    )
    claim_level = "partial_gt_render_signal_only"
    if fine_gate and confidence_gate and float(metrics.get("median_translation_error_m", 1e9)) <= 0.03:
        claim_level = "measurement_gate_candidate"
    return {
        "stage": "measurement_v1_legacy_report",
        "eval_dir": str(root),
        "metrics": metrics,
        "measurement": measurement,
        "calibration": calibration,
        "oracle": oracle,
        "assessment": {
            "fine_measurement_gate_passed": fine_gate,
            "confidence_gate_passed": confidence_gate,
            "claim_level": claim_level,
            "notes": [
                "This report summarizes legacy artifacts; it is not a measurement_v1 full run.",
                "A failed fine gate means query-side continuous measurement is not yet demonstrated.",
                "A large oracle_fine_gap means geometry can improve if measurement is corrected.",
            ],
        },
    }


def report_to_markdown(report: Mapping[str, Any]) -> str:
    measurement = dict(report.get("measurement", {}))
    calibration = dict(report.get("calibration", {}))
    oracle = dict(report.get("oracle", {}))
    assessment = dict(report.get("assessment", {}))
    lines = [
        "# measurement_v1 legacy diagnostic report",
        "",
        f"- eval_dir: `{report.get('eval_dir')}`",
        f"- claim_level: `{assessment.get('claim_level')}`",
        f"- fine gate: `{assessment.get('fine_measurement_gate_passed')}`",
        f"- confidence gate: `{assessment.get('confidence_gate_passed')}`",
        "",
        "## Measurement",
        f"- before median px: `{measurement.get('before_median_px')}`",
        f"- after median px: `{measurement.get('after_median_px')}`",
        f"- fine-after-better ratio: `{measurement.get('fine_after_better_ratio')}`",
        f"- after recall 1px / 5px: `{measurement.get('after_recall_1px')}` / `{measurement.get('after_recall_5px')}`",
        "",
        "## Calibration",
        f"- confidence mean: `{calibration.get('confidence_mean')}`",
        f"- confidence inlier gap: `{calibration.get('confidence_inlier_gap')}`",
        f"- match validity ECE 5px: `{calibration.get('match_validity_ece_5px')}`",
        "",
        "## Oracle",
        f"- normal median translation m: `{oracle.get('normal_median_translation_error_m')}`",
        f"- oracle fine median translation m: `{oracle.get('oracle_fine_median_translation_error_m')}`",
        f"- oracle fine gap m: `{oracle.get('oracle_fine_gap_m')}`",
        "",
    ]
    return "\n".join(lines)
