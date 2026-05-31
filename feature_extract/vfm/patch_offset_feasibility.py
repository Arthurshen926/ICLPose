"""Diagnostics for patch-to-pixel offset feasibility experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _basic_error_stats(errors: np.ndarray, confidences: np.ndarray | None = None) -> dict[str, object]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "p75_error_px": None,
            "mean_confidence": None,
        }
    stats = {
        "count": int(values.size),
        "mean_error_px": float(np.mean(values)),
        "median_error_px": float(np.median(values)),
        "p75_error_px": float(np.quantile(values, 0.75)),
        "mean_confidence": None,
    }
    if confidences is not None:
        conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
        conf = conf[np.isfinite(conf)]
        stats["mean_confidence"] = None if conf.size == 0 else float(np.mean(conf))
    return stats


def calibration_summary(
    offset_errors_px: Sequence[float] | np.ndarray,
    confidences: Sequence[float] | np.ndarray,
    good_threshold_px: float = 4.0,
    bin_count: int = 10,
) -> dict[str, object]:
    """Summarize whether predicted confidence tracks low offset error."""

    errors = np.asarray(offset_errors_px, dtype=np.float64).reshape(-1)
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if errors.shape[0] != conf.shape[0]:
        raise ValueError("offset_errors_px and confidences must have the same length")
    valid = np.isfinite(errors) & np.isfinite(conf)
    errors = errors[valid]
    conf = np.clip(conf[valid], 0.0, 1.0)
    good = errors <= float(good_threshold_px)
    if errors.size == 0:
        return {
            "count": 0,
            "good_threshold_px": float(good_threshold_px),
            "ece": None,
            "brier": None,
            "precision_at_top10_percent": None,
            "bins": [],
        }
    brier = float(np.mean((conf - good.astype(np.float64)) ** 2))
    top_count = max(1, int(np.ceil(0.10 * float(errors.size))))
    top_idx = np.argsort(-conf)[:top_count]
    bins = []
    ece = 0.0
    for bin_idx in range(int(bin_count)):
        low = bin_idx / float(bin_count)
        high = (bin_idx + 1) / float(bin_count)
        if bin_idx + 1 == int(bin_count):
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)
        if not bool(np.any(mask)):
            bins.append(
                {
                    "bin": int(bin_idx),
                    "low": float(low),
                    "high": float(high),
                    "count": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                    "mean_error_px": None,
                }
            )
            continue
        accuracy = float(np.mean(good[mask]))
        mean_confidence = float(np.mean(conf[mask]))
        mean_error = float(np.mean(errors[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "bin": int(bin_idx),
                "low": float(low),
                "high": float(high),
                "count": int(np.sum(mask)),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "mean_error_px": mean_error,
            }
        )
    return {
        "count": int(errors.size),
        "good_threshold_px": float(good_threshold_px),
        "ece": float(ece),
        "brier": brier,
        "precision_at_top10_percent": float(np.mean(good[top_idx])),
        "bins": bins,
    }


def offset_error_bucket_summary(
    rows: Sequence[Mapping[str, Any]],
    offset_errors_px: Sequence[float] | np.ndarray,
    confidences: Sequence[float] | np.ndarray | None = None,
) -> dict[str, object]:
    """Bucket learned offset error by match labels and simple observable cues."""

    errors = np.asarray(offset_errors_px, dtype=np.float64).reshape(-1)
    if errors.shape[0] != len(rows):
        raise ValueError("offset_errors_px must have one value per row")
    conf = None if confidences is None else np.asarray(confidences, dtype=np.float64).reshape(-1)
    if conf is not None and conf.shape[0] != len(rows):
        raise ValueError("confidences must have one value per row")

    margins = np.asarray([
        0.0 if row.get("similarity_margin") is None else float(row.get("similarity_margin", 0.0))
        for row in rows
    ], dtype=np.float64)
    finite_margins = margins[np.isfinite(margins)]
    score_source = "similarity_margin"
    if finite_margins.size == 0 or float(np.max(finite_margins) - np.min(finite_margins)) < 1e-8:
        margins = np.asarray([
            0.0 if row.get("similarity") is None else float(row.get("similarity", 0.0))
            for row in rows
        ], dtype=np.float64)
        finite_margins = margins[np.isfinite(margins)]
        score_source = "similarity"
    margin_median = 0.0 if finite_margins.size == 0 else float(np.median(finite_margins))

    def mask_for(name: str) -> np.ndarray:
        if name == "patch_positive":
            return np.asarray([bool(row.get("patch_positive_label", row.get("patch_correct", False))) for row in rows])
        if name == "patch_negative":
            return ~mask_for("patch_positive")
        if name == "first_pass_inlier":
            return np.asarray([bool(row.get("pnp_inlier", False)) for row in rows])
        if name == "first_pass_outlier":
            return ~mask_for("first_pass_inlier")
        if name == "high_margin":
            return margins >= margin_median
        if name == "low_margin":
            return margins < margin_median
        if name == "single_positive_patch":
            return np.asarray([int(row.get("positive_count", 0) or 0) == 1 for row in rows])
        if name == "multi_positive_patch":
            return np.asarray([int(row.get("positive_count", 0) or 0) > 1 for row in rows])
        raise ValueError(f"unknown bucket: {name}")

    summary = {"margin_median": margin_median, "margin_score_source": score_source}
    for name in (
        "patch_positive",
        "patch_negative",
        "first_pass_inlier",
        "first_pass_outlier",
        "high_margin",
        "low_margin",
        "single_positive_patch",
        "multi_positive_patch",
    ):
        mask = mask_for(name)
        summary[name] = _basic_error_stats(errors[mask], None if conf is None else conf[mask])
    return summary


def pose_metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Aggregate pose metrics for one offset variant."""

    count = len(rows)
    if count == 0:
        return {
            "query_count": 0,
            "solve_rate": None,
            "success_10cm_5deg": None,
            "success_25cm_10deg": None,
            "success_50cm_10deg": None,
            "median_translation_error_m": None,
            "median_rotation_error_deg": None,
            "mean_inlier_count": None,
        }
    translations = [_finite_float(row.get("translation_error_m")) for row in rows]
    rotations = [_finite_float(row.get("rotation_error_deg")) for row in rows]
    present_t = [value for value in translations if value is not None]
    present_r = [value for value in rotations if value is not None]

    def success_rate(t_thr: float, r_thr: float) -> float:
        flags = [
            t is not None and r is not None and float(t) <= float(t_thr) and float(r) <= float(r_thr)
            for t, r in zip(translations, rotations)
        ]
        return float(np.mean(flags))

    return {
        "query_count": int(count),
        "solve_rate": float(np.mean([bool(row.get("success", False)) for row in rows])),
        "success_10cm_5deg": success_rate(0.10, 5.0),
        "success_25cm_10deg": success_rate(0.25, 10.0),
        "success_50cm_10deg": success_rate(0.50, 10.0),
        "median_translation_error_m": None if not present_t else float(np.median(present_t)),
        "median_rotation_error_deg": None if not present_r else float(np.median(present_r)),
        "mean_inlier_count": float(np.mean([int(row.get("inlier_count", 0) or 0) for row in rows])),
    }
