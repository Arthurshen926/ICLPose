from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_training import _optional_bool, _read_csv


TEACHER_FIELDNAMES = [
    "row_index",
    "query_id",
    "support_image_id",
    "track_id",
    "support_track_id",
    "target_is_dustbin",
    "baseline_epe_px",
    "lk_epe_px",
    "fallback_epe_px",
    "lk_improved",
    "lk_applied",
    "lk_reason",
    "center_x",
    "center_y",
    "query_gt_x",
    "query_gt_y",
    "lk_pred_x",
    "lk_pred_y",
    "lk_dx",
    "lk_dy",
    "support_x",
    "support_y",
    "lk_error",
    "fb_error_px",
    "flow_from_center_px",
    "support_grad_mean",
    "query_center_grad_mean",
    "center_patch_ncc",
    "center_patch_mae",
    "requested_residual_px",
]


@dataclass(frozen=True)
class LKAlignmentConfig:
    win_size_px: int = 21
    max_level: int = 2
    criteria_count: int = 30
    criteria_eps: float = 0.01
    min_eig_threshold: float = 1e-4
    max_lk_error: float | None = 40.0
    max_flow_from_center_px: float | None = 8.0
    fb_max_error_px: float | None = 1.0

    def __post_init__(self) -> None:
        if int(self.win_size_px) <= 2:
            raise ValueError("win_size_px must be > 2")
        if int(self.max_level) < 0:
            raise ValueError("max_level must be non-negative")
        if int(self.criteria_count) <= 0 or float(self.criteria_eps) <= 0.0:
            raise ValueError("criteria_count/criteria_eps are invalid")
        if float(self.min_eig_threshold) < 0.0:
            raise ValueError("min_eig_threshold must be non-negative")
        if self.max_lk_error is not None and float(self.max_lk_error) < 0.0:
            raise ValueError("max_lk_error must be non-negative")
        if self.max_flow_from_center_px is not None and float(self.max_flow_from_center_px) < 0.0:
            raise ValueError("max_flow_from_center_px must be non-negative")
        if self.fb_max_error_px is not None and float(self.fb_max_error_px) < 0.0:
            raise ValueError("fb_max_error_px must be non-negative")


@dataclass(frozen=True)
class LKAlignmentResult:
    pred_xy: np.ndarray
    applied: bool
    reason: str
    lk_error: float | None = None
    fb_error_px: float | None = None
    flow_from_center_px: float | None = None


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEACHER_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in TEACHER_FIELDNAMES})


def _float(row: Mapping[str, object], key: str) -> float:
    return float(str(row.get(key, "")).strip())


def _optional_float(row: Mapping[str, object], key: str) -> float | None:
    text = str(row.get(key, "")).strip()
    if not text:
        return None
    return float(text)


def _load_gray(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(Path(path)).convert("RGB"), dtype=np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def _inside_margin(xy: np.ndarray, image: np.ndarray, margin: float) -> bool:
    x, y = float(xy[0]), float(xy[1])
    h, w = int(image.shape[0]), int(image.shape[1])
    return float(margin) <= x < float(w) - float(margin) and float(margin) <= y < float(h) - float(margin)


def _extract_patch(image: np.ndarray, xy: np.ndarray, radius_px: int) -> np.ndarray | None:
    values = np.asarray(image, dtype=np.float32)
    xy_value = np.asarray(xy, dtype=np.float64).reshape(2)
    x = int(round(float(xy_value[0])))
    y = int(round(float(xy_value[1])))
    radius = int(radius_px)
    if x - radius < 0 or y - radius < 0 or x + radius >= values.shape[1] or y + radius >= values.shape[0]:
        return None
    return values[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32, copy=True)


def _gradient_mean(patch: np.ndarray | None) -> float | None:
    if patch is None:
        return None
    values = np.asarray(patch, dtype=np.float32)
    if values.size == 0:
        return None
    gy, gx = np.gradient(values)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _patch_ncc(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-8:
        return None
    return float(np.dot(aa, bb) / denom)


def patch_observability(
    *,
    support_gray: np.ndarray,
    query_gray: np.ndarray,
    support_xy: np.ndarray,
    query_xy: np.ndarray,
    radius_px: int,
) -> dict[str, float | None]:
    support_patch = _extract_patch(support_gray, support_xy, int(radius_px))
    query_patch = _extract_patch(query_gray, query_xy, int(radius_px))
    mae = None
    if support_patch is not None and query_patch is not None:
        mae = float(np.mean(np.abs(support_patch.astype(np.float32) - query_patch.astype(np.float32))))
    return {
        "support_grad_mean": _gradient_mean(support_patch),
        "query_grad_mean": _gradient_mean(query_patch),
        "patch_ncc": _patch_ncc(support_patch, query_patch),
        "photometric_mae": mae,
    }


def estimate_lk_alignment(
    *,
    support_gray: np.ndarray,
    query_gray: np.ndarray,
    support_xy: np.ndarray,
    center_xy: np.ndarray,
    config: LKAlignmentConfig,
) -> LKAlignmentResult:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for dense LK alignment") from exc

    support = np.asarray(support_gray)
    query = np.asarray(query_gray)
    if support.ndim != 2 or query.ndim != 2:
        raise ValueError("support_gray/query_gray must be HxW")
    support_point = np.asarray(support_xy, dtype=np.float32).reshape(1, 1, 2)
    initial_query_point = np.asarray(center_xy, dtype=np.float32).reshape(1, 1, 2)
    margin = max(float(config.win_size_px), 2.0)
    if not _inside_margin(support_point.reshape(2), support, margin):
        return LKAlignmentResult(initial_query_point.reshape(2).astype(np.float64), False, "support_oob")
    if not _inside_margin(initial_query_point.reshape(2), query, margin):
        return LKAlignmentResult(initial_query_point.reshape(2).astype(np.float64), False, "center_oob")
    criteria = (
        int(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT),
        int(config.criteria_count),
        float(config.criteria_eps),
    )
    next_point, status, error = cv2.calcOpticalFlowPyrLK(
        support,
        query,
        support_point,
        initial_query_point.copy(),
        winSize=(int(config.win_size_px), int(config.win_size_px)),
        maxLevel=int(config.max_level),
        criteria=criteria,
        flags=int(cv2.OPTFLOW_USE_INITIAL_FLOW),
        minEigThreshold=float(config.min_eig_threshold),
    )
    if next_point is None or status is None or int(status.reshape(-1)[0]) == 0:
        return LKAlignmentResult(initial_query_point.reshape(2).astype(np.float64), False, "lk_status_failed")
    pred = np.asarray(next_point, dtype=np.float64).reshape(2)
    lk_error = None if error is None else float(np.asarray(error).reshape(-1)[0])
    flow_from_center = float(np.linalg.norm(pred - np.asarray(center_xy, dtype=np.float64).reshape(2)))
    if config.max_lk_error is not None and lk_error is not None and lk_error > float(config.max_lk_error):
        return LKAlignmentResult(pred, False, "high_lk_error", lk_error=lk_error, flow_from_center_px=flow_from_center)
    if config.max_flow_from_center_px is not None and flow_from_center > float(config.max_flow_from_center_px):
        return LKAlignmentResult(pred, False, "large_flow_from_center", lk_error=lk_error, flow_from_center_px=flow_from_center)
    fb_error = None
    if config.fb_max_error_px is not None:
        backward, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
            query,
            support,
            pred.astype(np.float32).reshape(1, 1, 2),
            support_point.copy(),
            winSize=(int(config.win_size_px), int(config.win_size_px)),
            maxLevel=int(config.max_level),
            criteria=criteria,
            flags=int(cv2.OPTFLOW_USE_INITIAL_FLOW),
            minEigThreshold=float(config.min_eig_threshold),
        )
        if backward is None or backward_status is None or int(backward_status.reshape(-1)[0]) == 0:
            return LKAlignmentResult(pred, False, "fb_status_failed", lk_error=lk_error, flow_from_center_px=flow_from_center)
        fb_error = float(np.linalg.norm(np.asarray(backward, dtype=np.float64).reshape(2) - support_point.reshape(2)))
        if fb_error > float(config.fb_max_error_px):
            return LKAlignmentResult(
                pred,
                False,
                "high_fb_error",
                lk_error=lk_error,
                fb_error_px=fb_error,
                flow_from_center_px=flow_from_center,
            )
    return LKAlignmentResult(
        pred,
        True,
        "applied",
        lk_error=lk_error,
        fb_error_px=fb_error,
        flow_from_center_px=flow_from_center,
    )


def _recall(values: np.ndarray, threshold: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values <= float(threshold)))


def _median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def _p90(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, 90.0))


def _finite_array(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=np.float64)


def _oracle_gated_epe(
    *,
    baseline_values: Sequence[float],
    applied_values: Sequence[float | None],
) -> np.ndarray:
    if len(baseline_values) != len(applied_values):
        raise ValueError("baseline_values and applied_values must have the same length")
    out = []
    for baseline, applied in zip(baseline_values, applied_values):
        baseline_value = float(baseline)
        if applied is None:
            out.append(baseline_value)
            continue
        applied_value = float(applied)
        out.append(applied_value if applied_value < baseline_value else baseline_value)
    return np.asarray(out, dtype=np.float64)


def _center_preserving_hybrid_epe(
    *,
    baseline_values: Sequence[float],
    applied_values: Sequence[float | None],
    requested_residual_values: Sequence[float | None],
    preserve_below_px: float,
) -> np.ndarray:
    if len(baseline_values) != len(applied_values) or len(baseline_values) != len(requested_residual_values):
        raise ValueError("baseline/applied/requested residual values must have the same length")
    out = []
    threshold = float(preserve_below_px)
    for baseline, applied, requested in zip(baseline_values, applied_values, requested_residual_values):
        baseline_value = float(baseline)
        if requested is not None and float(requested) <= threshold:
            out.append(baseline_value)
            continue
        if applied is None:
            out.append(baseline_value)
        else:
            out.append(float(applied))
    return np.asarray(out, dtype=np.float64)


def _metric_block(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    return {
        f"{prefix}_median_px": _median(values),
        f"{prefix}_p90_px": _p90(values),
        f"{prefix}_recall_0p5px": _recall(values, 0.5),
        f"{prefix}_recall_1px": _recall(values, 1.0),
        f"{prefix}_recall_2px": _recall(values, 2.0),
        f"{prefix}_recall_5px": _recall(values, 5.0),
    }


def _optional_metric_summary(values: Sequence[float | None], *, prefix: str) -> dict[str, float | int | None]:
    arr = _finite_array(values)
    out: dict[str, float | int | None] = {f"{prefix}_count": int(arr.size)}
    out.update(_metric_block(prefix, arr))
    return out


def _residual_bin_metric_block(
    rows: Sequence[Mapping[str, object]],
    values: Sequence[float | None],
    baselines: Sequence[float],
    *,
    prefix: str,
) -> dict[str, float | int | None]:
    indices_by_bin: dict[float, list[int]] = {}
    for index, row in enumerate(rows):
        requested = _optional_float(row, "requested_residual_px")
        if requested is not None:
            indices_by_bin.setdefault(float(requested), []).append(int(index))
    out: dict[str, float | int | None] = {}
    for bin_value in sorted(indices_by_bin):
        indices = indices_by_bin[bin_value]
        bin_values = _finite_array([values[index] for index in indices])
        bin_baselines = np.asarray([float(baselines[index]) for index in indices], dtype=np.float64)
        key = f"{prefix}_bin_{bin_value:.3f}".replace(".", "p")
        out[f"{key}_count"] = int(len(indices))
        out[f"{key}_applied_count"] = int(bin_values.size)
        out[f"{key}_applied_ratio"] = float(bin_values.size / max(len(indices), 1))
        out.update(_metric_block(f"{key}_epe", bin_values))
        finite_positions = [position for position, index in enumerate(indices) if values[index] is not None]
        if finite_positions:
            finite_baselines = bin_baselines[np.asarray(finite_positions, dtype=np.int64)]
            out[f"{key}_improve_ratio"] = float(np.mean(bin_values < finite_baselines))
        else:
            out[f"{key}_improve_ratio"] = None
    return out


def _threshold_gated_epe(
    *,
    baseline_values: Sequence[float],
    applied_values: Sequence[float | None],
    score_values: Sequence[float | None],
    threshold: float,
    direction: str,
) -> np.ndarray:
    if len(baseline_values) != len(applied_values) or len(baseline_values) != len(score_values):
        raise ValueError("baseline/applied/score values must have the same length")
    out = []
    for baseline, applied, score in zip(baseline_values, applied_values, score_values):
        baseline_value = float(baseline)
        if applied is None or score is None:
            out.append(baseline_value)
            continue
        score_value = float(score)
        if direction == "le":
            use_applied = score_value <= float(threshold)
        elif direction == "ge":
            use_applied = score_value >= float(threshold)
        else:
            raise ValueError("direction must be 'le' or 'ge'")
        out.append(float(applied) if use_applied else baseline_value)
    return np.asarray(out, dtype=np.float64)


def _add_threshold_sweep_metrics(
    metrics: dict[str, float | int | None],
    *,
    prefix: str,
    baseline_values: Sequence[float],
    applied_values: Sequence[float | None],
    score_values: Sequence[float | None],
    thresholds: Sequence[float],
    direction: str = "le",
) -> None:
    baseline = np.asarray(baseline_values, dtype=np.float64)
    best_key = None
    best_median = None
    best_improve = None
    for threshold in thresholds:
        values = _threshold_gated_epe(
            baseline_values=baseline_values,
            applied_values=applied_values,
            score_values=score_values,
            threshold=float(threshold),
            direction=direction,
        )
        key = f"{prefix}_thr_{float(threshold):.3f}".replace(".", "p")
        metrics.update(_metric_block(key, values))
        improve = float(np.mean(values < baseline)) if values.size else None
        metrics[f"{key}_improve_ratio"] = improve
        median = _median(values)
        if median is not None and (best_median is None or float(median) < float(best_median)):
            best_key = key
            best_median = median
            best_improve = improve
    metrics[f"{prefix}_best_key"] = best_key
    metrics[f"{prefix}_best_median_px"] = best_median
    metrics[f"{prefix}_best_improve_ratio"] = best_improve


def _add_center_preserve_sweep_metrics(
    metrics: dict[str, float | int | None],
    *,
    baseline_values: Sequence[float],
    applied_values: Sequence[float | None],
    requested_residual_values: Sequence[float | None],
    thresholds: Sequence[float],
    prefix: str = "lk_center_preserve",
) -> None:
    baseline = np.asarray(baseline_values, dtype=np.float64)
    best_key = None
    best_median = None
    best_improve = None
    for threshold in thresholds:
        values = _center_preserving_hybrid_epe(
            baseline_values=baseline_values,
            applied_values=applied_values,
            requested_residual_values=requested_residual_values,
            preserve_below_px=float(threshold),
        )
        key = f"{prefix}_thr_{float(threshold):.3f}".replace(".", "p")
        metrics.update(_metric_block(key, values))
        improve = float(np.mean(values < baseline)) if values.size else None
        metrics[f"{key}_improve_ratio"] = improve
        median = _median(values)
        if median is not None and (best_median is None or float(median) < float(best_median)):
            best_key = key
            best_median = median
            best_improve = improve
    metrics[f"{prefix}_best_key"] = best_key
    metrics[f"{prefix}_best_median_px"] = best_median
    metrics[f"{prefix}_best_improve_ratio"] = best_improve


def export_dense_alignment_teacher_diagnostics(
    *,
    rows_csv: Path,
    image_root: Path,
    output_dir: Path,
    max_rows: int | None = None,
    win_size_px: int = 21,
    max_level: int = 2,
    criteria_count: int = 30,
    criteria_eps: float = 0.01,
    min_eig_threshold: float = 1e-4,
    max_lk_error: float | None = 40.0,
    max_flow_from_center_px: float | None = 8.0,
    fb_max_error_px: float | None = 1.0,
    center_preserve_below_px: float | None = None,
) -> dict[str, Any]:
    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    if not rows:
        raise ValueError("rows_csv contains no rows")
    config = LKAlignmentConfig(
        win_size_px=int(win_size_px),
        max_level=int(max_level),
        criteria_count=int(criteria_count),
        criteria_eps=float(criteria_eps),
        min_eig_threshold=float(min_eig_threshold),
        max_lk_error=max_lk_error,
        max_flow_from_center_px=max_flow_from_center_px,
        fb_max_error_px=fb_max_error_px,
    )
    image_cache: dict[str, np.ndarray] = {}

    def get_gray(image_id: str) -> np.ndarray:
        key = str(image_id)
        if key not in image_cache:
            image_cache[key] = _load_gray(Path(image_root) / key)
        return image_cache[key]

    diagnostic_rows: list[dict[str, object]] = []
    positive_metric_rows: list[Mapping[str, object]] = []
    baseline_values: list[float] = []
    fallback_values: list[float] = []
    applied_values: list[float | None] = []
    lk_error_values: list[float | None] = []
    fb_error_values: list[float | None] = []
    flow_values: list[float | None] = []
    requested_residual_values: list[float | None] = []
    support_grad_values: list[float | None] = []
    query_grad_values: list[float | None] = []
    center_ncc_values: list[float | None] = []
    center_mae_values: list[float | None] = []
    applied_flags: list[bool] = []
    reason_counts: dict[str, int] = {}
    applied_count = 0
    dustbin_count = 0
    for row_index, row in enumerate(rows):
        query_id = str(row.get("query_id", "")).strip()
        support_id = str(row.get("support_image_id", "")).strip()
        if not query_id:
            raise ValueError("row missing query_id")
        if not support_id:
            raise ValueError("row missing support_image_id")
        target_is_dustbin = bool(_optional_bool(row, "target_is_dustbin") or False)
        if target_is_dustbin:
            dustbin_count += 1
        support_xy = np.asarray(
            [
                _float(row, "support_x") if str(row.get("support_x", "")).strip() else _float(row, "render_x"),
                _float(row, "support_y") if str(row.get("support_y", "")).strip() else _float(row, "render_y"),
            ],
            dtype=np.float64,
        )
        center = np.asarray([_float(row, "center_x"), _float(row, "center_y")], dtype=np.float64)
        gt = np.asarray([_float(row, "query_gt_x"), _float(row, "query_gt_y")], dtype=np.float64)
        baseline = float(np.linalg.norm(gt - center))
        result = LKAlignmentResult(center.copy(), False, "target_dustbin")
        lk_epe: float | None = None
        lk_dxdy: np.ndarray | None = None
        if not target_is_dustbin:
            support_gray = get_gray(support_id)
            query_gray = get_gray(query_id)
            observability = patch_observability(
                support_gray=support_gray,
                query_gray=query_gray,
                support_xy=support_xy,
                query_xy=center,
                radius_px=max(2, int(round(float(win_size_px) / 2.0))),
            )
            result = estimate_lk_alignment(
                support_gray=support_gray,
                query_gray=query_gray,
                support_xy=support_xy,
                center_xy=center,
                config=config,
            )
            if result.applied:
                lk_epe = float(np.linalg.norm(result.pred_xy - gt))
                lk_dxdy = result.pred_xy - center
                applied_count += 1
        fallback_epe = lk_epe if lk_epe is not None else baseline
        if not target_is_dustbin:
            positive_metric_rows.append(row)
            baseline_values.append(baseline)
            fallback_values.append(float(fallback_epe))
            applied_values.append(lk_epe)
            lk_error_values.append(result.lk_error if result.applied else None)
            fb_error_values.append(result.fb_error_px if result.applied else None)
            flow_values.append(result.flow_from_center_px if result.applied else None)
            requested_residual_values.append(_optional_float(row, "requested_residual_px"))
            support_grad_values.append(observability["support_grad_mean"])
            query_grad_values.append(observability["query_grad_mean"])
            center_ncc_values.append(observability["patch_ncc"])
            center_mae_values.append(observability["photometric_mae"])
            applied_flags.append(bool(result.applied))
        else:
            observability = {
                "support_grad_mean": None,
                "query_grad_mean": None,
                "patch_ncc": None,
                "photometric_mae": None,
            }
        reason_counts[str(result.reason)] = reason_counts.get(str(result.reason), 0) + 1
        diagnostic_rows.append(
            {
                "row_index": int(row_index),
                "query_id": query_id,
                "support_image_id": support_id,
                "track_id": str(row.get("track_id", "")),
                "support_track_id": str(row.get("support_track_id", "")),
                "target_is_dustbin": str(target_is_dustbin),
                "baseline_epe_px": baseline,
                "lk_epe_px": "" if lk_epe is None else lk_epe,
                "fallback_epe_px": fallback_epe,
                "lk_improved": bool(lk_epe is not None and lk_epe < baseline),
                "lk_applied": bool(result.applied),
                "lk_reason": str(result.reason),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "query_gt_x": float(gt[0]),
                "query_gt_y": float(gt[1]),
                "lk_pred_x": "" if not result.applied else float(result.pred_xy[0]),
                "lk_pred_y": "" if not result.applied else float(result.pred_xy[1]),
                "lk_dx": "" if lk_dxdy is None else float(lk_dxdy[0]),
                "lk_dy": "" if lk_dxdy is None else float(lk_dxdy[1]),
                "support_x": float(support_xy[0]),
                "support_y": float(support_xy[1]),
                "lk_error": "" if result.lk_error is None else float(result.lk_error),
                "fb_error_px": "" if result.fb_error_px is None else float(result.fb_error_px),
                "flow_from_center_px": "" if result.flow_from_center_px is None else float(result.flow_from_center_px),
                "support_grad_mean": "" if observability["support_grad_mean"] is None else float(observability["support_grad_mean"]),
                "query_center_grad_mean": "" if observability["query_grad_mean"] is None else float(observability["query_grad_mean"]),
                "center_patch_ncc": "" if observability["patch_ncc"] is None else float(observability["patch_ncc"]),
                "center_patch_mae": "" if observability["photometric_mae"] is None else float(observability["photometric_mae"]),
                "requested_residual_px": str(row.get("requested_residual_px", "")),
            }
        )
    output = Path(output_dir)
    _write_csv(output / "teacher_rows.csv", diagnostic_rows)
    baseline_arr = np.asarray(baseline_values, dtype=np.float64)
    fallback_arr = np.asarray(fallback_values, dtype=np.float64)
    applied_arr = _finite_array(applied_values)
    applied_baselines = np.asarray(
        [baseline for baseline, value in zip(baseline_values, applied_values) if value is not None],
        dtype=np.float64,
    )
    metrics: dict[str, float | int | None] = {
        "baseline_median_px": _median(baseline_arr),
        "baseline_p90_px": _p90(baseline_arr),
        "lk_fallback_improve_ratio": None if fallback_arr.size == 0 else float(np.mean(fallback_arr < baseline_arr)),
        "lk_applied_improve_ratio": (
            None if applied_arr.size == 0 else float(np.mean(applied_arr < applied_baselines))
        ),
    }
    metrics.update(_metric_block("lk_applied", applied_arr))
    metrics.update(_metric_block("lk_fallback", fallback_arr))
    oracle_gated_arr = _oracle_gated_epe(baseline_values=baseline_values, applied_values=applied_values)
    metrics.update(_metric_block("lk_oracle_gated", oracle_gated_arr))
    metrics["lk_oracle_gated_improve_ratio"] = (
        None if oracle_gated_arr.size == 0 else float(np.mean(oracle_gated_arr < baseline_arr))
    )
    if center_preserve_below_px is not None:
        center_preserve_arr = _center_preserving_hybrid_epe(
            baseline_values=baseline_values,
            applied_values=applied_values,
            requested_residual_values=requested_residual_values,
            preserve_below_px=float(center_preserve_below_px),
        )
        metrics.update(_metric_block("lk_center_preserve", center_preserve_arr))
        metrics["lk_center_preserve_improve_ratio"] = (
            None if center_preserve_arr.size == 0 else float(np.mean(center_preserve_arr < baseline_arr))
        )
    _add_center_preserve_sweep_metrics(
        metrics,
        baseline_values=baseline_values,
        applied_values=applied_values,
        requested_residual_values=requested_residual_values,
        thresholds=(0.25, 0.5, 0.75, 1.0, 1.5),
    )
    _add_threshold_sweep_metrics(
        metrics,
        prefix="lk_fb_gated",
        baseline_values=baseline_values,
        applied_values=applied_values,
        score_values=fb_error_values,
        thresholds=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        direction="le",
    )
    _add_threshold_sweep_metrics(
        metrics,
        prefix="lk_error_gated",
        baseline_values=baseline_values,
        applied_values=applied_values,
        score_values=lk_error_values,
        thresholds=(5.0, 10.0, 20.0, 40.0),
        direction="le",
    )
    _add_threshold_sweep_metrics(
        metrics,
        prefix="lk_flow_gated",
        baseline_values=baseline_values,
        applied_values=applied_values,
        score_values=flow_values,
        thresholds=(1.0, 2.0, 4.0, 8.0),
        direction="le",
    )
    _add_threshold_sweep_metrics(
        metrics,
        prefix="lk_support_grad_gated",
        baseline_values=baseline_values,
        applied_values=applied_values,
        score_values=support_grad_values,
        thresholds=(2.0, 5.0, 8.0, 12.0, 16.0),
        direction="ge",
    )
    _add_threshold_sweep_metrics(
        metrics,
        prefix="lk_center_ncc_gated",
        baseline_values=baseline_values,
        applied_values=applied_values,
        score_values=center_ncc_values,
        thresholds=(0.50, 0.70, 0.85, 0.90, 0.95),
        direction="ge",
    )
    metrics.update(_residual_bin_metric_block(positive_metric_rows, applied_values, baseline_values, prefix="lk_applied"))
    metrics.update(_residual_bin_metric_block(positive_metric_rows, fallback_values, baseline_values, prefix="lk_fallback"))
    applied_mask = np.asarray(applied_flags, dtype=bool)
    observability_summary = {
        **_optional_metric_summary(support_grad_values, prefix="support_grad_mean"),
        **_optional_metric_summary(query_grad_values, prefix="query_center_grad_mean"),
        **_optional_metric_summary(center_ncc_values, prefix="center_patch_ncc"),
        **_optional_metric_summary(center_mae_values, prefix="center_patch_mae"),
    }
    if applied_mask.size:
        observability_summary.update(
            {
                **_optional_metric_summary(
                    [value for value, flag in zip(support_grad_values, applied_flags) if flag],
                    prefix="applied_support_grad_mean",
                ),
                **_optional_metric_summary(
                    [value for value, flag in zip(support_grad_values, applied_flags) if not flag],
                    prefix="rejected_support_grad_mean",
                ),
                **_optional_metric_summary(
                    [value for value, flag in zip(center_ncc_values, applied_flags) if flag],
                    prefix="applied_center_patch_ncc",
                ),
                **_optional_metric_summary(
                    [value for value, flag in zip(center_ncc_values, applied_flags) if not flag],
                    prefix="rejected_center_patch_ncc",
                ),
            }
        )
    summary = {
        "stage": "measurement_v1_dense_alignment_teacher_diagnostics",
        "rows_csv": str(rows_csv),
        "row_count": int(len(rows)),
        "positive_count": int(len(positive_metric_rows)),
        "dustbin_count": int(dustbin_count),
        "applied_count": int(applied_count),
        "applied_ratio": float(applied_count / max(len(positive_metric_rows), 1)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "config": {
            "win_size_px": int(config.win_size_px),
            "max_level": int(config.max_level),
            "criteria_count": int(config.criteria_count),
            "criteria_eps": float(config.criteria_eps),
            "min_eig_threshold": float(config.min_eig_threshold),
            "max_lk_error": None if config.max_lk_error is None else float(config.max_lk_error),
            "max_flow_from_center_px": None
            if config.max_flow_from_center_px is None
            else float(config.max_flow_from_center_px),
            "fb_max_error_px": None if config.fb_max_error_px is None else float(config.fb_max_error_px),
            "center_preserve_below_px": None if center_preserve_below_px is None else float(center_preserve_below_px),
        },
        "metrics": metrics,
        "observability": observability_summary,
        "outputs": {
            "teacher_rows": str(output / "teacher_rows.csv"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
