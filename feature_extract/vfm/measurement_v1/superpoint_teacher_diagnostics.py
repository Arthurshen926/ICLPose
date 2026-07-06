from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from feature_extract.vfm.lowlevel_offset_sidecar import (
    HLocSuperPointKeypointDetector,
    LandmarkConditionedKeypointSelectorConfig,
    SuperPointKeypointSet,
    select_landmark_conditioned_keypoint,
    select_support_superpoint_keypoint,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import _optional_bool, _read_csv


TEACHER_FIELDNAMES = [
    "row_index",
    "query_id",
    "support_image_id",
    "track_id",
    "support_track_id",
    "target_is_dustbin",
    "baseline_epe_px",
    "teacher_epe_px",
    "fallback_epe_px",
    "teacher_improved",
    "teacher_applied",
    "teacher_reason",
    "center_x",
    "center_y",
    "query_gt_x",
    "query_gt_y",
    "teacher_pred_x",
    "teacher_pred_y",
    "teacher_dx",
    "teacher_dy",
    "support_x",
    "support_y",
    "support_keypoint_x",
    "support_keypoint_y",
    "support_delta_x",
    "support_delta_y",
    "support_keypoint_score",
    "support_keypoint_distance_px",
    "teacher_score",
    "teacher_confidence",
    "query_keypoint_count",
    "support_keypoint_count",
    "oracle_query_sp_distance_px",
    "requested_residual_px",
]


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


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(Path(path)).convert("RGB"))


def _keypoint_cache_path(cache_dir: Path, image_id: str) -> Path:
    digest = hashlib.sha1(str(image_id).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.npz"


def _save_keypoints(path: Path, keypoints: SuperPointKeypointSet, *, image_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = (
        np.zeros((0, 0), dtype=np.float32)
        if keypoints.descriptors is None
        else np.asarray(keypoints.descriptors, dtype=np.float32)
    )
    np.savez_compressed(
        path,
        image_id=np.asarray(str(image_id)),
        xy=np.asarray(keypoints.xy, dtype=np.float32),
        scores=np.asarray(keypoints.scores, dtype=np.float32),
        descriptors=descriptors,
        has_descriptors=np.asarray(keypoints.descriptors is not None),
    )


def _load_keypoints(path: Path, *, expected_image_id: str) -> SuperPointKeypointSet | None:
    if not Path(path).exists():
        return None
    with np.load(Path(path), allow_pickle=False) as data:
        image_id = str(data["image_id"].item())
        if image_id != str(expected_image_id):
            return None
        descriptors = None
        if bool(data["has_descriptors"].item()):
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
        return SuperPointKeypointSet(
            xy=np.asarray(data["xy"], dtype=np.float32),
            scores=np.asarray(data["scores"], dtype=np.float32),
            descriptors=descriptors,
        )


def _nearest_keypoint_distance(xy: np.ndarray, keypoints: SuperPointKeypointSet) -> float | None:
    keypoint_xy = np.asarray(keypoints.xy, dtype=np.float64).reshape(-1, 2)
    if keypoint_xy.shape[0] == 0:
        return None
    target = np.asarray(xy, dtype=np.float64).reshape(2)
    distances = np.linalg.norm(keypoint_xy - target[None, :], axis=1)
    return float(np.min(distances))


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


def _metric_block(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    return {
        f"{prefix}_median_px": _median(values),
        f"{prefix}_p90_px": _p90(values),
        f"{prefix}_recall_0p5px": _recall(values, 0.5),
        f"{prefix}_recall_1px": _recall(values, 1.0),
        f"{prefix}_recall_2px": _recall(values, 2.0),
        f"{prefix}_recall_5px": _recall(values, 5.0),
    }


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
        out.update(_metric_block(f"{key}_epe", bin_values))
        finite_positions = [position for position, index in enumerate(indices) if values[index] is not None]
        out[f"{key}_applied_count"] = int(bin_values.size)
        out[f"{key}_applied_ratio"] = float(bin_values.size / max(len(indices), 1))
        if finite_positions:
            finite_baselines = bin_baselines[np.asarray(finite_positions, dtype=np.int64)]
            out[f"{key}_improve_ratio"] = float(np.mean(bin_values < finite_baselines))
        else:
            out[f"{key}_improve_ratio"] = None
    return out


def export_superpoint_teacher_diagnostics(
    *,
    rows_csv: Path,
    image_root: Path,
    output_dir: Path,
    keypoints_by_image: Mapping[str, SuperPointKeypointSet] | None = None,
    keypoint_cache_dir: Path | None = None,
    detector: Any | None = None,
    device: str = "cuda",
    max_rows: int | None = None,
    candidate_radius_px: float = 8.0,
    support_radius_px: float = 8.0,
    min_query_score: float = 0.005,
    min_support_score: float = 0.005,
    descriptor_weight: float = 1.0,
    query_score_weight: float = 0.25,
    center_penalty_weight: float = 0.15,
    support_distance_penalty_weight: float = 0.10,
    score_threshold: float = 0.2,
    support_selection_strategy: str = "nearest",
    superpoint_nms_radius: int = 4,
    superpoint_keypoint_threshold: float = 0.005,
    superpoint_max_keypoints: int = -1,
    superpoint_remove_borders: int = 4,
) -> dict[str, Any]:
    """Evaluate SuperPoint descriptor-conditioned local measurement teachers.

    The teacher is intentionally diagnostic-only: it uses known support/query
    row pairings and predicts the query-side measurement by selecting a query
    SuperPoint keypoint whose descriptor matches the support keypoint near the
    immutable anchor. Rows with no confident selection remain invalid and use
    the original center as a fallback metric.
    """

    rows = _read_csv(Path(rows_csv), max_rows=max_rows)
    if not rows:
        raise ValueError("rows_csv contains no rows")
    keypoint_cache: dict[str, SuperPointKeypointSet] = dict(keypoints_by_image or {})
    image_cache: dict[str, np.ndarray] = {}
    cache_dir = None if keypoint_cache_dir is None or not str(keypoint_cache_dir) else Path(keypoint_cache_dir)
    detector_instance = detector

    def get_keypoints(image_id: str) -> SuperPointKeypointSet:
        nonlocal detector_instance
        key = str(image_id)
        if key not in keypoint_cache:
            if cache_dir is not None:
                cached = _load_keypoints(_keypoint_cache_path(cache_dir, key), expected_image_id=key)
                if cached is not None:
                    keypoint_cache[key] = cached
                    return keypoint_cache[key]
            if detector_instance is None:
                detector_instance = HLocSuperPointKeypointDetector(
                    device=str(device),
                    nms_radius=int(superpoint_nms_radius),
                    keypoint_threshold=float(superpoint_keypoint_threshold),
                    max_keypoints=int(superpoint_max_keypoints),
                    remove_borders=int(superpoint_remove_borders),
                )
            path = Path(image_root) / key
            if key not in image_cache:
                image_cache[key] = _load_rgb(path)
            keypoint_cache[key] = detector_instance.detect(image_cache[key])
            if cache_dir is not None:
                _save_keypoints(_keypoint_cache_path(cache_dir, key), keypoint_cache[key], image_id=key)
        return keypoint_cache[key]

    config = LandmarkConditionedKeypointSelectorConfig(
        candidate_radius_px=float(candidate_radius_px),
        support_radius_px=float(support_radius_px),
        min_query_score=float(min_query_score),
        min_support_score=float(min_support_score),
        descriptor_weight=float(descriptor_weight),
        query_score_weight=float(query_score_weight),
        center_penalty_weight=float(center_penalty_weight),
        support_distance_penalty_weight=float(support_distance_penalty_weight),
        score_threshold=float(score_threshold),
        support_selection_strategy=str(support_selection_strategy),
    )
    diagnostic_rows: list[dict[str, object]] = []
    positive_metric_rows: list[Mapping[str, object]] = []
    baseline_values: list[float] = []
    fallback_values: list[float] = []
    applied_values: list[float | None] = []
    oracle_distances: list[float | None] = []
    reason_counts: dict[str, int] = {}
    applied_count = 0
    support_available_count = 0
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
        center = np.asarray([_float(row, "center_x"), _float(row, "center_y")], dtype=np.float64)
        gt = np.asarray([_float(row, "query_gt_x"), _float(row, "query_gt_y")], dtype=np.float64)
        support_xy = np.asarray(
            [
                _float(row, "support_x") if str(row.get("support_x", "")).strip() else _float(row, "render_x"),
                _float(row, "support_y") if str(row.get("support_y", "")).strip() else _float(row, "render_y"),
            ],
            dtype=np.float64,
        )
        baseline = float(np.linalg.norm(gt - center))
        query_keypoints = get_keypoints(query_id)
        support_keypoints = get_keypoints(support_id)
        oracle_distance = _nearest_keypoint_distance(gt, query_keypoints)
        support = select_support_superpoint_keypoint(
            track_id=int(float(str(row.get("track_id", "0")).strip() or 0.0)),
            support_image_id=support_id,
            observation_xy=support_xy,
            keypoints=support_keypoints,
            max_distance_px=float(support_radius_px),
            min_score=float(min_support_score),
            strategy=str(support_selection_strategy),
        )
        teacher_pred: np.ndarray | None = None
        teacher_epe: float | None = None
        teacher_dxdy: np.ndarray | None = None
        teacher_score: float | None = None
        teacher_confidence: float | None = None
        support_delta = np.zeros((2,), dtype=np.float64)
        teacher_reason = "target_dustbin" if target_is_dustbin else "missing_support"
        teacher_applied = False
        if support is not None and not target_is_dustbin:
            support_available_count += 1
            support_delta = support.observation_xy - support.xy if support.observation_xy is not None else support_xy - support.xy
            result = select_landmark_conditioned_keypoint(center, query_keypoints, support, config)
            teacher_reason = result.reason
            teacher_score = float(result.score)
            teacher_confidence = float(result.confidence)
            if result.applied:
                teacher_applied = True
                selected_query_keypoint = center + result.offset_xy.astype(np.float64)
                teacher_pred = selected_query_keypoint + support_delta
                teacher_dxdy = teacher_pred - center
                teacher_epe = float(np.linalg.norm(teacher_pred - gt))
                applied_count += 1
        fallback_epe = teacher_epe if teacher_epe is not None else baseline
        if not target_is_dustbin:
            positive_metric_rows.append(row)
            baseline_values.append(baseline)
            fallback_values.append(float(fallback_epe))
            applied_values.append(teacher_epe)
            oracle_distances.append(oracle_distance)
        diagnostic_rows.append(
            {
                "row_index": int(row_index),
                "query_id": query_id,
                "support_image_id": support_id,
                "track_id": str(row.get("track_id", "")),
                "support_track_id": str(row.get("support_track_id", "")),
                "target_is_dustbin": str(target_is_dustbin),
                "baseline_epe_px": baseline,
                "teacher_epe_px": "" if teacher_epe is None else teacher_epe,
                "fallback_epe_px": fallback_epe,
                "teacher_improved": bool(teacher_epe is not None and teacher_epe < baseline),
                "teacher_applied": bool(teacher_applied),
                "teacher_reason": teacher_reason,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "query_gt_x": float(gt[0]),
                "query_gt_y": float(gt[1]),
                "teacher_pred_x": "" if teacher_pred is None else float(teacher_pred[0]),
                "teacher_pred_y": "" if teacher_pred is None else float(teacher_pred[1]),
                "teacher_dx": "" if teacher_dxdy is None else float(teacher_dxdy[0]),
                "teacher_dy": "" if teacher_dxdy is None else float(teacher_dxdy[1]),
                "support_x": float(support_xy[0]),
                "support_y": float(support_xy[1]),
                "support_keypoint_x": "" if support is None else float(support.xy[0]),
                "support_keypoint_y": "" if support is None else float(support.xy[1]),
                "support_delta_x": float(support_delta[0]),
                "support_delta_y": float(support_delta[1]),
                "support_keypoint_score": "" if support is None else float(support.score),
                "support_keypoint_distance_px": "" if support is None else float(support.distance_to_observation_px),
                "teacher_score": "" if teacher_score is None else teacher_score,
                "teacher_confidence": "" if teacher_confidence is None else teacher_confidence,
                "query_keypoint_count": int(np.asarray(query_keypoints.xy).shape[0]),
                "support_keypoint_count": int(np.asarray(support_keypoints.xy).shape[0]),
                "oracle_query_sp_distance_px": "" if oracle_distance is None else float(oracle_distance),
                "requested_residual_px": str(row.get("requested_residual_px", "")),
            }
        )
        reason_counts[str(teacher_reason)] = reason_counts.get(str(teacher_reason), 0) + 1
    output = Path(output_dir)
    _write_csv(output / "teacher_rows.csv", diagnostic_rows)
    baseline_arr = np.asarray(baseline_values, dtype=np.float64)
    fallback_arr = np.asarray(fallback_values, dtype=np.float64)
    applied_arr = _finite_array(applied_values)
    oracle_arr = _finite_array(oracle_distances)
    metrics: dict[str, float | int | None] = {
        "baseline_median_px": _median(baseline_arr),
        "baseline_p90_px": _p90(baseline_arr),
        "teacher_fallback_improve_ratio": (
            None if fallback_arr.size == 0 else float(np.mean(fallback_arr < baseline_arr))
        ),
        "teacher_applied_improve_ratio": (
            None if applied_arr.size == 0 else float(np.mean(applied_arr < np.asarray([v for v, e in zip(baseline_values, applied_values) if e is not None], dtype=np.float64)))
        ),
        "oracle_query_sp_availability_at_0p5px": _recall(oracle_arr, 0.5),
        "oracle_query_sp_availability_at_1px": _recall(oracle_arr, 1.0),
        "oracle_query_sp_availability_at_2px": _recall(oracle_arr, 2.0),
        "oracle_query_sp_availability_at_5px": _recall(oracle_arr, 5.0),
        "oracle_query_sp_distance_median_px": _median(oracle_arr),
    }
    metrics.update(_metric_block("teacher_applied", applied_arr))
    metrics.update(_metric_block("teacher_fallback", fallback_arr))
    metrics.update(_residual_bin_metric_block(positive_metric_rows, applied_values, baseline_values, prefix="teacher_applied"))
    metrics.update(_residual_bin_metric_block(positive_metric_rows, fallback_values, baseline_values, prefix="teacher_fallback"))
    summary = {
        "stage": "measurement_v1_superpoint_teacher_diagnostics",
        "rows_csv": str(rows_csv),
        "row_count": int(len(rows)),
        "positive_count": int(len(positive_metric_rows)),
        "dustbin_count": int(dustbin_count),
        "applied_count": int(applied_count),
        "applied_ratio": float(applied_count / max(len(positive_metric_rows), 1)),
        "support_available_count": int(support_available_count),
        "support_available_ratio": float(support_available_count / max(len(positive_metric_rows), 1)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "config": {
            "keypoint_cache_dir": "" if cache_dir is None else str(cache_dir),
            "candidate_radius_px": float(candidate_radius_px),
            "support_radius_px": float(support_radius_px),
            "min_query_score": float(min_query_score),
            "min_support_score": float(min_support_score),
            "descriptor_weight": float(descriptor_weight),
            "query_score_weight": float(query_score_weight),
            "center_penalty_weight": float(center_penalty_weight),
            "support_distance_penalty_weight": float(support_distance_penalty_weight),
            "score_threshold": float(score_threshold),
            "support_selection_strategy": str(support_selection_strategy),
        },
        "metrics": metrics,
        "outputs": {
            "teacher_rows": str(output / "teacher_rows.csv"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
