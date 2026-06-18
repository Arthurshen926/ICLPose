"""Oracle coverage diagnostics for virtual reference poses."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    _world_y_yaw_rotation,
    parse_cambridge_pose_file,
    rotation_angle_deg,
)


def threshold_key(translation_m: float, rotation_deg: float) -> str:
    t_cm = int(round(float(translation_m) * 100.0))
    r_deg = int(round(float(rotation_deg)))
    return f"{t_cm}cm_{r_deg}deg"


def parse_thresholds(text: str) -> tuple[tuple[float, float], ...]:
    thresholds: list[tuple[float, float]] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(value.strip()) for value in item.split(",") if value.strip()]
        if len(parts) != 2:
            raise ValueError("thresholds must be formatted as translation_m,rotation_deg pairs")
        thresholds.append((float(parts[0]), float(parts[1])))
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return tuple(thresholds)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("at least one float value is required")
    return values


def _summary_from_best(
    best_translation: Sequence[float],
    best_rotation: Sequence[float],
    thresholds: Sequence[tuple[float, float]],
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    t = np.asarray(best_translation, dtype=np.float64)
    r = np.asarray(best_rotation, dtype=np.float64)
    summary: dict[str, object] = {
        "query_count": int(t.shape[0]),
        "min_translation_m_median": float(np.median(t)) if t.size else None,
        "min_rotation_deg_median": float(np.median(r)) if r.size else None,
        "min_translation_m_p90": float(np.percentile(t, 90)) if t.size else None,
        "min_rotation_deg_p90": float(np.percentile(r, 90)) if r.size else None,
    }
    summary.update(dict(extra or {}))
    for translation_m, rotation_deg in thresholds:
        key = threshold_key(translation_m, rotation_deg)
        summary[f"exist_recall_{key}"] = float(np.mean((t <= float(translation_m)) & (r <= float(rotation_deg))))
    return summary


def summarize_pose_file_oracle_coverage(
    query_pose_file: Path,
    reference_pose_file: Path,
    thresholds: Sequence[tuple[float, float]] = ((0.25, 5.0),),
) -> dict[str, object]:
    queries = parse_cambridge_pose_file(Path(query_pose_file))
    references = parse_cambridge_pose_file(Path(reference_pose_file))
    reference_centers = np.stack([record.camera_center for record in references], axis=0).astype(np.float64)
    reference_rotations = np.stack([record.rotation_w2c for record in references], axis=0).astype(np.float64)
    best_t: list[float] = []
    best_r: list[float] = []
    for query in queries:
        translations = np.linalg.norm(reference_centers - query.camera_center[None, :], axis=1)
        cos_angles = (np.einsum("ij,nij->n", query.rotation_w2c, reference_rotations) - 1.0) * 0.5
        rotations = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
        joint = translations / max(float(min(t for t, _r in thresholds)), 1e-6) + rotations / max(
            float(min(r for _t, r in thresholds)),
            1e-6,
        )
        idx = int(np.argmin(joint))
        best_t.append(float(translations[idx]))
        best_r.append(float(rotations[idx]))
    return _summary_from_best(
        best_t,
        best_r,
        thresholds,
        extra={
            "reference_count": int(len(references)),
            "coverage_mode": "pose_file",
        },
    )


def _inclusive_axis_values(min_value: float, max_value: float, step: float) -> np.ndarray:
    if float(step) <= 0.0:
        raise ValueError("grid_step_m must be positive")
    count = int(np.floor((float(max_value) - float(min_value)) / float(step) + 1e-9)) + 1
    values = float(min_value) + np.arange(max(count, 1), dtype=np.float64) * float(step)
    if values[-1] < float(max_value) - max(abs(float(step)) * 1e-6, 1e-9):
        values = np.concatenate([values, np.asarray([float(max_value)], dtype=np.float64)])
    return values


def _idw_height(
    reference_centers: np.ndarray,
    reference_xz: np.ndarray,
    xz: np.ndarray,
    *,
    mode: str,
    height_knn: int,
) -> tuple[float, np.ndarray]:
    dists2 = np.sum((reference_xz - xz[None, :]) ** 2, axis=1)
    order = np.argsort(dists2)
    if mode == "nearest":
        return float(reference_centers[int(order[0]), 1]), order
    if mode != "idw":
        raise ValueError("height_mode must be one of: nearest, idw")
    keep = order[: min(int(height_knn), int(order.shape[0]))]
    if float(dists2[int(keep[0])]) <= 1e-12:
        return float(reference_centers[int(keep[0]), 1]), order
    weights = 1.0 / np.maximum(dists2[keep], 1e-12)
    return float(np.sum(weights * reference_centers[keep, 1]) / np.sum(weights)), order


def summarize_virtual_grid_oracle_coverage(
    reference_pose_file: Path,
    query_pose_file: Path,
    grid_step_m: float,
    yaw_offsets_deg: Sequence[float] = (0.0,),
    height_mode: str = "nearest",
    height_knn: int = 4,
    height_offsets_m: Sequence[float] = (0.0,),
    orientation_knn: int = 1,
    grid_margin_m: float = 0.0,
    thresholds: Sequence[tuple[float, float]] = ((0.25, 5.0),),
) -> dict[str, object]:
    references = parse_cambridge_pose_file(Path(reference_pose_file))
    queries = parse_cambridge_pose_file(Path(query_pose_file))
    if int(height_knn) <= 0:
        raise ValueError("height_knn must be positive")
    if int(orientation_knn) <= 0:
        raise ValueError("orientation_knn must be positive")
    parsed_yaw = tuple(float(value) for value in yaw_offsets_deg)
    parsed_height_offsets = tuple(float(value) for value in height_offsets_m)
    if not parsed_yaw:
        raise ValueError("at least one yaw offset is required")
    if not parsed_height_offsets:
        raise ValueError("at least one height offset is required")
    reference_centers = np.stack([record.camera_center for record in references], axis=0).astype(np.float64)
    reference_rotations = np.stack([record.rotation_w2c for record in references], axis=0).astype(np.float64)
    reference_xz = reference_centers[:, [0, 2]]
    margin = float(grid_margin_m)
    x_values = _inclusive_axis_values(
        float(np.min(reference_centers[:, 0]) - margin),
        float(np.max(reference_centers[:, 0]) + margin),
        float(grid_step_m),
    )
    z_values = _inclusive_axis_values(
        float(np.min(reference_centers[:, 2]) - margin),
        float(np.max(reference_centers[:, 2]) + margin),
        float(grid_step_m),
    )
    min_t_threshold = max(float(min(t for t, _r in thresholds)), 1e-6)
    min_r_threshold = max(float(min(r for _t, r in thresholds)), 1e-6)
    max_t_threshold = float(max(t for t, _r in thresholds))
    yaw_rotations = tuple(_world_y_yaw_rotation(yaw_deg) for yaw_deg in parsed_yaw)
    best_t: list[float] = []
    best_r: list[float] = []
    for query in queries:
        candidate_t: list[float] = []
        candidate_r: list[float] = []
        local_x = x_values[(x_values >= query.camera_center[0] - max_t_threshold) & (x_values <= query.camera_center[0] + max_t_threshold)]
        local_z = z_values[(z_values >= query.camera_center[2] - max_t_threshold) & (z_values <= query.camera_center[2] + max_t_threshold)]
        if local_x.size == 0 or local_z.size == 0:
            best_t.append(float("inf"))
            best_r.append(float("inf"))
            continue
        for x_value in local_x:
            for z_value in local_z:
                xz = np.asarray([float(x_value), float(z_value)], dtype=np.float64)
                base_height, order = _idw_height(
                    reference_centers,
                    reference_xz,
                    xz,
                    mode=str(height_mode),
                    height_knn=int(height_knn),
                )
                orientation_order = order[: min(int(orientation_knn), len(references))]
                for height_offset in parsed_height_offsets:
                    center = np.asarray([float(x_value), base_height + float(height_offset), float(z_value)], dtype=np.float64)
                    translation = float(np.linalg.norm(center - query.camera_center))
                    if translation > max_t_threshold:
                        continue
                    for orientation_idx in orientation_order:
                        base_rotation = reference_rotations[int(orientation_idx)]
                        for yaw_rotation in yaw_rotations:
                            rotation = base_rotation @ yaw_rotation.T
                            candidate_t.append(translation)
                            candidate_r.append(rotation_angle_deg(query.rotation_w2c, rotation))
        if not candidate_t:
            best_t.append(float("inf"))
            best_r.append(float("inf"))
            continue
        t = np.asarray(candidate_t, dtype=np.float64)
        r = np.asarray(candidate_r, dtype=np.float64)
        joint = t / min_t_threshold + r / min_r_threshold
        idx = int(np.argmin(joint))
        best_t.append(float(t[idx]))
        best_r.append(float(r[idx]))
    finite_t = [value for value in best_t if np.isfinite(value)]
    finite_r = [value for value in best_r if np.isfinite(value)]
    summary = _summary_from_best(
        best_t,
        best_r,
        thresholds,
        extra={
            "coverage_mode": "virtual_grid",
            "reference_count": int(len(references)),
            "grid_step_m": float(grid_step_m),
            "grid_margin_m": float(grid_margin_m),
            "height_mode": str(height_mode),
            "height_knn": int(height_knn),
            "height_offsets_m": [float(value) for value in parsed_height_offsets],
            "orientation_knn": int(orientation_knn),
            "yaw_offsets_deg": [float(value) for value in parsed_yaw],
            "candidate_available_query_count": int(len(finite_t)),
            "best_joint_translation_m_median": float(np.median(finite_t)) if finite_t else None,
            "best_joint_rotation_deg_median": float(np.median(finite_r)) if finite_r else None,
        },
    )
    return summary
