"""Fit measurement confidence and uncertainty calibration from real residuals."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


@dataclass(frozen=True)
class MeasurementCalibrationSample:
    query_id: str
    measured_x: float
    measured_y: float
    gt_x: float
    gt_y: float
    confidence: float
    uncertainty_px: float
    residual_px: float


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _parse_xyz(value: object) -> np.ndarray | None:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    arr = np.asarray(parsed, dtype=np.float64).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def calibration_rows_from_matches(
    rows: Sequence[Mapping[str, object]],
    *,
    cameras_by_query: Mapping[str, ColmapCamera],
    pose_w2c_by_query: Mapping[str, np.ndarray],
) -> list[MeasurementCalibrationSample]:
    samples: list[MeasurementCalibrationSample] = []
    for row in rows:
        query_id = str(row.get("query_id", ""))
        camera = cameras_by_query.get(query_id)
        pose = pose_w2c_by_query.get(query_id)
        if camera is None or pose is None:
            continue
        measured_x = _finite_float(row.get("x"))
        measured_y = _finite_float(row.get("y"))
        confidence = _finite_float(row.get("patch_offset_confidence"))
        uncertainty = _finite_float(row.get("measurement_sigma_px"))
        xyz = _parse_xyz(row.get("xyz"))
        if measured_x is None or measured_y is None or confidence is None or uncertainty is None or xyz is None:
            continue
        gt_xy = project_world_to_image(xyz[None, :], np.asarray(pose, dtype=np.float64).reshape(4, 4), camera)[0]
        residual = float(np.linalg.norm(np.asarray([measured_x, measured_y], dtype=np.float64) - gt_xy))
        samples.append(
            MeasurementCalibrationSample(
                query_id=query_id,
                measured_x=float(measured_x),
                measured_y=float(measured_y),
                gt_x=float(gt_xy[0]),
                gt_y=float(gt_xy[1]),
                confidence=float(np.clip(confidence, 1e-6, 1.0 - 1e-6)),
                uncertainty_px=float(max(uncertainty, 1e-6)),
                residual_px=residual,
            )
        )
    return samples


def _binary_cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(target * np.log(probs) + (1.0 - target) * np.log(1.0 - probs)))


def fit_confidence_temperature_bias(
    samples: Sequence[MeasurementCalibrationSample],
    *,
    inlier_threshold_px: float = 2.0,
    temperatures: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    biases: Sequence[float] = tuple(np.linspace(-12.0, 6.0, 73)),
) -> dict[str, float | int]:
    if not samples:
        return {"sample_count": 0, "confidence_temperature": 1.0, "confidence_bias": 0.0, "bce_before": float("nan"), "bce_after": float("nan")}
    confidences = np.asarray([sample.confidence for sample in samples], dtype=np.float64)
    labels = np.asarray([sample.residual_px <= float(inlier_threshold_px) for sample in samples], dtype=np.float64)
    logits = np.log(np.clip(confidences, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - confidences, 1e-6, 1.0))
    before = _binary_cross_entropy(confidences, labels)
    best = (before, 1.0, 0.0)
    for temperature in temperatures:
        temp = max(float(temperature), 1e-6)
        for bias in biases:
            probs = 1.0 / (1.0 + np.exp(-((logits + float(bias)) / temp)))
            bce = _binary_cross_entropy(probs, labels)
            if bce < best[0]:
                best = (bce, temp, float(bias))
    return {
        "sample_count": int(len(samples)),
        "inlier_threshold_px": float(inlier_threshold_px),
        "confidence_temperature": float(best[1]),
        "confidence_bias": float(best[2]),
        "bce_before": float(before),
        "bce_after": float(best[0]),
    }


def fit_uncertainty_scale_floor(
    samples: Sequence[MeasurementCalibrationSample],
    *,
    floors: Sequence[float] = tuple(np.linspace(0.0, 4.0, 17)),
    max_residual_px: float | None = None,
) -> dict[str, float | int]:
    values = list(samples)
    if max_residual_px is not None:
        values = [sample for sample in values if sample.residual_px <= float(max_residual_px)]
    if not values:
        return {
            "sample_count": 0,
            "excluded_sample_count": int(len(samples)),
            "max_residual_px": None if max_residual_px is None else float(max_residual_px),
            "uncertainty_scale": 1.0,
            "uncertainty_floor_px": 0.0,
            "mae_before": float("nan"),
            "mae_after": float("nan"),
        }
    sigma = np.asarray([sample.uncertainty_px for sample in values], dtype=np.float64)
    residual = np.asarray([sample.residual_px for sample in values], dtype=np.float64)
    before = float(np.mean(np.abs(sigma - residual)))
    best = (before, 1.0, 0.0)
    for floor in floors:
        adjusted = np.maximum(residual - float(floor), 0.0)
        denom = max(float(np.dot(sigma, sigma)), 1e-12)
        scale = max(float(np.dot(sigma, adjusted) / denom), 1e-6)
        predicted = scale * sigma + float(floor)
        mae = float(np.mean(np.abs(predicted - residual)))
        if mae < best[0]:
            best = (mae, scale, float(floor))
    return {
        "sample_count": int(len(values)),
        "excluded_sample_count": int(len(samples) - len(values)),
        "max_residual_px": None if max_residual_px is None else float(max_residual_px),
        "uncertainty_scale": float(best[1]),
        "uncertainty_floor_px": float(best[2]),
        "mae_before": float(before),
        "mae_after": float(best[0]),
        "residual_median_px": float(np.median(residual)),
        "uncertainty_median_px": float(np.median(sigma)),
    }
