"""Fit measurement confidence and uncertainty calibration from real residuals."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
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
    similarity: float | None = None
    similarity_margin: float | None = None
    query_heatmap_score: float | None = None
    observation_count: int | None = None
    landmark_reprojection_error: float | None = None
    local_consistency_score: float | None = None
    coarse_pnp_residual_px: float | None = None


GEOMETRY_PROBABILITY_FEATURES: tuple[str, ...] = (
    "measurement_logit",
    "uncertainty_score",
    "similarity",
    "similarity_margin",
    "query_heatmap_score",
    "observation_quality",
    "landmark_reprojection_quality",
    "coarse_pnp_consistency",
    "coarse_pnp_residual_score",
)


@dataclass(frozen=True)
class GeometryProbabilityModel:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    inlier_threshold_px: float

    def predict_sample(self, sample: MeasurementCalibrationSample) -> float:
        return self.predict_values(_geometry_feature_values(sample))

    def predict_match(self, match: object) -> float:
        return self.predict_values(_geometry_feature_values(match))

    def predict_values(self, values: Mapping[str, float]) -> float:
        vector = np.asarray([float(values.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        standardized = (vector - mean) / np.maximum(scale, 1e-6)
        logit = float(np.dot(standardized, weights) + float(self.bias))
        return _sigmoid(logit)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "measurement_geometry_probability_v1",
            "feature_names": list(self.feature_names),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "weights": [float(value) for value in self.weights],
            "bias": float(self.bias),
            "inlier_threshold_px": float(self.inlier_threshold_px),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GeometryProbabilityModel":
        if payload.get("format") not in {None, "measurement_geometry_probability_v1"}:
            raise ValueError("unsupported geometry probability model format")
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            weights=tuple(float(value) for value in payload["weights"]),
            bias=float(payload["bias"]),
            inlier_threshold_px=float(payload.get("inlier_threshold_px", 4.0)),
        )


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _optional_int(value: object) -> int | None:
    out = _finite_float(value)
    if out is None:
        return None
    return int(out)


def _clip01(value: float | None, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(np.clip(float(value), 0.0, 1.0))


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(value / (1.0 - value)))


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    out = np.empty_like(arr)
    positive = arr >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_value = np.exp(arr[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    if np.isscalar(value):
        return float(out.reshape(-1)[0])
    return out


def _attribute_or_none(source: object, name: str) -> float | None:
    if isinstance(source, Mapping):
        return _finite_float(source.get(name))
    return _finite_float(getattr(source, name, None))


def _geometry_feature_values(source: object) -> dict[str, float]:
    confidence = _attribute_or_none(source, "confidence")
    if confidence is None:
        confidence = _attribute_or_none(source, "patch_offset_confidence")
    uncertainty = _attribute_or_none(source, "uncertainty_px")
    if uncertainty is None:
        uncertainty = _attribute_or_none(source, "measurement_sigma_px")
    observation_count = _attribute_or_none(source, "observation_count")
    reprojection_error = _attribute_or_none(source, "landmark_reprojection_error")
    local_consistency = _attribute_or_none(source, "local_consistency_score")
    coarse_residual = _attribute_or_none(source, "coarse_pnp_residual_px")
    if coarse_residual is None:
        coarse_residual = _attribute_or_none(source, "patch_offset_consistency_before_px")
    sigma = 16.0 if uncertainty is None else max(float(uncertainty), 1e-6)
    obs = 1.0 if observation_count is None else max(float(observation_count), 1.0)
    reproj = 4.0 if reprojection_error is None else max(float(reprojection_error), 0.0)
    coarse = 16.0 if coarse_residual is None else max(float(coarse_residual), 0.0)
    return {
        "measurement_logit": _logit(_clip01(confidence, default=0.5)),
        "uncertainty_score": float(1.0 / (1.0 + sigma / 4.0)),
        "similarity": _clip01(_attribute_or_none(source, "similarity"), default=0.5),
        "similarity_margin": _clip01(_attribute_or_none(source, "similarity_margin"), default=0.0),
        "query_heatmap_score": _clip01(_attribute_or_none(source, "query_heatmap_score"), default=0.5),
        "observation_quality": float(np.clip(np.log1p(obs) / np.log1p(20.0), 0.0, 1.0)),
        "landmark_reprojection_quality": float(1.0 / (1.0 + reproj)),
        "coarse_pnp_consistency": _clip01(local_consistency, default=0.0),
        "coarse_pnp_residual_score": float(1.0 / (1.0 + coarse / 8.0)),
    }


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
                similarity=_finite_float(row.get("similarity")),
                similarity_margin=_finite_float(row.get("similarity_margin")),
                query_heatmap_score=_finite_float(row.get("query_heatmap_score")),
                observation_count=_optional_int(row.get("observation_count")),
                landmark_reprojection_error=_finite_float(row.get("landmark_reprojection_error")),
                local_consistency_score=_finite_float(row.get("local_consistency_score")),
                coarse_pnp_residual_px=_finite_float(row.get("patch_offset_consistency_before_px")),
            )
        )
    return samples


def _binary_cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(target * np.log(probs) + (1.0 - target) * np.log(1.0 - probs)))


def _geometry_design_matrix(samples: Sequence[MeasurementCalibrationSample]) -> np.ndarray:
    return np.asarray(
        [
            [float(_geometry_feature_values(sample)[name]) for name in GEOMETRY_PROBABILITY_FEATURES]
            for sample in samples
        ],
        dtype=np.float64,
    )


def fit_geometry_probability_model(
    samples: Sequence[MeasurementCalibrationSample],
    *,
    inlier_threshold_px: float = 4.0,
    l2: float = 1e-3,
    learning_rate: float = 0.2,
    max_iter: int = 1000,
) -> dict[str, object]:
    values = list(samples)
    if not values:
        model = GeometryProbabilityModel(
            feature_names=GEOMETRY_PROBABILITY_FEATURES,
            mean=tuple(0.0 for _ in GEOMETRY_PROBABILITY_FEATURES),
            scale=tuple(1.0 for _ in GEOMETRY_PROBABILITY_FEATURES),
            weights=tuple(0.0 for _ in GEOMETRY_PROBABILITY_FEATURES),
            bias=0.0,
            inlier_threshold_px=float(inlier_threshold_px),
        )
        return {
            "sample_count": 0,
            "inlier_threshold_px": float(inlier_threshold_px),
            "positive_count": 0,
            "bce_before": float("nan"),
            "bce_after": float("nan"),
            "best_f1_threshold": 0.5,
            "best_f1": float("nan"),
            "model": model,
        }
    x = _geometry_design_matrix(values)
    y = np.asarray([sample.residual_px <= float(inlier_threshold_px) for sample in values], dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_std = (x - mean[None, :]) / scale[None, :]
    positive_rate = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    bias = _logit(positive_rate)
    weights = np.zeros((x_std.shape[1],), dtype=np.float64)
    before = _binary_cross_entropy(np.full_like(y, positive_rate), y)
    for _idx in range(max(1, int(max_iter))):
        logits = x_std @ weights + bias
        probs = np.asarray(_sigmoid(logits), dtype=np.float64)
        error = probs - y
        weights -= float(learning_rate) * ((x_std.T @ error) / max(len(values), 1) + float(l2) * weights)
        bias -= float(learning_rate) * float(np.mean(error))
    probs = np.asarray(_sigmoid(x_std @ weights + bias), dtype=np.float64)
    after = _binary_cross_entropy(probs, y)
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.05, 0.95, 19):
        predicted = probs >= float(threshold)
        tp = float(np.count_nonzero(predicted & (y > 0.5)))
        fp = float(np.count_nonzero(predicted & (y <= 0.5)))
        fn = float(np.count_nonzero((~predicted) & (y > 0.5)))
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    model = GeometryProbabilityModel(
        feature_names=GEOMETRY_PROBABILITY_FEATURES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        bias=float(bias),
        inlier_threshold_px=float(inlier_threshold_px),
    )
    return {
        "sample_count": int(len(values)),
        "inlier_threshold_px": float(inlier_threshold_px),
        "positive_count": int(np.count_nonzero(y > 0.5)),
        "bce_before": float(before),
        "bce_after": float(after),
        "best_f1_threshold": float(best_threshold),
        "best_f1": float(best_f1),
        "model": model,
    }


def load_geometry_probability_model(path: str | Path) -> GeometryProbabilityModel:
    payload = json.loads(Path(path).read_text())
    model_payload = payload.get("geometry_probability", payload)
    if isinstance(model_payload, Mapping) and "model" in model_payload:
        model_payload = model_payload["model"]
    if not isinstance(model_payload, Mapping):
        raise ValueError("geometry probability model JSON must contain a model object")
    return GeometryProbabilityModel.from_dict(model_payload)


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
