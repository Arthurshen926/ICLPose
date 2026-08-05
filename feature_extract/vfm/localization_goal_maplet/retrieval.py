"""Calibrated canonical-field retrieval for Goal-Maplet."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.optimize import minimize


CALIBRATION_SCHEMA = "goal_maplet_validity_calibration_v1"


def retrieve_maplet_posterior(
    descriptor: np.ndarray,
    map_descriptor: np.ndarray,
    maplet_ids: np.ndarray,
    valid_maplets: np.ndarray,
    *,
    maximum_candidates: int,
    temperature: float,
    null_similarity_center: float,
    null_similarity_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = descriptor / np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-8)
    score = query @ map_descriptor.T
    score[:, ~valid_maplets] = -np.inf
    count = min(int(maximum_candidates), int(np.sum(valid_maplets)))
    columns = np.argpartition(-score, kth=count - 1, axis=1)[:, :count]
    order = np.argsort(-np.take_along_axis(score, columns, axis=1), axis=1, kind="stable")
    columns = np.take_along_axis(columns, order, axis=1)
    selected_score = np.take_along_axis(score, columns, axis=1)
    maximum = np.max(score, axis=1, keepdims=True)
    exponential = np.exp((score - maximum) / max(float(temperature), 1e-4))
    exponential[:, ~valid_maplets] = 0.0
    normalizer = np.sum(exponential, axis=1, keepdims=True)
    conditional = np.take_along_axis(exponential / np.maximum(normalizer, 1e-12), columns, axis=1)
    best = selected_score[:, 0]
    predicted_valid = 1.0 / (
        1.0 + np.exp(-(best - float(null_similarity_center)) / max(float(null_similarity_scale), 1e-4))
    )
    probability = predicted_valid[:, None] * conditional
    null = np.clip(1.0 - np.sum(probability, axis=1), 0.0, 1.0)
    return maplet_ids[columns], probability, null, best


@dataclass(frozen=True)
class ValidityCalibration:
    center: float
    scale: float
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not np.isfinite(self.center) or not np.isfinite(self.scale) or float(self.scale) <= 0.0:
            raise ValueError("invalid validity calibration")
        metadata = dict(self.metadata)
        if metadata.get("artifact_type", CALIBRATION_SCHEMA) != CALIBRATION_SCHEMA:
            raise ValueError("not a Goal-Maplet validity calibration")
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            {"center": float(self.center), "scale": float(self.scale), "metadata": dict(self.metadata)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf8")
        return hashlib.sha256(payload).hexdigest()

    def predict_valid(self, similarity: np.ndarray) -> np.ndarray:
        value = (np.asarray(similarity, dtype=np.float64) - float(self.center)) / float(self.scale)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))

    def save_json(self, path: Path) -> None:
        payload = {
            "artifact_type": CALIBRATION_SCHEMA,
            "center": float(self.center),
            "scale": float(self.scale),
            "metadata": dict(self.metadata),
            "content_sha256": self.content_sha256,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_json(cls, path: Path) -> "ValidityCalibration":
        payload = json.loads(Path(path).read_text())
        result = cls(float(payload["center"]), float(payload["scale"]), payload["metadata"])
        if str(payload.get("content_sha256", "")) != result.content_sha256:
            raise ValueError("validity calibration content hash mismatch")
        return result


def fit_validity_calibration(
    similarity: np.ndarray,
    valid_target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ValidityCalibration:
    score = np.asarray(similarity, dtype=np.float64).reshape(-1)
    target = np.asarray(valid_target, dtype=np.float64).reshape(-1)
    weight = np.ones_like(score) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if score.shape != target.shape or weight.shape != score.shape or score.size == 0:
        raise ValueError("calibration arrays differ or are empty")
    if np.any(~np.isfinite(score)) or np.any((target < 0.0) | (target > 1.0)) or np.any(weight <= 0.0):
        raise ValueError("invalid calibration samples")
    weight = weight / np.sum(weight)

    def objective(parameters: np.ndarray) -> float:
        center, log_scale = float(parameters[0]), float(parameters[1])
        scale = float(np.exp(log_scale))
        probability = 1.0 / (1.0 + np.exp(-np.clip((score - center) / scale, -50.0, 50.0)))
        loss = -target * np.log(np.maximum(probability, 1e-8))
        loss -= (1.0 - target) * np.log(np.maximum(1.0 - probability, 1e-8))
        return float(np.sum(weight * loss))

    initial_center = float(np.median(score))
    result = minimize(
        objective,
        np.asarray([initial_center, np.log(0.08)], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((-1.0, 1.0), (np.log(0.005), np.log(1.0))),
    )
    if not result.success:
        raise RuntimeError(f"validity calibration failed: {result.message}")
    return ValidityCalibration(
        center=float(result.x[0]),
        scale=float(np.exp(result.x[1])),
        metadata={"artifact_type": CALIBRATION_SCHEMA, **dict(metadata or {})},
    )
