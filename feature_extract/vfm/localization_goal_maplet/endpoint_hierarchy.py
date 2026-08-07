"""Properly calibrated probabilities for the Goal-Maplet endpoint hierarchy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.optimize import minimize, minimize_scalar


SCHEMA = "goal_maplet_endpoint_hierarchy_calibration_v1"


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(value) - np.log1p(-value)


def _power_distribution(probability: np.ndarray, temperature: float) -> np.ndarray:
    value = np.maximum(np.asarray(probability, dtype=np.float64), 0.0)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("categorical probability must be a non-empty vector")
    total = float(np.sum(value))
    if total <= 1e-15:
        output = np.zeros_like(value)
        output[-1] = 1.0
        return output
    value /= total
    log_value = np.log(np.maximum(value, 1e-12)) / max(float(temperature), 1e-4)
    log_value -= float(np.max(log_value))
    output = np.exp(log_value)
    return output / max(float(np.sum(output)), 1e-15)


def conditional_with_tail(probability: np.ndarray, root_mass: float) -> np.ndarray:
    """Convert sparse unconditional masses into a conditional + tail vector."""

    value = np.maximum(np.asarray(probability, dtype=np.float64).reshape(-1), 0.0)
    root = max(float(root_mass), 1e-12)
    conditional = value / root
    if float(np.sum(conditional)) > 1.0:
        conditional /= float(np.sum(conditional))
    tail = max(1.0 - float(np.sum(conditional)), 0.0)
    output = np.concatenate([conditional, np.asarray([tail])])
    return output / max(float(np.sum(output)), 1e-15)


@dataclass(frozen=True)
class EndpointHierarchyCalibration:
    support_logit_scale: float
    support_logit_bias: float
    parent_temperature: float
    child_temperature: float
    mode_temperature: float
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray([
            self.support_logit_scale, self.support_logit_bias,
            self.parent_temperature, self.child_temperature, self.mode_temperature,
        ], dtype=np.float64)
        if (
            np.any(~np.isfinite(values)) or float(self.support_logit_scale) <= 0.0
            or np.any(values[2:] <= 0.0)
        ):
            raise ValueError("invalid endpoint hierarchy calibration")
        metadata = dict(self.metadata)
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not an endpoint hierarchy calibration")
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(self.to_payload(include_hash=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf8")).hexdigest()

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "artifact_type": SCHEMA,
            "support_logit_scale": float(self.support_logit_scale),
            "support_logit_bias": float(self.support_logit_bias),
            "parent_temperature": float(self.parent_temperature),
            "child_temperature": float(self.child_temperature),
            "mode_temperature": float(self.mode_temperature),
            "metadata": dict(self.metadata),
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def save_json(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_json(cls, path: Path) -> "EndpointHierarchyCalibration":
        payload = json.loads(Path(path).read_text())
        result = cls(
            float(payload["support_logit_scale"]), float(payload["support_logit_bias"]),
            float(payload["parent_temperature"]), float(payload["child_temperature"]),
            float(payload["mode_temperature"]), payload["metadata"],
        )
        if str(payload.get("content_sha256", "")) != result.content_sha256:
            raise ValueError("endpoint hierarchy calibration content hash mismatch")
        return result

    def calibrate_support(self, probability: np.ndarray) -> np.ndarray:
        score = float(self.support_logit_scale) * _logit(probability) + float(self.support_logit_bias)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))

    def calibrate_distribution(self, probability: np.ndarray, *, level: str) -> np.ndarray:
        temperature = {
            "parent": self.parent_temperature,
            "child": self.child_temperature,
            "mode": self.mode_temperature,
        }.get(str(level))
        if temperature is None:
            raise ValueError("unknown endpoint hierarchy level")
        return _power_distribution(probability, float(temperature))


def fit_endpoint_hierarchy_calibration(
    support_probability: np.ndarray,
    support_target: np.ndarray,
    parent_distributions: list[np.ndarray],
    parent_targets: np.ndarray,
    child_distributions: list[np.ndarray],
    child_targets: np.ndarray,
    mode_distributions: list[np.ndarray],
    mode_targets: np.ndarray,
    *,
    metadata: Mapping[str, object] | None = None,
) -> EndpointHierarchyCalibration:
    """Fit binary and categorical temperatures with proper log scores."""

    support = np.asarray(support_probability, dtype=np.float64).reshape(-1)
    target = np.asarray(support_target, dtype=np.float64).reshape(-1)
    if support.shape != target.shape or support.size == 0 or np.any((target < 0) | (target > 1)):
        raise ValueError("support calibration samples differ")

    def binary_objective(parameters: np.ndarray) -> float:
        scale = float(np.exp(parameters[0]))
        score = scale * _logit(support) + float(parameters[1])
        predicted = 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))
        return float(np.mean(
            -target * np.log(np.maximum(predicted, 1e-12))
            -(1.0 - target) * np.log(np.maximum(1.0 - predicted, 1e-12))
        ))

    binary = minimize(
        binary_objective, np.zeros((2,), dtype=np.float64), method="L-BFGS-B",
        bounds=((np.log(0.05), np.log(20.0)), (-10.0, 10.0)),
    )
    if not binary.success:
        raise RuntimeError(f"support hierarchy calibration failed: {binary.message}")

    def temperature_fit(distributions: list[np.ndarray], targets: np.ndarray) -> float:
        labels = np.asarray(targets, dtype=np.int64).reshape(-1)
        if len(distributions) != labels.size or labels.size == 0:
            raise ValueError("categorical hierarchy calibration samples differ")

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            losses = []
            for probability, label in zip(distributions, labels.tolist()):
                calibrated = _power_distribution(probability, temperature)
                if label < 0 or label >= calibrated.size:
                    raise ValueError("categorical hierarchy target is invalid")
                losses.append(-np.log(max(float(calibrated[label]), 1e-12)))
            return float(np.mean(losses))

        result = minimize_scalar(
            objective, bounds=(np.log(0.05), np.log(10.0)), method="bounded",
            options={"xatol": 1e-5},
        )
        if not result.success:
            raise RuntimeError("categorical hierarchy calibration failed")
        return float(np.exp(result.x))

    return EndpointHierarchyCalibration(
        support_logit_scale=float(np.exp(binary.x[0])),
        support_logit_bias=float(binary.x[1]),
        parent_temperature=temperature_fit(parent_distributions, parent_targets),
        child_temperature=temperature_fit(child_distributions, child_targets),
        mode_temperature=temperature_fit(mode_distributions, mode_targets),
        metadata={"artifact_type": SCHEMA, **dict(metadata or {})},
    )

