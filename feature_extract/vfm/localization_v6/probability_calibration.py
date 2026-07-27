"""Frozen null calibration for V6 identity and spatial posteriors.

The calibration consumes only summary statistics of a complete candidate
distribution.  It is deliberately separate from descriptor training so the
trajectory-disjoint calibration split and artifact lineage remain explicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import expit, logsumexp


NULL_FEATURE_NAMES = (
    "bias",
    "best_score",
    "top1_top2_margin",
    "normalized_entropy",
    "log_candidate_count",
)


def null_calibration_features(scores: np.ndarray) -> np.ndarray:
    """Return stable features for one or more candidate-score rows."""

    value = np.asarray(scores, dtype=np.float64)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError("null calibration scores must have shape (N,K), K>=1")
    ordered = np.sort(value, axis=1)
    best = ordered[:, -1]
    second = ordered[:, -2] if value.shape[1] > 1 else ordered[:, -1]
    probability = np.exp(value - logsumexp(value, axis=1, keepdims=True))
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=1
    )
    entropy /= max(float(np.log(max(value.shape[1], 2))), 1e-8)
    return np.stack(
        [
            np.ones(value.shape[0], dtype=np.float64),
            best,
            best - second,
            entropy,
            np.full(
                value.shape[0],
                np.log(float(value.shape[1])),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )


@dataclass(frozen=True)
class DistributionNullCalibration:
    """Logistic probability that a candidate distribution is unresolved."""

    weights: np.ndarray
    minimum_probability: float = 1e-4
    maximum_probability: float = 1.0 - 1e-4

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if weights.shape != (len(NULL_FEATURE_NAMES),):
            raise ValueError("null calibration weight dimension differs")
        if not np.all(np.isfinite(weights)):
            raise ValueError("null calibration weights must be finite")
        lower = float(self.minimum_probability)
        upper = float(self.maximum_probability)
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("invalid null probability bounds")
        object.__setattr__(self, "weights", weights)

    def probability(self, scores: np.ndarray) -> np.ndarray:
        features = null_calibration_features(scores)
        probability = expit(features @ self.weights)
        return np.clip(
            probability,
            float(self.minimum_probability),
            float(self.maximum_probability),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "feature_names": list(NULL_FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "minimum_probability": float(self.minimum_probability),
            "maximum_probability": float(self.maximum_probability),
        }

    @classmethod
    def from_json(
        cls, value: Mapping[str, object]
    ) -> "DistributionNullCalibration":
        names = tuple(str(item) for item in value["feature_names"])
        if names != NULL_FEATURE_NAMES:
            raise ValueError("null calibration feature contract differs")
        return cls(
            weights=np.asarray(value["weights"], dtype=np.float64),
            minimum_probability=float(value["minimum_probability"]),
            maximum_probability=float(value["maximum_probability"]),
        )


@dataclass(frozen=True)
class V6ProbabilityCalibration:
    """Identity-null and conditional spatial-null calibration artifact."""

    identity: DistributionNullCalibration
    spatial: DistributionNullCalibration
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        if metadata.get("artifact_type") != "v6_probability_calibration":
            raise ValueError("invalid V6 probability calibration artifact")
        if not metadata.get("calibration_trajectory_ids"):
            raise ValueError("calibration trajectories must be recorded")
        if metadata.get("strict_holdout_trajectory_ids") is None:
            raise ValueError("strict holdout trajectories must be recorded")
        if (
            metadata.get("spatial_calibration_condition")
            != "identity_is_true_and_atlas_available"
        ):
            raise ValueError(
                "spatial null must be calibrated conditionally on a true "
                "identity with an available atlas"
            )
        if (
            metadata.get("calibration_objective")
            != "unweighted_bernoulli_nll"
        ):
            raise ValueError(
                "null probabilities require a proper unweighted scoring rule"
            )
        overlap = set(metadata["calibration_trajectory_ids"]) & set(
            metadata["strict_holdout_trajectory_ids"]
        )
        if overlap:
            raise ValueError(
                f"strict holdout leaked into probability calibration: {overlap}"
            )
        object.__setattr__(self, "metadata", metadata)

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_type": "v6_probability_calibration",
            "identity": self.identity.to_json(),
            "spatial": self.spatial.to_json(),
            "metadata": dict(self.metadata),
        }

    def save_json(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load_json(cls, path: Path) -> "V6ProbabilityCalibration":
        value = json.loads(Path(path).read_text())
        if value.get("artifact_type") != "v6_probability_calibration":
            raise ValueError("invalid V6 probability calibration file")
        return cls(
            identity=DistributionNullCalibration.from_json(value["identity"]),
            spatial=DistributionNullCalibration.from_json(value["spatial"]),
            metadata=value["metadata"],
        )


def probability_calibration_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
