"""Paired pose-likelihood ratio for fixed child-local VFM factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from .child_local_factor import FEATURE_NAMES


@dataclass(frozen=True)
class PoseLikelihoodRatioArtifact:
    """A decomposable scalar factor learned from within-group contrasts.

    The ranking direction is learned only from ``x_positive - x_negative``
    pairs.  The affine calibration gives that otherwise translation-invariant
    direction a zero point, so independent group scores can be added and an
    explicit null hypothesis can use log-ratio zero.
    """

    pair_estimator: object
    calibration_scale: float
    calibration_intercept: float
    metadata: Mapping[str, object]

    def score_log_likelihood_ratio(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
            raise ValueError("pose-likelihood feature dimension differs")
        raw = np.asarray(self.pair_estimator.decision_function(value), dtype=np.float64).reshape(-1)
        score = float(self.calibration_scale) * raw + float(self.calibration_intercept)
        if not np.all(np.isfinite(score)):
            raise ValueError("pose-likelihood ratio contains non-finite values")
        return score

    def probability_correct(self, features: np.ndarray) -> np.ndarray:
        score = self.score_log_likelihood_ratio(features)
        positive = np.exp(-np.logaddexp(0.0, -score))
        return np.stack([1.0 - positive, positive], axis=1)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pair_estimator": self.pair_estimator,
            "calibration_scale": float(self.calibration_scale),
            "calibration_intercept": float(self.calibration_intercept),
            "metadata": dict(self.metadata),
        }, Path(path))

    @classmethod
    def load(cls, path: Path) -> "PoseLikelihoodRatioArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        if metadata.get("artifact_type") != "goal_maplet_pose_likelihood_ratio_v1":
            raise ValueError("not a Goal-Maplet pose-likelihood ratio")
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("pose-likelihood feature contract differs")
        if metadata.get("pairing_contract") != "same_image_same_query_group_exact_v1":
            raise ValueError("pose-likelihood pairing contract differs")
        if metadata.get("feature_input_contract") != "deployment_query_group_center_v1":
            raise ValueError("pose-likelihood deployment-replay contract differs")
        if int(metadata.get("runtime_maximum_children", -1)) <= 0:
            raise ValueError("pose-likelihood runtime child contract differs")
        return cls(
            pair_estimator=payload["pair_estimator"],
            calibration_scale=float(payload["calibration_scale"]),
            calibration_intercept=float(payload["calibration_intercept"]),
            metadata=metadata,
        )
