"""Calibrated typed-null evidence for a child-local multi-modal factor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from .child_local_likelihood import ChildLocalSurfaceLikelihood
from .physical_map import GoalMapletPhysicalMap


NULL_TYPES = (
    "valid",
    "wrong_child",
    "pose_incompatible",
    "unresolved",
    "field_missing",
)

FEATURE_NAMES = (
    "maximum_similarity",
    "top1_mode_probability",
    "top1_top2_probability_margin",
    "topm_probability_mass",
    "normalized_mode_entropy",
    "canonical_feature_coverage",
    "coverage_null_probability",
    "minimum_normalized_reprojection",
    "probability_weighted_normalized_reprojection",
    "visible_mode_fraction",
    "maximum_view_normal_incidence",
    "maximum_geometry_log_likelihood",
    "normalized_mode_spread",
    "log_child_primitive_count",
    "retrieval_parent_probability",
    "retrieval_child_probability",
    "retrieval_parent_null_probability",
    "normalized_query_scale",
)


def child_local_factor_runtime_features(
    likelihood: ChildLocalSurfaceLikelihood,
    mode_features: np.ndarray,
    mode_valid: np.ndarray,
    child_rows: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    parent_probability: np.ndarray,
    child_probability: np.ndarray,
    parent_null_probability: np.ndarray,
    query_scale_px: np.ndarray,
    image_diagonal_px: float,
) -> np.ndarray:
    """Summarize one multi-modal child likelihood without absolute position."""

    feature = np.asarray(mode_features, dtype=np.float64)
    valid = np.asarray(mode_valid, dtype=bool)
    children = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    probability = np.asarray(likelihood.mode_probabilities, dtype=np.float64)
    if feature.ndim != 3 or feature.shape[:2] != valid.shape:
        raise ValueError("child-local factor mode arrays differ")
    if probability.shape != valid.shape or children.shape != (valid.shape[0],):
        raise ValueError("child-local factor likelihood arrays differ")
    count, modes = valid.shape
    safe_probability = np.where(valid, probability, 0.0)
    mass = np.sum(safe_probability, axis=1)
    normalized = safe_probability / np.maximum(mass[:, None], 1e-12)
    ordered = np.sort(safe_probability, axis=1)[:, ::-1]
    top1 = ordered[:, 0]
    top2 = ordered[:, 1] if modes > 1 else np.zeros((count,), dtype=np.float64)
    entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)), axis=1)
    valid_count = np.sum(valid, axis=1)
    entropy /= np.maximum(np.log(np.maximum(valid_count, 2)), 1e-12)

    residual = np.where(valid, feature[:, :, 4], np.inf)
    geometry_log = np.where(valid, feature[:, :, 5], -np.inf)
    incidence = np.where(valid, feature[:, :, 7], -np.inf)
    minimum_residual = np.min(residual, axis=1)
    minimum_residual[~np.isfinite(minimum_residual)] = 1e3
    weighted_residual = np.sum(normalized * np.where(valid, feature[:, :, 4], 0.0), axis=1)
    maximum_geometry = np.max(geometry_log, axis=1)
    maximum_geometry[~np.isfinite(maximum_geometry)] = -1e3
    maximum_incidence = np.max(incidence, axis=1)
    maximum_incidence[~np.isfinite(maximum_incidence)] = -1.0

    primitive = np.maximum(np.asarray(likelihood.mode_primitive_rows, dtype=np.int64), 0)
    points = physical.primitive_centers[primitive]
    center = np.sum(normalized[:, :, None] * points, axis=1)
    spread = np.sqrt(np.sum(
        normalized * np.sum(np.square(points - center[:, None]), axis=2), axis=1
    ))
    child_scale = np.maximum(np.linalg.norm(physical.child_extents[children, :2], axis=1), 0.05)
    spread /= child_scale
    member_count = physical.child_member_offsets[children + 1] - physical.child_member_offsets[children]

    output = np.stack([
        np.asarray(likelihood.maximum_similarity, dtype=np.float64),
        top1,
        top1 - top2,
        mass,
        entropy,
        np.asarray(likelihood.feature_coverage, dtype=np.float64),
        np.asarray(likelihood.null_probabilities, dtype=np.float64),
        minimum_residual,
        weighted_residual,
        valid_count / max(modes, 1),
        maximum_incidence,
        maximum_geometry,
        spread,
        np.log1p(member_count),
        np.asarray(parent_probability, dtype=np.float64).reshape(-1),
        np.asarray(child_probability, dtype=np.float64).reshape(-1),
        np.asarray(parent_null_probability, dtype=np.float64).reshape(-1),
        np.asarray(query_scale_px, dtype=np.float64).reshape(-1) / max(float(image_diagonal_px), 1.0),
    ], axis=1)
    if output.shape != (count, len(FEATURE_NAMES)) or not np.all(np.isfinite(output)):
        raise ValueError("child-local factor features are invalid")
    return output.astype(np.float32)


@dataclass(frozen=True)
class ChildLocalFactorCalibratorArtifact:
    estimator: object
    metadata: Mapping[str, object]

    def predict_typed_probabilities(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
            raise ValueError("child-local factor feature dimension differs")
        probability = np.asarray(self.estimator.predict_proba(value), dtype=np.float64)
        classes = np.asarray(self.estimator.classes_, dtype=np.int64)
        output = np.zeros((value.shape[0], len(NULL_TYPES)), dtype=np.float64)
        output[:, classes] = probability
        temperature = max(float(self.metadata.get("calibration_temperature", 1.0)), 1e-4)
        logits = np.log(np.maximum(output, 1e-12)) / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        output = np.exp(np.clip(logits, -60.0, 0.0))
        output /= np.maximum(np.sum(output, axis=1, keepdims=True), 1e-12)
        return output

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "metadata": dict(self.metadata)}, Path(path))

    @classmethod
    def load(cls, path: Path) -> "ChildLocalFactorCalibratorArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        if metadata.get("artifact_type") != "goal_maplet_child_local_factor_calibrator_v2":
            raise ValueError("not a child-local factor calibrator")
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("child-local factor feature contract differs")
        if tuple(metadata.get("null_types", ())) != NULL_TYPES:
            raise ValueError("child-local factor null contract differs")
        return cls(payload["estimator"], metadata)
