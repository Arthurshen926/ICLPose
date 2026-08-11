"""Low-capacity Stage-C likelihood for VFM phase and dense 2DGS geometry.

The map still stores one canonical VFM field.  Dense query geometry is a
regenerable RADIO readout and the map-side depth/normal are rendered directly
from the 2DGS primitives.  All measurements are normalized within the frozen
candidate set so a policy calibrated on one map fold can transfer to another
without depending on absolute descriptor-score scales.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


COMPONENT_NAMES = (
    "proposal_identity",
    "surface_phase_policy_score",
    "fractional_observation_quality",
    "dense_scale_marginalized_geometry",
)
DEFAULT_SCALE_FLOORS = np.asarray((0.05, 0.02, 0.02, 0.02), dtype=np.float64)


def fractional_observation_quality(evidence: dict[str, float]) -> float:
    return float(
        float(evidence["jacobian_observability"])
        - float(evidence.get("missing_fraction_mean", 0.0))
        - 0.5 * float(evidence.get("mixed_surface_fraction", 0.0))
    )


def candidate_measurements(
    proposal_identity: np.ndarray,
    phase_policy_score: np.ndarray,
    phase_evidence: list[dict[str, float]],
    dense_geometry_evidence: list[dict[str, float]],
) -> np.ndarray:
    identity = np.asarray(proposal_identity, dtype=np.float64)
    phase_score = np.asarray(phase_policy_score, dtype=np.float64)
    count = int(identity.size)
    if identity.ndim != 1 or phase_score.shape != identity.shape:
        raise ValueError("joint likelihood identity and phase scores differ")
    if len(phase_evidence) != count or len(dense_geometry_evidence) != count:
        raise ValueError("joint likelihood candidate evidence differs")
    observation = np.asarray(
        [fractional_observation_quality(item) for item in phase_evidence],
        dtype=np.float64,
    )
    geometry = np.asarray(
        [
            0.0 if item.get("score") is None else float(item["score"])
            for item in dense_geometry_evidence
        ],
        dtype=np.float64,
    )
    result = np.stack((identity, phase_score, observation, geometry), axis=1)
    if np.any(~np.isfinite(result)):
        raise ValueError("joint phase-geometry measurements are not finite")
    return result


def normalize_candidate_measurements(
    measurements: np.ndarray,
    *,
    scale_floors: np.ndarray = DEFAULT_SCALE_FLOORS,
) -> np.ndarray:
    value = np.asarray(measurements, dtype=np.float64)
    floors = np.asarray(scale_floors, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(COMPONENT_NAMES):
        raise ValueError("joint likelihood expects [candidate,4] measurements")
    if floors.shape != (len(COMPONENT_NAMES),) or np.any(floors <= 0.0):
        raise ValueError("invalid joint-likelihood scale floors")
    if not value.shape[0]:
        return value.copy()
    center = np.median(value, axis=0)
    q25, q75 = np.percentile(value, (25.0, 75.0), axis=0)
    scale = np.maximum((q75 - q25) / 1.349, floors)
    return (value - center[None]) / scale[None]


@dataclass(frozen=True)
class JointPhaseGeometryLikelihood:
    weights: np.ndarray
    scale_floors: np.ndarray
    metadata: dict[str, object]
    null_logit: float | None = None

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        floors = np.asarray(self.scale_floors, dtype=np.float64)
        if weights.shape != (len(COMPONENT_NAMES),) or np.any(~np.isfinite(weights)):
            raise ValueError("invalid joint-likelihood weights")
        if np.any(weights < 0.0):
            raise ValueError("joint-likelihood evidence weights must be monotonic")
        if floors.shape != weights.shape or np.any(~np.isfinite(floors)) or np.any(floors <= 0.0):
            raise ValueError("invalid joint-likelihood scale floors")
        if self.null_logit is not None and not np.isfinite(float(self.null_logit)):
            raise ValueError("invalid joint-likelihood null logit")

    def raw_logits(self, measurements: np.ndarray) -> np.ndarray:
        normalized = normalize_candidate_measurements(
            measurements, scale_floors=self.scale_floors,
        )
        return normalized @ np.asarray(self.weights, dtype=np.float64)

    def logits(self, measurements: np.ndarray) -> np.ndarray:
        logits = self.raw_logits(measurements)
        if logits.size:
            logits -= float(np.max(logits))
        return logits

    def posterior(self, measurements: np.ndarray) -> np.ndarray:
        logits = self.logits(measurements)
        probability = np.exp(logits)
        return probability / max(float(np.sum(probability)), 1.0e-12)

    def posterior_with_null(self, measurements: np.ndarray) -> tuple[np.ndarray, float]:
        """Return absolute candidate mass and one typed all-candidates-wrong mass."""

        candidate_logits = self.raw_logits(measurements)
        if self.null_logit is None:
            return self.posterior(measurements), 0.0
        logits = np.concatenate(
            (candidate_logits, np.asarray([float(self.null_logit)], dtype=np.float64))
        )
        logits -= float(np.max(logits))
        probability = np.exp(logits)
        probability /= max(float(np.sum(probability)), 1.0e-12)
        return probability[:-1], float(probability[-1])


def load_joint_phase_geometry_likelihood(path: Path) -> JointPhaseGeometryLikelihood:
    payload = json.loads(Path(path).read_text())
    artifact_type = payload.get("artifact_type")
    if artifact_type not in {
        "goal_maplet_joint_phase_geometry_likelihood_v1",
        "goal_maplet_joint_phase_geometry_likelihood_v2",
    }:
        raise ValueError("not a Goal-Maplet joint phase-geometry likelihood")
    if tuple(payload.get("component_names", ())) != COMPONENT_NAMES:
        raise ValueError("joint phase-geometry measurement contract differs")
    if payload.get("normalization") != "per_query_median_iqr_with_fixed_floors_v1":
        raise ValueError("joint phase-geometry normalization differs")
    null_logit = payload.get("null_logit")
    if artifact_type.endswith("_v2") and null_logit is None:
        raise ValueError("v2 joint likelihood requires a typed null logit")
    return JointPhaseGeometryLikelihood(
        weights=np.asarray(payload.get("weights", ()), dtype=np.float64),
        scale_floors=np.asarray(payload.get("scale_floors", ()), dtype=np.float64),
        metadata=payload,
        null_logit=(None if null_logit is None else float(null_logit)),
    )
