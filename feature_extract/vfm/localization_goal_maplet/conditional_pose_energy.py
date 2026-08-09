"""Low-capacity conditional energy for frozen Goal-Maplet pose candidates.

The model deliberately keeps physical identity, VFM spatial phase and
fractional observation quality as separate measurements.  It normalizes each
measurement only within the frozen candidate set, applies non-negative
weights, and normalizes candidates together with one explicit null state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


COMPONENT_NAMES = (
    "proposal_identity",
    "jacobian_phase_visible",
    "fractional_observation_quality",
)
DEFAULT_SCALE_FLOORS = np.asarray((0.05, 0.02, 0.02), dtype=np.float64)


def fractional_observation_quality(evidence: dict[str, float]) -> float:
    """Return a fixed, physically typed observation-quality measurement.

    Jacobian observability already uses query-only carrier mass and fractional
    feature/purity support.  Missing canonical features and mixed-surface
    boundaries are distinct failure modes and therefore subtract evidence.
    Background is represented by the absent carrier support rather than being
    counted a second time.
    """

    return float(
        float(evidence["jacobian_observability"])
        - float(evidence.get("missing_fraction_mean", 0.0))
        - 0.5 * float(evidence.get("mixed_surface_fraction", 0.0))
    )


def candidate_measurements(
    proposal_identity: np.ndarray,
    phase_evidence: list[dict[str, float]],
) -> np.ndarray:
    identity = np.asarray(proposal_identity, dtype=np.float64)
    if identity.ndim != 1 or len(identity) != len(phase_evidence):
        raise ValueError("candidate identity and phase evidence differ")
    phase = np.asarray(
        [item["jacobian_phase_visible"] for item in phase_evidence], dtype=np.float64,
    )
    observation = np.asarray(
        [fractional_observation_quality(item) for item in phase_evidence],
        dtype=np.float64,
    )
    result = np.stack((identity, phase, observation), axis=1)
    if np.any(~np.isfinite(result)):
        raise ValueError("conditional-energy measurements are not finite")
    return result


def normalize_candidate_measurements(
    measurements: np.ndarray,
    *,
    scale_floors: np.ndarray = DEFAULT_SCALE_FLOORS,
) -> np.ndarray:
    """Robust query-local normalization without changing component ordering."""

    value = np.asarray(measurements, dtype=np.float64)
    floors = np.asarray(scale_floors, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(COMPONENT_NAMES):
        raise ValueError("conditional energy expects [candidate,3] measurements")
    if floors.shape != (len(COMPONENT_NAMES),) or np.any(floors <= 0.0):
        raise ValueError("invalid conditional-energy scale floors")
    if value.shape[0] == 0:
        return value.copy()
    center = np.median(value, axis=0)
    q25, q75 = np.percentile(value, (25.0, 75.0), axis=0)
    robust_scale = (q75 - q25) / 1.349
    scale = np.maximum(robust_scale, floors)
    return (value - center[None]) / scale[None]


@dataclass(frozen=True)
class ConditionalPoseEnergyPolicy:
    weights: np.ndarray
    candidate_bias: float
    scale_floors: np.ndarray
    null_phase_threshold: float
    null_phase_scale: float
    null_phase_slope: float
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        floors = np.asarray(self.scale_floors, dtype=np.float64)
        if weights.shape != (len(COMPONENT_NAMES),) or np.any(~np.isfinite(weights)):
            raise ValueError("invalid conditional-energy weights")
        if np.any(weights < 0.0):
            raise ValueError("conditional-energy component weights must be monotonic")
        if floors.shape != weights.shape or np.any(~np.isfinite(floors)) or np.any(floors <= 0.0):
            raise ValueError("invalid conditional-energy scale floors")
        if not np.isfinite(float(self.candidate_bias)):
            raise ValueError("invalid conditional-energy null offset")
        if not np.isfinite(float(self.null_phase_threshold)):
            raise ValueError("invalid conditional-energy phase-support threshold")
        if not np.isfinite(float(self.null_phase_scale)) or self.null_phase_scale <= 0.0:
            raise ValueError("invalid conditional-energy phase-support scale")
        if not np.isfinite(float(self.null_phase_slope)) or self.null_phase_slope <= 0.0:
            raise ValueError("invalid conditional-energy phase-support slope")

    def energies(self, measurements: np.ndarray) -> tuple[np.ndarray, float]:
        normalized = normalize_candidate_measurements(
            measurements, scale_floors=self.scale_floors,
        )
        candidate = normalized @ np.asarray(self.weights, dtype=np.float64)
        if candidate.size:
            candidate -= float(np.max(candidate))
        phase_peak = float(np.max(np.asarray(measurements, dtype=np.float64)[:, 1]))
        presence = float(self.null_phase_slope) * (
            phase_peak - float(self.null_phase_threshold)
        ) / float(self.null_phase_scale)
        candidate = candidate + presence + float(self.candidate_bias)
        return candidate, 0.0

    def posterior(self, measurements: np.ndarray) -> tuple[np.ndarray, float]:
        candidate, null = self.energies(measurements)
        logits = np.r_[candidate, float(null)]
        maximum = float(np.max(logits))
        probability = np.exp(logits - maximum)
        probability /= max(float(np.sum(probability)), 1.0e-12)
        return probability[:-1], float(probability[-1])


def load_conditional_pose_energy(path: Path) -> ConditionalPoseEnergyPolicy:
    payload = json.loads(Path(path).read_text())
    if payload.get("artifact_type") != "goal_maplet_conditional_pose_energy_v1":
        raise ValueError("not a Goal-Maplet conditional pose energy")
    if tuple(payload.get("component_names", ())) != COMPONENT_NAMES:
        raise ValueError("conditional pose energy has a different measurement contract")
    if payload.get("normalization") != "per_query_median_iqr_with_fixed_floors_v1":
        raise ValueError("conditional pose energy has an unknown normalization")
    if payload.get("null_hypothesis") != "candidate_set_phase_support_null_logit_zero_v2":
        raise ValueError("conditional pose energy lacks the typed null state")
    return ConditionalPoseEnergyPolicy(
        weights=np.asarray(payload.get("weights", ()), dtype=np.float64),
        candidate_bias=float(payload.get("candidate_bias", 0.0)),
        scale_floors=np.asarray(payload.get("scale_floors", ()), dtype=np.float64),
        null_phase_threshold=float(payload.get("null_phase_threshold", np.nan)),
        null_phase_scale=float(payload.get("null_phase_scale", np.nan)),
        null_phase_slope=float(payload.get("null_phase_slope", np.nan)),
        metadata=payload,
    )
