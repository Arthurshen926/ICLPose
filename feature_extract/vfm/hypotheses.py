"""Candidate hypothesis schema for map-conditioned verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class PoseCost:
    """Pose error measured against the evaluation ground truth."""

    translation_m: float
    rotation_deg: float

    def combined(self, max_translation_m: float, max_rotation_deg: float) -> float:
        if max_translation_m <= 0.0 or max_rotation_deg <= 0.0:
            raise ValueError("normalizers must be positive")
        t = max(0.0, self.translation_m) / max_translation_m
        r = max(0.0, self.rotation_deg) / max_rotation_deg
        return 0.5 * (t + r)

    def in_basin(self, translation_threshold_m: float, rotation_threshold_deg: float) -> bool:
        return (
            self.translation_m <= translation_threshold_m
            and self.rotation_deg <= rotation_threshold_deg
        )


@dataclass(frozen=True)
class CandidateHypothesis:
    """A pose, reference, place, or solver hypothesis to be verified."""

    candidate_id: str
    candidate_type: str
    pose_error: PoseCost
    prior_score: Optional[float] = None
    pose: Optional[object] = None
    reference_image: Optional[str] = None
    submap_id: Optional[str] = None
    solver_success: Optional[bool] = None
    hard_case_type: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def basin_label(self, translation_threshold_m: float, rotation_threshold_deg: float) -> bool:
        return self.pose_error.in_basin(translation_threshold_m, rotation_threshold_deg)


def assign_basin_labels(
    hypotheses: Iterable[CandidateHypothesis],
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> Dict[str, bool]:
    """Return solver-basin labels keyed by candidate id."""

    return {
        h.candidate_id: h.basin_label(translation_threshold_m, rotation_threshold_deg)
        for h in hypotheses
    }
