"""Map-conditioned hypothesis verifier interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class HypothesisEvidence:
    feature_similarity: float
    visibility_fraction: float
    geometry_consistency: float
    uncertainty: float
    candidate_prior: Optional[float] = None


@dataclass(frozen=True)
class VerificationScore:
    score: float
    risk: float
    accepted: bool


class LinearEvidenceVerifier:
    """Small transparent verifier for score tables and smoke tests."""

    def __init__(self, prior_weight: float = 0.1, accept_threshold: float = 0.5):
        self.prior_weight = float(prior_weight)
        self.accept_threshold = float(accept_threshold)

    def score(self, evidence: HypothesisEvidence) -> VerificationScore:
        prior = 0.0 if evidence.candidate_prior is None else float(evidence.candidate_prior)
        raw_score = (
            0.45 * evidence.feature_similarity
            + 0.25 * evidence.visibility_fraction
            + 0.20 * evidence.geometry_consistency
            + self.prior_weight * prior
            - 0.20 * evidence.uncertainty
        )
        score = float(np.clip(raw_score, 0.0, 1.0))
        risk = float(np.clip(1.0 - score + evidence.uncertainty, 0.0, 1.0))
        return VerificationScore(
            score=score,
            risk=risk,
            accepted=score >= self.accept_threshold,
        )
