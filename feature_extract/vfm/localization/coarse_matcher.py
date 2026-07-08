"""Coarse matching wrappers for real query/reference descriptor maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from feature_extract.vfm.localization.schemas import CoarseProposal
from feature_extract.vfm.matcha_coarse_to_fine import matcha_coarse_dual_softmax_matches


class CoarseMatcher(Protocol):
    """Produce coarse query/reference cell proposals from descriptor maps."""

    def match(
        self,
        query_descriptors: np.ndarray,
        reference_descriptors: np.ndarray,
        *,
        query_image_size: tuple[int, int],
        reference_image_size: tuple[int, int],
    ) -> list[CoarseProposal]:
        """Return coarse proposals in descending confidence order."""


def _proposal_from_match(match) -> CoarseProposal:
    return CoarseProposal(
        query_index=int(match.query_index),
        reference_index=int(match.render_index),
        query_xy=np.asarray(match.query_xy, dtype=np.float32),
        reference_xy=np.asarray(match.render_xy, dtype=np.float32),
        score=float(match.similarity),
        confidence=None if match.dual_softmax_confidence is None else float(match.dual_softmax_confidence),
        rank=None if match.coarse_rank is None else int(match.coarse_rank),
        metadata={
            "similarity_margin": match.similarity_margin,
            "mutual_rank": match.mutual_rank,
            "coarse_score": match.coarse_score,
            "coarse_score_gap": match.coarse_score_gap,
        },
    )


@dataclass(frozen=True)
class MatchaCoarseMatcher:
    """MATCHA-style coarse dual-softmax matcher without local fine refinement."""

    logit_scale: float = 10.0
    min_confidence: float = 0.0
    min_similarity: float = -1.0
    max_matches: int | None = None
    mutual: bool = True

    def match(
        self,
        query_descriptors: np.ndarray,
        reference_descriptors: np.ndarray,
        *,
        query_image_size: tuple[int, int],
        reference_image_size: tuple[int, int],
    ) -> list[CoarseProposal]:
        matches = matcha_coarse_dual_softmax_matches(
            query_descriptors,
            reference_descriptors,
            query_image_width=int(query_image_size[0]),
            query_image_height=int(query_image_size[1]),
            render_image_width=int(reference_image_size[0]),
            render_image_height=int(reference_image_size[1]),
            logit_scale=float(self.logit_scale),
            min_confidence=float(self.min_confidence),
            min_similarity=float(self.min_similarity),
            max_matches=self.max_matches,
            mutual=bool(self.mutual),
        )
        return [_proposal_from_match(match) for match in matches]
