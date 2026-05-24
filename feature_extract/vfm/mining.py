"""Hard-case mining helpers."""

from __future__ import annotations

from typing import Iterable, List

from feature_extract.vfm.hypotheses import CandidateHypothesis


def mine_false_accepts(
    hypotheses: Iterable[CandidateHypothesis],
    scores: Iterable[float],
    accept_threshold: float,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> List[CandidateHypothesis]:
    """Return high-scoring candidates outside the solver basin."""

    hard: List[CandidateHypothesis] = []
    for hypothesis, score in zip(hypotheses, scores):
        if score < accept_threshold:
            continue
        if hypothesis.basin_label(translation_threshold_m, rotation_threshold_deg):
            continue
        hard.append(hypothesis)
    return hard
