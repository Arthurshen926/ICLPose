"""Baseline scoring from fixed candidate-bank metadata."""

from __future__ import annotations

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.score_table import ScoreRow


def _retrieval_order_score(candidate) -> float:
    rank = candidate.metadata.get("retrieval_rank")
    if rank is None:
        raise ValueError(f"candidate {candidate.candidate_id} has no retrieval_rank metadata")
    return -float(rank)


def _prior_score(candidate) -> float:
    if candidate.prior_score is None:
        raise ValueError(f"candidate {candidate.candidate_id} has no prior_score")
    return float(candidate.prior_score)


def score_candidate_bank_by_metadata(
    bank: CandidateHypothesisBank,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> list[ScoreRow]:
    """Convert a labeled candidate bank into a baseline score table."""

    scorers = {
        "retrieval_order": _retrieval_order_score,
        "candidate_prior": _prior_score,
    }
    if method not in scorers:
        raise ValueError(f"unsupported metadata scoring method: {method}")
    rows: list[ScoreRow] = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=scorers[method](candidate),
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    translation_threshold_m=translation_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                ),
                protocol_kind=bank.protocol_kind,
                method=method,
            )
        )
    return rows
