"""Join labels from score-table candidates onto deployable candidate banks."""

from __future__ import annotations

from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


def _retrieval_rank(candidate: CandidateHypothesis) -> int | None:
    value = candidate.metadata.get("retrieval_rank")
    if value is None or value == "":
        return None
    return int(value)


def label_candidate_bank_from_score_table(
    unlabeled_bank: CandidateHypothesisBank,
    score_table_bank: CandidateHypothesisBank,
    protocol_name: str,
) -> CandidateHypothesisBank:
    """Copy pose labels/score metadata from a score-table bank by query/rank.

    Real retrieval exports can carry deployable fields such as reference_image
    and hypothesis pose, while offline score tables carry pose_error labels.
    Matching on `(query_id, retrieval_rank)` preserves the deployable candidate
    record and adds the evaluation labels needed for fixed-candidate scoring.
    """

    label_index: dict[tuple[str, int], CandidateHypothesis] = {}
    for candidate in score_table_bank.candidates:
        if candidate.query_id is None or candidate.pose_error is None:
            continue
        rank = _retrieval_rank(candidate)
        if rank is None:
            continue
        label_index[(candidate.query_id, rank)] = candidate

    labeled: list[CandidateHypothesis] = []
    for candidate in unlabeled_bank.candidates:
        if candidate.query_id is None:
            continue
        rank = _retrieval_rank(candidate)
        if rank is None:
            continue
        label = label_index.get((candidate.query_id, rank))
        if label is None or label.pose_error is None:
            continue
        metadata = dict(candidate.metadata)
        metadata.update(dict(label.metadata))
        labeled.append(
            CandidateHypothesis(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                candidate_type=candidate.candidate_type,
                pose_error=label.pose_error,
                prior_score=label.prior_score if label.prior_score is not None else candidate.prior_score,
                pose=candidate.pose,
                reference_image=candidate.reference_image,
                submap_id=candidate.submap_id,
                solver_success=candidate.solver_success,
                hard_case_type=candidate.hard_case_type,
                metadata=metadata,
            )
        )
    return CandidateHypothesisBank.from_candidates(
        protocol_name=protocol_name,
        protocol_kind=unlabeled_bank.protocol_kind,
        candidates=labeled,
        protocol_fingerprint=unlabeled_bank.protocol_fingerprint,
    )
