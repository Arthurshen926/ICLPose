"""Metrics for fixed descriptor-retrieval candidate banks."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


DEFAULT_RETRIEVAL_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (0.10, 5.0),
    (0.25, 10.0),
    (0.50, 10.0),
    (1.00, 10.0),
    (2.00, 20.0),
)


def _candidate_rank(candidate: CandidateHypothesis, fallback: int) -> int:
    rank = candidate.metadata.get("retrieval_rank")
    if rank is None:
        return fallback
    return int(rank)


def _group_by_query(bank: CandidateHypothesisBank) -> dict[str, list[CandidateHypothesis]]:
    grouped: dict[str, list[tuple[int, int, CandidateHypothesis]]] = defaultdict(list)
    for order, candidate in enumerate(bank.candidates):
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        grouped[candidate.query_id].append((_candidate_rank(candidate, order + 1), order, candidate))
    return {
        query_id: [candidate for _rank, _order, candidate in sorted(rows)]
        for query_id, rows in grouped.items()
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(values))


def _success(candidate: CandidateHypothesis, translation_m: float, rotation_deg: float) -> bool:
    if candidate.pose_error is None:
        raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
    return (
        float(candidate.pose_error.translation_m) <= float(translation_m)
        and float(candidate.pose_error.rotation_deg) <= float(rotation_deg)
    )


def summarize_descriptor_retrieval_bank(
    bank: CandidateHypothesisBank,
    thresholds: Sequence[tuple[float, float]] = DEFAULT_RETRIEVAL_THRESHOLDS,
    recall_at: Sequence[int] = (1, 5, 10, 20),
) -> Mapping[str, object]:
    """Summarize pose-label quality of a descriptor-retrieval candidate bank.

    Candidate generation is assumed fixed before this function is called. The
    pose labels are used only for evaluation of retrieval/localization recall.
    """

    grouped = _group_by_query(bank)
    query_rows = list(grouped.values())
    top1_translation = [float(rows[0].pose_error.translation_m) for rows in query_rows]
    top1_rotation = [float(rows[0].pose_error.rotation_deg) for rows in query_rows]
    oracle_translation = [
        min(float(candidate.pose_error.translation_m) for candidate in rows)
        for rows in query_rows
    ]
    oracle_rotation = [
        min(float(candidate.pose_error.rotation_deg) for candidate in rows)
        for rows in query_rows
    ]
    candidate_counts = [len(rows) for rows in query_rows]

    threshold_reports: dict[str, dict[str, float]] = {}
    for translation_m, rotation_deg in thresholds:
        label = f"{translation_m:g}m_{rotation_deg:g}deg"
        threshold_reports[label] = {}
        for k in recall_at:
            cutoff = int(k)
            hits = [
                any(_success(candidate, translation_m, rotation_deg) for candidate in rows[:cutoff])
                for rows in query_rows
            ]
            threshold_reports[label][f"recall_at_{cutoff}"] = _mean([1.0 if hit else 0.0 for hit in hits])

    return {
        "protocol_name": bank.protocol_name,
        "protocol_kind": bank.protocol_kind.value,
        "query_count": len(query_rows),
        "candidate_count": len(bank.candidates),
        "min_candidates_per_query": int(min(candidate_counts)) if candidate_counts else 0,
        "max_candidates_per_query": int(max(candidate_counts)) if candidate_counts else 0,
        "mean_candidates_per_query": _mean([float(count) for count in candidate_counts]),
        "median_top1_translation_m": _median(top1_translation),
        "median_top1_rotation_deg": _median(top1_rotation),
        "mean_top1_translation_m": _mean(top1_translation),
        "mean_top1_rotation_deg": _mean(top1_rotation),
        "median_oracle_translation_m": _median(oracle_translation),
        "median_oracle_rotation_deg": _median(oracle_rotation),
        "mean_oracle_translation_m": _mean(oracle_translation),
        "mean_oracle_rotation_deg": _mean(oracle_rotation),
        "thresholds": threshold_reports,
    }
