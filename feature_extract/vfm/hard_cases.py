"""Hard-case subset construction for localization verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class HardCaseCandidate:
    query_id: str
    candidate_id: str
    cost_m: float
    basin_label: bool
    retrieval_rank: int | None = None
    verifier_score: float | None = None
    pnp_score: float | None = None
    identity_delta_m: float | None = None


@dataclass(frozen=True)
class HardCaseSplits:
    retrieval_top1_wrong: Tuple[str, ...]
    near_identity_false_positive: Tuple[str, ...]
    pnp_high_score_wrong: Tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "retrieval_top1_wrong": list(self.retrieval_top1_wrong),
            "near_identity_false_positive": list(self.near_identity_false_positive),
            "pnp_high_score_wrong": list(self.pnp_high_score_wrong),
        }


def _group_by_query(rows: Iterable[HardCaseCandidate]) -> Dict[str, list[HardCaseCandidate]]:
    grouped: Dict[str, list[HardCaseCandidate]] = {}
    for row in rows:
        grouped.setdefault(row.query_id, []).append(row)
    if not grouped:
        raise ValueError("hard-case builder requires at least one candidate row")
    return grouped


def build_hard_case_splits(
    rows: Iterable[HardCaseCandidate],
    accept_threshold: float,
    pnp_score_threshold: float,
    near_identity_threshold_m: float,
) -> HardCaseSplits:
    """Build query-level hard-case subsets from fixed candidate rows.

    A query is a retrieval-top1-wrong case when the rank-1 candidate is outside
    the target basin while another candidate for the same query is inside it.
    The other subsets track high-confidence wrong hypotheses from verification
    and PnP-style metadata.
    """

    grouped = _group_by_query(rows)
    retrieval_top1_wrong: list[str] = []
    near_identity_false_positive: list[str] = []
    pnp_high_score_wrong: list[str] = []

    for query_id, group in grouped.items():
        top1 = [row for row in group if row.retrieval_rank == 1]
        has_correct_candidate = any(row.basin_label for row in group)
        if top1 and has_correct_candidate and not top1[0].basin_label:
            retrieval_top1_wrong.append(query_id)

        if any(
            not row.basin_label
            and row.verifier_score is not None
            and row.verifier_score >= accept_threshold
            and row.identity_delta_m is not None
            and row.identity_delta_m <= near_identity_threshold_m
            for row in group
        ):
            near_identity_false_positive.append(query_id)

        if any(
            not row.basin_label
            and row.pnp_score is not None
            and row.pnp_score >= pnp_score_threshold
            for row in group
        ):
            pnp_high_score_wrong.append(query_id)

    return HardCaseSplits(
        retrieval_top1_wrong=tuple(sorted(retrieval_top1_wrong)),
        near_identity_false_positive=tuple(sorted(near_identity_false_positive)),
        pnp_high_score_wrong=tuple(sorted(pnp_high_score_wrong)),
    )
