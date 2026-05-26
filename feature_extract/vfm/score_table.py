"""Score-table evaluation for candidate hypothesis verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from feature_extract.vfm.metrics import ranking_summary
from feature_extract.vfm.protocols import ProtocolKind


@dataclass(frozen=True)
class ScoreRow:
    query_id: str
    candidate_id: str
    score: float
    cost_m: float
    basin_label: bool
    protocol_kind: ProtocolKind
    method: str
    risk: float = 0.0
    mean_similarity: Optional[float] = None
    inlier_fraction: Optional[float] = None
    match_count: Optional[int] = None
    visibility_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_kind", ProtocolKind(self.protocol_kind))


@dataclass(frozen=True)
class ScoreTableReport:
    method: str
    protocol_kind: ProtocolKind
    query_count: int
    mean_pred_cost_m: float
    mean_oracle_cost_m: float
    mean_oracle_gap_m: float
    mean_top1_acc: float
    mean_spearman: float
    mean_ndcg_at_10: float
    mean_basin_recall_at_1: float
    mean_basin_recall_at_5: float
    mean_basin_recall_at_10: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "protocol_kind": self.protocol_kind.value,
            "query_count": self.query_count,
            "mean_pred_cost_m": self.mean_pred_cost_m,
            "mean_oracle_cost_m": self.mean_oracle_cost_m,
            "mean_oracle_gap_m": self.mean_oracle_gap_m,
            "mean_top1_acc": self.mean_top1_acc,
            "mean_spearman": self.mean_spearman,
            "mean_ndcg_at_10": self.mean_ndcg_at_10,
            "mean_basin_recall_at_1": self.mean_basin_recall_at_1,
            "mean_basin_recall_at_5": self.mean_basin_recall_at_5,
            "mean_basin_recall_at_10": self.mean_basin_recall_at_10,
        }


def group_rows_by_query(rows: Iterable[ScoreRow]) -> Dict[str, List[ScoreRow]]:
    grouped: Dict[str, List[ScoreRow]] = {}
    for row in rows:
        grouped.setdefault(row.query_id, []).append(row)
    if not grouped:
        raise ValueError("score table must contain at least one row")
    return grouped


def validate_candidate_groups(rows: Iterable[ScoreRow], min_candidates: int) -> None:
    if min_candidates <= 0:
        raise ValueError("min_candidates must be positive")
    for query_id, group in group_rows_by_query(rows).items():
        if len(group) < min_candidates:
            raise ValueError(
                f"query '{query_id}' has {len(group)} candidate(s), expected at least {min_candidates}"
            )


def _ensure_single_protocol_and_method(rows: List[ScoreRow]) -> tuple[ProtocolKind, str]:
    protocols = {row.protocol_kind for row in rows}
    methods = {row.method for row in rows}
    if len(protocols) != 1:
        raise ValueError("score table mixes protocol kinds")
    if len(methods) != 1:
        raise ValueError("score table mixes methods")
    return next(iter(protocols)), next(iter(methods))


def evaluate_score_table(rows: Iterable[ScoreRow]) -> ScoreTableReport:
    row_list = list(rows)
    protocol, method = _ensure_single_protocol_and_method(row_list)
    grouped = group_rows_by_query(row_list)
    summaries = []
    for group in grouped.values():
        summaries.append(
            ranking_summary(
                scores=[row.score for row in group],
                costs_m=[row.cost_m for row in group],
                basin_labels=[row.basin_label for row in group],
            )
        )

    def mean(field: str) -> float:
        return float(np.mean([getattr(summary, field) for summary in summaries]))

    return ScoreTableReport(
        method=method,
        protocol_kind=protocol,
        query_count=len(summaries),
        mean_pred_cost_m=mean("pred_cost_m"),
        mean_oracle_cost_m=mean("oracle_cost_m"),
        mean_oracle_gap_m=mean("oracle_gap_m"),
        mean_top1_acc=mean("top1_acc"),
        mean_spearman=mean("spearman"),
        mean_ndcg_at_10=mean("ndcg_at_10"),
        mean_basin_recall_at_1=mean("basin_recall_at_1"),
        mean_basin_recall_at_5=mean("basin_recall_at_5"),
        mean_basin_recall_at_10=mean("basin_recall_at_10"),
    )


def rows_from_arrays(
    query_ids: Iterable[str],
    scores: np.ndarray,
    costs_m: np.ndarray,
    basin_labels: np.ndarray,
    protocol_kind: ProtocolKind,
    method: str,
) -> List[ScoreRow]:
    score_array = np.asarray(scores, dtype=np.float64)
    cost_array = np.asarray(costs_m, dtype=np.float64)
    basin_array = np.asarray(basin_labels, dtype=bool)
    query_list = list(query_ids)
    if score_array.shape != cost_array.shape or score_array.shape != basin_array.shape:
        raise ValueError("scores, costs_m, and basin_labels must have the same shape")
    if score_array.ndim != 2 or score_array.shape[0] != len(query_list):
        raise ValueError("arrays must have shape (query_count, candidates_per_query)")

    rows: List[ScoreRow] = []
    for q_idx, query_id in enumerate(query_list):
        for c_idx in range(score_array.shape[1]):
            rows.append(
                ScoreRow(
                    query_id=query_id,
                    candidate_id=f"c{c_idx}",
                    score=float(score_array[q_idx, c_idx]),
                    cost_m=float(cost_array[q_idx, c_idx]),
                    basin_label=bool(basin_array[q_idx, c_idx]),
                    protocol_kind=protocol_kind,
                    method=method,
                )
            )
    return rows
