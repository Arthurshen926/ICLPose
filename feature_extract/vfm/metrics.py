"""Metrics for hypothesis verification and feature-selection gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RankingSummary:
    pred_cost_m: float
    oracle_cost_m: float
    oracle_gap_m: float
    top1_acc: float
    spearman: float
    ndcg_at_10: float
    basin_recall_at_1: float
    basin_recall_at_2: float
    basin_recall_at_5: float
    basin_recall_at_10: float


def _as_1d(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _order_by_score(scores: Sequence[float]) -> np.ndarray:
    scores_array = _as_1d(scores, "scores")
    return np.argsort(-scores_array, kind="mergesort")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_rank(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = _as_1d(x, "x")
    y_array = _as_1d(y, "y")
    if x_array.size != y_array.size:
        raise ValueError("x and y must have the same length")
    rx = _rankdata(x_array)
    ry = _rankdata(y_array)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0.0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


def ndcg_at_k(scores: Sequence[float], relevance: Sequence[float], k: int) -> float:
    score_array = _as_1d(scores, "scores")
    rel = _as_1d(relevance, "relevance")
    if score_array.size != rel.size:
        raise ValueError("scores and relevance must have the same length")
    if k <= 0:
        raise ValueError("k must be positive")
    shifted = rel - rel.min()
    order = _order_by_score(score_array)[:k]
    ideal = np.argsort(-shifted, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, order.size + 2, dtype=np.float64))
    dcg = float(np.sum(shifted[order] * discounts))
    idcg = float(np.sum(shifted[ideal] * discounts))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def basin_recall_at_k(scores: Sequence[float], basin_labels: Sequence[bool], k: int) -> float:
    labels = np.asarray(basin_labels, dtype=bool).reshape(-1)
    if labels.size == 0:
        raise ValueError("basin_labels must be non-empty")
    order = _order_by_score(scores)
    if order.size != labels.size:
        raise ValueError("scores and basin_labels must have the same length")
    return float(np.any(labels[order[: max(1, k)]]))


def oracle_gap(scores: Sequence[float], costs_m: Sequence[float]) -> float:
    score_array = _as_1d(scores, "scores")
    costs = _as_1d(costs_m, "costs_m")
    if score_array.size != costs.size:
        raise ValueError("scores and costs_m must have the same length")
    pred_cost = float(costs[_order_by_score(score_array)[0]])
    oracle_cost = float(costs.min())
    return pred_cost - oracle_cost


def hard_false_accept_rate(
    scores: Sequence[float],
    basin_labels: Sequence[bool],
    accept_threshold: float,
) -> float:
    score_array = _as_1d(scores, "scores")
    labels = np.asarray(basin_labels, dtype=bool).reshape(-1)
    if score_array.size != labels.size:
        raise ValueError("scores and basin_labels must have the same length")
    accepted = score_array >= accept_threshold
    if not np.any(accepted):
        return 0.0
    return float(np.mean(~labels[accepted]))


def ranking_summary(
    scores: Sequence[float],
    costs_m: Sequence[float],
    basin_labels: Sequence[bool],
) -> RankingSummary:
    score_array = _as_1d(scores, "scores")
    costs = _as_1d(costs_m, "costs_m")
    labels = np.asarray(basin_labels, dtype=bool).reshape(-1)
    if not (score_array.size == costs.size == labels.size):
        raise ValueError("scores, costs_m, and basin_labels must have the same length")
    order = _order_by_score(score_array)
    pred_cost = float(costs[order[0]])
    oracle_cost = float(costs.min())
    return RankingSummary(
        pred_cost_m=pred_cost,
        oracle_cost_m=oracle_cost,
        oracle_gap_m=pred_cost - oracle_cost,
        top1_acc=float(labels[order[0]]),
        spearman=spearman_rank(score_array, -costs),
        ndcg_at_10=ndcg_at_k(score_array, -costs, min(10, score_array.size)),
        basin_recall_at_1=basin_recall_at_k(score_array, labels, 1),
        basin_recall_at_2=basin_recall_at_k(score_array, labels, 2),
        basin_recall_at_5=basin_recall_at_k(score_array, labels, 5),
        basin_recall_at_10=basin_recall_at_k(score_array, labels, 10),
    )
