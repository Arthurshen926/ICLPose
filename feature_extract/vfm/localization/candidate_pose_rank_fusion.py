"""Target-free rank-percentile fusion primitives for frozen pose hypotheses.

The base S0 selector and the high-resolution RGB likelihood have unrelated
raw scales.  This module deliberately combines only their within-query ranks,
so a train-only calibration can choose one bounded scalar without asserting
that either raw score is already a calibrated probability.
"""

from __future__ import annotations

import math

import numpy as np


RANK_PERCENTILE_FUSION_POLICY = "base_rank_percentile_plus_alpha_visual_rank_percentile_v1"


def validate_fusion_alpha(alpha: float) -> float:
    """Validate the non-negative visual rank weight."""

    value = float(alpha)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("rank-percentile fusion alpha must be finite and non-negative")
    return value


def rank_percentiles(values: np.ndarray, tie_break_orders: np.ndarray) -> np.ndarray:
    """Return higher-is-better within-query rank percentiles with shared ties."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    ties = np.asarray(tie_break_orders, dtype=np.int64).reshape(-1)
    if (
        len(scores) == 0
        or scores.shape != ties.shape
        or not np.isfinite(scores).all()
        or len(np.unique(ties)) != len(ties)
    ):
        raise ValueError("rank-percentile fusion inputs are invalid")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty((len(scores),), dtype=np.float64)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and scores[order[end]] == scores[order[begin]]:
            end += 1
        ranks[order[begin:end]] = 0.5 * float(begin + end - 1)
        begin = end
    return ranks / float(max(len(scores) - 1, 1))


def descending_ranks(values: np.ndarray, tie_break_orders: np.ndarray) -> np.ndarray:
    """Return deterministic one-based higher-is-better ranks."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    ties = np.asarray(tie_break_orders, dtype=np.int64).reshape(-1)
    if (
        len(scores) == 0
        or scores.shape != ties.shape
        or not np.isfinite(scores).all()
        or len(np.unique(ties)) != len(ties)
    ):
        raise ValueError("descending-rank fusion inputs are invalid")
    order = np.lexsort((ties, -scores))
    output = np.empty((len(scores),), dtype=np.int64)
    output[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return output


def selected_position(values: np.ndarray, tie_break_orders: np.ndarray) -> int:
    """Return the deterministic top-1 position for one query's hypotheses."""

    ranks = descending_ranks(values, tie_break_orders)
    selected = np.flatnonzero(ranks == 1)
    if selected.shape != (1,):
        raise RuntimeError("rank-percentile fusion did not produce one top hypothesis")
    return int(selected[0])


def fuse_rank_percentiles(
    *,
    baseline_scores: np.ndarray,
    visual_scores: np.ndarray,
    tie_break_orders: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse fixed S0 and visual evidence after per-query rank normalization."""

    weight = validate_fusion_alpha(alpha)
    baseline = rank_percentiles(baseline_scores, tie_break_orders)
    visual = rank_percentiles(visual_scores, tie_break_orders)
    output = baseline + weight * visual
    if not np.isfinite(output).all():  # pragma: no cover - guarded input range.
        raise RuntimeError("rank-percentile fusion produced non-finite scores")
    return output.astype(np.float64, copy=False)
