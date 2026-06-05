"""Array-level metrics for dense-context oracle diagnostics."""

from __future__ import annotations

import numpy as np


def _auc(labels: np.ndarray, scores: np.ndarray, valid_mask: np.ndarray | None = None) -> float | None:
    y = np.asarray(labels, dtype=bool).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        y = y[valid]
        s = s[valid]
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    pos = int(np.sum(y))
    neg = int(y.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float64)
    return float((np.sum(ranks[y]) - pos * (pos + 1) / 2.0) / max(pos * neg, 1))


def _ap(labels: np.ndarray, scores: np.ndarray, valid_mask: np.ndarray | None = None) -> float | None:
    y = np.asarray(labels, dtype=bool).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        y = y[valid]
        s = s[valid]
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    pos = int(np.sum(y))
    if pos == 0:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted.astype(np.float64))
    precision = tp / np.arange(1, y_sorted.size + 1, dtype=np.float64)
    return float(np.sum(precision[y_sorted]) / max(pos, 1))


def score_metrics_from_matrices(
    labels: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute candidate-level and token-level ranking metrics from NxK matrices."""

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 2:
        raise ValueError("labels and scores must have the same shape (N, K)")
    if valid_mask is None:
        valid = np.ones(y.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != y.shape:
            raise ValueError("valid_mask must have the same shape as labels")
    token_count = int(y.shape[0])
    valid_by_token = np.any(valid, axis=1)
    positive_by_token = np.any(y & valid, axis=1)
    masked_scores = np.where(valid, s, -np.inf)
    top_indices = np.argmax(masked_scores, axis=1) if token_count else np.zeros((0,), dtype=np.int64)
    token_rows = np.flatnonzero(valid_by_token)
    top1 = (
        y[token_rows, top_indices[token_rows]]
        if token_rows.size
        else np.zeros((0,), dtype=bool)
    )
    positive_best_ranks = []
    for row in np.flatnonzero(positive_by_token).tolist():
        order = np.argsort(-masked_scores[row])
        order = order[valid[row, order]]
        positive_positions = np.flatnonzero(y[row, order])
        if positive_positions.size:
            positive_best_ranks.append(float(positive_positions[0] + 1))
    return {
        "candidate_count": int(np.sum(valid)),
        "positive_count": int(np.sum(y & valid)),
        "positive_prior": float(np.sum(y & valid) / max(int(np.sum(valid)), 1)),
        "token_count": int(np.sum(valid_by_token)),
        "positive_token_count": int(np.sum(positive_by_token)),
        "auroc": _auc(y, s, valid),
        "auprc": _ap(y, s, valid),
        "top1_accuracy": float(np.mean(top1)) if top1.size else 0.0,
        "mean_best_positive_rank": None if not positive_best_ranks else float(np.mean(positive_best_ranks)),
        "median_best_positive_rank": None if not positive_best_ranks else float(np.median(positive_best_ranks)),
    }
