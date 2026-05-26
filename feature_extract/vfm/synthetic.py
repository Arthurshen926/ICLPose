"""Synthetic validation for the VFM-MapLoc reporting pipeline.

This is a code-level positive control. It validates that metrics, controls, and
reporting can expose a known selected-feature signal. It is not a localization
benchmark result.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import evaluate_score_table, rows_from_arrays


def _method_report(
    query_ids,
    scores: np.ndarray,
    costs: np.ndarray,
    basin: np.ndarray,
    method: str,
) -> Dict[str, object]:
    rows = rows_from_arrays(
        query_ids=query_ids,
        scores=scores,
        costs_m=costs,
        basin_labels=basin,
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        method=method,
    )
    return evaluate_score_table(rows).to_dict()


def run_synthetic_feature_utility_validation(
    query_count: int = 64,
    candidates_per_query: int = 8,
    seed: int = 0,
) -> Dict[str, Dict[str, object]]:
    """Generate a positive-control feature utility experiment."""

    if query_count <= 0 or candidates_per_query < 2:
        raise ValueError("query_count must be positive and candidates_per_query >= 2")
    rng = np.random.default_rng(seed)
    query_ids = [f"q{i}" for i in range(query_count)]

    costs = rng.uniform(0.20, 1.00, size=(query_count, candidates_per_query))
    positive_idx = rng.integers(0, candidates_per_query, size=query_count)
    costs[np.arange(query_count), positive_idx] = rng.uniform(0.03, 0.09, size=query_count)
    basin = costs <= 0.15

    selected_scores = -costs + rng.normal(0.0, 0.01, size=costs.shape)
    raw_scores = -costs + rng.normal(0.0, 0.35, size=costs.shape)
    metadata_scores = rng.normal(0.0, 0.20, size=costs.shape) - 0.02 * np.arange(candidates_per_query)
    shuffle_scores = rng.permutation(selected_scores.reshape(-1)).reshape(selected_scores.shape)

    return {
        "selected_feature": _method_report(query_ids, selected_scores, costs, basin, "selected_feature"),
        "raw_vfm": _method_report(query_ids, raw_scores, costs, basin, "raw_vfm"),
        "metadata_only": _method_report(query_ids, metadata_scores, costs, basin, "metadata_only"),
        "query_shuffle": _method_report(query_ids, shuffle_scores, costs, basin, "query_shuffle"),
    }
