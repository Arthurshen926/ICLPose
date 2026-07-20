from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_pose_evidence import (
    _baseline_rows,
    _per_query_rows,
    _validate_alpha_zero,
    descending_ranks,
    parse_alpha_grid,
    rank_percentiles,
)


def test_rank_percentiles_preserve_equal_scores_and_descending_tie_break() -> None:
    scores = np.asarray([0.2, 0.8, 0.8, 0.1], dtype=np.float64)
    order = np.asarray([4, 3, 2, 1], dtype=np.int64)
    percentiles = rank_percentiles(scores, order)
    assert percentiles[1] == pytest.approx(percentiles[2])
    assert descending_ranks(scores, order).tolist() == [3, 2, 1, 4]


def test_alpha_zero_baseline_rows_reproduce_immutable_selection() -> None:
    keys = (
        ("validation", "label", "q0", 0),
        ("validation", "label", "q0", 1),
        ("validation", "label", "q1", 0),
        ("validation", "label", "q1", 1),
    )
    baseline = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float64)
    translation = np.asarray([0.5, 0.1, 0.2, 0.05], dtype=np.float64)
    rotation = np.zeros((4,), dtype=np.float64)
    tie_order = np.arange(4, dtype=np.int64)
    rows = _baseline_rows(
        keys=keys,
        baseline_scores=baseline,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_order,
    )
    _validate_alpha_zero(
        source_top1=np.asarray([False, True, True, False]),
        keys=keys,
        baseline_rows=rows,
    )
    assert [row["selected_hypothesis_index"] for row in rows] == [1, 0]
    assert [row["oracle_score_rank"] for row in rows] == [1, 2]


def test_per_query_rows_reports_best_correct_rank_without_using_it_for_selection() -> None:
    keys = (
        ("validation", "label", "q", 0),
        ("validation", "label", "q", 1),
        ("validation", "label", "q", 2),
    )
    rows = _per_query_rows(
        keys=keys,
        scores=np.asarray([0.9, 0.8, 0.1]),
        baseline_scores=np.asarray([0.9, 0.8, 0.1]),
        translation_m=np.asarray([0.4, 0.08, 0.01]),
        rotation_deg=np.asarray([0.0, 0.0, 0.0]),
        tie_break_orders=np.arange(3, dtype=np.int64),
        family="f",
        score_mode="standalone",
        alpha=None,
    )
    assert rows[0]["selected_hypothesis_index"] == 0
    assert rows[0]["best_10cm_rank"] == 2
    assert rows[0]["best_3cm_rank"] == 3


def test_alpha_grid_requires_zero_and_nonnegative_values() -> None:
    assert parse_alpha_grid("1,0,0.5") == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        parse_alpha_grid("0.5")
    with pytest.raises(ValueError):
        parse_alpha_grid("0,-1")

