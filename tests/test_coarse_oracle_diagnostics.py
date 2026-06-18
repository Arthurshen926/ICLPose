from __future__ import annotations

import numpy as np

from feature_extract.vfm.coarse_oracle_diagnostics import (
    CoarseOracleRankAccumulator,
    coarse_oracle_candidate_rows,
    coarse_oracle_rank_rows,
    summarize_coarse_oracle_rank_rows,
)


def test_coarse_oracle_rank_rows_report_rank_and_cell_distance() -> None:
    query = np.zeros((2, 1, 2), dtype=np.float32)
    render = np.zeros((2, 1, 3), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0]
    render[:, 0, 0] = [0.0, 1.0]
    render[:, 0, 1] = [1.0, 0.0]
    render[:, 0, 2] = [0.8, 0.2]

    rows = coarse_oracle_rank_rows(
        query,
        render,
        oracle_render_indices=np.asarray([2, 0], dtype=np.int64),
        logit_scale=10.0,
    )

    assert len(rows) == 2
    assert rows[0]["query_index"] == 0
    assert rows[0]["coarse_top1_render_index"] == 1
    assert rows[0]["oracle_render_index"] == 2
    assert rows[0]["cell_distance_top1_to_oracle"] == 1
    assert rows[0]["oracle_rank_by_similarity"] == 2
    assert rows[0]["oracle_rank_by_dual_softmax"] == 2
    assert rows[1]["oracle_rank_by_similarity"] == 1
    assert rows[1]["coarse_hit_radius0"] is True


def test_summarize_coarse_oracle_rank_rows_reports_hit_rates() -> None:
    rows = [
        {"oracle_rank_by_dual_softmax": 1, "coarse_hit_radius0": True, "coarse_hit_radius1": True, "coarse_hit_radius2": True, "mutual_dropped_oracle": False},
        {"oracle_rank_by_dual_softmax": 3, "coarse_hit_radius0": False, "coarse_hit_radius1": True, "coarse_hit_radius2": True, "mutual_dropped_oracle": True},
        {"oracle_rank_by_dual_softmax": 21, "coarse_hit_radius0": False, "coarse_hit_radius1": False, "coarse_hit_radius2": False, "mutual_dropped_oracle": True},
    ]

    summary = summarize_coarse_oracle_rank_rows(rows)

    assert summary["coarse_oracle_count"] == 3
    assert summary["coarse_hit_radius0"] == 1 / 3
    assert summary["coarse_hit_radius1"] == 2 / 3
    assert summary["oracle_rank_at_1"] == 1 / 3
    assert summary["oracle_rank_at_5"] == 2 / 3
    assert summary["mutual_dropped_oracle_rate"] == 2 / 3


def test_coarse_oracle_rank_accumulator_matches_batch_summary() -> None:
    rows = [
        {
            "oracle_rank_by_dual_softmax": 1,
            "coarse_hit_radius0": True,
            "coarse_hit_radius1": True,
            "coarse_hit_radius2": True,
            "mutual_dropped_oracle": False,
            "score_gap_top1_minus_oracle": 0.0,
        },
        {
            "oracle_rank_by_dual_softmax": 4,
            "coarse_hit_radius0": False,
            "coarse_hit_radius1": True,
            "coarse_hit_radius2": True,
            "mutual_dropped_oracle": True,
            "score_gap_top1_minus_oracle": 0.3,
        },
        {
            "oracle_rank_by_dual_softmax": 30,
            "coarse_hit_radius0": False,
            "coarse_hit_radius1": False,
            "coarse_hit_radius2": False,
            "mutual_dropped_oracle": True,
            "score_gap_top1_minus_oracle": 0.6,
        },
    ]
    accumulator = CoarseOracleRankAccumulator()

    accumulator.update(rows[:1])
    accumulator.update(rows[1:])

    assert accumulator.summary() == summarize_coarse_oracle_rank_rows(rows)


def test_coarse_oracle_candidate_rows_exports_topk_labels_for_ranker() -> None:
    query = np.zeros((2, 1, 1), dtype=np.float32)
    render = np.zeros((2, 1, 3), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 1] = [0.8, 0.2]
    render[:, 0, 2] = [0.2, 0.8]

    rows = coarse_oracle_candidate_rows(
        query,
        render,
        oracle_render_indices=np.asarray([1], dtype=np.int64),
        top_k=2,
        positive_radius=0,
        logit_scale=10.0,
    )

    assert len(rows) == 2
    assert [row["coarse_rank"] for row in rows] == [0, 1]
    assert rows[0]["candidate_render_index"] == 0
    assert rows[1]["candidate_render_index"] == 1
    assert rows[0]["patch_correct"] is False
    assert rows[1]["patch_correct"] is True
    assert rows[1]["cell_distance_to_oracle"] == 0
