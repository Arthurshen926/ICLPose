from __future__ import annotations

import numpy as np

from feature_extract.vfm.measurement_v1.candidate_measurement_selection import (
    _select_candidate_columns,
    audit_candidate_measurement_selection,
)


def test_selection_preserves_frozen_candidate_and_adds_score_alternatives() -> None:
    scores = np.asarray([[0.9, 0.8, 0.7], [0.4, 0.6, 0.5]], dtype=np.float32)
    valid = np.ones_like(scores, dtype=bool)
    columns, roles = _select_candidate_columns(
        scores=scores,
        valid=valid,
        frozen_selected_columns=np.asarray([1, 0]),
        candidates_per_token=3,
    )
    assert columns.tolist() == [[1, 0, 2], [0, 1, 2]]
    assert roles[:, 0].tolist() == ["frozen_selected", "frozen_selected"]


def test_rescue_audit_distinguishes_selected_top_m_and_pool() -> None:
    measured = np.asarray(
        [[10.0, 1.0, 9.0], [1.0, 8.0, 9.0], [20.0, 30.0, 40.0]],
        dtype=np.float32,
    )
    pool = np.asarray(
        [[10.0, 1.0, 9.0, 12.0], [1.0, 8.0, 9.0, 10.0], [20.0, 30.0, 40.0, 1.5]],
        dtype=np.float32,
    )
    report = audit_candidate_measurement_selection(
        query_ids=np.asarray(["seq1/a.png", "seq1/a.png", "seq2/b.png"]),
        split_names=np.asarray(["validation", "validation", "validation"]),
        selected_residuals=measured,
        pool_residuals=pool,
        valid=np.ones_like(measured, dtype=bool),
        thresholds_px=(2.0,),
    )
    metrics = report["split_metrics"]["validation"]["2.0"]
    assert metrics["selected_correct_count"] == 1
    assert metrics["measured_top_m_correct_count"] == 2
    assert metrics["oracle_pool_correct_count"] == 3
    assert metrics["rescue_eligible_count"] == 2
    assert metrics["rescued_count"] == 1
    assert metrics["rescue_recall"] == 0.5
