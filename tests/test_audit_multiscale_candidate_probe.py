from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_multiscale_candidate_probe import (
    _paired_rank_audit,
    _rank2_to_l_rescue_audit,
    _set_valued_metrics,
)


def test_set_valued_metrics_credit_all_geometric_positives_and_null() -> None:
    probability = np.asarray(
        [[0.35, 0.35, 0.0], [0.20, 0.10, 0.0]], dtype=np.float64
    )
    null = np.asarray([0.30, 0.70], dtype=np.float64)
    labels = np.asarray([[True, True, False], [False, False, False]])
    valid = np.asarray([[True, True, False], [True, True, False]])
    metrics = _set_valued_metrics(
        probability,
        null,
        labels=labels,
        valid=valid,
        row_mask=np.asarray([True, True]),
    )
    assert metrics["positive_row_count"] == 1
    assert metrics["positive_candidate_count"] == 2
    assert metrics["group_argmax_correct_rate"] == 1.0
    assert np.isclose(metrics["group_target_nll"], -np.log(0.70))


def test_paired_rank_audit_counts_rescue_and_harm() -> None:
    baseline = np.asarray([[0.2, 0.7], [0.8, 0.1]], dtype=np.float64)
    probe = np.asarray([[0.8, 0.1], [0.1, 0.7]], dtype=np.float64)
    labels = np.asarray([[True, False], [True, False]])
    valid = np.ones_like(labels, dtype=bool)
    paired = _paired_rank_audit(
        baseline,
        probe,
        labels=labels,
        valid=valid,
        row_mask=np.asarray([True, True]),
    )
    assert paired["positive_row_count"] == 2
    assert paired["rank_win_count"] == 1
    assert paired["rank_loss_count"] == 1
    assert paired["top1_rescue_count"] == 1
    assert paired["top1_harm_count"] == 1


def test_rank2_to_l_rescue_audit_is_limited_to_baseline_missed_positives() -> None:
    baseline = np.asarray([[0.2, 0.7], [0.8, 0.1]], dtype=np.float64)
    probe = np.asarray([[0.8, 0.1], [0.7, 0.2]], dtype=np.float64)
    labels = np.asarray([[True, False], [True, False]])
    report = _rank2_to_l_rescue_audit(
        baseline,
        probe,
        labels=labels,
        valid=np.ones_like(labels, dtype=bool),
        row_mask=np.asarray([True, True]),
    )
    assert report["eligible_positive_row_count"] == 1
    assert report["baseline"]["median_first_positive_rank"] == 2.0
    assert report["probe"]["median_first_positive_rank"] == 1.0
    assert report["paired_rank"]["top1_rescue_count"] == 1
