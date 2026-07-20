from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_appearance import (
    _paired_rank,
    _predeclared_identity_gate,
    _rank_metrics,
    _thresholds,
)


def test_rank_metrics_use_the_same_explicit_candidate_coverage_for_both_scores() -> None:
    labels = np.asarray([[False, True, False], [True, False, False]], dtype=bool)
    valid = np.asarray([[True, True, False], [True, False, False]], dtype=bool)
    baseline = np.asarray([[0.9, 0.8, -9.0], [0.7, -9.0, -9.0]])
    appearance = np.asarray([[0.1, 0.8, np.nan], [0.9, np.nan, np.nan]])

    row_mask = np.asarray([True, True])
    metrics = _rank_metrics(
        scores=appearance,
        labels=labels & valid,
        candidate_valid=valid,
        row_mask=row_mask,
    )
    paired = _paired_rank(
        baseline_scores=baseline,
        appearance_scores=appearance,
        labels=labels,
        candidate_valid=valid,
        row_mask=row_mask,
    )

    assert metrics["positive_row_count"] == 2
    assert metrics["top1_positive_rate_given_positive"] == 1.0
    assert paired["rank_win_count"] == 1
    assert paired["rank_loss_count"] == 0
    assert np.array_equal(row_mask, np.asarray([True, True]))


def test_threshold_parser_is_strict() -> None:
    assert _thresholds("5,2") == (2.0, 5.0)
    for value in ("", "2,2", "0", "nan"):
        try:
            _thresholds(value)
        except ValueError:
            continue
        raise AssertionError(f"expected threshold parser to reject {value!r}")


def test_predeclared_gate_requires_rank_tail_and_paired_safety() -> None:
    audit = {
        "exact_registered_identity": {
            "baseline_common_coverage": {
                "median_first_positive_rank": 4.0,
                "p90_first_positive_rank": 10.0,
                "top1_positive_rate_given_positive": 0.2,
            },
            "appearance": {
                "median_first_positive_rank": 3.0,
                "p90_first_positive_rank": 10.0,
                "top1_positive_rate_given_positive": 0.25,
            },
            "paired_rank": {
                "rank_win_count": 8,
                "rank_loss_count": 3,
                "top1_rescue_count": 4,
                "top1_harm_count": 1,
            },
        }
    }
    assert _predeclared_identity_gate(audit)["passed"] is True
    audit["exact_registered_identity"]["paired_rank"]["top1_harm_count"] = 5
    assert _predeclared_identity_gate(audit)["passed"] is False
