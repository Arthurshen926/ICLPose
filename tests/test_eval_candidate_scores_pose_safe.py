from __future__ import annotations

import argparse

import numpy as np

from feature_extract.tools.vfm.eval_candidate_scores_pose_safe import (
    _finite_float_list,
    _score_key_map,
    _switch_residual_audit,
)


def test_score_key_map_parses_decoupled_assignment_and_confidence_keys() -> None:
    assert _score_key_map("resolved=p5,baseline=baseline") == {
        "resolved": "p5",
        "baseline": "baseline",
    }


def test_score_key_map_rejects_ambiguous_entries() -> None:
    for value in ("missing_separator", "a=b,a=c", "=b", "a="):
        try:
            _score_key_map(value)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"expected invalid score map: {value}")


def test_finite_float_list_allows_empty_and_rejects_nonfinite_values() -> None:
    assert _finite_float_list("") == ()
    assert _finite_float_list("0.1, 0.5") == (0.1, 0.5)
    for value in ("nan", "inf", "-inf"):
        try:
            _finite_float_list(value)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"expected invalid float list: {value}")


def test_switch_residual_audit_reports_update_risk() -> None:
    audit = _switch_residual_audit(
        np.asarray([[10.0, 2.0], [1.0, 4.0], [np.inf, 3.0]], dtype=np.float32),
        np.asarray([[0.9, 0.8], [0.9, 0.8], [0.9, 0.8]], dtype=np.float32),
        np.asarray([[-np.inf, 0.9], [-np.inf, 0.9], [-np.inf, 0.9]], dtype=np.float32),
        row_mask=np.asarray([True, True, True]),
    )
    assert audit["switch_count"] == 3
    assert audit["true_rescue_count"] == 2
    assert audit["improved_count"] == 2
    assert audit["worsened_count"] == 1
    assert audit["baseline_valid_false_switch_count"] == 1
    assert audit["finite_residual_delta_count"] == 2
    assert audit["median_selected_minus_baseline_residual_px"] == -2.5
