from __future__ import annotations

import pytest

from feature_extract.vfm.localization_goal_maplet.pose_ambiguity_calibration import (
    calibrate_zero_false_accept_margin,
    evaluate_ambiguity_gate,
)


def _row(margin, loose, strict=False, retained=True):
    return {
        "image_id": str(margin),
        "final_distinct_score_margin": margin,
        "final_loose_1m_10deg": loose,
        "final_strict_0_5m_5deg": strict,
        "final_any_loose_1m_10deg": retained,
    }


def test_zero_false_accept_gate_uses_strict_maximum_error_margin():
    rows = [_row(0.4, True, True), _row(0.2, False), _row(0.25, False)]
    threshold = calibrate_zero_false_accept_margin(rows)
    assert threshold == 0.25
    result = evaluate_ambiguity_gate(rows, threshold)
    assert result["accepted_single_pose_count"] == 1
    assert result["accepted_loose_accuracy"] == 1.0
    assert result["accepted_loose_error_count"] == 0
    assert result["unresolved_set_loose_retention_rate"] == 1.0
    assert result["rows"][2]["single_pose_accepted"] is False


def test_ambiguity_gate_rejects_invalid_or_uninformative_calibration():
    with pytest.raises(ValueError, match="observed errors"):
        calibrate_zero_false_accept_margin([_row(0.4, True)])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_ambiguity_gate([_row(float("nan"), True)], 0.2)
