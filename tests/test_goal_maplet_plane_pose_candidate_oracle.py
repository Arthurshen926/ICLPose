from __future__ import annotations

import pytest

from feature_extract.tools.vfm.evaluate_goal_maplet_plane_pose_candidate_oracle import (
    evaluate_candidate_oracle,
)


def _row(t: float, r: float, usable: bool = True) -> dict[str, object]:
    return {"translation_error_m": t, "rotation_error_deg": r, "usable": usable}


def test_oracle_reports_threshold_union_and_selector_recoverable_miss() -> None:
    first = {"a": _row(.08, .8), "b": _row(.6, 6.0)}
    second = {"a": _row(.2, 1.5), "b": _row(.2, 1.5)}
    selected = {"a": _row(.2, 1.5), "b": _row(.6, 6.0)}
    result = evaluate_candidate_oracle([first, second], ["wide", "strict"], selected)
    assert result["threshold_union_hit_counts"]["0.1m_1deg"] == 1
    assert result["threshold_union_hit_counts"]["0.25m_2deg"] == 2
    assert result["selected_threshold_recoverable_miss_counts"]["0.25m_2deg"] == 1
    assert result["composite_oracle_branch_counts"] == {"wide": 1, "strict": 1}


def test_oracle_rejects_inventory_mismatch() -> None:
    with pytest.raises(ValueError, match="inventories differ"):
        evaluate_candidate_oracle([{"a": _row(.1, 1)}, {"b": _row(.1, 1)}], ["a", "b"])
