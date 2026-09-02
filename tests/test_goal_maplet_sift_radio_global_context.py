from __future__ import annotations

from feature_extract.tools.vfm.select_goal_maplet_sift_radio_by_global_context import (
    _select_sift,
)


def test_global_context_selector_is_fail_closed_and_ties_to_radio() -> None:
    assert not _select_sift(0.8, 0.9, True, False)
    assert _select_sift(-1.0, 0.1, False, True)
    assert not _select_sift(0.8, 0.8, True, True)
    assert _select_sift(0.8, 0.81, True, True)
