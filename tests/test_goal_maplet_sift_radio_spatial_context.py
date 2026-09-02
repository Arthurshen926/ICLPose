from __future__ import annotations

from feature_extract.tools.vfm.select_goal_maplet_sift_radio_by_spatial_context import (
    _select_sift,
)


def test_spatial_context_selector_is_fail_closed_and_ties_to_radio() -> None:
    assert not _select_sift((4, 0.8), (9, 0.9), True, False)
    assert _select_sift((0, 0.0), (1, 0.1), False, True)
    assert not _select_sift((8, 0.8), (8, 0.8), True, True)
    assert _select_sift((8, 0.8), (9, 0.1), True, True)
    assert _select_sift((8, 0.8), (8, 0.81), True, True)
