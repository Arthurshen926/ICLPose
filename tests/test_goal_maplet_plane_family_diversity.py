from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.diversify_goal_maplet_direct_radio_plane_ranking import (
    _family_first,
)


def test_family_first_preserves_score_order_within_unique_and_duplicate_passes() -> None:
    family = np.asarray([0, 0, 1, 2, 1, 3], np.int32)
    assert _family_first([0, 1, 2, 4, 3, 5], family) == [0, 2, 3, 5, 1, 4]


def test_family_first_can_keep_two_spatial_fragments_per_family() -> None:
    family = np.asarray([0, 0, 0, 1, 2, 1], np.int32)
    assert _family_first(
        [0, 1, 2, 3, 5, 4], family, maximum_per_family=2
    ) == [0, 1, 3, 5, 4, 2]


def test_family_first_rejects_nonpositive_cap() -> None:
    family = np.asarray([0], np.int32)
    try:
        _family_first([0], family, maximum_per_family=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive family cap should fail closed")
