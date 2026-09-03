from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.select_goal_maplet_dual_surface_geometry_consensus import (
    _select_by_cross_geometry_inlier_ratio,
)


def test_dual_surface_selection_grades_both_geometries_and_breaks_ties_stably() -> None:
    ratio = np.asarray([
        [[0.8, 0.6], [0.7, 0.9]],
        [[0.7, 0.7], [0.7, 0.7]],
        [[0.9, 0.8], [0.9, 0.7]],
    ])
    np.testing.assert_array_equal(_select_by_cross_geometry_inlier_ratio(ratio), [1, 0, 0])


def test_dual_surface_selection_rejects_nonfinite_or_wrong_shape() -> None:
    with pytest.raises(ValueError):
        _select_by_cross_geometry_inlier_ratio(np.zeros((2, 2)))
    bad = np.zeros((1, 2, 2)); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        _select_by_cross_geometry_inlier_ratio(bad)
