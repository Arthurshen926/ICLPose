from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_reliability_cascade import (
    _select_spatial_candidate,
    _spatial_refinement_applied,
    _strict_inlier_improvement,
)


def test_spatial_cascade_uses_only_refinements_that_passed_upstream_gate() -> None:
    np.testing.assert_array_equal(
        _spatial_refinement_applied(np.asarray([70, 71, 72])),
        [False, True, True],
    )


def test_spatial_cascade_rejects_unknown_branch_semantics() -> None:
    with pytest.raises(ValueError, match="branch semantics"):
        _spatial_refinement_applied(np.asarray([70, 99]))


def test_multiscale_cascade_selects_maximum_valid_inliers_with_stable_tie() -> None:
    selected, valid = _select_spatial_candidate(
        np.asarray([[True, True, False], [False, False, False], [True, True, True]]),
        np.asarray([[10, 12, 99], [20, 30, 40], [8, 8, 7]]),
    )
    np.testing.assert_array_equal(selected, [1, 0, 0])
    np.testing.assert_array_equal(valid, [True, False, True])


def test_multiscale_cascade_requires_strict_inlier_improvement() -> None:
    np.testing.assert_array_equal(
        _strict_inlier_improvement(
            np.asarray([True, True, False]),
            np.asarray([11, 10, 99]),
            np.asarray([10, 10, 1]),
        ),
        [True, False, False],
    )
