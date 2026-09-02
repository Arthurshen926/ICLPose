from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_budget_consensus import (
    _agreement_shrunk_fraction,
    _interpolate_pose,
)
from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_map_density import (
    _midpoint_pose,
)


def test_budget_consensus_midpoint_is_symmetric() -> None:
    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    right[0, 3] = -0.4
    assert np.allclose(_midpoint_pose(left, right), _midpoint_pose(right, left))


def test_interpolate_pose_endpoints_and_midpoint() -> None:
    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    right[:3, :3] = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    right_center = np.asarray([2.0, -1.0, 0.5])
    right[:3, 3] = -right[:3, :3] @ right_center
    assert np.allclose(_interpolate_pose(left, right, 0.0), left)
    assert np.allclose(_interpolate_pose(left, right, 1.0), right)
    assert np.allclose(_interpolate_pose(left, right, 0.5), _midpoint_pose(left, right))


def test_agreement_shrink_is_bounded_monotonic_and_fails_closed() -> None:
    assert _agreement_shrunk_fraction(0.0, 0.0) == 0.5
    near = _agreement_shrunk_fraction(0.1, 0.5)
    far = _agreement_shrunk_fraction(0.4, 4.0)
    assert 0.0 < far < near < 0.5
    assert _agreement_shrunk_fraction(0.51, 0.0) == 0.0
    assert _agreement_shrunk_fraction(0.0, 5.1) == 0.0
    with pytest.raises(ValueError):
        _agreement_shrunk_fraction(0.0, 0.0, translation_limit_m=0.0)
