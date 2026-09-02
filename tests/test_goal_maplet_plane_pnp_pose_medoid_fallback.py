from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_pose_medoid_fallback import (
    _medoid_index,
    _pose_distance,
)


def _pose(center_x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = -center_x
    return pose


def test_pose_distance_uses_metric_camera_centres() -> None:
    assert _pose_distance(_pose(0.0), _pose(0.5)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        _pose_distance(_pose(0.0), _pose(0.5), translation_scale_m=0.0)


def test_medoid_prefers_consensus_and_ties_prefer_primary() -> None:
    index, distances = _medoid_index([_pose(0.0), _pose(1.0), _pose(1.1)])
    assert index == 1
    assert distances.shape == (3, 3)
    tie, _ = _medoid_index([_pose(0.0), _pose(1.0)])
    assert tie == 0
