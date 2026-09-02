from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_direct_plane_pnp_union_closure import (
    _closure_enabled,
    _normalized_pose_distance,
)


def _pose(center_x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = -center_x
    return pose


def test_normalized_pose_distance_uses_frozen_gate_units() -> None:
    assert _normalized_pose_distance(_pose(0.0), _pose(0.5)) == 1.0


def test_closure_gate_requires_low_support_and_bounded_motion() -> None:
    assert _closure_enabled(0.59, 1.99)
    assert not _closure_enabled(0.60, 1.99)
    assert not _closure_enabled(0.59, 2.0)
    assert not _closure_enabled(np.nan, 1.0)
