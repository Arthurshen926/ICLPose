from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_phase_topk_physical_basins import (
    _physical_nms,
)


def _pose(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.radians(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    center = np.asarray([x, 0.0, 0.0])
    value[:3, 3] = -value[:3, :3] @ center
    return value


def test_phase_nms_excludes_anchor_and_removes_only_joint_physical_duplicates():
    poses = np.stack([
        _pose(99.0),  # diagnostic anchor, must never be selected
        _pose(0.0), _pose(0.25, 2.0), _pose(0.25, 8.0), _pose(1.0),
    ])
    retained, duplicates = _physical_nms(
        poses, np.asarray([100.0, 5.0, 4.0, 3.0, 2.0]), np.ones(5, dtype=bool),
        maximum_count=3, translation_threshold_m=0.5, rotation_threshold_deg=5.0,
    )
    assert retained == [1, 3, 4]
    assert duplicates == 1


def test_phase_nms_is_stable_at_score_ties_and_closed_boundaries():
    poses = np.stack([_pose(99.0), _pose(0.0), _pose(0.5, 5.0), _pose(0.50001, 5.0)])
    retained, duplicates = _physical_nms(
        poses, np.asarray([0.0, 1.0, 1.0, 1.0]), np.ones(4, dtype=bool),
        maximum_count=2, translation_threshold_m=0.5, rotation_threshold_deg=5.0,
    )
    assert retained == [1, 3]
    assert duplicates == 1
