from __future__ import annotations

from dataclasses import replace

import numpy as np

from feature_extract.vfm.localization_goal_maplet.pose_transport_hierarchy import (
    build_pose_transport_hierarchy,
)
from test_goal_maplet_pure_retrieval import _physical


def test_geometry_hierarchy_is_symmetric_local_and_does_not_invent_support():
    physical = _physical()
    # Reuse the fully validated fixture while supplying a small interpretable
    # child geometry: 0--1 touch, while 2 and 3 are far away.
    physical = replace(
        physical,
        child_parent_rows=np.asarray([0, 0, 0, 0]),
        child_centers=np.asarray([[0, 0, 0], [0.6, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float),
        child_normals=np.asarray([[0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, 1]], dtype=float),
        child_frames=np.repeat(np.eye(3)[None], 4, axis=0),
        child_extents=np.ones((4, 3), dtype=float) * 0.5,
        maplet_child_offsets=np.asarray([0, 4], dtype=np.int64),
        child_member_offsets=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        child_member_primitive_rows=np.asarray([0, 1, 2, 3], dtype=np.int64),
        child_member_weights=np.ones(4, dtype=np.float32),
        child_member_local_uv=np.zeros((4, 2), dtype=np.float32),
    )
    hierarchy = build_pose_transport_hierarchy(physical, adjacency_gap_m=0.2)
    assert np.all(hierarchy.child_support_ids == -1)
    adjacency = [
        set(hierarchy.adjacency_child_rows[hierarchy.adjacency_offsets[i]:hierarchy.adjacency_offsets[i+1]].tolist())
        for i in range(4)
    ]
    assert adjacency == [{1}, {0}, set(), set()]


def test_geometry_hierarchy_rejects_unbounded_policy():
    with np.testing.assert_raises_regex(ValueError, "adjacency_gap"):
        build_pose_transport_hierarchy(_physical(), adjacency_gap_m=float("inf"))
