from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_plane_uv_pose_by_view_geometry import (
    _compatible_rows,
)


def test_view_geometry_gate_keeps_same_hemisphere_and_broad_scale() -> None:
    pose = np.eye(4, dtype=np.float64)
    world = np.asarray([[0, 0, 2], [0, 0, 2], [0, 0, 2], [0, 0, 2]], np.float64)
    # Surface-to-camera direction is -z and range is two metres.
    direction = np.asarray([
        [0, 0, -1], [0, 0, 1], [0, 0, -1], [0, 0, -1],
    ], np.float64)
    ranges = np.asarray([2.0, 2.0, 0.5, 8.0])
    rows, cosine, ratio = _compatible_rows(
        pose, world, np.arange(4), direction, ranges,
    )
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_allclose(cosine, [1, -1, 1, 1])
    np.testing.assert_allclose(ratio, [1, 1, 4, 0.25])
