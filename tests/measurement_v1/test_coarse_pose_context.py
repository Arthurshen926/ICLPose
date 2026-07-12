from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_selected_policy_coarse_pose_context import (
    _project_xyz,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera


def test_project_xyz_uses_only_estimated_pose_and_camera_intrinsics() -> None:
    camera = ColmapCamera(
        camera_id=1,
        model_id=1,
        width=100,
        height=80,
        params=(50.0, 50.0, 50.0, 40.0),
    )
    pose = np.eye(4, dtype=np.float64)
    xyz = np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 0.0, -1.0]])

    projected, in_front = _project_xyz(xyz, pose, camera)

    assert np.allclose(projected[:2], [[50.0, 40.0], [60.0, 40.0]])
    assert in_front.tolist() == [True, True, False]
