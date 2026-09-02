import numpy as np

from feature_extract.tools.vfm.fuse_goal_maplet_hloc_plane_agreement import (
    _fuse_hloc_with_plane_agreement,
)


def _pose(center_x):
    output = np.eye(4)
    output[0, 3] = -float(center_x)
    return output


def test_hloc_is_default_and_only_same_basin_pose_is_midpoint():
    hloc = np.stack([_pose(0.0), _pose(0.0), _pose(0.0)])
    plane = np.stack([_pose(0.4), _pose(1.0), _pose(0.2)])
    output, selected, translation, rotation = _fuse_hloc_with_plane_agreement(
        hloc, plane, np.asarray([True, True, False]), 0.5, 5.0
    )
    np.testing.assert_array_equal(selected, [True, False, False])
    np.testing.assert_allclose(output[0], _pose(0.2), atol=1e-12)
    np.testing.assert_allclose(output[1], hloc[1])
    np.testing.assert_allclose(output[2], hloc[2])
    np.testing.assert_allclose(translation[:2], [0.4, 1.0])
    assert np.isinf(translation[2])
    np.testing.assert_allclose(rotation[:2], 0.0)
