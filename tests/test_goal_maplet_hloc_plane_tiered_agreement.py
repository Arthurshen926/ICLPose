import numpy as np

from feature_extract.tools.vfm.fuse_goal_maplet_hloc_plane_tiered_agreement import (
    _fuse_hloc_with_plane_tiered_agreement,
)


def _pose(center_x):
    output = np.eye(4)
    output[0, 3] = -float(center_x)
    return output


def test_tiered_fusion_uses_midpoint_quarter_damping_and_hloc_default():
    hloc = np.stack([_pose(0.0)] * 4)
    plane = np.stack([_pose(0.4), _pose(1.0), _pose(3.0), _pose(0.2)])
    output, branch, fraction, translation, rotation = (
        _fuse_hloc_with_plane_tiered_agreement(
            hloc, plane, np.asarray([True, True, True, False]),
            0.5, 5.0, 2.0, 5.0, 0.25,
        )
    )
    np.testing.assert_array_equal(branch, [111, 112, 110, 110])
    np.testing.assert_allclose(fraction, [0.5, 0.25, 0.0, 0.0])
    np.testing.assert_allclose(output[0], _pose(0.2), atol=1e-12)
    np.testing.assert_allclose(output[1], _pose(0.25), atol=1e-12)
    np.testing.assert_allclose(output[2:], hloc[2:])
    np.testing.assert_allclose(translation[:3], [0.4, 1.0, 3.0])
    assert np.isinf(translation[3])
    np.testing.assert_allclose(rotation[:3], 0.0)
