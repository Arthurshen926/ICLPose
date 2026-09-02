import numpy as np

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_radio_sift_agreement import (
    _interpolate_pose,
    _pose_distance,
)


def _pose(center, yaw_deg):
    yaw = np.deg2rad(float(yaw_deg))
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    output = np.eye(4)
    output[:3, :3] = rotation
    output[:3, 3] = -rotation @ np.asarray(center, np.float64)
    return output


def test_pose_distance_and_midpoint_use_camera_centers_and_slerp():
    left = _pose([0.0, 0.0, 0.0], 0.0)
    right = _pose([0.4, 0.0, 0.0], 4.0)
    translation, rotation = _pose_distance(left, right)
    assert np.isclose(translation, 0.4)
    assert np.isclose(rotation, 4.0)
    midpoint = _interpolate_pose(left, right, 0.5)
    expected = _pose([0.2, 0.0, 0.0], 2.0)
    assert np.allclose(midpoint, expected, atol=1e-12)


def test_interpolation_endpoints_replay_inputs():
    left = _pose([1.0, 2.0, 3.0], -7.0)
    right = _pose([-2.0, 4.0, 5.0], 11.0)
    assert np.allclose(_interpolate_pose(left, right, 0.0), left, atol=1e-12)
    assert np.allclose(_interpolate_pose(left, right, 1.0), right, atol=1e-12)
