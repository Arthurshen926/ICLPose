from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_pose_by_moge3_surface_registration import (
    _fit_surface_sim3,
    _token_points,
)


def test_surface_sim3_recovers_scale_and_camera_center() -> None:
    rng = np.random.default_rng(4)
    query = rng.normal(size=(80, 3)); query[:, 2] += 5.0
    center = np.asarray([0.2, -0.1, 0.3])
    world = 1.5 * query + center
    pose, scale, accepted, row = _fit_surface_sim3(np.eye(4), query, world)
    assert accepted
    np.testing.assert_allclose(scale, 1.5, atol=1e-6)
    np.testing.assert_allclose(-pose[:3, :3].T @ pose[:3, 3], center, atol=1e-6)
    assert row["final_median_m"] < 1e-6


def test_token_points_use_robust_block_center_and_reject_empty_block() -> None:
    points = np.zeros((144, 256, 3)); valid = np.zeros((144, 256), bool)
    points[:4, :4] = [1, 2, 3]; valid[:4, :4] = True
    output, keep = _token_points(points, valid, np.asarray([0, 1]))
    np.testing.assert_allclose(output[0], [1, 2, 3])
    np.testing.assert_array_equal(keep, [True, False])
