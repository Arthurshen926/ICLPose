from __future__ import annotations

import cv2
import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_moge3 import (
    _greedy_plane_associations,
    _refine_pose_scale,
)


def test_plane_associations_are_distinct_and_support_ordered() -> None:
    provenance = np.asarray(
        [[0, 4, 0]] * 6 + [[0, 5, 1]] * 4 + [[1, 4, 2]] * 5 + [[1, 6, 3]] * 3,
        np.int64,
    )
    result = _greedy_plane_associations(provenance, np.arange(len(provenance)))
    np.testing.assert_array_equal(result, [[0, 4, 6], [1, 6, 3]])


def test_joint_plane_scale_refinement_recovers_two_plane_pose() -> None:
    first = np.asarray([
        [-1.0, -1.0, 5.0], [0.0, -1.0, 5.0], [1.0, -1.0, 5.0],
        [-1.0, 1.0, 5.0], [0.0, 1.0, 5.0], [1.0, 1.0, 5.0],
    ])
    second = np.asarray([
        [1.0, -1.0, 3.0], [1.0, 0.0, 3.5], [1.0, 1.0, 4.0],
        [1.0, -1.0, 5.5], [1.0, 0.0, 6.0], [1.0, 1.0, 6.5],
    ])
    world = np.r_[first, second]
    K = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    pixel, _ = cv2.projectPoints(world, np.zeros(3), np.zeros(3), K, np.zeros(5))
    pixel = pixel.reshape(-1, 2)
    initial = np.eye(4)
    initial[:3, :3] = cv2.Rodrigues(np.asarray([0.0, np.deg2rad(1.0), 0.0]))[0]
    initial[:3, 3] = [0.04, -0.02, 0.03]
    provenance = np.c_[
        np.r_[np.zeros(6, np.int64), np.ones(6, np.int64)],
        np.r_[np.zeros(6, np.int64), np.ones(6, np.int64)],
        np.arange(12),
    ]
    refined, scale, accepted, _ = _refine_pose_scale(
        initial, world, pixel, provenance, np.arange(12),
        np.asarray([[0, 0, 6], [1, 1, 6]], np.int64),
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.asarray([5.0, 1.0]),
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.asarray([5.0, 1.0]),
        K, 0.0,
    )
    assert accepted
    assert np.linalg.norm(refined[:3, 3]) < np.linalg.norm(initial[:3, 3])
    assert abs(scale - 1.0) < 1e-3
