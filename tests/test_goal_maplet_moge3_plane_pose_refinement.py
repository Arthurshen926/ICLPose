from __future__ import annotations

import cv2
import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_moge3 import (
    _conditional_scalar_information,
    _greedy_plane_associations,
    _many_to_one_plane_associations,
    _map_plane_balanced_association_weights,
    _reprojection_rows,
    _refine_pose_scale,
)


def test_plane_associations_are_distinct_and_support_ordered() -> None:
    provenance = np.asarray(
        [[0, 4, 0]] * 6 + [[0, 5, 1]] * 4 + [[1, 4, 2]] * 5 + [[1, 6, 3]] * 3,
        np.int64,
    )
    result = _greedy_plane_associations(provenance, np.arange(len(provenance)))
    np.testing.assert_array_equal(result, [[0, 4, 6], [1, 6, 3]])


def test_many_to_one_keeps_occlusion_fragments_without_double_assigning_region() -> None:
    provenance = np.asarray(
        [[0, 4, 0]] * 6 + [[0, 5, 1]] * 4 + [[1, 4, 2]] * 5 + [[2, 4, 3]] * 3,
        np.int64,
    )
    result = _many_to_one_plane_associations(provenance, np.arange(len(provenance)))
    np.testing.assert_array_equal(result, [[0, 4, 6], [1, 4, 5], [2, 4, 3]])


def test_many_to_one_fragment_weights_preserve_one_map_plane_mass() -> None:
    association = np.asarray([[0, 4, 6], [1, 4, 3], [2, 5, 7]], np.int64)
    weight = _map_plane_balanced_association_weights(association)
    np.testing.assert_allclose(weight * weight, [2.0 / 3.0, 1.0 / 3.0, 1.0])
    assert np.isclose(np.sum(np.square(weight[association[:, 1] == 4])), 1.0)


def test_reprojection_rows_fail_closed_for_nonfinite_unusable_pose() -> None:
    pose = np.eye(4)
    pose[0, 3] = np.nan
    world = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    pixel = np.asarray([[10.0, 10.0], [20.0, 10.0]])
    K = np.asarray([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    rows, error = _reprojection_rows(pose, world, pixel, K, 0.0)
    assert rows.dtype == np.int64
    assert len(rows) == 0
    assert np.all(np.isinf(error))


def test_conditional_scale_information_removes_pose_confounding() -> None:
    # Scalar is an exact copy of the nuisance column: no conditional evidence.
    confounded = np.asarray([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]])
    assert _conditional_scalar_information(confounded) < 1e-12
    # Orthogonal scalar/nuisance directions retain unit information.
    independent = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert abs(_conditional_scalar_information(independent) - 1.0) < 1e-12


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
