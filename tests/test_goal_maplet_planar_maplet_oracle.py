from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.localization_goal_maplet.planar_maplet_oracle import (
    fit_weighted_primitive_plane,
    solve_metric_translation,
    solve_rotation_from_plane_normals,
    solve_scaled_translation,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def test_plane_correspondences_recover_metric_pose_and_scale() -> None:
    normals_world = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 2.0, 1.0],
    ], np.float64)
    normals_world /= np.linalg.norm(normals_world, axis=1, keepdims=True)
    center_world = np.asarray([2.0, -3.0, 1.5])
    rotation_c2w = _rotation_z(0.37)
    offsets_world = np.asarray([4.0, -1.0, 3.0, 2.5, -2.0])
    offsets_query = offsets_world - normals_world @ center_world
    normals_query = (rotation_c2w.T @ normals_world.T).T
    weights = np.asarray([1.0, 2.0, 3.0, 1.5, 0.75])

    estimated_rotation, _ = solve_rotation_from_plane_normals(
        normals_query, normals_world, weights,
    )
    estimated_center, rank3, _ = solve_metric_translation(
        normals_world, offsets_world, offsets_query, weights,
    )
    scaled_center, scale, rank4, _ = solve_scaled_translation(
        normals_world, offsets_world, offsets_query, weights,
    )

    np.testing.assert_allclose(estimated_rotation, rotation_c2w, atol=1.0e-12)
    np.testing.assert_allclose(estimated_center, center_world, atol=1.0e-12)
    np.testing.assert_allclose(scaled_center, center_world, atol=1.0e-12)
    assert abs(scale - 1.0) < 1.0e-12
    assert rank3 == 3
    assert rank4 == 4


def test_scale_aware_plane_pose_needs_rank_four_not_three_matches() -> None:
    normals = np.eye(3, dtype=np.float64)
    offsets_world = np.asarray([2.0, 3.0, 4.0])
    offsets_query = np.asarray([0.5, 1.0, 1.5])
    _, _, rank, singular = solve_scaled_translation(
        normals, offsets_world, offsets_query, np.ones(3),
    )
    assert rank == 3
    assert singular.shape == (3,)


def test_single_2dgs_ellipse_has_a_well_defined_plane_normal() -> None:
    physical = SimpleNamespace(
        primitive_centers=np.asarray([[1.0, 2.0, 3.0]]),
        primitive_tangent1=np.asarray([[1.0, 0.0, 0.0]]),
        primitive_tangent2=np.asarray([[0.0, 1.0, 0.0]]),
        primitive_scale1=np.asarray([2.0]),
        primitive_scale2=np.asarray([1.0]),
    )
    normal, offset, eigenvalues = fit_weighted_primitive_plane(
        physical, np.asarray([0]), np.asarray([1.0]), np.asarray([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1.0e-15)
    assert offset == 3.0
    np.testing.assert_allclose(eigenvalues, [0.0, 0.25, 1.0], atol=1.0e-15)
