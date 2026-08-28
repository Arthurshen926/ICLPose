from types import SimpleNamespace

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_map_conditioned_plane_offset import (
    _fit as fit_map_region_offset,
    _fit_normal_calibration as fit_normal_calibration,
    _predict as predict_map_region_offset,
)

from feature_extract.vfm.localization_goal_maplet.planar_maplet_oracle import (
    convex_polygon_gap,
    fit_weighted_primitive_plane,
    select_normal_diverse_planes,
    solve_metric_translation,
    solve_robust_metric_translation,
    solve_rotation_from_plane_normals,
    solve_scaled_translation,
    solve_affine_scaled_translation,
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


def test_affine_scale_shift_plane_offsets_recover_metric_center() -> None:
    normals = np.asarray([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
    ])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    center = np.asarray([2.0, -1.0, 3.0])
    map_offset = np.asarray([4., 3., 8., 2., 6., 5.])
    scale, shift = 1.7, -0.4
    query_offset = (map_offset - normals @ center - shift) / scale
    estimate, got_scale, got_shift, rank, _ = solve_affine_scaled_translation(
        normals, map_offset, query_offset, np.ones(6),
    )
    np.testing.assert_allclose(estimate, center, atol=1e-12)
    assert abs(got_scale - scale) < 1e-12
    assert abs(got_shift - shift) < 1e-12
    assert rank == 5


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


def test_convex_polygon_gap_uses_boundaries_not_bounding_circles() -> None:
    square = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    separated = square + np.asarray([2.0, 0.0])
    crossing = np.asarray([[.5, -.5], [1.5, -.5], [1.5, .5], [.5, .5]])
    assert abs(convex_polygon_gap(square, separated) - 1.0) < 1.0e-12
    assert convex_polygon_gap(square, crossing) == 0.0


def test_diverse_selection_and_robust_translation_reject_offset_outlier() -> None:
    normals = np.asarray([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
    ])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    center = np.asarray([2.0, -1.0, 3.0])
    map_offset = np.linspace(1.0, 3.5, normals.shape[0])
    query_offset = map_offset - normals @ center
    query_offset[-1] += 4.0
    selected = select_normal_diverse_planes(normals, np.ones(6), 4)
    assert selected.size == 4
    assert np.linalg.matrix_rank(normals[selected]) == 3
    ordinary, _, _ = solve_metric_translation(normals, map_offset, query_offset, np.ones(6))
    robust, rank, _, _ = solve_robust_metric_translation(
        normals, map_offset, query_offset, np.ones(6), iterations=20,
    )
    assert rank == 3
    assert np.linalg.norm(robust - center) < np.linalg.norm(ordinary - center)


def test_map_region_offset_calibrator_has_explicit_unseen_fallback() -> None:
    data = {
        "query_offsets": np.asarray([0, 2, 4], np.int64),
        "region_rows": np.asarray([3, 7, 3, 7], np.int64),
        "query_offset": np.asarray([1.0, 2.0, 2.0, 3.0]),
        "ideal_query_offset": np.asarray([2.0, 1.0, 3.0, 2.0]),
        "weight": np.ones(4),
    }
    model = fit_map_region_offset(data, np.asarray([True, True]), 1.0e-6)
    predicted, seen = predict_map_region_offset(
        model, np.asarray([4.0, 4.0, 4.0]), np.asarray([3, 7, 99]),
    )
    assert seen.tolist() == [True, True, False]
    assert predicted[0] > predicted[1]
    assert np.isfinite(predicted).all()


def test_plane_normal_calibration_recovers_fixed_linear_bias() -> None:
    predicted = np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
    transform = np.asarray([[1., .1, 0.], [0., 1., .2], [.1, 0., 1.]])
    data = {"query_normal": predicted, "ideal_query_normal": predicted @ transform, "weight": np.ones(4)}
    recovered = fit_normal_calibration(data)
    np.testing.assert_allclose(recovered, transform, atol=1.0e-12)
