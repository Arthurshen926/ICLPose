import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_cross_coordinate_moge3_geometry_pose import (
    _frozen_plane_signs,
    _plane_geometry_objective,
    _select_surface,
)
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_moge3 import (
    _many_to_one_plane_associations,
    _map_plane_balanced_association_weights,
)


def test_surface_selection_is_strict_and_ties_fall_back_to_point():
    selected = _select_surface(
        np.asarray([1.0, 1.0, np.inf]),
        np.asarray([0.9, 1.0, np.inf]),
    )
    assert selected.tolist() == [1, 0, 0]


def test_frozen_sign_makes_plane_representation_sign_invariant():
    pose = np.eye(4)
    map_normal = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    query_normal = -map_normal
    signs = _frozen_plane_signs(pose, map_normal, query_normal)
    assert signs.tolist() == [-1.0, -1.0]
    first = _plane_geometry_objective(
        pose, map_normal, np.asarray([1.0, 2.0]), query_normal,
        np.asarray([-1.0, -2.0]), np.ones(2), signs,
    )
    second = _plane_geometry_objective(
        pose, -map_normal, -np.asarray([1.0, 2.0]), query_normal,
        np.asarray([-1.0, -2.0]), np.ones(2), -signs,
    )
    np.testing.assert_allclose(first[:2], second[:2], atol=1e-12)


def test_correct_pose_has_lower_two_plane_geometry_objective():
    normals = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offsets = np.asarray([-2.0, -3.0])
    query_normals = normals.copy()
    query_offsets = offsets.copy()
    correct = np.eye(4)
    shifted = np.eye(4); shifted[:3, 3] = [0.5, -0.5, 0.0]
    signs = _frozen_plane_signs(correct, normals, query_normals)
    correct_score, _, _ = _plane_geometry_objective(
        correct, normals, offsets, query_normals, query_offsets, np.ones(2), signs,
    )
    shifted_score, _, _ = _plane_geometry_objective(
        shifted, normals, offsets, query_normals, query_offsets, np.ones(2), signs,
    )
    assert correct_score < shifted_score


def test_single_plane_is_rejected_as_geometrically_degenerate():
    score, scale, detail = _plane_geometry_objective(
        np.eye(4), np.asarray([[1.0, 0.0, 0.0]]), np.asarray([-2.0]),
        np.asarray([[1.0, 0.0, 0.0]]), np.asarray([-2.0]),
        np.ones(1), np.ones(1),
    )
    assert np.isinf(score)
    assert scale == 1.0
    assert np.isinf(detail["normal_median_deg"])


def test_fragmented_query_regions_share_one_physical_plane_without_extra_weight():
    provenance = np.asarray([
        [0, 4, 1], [0, 4, 2], [0, 4, 3],
        [1, 4, 4], [1, 4, 5], [1, 4, 6],
        [2, 7, 7], [2, 7, 8], [2, 7, 9],
    ])
    associations = _many_to_one_plane_associations(
        provenance, np.arange(len(provenance)), minimum_support=3,
    )
    assert associations.tolist() == [[0, 4, 3], [1, 4, 3], [2, 7, 3]]
    weights = _map_plane_balanced_association_weights(associations)
    np.testing.assert_allclose(np.sum(np.square(weights[:2])), 1.0)
    np.testing.assert_allclose(np.sum(np.square(weights[2:])), 1.0)
