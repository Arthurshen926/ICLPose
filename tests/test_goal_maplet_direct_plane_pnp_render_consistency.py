from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _metric_depth_normal_agreement,
)
from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_pair_plan import (
    _pose_distance,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_selected_direct_plane_pnp import (
    _summary,
)
from feature_extract.tools.vfm.build_goal_maplet_pnp_pose_conditioned_view_context import (
    _candidate_context_score,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _choice_key,
    _score,
)
from feature_extract.tools.vfm.build_goal_maplet_pnp_pose_conditioned_spatial_radio import (
    _spatial_match_score,
)


def test_render_consistency_is_invariant_to_one_query_depth_scale() -> None:
    query = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    rendered = query * 7.0
    normal = np.zeros((2, 2, 3)); normal[..., 2] = 1.0
    result = _metric_depth_normal_agreement(
        rendered, normal, query, -normal, np.ones((2, 2), bool),
    )
    # Fewer than 16 pixels is deliberately fail-closed.
    assert result["fitted_query_depth_scale"] is None

    query = np.tile(query, (4, 4))
    rendered = query * 7.0
    normal = np.zeros((*query.shape, 3)); normal[..., 2] = 1.0
    result = _metric_depth_normal_agreement(
        rendered, normal, query, -normal, np.ones(query.shape, bool),
    )
    assert result["fitted_query_depth_scale"] == 7.0
    assert result["absolute_log_depth_p90"] < 1e-12
    assert result["normal_within_20deg"] == 1.0


def test_render_consistency_detects_spatial_depth_shape_error() -> None:
    query = np.ones((4, 4), np.float64)
    rendered = np.ones((4, 4), np.float64)
    rendered[:, 2:] = 2.0
    normal = np.zeros((4, 4, 3)); normal[..., 2] = 1.0
    result = _metric_depth_normal_agreement(
        rendered, normal, query, normal, np.ones((4, 4), bool),
    )
    assert result["absolute_log_depth_median"] > 0.3
    # The fitted median scale is 1.5 for an exactly tied bimodal ratio; both
    # halves remain outside the 20% band, so the spatial inconsistency cannot
    # be hidden by the single scale parameter.
    assert result["depth_ratio_within_20pct"] == 0.0


def test_render_consistency_separates_metric_scale_and_affine_depth_shape() -> None:
    query = np.geomspace(1.0, 12.0, 64).reshape(8, 8)
    rendered = 3.0 * query ** 1.2
    normal = np.zeros((8, 8, 3)); normal[..., 2] = 1.0
    result = _metric_depth_normal_agreement(
        rendered, normal, query, normal, np.ones(query.shape, bool),
    )
    assert result["metric_query_depth_scale_log_bias"] > 1.0
    assert result["absolute_log_depth_p90"] > 0.1
    assert abs(result["affine_log_depth_slope"] - 1.2) < 1e-10
    assert result["affine_log_depth_p90"] < 1e-10
    assert result["relative_log_depth_correlation"] > 0.999999


def test_render_coverage_denominator_is_full_query_valid_domain() -> None:
    query = np.ones((4, 8), np.float64)
    rendered = np.ones((4, 8), np.float64)
    rendered[:, 4:] = 0.0
    normal = np.zeros((4, 8, 3)); normal[..., 2] = 1.0
    result = _metric_depth_normal_agreement(
        rendered, normal, query, normal, np.ones(query.shape, bool),
    )
    assert result["query_valid_pixel_count"] == 32
    assert result["common_valid_pixel_count"] == 16
    assert result["render_coverage_of_query_valid"] == 0.5


def test_pair_plan_pose_distance_uses_camera_centers_and_rotation() -> None:
    left = np.eye(4)
    right = np.eye(4)
    right[0, 3] = -0.5
    angle = np.deg2rad(10.0)
    right[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]]
    # Set t=-R*C for a fixed world camera center C=(0.5,0,0).
    right[:3, 3] = -right[:3, :3] @ np.asarray([0.5, 0.0, 0.0])
    translation, rotation = _pose_distance(left, right)
    np.testing.assert_allclose(translation, 0.5, atol=1e-12)
    np.testing.assert_allclose(rotation, 10.0, atol=1e-12)


def test_selected_pose_summary_keeps_precision_and_system_recall_distinct() -> None:
    rows = [
        {"usable": True, "translation_error_m": 0.1, "rotation_error_deg": 1.0,
         "selected_inlier_ratio": 0.2, "selected_branch": 5},
        {"usable": True, "translation_error_m": 5.0, "rotation_error_deg": 1.0,
         "selected_inlier_ratio": 0.2, "selected_branch": 10},
        {"usable": True, "translation_error_m": 0.2, "rotation_error_deg": 1.0,
         "selected_inlier_ratio": 0.1, "selected_branch": 5},
    ]
    result = _summary(rows, 0.15)
    assert result["raw_recall_2m45"] == 2 / 3
    assert result["accepted_precision_2m45"] == 0.5
    assert result["selective_system_recall_2m45"] == 1 / 3


def test_pose_conditioned_context_uses_only_nearby_similarly_oriented_views() -> None:
    pose = np.eye(4)
    query = np.asarray([1.0, 0.0])
    centers = np.asarray([[1.0, 0.0, 0.0], [20.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    forwards = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    descriptors = np.asarray([[0.8, 0.6], [1.0, 0.0], [1.0, 0.0]])
    score, count = _candidate_context_score(
        pose, query, centers, forwards, descriptors,
        maximum_distance_m=10.0, maximum_direction_degrees=45.0, top_views=4,
    )
    assert count == 1
    np.testing.assert_allclose(score, 0.8)


def test_multihypothesis_balanced_support_does_not_prefer_one_dense_region() -> None:
    dense = {
        "inlier_count": 100, "region_capped_support": 8, "plane_capped_support": 8,
        "view_capped_support": 8, "supported_region_count": 1,
        "supported_plane_count": 1, "supported_view_count": 1,
        "reprojection_median_px": 0.5,
    }
    distributed = {
        "inlier_count": 30, "region_capped_support": 24, "plane_capped_support": 24,
        "view_capped_support": 24, "supported_region_count": 3,
        "supported_plane_count": 3, "supported_view_count": 3,
        "reprojection_median_px": 1.0,
    }
    assert _choice_key(dense, "raw_inliers") > _choice_key(distributed, "raw_inliers")
    assert _choice_key(distributed, "balanced_support") > _choice_key(dense, "balanced_support")
    assert _choice_key(distributed, "supported_entities") > _choice_key(dense, "supported_entities")


def test_spatial_radio_score_rewards_geometrically_consistent_matches() -> None:
    generator = np.random.default_rng(7)
    query = generator.normal(size=(18 * 32, 128)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    matched_count, _ = _spatial_match_score(query, query.copy())
    shuffled = query[generator.permutation(len(query))]
    shuffled_count, _ = _spatial_match_score(query, shuffled)
    assert matched_count == len(query)
    assert shuffled_count < matched_count // 4


def test_multihypothesis_score_counts_each_query_token_once() -> None:
    pose = np.eye(4)
    K = np.eye(3)
    world = np.asarray([
        [1.5, 1.5, 1.0],
        [1.5, 1.5, 1.0],  # duplicate 3D hypothesis for the same token
        [5.5, 1.5, 1.0],
    ])
    tokens = np.asarray([0, 0, 1])
    provenance = np.asarray([[0, 0, 0], [0, 1, 1], [1, 2, 2]])
    result = _score(pose, world, tokens, provenance, K, 0.0)
    assert result["inlier_count"] == 2
    assert result["inlier_ratio"] == 1.0
