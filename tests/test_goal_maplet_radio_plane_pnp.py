from __future__ import annotations

import numpy as np
import json
import pytest
import cv2

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _choose_token_hypotheses,
    _homography_filter,
    _homography_filter_with_bidirectional_projections,
    _homography_filter_with_query_projection,
    _homography_source_world_points,
    _pose_diagnostics,
    _pnp,
    _project_points_to_plane,
    _records,
    _region_tokens,
    _region_token_support,
    _scaled_intrinsics,
)
from feature_extract.tools.vfm.select_goal_maplet_sparse_occlusion_pnp_branch import (
    _carrier_pareto_mask,
)
from feature_extract.tools.vfm.build_goal_maplet_direct_radio_plane_ranking import (
    _base_to_carrier_regions,
    _consensus_supplement_ranking,
)
from feature_extract.tools.vfm.select_goal_maplet_sparse_occlusion_pnp_by_cross_island import (
    _cross_island_support,
)


def test_multiview_consensus_prioritizes_a_repeated_world_hypothesis() -> None:
    query_tokens = np.asarray([7, 7, 7, 7], np.int64)
    world_points = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        np.float64,
    )
    match_scores = np.asarray([0.80, 0.70, 0.95, 0.60], np.float64)
    # (query region, physical plane, source-atlas row).  The first two
    # observations come from independent views but support one world point.
    provenance = np.asarray([[1, 11, 0], [1, 11, 1], [1, 11, 2], [1, 11, 3]])
    view_names = np.asarray(["view_a", "view_b", "view_c", "view_d"])

    consensus = _choose_token_hypotheses(
        query_tokens,
        world_points,
        match_scores,
        provenance,
        view_names,
        maximum_per_token=2,
        selection="multiview_consensus",
        consensus_radius_m=0.5,
    )
    score_only = _choose_token_hypotheses(
        query_tokens,
        world_points,
        match_scores,
        provenance,
        view_names,
        maximum_per_token=2,
        selection="score",
        consensus_radius_m=0.5,
    )

    np.testing.assert_array_equal(consensus, [0, 2])
    np.testing.assert_array_equal(score_only, [2, 0])


def test_scaled_intrinsics_preserve_area_resize_pixel_center_phase() -> None:
    K, k1 = _scaled_intrinsics(2, np.asarray([800.0, 511.5, 287.5, 0.01]), 1024, 576)
    np.testing.assert_allclose(K, [[200.0, 0.0, 127.5], [0.0, 200.0, 71.5], [0.0, 0.0, 1.0]])
    assert k1 == 0.01


def test_pnp_recovers_identity_pose_from_token_centers() -> None:
    K = np.asarray([[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]])
    tokens = np.asarray([10 * 64 + 20, 10 * 64 + 40, 20 * 64 + 25, 20 * 64 + 45, 28 * 64 + 30, 28 * 64 + 50])
    pixels = np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
    depth = np.asarray([4.0, 5.0, 4.5, 6.0, 5.5, 7.0])
    world = np.c_[(pixels[:, 0] - K[0, 2]) / K[0, 0] * depth, (pixels[:, 1] - K[1, 2]) / K[1, 1] * depth, depth]
    pose, inlier = _pnp(world, tokens, K, 0.0)
    assert pose is not None and len(inlier) == len(tokens)
    np.testing.assert_allclose(pose, np.eye(4), atol=2e-5)


def test_pnp_recovers_identity_pose_on_68x120_radio_grid() -> None:
    grid = (68, 120)
    K = np.asarray([[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]])
    tokens = np.asarray([8 * 120 + 12, 8 * 120 + 90, 30 * 120 + 20,
                         30 * 120 + 100, 55 * 120 + 35, 55 * 120 + 105])
    pixels = np.c_[(tokens % 120 + .5) * 256 / 120 - .5,
                   (tokens // 120 + .5) * 144 / 68 - .5]
    depth = np.asarray([4.0, 5.0, 4.5, 6.0, 5.5, 7.0])
    world = np.c_[(pixels[:, 0] - K[0, 2]) / K[0, 0] * depth,
                  (pixels[:, 1] - K[1, 2]) / K[1, 1] * depth, depth]
    pose, inlier = _pnp(world, tokens, K, 0.0, token_grid=grid)
    assert pose is not None and len(inlier) == len(tokens)
    np.testing.assert_allclose(pose, np.eye(4), atol=2e-5)


def test_homography_projection_preserves_filter_and_returns_continuous_query_xy() -> None:
    query_xy = np.asarray([
        [5, 5], [10, 5], [15, 5], [5, 10], [10, 10], [15, 10],
        [5, 15], [10, 15], [15, 15], [20, 10], [20, 15], [20, 20],
    ], np.int64)
    # Integer token observations sampled from a slightly projective mapping
    # intentionally yield continuous inverse projections.
    homography = np.asarray([
        [1.05, 0.08, 3.2], [-0.04, 0.96, 2.7], [0.0015, -0.001, 1.0]
    ])
    mapped = cv2.perspectiveTransform(
        query_xy.astype(np.float64).reshape(-1, 1, 2), homography,
    ).reshape(-1, 2)
    map_xy = np.rint(mapped).astype(np.int64)
    query_token = query_xy[:, 1] * 64 + query_xy[:, 0]
    map_token = map_xy[:, 1] * 64 + map_xy[:, 0]

    legacy = _homography_filter(query_token, map_token)
    keep, projected = _homography_filter_with_query_projection(query_token, map_token)

    np.testing.assert_array_equal(keep, legacy)
    assert int(np.sum(keep)) >= 10
    assert np.all(np.isfinite(projected[keep]))
    assert np.any(np.abs(projected[keep] - query_xy[keep]) > 1e-3)


def test_homography_returns_continuous_forward_mapping_coordinates() -> None:
    query_xy = np.asarray([
        [5, 5], [10, 5], [15, 5], [5, 10], [10, 10], [15, 10],
        [5, 15], [10, 15], [15, 15], [20, 10], [20, 15], [20, 20],
    ], np.int64)
    transform = np.asarray([
        [1.02, 0.04, 2.3], [-0.02, 0.98, 1.7], [0.001, -0.0015, 1.0],
    ])
    continuous = cv2.perspectiveTransform(
        query_xy.astype(np.float64).reshape(-1, 1, 2), transform,
    ).reshape(-1, 2)
    map_xy = np.rint(continuous).astype(np.int64)
    query_token = query_xy[:, 1] * 64 + query_xy[:, 0]
    map_token = map_xy[:, 1] * 64 + map_xy[:, 0]
    keep, _, projected_map = _homography_filter_with_bidirectional_projections(
        query_token, map_token,
    )
    assert int(np.sum(keep)) >= 10
    assert np.all(np.isfinite(projected_map[keep]))
    assert np.any(np.abs(projected_map[keep] - map_xy[keep]) > 1e-3)


def test_homography_source_lift_uses_continuous_pixel_and_fails_closed_at_depth_edge() -> None:
    depth = np.full((8, 12), 5.0, np.float64)
    K = np.asarray([[10.0, 0.0, 5.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]])
    map_xy = np.asarray([[2.25, 1.5], [4.0, 2.0]])
    token_grid = (4, 6)
    pixel = np.c_[(map_xy[:, 0] + .5) * 12 / 6 - .5,
                  (map_xy[:, 1] + .5) * 8 / 4 - .5]
    expected = np.c_[(pixel[:, 0] - K[0, 2]) / K[0, 0] * 5.0,
                     (pixel[:, 1] - K[1, 2]) / K[1, 1] * 5.0,
                     np.full(2, 5.0)]
    reference = expected.copy()
    # The second sample is surrounded by a different surface and must retain
    # its original token-median point rather than crossing that discontinuity.
    x0, y0 = np.floor(pixel[1]).astype(int)
    depth[y0:y0 + 2, x0:x0 + 2] = 12.0
    lifted, used = _homography_source_world_points(
        depth, np.eye(4), K, 0.0, map_xy, reference, token_grid,
    )
    np.testing.assert_allclose(lifted[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(lifted[1], reference[1], atol=1e-12)
    assert used.tolist() == [True, False]


def test_pnp_accepts_continuous_query_pixels() -> None:
    K = np.asarray([[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]])
    tokens = np.asarray([10 * 64 + 20, 10 * 64 + 40, 20 * 64 + 25,
                         20 * 64 + 45, 28 * 64 + 30, 28 * 64 + 50])
    pixels = np.c_[(tokens % 64) * 4 + 1.8, (tokens // 64) * 4 + 1.2]
    depth = np.asarray([4.0, 5.0, 4.5, 6.0, 5.5, 7.0])
    world = np.c_[(pixels[:, 0] - K[0, 2]) / K[0, 0] * depth,
                  (pixels[:, 1] - K[1, 2]) / K[1, 1] * depth, depth]
    pose, inlier = _pnp(world, tokens, K, 0.0, query_pixel=pixels)
    assert pose is not None and len(inlier) == len(tokens)
    np.testing.assert_allclose(pose, np.eye(4), atol=2e-5)


def test_plane_projection_removes_only_normal_residual() -> None:
    points = np.asarray([[1.0, 2.0, 3.2], [-4.0, 5.0, 2.7]])
    projected = _project_points_to_plane(points, np.asarray([0.0, 0.0, 2.0]), 6.0)
    np.testing.assert_allclose(projected, [[1.0, 2.0, 3.0], [-4.0, 5.0, 3.0]])
    np.testing.assert_allclose(projected[:, :2], points[:, :2])


def test_region_tokens_use_area_projection_for_68x120() -> None:
    labels = np.full((144, 256), -1, np.int32)
    labels[:, :128] = 0
    tokens = _region_tokens(labels, 0, token_grid=(68, 120))
    assert len(tokens) == 68 * 60
    assert np.all(tokens.reshape(68, 60) % 120 < 60)


def test_region_token_support_retains_only_observed_fraction() -> None:
    labels = np.full((8, 8), -1, np.int32)
    labels[:4, :4] = 0
    labels[:2, 4:8] = 0
    token, weight = _region_token_support(labels, 0, token_grid=(2, 2))
    assert token.tolist() == [0, 1]
    assert np.allclose(weight, [1.0, 0.5])


def test_sparse_occlusion_carrier_requires_inlier_pareto_improvement() -> None:
    selected = _carrier_pareto_mask(
        np.asarray([True, True, True, False, True]),
        np.asarray([100, 100, 100, 100, 100]),
        np.asarray([50, 50, 50, 0, 50]),
        np.asarray([True, True, True, True, False]),
        np.asarray([100, 80, 120, 100, 100]),
        np.asarray([51, 50, 51, 8, 80]),
    )
    # 0 strictly improves count and ratio; 1 only improves ratio; 2 only
    # improves count; 3 is the usable fallback; 4 is unusable.
    assert selected.tolist() == [True, False, False, True, False]


def test_candidate_sharing_maps_connected_regions_without_filling_gap() -> None:
    base = np.asarray([[0, 0, -1, 1, 1], [0, 0, -1, 1, 1]], np.int32)
    carrier = np.asarray([[0, 0, -1, 0, 0], [0, 0, -1, 0, 0]], np.int32)
    assert _base_to_carrier_regions(base, carrier).tolist() == [0, 0]
    with pytest.raises(ValueError, match="observed support"):
        _base_to_carrier_regions(base, np.where(carrier < 0, 0, carrier))


def test_multi_island_consensus_preserves_top5_and_rejects_single_island_vote() -> None:
    own = np.asarray([.95, .90, .85, .80, .75, .10, .20, .05])
    partner = np.asarray([.05, .10, .20, .30, .40, .99, .15, .02])
    # Plane 5 is high only on the partner and must not supplement the base.
    ranking, supplement, votes = _consensus_supplement_ranking(
        own, np.stack((own, partner)), topk=6, preserve=5,
        vote_depth=3, minimum_votes=2,
    )
    assert ranking[:5].tolist() == [0, 1, 2, 3, 4]
    assert supplement == -1 and votes == 0

    second = partner.copy(); second[6] = 1.0
    own_with_vote = own.copy(); own_with_vote[6] = .74
    ranking, supplement, votes = _consensus_supplement_ranking(
        own_with_vote, np.stack((own_with_vote, second)), topk=6, preserve=5,
        vote_depth=6, minimum_votes=2,
    )
    assert ranking[:5].tolist() == [0, 1, 2, 3, 4]
    assert supplement == 6 and votes == 2
    assert ranking[5] == 6


def test_cross_island_support_requires_two_regions_on_same_physical_plane() -> None:
    # Regions 0 and 1 share carrier 0 and plane 7 with two inliers each.
    provenance = np.asarray([
        [0, 7, 10], [0, 7, 11], [1, 7, 12], [1, 7, 13],
        [2, 8, 14], [2, 8, 15], [3, 9, 16], [3, 9, 17],
    ], np.int64)
    result = _cross_island_support(
        np.arange(len(provenance)), provenance, np.asarray([0, 0, 1, 1]),
    )
    assert result == {
        "cross_island_plane_group_count": 1,
        "cross_island_inlier_count": 4,
        "cross_island_region_count": 2,
    }


def test_pose_diagnostics_measure_image_and_world_support_without_gt() -> None:
    K = np.asarray([[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]])
    tokens = np.asarray([5 * 64 + 5, 5 * 64 + 50, 30 * 64 + 8, 30 * 64 + 55])
    pixels = np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
    depth = np.asarray([4.0, 5.0, 6.0, 7.0])
    world = np.c_[
        (pixels[:, 0] - K[0, 2]) / K[0, 0] * depth,
        (pixels[:, 1] - K[1, 2]) / K[1, 1] * depth,
        depth,
    ]
    rows = [(0, 10, 1), (1, 11, 1), (2, 12, 2), (3, 13, 2)]
    view_names = np.asarray(["unused", "a", "b"])
    result = _pose_diagnostics(
        np.eye(4), np.arange(4), world, tokens, rows, K, 0.0,
        atlas_view_names=view_names,
    )
    assert result["inlier_query_hull_fraction"] > 0.25
    assert result["inlier_query_bbox_fraction"] > 0.25
    assert result["inlier_region_count"] == 4
    assert result["inlier_plane_count"] == 4
    assert result["inlier_source_view_count"] == 2
    assert result["inlier_reprojection_p90_px"] < 1e-8


def test_radio_manifests_must_be_disjoint(tmp_path) -> None:
    row = {"image_id": "seq1/frame00001.png"}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"records": [row]}))
    second.write_text(json.dumps({"records": [row]}))
    with pytest.raises(ValueError, match="duplicate"):
        _records([first, second])
