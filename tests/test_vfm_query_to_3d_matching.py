import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkQualityConfig,
    LandmarkAmbiguityPruningConfig,
    LandmarkMapIndex,
    LocalGeometricConsistencyConfig,
    MapReliabilityConfig,
    PoseRiskConfig,
    QueryTo3DMatch,
    QueryTo3DMatchingConfig,
    SpatialDiversityPnPConfig,
    deduplicate_pnp_matches,
    estimate_pose_pnp_fixed,
    estimate_pose_pnp_fixed_robust,
    estimate_pose_pnp_ransac,
    filter_matches_by_local_geometric_consistency,
    filter_landmarks_by_reference_images,
    landmark_submap_ambiguity_scores,
    map_reliability_scores,
    pose_risk_score,
    prune_ambiguous_landmarks,
    select_pnp_matches_by_spatial_diversity,
    selective_localization_summary,
    select_pnp_matches_by_map_reliability,
    match_query_tokens_to_landmarks,
    soft_order_pnp_matches,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    reprojection_error_stats,
    refit_pose_with_unique_query_inliers,
    select_unique_query_inlier_mask,
    match_spatial_distribution_stats,
    token_grid_xy,
    with_landmark_ambiguity_scores,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _xyz_from_xy(xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (xy[:, 0] - 50.0) / 80.0 * z
    y = (xy[:, 1] - 50.0) / 80.0 * z
    return np.stack([x, y, z], axis=1).astype(np.float64)


def _feature(dim: int, idx: int) -> np.ndarray:
    value = np.zeros((dim,), dtype=np.float32)
    value[idx] = 1.0
    return value


def test_token_grid_xy_supports_patch_center_coordinates() -> None:
    grid_xy = token_grid_xy(token_width=4, token_height=2, image_width=100, image_height=50, coordinate_mode="center")

    np.testing.assert_allclose(grid_xy[0], [12.0, 12.0])
    np.testing.assert_allclose(grid_xy[-1], [87.0, 37.0])
    assert np.all(grid_xy[:, 0] > 0.0)
    assert np.all(grid_xy[:, 0] < 99.0)
    assert np.all(grid_xy[:, 1] > 0.0)
    assert np.all(grid_xy[:, 1] < 49.0)


def test_token_grid_xy_defaults_to_legacy_edge_coordinates() -> None:
    grid_xy = token_grid_xy(token_width=4, token_height=2, image_width=100, image_height=50)

    np.testing.assert_allclose(grid_xy[0], [0.0, 0.0])
    np.testing.assert_allclose(grid_xy[-1], [99.0, 49.0])


def test_deduplicate_pnp_matches_keeps_first_track_occurrence() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([10.0, 10.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([20.0, 10.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([1.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.8,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=2,
            xy=np.asarray([90.0, 90.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.1,
            ratio=0.0,
            landmark_variance=0.0,
        ),
    ]

    unique_matches, original_indices = deduplicate_pnp_matches(matches)

    assert [match.track_id for match in unique_matches] == [1, 2]
    assert original_indices.tolist() == [0, 1]


def test_select_unique_query_inlier_mask_keeps_lowest_residual_candidate() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([50.0, 50.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.7,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([50.0, 50.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([1.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.99,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([66.0, 50.0], dtype=np.float64),
            track_id=3,
            xyz=np.asarray([1.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.8,
            ratio=0.0,
            landmark_variance=0.0,
        ),
    ]

    mask = select_unique_query_inlier_mask(matches, np.eye(4, dtype=np.float64), _camera(), np.ones((3,), dtype=bool))

    assert mask.tolist() == [True, False, True]


def test_refit_pose_with_unique_query_inliers_remaps_mask() -> None:
    xy = np.asarray(
        [
            [20.0, 20.0],
            [80.0, 20.0],
            [20.0, 80.0],
            [80.0, 80.0],
            [50.0, 35.0],
        ],
        dtype=np.float64,
    )
    xyz = _xyz_from_xy(xy, np.full((xy.shape[0],), 5.0, dtype=np.float64))
    matches = [
        QueryTo3DMatch(idx, xy[idx], idx + 10, xyz[idx], 0.9, 0.0, 0.0)
        for idx in range(xy.shape[0])
    ]
    bad_duplicate = QueryTo3DMatch(
        token_index=0,
        xy=xy[0],
        track_id=99,
        xyz=np.asarray([1.0, 0.0, 5.0], dtype=np.float64),
        similarity=0.99,
        ratio=0.0,
        landmark_variance=0.0,
    )
    matches.insert(1, bad_duplicate)
    initial = estimate_pose_pnp_fixed([matches[0], *matches[2:]], _camera(), refine_method="LM")
    initial_mask = np.ones((len(matches),), dtype=bool)

    refit = refit_pose_with_unique_query_inliers(
        matches,
        _camera(),
        initial.pose_w2c,
        initial_mask,
        min_inliers=4,
    )

    assert refit.success
    assert refit.inlier_mask.tolist() == [True, False, True, True, True, True]
    error = pnp_pose_error(refit.pose_w2c, np.eye(4, dtype=np.float64))
    assert error.translation_m < 1e-4


def test_query_tokens_match_landmarks_and_support_pnp() -> None:
    dim = 8
    height = 4
    width = 4
    grid_xy = token_grid_xy(width, height, image_width=100, image_height=100)
    chosen_token_indices = np.asarray([0, 3, 5, 6, 9, 12], dtype=np.int64)
    chosen_xy = grid_xy[chosen_token_indices]
    xyz = _xyz_from_xy(chosen_xy, np.asarray([4.0, 4.6, 5.2, 5.8, 6.4, 7.0], dtype=np.float64))
    query_map = np.zeros((dim, height, width), dtype=np.float32)
    features = []
    track_ids = []
    for local_idx, token_idx in enumerate(chosen_token_indices):
        track_id = 100 + local_idx
        track_ids.append(track_id)
        feat = _feature(dim, local_idx)
        features.append(feat)
        y_idx, x_idx = divmod(int(token_idx), width)
        query_map[:, y_idx, x_idx] = feat
    index = LandmarkMapIndex(
        track_ids=np.asarray(track_ids, dtype=np.int64),
        xyz=xyz,
        features=np.stack(features, axis=0),
        mean_variances=np.zeros((len(track_ids),), dtype=np.float32),
        observation_counts=np.full((len(track_ids),), 3, dtype=np.int64),
        observation_image_ids=tuple((("ref_a.png",),) for _ in track_ids),
    )

    matches = match_query_tokens_to_landmarks(
        query_map,
        index,
        QueryTo3DMatchingConfig(top_k=2, ratio_threshold=0.8, min_similarity=0.5, mutual=True),
        image_width=100,
        image_height=100,
    )

    assert len(matches) == len(track_ids)
    assert {match.track_id for match in matches} == set(track_ids)
    result = estimate_pose_pnp_ransac(matches, _camera(), reprojection_error_px=2.0, iterations=200)
    assert result.success
    assert result.inlier_count >= 5
    error = pnp_pose_error(result.pose_w2c, np.eye(4, dtype=np.float64))
    assert error.translation_m < 1e-4
    assert error.rotation_deg < 1e-3

    strict_result = estimate_pose_pnp_ransac(matches, _camera(), reprojection_error_px=2.0, iterations=200, min_inliers=7)
    assert not strict_result.success
    assert strict_result.pose_w2c is None
    assert strict_result.inlier_count == 0

    refined_result = estimate_pose_pnp_ransac(
        matches,
        _camera(),
        reprojection_error_px=2.0,
        iterations=200,
        pnp_method="EPNP",
        refine_method="LM",
    )
    assert refined_result.success
    refined_error = pnp_pose_error(refined_result.pose_w2c, np.eye(4, dtype=np.float64))
    assert refined_error.translation_m < 1e-4

    fixed_result = estimate_pose_pnp_fixed(matches, _camera(), pnp_method="EPNP", refine_method="LM")
    assert fixed_result.success
    assert fixed_result.inlier_count == len(matches)
    assert fixed_result.inlier_mask.tolist() == [True] * len(matches)
    fixed_error = pnp_pose_error(fixed_result.pose_w2c, np.eye(4, dtype=np.float64))
    assert fixed_error.translation_m < 1e-4
    assert fixed_error.rotation_deg < 1e-3

    with pytest.raises(ValueError, match="unsupported PnP method"):
        estimate_pose_pnp_ransac(matches, _camera(), pnp_method="BAD")


def test_robust_fixed_pnp_refinement_downweights_bad_measurements() -> None:
    rng = np.random.default_rng(7)
    clean_xy = np.asarray(
        [
            [20.0, 20.0],
            [80.0, 22.0],
            [25.0, 75.0],
            [75.0, 78.0],
            [50.0, 18.0],
            [18.0, 50.0],
            [82.0, 50.0],
            [50.0, 82.0],
            [35.0, 35.0],
            [65.0, 65.0],
        ],
        dtype=np.float64,
    )
    z = np.linspace(4.0, 8.0, clean_xy.shape[0], dtype=np.float64)
    xyz = _xyz_from_xy(clean_xy, z)
    measured = clean_xy + rng.normal(0.0, 0.25, size=clean_xy.shape)
    measured[-2:] += np.asarray([[25.0, -20.0], [-28.0, 18.0]], dtype=np.float64)
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=measured[idx],
            track_id=idx + 1,
            xyz=xyz[idx],
            similarity=0.9,
            ratio=0.0,
            landmark_variance=0.0,
        )
        for idx in range(clean_xy.shape[0])
    ]
    initial = np.eye(4, dtype=np.float64)
    initial[:3, 3] = np.asarray([0.08, -0.06, 0.04], dtype=np.float64)
    weights = np.ones((len(matches),), dtype=np.float64)
    weights[-2:] = 0.02

    unweighted = estimate_pose_pnp_fixed_robust(
        matches,
        _camera(),
        initial_pose_w2c=initial,
        weights=np.ones_like(weights),
        loss="linear",
        max_nfev=80,
    )
    weighted = estimate_pose_pnp_fixed_robust(
        matches,
        _camera(),
        initial_pose_w2c=initial,
        weights=weights,
        loss="huber",
        f_scale_px=2.0,
        max_nfev=80,
    )

    assert weighted.success
    assert weighted.inlier_count == len(matches)
    unweighted_error = pnp_pose_error(unweighted.pose_w2c, np.eye(4, dtype=np.float64))
    weighted_error = pnp_pose_error(weighted.pose_w2c, np.eye(4, dtype=np.float64))
    assert weighted_error.translation_m < unweighted_error.translation_m
    assert weighted_error.translation_m < 0.05


def test_soft_order_pnp_matches_keeps_matches_but_prioritizes_confident_inputs() -> None:
    base = QueryTo3DMatch(
        token_index=0,
        xy=np.array([0.0, 0.0], dtype=np.float64),
        track_id=1,
        xyz=np.array([0.0, 0.0, 3.0], dtype=np.float64),
        similarity=0.9,
        ratio=0.5,
        landmark_variance=0.1,
        similarity_margin=0.01,
        map_reliability=0.2,
    )
    reliable = QueryTo3DMatch(
        token_index=1,
        xy=np.array([1.0, 0.0], dtype=np.float64),
        track_id=2,
        xyz=np.array([1.0, 0.0, 3.0], dtype=np.float64),
        similarity=0.8,
        ratio=0.5,
        landmark_variance=0.1,
        similarity_margin=0.2,
        map_reliability=0.9,
    )

    ordered = soft_order_pnp_matches([base, reliable], mode="composite")
    assert [match.track_id for match in ordered] == [2, 1]
    assert ordered[0].pnp_soft_score is not None
    assert ordered[0].pnp_soft_score > ordered[1].pnp_soft_score

    top1 = soft_order_pnp_matches([base, reliable], mode="composite", max_matches=1)
    assert [match.track_id for match in top1] == [2]

    with pytest.raises(ValueError, match="unsupported soft PnP ordering mode"):
        soft_order_pnp_matches([base], mode="bad")


def test_soft_order_pnp_matches_uncertainty_penalizes_noisy_measurements() -> None:
    noisy = QueryTo3DMatch(
        token_index=0,
        xy=np.array([0.0, 0.0], dtype=np.float64),
        track_id=1,
        xyz=np.array([0.0, 0.0, 3.0], dtype=np.float64),
        similarity=0.9,
        ratio=0.5,
        landmark_variance=0.1,
        similarity_margin=0.2,
        measurement_sigma_px=16.0,
        pnp_uncertainty_scale=2.0,
    )
    crisp = QueryTo3DMatch(
        token_index=1,
        xy=np.array([1.0, 0.0], dtype=np.float64),
        track_id=2,
        xyz=np.array([1.0, 0.0, 3.0], dtype=np.float64),
        similarity=0.86,
        ratio=0.5,
        landmark_variance=0.1,
        similarity_margin=0.2,
        measurement_sigma_px=2.0,
        pnp_uncertainty_scale=1.0,
    )

    ordered = soft_order_pnp_matches([noisy, crisp], mode="uncertainty")

    assert [match.track_id for match in ordered] == [2, 1]
    assert ordered[0].pnp_soft_score is not None
    assert ordered[0].pnp_soft_score > ordered[1].pnp_soft_score


def test_matching_filters_by_ratio_mutual_and_landmark_variance() -> None:
    query_map = np.zeros((4, 1, 3), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    query_map[:, 0, 2] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        xyz=np.zeros((4, 3), dtype=np.float64),
        features=np.asarray(
            [
                [0.95, 0.3122499, 0.0, 0.0],
                [0.94, 0.341174, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mean_variances=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        observation_counts=np.ones((4,), dtype=np.int64),
        observation_image_ids=(("a",), ("a",), ("a",), ("a",)),
    )

    matches = match_query_tokens_to_landmarks(
        query_map,
        index,
        QueryTo3DMatchingConfig(
            top_k=2,
            ratio_threshold=0.8,
            min_similarity=0.5,
            mutual=True,
            max_landmark_variance=0.5,
        ),
        image_width=30,
        image_height=10,
    )

    assert [match.track_id for match in matches] == [3]


def test_landmark_quality_can_reweight_candidate_selection() -> None:
    query_map = np.zeros((2, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.zeros((2, 3), dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.90, 0.4358899]], dtype=np.float32),
        mean_variances=np.asarray([1.0, 0.0], dtype=np.float32),
        observation_counts=np.asarray([1, 8], dtype=np.int64),
        observation_image_ids=(("a",), ("a",) * 8),
        reprojection_errors=np.asarray([4.0, 0.1], dtype=np.float32),
    )

    matches = match_query_tokens_to_landmarks(
        query_map,
        index,
        QueryTo3DMatchingConfig(
            top_k=2,
            min_similarity=0.1,
            ratio_threshold=None,
            landmark_quality=LandmarkQualityConfig(enabled=True, min_score=0.0),
        ),
        image_width=10,
        image_height=10,
    )

    assert [match.track_id for match in matches] == [2]
    assert matches[0].landmark_quality is not None
    assert matches[0].quality_weighted_similarity is not None
    assert matches[0].landmark_quality > 0.75


def test_min_similarity_margin_rejects_ambiguous_matches() -> None:
    query_map = np.zeros((2, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.zeros((2, 3), dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.99995, 0.01]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64),
        observation_image_ids=(("a",), ("a",)),
    )

    matches = match_query_tokens_to_landmarks(
        query_map,
        index,
        QueryTo3DMatchingConfig(
            top_k=2,
            min_similarity=0.1,
            ratio_threshold=None,
            min_similarity_margin=0.01,
        ),
        image_width=10,
        image_height=10,
    )

    assert matches == []


def test_landmark_filters_use_reprojection_ambiguity_and_boundary() -> None:
    query_map = np.zeros((3, 1, 3), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    query_map[:, 0, 2] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.zeros((3, 3), dtype=np.float64),
        features=np.eye(3, dtype=np.float32),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64) * 3,
        observation_image_ids=(("a",), ("a",), ("a",)),
        reprojection_errors=np.asarray([0.1, 4.0, 0.1], dtype=np.float32),
        feature_ambiguities=np.asarray([0.1, 0.1, 0.95], dtype=np.float32),
    )

    matches = match_query_tokens_to_landmarks(
        query_map,
        index,
        QueryTo3DMatchingConfig(
            top_k=1,
            min_similarity=0.5,
            max_landmark_reprojection_error=1.0,
            max_landmark_ambiguity=0.5,
            min_distance_to_boundary_px=1.0,
        ),
        image_width=30,
        image_height=10,
    )

    assert matches == []


def test_scene_level_ambiguity_scores_survive_subsetting() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.zeros((3, 3), dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64),
        observation_image_ids=(("a",), ("a",), ("a",)),
    )

    scored = with_landmark_ambiguity_scores(index, reference_size=3, block_size=2)
    subset = scored.subset([0, 2])

    assert subset.feature_ambiguities[0] > subset.feature_ambiguities[1]


def test_reference_visibility_submap_keeps_only_tracks_observed_by_references() -> None:
    bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(1, np.ones((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32), 2, 1.0, ("ref_a.png",)),
            2: TrackFeature(2, np.ones((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32), 2, 1.0, ("ref_b.png",)),
            3: TrackFeature(3, np.ones((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32), 2, 1.0, ("ref_c.png",)),
        },
        feature_dim=2,
    )
    index = LandmarkMapIndex.from_track_bank(
        bank,
        xyz_by_track={1: np.zeros((3,)), 2: np.ones((3,)), 3: np.full((3,), 2.0)},
    )

    subset = filter_landmarks_by_reference_images(index, {"ref_b.png", "missing.png"})

    assert subset.track_ids.tolist() == [2]
    assert subset.xyz.tolist() == [[1.0, 1.0, 1.0]]


def test_reprojection_error_stats_reports_distribution_and_pnp_inlier_quality() -> None:
    grid_xy = np.asarray(
        [
            [20.0, 20.0],
            [80.0, 20.0],
            [20.0, 80.0],
            [80.0, 80.0],
        ],
        dtype=np.float64,
    )
    xyz = _xyz_from_xy(grid_xy, np.full((4,), 5.0, dtype=np.float64))
    matches = []
    for idx in range(4):
        xy = grid_xy[idx].copy()
        if idx == 3:
            xy += np.asarray([30.0, 0.0], dtype=np.float64)
        matches.append(
            type(
                "Match",
                (),
                {
                    "xyz": xyz[idx],
                    "xy": xy,
                },
            )()
        )

    stats = reprojection_error_stats(
        matches,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
        thresholds_px=(5.0, 16.0, 32.0),
        pnp_inlier_mask=np.asarray([True, True, False, True], dtype=bool),
    )

    assert stats["match_count"] == 4
    assert stats["gt_precision_5px"] == 0.75
    assert stats["gt_precision_16px"] == 0.75
    assert stats["gt_precision_32px"] == 1.0
    assert stats["gt_reproj_median_px"] == 0.0
    assert stats["pnp_inlier_count"] == 3
    assert stats["pnp_inlier_gt_precision_5px"] == 2.0 / 3.0
    assert stats["pnp_inlier_gt_precision_16px"] == 2.0 / 3.0
    assert stats["pnp_inlier_gt_precision_32px"] == 1.0


def test_pnp_residual_and_spatial_distribution_stats_expose_degeneracy() -> None:
    grid_xy = np.asarray(
        [
            [10.0, 10.0],
            [12.0, 12.0],
            [14.0, 14.0],
            [16.0, 16.0],
            [80.0, 80.0],
        ],
        dtype=np.float64,
    )
    xyz = _xyz_from_xy(grid_xy, np.full((5,), 5.0, dtype=np.float64))
    matches = [
        type("Match", (), {"xyz": xyz[idx], "xy": grid_xy[idx]})()
        for idx in range(grid_xy.shape[0])
    ]
    inlier_mask = np.asarray([True, True, True, True, False], dtype=bool)

    residual = pnp_reprojection_residual_stats(matches, np.eye(4, dtype=np.float64), _camera(), inlier_mask)
    spatial = match_spatial_distribution_stats(matches, 100, 100, inlier_mask)

    assert residual["pnp_reproj_inlier_count"] == 4
    assert residual["pnp_reproj_inlier_median_px"] < 1e-6
    assert spatial["count"] == 4
    assert spatial["bbox_area_frac"] < 0.01
    assert spatial["grid_4x4_occupancy_frac"] == 1.0 / 16.0
    assert spatial["xy_pca_minor_major_ratio"] < 1e-6


def test_map_reliability_scores_rank_stable_distinctive_tracks() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.zeros((3, 3), dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.8, 0.05, 0.02], dtype=np.float32),
        observation_counts=np.asarray([1, 8, 12], dtype=np.int64),
        observation_image_ids=(("a",), tuple("abcdefgh"), tuple("abcdefghijkl")),
        reprojection_errors=np.asarray([4.0, 0.5, 0.1], dtype=np.float32),
        feature_ambiguities=np.asarray([0.9, 0.6, 0.05], dtype=np.float32),
    )

    scores = map_reliability_scores(
        index,
        index.features,
        MapReliabilityConfig(
            enabled=True,
            track_weight=1.0,
            variance_weight=1.0,
            reprojection_weight=1.0,
            idf_weight=1.0,
            ambiguity_weight=1.0,
        ),
    )

    assert scores.shape == (3,)
    assert 0.0 <= float(np.min(scores)) <= float(np.max(scores)) <= 1.0
    assert int(index.track_ids[int(np.argmax(scores))]) == 3
    assert scores[2] > scores[1] > scores[0]


def test_select_pnp_matches_by_map_reliability_keeps_descriptor_order() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=np.asarray([float(idx), 0.0], dtype=np.float64),
            track_id=idx,
            xyz=np.asarray([0.0, 0.0, 5.0 + idx], dtype=np.float64),
            similarity=1.0 - idx * 0.1,
            ratio=0.0,
            landmark_variance=0.0,
            map_reliability=reliability,
        )
        for idx, reliability in [(1, 0.2), (2, 0.9), (3, 0.8), (4, 0.1)]
    ]

    selected = select_pnp_matches_by_map_reliability(matches, keep_fraction=0.5)

    assert [match.track_id for match in selected] == [2, 3]


def test_ambiguity_pruning_drops_duplicate_landmarks_before_matching() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        xyz=np.zeros((4, 3), dtype=np.float64),
        features=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.999, 0.01, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        mean_variances=np.zeros((4,), dtype=np.float32),
        observation_counts=np.ones((4,), dtype=np.int64) * 3,
        observation_image_ids=(("a",), ("a",), ("a",), ("a",)),
    )

    scores = landmark_submap_ambiguity_scores(index, close_similarity_threshold=0.95, block_size=2)
    pruned = prune_ambiguous_landmarks(
        index,
        LandmarkAmbiguityPruningConfig(enabled=True, drop_fraction=0.5, close_similarity_threshold=0.95),
    )

    assert scores[0] > scores[2]
    assert scores[1] > scores[3]
    assert pruned.track_ids.tolist() == [3, 4]


def test_spatial_diversity_pnp_selection_limits_matches_per_image_cell() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=np.asarray(xy, dtype=np.float64),
            track_id=idx,
            xyz=np.asarray([float(idx), 0.0, 5.0], dtype=np.float64),
            similarity=similarity,
            ratio=0.0,
            landmark_variance=0.0,
            similarity_margin=margin,
        )
        for idx, xy, similarity, margin in [
            (1, [10.0, 10.0], 0.9, 0.05),
            (2, [20.0, 20.0], 0.8, 0.20),
            (3, [80.0, 10.0], 0.7, 0.10),
            (4, [90.0, 20.0], 0.6, 0.30),
        ]
    ]

    selected = select_pnp_matches_by_spatial_diversity(
        matches,
        image_width=100,
        image_height=100,
        config=SpatialDiversityPnPConfig(enabled=True, grid_rows=1, grid_cols=2, max_per_cell=1, score_mode="margin"),
    )

    assert [match.track_id for match in selected] == [2, 4]


def test_spatial_diversity_pnp_selection_can_fill_for_depth_diversity() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=np.asarray([10.0 + float(idx), 10.0], dtype=np.float64),
            track_id=idx,
            xyz=np.asarray([float(idx), 0.0, depth], dtype=np.float64),
            similarity=0.9 - 0.01 * idx,
            ratio=0.0,
            landmark_variance=0.0,
            similarity_margin=0.5 - 0.01 * idx,
        )
        for idx, depth in [(1, 5.0), (2, 5.1), (3, 9.5)]
    ]

    selected = select_pnp_matches_by_spatial_diversity(
        matches,
        image_width=100,
        image_height=100,
        config=SpatialDiversityPnPConfig(
            enabled=True,
            grid_rows=1,
            grid_cols=1,
            max_per_cell=1,
            score_mode="margin",
            min_depth_range_m=3.0,
            max_matches=3,
        ),
    )

    assert [match.track_id for match in selected] == [1, 3]


def test_spatial_diversity_pnp_selection_can_fill_for_3d_covariance() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=np.asarray([10.0 + float(idx), 10.0 + float(idx)], dtype=np.float64),
            track_id=idx,
            xyz=np.asarray(xyz, dtype=np.float64),
            similarity=0.9 - 0.01 * idx,
            ratio=0.0,
            landmark_variance=0.0,
            similarity_margin=0.5 - 0.01 * idx,
        )
        for idx, xyz in [
            (1, [0.0, 0.0, 5.0]),
            (2, [1.0, 0.0, 5.0]),
            (3, [0.0, 1.0, 5.0]),
            (4, [0.0, 0.0, 7.0]),
        ]
    ]

    selected = select_pnp_matches_by_spatial_diversity(
        matches,
        image_width=100,
        image_height=100,
        config=SpatialDiversityPnPConfig(
            enabled=True,
            grid_rows=1,
            grid_cols=1,
            max_per_cell=3,
            score_mode="margin",
            min_planarity_ratio=0.01,
            max_matches=4,
        ),
    )

    assert [match.track_id for match in selected] == [1, 2, 3, 4]


def test_selective_localization_summary_reports_low_risk_coverage_metrics() -> None:
    rows = [
        {
            "pose_risk": 0.1,
            "success_25cm_10deg": True,
            "translation_error_m": 0.10,
            "rotation_error_deg": 1.0,
        },
        {
            "pose_risk": 0.2,
            "success_25cm_10deg": True,
            "translation_error_m": 0.20,
            "rotation_error_deg": 2.0,
        },
        {
            "pose_risk": 0.9,
            "success_25cm_10deg": False,
            "translation_error_m": 2.00,
            "rotation_error_deg": 20.0,
        },
    ]

    summary = selective_localization_summary(rows, coverages=(2.0 / 3.0, 1.0), success_key="success_25cm_10deg")

    assert summary["coverage_0.667"]["success_rate"] == 1.0
    assert np.isclose(summary["coverage_0.667"]["median_translation_error_m"], 0.15)
    assert summary["coverage_1.000"]["success_rate"] == 2.0 / 3.0


def test_pose_risk_score_increases_for_low_inlier_and_poor_spatial_coverage() -> None:
    good = pose_risk_score(
        {
            "pnp_inlier_count": 80,
            "pnp_inlier_ratio": 0.6,
            "patch_geometry": {"pnp_inlier_patch_at_1": 0.8},
            "pnp_reprojection": {"pnp_reproj_inlier_median_px": 2.0},
            "pnp_inlier_spatial": {"grid_4x4_occupancy_frac": 0.5, "depth_range_m": 4.0, "xyz_planarity_ratio": 0.1},
            "map_reliability": {"pnp_inliers": {"mean": 0.8}},
        },
        PoseRiskConfig(),
    )
    bad = pose_risk_score(
        {
            "pnp_inlier_count": 5,
            "pnp_inlier_ratio": 0.05,
            "patch_geometry": {"pnp_inlier_patch_at_1": 0.1},
            "pnp_reprojection": {"pnp_reproj_inlier_median_px": 40.0},
            "pnp_inlier_spatial": {"grid_4x4_occupancy_frac": 0.05, "depth_range_m": 0.1, "xyz_planarity_ratio": 0.0},
            "map_reliability": {"pnp_inliers": {"mean": 0.2}},
        },
        PoseRiskConfig(),
    )

    assert 0.0 <= good < bad <= 1.0


def test_local_geometric_consistency_filters_isolated_3d_outlier_without_reranking() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([10.0, 10.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.99,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=2,
            xy=np.asarray([25.0, 12.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([0.2, 0.0, 5.1], dtype=np.float64),
            similarity=0.95,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=3,
            xy=np.asarray([18.0, 24.0], dtype=np.float64),
            track_id=3,
            xyz=np.asarray([0.1, 0.2, 5.0], dtype=np.float64),
            similarity=0.90,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=4,
            xy=np.asarray([16.0, 18.0], dtype=np.float64),
            track_id=99,
            xyz=np.asarray([20.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.89,
            ratio=0.0,
            landmark_variance=0.0,
        ),
    ]

    filtered = filter_matches_by_local_geometric_consistency(
        matches,
        LocalGeometricConsistencyConfig(
            enabled=True,
            image_radius_px=32.0,
            xyz_radius_m=1.0,
            min_support=1,
        ),
    )

    assert [match.track_id for match in filtered] == [1, 2, 3]
    assert [match.local_consistency_support for match in filtered] == [2, 2, 2]
    assert all(match.local_consistency_score is not None for match in filtered)


def test_local_geometric_consistency_can_cap_input_matches_before_quadratic_filter() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=np.asarray([float(idx), 0.0], dtype=np.float64),
            track_id=idx,
            xyz=np.asarray([float(idx) * 0.1, 0.0, 5.0], dtype=np.float64),
            similarity=1.0 - idx * 0.01,
            ratio=0.0,
            landmark_variance=0.0,
        )
        for idx in range(6)
    ]

    filtered = filter_matches_by_local_geometric_consistency(
        matches,
        LocalGeometricConsistencyConfig(
            enabled=True,
            image_radius_px=10.0,
            xyz_radius_m=1.0,
            min_support=None,
            max_input_matches=3,
        ),
    )

    assert [match.track_id for match in filtered] == [0, 1, 2]
