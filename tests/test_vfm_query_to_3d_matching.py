import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkQualityConfig,
    LandmarkMapIndex,
    QueryTo3DMatchingConfig,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_query_tokens_to_landmarks,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    reprojection_error_stats,
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
