import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    PatchTo3DMatchingConfig,
    build_patch_positive_sets,
    evaluate_patch_matches,
    filter_landmarks_by_projected_visibility,
    match_query_patches_to_landmarks,
    patch_positive_set_stats,
    token_patch_boxes,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    LandmarkQualityConfig,
    LocalGeometricConsistencyConfig,
    MapReliabilityConfig,
    token_grid_xy,
)


def test_patch_level_helpers_are_exported_from_vfm_package() -> None:
    from feature_extract.vfm import filter_landmarks_by_projected_visibility as exported_filter
    from feature_extract.vfm import patch_positive_set_stats as exported_stats

    assert exported_filter is filter_landmarks_by_projected_visibility
    assert exported_stats is patch_positive_set_stats


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _xyz_from_xy(xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (xy[:, 0] - 50.0) / 80.0 * z
    y = (xy[:, 1] - 50.0) / 80.0 * z
    return np.stack([x, y, z], axis=1).astype(np.float64)


def test_patch_positive_sets_count_any_visible_landmark_inside_token_patch() -> None:
    token_xy = token_grid_xy(3, 3, image_width=100, image_height=100)
    center_token = 4
    projected_xy = token_xy[center_token] + np.asarray([12.0, 0.0], dtype=np.float64)
    xyz = _xyz_from_xy(projected_xy[None, :], np.asarray([5.0], dtype=np.float64))
    index = LandmarkMapIndex(
        track_ids=np.asarray([10], dtype=np.int64),
        xyz=xyz,
        features=np.ones((1, 4), dtype=np.float32),
        mean_variances=np.zeros((1,), dtype=np.float32),
        observation_counts=np.ones((1,), dtype=np.int64),
        observation_image_ids=(("ref.png",),),
    )

    positives = build_patch_positive_sets(
        index,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
        token_width=3,
        token_height=3,
    )

    assert positives.by_token[center_token].track_ids == {10}
    assert positives.by_token[center_token].count == 1


def test_patch_positive_set_stats_report_density_and_empty_ratio() -> None:
    token_xy = token_grid_xy(3, 3, image_width=100, image_height=100)
    xyz = _xyz_from_xy(
        np.stack([token_xy[4], token_xy[4] + np.asarray([10.0, 0.0]), token_xy[0]], axis=0),
        np.asarray([5.0, 5.0, 5.0], dtype=np.float64),
    )
    index = LandmarkMapIndex(
        track_ids=np.asarray([10, 11, 12], dtype=np.int64),
        xyz=xyz,
        features=np.ones((3, 4), dtype=np.float32),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64),
        observation_image_ids=(("ref.png",), ("ref.png",), ("ref.png",)),
    )

    positives = build_patch_positive_sets(index, np.eye(4, dtype=np.float64), _camera(), 3, 3)
    stats = patch_positive_set_stats(positives)

    assert stats["token_count"] == 9
    assert stats["positive_landmark_count"] == 3
    assert stats["max_positives_per_token"] >= 2
    assert stats["zero_positive_token_ratio"] > 0.0
    assert stats["mean_positives_per_nonempty_token"] >= 1.0


def test_patch_evaluator_counts_patch_correct_when_pixel_gt5_fails() -> None:
    query_map = np.zeros((2, 3, 3), dtype=np.float32)
    query_map[:, 1, 1] = np.asarray([1.0, 0.0], dtype=np.float32)
    center_xy = token_grid_xy(3, 3, image_width=100, image_height=100)[4]
    projected_xy = center_xy + np.asarray([12.0, 0.0], dtype=np.float64)
    index = LandmarkMapIndex(
        track_ids=np.asarray([10], dtype=np.int64),
        xyz=_xyz_from_xy(projected_xy[None, :], np.asarray([5.0], dtype=np.float64)),
        features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((1,), dtype=np.float32),
        observation_counts=np.ones((1,), dtype=np.int64),
        observation_image_ids=(("ref.png",),),
    )
    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(top_k=1, min_similarity=0.5, match_mode="nn"),
        image_width=100,
        image_height=100,
    )
    positives = build_patch_positive_sets(index, np.eye(4, dtype=np.float64), _camera(), 3, 3)

    stats = evaluate_patch_matches(matches, positives, np.eye(4, dtype=np.float64), _camera(), stride_px=49.5)

    assert stats["patch_at_1"] == 1.0
    assert stats["gt_precision_5px"] == 0.0
    assert stats["gt_precision_stride"] == 1.0


def test_patch_evaluator_reports_pnp_inlier_patch_at_k() -> None:
    query_map = np.zeros((2, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.0, 1.0], dtype=np.float32)
    xy = token_grid_xy(2, 1, image_width=100, image_height=100)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=_xyz_from_xy(xy, np.asarray([5.0, 5.0], dtype=np.float64)),
        features=np.eye(2, dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64),
        observation_image_ids=(("ref.png",), ("ref.png",)),
    )
    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(top_k=1, min_similarity=0.5, match_mode="nn"),
        image_width=100,
        image_height=100,
    )
    positives = build_patch_positive_sets(index, np.eye(4, dtype=np.float64), _camera(), 2, 1)

    stats = evaluate_patch_matches(
        matches,
        positives,
        np.eye(4, dtype=np.float64),
        _camera(),
        stride_px=99.0,
        pnp_inlier_mask=np.asarray([True, False]),
        top_k=5,
    )

    assert stats["pnp_inlier_patch_at_1"] == 1.0
    assert stats["pnp_inlier_patch_at_5"] == 1.0


def test_soft_mutual_topk_keeps_patch_level_many_to_one_candidates() -> None:
    query_map = np.zeros((2, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.99, 0.01], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.zeros((2, 3), dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64),
        observation_image_ids=(("a",), ("a",)),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(top_k=1, mutual_top_k=2, min_similarity=0.5, match_mode="soft_mutual"),
        image_width=100,
        image_height=10,
    )

    assert [(match.token_index, match.track_id) for match in matches] == [(0, 1), (1, 1)]


def test_patch_matching_can_rank_by_landmark_quality_weighted_similarity() -> None:
    query_map = np.zeros((2, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.92, 0.39]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([1, 20], dtype=np.int64),
        observation_image_ids=(("ref.png",), tuple(f"ref_{idx}.png" for idx in range(20))),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=2,
            match_mode="nn",
            min_similarity=0.0,
            ratio_threshold=None,
            landmark_quality=LandmarkQualityConfig(
                enabled=True,
                track_weight=1.0,
                variance_weight=0.0,
                reprojection_weight=0.0,
                idf_weight=0.0,
                ambiguity_weight=0.0,
            ),
        ),
        image_width=100,
        image_height=100,
    )

    assert len(matches) == 1
    assert matches[0].track_id == 2
    assert matches[0].landmark_quality is not None
    assert matches[0].quality_weighted_similarity is not None


def test_patch_matching_can_rank_by_similarity_even_when_quality_is_enabled() -> None:
    query_map = np.zeros((2, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.92, 0.39]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([1, 20], dtype=np.int64),
        observation_image_ids=(("ref.png",), tuple(f"ref_{idx}.png" for idx in range(20))),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=2,
            match_mode="nn",
            min_similarity=0.0,
            ratio_threshold=None,
            match_score_mode="similarity",
            landmark_quality=LandmarkQualityConfig(
                enabled=True,
                track_weight=1.0,
                variance_weight=0.0,
                reprojection_weight=0.0,
                idf_weight=0.0,
                ambiguity_weight=0.0,
            ),
        ),
        image_width=100,
        image_height=100,
    )

    assert len(matches) == 1
    assert matches[0].track_id == 1


def test_patch_matching_can_rank_by_landmark_quality_only() -> None:
    query_map = np.zeros((2, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.7, 0.71]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([1, 20], dtype=np.int64),
        observation_image_ids=(("ref.png",), tuple(f"ref_{idx}.png" for idx in range(20))),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=2,
            match_mode="nn",
            min_similarity=0.0,
            ratio_threshold=None,
            match_score_mode="landmark_quality",
            landmark_quality=LandmarkQualityConfig(
                enabled=True,
                track_weight=1.0,
                variance_weight=0.0,
                reprojection_weight=0.0,
                idf_weight=0.0,
                ambiguity_weight=0.0,
            ),
        ),
        image_width=100,
        image_height=100,
    )

    assert len(matches) == 1
    assert matches[0].track_id == 2


def test_patch_matching_pairwise_head_can_filter_without_reranking() -> None:
    class DummyPairwiseScorer:
        def score_pairs(self, query_descriptors, landmark_descriptors):
            del query_descriptors
            return np.where(np.asarray(landmark_descriptors)[:, 1] > 0.5, 4.0, -4.0).astype(np.float32)

    query_map = np.zeros((2, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.0, 1.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64) * 3,
        observation_image_ids=(("a",), ("b",)),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=1,
            mutual_top_k=1,
            match_mode="mnn",
            min_similarity=0.0,
            ratio_threshold=None,
            pairwise_filter_keep_fraction=0.5,
        ),
        image_width=100,
        image_height=10,
        pairwise_inlier_scorer=DummyPairwiseScorer(),
    )

    assert [match.track_id for match in matches] == [2]
    assert matches[0].pairwise_inlier_logit == 4.0


def test_patch_matching_map_reliability_filter_does_not_rerank_descriptor_matches() -> None:
    query_map = np.zeros((2, 1, 3), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.8, 0.6], dtype=np.float32)
    query_map[:, 0, 2] = np.asarray([0.0, 1.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0], [0.0, 0.0, 7.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.8, 0.05, 0.02], dtype=np.float32),
        observation_counts=np.asarray([1, 4, 8], dtype=np.int64),
        observation_image_ids=(("a",), ("a", "b", "c", "d"), tuple("abcdefgh")),
        reprojection_errors=np.asarray([4.0, 0.4, 0.1], dtype=np.float32),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=1,
            mutual_top_k=1,
            match_mode="nn",
            min_similarity=0.0,
            ratio_threshold=None,
            match_score_mode="similarity",
            map_reliability=MapReliabilityConfig(
                enabled=True,
                track_weight=1.0,
                variance_weight=1.0,
                reprojection_weight=1.0,
                ambiguity_weight=0.0,
                filter_keep_fraction=2.0 / 3.0,
                uncertainty_min_scale=0.5,
                uncertainty_max_scale=2.0,
            ),
        ),
        image_width=100,
        image_height=10,
    )

    assert [match.track_id for match in matches] == [2, 3]
    assert matches[0].similarity > matches[1].similarity
    assert matches[0].map_reliability is not None
    assert matches[0].pnp_uncertainty_scale is not None
    assert matches[0].pnp_uncertainty_scale < 2.0


def test_patch_matching_map_reliability_min_score_filters_after_descriptor_selection() -> None:
    query_map = np.zeros((2, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.92, 0.39]], dtype=np.float32),
        mean_variances=np.asarray([1.0, 0.0], dtype=np.float32),
        observation_counts=np.asarray([1, 20], dtype=np.int64),
        observation_image_ids=(("a",), tuple(f"ref_{idx}" for idx in range(20))),
        reprojection_errors=np.asarray([5.0, 0.1], dtype=np.float32),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=1,
            match_mode="nn",
            min_similarity=0.0,
            ratio_threshold=None,
            match_score_mode="similarity",
            map_reliability=MapReliabilityConfig(enabled=True, min_score=0.5),
        ),
        image_width=100,
        image_height=100,
    )

    assert matches == []


def test_soft_mutual_topk_local_consistency_keeps_geometrically_supported_candidates() -> None:
    query_map = np.zeros((4, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 101, 102], dtype=np.int64),
        xyz=np.asarray(
            [
                [0.0, 0.0, 5.0],
                [0.2, 0.0, 5.0],
                [20.0, 0.0, 5.0],
                [-20.0, 0.0, 5.0],
            ],
            dtype=np.float64,
        ),
        features=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0],
                [0.01, 0.99, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mean_variances=np.zeros((4,), dtype=np.float32),
        observation_counts=np.ones((4,), dtype=np.int64) * 3,
        observation_image_ids=(("a",), ("a",), ("a",), ("a",)),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=2,
            mutual_top_k=2,
            match_mode="soft_mutual",
            min_similarity=0.0,
            ratio_threshold=None,
            match_score_mode="similarity",
            local_geometric_consistency=LocalGeometricConsistencyConfig(
                enabled=True,
                image_radius_px=120.0,
                xyz_radius_m=1.0,
                min_support=1,
            ),
        ),
        image_width=100,
        image_height=10,
    )

    assert [match.track_id for match in matches] == [1, 2]
    assert all(match.local_consistency_support == 1 for match in matches)


def test_token_patch_boxes_cover_stride_sized_regions() -> None:
    boxes = token_patch_boxes(token_width=3, token_height=3, image_width=100, image_height=100)

    center = boxes[4]
    assert center.center.tolist() == [49.5, 49.5]
    assert center.x0 < 49.5 < center.x1
    assert center.y0 < 49.5 < center.y1


def test_filter_landmarks_by_projected_visibility_keeps_only_gt_visible_points() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [10.0, 0.0, 5.0]], dtype=np.float64),
        features=np.ones((3, 2), dtype=np.float32),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64),
        observation_image_ids=(("a",), ("a",), ("a",)),
    )

    visible = filter_landmarks_by_projected_visibility(index, np.eye(4, dtype=np.float64), _camera())

    assert visible.track_ids.tolist() == [1]
