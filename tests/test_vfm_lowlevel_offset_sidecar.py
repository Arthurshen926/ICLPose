from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.lowlevel_offset_sidecar import (
    LandmarkConditionedKeypointSelectorConfig,
    SupportSuperPointKeypoint,
    SuperPointKeypointSet,
    SuperPointSnapConfig,
    LowLevelOffsetSidecarConfig,
    LowLevelSupportBank,
    build_landmark_conditioned_superpoint_candidate_rows,
    apply_superpoint_snaps_to_matches,
    apply_superpoint_gt_oracle_snaps_to_matches,
    apply_landmark_conditioned_superpoint_snaps_to_matches,
    apply_lowlevel_offsets_to_matches,
    estimate_ncc_patch_offset,
    select_landmark_conditioned_keypoint,
    select_support_superpoint_keypoint,
    superpoint_availability_summary,
    select_support_observation,
    snap_xy_to_superpoint_keypoint,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


def _match(track_id: int = 1, xy: tuple[float, float] = (16.0, 16.0)) -> QueryTo3DMatch:
    return QueryTo3DMatch(
        token_index=0,
        xy=np.asarray(xy, dtype=np.float64),
        track_id=track_id,
        xyz=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        similarity=0.9,
        ratio=0.0,
        landmark_variance=0.0,
    )


def _observation(
    track_id: int,
    image_id: str,
    ray: tuple[float, float, float],
    xy: tuple[float, float] = (10.0, 10.0),
) -> ColmapTrackObservation:
    return ColmapTrackObservation(
        track_id=track_id,
        image_id=image_id,
        point2d_idx=0,
        xy=xy,
        xyz=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        track_length=2,
        reprojection_error=0.5,
        viewing_ray=np.asarray(ray, dtype=np.float64),
    )


def test_support_bank_selects_closest_viewing_ray() -> None:
    bank = LowLevelSupportBank.from_observations(
        [
            _observation(1, "a.png", (1.0, 0.0, 0.0)),
            _observation(1, "b.png", (0.0, 0.0, 1.0)),
        ]
    )

    selected = select_support_observation(bank, 1, query_viewing_ray=np.asarray([0.0, 0.0, 1.0]))

    assert selected is not None
    assert selected.image_id == "b.png"


def test_estimate_ncc_patch_offset_recovers_integer_shift() -> None:
    query = np.zeros((33, 33), dtype=np.float32)
    support = np.zeros((33, 33), dtype=np.float32)
    support[14:19, 14:19] = 1.0
    query[16:21, 11:16] = 1.0

    result = estimate_ncc_patch_offset(
        query,
        support,
        query_xy=np.asarray([16.0, 16.0]),
        support_xy=np.asarray([16.0, 16.0]),
        config=LowLevelOffsetSidecarConfig(template_radius_px=4, search_radius_px=6, search_step_px=1),
    )

    assert result.applied
    np.testing.assert_allclose(result.offset_xy, [-3.0, 2.0], atol=1e-6)
    assert result.confidence > 0.0


def test_apply_lowlevel_offsets_keeps_low_confidence_and_non_inliers_at_patch_center() -> None:
    query = np.zeros((33, 33), dtype=np.float32)
    support = np.zeros((33, 33), dtype=np.float32)
    support[14:19, 14:19] = 1.0
    query[16:21, 11:16] = 1.0
    bank = LowLevelSupportBank.from_observations([_observation(1, "ref.png", (0.0, 0.0, 1.0), xy=(16.0, 16.0))])
    images = {"q.png": query, "ref.png": support}
    matches = [_match(1), _match(2, (20.0, 20.0))]

    refined, summary = apply_lowlevel_offsets_to_matches(
        matches,
        query_image_id="q.png",
        image_by_id=images,
        support_bank=bank,
        inlier_mask=np.asarray([True, False]),
        config=LowLevelOffsetSidecarConfig(template_radius_px=4, search_radius_px=6, search_step_px=1, min_confidence=0.01),
        query_viewing_ray_by_track={1: np.asarray([0.0, 0.0, 1.0])},
    )

    np.testing.assert_allclose(refined[0].xy, [13.0, 18.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [20.0, 20.0], atol=1e-6)
    assert summary["applied_count"] == 1
    assert summary["inlier_count"] == 1
    assert summary["offset_applied_ratio"] == 1.0
    assert summary["skipped_by_inlier_count"] == 1


def test_snap_xy_to_superpoint_keypoint_picks_high_score_nearby_keypoint() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0], [15.0, 15.0], [30.0, 30.0]], dtype=np.float32),
        scores=np.asarray([0.8, 0.3, 0.99], dtype=np.float32),
    )

    result = snap_xy_to_superpoint_keypoint(
        np.asarray([16.0, 16.0], dtype=np.float64),
        keypoints,
        SuperPointSnapConfig(max_offset_px=6.0, min_score=0.5),
    )

    assert result.applied
    np.testing.assert_allclose(result.offset_xy, [-3.0, 2.0], atol=1e-6)
    assert np.isclose(result.score, 0.8)


def test_superpoint_snap_respects_offset_bound_and_score_threshold() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[24.0, 16.0], [15.0, 15.0]], dtype=np.float32),
        scores=np.asarray([0.9, 0.2], dtype=np.float32),
    )

    result = snap_xy_to_superpoint_keypoint(
        np.asarray([16.0, 16.0], dtype=np.float64),
        keypoints,
        SuperPointSnapConfig(max_offset_px=4.0, min_score=0.5),
    )

    assert not result.applied
    np.testing.assert_allclose(result.offset_xy, [0.0, 0.0], atol=1e-6)
    assert result.reason == "no_keypoint_within_radius"


def test_superpoint_snap_can_prefer_nearest_keypoint_over_highest_score() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[10.0, 16.0], [15.0, 15.0]], dtype=np.float32),
        scores=np.asarray([0.9, 0.2], dtype=np.float32),
    )

    result = snap_xy_to_superpoint_keypoint(
        np.asarray([16.0, 16.0], dtype=np.float64),
        keypoints,
        SuperPointSnapConfig(max_offset_px=8.0, min_score=0.1, selection_strategy="nearest"),
    )

    assert result.applied
    np.testing.assert_allclose(result.offset_xy, [-1.0, -1.0], atol=1e-6)
    assert np.isclose(result.score, 0.2)


def test_apply_superpoint_snaps_only_refines_first_pass_inliers() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0], [20.0, 23.0]], dtype=np.float32),
        scores=np.asarray([0.9, 0.95], dtype=np.float32),
    )
    matches = [_match(1), _match(2, (20.0, 20.0))]

    refined, summary = apply_superpoint_snaps_to_matches(
        matches,
        keypoints,
        inlier_mask=np.asarray([True, False]),
        config=SuperPointSnapConfig(max_offset_px=6.0, min_score=0.5),
    )

    np.testing.assert_allclose(refined[0].xy, [13.0, 18.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [20.0, 20.0], atol=1e-6)
    assert summary["mode"] == "superpoint_snap"
    assert summary["applied_count"] == 1
    assert summary["inlier_count"] == 1
    assert summary["skipped_by_inlier_count"] == 1
    assert summary["offset_applied_ratio"] == 1.0


def test_superpoint_availability_summary_reports_radius_rates() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[10.0, 10.0], [30.0, 30.0]], dtype=np.float32),
        scores=np.asarray([0.8, 0.7], dtype=np.float32),
    )
    gt_xy = [np.asarray([12.0, 10.0]), np.asarray([40.0, 30.0]), None]
    mask = np.asarray([True, True, True])

    summary = superpoint_availability_summary(gt_xy, keypoints, mask, radii_px=(2.0, 8.0, 16.0))

    assert summary["availability_count"] == 2
    assert summary["sp_availability_at_2px"] == 0.5
    assert summary["sp_availability_at_8px"] == 0.5
    assert summary["sp_availability_at_16px"] == 1.0
    assert np.isclose(summary["nearest_sp_distance_px_mean"], 6.0)


def test_superpoint_gt_oracle_snaps_to_keypoint_nearest_gt_projection() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0], [15.0, 15.0], [30.0, 30.0]], dtype=np.float32),
        scores=np.asarray([0.2, 0.9, 0.99], dtype=np.float32),
    )
    matches = [_match(1), _match(2, (20.0, 20.0))]
    gt_xy = [np.asarray([13.2, 18.1]), np.asarray([20.0, 23.0])]

    refined, summary = apply_superpoint_gt_oracle_snaps_to_matches(
        matches,
        keypoints,
        inlier_mask=np.asarray([True, False]),
        gt_xy_by_match=gt_xy,
        max_distance_px=16.0,
    )

    np.testing.assert_allclose(refined[0].xy, [13.0, 18.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [20.0, 20.0], atol=1e-6)
    assert summary["mode"] == "superpoint_gt_oracle"
    assert summary["applied_count"] == 1
    assert summary["skipped_by_inlier_count"] == 1
    assert summary["mean_gt_to_sp_distance_px"] < 0.3


def test_superpoint_gt_oracle_can_restrict_to_patch_positive_matches() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0], [20.0, 23.0]], dtype=np.float32),
        scores=np.asarray([0.9, 0.95], dtype=np.float32),
    )
    matches = [_match(1), _match(2, (20.0, 20.0))]
    gt_xy = [np.asarray([13.0, 18.0]), np.asarray([20.0, 23.0])]

    refined, summary = apply_superpoint_gt_oracle_snaps_to_matches(
        matches,
        keypoints,
        inlier_mask=np.asarray([True, True]),
        gt_xy_by_match=gt_xy,
        max_distance_px=16.0,
        patch_positive_by_token={0: {1}},
        require_patch_positive=True,
    )

    np.testing.assert_allclose(refined[0].xy, [13.0, 18.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [20.0, 20.0], atol=1e-6)
    assert summary["applied_count"] == 1
    assert summary["skipped_by_patch_positive_count"] == 1


def test_select_support_superpoint_keypoint_binds_observation_to_descriptor() -> None:
    keypoints = SuperPointKeypointSet(
        xy=np.asarray([[12.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        scores=np.asarray([0.7, 0.9], dtype=np.float32),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    support = select_support_superpoint_keypoint(
        track_id=3,
        support_image_id="ref.png",
        observation_xy=np.asarray([10.0, 10.0], dtype=np.float64),
        keypoints=keypoints,
        max_distance_px=4.0,
    )

    assert support is not None
    assert support.track_id == 3
    assert support.support_image_id == "ref.png"
    np.testing.assert_allclose(support.xy, [12.0, 10.0], atol=1e-6)
    np.testing.assert_allclose(support.descriptor, [1.0, 0.0], atol=1e-6)


def test_landmark_conditioned_selector_uses_support_descriptor_not_strongest_keypoint() -> None:
    query_keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0], [15.0, 15.0]], dtype=np.float32),
        scores=np.asarray([0.2, 0.9], dtype=np.float32),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    support = SupportSuperPointKeypoint(
        track_id=1,
        support_image_id="ref.png",
        xy=np.asarray([11.0, 10.0], dtype=np.float64),
        descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
        score=0.8,
        distance_to_observation_px=1.0,
    )

    result = select_landmark_conditioned_keypoint(
        query_xy=np.asarray([16.0, 16.0], dtype=np.float64),
        query_keypoints=query_keypoints,
        support=support,
        config=LandmarkConditionedKeypointSelectorConfig(
            candidate_radius_px=8.0,
            min_query_score=0.0,
            descriptor_weight=2.0,
            query_score_weight=0.1,
            center_penalty_weight=0.0,
            score_threshold=0.5,
        ),
    )

    assert result.applied
    np.testing.assert_allclose(result.offset_xy, [-3.0, 2.0], atol=1e-6)
    assert result.reason == "applied"


def test_landmark_conditioned_selector_returns_no_snap_when_score_low() -> None:
    query_keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0]], dtype=np.float32),
        scores=np.asarray([0.2], dtype=np.float32),
        descriptors=np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    support = SupportSuperPointKeypoint(
        track_id=1,
        support_image_id="ref.png",
        xy=np.asarray([11.0, 10.0], dtype=np.float64),
        descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
        score=0.8,
        distance_to_observation_px=1.0,
    )

    result = select_landmark_conditioned_keypoint(
        query_xy=np.asarray([16.0, 16.0], dtype=np.float64),
        query_keypoints=query_keypoints,
        support=support,
        config=LandmarkConditionedKeypointSelectorConfig(candidate_radius_px=8.0, score_threshold=0.5),
    )

    assert not result.applied
    assert result.reason == "selector_below_threshold"


def test_apply_landmark_conditioned_snaps_reports_selection_accuracy() -> None:
    query_keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0], [20.0, 23.0]], dtype=np.float32),
        scores=np.asarray([0.8, 0.9], dtype=np.float32),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    support_by_track = {
        1: SupportSuperPointKeypoint(
            track_id=1,
            support_image_id="ref.png",
            xy=np.asarray([11.0, 10.0], dtype=np.float64),
            descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
            score=0.8,
            distance_to_observation_px=1.0,
        )
    }
    matches = [_match(1), _match(2, (20.0, 20.0))]

    refined, summary = apply_landmark_conditioned_superpoint_snaps_to_matches(
        matches,
        query_keypoints,
        support_by_track,
        inlier_mask=np.asarray([True, False]),
        config=LandmarkConditionedKeypointSelectorConfig(candidate_radius_px=8.0, score_threshold=0.1),
        gt_xy_by_match=[np.asarray([13.0, 18.0]), np.asarray([20.0, 23.0])],
    )

    np.testing.assert_allclose(refined[0].xy, [13.0, 18.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [20.0, 20.0], atol=1e-6)
    assert summary["mode"] == "landmark_conditioned_superpoint"
    assert summary["applied_count"] == 1
    assert summary["snap_selection_accuracy_at_4px"] == 1.0
    assert summary["sp_descriptor_top1_accuracy"] == 1.0


def test_landmark_conditioned_candidate_rows_include_no_snap_and_support_residual() -> None:
    query_keypoints = SuperPointKeypointSet(
        xy=np.asarray([[13.0, 18.0], [21.0, 21.0]], dtype=np.float32),
        scores=np.asarray([0.8, 0.9], dtype=np.float32),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    support_by_track = {
        1: SupportSuperPointKeypoint(
            track_id=1,
            support_image_id="ref.png",
            xy=np.asarray([11.0, 10.0], dtype=np.float64),
            observation_xy=np.asarray([10.0, 10.0], dtype=np.float64),
            descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
            score=0.8,
            distance_to_observation_px=1.0,
        )
    }

    rows = build_landmark_conditioned_superpoint_candidate_rows(
        matches=[_match(1)],
        query_keypoints=query_keypoints,
        support_by_track=support_by_track,
        inlier_mask=np.asarray([True]),
        config=LandmarkConditionedKeypointSelectorConfig(
            candidate_radius_px=8.0,
            min_query_score=0.0,
            descriptor_weight=2.0,
            query_score_weight=0.1,
            center_penalty_weight=0.0,
            support_distance_penalty_weight=0.0,
            score_threshold=0.1,
        ),
        gt_xy_by_match=[np.asarray([12.0, 18.0], dtype=np.float64)],
        query_id="query.png",
        stride_px=16.0,
    )

    no_snap = [row for row in rows if row["action"] == "no_snap"]
    candidates = [row for row in rows if row["action"] == "snap"]
    assert len(no_snap) == 1
    assert len(candidates) == 2
    assert candidates[0]["selected_by_heuristic"]
    np.testing.assert_allclose(candidates[0]["support_delta_xy"], [-1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(candidates[0]["residual_xy"], [12.0, 18.0], atol=1e-6)
    assert candidates[0]["snap_improves_center"]
    assert candidates[0]["snap_improvement_px"] > 0.0
