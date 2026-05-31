from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.lowlevel_offset_sidecar import (
    SuperPointKeypointSet,
    SuperPointSnapConfig,
    LowLevelOffsetSidecarConfig,
    LowLevelSupportBank,
    apply_superpoint_snaps_to_matches,
    apply_lowlevel_offsets_to_matches,
    estimate_ncc_patch_offset,
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
