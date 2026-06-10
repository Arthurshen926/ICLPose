from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch
from feature_extract.vfm.rendered_pose_scoring import coverage_preserving_match_filter, score_pose_hypothesis


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _matches(sigma: float = 1.0) -> list[QueryTo3DMatch]:
    xyzs = [
        np.asarray([-1.0, -1.0, 8.0]),
        np.asarray([1.0, -1.0, 8.0]),
        np.asarray([-1.0, 1.0, 8.0]),
        np.asarray([1.0, 1.0, 8.0]),
    ]
    xys = [
        np.asarray([40.0, 40.0]),
        np.asarray([60.0, 40.0]),
        np.asarray([40.0, 60.0]),
        np.asarray([60.0, 60.0]),
    ]
    return [
        QueryTo3DMatch(
            token_index=idx,
            xy=xy,
            track_id=idx,
            xyz=xyz,
            similarity=0.9,
            ratio=0.0,
            landmark_variance=0.0,
            pnp_soft_score=0.8,
            measurement_sigma_px=sigma,
        )
        for idx, (xy, xyz) in enumerate(zip(xys, xyzs))
    ]


def test_score_pose_hypothesis_prefers_low_uncertainty_residuals() -> None:
    pose = np.eye(4, dtype=np.float64)
    good = score_pose_hypothesis(_matches(), pose, _camera(), inlier_threshold_px=8.0)
    shifted = pose.copy()
    shifted[0, 3] = 0.5
    bad = score_pose_hypothesis(_matches(), shifted, _camera(), inlier_threshold_px=8.0)

    assert good.score > bad.score
    assert good.inlier_count == 4


def test_score_pose_hypothesis_uses_measurement_uncertainty() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 0.25
    tight = score_pose_hypothesis(_matches(sigma=1.0), pose, _camera(), inlier_threshold_px=8.0)
    loose = score_pose_hypothesis(_matches(sigma=8.0), pose, _camera(), inlier_threshold_px=8.0)

    assert loose.score > tight.score


def test_coverage_preserving_match_filter_keeps_top_matches_per_grid_cell() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=np.asarray(xy, dtype=np.float64),
            track_id=idx,
            xyz=np.asarray([0.0, 0.0, 4.0 + idx], dtype=np.float64),
            similarity=float(sim),
            ratio=0.0,
            landmark_variance=0.0,
            pnp_soft_score=float(conf),
        )
        for idx, (xy, sim, conf) in enumerate(
            [
                ((10.0, 10.0), 0.1, 0.2),
                ((12.0, 12.0), 0.2, 0.9),
                ((80.0, 10.0), 0.3, 0.4),
                ((82.0, 12.0), 0.4, 0.8),
                ((80.0, 80.0), 0.5, 0.7),
            ]
        )
    ]

    kept = coverage_preserving_match_filter(matches, _camera(), grid_size=2, max_per_cell=1)

    assert [match.track_id for match in kept] == [1, 3, 4]


def test_coverage_preserving_match_filter_applies_min_confidence_without_emptying_everything() -> None:
    matches = _matches()
    low = [
        QueryTo3DMatch(
            token_index=match.token_index,
            xy=match.xy,
            track_id=match.track_id,
            xyz=match.xyz,
            similarity=match.similarity,
            ratio=match.ratio,
            landmark_variance=match.landmark_variance,
            pnp_soft_score=0.05,
        )
        for match in matches
    ]

    kept = coverage_preserving_match_filter(low, _camera(), grid_size=2, max_per_cell=2, min_confidence=0.5)

    assert len(kept) == len(low)
