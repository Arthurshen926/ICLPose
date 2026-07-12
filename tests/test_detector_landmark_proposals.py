from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.probe_detector_maplet_geometry import (
    _identity_metrics,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.detector_landmark_proposals import (
    candidate_reprojection_residuals,
    geometry_oracle_scores,
    nearest_visible_landmarks,
    rank_candidate_pool,
    summarize_detector_proposal_geometry,
    summarize_query_proposal_difficulty,
    summarize_ranked_detector_proposal_geometry,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _index() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 20], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.full((2,), 2, dtype=np.int64),
        observation_image_ids=(("a", "b"), ("a", "b")),
        reprojection_errors=np.zeros((2,), dtype=np.float32),
        feature_ambiguities=np.zeros((2,), dtype=np.float32),
        prototype_ids=np.zeros((2,), dtype=np.int64),
    )


def test_detector_geometry_uses_visible_projected_tracks() -> None:
    camera = ColmapCamera(1, 1, 100, 100, (50.0, 50.0, 50.0, 50.0))
    pose = np.eye(4, dtype=np.float64)
    points = np.asarray([[50.0, 50.0], [60.0, 50.0]], dtype=np.float32)
    rows, tracks, distances = nearest_visible_landmarks(points, _index(), pose, camera)
    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_array_equal(tracks, [10, 20])
    np.testing.assert_allclose(distances, 0.0, atol=1e-5)
    residuals = candidate_reprojection_residuals(
        points,
        np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        _index(),
        pose,
        camera,
    )
    np.testing.assert_allclose(residuals, [[10.0, 0.0], [10.0, 0.0]], atol=1e-5)


def test_detector_summary_conditions_recall_on_mappability() -> None:
    summary = summarize_detector_proposal_geometry(
        nearest_landmark_residuals=np.asarray([0.5, 1.5, 9.0]),
        candidate_residuals=np.asarray([[4.0, 0.5], [8.0, 1.5], [0.5, 9.0]]),
        query_ids=("a", "a", "b"),
        thresholds_px=(2.0,),
        top_ls=(1, 2),
    )
    metrics = summary["thresholds_px"]["2"]
    assert metrics["mappable_point_count"] == 2
    assert metrics["recall_at_1_given_mappable"] == 0.0
    assert metrics["recall_at_2_given_mappable"] == 1.0


def test_ranked_detector_summary_reorders_fixed_candidates() -> None:
    summary = summarize_ranked_detector_proposal_geometry(
        nearest_landmark_residuals=np.asarray([0.5]),
        candidate_residuals=np.asarray([[0.5, 9.0]], dtype=np.float32),
        candidate_scores=np.asarray([[0.1, 0.9]], dtype=np.float32),
        query_ids=("a",),
        thresholds_px=(1.0,),
        top_ls=(1, 2),
    )
    metrics = summary["thresholds_px"]["1"]
    assert metrics["recall_at_1_given_mappable"] == 0.0
    assert metrics["recall_at_2_given_mappable"] == 1.0


def test_identity_metrics_separates_pool_availability_from_score_ranking() -> None:
    summary = _identity_metrics(
        nearest_residuals=np.asarray([0.5], dtype=np.float32),
        candidate_residuals=np.asarray([[0.5, 9.0]], dtype=np.float32),
        scores=np.asarray([[0.1, 0.9]], dtype=np.float32),
        query_ids=np.asarray(["q"]),
        labels=np.asarray([[True, False]]),
        valid_edges=np.asarray([[True, True]]),
    )

    assert summary["geometry"]["thresholds_px"]["1"][
        "recall_at_1_given_mappable"
    ] == 0.0
    assert summary["proposal_pool_availability"]["thresholds_px"]["1"][
        "recall_at_1_given_mappable"
    ] == 1.0


def test_rank_candidate_pool_keeps_aligned_arrays_and_oracle_rejects_bad_rows() -> None:
    scores, tracks, residuals = rank_candidate_pool(
        candidate_scores=np.asarray([[0.1, 0.9, 0.5], [0.8, -np.inf, 0.7]]),
        top_l=2,
        arrays=(
            np.asarray([[10, 20, 30], [40, -1, 60]], dtype=np.int64),
            np.asarray([[0.5, 8.0, 1.5], [7.0, np.inf, 0.25]], dtype=np.float32),
        ),
    )
    np.testing.assert_allclose(scores, [[0.9, 0.5], [0.8, 0.7]])
    np.testing.assert_array_equal(tracks, [[20, 30], [40, 60]])
    np.testing.assert_allclose(residuals, [[8.0, 1.5], [7.0, 0.25]])
    oracle = geometry_oracle_scores(residuals, threshold_px=2.0)
    assert np.isneginf(oracle[0, 0])
    assert oracle[0, 1] == np.float32(-1.5)
    assert np.isneginf(oracle[1, 0])
    assert oracle[1, 1] == np.float32(-0.25)


def test_query_difficulty_reports_unique_tracks_and_grid_coverage() -> None:
    rows = summarize_query_proposal_difficulty(
        nearest_landmark_residuals=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        ranked_candidate_residuals=np.asarray(
            [[9.0, 0.5], [0.25, 8.0], [0.4, 0.3]], dtype=np.float32
        ),
        ranked_candidate_track_ids=np.asarray(
            [[99, 10], [10, 20], [10, 30]], dtype=np.int64
        ),
        query_ids=("q", "q", "q"),
        query_xy=np.asarray([[10.0, 10.0], [70.0, 10.0], [70.0, 70.0]]),
        image_sizes={"q": (100, 100)},
        thresholds_px=(1.0,),
    )
    metrics = rows[0]["thresholds_px"]["1"]
    assert metrics["positive_point_count"] == 3
    assert metrics["positive_unique_track_count"] == 2
    assert metrics["median_first_positive_rank"] == 1.0
    assert metrics["grid_cell_count"] == 3
    assert metrics["grid_coverage"] == 3.0 / 16.0
