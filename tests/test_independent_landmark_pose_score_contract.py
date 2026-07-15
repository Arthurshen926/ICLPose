import numpy as np

from feature_extract.tools.vfm.eval_independent_landmark_pose_scores import (
    _best_correct_rank,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _maplet_purged_tracks,
    _verification_points_for_query,
)


def test_verification_points_are_disjoint_and_purge_all_fit_top_l_tracks() -> None:
    detector = {
        "image_ids": np.asarray(["query.png"]),
        "offsets": np.asarray([0, 4], dtype=np.int64),
        "xy": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        ),
        "global_descriptors": np.eye(4, dtype=np.float32),
        "detector_scores": np.ones((4,), dtype=np.float32),
    }
    proposals = {
        "query_ids": np.asarray(["query.png"] * 4),
        "coarse_scores": np.asarray(
            [[0.9, 0.8], [0.8, 0.7], [0.2, 0.1], [0.7, 0.6]],
            dtype=np.float32,
        ),
        "candidate_track_ids": np.asarray(
            [[10, 11], [12, 13], [14, 15], [16, 17]], dtype=np.int64
        ),
    }

    points, excluded_tracks, audit = _verification_points_for_query(
        "query.png",
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0, 1], dtype=np.int64),
        point_count=1,
        detector_log_merit_weight=0.0,
    )

    np.testing.assert_array_equal(points.source_row_indices, np.asarray([3]))
    np.testing.assert_array_equal(
        excluded_tracks, np.asarray([10, 11, 12, 13], dtype=np.int64)
    )
    assert audit == {
        "fit_query_point_count": 2,
        "available_unused_query_point_count": 2,
        "selected_verification_point_count": 1,
    }


def test_verification_points_use_aligned_learned_candidate_posterior() -> None:
    detector = {
        "image_ids": np.asarray(["query.png"]),
        "offsets": np.asarray([0, 2], dtype=np.int64),
        "xy": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "global_descriptors": np.eye(2, dtype=np.float32),
        "detector_scores": np.ones((2,), dtype=np.float32),
    }
    tracks = np.asarray([[10, 11], [12, 13]], dtype=np.int64)
    proposals = {
        "query_ids": np.asarray(["query.png", "query.png"]),
        "coarse_scores": np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        "candidate_track_ids": tracks,
    }
    overlay = {
        "candidate_track_ids": tracks.copy(),
        "candidate_probabilities": np.asarray(
            [[0.1, 0.2], [0.3, 0.4]], dtype=np.float32
        ),
        "null_probabilities": np.asarray([0.7, 0.3], dtype=np.float32),
    }

    points, _excluded, _audit = _verification_points_for_query(
        "query.png",
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0], dtype=np.int64),
        point_count=1,
        detector_log_merit_weight=0.0,
        candidate_prior_overlay=overlay,
    )

    np.testing.assert_allclose(points.candidate_descriptor_scores, [[0.3, 0.4]])
    np.testing.assert_allclose(points.candidate_null_probabilities, [0.3])


def test_maplet_purge_expands_only_clusters_touched_by_fit_tracks() -> None:
    purged = _maplet_purged_tracks(
        np.asarray([20, 99], dtype=np.int64),
        maplet_track_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
        maplet_cluster_ids=np.asarray([1, 1, 2, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(purged, np.asarray([10, 20, 99]))


def test_best_correct_rank_uses_frozen_score_order() -> None:
    rank = _best_correct_rank(
        np.asarray([0.9, 0.8, 0.7], dtype=np.float64),
        np.asarray([0.20, 0.03, 0.01], dtype=np.float64),
        np.asarray([0.1, 6.0, 0.2], dtype=np.float64),
        0.05,
    )

    assert rank == 3
