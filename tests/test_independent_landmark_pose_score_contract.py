import numpy as np

from feature_extract.tools.vfm.eval_independent_landmark_pose_scores import (
    _best_correct_rank,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
    _load_npz_fields,
    _maplet_purged_tracks,
    _strict_absolute_likelihood_config,
    _verification_points_for_query,
    parse_args,
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


def test_absolute_score_cli_enforces_fixed_topl_with_explicit_null() -> None:
    args = parse_args(
        [
            "--hypothesis_artifacts",
            "hypotheses.npz",
            "--detector_query_cache",
            "detector.npz",
            "--proposals",
            "proposals.npz",
            "--candidate_artifact",
            "candidate.npz",
            "--fixed_candidate_prior_overlay",
            "prior.npz",
            "--projected_landmark_bank",
            "bank.npz",
            "--support_geometry_index",
            "views.npz",
            "--colmap_model_dir",
            "model",
            "--output_dir",
            "output",
        ]
    )

    config = _strict_absolute_likelihood_config(args)

    assert config.candidate_mode == "fixed_global_topl"
    assert config.fixed_candidate_prior_source == "learned_probability"
    assert config.maximum_view_angle_deg == 90.0


def test_absolute_score_cli_refuses_missing_explicit_null_overlay() -> None:
    with np.testing.assert_raises(SystemExit):
        parse_args(
            [
                "--hypothesis_artifacts",
                "hypotheses.npz",
                "--detector_query_cache",
                "detector.npz",
                "--proposals",
                "proposals.npz",
                "--candidate_artifact",
                "candidate.npz",
                "--projected_landmark_bank",
                "bank.npz",
                "--support_geometry_index",
                "views.npz",
                "--colmap_model_dir",
                "model",
                "--output_dir",
                "output",
            ]
        )


def test_inference_field_loader_does_not_materialize_supervision(tmp_path) -> None:
    path = tmp_path / "artifact.npz"
    np.savez(
        path,
        query_ids=np.asarray(["q"]),
        candidate_gt_residuals_px=np.asarray([[1.0]], dtype=np.float32),
        metadata_json=np.asarray('{"format":"test"}'),
    )

    arrays, metadata = _load_npz_fields(path, ("query_ids",))

    assert set(arrays) == {"query_ids"}
    assert metadata == {"format": "test"}


def test_strict_scorer_accepts_target_free_image_context_overlay(tmp_path) -> None:
    proposals_path = tmp_path / "proposals.npz"
    tracks = np.asarray([[10, 11]], dtype=np.int64)
    np.savez(proposals_path, candidate_track_ids=tracks)
    from feature_extract.vfm.artifacts import file_sha256_short

    overlay_path = tmp_path / "overlay.npz"
    metadata = {
        "format": "candidate_image_context_prior_overlay_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "probability_semantics": (
            "candidate_identity_probability_plus_explicit_null_equals_one"
        ),
        "proposals_sha256": file_sha256_short(proposals_path),
    }
    np.savez(
        overlay_path,
        candidate_track_ids=tracks,
        candidate_probabilities=np.asarray([[0.2, 0.3]], dtype=np.float32),
        null_probabilities=np.asarray([0.5], dtype=np.float32),
        metadata_json=np.asarray(__import__("json").dumps(metadata)),
    )

    arrays, loaded_metadata = _load_candidate_prior_overlay(
        overlay_path,
        proposals_path=proposals_path,
        proposals={"candidate_track_ids": tracks},
    )

    np.testing.assert_allclose(arrays["null_probabilities"], [0.5])
    assert loaded_metadata["format"] == "candidate_image_context_prior_overlay_v1"
