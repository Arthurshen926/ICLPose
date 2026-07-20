import numpy as np
import pytest

from feature_extract.tools.vfm.eval_independent_landmark_pose_scores import (
    _best_correct_rank,
    _selection_score_contract,
    _validate_candidate_spatial_materialization_contract,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _merge_hypothesis_artifacts,
    _mask_fixed_candidate_posterior_topk,
    _load_candidate_prior_overlay,
    _mixed_verification_points_for_query,
    _load_npz_fields,
    _maplet_purged_tracks,
    _spatial_materialization_audit,
    _strict_absolute_likelihood_config,
    _verification_points_for_query,
    parse_args,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    MixedVerificationPoints,
)
from feature_extract.tools.vfm.select_independent_landmark_pose_score_profile import (
    reselect_score_profile,
)


def test_hypothesis_merge_ignores_variable_width_diagnostic_edges(tmp_path) -> None:
    """The scorer's documented multi-shard input must not merge edge tensors."""

    def write_shard(path, *, query_id: str, hypothesis_index: int, edge_count: int) -> None:
        metadata = {
            "format": "grouped_pose_hypotheses_inference_only_v1",
            "contains_target_fields": False,
            "pose_or_ground_truth_used_for_generation": False,
            "row_count": 1,
            "inputs": {"candidate_artifact_sha256": "fixed"},
            "grouped_config": {"profile": "fixed"},
        }
        np.savez_compressed(
            path,
            query_ids=np.asarray([query_id]),
            split_names=np.asarray(["validation"]),
            evaluation_labels=np.asarray(["label"]),
            hypothesis_indices=np.asarray([hypothesis_index], dtype=np.int64),
            generation_profiles=np.asarray(["profile"]),
            selection_modes=np.asarray(["mode"]),
            shortlisted_for_verification=np.asarray([True]),
            chosen_for_optional_pose=np.asarray([False]),
            poses_w2c=np.eye(4, dtype=np.float64)[None],
            preliminary_log_likelihood_means=np.asarray([0.0], dtype=np.float64),
            shortlist_log_likelihood_means=np.asarray([0.0], dtype=np.float64),
            verification_log_likelihood_means=np.asarray([0.0], dtype=np.float64),
            # A real relation diagnostic has this shard-local variable edge
            # dimension.  It must never enter the pose scorer merge.
            verification_relation_feature_edge_histograms=np.zeros(
                (1, edge_count, 2), dtype=np.float32
            ),
            metadata_json=np.asarray(__import__("json").dumps(metadata)),
        )

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_shard(first, query_id="q1.png", hypothesis_index=0, edge_count=3)
    write_shard(second, query_id="q2.png", hypothesis_index=1, edge_count=5)

    merged, metadata, compatibility = _merge_hypothesis_artifacts((first, second))

    assert metadata["row_count"] == 1
    assert compatibility
    assert set(merged) == {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "shortlisted_for_verification",
        "chosen_for_optional_pose",
        "preliminary_log_likelihood_means",
        "poses_w2c",
    }
    assert merged["query_ids"].tolist() == ["q1.png", "q2.png"]


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


def test_mixed_verification_points_keep_their_frozen_candidates_and_fit_exclusion() -> None:
    metadata = {
        "format": MIXED_VERIFICATION_POINTS_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    mixed = MixedVerificationPoints(
        source_point_ids=np.asarray([100, 101], dtype=np.int64),
        query_ids=np.asarray(["query.png", "query.png"]),
        split_names=np.asarray(["validation", "validation"]),
        xy=np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        point_sources=np.asarray([POINT_SOURCE_ALIKE, POINT_SOURCE_RADIO_FINAL]),
        source_detector_rows=np.asarray([2, -1], dtype=np.int64),
        descriptors=np.eye(2, dtype=np.float32),
        candidate_bank_rows=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        candidate_track_ids=np.asarray([[20, 21], [22, 23]], dtype=np.int64),
        candidate_prototype_ids=np.zeros((2, 2), dtype=np.int64),
        candidate_coarse_similarities=np.asarray(
            [[0.9, 0.8], [0.7, 0.6]], dtype=np.float32
        ),
        candidate_prior_probabilities=np.asarray(
            [[0.4, 0.4], [0.5, 0.3]], dtype=np.float32
        ),
        null_probabilities=np.asarray([0.2, 0.2], dtype=np.float32),
        metadata=metadata,
    )
    detector = {
        "image_ids": np.asarray(["query.png"]),
        "offsets": np.asarray([0, 4], dtype=np.int64),
    }
    proposals = {
        "query_ids": np.asarray(["query.png"] * 4),
        "candidate_track_ids": np.asarray(
            [[10, 11], [12, 13], [14, 15], [16, 17]], dtype=np.int64
        ),
    }

    points, excluded, audit = _mixed_verification_points_for_query(
        "query.png",
        mixed_points=mixed,
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0, 1], dtype=np.int64),
    )

    np.testing.assert_array_equal(points.source_row_indices, [100, 101])
    np.testing.assert_allclose(points.candidate_descriptor_scores, [[0.4, 0.4], [0.5, 0.3]])
    np.testing.assert_allclose(points.candidate_null_probabilities, [0.2, 0.2])
    np.testing.assert_allclose(points.descriptor_reference_scores, [0.9, 0.7])
    np.testing.assert_array_equal(excluded, [10, 11, 12, 13])
    assert audit == {
        "fit_query_point_count": 2,
        "available_unused_query_point_count": 2,
        "selected_verification_point_count": 2,
        "mixed_alike_verification_point_count": 1,
        "mixed_radio_intermediate_verification_point_count": 0,
        "mixed_radio_final_verification_point_count": 1,
    }


def test_fixed_candidate_topk_transfers_only_removed_mass_to_null() -> None:
    tracks = np.asarray([[10, 11, 12, -1], [20, 21, 22, 23]], dtype=np.int64)
    probabilities = np.asarray(
        [[0.1, 0.4, 0.2, 0.0], [0.2, 0.1, 0.0, 0.1]], dtype=np.float32
    )
    null = np.asarray([0.3, 0.6], dtype=np.float32)

    masked, masked_null, retained = _mask_fixed_candidate_posterior_topk(
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=null,
        top_k=2,
    )

    np.testing.assert_array_equal(
        retained,
        np.asarray([[False, True, True, False], [True, True, False, False]]),
    )
    np.testing.assert_allclose(masked, [[0.0, 0.4, 0.2, 0.0], [0.2, 0.1, 0.0, 0.0]])
    np.testing.assert_allclose(masked_null, [0.4, 0.7])
    np.testing.assert_allclose(masked.sum(axis=1) + masked_null, 1.0)

    unchanged, unchanged_null, unchanged_retained = _mask_fixed_candidate_posterior_topk(
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=null,
        top_k=4,
    )
    np.testing.assert_array_equal(unchanged, probabilities)
    np.testing.assert_array_equal(unchanged_null, null)
    np.testing.assert_array_equal(unchanged_retained, tracks >= 0)
    with pytest.raises(ValueError, match="fixed_candidate_top_k"):
        _mask_fixed_candidate_posterior_topk(
            candidate_track_ids=tracks,
            candidate_probabilities=probabilities,
            null_probabilities=null,
            top_k=0,
        )


def test_spatial_artifact_checks_full_posterior_before_topk_mask() -> None:
    detector = {
        "image_ids": np.asarray(["query.png"]),
        "offsets": np.asarray([0, 2], dtype=np.int64),
        "xy": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "global_descriptors": np.eye(2, dtype=np.float32),
        "detector_scores": np.ones((2,), dtype=np.float32),
    }
    proposals = {
        "query_ids": np.asarray(["query.png", "query.png"]),
        "coarse_scores": np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        "candidate_track_ids": np.asarray([[10, 11], [12, 13]], dtype=np.int64),
    }
    # Candidate 13 is removed by a top-1 posterior ablation.  Its RGB mode
    # still carries the full frozen posterior provenance, while only the
    # effective verifier posterior receives zero mass and a larger null.
    effective_overlay = {
        "candidate_track_ids": proposals["candidate_track_ids"].copy(),
        "candidate_probabilities": np.asarray(
            [[0.4, 0.2], [0.3, 0.0]], dtype=np.float32
        ),
        "null_probabilities": np.asarray([0.4, 0.7], dtype=np.float32),
    }
    source_probabilities = np.asarray(
        [[0.4, 0.2], [0.3, 0.4]], dtype=np.float32
    )
    mode_index = {
        (1, 12): [(0, 1.0, 0.3, 1.0, 0.0, 0.0, 0, 0)],
        (1, 13): [(0, 1.0, 0.4, 1.0, 0.0, 0.0, 0, 1)],
    }
    log_maps = [np.zeros((2, 1), dtype=np.float16)]

    points, _excluded, _audit = _verification_points_for_query(
        "query.png",
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0], dtype=np.int64),
        point_count=1,
        detector_log_merit_weight=0.0,
        candidate_prior_overlay=effective_overlay,
        candidate_spatial_mode_index=mode_index,
        candidate_spatial_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
        candidate_spatial_log_probability_arrays=log_maps,
        candidate_spatial_source_probabilities=source_probabilities,
    )

    np.testing.assert_allclose(points.candidate_descriptor_scores, [[0.3, 0.0]])
    np.testing.assert_allclose(points.candidate_null_probabilities, [0.7])
    assert points.candidate_spatial_valid_mask is not None
    np.testing.assert_array_equal(points.candidate_spatial_valid_mask, [[[True], [True]]])

    with pytest.raises(ValueError, match="identity prior differs"):
        _verification_points_for_query(
            "query.png",
            detector=detector,
            proposals=proposals,
            selected_rows=np.asarray([0], dtype=np.int64),
            point_count=1,
            detector_log_merit_weight=0.0,
            candidate_prior_overlay=effective_overlay,
            candidate_spatial_mode_index=mode_index,
            candidate_spatial_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
            candidate_spatial_log_probability_arrays=log_maps,
        )


def test_verification_points_accept_fixed_target_free_selector_rows() -> None:
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

    points, _excluded, audit = _verification_points_for_query(
        "query.png",
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0], dtype=np.int64),
        point_count=2,
        detector_log_merit_weight=0.0,
        preselected_verification_rows=np.asarray([2, 3], dtype=np.int64),
    )

    np.testing.assert_array_equal(points.source_row_indices, [2, 3])
    assert audit["selected_verification_point_count"] == 2
    with pytest.raises(ValueError, match="held-out row set"):
        _verification_points_for_query(
            "query.png",
            detector=detector,
            proposals=proposals,
            selected_rows=np.asarray([0], dtype=np.int64),
            point_count=2,
            detector_log_merit_weight=0.0,
            preselected_verification_rows=np.asarray([0], dtype=np.int64),
        )


def test_spatial_materialization_audit_counts_points_and_views() -> None:
    from feature_extract.vfm.localization.independent_landmark_pose_likelihood import (
        IndependentVerificationPoints,
    )

    points = IndependentVerificationPoints(
        xy=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        descriptors=np.eye(2, dtype=np.float32),
        descriptor_reference_scores=np.ones((2,), dtype=np.float32),
        candidate_track_ids=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        candidate_descriptor_scores=np.full((2, 2), 0.25, dtype=np.float32),
        candidate_null_probabilities=np.asarray([0.5, 0.5], dtype=np.float32),
        candidate_spatial_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
        candidate_spatial_log_probabilities=np.zeros((2, 2, 2, 1), dtype=np.float32),
        candidate_spatial_dustbin_probabilities=np.ones((2, 2, 2), dtype=np.float32),
        candidate_support_view_probabilities=np.zeros((2, 2, 2), dtype=np.float32),
        candidate_spatial_valid_mask=np.asarray(
            [[[True, False], [False, False]], [[False, False], [True, True]]],
            dtype=bool,
        ),
    )

    assert _spatial_materialization_audit(points) == {
        "materialized_verification_point_count": 2,
        "materialized_candidate_view_count": 3,
    }


def test_eval_refuses_claimed_rgb_spatial_evidence_without_heldout_modes(tmp_path) -> None:
    path = tmp_path / "scores.npz"
    contract = {
        "candidate_specific_rgb_spatial_modes": True,
        "candidate_spatial_dustbin_and_missing_pose_independent": True,
        "candidate_spatial_omitted_topk_mass_is_null": True,
        "candidate_spatial_query_materialization": (
            "required_at_least_one_heldout_verification_point"
        ),
    }
    arrays = {
        "candidate_spatial_materialized_verification_point_counts": np.asarray(
            [4, 0], dtype=np.int64
        ),
        "candidate_spatial_materialized_candidate_view_counts": np.asarray(
            [12, 0], dtype=np.int64
        ),
    }

    with np.testing.assert_raises_regex(ValueError, "unmaterialized"):
        _validate_candidate_spatial_materialization_contract(
            path, arrays, contract, row_count=2
        )

    arrays["candidate_spatial_materialized_verification_point_counts"] = (
        np.asarray([4, 5], dtype=np.int64)
    )
    arrays["candidate_spatial_materialized_candidate_view_counts"] = np.asarray(
        [12, 15], dtype=np.int64
    )
    _validate_candidate_spatial_materialization_contract(
        path, arrays, contract, row_count=2
    )


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
    assert args.selection_statistic == "mean"


def test_explicit_selection_score_contract_is_not_silently_replaced() -> None:
    field, statistic = _selection_score_contract(
        {
            "selection": {
                "statistic": "spatial_median_of_means_2x2",
                "score_field": "independent_selection_scores",
            }
        },
        {"independent_log_likelihood_means", "independent_selection_scores"},
    )

    assert field == "independent_selection_scores"
    assert statistic == "spatial_median_of_means_2x2"
    with np.testing.assert_raises_regex(ValueError, "configured selection field"):
        _selection_score_contract(
            {"selection": {"score_field": "missing_scores"}},
            {"independent_log_likelihood_means"},
        )


def test_reselect_score_profile_changes_only_frozen_top1_and_score_field() -> None:
    arrays = {
        "query_ids": np.asarray(["q1", "q1", "q2", "q2"]),
        "split_names": np.asarray(["validation"] * 4),
        "evaluation_labels": np.asarray(["label"] * 4),
        "hypothesis_indices": np.asarray([3, 4, 5, 6], dtype=np.int64),
        "independent_score_top1": np.asarray([True, False, True, False]),
        "independent_selection_scores": np.asarray([0.8, 0.7, 0.9, 0.6]),
        "independent_log_likelihood_means": np.asarray([0.8, 0.7, 0.9, 0.6]),
        "independent_log_likelihood_medians": np.asarray([0.1, 0.2, 0.4, 0.3]),
    }
    metadata = {
        "selection": {
            "statistic_score_fields": {
                "mean": "independent_log_likelihood_means",
                "median": "independent_log_likelihood_medians",
            }
        }
    }

    output, output_metadata = reselect_score_profile(
        arrays, metadata, statistic="median"
    )

    np.testing.assert_array_equal(
        output["independent_score_top1"], [False, True, True, False]
    )
    np.testing.assert_allclose(output["independent_selection_scores"], [0.1, 0.2, 0.4, 0.3])
    assert output_metadata["selection"]["source_score_field"] == (
        "independent_log_likelihood_medians"
    )
    assert output_metadata["profile_transform"]["target_free"] is True


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


@pytest.mark.parametrize(
    ("overlay_format", "probability_semantics"),
    (
        (
            "candidate_image_context_prior_overlay_v1",
            "candidate_identity_probability_plus_explicit_null_equals_one",
        ),
        (
            "multiscale_candidate_probe_prior_overlay_v1",
            "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one",
        ),
        (
            "multiscale_candidate_probe_prior_overlay_v2",
            "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one",
        ),
        (
            "multiscale_candidate_probe_prior_overlay_v2",
            "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one",
        ),
    ),
)
def test_strict_scorer_accepts_target_free_image_context_overlay(
    tmp_path, overlay_format: str, probability_semantics: str
) -> None:
    proposals_path = tmp_path / "proposals.npz"
    tracks = np.asarray([[10, 11]], dtype=np.int64)
    np.savez(proposals_path, candidate_track_ids=tracks)
    from feature_extract.vfm.artifacts import file_sha256_short

    overlay_path = tmp_path / "overlay.npz"
    metadata = {
        "format": overlay_format,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "probability_semantics": probability_semantics,
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
    assert loaded_metadata["format"] == overlay_format
