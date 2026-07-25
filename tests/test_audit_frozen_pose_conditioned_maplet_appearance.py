from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_frozen_pose_conditioned_maplet_appearance import (
    _assert_visual_control_pair,
    _assert_strict_contract,
    _canonical_hash,
    _score_contract,
)


def _metadata(*, variant: str) -> dict[str, object]:
    return {
        "query_id": "query.png",
        "inputs": {
            key: {"sha256": "same"}
            for key in (
                "detector_query_cache",
                "proposals",
                "candidate_artifact",
                "fixed_candidate_prior_overlay",
                "fixed_candidate_support_view_overlay",
                "maplet_support_index",
                "support_geometry_index",
                "projected_landmark_bank",
                "neighbor_topology_cache",
                "radio_final_context_cache",
                "radio_intermediate_context_cache",
                "alike_spatial_context_cache",
                "colmap_cameras_bin",
                "colmap_images_bin_camera_ownership_only",
            )
        },
        "strict_frozen_maplet_appearance_contract": {
            "heldout_query_rows": True,
            "heldout_query_image_content_excludes_pnp_fit_neighborhoods": True,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": 20,
            "candidate_identity_fixed_across_hypotheses": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "candidate_support_view_posterior_fixed_before_pose_scoring": True,
            "candidate_group_latent_identity_marginalized": True,
            "candidate_group_topl_denominator_fixed": True,
            "candidate_group_explicit_null": True,
            "support_maplet_center_excluded": True,
            "support_maplet_topology_fixed": True,
            "maplet_neighbor_identity_fixed": True,
            "pose_dependent_correspondence_selection": False,
            "fit_neighborhood_missing_evidence_penalized": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "raw_scores_calibrated_or_promoted": False,
            "raw_scores_must_not_feed_pnp": True,
            "support_descriptor_permutation_control": variant == "support_descriptor_permutation_control",
            "xyz_permutation_control": False,
        },
        "maplet_config": {"profiles": ["profile"], "fit_exclusion_mask_fraction": 0.1},
        "input_metadata": {"format": "test"},
        "evidence_layout_digest": {"fixed": "digest"},
        "profile_static_layout": {
            "profile": {
                "profile": "profile",
                "static_neighbor_count": 10,
                "static_candidate_view_with_neighbor_count": 5,
                "neighbor_track_ids_sha256": "track",
                "neighbor_xyz_sha256": "xyz",
                "neighbor_support_xy_sha256": "xy",
                "neighbor_valid_sha256": "valid",
                "support_descriptor_derangement": None
                if variant == "visual"
                else "permutation",
            }
        },
    }


def _arrays(*, score: float) -> dict[str, np.ndarray]:
    return {
        "query_ids": np.asarray(["query.png"]),
        "split_names": np.asarray(["validation"]),
        "evaluation_labels": np.asarray(["label"]),
        "hypothesis_indices": np.asarray([0], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray([False]),
        "baseline_score_top1": np.asarray([True]),
        "baseline_selection_scores": np.asarray([0.0]),
        "verification_source_row_indices": np.asarray([1], dtype=np.int64),
        "verification_xy": np.asarray([[2.0, 3.0]], dtype=np.float32),
        "fit_query_xy": np.asarray([[4.0, 5.0]], dtype=np.float32),
        "candidate_track_ids": np.arange(20, dtype=np.int64)[None],
        "candidate_probabilities": np.full((1, 20), 0.05, dtype=np.float32),
        "null_probabilities": np.asarray([0.0], dtype=np.float32),
        "support_view_probabilities": np.full((1, 20, 2), 0.5, dtype=np.float32),
        "support_image_ids": np.full((1, 20, 2), "support.png"),
        "family_names": np.asarray(["family"]),
        "family_log_likelihood_means": np.asarray([[score]], dtype=np.float64),
        "family_log_likelihood_medians": np.asarray([[score]], dtype=np.float64),
        "family_log_likelihood_worst_quartile_means": np.asarray([[score]], dtype=np.float64),
        "family_spatial_median_of_means_2x2": np.asarray([[score]], dtype=np.float64),
    }


def test_visual_control_pair_requires_same_frozen_layout_and_changed_scores() -> None:
    visual = _arrays(score=0.0)
    control = _arrays(score=1.0)
    _assert_visual_control_pair(
        visual_arrays=visual,
        visual_metadata=_metadata(variant="visual"),
        control_arrays=control,
        control_metadata=_metadata(variant="support_descriptor_permutation_control"),
    )
    control["candidate_track_ids"][0, 0] = 999
    with pytest.raises(ValueError, match="frozen field differs"):
        _assert_visual_control_pair(
            visual_arrays=visual,
            visual_metadata=_metadata(variant="visual"),
            control_arrays=control,
            control_metadata=_metadata(variant="support_descriptor_permutation_control"),
        )


def test_strict_contract_rejects_wrong_control_flag() -> None:
    metadata = _metadata(variant="visual")
    _assert_strict_contract(metadata, expected_variant="visual")
    with pytest.raises(ValueError, match="control flags"):
        _assert_strict_contract(metadata, expected_variant="support_descriptor_permutation_control")


def test_score_contract_keeps_fit_mask_coverage_per_query_not_cross_shard_config() -> None:
    first = _metadata(variant="visual")
    second = _metadata(variant="visual")
    second["maplet_config"] = {"profiles": ["profile"], "fit_exclusion_mask_fraction": 0.4}
    assert _canonical_hash(_score_contract(first)) == _canonical_hash(_score_contract(second))
