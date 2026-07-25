from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_frozen_lifted_loftr_map_to_query_pose_evidence import (
    _implementation_hashes,
    _primary_gate,
    _validate_visual_control_pairing,
)


def _metadata(*, control: bool) -> dict[str, object]:
    strict = {
        "heldout_query_image_content_excludes_pnp_fit_neighborhoods": True,
        "fixed_pnp_fit_topl_maplet_union": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_group_latent_identity_marginalized": True,
        "candidate_group_topl_denominator_fixed": True,
        "candidate_group_explicit_null": True,
        "candidate_group_identity_prior_fixed_before_pose_scoring": True,
        "candidate_group_identity_prior_target_free": True,
        "candidate_groups_without_lifted_evidence_fixed_null_only": True,
        "candidate_group_active_mask_fixed_across_hypotheses": True,
        "pnp_query_center_used_for_group_scoring": False,
        "support_maplet_prefix_fixed": True,
        "support_view_descriptor_averaging": False,
        "support_view_endpoints_averaged_before_likelihood": False,
        "support_view_endpoint_mixture_explicit": True,
        "cross_view_modes_marginalized_not_argmaxed": True,
        "query_center_used_for_loftr_mode_selection": False,
        "pose_dependent_correspondence_selection": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "full_mapping_image_pair_cache": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "out_of_image_projection_is_negative_likelihood": True,
        "raw_scores_calibrated_or_promoted": False,
        "raw_scores_must_not_feed_pnp": True,
        "xyz_permutation_control": control,
    }
    return {
        "query_id": "q.png",
        "evidence_variant": "xyz_permutation_control" if control else "visual",
        "strict_frozen_lifted_map_to_query_contract": strict,
        "candidate_maplet_union": {"candidate_union_digest": "union"},
        "lifting_config": {"sigma": 4},
        "coordinate_contract": {"version": 1},
        "pair_cache_contract": {"version": 1},
        "input_metadata_hashes": {"bank": "x"},
        "canonical_mode_layout_sha256": "layout",
        "canonical_xyz_sha256": "xyz",
        "score_profiles": [{"name": "sigma4"}],
        "xyz_permutation": [1, 0] if control else None,
    }


def _arrays(*, control: bool) -> dict[str, np.ndarray]:
    canonical_xyz = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    return {
        "query_ids": np.asarray(["q.png"]),
        "split_names": np.asarray(["validation"]),
        "evaluation_labels": np.asarray(["label"]),
        "hypothesis_indices": np.asarray([0], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray([True]),
        "baseline_score_top1": np.asarray([True]),
        "baseline_selection_scores": np.asarray([1.0]),
        "profile_names": np.asarray(["sigma4"]),
        "fit_query_xy": np.zeros((128, 2), dtype=np.float32),
        "candidate_group_track_indices": np.tile(np.asarray([[0, 1] + [-1] * 18]), (128, 1)),
        "candidate_group_reference_xy": np.zeros((128, 2), dtype=np.float32),
        "candidate_group_active_mask": np.ones((128,), dtype=bool),
        "candidate_group_identity_probabilities": np.tile(
            np.asarray([[0.4, 0.1] + [0.0] * 18], dtype=np.float32), (128, 1)
        ),
        "candidate_group_null_probabilities": np.full((128,), 0.5, dtype=np.float32),
        "canonical_track_ids": np.asarray([7, 9], dtype=np.int64),
        "canonical_xyz": canonical_xyz,
        "mode_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "mode_query_xy": np.zeros((2, 2), dtype=np.float32),
        "mode_weights": np.asarray([1.0, 1.0], dtype=np.float32),
        "mode_support_image_ids": np.asarray(["a.png", "b.png"]),
        "mode_support_view_counts": np.asarray([2, 2], dtype=np.int64),
        "mode_confidence_sums": np.asarray([1.0, 1.0], dtype=np.float32),
        "track_reliabilities": np.asarray([0.8, 0.8], dtype=np.float32),
        "track_reference_xy": np.zeros((2, 2), dtype=np.float32),
        "active_xyz": canonical_xyz[[1, 0]] if control else canonical_xyz,
    }


def _candidate_view_metadata(*, control: bool) -> dict[str, object]:
    metadata = _metadata(control=control)
    metadata["evidence_layout"] = "candidate_specific_support_view_v1"
    strict = dict(metadata["strict_frozen_lifted_map_to_query_contract"])
    strict.update(
        {
            "candidate_specific_support_view_posterior": True,
            "candidate_support_view_posterior_fixed_before_pose_scoring": True,
            "candidate_support_view_posterior_target_free": True,
            "missing_support_view_evidence_is_neutral_ratio": True,
        }
    )
    metadata["strict_frozen_lifted_map_to_query_contract"] = strict
    return metadata


def _candidate_view_arrays(*, control: bool) -> dict[str, np.ndarray]:
    arrays = _arrays(control=control)
    for field in (
        "mode_support_image_ids",
        "mode_support_view_counts",
        "track_reliabilities",
        "track_reference_xy",
    ):
        arrays.pop(field)
    slots = np.full((128, 20, 2), -1, dtype=np.int64)
    slots[:, 0] = np.asarray([0, 1], dtype=np.int64)
    support_ids = np.full((128, 20, 2), "unused.png")
    support_ids[:, 0] = np.asarray(["a.png", "b.png"])
    arrays.update(
        {
            "candidate_group_support_slot_indices": slots,
            "candidate_group_support_image_ids": support_ids,
            "candidate_group_support_view_probabilities": np.full(
                (128, 20, 2), 0.5, dtype=np.float32
            ),
            "support_slot_track_indices": np.asarray([0, 0], dtype=np.int64),
            "support_slot_image_ids": np.asarray(["a.png", "b.png"]),
            "mode_offsets": np.asarray([0, 1, 2], dtype=np.int64),
            "mode_query_xy": np.zeros((2, 2), dtype=np.float32),
            "mode_weights": np.asarray([1.0, 1.0], dtype=np.float32),
            "mode_confidence_sums": np.asarray([1.0, 1.0], dtype=np.float32),
            "support_reliabilities": np.asarray([0.8, 0.8], dtype=np.float32),
            "support_reference_xy": np.zeros((2, 2), dtype=np.float32),
        }
    )
    return arrays


def test_visual_control_pairing_allows_only_xyz_derangement() -> None:
    _validate_visual_control_pairing(
        visual_artifacts=[(_arrays(control=False), _metadata(control=False))],
        control_artifacts=[(_arrays(control=True), _metadata(control=True))],
    )


def test_candidate_support_view_pairing_allows_only_xyz_derangement() -> None:
    _validate_visual_control_pairing(
        visual_artifacts=[
            (_candidate_view_arrays(control=False), _candidate_view_metadata(control=False))
        ],
        control_artifacts=[
            (_candidate_view_arrays(control=True), _candidate_view_metadata(control=True))
        ],
    )


def test_visual_control_pairing_rejects_mixed_evidence_layouts() -> None:
    with pytest.raises(ValueError, match="provenance"):
        _validate_visual_control_pairing(
            visual_artifacts=[(_arrays(control=False), _metadata(control=False))],
            control_artifacts=[
                (_candidate_view_arrays(control=True), _candidate_view_metadata(control=True))
            ],
        )


def test_visual_control_pairing_rejects_non_deranged_control() -> None:
    metadata = _metadata(control=True)
    metadata["xyz_permutation"] = [0, 1]
    with pytest.raises(ValueError, match="derangement"):
        _validate_visual_control_pairing(
            visual_artifacts=[(_arrays(control=False), _metadata(control=False))],
            control_artifacts=[(_arrays(control=True), metadata)],
        )


def test_implementation_hashes_ignore_invocation_path() -> None:
    relative = {
        "implementation": {
            "script_path": "feature_extract/tools/vfm/score.py",
            "script_sha256": "script",
            "lifted_mode_module_sha256": "module",
        }
    }
    absolute = {
        "implementation": {
            "script_path": "/root/ICLPose/feature_extract/tools/vfm/score.py",
            "script_sha256": "script",
            "lifted_mode_module_sha256": "module",
        }
    }
    assert _implementation_hashes(relative) == _implementation_hashes(absolute)


def test_gate_requires_rank_signal_and_tail_safety() -> None:
    visual = {
        "median_oracle_score_rank": 20.0,
        "p90_oracle_score_rank": 40.0,
        "p90_selected_translation_cm": 50.0,
        "catastrophic_1m_count": 0,
    }
    control = {
        "median_oracle_score_rank": 60.0,
        "p90_oracle_score_rank": 45.0,
    }
    baseline = {
        "p90_oracle_score_rank": 45.0,
        "p90_selected_translation_cm": 60.0,
        "catastrophic_1m_count": 1,
    }
    passed = _primary_gate(
        visual=visual,
        control=control,
        baseline=baseline,
        paired_rank={"oracle_rank_wins": 8, "oracle_rank_losses": 2},
    )
    assert passed["raw_p1_gate_passed"] is True
    assert passed["eligible_for_pose_or_pnp_promotion"] is False
    failed = _primary_gate(
        visual={**visual, "catastrophic_1m_count": 2},
        control=control,
        baseline=baseline,
        paired_rank={"oracle_rank_wins": 8, "oracle_rank_losses": 2},
    )
    assert failed["raw_p1_gate_passed"] is False
