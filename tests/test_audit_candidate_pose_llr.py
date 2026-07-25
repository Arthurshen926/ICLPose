from __future__ import annotations

import numpy as np
import pytest
import sys

from feature_extract.tools.vfm.audit_candidate_pose_llr import (
    _assert_visual_control_pair,
    _primary_gate,
    _validate_target_lineage,
)


def test_direct_script_import_bootstrap_resolves_repository_root() -> None:
    import feature_extract.tools.vfm.audit_candidate_pose_llr as module

    assert str(module._REPOSITORY_ROOT) in sys.path


def _arrays(*, score: float) -> dict[str, np.ndarray]:
    return {
        "query_ids": np.asarray(["query.png"]),
        "split_names": np.asarray(["validation"]),
        "evaluation_labels": np.asarray(["frozen"]),
        "hypothesis_indices": np.asarray([0], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray([False]),
        "baseline_score_top1": np.asarray([True]),
        "baseline_selection_scores": np.asarray([0.0]),
        "pose_log_likelihood_ratios": np.asarray([score], dtype=np.float32),
        "point_log_likelihood_ratios": np.asarray([[score, score]], dtype=np.float32),
        "point_effective_candidate_counts": np.asarray([[1, 1]], dtype=np.int16),
        "point_effective_view_masses": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "source_names": np.asarray(["alike"]),
        "source_log_likelihood_means": np.asarray([[score]], dtype=np.float32),
        "source_effective_point_counts": np.asarray([[2]], dtype=np.int16),
        "verification_source_point_ids": np.asarray([1, 2], dtype=np.int64),
        "verification_point_sources": np.asarray(["alike", "alike"]),
        "verification_source_detector_rows": np.asarray([4, 5], dtype=np.int64),
        "verification_xy": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "candidate_track_ids": np.asarray([[10], [11]], dtype=np.int64),
        "candidate_probabilities": np.asarray([[0.8], [0.8]], dtype=np.float32),
        "null_probabilities": np.asarray([0.2, 0.2], dtype=np.float32),
        "candidate_view_weights": np.asarray([[[1.0]], [[1.0]]], dtype=np.float32),
        "candidate_support_image_ids": np.asarray([[["support.png"]], [["support.png"]]]),
    }


def _metadata(*, variant: str) -> dict[str, object]:
    return {
        "format": "candidate_specific_pose_llr_scores_v3",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "evidence_variant": variant,
        "frozen_query_evidence_sha256": "same",
        "strict_candidate_pose_llr_contract": {
            "heldout_query_rows": True,
            "formal_p1_mixed_multiscale_points": True,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": 20,
            "candidate_identity_fixed_across_hypotheses": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "fixed_support_view_count": 2,
            "explicit_null": True,
            "candidate_projection_is_only_pose_dependent_encoder_input": True,
            "candidate_pose_matrix_excluded_from_encoder": True,
            "residual_and_target_excluded_from_encoder": True,
            "support_descriptor_permutation_control": variant
            == "support_descriptor_permutation_control",
            "support_image_appearance_derangement_control": variant
            == "support_descriptor_permutation_control",
            "appearance_control_geometry_fixed": True,
            "no_pnp": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }


def test_visual_control_pair_rejects_changed_frozen_candidate_layout() -> None:
    visual = _arrays(score=0.5)
    control = _arrays(score=-0.5)
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


def test_visual_control_pair_rejects_non_geometry_fixed_v2_control() -> None:
    visual = _arrays(score=0.5)
    control = _arrays(score=-0.5)
    control_metadata = _metadata(variant="support_descriptor_permutation_control")
    strict = dict(control_metadata["strict_candidate_pose_llr_contract"])
    strict["appearance_control_geometry_fixed"] = False
    control_metadata["strict_candidate_pose_llr_contract"] = strict
    with pytest.raises(ValueError, match="geometry-fixed"):
        _assert_visual_control_pair(
            visual_arrays=visual,
            visual_metadata=_metadata(variant="visual"),
            control_arrays=control,
            control_metadata=control_metadata,
        )


def test_primary_gate_requires_rank_gain_without_tail_regression() -> None:
    visual = {
        "median_best_10cm_rank": 12.0,
        "p90_best_10cm_rank": 30.0,
        "p90_selected_translation_cm": 20.0,
        "catastrophic_1m_count": 0,
    }
    control = {
        "median_best_10cm_rank": 30.0,
        "p90_best_10cm_rank": 32.0,
        "p90_selected_translation_cm": 25.0,
        "catastrophic_1m_count": 0,
    }
    paired = {"best_10cm_rank_wins": 8, "best_10cm_rank_losses": 3}
    assert all(_primary_gate(visual=visual, control=control, paired_rank=paired).values())

    visual["catastrophic_1m_count"] = 1
    assert not all(_primary_gate(visual=visual, control=control, paired_rank=paired).values())


def test_target_lineage_accepts_grouped_inference_artifact_manifest() -> None:
    _validate_target_lineage(
        score_metadata=[
            {
                "inputs": {
                    "hypothesis_artifact": {"sha256": "hypothesis"},
                    "baseline_score_artifact": {"sha256": "baseline"},
                }
            }
        ],
        target_metadata={
            "format": "grouped_pose_hypothesis_targets_v1",
            "targets_joined_after_inference": True,
            "inference_artifact_sha256": ["hypothesis"],
        },
    )
