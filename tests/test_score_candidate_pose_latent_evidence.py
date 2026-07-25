from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.score_candidate_pose_latent_evidence import (
    baseline_top1_for_scored_rows,
    materialize_support_image_ids,
    native_frozen_hypothesis_baseline,
    parse_evidence_variants,
    ragged_offsets,
    select_scored_query_groups,
    select_heldout_query_groups,
    validate_checkpoint_for_target_free_scoring,
)


def _metadata() -> dict[str, object]:
    verification_compatibility = {
        "format": "mixed_multiscale_verification_points_scoring_compatibility_v1",
        "candidate_top_k": 20,
        "matcha_joint_checkpoint_sha256": "mapper",
        "projected_landmark_bank_sha256": "bank",
        "point_sources": {"alike_high_detail": {"count_per_image": 64}},
    }
    return {
        "format": "candidate_pose_latent_evidence_checkpoint_v1",
        "model_format": "candidate_pose_latent_evidence_v1",
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "fixed_support_view_count": 2,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "explicit_null": True,
        "identity_candidate_mixture": "soft_observed_coordinate_posterior_detached_before_pose",
        "alignment_token_weights": "identity_max_conditional_posterior_detached_before_pose",
        "verification_points_scoring_compatibility": verification_compatibility,
        "inputs": {
            name: (
                {
                    "sha256": value,
                    "scoring_compatibility": verification_compatibility,
                }
                if name == "verification_points"
                else {"sha256": value}
            )
            for name, value in zip(
                (
                    "verification_points",
                    "maplet_support_index",
                    "support_geometry_index",
                    "projected_landmark_bank",
                    "radio_final_context_cache",
                    "radio_intermediate_context_cache",
                    "alike_spatial_context_cache",
                    "colmap_cameras_bin",
                    "colmap_images_bin",
                ),
                "abcdefghi",
            )
        },
    }


def test_latent_checkpoint_contract_requires_current_static_inputs() -> None:
    metadata = _metadata()
    expected = dict(metadata["inputs"])

    validate_checkpoint_for_target_free_scoring(metadata, expected_inputs=expected)

    heldout_cache = {name: dict(value) for name, value in expected.items()}
    heldout_cache["verification_points"]["sha256"] = "different-test-cache"
    validate_checkpoint_for_target_free_scoring(
        metadata, expected_inputs=heldout_cache
    )

    stale = {name: dict(value) for name, value in expected.items()}
    stale["radio_intermediate_context_cache"] = {"sha256": "stale"}
    with pytest.raises(ValueError, match="stale"):
        validate_checkpoint_for_target_free_scoring(metadata, expected_inputs=stale)

    stale["radio_intermediate_context_cache"] = dict(
        expected["radio_intermediate_context_cache"]
    )
    stale["verification_points"]["sha256"] = "different-test-cache"
    stale["verification_points"]["scoring_compatibility"] = {
        **dict(stale["verification_points"]["scoring_compatibility"]),
        "projected_landmark_bank_sha256": "stale-bank",
    }
    with pytest.raises(ValueError, match="semantic"):
        validate_checkpoint_for_target_free_scoring(metadata, expected_inputs=stale)


def test_score_variants_require_the_paired_visual_control_contract() -> None:
    assert parse_evidence_variants(
        "visual,support_descriptor_permutation_control"
    ) == ("visual", "support_descriptor_permutation_control")
    with pytest.raises(ValueError, match="paired"):
        parse_evidence_variants("visual")


def test_heldout_query_groups_drop_train_rows_without_reordering_queries() -> None:
    groups = select_heldout_query_groups(
        query_ids=["train/a.png", "validation/a.png", "validation/a.png", "test/a.png"],
        split_names=["train", "validation", "validation", "test"],
    )

    assert groups == (
        ("test/a.png", "test", (3,)),
        ("validation/a.png", "validation", (1, 2)),
    )


def test_scored_query_groups_keep_whole_heldout_groups_with_per_group_limit() -> None:
    groups = select_scored_query_groups(
        query_ids=[
            "train/a.png",
            "validation/a.png",
            "validation/a.png",
            "validation/a.png",
            "test/a.png",
            "test/a.png",
            "test/a.png",
        ],
        split_names=[
            "train",
            "validation",
            "validation",
            "validation",
            "test",
            "test",
            "test",
        ],
        score_splits=("validation", "test"),
        hypothesis_limit=2,
    )

    assert groups == (
        ("test/a.png", "test", (4, 5)),
        ("validation/a.png", "validation", (1, 2)),
    )


def test_native_frozen_hypothesis_baseline_replaces_unverified_scores_without_changing_top1() -> None:
    scores, top1 = native_frozen_hypothesis_baseline(
        query_ids=np.asarray(["validation/a.png"] * 3 + ["test/a.png"] * 2),
        split_names=np.asarray(["validation"] * 3 + ["test"] * 2),
        evaluation_labels=np.asarray(["fixed"] * 5),
        hypothesis_indices=np.asarray([7, 3, 8, 4, 2]),
        verification_log_likelihood_means=np.asarray(
            [np.nan, 0.25, 0.10, -3.0, np.nan]
        ),
        chosen_for_optional_pose=np.asarray([False, True, False, True, False]),
    )

    assert np.isfinite(scores).all()
    assert scores[0] < scores[2]
    assert scores[4] < scores[3]
    assert top1.tolist() == [False, True, False, True, False]


def test_native_frozen_hypothesis_baseline_rejects_nonmaximal_chosen_row() -> None:
    with pytest.raises(ValueError, match="does not reproduce"):
        native_frozen_hypothesis_baseline(
            query_ids=np.asarray(["validation/a.png", "validation/a.png"]),
            split_names=np.asarray(["validation", "validation"]),
            evaluation_labels=np.asarray(["fixed", "fixed"]),
            hypothesis_indices=np.asarray([0, 1]),
            verification_log_likelihood_means=np.asarray([0.5, 0.75]),
            chosen_for_optional_pose=np.asarray([True, False]),
        )


def test_ragged_offsets_keep_empty_segments_and_reject_negative_lengths() -> None:
    assert ragged_offsets([2, 0, 3]).tolist() == [0, 2, 2, 5]
    with pytest.raises(ValueError, match="ragged"):
        ragged_offsets([1, -1])


def test_scored_row_baseline_top1_is_recomputed_inside_a_development_prefix() -> None:
    top1 = baseline_top1_for_scored_rows(
        query_ids=np.asarray(["validation/a.png", "validation/a.png", "test/a.png"]),
        split_names=np.asarray(["validation", "validation", "test"]),
        evaluation_labels=np.asarray(["fixed", "fixed", "fixed"]),
        hypothesis_indices=np.asarray([7, 3, 1]),
        baseline_selection_scores=np.asarray([0.2, 0.4, -1.0]),
    )

    assert top1.tolist() == [False, True, True]


def test_materialize_support_image_ids_preserves_full_ids_and_blanks_invalid_views() -> None:
    image_ids = materialize_support_image_ids(
        support_image_indices=np.asarray([[[0, 1], [-1, 0]]]),
        support_view_valid=np.asarray([[[True, True], [False, False]]]),
        cache_image_ids=np.asarray(["seq1/frame00051.png", "seq6/frame00044.png"]),
    )

    assert image_ids.tolist() == [
        [
            ["seq1/frame00051.png", "seq6/frame00044.png"],
            ["", ""],
        ],
    ]
