from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _is_better_inner_validation_epoch,
    _evaluate_inner_validation_target_free_static_selector,
    fixed_final_epoch_checkpoint_selection,
    _hard_repeat_batch_from_group,
    _select_group_points,
    _partition_train_queries_for_inner_validation,
    _validate_training_args,
    build_train_query_groups,
    coherent_hard_repeat_context_margin_loss,
    coherent_hard_repeat_edge_margin_loss,
    CHECKPOINT_FORMAT,
    current_inner_gate_evaluator_manifest,
    configure_rgb_cost_volume_only_trainable_parameters,
    exact_identity_support_appearance_contrastive_loss,
    HardRepeatBatch,
    HardRepeatQueryTargets,
    hard_repeat_loss_weight_for_epoch,
    hard_repeat_training_gate_decision,
    load_context_observation_pretrain_initialization_checkpoint,
    load_hard_repeat_gated_rgb_texture_initialization_checkpoint,
    load_hard_pose_pretrain_initialization_checkpoint,
    load_identity_llr_texture_pretrain_initialization_checkpoint,
    load_observation_pretrain_initialization_checkpoint,
    load_texture_observation_pretrain_component_checkpoint,
    load_target_free_initialization_checkpoint,
    rgb_coordinate_scale,
    support_permutation_contrastive_loss,
    train_support_permutation_shift,
    train_query_partition_manifest,
    validate_current_system_mined_hard_repeat_targets,
    resolve_current_system_mined_top_h_mode_positions,
    TrainQueryGroup,
    training_gate_decision,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
    parse_args,
    registered_identity_batch_for_geometry_selection,
    registered_identity_observed_source_point_ids,
    SOURCE_SAFE_EDGE_AVAILABILITY,
    validate_registered_identity_targets_for_geometry,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    CandidatePoseRGBSpatialEdgePrediction,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_support_appearance,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CandidatePoseRGBSpatialIdentityLLR,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    CandidatePoseRGBSpatialTrainingTargets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
    CandidatePoseRGBSpatialHardRepeatTargets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_system_hard import (
    CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT,
    CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT,
)


def _layout() -> CandidatePoseRGBSpatialLayout:
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        query_ids=np.asarray(["q/a.png", "q/a.png", "q/b.png", "q/b.png"]),
        split_names=np.asarray(["train", "train", "train", "train"]),
        xy=np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], dtype=np.float32),
        point_sources=np.asarray(["p", "p", "p", "p"]),
        candidate_track_ids=np.asarray([[101], [102], [103], [104]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0], [1], [2], [3]], dtype=np.int64),
        candidate_coarse_similarities=np.full((4, 1), 0.8, dtype=np.float32),
        candidate_prior_probabilities=np.full((4, 1), 0.8, dtype=np.float32),
        null_probabilities=np.full((4,), 0.2, dtype=np.float32),
        support_image_ids=np.asarray(
            [[ ["map/a.png"] ], [["map/b.png"]], [["map/c.png"]], [["map/d.png"]]]
        ),
        support_xy=np.asarray(
            [[[[1.0, 1.0]]], [[[2.0, 2.0]]], [[[3.0, 3.0]]], [[[4.0, 4.0]]]],
            dtype=np.float32,
        ),
        support_view_valid=np.ones((4, 1, 1), dtype=bool),
        support_view_weights=np.ones((4, 1, 1), dtype=np.float32),
        support_coverage_counts=np.ones((4, 1, 1), dtype=np.int32),
        metadata={
            "format": "candidate_pose_rgb_spatial_layout_v1",
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "verification_points_sha256": "points",
            "maplet_support_index_sha256": "maplet",
            "support_geometry_index_sha256": "geometry",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
        },
    )


def _minimal_training_cli_args() -> list[str]:
    return [
        "--rgb-spatial-layout",
        "layout.npz",
        "--training-targets",
        "targets.npz",
        "--radio-final-context-cache",
        "radio_final.npz",
        "--radio-intermediate-context-cache",
        "radio_intermediate.npz",
        "--alike-spatial-context-cache",
        "alike.npz",
        "--image-root",
        "images",
        "--output-dir",
        "output",
    ]


def test_full_likelihood_accepts_direct_context_identity_objectives() -> None:
    args = parse_args(
        _minimal_training_cli_args()
        + [
            "--context-identity-loss-weight",
            "0.5",
            "--context-identity-support-permutation-loss-weight",
            "0.25",
            "--context-identity-support-permutation-margin",
            "0.3",
        ]
    )
    assert _validate_training_args(args) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }


def test_separate_registered_identity_targets_require_an_exact_identity_objective() -> None:
    unused = parse_args(
        _minimal_training_cli_args()
        + ["--registered-identity-targets", "registered_identity.npz"]
    )
    with pytest.raises(ValueError, match="positive exact-identity objective"):
        _validate_training_args(unused)

    accepted = parse_args(
        _minimal_training_cli_args()
        + [
            "--registered-identity-targets",
            "registered_identity.npz",
            "--context-identity-loss-weight",
            "0.5",
        ]
    )
    assert _validate_training_args(accepted) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }

    introduction_without_parent = parse_args(
        _minimal_training_cli_args()
        + [
            "--registered-identity-targets",
            "registered_identity.npz",
            "--context-identity-loss-weight",
            "0.5",
            "--allow-registered-identity-sidecar-introduction-from-identity-free-init",
        ]
    )
    with pytest.raises(ValueError, match="sidecar introduction requires"):
        _validate_training_args(introduction_without_parent)


def test_current_system_mined_context_loss_is_checkpoint_bound_and_not_rgb_only() -> None:
    missing_targets = parse_args(
        _minimal_training_cli_args()
        + ["--mined-hard-repeat-context-loss-weight", "0.2"]
    )
    with pytest.raises(ValueError, match="current-system target artifact"):
        _validate_training_args(missing_targets)

    accepted = parse_args(
        _minimal_training_cli_args()
        + [
            "--hard-repeat-targets",
            "static_hard_repeat.npz",
            "--mined-hard-repeat-targets",
            "current_system_hard_repeat.npz",
            "--mined-registered-identity-targets",
            "registered_identity.npz",
            "--init-checkpoint",
            "frozen_mining_checkpoint.pt",
            "--mined-hard-repeat-context-loss-weight",
            "0.2",
            "--mined-hard-repeat-context-margin",
            "0.3",
        ]
    )
    assert _validate_training_args(accepted) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }
    assert accepted.mined_hard_repeat_loss_weight == pytest.approx(0.0)
    assert accepted.mined_hard_repeat_context_loss_weight == pytest.approx(0.2)
    assert accepted.mined_hard_repeat_context_margin == pytest.approx(0.3)

    rgb_only = parse_args(
        _minimal_training_cli_args()
        + [
            "--rgb-cost-volume-only",
            "--dustbin-loss-weight",
            "0",
            "--hard-repeat-targets",
            "static_hard_repeat.npz",
            "--mined-hard-repeat-targets",
            "current_system_hard_repeat.npz",
            "--mined-registered-identity-targets",
            "registered_identity.npz",
            "--init-checkpoint",
            "frozen_mining_checkpoint.pt",
            "--mined-hard-repeat-context-loss-weight",
            "0.2",
        ]
    )
    with pytest.raises(ValueError, match="context-only"):
        _validate_training_args(rgb_only)


def test_context_only_initializer_requires_and_accepts_source_pairs() -> None:
    missing_pairs = parse_args(
        _minimal_training_cli_args()
        + ["--context-observation-pretrain-checkpoint", "context.pt"]
    )
    with pytest.raises(ValueError, match="source observation-pair"):
        _validate_training_args(missing_pairs)

    accepted = parse_args(
        _minimal_training_cli_args()
        + [
            "--context-observation-pretrain-checkpoint",
            "context.pt",
            "--context-observation-pairs",
            "pairs.npz",
        ]
    )
    assert _validate_training_args(accepted) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }


def test_full_likelihood_accepts_full_pool_permutation_soft_hard_arguments() -> None:
    args = parse_args(
        _minimal_training_cli_args()
        + [
            "--pose-pool-permutation-soft-hard-loss-weight",
            "0.5",
            "--pose-pool-permutation-soft-hard-margin",
            "0.1",
            "--pose-pool-permutation-soft-hard-temperature",
            "0.25",
        ]
    )
    assert _validate_training_args(args) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }
    assert args.pose_pool_permutation_soft_hard_loss_weight == pytest.approx(0.5)
    assert args.pose_pool_permutation_soft_hard_margin == pytest.approx(0.1)
    assert args.pose_pool_permutation_soft_hard_temperature == pytest.approx(0.25)


def test_rgb_only_rejects_direct_context_identity_objectives() -> None:
    args = parse_args(
        _minimal_training_cli_args()
        + [
            "--rgb-cost-volume-only",
            "--dustbin-loss-weight",
            "0",
            "--context-identity-loss-weight",
            "0.5",
        ]
    )
    with pytest.raises(ValueError, match="context-only"):
        _validate_training_args(args)


def test_hard_repeat_texture_initializer_is_exclusive_and_requires_matching_targets() -> None:
    missing_targets = parse_args(
        _minimal_training_cli_args()
        + [
            "--rgb-hard-repeat-texture-checkpoint",
            "rgb_hard_repeat.pt",
        ]
    )
    with pytest.raises(ValueError, match="matching train-only hard-repeat"):
        _validate_training_args(missing_targets)

    accepted = parse_args(
        _minimal_training_cli_args()
        + [
            "--hard-repeat-targets",
            "hard_repeat_targets.npz",
            "--rgb-hard-repeat-texture-checkpoint",
            "rgb_hard_repeat.pt",
        ]
    )
    assert _validate_training_args(accepted) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }

    mixed = parse_args(
        _minimal_training_cli_args()
        + [
            "--hard-repeat-targets",
            "hard_repeat_targets.npz",
            "--rgb-hard-repeat-texture-checkpoint",
            "rgb_hard_repeat.pt",
            "--context-observation-pretrain-checkpoint",
            "context.pt",
        ]
    )
    with pytest.raises(ValueError, match="without other component initializers"):
        _validate_training_args(mixed)


def test_gate_passing_inner_epoch_outranks_lower_loss_gate_failure() -> None:
    failed = {
        "normal_query_grouped_loss": 0.10,
        "normal_correct_win_fraction": 0.90,
    }
    passed = {
        "normal_query_grouped_loss": 0.11,
        "normal_correct_win_fraction": 0.80,
    }
    assert _is_better_inner_validation_epoch(
        candidate=passed,
        incumbent=failed,
        candidate_gate_passed=True,
        incumbent_gate_passed=False,
    )
    assert not _is_better_inner_validation_epoch(
        candidate=failed,
        incumbent=passed,
        candidate_gate_passed=False,
        incumbent_gate_passed=True,
    )


def test_full_likelihood_checkpoint_selection_is_fixed_to_final_epoch() -> None:
    assert fixed_final_epoch_checkpoint_selection(epochs=6) == {
        "policy": "fixed_final_epoch_without_inner_validation_model_selection_v1",
        "selected_epoch": 6,
        "inner_validation_used_for_model_selection": False,
    }


def test_train_query_partition_manifest_is_exact_and_tamper_evident() -> None:
    manifest = train_query_partition_manifest(
        all_query_ids=("q/c.png", "q/a.png", "q/b.png"),
        inner_train_query_ids=("q/c.png", "q/a.png"),
        inner_validation_query_ids=("q/b.png",),
        fold_count=3,
        fold_index=1,
    )
    assert manifest["assignment"] == "sorted_unique_query_index_modulo_fold_count_v1"
    assert manifest["inner_train"]["query_ids"] == ["q/a.png", "q/c.png"]
    assert manifest["inner_validation"]["query_ids"] == ["q/b.png"]
    assert manifest["all_train"]["query_count"] == 3
    with pytest.raises(ValueError, match="inconsistent"):
        train_query_partition_manifest(
            all_query_ids=("q/a.png", "q/b.png"),
            inner_train_query_ids=("q/a.png",),
            inner_validation_query_ids=("q/a.png",),
            fold_count=2,
            fold_index=0,
        )


def _current_system_mined_targets(
    *,
    query_id: str,
    partition: dict[str, object],
    checkpoint_sha256: str,
    registered_identity_targets_sha256: str,
) -> CandidatePoseRGBSpatialHardRepeatTargets:
    config = {"context_encoder_arch": "absolute_cross_attention_v3", "step_px": 1.0}
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return CandidatePoseRGBSpatialHardRepeatTargets(
        source_point_ids=np.asarray([1], dtype=np.int64),
        query_ids=np.asarray([query_id]),
        pair_ids=np.asarray([7], dtype=np.int64),
        positive_candidate_indices=np.asarray([0], dtype=np.int64),
        negative_candidate_indices=np.asarray([1], dtype=np.int64),
        positive_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
        negative_offsets_xy=np.asarray([[1.0, 1.0]], dtype=np.float32),
        metadata={
            "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "rgb_spatial_layout_sha256": "layout",
            "rgb_spatial_targets_sha256": "targets",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "candidate_count": 2,
            "positive_radius_px": 2.0,
            "negative_radius_px": 2.0,
            "mining_format": CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMAT,
            "score_component": "combined",
            "frozen_checkpoint": {
                "sha256": checkpoint_sha256,
                "train_only_inner_gate_passed": True,
                "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
            },
            "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
            "mining_checkpoint_sha256": checkpoint_sha256,
            "mining_checkpoint_config": config,
            "mining_checkpoint_config_sha256": config_sha256,
            "train_query_partition": partition,
            "registered_identity_targets_sha256": registered_identity_targets_sha256,
        },
    )


def _registered_identity_targets(*, query_id: str) -> CandidatePoseRGBSpatialTrainingTargets:
    return CandidatePoseRGBSpatialTrainingTargets(
        source_point_ids=np.asarray([1], dtype=np.int64),
        query_ids=np.asarray([query_id]),
        spatial_target_offsets_xy=np.zeros((1, 2, 2), dtype=np.float32),
        spatial_target_observed=np.asarray([[True, False]], dtype=bool),
        spatial_target_dustbin=np.asarray([[False, True]], dtype=bool),
        spatial_target_supervised=np.asarray([[True, True]], dtype=bool),
        pair_query_ids=np.asarray([query_id]),
        pair_ids=np.asarray([7], dtype=np.int64),
        pair_point_offsets=np.asarray([0, 1], dtype=np.int64),
        pair_source_point_ids=np.asarray([1], dtype=np.int64),
        correct_projection_offsets_xy=np.zeros((1, 2, 2), dtype=np.float32),
        correct_projection_valid=np.ones((1, 2), dtype=bool),
        coherent_wrong_projection_offsets_xy=np.ones((1, 2, 2), dtype=np.float32),
        coherent_wrong_projection_valid=np.ones((1, 2), dtype=bool),
        metadata={
            "format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "rgb_spatial_layout_sha256": "layout",
            "train_pairs_sha256": "pairs",
            "support_geometry_index_sha256": "geometry",
            "projected_landmark_bank_sha256": "bank",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "spatial_search_radius_px": 2.0,
            "spatial_supervision_mode": "registered_exact_identity",
            "spatial_target_semantics": (
                "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
            ),
            "spatial_class_balance": "per_batch_observed_dustbin_mean_v1",
            "registered_identity_radius_px": 2.0,
            "colmap_images_sha256": "images",
        },
    )


def test_current_system_mined_targets_are_inner_train_only_and_checkpoint_bound(tmp_path) -> None:
    checkpoint = tmp_path / "mining.pt"
    checkpoint.write_bytes(b"frozen-mining-checkpoint")
    checkpoint_sha256 = file_sha256_short(checkpoint)
    identity_sha256 = "registered-identity-targets"
    partition = train_query_partition_manifest(
        all_query_ids=("q/a.png", "q/b.png"),
        inner_train_query_ids=("q/a.png",),
        inner_validation_query_ids=("q/b.png",),
        fold_count=2,
        fold_index=1,
    )
    targets = _current_system_mined_targets(
        query_id="q/a.png",
        partition=partition,
        checkpoint_sha256=checkpoint_sha256,
        registered_identity_targets_sha256=identity_sha256,
    )
    registered_identity = _registered_identity_targets(query_id="q/a.png")
    groups = {
        "q/a.png": HardRepeatQueryTargets(
            query_id="q/a.png",
            source_point_ids=np.asarray([1], dtype=np.int64),
            pair_ids=np.asarray([7], dtype=np.int64),
            positive_candidate_indices=np.asarray([0], dtype=np.int64),
            negative_candidate_indices=np.asarray([1], dtype=np.int64),
            positive_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
            negative_offsets_xy=np.asarray([[1.0, 1.0]], dtype=np.float32),
        )
    }
    lineage = validate_current_system_mined_hard_repeat_targets(
        mined_targets=targets,
        mined_groups=groups,
        registered_identity_targets=registered_identity,
        registered_identity_targets_sha256=identity_sha256,
        expected_partition=partition,
        initialization_checkpoint_path=checkpoint,
    )
    assert lineage["selection_scope"] == "current_model_inner_train_only_v1"
    assert lineage["mined_edge_count"] == 1

    multi_mode_targets = CandidatePoseRGBSpatialHardRepeatTargets(
        **{
            **targets.__dict__,
            "metadata": {
                **targets.metadata,
                "mining_format": CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT,
                "wrong_mode_selection": {
                    "policy": "target_free_combined_pose_llr_descending_pair_id_tiebreak_top_h_v1",
                    "max_wrong_modes_per_query": 4,
                    "target_join_after_mode_ranking": True,
                    "label_based_mode_backfill": False,
                },
                "target_free_ranked_wrong_modes": {
                    "format": CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT,
                    "query_count": 1,
                    "ranked_modes_by_query": {
                        "q/a.png": [
                            {"mode_rank": 0, "mode_index": 0, "pair_id": 7}
                        ]
                    },
                    "selection_before_train_only_target_join": True,
                },
            },
        }
    )
    multi_mode_lineage = validate_current_system_mined_hard_repeat_targets(
        mined_targets=multi_mode_targets,
        mined_groups=groups,
        registered_identity_targets=registered_identity,
        registered_identity_targets_sha256=identity_sha256,
        expected_partition=partition,
        initialization_checkpoint_path=checkpoint,
    )
    assert multi_mode_lineage["mined_edge_count"] == 1

    malformed_multi_mode_targets = CandidatePoseRGBSpatialHardRepeatTargets(
        **{
            **multi_mode_targets.__dict__,
            "metadata": {
                **multi_mode_targets.metadata,
                "wrong_mode_selection": {
                    **multi_mode_targets.metadata["wrong_mode_selection"],
                    "label_based_mode_backfill": True,
                },
            },
        }
    )
    with pytest.raises(ValueError, match="multi-mode.*contract"):
        validate_current_system_mined_hard_repeat_targets(
            mined_targets=malformed_multi_mode_targets,
            mined_groups=groups,
            registered_identity_targets=registered_identity,
            registered_identity_targets_sha256=identity_sha256,
            expected_partition=partition,
            initialization_checkpoint_path=checkpoint,
        )

    heldout = _current_system_mined_targets(
        query_id="q/b.png",
        partition=partition,
        checkpoint_sha256=checkpoint_sha256,
        registered_identity_targets_sha256=identity_sha256,
    )
    heldout_groups = {
        "q/b.png": HardRepeatQueryTargets(
            query_id="q/b.png",
            source_point_ids=np.asarray([1], dtype=np.int64),
            pair_ids=np.asarray([7], dtype=np.int64),
            positive_candidate_indices=np.asarray([0], dtype=np.int64),
            negative_candidate_indices=np.asarray([1], dtype=np.int64),
            positive_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
            negative_offsets_xy=np.asarray([[1.0, 1.0]], dtype=np.float32),
        )
    }
    with pytest.raises(ValueError, match="non-inner-train"):
        validate_current_system_mined_hard_repeat_targets(
            mined_targets=heldout,
            mined_groups=heldout_groups,
            registered_identity_targets=_registered_identity_targets(query_id="q/b.png"),
            registered_identity_targets_sha256=identity_sha256,
            expected_partition=partition,
            initialization_checkpoint_path=checkpoint,
        )

    wrong_checkpoint = tmp_path / "different.pt"
    wrong_checkpoint.write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="exact mining checkpoint"):
        validate_current_system_mined_hard_repeat_targets(
            mined_targets=targets,
            mined_groups=groups,
            registered_identity_targets=registered_identity,
            registered_identity_targets_sha256=identity_sha256,
            expected_partition=partition,
            initialization_checkpoint_path=wrong_checkpoint,
        )


def test_current_system_top_h_pose_pool_uses_frozen_target_free_mode_order() -> None:
    query_id = "q/a.png"
    partition = train_query_partition_manifest(
        all_query_ids=(query_id, "q/b.png"),
        inner_train_query_ids=(query_id,),
        inner_validation_query_ids=("q/b.png",),
        fold_count=2,
        fold_index=1,
    )
    targets = _current_system_mined_targets(
        query_id=query_id,
        partition=partition,
        checkpoint_sha256="checkpoint",
        registered_identity_targets_sha256="identity",
    )
    top_h_targets = CandidatePoseRGBSpatialHardRepeatTargets(
        **{
            **targets.__dict__,
            "metadata": {
                **targets.metadata,
                "mining_format": CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT,
                "wrong_mode_selection": {
                    "policy": "target_free_combined_pose_llr_descending_pair_id_tiebreak_top_h_v1",
                    "max_wrong_modes_per_query": 2,
                    "target_join_after_mode_ranking": True,
                    "label_based_mode_backfill": False,
                },
                "target_free_ranked_wrong_modes": {
                    "format": CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT,
                    "query_count": 1,
                    "ranked_modes_by_query": {
                        query_id: [
                            {"mode_rank": 0, "mode_index": 1, "pair_id": 7},
                            {"mode_rank": 1, "mode_index": 0, "pair_id": 9},
                        ]
                    },
                    "selection_before_train_only_target_join": True,
                },
            },
        }
    )
    geometry = TrainQueryGroup(
        query_id=query_id,
        source_point_ids=np.asarray([1], dtype=np.int64),
        layout_rows=np.asarray([0], dtype=np.int64),
        target_rows=np.asarray([0], dtype=np.int64),
        spatial_target_offsets_xy=np.zeros((1, 2, 2), dtype=np.float32),
        spatial_target_observed=np.asarray([[True, False]], dtype=bool),
        spatial_target_supervised=np.ones((1, 2), dtype=bool),
        spatial_target_dustbin=np.zeros((1, 2), dtype=bool),
        correct_projection_offsets_xy=np.zeros((1, 2, 2), dtype=np.float32),
        correct_projection_valid=np.ones((1, 2), dtype=bool),
        wrong_pair_ids=np.asarray([9, 7], dtype=np.int64),
        wrong_projection_offsets_xy=np.zeros((2, 1, 2, 2), dtype=np.float32),
        wrong_projection_valid=np.ones((2, 1, 2), dtype=bool),
    )
    hard_group = HardRepeatQueryTargets(
        query_id=query_id,
        source_point_ids=np.asarray([1], dtype=np.int64),
        pair_ids=np.asarray([7], dtype=np.int64),
        positive_candidate_indices=np.asarray([0], dtype=np.int64),
        negative_candidate_indices=np.asarray([1], dtype=np.int64),
        positive_offsets_xy=np.zeros((1, 2), dtype=np.float32),
        negative_offsets_xy=np.ones((1, 2), dtype=np.float32),
    )
    resolved = resolve_current_system_mined_top_h_mode_positions(
        mined_targets=top_h_targets,
        mined_groups={query_id: hard_group},
        geometry_groups={query_id: geometry},
    )
    np.testing.assert_array_equal(resolved[query_id], np.asarray([1, 0], dtype=np.int64))

    query_without_edges = "q/b.png"
    geometry_without_edges = TrainQueryGroup(
        query_id=query_without_edges,
        source_point_ids=np.asarray([2], dtype=np.int64),
        layout_rows=np.asarray([1], dtype=np.int64),
        target_rows=np.asarray([1], dtype=np.int64),
        spatial_target_offsets_xy=np.zeros((1, 2, 2), dtype=np.float32),
        spatial_target_observed=np.asarray([[True, False]], dtype=bool),
        spatial_target_supervised=np.ones((1, 2), dtype=bool),
        spatial_target_dustbin=np.zeros((1, 2), dtype=bool),
        correct_projection_offsets_xy=np.zeros((1, 2, 2), dtype=np.float32),
        correct_projection_valid=np.ones((1, 2), dtype=bool),
        wrong_pair_ids=np.asarray([5, 3], dtype=np.int64),
        wrong_projection_offsets_xy=np.zeros((2, 1, 2, 2), dtype=np.float32),
        wrong_projection_valid=np.ones((2, 1, 2), dtype=bool),
    )
    top_h_with_unmaterialized_query = CandidatePoseRGBSpatialHardRepeatTargets(
        **{
            **top_h_targets.__dict__,
            "metadata": {
                **top_h_targets.metadata,
                "target_free_ranked_wrong_modes": {
                    **top_h_targets.metadata["target_free_ranked_wrong_modes"],
                    "query_count": 2,
                    "ranked_modes_by_query": {
                        **top_h_targets.metadata["target_free_ranked_wrong_modes"][
                            "ranked_modes_by_query"
                        ],
                        query_without_edges: [
                            {"mode_rank": 0, "mode_index": 0, "pair_id": 5},
                            {"mode_rank": 1, "mode_index": 1, "pair_id": 3},
                        ],
                    },
                },
            },
        }
    )
    resolved_with_unmaterialized_query = resolve_current_system_mined_top_h_mode_positions(
        mined_targets=top_h_with_unmaterialized_query,
        mined_groups={query_id: hard_group},
        geometry_groups={query_id: geometry, query_without_edges: geometry_without_edges},
    )
    np.testing.assert_array_equal(
        resolved_with_unmaterialized_query[query_without_edges],
        np.asarray([0, 1], dtype=np.int64),
    )

    malformed = CandidatePoseRGBSpatialHardRepeatTargets(
        **{
            **top_h_targets.__dict__,
            "metadata": {
                **top_h_targets.metadata,
                "target_free_ranked_wrong_modes": {
                    **top_h_targets.metadata["target_free_ranked_wrong_modes"],
                    "ranked_modes_by_query": {
                        query_id: [
                            {"mode_rank": 0, "mode_index": 1, "pair_id": 9},
                            {"mode_rank": 1, "mode_index": 0, "pair_id": 7},
                        ]
                    },
                },
            },
        }
    )
    with pytest.raises(ValueError, match="differs from frozen geometry"):
        resolve_current_system_mined_top_h_mode_positions(
            mined_targets=malformed,
            mined_groups={query_id: hard_group},
            geometry_groups={query_id: geometry},
        )
def test_rgb_checkpoint_gate_selects_target_free_layout_before_target_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood as module

    events: list[str] = []
    group = SimpleNamespace(layout_rows=np.asarray([40, 41, 42, 43, 44, 45], dtype=np.int64))
    registered_identity_group = SimpleNamespace()
    registered_observed = torch.tensor(
        [[True, False], [False, True], [True, False], [False, True]], dtype=torch.bool
    )
    normal_runtime = SimpleNamespace(
        support_image_indices=torch.tensor([0]), support_xy=torch.zeros((1, 2))
    )
    permuted_runtime = SimpleNamespace(
        support_image_indices=torch.tensor([1]), support_xy=torch.zeros((1, 2))
    )
    batch = SimpleNamespace(
        runtime=normal_runtime,
        correct_projection_offsets_xy=torch.zeros((4, 2, 2), dtype=torch.float32),
        correct_projection_valid=torch.ones((4, 2), dtype=torch.bool),
        wrong_projection_offsets_xy=torch.zeros((1, 4, 2, 2), dtype=torch.float32),
        wrong_projection_valid=torch.ones((1, 4, 2), dtype=torch.bool),
    )

    def fake_selector_input(*, layout: object, rows: np.ndarray) -> object:
        assert layout == "target-free-layout"
        np.testing.assert_array_equal(rows, group.layout_rows)
        events.append("selector_input")
        return SimpleNamespace(point_count=6)

    def fake_selector_scores(*, selector_input: object, policy: str) -> np.ndarray:
        assert selector_input.point_count == 6
        assert policy == "coarse_margin"
        events.append("selector_scores")
        return np.linspace(0.0, 1.0, 6, dtype=np.float32)

    def fake_select(**kwargs: object) -> np.ndarray:
        assert kwargs["point_budget"] == 4
        events.append("select")
        return np.asarray([1, 2, 4, 5], dtype=np.int64)

    def fake_batch(**kwargs: object) -> object:
        np.testing.assert_array_equal(
            kwargs["point_positions"], np.asarray([1, 2, 4, 5], dtype=np.int64)
        )
        events.append("target_join")
        return batch

    def fake_crop(**kwargs: object) -> tuple[object, object]:
        assert kwargs["runtime"] is normal_runtime
        events.append("crop")
        return "query-patches", "support-patches"

    def fake_permute(runtime: object, *, shift: int) -> object:
        assert runtime is normal_runtime
        assert shift == 1
        return permuted_runtime

    def fake_permuted_crop(**kwargs: object) -> object:
        assert kwargs["permuted_runtime"] is permuted_runtime
        assert kwargs["normal_query_patches"] == "query-patches"
        events.append("permuted_crop")
        return "permuted-support-patches"

    def fake_geometry_assert(**kwargs: object) -> None:
        assert kwargs["runtime"] is normal_runtime
        assert kwargs["permuted_runtime"] is permuted_runtime
        events.append("geometry_assert")

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def __call__(self, *, runtime: object, **kwargs: object) -> object:
            assert torch.is_grad_enabled() is False
            assert kwargs["rgb_cost_volume_only"] is False
            events.append("forward:normal" if runtime is normal_runtime else "forward:permuted")
            return runtime

    def fake_score(**kwargs: object) -> object:
        events.append(
            "score:normal" if kwargs["runtime"] is normal_runtime else "score:permuted"
        )
        return SimpleNamespace(pose_log_likelihood_ratios=torch.tensor([0.3]))

    def fake_margin(**kwargs: object) -> tuple[torch.Tensor, dict[str, float]]:
        assert kwargs["correct_scores"].shape == (1,)
        assert kwargs["coherent_wrong_scores"].shape == (1, 1)
        return torch.tensor(0.2), {
            "query_mean_correct_minus_hardest_wrong": 0.1,
            "query_correct_win_fraction": 1.0,
        }

    def fake_registered_join(**kwargs: object) -> object:
        assert kwargs["geometry_group"] is group
        assert kwargs["registered_identity_group"] is registered_identity_group
        np.testing.assert_array_equal(
            kwargs["point_positions"], np.asarray([1, 2, 4, 5], dtype=np.int64)
        )
        events.append("registered_identity_join")
        return SimpleNamespace(target_observed=registered_observed)

    def fake_context_identity(**kwargs: object) -> tuple[torch.Tensor, dict[str, float]]:
        assert torch.equal(kwargs["target_observed"], registered_observed)
        events.append("context_identity")
        return torch.tensor(0.0), {
            "context_identity_cross_entropy": 0.2,
            "context_identity_top1_accuracy": 0.5,
            "context_identity_mean_margin": 0.1,
            "context_identity_active_rows": 4.0,
        }

    def fake_context_permutation(**kwargs: object) -> tuple[torch.Tensor, dict[str, float]]:
        assert torch.equal(kwargs["target_observed"], registered_observed)
        events.append("context_permutation")
        return torch.tensor(0.0), {
            "context_identity_permutation_margin_loss": 0.0,
            "context_identity_permutation_mean_gap": 0.2,
            "context_identity_permutation_win_fraction": 1.0,
            "context_identity_permutation_active_rows": 4.0,
        }

    monkeypatch.setattr(module, "selector_input_from_target_free_layout", fake_selector_input)
    monkeypatch.setattr(module, "target_free_selector_scores", fake_selector_scores)
    monkeypatch.setattr(module, "select_target_free_spatial_quota", fake_select)
    monkeypatch.setattr(module, "_query_batch_from_group", fake_batch)
    monkeypatch.setattr(module, "_crop_runtime_rgb_patches", fake_crop)
    monkeypatch.setattr(module, "permute_runtime_support_image_appearance_only", fake_permute)
    monkeypatch.setattr(module, "_crop_geometry_fixed_permuted_support_patches", fake_permuted_crop)
    monkeypatch.setattr(module, "_assert_geometry_fixed_support_image_control", fake_geometry_assert)
    monkeypatch.setattr(module, "score_candidate_pose_rgb_spatial_batch", fake_score)
    monkeypatch.setattr(module, "query_grouped_pose_margin_loss", fake_margin)
    monkeypatch.setattr(module, "registered_identity_batch_for_geometry_selection", fake_registered_join)
    monkeypatch.setattr(module, "context_identity_cross_entropy_loss", fake_context_identity)
    monkeypatch.setattr(
        module,
        "context_identity_support_permutation_margin_loss",
        fake_context_permutation,
    )

    metrics = _evaluate_inner_validation_target_free_static_selector(
        model=FakeModel(),
        layout="target-free-layout",  # type: ignore[arg-type]
        groups={"query": group},
        complete_runtime="complete-runtime",
        query_ids=("query",),
        image_ids=np.asarray(["image.png"]),
        image_root=module.Path("/unused"),
        coordinate_image_size=(1024, 576),
        rgb_image_size=(1920, 1080),
        radius_px=20.0,
        step_px=0.5,
        cache=SimpleNamespace(),
        state=SimpleNamespace(
            world_size=1,
            rank=0,
            device=torch.device("cpu"),
            enabled=False,
        ),
        selector_policy="coarse_margin",
        selector_point_budget=4,
        selector_grid_rows=2,
        selector_grid_columns=2,
        pose_margin=0.25,
        missing_edge_log_likelihood_ratio=0.0,
        max_abs_pose_log_ratio=6.0,
        amp_enabled=False,
        permutation_control_shift=1,
        seed=7,
        rgb_cost_volume_only=False,
        registered_identity_groups={"query": registered_identity_group},
        include_context_identity_diagnostics=True,
    )

    assert events[:5] == [
        "selector_input",
        "selector_scores",
        "select",
        "target_join",
        "crop",
    ]
    assert "permuted_crop" in events
    assert "geometry_assert" in events
    assert events.index("registered_identity_join") > events.index("forward:permuted")
    assert events.count("context_identity") == 2
    assert "context_permutation" in events
    assert metrics["normal_mean_correct_minus_hardest_wrong"] == pytest.approx(0.1)
    assert metrics["permuted_mean_correct_minus_hardest_wrong"] == pytest.approx(0.1)


def _targets() -> CandidatePoseRGBSpatialTrainingTargets:
    return CandidatePoseRGBSpatialTrainingTargets(
        source_point_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        query_ids=np.asarray(["q/a.png", "q/a.png", "q/b.png", "q/b.png"]),
        spatial_target_offsets_xy=np.zeros((4, 1, 2), dtype=np.float32),
        spatial_target_observed=np.ones((4, 1), dtype=bool),
        spatial_target_dustbin=np.zeros((4, 1), dtype=bool),
        pair_query_ids=np.asarray(["q/a.png", "q/b.png"]),
        pair_ids=np.asarray([5, 7], dtype=np.int64),
        pair_point_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        pair_source_point_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        correct_projection_offsets_xy=np.zeros((4, 1, 2), dtype=np.float32),
        correct_projection_valid=np.ones((4, 1), dtype=bool),
        coherent_wrong_projection_offsets_xy=np.full((4, 1, 2), 1.0, dtype=np.float32),
        coherent_wrong_projection_valid=np.ones((4, 1), dtype=bool),
        metadata={
            "format": "candidate_pose_rgb_spatial_targets_v1",
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "rgb_spatial_layout_sha256": "layout-sha",
            "train_pairs_sha256": "pairs-sha",
            "support_geometry_index_sha256": "geometry-sha",
            "projected_landmark_bank_sha256": "bank-sha",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "spatial_search_radius_px": 2.0,
        },
    )


def test_train_only_lineage_rejects_layout_hash_or_split_drift() -> None:
    layout = _layout()
    targets = _targets()
    validate_training_layout_and_targets(
        layout=layout, targets=targets, layout_sha256="layout-sha"
    )
    with pytest.raises(ValueError, match="hash"):
        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256="other-layout-sha"
        )

    validation_layout = CandidatePoseRGBSpatialLayout(
        **{**layout.__dict__, "split_names": np.asarray(["validation", "train", "train", "train"])}
    )
    with pytest.raises(ValueError, match="not train"):
        validate_training_layout_and_targets(
            layout=validation_layout, targets=targets, layout_sha256="layout-sha"
        )


def test_query_groups_keep_all_wrong_modes_and_source_order() -> None:
    targets = _targets()
    doubled = CandidatePoseRGBSpatialTrainingTargets(
        **{
            **targets.__dict__,
            "pair_query_ids": np.asarray(["q/a.png", "q/a.png", "q/b.png"]),
            "pair_ids": np.asarray([5, 6, 7], dtype=np.int64),
            "pair_point_offsets": np.asarray([0, 2, 4, 6], dtype=np.int64),
            "pair_source_point_ids": np.asarray([1, 2, 1, 2, 3, 4], dtype=np.int64),
            "correct_projection_offsets_xy": np.zeros((6, 1, 2), dtype=np.float32),
            "correct_projection_valid": np.ones((6, 1), dtype=bool),
            "coherent_wrong_projection_offsets_xy": np.full((6, 1, 2), 1.0, dtype=np.float32),
            "coherent_wrong_projection_valid": np.ones((6, 1), dtype=bool),
        }
    )
    groups = build_train_query_groups(layout=_layout(), targets=doubled)

    assert list(groups) == ["q/a.png", "q/b.png"]
    assert groups["q/a.png"].wrong_projection_offsets_xy.shape == (2, 2, 1, 2)
    np.testing.assert_array_equal(groups["q/a.png"].wrong_pair_ids, [5, 6])
    np.testing.assert_array_equal(groups["q/a.png"].source_point_ids, [1, 2])
    assert groups["q/b.png"].wrong_projection_offsets_xy.shape == (1, 2, 1, 2)


def _alignment_group(
    *,
    query_id: str,
    source_point_ids: list[int],
    candidate_count: int = 2,
    observed_candidate_indices: list[int] | None = None,
) -> TrainQueryGroup:
    point_count = len(source_point_ids)
    observed = np.zeros((point_count, candidate_count), dtype=bool)
    if observed_candidate_indices is not None:
        observed[np.arange(point_count), np.asarray(observed_candidate_indices, dtype=np.int64)] = True
    offsets = np.zeros((point_count, candidate_count, 2), dtype=np.float32)
    for row, source_id in enumerate(source_point_ids):
        offsets[row, :, 0] = float(source_id)
        offsets[row, :, 1] = float(-source_id)
    return TrainQueryGroup(
        query_id=query_id,
        source_point_ids=np.asarray(source_point_ids, dtype=np.int64),
        layout_rows=np.arange(point_count, dtype=np.int64),
        target_rows=np.arange(point_count, dtype=np.int64),
        spatial_target_offsets_xy=offsets,
        spatial_target_observed=observed,
        spatial_target_supervised=np.ones((point_count, candidate_count), dtype=bool),
        spatial_target_dustbin=np.zeros((point_count, candidate_count), dtype=bool),
        correct_projection_offsets_xy=np.zeros((point_count, candidate_count, 2), dtype=np.float32),
        correct_projection_valid=np.ones((point_count, candidate_count), dtype=bool),
        wrong_pair_ids=np.asarray([1], dtype=np.int64),
        wrong_projection_offsets_xy=np.zeros(
            (1, point_count, candidate_count, 2), dtype=np.float32
        ),
        wrong_projection_valid=np.ones((1, point_count, candidate_count), dtype=bool),
    )


def test_registered_identity_join_reorders_by_source_and_rejects_mixed_groups() -> None:
    geometry = _alignment_group(
        query_id="q/a.png",
        source_point_ids=[20, 10, 30, 40],
        observed_candidate_indices=[0, 0, 0, 0],
    )
    identity = _alignment_group(
        query_id="q/a.png",
        source_point_ids=[10, 20, 40, 30],
        observed_candidate_indices=[1, 0, 1, 0],
    )
    batch = registered_identity_batch_for_geometry_selection(
        geometry_group=geometry,
        registered_identity_group=identity,
        point_positions=np.asarray([1, 0, 3], dtype=np.int64),
        device=torch.device("cpu"),
    )
    # Geometry positions [1, 0, 3] identify sources [10, 20, 40], regardless
    # of the different registered-target row order.
    np.testing.assert_allclose(
        batch.target_offsets_xy.numpy()[:, 0, 0], np.asarray([10.0, 20.0, 40.0])
    )
    np.testing.assert_array_equal(
        batch.target_observed.numpy(),
        np.asarray([[False, True], [True, False], [False, True]]),
    )
    np.testing.assert_array_equal(
        registered_identity_observed_source_point_ids(identity),
        np.asarray([10, 20, 40, 30], dtype=np.int64),
    )

    missing_source = _alignment_group(
        query_id="q/a.png",
        source_point_ids=[10, 20, 40, 99],
        observed_candidate_indices=[0, 0, 0, 0],
    )
    with pytest.raises(ValueError, match="source-point universe"):
        registered_identity_batch_for_geometry_selection(
            geometry_group=geometry,
            registered_identity_group=missing_source,
            point_positions=np.asarray([0], dtype=np.int64),
            device=torch.device("cpu"),
        )
    wrong_query = _alignment_group(
        query_id="q/b.png",
        source_point_ids=[10, 20, 40, 30],
        observed_candidate_indices=[0, 0, 0, 0],
    )
    with pytest.raises(ValueError, match="query group"):
        registered_identity_batch_for_geometry_selection(
            geometry_group=geometry,
            registered_identity_group=wrong_query,
            point_positions=np.asarray([0], dtype=np.int64),
            device=torch.device("cpu"),
        )


def test_registered_identity_contract_allows_smaller_identity_window_only() -> None:
    geometry = _targets()
    identity_metadata = {
        **geometry.metadata,
        "format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
        "spatial_supervision_mode": "registered_exact_identity",
        "spatial_target_semantics": (
            "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
        ),
        "spatial_class_balance": "per_batch_observed_dustbin_mean_v1",
        "registered_identity_radius_px": 1.5,
        "colmap_images_sha256": "images",
    }
    identity = CandidatePoseRGBSpatialTrainingTargets(
        **{**geometry.__dict__, "metadata": identity_metadata}
    )
    validate_registered_identity_targets_for_geometry(
        layout=_layout(),
        geometry_targets=geometry,
        registered_identity_targets=identity,
        layout_sha256="layout-sha",
    )
    oversized = CandidatePoseRGBSpatialTrainingTargets(
        **{
            **identity.__dict__,
            "metadata": {**identity_metadata, "registered_identity_radius_px": 2.5},
        }
    )
    with pytest.raises(ValueError, match="radius"):
        validate_registered_identity_targets_for_geometry(
            layout=_layout(),
            geometry_targets=geometry,
            registered_identity_targets=oversized,
            layout_sha256="layout-sha",
        )


def test_point_selection_keeps_sparse_observed_and_required_repeat_rows() -> None:
    group = TrainQueryGroup(
        query_id="q/one.png",
        source_point_ids=np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64),
        layout_rows=np.arange(6, dtype=np.int64),
        target_rows=np.arange(6, dtype=np.int64),
        spatial_target_offsets_xy=np.zeros((6, 2, 2), dtype=np.float32),
        spatial_target_observed=np.asarray(
            [[False, False], [True, False], [False, False], [False, True], [False, False], [False, False]]
        ),
        spatial_target_supervised=np.ones((6, 2), dtype=bool),
        spatial_target_dustbin=np.zeros((6, 2), dtype=bool),
        correct_projection_offsets_xy=np.zeros((6, 2, 2), dtype=np.float32),
        correct_projection_valid=np.ones((6, 2), dtype=bool),
        wrong_pair_ids=np.asarray([1], dtype=np.int64),
        wrong_projection_offsets_xy=np.zeros((1, 6, 2, 2), dtype=np.float32),
        wrong_projection_valid=np.ones((1, 6, 2), dtype=bool),
    )
    selected = _select_group_points(
        group=group,
        max_points=4,
        seed=7,
        required_source_point_ids=np.asarray([12], dtype=np.int64),
    )
    np.testing.assert_array_equal(group.source_point_ids[selected], [11, 12, 13, 14])
    dense_required = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    dense_selected = _select_group_points(
        group=group,
        max_points=4,
        seed=7,
        required_source_point_ids=dense_required,
    )
    repeated_dense_selected = _select_group_points(
        group=group,
        max_points=4,
        seed=7,
        required_source_point_ids=dense_required,
    )
    assert len(dense_selected) == 4
    assert set(group.source_point_ids[dense_selected]).issubset(set(dense_required))
    np.testing.assert_array_equal(dense_selected, repeated_dense_selected)


def test_hard_repeat_edge_batch_balances_coherent_wrong_pose_modes() -> None:
    group = TrainQueryGroup(
        query_id="q/repeat.png",
        source_point_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        layout_rows=np.arange(4, dtype=np.int64),
        target_rows=np.arange(4, dtype=np.int64),
        spatial_target_offsets_xy=np.zeros((4, 2, 2), dtype=np.float32),
        spatial_target_observed=np.asarray([[True, False]] * 4),
        spatial_target_supervised=np.ones((4, 2), dtype=bool),
        spatial_target_dustbin=np.zeros((4, 2), dtype=bool),
        correct_projection_offsets_xy=np.zeros((4, 2, 2), dtype=np.float32),
        correct_projection_valid=np.ones((4, 2), dtype=bool),
        wrong_pair_ids=np.asarray([7, 9], dtype=np.int64),
        wrong_projection_offsets_xy=np.zeros((2, 4, 2, 2), dtype=np.float32),
        wrong_projection_valid=np.ones((2, 4, 2), dtype=bool),
    )
    hard = HardRepeatQueryTargets(
        query_id="q/repeat.png",
        source_point_ids=np.tile(group.source_point_ids, 2),
        pair_ids=np.repeat(np.asarray([7, 9], dtype=np.int64), 4),
        positive_candidate_indices=np.zeros((8,), dtype=np.int64),
        negative_candidate_indices=np.ones((8,), dtype=np.int64),
        positive_offsets_xy=np.zeros((8, 2), dtype=np.float32),
        negative_offsets_xy=np.ones((8, 2), dtype=np.float32),
    )
    batch = _hard_repeat_batch_from_group(
        hard_targets=hard,
        group=group,
        point_positions=np.arange(4, dtype=np.int64),
        device=torch.device("cpu"),
        max_edges=4,
        seed=11,
    )
    assert batch is not None
    assert batch.pair_ids is not None
    pair_ids = batch.pair_ids.cpu().numpy()
    assert len(pair_ids) == 4
    assert int(np.count_nonzero(pair_ids == 7)) == 2
    assert int(np.count_nonzero(pair_ids == 9)) == 2


def test_rgb_only_trainable_scope_freezes_context_and_learned_heads() -> None:
    source_grids = {
        "radio_final": torch.nn.functional.normalize(torch.randn(10, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
    )
    names = configure_rgb_cost_volume_only_trainable_parameters(model)
    assert names and all(name.startswith("texture_encoder.") for name in names)
    assert all(
        parameter.requires_grad == name.startswith("texture_encoder.")
        for name, parameter in model.named_parameters()
    )


def test_inner_fold_is_stable_and_gate_requires_visual_signal() -> None:
    train, validation = _partition_train_queries_for_inner_validation(
        query_ids=[f"q/{index}.png" for index in range(10)], fold_count=5, fold_index=1
    )
    assert train and validation and set(train).isdisjoint(validation)
    assert (train, validation) == _partition_train_queries_for_inner_validation(
        query_ids=[f"q/{index}.png" for index in range(10)], fold_count=5, fold_index=1
    )

    passed = training_gate_decision(
        {
            "normal_correct_win_fraction": 0.75,
            "normal_mean_correct_minus_hardest_wrong": 0.30,
            "permuted_mean_correct_minus_hardest_wrong": 0.02,
        },
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
    )
    assert passed["passed"] is True
    failed = training_gate_decision(
        {
            "normal_correct_win_fraction": 0.75,
            "normal_mean_correct_minus_hardest_wrong": 0.30,
            "permuted_mean_correct_minus_hardest_wrong": 0.29,
        },
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
    )
    assert failed["passed"] is False
    assert failed["normal_minus_permuted_gap"] == pytest.approx(0.01)


def test_current_hard_repeat_gate_rejects_easy_generic_pose_ranking() -> None:
    base_metrics = {
        "normal_correct_win_fraction": 0.90,
        "normal_mean_correct_minus_hardest_wrong": 0.30,
        "permuted_mean_correct_minus_hardest_wrong": 0.05,
        "hard_repeat_eligible_query_fraction": 1.0,
        "hard_repeat_correct_win_fraction": 0.30,
        "hard_repeat_mean_correct_minus_coherent_wrong": -0.20,
        "hard_repeat_permuted_mean_correct_minus_coherent_wrong": -0.10,
    }
    rejected = hard_repeat_training_gate_decision(
        base_metrics,
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
        require_hard_repeat=True,
        minimum_hard_repeat_eligible_query_fraction=0.90,
        minimum_hard_repeat_win_fraction=0.55,
        minimum_hard_repeat_gap=0.05,
        minimum_hard_repeat_visual_gap_delta=0.05,
    )
    assert rejected["hard_repeat_passed"] is False
    assert rejected["passed"] is False

    promoted_metrics = {
        **base_metrics,
        "hard_repeat_correct_win_fraction": 0.75,
        "hard_repeat_mean_correct_minus_coherent_wrong": 0.20,
        "hard_repeat_permuted_mean_correct_minus_coherent_wrong": 0.05,
    }
    accepted = hard_repeat_training_gate_decision(
        promoted_metrics,
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
        require_hard_repeat=True,
        minimum_hard_repeat_eligible_query_fraction=0.90,
        minimum_hard_repeat_win_fraction=0.55,
        minimum_hard_repeat_gap=0.05,
        minimum_hard_repeat_visual_gap_delta=0.05,
    )
    assert accepted["hard_repeat_passed"] is True
    assert accepted["passed"] is True


def test_hard_repeat_warmup_is_epoch_only_and_reaches_configured_weight() -> None:
    assert hard_repeat_loss_weight_for_epoch(
        base_weight=0.25, epoch_index=0, warmup_epochs=3
    ) == pytest.approx(0.25 / 3.0)
    assert hard_repeat_loss_weight_for_epoch(
        base_weight=0.25, epoch_index=1, warmup_epochs=3
    ) == pytest.approx(0.25 * 2.0 / 3.0)
    assert hard_repeat_loss_weight_for_epoch(
        base_weight=0.25, epoch_index=2, warmup_epochs=3
    ) == pytest.approx(0.25)
    assert hard_repeat_loss_weight_for_epoch(
        base_weight=0.25, epoch_index=7, warmup_epochs=3
    ) == pytest.approx(0.25)
    assert hard_repeat_loss_weight_for_epoch(
        base_weight=0.25, epoch_index=0, warmup_epochs=0
    ) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="invalid"):
        hard_repeat_loss_weight_for_epoch(
            base_weight=0.25, epoch_index=-1, warmup_epochs=3
        )


def test_rgb_coordinate_bridge_scales_centers_and_offsets_only_isotropically() -> None:
    assert rgb_coordinate_scale(
        coordinate_image_size=(1024, 576), rgb_image_size=(1920, 1080)
    ) == pytest.approx(1.875)
    with pytest.raises(ValueError, match="isotropically"):
        rgb_coordinate_scale(
            coordinate_image_size=(1024, 576), rgb_image_size=(1920, 1000)
        )


def test_rgb_coordinate_bridge_requires_cache_lineage_for_resized_real_rgb() -> None:
    bridge = validate_rgb_coordinate_bridge(
        source_metadata={
            "coordinate_bridge": {
                "format": "normalized_adaptive_grid_coordinate_bridge_v1",
                "aligned_coordinate_size": [1024, 576],
                "raw_rgb_size": [1920, 1080],
                "raw_pixels_per_aligned_pixel": 1.875,
            }
        },
        coordinate_image_size=(1024, 576),
        rgb_image_size=(1920, 1080),
    )
    assert bridge["raw_pixels_per_aligned_pixel"] == pytest.approx(1.875)
    with pytest.raises(ValueError, match="differs"):
        validate_rgb_coordinate_bridge(
            source_metadata={"coordinate_bridge": bridge},
            coordinate_image_size=(1024, 576),
            rgb_image_size=(2048, 1152),
        )


def test_support_permutation_contrastive_loss_requires_visual_gap_and_has_gradients() -> None:
    normal_correct = torch.tensor([0.80], requires_grad=True)
    normal_wrong = torch.tensor([[0.50]])
    permuted_correct = torch.tensor([0.60], requires_grad=True)
    permuted_wrong = torch.tensor([[0.50]])

    loss, metrics = support_permutation_contrastive_loss(
        normal_correct_scores=normal_correct,
        normal_wrong_scores=normal_wrong,
        permuted_correct_scores=permuted_correct,
        permuted_wrong_scores=permuted_wrong,
        margin=0.25,
    )

    # Normal gap=.30, deranged gap=.10, so it misses the .25 visual margin by .05.
    assert float(loss.detach()) == pytest.approx(0.05)
    assert metrics["support_permutation_gap_delta"] == pytest.approx(0.20)
    loss.backward()
    assert float(normal_correct.grad) < 0.0
    assert float(permuted_correct.grad) > 0.0


def test_exact_identity_support_contrastive_uses_only_appearance_difference() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[10.0, 10.0]]),
        support_image_indices=torch.tensor([[[1, 2], [3, 4]]]),
        support_xy=torch.tensor(
            [[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]]]
        ),
        support_view_valid=torch.ones((1, 2, 2), dtype=torch.bool),
        candidate_view_weights=torch.full((1, 2, 2), 0.5),
        candidate_probabilities=torch.tensor([[0.45, 0.45]]),
        null_probabilities=torch.tensor([0.10]),
    )
    permuted_runtime = permute_runtime_support_appearance(runtime, shift=2)
    offsets = torch.tensor(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    probabilities = torch.full((10,), 0.1)
    joint = probabilities.log().reshape(1, 1, 1, 10).expand(1, 2, 2, -1)
    # The local distribution is intentionally identical.  The only trainable
    # difference is the candidate-specific appearance context LLR.
    normal_context = torch.full((1, 2, 2), 0.40, requires_grad=True)
    permuted_context = torch.zeros((1, 2, 2), requires_grad=True)
    normal_prediction = CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=torch.zeros((1, 2, 2, 9)),
        non_dustbin_logits=torch.zeros((1, 2, 2)),
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        context_log_likelihood_ratios=normal_context,
        edge_usable=torch.ones((1, 2, 2), dtype=torch.bool),
    )
    permuted_prediction = CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=torch.zeros((1, 2, 2, 9)),
        non_dustbin_logits=torch.zeros((1, 2, 2)),
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        context_log_likelihood_ratios=permuted_context,
        edge_usable=torch.ones((1, 2, 2), dtype=torch.bool),
    )

    loss, metrics = exact_identity_support_appearance_contrastive_loss(
        runtime=runtime,
        prediction=normal_prediction,
        permuted_runtime=permuted_runtime,
        permuted_prediction=permuted_prediction,
        target_offsets_xy=torch.zeros((1, 2, 2)),
        target_observed=torch.tensor([[True, False]]),
        margin=0.50,
        missing_edge_log_likelihood_ratio=0.0,
        max_abs_pose_log_ratio=6.0,
    )

    assert metrics["identity_support_active_edges"] == 1.0
    # The scorer bounds raw edge LLRs, so a 0.40 context delta is slightly
    # compressed but must remain a positive appearance-only margin.
    assert 0.35 < metrics["identity_support_mean_gap"] < 0.40
    assert metrics["identity_support_normal_win_fraction"] == 1.0
    loss.backward()
    assert float(normal_context.grad[0, 0, 0]) < 0.0
    assert float(permuted_context.grad[0, 0, 0]) > 0.0
    with pytest.raises(ValueError, match="do not match"):
        exact_identity_support_appearance_contrastive_loss(
            runtime=runtime,
            prediction=normal_prediction,
            permuted_runtime=permuted_runtime,
            permuted_prediction=permuted_prediction,
            target_offsets_xy=torch.zeros((1, 2, 2)),
            target_observed=torch.tensor([[True, True]]),
            margin=0.50,
            missing_edge_log_likelihood_ratio=0.0,
            max_abs_pose_log_ratio=6.0,
        )


def test_train_support_permutation_shift_excludes_the_reserved_inner_gate_control() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[10.0, 10.0]]),
        support_image_indices=torch.tensor([[[1, 2, 3]]]),
        support_xy=torch.tensor([[[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]]),
        support_view_valid=torch.ones((1, 1, 3), dtype=torch.bool),
        candidate_view_weights=torch.full((1, 1, 3), 1.0 / 3.0),
        candidate_probabilities=torch.tensor([[0.8]]),
        null_probabilities=torch.tensor([0.2]),
    )
    shift = train_support_permutation_shift(
        runtime=runtime,
        query_id="query/a.png",
        epoch=3,
        reserved_control_shift=1,
    )
    assert shift % 3 not in {0, 1}
    with pytest.raises(ValueError, match="leaves"):
        train_support_permutation_shift(
            runtime=runtime,
            query_id="query/a.png",
            epoch=3,
            reserved_control_shift=3,
        )


def test_coherent_hard_repeat_loss_ranks_distinct_candidate_views() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[10.0, 10.0]]),
        support_image_indices=torch.tensor([[[1], [2]]]),
        support_xy=torch.zeros((1, 2, 1, 2)),
        support_view_valid=torch.ones((1, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 2, 1)),
        candidate_probabilities=torch.tensor([[0.4, 0.4]]),
        null_probabilities=torch.tensor([0.2]),
    )
    local = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    joint = torch.cat([local, torch.tensor([0.55])]).log().reshape(1, 1, 1, 10)
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=torch.zeros((1, 2, 1, 9)),
        non_dustbin_logits=torch.zeros((1, 2, 1)),
        joint_log_probabilities=joint.expand(1, 2, 1, -1).clone(),
        offsets_xy=torch.tensor(
            [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        context_log_likelihood_ratios=torch.zeros((1, 2, 1)),
        edge_usable=torch.ones((1, 2, 1), dtype=torch.bool),
    )
    loss, metrics = coherent_hard_repeat_edge_margin_loss(
        runtime=runtime,
        prediction=prediction,
        hard_batch=HardRepeatBatch(
            point_indices=torch.tensor([0]),
            positive_candidate_indices=torch.tensor([1]),
            negative_candidate_indices=torch.tensor([0]),
            positive_offsets_xy=torch.tensor([[1.0, 1.0]]),
            negative_offsets_xy=torch.tensor([[-1.0, -1.0]]),
        ),
        margin=0.25,
        missing_edge_log_likelihood_ratio=0.0,
        max_abs_pose_log_ratio=6.0,
    )
    assert float(loss) < 0.2
    assert metrics["hard_repeat_active_edges"] == 1.0
    assert metrics["hard_repeat_correct_win_fraction"] == 1.0


def test_coherent_hard_repeat_context_loss_cannot_use_rgb_density_shortcut() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[10.0, 10.0]]),
        support_image_indices=torch.tensor([[[1], [2]]]),
        support_xy=torch.zeros((1, 2, 1, 2)),
        support_view_valid=torch.ones((1, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 2, 1)),
        candidate_probabilities=torch.tensor([[0.4, 0.4]]),
        null_probabilities=torch.tensor([0.2]),
    )
    local = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    joint = torch.cat([local, torch.tensor([0.55])]).log().reshape(1, 1, 1, 10)
    context = torch.tensor([[[-1.0], [1.0]]], requires_grad=True)
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=torch.zeros((1, 2, 1, 9)),
        non_dustbin_logits=torch.zeros((1, 2, 1)),
        joint_log_probabilities=joint.expand(1, 2, 1, -1).clone(),
        offsets_xy=torch.tensor(
            [
                [-1.0, -1.0],
                [0.0, -1.0],
                [1.0, -1.0],
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        context_log_likelihood_ratios=context,
        edge_usable=torch.ones((1, 2, 1), dtype=torch.bool),
    )
    loss, metrics = coherent_hard_repeat_context_margin_loss(
        runtime=runtime,
        prediction=prediction,
        hard_batch=HardRepeatBatch(
            point_indices=torch.tensor([0]),
            positive_candidate_indices=torch.tensor([1]),
            negative_candidate_indices=torch.tensor([0]),
            positive_offsets_xy=torch.tensor([[1.0, 1.0]]),
            negative_offsets_xy=torch.tensor([[-1.0, -1.0]]),
        ),
        margin=0.25,
        missing_edge_log_likelihood_ratio=0.0,
    )
    assert float(loss) < 0.2
    assert metrics["hard_repeat_context_active_edges"] == 1.0
    assert metrics["hard_repeat_context_correct_win_fraction"] == 1.0
    loss.backward()
    assert float(context.grad[0, 1, 0]) < 0.0
    assert float(context.grad[0, 0, 0]) > 0.0


def test_target_free_initialization_checkpoint_requires_matching_lineage_and_config(tmp_path) -> None:
    torch.manual_seed(13)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(4, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(4, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(4, 32, 32, 4), dim=-1),
    }
    model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
    )
    path = tmp_path / "init.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "metadata": {
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "runtime_layout_is_target_free": True,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "fixed_candidate_top_k": 2,
                "fixed_support_view_count": 1,
                "train_only_inner_gate_passed": True,
                "holdout_evaluation_allowed": True,
                "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
                "lineage": {
                    "layout_sha256": "layout",
                    "training_targets_sha256": "targets",
                    "registered_identity_targets_sha256": "registered-identity",
                },
                "config": {
                    "search_radius_px": 2.0,
                    "context_radius_px": 2.0,
                    "step_px": 1.0,
                    "texture_feature_dim": 4,
                    "hidden_dim": 8,
                    "max_abs_context_log_ratio": 3.0,
                    "context_encoder_arch": "conv_v1",
                },
                "training": {"inner_validation": {"selected_epoch": 3}},
            },
        },
        path,
    )
    fresh = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
    )
    result = load_target_free_initialization_checkpoint(
        path=path,
        model=fresh,
        layout_sha256="layout",
        targets_sha256="targets",
        candidate_count=2,
        support_view_count=1,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
    )
    assert result["selected_epoch"] == 3
    assert result["target_lineage"] == "exact_target_hash"
    assert result["inner_gate_evaluator_manifest"] == current_inner_gate_evaluator_manifest()
    matched_identity = load_target_free_initialization_checkpoint(
        path=path,
        model=fresh,
        layout_sha256="layout",
        targets_sha256="targets",
        registered_identity_targets_sha256="registered-identity",
        candidate_count=2,
        support_view_count=1,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
    )
    assert matched_identity["checkpoint_registered_identity_targets_sha256"] == "registered-identity"
    with pytest.raises(ValueError, match="registered-identity lineage"):
        load_target_free_initialization_checkpoint(
            path=path,
            model=fresh,
            layout_sha256="layout",
            targets_sha256="targets",
            registered_identity_targets_sha256="different-identity",
            candidate_count=2,
            support_view_count=1,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )

    identity_free_path = tmp_path / "identity_free_init.pt"
    identity_free_payload = torch.load(path, map_location="cpu", weights_only=False)
    identity_free_lineage = identity_free_payload["metadata"]["lineage"]
    identity_free_lineage.pop("registered_identity_targets_sha256")
    identity_free_payload["metadata"]["training"] = {
        "inner_validation": {"selected_epoch": 3},
        "registered_identity_supervision": {"enabled": False},
    }
    torch.save(identity_free_payload, identity_free_path)
    with pytest.raises(ValueError, match="registered-identity lineage"):
        load_target_free_initialization_checkpoint(
            path=identity_free_path,
            model=fresh,
            layout_sha256="layout",
            targets_sha256="targets",
            registered_identity_targets_sha256="registered-identity",
            candidate_count=2,
            support_view_count=1,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )
    introduced = load_target_free_initialization_checkpoint(
        path=identity_free_path,
        model=fresh,
        layout_sha256="layout",
        targets_sha256="targets",
        registered_identity_targets_sha256="registered-identity",
        allow_registered_identity_sidecar_introduction_from_identity_free_parent=True,
        candidate_count=2,
        support_view_count=1,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
    )
    assert introduced["registered_identity_lineage_transition"] == (
        "explicit_sidecar_introduction_from_identity_free_parent_v1"
    )

    identity_free_payload["metadata"]["training"] = {
        "inner_validation": {"selected_epoch": 3},
        "registered_candidate_context_identity": {"enabled": False},
        "registered_candidate_context_support_appearance_derangement": {
            "enabled": False
        },
        "exact_identity_support_appearance_contrastive": {"enabled": False},
    }
    torch.save(identity_free_payload, identity_free_path)
    legacy_introduced = load_target_free_initialization_checkpoint(
        path=identity_free_path,
        model=fresh,
        layout_sha256="layout",
        targets_sha256="targets",
        registered_identity_targets_sha256="registered-identity",
        allow_registered_identity_sidecar_introduction_from_identity_free_parent=True,
        candidate_count=2,
        support_view_count=1,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
    )
    assert legacy_introduced["registered_identity_lineage_transition"] == (
        "explicit_sidecar_introduction_from_identity_free_parent_v1"
    )

    identity_free_payload["metadata"]["training"]["registered_identity_supervision"] = {
        "enabled": True
    }
    torch.save(identity_free_payload, identity_free_path)
    with pytest.raises(ValueError, match="identity-free parent status is unproven"):
        load_target_free_initialization_checkpoint(
            path=identity_free_path,
            model=fresh,
            layout_sha256="layout",
            targets_sha256="targets",
            registered_identity_targets_sha256="registered-identity",
            allow_registered_identity_sidecar_introduction_from_identity_free_parent=True,
            candidate_count=2,
            support_view_count=1,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )
    with pytest.raises(ValueError, match="stale"):
        load_target_free_initialization_checkpoint(
            path=path,
            model=fresh,
            layout_sha256="layout",
            targets_sha256="expanded-targets",
            candidate_count=2,
            support_view_count=1,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )
    continued = load_target_free_initialization_checkpoint(
        path=path,
        model=fresh,
        layout_sha256="layout",
        targets_sha256="expanded-targets",
        permitted_parent_target_sha256s=("targets",),
        candidate_count=2,
        support_view_count=1,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
    )
    assert continued["target_lineage"] == "declared_train_only_parent_target_continuation"
    assert continued["checkpoint_training_targets_sha256"] == "targets"
    with pytest.raises(ValueError, match="stale"):
        load_target_free_initialization_checkpoint(
            path=path,
            model=fresh,
            layout_sha256="different",
            targets_sha256="targets",
            candidate_count=2,
            support_view_count=1,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )
    with pytest.raises(ValueError, match="context architecture differs"):
        load_target_free_initialization_checkpoint(
            path=path,
            model=fresh,
            layout_sha256="layout",
            targets_sha256="targets",
            candidate_count=2,
            support_view_count=1,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_encoder_arch="cross_attention_v2",
        )

    stale_evaluator_payload = torch.load(path, map_location="cpu", weights_only=False)
    stale_evaluator_payload["metadata"]["inner_gate_evaluator_manifest"] = {
        "format": "stale"
    }
    stale_evaluator_path = tmp_path / "stale_evaluator.pt"
    torch.save(stale_evaluator_payload, stale_evaluator_path)
    with pytest.raises(ValueError, match="evaluator manifest"):
        load_target_free_initialization_checkpoint(
            path=stale_evaluator_path,
            model=fresh,
            layout_sha256="layout",
            targets_sha256="targets",
            candidate_count=2,
            support_view_count=1,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )


def test_observation_pretrain_initializer_requires_passed_gate_and_exact_visual_lineage(
    tmp_path,
) -> None:
    torch.manual_seed(17)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(4, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(4, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(4, 32, 32, 4), dim=-1),
    }
    context_windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    source_cache_paths = {
        "radio_final": tmp_path / "radio_final.npz",
        "radio_intermediate": tmp_path / "radio_intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for name, path in source_cache_paths.items():
        path.write_bytes(name.encode("ascii"))
    bridge = {
        "format": "identity_rgb_coordinate_bridge_v1",
        "aligned_coordinate_size": [64, 64],
        "raw_rgb_size": [64, 64],
        "raw_pixels_per_aligned_pixel": 1.0,
    }
    metadata = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "p1_finetune_allowed": True,
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
        ],
        "config": {
            "search_radius_px": 2.0,
            "context_radius_px": 2.0,
            "step_px": 1.0,
            "texture_feature_dim": 4,
            "hidden_dim": 8,
            "max_abs_context_log_ratio": 3.0,
            "context_windows": context_windows,
            "context_encoder_arch": "conv_v1",
        },
        "training": {
            "inner_validation": {"selected_epoch": 2, "gate": {"passed": True}}
        },
        "lineage": {
            "radio_final_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_final"]
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_intermediate"]
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(
                source_cache_paths["alike"]
            ),
            "source_image_manifest_sha256": "manifest",
            "rgb_coordinate_bridge": bridge,
        },
    }
    path = tmp_path / "observation_pretrain.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    fresh = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    result = load_observation_pretrain_initialization_checkpoint(
        path=path,
        model=fresh,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        rgb_coordinate_bridge=bridge,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=context_windows,
    )
    assert result["kind"] == "gate_approved_observation_pretrain"
    assert result["selected_epoch"] == 2

    blocked_path = tmp_path / "blocked_observation_pretrain.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "metadata": {**metadata, "p1_finetune_allowed": False},
        },
        blocked_path,
    )
    with pytest.raises(ValueError, match="not eligible"):
        load_observation_pretrain_initialization_checkpoint(
            path=blocked_path,
            model=fresh,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
        )
    different_radio_final = tmp_path / "different_radio_final.npz"
    different_radio_final.write_bytes(b"different")
    with pytest.raises(ValueError, match="source cache lineage"):
        load_observation_pretrain_initialization_checkpoint(
            path=path,
            model=fresh,
            source_cache_paths={
                **source_cache_paths,
                "radio_final": different_radio_final,
            },
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
        )


def test_texture_observation_component_initializer_loads_only_texture_with_exact_geometry(
    tmp_path,
) -> None:
    torch.manual_seed(23)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(4, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(4, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(4, 32, 32, 4), dim=-1),
    }
    context_windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    source_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    with torch.no_grad():
        for parameter in source_model.texture_encoder.parameters():
            parameter.fill_(0.625)
        for parameter in source_model.context_encoders.parameters():
            parameter.fill_(0.125)
    source_cache_paths = {
        "radio_final": tmp_path / "radio_final.npz",
        "radio_intermediate": tmp_path / "radio_intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for name, source_path in source_cache_paths.items():
        source_path.write_bytes(name.encode("ascii"))
    bridge = {
        "format": "identity_rgb_coordinate_bridge_v1",
        "aligned_coordinate_size": [64, 64],
        "raw_rgb_size": [64, 64],
        "raw_pixels_per_aligned_pixel": 1.0,
    }
    metadata = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "p1_finetune_allowed": True,
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
        ],
        "config": {
            "search_radius_px": 2.0,
            "context_radius_px": 2.0,
            "step_px": 1.0,
            "texture_feature_dim": 4,
            "hidden_dim": 8,
            "max_abs_context_log_ratio": 3.0,
            "rgb_cost_volume_only": True,
        },
        "training": {
            "inner_validation": {"selected_epoch": 2, "gate": {"passed": True}}
        },
        "lineage": {
            "radio_final_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_final"]
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_intermediate"]
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(
                source_cache_paths["alike"]
            ),
            "source_image_manifest_sha256": "manifest",
            "rgb_coordinate_bridge": bridge,
        },
    }
    checkpoint_path = tmp_path / "texture_observation_pretrain.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": source_model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    torch.manual_seed(29)
    target_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    context_before = next(target_model.context_encoders.parameters()).detach().clone()
    result = load_texture_observation_pretrain_component_checkpoint(
        path=checkpoint_path,
        model=target_model,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        rgb_coordinate_bridge=bridge,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
    )
    assert result["loaded_parameter_scope"] == "texture_encoder_only"
    assert torch.allclose(
        next(target_model.texture_encoder.parameters()),
        torch.full_like(next(target_model.texture_encoder.parameters()), 0.625),
    )
    assert torch.equal(next(target_model.context_encoders.parameters()), context_before)
    with pytest.raises(ValueError, match="config differs"):
        load_texture_observation_pretrain_component_checkpoint(
            path=checkpoint_path,
            model=target_model,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=0.5,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )


def test_hard_repeat_gated_rgb_texture_initializer_transfers_only_texture_with_exact_p1_lineage(
    tmp_path,
) -> None:
    torch.manual_seed(43)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(4, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(4, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(4, 32, 32, 4), dim=-1),
    }
    context_windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    source_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    with torch.no_grad():
        for parameter in source_model.texture_encoder.parameters():
            parameter.fill_(0.875)
        for parameter in source_model.context_encoders.parameters():
            parameter.fill_(0.125)
    source_cache_paths = {
        "radio_final": tmp_path / "radio_final.npz",
        "radio_intermediate": tmp_path / "radio_intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for name, source_path in source_cache_paths.items():
        source_path.write_bytes(name.encode("ascii"))
    layout_path = tmp_path / "layout.npz"
    target_path = tmp_path / "targets.npz"
    hard_repeat_path = tmp_path / "hard_repeat_targets.npz"
    layout_path.write_bytes(b"layout")
    target_path.write_bytes(b"targets")
    hard_repeat_path.write_bytes(b"hard-repeat")
    bridge = {
        "format": "identity_rgb_coordinate_bridge_v1",
        "aligned_coordinate_size": [64, 64],
        "raw_rgb_size": [64, 64],
        "raw_pixels_per_aligned_pixel": 1.0,
    }
    layout_sha256 = file_sha256_short(layout_path)
    targets_sha256 = file_sha256_short(target_path)
    hard_repeat_targets_sha256 = file_sha256_short(hard_repeat_path)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "architecture": "real_rgb_fpn_candidate_specific_high_resolution_cost_volume_only_v1",
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 1,
        "fixed_support_view_count": 1,
        "explicit_null": True,
        "projection_after_network_only": True,
        "out_of_window_projection_semantics": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
        "edge_source_availability": dict(SOURCE_SAFE_EDGE_AVAILABILITY),
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "train_only_inner_gate_passed": False,
        "holdout_evaluation_allowed": False,
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
        ],
        "config": {
            "search_radius_px": 2.0,
            "context_radius_px": 2.0,
            "step_px": 1.0,
            "max_abs_context_log_ratio": 3.0,
            "texture_feature_dim": 4,
            "hidden_dim": 8,
            "rgb_cost_volume_only": True,
            "trainable_parameter_scope": "texture_encoder_only",
            "context_windows": context_windows,
            "context_encoder_arch": "conv_v1",
            "hard_repeat_inner_gate": {
                "required_when_targets_loaded": True,
                "common_normal_permuted_availability_only": True,
            },
        },
        "training": {
            "inner_validation": {
                "selected_epoch": 2,
                "gate": {
                    "passed": False,
                    "hard_repeat_required": True,
                    "hard_repeat_passed": True,
                    "hard_repeat_correct_win_fraction": 0.70,
                    "hard_repeat_mean_correct_minus_coherent_wrong": 0.60,
                    "hard_repeat_minus_permuted_gap": 0.12,
                },
            }
        },
        "lineage": {
            "layout_sha256": layout_sha256,
            "training_targets_sha256": targets_sha256,
            "hard_repeat_targets_sha256": hard_repeat_targets_sha256,
            "source_image_manifest_sha256": "manifest",
            "rgb_coordinate_bridge": bridge,
        },
        "inputs": {
            "rgb_spatial_layout": {"sha256": layout_sha256},
            "training_targets": {"sha256": targets_sha256},
            "hard_repeat_targets": {"sha256": hard_repeat_targets_sha256},
            "radio_final_context_cache": {
                "sha256": file_sha256_short(source_cache_paths["radio_final"])
            },
            "radio_intermediate_context_cache": {
                "sha256": file_sha256_short(source_cache_paths["radio_intermediate"])
            },
            "alike_spatial_context_cache": {
                "sha256": file_sha256_short(source_cache_paths["alike"])
            },
        },
    }
    checkpoint_path = tmp_path / "hard_repeat_rgb_texture.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": source_model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    torch.manual_seed(47)
    target_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
    )
    context_before = next(target_model.context_encoders.parameters()).detach().clone()
    result = load_hard_repeat_gated_rgb_texture_initialization_checkpoint(
        path=checkpoint_path,
        model=target_model,
        layout_sha256=layout_sha256,
        targets_sha256=targets_sha256,
        hard_repeat_targets_sha256=hard_repeat_targets_sha256,
        candidate_count=1,
        support_view_count=1,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        rgb_coordinate_bridge=bridge,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
    )
    assert result["kind"] == "hard_repeat_gated_rgb_texture_component_initializer_v1"
    assert result["loaded_parameter_scope"] == "texture_encoder_only"
    assert result["source_checkpoint_overall_gate_passed"] is False
    assert torch.allclose(
        next(target_model.texture_encoder.parameters()),
        torch.full_like(next(target_model.texture_encoder.parameters()), 0.875),
    )
    assert torch.equal(next(target_model.context_encoders.parameters()), context_before)
    with pytest.raises(ValueError, match="lineage is stale"):
        load_hard_repeat_gated_rgb_texture_initialization_checkpoint(
            path=checkpoint_path,
            model=target_model,
            layout_sha256=layout_sha256,
            targets_sha256=targets_sha256,
            hard_repeat_targets_sha256="different",
            candidate_count=1,
            support_view_count=1,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )

    blocked_path = tmp_path / "hard_repeat_rgb_texture_blocked.pt"
    blocked_gate = {
        **metadata["training"]["inner_validation"]["gate"],
        "hard_repeat_passed": False,
    }
    blocked_metadata = {
        **metadata,
        "training": {
            **metadata["training"],
            "inner_validation": {
                **metadata["training"]["inner_validation"],
                "gate": blocked_gate,
            },
        },
    }
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": source_model.state_dict(),
            "metadata": blocked_metadata,
        },
        blocked_path,
    )
    with pytest.raises(ValueError, match="did not pass its direct hard-repeat gate"):
        load_hard_repeat_gated_rgb_texture_initialization_checkpoint(
            path=blocked_path,
            model=target_model,
            layout_sha256=layout_sha256,
            targets_sha256=targets_sha256,
            hard_repeat_targets_sha256=hard_repeat_targets_sha256,
            candidate_count=1,
            support_view_count=1,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
        )


def test_identity_llr_texture_initializer_requires_v5_gate_and_loads_only_texture(tmp_path) -> None:
    torch.manual_seed(37)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(4, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(4, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(4, 32, 32, 4), dim=-1),
    }
    context_windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    identity_model = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        rgb_context_radius_px=4.0,
        rgb_step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    with torch.no_grad():
        for parameter in identity_model.texture_encoder.parameters():
            parameter.fill_(0.625)
    source_cache_paths = {
        "radio_final": tmp_path / "radio_final.npz",
        "radio_intermediate": tmp_path / "radio_intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for name, source_path in source_cache_paths.items():
        source_path.write_bytes(name.encode("ascii"))
    bridge = {
        "format": "identity_rgb_coordinate_bridge_v1",
        "aligned_coordinate_size": [64, 64],
        "raw_rgb_size": [64, 64],
        "raw_pixels_per_aligned_pixel": 1.0,
    }
    metadata = {
        "format": "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v2",
        "model_format": "candidate_pose_rgb_spatial_identity_llr_v3",
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "candidate_slot_permutation_equivariant": True,
        "p1_initialization_allowed": True,
        "fixed_candidate_count": 1,
        "visual_evidence_gate_version": (
            "combined_visual_controls_plus_conditional_source_visual_ablation_and_hard_pose_v5"
        ),
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
        ],
        "config": {
            "rgb_context_radius_px": 4.0,
            "rgb_step_px": 1.0,
            "texture_feature_dim": 4,
            "hidden_dim": 8,
            "max_abs_log_ratio": 4.0,
        },
        "training": {
            "inner_validation": {"selected_epoch": 1, "gate": {"passed": True}}
        },
        "lineage": {
            "radio_final_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_final"]
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_intermediate"]
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(
                source_cache_paths["alike"]
            ),
            "source_image_manifest_sha256": "manifest",
            "rgb_coordinate_bridge": bridge,
        },
    }
    checkpoint_path = tmp_path / "identity_llr_texture.pt"
    torch.save(
        {
            "format": "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v2",
            "state_dict": identity_model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    torch.manual_seed(41)
    target_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=0.5,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    context_before = next(target_model.context_encoders.parameters()).detach().clone()
    result = load_identity_llr_texture_pretrain_initialization_checkpoint(
        path=checkpoint_path,
        model=target_model,
        candidate_count=1,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        rgb_coordinate_bridge=bridge,
        texture_feature_dim=4,
        hidden_dim=8,
    )
    assert result["loaded_parameter_scope"] == "texture_encoder_only"
    assert result["source_rgb_patch_geometry"] == {"context_radius_px": 4.0, "step_px": 1.0}
    assert torch.allclose(
        next(target_model.texture_encoder.parameters()),
        torch.full_like(next(target_model.texture_encoder.parameters()), 0.625),
    )
    assert torch.equal(next(target_model.context_encoders.parameters()), context_before)

    blocked_path = tmp_path / "identity_llr_texture_blocked.pt"
    torch.save(
        {
            "format": "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v2",
            "state_dict": identity_model.state_dict(),
            "metadata": {**metadata, "visual_evidence_gate_version": "wrong_gate"},
        },
        blocked_path,
    )
    with pytest.raises(ValueError, match="not eligible"):
        load_identity_llr_texture_pretrain_initialization_checkpoint(
            path=blocked_path,
            model=target_model,
            candidate_count=1,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            texture_feature_dim=4,
            hidden_dim=8,
        )

    mismatched_identity_model = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        rgb_context_radius_px=4.0,
        rgb_step_px=1.0,
        texture_feature_dim=5,
        hidden_dim=8,
        context_windows=context_windows,
    )
    mismatch_path = tmp_path / "identity_llr_texture_shape_mismatch.pt"
    torch.save(
        {
            "format": "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v2",
            "state_dict": mismatched_identity_model.state_dict(),
            # Keep the claimed config compatible to ensure the state contract,
            # rather than only metadata, rejects incompatible tensors.
            "metadata": metadata,
        },
        mismatch_path,
    )
    with pytest.raises(ValueError, match="state dict is incompatible"):
        load_identity_llr_texture_pretrain_initialization_checkpoint(
            path=mismatch_path,
            model=target_model,
            candidate_count=1,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            texture_feature_dim=4,
            hidden_dim=8,
        )


def test_context_observation_initializer_loads_only_context_with_exact_lineage(tmp_path) -> None:
    torch.manual_seed(31)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(4, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(4, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(4, 32, 32, 4), dim=-1),
    }
    context_windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    source_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
    )
    with torch.no_grad():
        for parameter in source_model.context_encoders.parameters():
            parameter.fill_(0.125)
        for parameter in source_model.context_identity_head.parameters():
            parameter.fill_(0.25)
        for parameter in source_model.texture_encoder.parameters():
            parameter.fill_(0.875)
    source_cache_paths = {
        "radio_final": tmp_path / "radio_final.npz",
        "radio_intermediate": tmp_path / "radio_intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for name, source_path in source_cache_paths.items():
        source_path.write_bytes(name.encode("ascii"))
    metadata = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "p1_context_transfer_allowed": True,
        "raw_scores_must_not_feed_pnp": True,
        "appearance_control_geometry_fixed": True,
        "checkpoint_selection_policy": (
            "fixed_final_epoch_without_inner_validation_model_selection_v1"
        ),
        "inner_validation_used_for_model_selection": False,
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
            "rgb_patch",
        ],
        "config": {
            "search_radius_px": 2.0,
            "context_radius_px": 2.0,
            "step_px": 1.0,
            "texture_feature_dim": 4,
            "hidden_dim": 8,
            "max_abs_context_log_ratio": 3.0,
            "context_windows": context_windows,
            "context_encoder_arch": "absolute_cross_attention_v3",
            "context_only": True,
        },
        "training": {
            "inner_validation": {
                "selected_epoch": 2,
                "checkpoint_selection": {
                    "policy": "fixed_final_epoch_without_inner_validation_model_selection_v1",
                    "selected_epoch": 2,
                    "inner_validation_used_for_model_selection": False,
                },
                "gate": {"passed": True},
            }
        },
        "lineage": {
            "radio_final_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_final"]
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_intermediate"]
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(
                source_cache_paths["alike"]
            ),
            "source_image_manifest_sha256": "manifest",
        },
    }
    checkpoint_path = tmp_path / "context_observation_pretrain.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": source_model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    torch.manual_seed(37)
    target_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
    )
    texture_before = next(target_model.texture_encoder.parameters()).detach().clone()
    result = load_context_observation_pretrain_initialization_checkpoint(
        path=checkpoint_path,
        model=target_model,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
    )
    assert result["kind"] == "gate_approved_context_observation_pretrain_context_only"
    assert result["loaded_parameter_scope"] == "context_encoders_and_context_identity_head_only"
    assert torch.allclose(next(target_model.context_encoders.parameters()), torch.full_like(
        next(target_model.context_encoders.parameters()), 0.125
    ))
    assert torch.allclose(next(target_model.context_identity_head.parameters()), torch.full_like(
        next(target_model.context_identity_head.parameters()), 0.25
    ))
    assert torch.equal(next(target_model.texture_encoder.parameters()), texture_before)

    pair_path = tmp_path / "observation_pairs.npz"
    np.savez(
        pair_path,
        query_image_ids=np.asarray(["train-a", "train-a", "val-b"], dtype="U16"),
        split_names=np.asarray(["inner_train", "inner_train", "inner_validation"], dtype="U16"),
    )
    split_metadata = {
        **metadata,
        "lineage": {
            **metadata["lineage"],
            "observation_pairs_sha256": file_sha256_short(pair_path),
        },
    }
    split_checkpoint_path = tmp_path / "context_observation_pretrain_split.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": source_model.state_dict(),
            "metadata": split_metadata,
        },
        split_checkpoint_path,
    )
    split_result = load_context_observation_pretrain_initialization_checkpoint(
        path=split_checkpoint_path,
        model=CandidatePoseRGBSpatialLikelihood(
            sources=sources,
            image_sizes=torch.full((4, 2), 64.0),
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            context_windows=context_windows,
            context_encoder_arch="absolute_cross_attention_v3",
        ),
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
        observation_pairs_path=pair_path,
        expected_pretrain_train_query_ids=("train-a",),
        expected_pretrain_validation_query_ids=("val-b",),
    )
    assert split_result["query_split_lineage"] == {
        "observation_pairs_path": str(pair_path),
        "observation_pairs_sha256": file_sha256_short(pair_path),
        "inner_train_query_count": 1,
        "inner_validation_query_count": 1,
        "exact_p1_fold_match": True,
    }
    with pytest.raises(ValueError, match="does not match the P1 fold"):
        load_context_observation_pretrain_initialization_checkpoint(
            path=split_checkpoint_path,
            model=CandidatePoseRGBSpatialLikelihood(
                sources=sources,
                image_sizes=torch.full((4, 2), 64.0),
                search_radius_px=2.0,
                context_radius_px=2.0,
                step_px=1.0,
                texture_feature_dim=4,
                hidden_dim=8,
                context_windows=context_windows,
                context_encoder_arch="absolute_cross_attention_v3",
            ),
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
            context_encoder_arch="absolute_cross_attention_v3",
            observation_pairs_path=pair_path,
            expected_pretrain_train_query_ids=("val-b",),
            expected_pretrain_validation_query_ids=("train-a",),
        )

    step_mismatch_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=0.5,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
    )
    mismatch_texture_before = next(step_mismatch_model.texture_encoder.parameters()).detach().clone()
    with pytest.raises(ValueError, match="config differs"):
        load_context_observation_pretrain_initialization_checkpoint(
            path=checkpoint_path,
            model=step_mismatch_model,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=0.5,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
            context_encoder_arch="absolute_cross_attention_v3",
        )
    step_result = load_context_observation_pretrain_initialization_checkpoint(
        path=checkpoint_path,
        model=step_mismatch_model,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=0.5,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
        allow_rgb_step_mismatch=True,
    )
    assert step_result["rgb_step_px_component_invariant_mismatch"] is True
    assert torch.allclose(
        next(step_mismatch_model.context_encoders.parameters()),
        torch.full_like(next(step_mismatch_model.context_encoders.parameters()), 0.125),
    )
    assert torch.equal(next(step_mismatch_model.texture_encoder.parameters()), mismatch_texture_before)

    radius_mismatch_model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=3.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
    )
    with pytest.raises(ValueError, match="config differs"):
        load_context_observation_pretrain_initialization_checkpoint(
            path=checkpoint_path,
            model=radius_mismatch_model,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            search_radius_px=3.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
            context_encoder_arch="absolute_cross_attention_v3",
        )
    radius_result = load_context_observation_pretrain_initialization_checkpoint(
        path=checkpoint_path,
        model=radius_mismatch_model,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        search_radius_px=3.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=context_windows,
        context_encoder_arch="absolute_cross_attention_v3",
        allow_spatial_search_radius_mismatch=True,
    )
    assert radius_result["spatial_search_radius_component_invariant_mismatch"] is True

    blocked_path = tmp_path / "blocked_context_observation_pretrain.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": source_model.state_dict(),
            "metadata": {**metadata, "p1_context_transfer_allowed": False},
        },
        blocked_path,
    )
    with pytest.raises(ValueError, match="not eligible"):
        load_context_observation_pretrain_initialization_checkpoint(
            path=blocked_path,
            model=target_model,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
            context_encoder_arch="absolute_cross_attention_v3",
        )


def test_hard_pose_pretrain_initializer_rejects_other_pretrain_formats(tmp_path) -> None:
    torch.manual_seed(19)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(4, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(4, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(4, 32, 32, 4), dim=-1),
    }
    context_windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    model = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    source_cache_paths = {
        "radio_final": tmp_path / "radio_final.npz",
        "radio_intermediate": tmp_path / "radio_intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for name, source_path in source_cache_paths.items():
        source_path.write_bytes(name.encode("ascii"))
    bridge = {
        "format": "identity_rgb_coordinate_bridge_v1",
        "aligned_coordinate_size": [64, 64],
        "raw_rgb_size": [64, 64],
        "raw_pixels_per_aligned_pixel": 1.0,
    }
    metadata = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "p1_finetune_allowed": True,
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
        ],
        "config": {
            "search_radius_px": 2.0,
            "context_radius_px": 2.0,
            "step_px": 1.0,
            "texture_feature_dim": 4,
            "hidden_dim": 8,
            "max_abs_context_log_ratio": 3.0,
            "context_windows": context_windows,
            "context_encoder_arch": "conv_v1",
            "fixed_candidate_null_mass": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS,
        },
        "training": {
            "objective": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_OBJECTIVE,
            "inner_validation": {"selected_epoch": 3, "gate": {"passed": True}},
        },
        "lineage": {
            "radio_final_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_final"]
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_intermediate"]
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(
                source_cache_paths["alike"]
            ),
            "source_image_manifest_sha256": "manifest",
            "rgb_coordinate_bridge": bridge,
        },
    }
    path = tmp_path / "hard_pose_pretrain.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    fresh = CandidatePoseRGBSpatialLikelihood(
        sources=sources,
        image_sizes=torch.full((4, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        context_windows=context_windows,
    )
    result = load_hard_pose_pretrain_initialization_checkpoint(
        path=path,
        model=fresh,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        rgb_coordinate_bridge=bridge,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=context_windows,
    )
    assert result["kind"] == "gate_approved_hard_pose_pretrain"
    diagnostic_path = tmp_path / "context_identity_diagnostic.pt"
    diagnostic_metadata = {
        **metadata,
        "p1_finetune_allowed": False,
        "config": {**metadata["config"], "context_only": True},
        "training": {
            "objective": CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE,
            "inner_validation": {"selected_epoch": 1, "gate": {"passed": False}},
        },
    }
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "metadata": diagnostic_metadata,
        },
        diagnostic_path,
    )
    with pytest.raises(ValueError, match="not eligible"):
        load_hard_pose_pretrain_initialization_checkpoint(
            path=diagnostic_path,
            model=fresh,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
        )
    diagnostic = load_hard_pose_pretrain_initialization_checkpoint(
        path=diagnostic_path,
        model=fresh,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        rgb_coordinate_bridge=bridge,
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=context_windows,
        allow_diagnostic_ineligible=True,
    )
    assert diagnostic["kind"] == "diagnostic_ineligible_hard_pose_pretrain"
    assert diagnostic["eligible_for_p1_finetune"] is False
    assert diagnostic["diagnostic_override"] is True
    wrong_format_path = tmp_path / "wrong_format.pt"
    torch.save(
        {
            "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "metadata": {
                **metadata,
                "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            },
        },
        wrong_format_path,
    )
    with pytest.raises(ValueError, match="not eligible"):
        load_hard_pose_pretrain_initialization_checkpoint(
            path=wrong_format_path,
            model=fresh,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            rgb_coordinate_bridge=bridge,
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=context_windows,
        )
