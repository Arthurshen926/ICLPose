from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_identity_llr import (
    CHECKPOINT_FORMAT,
    OBSERVATION_IDENTITY_GATE_VERSION,
    _combined_component_gain_statistics,
    _component_visual_ablation_scales,
    _fixed_view_prediction,
    _load_identity_observation_initializer,
    _load_visual_representation_initializer,
    _hard_pose_identity_statistics,
    _identity_pretrain_gate,
    _permuted_hard_pose_negative_mask,
    _row_identity_statistics,
    _support_derangement_loss,
    _visual_content_ablation_loss,
    _conditional_visual_ablation_loss,
    conditional_source_visual_ablation_gate,
    independent_source_masked_observation_gate,
    observation_identity_gate,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_V2_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES,
    CandidatePoseRGBSpatialIdentityLLR,
    CandidatePoseRGBSpatialIdentityLLREdgePrediction,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[32.0, 32.0], [48.0, 48.0]]),
        support_image_indices=torch.tensor([[[2], [3]], [[2], [3]]]),
        support_xy=torch.full((2, 2, 1, 2), 48.0),
        support_view_valid=torch.ones((2, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 2, 1), dtype=torch.float32),
        candidate_probabilities=torch.full((2, 2), 0.5),
        null_probabilities=torch.zeros((2,), dtype=torch.float32),
    )


def _prediction(values: torch.Tensor) -> CandidatePoseRGBSpatialIdentityLLREdgePrediction:
    return CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=values.unsqueeze(2),
        edge_usable=torch.ones((*values.shape, 1), dtype=torch.bool),
    )


def test_observation_identity_statistics_and_derangement_are_target_joined_after_scoring() -> None:
    runtime = _runtime()
    targets = torch.tensor([[True, False], [False, True]])
    normal = _prediction(torch.tensor([[2.0, -1.0], [-1.0, 3.0]]))
    deranged = _prediction(torch.tensor([[0.25, -1.0], [-1.0, 0.5]]))
    ce, margin, top1, active = _row_identity_statistics(
        runtime=runtime, prediction=normal, targets=targets
    )
    loss, metrics = _support_derangement_loss(
        runtime=runtime,
        prediction=normal,
        deranged_runtime=runtime,
        deranged_prediction=deranged,
        targets=targets,
        margin=0.25,
    )
    assert active.tolist() == [True, True]
    assert torch.all(margin > 0.0)
    assert top1.tolist() == [True, True]
    assert torch.all(ce > 0.0)
    assert float(loss) > 0.0
    assert metrics["mean_gap"] > 0.0
    position_loss, position_metrics = _visual_content_ablation_loss(
        runtime=runtime,
        prediction=normal,
        position_only_prediction=deranged,
        targets=targets,
        margin=0.25,
    )
    assert float(position_loss) > 0.0
    assert position_metrics == metrics


def test_conditional_source_gain_uses_shared_target_joined_rows_only() -> None:
    margin_gain, top1_gain, margin_win, active = _combined_component_gain_statistics(
        combined_margin=torch.tensor([0.8, -0.5, 0.1]),
        combined_top1=torch.tensor([True, False, True]),
        combined_active=torch.tensor([True, True, True]),
        component_margin=torch.tensor([0.2, 0.3, 0.1]),
        component_top1=torch.tensor([False, False, True]),
        component_active=torch.tensor([True, True, False]),
    )
    assert active.tolist() == [True, True, False]
    assert margin_gain.tolist() == pytest.approx([0.6, -0.8, 0.0])
    assert top1_gain.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert margin_win.tolist() == [True, False, False]

    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=torch.ones((1, 2, 1)),
        edge_usable=torch.ones((1, 2, 1), dtype=torch.bool),
        support_view_logits=torch.full((1, 2, 1), 3.0),
        point_null_log_likelihood_ratios=torch.full((1,), 2.0),
    )
    fixed = _fixed_view_prediction(
        prediction=prediction,
        edge_usable_override=torch.tensor([[[True], [False]]]),
    )
    assert fixed.edge_usable.tolist() == [[[True], [False]]]
    assert fixed.support_view_logits is not None
    assert fixed.point_null_log_likelihood_ratios is not None
    assert torch.count_nonzero(fixed.support_view_logits) == 0
    assert torch.count_nonzero(fixed.point_null_log_likelihood_ratios) == 0


def test_component_visual_ablation_preserves_the_other_appearance_factor() -> None:
    rgb = _component_visual_ablation_scales("rgb_identity")
    context = _component_visual_ablation_scales("context_coherence")
    assert set(rgb) == set(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES)
    assert set(context) == set(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES)
    assert rgb["rgb"] == 0.0
    assert all(rgb[name] == 1.0 for name in rgb if name != "rgb")
    assert context["rgb"] == 1.0
    assert all(context[name] == 0.0 for name in context if name != "rgb")
    with pytest.raises(ValueError, match="component"):
        _component_visual_ablation_scales("invalid")


def test_conditional_visual_ablation_trains_joint_margin_without_solo_source_gate() -> None:
    runtime = _runtime()
    targets = torch.tensor([[True, False], [False, True]])
    normal = _prediction(torch.tensor([[1.5, -1.0], [-1.0, 1.5]]))
    ablated = _prediction(torch.zeros((2, 2)))
    loss, metrics = _conditional_visual_ablation_loss(
        runtime=runtime,
        normal_prediction=normal,
        source_ablated_prediction=ablated,
        targets=targets,
        margin=0.25,
    )
    assert float(loss) > 0.0
    assert metrics["active_rows"] == 2.0
    assert metrics["mean_gap"] > 0.0
    assert metrics["top1_delta"] > 0.0

    combined = {
        "normal_mean_margin": 0.2,
        "normal_win_fraction": 0.8,
        "support_permuted_mean_margin": 0.0,
        "position_only_mean_margin": 0.0,
        "normal_minus_support_permuted_correct_candidate_score": 0.2,
        "normal_minus_position_only_correct_candidate_score": 0.2,
    }
    metrics_for_gate = {
        **combined,
        **{
            f"{component}_{field}": value
            for component in ("rgb_identity", "context_coherence")
            for field, value in (
                ("conditional_visual_content_minus_ablated_mean_margin", 0.1),
                ("conditional_visual_content_margin_win_fraction", 0.6),
                ("conditional_visual_content_minus_ablated_top1_delta", 0.02),
            )
        },
    }
    gate = conditional_source_visual_ablation_gate(
        metrics_for_gate,
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
        minimum_support_correct_score_gap=0.05,
        minimum_position_correct_score_gap=0.05,
        minimum_source_margin_gain=0.05,
        minimum_source_margin_win_fraction=0.55,
        minimum_source_top1_delta=0.0,
    )
    assert gate["passed"] is True


def test_observation_identity_gate_rejects_coordinate_only_explanations() -> None:
    baseline = {
        "normal_mean_margin": 0.2,
        "normal_win_fraction": 0.8,
        "support_permuted_mean_margin": 0.0,
        "position_only_mean_margin": 0.01,
        "normal_minus_support_permuted_correct_candidate_score": 0.2,
        "normal_minus_position_only_correct_candidate_score": 0.19,
    }
    passed = observation_identity_gate(
        baseline,
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
        minimum_support_correct_score_gap=0.05,
        minimum_position_correct_score_gap=0.05,
    )
    coordinate_only = observation_identity_gate(
        {
            **baseline,
            "position_only_mean_margin": 0.19,
            "normal_minus_position_only_correct_candidate_score": 0.01,
        },
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
        minimum_support_correct_score_gap=0.05,
        minimum_position_correct_score_gap=0.05,
    )
    assert passed["passed"] is True
    assert coordinate_only["passed"] is False
    assert coordinate_only["checks"]["visual_over_position_gap"] is False
    assert coordinate_only["checks"]["position_correct_score_gap"] is False


def test_independent_source_masked_gate_rejects_dead_rgb_expert() -> None:
    combined = {
        "normal_mean_margin": 0.2,
        "normal_win_fraction": 0.8,
        "support_permuted_mean_margin": 0.0,
        "position_only_mean_margin": 0.0,
        "normal_minus_support_permuted_correct_candidate_score": 0.2,
        "normal_minus_position_only_correct_candidate_score": 0.2,
    }
    metrics = {
        **combined,
        **{f"rgb_identity_{name}": value for name, value in combined.items()},
        **{f"context_coherence_{name}": value for name, value in combined.items()},
    }
    passed = independent_source_masked_observation_gate(
        metrics,
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
        minimum_support_correct_score_gap=0.05,
        minimum_position_correct_score_gap=0.05,
    )
    dead_rgb = independent_source_masked_observation_gate(
        {
            **metrics,
            "rgb_identity_normal_mean_margin": 0.0,
            "rgb_identity_normal_win_fraction": 0.05,
            "rgb_identity_normal_minus_support_permuted_correct_candidate_score": 0.0,
            "rgb_identity_normal_minus_position_only_correct_candidate_score": 0.0,
        },
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
        minimum_support_correct_score_gap=0.05,
        minimum_position_correct_score_gap=0.05,
    )
    assert passed["passed"] is True
    assert dead_rgb["passed"] is False
    assert dead_rgb["checks"]["rgb_identity_normal_margin"] is False


def test_hard_pose_curriculum_initializer_requires_visual_lineage_not_target_lineage(tmp_path) -> None:
    torch.manual_seed(41)
    sources = {
        name: torch.nn.functional.normalize(torch.randn(3, side, side, dimension), dim=-1)
        for name, dimension, side in (
            ("radio_final", 8, 5),
            ("radio_intermediate", 8, 5),
            ("alike", 6, 7),
        )
    }
    windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    model = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        max_abs_log_ratio=4.0,
        context_windows=windows,
    )
    source_lineage = {
        "radio_final_context_cache_sha256": "final",
        "radio_intermediate_context_cache_sha256": "intermediate",
        "alike_spatial_context_cache_sha256": "alike",
        "source_image_manifest_sha256": "images",
        "rgb_coordinate_bridge": {"format": "bridge"},
    }
    args = SimpleNamespace(
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        max_abs_log_ratio=4.0,
    )
    checkpoint = tmp_path / "initializer.pt"
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "p1_initialization_allowed": True,
        "visual_evidence_gate_version": OBSERVATION_IDENTITY_GATE_VERSION,
        "fixed_candidate_count": 2,
        "config": {
            "rgb_context_radius_px": 8.0,
            "rgb_step_px": 1.0,
            "texture_feature_dim": 8,
            "hidden_dim": 8,
            "max_abs_log_ratio": 4.0,
            "context_windows": windows,
        },
        "lineage": {**source_lineage, "observation_pairs_sha256": "old-targets"},
        "training": {"inner_validation": {"selected_epoch": 2, "gate": {"passed": True}}},
    }
    torch.save({"format": CHECKPOINT_FORMAT, "metadata": metadata, "state_dict": model.state_dict()}, checkpoint)
    target = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        max_abs_log_ratio=4.0,
        context_windows=windows,
    )
    loaded = _load_identity_observation_initializer(
        model=target,
        path=checkpoint,
        candidate_count=2,
        source_lineage=source_lineage,
        windows=windows,
        args=args,
    )
    assert loaded["source_observation_pairs_sha256"] == "old-targets"
    changed_lineage = {**source_lineage, "radio_final_context_cache_sha256": "other"}
    with pytest.raises(ValueError, match="lineage"):
        _load_identity_observation_initializer(
            model=target,
            path=checkpoint,
            candidate_count=2,
            source_lineage=changed_lineage,
            windows=windows,
            args=args,
        )


def test_representation_initializer_copies_only_visual_backbone_from_v2_checkpoint(tmp_path) -> None:
    torch.manual_seed(43)
    sources = {
        name: torch.nn.functional.normalize(torch.randn(3, side, side, dimension), dim=-1)
        for name, dimension, side in (
            ("radio_final", 8, 5),
            ("radio_intermediate", 8, 5),
            ("alike", 6, 7),
        )
    }
    windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    source = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        max_abs_log_ratio=4.0,
        context_windows=windows,
    )
    with torch.no_grad():
        source.edge_head[-1].weight.fill_(3.0)
        source.edge_head[-1].bias.fill_(2.0)
    source_state = source.state_dict()
    representation_prefixes = (
        "context_encoders.",
        "global_projections.",
        "texture_encoder.",
        "context_projector.",
        "rgb_projector.",
    )
    # Match the old v2 surface: shared visual state plus a single fused edge
    # head, with no v3 RGB/context/view/null scalar head slots.
    v2_like_state = {
        name: value.clone()
        for name, value in source_state.items()
        if name.startswith(representation_prefixes) or name.startswith("edge_head.")
    }
    source_lineage = {
        "radio_final_context_cache_sha256": "final",
        "radio_intermediate_context_cache_sha256": "intermediate",
        "alike_spatial_context_cache_sha256": "alike",
        "source_image_manifest_sha256": "images",
        "rgb_coordinate_bridge": {"format": "bridge"},
    }
    args = SimpleNamespace(
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        max_abs_log_ratio=4.0,
    )
    checkpoint = tmp_path / "v2_representation_initializer.pt"
    metadata = {
        "format": "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v1",
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_V2_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "candidate_slot_permutation_equivariant": True,
        "fixed_candidate_count": 2,
        "config": {
            "rgb_context_radius_px": 8.0,
            "rgb_step_px": 1.0,
            "texture_feature_dim": 8,
            "hidden_dim": 8,
            "max_abs_log_ratio": 4.0,
            "context_windows": windows,
        },
        "lineage": {**source_lineage, "observation_pairs_sha256": "old-pairs"},
        "training": {"inner_validation": {"selected_epoch": 3, "gate": {"passed": True}}},
    }
    torch.save(
        {
            "format": metadata["format"],
            "metadata": metadata,
            "state_dict": v2_like_state,
        },
        checkpoint,
    )
    target = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        max_abs_log_ratio=4.0,
        context_windows=windows,
    )
    scalar_before = {
        name: value.clone()
        for name, value in target.state_dict().items()
        if name.startswith(
            ("edge_head.", "rgb_identity_head.", "context_coherence_head.", "support_view_head.", "null_head.")
        )
    }
    loaded = _load_visual_representation_initializer(
        model=target,
        path=checkpoint,
        candidate_count=2,
        source_lineage=source_lineage,
        windows=windows,
        args=args,
    )
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name.startswith(representation_prefixes):
            assert torch.equal(target_state[name], value)
    for name, value in scalar_before.items():
        assert torch.equal(target_state[name], value)
    assert loaded["kind"] == "representation_only_visual_initializer"
    assert loaded["source_model_format"] == CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_V2_FORMAT
    assert loaded["scalar_heads_transferred"] is False
    assert loaded["source_observation_pairs_sha256"] == "old-pairs"
    with pytest.raises(ValueError, match="lineage"):
        _load_visual_representation_initializer(
            model=target,
            path=checkpoint,
            candidate_count=2,
            source_lineage={**source_lineage, "source_image_manifest_sha256": "other"},
            windows=windows,
            args=args,
        )


def test_direct_hard_pose_statistics_and_gate_require_visual_control() -> None:
    values = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
    usable = torch.ones_like(values, dtype=torch.bool)
    positive = torch.tensor([[True, False], [False, True]])
    hard = torch.tensor([[False, True], [True, False]])
    gaps, wins, active = _hard_pose_identity_statistics(
        candidate_values=values,
        candidate_usable=usable,
        positive_targets=positive,
        hard_negative_mask=hard,
    )
    assert active.tolist() == [True, True]
    assert wins.tolist() == [True, True]
    assert gaps.tolist() == [3.0, 2.0]
    args = SimpleNamespace(
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
        minimum_support_correct_score_gap=0.05,
        minimum_position_correct_score_gap=0.05,
        minimum_hard_pose_eligible_query_fraction=0.9,
        minimum_hard_pose_gap=0.05,
        minimum_hard_pose_win_fraction=0.55,
        minimum_hard_pose_visual_gap=0.05,
        minimum_conditional_visual_ablation_margin=0.05,
        minimum_conditional_visual_ablation_win_fraction=0.55,
        minimum_conditional_visual_ablation_top1_delta=0.0,
        minimum_conditional_visual_ablation_hard_pose_gap=0.05,
        minimum_conditional_visual_ablation_hard_pose_win_fraction=0.55,
    )
    metrics = {
        "normal_mean_margin": 0.2,
        "normal_win_fraction": 0.8,
        "support_permuted_mean_margin": 0.0,
        "position_only_mean_margin": 0.0,
        "normal_minus_support_permuted_correct_candidate_score": 0.2,
        "normal_minus_position_only_correct_candidate_score": 0.2,
        "hard_pose_eligible_query_fraction": 1.0,
        "hard_pose_mean_gap": 0.2,
        "hard_pose_win_fraction": 0.8,
        "hard_pose_support_permuted_mean_gap": 0.0,
        "hard_pose_position_only_mean_gap": 0.0,
    }
    metrics.update(
        {
            f"{component}_{name}": value
            for component in ("rgb_identity", "context_coherence")
            for name, value in (
                ("normal_mean_margin", 0.2),
                ("normal_win_fraction", 0.8),
                ("support_permuted_mean_margin", 0.0),
                ("position_only_mean_margin", 0.0),
                ("normal_minus_support_permuted_correct_candidate_score", 0.2),
                ("normal_minus_position_only_correct_candidate_score", 0.2),
            )
        }
    )
    metrics.update(
        {
            f"{component}_{name}": value
            for component in ("rgb_identity", "context_coherence")
            for name, value in (
                ("conditional_visual_content_minus_ablated_mean_margin", 0.2),
                ("conditional_visual_content_margin_win_fraction", 0.8),
                ("conditional_visual_content_minus_ablated_top1_delta", 0.1),
                ("conditional_visual_content_minus_ablated_hard_pose_mean_gap", 0.2),
                ("conditional_visual_content_hard_pose_gap_win_fraction", 0.8),
            )
        }
    )
    passed = _identity_pretrain_gate(metrics=metrics, args=args, hard_pose_enabled=True)
    failed = _identity_pretrain_gate(
        metrics={**metrics, "hard_pose_position_only_mean_gap": 0.19},
        args=args,
        hard_pose_enabled=True,
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["hard_pose_position_visual_gap"] is False


def test_direct_hard_pose_mask_moves_with_target_free_candidate_permutation() -> None:
    pairs = SimpleNamespace(
        negative_count=2,
        anchor_ids=torch.tensor([10, 11]).numpy(),
    )
    mask = _permuted_hard_pose_negative_mask(
        pairs=pairs,
        rows=torch.tensor([0, 1]).numpy(),
        permutations=torch.tensor([[2, 0, 1], [1, 2, 0]]),
        lookup={
            10: torch.tensor([False, True, False]).numpy(),
            11: torch.tensor([False, False, True]).numpy(),
        },
        device=torch.device("cpu"),
    )
    assert mask is not None
    # Runtime slot i now contains original slot permutations[i], so target
    # masks must follow the same target-free reordering.
    assert mask.tolist() == [[False, False, True], [False, True, False]]
