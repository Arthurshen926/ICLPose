from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    _candidate_scores,
    _component_edge_only_prediction,
    _candidate_plus_null_logits,
    _checkpoint_gate,
    _configure_identity_llr_feature_training,
    _identity_llr_scalar_gradient_diagnostics,
    _identity_llr_scalar_head_parameters,
    _identity_llr_expert_update_l2,
    _hard_pose_group_metrics,
    _hard_pose_group_permutation_metrics,
    _hard_repeat_metrics,
    _identity_llr_optimizer_parameter_groups,
    _is_better,
    _jitter_runtime_query_anchors,
    _load_identity_observation_pretrain,
    _prepare_output_directory,
    _query_anchor_jitter_seed,
    _registered_identity_metrics,
    _registered_identity_or_null_posterior_cross_entropy_metrics,
    _registered_identity_posterior_cross_entropy_metrics,
    _registered_permutation_metrics,
    _reset_identity_l0_scalar_head_finals,
    _reset_identity_llr_final_edge_head,
    _resolve_feature_training_mode,
    _select_identity_l0_group_points,
    _set_identity_llr_train_mode,
    _validate_args,
    _validate_identity_target_contract,
    _write_json_atomically,
    parse_args,
)
from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_identity_llr import (
    CHECKPOINT_FORMAT as OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    OBSERVATION_IDENTITY_GATE_VERSION,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatBatch,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
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
        support_image_indices=torch.tensor([[[2, 3], [2, 3]], [[2, 3], [2, 3]]]),
        support_xy=torch.full((2, 2, 2, 2), 48.0),
        support_view_valid=torch.ones((2, 2, 2), dtype=torch.bool),
        candidate_view_weights=torch.full((2, 2, 2), 0.5),
        candidate_probabilities=torch.full((2, 2), 0.4),
        null_probabilities=torch.full((2,), 0.2),
    )


def _prediction(values: torch.Tensor) -> CandidatePoseRGBSpatialIdentityLLREdgePrediction:
    return CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=values,
        edge_usable=torch.ones_like(values, dtype=torch.bool),
    )


class _GateArgs:
    training_stage = "hard_pose_contrastive"
    minimum_win_fraction = 0.55
    minimum_posterior_candidate_top1_fraction = 0.10
    minimum_rgb_identity_posterior_top1_fraction = 0.10
    minimum_rgb_identity_permutation_gap = 0.05
    minimum_context_coherence_hard_repeat_gap = 0.05
    minimum_context_coherence_permutation_gap = 0.05
    minimum_normal_gap = 0.05
    minimum_visual_gap_delta = 0.05
    minimum_position_visual_gap = 0.05
    minimum_inner_eligible_query_fraction = 0.9
    minimum_hard_repeat_win_fraction = 0.55
    minimum_hard_repeat_gap = 0.05
    minimum_inner_hard_repeat_eligible_query_fraction = 0.9
    minimum_inner_hard_pose_eligible_query_fraction = 0.9
    minimum_hard_pose_win_fraction = 0.55
    minimum_hard_pose_gap = 0.05
    minimum_hard_pose_visual_gap_delta = 0.05


def _gate_metrics() -> dict[str, float]:
    return {
        "normal_correct_win_fraction": 1.0,
        "posterior_candidate_top1_fraction": 0.20,
        "rgb_identity_posterior_top1_fraction": 0.20,
        "rgb_identity_permutation_mean_gap": 0.20,
        "context_coherence_hard_repeat_mean_gap": 0.20,
        "context_coherence_permutation_mean_gap": 0.20,
        "normal_mean_correct_minus_hardest_wrong": 0.2,
        "permuted_mean_correct_minus_hardest_wrong": 0.0,
        "position_only_mean_correct_minus_hardest_wrong": 0.0,
        "eligible_query_fraction": 1.0,
        "hard_repeat_eligible_query_fraction": 1.0,
        "hard_repeat_correct_win_fraction": 1.0,
        "hard_repeat_mean_correct_minus_coherent_wrong": 0.2,
        "hard_repeat_position_only_mean_correct_minus_coherent_wrong": 0.0,
        "hard_pose_group_eligible_query_fraction": 1.0,
        "hard_pose_group_correct_win_fraction": 1.0,
        "hard_pose_group_mean_correct_minus_coherent_wrong": 0.2,
        "hard_pose_group_permuted_mean_correct_minus_coherent_wrong": 0.0,
        "hard_pose_group_position_only_mean_correct_minus_coherent_wrong": 0.0,
    }


def test_registered_identity_margin_uses_distinct_candidate_as_negative() -> None:
    prediction = _prediction(
        torch.tensor([[[2.0, 2.0], [-1.0, -1.0]], [[-1.0, -1.0], [3.0, 3.0]]])
    )
    loss, metrics = _registered_identity_metrics(
        runtime=_runtime(),
        prediction=prediction,
        observed_candidate_mask=torch.tensor([[True, False], [False, True]]),
        margin=0.25,
    )
    assert float(loss) > 0.0
    assert metrics["active"] == 2.0
    assert metrics["mean_gap"] == 3.5
    assert metrics["win_fraction"] == 1.0


def test_fixed_candidate_prior_is_added_only_after_visual_llr_emission() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[32.0, 32.0], [48.0, 48.0]]),
        support_image_indices=torch.tensor([[[2], [3]], [[2], [3]]]),
        support_xy=torch.full((2, 2, 1, 2), 48.0),
        support_view_valid=torch.ones((2, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 2, 1)),
        candidate_probabilities=torch.tensor([[0.8, 0.1], [0.7, 0.2]]),
        null_probabilities=torch.tensor([0.1, 0.1]),
    )
    prediction = _prediction(torch.zeros((2, 2, 1)))
    prior_free, usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=0.0,
    )
    residual, residual_usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=1.0,
    )
    assert torch.equal(usable, residual_usable)
    assert torch.allclose(prior_free, torch.zeros_like(prior_free))
    assert torch.allclose(
        residual,
        torch.log(runtime.candidate_probabilities),
    )


def test_component_edge_adapter_cannot_reuse_fused_view_or_null_outputs() -> None:
    rgb = torch.full((2, 2, 2), 1.5)
    context = torch.full((2, 2, 2), -0.75)
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=torch.full((2, 2, 2), 3.0),
        edge_usable=torch.ones((2, 2, 2), dtype=torch.bool),
        support_view_logits=torch.full((2, 2, 2), 4.0),
        point_null_log_likelihood_ratios=torch.full((2,), 2.0),
        rgb_identity_edge_log_likelihood_ratios=rgb,
        context_coherence_edge_log_likelihood_ratios=context,
    )
    rgb_only = _component_edge_only_prediction(
        prediction=prediction,
        component="rgb_identity",
    )
    context_only = _component_edge_only_prediction(
        prediction=prediction,
        component="context_coherence",
    )
    assert torch.equal(rgb_only.edge_log_likelihood_ratios, rgb)
    assert torch.equal(context_only.edge_log_likelihood_ratios, context)
    assert torch.equal(rgb_only.support_view_logits, torch.zeros_like(rgb))
    assert torch.equal(context_only.support_view_logits, torch.zeros_like(context))
    assert torch.equal(
        rgb_only.point_null_log_likelihood_ratios,
        torch.zeros((2,), dtype=rgb.dtype),
    )
    assert torch.equal(
        context_only.point_null_log_likelihood_ratios,
        torch.zeros((2,), dtype=context.dtype),
    )
    with pytest.raises(ValueError, match="component name"):
        _component_edge_only_prediction(prediction=prediction, component="fused")


def test_component_edge_adapter_preserves_source_specific_availability() -> None:
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=torch.tensor([[[1.0], [2.0]]]),
        edge_usable=torch.tensor([[[True], [True]]]),
        rgb_identity_edge_log_likelihood_ratios=torch.tensor([[[1.0], [0.0]]]),
        context_coherence_edge_log_likelihood_ratios=torch.tensor([[[0.0], [2.0]]]),
        rgb_edge_usable=torch.tensor([[[True], [False]]]),
        context_edge_usable=torch.tensor([[[False], [True]]]),
    )
    rgb = _component_edge_only_prediction(prediction=prediction, component="rgb_identity")
    context = _component_edge_only_prediction(
        prediction=prediction, component="context_coherence"
    )
    assert rgb.edge_usable.tolist() == [[[True], [False]]]
    assert context.edge_usable.tolist() == [[[False], [True]]]
    assert rgb.rgb_edge_usable.tolist() == [[[True], [False]]]
    assert context.context_edge_usable.tolist() == [[[False], [True]]]


def test_posterior_cross_entropy_uses_immutable_null_and_visual_residual() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[32.0, 32.0]]),
        support_image_indices=torch.tensor([[[2], [3]]]),
        support_xy=torch.full((1, 2, 1, 2), 48.0),
        support_view_valid=torch.ones((1, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 2, 1)),
        candidate_probabilities=torch.tensor([[0.8, 0.1]]),
        null_probabilities=torch.tensor([0.1]),
    )
    # Candidate 1 starts with lower prior but receives enough visual residual
    # to become the correct posterior mode.
    prediction = _prediction(torch.tensor([[[0.0], [4.0]]]))
    loss, metrics = _registered_identity_posterior_cross_entropy_metrics(
        runtime=runtime,
        prediction=prediction,
        observed_candidate_mask=torch.tensor([[False, True]]),
        candidate_prior_logit_weight=1.0,
    )
    assert float(loss) < 0.2
    assert metrics["active"] == 1.0
    assert metrics["top1_fraction"] == 1.0
    assert metrics["mean_target_probability"] > 0.8


def test_candidate_or_null_posterior_supervises_explicit_null_rows() -> None:
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=torch.tensor(
            [[[2.0, 2.0], [-2.0, -2.0]], [[-2.0, -2.0], [-2.0, -2.0]]]
        ),
        edge_usable=torch.ones((2, 2, 2), dtype=torch.bool),
        point_null_log_likelihood_ratios=torch.tensor([-2.0, 4.0]),
    )
    logits, _ = _candidate_plus_null_logits(
        runtime=_runtime(), prediction=prediction, candidate_prior_logit_weight=1.0
    )
    assert int(logits[0].argmax()) == 0
    assert int(logits[1].argmax()) == 2
    loss, metrics = _registered_identity_or_null_posterior_cross_entropy_metrics(
        runtime=_runtime(),
        prediction=prediction,
        observed_candidate_mask=torch.tensor([[True, False], [False, False]]),
        target_dustbin=torch.tensor([[False, True], [True, True]]),
        target_supervised=torch.ones((2, 2), dtype=torch.bool),
        candidate_prior_logit_weight=1.0,
    )
    assert float(loss) < 0.2
    assert metrics["active"] == 2.0
    assert metrics["candidate_active"] == 1.0
    assert metrics["null_active"] == 1.0
    assert metrics["candidate_top1_fraction"] == 1.0
    assert metrics["null_top1_fraction"] == 1.0
    assert metrics["optimization_loss"] == pytest.approx(metrics["loss"])
    with pytest.raises(ValueError, match="complete candidate rows"):
        _registered_identity_or_null_posterior_cross_entropy_metrics(
            runtime=_runtime(),
            prediction=prediction,
            observed_candidate_mask=torch.tensor([[True, False], [False, False]]),
            target_dustbin=torch.tensor([[False, True], [True, True]]),
            target_supervised=torch.tensor([[True, False], [True, True]]),
            candidate_prior_logit_weight=1.0,
        )


def test_candidate_or_null_posterior_weights_change_only_optimization_balance() -> None:
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=torch.tensor(
            [[[-4.0, -4.0], [4.0, 4.0]], [[-4.0, -4.0], [-4.0, -4.0]]]
        ),
        edge_usable=torch.ones((2, 2, 2), dtype=torch.bool),
        point_null_log_likelihood_ratios=torch.tensor([-4.0, 4.0]),
    )
    kwargs = {
        "runtime": _runtime(),
        "prediction": prediction,
        "observed_candidate_mask": torch.tensor([[True, False], [False, False]]),
        "target_dustbin": torch.tensor([[False, True], [True, True]]),
        "target_supervised": torch.ones((2, 2), dtype=torch.bool),
        "candidate_prior_logit_weight": 1.0,
    }
    default_loss, default = _registered_identity_or_null_posterior_cross_entropy_metrics(
        **kwargs
    )
    candidate_weighted_loss, candidate_weighted = (
        _registered_identity_or_null_posterior_cross_entropy_metrics(
            **kwargs, candidate_loss_weight=4.0, null_loss_weight=1.0
        )
    )
    assert candidate_weighted["candidate_loss"] > candidate_weighted["null_loss"]
    assert candidate_weighted["loss"] == pytest.approx(default["loss"])
    assert candidate_weighted["optimization_loss"] > default["optimization_loss"]
    assert float(candidate_weighted_loss) > float(default_loss)


def test_identity_l0_sampler_reserves_explicit_null_rows() -> None:
    group = SimpleNamespace(
        point_count=8,
        query_id="train/query",
        source_point_ids=torch.arange(100, 108).numpy(),
        spatial_target_observed=torch.tensor(
            [[True, False], [True, False], [False, False], [False, False], [False, False], [False, False], [False, False], [False, False]]
        ).numpy(),
        spatial_target_supervised=torch.tensor(
            [[True, True], [True, True], [True, True], [True, True], [False, False], [False, False], [False, False], [False, False]]
        ).numpy(),
        spatial_target_dustbin=torch.tensor(
            [[False, True], [False, True], [True, True], [True, True], [False, False], [False, False], [False, False], [False, False]]
        ).numpy(),
    )
    selected = _select_identity_l0_group_points(
        group=group,
        max_points=4,
        seed=17,
        required_source_point_ids=[104],
    )
    assert len(selected) == 4
    assert {0, 1, 4}.issubset(set(selected.tolist()))
    assert bool(set(selected.tolist()) & {2, 3})


def test_identity_l0_freezes_feature_extractors_but_keeps_scalar_heads_trainable() -> None:
    model = CandidatePoseRGBSpatialIdentityLLR(
        sources={
            "radio_final": torch.nn.functional.normalize(torch.randn(3, 5, 5, 6), dim=-1),
            "radio_intermediate": torch.nn.functional.normalize(torch.randn(3, 5, 5, 6), dim=-1),
            "alike": torch.nn.functional.normalize(torch.randn(3, 5, 5, 6), dim=-1),
        },
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        context_windows={"radio_final": 3, "radio_intermediate": 3, "alike": 3},
    )
    info = _configure_identity_llr_feature_training(
        model=model,
        mode="frozen_scalar_heads",
    )
    assert info["mode"] == "frozen_scalar_heads"
    assert info["activation_checkpointing"] is False
    for name, parameter in model.named_parameters():
        expected = name.startswith(
            (
                "rgb_identity_head.",
                "context_coherence_head.",
                "support_view_head.",
                "null_head.",
            )
        )
        assert parameter.requires_grad is expected
    _set_identity_llr_train_mode(model=model, feature_training_mode="frozen_scalar_heads")
    assert model.edge_head.training
    assert not model.context_encoders.training
    assert not model.texture_encoder.training
    _configure_identity_llr_feature_training(model=model, mode="full")
    assert all(
        parameter.requires_grad is (not name.startswith("edge_head."))
        for name, parameter in model.named_parameters()
    )
    assert model.activation_checkpointing is True
    assert _resolve_feature_training_mode(
        SimpleNamespace(feature_training_mode="auto", training_stage="identity_l0")
    ) == "frozen_scalar_heads"
    assert _resolve_feature_training_mode(
        SimpleNamespace(feature_training_mode="auto", training_stage="hard_pose_contrastive")
    ) == "full"
    scalar = _identity_llr_scalar_head_parameters(model)
    for parameter in scalar.values():
        parameter.grad = torch.ones_like(parameter)
    gradient_l2, gradients_finite = _identity_llr_scalar_gradient_diagnostics(model)
    assert gradients_finite
    assert gradient_l2 > 0.0
    next(iter(scalar.values())).grad = torch.full_like(next(iter(scalar.values())), float("nan"))
    _, gradients_finite = _identity_llr_scalar_gradient_diagnostics(model)
    assert not gradients_finite
    before = {name: parameter.detach().clone() for name, parameter in scalar.items()}
    rgb_final = model.rgb_identity_head[-1]
    assert isinstance(rgb_final, torch.nn.Linear)
    with torch.no_grad():
        rgb_final.weight.add_(0.01)
    assert _identity_llr_expert_update_l2(
        model=model,
        before=before,
        expert="rgb_identity",
    ) > 0.0
    assert _identity_llr_expert_update_l2(
        model=model,
        before=before,
        expert="context_coherence",
    ) == 0.0


def test_train_query_anchor_jitter_changes_only_query_coordinates() -> None:
    runtime = _runtime()
    seed = _query_anchor_jitter_seed(seed=31, epoch=2, query_id="train/query.png")
    jittered = _jitter_runtime_query_anchors(
        runtime=runtime,
        radius_px=4.0,
        coordinate_image_size=(128, 96),
        interior_margin_px=8.0,
        seed=seed,
    )
    repeated = _jitter_runtime_query_anchors(
        runtime=runtime,
        radius_px=4.0,
        coordinate_image_size=(128, 96),
        interior_margin_px=8.0,
        seed=seed,
    )
    assert torch.allclose(jittered.query_xy, repeated.query_xy)
    assert not torch.allclose(jittered.query_xy, runtime.query_xy)
    assert torch.all(jittered.query_xy[:, 0] >= 8.0)
    assert torch.all(jittered.query_xy[:, 0] <= 119.0)
    assert torch.all(jittered.query_xy[:, 1] >= 8.0)
    assert torch.all(jittered.query_xy[:, 1] <= 87.0)
    assert torch.all(torch.linalg.vector_norm(jittered.query_xy - runtime.query_xy, dim=1) <= 4.0001)
    assert torch.equal(jittered.query_image_indices, runtime.query_image_indices)
    assert torch.equal(jittered.support_image_indices, runtime.support_image_indices)
    assert torch.equal(jittered.support_xy, runtime.support_xy)
    assert torch.equal(jittered.support_view_valid, runtime.support_view_valid)
    assert torch.equal(jittered.candidate_view_weights, runtime.candidate_view_weights)
    assert torch.equal(jittered.candidate_probabilities, runtime.candidate_probabilities)
    assert torch.equal(jittered.null_probabilities, runtime.null_probabilities)
    assert _jitter_runtime_query_anchors(
        runtime=runtime,
        radius_px=0.0,
        coordinate_image_size=(128, 96),
        interior_margin_px=8.0,
        seed=seed,
    ) is runtime


def test_hard_repeat_and_permutation_losses_are_candidate_specific() -> None:
    runtime = _runtime()
    normal = _prediction(
        torch.tensor([[[2.0, 2.0], [-2.0, -2.0]], [[1.0, 1.0], [-1.0, -1.0]]])
    )
    permuted = _prediction(
        torch.tensor([[[0.5, 0.5], [-2.0, -2.0]], [[0.5, 0.5], [-1.0, -1.0]]])
    )
    hard_loss, hard = _hard_repeat_metrics(
        runtime=runtime,
        prediction=normal,
        hard_batch=HardRepeatBatch(
            point_indices=torch.tensor([0, 1]),
            positive_candidate_indices=torch.tensor([0, 0]),
            negative_candidate_indices=torch.tensor([1, 1]),
            positive_offsets_xy=torch.zeros((2, 2)),
            negative_offsets_xy=torch.zeros((2, 2)),
        ),
        margin=0.25,
    )
    permutation_loss, permutation = _registered_permutation_metrics(
        runtime=runtime,
        prediction=normal,
        permuted_runtime=runtime,
        permuted_prediction=permuted,
        observed_candidate_mask=torch.tensor([[True, False], [True, False]]),
        margin=0.05,
    )
    assert float(hard_loss) > 0.0
    assert hard["mean_gap"] == 3.0
    assert permutation["mean_gap"] == 1.0
    assert float(permutation_loss) > 0.0


def test_hard_repeat_loss_uses_the_strongest_distinct_wrong_candidate_per_point() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[32.0, 32.0]]),
        support_image_indices=torch.tensor([[[2], [2], [2]]]),
        support_xy=torch.full((1, 3, 1, 2), 48.0),
        support_view_valid=torch.ones((1, 3, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 3, 1)),
        candidate_probabilities=torch.full((1, 3), 0.3),
        null_probabilities=torch.tensor([0.1]),
    )
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=torch.tensor([[[1.0], [0.0], [3.0]]]),
        edge_usable=torch.ones((1, 3, 1), dtype=torch.bool),
    )
    loss, metrics = _hard_repeat_metrics(
        runtime=runtime,
        prediction=prediction,
        hard_batch=HardRepeatBatch(
            point_indices=torch.tensor([0, 0]),
            positive_candidate_indices=torch.tensor([0, 0]),
            negative_candidate_indices=torch.tensor([1, 2]),
            positive_offsets_xy=torch.zeros((2, 2)),
            negative_offsets_xy=torch.zeros((2, 2)),
        ),
        margin=0.25,
    )
    assert float(loss) > 0.0
    assert metrics["active"] == 1.0
    assert metrics["mean_gap"] == -2.0


def test_hard_pose_group_uses_pair_ids_and_counts_each_source_once() -> None:
    runtime = _runtime()
    prediction = _prediction(
        torch.tensor([[[3.0, 3.0], [1.0, 1.0]], [[2.0, 2.0], [1.0, 1.0]]])
    )
    batch = HardRepeatBatch(
        # The first source has two rows for the same pair.  They must collapse
        # before the pair score is averaged with the second source.
        point_indices=torch.tensor([0, 0, 1]),
        positive_candidate_indices=torch.tensor([0, 0, 0]),
        negative_candidate_indices=torch.tensor([1, 1, 1]),
        positive_offsets_xy=torch.zeros((3, 2)),
        negative_offsets_xy=torch.zeros((3, 2)),
        pair_ids=torch.tensor([17, 17, 17]),
    )
    loss, metrics = _hard_pose_group_metrics(
        runtime=runtime,
        prediction=prediction,
        hard_batch=batch,
        margin=0.25,
        minimum_points=2,
    )
    assert float(loss) > 0.0
    assert metrics["active"] == 1.0
    assert metrics["active_points"] == 2.0
    assert metrics["mean_gap"] == 1.5
    assert metrics["win_fraction"] == 1.0


def test_hard_pose_group_can_rank_in_fixed_prior_residual_score_space() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[32.0, 32.0], [48.0, 48.0]]),
        support_image_indices=torch.tensor([[[2], [3]], [[2], [3]]]),
        support_xy=torch.full((2, 2, 1, 2), 48.0),
        support_view_valid=torch.ones((2, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 2, 1)),
        candidate_probabilities=torch.tensor([[0.8, 0.1], [0.7, 0.2]]),
        null_probabilities=torch.tensor([0.1, 0.1]),
    )
    prediction = _prediction(torch.zeros((2, 2, 1)))
    _, metrics = _hard_pose_group_metrics(
        runtime=runtime,
        prediction=prediction,
        hard_batch=HardRepeatBatch(
            point_indices=torch.tensor([0, 1]),
            positive_candidate_indices=torch.tensor([0, 0]),
            negative_candidate_indices=torch.tensor([1, 1]),
            positive_offsets_xy=torch.zeros((2, 2)),
            negative_offsets_xy=torch.zeros((2, 2)),
            pair_ids=torch.tensor([5, 5]),
        ),
        margin=0.25,
        minimum_points=2,
        candidate_prior_logit_weight=1.0,
    )
    expected = (torch.log(torch.tensor(0.8 / 0.1)) + torch.log(torch.tensor(0.7 / 0.2))) / 2.0
    assert metrics["active"] == 1.0
    assert metrics["mean_gap"] == pytest.approx(float(expected))
    assert metrics["win_fraction"] == 1.0


def test_hard_pose_group_requires_pair_ids_and_shared_control_availability() -> None:
    runtime = _runtime()
    normal = _prediction(
        torch.tensor([[[3.0, 3.0], [1.0, 1.0]], [[2.0, 2.0], [1.0, 1.0]]])
    )
    permuted = _prediction(
        torch.tensor([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]])
    )
    without_pair_ids = HardRepeatBatch(
        point_indices=torch.tensor([0, 1]),
        positive_candidate_indices=torch.tensor([0, 0]),
        negative_candidate_indices=torch.tensor([1, 1]),
        positive_offsets_xy=torch.zeros((2, 2)),
        negative_offsets_xy=torch.zeros((2, 2)),
    )
    _, missing = _hard_pose_group_metrics(
        runtime=runtime,
        prediction=normal,
        hard_batch=without_pair_ids,
        margin=0.25,
        minimum_points=2,
    )
    assert missing["active"] == 0.0
    with_pair_ids = HardRepeatBatch(
        point_indices=torch.tensor([0, 1]),
        positive_candidate_indices=torch.tensor([0, 0]),
        negative_candidate_indices=torch.tensor([1, 1]),
        positive_offsets_xy=torch.zeros((2, 2)),
        negative_offsets_xy=torch.zeros((2, 2)),
        pair_ids=torch.tensor([23, 23]),
    )
    loss, metrics = _hard_pose_group_permutation_metrics(
        runtime=runtime,
        prediction=normal,
        permuted_runtime=runtime,
        permuted_prediction=permuted,
        hard_batch=with_pair_ids,
        margin=0.05,
        minimum_points=2,
        common_edge_usable=normal.edge_usable & permuted.edge_usable,
    )
    assert float(loss) > 0.0
    assert metrics["active"] == 1.0
    assert metrics["active_points"] == 2.0
    assert metrics["mean_gap"] == 2.5


def test_checkpoint_gate_requires_registered_coverage_in_addition_to_visual_metrics() -> None:
    metrics = _gate_metrics()
    metrics["eligible_query_fraction"] = 0.5
    gate = _checkpoint_gate(metrics=metrics, args=_GateArgs())
    assert gate["passed"] is False
    assert gate["eligible_query_fraction"] == 0.5


def test_checkpoint_gate_requires_current_coherent_repeat_evidence() -> None:
    metrics = _gate_metrics()
    metrics["hard_pose_group_eligible_query_fraction"] = 0.5
    metrics["hard_pose_group_correct_win_fraction"] = 0.4
    metrics["hard_pose_group_mean_correct_minus_coherent_wrong"] = -0.1
    gate = _checkpoint_gate(metrics=metrics, args=_GateArgs())
    assert gate["passed"] is False
    assert gate["hard_pose_group_eligible_query_fraction"] == 0.5


def test_identity_l0_gate_uses_direct_repeat_not_pose_groups() -> None:
    args = _GateArgs()
    args.training_stage = "identity_l0"
    metrics = _gate_metrics()
    metrics["hard_pose_group_eligible_query_fraction"] = 0.0
    metrics["hard_pose_group_correct_win_fraction"] = 0.0
    metrics["hard_pose_group_mean_correct_minus_coherent_wrong"] = -1.0
    gate = _checkpoint_gate(metrics=metrics, args=args)
    assert gate["passed"] is True
    assert gate["training_stage"] == "identity_l0"


def test_identity_l0_gate_rejects_null_dominant_candidate_posterior() -> None:
    args = _GateArgs()
    args.training_stage = "identity_l0"
    metrics = _gate_metrics()
    metrics["posterior_candidate_top1_fraction"] = 0.05
    gate = _checkpoint_gate(metrics=metrics, args=args)
    assert gate["passed"] is False
    assert gate["posterior_candidate_top1_fraction"] == 0.05


def test_identity_l0_gate_requires_each_independent_expert_control() -> None:
    args = _GateArgs()
    args.training_stage = "identity_l0"
    metrics = _gate_metrics()
    metrics["rgb_identity_permutation_mean_gap"] = 0.0
    gate = _checkpoint_gate(metrics=metrics, args=args)
    assert gate["passed"] is False
    assert gate["rgb_identity_permutation_mean_gap"] == 0.0
    metrics = _gate_metrics()
    metrics["context_coherence_hard_repeat_mean_gap"] = 0.0
    gate = _checkpoint_gate(metrics=metrics, args=args)
    assert gate["passed"] is False
    assert gate["context_coherence_hard_repeat_mean_gap"] == 0.0


def test_identity_l0_checkpoint_selection_preserves_candidate_floor() -> None:
    candidate = _gate_metrics()
    candidate["posterior_candidate_top1_fraction"] = 0.05
    candidate["hard_repeat_mean_correct_minus_coherent_wrong"] = 1.0
    incumbent = _gate_metrics()
    incumbent["posterior_candidate_top1_fraction"] = 0.12
    assert not _is_better(
        candidate=candidate,
        incumbent=incumbent,
        candidate_passed=False,
        incumbent_passed=False,
        training_stage="identity_l0",
        minimum_posterior_candidate_top1_fraction=0.10,
    )


def test_checkpoint_selection_keeps_a_passing_zero_shot_initializer() -> None:
    """A sparse P1 update cannot displace a gate-passing initializer by loss alone."""

    initializer = _gate_metrics()
    initializer["normal_correct_win_fraction"] = 0.7
    initializer["hard_pose_group_correct_win_fraction"] = 0.7
    regressed_update = _gate_metrics()
    regressed_update["normal_mean_correct_minus_hardest_wrong"] = 2.0
    regressed_update["hard_pose_group_mean_correct_minus_coherent_wrong"] = 2.0
    assert not _is_better(
        candidate=regressed_update,
        incumbent=initializer,
        candidate_passed=False,
        incumbent_passed=True,
    )


def test_initialization_only_accepts_zero_epochs_but_training_does_not() -> None:
    required = [
        "--rgb-spatial-layout",
        "layout.npz",
        "--training-targets",
        "targets.npz",
        "--hard-repeat-targets",
        "hard_repeat.npz",
        "--radio-final-context-cache",
        "final.npz",
        "--radio-intermediate-context-cache",
        "intermediate.npz",
        "--alike-spatial-context-cache",
        "alike.npz",
        "--image-root",
        "images",
        "--output-dir",
        "run",
        "--texture-observation-pretrain-checkpoint",
        "texture.pt",
        "--epochs",
        "0",
    ]
    with pytest.raises(ValueError, match="training arguments"):
        _validate_args(parse_args(required))
    windows = _validate_args(parse_args([*required, "--initialization-only"]))
    assert windows == {"radio_final": 15, "radio_intermediate": 15, "alike": 21}


def test_identity_llr_accepts_zero_hard_repeat_limit_for_full_pool_gate() -> None:
    args = parse_args(
        [
            "--rgb-spatial-layout",
            "layout.npz",
            "--training-targets",
            "targets.npz",
            "--hard-repeat-targets",
            "hard_repeat.npz",
            "--radio-final-context-cache",
            "final.npz",
            "--radio-intermediate-context-cache",
            "intermediate.npz",
            "--alike-spatial-context-cache",
            "alike.npz",
            "--image-root",
            "images",
            "--output-dir",
            "run",
            "--texture-observation-pretrain-checkpoint",
            "texture.pt",
            "--epochs",
            "1",
            "--max-hard-repeat-edges-per-query",
            "0",
            "--reset-current-p1-scalar-head-finals",
        ]
    )
    assert args.reset_current_p1_scalar_head_finals is True
    assert _validate_args(args) == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 21,
    }


def test_checkpoint_gate_rejects_position_only_shortcut() -> None:
    metrics = _gate_metrics()
    metrics["position_only_mean_correct_minus_hardest_wrong"] = 0.2
    metrics["hard_pose_group_position_only_mean_correct_minus_coherent_wrong"] = 0.2
    gate = _checkpoint_gate(metrics=metrics, args=_GateArgs())
    assert gate["passed"] is False
    assert gate["normal_minus_position_only_gap"] == 0.0


def test_identity_target_contract_rejects_projection_only_main_targets() -> None:
    hard_metadata = {
        "positive_requires_registered_exact_track": True,
        "selection": "registered_exact_track_correct_local_candidate_vs_different_coherent_wrong_local_candidate_min_linf_then_frozen_prior_v1",
    }
    _validate_identity_target_contract(
        targets=SimpleNamespace(metadata={"spatial_supervision_mode": "registered_exact_identity"}),
        hard_targets=SimpleNamespace(metadata=hard_metadata),
    )
    with pytest.raises(ValueError, match="registered-exact"):
        _validate_identity_target_contract(
            targets=SimpleNamespace(metadata={"spatial_supervision_mode": "gt_projection"}),
            hard_targets=SimpleNamespace(metadata=hard_metadata),
        )


def test_prior_residual_reset_only_neutralizes_final_llr_layer() -> None:
    torch.manual_seed(31)
    sources = {
        name: torch.nn.functional.normalize(torch.randn(3, side, side, dimension), dim=-1)
        for name, dimension, side in (
            ("radio_final", 8, 5),
            ("radio_intermediate", 8, 5),
            ("alike", 6, 7),
        )
    }
    model = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        context_windows={"radio_final": 3, "radio_intermediate": 3, "alike": 3},
    )
    final = model.edge_head[-1]
    rgb_final = model.rgb_identity_head[-1]
    context_final = model.context_coherence_head[-1]
    assert isinstance(final, torch.nn.Linear)
    assert isinstance(rgb_final, torch.nn.Linear)
    assert isinstance(context_final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.fill_(0.75)
        final.bias.fill_(0.25)
        rgb_final.weight.fill_(0.5)
        rgb_final.bias.fill_(0.2)
        context_final.weight.fill_(0.4)
        context_final.bias.fill_(0.1)
    projector_before = model.context_projector[1].weight.detach().clone()
    _reset_identity_llr_final_edge_head(model)
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0
    assert torch.count_nonzero(rgb_final.weight) == 0
    assert torch.count_nonzero(rgb_final.bias) == 0
    assert torch.count_nonzero(context_final.weight) == 0
    assert torch.count_nonzero(context_final.bias) == 0
    assert torch.equal(model.context_projector[1].weight, projector_before)


def test_identity_l0_reset_neutralizes_view_and_null_calibration_too() -> None:
    model = CandidatePoseRGBSpatialIdentityLLR(
        sources={
            name: torch.nn.functional.normalize(torch.randn(3, side, side, dimension), dim=-1)
            for name, dimension, side in (
                ("radio_final", 8, 5),
                ("radio_intermediate", 8, 5),
                ("alike", 6, 7),
            )
        },
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        context_windows={"radio_final": 3, "radio_intermediate": 3, "alike": 3},
    )
    with torch.no_grad():
        for name in (
            "rgb_identity_head",
            "context_coherence_head",
            "support_view_head",
            "null_head",
        ):
            final = getattr(model, name)[-1]
            assert isinstance(final, torch.nn.Linear)
            final.weight.fill_(0.5)
            final.bias.fill_(0.25)
    assert _reset_identity_l0_scalar_head_finals(model) == (
        "rgb_identity_head",
        "context_coherence_head",
        "support_view_head",
        "null_head",
    )
    for name in (
        "rgb_identity_head",
        "context_coherence_head",
        "support_view_head",
        "null_head",
    ):
        final = getattr(model, name)[-1]
        assert isinstance(final, torch.nn.Linear)
        assert torch.count_nonzero(final.weight) == 0
        assert torch.count_nonzero(final.bias) == 0


def test_optimizer_groups_train_neutral_head_faster_than_pretrained_texture() -> None:
    torch.manual_seed(23)
    sources = {
        name: torch.nn.functional.normalize(torch.randn(3, side, side, dimension), dim=-1)
        for name, dimension, side in (
            ("radio_final", 8, 5),
            ("radio_intermediate", 8, 5),
            ("alike", 6, 7),
        )
    }
    model = CandidatePoseRGBSpatialIdentityLLR(
        sources=sources,
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(3, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        context_windows={"radio_final": 3, "radio_intermediate": 3, "alike": 3},
    )
    args = SimpleNamespace(
        learning_rate=1e-4,
        head_learning_rate=1e-3,
        projector_learning_rate=2e-4,
        context_learning_rate=5e-5,
        texture_learning_rate=1e-5,
    )
    groups = _identity_llr_optimizer_parameter_groups(model=model, args=args)
    rates = {str(group["group_name"]): float(group["lr"]) for group in groups}
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert rates == {
        "edge_head": 1e-3,
        "projectors": 2e-4,
        "context": 5e-5,
        "texture": 1e-5,
    }
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_atomic_history_write_never_leaves_the_temporary_payload(tmp_path) -> None:
    path = tmp_path / "history.partial.json"
    _write_json_atomically(path, {"epoch": 1, "loss": 0.25})
    assert path.read_text().strip() == '{\n  "epoch": 1,\n  "loss": 0.25\n}'
    assert not (tmp_path / ".history.partial.json.tmp").exists()


def test_ddp_output_directory_uses_rank_zero_decision(monkeypatch, tmp_path) -> None:
    """A non-owner rank must not reject rank zero's newly-created directory."""

    output = tmp_path / "identity_run"
    output.mkdir()
    broadcasts: list[int] = []

    def fake_broadcast(value: torch.Tensor, src: int) -> None:
        assert src == 0
        broadcasts.append(int(value.item()))
        value.fill_(0)  # Rank zero accepted a fresh run before creating it.

    barriers: list[bool] = []
    monkeypatch.setattr(
        "feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr.distributed.broadcast",
        fake_broadcast,
    )
    monkeypatch.setattr(
        "feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr.distributed.barrier",
        lambda: barriers.append(True),
    )
    _prepare_output_directory(
        output_dir=output,
        force=False,
        state=SimpleNamespace(rank=1, enabled=True, device=torch.device("cpu")),
    )
    assert broadcasts == [0]
    assert barriers == [True]
    assert not (output / "progress.json").exists()


def test_output_directory_refusal_is_owned_by_rank_zero(tmp_path) -> None:
    output = tmp_path / "existing_identity_run"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _prepare_output_directory(
            output_dir=output,
            force=False,
            state=SimpleNamespace(rank=0, enabled=False, device=torch.device("cpu")),
        )


def test_fold_matched_broad_identity_initialization_requires_v3_source_masked_checkpoint(tmp_path) -> None:
    torch.manual_seed(29)
    sources = {
        name: torch.nn.functional.normalize(torch.randn(3, side, side, dimension), dim=-1)
        for name, dimension, side in (
            ("radio_final", 8, 5),
            ("radio_intermediate", 8, 5),
            ("alike", 6, 7),
        )
    }
    windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}
    source_lineage = {
        "radio_final_context_cache_sha256": "final-cache",
        "radio_intermediate_context_cache_sha256": "intermediate-cache",
        "alike_spatial_context_cache_sha256": "alike-cache",
        "source_image_manifest_sha256": "image-manifest",
        "rgb_coordinate_bridge": {"format": "test-bridge"},
    }
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
    checkpoint = tmp_path / "identity_observation.pt"
    metadata = {
        "format": OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "candidate_slot_permutation_equivariant": True,
        "visual_evidence_gate_version": OBSERVATION_IDENTITY_GATE_VERSION,
        "p1_initialization_allowed": True,
        "fixed_candidate_count": 2,
        "observation_pair_inner_validation": {"fold_count": 5, "fold_index": 1},
        "lineage": source_lineage,
        "config": {
            "rgb_context_radius_px": 8.0,
            "rgb_step_px": 1.0,
            "texture_feature_dim": 8,
            "hidden_dim": 8,
            "max_abs_log_ratio": 4.0,
            "context_windows": windows,
        },
        "training": {"inner_validation": {"selected_epoch": 3, "gate": {"passed": True}}},
    }
    torch.save(
        {"format": metadata["format"], "metadata": metadata, "state_dict": source.state_dict()},
        checkpoint,
    )
    args = SimpleNamespace(
        inner_validation_fold_count=5,
        inner_validation_fold_index=1,
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        max_abs_log_ratio=4.0,
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
    loaded = _load_identity_observation_pretrain(
        model=target,
        path=checkpoint,
        candidate_count=2,
        context_windows=windows,
        source_lineage=source_lineage,
        args=args,
    )
    assert loaded["kind"] == "fold_matched_broad_identity_observation_pretrain"
    assert loaded["independent_expert_initialization"] == (
        "loaded_from_gate_approved_v3_source_masked_experts_v1"
    )
    for expert_name in ("rgb_identity_head", "context_coherence_head"):
        expert_state = getattr(target, expert_name).state_dict()
        source_state = getattr(source, expert_name).state_dict()
        assert expert_state.keys() == source_state.keys()
        for name, value in expert_state.items():
            assert torch.equal(value, source_state[name])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["metadata"]["visual_evidence_gate_version"] = "legacy_margin_only_v1"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="strict initialization"):
        _load_identity_observation_pretrain(
            model=target,
            path=checkpoint,
            candidate_count=2,
            context_windows=windows,
            source_lineage=source_lineage,
            args=args,
        )
    payload["metadata"]["visual_evidence_gate_version"] = OBSERVATION_IDENTITY_GATE_VERSION
    payload["metadata"]["model_format"] = "candidate_pose_rgb_spatial_identity_llr_v2"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="strict initialization"):
        _load_identity_observation_pretrain(
            model=target,
            path=checkpoint,
            candidate_count=2,
            context_windows=windows,
            source_lineage=source_lineage,
            args=args,
        )
    payload["metadata"]["model_format"] = CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT
    torch.save(payload, checkpoint)
    args.inner_validation_fold_index = 0
    with pytest.raises(ValueError, match="fold"):
        _load_identity_observation_pretrain(
            model=target,
            path=checkpoint,
            candidate_count=2,
            context_windows=windows,
            source_lineage=source_lineage,
            args=args,
        )
