from __future__ import annotations

import torch

from feature_extract.tools.vfm.train_candidate_multiscale_context_llr import (
    appearance_control_margin_loss,
    configure_single_source_trainable_parameters,
    fixed_final_epoch_checkpoint_selection,
    hard_repeat_control_metrics,
    inner_gate_decision,
    parse_args,
    registered_identity_loss,
    source_candidate_plus_null_logits,
    source_weights,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatBatch,
)
from feature_extract.vfm.localization.candidate_multiscale_context_llr import (
    CandidateMultiscaleContextLLR,
    CandidateMultiscaleContextLLRPrediction,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[40.0, 30.0]]),
        support_image_indices=torch.tensor([[[1], [2]]]),
        support_xy=torch.tensor([[[[40.0, 30.0]], [[60.0, 45.0]]]]),
        support_view_valid=torch.ones((1, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 2, 1)),
        candidate_probabilities=torch.tensor([[0.45, 0.35]]),
        null_probabilities=torch.tensor([0.20]),
    )


def _prediction(
    *, normal: tuple[float, float] = (1.0, -1.0)
) -> CandidateMultiscaleContextLLRPrediction:
    values = torch.tensor(normal, dtype=torch.float32).reshape(1, 2, 1)
    return CandidateMultiscaleContextLLRPrediction(
        source_edge_log_likelihood_ratios={
            "radio_final": values,
            "radio_intermediate": torch.zeros_like(values),
            "alike": torch.zeros_like(values),
        },
        source_edge_usable={
            name: torch.ones_like(values, dtype=torch.bool)
            for name in ("radio_final", "radio_intermediate", "alike")
        },
    )


def test_single_source_scope_and_fixed_prior_null_logits() -> None:
    torch.manual_seed(4)
    model = CandidateMultiscaleContextLLR(
        sources={
            "radio_final": torch.nn.functional.normalize(torch.randn(3, 7, 7, 8), dim=-1),
            "radio_intermediate": torch.nn.functional.normalize(torch.randn(3, 11, 11, 8), dim=-1),
            "alike": torch.nn.functional.normalize(torch.randn(3, 15, 15, 6), dim=-1),
        },
        image_sizes=torch.tensor([[96.0, 72.0]]).repeat(3, 1),
        hidden_dim=8,
    )
    selected = configure_single_source_trainable_parameters(model=model, source="alike")
    assert selected
    assert all(name.startswith("heads.alike.") for name in selected)
    assert all(
        parameter.requires_grad == name.startswith("heads.alike.")
        for name, parameter in model.named_parameters()
    )
    logits, usable = source_candidate_plus_null_logits(
        runtime=_runtime(), prediction=_prediction(normal=(0.0, 0.0)), source="radio_final"
    )
    assert torch.all(usable)
    assert torch.allclose(torch.softmax(logits, dim=1), torch.tensor([[0.45, 0.35, 0.20]]))


def test_registered_identity_is_exact_and_uses_no_null_target_leakage() -> None:
    loss, metrics = registered_identity_loss(
        runtime=_runtime(),
        prediction=_prediction(),
        source="radio_final",
        observed_candidate_mask=torch.tensor([[True, False]]),
    )
    assert torch.isfinite(loss)
    assert metrics["active"] == 1.0
    assert metrics["top1"] == 1.0
    try:
        registered_identity_loss(
            runtime=_runtime(),
            prediction=_prediction(),
            source="radio_final",
            observed_candidate_mask=torch.tensor([[True, True]]),
        )
    except ValueError as error:
        assert "fixed candidate layout" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("ambiguous identity targets must be rejected")


def test_hard_repeat_controls_share_common_availability() -> None:
    hard = HardRepeatBatch(
        point_indices=torch.tensor([0]),
        positive_candidate_indices=torch.tensor([0]),
        negative_candidate_indices=torch.tensor([1]),
        positive_offsets_xy=torch.zeros((1, 2)),
        negative_offsets_xy=torch.zeros((1, 2)),
    )
    normal, permuted, zero = hard_repeat_control_metrics(
        runtime=_runtime(),
        normal_prediction=_prediction(normal=(1.0, -1.0)),
        permuted_runtime=_runtime(),
        permuted_prediction=_prediction(normal=(0.25, -0.25)),
        zero_prediction=_prediction(normal=(0.0, 0.0)),
        source="radio_final",
        hard_batch=hard,
    )
    assert torch.allclose(normal, torch.tensor([2.0]))
    assert torch.allclose(permuted, torch.tensor([0.5]))
    assert torch.allclose(zero, torch.tensor([0.0]))


def test_appearance_control_backpropagates_only_through_visual_gap() -> None:
    normal = torch.tensor([0.20, 0.40], requires_grad=True)
    permuted = torch.tensor([0.05, 0.10])
    zero = torch.tensor([0.00, 0.00])
    loss, metrics = appearance_control_margin_loss(
        normal_gaps=normal, permuted_gaps=permuted, zero_gaps=zero, margin=0.10
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(normal.grad < 0.0)
    assert metrics["normal_minus_permuted"] > 0.0
    assert metrics["normal_minus_zero"] > 0.0


def test_inner_gate_requires_zero_visual_and_permutation_evidence() -> None:
    args = parse_args(
        [
            "--rgb-spatial-layout", "layout.npz",
            "--training-targets", "targets.npz",
            "--hard-repeat-targets", "hard.npz",
            "--radio-final-context-cache", "final.npz",
            "--radio-intermediate-context-cache", "intermediate.npz",
            "--alike-spatial-context-cache", "alike.npz",
            "--output-dir", "out",
            "--source", "radio_final",
        ]
    )
    metrics = {
        "normal_pose_win_fraction": 0.70,
        "normal_pose_gap": 0.20,
        "normal_minus_permuted_pose_gap": 0.10,
        "normal_minus_zero_visual_pose_gap": 0.10,
        "hard_repeat_eligible_query_fraction": 1.0,
        "hard_repeat_win_fraction": 0.70,
        "hard_repeat_gap": 0.20,
        "hard_repeat_normal_minus_permuted_gap": 0.10,
        "hard_repeat_normal_minus_zero_visual_gap": 0.10,
    }
    assert inner_gate_decision(metrics=metrics, args=args)["passed"] is True
    metrics["hard_repeat_normal_minus_zero_visual_gap"] = 0.0
    assert inner_gate_decision(metrics=metrics, args=args)["passed"] is False
    assert source_weights("radio_final") == {
        "radio_final": 1.0,
        "radio_intermediate": 0.0,
        "alike": 0.0,
    }


def test_context_checkpoint_selection_is_fixed_to_final_epoch() -> None:
    selection = fixed_final_epoch_checkpoint_selection(epochs=12)
    assert selection["selected_epoch"] == 12
    assert selection["inner_validation_used_for_model_selection"] is False
