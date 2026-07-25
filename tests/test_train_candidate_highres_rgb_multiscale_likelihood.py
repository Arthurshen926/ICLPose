from __future__ import annotations

import torch

from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    appearance_control_margin_loss,
    balanced_ddp_query_schedules,
    configure_trainable_parameters,
    ddp_owner_cost_balance_metrics,
    fixed_final_epoch_checkpoint_selection,
    inner_gate_decision,
    parse_args,
    pose_margin_terms,
    registered_observation_appearance_control_terms,
    source_weights,
)
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscaleLikelihood,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_support_patch_appearance,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[45.0, 35.0], [52.0, 42.0]]),
        support_image_indices=torch.tensor([[[2, 3], [4, 5]], [[2, 3], [4, 5]]]),
        support_xy=torch.tensor(
            [
                [[[44.0, 35.0], [49.0, 39.0]], [[58.0, 44.0], [61.0, 49.0]]],
                [[[47.0, 37.0], [51.0, 42.0]], [[57.0, 46.0], [63.0, 52.0]]],
            ]
        ),
        support_view_valid=torch.ones((2, 2, 2), dtype=torch.bool),
        candidate_view_weights=torch.full((2, 2, 2), 0.5),
        candidate_probabilities=torch.tensor([[0.45, 0.35], [0.50, 0.30]]),
        null_probabilities=torch.tensor([0.20, 0.20]),
    )


def _patches(model: CandidateHighresRGBMultiscaleLikelihood) -> tuple[torch.Tensor, torch.Tensor]:
    runtime = _runtime()
    torch.manual_seed(31)
    return (
        torch.rand((runtime.point_count, 3, model.patch_side, model.patch_side)),
        torch.rand(
            (
                runtime.point_count,
                runtime.candidate_count,
                runtime.support_view_count,
                3,
                model.patch_side,
                model.patch_side,
            )
        ),
    )


def _args():
    return parse_args(
        [
            "--rgb-spatial-layout",
            "layout.npz",
            "--training-targets",
            "targets.npz",
            "--hard-repeat-targets",
            "hard.npz",
            "--radio-final-context-cache",
            "final.npz",
            "--radio-intermediate-context-cache",
            "intermediate.npz",
            "--alike-spatial-context-cache",
            "alike.npz",
            "--image-root",
            "rgb",
            "--output-dir",
            "out",
            "--source",
            "broad",
        ]
    )


def test_selected_source_scope_keeps_shared_texture_and_one_calibrator_trainable() -> None:
    model = CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.tensor([[96.0, 72.0]]),
        fine_search_radius_px=2.0,
        fine_context_radius_px=2.0,
        broad_search_radius_px=2.0,
        broad_context_radius_px=4.0,
        texture_feature_dim=8,
        hidden_dim=8,
    )
    selected = configure_trainable_parameters(model=model, source="broad")
    assert selected
    assert any(name.startswith("texture_encoder.") for name in selected)
    assert any(name.startswith("calibrators.broad.") for name in selected)
    assert all(
        parameter.requires_grad
        == (name.startswith("texture_encoder.") or name.startswith("calibrators.broad."))
        for name, parameter in model.named_parameters()
    )
    assert source_weights("combined") == {"fine": 0.5, "broad": 0.5}


def test_pose_and_appearance_losses_backpropagate_only_from_visual_gaps() -> None:
    correct = torch.tensor([0.4], requires_grad=True)
    wrong = torch.tensor([0.1, 0.2])
    hardest, soft, gap = pose_margin_terms(
        correct_scores=correct, wrong_scores=wrong, margin=0.2, temperature=0.3
    )
    control, metrics = appearance_control_margin_loss(
        normal_gaps=gap,
        permuted_gaps=torch.tensor([0.0]),
        margin=0.05,
    )
    (hardest + soft + control).backward()
    assert torch.isfinite(hardest + soft + control)
    assert correct.grad is not None and correct.grad.item() < 0.0
    assert metrics["normal_minus_permuted"] > 0.0


def test_inner_gate_requires_permutation_and_zero_visual_separation() -> None:
    args = _args()
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
    metrics["normal_minus_zero_visual_pose_gap"] = 0.0
    assert inner_gate_decision(metrics=metrics, args=args)["passed"] is False


def test_highres_trainer_defaults_to_cpu_full_image_cache_and_larger_edge_chunks() -> None:
    args = _args()
    assert args.rgb_cache_device == "cpu"
    assert args.rgb_cache_dtype == "uint8"
    assert args.edge_chunk_size == 128
    assert args.amp_init_scale == 4096.0


def test_checkpoint_selection_is_fixed_to_the_predeclared_final_epoch() -> None:
    selection = fixed_final_epoch_checkpoint_selection(epochs=4)
    assert selection["selected_epoch"] == 4
    assert selection["inner_validation_used_for_model_selection"] is False


def test_registered_observation_control_uses_only_observed_in_window_candidates() -> None:
    runtime = _runtime()
    model = CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.tensor([[96.0, 72.0]]).repeat(6, 1),
        fine_search_radius_px=2.0,
        fine_context_radius_px=2.0,
        broad_search_radius_px=2.0,
        broad_context_radius_px=4.0,
        texture_feature_dim=8,
        hidden_dim=8,
    ).train()
    query, support = _patches(model)
    normal = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    permuted = model(
        runtime=runtime,
        query_rgb_patches=query,
        support_rgb_patches=permute_support_patch_appearance(
            runtime=runtime, support_patches=support, shift=1
        ),
    )
    offsets = torch.tensor([[[0.0, 0.0], [99.0, 99.0]], [[0.0, 0.0], [0.0, 0.0]]])
    observed = torch.tensor([[True, True], [False, True]])
    supervised = torch.ones((2, 2), dtype=torch.bool)
    loss, metrics = registered_observation_appearance_control_terms(
        runtime=runtime,
        normal_prediction=normal,
        permuted_prediction=permuted,
        target_offsets_xy=offsets,
        target_observed=observed,
        target_supervised=supervised,
        source="fine",
        max_abs_pose_log_ratio=6.0,
        margin=0.05,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["observed_candidate_count"] == 3.0
    assert metrics["usable_candidate_count"] == 2.0
    assert model.calibrators["fine"].network[-1].bias.grad is not None


def test_cost_balanced_ddp_schedule_preserves_query_coverage_and_pairs_similar_work() -> None:
    owner_costs = {"q0": 100, "q1": 103, "q2": 190, "q3": 193, "q4": 310, "q5": 312}
    schedules = balanced_ddp_query_schedules(
        query_ids=tuple(owner_costs), owner_costs=owner_costs, world_size=2, seed=17
    )
    assert len(schedules) == 2
    assert sorted((*schedules[0], *schedules[1])) == sorted(owner_costs)
    metrics = ddp_owner_cost_balance_metrics(schedules=schedules, owner_costs=owner_costs)
    assert metrics["mean_abs_owner_cost_difference"] <= 3.0
    assert metrics["max_abs_owner_cost_difference"] <= 3.0
