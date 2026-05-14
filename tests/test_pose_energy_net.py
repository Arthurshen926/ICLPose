import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.runtime import apply_pose_delta

from feature_extract.students.pose_energy_net import (
    PoseEnergyNet,
    PoseFeatureDomainAdapter,
    compose_w2c_from_center_and_rotation,
    pose_costs_and_residual_targets,
    pose_energy_losses,
    pose_energy_factorized_selection,
    pose_energy_selection_scores,
)
from feature_extract.tools.train_pose_energy import (
    apply_config_defaults,
    build_candidate_bank,
    candidate_center_pose,
    parse_candidate_center_buckets,
    parse_float_csv,
)
from feature_extract.train_impl import pose_error_tensors


def test_pose_energy_net_forward_shapes():
    model = PoseEnergyNet(
        vector_dim=5,
        score_map_channels=3,
        map_channels=4,
        grid_size=2,
        hidden_dim=16,
        context_layers=1,
        context_heads=1,
    )
    score_maps = torch.randn(2, 7, 3, 8, 10)
    vector_features = torch.randn(2, 7, 5)

    out = model(score_maps, vector_features)

    assert out["energy_logits"].shape == (2, 7)
    assert out["residual_delta"].shape == (2, 7, 6)
    assert out["confidence_logits"].shape == (2, 7)


def test_pose_energy_net_factorized_heads_return_component_logits():
    model = PoseEnergyNet(
        vector_dim=5,
        score_map_channels=3,
        map_channels=4,
        grid_size=2,
        hidden_dim=16,
        context_layers=0,
        factorized_heads=True,
    )
    score_maps = torch.randn(2, 7, 3, 8, 10)
    vector_features = torch.randn(2, 7, 5)

    out = model(score_maps, vector_features)

    assert out["translation_energy_logits"].shape == (2, 7)
    assert out["rotation_energy_logits"].shape == (2, 7)
    assert out["joint_energy_logits"].shape == (2, 7)
    assert out["energy_logits"].shape == (2, 7)


def test_pose_energy_factorized_selection_composes_center_and_rotation_heads():
    angle = torch.tensor(0.30)
    wrong_rot = torch.eye(4)
    wrong_rot[:3, :3] = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    wrong_center = torch.eye(4)
    wrong_center[:3, 3] = torch.tensor([-1.0, 0.0, 0.0])
    bad = torch.eye(4)
    bad[:3, 3] = torch.tensor([0.0, -1.0, 0.0])
    candidates = torch.stack([wrong_rot, wrong_center, bad], dim=0).view(1, 3, 4, 4)
    outputs = {
        "energy_logits": torch.zeros(1, 3),
        "translation_energy_logits": torch.tensor([[4.0, 0.0, -1.0]]),
        "rotation_energy_logits": torch.tensor([[0.0, 4.0, -1.0]]),
    }

    selected, trans_idx, rot_idx = pose_energy_factorized_selection(outputs, candidates)

    assert trans_idx.item() == 0
    assert rot_idx.item() == 1
    assert torch.allclose(selected[0], torch.eye(4), atol=1e-5)


def test_compose_w2c_from_center_and_rotation_preserves_center_and_rotation():
    center_pose = torch.eye(4).view(1, 4, 4)
    center_pose[:, :3, 3] = torch.tensor([[-2.0, 0.0, 0.0]])
    angle = torch.tensor(0.25)
    rotation_pose = torch.eye(4).view(1, 4, 4)
    rotation_pose[:, :3, :3] = torch.tensor(
        [
            [
                [torch.cos(angle), -torch.sin(angle), 0.0],
                [torch.sin(angle), torch.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )

    composed = compose_w2c_from_center_and_rotation(center_pose, rotation_pose)
    center = -(composed[:, :3, :3].transpose(-1, -2) @ composed[:, :3, 3:4]).squeeze(-1)
    expected_center = -(center_pose[:, :3, :3].transpose(-1, -2) @ center_pose[:, :3, 3:4]).squeeze(-1)

    assert torch.allclose(composed[:, :3, :3], rotation_pose[:, :3, :3], atol=1e-6)
    assert torch.allclose(center, expected_center, atol=1e-6)


def test_pose_energy_net_can_zero_init_residual_without_flat_energy():
    torch.manual_seed(5)
    model = PoseEnergyNet(
        vector_dim=0,
        score_map_channels=3,
        map_channels=4,
        grid_size=2,
        hidden_dim=16,
        context_layers=0,
        zero_init_residual_head=True,
    )
    score_maps = torch.randn(2, 7, 3, 8, 10)

    out = model(score_maps)

    assert torch.allclose(out["residual_delta"], torch.zeros_like(out["residual_delta"]))
    assert out["energy_logits"].std() > 0.0


def test_pose_feature_domain_adapter_projects_query_and_render_without_shape_change():
    adapter = PoseFeatureDomainAdapter(
        channels=8,
        hidden_dim=12,
        residual_scale=0.0,
        zero_init=True,
        l2_normalize=True,
    )
    query = torch.randn(2, 8, 5, 6)
    render = torch.randn(14, 8, 5, 6)

    query_loc = adapter.project_query(query)
    render_loc = adapter.project_render(render)

    assert query_loc.shape == query.shape
    assert render_loc.shape == render.shape
    assert torch.allclose(torch.linalg.norm(query_loc, dim=1).mean(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(torch.linalg.norm(render_loc, dim=1).mean(), torch.tensor(1.0), atol=1e-4)


def test_pose_feature_domain_adapter_has_independent_query_and_render_params():
    adapter = PoseFeatureDomainAdapter(channels=4, hidden_dim=4, zero_init=True)

    assert adapter.query_adapter is not adapter.render_adapter
    assert next(adapter.query_adapter.parameters()) is not next(adapter.render_adapter.parameters())


def test_pose_feature_domain_adapter_can_predict_query_and_render_uncertainty():
    adapter = PoseFeatureDomainAdapter(channels=4, hidden_dim=4, zero_init=True, uncertainty_enabled=True)
    query = torch.randn(2, 4, 5, 6)
    render = torch.randn(3, 4, 5, 6)

    query_loc, query_unc = adapter.project_query_with_uncertainty(query)
    render_loc, render_unc = adapter.project_render_with_uncertainty(render)

    assert query_loc.shape == query.shape
    assert render_loc.shape == render.shape
    assert query_unc.shape == (2, 1, 5, 6)
    assert render_unc.shape == (3, 1, 5, 6)
    assert torch.all((query_unc >= 0.0) & (query_unc <= 1.0))
    assert torch.all((render_unc >= 0.0) & (render_unc <= 1.0))


def test_pose_costs_and_residual_targets_match_left_update_convention():
    pose_gt = torch.eye(4).view(1, 4, 4)
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.10],
        ],
        dtype=torch.float32,
    )
    candidates = apply_pose_delta(pose_gt.expand(3, -1, -1), deltas).view(1, 3, 4, 4)

    costs, residuals, trans_err, rot_err = pose_costs_and_residual_targets(
        candidates,
        pose_gt,
        rot_cost_weight=0.5,
    )

    assert costs.shape == (1, 3)
    assert residuals.shape == (1, 3, 6)
    assert torch.argmin(costs, dim=1).item() == 0
    assert torch.allclose(residuals[0, 0], torch.zeros(6), atol=1e-5)

    corrected = apply_pose_delta(
        candidates.reshape(3, 4, 4),
        residuals.reshape(3, 6),
    ).view(1, 3, 4, 4)
    assert torch.allclose(corrected, pose_gt[:, None].expand_as(corrected), atol=1e-4)
    assert trans_err[0, 1] > trans_err[0, 0]
    assert rot_err[0, 2] > rot_err[0, 0]


def test_pose_energy_losses_reward_low_energy_for_low_pose_cost():
    pose_gt = torch.eye(4).view(1, 4, 4)
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.50, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    candidates = apply_pose_delta(pose_gt.expand(3, -1, -1), deltas).view(1, 3, 4, 4)
    good_outputs = {
        "energy_logits": torch.tensor([[4.0, 0.0, -4.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }
    bad_outputs = {
        "energy_logits": torch.tensor([[-4.0, 0.0, 4.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }

    good = pose_energy_losses(
        good_outputs,
        candidates,
        pose_gt,
        target_temperature_m=0.05,
        residual_weight=0.0,
        improve_weight=0.0,
    )
    bad = pose_energy_losses(
        bad_outputs,
        candidates,
        pose_gt,
        target_temperature_m=0.05,
        residual_weight=0.0,
        improve_weight=0.0,
    )

    assert good["energy_loss"] < bad["energy_loss"]
    assert good["target_index"].item() == 0
    assert good["pred_index"].item() == 0
    assert bad["pred_index"].item() == 2


def test_pose_energy_losses_train_factorized_translation_and_rotation_heads():
    pose_gt = torch.eye(4).view(1, 4, 4)
    candidates = apply_pose_delta(
        pose_gt.expand(3, -1, -1),
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.30],
                [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0, 0.0, 0.0, 0.05],
            ],
            dtype=torch.float32,
        ),
    ).view(1, 3, 4, 4)
    good_outputs = {
        "energy_logits": torch.tensor([[0.0, 0.0, 3.0]]),
        "translation_energy_logits": torch.tensor([[3.0, -3.0, 2.0]]),
        "rotation_energy_logits": torch.tensor([[-3.0, 3.0, 2.0]]),
        "joint_energy_logits": torch.tensor([[0.0, 0.0, 3.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }
    bad_outputs = {
        "energy_logits": torch.tensor([[0.0, 0.0, 3.0]]),
        "translation_energy_logits": torch.tensor([[-3.0, 3.0, 0.0]]),
        "rotation_energy_logits": torch.tensor([[3.0, -3.0, 0.0]]),
        "joint_energy_logits": torch.tensor([[0.0, 0.0, 3.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }

    good = pose_energy_losses(
        good_outputs,
        candidates,
        pose_gt,
        target_temperature_m=0.05,
        rot_cost_weight=0.5,
        residual_weight=0.0,
        translation_energy_weight=1.0,
        rotation_energy_weight=1.0,
        joint_energy_weight=1.0,
    )
    bad = pose_energy_losses(
        bad_outputs,
        candidates,
        pose_gt,
        target_temperature_m=0.05,
        rot_cost_weight=0.5,
        residual_weight=0.0,
        translation_energy_weight=1.0,
        rotation_energy_weight=1.0,
        joint_energy_weight=1.0,
    )

    assert good["translation_energy_loss"] < bad["translation_energy_loss"]
    assert good["rotation_energy_loss"] < bad["rotation_energy_loss"]
    assert good["loss"] < bad["loss"]


def test_pose_energy_losses_include_component_hard_ce_for_factorized_heads():
    pose_gt = torch.eye(4).view(1, 4, 4)
    candidates = apply_pose_delta(
        pose_gt.expand(3, -1, -1),
        torch.tensor(
            [
                [0.00, 0.0, 0.0, 0.0, 0.0, 0.30],
                [0.25, 0.0, 0.0, 0.0, 0.0, 0.00],
                [0.10, 0.0, 0.0, 0.0, 0.0, 0.10],
            ],
            dtype=torch.float32,
        ),
    ).view(1, 3, 4, 4)
    bad_outputs = {
        "energy_logits": torch.zeros(1, 3),
        "translation_energy_logits": torch.tensor([[-2.0, 2.0, 0.0]]),
        "rotation_energy_logits": torch.tensor([[2.0, -2.0, 0.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }
    good_outputs = {
        "energy_logits": torch.zeros(1, 3),
        "translation_energy_logits": torch.tensor([[2.0, -2.0, 0.0]]),
        "rotation_energy_logits": torch.tensor([[-2.0, 2.0, 0.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }

    bad = pose_energy_losses(
        bad_outputs,
        candidates,
        pose_gt,
        residual_weight=0.0,
        component_hard_ce_weight=1.0,
    )
    good = pose_energy_losses(
        good_outputs,
        candidates,
        pose_gt,
        residual_weight=0.0,
        component_hard_ce_weight=1.0,
    )

    assert bad["component_hard_ce_loss"] > good["component_hard_ce_loss"]
    assert bad["loss"] > good["loss"]


def test_pose_energy_losses_include_hard_ce_term():
    pose_gt = torch.eye(4).view(1, 4, 4)
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    candidates = apply_pose_delta(pose_gt.expand(2, -1, -1), deltas).view(1, 2, 4, 4)
    outputs = {
        "energy_logits": torch.tensor([[-2.0, 2.0]]),
        "residual_delta": torch.zeros(1, 2, 6),
        "confidence_logits": torch.zeros(1, 2),
    }

    without_hard = pose_energy_losses(
        outputs,
        candidates,
        pose_gt,
        target_temperature_m=0.05,
        hard_ce_weight=0.0,
        residual_weight=0.0,
        improve_weight=0.0,
    )
    with_hard = pose_energy_losses(
        outputs,
        candidates,
        pose_gt,
        target_temperature_m=0.05,
        hard_ce_weight=1.0,
        residual_weight=0.0,
        improve_weight=0.0,
    )

    assert with_hard["hard_ce_loss"] > 0
    assert with_hard["loss"] > without_hard["loss"]


def test_pose_energy_losses_penalize_identity_when_better_non_identity_exists():
    pose_gt = torch.eye(4).view(1, 4, 4)
    init_pose = apply_pose_delta(
        pose_gt,
        torch.tensor([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    closer_pose = apply_pose_delta(
        pose_gt,
        torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    candidates = torch.stack([init_pose[0], closer_pose[0]], dim=0).view(1, 2, 4, 4)
    identity_biased = {
        "energy_logits": torch.tensor([[3.0, 0.0]]),
        "residual_delta": torch.zeros(1, 2, 6),
        "confidence_logits": torch.zeros(1, 2),
    }
    corrected = {
        "energy_logits": torch.tensor([[0.0, 3.0]]),
        "residual_delta": torch.zeros(1, 2, 6),
        "confidence_logits": torch.zeros(1, 2),
    }

    bad = pose_energy_losses(
        identity_biased,
        candidates,
        pose_gt,
        residual_weight=0.0,
        improve_weight=0.0,
        anti_identity_weight=1.0,
        anti_identity_margin=0.5,
        anti_identity_min_gap_m=0.03,
        identity_index=0,
    )
    good = pose_energy_losses(
        corrected,
        candidates,
        pose_gt,
        residual_weight=0.0,
        improve_weight=0.0,
        anti_identity_weight=1.0,
        anti_identity_margin=0.5,
        anti_identity_min_gap_m=0.03,
        identity_index=0,
    )

    assert bad["anti_identity_loss"] > 0.0
    assert good["anti_identity_loss"] == 0.0
    assert bad["loss"] > good["loss"]


def test_pose_energy_losses_pairwise_rank_pushes_best_above_worse_candidates():
    pose_gt = torch.eye(4).view(1, 4, 4)
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    candidates = apply_pose_delta(pose_gt.expand(3, -1, -1), deltas).view(1, 3, 4, 4)
    bad_rank = {
        "energy_logits": torch.tensor([[0.0, 2.0, 3.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }
    good_rank = {
        "energy_logits": torch.tensor([[3.0, 1.0, 0.0]]),
        "residual_delta": torch.zeros(1, 3, 6),
        "confidence_logits": torch.zeros(1, 3),
    }

    bad = pose_energy_losses(
        bad_rank,
        candidates,
        pose_gt,
        residual_weight=0.0,
        improve_weight=0.0,
        pairwise_rank_weight=1.0,
        pairwise_rank_min_gap_m=0.03,
        pairwise_rank_logit_margin=0.5,
    )
    good = pose_energy_losses(
        good_rank,
        candidates,
        pose_gt,
        residual_weight=0.0,
        improve_weight=0.0,
        pairwise_rank_weight=1.0,
        pairwise_rank_min_gap_m=0.03,
        pairwise_rank_logit_margin=0.5,
    )

    assert bad["pairwise_rank_loss"] > good["pairwise_rank_loss"]
    assert bad["loss"] > good["loss"]


def test_pose_energy_losses_can_train_confidence_head():
    pose_gt = torch.eye(4).view(1, 4, 4)
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.30, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    candidates = apply_pose_delta(pose_gt.expand(2, -1, -1), deltas).view(1, 2, 4, 4)
    confidence_logits = torch.zeros(1, 2, requires_grad=True)
    outputs = {
        "energy_logits": torch.zeros(1, 2, requires_grad=True),
        "residual_delta": torch.zeros(1, 2, 6, requires_grad=True),
        "confidence_logits": confidence_logits,
    }

    losses = pose_energy_losses(
        outputs,
        candidates,
        pose_gt,
        residual_weight=0.0,
        improve_weight=0.0,
        confidence_weight=1.0,
        confidence_temperature_m=0.05,
    )
    losses["loss"].backward()

    assert losses["confidence_loss"] > 0.0
    assert confidence_logits.grad is not None
    assert confidence_logits.grad[0, 0] < 0.0
    assert confidence_logits.grad[0, 1] > 0.0


def test_pose_energy_selection_scores_can_use_confidence_and_residual_norm():
    outputs = {
        "energy_logits": torch.tensor([[0.0, 0.0, 0.0]]),
        "confidence_logits": torch.tensor([[0.0, 2.0, 0.0]]),
        "residual_delta": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.50, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
    }

    confidence_scores = pose_energy_selection_scores(outputs, confidence_weight=1.0)
    residual_scores = pose_energy_selection_scores(
        outputs,
        residual_norm_weight=1.0,
        residual_trans_scale_m=0.25,
        residual_rot_scale_rad=0.1,
    )

    assert confidence_scores.argmax(dim=1).item() == 1
    assert residual_scores[0, 0] > residual_scores[0, 2]


def test_pose_energy_residual_loss_uses_separate_trans_rot_scales():
    pose_gt = torch.eye(4).view(1, 4, 4)
    candidate = apply_pose_delta(
        pose_gt,
        torch.tensor([[0.10, 0.0, 0.0, 0.0, 0.0, 0.10]], dtype=torch.float32),
    ).view(1, 1, 4, 4)
    outputs = {
        "energy_logits": torch.zeros(1, 1),
        "residual_delta": torch.zeros(1, 1, 6),
        "confidence_logits": torch.zeros(1, 1),
    }

    loose = pose_energy_losses(
        outputs,
        candidate,
        pose_gt,
        residual_weight=1.0,
        improve_weight=0.0,
        residual_trans_scale_m=1.0,
        residual_rot_scale_rad=1.0,
    )
    tight = pose_energy_losses(
        outputs,
        candidate,
        pose_gt,
        residual_weight=1.0,
        improve_weight=0.0,
        residual_trans_scale_m=0.05,
        residual_rot_scale_rad=0.05,
    )

    assert tight["residual_loss"] > loose["residual_loss"]


def test_pose_energy_improve_loss_has_finite_residual_gradients():
    pose_gt = torch.eye(4).view(1, 4, 4)
    candidates = apply_pose_delta(
        pose_gt.expand(3, -1, -1),
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.10, 0.0, 0.0, 0.0, 0.0, 0.05],
                [0.25, 0.0, 0.0, 0.0, 0.0, 0.10],
            ],
            dtype=torch.float32,
        ),
    ).view(1, 3, 4, 4)
    residual_delta = torch.zeros(1, 3, 6, requires_grad=True)
    outputs = {
        "energy_logits": torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True),
        "residual_delta": residual_delta,
        "confidence_logits": torch.zeros(1, 3),
    }

    losses = pose_energy_losses(
        outputs,
        candidates,
        pose_gt,
        residual_weight=1.0,
        improve_weight=1.0,
        residual_trans_scale_m=0.25,
        residual_rot_scale_rad=0.1,
    )
    losses["loss"].backward()

    assert residual_delta.grad is not None
    assert torch.isfinite(residual_delta.grad).all()
    assert residual_delta.grad.abs().max() > 0.0


def test_pose_energy_residual_top1_mode_ignores_worse_candidates():
    pose_gt = torch.eye(4).view(1, 4, 4)
    candidates = apply_pose_delta(
        pose_gt.expand(3, -1, -1),
        torch.tensor(
            [
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.20, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.40, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    ).view(1, 3, 4, 4)
    residual_delta = torch.zeros(1, 3, 6, requires_grad=True)
    outputs = {
        "energy_logits": torch.tensor([[2.0, 0.0, -2.0]], requires_grad=True),
        "residual_delta": residual_delta,
        "confidence_logits": torch.zeros(1, 3),
    }

    losses = pose_energy_losses(
        outputs,
        candidates,
        pose_gt,
        residual_weight=1.0,
        improve_weight=0.0,
        residual_target_mode="top1",
    )
    losses["residual_loss"].backward()

    assert residual_delta.grad is not None
    assert residual_delta.grad[0, 0].abs().sum() > 0.0
    assert torch.allclose(residual_delta.grad[0, 1:], torch.zeros_like(residual_delta.grad[0, 1:]))


def test_pose_energy_train_args_use_config_defaults():
    args = Namespace(
        topk=None,
        lattice_trans_cm=None,
        lattice_rot_deg=None,
        lattice_direction_mode=None,
        combine_trans_rot=None,
        limit_strategy=None,
        target_temperature_m=None,
        rot_cost_weight=None,
        hard_ce_weight=None,
        residual_weight=None,
        improve_weight=None,
        improve_margin_m=None,
        auc_good_m=None,
        auc_bad_m=None,
        score_radius=None,
        score_preprocess=None,
        score_highpass_kernel=None,
        score_map_mode=None,
        use_candidate_delta=None,
        use_delta_vector=None,
        hidden_dim=None,
        map_channels=None,
        grid_size=None,
        context_layers=None,
        context_heads=None,
        synthetic_ratio=None,
        synthetic_trans_cm=None,
        synthetic_rot_deg=None,
        pose_feature_adapter_enabled=None,
        pose_feature_adapter_hidden_dim=None,
        pose_feature_adapter_residual_scale=None,
        pose_feature_adapter_zero_init=None,
        pose_feature_adapter_l2_normalize=None,
        residual_trans_scale_m=None,
        residual_rot_scale_deg=None,
        anti_identity_weight=None,
        anti_identity_margin=None,
        anti_identity_min_gap_m=None,
        identity_index=None,
        pairwise_rank_weight=None,
        pairwise_rank_min_gap_m=None,
        pairwise_rank_logit_margin=None,
    )
    cfg = {
        "pose_energy": {
            "topk": 32,
            "lattice_trans_cm": [0, 10, 25],
            "synthetic_ratio": 0.5,
        }
    }

    resolved = apply_config_defaults(args, cfg)

    assert resolved.topk == 32
    assert parse_float_csv(resolved.lattice_trans_cm) == [0.0, 10.0, 25.0]
    assert resolved.synthetic_ratio == 0.5
    assert resolved.lattice_direction_mode == "cube"
    assert resolved.auc_good_m == 0.05


def test_candidate_center_pose_accepts_bucket_mixture():
    args = Namespace(
        candidate_center_mode="noisy_init",
        candidate_center_noise_mode="fixed",
        candidate_center_trans_cm=0.0,
        candidate_center_rot_deg=0.0,
        candidate_center_buckets="50:10:1",
    )
    pose_gt = torch.eye(4).view(1, 4, 4).repeat(4, 1, 1)

    init_pose = candidate_center_pose(pose_gt, args)
    _rot_loss, delta_r, delta_t = pose_error_tensors(init_pose, pose_gt)

    assert torch.allclose(delta_t, torch.full_like(delta_t, 0.5), atol=1e-3)
    assert torch.allclose(delta_r, torch.full_like(delta_r, 10.0), atol=1e-4)


def test_parse_candidate_center_buckets_supports_weighted_strings():
    buckets = parse_candidate_center_buckets("10:2:0.25,25:5:0.75")

    assert buckets == [(10.0, 2.0, 0.25), (25.0, 5.0, 0.75)]


def test_pose_energy_balanced_candidate_bank_includes_joint_candidates():
    args = Namespace(
        topk=16,
        lattice_trans_cm="10",
        lattice_rot_deg="5",
        lattice_direction_mode="axis",
        combine_trans_rot=False,
        limit_strategy="first",
        candidate_bank_mode="balanced",
    )
    pose_gt = torch.eye(4).view(1, 4, 4)

    bank = build_candidate_bank(pose_gt, args)
    delta_t = torch.linalg.norm(bank[:, :, :3, 3] - pose_gt[:, None, :3, 3], dim=-1)
    rot_trace = bank[:, :, 0, 0] + bank[:, :, 1, 1] + bank[:, :, 2, 2]
    delta_r = torch.acos(torch.clamp((rot_trace - 1.0) * 0.5, -1.0, 1.0))

    assert ((delta_t > 0.05) & (delta_r > 0.01)).any()


def test_pose_energy_config_defaults_allow_partial_namespace():
    args = Namespace(topk=None)

    resolved = apply_config_defaults(args, {"pose_energy": {"synthetic_ratio": 0.25}})

    assert resolved.topk == 64
    assert resolved.synthetic_ratio == 0.25
    assert resolved.target_temperature_m == 0.05
