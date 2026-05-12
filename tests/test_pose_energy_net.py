import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.runtime import apply_pose_delta

from feature_extract.students.pose_energy_net import (
    PoseEnergyNet,
    pose_costs_and_residual_targets,
    pose_energy_losses,
)
from feature_extract.tools.train_pose_energy import apply_config_defaults, parse_float_csv


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
        hidden_dim=None,
        map_channels=None,
        grid_size=None,
        context_layers=None,
        context_heads=None,
        synthetic_ratio=None,
        synthetic_trans_cm=None,
        synthetic_rot_deg=None,
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


def test_pose_energy_config_defaults_allow_partial_namespace():
    args = Namespace(topk=None)

    resolved = apply_config_defaults(args, {"pose_energy": {"synthetic_ratio": 0.25}})

    assert resolved.topk == 64
    assert resolved.synthetic_ratio == 0.25
    assert resolved.target_temperature_m == 0.05
