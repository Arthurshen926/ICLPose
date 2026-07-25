from __future__ import annotations

import pytest
import torch

from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_likelihood import (
    observation_pair_margin_loss,
    observation_pretrain_gate,
)


def test_observation_pair_margin_prefers_same_track_candidate_and_has_gradients() -> None:
    positive = torch.tensor([0.20, 0.70], requires_grad=True)
    negatives = torch.tensor([[0.10, -0.20], [0.60, 0.20]], requires_grad=True)
    loss, metrics = observation_pair_margin_loss(
        scores=torch.cat([positive[:, None], negatives], dim=1),
        usable=torch.ones((2, 3), dtype=torch.bool),
        margin=0.25,
    )

    # Both rows select a distinct hardest wrong candidate; gradients must raise
    # the same-track score and lower the currently hardest negative.
    assert metrics["correct_win_fraction"] == pytest.approx(1.0)
    assert metrics["mean_positive_minus_hardest_negative"] == pytest.approx(0.10)
    loss.backward()
    assert float(positive.grad[1]) < 0.0
    assert float(negatives.grad[1, 0]) > 0.0


def test_observation_pretrain_gate_requires_permutation_separation() -> None:
    passed = observation_pretrain_gate(
        {
            "normal_correct_win_fraction": 0.75,
            "normal_mean_gap": 0.20,
            "visual_gap_delta": 0.08,
        },
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.02,
    )
    assert passed["passed"] is True
    failed = observation_pretrain_gate(
        {
            "normal_correct_win_fraction": 0.75,
            "normal_mean_gap": 0.20,
            "visual_gap_delta": 0.01,
        },
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.02,
    )
    assert failed["passed"] is False
