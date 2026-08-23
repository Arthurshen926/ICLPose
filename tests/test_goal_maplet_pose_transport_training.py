from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization_goal_maplet.pose_transport_training import (
    PoseTransportTrainingConfig,
    leave_one_view_out_pose_field,
    normalized_joint_pose_error,
    pose_transport_energy_landscape_loss,
)


def test_leave_one_view_out_subtracts_self_observation_without_fallback():
    total_sum = np.asarray([[[5.0, 1.0], [2.0, 4.0]]])
    total_weight = np.asarray([[3.0, 1.0]])
    held_sum = np.asarray([[[2.0, 1.0], [2.0, 4.0]]])
    held_weight = np.asarray([[1.0, 1.0]])
    result = leave_one_view_out_pose_field(
        total_sum, total_weight, held_sum, held_weight
    )
    np.testing.assert_allclose(result.mean[0, 0], [1.5, 0.0])
    assert result.valid.tolist() == [[True, False]]
    np.testing.assert_array_equal(result.mean[0, 1], np.zeros(2))


def test_leave_one_view_out_rejects_contribution_not_in_total():
    with pytest.raises(ValueError, match="subset"):
        leave_one_view_out_pose_field(
            np.zeros((1, 2)), np.asarray([0.5]),
            np.zeros((1, 2)), np.asarray([0.75]),
        )


def test_stage_normalized_joint_pose_error_uses_joint_maximum():
    translation = torch.tensor([[0.5, 2.0]])
    rotation = torch.tensor([[45.0, 10.0]])
    error = normalized_joint_pose_error(translation, rotation, stage="coarse")
    torch.testing.assert_close(error, torch.tensor([[1.0, 1.0]]))


def test_energy_landscape_loss_penalizes_score_improving_error_worsening_step():
    score = torch.tensor([[0.0, 0.5]], requires_grad=True)
    total, report = pose_transport_energy_landscape_loss(
        score,
        torch.tensor([[0.2, 1.4]]),
        torch.tensor([[2.0, 20.0]]),
        torch.ones((1, 2), dtype=torch.bool),
        config=PoseTransportTrainingConfig(stage="coarse"),
        drift_negative_pairs=torch.tensor([[0, 0, 1]]),
    )
    assert report["drift_negative"] > 0.0
    total.backward()
    assert score.grad[0, 0] < 0.0
    assert score.grad[0, 1] > 0.0


def test_soft_attribution_kl_requires_explicit_sink_normalization():
    score = torch.tensor([[0.2, 0.1]], requires_grad=True)
    common = dict(
        candidate_score=score,
        translation_error_m=torch.tensor([[0.1, 0.8]]),
        rotation_error_deg=torch.tensor([[1.0, 8.0]]),
        candidate_valid=torch.ones((1, 2), dtype=torch.bool),
        config=PoseTransportTrainingConfig(stage="medium"),
    )
    with pytest.raises(ValueError, match="sum to one"):
        pose_transport_energy_landscape_loss(
            **common,
            attribution_probability=torch.tensor([[0.6, 0.2]]),
            attribution_target=torch.tensor([[0.7, 0.3]]),
        )
    loss, report = pose_transport_energy_landscape_loss(
        **common,
        attribution_probability=torch.tensor([[0.6, 0.3, 0.1]]),
        attribution_target=torch.tensor([[0.8, 0.1, 0.1]]),
    )
    assert report["attribution_kl"] > 0.0
    assert torch.isfinite(loss)


def test_synthetic_score_parameters_learn_correct_pose_order():
    score = torch.nn.Parameter(torch.zeros((1, 4)))
    optimizer = torch.optim.Adam([score], lr=0.08)
    translation = torch.tensor([[0.05, 0.4, 0.9, 1.8]])
    rotation = torch.tensor([[0.5, 4.0, 9.0, 25.0]])
    valid = torch.ones((1, 4), dtype=torch.bool)
    path = torch.tensor([[0, 0, 1], [0, 1, 2], [0, 2, 3]])
    drift = torch.tensor([[0, 0, 3]])
    config = PoseTransportTrainingConfig(stage="coarse")
    for _ in range(60):
        optimizer.zero_grad()
        loss, _report = pose_transport_energy_landscape_loss(
            score, translation, rotation, valid, config=config,
            monotonic_pairs=path, drift_negative_pairs=drift,
        )
        loss.backward()
        optimizer.step()
    assert torch.all(score[0, :-1] > score[0, 1:])


def test_invalid_candidate_is_excluded_from_listwise_target():
    score = torch.tensor([[0.1, 1000.0]], requires_grad=True)
    loss, _ = pose_transport_energy_landscape_loss(
        score,
        torch.tensor([[0.1, 0.0]]), torch.tensor([[1.0, 0.0]]),
        torch.tensor([[True, False]]),
        config=PoseTransportTrainingConfig(stage="fine"),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert score.grad[0, 1] == 0.0
