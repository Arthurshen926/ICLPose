from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet.relative_pose_correction import (
    FullTokenRelativePoseCorrectionNet,
    apply_left_pose_correction_coordinate,
    left_pose_correction_coordinate,
    relative_pose_mixture_loss,
)


def test_relative_pose_coordinate_round_trips_coupled_left_se3():
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = np.asarray([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    candidate[:3, 3] = [0.7, -0.2, 1.3]
    coordinate = np.asarray([0.2, -0.3, 0.1, 0.15, -0.25, 0.2])
    target = apply_left_pose_correction_coordinate(candidate, coordinate)
    recovered = left_pose_correction_coordinate(candidate, target)
    np.testing.assert_allclose(recovered, coordinate, atol=2.0e-15, rtol=0.0)
    np.testing.assert_allclose(
        apply_left_pose_correction_coordinate(candidate, recovered),
        target, atol=3.0e-15, rtol=0.0,
    )


def test_multimodal_relative_head_and_loss_are_finite_and_trainable():
    torch.manual_seed(3)
    model = FullTokenRelativePoseCorrectionNet(mode_count=4)
    features = torch.randn(3, 9, 36, 64)
    target = torch.tensor([
        [0.2, 0.0, -0.1, 0.1, 0.0, 0.0],
        [-0.4, 0.3, 0.0, 0.0, -0.2, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    coordinate, logit = model(features)
    assert coordinate.shape == (3, 4, 6)
    assert logit.shape == (3, 4)
    assert torch.all(torch.abs(coordinate) <= 1.0)
    assert torch.all(torch.linalg.vector_norm(coordinate[:, :, 3:], dim=2) <= 1.0 + 1e-6)
    loss, diagnostics = relative_pose_mixture_loss(coordinate, logit, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["mean_best_mode_smooth_l1"] >= 0.0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_relative_mixture_rewards_a_mode_at_the_target():
    target = torch.zeros(1, 6)
    bad = torch.full((1, 2, 6), 0.8)
    good = bad.clone()
    good[0, 1] = 0.0
    logits = torch.zeros(1, 2)
    bad_loss, _ = relative_pose_mixture_loss(bad, logits, target)
    good_loss, _ = relative_pose_mixture_loss(good, logits, target)
    assert good_loss < bad_loss
