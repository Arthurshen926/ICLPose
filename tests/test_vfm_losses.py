import torch

from feature_extract.vfm.losses import (
    basin_bce_loss,
    group_sparsity_loss,
    hard_negative_contrastive_loss,
    listwise_pose_rank_loss,
    track_consistency_loss,
    uncertainty_calibration_loss,
)


def test_track_consistency_loss_prefers_same_track_features():
    same_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    same_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    shifted = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    assert track_consistency_loss(same_a, same_b) < track_consistency_loss(same_a, shifted)


def test_hard_negative_contrastive_loss_has_gradients():
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[0.9, 0.1]])
    negatives = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])

    loss = hard_negative_contrastive_loss(query, positive, negatives, temperature=0.1)
    loss.backward()

    assert loss.item() < 0.1
    assert query.grad is not None


def test_listwise_pose_rank_loss_rewards_low_cost_high_score():
    good_scores = torch.tensor([[2.0, 0.0, -1.0]])
    bad_scores = torch.tensor([[-1.0, 0.0, 2.0]])
    costs = torch.tensor([[0.05, 0.30, 0.90]])

    assert listwise_pose_rank_loss(good_scores, costs) < listwise_pose_rank_loss(bad_scores, costs)


def test_basin_sparsity_and_calibration_losses_are_finite():
    logits = torch.tensor([2.0, -2.0, 0.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    gates = torch.tensor([0.9, 0.1, 0.5], requires_grad=True)
    uncertainty = torch.tensor([0.1, 0.7, 0.2], requires_grad=True)

    loss = (
        basin_bce_loss(logits, labels)
        + group_sparsity_loss(gates)
        + uncertainty_calibration_loss(torch.sigmoid(logits), labels, uncertainty)
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert gates.grad is not None
    assert uncertainty.grad is not None
