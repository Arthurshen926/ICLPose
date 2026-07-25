from __future__ import annotations

import math

import pytest
import torch

from feature_extract.tools.vfm.train_candidate_pose_latent_evidence import (
    conditional_identity_supervision_loss,
    direct_candidate_alignment_margin_loss,
    exact_identity_alignment_weights,
    identity_target_class_masks,
)


def test_identity_supervision_learns_exact_track_and_uniforms_fixed_null_rows() -> None:
    conditional = torch.tensor(
        [[0.80, 0.20], [0.95, 0.05], [0.10, 0.90]], dtype=torch.float32
    )
    # Candidate count is two.  Class two is a registered track absent from the
    # frozen top-L and class -1 is an unlabelled detector anchor.
    target_classes = torch.tensor([0, 2, -1], dtype=torch.long)

    loss, metrics = conditional_identity_supervision_loss(
        conditional_probabilities=conditional,
        target_classes=target_classes,
        null_uniformity_weight=1.0,
    )

    expected_exact = -math.log(0.80)
    expected_null_uniform_kl = -0.5 * (math.log(0.95) + math.log(0.05)) - math.log(2.0)
    assert loss.item() == pytest.approx(expected_exact + expected_null_uniform_kl)
    assert metrics["exact_candidate_target_count"] == 1.0
    assert metrics["fixed_null_target_count"] == 1.0
    assert metrics["unlabelled_target_count"] == 1.0
    assert metrics["exact_candidate_top1_accuracy"] == 1.0


def test_alignment_training_weights_exclude_null_and_unlabelled_tokens() -> None:
    selector = torch.tensor([0.8, 0.6, 0.4, 0.2], dtype=torch.float32)
    target_classes = torch.tensor([1, 3, -1, 0], dtype=torch.long)

    weights, metrics = exact_identity_alignment_weights(
        selector_weights=selector,
        target_classes=target_classes,
        candidate_count=3,
    )

    torch.testing.assert_close(weights, torch.tensor([0.8, 0.0, 0.0, 0.2]))
    assert metrics["exact_identity_alignment_token_count"] == 2.0
    assert metrics["exact_identity_alignment_weight_sum"] == pytest.approx(1.0)


def test_identity_target_masks_do_not_treat_unlabelled_minus_one_as_exact() -> None:
    exact, fixed_null, unlabelled = identity_target_class_masks(
        target_classes=torch.tensor([-1, 0, 2, 3], dtype=torch.long),
        candidate_count=3,
    )

    torch.testing.assert_close(exact, torch.tensor([False, True, True, False]))
    torch.testing.assert_close(fixed_null, torch.tensor([False, False, False, True]))
    torch.testing.assert_close(unlabelled, torch.tensor([True, False, False, False]))


def test_direct_candidate_alignment_margin_uses_true_candidate_before_mixture() -> None:
    candidate_alignment = torch.tensor(
        [
            [[0.0, 2.0], [3.0, 0.0], [9.0, 9.0]],
            [[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]],
            [[0.0, 1.0], [2.0, 0.0], [9.0, 9.0]],
        ],
        dtype=torch.float32,
    )
    # The first two tokens are true candidates 1 and 0. The final token is a
    # fixed-null target and cannot supervise an edge-specific pose loss.
    target_classes = torch.tensor([1, 0, 2], dtype=torch.long)

    loss, metrics = direct_candidate_alignment_margin_loss(
        candidate_alignment=candidate_alignment,
        target_classes=target_classes,
        margin=0.25,
    )

    assert loss.item() > 0.0
    assert metrics["exact_candidate_token_count"] == 2.0
    assert metrics["correct_win_fraction"] == 1.0
    assert metrics["mean_correct_minus_hardest_wrong"] == pytest.approx(1.0)
