from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.pretrain_candidate_pose_context_identity_l0 import (
    _query_grouped_means,
    candidate_slot_permutations_from_anchor_ids,
    fixed_final_epoch_checkpoint_selection,
    observation_context_identity_gate,
    positive_targets_after_candidate_slot_permutation,
)


def test_observation_l0_checkpoint_selection_is_fixed_final_epoch() -> None:
    selection = fixed_final_epoch_checkpoint_selection(epochs=6)
    assert selection == {
        "policy": "fixed_final_epoch_without_inner_validation_model_selection_v1",
        "selected_epoch": 6,
        "inner_validation_used_for_model_selection": False,
    }


def test_candidate_slot_permutations_are_anchor_stable_and_move_positive_labels() -> None:
    anchors = np.asarray([17, 23, 31], dtype=np.int64)
    first = candidate_slot_permutations_from_anchor_ids(
        anchor_ids=anchors, candidate_count=4, seed=7
    )
    second = candidate_slot_permutations_from_anchor_ids(
        anchor_ids=anchors, candidate_count=4, seed=7
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        torch.sort(first, dim=1).values,
        torch.arange(4).expand(len(anchors), -1),
    )
    targets = positive_targets_after_candidate_slot_permutation(first)
    assert torch.all(targets.sum(dim=1) == 1)
    labels = torch.argmax(targets.to(dtype=torch.long), dim=1)
    assert torch.all(first.gather(1, labels[:, None]).squeeze(1) == 0)


def test_context_observation_gate_requires_visual_and_nonpositional_signal() -> None:
    metrics = {
        "normal_mean_margin": 0.20,
        "normal_win_fraction": 0.80,
        "support_permuted_mean_margin": 0.08,
        "position_only_mean_margin": 0.09,
    }
    passed = observation_context_identity_gate(
        metrics,
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
    )
    assert passed["passed"] is True

    position_leak = observation_context_identity_gate(
        {**metrics, "position_only_mean_margin": 0.19},
        minimum_win_fraction=0.55,
        minimum_normal_margin=0.05,
        minimum_support_visual_gap=0.05,
        minimum_position_visual_gap=0.05,
    )
    assert position_leak["passed"] is False
    assert position_leak["checks"]["descriptor_over_position_gap"] is False


def test_query_grouped_means_ignores_inactive_rows_without_cross_query_weighting() -> None:
    values = _query_grouped_means(
        query_ids=["a", "a", "b", "b"],
        values=np.asarray([1.0, 3.0, 100.0, 5.0]),
        active=np.asarray([True, True, False, True]),
    )
    np.testing.assert_allclose(values, [2.0, 5.0])
    duplicate = candidate_slot_permutations_from_anchor_ids(
        anchor_ids=np.asarray([1, 1]), candidate_count=4, seed=3
    )
    torch.testing.assert_close(duplicate[0], duplicate[1])
    torch.testing.assert_close(
        torch.sort(duplicate, dim=1).values,
        torch.arange(4).expand(2, -1),
    )
