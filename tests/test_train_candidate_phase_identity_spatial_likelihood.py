from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_candidate_phase_identity_spatial_likelihood import (
    _required_hard_source_ids,
    _strict_hard_repeat_objective_terms,
    hybrid_inner_gate_decision,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatQueryTargets,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        minimum_pose_win_fraction=0.55,
        minimum_pose_gap=0.05,
        minimum_both_visual_pose_delta=0.05,
        minimum_rgb_visual_pose_delta=0.005,
        minimum_hard_eligible_query_fraction=0.90,
        minimum_hard_win_fraction=0.55,
        minimum_hard_gap=0.05,
        minimum_both_visual_hard_delta=0.05,
        minimum_rgb_visual_hard_delta=0.005,
    )


def _metrics() -> dict[str, float]:
    return {
        "normal_pose_win_fraction": 0.70,
        "normal_pose_gap": 0.12,
        "normal_minus_both_deranged_pose_gap": 0.07,
        "normal_minus_rgb_deranged_pose_gap": 0.012,
        "hard_eligible_query_fraction": 1.0,
        "hard_win_fraction": 0.65,
        "hard_gap": 0.14,
        "hard_normal_minus_both_deranged_gap": 0.08,
        "hard_normal_minus_rgb_deranged_gap": 0.011,
    }


def test_hybrid_gate_requires_an_observable_rgb_contribution() -> None:
    accepted = hybrid_inner_gate_decision(metrics=_metrics(), args=_args())
    assert accepted["passed"] is True
    rejected = hybrid_inner_gate_decision(
        metrics={**_metrics(), "normal_minus_rgb_deranged_pose_gap": 0.0}, args=_args()
    )
    assert rejected["rgb_visual_pose"] is False
    assert rejected["passed"] is False


def _hard_targets(*, source_ids: list[int]) -> HardRepeatQueryTargets:
    count = len(source_ids)
    return HardRepeatQueryTargets(
        query_id="train/q0.png",
        source_point_ids=np.asarray(source_ids, dtype=np.int64),
        pair_ids=np.arange(count, dtype=np.int64),
        positive_candidate_indices=np.zeros((count,), dtype=np.int64),
        negative_candidate_indices=np.ones((count,), dtype=np.int64),
        positive_offsets_xy=np.zeros((count, 2), dtype=np.float32),
        negative_offsets_xy=np.ones((count, 2), dtype=np.float32),
    )


def test_hybrid_training_selects_the_union_of_current_and_static_hard_sources() -> None:
    current = _hard_targets(source_ids=[17, 5, 17])
    static = _hard_targets(source_ids=[9, 5])
    selected = _required_hard_source_ids(current, static)
    assert selected is not None
    np.testing.assert_array_equal(selected, np.asarray([5, 9, 17], dtype=np.int64))
    assert _required_hard_source_ids() is None


def test_hybrid_hard_objective_requires_common_rgb_counterfactual_evidence() -> None:
    normal = torch.tensor([0.30, 0.80], requires_grad=True)
    both = torch.tensor([0.05, 0.10], requires_grad=True)
    rgb = torch.tensor([0.15, 0.20], requires_grad=True)
    usable = torch.tensor([True, True])
    rgb_usable = torch.tensor([True, False])
    hard_loss, both_control, rgb_control, gap, active = _strict_hard_repeat_objective_terms(
        normal_values=normal,
        normal_usable=usable,
        both_deranged_values=both,
        both_deranged_usable=usable,
        rgb_deranged_values=rgb,
        rgb_deranged_usable=rgb_usable,
        hard_margin=0.20,
        appearance_control_margin=0.05,
        anchor=normal,
    )
    (hard_loss + both_control + rgb_control).backward()
    assert active == pytest.approx(1.0)
    assert gap.item() == pytest.approx(0.30)
    assert normal.grad is not None
    assert both.grad is not None
    assert rgb.grad is not None
