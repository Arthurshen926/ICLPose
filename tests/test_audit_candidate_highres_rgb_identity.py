from __future__ import annotations

import pytest
import torch

from feature_extract.tools.vfm.audit_candidate_highres_rgb_identity import (
    candidate_identity_rank_gate,
    candidate_identity_rank_metrics,
    stable_candidate_ranks,
)


def test_stable_candidate_ranks_use_candidate_slot_only_as_tie_break() -> None:
    logits = torch.tensor([[0.5, 0.5, 0.1], [0.3, 0.2, 0.3]])
    supported = torch.ones_like(logits, dtype=torch.bool)
    ranks = stable_candidate_ranks(
        candidate_logits=logits,
        target_candidate_indices=torch.tensor([1, 2]),
        candidate_supported=supported,
    )
    # Equal scores are deterministic for auditing, but the slot index is never
    # supplied to the RGB model as a feature.
    torch.testing.assert_close(ranks, torch.tensor([2, 2]))


def test_candidate_identity_metrics_keep_explicit_null_out_of_track_rank() -> None:
    logits = torch.log(torch.tensor([[0.40, 0.20, 0.10, 0.30], [0.10, 0.60, 0.10, 0.20]]))
    metrics = candidate_identity_rank_metrics(
        candidate_null_logits=logits,
        target_candidate_indices=torch.tensor([0, 1]),
        candidate_supported=torch.ones((2, 3), dtype=torch.bool),
    )
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["mean_rank"] == pytest.approx(1.0)
    assert metrics["correct_probability"] == pytest.approx(0.50)


def test_rank_gate_requires_real_visual_gain_over_support_permutation() -> None:
    base = {"observed_count": 20.0, "top1": 0.30, "mean_rank": 5.0, "correct_probability": 0.12}
    normal = {"observed_count": 20.0, "top1": 0.38, "mean_rank": 4.2, "correct_probability": 0.17}
    permuted = {"observed_count": 20.0, "top1": 0.31, "mean_rank": 4.9, "correct_probability": 0.125}
    thresholds = {
        "minimum_eligible_fraction": 0.9,
        "minimum_top1_lift": 0.02,
        "minimum_mean_rank_reduction": 0.25,
        "minimum_correct_probability_lift": 0.01,
        "minimum_control_top1_delta": 0.01,
        "minimum_control_rank_reduction_delta": 0.10,
        "minimum_control_correct_probability_delta": 0.005,
    }
    passed = candidate_identity_rank_gate(
        base=base,
        normal=normal,
        permuted=permuted,
        eligible_fraction=0.95,
        thresholds=thresholds,
    )
    assert passed["passed"] is True
    rejected = candidate_identity_rank_gate(
        base=base,
        normal=normal,
        permuted=normal,
        eligible_fraction=0.95,
        thresholds=thresholds,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["control_top1"] is False
