from __future__ import annotations

from feature_extract.tools.vfm.train_candidate_group_phase_identity_llr import (
    group_identity_gate_decision,
)


def _hard_metrics() -> dict[str, float]:
    return {
        "eligible_query_fraction": 1.0,
        "normal_win_fraction": 0.70,
        "normal_mean_positive_minus_negative": 0.14,
        "permuted_mean_positive_minus_negative": 0.02,
    }


def _rank_metrics() -> dict[str, float]:
    return {
        "base_top1": 0.40,
        "group_top1": 0.44,
        "base_mean_rank": 5.0,
        "group_mean_rank": 4.5,
        "base_correct_probability": 0.18,
        "group_correct_probability": 0.20,
        "observed_count": 100.0,
    }


def _gate(**changes: float) -> dict[str, object]:
    rank = {**_rank_metrics(), **changes}
    return group_identity_gate_decision(
        hard_metrics=_hard_metrics(),
        rank_metrics=rank,
        minimum_hard_eligible_query_fraction=0.90,
        minimum_hard_win_fraction=0.55,
        minimum_hard_gap=0.05,
        minimum_hard_visual_gap_delta=0.05,
        minimum_identity_top1_lift=0.02,
        minimum_identity_mean_rank_reduction=0.25,
        minimum_identity_correct_probability_lift=0.01,
    )


def test_gate_requires_both_visual_hard_evidence_and_identity_rank_lift() -> None:
    accepted = _gate()
    assert accepted["hard_repeat_passed"] is True
    assert accepted["posterior_rank_passed"] is True
    assert accepted["passed"] is True

    rejected = _gate(group_top1=0.41)
    assert rejected["hard_repeat_passed"] is True
    assert rejected["posterior_rank_passed"] is False
    assert rejected["passed"] is False


def test_gate_rejects_identity_lift_when_visual_control_fails() -> None:
    hard = {**_hard_metrics(), "permuted_mean_positive_minus_negative": 0.11}
    result = group_identity_gate_decision(
        hard_metrics=hard,
        rank_metrics=_rank_metrics(),
        minimum_hard_eligible_query_fraction=0.90,
        minimum_hard_win_fraction=0.55,
        minimum_hard_gap=0.05,
        minimum_hard_visual_gap_delta=0.05,
        minimum_identity_top1_lift=0.02,
        minimum_identity_mean_rank_reduction=0.25,
        minimum_identity_correct_probability_lift=0.01,
    )
    assert result["hard_repeat_passed"] is False
    assert result["posterior_rank_passed"] is True
    assert result["passed"] is False
