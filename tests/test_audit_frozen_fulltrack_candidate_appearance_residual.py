from __future__ import annotations

from feature_extract.tools.vfm.audit_frozen_fulltrack_candidate_appearance_residual import (
    fulltrack_residual_candidate_gate,
)


def _metrics(*, hard_lower: float = 0.6, hard_gap: float = 0.2):
    return {
        "exact_registered_identity": {
            "baseline": {
                "candidate_pair_average_precision": 0.4,
                "top1_positive_rate_given_positive": 0.4,
                "p90_first_positive_rank": 10.0,
            },
            "probe": {
                "candidate_pair_average_precision": 0.5,
                "top1_positive_rate_given_positive": 0.5,
                "p90_first_positive_rank": 8.0,
            },
            "paired_rank": {
                "rank_win_count": 20,
                "rank_loss_count": 4,
                "top1_rescue_count": 8,
                "top1_harm_count": 2,
            },
        },
        "exact_registered_rank2_to_l": {
            "baseline": {
                "median_first_positive_rank": 5.0,
                "p90_first_positive_rank": 12.0,
            },
            "probe": {
                "median_first_positive_rank": 2.0,
                "p90_first_positive_rank": 9.0,
            },
            "paired_rank": {
                "positive_row_count": 80,
                "rank_win_count": 30,
                "rank_loss_count": 3,
                "top1_rescue_count": 12,
                "top1_harm_count": 1,
            },
        },
        "rank2_to_top1_wrong_hard_pairs": {
            "baseline_pairwise": {"median_correct_minus_wrong": -0.5},
            "probe_pairwise": {
                "usable_pair_count": 80,
                "win_rate_wilson95_lower": hard_lower,
                "median_correct_minus_wrong": hard_gap,
            },
            "paired_rank": {
                "rank_win_count": 30,
                "rank_loss_count": 3,
                "top1_rescue_count": 12,
                "top1_harm_count": 1,
            },
        },
    }


def test_fulltrack_residual_gate_requires_hard_pair_evidence_not_only_rank_gain() -> None:
    assert fulltrack_residual_candidate_gate(_metrics())["passed"] is True
    rejected = fulltrack_residual_candidate_gate(_metrics(hard_lower=0.49))
    assert rejected["passed"] is False
    assert rejected["checks"]["hard_pair_wilson95_lower_above_chance"] is False
