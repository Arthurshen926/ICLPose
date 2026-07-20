from __future__ import annotations

import pytest

from feature_extract.tools.vfm.audit_frozen_fulltrack_candidate_appearance_residual import (
    fulltrack_residual_candidate_gate,
)
from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_candidate_probe import (
    _validate_raw_top4_evidence_contract,
    fulltrack_per_view_candidate_gate,
    parse_args,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
)


def _metrics() -> dict[str, object]:
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
                "win_rate_wilson95_lower": 0.6,
                "median_correct_minus_wrong": 0.2,
            },
            "paired_rank": {
                "rank_win_count": 30,
                "rank_loss_count": 3,
                "top1_rescue_count": 12,
                "top1_harm_count": 1,
            },
        },
    }


def test_per_view_gate_keeps_the_strict_candidate_checks() -> None:
    metrics = _metrics()
    expected = fulltrack_residual_candidate_gate(metrics)
    actual = fulltrack_per_view_candidate_gate(metrics)
    assert actual["passed"] is True
    assert actual["checks"] == expected["checks"]
    assert "frozen per-view candidate-evidence" in str(actual["policy"])


def test_train_audit_split_requires_an_explicit_nondefault_choice() -> None:
    common = [
        "--appearance-artifacts",
        "appearance.npz",
        "--predictions",
        "predictions.npz",
        "--colmap-model-dir",
        "model",
        "--projected-landmark-bank",
        "bank.npz",
        "--output-dir",
        "out",
    ]
    assert parse_args(common).audit_split == "validation"
    assert parse_args([*common, "--audit-split", "train"]).audit_split == "train"


def test_bounded_raw_top4_contract_requires_a_real_cap_and_provenance() -> None:
    contract = {
        "postfit_residual_scale": 0.6,
        "postfit_scale_provenance": "train-only calibration",
        "raw_topk_aggregation": "uniform_mean",
        "postfit_residual_cap": 3.0,
        "postfit_cap_provenance": "train-only cap selection",
    }
    _validate_raw_top4_evidence_contract(
        architecture=RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        raw_topk_aggregation="uniform_mean",
        contract=contract,
    )
    for key, value in (
        ("postfit_residual_cap", None),
        ("postfit_residual_cap", 0.0),
        ("postfit_cap_provenance", ""),
    ):
        invalid = {**contract, key: value}
        with pytest.raises(ValueError, match="cap contract"):
            _validate_raw_top4_evidence_contract(
                architecture=RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
                raw_topk_aggregation="uniform_mean",
                contract=invalid,
            )
