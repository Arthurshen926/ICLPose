from __future__ import annotations

import pytest

from feature_extract.tools.vfm.evaluate_multiscale_candidate_probe_gate import (
    evaluate_multiscale_candidate_probe_gate,
)


def _audit(*, test_top1_delta: float = 0.01) -> dict[str, object]:
    def split() -> dict[str, object]:
        return {
            "geometry_set": {
                "baseline": {
                    "candidate_pair_average_precision": 0.2,
                    "group_target_nll": 1.0,
                    "top1_geometry_valid_rate_given_positive": 0.4,
                },
                "probe": {
                    "candidate_pair_average_precision": 0.25,
                    "group_target_nll": 0.9,
                    "top1_geometry_valid_rate_given_positive": 0.4 + test_top1_delta,
                },
                "paired_rank": {"rank_win_count": 8, "rank_loss_count": 4},
            },
            "exact_registered_identity": {
                "baseline": {
                    "exact_candidate_pair_average_precision": 0.3,
                    "top1_exact_rate_given_retrieved": 0.4,
                    "median_exact_rank_when_retrieved": 2.0,
                    "p90_exact_rank_when_retrieved": 5.0,
                },
                "probe": {
                    "exact_candidate_pair_average_precision": 0.35,
                    "top1_exact_rate_given_retrieved": 0.45,
                    "median_exact_rank_when_retrieved": 2.0,
                    "p90_exact_rank_when_retrieved": 4.0,
                },
                "paired_rank": {"rank_win_count": 5, "rank_loss_count": 4},
            },
        }

    return {
        "protocol": {
            "fit_uses_train_geometric_targets_only": True,
            "prediction_artifact_frozen_before_validation_test_label_join": True,
            "test_used_for_model_selection": False,
            "image_retrieval_or_submap_used": False,
            "whole_image_summary_or_global_used": False,
        },
        "families": {"primary": {"splits": {"validation": split(), "test": split()}}},
    }


def test_cross_split_gate_requires_all_predeclared_nonregression_checks() -> None:
    passed = evaluate_multiscale_candidate_probe_gate(_audit(), family="primary")
    assert passed["pass"] is True
    assert passed["next_action"] == "run_fixed_hypothesis_pose_rank_gate"

    failed = evaluate_multiscale_candidate_probe_gate(
        _audit(test_top1_delta=-0.01), family="primary"
    )
    assert failed["pass"] is False
    assert failed["splits"]["test"]["checks"]["geometry_top1_not_worse"] is False


def test_cross_split_gate_accepts_train_only_registered_identity_supervision() -> None:
    audit = _audit()
    audit["protocol"]["fit_uses_train_geometric_targets_only"] = False
    audit["protocol"]["fit_uses_train_targets_only"] = True
    audit["protocol"]["fit_uses_train_registered_identity_targets_only"] = True
    audit["protocol"]["training_supervision_mode"] = "registered_track_identity"
    audit["protocol"]["prediction_probability_semantics"] = (
        "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
    )

    result = evaluate_multiscale_candidate_probe_gate(
        audit, family="primary", gate_mode="identity_only"
    )

    assert result["pass"] is True
    assert result["protocol_checks"]["train_only_fit"] is True
    assert result["protocol_checks"]["registered_identity_train_only"] is True
    assert result["next_action"] == "run_fixed_hypothesis_pose_rank_gate_with_identity_prior"


def test_cross_split_gate_requires_context_only_attribution_control_when_requested() -> None:
    audit = _audit()
    context_family = "structured_candidate_specific_context_only"
    audit["families"][context_family] = audit["families"]["primary"]
    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family="primary",
        candidate_specific_family=context_family,
    )
    assert passed["pass"] is True
    assert passed["candidate_specific_attribution_control"]["family"] == context_family

    failing = _audit(test_top1_delta=-0.01)
    audit["families"][context_family] = failing["families"]["primary"]
    blocked = evaluate_multiscale_candidate_probe_gate(
        audit,
        family="primary",
        candidate_specific_family=context_family,
    )
    assert blocked["pass"] is False
    assert blocked["candidate_specific_attribution_control"]["pass"] is False


def test_cross_split_gate_accepts_predeclared_cost_volume_context_control() -> None:
    audit = _audit()
    context_family = "cost_volume_candidate_specific_context_only"
    audit["families"][context_family] = audit["families"]["primary"]
    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family="primary",
        candidate_specific_family=context_family,
    )
    assert passed["pass"] is True
    assert passed["candidate_specific_attribution_control"]["family"] == context_family


def test_cross_split_gate_accepts_final_only_cost_volume_context_control() -> None:
    audit = _audit()
    context_family = "cost_volume_radio_final_context_only"
    audit["families"][context_family] = audit["families"]["primary"]
    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family="primary",
        candidate_specific_family=context_family,
    )
    assert passed["pass"] is True


def test_cross_split_gate_requires_the_matching_context_only_control_for_composites() -> None:
    audit = _audit()
    primary = "wide_fullcorr_radio_final_large_context"
    control = "wide_fullcorr_radio_final_context_only"
    audit["families"][primary] = audit["families"]["primary"]
    audit["families"][control] = audit["families"]["primary"]

    with pytest.raises(ValueError, match="requires the predeclared"):
        evaluate_multiscale_candidate_probe_gate(audit, family=primary)

    with pytest.raises(ValueError, match="requires attribution control"):
        evaluate_multiscale_candidate_probe_gate(
            audit,
            family=primary,
            candidate_specific_family="wide_fullcorr_candidate_specific_context_only",
        )

    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family=primary,
        candidate_specific_family=control,
    )
    assert passed["pass"] is True
    assert passed["required_candidate_specific_attribution_control"] == control


def test_cross_split_gate_accepts_wide_full_correlation_context_control() -> None:
    audit = _audit()
    context_family = "wide_fullcorr_candidate_specific_context_only"
    audit["families"][context_family] = audit["families"]["primary"]
    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family="primary",
        candidate_specific_family=context_family,
    )
    assert passed["pass"] is True
    assert passed["candidate_specific_attribution_control"]["family"] == context_family


def test_cross_split_gate_allows_only_explicit_soft_global_context_factor() -> None:
    audit = _audit()
    audit["protocol"].update(
        {
            "whole_image_summary_or_global_used": True,
            "soft_global_context_factor_used": True,
            "global_context_hard_retrieval_or_candidate_reselection": False,
        }
    )
    context_family = "global_context_candidate_specific_context_only"
    audit["families"][context_family] = audit["families"]["primary"]
    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family="primary",
        candidate_specific_family=context_family,
    )
    assert passed["pass"] is True
    assert passed["protocol_checks"]["no_retrieval_or_global_summary"] is True

    audit["protocol"]["global_context_hard_retrieval_or_candidate_reselection"] = True
    blocked = evaluate_multiscale_candidate_probe_gate(
        audit,
        family="primary",
        candidate_specific_family=context_family,
    )
    assert blocked["pass"] is False
    assert blocked["protocol_checks"]["no_retrieval_or_global_summary"] is False


def test_cross_split_gate_rejects_an_anchor_control_masquerading_as_context() -> None:
    with pytest.raises(ValueError, match="candidate-specific attribution control"):
        evaluate_multiscale_candidate_probe_gate(
            _audit(),
            family="primary",
            candidate_specific_family="structured_existing_anchor_control",
        )


def test_context_attention_primary_requires_its_masked_context_control() -> None:
    audit = _audit()
    primary = "context_attention_multiscale_with_anchor"
    control = "context_attention_multiscale_context_only"
    audit["families"][primary] = audit["families"]["primary"]
    audit["families"][control] = audit["families"]["primary"]

    with pytest.raises(ValueError, match="requires the predeclared"):
        evaluate_multiscale_candidate_probe_gate(audit, family=primary)

    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family=primary,
        candidate_specific_family=control,
    )
    assert passed["pass"] is True
    assert passed["required_candidate_specific_attribution_control"] == control


def test_absolute_global_transport_primary_requires_visual_context_control() -> None:
    audit = _audit()
    primary = "absolute_global_transport_with_anchor"
    control = "absolute_global_transport_context_only"
    audit["families"][primary] = audit["families"]["primary"]
    audit["families"][control] = audit["families"]["primary"]

    with pytest.raises(ValueError, match="requires the predeclared"):
        evaluate_multiscale_candidate_probe_gate(audit, family=primary)

    passed = evaluate_multiscale_candidate_probe_gate(
        audit,
        family=primary,
        candidate_specific_family=control,
    )
    assert passed["pass"] is True
    assert passed["required_candidate_specific_attribution_control"] == control
