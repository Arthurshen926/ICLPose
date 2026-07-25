from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_multiscale_candidate_probe import (
    _BIDIRECTIONAL_VISUAL_CONTROL_FAMILIES,
    _allowed_prediction_soft_global_context,
    _paired_rank_audit,
    _per_scale_visual_control_diagnostic,
    _rank2_to_l_rescue_audit,
    _set_valued_metrics,
    _visual_vs_position_control_pre_gate,
)


def test_v2_global_prediction_protocol_must_remain_candidate_conditioned() -> None:
    allowed = {
        "source_feature_protocol": {
            "whole_image_summary_or_global_used": True,
            "soft_global_context_factor_used": True,
            "candidate_conditioned_full_image_region_tokens": True,
            "global_context_hard_retrieval_or_candidate_reselection": False,
            "image_retrieval_or_submap_used": False,
        }
    }
    disallowed = {
        "source_feature_protocol": {
            **allowed["source_feature_protocol"],
            "global_context_hard_retrieval_or_candidate_reselection": True,
        }
    }

    assert _allowed_prediction_soft_global_context(allowed)
    assert not _allowed_prediction_soft_global_context(disallowed)


def test_paired_candidate_gate_declares_the_raw_v3_visual_control_profile() -> None:
    assert (
        "bidirectional_absolute_raw_v3",
        "bidirectional_absolute_raw_visual_v3",
        "bidirectional_absolute_raw_position_control_v3",
    ) in _BIDIRECTIONAL_VISUAL_CONTROL_FAMILIES
    assert (
        "bidirectional_absolute_raw_layout_v4",
        "bidirectional_absolute_raw_layout_visual_v4",
        "bidirectional_absolute_raw_layout_position_control_v4",
    ) in _BIDIRECTIONAL_VISUAL_CONTROL_FAMILIES
    assert (
        "bidirectional_absolute_dual_head_raw_layout_v5",
        "bidirectional_absolute_dual_head_raw_layout_visual_v5",
        "bidirectional_absolute_dual_head_raw_layout_position_control_v5",
    ) in _BIDIRECTIONAL_VISUAL_CONTROL_FAMILIES


def test_set_valued_metrics_credit_all_geometric_positives_and_null() -> None:
    probability = np.asarray(
        [[0.35, 0.35, 0.0], [0.20, 0.10, 0.0]], dtype=np.float64
    )
    null = np.asarray([0.30, 0.70], dtype=np.float64)
    labels = np.asarray([[True, True, False], [False, False, False]])
    valid = np.asarray([[True, True, False], [True, True, False]])
    metrics = _set_valued_metrics(
        probability,
        null,
        labels=labels,
        valid=valid,
        row_mask=np.asarray([True, True]),
    )
    assert metrics["positive_row_count"] == 1
    assert metrics["positive_candidate_count"] == 2
    assert metrics["group_argmax_correct_rate"] == 1.0
    assert np.isclose(metrics["group_target_nll"], -np.log(0.70))


def test_paired_rank_audit_counts_rescue_and_harm() -> None:
    baseline = np.asarray([[0.2, 0.7], [0.8, 0.1]], dtype=np.float64)
    probe = np.asarray([[0.8, 0.1], [0.1, 0.7]], dtype=np.float64)
    labels = np.asarray([[True, False], [True, False]])
    valid = np.ones_like(labels, dtype=bool)
    paired = _paired_rank_audit(
        baseline,
        probe,
        labels=labels,
        valid=valid,
        row_mask=np.asarray([True, True]),
    )
    assert paired["positive_row_count"] == 2
    assert paired["rank_win_count"] == 1
    assert paired["rank_loss_count"] == 1
    assert paired["top1_rescue_count"] == 1
    assert paired["top1_harm_count"] == 1


def test_rank2_to_l_rescue_audit_is_limited_to_baseline_missed_positives() -> None:
    baseline = np.asarray([[0.2, 0.7], [0.8, 0.1]], dtype=np.float64)
    probe = np.asarray([[0.8, 0.1], [0.7, 0.2]], dtype=np.float64)
    labels = np.asarray([[True, False], [True, False]])
    report = _rank2_to_l_rescue_audit(
        baseline,
        probe,
        labels=labels,
        valid=np.ones_like(labels, dtype=bool),
        row_mask=np.asarray([True, True]),
    )
    assert report["eligible_positive_row_count"] == 1
    assert report["baseline"]["median_first_positive_rank"] == 2.0
    assert report["probe"]["median_first_positive_rank"] == 1.0
    assert report["paired_rank"]["top1_rescue_count"] == 1


def test_visual_vs_position_control_gate_requires_visual_rank_rescues() -> None:
    position = np.asarray([[0.8, 0.1], [0.7, 0.2]], dtype=np.float64)
    visual = np.asarray([[0.1, 0.8], [0.1, 0.8]], dtype=np.float64)
    labels = np.asarray([[False, True], [False, True]])
    report = _visual_vs_position_control_pre_gate(
        position_probability=position,
        position_null=np.asarray([0.1, 0.1]),
        visual_probability=visual,
        visual_null=np.asarray([0.1, 0.1]),
        geometry_labels=labels,
        exact_labels=labels,
        exact_supervised=np.asarray([True, True]),
        candidate_valid=np.ones_like(labels, dtype=bool),
    )

    gate = report["candidate_pre_gate"]
    assert gate["geometry_nll_improved"]
    assert gate["rank2_to_l_wins_exceed_losses"]
    assert gate["passed"]


def test_per_scale_diagnostic_keeps_a_common_base_null_reference() -> None:
    # Shape: query row, candidate, support view, frozen source scale.
    visual = np.zeros((2, 2, 1, 2), dtype=np.float64)
    visual[:, 1, 0, 0] = 3.0
    position = np.zeros_like(visual)
    labels = np.asarray([[False, True], [False, True]])
    report = _per_scale_visual_control_diagnostic(
        visual_per_scale_view_logits=visual,
        position_per_scale_view_logits=position,
        scale_names=("radio_final", "alike"),
        base_candidate=np.asarray([[0.45, 0.45], [0.45, 0.45]]),
        base_null=np.asarray([0.10, 0.10]),
        view_valid=np.ones((2, 2, 1), dtype=bool),
        geometry_labels=labels,
        exact_labels=labels,
        exact_supervised=np.asarray([True, True]),
        candidate_valid=np.ones_like(labels, dtype=bool),
    )

    assert report["diagnostic_only"]
    assert report["promotion_allowed"] is False
    assert (
        report["null_reference"]
        == "immutable_base_null_only_no_learned_joint_null_head"
    )
    assert report["combinations"]["radio_final"]["candidate_pre_gate"]["passed"]


def test_dual_head_audit_uses_identity_probabilities_only_for_exact_metrics() -> None:
    position_geometry = np.asarray([[0.15, 0.75], [0.15, 0.75]], dtype=np.float64)
    visual_geometry = np.asarray([[0.75, 0.15], [0.75, 0.15]], dtype=np.float64)
    # The geometry semantics intentionally prefer candidate 0.  The strict
    # SfM identity side head instead distinguishes candidate 1, proving that
    # exact-track metrics cannot silently reuse geometry logits in V5.
    position_identity = np.asarray([[0.75, 0.15], [0.75, 0.15]], dtype=np.float64)
    visual_identity = np.asarray([[0.15, 0.75], [0.15, 0.75]], dtype=np.float64)
    geometry_labels = np.asarray([[True, False], [True, False]])
    exact_labels = np.asarray([[False, True], [False, True]])
    report = _visual_vs_position_control_pre_gate(
        position_probability=position_geometry,
        position_null=np.asarray([0.10, 0.10]),
        visual_probability=visual_geometry,
        visual_null=np.asarray([0.10, 0.10]),
        geometry_labels=geometry_labels,
        exact_labels=exact_labels,
        exact_supervised=np.asarray([True, True]),
        candidate_valid=np.ones_like(geometry_labels, dtype=bool),
        position_identity_probability=position_identity,
        position_identity_null=np.asarray([0.10, 0.10]),
        visual_identity_probability=visual_identity,
        visual_identity_null=np.asarray([0.10, 0.10]),
    )

    assert report["exact_identity_probability_source"] == "separate_identity_side_head"
    assert (
        report["visual"]["exact_registered_identity"]["top1_exact_rate_given_retrieved"]
        == 1.0
    )
    assert (
        report["position_control"]["exact_registered_identity"]["top1_exact_rate_given_retrieved"]
        == 0.0
    )
