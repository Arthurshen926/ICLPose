from feature_extract.tools.vfm.summarize_goal_maplet_g20 import (
    _continuous_gate,
    _discrete_gate,
)


def test_discrete_gate_requires_tail_success_and_catastrophe_non_regression():
    baseline = {
        "translation_p90_m": 1.0,
        "strict_0.5m_5deg": 0.5,
        "within_1m_10deg": 0.8,
        "catastrophic_rate": 0.1,
    }
    improved = {
        "translation_p90_m": 0.9,
        "strict_0.5m_5deg": 0.6,
        "within_1m_10deg": 0.9,
        "catastrophic_rate": 0.0,
    }
    report = {
        "selected_variant": "full",
        "map_query_overlap_audit": {"map_query_image_disjoint": True},
        "variants": {
            "identity_only": {"outer_loto_metrics": baseline},
            "full": {"outer_loto_metrics": improved},
        },
    }
    decision = _discrete_gate(report)
    assert decision["self_map_query_trajectory_loto_diagnostic_pass"]
    assert decision["map_disjoint_cross_acquisition_pass"]
    assert not decision["production_promotion_allowed"]


def test_continuous_gate_identifies_scale_as_normal_complement():
    axes = ("tangent1", "tangent2", "normal", "roll", "pitch", "yaw")
    summary = {axis: {"gt_strict_local_max_fraction": 1.0} for axis in axes}
    summary["gate"] = {
        "translation_0.25m_local_max_fraction": 1.0,
        "translation_0.5m_local_max_fraction": 1.0,
        "rotation_3deg_local_max_fraction": 1.0,
    }
    component = {
        "jacobian_phase_visible": {
            "normal": {"gt_strict_local_max_fraction": 0.2},
        },
        "jacobian_log_scale_agreement": {
            "normal": {"gt_strict_local_max_fraction": 0.7},
        },
    }
    decision = _continuous_gate({
        "cross_acquisition_summary": summary,
        "cross_acquisition_component_summary": component,
    }, map_query_image_disjoint=True)
    assert decision["g21_continuous_refinement_open"]
    assert decision["normal_evidence_diagnosis"] == "vfm_gradient_scale_is_complementary"


def test_map_overlap_blocks_cross_acquisition_and_g21_promotion():
    baseline = {
        "translation_p90_m": 1.0,
        "strict_0.5m_5deg": 0.5,
        "within_1m_10deg": 0.8,
        "catastrophic_rate": 0.1,
    }
    report = {
        "selected_variant": "full",
        "map_query_overlap_audit": {"map_query_image_disjoint": False},
        "variants": {
            "identity_only": {"outer_loto_metrics": baseline},
            "full": {"outer_loto_metrics": baseline},
        },
    }
    decision = _discrete_gate(report)
    assert decision["self_map_query_trajectory_loto_diagnostic_pass"]
    assert not decision["map_disjoint_cross_acquisition_pass"]

    axes = ("tangent1", "tangent2", "normal", "roll", "pitch", "yaw")
    summary = {axis: {"gt_strict_local_max_fraction": 1.0} for axis in axes}
    summary["gate"] = {
        "translation_0.25m_local_max_fraction": 1.0,
        "translation_0.5m_local_max_fraction": 1.0,
        "rotation_3deg_local_max_fraction": 1.0,
    }
    component = {
        name: {"normal": {"gt_strict_local_max_fraction": 1.0}}
        for name in ("jacobian_phase_visible", "jacobian_log_scale_agreement")
    }
    basin = _continuous_gate({
        "cross_acquisition_summary": summary,
        "cross_acquisition_component_summary": component,
    })
    assert basin["self_map_basin_gate_pass"]
    assert not basin["g21_continuous_refinement_open"]
