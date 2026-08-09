import copy
from types import SimpleNamespace

import numpy as np

from feature_extract.tools.vfm.audit_goal_maplet_phase_basin import (
    _phase_evidence_for_policy,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_phase_operator_crossfit import (
    _compare_rows,
)
from feature_extract.tools.vfm.rebind_goal_maplet_phase_policy import (
    _rebound_metadata,
)
from feature_extract.tools.vfm.merge_goal_maplet_phase_operator_crossfit import (
    _operator_decision,
)
from feature_extract.tools.vfm.merge_goal_maplet_map_crossfit_basin import (
    _merge_reports,
    _validate_crossfit_lineage,
)
from feature_extract.tools.vfm.create_goal_maplet_map_crossfit_phase_policy import (
    _policy,
)


def _candidate(x, translation, rotation, score):
    return {
        "pose_w2c": [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "translation_m": translation,
        "rotation_deg": rotation,
        "surface_alignment_score": score,
    }


def _report(rows):
    return {"rows": [
        {
            "image_id": image_id,
            "mode_details": {"actual_parent_actual_child": details},
        }
        for image_id, details in rows
    ]}


def test_rebind_changes_only_map_lineage_and_records_frozen_operator():
    source = {
        "artifact_type": "goal_maplet_phase_readout_policy_v1",
        "component_names": ["horizontal_phase", "vertical_phase"],
        "coefficient": [1.25, 0.35],
        "standardizer_scale": [0.04, 0.06],
        "physical_map_sha256": "old-map",
        "canonical_field_sha256": "old-field",
        "physical_instance_readout_sha256": "old-readout",
    }
    rebound = _rebound_metadata(
        source,
        source_sha256="source",
        physical_map_sha256="new-map",
        canonical_field_sha256="new-field",
        physical_instance_readout_sha256="new-readout",
    )
    assert rebound["coefficient"] == source["coefficient"]
    assert rebound["standardizer_scale"] == source["standardizer_scale"]
    assert rebound["operator_parameters_refit"] is False
    assert rebound["map_crossfit_lineage_rebind"] is True
    assert rebound["source_map_lineage"]["canonical_field_sha256"] == "old-field"
    assert rebound["canonical_field_sha256"] == "new-field"


def test_phase_comparison_aligns_candidate_poses_and_reports_complementarity():
    good = _candidate(0.0, 0.2, 1.0, 0.9)
    medium = _candidate(1.0, 0.7, 2.0, 0.4)
    bad = _candidate(2.0, 1.5, 4.0, 0.1)
    directional = _report([
        ("seq12/a.png", [good, medium, bad]),
        ("seq12/b.png", [medium, good, bad]),
    ])
    jacobian = _report([
        ("seq12/a.png", [copy.deepcopy(medium), copy.deepcopy(good), copy.deepcopy(bad)]),
        ("seq12/b.png", [copy.deepcopy(good), copy.deepcopy(medium), copy.deepcopy(bad)]),
    ])
    # Give the reordered Jacobian candidates scores consistent with their list.
    jacobian["rows"][0]["mode_details"]["actual_parent_actual_child"][0][
        "surface_alignment_score"
    ] = 1.0
    jacobian["rows"][0]["mode_details"]["actual_parent_actual_child"][1][
        "surface_alignment_score"
    ] = 0.5
    result = _compare_rows(directional, jacobian)
    complement = result["top1_complementarity"]
    assert complement["directional_right_jacobian_wrong"] == 1
    assert complement["jacobian_right_directional_wrong"] == 1
    assert complement["query_count"] == 2
    assert result["metrics"]["directional"]["strict_0.5m_5deg"] == 0.5
    assert result["metrics"]["jacobian"]["strict_0.5m_5deg"] == 0.5
    assert result["pose_quality_pairwise_concordance"]["directional"]["total"] > 0
    headroom = result["candidate_coverage_and_ranking_headroom"]
    assert headroom["strict"]["phase_evaluated_oracle_rate"] == 1.0
    assert headroom["strict"]["phase_ranking_miss_count"] == 1


def test_basin_dispatch_computes_directional_and_jacobian_operators_separately():
    query = np.zeros((2, 4, 4), dtype=np.float32)
    query[0, :, 1:] = 1.0
    query[1, 1:, :] = 1.0
    rendered = query.copy()
    valid = np.ones((4, 4), dtype=bool)
    directional = _phase_evidence_for_policy(
        query,
        rendered,
        valid,
        SimpleNamespace(metadata={"artifact_type": "goal_maplet_phase_readout_policy_v1"}),
    )
    jacobian = _phase_evidence_for_policy(
        query,
        rendered,
        valid,
        SimpleNamespace(metadata={"artifact_type": "goal_maplet_phase_readout_policy_v2"}),
    )
    assert directional.horizontal_phase > 0.0
    assert directional.vertical_phase > 0.0
    assert directional.jacobian_phase_visible > 0.0
    assert jacobian.horizontal_phase == 0.0
    assert jacobian.vertical_phase == 0.0
    assert jacobian.jacobian_phase_visible > 0.0


def test_phase_operator_decision_requires_foldwise_tail_noninferiority():
    fold_a = {"metrics": {
        "directional": {"strict_0.5m_5deg": 0.8, "within_1m_10deg": 1.0,
                        "catastrophic_rate": 0.0},
        "jacobian": {"strict_0.5m_5deg": 0.9, "within_1m_10deg": 1.0,
                     "catastrophic_rate": 0.0},
    }}
    fold_b = {"metrics": {
        "directional": {"strict_0.5m_5deg": 0.9, "within_1m_10deg": 1.0,
                        "catastrophic_rate": 0.0},
        "jacobian": {"strict_0.5m_5deg": 0.8, "within_1m_10deg": 0.9,
                     "catastrophic_rate": 0.0},
    }}
    decision = _operator_decision([fold_a, fold_b])
    assert decision["selected_phase_operator"] == "directional"
    assert not decision["directional_foldwise_noninferior"]
    assert not decision["jacobian_foldwise_noninferior"]
    assert decision["phase_combination_training_allowed"] is False


def test_crossfit_phase_policies_are_parameter_free_and_keep_observability_out_of_score():
    directional = _policy(
        "directional", physical_map_sha256="map",
        canonical_field_sha256="field",
        physical_instance_readout_sha256="readout",
    )
    jacobian = _policy(
        "jacobian", physical_map_sha256="map",
        canonical_field_sha256="field",
        physical_instance_readout_sha256="readout",
    )
    assert directional["coefficient"] == [0.5, 0.5]
    assert directional["standardizer_scale"] == [1.0, 1.0]
    assert jacobian["component_names"] == ["jacobian_phase_visible"]
    assert "jacobian_observability" not in jacobian["component_names"]
    assert directional["operator_parameters_refit"] is False
    assert jacobian["null_or_abstention_included"] is False
    assert directional["feature_pipeline_crossfit_binding"] is True


def _basin_report(trajectory, field):
    magnitudes_t = [0.1, 0.25, 0.5]
    magnitudes_r = [1.0, 3.0]
    image_id = f"{trajectory}/a.png"
    ground_truth = [{
        "image_id": image_id,
        "phase": {
            "jacobian_phase_visible": 1.0,
            "jacobian_log_scale_agreement": 1.0,
            "jacobian_observability": 1.0,
        },
    }]
    rows = []
    for axis in ("tangent1", "tangent2", "normal"):
        for magnitude in magnitudes_t:
            for sign in (-1, 1):
                score = 1.0 - magnitude
                rows.append({
                    "image_id": image_id, "axis": axis, "magnitude": magnitude,
                    "sign": sign, "gt_score": 1.0, "score": score,
                    "phase": {
                        "jacobian_phase_visible": score,
                        "jacobian_log_scale_agreement": score,
                        "jacobian_observability": score,
                    },
                })
    for axis in ("roll", "pitch", "yaw"):
        for magnitude in magnitudes_r:
            for sign in (-1, 1):
                score = 1.0 - magnitude / 10.0
                rows.append({
                    "image_id": image_id, "axis": axis, "magnitude": magnitude,
                    "sign": sign, "gt_score": 1.0, "score": score,
                    "phase": {
                        "jacobian_phase_visible": score,
                        "jacobian_log_scale_agreement": score,
                        "jacobian_observability": score,
                    },
                })
    return {
        "render_supersample_factor": 2,
        "physical_map_sha256": "map",
        "canonical_field_sha256": field,
        "phase_readout_artifact_type": "goal_maplet_phase_readout_policy_v2",
        "translation_magnitudes": magnitudes_t,
        "rotation_magnitudes_deg": magnitudes_r,
        "ground_truth": ground_truth,
        "rows": rows,
    }


def test_basin_merge_recomputes_exact_overall_and_per_trajectory_metrics():
    merged = _merge_reports([
        _basin_report("seq12", "field12"),
        _basin_report("seq14", "field14"),
    ])
    assert merged["query_count"] == 2
    assert set(merged["per_trajectory"]) == {"seq12", "seq14"}
    assert merged["summary"]["gate"]["translation_0.25m_local_max_fraction"] == 1.0
    assert merged["summary"]["normal"]["fully_monotonic_ray_fraction"] == 1.0


def test_basin_merge_requires_matching_audited_feature_crossfit_fold():
    report = _basin_report("seq12", "field12")
    evaluation = {
        "stage": "goal_maplet_phase_operator_outer_feature_crossfit_evaluation",
        "feature_pipeline_outer_crossfit": True,
        "geometry_was_outer_crossfit": False,
        "heldout_trajectories": ["seq12"],
        "candidate_generator_contract": {
            "canonical_field_sha256": "field12",
            "physical_map_sha256": "map",
        },
        "query_results": [{"image_id": "seq12/a.png"}],
    }
    assert _validate_crossfit_lineage([report], [evaluation]) is False
    invalid = copy.deepcopy(evaluation)
    invalid["query_results"] = [{"image_id": "seq12/other.png"}]
    with np.testing.assert_raises(ValueError):
        _validate_crossfit_lineage([report], [invalid])
