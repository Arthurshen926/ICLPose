import numpy as np
import json

from feature_extract.vfm.postselection_success_calibration import (
    FEATURE_NAMES,
    candidate_source_postselection_evidence,
    postselection_feature_row,
    predict_sigmoid_logistic_payload,
    typed_null_diagnostics,
)
from feature_extract.tools.vfm.calibrate_goal_maplet_postselection_success_oof import (
    _nested_risk_threshold_evaluation,
    _select_regularization_nested,
    _select_risk_threshold,
    main as calibrate_main,
)
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def test_risk_threshold_maximizes_tie_safe_coverage_under_wilson_constraint():
    probability = np.asarray([0.95] * 10 + [0.7] * 10 + [0.2] * 10)
    label = np.asarray([1] * 10 + [1] * 8 + [0] * 2 + [0] * 10, dtype=bool)
    selected = _select_risk_threshold(
        probability, label, minimum_success=0.65
    )
    assert selected["probability_threshold"] == 0.7
    assert selected["accepted_count"] == 20


def test_nested_risk_threshold_never_uses_held_route_for_selection():
    probability = np.asarray([0.9, 0.8, 0.7, 0.6])
    label = np.asarray([1, 1, 0, 0], dtype=bool)
    trajectory = np.asarray(["a", "a", "b", "b"])
    result = _nested_risk_threshold_evaluation(
        probability, label, trajectory,
        [
            {"fold_id": "f0", "held_query_trajectories": ["a"]},
            {"fold_id": "f1", "held_query_trajectories": ["b"]},
        ],
        minimum_success=0.5,
    )
    assert result["eligible_for_unbiased_train_side_evaluation"] is True
    assert all(
        fold["threshold_selected_without_held_route_outcomes"]
        for fold in result["folds"]
    )


def test_regularization_is_selected_only_by_inner_route_predictions():
    features = np.asarray([
        [0.0], [0.1], [1.0], [1.1],
        [0.2], [0.3], [1.2], [1.3],
    ])
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool)
    trajectories = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
    result = _select_regularization_nested(
        features, labels, trajectories,
        [
            {"fold_id": "f0", "held_query_trajectories": ["a"]},
            {"fold_id": "f1", "held_query_trajectories": ["b"]},
        ],
        [0.01, 0.1],
    )
    assert result["selected_C"] in {0.01, 0.1}
    assert all(value["validation_count"] == len(labels) for value in result["candidates"])


def _candidate(index, rank, score, validation):
    return {
        "union_candidate_index": index,
        "common_initial_rank": rank,
        "initial_score": score - 0.1,
        "final_score": score,
        "validation_score": validation,
    }


def test_postselection_features_resolve_selected_and_baseline_by_union_index():
    row = {
        "selected_union_candidate_index": 5,
        "candidate_refinements": [
            _candidate(5, 2, 0.9, 0.8),
            _candidate(0, 7, 0.7, 0.9),
            _candidate(3, 1, 0.8, 0.7),
        ],
    }
    feature = postselection_feature_row(row)
    assert feature.shape == (len(FEATURE_NAMES),)
    np.testing.assert_allclose(feature[:5], [0.9, 0.1, 0.2, 0.1, -0.1])
    assert feature[5] == 0.0


def test_postselection_passthrough_has_explicit_neutral_diagnostics():
    feature = postselection_feature_row(
        {"final_score": 0.4, "candidate_refinements": []}
    )
    np.testing.assert_allclose(feature, [0.4, 0, 0, 0, 0, 1, *([0] * 11)])


def test_source_evidence_and_typed_null_are_query_local_only():
    source = {
        "posterior_mass": {
            "out_of_map_mean": 0.2,
            "truncated_in_map_tail_mean": 0.1,
        },
        "query_geometry": {"confidence_mean": 0.75},
        "proposal_diagnostics": {
            "actual_parent_actual_child": {
                "mapping_view_posterior": {
                    "typed_null_probability": 0.3,
                    "anchor_scores": [0.0, 0.0],
                },
                "disconnected_seed_vfm_likelihood": {
                    "rendered_coverage_mean": 0.8,
                    "feature_coverage_mean": 0.4,
                },
            }
        },
        # These oracle-only fields must have no effect on extracted evidence.
        "translation_m": 999.0,
        "rotation_deg": 999.0,
    }
    evidence = candidate_source_postselection_evidence(source)
    assert evidence["parent_out_of_map_mean"] == 0.2
    assert evidence["mapping_view_anchor_entropy_normalized"] == 1.0
    row = {
        "selected_union_candidate_index": 0,
        "postselection_source_evidence": evidence,
        "candidate_refinements": [{
            **_candidate(0, 1, 0.8, 0.8),
            "pose_w2c": np.eye(4).tolist(),
        }],
    }
    typed = typed_null_diagnostics(row)
    assert typed["measurements"][
        "in_map_but_canonical_field_unsupported_fraction"
    ] == 0.5
    assert typed["measurements"]["low_quality_query"] == 0.25


def test_serialized_logistic_inference_uses_frozen_feature_contract():
    feature = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "standardization_mean": [0.0] * len(FEATURE_NAMES),
        "standardization_scale": [1.0] * len(FEATURE_NAMES),
        "coefficient": [0.0] * len(FEATURE_NAMES),
        "intercept": 0.0,
    }
    np.testing.assert_allclose(
        predict_sigmoid_logistic_payload(payload, feature), [0.5, 0.5]
    )


def test_complete_route_oof_calibrator_fits_one_joint_model(tmp_path):
    rows = []
    image_ids = []
    for route in range(5):
        for local, error in enumerate((0.1, 2.0)):
            image_id = f"seq{route}/frame{local:05d}.png"
            image_ids.append(image_id)
            rows.append({
                "image_id": image_id,
                "final_translation_m": error,
                "final_rotation_deg": error * 10.0,
                "selected_union_candidate_index": 0,
                "postselection_source_evidence": {
                    name: 0.1 + 0.1 * local
                    for name in (
                        "parent_out_of_map_mean",
                        "parent_truncated_in_map_tail_mean",
                        "mapping_view_typed_null_probability",
                        "selected_exact_rendered_coverage",
                        "selected_exact_feature_coverage",
                        "query_geometry_confidence_mean",
                        "mapping_view_anchor_entropy_normalized",
                    )
                },
                "candidate_refinements": [{
                    **_candidate(0, 1, 0.8 - 0.1 * local, 0.8),
                    "pose_w2c": np.eye(4).tolist(),
                }],
            })
    protocol = {
        "official_train": {
            "count": len(rows),
            "image_ids_sha256": ordered_id_sha256(image_ids),
        },
        "development": {"folds": [{
            "fold_id": f"fold{route}",
            "held_query_trajectories": [f"seq{route}"],
        } for route in range(5)]},
    }
    report = {
        "postselection_evidence_contract": (
            "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ),
        "eligible_for_unbiased_train_side_evaluation": True,
        "selection_semantics": (
            "nested_route_crossfit_without_held_route_outcomes"
        ),
        "rows": rows,
    }
    protocol_path = tmp_path / "protocol.json"
    report_path = tmp_path / "report.json"
    final_fit_path = tmp_path / "final_fit.json"
    output_path = tmp_path / "calibration.json"
    protocol_path.write_text(json.dumps(protocol))
    report_path.write_text(json.dumps(report))
    final_fit_path.write_text(json.dumps({
        "postselection_evidence_contract": report["postselection_evidence_contract"],
        "rows": rows,
    }))
    calibrate_main([
        "--refinement_reports", str(report_path),
        "--final_fit_refinement_report", str(final_fit_path),
        "--require_nested_policy_evaluation",
        "--protocol_json", str(protocol_path),
        "--output_json", str(output_path),
    ])
    result = json.loads(output_path.read_text())
    assert result["feature_count"] == len(FEATURE_NAMES)
    assert result["calibrator_evaluation_is_route_grouped_oof"]
    assert result["policy_selection_is_nested_for_calibrator_evaluation"]
    assert not result["final_fit_outcomes_used_for_crossfit_metrics"]
    assert result["regularization"]["selection"].startswith("nested_inner_route")
    assert all(
        "regularization_selection" in value
        for value in result["heads"]["strict_0.5m_5deg"]["fold_models"]
    )
    strict_head = result["heads"]["strict_0.5m_5deg"]
    assert not strict_head["final_policy_oof_fit_diagnostic"][
        "eligible_for_unbiased_train_side_evaluation"
    ]
    operating_point = strict_head["selective_operating_points"][
        "wilson95_min_success_0.8"
    ]
    assert "deployment_policy" in operating_point[
        "final_threshold_selection_semantics"
    ]
    assert len(result["final_policy_fit_only_predictions"]) == len(rows)
    assert result["heads"]["strict_0.5m_5deg"]["metrics"]["count"] == 10
