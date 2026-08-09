import json

import numpy as np

from feature_extract.vfm.localization_goal_maplet.conditional_pose_energy import (
    ConditionalPoseEnergyPolicy,
    candidate_measurements,
    fractional_observation_quality,
    load_conditional_pose_energy,
)
from feature_extract.tools.vfm.fit_evaluate_goal_maplet_conditional_energy import (
    _fractional_mass_audit,
    _fit,
    _load_queries,
    _map_query_overlap_audit,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_conditional_energy import (
    _candidate_generator_contract,
)


def _policy(weights=(1.0, 1.0, 1.0), bias=0.0):
    return ConditionalPoseEnergyPolicy(
        weights=np.asarray(weights, dtype=np.float64),
        candidate_bias=bias,
        scale_floors=np.asarray((0.05, 0.02, 0.02), dtype=np.float64),
        null_phase_threshold=0.0,
        null_phase_scale=1.0,
        null_phase_slope=1.0,
        metadata={},
    )


def test_conditional_energy_is_monotonic_in_each_typed_measurement():
    policy = _policy()
    measurements = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    energy, _ = policy.energies(measurements)
    assert np.all(energy[1:] > energy[0])


def test_conditional_energy_normalizes_candidates_with_one_null():
    candidate, null = _policy(bias=-10.0).posterior(np.zeros((3, 3)))
    assert np.isclose(float(np.sum(candidate)) + null, 1.0)
    assert null > float(np.max(candidate))


def test_fractional_observation_penalizes_missing_and_mixed_mass():
    clean = {"jacobian_observability": 0.8, "missing_fraction_mean": 0.0,
             "mixed_surface_fraction": 0.0}
    mixed = {"jacobian_observability": 0.8, "missing_fraction_mean": 0.2,
             "mixed_surface_fraction": 0.4}
    assert fractional_observation_quality(clean) > fractional_observation_quality(mixed)


def test_candidate_measurements_keep_identity_phase_observation_separate():
    evidence = [{
        "jacobian_phase_visible": 0.4,
        "jacobian_observability": 0.7,
        "missing_fraction_mean": 0.1,
        "mixed_surface_fraction": 0.2,
    }]
    value = candidate_measurements(np.asarray([2.0]), evidence)
    assert np.allclose(value, [[2.0, 0.4, 0.5]])


def test_loader_fails_closed_on_negative_weight(tmp_path):
    path = tmp_path / "energy.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_conditional_pose_energy_v1",
        "component_names": [
            "proposal_identity", "jacobian_phase_visible",
            "fractional_observation_quality",
        ],
        "normalization": "per_query_median_iqr_with_fixed_floors_v1",
        "null_hypothesis": "candidate_set_phase_support_null_logit_zero_v2",
        "weights": [1.0, -1.0, 0.0],
        "candidate_bias": 0.0,
        "scale_floors": [0.05, 0.02, 0.02],
        "null_phase_threshold": 0.0,
        "null_phase_scale": 1.0,
        "null_phase_slope": 1.0,
    }))
    try:
        load_conditional_pose_energy(path)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("negative conditional-energy weight must fail closed")


def test_training_report_inverts_runtime_ranking_before_pairing_phase():
    def phase(value):
        return {
            "jacobian_phase_visible": value,
            "jacobian_observability": 0.8,
            "missing_fraction_mean": 0.0,
            "mixed_surface_fraction": 0.0,
        }

    report = {"rows": [{
        "image_id": "seq1/frame.png",
        # Runtime order is source candidate 1, then source candidate 0.
        "mode_details": {"mode": [
            {"pre_surface_score": 1.0, "translation_m": 2.0, "rotation_deg": 1.0},
            {"pre_surface_score": 0.0, "translation_m": 0.2, "rotation_deg": 1.0},
        ]},
        "ranking_diagnostics": {"mode": {
            "surface_phase_components_preorder": [phase(0.1), phase(0.9)],
            "surface_alignment_original_indices": [1, 0],
            "surface_alignment_evaluated_count": 2,
        }},
    }]}
    rows = _load_queries(
        report, mode_name="mode", success_translation_m=1.0,
        success_rotation_deg=10.0,
    )
    assert rows[0].target_index == 0
    np.testing.assert_allclose(rows[0].translation_m, [0.2, 2.0])
    # Phase preorder remains source-aligned after inversion.
    assert rows[0].normalized[0, 1] < rows[0].normalized[1, 1]


def test_fitted_conditional_energy_preserves_monotonic_constraints():
    report_rows = []
    for index in range(4):
        report_rows.append({
            "image_id": f"seq{index + 1}/frame.png",
            "mode_details": {"mode": [
                {"pre_surface_score": 1.0, "translation_m": 0.1, "rotation_deg": 1.0},
                {"pre_surface_score": 0.0, "translation_m": 2.0, "rotation_deg": 20.0},
            ]},
            "ranking_diagnostics": {"mode": {
                "surface_phase_components_preorder": [
                    {"jacobian_phase_visible": 0.9, "jacobian_observability": 0.9,
                     "missing_fraction_mean": 0.0, "mixed_surface_fraction": 0.0},
                    {"jacobian_phase_visible": 0.1, "jacobian_observability": 0.2,
                     "missing_fraction_mean": 0.2, "mixed_surface_fraction": 0.2},
                ],
                "surface_alignment_original_indices": [0, 1],
                "surface_alignment_evaluated_count": 2,
            }},
        })
    rows = _load_queries(
        {"rows": report_rows}, mode_name="mode", success_translation_m=1.0,
        success_rotation_deg=10.0,
    )
    weight, _bias, optimization = _fit(
        rows, active=(True, True, True), regularization=1e-3,
    )
    assert optimization["success"]
    assert np.all(weight >= 0.0)


def test_fractional_mass_audit_checks_both_closure_equations():
    evidence = {
        "feature_fraction_mean": 0.6,
        "visibility_fraction_mean": 0.8,
        "missing_fraction_mean": 0.2,
        "background_fraction_mean": 0.2,
        "dominant_surface_fraction_mean": 0.9,
        "mixed_surface_fraction": 0.1,
    }
    report = {"rows": [{"ranking_diagnostics": {"mode": {
        "surface_phase_components_preorder": [evidence],
    }}}]}
    audit = _fractional_mass_audit(report, mode_name="mode")
    assert audit["candidate_count"] == 1
    assert audit["maximum_feature_missing_background_residual"] < 1e-7
    assert audit["maximum_feature_missing_visibility_residual"] < 1e-7


def test_map_query_overlap_audit_detects_exact_self_map_image(tmp_path):
    contributors = tmp_path / "contributors"
    contributors.mkdir()
    metadata = {
        "image_id": "seq6/frame.png",
        "trajectory_id": "seq6",
    }
    np.savez_compressed(
        contributors / "frame.npz",
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    summary = tmp_path / "field.json"
    summary.write_text(json.dumps({
        "canonical_field_sha256": "field",
        "mapping_image_count": 1,
        "mapping_trajectory_ids": ["seq6"],
    }))
    audit = _map_query_overlap_audit(
        {
            "canonical_field_sha256": "field",
            "rows": [{"image_id": "seq6/frame.png"}],
        },
        canonical_field_summary=summary,
        mapping_contributors=contributors,
    )
    assert audit["exact_image_overlap_count"] == 1
    assert not audit["map_query_image_disjoint"]


def test_frozen_transfer_contract_includes_seed_and_geometry_lineage():
    common = {
        "proposal_method": "graph",
        "proposal_seed_policy": "sha256_image_id_uint31_little_endian_v1",
        "identity_render_mode": "child_splat",
        "render_identity_rerank": True,
        "cascade_contract": {"topk": 4, "always_exact": False},
        "physical_map_sha256": "map",
        "canonical_field_sha256": "field",
        "field_feature_contract_sha256": "feature-contract",
        "physical_instance_readout_sha256": "readout",
        "typed_graph_sha256": "graph-v1",
        "validity_calibration_sha256": "validity-v1",
        "maximum_modes": 32,
        "surface_verification_contract": {"maximum_modes": 16},
    }
    same = dict(common)
    changed_seed = dict(common, proposal_seed_policy=None)
    changed_surface = {
        **common,
        "surface_verification_contract": {"maximum_modes": 8},
    }
    changed_graph = dict(common, typed_graph_sha256="graph-v2")
    assert _candidate_generator_contract(common) == _candidate_generator_contract(same)
    assert _candidate_generator_contract(common) != _candidate_generator_contract(
        changed_seed
    )
    assert _candidate_generator_contract(common) != _candidate_generator_contract(
        changed_surface
    )
    assert _candidate_generator_contract(common) != _candidate_generator_contract(
        changed_graph
    )
