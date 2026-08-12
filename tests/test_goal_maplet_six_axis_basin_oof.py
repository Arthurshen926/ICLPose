import json

import pytest

from feature_extract.tools.vfm.aggregate_goal_maplet_six_axis_basin_oof import (
    EXPECTED_CONFIGURATION,
    aggregate_reports,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_primitive_refinement_six_axis_basin import (
    _trial_definitions,
)
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def _write_fixture(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "artifact_type": "goal_maplet_official_train_oof_protocol_v1",
        "development": {"folds": [{
            "fold_id": "fold0", "held_query_trajectories": ["seq2"],
        }]},
    }))
    image_ids = [f"seq2/frame{index:05d}.png" for index in range(4)]
    rows = []
    trials = _trial_definitions(
        EXPECTED_CONFIGURATION["translation_levels_m"],
        EXPECTED_CONFIGURATION["rotation_levels_deg"],
        include_zero=True,
    )
    for image_id in image_ids:
        for trial in trials:
            rows.append({
                "image_id": image_id,
                "perturbation_axis": trial["axis"],
                "perturbation_magnitude": trial["magnitude"],
                "perturbation_sign": trial["sign"],
                "final_translation_m": 0.1,
                "final_rotation_deg": 1.0,
                "improved_normalized_error": True,
                "exact_score_monotonic": True,
                "accepted_steps": 1,
            })
    report = tmp_path / "fold0.json"
    report.write_text(json.dumps({
        "artifact_type": "goal_maplet_primitive_vfm_six_axis_basin_v1",
        "oracle_only_initialization": True,
        "refinement_uses_ground_truth": False,
        "exact_score_acceptance_is_monotonic": True,
        "surface_score_semantics": (
            "all_geometry_occludes_fixed_query_grid_primitive_vfm"
        ),
        "configuration": EXPECTED_CONFIGURATION,
        "query_count": len(image_ids),
        "query_image_ids_sha256": ordered_id_sha256(image_ids),
        "physical_map_sha256": "physical",
        "canonical_field_sha256": "field",
        "surface_mapper_sha256": "mapper",
        "rows": rows,
    }))
    return protocol, report


def test_six_axis_oof_aggregation_requires_complete_balanced_trials(tmp_path):
    protocol, report = _write_fixture(tmp_path)
    result = aggregate_reports(protocol, {"fold0": report})
    assert result["integrity"]["query_count"] == 4
    assert result["integrity"]["trial_count"] == 4 * 55
    assert result["summary"]["strict_success_rate"] == 1.0


def test_six_axis_oof_aggregation_rejects_non_monotonic_trial(tmp_path):
    protocol, report = _write_fixture(tmp_path)
    payload = json.loads(report.read_text())
    payload["rows"][0]["exact_score_monotonic"] = False
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-monotonic"):
        aggregate_reports(protocol, {"fold0": report})
