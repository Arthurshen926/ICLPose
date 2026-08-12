import json

import pytest

from feature_extract.tools.vfm.freeze_goal_maplet_g23_configuration import (
    DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS,
    FINAL_FIT_RECIPE,
    _candidate_configuration,
    _canonical_sha256,
    verify_frozen,
)
from feature_extract.tools.vfm.render_goal_maplet_frozen_candidate_args import main as render_args
from feature_extract.tools.vfm.render_goal_maplet_frozen_refinement_args import (
    frozen_refinement_arguments,
)
from feature_extract.tools.vfm.run_goal_maplet_strict_matcha_fold import (
    MATCHA_PATCH_CONTRACT,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.lineage import capture_repository_state


def _candidate():
    keys = (
        "alike_detector_only", "cascade_contract", "child_mode",
        "detector_radio_refine_topn", "geometry_pair_beam",
        "geometry_proposal_confidence", "graph_seed_parent_count",
        "graph_seed_parent_pair_count", "graph_support_anchor_count",
        "graph_support_anchor_pair_count", "identity_render_mode",
        "local_evidence_weight", "maximum_group_diameter_tokens",
        "maximum_modes", "parent_conditioned_child_enumeration",
        "parent_message_passing_iterations", "parent_mode",
        "posterior_mass_semantics", "proposal_method", "proposal_seed_policy",
        "proposal_trials", "render_identity_rerank", "rotation_nms_deg",
        "support_grouping", "translation_nms_m",
    )
    result = {key: 0 for key in keys}
    result["mapping_view_contract"] = {
        "anchor_count": 16, "candidate_budget": 128,
        "candidate_count": 101, "graph_view_node_count": 1135,
    }
    result["run_manifest"] = {
        "numeric_contract": {"feature_dtype": "float32"},
        "configuration": {
            key: False if key in {
                "view_geometry_disable_seed_vfm", "view_geometry_visibility_chart",
                "render_identity_rerank", "cascade_always_exact",
            } else 0
            for key in DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS
        },
    }
    return result


def test_candidate_freeze_removes_fold_dynamic_view_counts():
    one, two = _candidate(), _candidate()
    two["mapping_view_contract"]["candidate_count"] = 99
    assert _candidate_configuration(one) == _candidate_configuration(two)


def test_verify_rejects_recipe_drift(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text("{}")
    lineage = tmp_path / "lineage.json"
    lineage.write_text("{}")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({
        "artifact_type": "goal_maplet_g23_frozen_configuration_v1",
        "protocol_sha256": file_sha256(protocol),
        "final_fit_recipe_sha256": "drift",
        "success_calibration": {"path": str(calibrator), "sha256": file_sha256(calibrator)},
        "lineage": {"x": {"path": str(lineage), "sha256": file_sha256(lineage)}},
    }))
    with pytest.raises(ValueError, match="recipe differs"):
        verify_frozen(frozen, protocol)


def test_verify_rejects_implementation_source_drift(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text("{}")
    lineage = tmp_path / "lineage.json"
    lineage.write_text("{}")
    source = capture_repository_state(tmp_path.parents[1])
    source_identity = {
        key: source[key]
        for key in (
            "git_commit_sha", "git_tracked_status_sha256",
            "git_tracked_diff_sha256", "untracked_source_file_count",
            "untracked_source_files_sha256", "repository_state_capture",
        )
    }
    source_identity["git_commit_sha"] = "different"
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({
        "artifact_type": "goal_maplet_g23_frozen_configuration_v1",
        "protocol_sha256": file_sha256(protocol),
        "final_fit_recipe_sha256": _canonical_sha256(FINAL_FIT_RECIPE),
        "implementation_source_identity": source_identity,
        "success_calibration": {
            "path": str(calibrator), "sha256": file_sha256(calibrator),
        },
        "lineage": {"x": {"path": str(lineage), "sha256": file_sha256(lineage)}},
    }))
    with pytest.raises(ValueError, match="implementation differs"):
        verify_frozen(frozen, protocol)


def test_recipe_hash_is_canonical():
    assert _canonical_sha256(FINAL_FIT_RECIPE) == _canonical_sha256(
        json.loads(json.dumps(FINAL_FIT_RECIPE))
    )


def test_final_recipe_binds_the_versioned_matcha_patch():
    assert FINAL_FIT_RECIPE["strict_geometry"]["source_patch_sha256"] == (
        file_sha256(MATCHA_PATCH_CONTRACT)
    )


def test_frozen_candidate_renderer_emits_complete_path_free_args(tmp_path, capsys):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({
        "candidate_configuration": _candidate_configuration(_candidate())
    }))
    render_args(["--frozen", str(frozen)])
    tokens = capsys.readouterr().out.splitlines()
    assert "--mapping_view_candidates" in tokens
    assert "--view_geometry_exact_pool_semantics" in tokens
    assert "--contributors" not in tokens


def test_frozen_refinement_renderer_uses_recorded_optimizer_not_defaults():
    payload = {
        "artifact_type": "goal_maplet_g23_frozen_configuration_v1",
        "refinement_execution_configuration": {
            "mode_name": "actual_parent_actual_child",
            "refine_topk": 32,
            "refinement_candidate_policy": "score_topk",
            "refinement_score_margin": -1.0,
            "minimum_refinement_candidates": 1,
            "refinement_basin_translation_m": 0.5,
            "refinement_basin_rotation_deg": 5.0,
            "passthrough_without_additional_expert": False,
            "baseline_mode_count": 0,
            "additional_expert_mode_count": 0,
            "maximum_splat_radius_tokens": 0,
            "validation_splat_radius_tokens": 1,
            "require_cross_splat_winner_consistency": True,
            "refinement_optimizer": {
                "translation_steps_m": [0.6, 0.4],
                "rotation_steps_deg": [5.0, 3.0],
                "iterations_per_scale": 2,
                "minimum_score_improvement": 1.0e-6,
            },
        },
    }
    tokens = frozen_refinement_arguments(payload)
    assert tokens[tokens.index("--translation_steps_m") + 1] == "0.6,0.4"
    assert "--require_cross_splat_winner_consistency" in tokens
    assert "--passthrough_without_additional_expert" not in tokens
