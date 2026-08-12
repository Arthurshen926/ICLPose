"""Freeze or verify the train-only selected G23 deployment configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import (
    capture_repository_state,
    file_sha256,
)


FINAL_FIT_RECIPE = {
    "strict_geometry": {
        "builder": "MAtCha_b119fd96e484fc81eb40623c1ea92ad3dbd3c21e",
        "source_patch_sha256": "23a3cd4664c3713f936ea562a25fa56aa016f4c60df557b933242538bc8c9aa9",
        "chart_count": 64, "chart_selection": "route_proportional_uniform",
        "dense_supervision": "all_1487_official_train_images",
        "mast3r_coarse_iterations": 1000,
        "mast3r_refinement_iterations": 1000,
        "gaussian_iterations": 30000,
        "virtual_surface_cells": False,
        "surface_min_opacity": 0.05,
        "surface_adjacency_radius": 0.05,
        "surface_adjacency_element_radius_cap": 0.1,
        "surface_normal_cosine_threshold": 0.8,
        "canonical_token_supply": "grid_top",
    },
    "contributors": {
        "width": 256, "height": 144, "top_k": 4,
        "occlusion_geometry": "all_declared_clean_surface_elements",
    },
    "surface_mapper": {
        "checkpoint_protocol": "fixed_epoch_no_selection", "epochs": 120,
        "steps_per_epoch": 8, "batch_maplets": 64, "hidden_dim": 256,
        "output_dim": 128, "dropout": 0.0, "learning_rate": 0.0002,
        "weight_decay": 0.0001, "temperature": 0.07,
        "hard_negative_radius": 0.75, "hard_negative_margin": 0.25,
        "hard_negative_weight": 0.20, "seed": 2350,
    },
    "geometry_head": {
        "checkpoint_protocol": "fixed_epoch_no_selection", "epochs": 30,
        "batch_size": 16, "hidden_channels": 128,
        "architecture": "separate_decoders", "task": "multitask",
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "normal_weight": 1.0, "confidence_weight": 0.1,
        "amp": True, "seed": 2450,
    },
    "physical_readout": {
        "checkpoint_protocol": "fixed_step_no_selection", "steps": 800,
        "batch_size": 128, "learning_rate": 0.0002,
        "selection_trajectory_count": 0, "validation_trajectory_count": 0,
    },
    "validity": {
        "pooling": "current_1x1_3x3_5x5_9x9",
        "target_algorithm": "exact_owned_contributor_mass_integral_image_v1",
    },
    "final_training_trajectories": [
        "seq1", "seq2", "seq4", "seq6", "seq7", "seq8", "seq9",
        "seq10", "seq11", "seq12", "seq14",
    ],
    "strict_test_trajectories": ["seq3", "seq5", "seq13"],
}


DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS = (
    "parent_mode", "child_mode", "parent_candidates", "child_candidates",
    "child_temperature", "grouping_cosine", "support_grouping",
    "maximum_group_diameter_tokens", "parent_message_passing_iterations",
    "maximum_modes", "proposal_trials", "proposal_method",
    "mapping_view_candidates", "mapping_view_anchors",
    "mapping_view_support_pairs", "mapping_view_hypotheses",
    "mapping_view_missing_probability", "mapping_view_temperature",
    "view_geometry_disable_seed_vfm", "view_geometry_prescore_per_anchor",
    "view_geometry_exact_verify_count", "view_geometry_exact_keep_per_anchor",
    "view_geometry_exact_protected_anchors",
    "view_geometry_exact_pool_semantics",
    "view_geometry_exact_basin_translation_m",
    "view_geometry_exact_basin_rotation_deg",
    "view_geometry_final_ranking_semantics",
    "view_geometry_geometry_candidate_count",
    "view_geometry_visibility_chart", "view_geometry_chart_iterations",
    "view_geometry_chart_candidate_count",
    "view_geometry_chart_maximum_translation_m",
    "view_geometry_chart_maximum_rotation_deg",
    "geometry_proposal_confidence", "geometry_pair_supports",
    "geometry_support_pairs", "geometry_pair_candidates",
    "geometry_extension_candidates", "geometry_pair_hypotheses",
    "geometry_preliminary_poses", "geometry_orientation_normal_weight",
    "soft_phase_anchors", "soft_phase_anchor_pairs",
    "soft_phase_keep_per_anchor", "soft_parent_candidates",
    "soft_edge_candidates", "soft_maximum_edges", "sparse_vfm_temperature",
    "sparse_vfm_batch_size", "sparse_vfm_maximum_splat_radius_tokens",
    "sparse_vfm_score_semantics", "sparse_vfm_primitives_per_child",
    "sparse_primitive_score_semantics", "graph_seed_parent_pair_count",
    "graph_seed_parent_count", "graph_support_anchor_count",
    "graph_support_anchor_pair_count", "local_evidence_weight",
    "translation_nms_m", "rotation_nms_deg", "render_identity_rerank",
    "identity_render_mode", "cascade_topk", "cascade_disagreement_m",
    "cascade_disagreement_deg", "cascade_margin", "cascade_always_exact",
    "detector_radio_refine_topn",
)

RUN_SOURCE_IDENTITY_KEYS = (
    "git_commit_sha", "git_tracked_status_sha256", "git_tracked_diff_sha256",
    "untracked_source_file_count", "untracked_source_files_sha256",
    "repository_state_capture",
)


def _refinement_execution_configuration(
    report: dict[str, object],
) -> dict[str, object]:
    keys = (
        "mode_name", "refine_topk", "refinement_candidate_policy",
        "refinement_score_margin", "minimum_refinement_candidates",
        "refinement_basin_translation_m", "refinement_basin_rotation_deg",
        "passthrough_without_additional_expert", "baseline_mode_count",
        "additional_expert_mode_count", "maximum_splat_radius_tokens",
        "validation_splat_radius_tokens",
        "require_cross_splat_winner_consistency",
    )
    missing = [key for key in keys if key not in report]
    optimizer = report.get("refinement_optimizer")
    if missing or not isinstance(optimizer, dict):
        raise ValueError(
            f"refinement report lacks frozen execution fields: {missing}"
        )
    result = {key: report[key] for key in keys}
    result["refinement_optimizer"] = dict(optimizer)
    return result


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate_configuration(report: dict[str, object]) -> dict[str, object]:
    mapping_view = dict(report["mapping_view_contract"])
    for key in ("candidate_count", "graph_view_node_count"):
        mapping_view.pop(key, None)
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
    missing = [key for key in keys if key not in report]
    if missing:
        raise ValueError(f"candidate report lacks frozen fields: {missing}")
    result = {key: report[key] for key in keys}
    result["mapping_view_contract"] = mapping_view
    run_manifest = report.get("run_manifest", {})
    if (
        not isinstance(run_manifest, dict)
        or "numeric_contract" not in run_manifest
        or not isinstance(run_manifest.get("configuration"), dict)
    ):
        raise ValueError("candidate report lacks numeric determinism contract")
    result["numeric_contract"] = run_manifest["numeric_contract"]
    run_configuration = run_manifest["configuration"]
    missing_arguments = [
        key for key in DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS
        if key not in run_configuration
    ]
    if missing_arguments:
        raise ValueError(
            f"candidate run manifest lacks inference arguments: {missing_arguments}"
        )
    result["inference_arguments"] = {
        key: run_configuration[key] for key in DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS
    }
    return result


def freeze_configuration(
    *, protocol_path: Path, map_audit_path: Path, candidate_evaluation_path: Path,
    adaptive_policy_path: Path, calibrator_path: Path, failure_taxonomy_path: Path,
    extrapolation_evaluation_path: Path, six_axis_basin_path: Path,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text())
    map_audit = json.loads(map_audit_path.read_text())
    candidates = json.loads(candidate_evaluation_path.read_text())
    policy = json.loads(adaptive_policy_path.read_text())
    calibrator = json.loads(calibrator_path.read_text())
    failure_taxonomy = json.loads(failure_taxonomy_path.read_text())
    extrapolation = json.loads(extrapolation_evaluation_path.read_text())
    six_axis_basin = json.loads(six_axis_basin_path.read_text())
    protocol_sha = file_sha256(protocol_path)
    if protocol.get("artifact_type") != "goal_maplet_official_train_oof_protocol_v1":
        raise ValueError("unsupported OOF protocol")
    if (
        map_audit.get("artifact_type") != "goal_maplet_official_oof_map_audit_v1"
        or not bool(map_audit["all_folds_fixed_checkpoint_without_held_selection"])
        or not bool(map_audit["all_folds_exclude_held_and_official_test_routes"])
        or not bool(map_audit.get("all_folds_strict_geometry_confirmation", False))
    ):
        raise ValueError("fold-map leakage audit is absent or failed")
    strict_geometry = map_audit.get("strict_geometry_configuration")
    expected_geometry = FINAL_FIT_RECIPE["strict_geometry"]
    if (
        not isinstance(strict_geometry, dict)
        or int(strict_geometry.get("chart_count", -1))
        != int(expected_geometry["chart_count"])
        or int(strict_geometry.get("gaussian_iterations", -1))
        != int(expected_geometry["gaussian_iterations"])
        or str(strict_geometry.get("matcha_commit"))
        != str(expected_geometry["builder"])[len("MAtCha_"):]
        or str(strict_geometry.get("source_patch_sha256"))
        != str(expected_geometry["source_patch_sha256"])
    ):
        raise ValueError("strict geometry differs from the frozen final-fit recipe")
    if candidates.get("artifact_type") != (
        "goal_maplet_official_train_oof_candidate_evaluation_v1"
    ):
        raise ValueError("unsupported OOF candidate evaluation")
    integrity = candidates["integrity"]
    if (
        int(integrity["oof_query_count"]) != int(protocol["official_train"]["count"])
        or not bool(integrity["each_official_train_frame_evaluated_exactly_once"])
    ):
        raise ValueError("candidate OOF coverage is incomplete")
    if policy.get("artifact_type") != "goal_maplet_adaptive_refinement_oof_selection_v2":
        raise ValueError("adaptive refinement policy is not nested-v2")
    if policy.get("recommended_configuration_semantics") != (
        "selected_on_all_train_oof_outcomes_for_final_freeze_only"
    ):
        raise ValueError("adaptive deployment policy has invalid selection semantics")
    refinement_paths = [Path(value) for value in policy.get("source_reports", [])]
    refinement_hashes = list(policy.get("source_report_sha256", []))
    if len(refinement_paths) != len(refinement_hashes) or not refinement_paths:
        raise ValueError("adaptive policy refinement lineage is incomplete")
    refinement_execution_configurations = []
    refinement_source_identities = []
    for path, expected_sha in zip(refinement_paths, refinement_hashes):
        refinement = json.loads(path.read_text())
        refinement_manifest = refinement.get("run_manifest", {})
        if (
            file_sha256(path) != str(expected_sha)
            or refinement.get("alignment_semantics") != (
                "hard_correspondence_free_pnp_free_direct_fixed_grid_surface_"
                "score_coordinate_search_with_monotonic_acceptance"
            )
            or not bool(refinement.get(
                "exact_score_acceptance_is_monotonic_by_construction"
            ))
            or bool(refinement.get("selection_uses_query_ground_truth", True))
            or not bool(refinement.get("ground_truth_pose_used_for_offline_error_only"))
            or refinement_manifest.get("schema") != "goal_maplet_run_manifest_v1"
            or refinement_manifest.get("repository_state_capture")
            != "process_start_before_long_computation"
            or refinement_manifest.get("numeric_contract", {}).get("acceptance")
            != "strict_monotonic_common_exact_surface_score"
            or refinement_manifest.get("numeric_contract", {}).get(
                "feature_and_render_dtype"
            ) != "float32"
            or refinement_manifest.get("numeric_contract", {}).get(
                "primitive_tie_break"
            ) != "minimum_stable_primitive_id_within_1e-5_depth"
        ):
            raise ValueError("adaptive policy source violates refinement semantics")
        if (
            len(refinement.get("candidate_reports", [])) != 1
            or refinement.get("baseline_null_calibration") is not None
        ):
            raise ValueError("adaptive policy source is not the single G23 candidate path")
        refinement_execution_configurations.append(
            _refinement_execution_configuration(refinement)
        )
        refinement_source_identities.append({
            key: refinement_manifest.get(key) for key in RUN_SOURCE_IDENTITY_KEYS
        })
    if any(
        value != refinement_execution_configurations[0]
        for value in refinement_execution_configurations[1:]
    ):
        raise ValueError("refinement execution configuration differs by OOF shard")
    if any(
        value != refinement_source_identities[0]
        for value in refinement_source_identities[1:]
    ):
        raise ValueError("refinement implementation changed between OOF shards")
    if calibrator.get("artifact_type") != (
        "goal_maplet_postselection_success_calibration_oof_v1"
    ):
        raise ValueError("unsupported success calibrator")
    if (
        not bool(calibrator["calibrator_evaluation_is_route_grouped_oof"])
        or not bool(calibrator["policy_selection_is_nested_for_calibrator_evaluation"])
        or calibrator["evaluation_policy_semantics"] != (
            "nested_route_crossfit_policy_selection_without_held_route_outcomes"
        )
        or calibrator.get("final_fit_policy_report") is None
    ):
        raise ValueError("success calibrator does not satisfy nested policy evaluation")
    if calibrator.get("regularization", {}).get("selection") != (
        "nested_inner_route_crossfit_for_each_outer_fold_and_"
        "all_train_route_crossfit_for_final_model"
    ):
        raise ValueError("success calibrator regularization was not nested by route")
    if set(calibrator.get("heads", {})) != {
        "strict_0.5m_5deg", "loose_1m_10deg",
    }:
        raise ValueError("success calibrator threshold heads differ")
    for head in calibrator["heads"].values():
        diagnostic = head.get("final_policy_oof_fit_diagnostic", {})
        if bool(diagnostic.get("eligible_for_unbiased_train_side_evaluation", True)):
            raise ValueError("deployment-policy fit diagnostic is mislabeled as unbiased")
        for operating_point in head.get("selective_operating_points", {}).values():
            if operating_point.get("final_threshold_selection_semantics") != (
                "selected_from_crossfit_probabilities_of_the_all_train_"
                "selected_deployment_policy_for_final_freeze_only"
            ):
                raise ValueError("deployment risk threshold uses the wrong policy")
    for artifact in (map_audit, candidates, policy, calibrator):
        if str(artifact.get("protocol_sha256")) != protocol_sha:
            raise ValueError("OOF artifact protocol hash drift")
    if (
        failure_taxonomy.get("artifact_type")
        != "goal_maplet_oof_failure_taxonomy_v1"
        or str(failure_taxonomy.get("protocol_sha256")) != protocol_sha
        or int(failure_taxonomy.get("integrity", {}).get("query_count", -1))
        != int(protocol["official_train"]["count"])
        or not bool(failure_taxonomy.get("oracle_only_attribution", False))
        or bool(failure_taxonomy.get("selection_uses_ground_truth", True))
    ):
        raise ValueError("complete oracle-only G/S/V/R failure taxonomy is absent")
    if (
        extrapolation.get("artifact_type")
        != "goal_maplet_oof_extrapolation_stratified_evaluation_v1"
        or str(extrapolation.get("protocol_sha256")) != protocol_sha
        or int(extrapolation.get("integrity", {}).get("query_count", -1))
        != int(protocol["official_train"]["count"])
        or bool(extrapolation.get("selection_uses_strata_or_ground_truth", True))
    ):
        raise ValueError("route/view extrapolation evaluation is absent or incomplete")
    expected_route_count = len(protocol["official_train"]["trajectory_counts"])
    six_axis_integrity = six_axis_basin.get("integrity", {})
    if (
        six_axis_basin.get("artifact_type")
        != "goal_maplet_primitive_vfm_six_axis_basin_oof_v1"
        or str(six_axis_basin.get("protocol_sha256")) != protocol_sha
        or not bool(six_axis_basin.get("oracle_only_initialization"))
        or bool(six_axis_basin.get("refinement_uses_ground_truth", True))
        or bool(six_axis_basin.get("selection_uses_basin_outcomes", True))
        or not bool(six_axis_basin.get("strict_fold_geometry_required"))
        or not bool(six_axis_basin.get("exact_score_acceptance_is_monotonic"))
        or int(six_axis_integrity.get("fold_count", -1)) != 5
        or len(six_axis_integrity.get("queries_per_trajectory", {}))
        != expected_route_count
        or not bool(six_axis_integrity.get("every_route_balanced"))
        or bool(six_axis_integrity.get("query_fold_overlap", True))
    ):
        raise ValueError("strict route-balanced six-axis basin audit is absent")

    candidate_paths = [Path(value["source_report"]) for value in candidates["folds"]]
    reports = [json.loads(path.read_text()) for path in candidate_paths]
    configurations = [_candidate_configuration(report) for report in reports]
    if any(value != configurations[0] for value in configurations[1:]):
        raise ValueError("candidate configuration differs between OOF folds")
    deployment_configuration = configurations[0]
    deployment_arguments = deployment_configuration["inference_arguments"]
    if (
        str(deployment_configuration["parent_mode"]) != "actual"
        or str(deployment_configuration["child_mode"]) != "actual"
        or str(deployment_configuration["proposal_method"]) != "view_geometry"
        or int(deployment_arguments["detector_radio_refine_topn"]) != 0
        or int(deployment_arguments["view_geometry_exact_verify_count"]) <= 0
    ):
        raise ValueError("candidate configuration is not the deployable G23 path")
    candidate_source_identities = []
    for report in reports:
        candidate_manifest = report.get("run_manifest", {})
        query_contract = report.get("query_input_contract", {})
        numeric_contract = candidate_manifest.get(
            "numeric_contract", {}
        )
        repository_state_capture = candidate_manifest.get(
            "repository_state_capture"
        )
        if (
            bool(query_contract.get(
                "actual_path_uses_query_contributor_identity_labels", True
            ))
            or bool(query_contract.get(
                "ground_truth_pose_used_for_candidate_selection", True
            ))
            or str(numeric_contract.get("sparse_visibility"))
            != "all_clean_primitives_depth_prepass"
            or str(numeric_contract.get("pose_dtype")) != "float64"
            or str(numeric_contract.get("feature_and_render_dtype")) != "float32"
            or str(numeric_contract.get("primitive_tie_break"))
            != "minimum_stable_primitive_id_within_1e-5_depth"
            or str(numeric_contract.get("random_seed_policy"))
            != "sha256_image_id_uint31_little_endian_v1"
            or repository_state_capture
            != "process_start_before_long_computation"
        ):
            raise ValueError("candidate OOF report violates deployed query semantics")
        candidate_source_identities.append({
            key: candidate_manifest.get(key) for key in RUN_SOURCE_IDENTITY_KEYS
        })
    if any(
        value != candidate_source_identities[0]
        for value in candidate_source_identities[1:]
    ):
        raise ValueError("candidate implementation changed between OOF folds")
    if candidate_source_identities[0] != refinement_source_identities[0]:
        raise ValueError("candidate and refinement used different implementations")
    selection_paths = {
        str(report.get("candidate_selection_report")) for report in reports
    }
    selection_hashes = {
        str(report.get("candidate_selection_report_sha256")) for report in reports
    }
    if (
        len(selection_paths) != 1 or len(selection_hashes) != 1
        or None in {report.get("candidate_selection_report") for report in reports}
    ):
        raise ValueError("strict OOF folds do not share one screening selection")
    screening_selection_path = Path(next(iter(selection_paths)))
    screening_selection_sha = next(iter(selection_hashes))
    if file_sha256(screening_selection_path) != screening_selection_sha:
        raise ValueError("screening candidate selection changed before freeze")
    screening_selection = json.loads(screening_selection_path.read_text())
    if (
        screening_selection.get("artifact_type")
        != "goal_maplet_candidate_oof_development_selection_v1"
        or screening_selection.get("recommended", {}).get("candidate_configuration")
        != configurations[0]
    ):
        raise ValueError("strict OOF configuration differs from screening selection")
    recipe_sha = _canonical_sha256(FINAL_FIT_RECIPE)
    return {
        "artifact_type": "goal_maplet_g23_frozen_configuration_v1",
        "freeze_source": "all_1487_official_train_route_grouped_oof_only",
        "official_test_metrics_observed_during_freeze": False,
        "protocol": str(protocol_path), "protocol_sha256": protocol_sha,
        "official_train_count": int(protocol["official_train"]["count"]),
        "official_train_image_ids_sha256": protocol["official_train"]["image_ids_sha256"],
        "official_test_count": int(protocol["official_test"]["count"]),
        "official_test_image_ids_sha256": protocol["official_test"]["image_ids_sha256"],
        "candidate_configuration": deployment_configuration,
        "refinement_configuration": policy["recommended_configuration"],
        "refinement_execution_configuration": (
            refinement_execution_configurations[0]
        ),
        "implementation_source_identity": candidate_source_identities[0],
        "selection_order": policy["selection_order"],
        "success_calibration": {
            "path": str(calibrator_path), "sha256": file_sha256(calibrator_path),
            "probability_semantics": calibrator["probability_semantics"],
            "feature_names": calibrator["feature_names"],
            "regularization": calibrator["regularization"],
            "heads": sorted(calibrator["heads"]),
            "selective_risk_targets": calibrator.get("selective_risk_targets", []),
            "selective_operating_points": {
                name: head.get("selective_operating_points", {})
                for name, head in calibrator["heads"].items()
            },
        },
        "final_fit_recipe": FINAL_FIT_RECIPE,
        "final_fit_recipe_sha256": recipe_sha,
        "lineage": {
            "map_audit": {"path": str(map_audit_path), "sha256": file_sha256(map_audit_path)},
            "candidate_evaluation": {"path": str(candidate_evaluation_path), "sha256": file_sha256(candidate_evaluation_path)},
            "adaptive_policy": {"path": str(adaptive_policy_path), "sha256": file_sha256(adaptive_policy_path)},
            "failure_taxonomy": {
                "path": str(failure_taxonomy_path),
                "sha256": file_sha256(failure_taxonomy_path),
            },
            "screening_candidate_selection": {
                "path": str(screening_selection_path),
                "sha256": screening_selection_sha,
            },
            "extrapolation_evaluation": {
                "path": str(extrapolation_evaluation_path),
                "sha256": file_sha256(extrapolation_evaluation_path),
            },
            "six_axis_refinement_basin": {
                "path": str(six_axis_basin_path),
                "sha256": file_sha256(six_axis_basin_path),
            },
            "candidate_fold_reports": [
                {"path": str(path), "sha256": file_sha256(path)} for path in candidate_paths
            ],
        },
        "claim_boundary": {
            "screening_geometry_publishable": False,
            "strict_fold_rebuilt_geometry_completed_before_freeze": True,
            "test_feedback_may_change_configuration": False,
        },
    }


def verify_frozen(path: Path, protocol_path: Path) -> dict[str, object]:
    frozen = json.loads(path.read_text())
    if frozen.get("artifact_type") != "goal_maplet_g23_frozen_configuration_v1":
        raise ValueError("unsupported frozen G23 configuration")
    if str(frozen.get("protocol_sha256")) != file_sha256(protocol_path):
        raise ValueError("frozen configuration protocol hash drift")
    if str(frozen.get("final_fit_recipe_sha256")) != _canonical_sha256(FINAL_FIT_RECIPE):
        raise ValueError("frozen final-fit recipe differs from executable recipe")
    current_source = capture_repository_state(Path(__file__).resolve().parents[3])
    current_identity = {
        key: current_source.get(key) for key in RUN_SOURCE_IDENTITY_KEYS
    }
    if frozen.get("implementation_source_identity") != current_identity:
        raise ValueError("current implementation differs from frozen OOF source")
    calibrator = frozen["success_calibration"]
    if file_sha256(Path(calibrator["path"])) != str(calibrator["sha256"]):
        raise ValueError("frozen success calibrator changed")
    for value in frozen["lineage"].values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if file_sha256(Path(item["path"])) != str(item["sha256"]):
                raise ValueError(f"frozen lineage changed: {item['path']}")
    return frozen


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--protocol", required=True)
    build.add_argument("--map_audit", required=True)
    build.add_argument("--candidate_evaluation", required=True)
    build.add_argument("--adaptive_policy", required=True)
    build.add_argument("--calibrator", required=True)
    build.add_argument("--failure_taxonomy", required=True)
    build.add_argument("--extrapolation_evaluation", required=True)
    build.add_argument("--six_axis_basin", required=True)
    build.add_argument("--output_json", required=True)
    build.add_argument("--force", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--frozen", required=True)
    verify.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        frozen = verify_frozen(Path(args.frozen), Path(args.protocol))
        print(json.dumps({
            "artifact_type": frozen["artifact_type"],
            "verified": True,
            "final_fit_recipe_sha256": frozen["final_fit_recipe_sha256"],
        }, indent=2, sort_keys=True))
        return
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite frozen configuration: {output}")
    frozen = freeze_configuration(
        protocol_path=Path(args.protocol), map_audit_path=Path(args.map_audit),
        candidate_evaluation_path=Path(args.candidate_evaluation),
        adaptive_policy_path=Path(args.adaptive_policy),
        calibrator_path=Path(args.calibrator),
        failure_taxonomy_path=Path(args.failure_taxonomy),
        extrapolation_evaluation_path=Path(args.extrapolation_evaluation),
        six_axis_basin_path=Path(args.six_axis_basin),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_type": frozen["artifact_type"], "output": str(output),
        "official_train_count": frozen["official_train_count"],
        "official_test_metrics_observed_during_freeze": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
