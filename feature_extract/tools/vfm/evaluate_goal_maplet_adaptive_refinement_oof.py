"""Select a bounded basin-aware refinement policy from complete train OOF runs.

The source run refines every score-ranked state up to Kmax once.  This tool
then replays cheaper policies using only pre-refinement scores, anchors and
poses to decide which already-computed outcomes would have been available.
Ground truth is used only after each policy has selected its winner.

Two outputs have deliberately different semantics.  Nested route-crossfit
rows select the policy on the other routes and are the only rows eligible for
an unbiased train-side risk estimate.  The policy selected on all OOF rows is
the frozen deployment policy; its replayed rows may train the final success
calibrator, but must not be reported as an unbiased OOF evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_refinement_k_curve import (
    _aggregate,
    _prefix_selection,
)
from feature_extract.tools.vfm.refine_goal_maplet_pose_modes_with_primitive_vfm import (
    _select_refinement_candidate_indices,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256
from feature_extract.vfm.postselection_success_calibration import typed_null_diagnostics


def _parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or result[0] < 1:
        raise ValueError("policy budgets must be positive")
    return result


def _parse_margins(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result or any(item < 0.0 for item in result):
        raise ValueError("adaptive margins must be non-negative")
    return result


def _policy_subset(
    row: dict[str, object],
    *,
    policy: str,
    budget: int,
    score_margin: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    refinements = sorted(
        list(row.get("candidate_refinements", [])),
        key=lambda value: int(value["union_candidate_index"]),
    )
    if not refinements:
        return [], {"policy": "gate_passthrough", "selected_count": 0}
    union_indices = [int(value["union_candidate_index"]) for value in refinements]
    if union_indices != list(range(len(refinements))):
        raise ValueError("adaptive replay requires a complete union-indexed Kmax run")
    scores = np.asarray([value["initial_score"] for value in refinements], dtype=np.float64)
    poses = np.asarray([value["initial_pose_w2c"] for value in refinements], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    candidate_details = [
        (
            int(value.get("source_report_index", 0)),
            int(value.get("source_mode_rank", index + 1)),
            {"mapping_view_anchor_label": int(value.get("mapping_view_anchor_label", -1))},
        )
        for index, value in enumerate(refinements)
    ]
    selected_indices, diagnostic = _select_refinement_candidate_indices(
        common_order=order,
        common_initial_scores=scores,
        candidate_poses=poses,
        candidate_details=candidate_details,
        maximum_count=int(budget),
        policy=str(policy),
        score_margin=float(score_margin),
        minimum_count=1,
        translation_radius_m=0.5,
        rotation_radius_deg=5.0,
    )
    baseline_guard_added = 0 not in selected_indices
    if baseline_guard_added:
        selected_indices.append(0)
    diagnostic = dict(diagnostic)
    diagnostic.update({
        "baseline_guard_added": bool(baseline_guard_added),
        "selected_count": len(selected_indices),
    })
    return [refinements[index] for index in selected_indices], diagnostic


def _replay_row(
    row: dict[str, object],
    *,
    policy: str,
    budget: int,
    score_margin: float,
    validation_enabled: bool,
    require_cross_splat_winner_consistency: bool,
) -> dict[str, object]:
    subset, diagnostic = _policy_subset(
        row, policy=policy, budget=budget, score_margin=score_margin
    )
    if not subset:
        result = copy.deepcopy(row)
        result["adaptive_policy_replay"] = diagnostic
        return result
    replay = dict(row)
    replay["candidate_refinements"] = subset
    replay["baseline_null_threshold"] = None
    selected, decision, refined_count = _prefix_selection(
        replay,
        refine_k=10**9,
        validation_enabled=bool(validation_enabled),
        require_cross_splat_winner_consistency=bool(
            require_cross_splat_winner_consistency
        ),
    )
    result = copy.deepcopy(row)
    result.update({
        "final_translation_m": float(selected["final_translation_m"]),
        "final_rotation_deg": float(selected["final_rotation_deg"]),
        "final_score": float(selected["final_score"]),
        "pose_w2c": selected["pose_w2c"],
        "selected_union_candidate_index": int(selected["union_candidate_index"]),
        "selected_source_report_index": int(selected.get("source_report_index", 0)),
        "selected_source_mode_rank": int(selected.get("source_mode_rank", 1)),
        "selection_decision": str(decision),
        "refined_mode_count": int(refined_count),
        "candidate_refinements": copy.deepcopy(subset),
        "adaptive_policy_replay": diagnostic,
    })
    source_evidence = dict(result.get("postselection_source_evidence", {}))
    if "rendered_coverage" in selected and "feature_coverage" in selected:
        source_evidence.update({
            "selected_exact_rendered_coverage": float(
                selected["rendered_coverage"]
            ),
            "selected_exact_feature_coverage": float(
                selected["feature_coverage"]
            ),
            "coverage_semantics": (
                "adaptive_replay_final_selected_pose_under_primary_exact_renderer"
            ),
        })
    result["postselection_source_evidence"] = source_evidence
    result["typed_null_diagnostics"] = typed_null_diagnostics(result)
    return result


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    compact = [{
        "translation_m": float(row["final_translation_m"]),
        "rotation_deg": float(row["final_rotation_deg"]),
        "refined_candidate_count": int(row.get("refined_mode_count", 0)),
    } for row in rows]
    return _aggregate(compact)


def _configuration_key(value: dict[str, object]) -> tuple[object, ...]:
    summary = value["summary"]
    return (
        int(summary["catastrophic_count"]),
        -int(summary["strict_count"]),
        -int(summary["loose_count"]),
        float(summary["translation_p90_m"]),
        float(summary["rotation_p90_deg"]),
        float(summary["mean_refined_candidate_count"]),
    )


def _configuration_on_rows(
    configuration: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Return a policy record whose objective is computed on ``rows`` only."""

    result = {
        key: copy.deepcopy(value)
        for key, value in configuration.items()
        if key not in {"summary", "per_route"}
    }
    result["summary"] = _summary(rows)
    trajectories = sorted({str(row["image_id"]).split("/", 1)[0] for row in rows})
    result["per_route"] = {
        trajectory: _summary([
            row for row in rows
            if str(row["image_id"]).split("/", 1)[0] == trajectory
        ])
        for trajectory in trajectories
    }
    return result


def _nested_route_crossfit_selection(
    *,
    configurations: list[dict[str, object]],
    replay_cache: dict[str, list[dict[str, object]]],
    folds: list[dict[str, object]],
    image_ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select a refinement policy without exposing a held route's outcomes."""

    replay_by_configuration = {
        configuration_id: {
            str(row["image_id"]): row for row in replayed
        }
        for configuration_id, replayed in replay_cache.items()
    }
    selected_by_image: dict[str, dict[str, object]] = {}
    fold_records = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        held = set(str(value) for value in fold["held_query_trajectories"])
        held_ids = [
            image_id for image_id in image_ids
            if image_id.split("/", 1)[0] in held
        ]
        held_id_set = set(held_ids)
        training_ids = [image_id for image_id in image_ids if image_id not in held_id_set]
        fold_configurations = []
        for configuration in configurations:
            configuration_id = str(configuration["configuration_id"])
            by_image = replay_by_configuration[configuration_id]
            fold_configurations.append(_configuration_on_rows(
                configuration, [by_image[image_id] for image_id in training_ids]
            ))
        selected = min(fold_configurations, key=_configuration_key)
        selected_id = str(selected["configuration_id"])
        held_rows = []
        for image_id in held_ids:
            row = copy.deepcopy(replay_by_configuration[selected_id][image_id])
            row["adaptive_policy_crossfit"] = {
                "selection_fold_id": fold_id,
                "held_query_trajectories": sorted(held),
                "configuration_id": selected_id,
                "policy_selected_without_this_query_or_trajectory": True,
            }
            selected_by_image[image_id] = row
            held_rows.append(row)
        fold_records.append({
            "fold_id": fold_id,
            "held_query_trajectories": sorted(held),
            "policy_training_count": len(training_ids),
            "held_query_count": len(held_ids),
            "selected_configuration": selected,
            "held_evaluation_summary": _summary(held_rows),
        })
    if set(selected_by_image) != set(image_ids):
        raise ValueError("nested policy selection did not cover every OOF query once")
    return [selected_by_image[image_id] for image_id in image_ids], fold_records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement_reports", required=True, nargs="+")
    parser.add_argument("--protocol_json", required=True)
    parser.add_argument("--score_topk_budgets", default="1,2,4,8,16,32")
    parser.add_argument("--basin_cover_budgets", default="8,16,32")
    parser.add_argument("--basin_cover_score_margins", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--selected_report_json", required=True)
    parser.add_argument(
        "--final_policy_report_json",
        default="",
        help=(
            "Optional replay under the policy selected on all train OOF rows. "
            "This is fit-only data and is not an unbiased OOF evaluation."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    selected_output = Path(args.selected_report_json)
    final_policy_output = (
        Path(args.final_policy_report_json) if str(args.final_policy_report_json) else None
    )
    outputs = [output, selected_output]
    if final_policy_output is not None:
        outputs.append(final_policy_output)
    if len(set(outputs)) != len(outputs):
        raise ValueError("adaptive policy output paths must be distinct")
    if any(path.exists() for path in outputs) and not bool(args.force):
        raise FileExistsError("refusing to overwrite adaptive OOF policy outputs")
    protocol_path = Path(args.protocol_json)
    protocol = json.loads(protocol_path.read_text())
    paths = [Path(value) for value in args.refinement_reports]
    reports = [json.loads(path.read_text()) for path in paths]
    rows = [row for report in reports for row in report.get("rows", [])]
    image_ids = [str(row["image_id"]) for row in rows]
    expected = protocol["official_train"]
    if (
        len(image_ids) != len(set(image_ids))
        or len(image_ids) != int(expected["count"])
        or ordered_id_sha256(image_ids) != str(expected["image_ids_sha256"])
    ):
        raise ValueError("adaptive refinement reports do not cover official train once")
    for report in reports:
        if str(report.get("refinement_candidate_policy")) != "score_topk":
            raise ValueError("adaptive replay source must be a score_topk Kmax run")
        if float(report.get("refinement_score_margin", -1.0)) >= 0.0:
            raise ValueError("adaptive replay source must have no score cutoff")
        if str(report.get("postselection_evidence_contract", "")) != (
            "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ):
            raise ValueError("refinement source lacks post-selection evidence contract")
    max_available = min(int(report["refine_topk"]) for report in reports)
    validation_enabled = all(
        int(report.get("validation_splat_radius_tokens", -1)) >= 0
        for report in reports
    )
    require_consistency = all(
        bool(report.get("require_cross_splat_winner_consistency", False))
        for report in reports
    )
    configurations = []
    definitions = []
    for budget in _parse_ints(str(args.score_topk_budgets)):
        definitions.append(("score_topk", budget, -1.0))
    for budget in _parse_ints(str(args.basin_cover_budgets)):
        for margin in _parse_margins(str(args.basin_cover_score_margins)):
            definitions.append(("anchor_basin_cover", budget, margin))
    replay_cache: dict[str, list[dict[str, object]]] = {}
    for policy, budget, margin in definitions:
        if budget > max_available:
            raise ValueError("requested adaptive policy budget exceeds Kmax source")
        key = f"{policy}_k{budget}_margin{margin:g}"
        replayed = [
            _replay_row(
                row,
                policy=policy,
                budget=budget,
                score_margin=margin,
                validation_enabled=validation_enabled,
                require_cross_splat_winner_consistency=require_consistency,
            )
            for row in rows
        ]
        replay_cache[key] = replayed
        per_route = {}
        for trajectory in sorted({value.split("/", 1)[0] for value in image_ids}):
            per_route[trajectory] = _summary([
                row for row in replayed
                if str(row["image_id"]).split("/", 1)[0] == trajectory
            ])
        configurations.append({
            "configuration_id": key,
            "policy": policy,
            "maximum_refinement_candidates": int(budget),
            "score_margin": float(margin),
            "summary": _summary(replayed),
            "per_route": per_route,
        })
    recommended = min(configurations, key=_configuration_key)
    final_policy_rows = replay_cache[str(recommended["configuration_id"])]
    nested_rows, nested_folds = _nested_route_crossfit_selection(
        configurations=configurations,
        replay_cache=replay_cache,
        folds=list(protocol["development"]["folds"]),
        image_ids=image_ids,
    )
    selected_frequency: dict[str, int] = {}
    for fold in nested_folds:
        configuration_id = str(fold["selected_configuration"]["configuration_id"])
        selected_frequency[configuration_id] = selected_frequency.get(configuration_id, 0) + 1
    payload = {
        "artifact_type": "goal_maplet_adaptive_refinement_oof_selection_v2",
        "protocol_json": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "source_reports": [str(path) for path in paths],
        "source_report_sha256": [file_sha256(path) for path in paths],
        "selection_uses_ground_truth_only_through_train_oof_metrics": True,
        "policy_replay_candidate_allocation_uses_ground_truth": False,
        "selection_order": [
            "minimum_catastrophic_count",
            "maximum_strict_success_count",
            "maximum_loose_success_count",
            "minimum_translation_p90",
            "minimum_rotation_p90",
            "minimum_mean_refined_candidate_count",
        ],
        "recommended_configuration": recommended,
        "recommended_configuration_semantics": (
            "selected_on_all_train_oof_outcomes_for_final_freeze_only"
        ),
        "nested_route_crossfit": {
            "held_route_outcomes_used_for_policy_selection": False,
            "eligible_for_unbiased_train_side_evaluation": True,
            "summary": _summary(nested_rows),
            "selected_configuration_frequency": selected_frequency,
            "folds": nested_folds,
        },
        "configurations": configurations,
    }
    selected_payload = {
        "artifact_type": "goal_maplet_nested_adaptive_refinement_oof_report_v2",
        "postselection_evidence_contract": (
            "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ),
        "typed_null_contract": (
            "diagnostic_intervention_vector_not_independent_likelihoods_v1"
        ),
        "policy_selection_report": str(output),
        "selection_semantics": "nested_route_crossfit_without_held_route_outcomes",
        "eligible_for_unbiased_train_side_evaluation": True,
        "selected_configuration_varies_by_fold": len(selected_frequency) > 1,
        "nested_route_crossfit": payload["nested_route_crossfit"],
        "source_reports": [str(path) for path in paths],
        "source_report_sha256": [file_sha256(path) for path in paths],
        "rows": nested_rows,
    }
    final_policy_payload = {
        "artifact_type": "goal_maplet_final_policy_refinement_training_report_v1",
        "postselection_evidence_contract": (
            "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ),
        "typed_null_contract": (
            "diagnostic_intervention_vector_not_independent_likelihoods_v1"
        ),
        "policy_selection_report": str(output),
        "selection_semantics": "selected_on_all_train_oof_outcomes_for_final_freeze",
        "eligible_for_unbiased_train_side_evaluation": False,
        "eligible_for_final_calibrator_fit": True,
        "selected_configuration": recommended,
        "source_reports": [str(path) for path in paths],
        "source_report_sha256": [file_sha256(path) for path in paths],
        "rows": final_policy_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    selected_output.write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n"
    )
    if final_policy_output is not None:
        final_policy_output.parent.mkdir(parents=True, exist_ok=True)
        final_policy_output.write_text(
            json.dumps(final_policy_payload, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "artifact_type": payload["artifact_type"],
        "query_count": len(rows),
        "recommended_configuration": recommended,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
