"""Phase-2 seq10 grid for parent-priority plus global-fallback positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_progressive_global_position_proposal import (
    _load_json_no_duplicates,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_global_all_parent_pose_support import (
    _load_route_poses,
    _query_ids_from_pose_free_manifest,
    _stats,
    _validate_protocol_and_audit,
)
from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    ORIENTATION_COUNT,
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.progressive_global_position_proposal import (
    FALLBACK_FRACTIONS,
    PRIORITY_FALLBACK_POSITION_BUDGETS,
    load_priority_fallback_query_proposal,
)


SCHEMA = "goal_maplet_priority_fallback_position_acquisition_v1"
SELECTION_THRESHOLD = 0.95


def _select_configuration(rows: list[dict], query_count: int):
    required = int(math.ceil(SELECTION_THRESHOLD * query_count - 1e-12))
    passing = [
        row for row in rows
        if row["configuration_valid_for_all_queries"]
        and int(row["position_acquisition_hits_2m"]) >= required
    ]
    if not passing:
        return None, required
    selected = min(passing, key=lambda row: (
        int(row["total_position_budget"]), float(row["fallback_fraction"]),
    ))
    return {
        "total_position_budget": int(selected["total_position_budget"]),
        "fallback_fraction_numerator": int(selected["fallback_fraction_numerator"]),
        "fallback_fraction_denominator": int(selected["fallback_fraction_denominator"]),
        "fallback_fraction": float(selected["fallback_fraction"]),
        "position_acquisition_hits_2m": int(selected["position_acquisition_hits_2m"]),
    }, required


def _validate_frozen(path: Path) -> dict:
    report = _load_json_no_duplicates(path)
    unhashed = {key: value for key, value in report.items() if key != "content_sha256"}
    gate = report.get("seq10_priority_fallback_selection", {})
    selected = gate.get("selected_configuration") or {}
    fraction = (
        int(selected.get("fallback_fraction_numerator", -1)),
        int(selected.get("fallback_fraction_denominator", -1)),
    )
    if (
        report.get("artifact_type") != SCHEMA
        or report.get("query_route") != "seq10"
        or report.get("content_sha256") != canonical_json_sha256(unhashed)
        or gate.get("decision") != "GO"
        or int(gate.get("required_hits", -1)) != 84
        or gate.get("selection_order")
        != "minimum_total_position_budget_then_minimum_fallback_fraction"
        or int(selected.get("total_position_budget", -1))
        not in PRIORITY_FALLBACK_POSITION_BUDGETS
        or fraction not in FALLBACK_FRACTIONS
        or int(selected.get("position_acquisition_hits_2m", -1)) < 84
    ):
        raise ValueError("held eval lacks passing frozen priority-fallback gate")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--global_support", required=True)
    parser.add_argument("--query_route", choices=("seq10", "seq12", "seq14"), required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--official_protocol", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--contributor_audit", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development_grid", action="store_true")
    mode.add_argument("--frozen_seq10_selection")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    route = str(args.query_route)
    if bool(args.development_grid) != (route == "seq10"):
        raise ValueError("only seq10 may open priority-fallback grid")
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite priority-fallback evaluation")
    started = time.perf_counter()
    proposal_path = Path(args.proposal).resolve()
    arrays, metadata = load_priority_fallback_query_proposal(proposal_path)
    if metadata.get("query_route") != route:
        raise ValueError("priority-fallback proposal route differs")
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    if (
        metadata.get("global_support", {}).get("file_sha256")
        != file_sha256(support_path)
        or metadata.get("global_support", {}).get("content_sha256")
        != global_metadata["content_sha256"]
    ):
        raise ValueError("priority-fallback proposal global-v2 binding differs")
    protocol_path = Path(args.official_protocol).resolve()
    audit_path = Path(args.contributor_audit).resolve()
    contributor_dir = Path(args.contributors).resolve()
    protocol, contributor_audit = _validate_protocol_and_audit(
        protocol_path, audit_path, contributor_dir,
    )
    token_manifest = Path(args.token_manifest).resolve()
    image_ids = _query_ids_from_pose_free_manifest(token_manifest, route, protocol)
    if image_ids != arrays["image_ids"].tolist():
        raise ValueError("priority-fallback proposal query inventory differs")
    frozen = None
    selected_only = None
    if args.frozen_seq10_selection:
        frozen_path = Path(args.frozen_seq10_selection).resolve()
        frozen = _validate_frozen(frozen_path)
        if (
            frozen.get("global_support_content_sha256")
            != global_metadata["content_sha256"]
            or frozen.get("parent_order_content_sha256")
            != metadata["parent_order"]["content_sha256"]
            or frozen.get("global_order_content_sha256")
            != metadata["global_order"]["content_sha256"]
        ):
            raise ValueError("held priority-fallback algorithms differ from seq10")
        selected_only = frozen["seq10_priority_fallback_selection"][
            "selected_configuration"
        ]
    target, contributor_bindings = _load_route_poses(
        contributor_dir, image_ids, route, contributor_audit,
    )
    centers = np.stack([camera_center_from_pose_w2c(pose) for pose in target])
    origin = np.asarray(global_arrays["lattice_origin_world"], dtype=np.float64)
    spacing = float(global_arrays["lattice_spacing_m"])
    positions = origin[None] + (
        np.asarray(global_arrays["cell_indices_world"], dtype=np.float64) + 0.5
    ) * spacing
    offsets = arrays["candidate_offsets"]
    candidates = arrays["candidate_cell_rows"]
    sources = arrays["candidate_source_queue"]
    counts = arrays["candidate_count_by_fallback_fraction"]
    fractions = list(FALLBACK_FRACTIONS)
    budgets = list(PRIORITY_FALLBACK_POSITION_BUDGETS)
    evaluated_fractions = fractions if selected_only is None else [(
        int(selected_only["fallback_fraction_numerator"]),
        int(selected_only["fallback_fraction_denominator"]),
    )]
    evaluated_budgets = budgets if selected_only is None else [
        int(selected_only["total_position_budget"])
    ]
    grid = []
    for budget in evaluated_budgets:
        for numerator, denominator in evaluated_fractions:
            fraction_index = fractions.index((numerator, denominator))
            valid = counts[:, fraction_index] >= budget
            invalid_count = int(np.sum(~valid))
            base = {
                "fallback_fraction_numerator": numerator,
                "fallback_fraction_denominator": denominator,
                "fallback_fraction": numerator / denominator,
                "total_position_budget": budget,
                "requested_parent_priority_count": budget - (budget * numerator) // denominator,
                "requested_outside_parent_union_fallback_count": (
                    budget * numerator
                ) // denominator,
                "configuration_valid_for_all_queries": invalid_count == 0,
                "invalid_query_count_due_exact_queue_capacity": invalid_count,
            }
            if invalid_count:
                grid.append({
                    **base,
                    "position_acquisition_hits_2m": None,
                    "position_acquisition_misses_2m": None,
                    "position_acquisition_rate_2m": None,
                    "best_translation_m": None,
                    "implicit_pose_factor_count": None,
                })
                continue
            nearest = np.full((len(image_ids),), np.inf, dtype=np.float64)
            for query in range(len(image_ids)):
                group = query * len(fractions) + fraction_index
                start = int(offsets[group])
                local_rows = candidates[start:start + budget]
                nearest[query] = np.sqrt(np.min(np.sum(
                    (positions[local_rows] - centers[query, None]) ** 2, axis=1,
                )))
                if int(np.sum(sources[start:start + budget] == 1)) != (
                    budget * numerator
                ) // denominator:
                    raise AssertionError("phase2 fallback source count differs")
            hit = nearest <= 2.0
            grid.append({
                **base,
                "position_acquisition_hits_2m": int(np.sum(hit)),
                "position_acquisition_misses_2m": int(np.sum(~hit)),
                "position_acquisition_rate_2m": float(np.mean(hit)),
                "best_translation_m": _stats(nearest),
                "implicit_pose_factor_count": budget * ORIENTATION_COUNT,
            })
    selection = None
    if route == "seq10":
        selected, required = _select_configuration(grid, len(image_ids))
        selection = {
            "metric": "absolute_position_acquisition_within_2m",
            "threshold": SELECTION_THRESHOLD,
            "required_hits": required,
            "query_count": len(image_ids),
            "selection_order": (
                "minimum_total_position_budget_then_minimum_fallback_fraction"
            ),
            "selected_configuration": selected,
            "decision": "GO" if selected is not None else "KILL",
            "all_12_grid_cells_reported": len(grid) == 12,
            "held_route_pose_labels_opened": False,
        }
    report = {
        "artifact_type": SCHEMA,
        "query_route": route,
        "query_count": len(image_ids),
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": metadata["content_sha256"],
        "global_support_file_sha256": file_sha256(support_path),
        "global_support_content_sha256": global_metadata["content_sha256"],
        "parent_order_content_sha256": metadata["parent_order"]["content_sha256"],
        "global_order_content_sha256": metadata["global_order"]["content_sha256"],
        "contributor_inventory": contributor_bindings,
        "evaluated_grid": grid,
        "seq10_priority_fallback_selection": selection,
        "frozen_seq10_selection": (
            {
                "path": str(Path(args.frozen_seq10_selection).resolve()),
                "file_sha256": file_sha256(
                    Path(args.frozen_seq10_selection).resolve()
                ),
                "content_sha256": frozen["content_sha256"],
                "selected_configuration": selected_only,
            }
            if frozen is not None else None
        ),
        "fallback_fraction_is_outside_complete_top64_parent_union": True,
        "invalid_configuration_evaluated_on_surviving_queries": False,
        "orientation_count_bound_but_not_evaluated_for_gate": ORIENTATION_COUNT,
        "position_orientation_cartesian_product_materialized": False,
        "main_gate_is_position_acquisition_only": True,
        "score_before_label_separation": {
            "proposal_frozen_before_route_pose_files_opened": True,
            "phase1_builder_has_no_contributor_or_gt_argument": True,
            "only_selected_route_pose_files_opened_in_phase2": True,
            "held_labels_used_for_selection": False,
        },
        "ranking_beyond_frozen_queue_orders_performed": False,
        "raw_acquisition_is_support_not_localization_success": True,
        "phase2_elapsed_seconds": float(time.perf_counter() - started),
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output),
        "content_sha256": report["content_sha256"],
        "query_route": route,
        "evaluated_grid": grid,
        "seq10_priority_fallback_selection": selection,
        "phase2_elapsed_seconds": report["phase2_elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
