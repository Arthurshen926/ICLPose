"""Phase-2 position-acquisition grid for progressive global-v2 prefixes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

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
    PARENT_PREFIX_BUDGETS,
    POSITION_BUDGETS,
    QUERY_PROPOSAL_SCHEMA,
    load_progressive_query_proposal,
)


SCHEMA = "goal_maplet_progressive_global_position_acquisition_v1"
ACQUISITION_DISTANCE_M = 2.0
SELECTION_THRESHOLD = 0.95


def _select_configuration(
    grid_rows: list[dict[str, object]], *, query_count: int,
) -> tuple[dict[str, int] | None, int]:
    required = int(math.ceil(SELECTION_THRESHOLD * query_count - 1e-12))
    passing = [
        row for row in grid_rows
        if int(row["position_acquisition_hits"]) >= required
    ]
    if not passing:
        return None, required
    selected = min(
        passing,
        key=lambda row: (
            int(row["total_position_budget"]),
            int(row["parent_prefix_budget"]),
        ),
    )
    return {
        "parent_prefix_budget": int(selected["parent_prefix_budget"]),
        "total_position_budget": int(selected["total_position_budget"]),
        "position_acquisition_hits": int(selected["position_acquisition_hits"]),
    }, required


def _validate_frozen_seq10_selection(path: Path) -> dict:
    report = json.loads(path.read_text())
    unhashed = {key: value for key, value in report.items() if key != "content_sha256"}
    selection = report.get("seq10_position_budget_selection", {})
    selected = selection.get("selected_configuration") or {}
    if (
        report.get("artifact_type") != SCHEMA
        or report.get("query_route") != "seq10"
        or report.get("content_sha256") != canonical_json_sha256(unhashed)
        or selection.get("decision") != "GO"
        or selection.get("selection_order")
        != "minimum_total_position_budget_then_minimum_parent_prefix"
        or int(selection.get("required_hits", -1)) != 84
        or float(selection.get("threshold", -1.0)) != SELECTION_THRESHOLD
        or int(selected.get("parent_prefix_budget", -1)) not in PARENT_PREFIX_BUDGETS
        or int(selected.get("total_position_budget", -1)) not in POSITION_BUDGETS
        or int(selected.get("position_acquisition_hits", -1)) < 84
    ):
        raise ValueError("held progressive evaluation lacks a passing frozen seq10 gate")
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
        raise ValueError("only seq10 may open the progressive position grid")
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite progressive acquisition report")
    started = time.perf_counter()
    proposal_path = Path(args.proposal).resolve()
    arrays, metadata = load_progressive_query_proposal(proposal_path)
    if metadata.get("query_route") != route:
        raise ValueError("progressive proposal query route differs")
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    if (
        metadata.get("artifact_type") != QUERY_PROPOSAL_SCHEMA
        or metadata.get("global_support", {}).get("file_sha256")
        != file_sha256(support_path)
        or metadata.get("global_support", {}).get("content_sha256")
        != global_metadata["content_sha256"]
    ):
        raise ValueError("progressive proposal global-v2 binding differs")
    protocol_path = Path(args.official_protocol).resolve()
    audit_path = Path(args.contributor_audit).resolve()
    contributor_dir = Path(args.contributors).resolve()
    protocol, contributor_audit = _validate_protocol_and_audit(
        protocol_path, audit_path, contributor_dir,
    )
    token_manifest = Path(args.token_manifest).resolve()
    image_ids = _query_ids_from_pose_free_manifest(token_manifest, route, protocol)
    if image_ids != arrays["image_ids"].tolist():
        raise ValueError("progressive proposal and official query inventories differ")
    frozen = None
    selected_only = None
    if args.frozen_seq10_selection:
        frozen_path = Path(args.frozen_seq10_selection).resolve()
        frozen = _validate_frozen_seq10_selection(frozen_path)
        if (
            frozen.get("global_support_content_sha256")
            != global_metadata["content_sha256"]
            or frozen.get("parent_order_content_sha256")
            != metadata["parent_order"]["content_sha256"]
            or frozen.get("proposal_semantics") != metadata["semantics"]
        ):
            raise ValueError("held proposal algorithm differs from frozen seq10 screen")
        selected_only = frozen["seq10_position_budget_selection"][
            "selected_configuration"
        ]
    # Phase 1 is fully loaded and frozen before the first pose-bearing file.
    target, contributor_bindings = _load_route_poses(
        contributor_dir, image_ids, route, contributor_audit,
    )
    target_centers = np.stack([
        camera_center_from_pose_w2c(pose) for pose in target
    ])
    origin = np.asarray(global_arrays["lattice_origin_world"], dtype=np.float64)
    spacing = float(global_arrays["lattice_spacing_m"])
    global_positions = origin[None] + (
        np.asarray(global_arrays["cell_indices_world"], dtype=np.float64) + 0.5
    ) * spacing
    offsets = arrays["candidate_offsets"]
    candidates = arrays["candidate_cell_rows"]
    parent_budgets = list(PARENT_PREFIX_BUDGETS)
    position_budgets = list(POSITION_BUDGETS)
    nearest_by_grid: dict[tuple[int, int], np.ndarray] = {}
    evaluated_parent_budgets = (
        parent_budgets if selected_only is None
        else [int(selected_only["parent_prefix_budget"])]
    )
    evaluated_position_budgets = (
        position_budgets if selected_only is None
        else [int(selected_only["total_position_budget"])]
    )
    for parent_budget in evaluated_parent_budgets:
        prefix_index = parent_budgets.index(parent_budget)
        values = {
            budget: np.full((len(image_ids),), np.inf, dtype=np.float64)
            for budget in evaluated_position_budgets
        }
        for query in range(len(image_ids)):
            group = query * len(parent_budgets) + prefix_index
            start, end = int(offsets[group]), int(offsets[group + 1])
            rows = candidates[start:end]
            positions = global_positions[rows]
            distance = np.sqrt(np.sum(
                (positions - target_centers[query, None]) ** 2, axis=1,
            ))
            cumulative = np.minimum.accumulate(distance)
            for budget in evaluated_position_budgets:
                values[budget][query] = float(cumulative[budget - 1])
        for budget, distance in values.items():
            nearest_by_grid[(parent_budget, budget)] = distance
    grid_rows = []
    for position_budget in evaluated_position_budgets:
        for parent_budget in evaluated_parent_budgets:
            distance = nearest_by_grid[(parent_budget, position_budget)]
            hit = distance <= ACQUISITION_DISTANCE_M
            grid_rows.append({
                "parent_prefix_budget": parent_budget,
                "total_position_budget": position_budget,
                "position_acquisition_distance_m": ACQUISITION_DISTANCE_M,
                "position_acquisition_hits": int(np.sum(hit)),
                "position_acquisition_misses": int(np.sum(~hit)),
                "position_acquisition_rate": float(np.mean(hit)),
                "best_translation_m": _stats(distance),
                "implicit_pose_factor_count": position_budget * ORIENTATION_COUNT,
            })
    selection = None
    if route == "seq10":
        selected, required = _select_configuration(grid_rows, query_count=len(image_ids))
        selection = {
            "metric": "absolute_position_acquisition_within_2m",
            "threshold": SELECTION_THRESHOLD,
            "required_hits": required,
            "query_count": len(image_ids),
            "selection_order": (
                "minimum_total_position_budget_then_minimum_parent_prefix"
            ),
            "selected_configuration": selected,
            "decision": "GO" if selected is not None else "KILL",
            "all_20_grid_cells_reported": len(grid_rows) == 20,
            "held_route_pose_labels_opened": False,
        }
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_route": route,
        "query_count": len(image_ids),
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": metadata["content_sha256"],
        "proposal_semantics": metadata["semantics"],
        "global_support": str(support_path),
        "global_support_file_sha256": file_sha256(support_path),
        "global_support_content_sha256": global_metadata["content_sha256"],
        "parent_order_content_sha256": metadata["parent_order"]["content_sha256"],
        "token_manifest": {
            "path": str(token_manifest), "file_sha256": file_sha256(token_manifest),
        },
        "official_protocol": {
            "path": str(protocol_path), "file_sha256": file_sha256(protocol_path),
            "declared_pose_file_sha256": protocol["official_train"]["pose_file_sha256"],
        },
        "contributor_audit": {
            "path": str(audit_path), "file_sha256": file_sha256(audit_path),
        },
        "contributor_inventory": contributor_bindings,
        "evaluated_grid": grid_rows,
        "seq10_position_budget_selection": selection,
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
        "orientation_count_bound_but_not_evaluated_for_gate": ORIENTATION_COUNT,
        "main_gate_is_position_acquisition_only": True,
        "position_orientation_cartesian_product_materialized": False,
        "candidate_cells_strictly_from_global_v2": True,
        "position_budget_prefixes_nested_within_parent_prefix": True,
        "parent_prefix_domains_required_nested": False,
        "score_before_label_separation": {
            "proposal_frozen_before_route_contributors_opened": True,
            "phase1_builder_has_no_contributor_or_gt_argument": True,
            "only_selected_route_pose_files_opened_in_phase2": True,
            "held_labels_used_for_budget_selection": False,
        },
        "ranking_beyond_scene_parent_order_performed": False,
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
        "query_count": len(image_ids),
        "evaluated_grid": grid_rows,
        "seq10_position_budget_selection": selection,
        "phase2_elapsed_seconds": report["phase2_elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
