"""Phase-2 raw support and seq10 parent-prefix gate for geometry positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_contributors,
)
from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    camera_centers_from_w2c,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.parent_geometry_pose_proposal import (
    ORIENTATION_COVER_RADIUS_DEG,
    PARENT_PREFIX_BUDGETS,
    load_parent_geometry_proposal,
)


SCHEMA = "goal_maplet_parent_geometry_analytic60_raw_support_v1"
SELECTION_THRESHOLD = 0.95
THRESHOLDS = (
    ("region_2m_45deg", 2.0, 45.0),
    ("loose_1m_10deg", 1.0, 10.0),
    ("strict_0_5m_5deg", 0.5, 5.0),
)


def _rotation_errors(codebook: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.asarray(codebook, dtype=np.float64) @ target[:3, :3].T
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
    )
    return np.degrees(np.arccos(cosine))


def _select_minimum_parent_prefix(
    budget_rows: list[dict[str, object]], *, query_count: int,
) -> tuple[int | None, int]:
    required = int(math.ceil(SELECTION_THRESHOLD * query_count - 1.0e-12))
    qualifying = [
        int(row["parent_prefix_budget"])
        for row in budget_rows
        if int(row["raw_support"]["region_2m_45deg"]["joint_hits"])
        >= required
    ]
    return (min(qualifying) if qualifying else None), required


def _stats(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(value)),
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p90": float(np.percentile(value, 90.0)),
        "maximum": float(np.max(value)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--contributors", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development_selection", action="store_true")
    mode.add_argument("--frozen_selection")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent-geometry Phase-2 report")
    started = time.perf_counter()
    proposal_path = Path(args.proposal).resolve()
    arrays, metadata = load_parent_geometry_proposal(proposal_path)
    query_route = str(metadata["query_route"])
    if bool(args.development_selection) != (query_route == "seq10"):
        raise ValueError("parent-geometry development selection is restricted to seq10")
    evaluated_budgets = list(PARENT_PREFIX_BUDGETS)
    frozen = None
    if args.frozen_selection:
        frozen_path = Path(args.frozen_selection).resolve()
        frozen = json.loads(frozen_path.read_text())
        unhashed = {key: value for key, value in frozen.items() if key != "content_sha256"}
        selection = frozen.get("parent_prefix_selection", {})
        if (
            frozen.get("artifact_type") != SCHEMA
            or frozen.get("query_route") != "seq10"
            or frozen.get("content_sha256") != canonical_json_sha256(unhashed)
            or selection.get("decision") != "GO"
            or selection.get("threshold") != SELECTION_THRESHOLD
            or int(selection.get("selected_parent_prefix_budget", -1))
            not in PARENT_PREFIX_BUDGETS
        ):
            raise ValueError("held parent-geometry eval lacks frozen seq10 selection")
        evaluated_budgets = [int(selection["selected_parent_prefix_budget"])]

    image_ids = np.asarray(arrays["image_ids"])
    contributors = _load_contributors(
        Path(args.contributors), required_image_ids=set(image_ids.tolist()),
    )
    target_rows, contributor_bindings = [], []
    for image_id in image_ids.tolist():
        path = Path(contributors[str(image_id)]).resolve()
        digest = file_sha256(path)
        with np.load(path, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], dtype=np.float64)
        if target.shape != (4, 4) or np.any(~np.isfinite(target)):
            raise ValueError("parent-geometry Phase-2 target pose differs")
        target_rows.append(target)
        contributor_bindings.append({
            "image_id": str(image_id), "path": str(path), "file_sha256": digest,
        })
    target = np.stack(target_rows)
    target_center = camera_centers_from_w2c(target)
    codebook = np.asarray(arrays["orientation_rotations_w2c"], dtype=np.float64)
    best_rotation = np.asarray([
        np.min(_rotation_errors(codebook, pose)) for pose in target
    ])
    if np.any(best_rotation > ORIENTATION_COVER_RADIUS_DEG + 1.0e-9):
        raise AssertionError("analytic60 empirical target exceeds its global certificate")
    origin = np.asarray(arrays["lattice_origin_world"], dtype=np.float64)
    spacing = float(arrays["lattice_spacing_m"])
    offsets = arrays["cell_offsets"]
    cells = arrays["cell_indices_world"]
    first_rank = arrays["cell_first_parent_rank"]
    best_translation_by_budget = {
        budget: np.full((image_ids.size,), np.inf, dtype=np.float64)
        for budget in evaluated_budgets
    }
    for query in range(image_ids.size):
        start, end = int(offsets[query]), int(offsets[query + 1])
        local_cell = cells[start:end]
        local_rank = first_rank[start:end]
        for budget in evaluated_budgets:
            selected = local_cell[local_rank <= budget]
            position = origin[None] + (selected.astype(np.float64) + 0.5) * spacing
            best_translation_by_budget[budget][query] = float(np.min(
                np.linalg.norm(position - target_center[query, None], axis=1)
            ))
    budget_rows = []
    for budget in evaluated_budgets:
        budget_index = PARENT_PREFIX_BUDGETS.index(budget)
        translation = best_translation_by_budget[budget]
        raw_support = {}
        for name, translation_limit, rotation_limit in THRESHOLDS:
            position_hit = translation <= translation_limit
            orientation_hit = best_rotation <= rotation_limit
            joint = position_hit & orientation_hit
            raw_support[name] = {
                "position_hits": int(np.sum(position_hit)),
                "orientation_hits": int(np.sum(orientation_hit)),
                "joint_hits": int(np.sum(joint)),
                "joint_rate": float(np.mean(joint)),
                "position_misses": int(np.sum(~position_hit)),
                "orientation_misses": int(np.sum(~orientation_hit)),
                "joint_misses": int(np.sum(~joint)),
            }
        if raw_support["region_2m_45deg"]["orientation_hits"] != image_ids.size:
            raise AssertionError("analytic60 did not remove the 45-degree route ceiling")
        position_count = arrays["unique_position_count_by_parent_prefix"][:, budget_index]
        implicit_count = arrays["implicit_pose_factor_count_by_parent_prefix"][:, budget_index]
        budget_rows.append({
            "parent_prefix_budget": budget,
            "raw_support": raw_support,
            "best_translation_m": _stats(translation),
            "unique_position_count": _stats(position_count),
            "implicit_pose_factor_count": _stats(implicit_count),
        })
    selected, required = _select_minimum_parent_prefix(
        budget_rows, query_count=int(image_ids.size),
    )
    selection = None
    if bool(args.development_selection):
        selection = {
            "selection_route": "seq10",
            "threshold_metric": "absolute_region_2m_45deg_joint_query_rate",
            "threshold": SELECTION_THRESHOLD,
            "required_hits": required,
            "query_count": int(image_ids.size),
            "candidate_parent_prefix_budgets": list(PARENT_PREFIX_BUDGETS),
            "selected_parent_prefix_budget": selected,
            "decision": "GO" if selected is not None else "KILL",
            "held_routes_opened_for_selection": False,
        }
    reported_budget = selected if selected is not None else max(evaluated_budgets)
    query_rows = []
    translation = best_translation_by_budget[int(reported_budget)]
    for query, image_id in enumerate(image_ids.tolist()):
        query_rows.append({
            "query_index": query,
            "image_id": str(image_id),
            "reported_parent_prefix_budget": int(reported_budget),
            "best_translation_m": float(translation[query]),
            "best_rotation_deg": float(best_rotation[query]),
        })
    elapsed = float(time.perf_counter() - started)
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_route": query_route,
        "query_count": int(image_ids.size),
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": str(metadata["content_sha256"]),
        "contributor_inventory": contributor_bindings,
        "evaluated_parent_prefix_budgets": evaluated_budgets,
        "budget_rows": budget_rows,
        "parent_prefix_selection": selection,
        "frozen_seq10_selection": (
            {
                "path": str(Path(args.frozen_selection).resolve()),
                "file_sha256": file_sha256(Path(args.frozen_selection).resolve()),
                "content_sha256": str(frozen["content_sha256"]),
            }
            if args.frozen_selection else None
        ),
        "orientation_cover_certificate": metadata["orientation_cover_certificate"],
        "main_2m45_orientation_complete_for_all_so3": True,
        "one_meter_10deg_and_half_meter_5deg_are_empirical_only": True,
        "parent_aabb_expansion_is_a_frozen_assumption_not_pose_containment_proof": True,
        "position_cells_are_not_claimed_physical_free_space": True,
        "phase2_elapsed_seconds": elapsed,
        "score_before_label_separation": {
            "proposal_frozen_before_contributors_opened": True,
            "phase1_builder_has_no_contributor_or_gt_argument": True,
            "target_pose_opened_only_in_this_phase2_evaluator": True,
            "held_labels_used_for_parent_prefix_selection": False,
        },
        "cross_position_orientation_product_materialized": False,
        "raw_support_is_implicit_upper_bound_not_localization_success": True,
        "parent_layout_guide_run": False,
        "query_rows_at_selected_or_max_budget": query_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "content_sha256": report["content_sha256"],
        "query_route": query_route,
        "query_count": report["query_count"],
        "budget_rows": budget_rows,
        "parent_prefix_selection": selection,
        "phase2_elapsed_seconds": elapsed,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
