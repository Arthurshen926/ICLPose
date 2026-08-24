"""Freeze a low-capacity parent-balanced child allocator on one route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _contributor_inventory,
    _json_without_duplicates,
    _load_contributor,
)
from feature_extract.vfm.localization_goal_maplet.hierarchical_child_allocator import (
    SEMANTICS,
    allocate_parent_balanced_scene_children,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
    aggregate_sparse_token_evidence,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    PhysicalIncidence,
    _primitive_union_for_children,
    _surface_set_metrics,
    remap_pinhole_contributors_to_raw_grid,
    token_primitive_visibility,
)
from feature_extract.vfm.tokens import compute_file_sha256


CONFIG_SCHEMA = "goal_maplet_hierarchical_child_allocator_config_v1"
REPORT_SCHEMA = "goal_maplet_hierarchical_child_allocator_tuning_v1"


def _content_sha256(payload: dict[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", nargs="+", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--tuning_route", required=True)
    parser.add_argument(
        "--parent_mass_fractions", nargs="+", type=float,
        default=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0],
    )
    parser.add_argument("--maximum_children", type=int, default=64)
    parser.add_argument("--maximum_primitive_iou", type=float, default=0.50)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.output_config)
    report_path = Path(args.output_report)
    if (config_path.exists() or report_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite allocator tuning artifacts")
    fractions = tuple(sorted({float(value) for value in args.parent_mass_fractions}))
    if not fractions or fractions[0] < 0.0 or fractions[-1] > 1.0:
        raise ValueError("parent mass fractions must lie in [0,1]")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    incidence = PhysicalIncidence.from_physical_map(physical)
    summaries = [Path(value).resolve() for value in args.retrieval_summary]
    records: list[dict[str, object]] = []
    for summary_path in summaries:
        summary = _json_without_duplicates(summary_path)
        if summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure RADIO retrieval run")
        routes = list(summary.get("query_split_audit", {}).get("query_trajectory_ids", []))
        if routes != [str(args.tuning_route)]:
            raise ValueError("tuning retrieval route differs")
        if summary.get("control_only") is not True:
            raise ValueError("route-overlap tuning retrieval must be sealed control-only")
        for key in (
            "uses_query_pose", "uses_query_ground_truth", "uses_alike", "uses_pnp",
            "uses_sfm_points", "uses_sfm_tracks", "uses_mapping_rgb",
            "uses_image_retrieval",
        ):
            if summary.get(key) is not False:
                raise ValueError(f"tuning retrieval requires {key}=false")
        records.extend(list(summary.get("rows", [])))
    records.sort(key=lambda value: str(value["image_id"]))
    if not records or len({str(value["image_id"]) for value in records}) != len(records):
        raise ValueError("invalid tuning query inventory")
    contributors = _contributor_inventory(Path(args.contributors))
    values: dict[float, list[dict[str, float]]] = {fraction: [] for fraction in fractions}
    for index, record in enumerate(records):
        image_id = str(record["image_id"])
        retrieval_path = Path(str(record["artifact"]))
        if compute_file_sha256(retrieval_path) != str(record["artifact_sha256"]):
            raise ValueError("tuning retrieval file hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        if retrieval.content_sha256 != str(record["content_sha256"]):
            raise ValueError("tuning retrieval content hash differs")
        contributor = contributors.get(image_id)
        if contributor is None:
            raise ValueError(f"missing tuning contributor for {image_id}")
        labels, model, width, height, params, _ = _load_contributor(contributor)
        raw_labels, _ = remap_pinhole_contributors_to_raw_grid(
            labels, camera_model_id=model, camera_width=width,
            camera_height=height, camera_params=params,
        )
        token_primitive, _, _ = token_primitive_visibility(
            raw_labels, physical,
            token_height=int(retrieval.metadata["token_height"]),
            token_width=int(retrieval.metadata["token_width"]),
        )
        scene_mass = np.asarray(token_primitive.sum(axis=0)).reshape(-1)
        child_truth = np.asarray(
            (token_primitive @ incidence.primitive_to_child).sum(axis=0)
        ).reshape(-1)
        total = max(float(np.sum(child_truth)), 1e-12)
        child_score = aggregate_sparse_token_evidence(
            retrieval.token_xy, retrieval.token_child_rows,
            retrieval.token_child_probabilities,
            entity_count=physical.child_parent_rows.size,
            token_height=int(retrieval.metadata["token_height"]),
            token_width=int(retrieval.metadata["token_width"]),
        )
        visible_parent = np.bincount(
            physical.child_parent_rows, weights=child_truth,
            minlength=physical.maplet_ids.size,
        ) > 0.0
        for fraction in fractions:
            allocation = allocate_parent_balanced_scene_children(
                retrieval.scene_parent_ids, retrieval.scene_parent_scores,
                child_score, physical, parent_mass_fraction=fraction,
                maximum_children=int(args.maximum_children),
                maximum_primitive_iou=float(args.maximum_primitive_iou),
            )
            selected = allocation.child_rows
            exact = float(np.sum(child_truth[selected]) / total)
            represented_parent = np.zeros(physical.maplet_ids.shape, dtype=bool)
            represented_parent[physical.child_parent_rows[selected]] = True
            same_parent = float(
                np.sum(child_truth[represented_parent[physical.child_parent_rows]])
                / total
            )
            surface = _surface_set_metrics(
                _primitive_union_for_children(physical, selected),
                scene_mass, physical, tolerances_m=(0.25, 0.5, 1.0),
            )
            values[fraction].append({
                "exact_visible_mass_recall": exact,
                "same_parent_visible_mass_recall": same_parent,
                "tolerant_visible_mass_recall_0.25m": float(
                    surface["tolerant_visible_mass_recall_0.25m"]
                ),
                "tolerant_visible_mass_recall_0.5m": float(
                    surface["tolerant_visible_mass_recall_0.5m"]
                ),
                "tolerant_visible_mass_recall_1m": float(
                    surface["tolerant_visible_mass_recall_1m"]
                ),
                "represented_parent_count": float(allocation.represented_parent_count),
                "selected_child_on_visible_parent_fraction_oracle": float(
                    np.mean(visible_parent[physical.child_parent_rows[selected]])
                ),
            })
        print(json.dumps({"index": index + 1, "count": len(records), "image_id": image_id}), flush=True)
    grid: dict[str, object] = {}
    for fraction, rows in values.items():
        grid[f"{fraction:.6g}"] = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
    # Region retrieval is explicitly high-tolerance: freeze by 0.5 m recall,
    # with exact recall and the lower-complexity fraction as deterministic ties.
    best = max(
        fractions,
        key=lambda fraction: (
            float(grid[f"{fraction:.6g}"]["tolerant_visible_mass_recall_0.5m"]),
            float(grid[f"{fraction:.6g}"]["exact_visible_mass_recall"]),
            -fraction,
        ),
    )
    binding = {
        "retrieval_summary_paths": [str(path) for path in summaries],
        "retrieval_summary_file_sha256": [compute_file_sha256(path) for path in summaries],
    }
    config: dict[str, object] = {
        "artifact_type": CONFIG_SCHEMA,
        "allocator_semantics": SEMANTICS,
        "parent_mass_fraction": float(best),
        "maximum_children": int(args.maximum_children),
        "maximum_primitive_iou": float(args.maximum_primitive_iou),
        "physical_map_sha256": physical.content_sha256,
        "physical_map_file_sha256": compute_file_sha256(physical_path),
        "tuning_route": str(args.tuning_route),
        "tuning_query_count": len(records),
        "tuning_retrieval": binding,
        "selection_objective": "mean_tolerant_visible_mass_recall_0.5m_then_exact_then_minimum_fraction_v1",
        "uses_gt_only_for_offline_route_disjoint_config_selection": True,
        "deployment_uses_gt": False,
        "uses_query_pose": False,
        "uses_alike": False,
        "uses_pnp": False,
    }
    config["content_sha256"] = _content_sha256(config)
    report: dict[str, object] = {
        "artifact_type": REPORT_SCHEMA,
        "config": config,
        "grid": grid,
        "selected_parent_mass_fraction": float(best),
        "tuning_route": str(args.tuning_route),
        "query_count": len(records),
        "claim_scope": {
            "tuning_only": True,
            "held_routes_not_opened": True,
            "not_pose_estimation": True,
        },
    }
    report["content_sha256"] = _content_sha256(report)
    for path, payload in ((config_path, config), (report_path, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    print(json.dumps({"config": config, "report_content_sha256": report["content_sha256"], "grid": grid}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
