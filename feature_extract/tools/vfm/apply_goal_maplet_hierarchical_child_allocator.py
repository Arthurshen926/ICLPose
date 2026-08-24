"""Apply a frozen route-disjoint parent-balanced child allocator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
)
from feature_extract.tools.vfm.tune_goal_maplet_hierarchical_child_allocator import (
    CONFIG_SCHEMA,
    _content_sha256,
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
from feature_extract.vfm.tokens import compute_file_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", nargs="+", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--allocator_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary_json).resolve()
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite allocated retrieval summary")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    config_path = Path(args.allocator_config).resolve()
    config = _json_without_duplicates(config_path)
    if (
        config.get("artifact_type") != CONFIG_SCHEMA
        or config.get("allocator_semantics") != SEMANTICS
        or str(config.get("physical_map_sha256", "")) != physical.content_sha256
        or str(config.get("content_sha256", "")) != _content_sha256(config)
        or config.get("deployment_uses_gt") is not False
        or config.get("uses_query_pose") is not False
    ):
        raise ValueError("allocator config lineage/contract differs")
    tuning_route = str(config.get("tuning_route", ""))
    if not tuning_route:
        raise ValueError("allocator config lacks a tuning route")
    source_paths = [Path(value).resolve() for value in args.retrieval_summary]
    records: list[dict[str, object]] = []
    source_hashes: list[str] = []
    template: dict[str, object] | None = None
    query_routes: set[str] = set()
    for source_path in source_paths:
        summary = _json_without_duplicates(source_path)
        if summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure RADIO retrieval run")
        if summary.get("promotion_eligible") is not True or summary.get("control_only") is not False:
            raise ValueError("allocator requires a promoted strict retrieval run")
        routes = set(
            str(value)
            for value in summary.get("query_split_audit", {}).get(
                "query_trajectory_ids", []
            )
        )
        if not routes or tuning_route in routes:
            raise ValueError("allocator tuning/query routes overlap")
        query_routes.update(routes)
        records.extend(list(summary.get("rows", [])))
        source_hashes.append(compute_file_sha256(source_path))
        if template is None:
            template = summary
    records.sort(key=lambda value: str(value["image_id"]))
    if not records or len({str(value["image_id"]) for value in records}) != len(records):
        raise ValueError("invalid strict retrieval inventory")
    config_file_hash = compute_file_sha256(config_path)
    output_rows: list[dict[str, object]] = []
    elapsed_values: list[float] = []
    for index, record in enumerate(records):
        source = Path(str(record["artifact"]))
        if compute_file_sha256(source) != str(record["artifact_sha256"]):
            raise ValueError("source retrieval file hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(source)
        if retrieval.content_sha256 != str(record["content_sha256"]):
            raise ValueError("source retrieval content hash differs")
        started = time.perf_counter()
        child_score = aggregate_sparse_token_evidence(
            retrieval.token_xy, retrieval.token_child_rows,
            retrieval.token_child_probabilities,
            entity_count=physical.child_parent_rows.size,
            token_height=int(retrieval.metadata["token_height"]),
            token_width=int(retrieval.metadata["token_width"]),
        )
        allocation = allocate_parent_balanced_scene_children(
            retrieval.scene_parent_ids, retrieval.scene_parent_scores,
            child_score, physical,
            parent_mass_fraction=float(config["parent_mass_fraction"]),
            maximum_children=int(config["maximum_children"]),
            maximum_primitive_iou=float(config["maximum_primitive_iou"]),
        )
        elapsed = float(time.perf_counter() - started)
        elapsed_values.append(elapsed)
        metadata = dict(retrieval.metadata)
        metadata.pop("content_sha256", None)
        metadata.update({
            "base_retrieval_content_sha256": retrieval.content_sha256,
            "base_retrieval_file_sha256": str(record["artifact_sha256"]),
            "scene_child_selection_semantics": SEMANTICS,
            "hierarchical_child_allocator_config_content_sha256": str(
                config["content_sha256"]
            ),
            "hierarchical_child_allocator_config_file_sha256": config_file_hash,
            "hierarchical_child_allocator_tuning_route": tuning_route,
            "hierarchical_child_allocator_parent_mass_fraction": float(
                config["parent_mass_fraction"]
            ),
            "hierarchical_child_allocator_seeded_parent_count": int(
                allocation.seeded_parent_rows.size
            ),
            "hierarchical_child_allocator_represented_parent_count": int(
                allocation.represented_parent_count
            ),
            "children_suppressed_by_primitive_iou_nms": int(
                allocation.suppressed_duplicate_count
            ),
            "promotion_eligible": True,
            "promotion_blockers": [],
            "control_only": False,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
        })
        result = PureRadioPhysicalRetrieval(
            **{
                **retrieval.__dict__,
                "scene_child_rows": allocation.child_rows,
                "scene_child_scores": allocation.child_scores,
                "metadata": metadata,
            }
        )
        output = output_dir / source.name
        if output.exists() and not bool(args.force):
            raise FileExistsError(f"refusing to overwrite {output}")
        result.save_npz(output)
        output_rows.append({
            "image_id": result.image_id,
            "artifact": str(output),
            "artifact_sha256": compute_file_sha256(output),
            "content_sha256": result.content_sha256,
            "base_artifact": str(source.resolve()),
            "base_artifact_sha256": str(record["artifact_sha256"]),
            "allocator_elapsed_seconds": elapsed,
            "seeded_parent_count": int(allocation.seeded_parent_rows.size),
            "represented_parent_count": int(allocation.represented_parent_count),
        })
        print(json.dumps({"index": index + 1, "count": len(records), "image_id": result.image_id}), flush=True)
    assert template is not None
    summary = {
        **{key: value for key, value in template.items() if key != "rows"},
        "query_count": len(output_rows),
        "rows": output_rows,
        "shard_count": 1,
        "shard_index": 0,
        "source_retrieval_summaries": [str(path) for path in source_paths],
        "source_retrieval_summary_sha256": source_hashes,
        "scene_child_selection_semantics": SEMANTICS,
        "hierarchical_child_allocator_config": str(config_path),
        "hierarchical_child_allocator_config_file_sha256": config_file_hash,
        "hierarchical_child_allocator_config_content_sha256": str(
            config["content_sha256"]
        ),
        "hierarchical_child_allocator_tuning_route": tuning_route,
        "query_split_audit": {
            **dict(template.get("query_split_audit", {})),
            "query_trajectory_ids": sorted(query_routes),
            "allocator_tuning_trajectory_ids": [tuning_route],
            "allocator_tuning_query_disjoint": True,
            "blockers": [],
            "disjoint": True,
        },
        "promotion_eligible": True,
        "promotion_blockers": [],
        "control_only": False,
        "allocator_runtime_seconds": {
            "median": float(np.median(elapsed_values)),
            "p90": float(np.quantile(elapsed_values, 0.90)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, summary_path)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
