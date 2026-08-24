"""Apply a frozen route-disjoint token-layout child allocator."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
)
from feature_extract.vfm.localization_goal_maplet.layout_child_allocator_config import (
    CONFIG_SCHEMA,
    SCORE_EVIDENCE_PER_AREA,
    SCORE_POLICIES,
    SCORE_RAW_EVIDENCE,
    config_content_sha256,
    load_validate_layout_child_allocator_config,
    retrieval_layout_source_signature,
    validate_layout_source_signature,
)
from feature_extract.vfm.localization_goal_maplet.connected_fine_support import (
    connected_fine_support_components,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    child_surface_area_m2,
)
from feature_extract.vfm.localization_goal_maplet.hierarchical_child_allocator import (
    SEMANTICS as ALLOCATOR_SEMANTICS,
    allocate_parent_balanced_scene_children,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.scene_child_evidence import (
    SCENE_CHILD_EVIDENCE_SEMANTICS,
    SCENE_PARENT_MASK_SEMANTICS,
    aggregate_scene_child_evidence,
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
        raise FileExistsError("refusing to overwrite layout retrieval summary")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    child_area = child_surface_area_m2(physical)
    config_path = Path(args.allocator_config).resolve()
    config, tuning_signature = load_validate_layout_child_allocator_config(
        config_path,
        physical,
        physical_path=physical_path,
    )
    evidence_semantics = str(config.get("child_evidence_semantics", ""))
    score_policy = str(config.get("child_score_policy", ""))
    if (
        config.get("artifact_type") != CONFIG_SCHEMA
        or config.get("allocator_semantics") != ALLOCATOR_SEMANTICS
        or evidence_semantics not in SCENE_CHILD_EVIDENCE_SEMANTICS
        or score_policy not in SCORE_POLICIES
        or config.get("candidate_parent_mask_semantics")
        != SCENE_PARENT_MASK_SEMANTICS
        or str(config.get("physical_map_sha256", "")) != physical.content_sha256
        or str(config.get("content_sha256", ""))
        != config_content_sha256(config)
        or config.get("deployment_uses_gt") is not False
        or config.get("uses_query_pose") is not False
    ):
        raise ValueError("layout allocator config lineage/contract differs")
    tuning_route = str(config.get("tuning_route", ""))
    if not tuning_route:
        raise ValueError("layout allocator config lacks a tuning route")

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
            raise ValueError("layout allocator requires a promoted strict run")
        routes = set(
            str(value)
            for value in summary.get("query_split_audit", {}).get(
                "query_trajectory_ids", []
            )
        )
        if not routes or tuning_route in routes:
            raise ValueError("layout allocator tuning/query routes overlap")
        query_routes.update(routes)
        records.extend(list(summary.get("rows", [])))
        source_hashes.append(compute_file_sha256(source_path))
        if template is None:
            template = summary
    records.sort(key=lambda value: str(value["image_id"]))
    if not records or len({str(value["image_id"]) for value in records}) != len(records):
        raise ValueError("invalid strict layout retrieval inventory")

    first_source = Path(str(records[0]["artifact"]))
    if compute_file_sha256(first_source) != str(records[0]["artifact_sha256"]):
        raise ValueError("first source retrieval file hash differs")
    first_retrieval = PureRadioPhysicalRetrieval.load_npz(first_source)
    validate_layout_source_signature(
        tuning_signature,
        retrieval_layout_source_signature(first_retrieval),
    )

    config_file_hash = compute_file_sha256(config_path)
    output_rows: list[dict[str, object]] = []
    elapsed_values: list[float] = []
    retained_values: list[float] = []
    component_values: list[int] = []
    for index, record in enumerate(records):
        source = Path(str(record["artifact"]))
        if compute_file_sha256(source) != str(record["artifact_sha256"]):
            raise ValueError("source retrieval file hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(source)
        if retrieval.content_sha256 != str(record["content_sha256"]):
            raise ValueError("source retrieval content hash differs")
        started = time.perf_counter()
        raw_score, evidence_audit = aggregate_scene_child_evidence(
            retrieval.token_xy,
            retrieval.token_child_rows,
            retrieval.token_child_probabilities,
            physical,
            token_height=int(retrieval.metadata["token_height"]),
            token_width=int(retrieval.metadata["token_width"]),
            semantics=evidence_semantics,
            local_block_size=int(config["child_evidence_local_block_size"]),
            scene_parent_ids=retrieval.scene_parent_ids,
        )
        if score_policy == SCORE_RAW_EVIDENCE:
            allocation_score = raw_score
        elif score_policy == SCORE_EVIDENCE_PER_AREA:
            allocation_score = np.divide(
                raw_score,
                child_area,
                out=np.zeros_like(raw_score),
                where=child_area > 0.0,
            )
        else:  # guarded above; retained for fail-closed readability.
            raise ValueError("unknown child score policy")
        allocation = allocate_parent_balanced_scene_children(
            retrieval.scene_parent_ids,
            retrieval.scene_parent_scores,
            allocation_score,
            physical,
            parent_mass_fraction=float(config["parent_mass_fraction"]),
            maximum_children=int(config["maximum_children"]),
            maximum_primitive_iou=float(config["maximum_primitive_iou"]),
        )
        components = connected_fine_support_components(
            allocation.child_rows,
            physical,
            maximum_normal_angle_degrees=float(
                config["maximum_normal_angle_degrees"]
            ),
            precomputed_child_surface_area_m2=child_area,
        )
        elapsed = float(time.perf_counter() - started)
        elapsed_values.append(elapsed)
        retained_values.append(float(
            evidence_audit["retained_child_evidence_fraction_after_parent_mask"]
        ))
        component_values.append(int(components.component_count))
        metadata = dict(retrieval.metadata)
        metadata.pop("content_sha256", None)
        metadata.update({
            "base_retrieval_content_sha256": retrieval.content_sha256,
            "base_retrieval_file_sha256": str(record["artifact_sha256"]),
            "scene_child_selection_semantics": ALLOCATOR_SEMANTICS,
            "scene_child_evidence_semantics": evidence_semantics,
            "scene_child_score_policy": score_policy,
            "scene_child_candidate_parent_mask_semantics": (
                SCENE_PARENT_MASK_SEMANTICS
            ),
            "layout_child_allocator_config_content_sha256": str(
                config["content_sha256"]
            ),
            "layout_child_allocator_config_file_sha256": config_file_hash,
            "layout_child_allocator_tuning_route": tuning_route,
            "layout_child_allocator_parent_mass_fraction": float(
                config["parent_mass_fraction"]
            ),
            "layout_child_allocator_seeded_parent_count": int(
                allocation.seeded_parent_rows.size
            ),
            "layout_child_allocator_represented_parent_count": int(
                allocation.represented_parent_count
            ),
            "layout_child_connected_component_count": int(
                components.component_count
            ),
            "layout_child_retained_evidence_fraction_after_parent_mask": float(
                evidence_audit[
                    "retained_child_evidence_fraction_after_parent_mask"
                ]
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
            "connected_component_count": int(components.component_count),
            "retained_evidence_fraction_after_parent_mask": float(
                evidence_audit[
                    "retained_child_evidence_fraction_after_parent_mask"
                ]
            ),
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
        "scene_child_selection_semantics": ALLOCATOR_SEMANTICS,
        "scene_child_evidence_semantics": evidence_semantics,
        "scene_child_score_policy": score_policy,
        "scene_child_candidate_parent_mask_semantics": (
            SCENE_PARENT_MASK_SEMANTICS
        ),
        "layout_child_allocator_config": str(config_path),
        "layout_child_allocator_config_file_sha256": config_file_hash,
        "layout_child_allocator_config_content_sha256": str(
            config["content_sha256"]
        ),
        "layout_child_allocator_tuning_route": tuning_route,
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
        "layout_support_diagnostic_mean": {
            "retained_evidence_fraction_after_parent_mask": float(
                np.mean(retained_values)
            ),
            "connected_component_count": float(np.mean(component_values)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, summary_path)
    print(json.dumps({
        key: value for key, value in summary.items() if key != "rows"
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
