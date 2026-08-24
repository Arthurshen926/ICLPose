"""Materialize the frozen layout allocator on its seq10 tuning route.

This is an explicitly non-promotable same-route diagnostic.  The production
layout-allocator apply tool intentionally requires tuning/query disjointness;
this separate entry point keeps that boundary intact while allowing a
subsequent seq10-only factor-budget preregistration experiment.  It accepts no
contributor, pose, ground-truth, or label input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
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
from feature_extract.vfm.localization_goal_maplet.layout_child_allocator_config import (
    CONFIG_SCHEMA,
    SCORE_EVIDENCE_PER_AREA,
    SCORE_RAW_EVIDENCE,
    config_content_sha256,
    load_validate_layout_child_allocator_config,
    retrieval_layout_source_signature,
    validate_layout_source_signature,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.scene_child_evidence import (
    SCENE_PARENT_MASK_SEMANTICS,
    aggregate_scene_child_evidence,
)
from feature_extract.vfm.tokens import compute_file_sha256


CONTROL_SEMANTICS = "seq10_same_route_layout_allocator_tuning_control_v1"
CALIBRATION_BLOCKER = "query_route_used_for_validity_calibration"
ALLOCATOR_BLOCKER = "layout_allocator_tuning_route_overlaps_query_route"
CONTROL_BLOCKERS = (CALIBRATION_BLOCKER, ALLOCATOR_BLOCKER)
_REQUIRED_FALSE = (
    "uses_query_pose",
    "uses_query_ground_truth",
    "uses_alike",
    "uses_pnp",
    "uses_sfm_points",
    "uses_sfm_tracks",
    "uses_mapping_rgb",
    "uses_image_retrieval",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", nargs="+", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--allocator_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _atomic_save_retrieval(
    retrieval: PureRadioPhysicalRetrieval, destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp.npz",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        retrieval.save_npz(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def _same_route_split(
    source_split: Mapping[str, object],
) -> dict[str, object]:
    """Return the explicit non-disjoint split attached to every output."""

    return {
        **dict(source_split),
        "query_trajectory_ids": ["seq10"],
        "validity_calibration_fit_trajectory_ids": ["seq10"],
        "allocator_tuning_trajectory_ids": ["seq10"],
        "allocator_tuning_query_disjoint": False,
        "allocator_tuning_same_route_control": True,
        "blockers": list(CONTROL_BLOCKERS),
        "disjoint": False,
    }


def _validate_source_contract(
    summaries: Sequence[Mapping[str, object]],
    summary_paths: Sequence[Path],
    summary_hashes: Sequence[str],
    config: Mapping[str, object],
    *,
    physical_map_sha256: str,
) -> list[dict[str, object]]:
    """Bind the supplied baseline inventory exactly to the signed config."""

    if (
        config.get("artifact_type") != CONFIG_SCHEMA
        or config.get("tuning_route") != "seq10"
        or str(config.get("physical_map_sha256", "")) != physical_map_sha256
        or str(config.get("content_sha256", ""))
        != config_content_sha256(config)
    ):
        raise ValueError("seq10 layout tuning-control config contract differs")
    binding = config.get("tuning_retrieval")
    if not isinstance(binding, Mapping):
        raise ValueError("seq10 layout config lacks its tuning retrieval binding")
    bound_paths = [
        Path(str(value)).resolve()
        for value in binding.get("retrieval_summary_paths", ())
    ]
    bound_hashes = [
        str(value)
        for value in binding.get("retrieval_summary_file_sha256", ())
    ]
    if (
        list(summary_paths) != bound_paths
        or list(summary_hashes) != bound_hashes
        or len(summaries) != len(summary_paths)
    ):
        raise ValueError(
            "supplied retrieval summaries differ from the allocator tuning binding"
        )

    rows: list[dict[str, object]] = []
    for summary in summaries:
        split = summary.get("query_split_audit")
        blockers = list(summary.get("promotion_blockers", ()))
        if (
            summary.get("artifact_type")
            != "goal_maplet_pure_radio_retrieval_run_v1"
            or summary.get("promotion_eligible") is not False
            or summary.get("control_only") is not True
            or blockers != [CALIBRATION_BLOCKER]
            or str(summary.get("physical_map_sha256", ""))
            != physical_map_sha256
            or any(summary.get(flag) is not False for flag in _REQUIRED_FALSE)
            or not isinstance(split, Mapping)
            or list(split.get("query_trajectory_ids", ())) != ["seq10"]
            or list(split.get("validity_calibration_fit_trajectory_ids", ()))
            != ["seq10"]
            or split.get("disjoint") is not False
            or list(split.get("blockers", ())) != [CALIBRATION_BLOCKER]
        ):
            raise ValueError("baseline is not the frozen seq10 calibration control")
        summary_rows = summary.get("rows")
        if (
            not isinstance(summary_rows, list)
            or int(summary.get("query_count", -1)) != len(summary_rows)
        ):
            raise ValueError("seq10 baseline retrieval inventory differs")
        rows.extend(dict(value) for value in summary_rows)
    rows.sort(key=lambda value: str(value.get("image_id", "")))
    image_ids = [str(value.get("image_id", "")) for value in rows]
    if (
        not rows
        or len(set(image_ids)) != len(image_ids)
        or any(not value.startswith("seq10/") for value in image_ids)
        or len(rows) != int(config.get("tuning_query_count", -1))
    ):
        raise ValueError("seq10 tuning-control query inventory differs")
    return rows


def _materialize_query(
    retrieval: PureRadioPhysicalRetrieval,
    *,
    source_file_sha256: str,
    config: Mapping[str, object],
    config_file_sha256: str,
    physical: GoalMapletPhysicalMap,
    child_area_m2: np.ndarray,
) -> tuple[PureRadioPhysicalRetrieval, dict[str, object]]:
    """Apply the frozen allocator without any pose/label interface."""

    source_split = retrieval.metadata.get("query_split_audit")
    if (
        retrieval.metadata.get("promotion_eligible") is not False
        or retrieval.metadata.get("control_only") is not True
        or list(retrieval.metadata.get("promotion_blockers", ()))
        != [CALIBRATION_BLOCKER]
        or not isinstance(source_split, Mapping)
    ):
        raise ValueError("seq10 source artifact control claims differ")
    evidence_semantics = str(config["child_evidence_semantics"])
    score_policy = str(config["child_score_policy"])
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
            child_area_m2,
            out=np.zeros_like(raw_score),
            where=child_area_m2 > 0.0,
        )
    else:
        raise ValueError("unknown frozen layout child score policy")
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
        precomputed_child_surface_area_m2=child_area_m2,
    )
    metadata = dict(retrieval.metadata)
    metadata.pop("content_sha256", None)
    metadata.update({
        "base_retrieval_content_sha256": retrieval.content_sha256,
        "base_retrieval_file_sha256": str(source_file_sha256),
        "scene_child_selection_semantics": ALLOCATOR_SEMANTICS,
        "scene_child_evidence_semantics": evidence_semantics,
        "scene_child_score_policy": score_policy,
        "scene_child_candidate_parent_mask_semantics": (
            SCENE_PARENT_MASK_SEMANTICS
        ),
        "layout_child_allocator_config_content_sha256": str(
            config["content_sha256"]
        ),
        "layout_child_allocator_config_file_sha256": str(config_file_sha256),
        "layout_child_allocator_tuning_route": "seq10",
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
        "seq10_same_route_layout_tuning_control": True,
        "eligible_for_held_route_promotion": False,
        "promotion_eligible": False,
        "promotion_blockers": list(CONTROL_BLOCKERS),
        "control_only": True,
        "query_split_audit": _same_route_split(source_split),
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
    })
    result = PureRadioPhysicalRetrieval(**{
        **retrieval.__dict__,
        "scene_child_rows": allocation.child_rows,
        "scene_child_scores": allocation.child_scores,
        "metadata": metadata,
    })
    return result, {
        "seeded_parent_count": int(allocation.seeded_parent_rows.size),
        "represented_parent_count": int(allocation.represented_parent_count),
        "connected_component_count": int(components.component_count),
        "retained_evidence_fraction_after_parent_mask": float(
            evidence_audit[
                "retained_child_evidence_fraction_after_parent_mask"
            ]
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary_json).resolve()
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite seq10 layout tuning control")

    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    child_area = child_surface_area_m2(physical)
    config_path = Path(args.allocator_config).resolve()
    config, tuning_signature = load_validate_layout_child_allocator_config(
        config_path, physical, physical_path=physical_path,
    )
    config_file_sha256 = compute_file_sha256(config_path)

    source_paths = [Path(value).resolve() for value in args.retrieval_summary]
    source_hashes = [compute_file_sha256(path) for path in source_paths]
    summaries = [_json_without_duplicates(path) for path in source_paths]
    rows = _validate_source_contract(
        summaries,
        source_paths,
        source_hashes,
        config,
        physical_map_sha256=physical.content_sha256,
    )
    template = summaries[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    elapsed_values: list[float] = []
    retained_values: list[float] = []
    component_values: list[int] = []
    output_rows: list[dict[str, object]] = []
    for index, record in enumerate(rows):
        image_id = str(record["image_id"])
        source = Path(str(record["artifact"])).resolve()
        source_file_sha256 = str(record["artifact_sha256"])
        if compute_file_sha256(source) != source_file_sha256:
            raise ValueError("seq10 source retrieval file hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(source)
        if (
            retrieval.image_id != image_id
            or retrieval.content_sha256 != str(record["content_sha256"])
            or retrieval.physical_map_sha256 != physical.content_sha256
        ):
            raise ValueError("seq10 source retrieval identity/content differs")
        validate_layout_source_signature(
            tuning_signature, retrieval_layout_source_signature(retrieval),
        )
        started = time.perf_counter()
        result, audit = _materialize_query(
            retrieval,
            source_file_sha256=source_file_sha256,
            config=config,
            config_file_sha256=config_file_sha256,
            physical=physical,
            child_area_m2=child_area,
        )
        elapsed = float(time.perf_counter() - started)
        destination = output_dir / source.name
        if destination.exists() and not bool(args.force):
            raise FileExistsError(f"refusing to overwrite {destination}")
        _atomic_save_retrieval(result, destination)
        elapsed_values.append(elapsed)
        retained_values.append(float(
            audit["retained_evidence_fraction_after_parent_mask"]
        ))
        component_values.append(int(audit["connected_component_count"]))
        output_rows.append({
            "image_id": image_id,
            "artifact": str(destination),
            "artifact_sha256": compute_file_sha256(destination),
            "content_sha256": result.content_sha256,
            "base_artifact": str(source),
            "base_artifact_sha256": source_file_sha256,
            "allocator_elapsed_seconds": elapsed,
            **audit,
        })
        print(json.dumps({
            "index": index + 1,
            "count": len(rows),
            "image_id": image_id,
        }), flush=True)

    split = _same_route_split(
        dict(template.get("query_split_audit", {}))
    )
    summary: dict[str, object] = {
        **{key: value for key, value in template.items() if key != "rows"},
        "query_count": len(output_rows),
        "rows": output_rows,
        "shard_count": 1,
        "shard_index": 0,
        "method": (
            "full_radio_retrieval_plus_seq10_same_route_layout_allocator_"
            "tuning_control"
        ),
        "source_retrieval_summaries": [str(path) for path in source_paths],
        "source_retrieval_summary_sha256": source_hashes,
        "scene_child_selection_semantics": ALLOCATOR_SEMANTICS,
        "scene_child_evidence_semantics": str(
            config["child_evidence_semantics"]
        ),
        "scene_child_score_policy": str(config["child_score_policy"]),
        "scene_child_candidate_parent_mask_semantics": (
            SCENE_PARENT_MASK_SEMANTICS
        ),
        "layout_child_allocator_config": str(config_path),
        "layout_child_allocator_config_file_sha256": config_file_sha256,
        "layout_child_allocator_config_content_sha256": str(
            config["content_sha256"]
        ),
        "layout_child_allocator_tuning_route": "seq10",
        "seq10_same_route_layout_tuning_control": True,
        "eligible_for_held_route_promotion": False,
        "query_split_audit": split,
        "promotion_eligible": False,
        "promotion_blockers": list(CONTROL_BLOCKERS),
        "control_only": True,
        "allocator_runtime_seconds": {
            "total": float(np.sum(elapsed_values)),
            "median": float(np.median(elapsed_values)),
            "p90": float(np.quantile(elapsed_values, 0.90)),
        },
        "layout_support_diagnostic_mean": {
            "retained_evidence_fraction_after_parent_mask": float(
                np.mean(retained_values)
            ),
            "connected_component_count": float(np.mean(component_values)),
        },
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_mapping_rgb": False,
        "uses_image_retrieval": False,
    }
    _atomic_json(summary, summary_path)
    print(json.dumps({
        key: value for key, value in summary.items() if key != "rows"
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
