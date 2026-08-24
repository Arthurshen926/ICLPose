"""Build pose-free layout-parent progressive prefixes over global-v2 cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.progressive_global_position_proposal import (
    MAXIMUM_PARENT_PREFIX,
    PARENT_PREFIX_BUDGETS,
    POSITION_BUDGETS,
    QUERY_PROPOSAL_SCHEMA,
    QUERY_PROPOSAL_SEMANTICS,
    build_progressive_query_proposal_arrays,
    load_parent_progressive_order,
    load_progressive_query_proposal,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)


FROZEN_LAYOUT_CONFIG_CONTENT_SHA256 = (
    "aba64eadf02cfd7668b8c722b1b155863a8af2b580e8d1ebe39b50cb0d375341"
)
FROZEN_LAYOUT_CONFIG_FILE_SHA256 = (
    "c5df2bd082ae879393b75e64056821f04dbb9d18710bc77792b2c4dcad2bfd68"
)
FORBIDDEN_TRUE_FLAGS = (
    "uses_alike", "uses_image_retrieval", "uses_mapping_rgb", "uses_pnp",
    "uses_query_ground_truth", "uses_query_pose", "uses_sfm_points",
    "uses_sfm_tracks",
)
EXPECTED_QUERY_COUNTS = {"seq10": 88, "seq12": 188, "seq14": 36}


def _load_json_no_duplicates(path: Path) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON member: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("layout retrieval summary is not a JSON object")
    return value


def _atomic_save(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_layout_parent_inventory(
    summary_paths: list[Path], route: str, physical_content_sha256: str,
    *, maximum_parent_prefix: int = MAXIMUM_PARENT_PREFIX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    parent_prefix = int(maximum_parent_prefix)
    if parent_prefix <= 0 or parent_prefix > 64:
        raise ValueError("layout parent inventory prefix differs")
    records, summary_bindings = [], []
    summary_split_audit = None
    for path in summary_paths:
        run = _load_json_no_duplicates(path)
        audit = run.get("query_split_audit", {})
        if (
            run.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1"
            or run.get("physical_map_sha256") != physical_content_sha256
            or set(map(str, audit.get("query_trajectory_ids", ()))) != {route}
            or run.get("layout_child_allocator_config_content_sha256")
            != FROZEN_LAYOUT_CONFIG_CONTENT_SHA256
            or run.get("layout_child_allocator_config_file_sha256")
            != FROZEN_LAYOUT_CONFIG_FILE_SHA256
            or run.get("layout_child_allocator_tuning_route") != "seq10"
            or any(run.get(flag) is not False for flag in FORBIDDEN_TRUE_FLAGS)
        ):
            raise ValueError("layout retrieval dependency contract differs")
        blockers = list(run.get("promotion_blockers", ()))
        if route == "seq10":
            if (
                run.get("promotion_eligible") is not False
                or run.get("control_only") is not True
                or blockers != [
                    "query_route_used_for_validity_calibration",
                    "layout_allocator_tuning_route_overlaps_query_route",
                ]
                or audit.get("allocator_tuning_same_route_control") is not True
            ):
                raise ValueError("seq10 layout retrieval is not the frozen control")
        elif (
            run.get("promotion_eligible") is not True
            or run.get("control_only") is not False
            or blockers or audit.get("disjoint") is not True
            or audit.get("allocator_tuning_query_disjoint") is not True
        ):
            raise ValueError("held layout retrieval is not strict promotion eligible")
        local_records = list(run.get("rows", ()))
        if int(run.get("query_count", -1)) != len(local_records):
            raise ValueError("layout retrieval summary query count differs")
        records.extend(local_records)
        if summary_split_audit is None:
            summary_split_audit = audit
        elif summary_split_audit != audit:
            raise ValueError("layout retrieval shards have different split audits")
        summary_bindings.append({
            "path": str(path), "file_sha256": file_sha256(path),
        })
    records.sort(key=lambda row: str(row.get("image_id", "")))
    image_ids = [str(row.get("image_id", "")) for row in records]
    if (
        not records or len(set(image_ids)) != len(records)
        or len(records) != EXPECTED_QUERY_COUNTS[route]
        or any(not value.startswith(route + "/") for value in image_ids)
    ):
        raise ValueError("layout retrieval query inventory differs")
    parent_ids, parent_scores, artifact_bindings = [], [], []
    for record in records:
        artifact = Path(str(record.get("artifact", ""))).resolve()
        if file_sha256(artifact) != str(record.get("artifact_sha256", "")):
            raise ValueError("layout retrieval artifact bytes differ")
        retrieval = PureRadioPhysicalRetrieval.load_npz(artifact)
        artifact_split_audit = dict(retrieval.metadata.get("query_split_audit", {}))
        split_core_exclusions = {
            "allocator_tuning_query_disjoint",
            "allocator_tuning_trajectory_ids",
            "allocator_tuning_same_route_control",
        }
        artifact_split_core = {
            key: value for key, value in artifact_split_audit.items()
            if key not in split_core_exclusions
        }
        summary_split_core = {
            key: value for key, value in summary_split_audit.items()
            if key not in split_core_exclusions
        }
        ids = np.asarray(
            retrieval.scene_parent_ids[:parent_prefix], dtype=np.int64,
        )
        scores = np.asarray(
            retrieval.scene_parent_scores[:parent_prefix], dtype=np.float64,
        )
        tied_out_of_order = any(
            scores[index] == scores[index + 1] and ids[index] > ids[index + 1]
            for index in range(parent_prefix - 1)
        )
        if (
            retrieval.content_sha256 != str(record.get("content_sha256", ""))
            or retrieval.image_id != str(record.get("image_id", ""))
            or retrieval.physical_map_sha256 != physical_content_sha256
            or retrieval.metadata.get("layout_child_allocator_config_content_sha256")
            != FROZEN_LAYOUT_CONFIG_CONTENT_SHA256
            or retrieval.metadata.get("layout_child_allocator_config_file_sha256")
            != FROZEN_LAYOUT_CONFIG_FILE_SHA256
            or retrieval.metadata.get("layout_child_allocator_tuning_route") != "seq10"
            or artifact_split_core != summary_split_core
            or ids.shape != (parent_prefix,)
            or np.unique(ids).size != parent_prefix
            or np.any(~np.isfinite(scores)) or np.any(scores <= 0.0)
            or np.any(np.diff(scores) > 1e-12) or tied_out_of_order
        ):
            raise ValueError("layout scene-parent ranked evidence differs")
        parent_ids.append(ids)
        parent_scores.append(scores)
        artifact_bindings.append({
            "image_id": retrieval.image_id,
            "file_sha256": str(record["artifact_sha256"]),
            "content_sha256": retrieval.content_sha256,
        })
    return (
        np.asarray(image_ids), np.stack(parent_ids), np.stack(parent_scores),
        summary_bindings, artifact_bindings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_support", required=True)
    parser.add_argument("--parent_order", required=True)
    parser.add_argument("--retrieval_summary", action="append", required=True)
    parser.add_argument("--query_route", choices=("seq10", "seq12", "seq14"), required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz).resolve()
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite progressive query proposal")
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    order_path = Path(args.parent_order).resolve()
    order_arrays, order_metadata = load_parent_progressive_order(order_path)
    if (
        order_metadata.get("global_support", {}).get("file_sha256")
        != file_sha256(support_path)
        or order_metadata.get("global_support", {}).get("content_sha256")
        != global_metadata["content_sha256"]
        or not np.array_equal(order_arrays["maplet_ids"], global_arrays["maplet_ids"])
    ):
        raise ValueError("progressive parent order is not bound to global-v2")
    route = str(args.query_route)
    summary_paths = [Path(value).resolve() for value in args.retrieval_summary]
    image_ids, parent_ids, parent_scores, summaries, artifacts = (
        _load_layout_parent_inventory(
            summary_paths, route,
            str(global_metadata["physical_map"]["content_sha256"]),
        )
    )
    started = time.perf_counter()
    arrays = build_progressive_query_proposal_arrays(
        image_ids, parent_ids, parent_scores, order_arrays,
        global_arrays["orientation_rotations_w2c"],
        global_position_count=int(global_arrays["cell_indices_world"].shape[0]),
    )
    elapsed = float(time.perf_counter() - started)
    metadata: dict[str, object] = {
        "artifact_type": QUERY_PROPOSAL_SCHEMA,
        "semantics": QUERY_PROPOSAL_SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "query_route": route,
        "query_count": int(image_ids.size),
        "global_support": {
            "path": str(support_path), "file_sha256": file_sha256(support_path),
            "content_sha256": global_metadata["content_sha256"],
        },
        "global_position_count": int(global_arrays["cell_indices_world"].shape[0]),
        "parent_order": {
            "path": str(order_path), "file_sha256": file_sha256(order_path),
            "content_sha256": order_metadata["content_sha256"],
        },
        "retrieval_summaries": summaries,
        "retrieval_artifact_inventory": artifacts,
        "layout_allocator_config_content_sha256": FROZEN_LAYOUT_CONFIG_CONTENT_SHA256,
        "layout_allocator_config_file_sha256": FROZEN_LAYOUT_CONFIG_FILE_SHA256,
        "parent_prefix_budgets": list(PARENT_PREFIX_BUDGETS),
        "total_position_budgets": list(POSITION_BUDGETS),
        "maximum_position_budget": max(POSITION_BUDGETS),
        "allocation": (
            "equal_ranked_round_robin_in_scene_parent_evidence_order_"
            "skip_cross_parent_duplicate_cells"
        ),
        "raw_parent_scores_used_as_probabilities": False,
        "candidate_cells_strictly_from_global_v2": True,
        "position_budget_prefixes_nested_within_parent_prefix": True,
        "parent_prefix_domains_required_nested": False,
        "uses_query_image_features": True,
        "uses_query_rgb_directly": False,
        "uses_query_retrieval": True,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_mapping_camera_position_seed": False,
        "orientation_count": int(global_arrays["orientation_rotations_w2c"].shape[0]),
        "orientation_source": "frozen_global_v2_analytic60",
        "position_orientation_cartesian_product_materialized": False,
        "generation_seconds": elapsed,
        "phase2_labels_opened": False,
        "ranking_beyond_scene_parent_order_performed": False,
    }
    _atomic_save(output, arrays, metadata)
    loaded, loaded_metadata = load_progressive_query_proposal(output)
    if (
        loaded_metadata["content_sha256"] != metadata["content_sha256"]
        or arrays_sha256(loaded) != metadata["content_sha256"]
    ):
        raise AssertionError("progressive query proposal round-trip differs")
    report = {
        "artifact_type": "goal_maplet_progressive_global_position_proposal_build_v1",
        "proposal": str(output),
        "proposal_file_sha256": file_sha256(output),
        "proposal_content_sha256": metadata["content_sha256"],
        "query_route": route,
        "query_count": int(image_ids.size),
        "stored_candidate_rows": int(arrays["candidate_cell_rows"].size),
        "generation_seconds": elapsed,
        "score_before_label_phase1": True,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
