"""Seq10 Phase-2 absolute position ceiling of complete parent-box unions."""

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
    load_parent_cropped_ceiling,
)


SCHEMA = "goal_maplet_parent_cropped_position_ceiling_seq10_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_inventory", required=True)
    parser.add_argument("--global_support", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--official_protocol", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--contributor_audit", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent-cropped ceiling report")
    started = time.perf_counter()
    inventory_path = Path(args.candidate_inventory).resolve()
    arrays, metadata = load_parent_cropped_ceiling(inventory_path)
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    if (
        metadata.get("global_support", {}).get("file_sha256")
        != file_sha256(support_path)
        or metadata.get("global_support", {}).get("content_sha256")
        != global_metadata["content_sha256"]
    ):
        raise ValueError("parent-cropped ceiling global-v2 binding differs")
    protocol_path = Path(args.official_protocol).resolve()
    audit_path = Path(args.contributor_audit).resolve()
    contributor_dir = Path(args.contributors).resolve()
    protocol, contributor_audit = _validate_protocol_and_audit(
        protocol_path, audit_path, contributor_dir,
    )
    token_manifest = Path(args.token_manifest).resolve()
    image_ids = _query_ids_from_pose_free_manifest(token_manifest, "seq10", protocol)
    if image_ids != arrays["image_ids"].tolist():
        raise ValueError("parent-cropped ceiling query inventory differs")
    target, contributor_bindings = _load_route_poses(
        contributor_dir, image_ids, "seq10", contributor_audit,
    )
    centers = np.stack([camera_center_from_pose_w2c(pose) for pose in target])
    origin = np.asarray(global_arrays["lattice_origin_world"], dtype=np.float64)
    spacing = float(global_arrays["lattice_spacing_m"])
    positions = origin[None] + (
        np.asarray(global_arrays["cell_indices_world"], dtype=np.float64) + 0.5
    ) * spacing
    budgets = arrays["parent_prefix_budgets"].tolist()
    offsets = arrays["candidate_offsets"]
    rows = arrays["candidate_cell_rows"]
    grid = []
    for budget_index, budget in enumerate(budgets):
        nearest = np.full((len(image_ids),), np.inf, dtype=np.float64)
        counts = np.zeros((len(image_ids),), dtype=np.int64)
        for query in range(len(image_ids)):
            group = query * len(budgets) + budget_index
            start, end = int(offsets[group]), int(offsets[group + 1])
            local = rows[start:end]
            counts[query] = local.size
            nearest[query] = np.sqrt(np.min(np.sum(
                (positions[local] - centers[query, None]) ** 2, axis=1,
            )))
        hit = nearest <= 2.0
        grid.append({
            "parent_prefix_budget": int(budget),
            "complete_union_position_hits_2m": int(np.sum(hit)),
            "complete_union_position_misses_2m": int(np.sum(~hit)),
            "complete_union_position_rate_2m": float(np.mean(hit)),
            "best_translation_m": _stats(nearest),
            "position_count": _stats(counts),
            "implicit_pose_factor_count": _stats(counts * ORIENTATION_COUNT),
        })
    required = int(math.ceil(0.95 * len(image_ids) - 1e-12))
    top64_hits = int(grid[-1]["complete_union_position_hits_2m"])
    decision = "GO_TO_TOP64_PROGRESSIVE_BUDGET_GRID" if top64_hits >= required else "KILL"
    report = {
        "artifact_type": SCHEMA,
        "query_route": "seq10",
        "query_count": len(image_ids),
        "candidate_inventory": str(inventory_path),
        "candidate_inventory_file_sha256": file_sha256(inventory_path),
        "candidate_inventory_content_sha256": metadata["content_sha256"],
        "global_support_file_sha256": file_sha256(support_path),
        "global_support_content_sha256": global_metadata["content_sha256"],
        "contributor_inventory": contributor_bindings,
        "complete_parent_union_ceiling_by_prefix": grid,
        "top64_gate": {
            "metric": "complete_top64_parent_box_union_position_acquisition_2m",
            "threshold": 0.95,
            "required_hits": required,
            "observed_hits": top64_hits,
            "decision": decision,
            "held_route_pose_labels_opened": False,
        },
        "score_before_label_separation": {
            "candidate_inventory_frozen_before_seq10_pose_files_opened": True,
            "phase1_builder_has_no_contributor_or_gt_argument": True,
            "held_pose_files_opened": False,
        },
        "parent_retrieval_crops_global_v2_support": True,
        "ceiling_is_independent_of_progressive_cell_order": True,
        "orientation_not_part_of_position_gate": True,
        "ranking_performed": False,
        "phase2_elapsed_seconds": float(time.perf_counter() - started),
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output),
        "content_sha256": report["content_sha256"],
        "complete_parent_union_ceiling_by_prefix": grid,
        "top64_gate": report["top64_gate"],
        "phase2_elapsed_seconds": report["phase2_elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
