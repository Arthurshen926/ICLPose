"""Build Top64-parent priority plus global-fallback progressive proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_progressive_global_position_proposal import (
    _atomic_save,
    _load_layout_parent_inventory,
)
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.progressive_global_position_proposal import (
    FALLBACK_FRACTIONS,
    PRIORITY_FALLBACK_POSITION_BUDGETS,
    PRIORITY_FALLBACK_SCHEMA,
    PRIORITY_FALLBACK_SEMANTICS,
    build_priority_fallback_query_arrays,
    load_parent_progressive_order,
    load_priority_fallback_query_proposal,
    load_sparse_global_progressive_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_support", required=True)
    parser.add_argument("--parent_order", required=True)
    parser.add_argument("--global_order", required=True)
    parser.add_argument("--retrieval_summary", action="append", required=True)
    parser.add_argument("--query_route", choices=("seq10", "seq12", "seq14"), required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz).resolve()
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite priority-fallback proposal")
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    support_file_sha = file_sha256(support_path)
    order_path = Path(args.parent_order).resolve()
    parent_arrays, parent_metadata = load_parent_progressive_order(order_path)
    global_order_path = Path(args.global_order).resolve()
    global_order_arrays, global_order_metadata = load_sparse_global_progressive_order(
        global_order_path,
    )
    if (
        parent_metadata.get("global_support", {}).get("file_sha256")
        != support_file_sha
        or parent_metadata.get("global_support", {}).get("content_sha256")
        != global_metadata["content_sha256"]
        or global_order_metadata.get("global_support", {}).get("file_sha256")
        != support_file_sha
        or global_order_metadata.get("global_support", {}).get("content_sha256")
        != global_metadata["content_sha256"]
    ):
        raise ValueError("priority/fallback orders are not bound to global-v2")
    route = str(args.query_route)
    image_ids, parent_ids, parent_scores, summaries, artifacts = (
        _load_layout_parent_inventory(
            [Path(value).resolve() for value in args.retrieval_summary], route,
            str(global_metadata["physical_map"]["content_sha256"]),
            maximum_parent_prefix=64,
        )
    )
    started = time.perf_counter()
    arrays = build_priority_fallback_query_arrays(
        image_ids, parent_ids, parent_scores, parent_arrays, global_order_arrays,
        global_arrays["orientation_rotations_w2c"],
        global_position_count=int(global_arrays["cell_indices_world"].shape[0]),
    )
    elapsed = float(time.perf_counter() - started)
    metadata = {
        "artifact_type": PRIORITY_FALLBACK_SCHEMA,
        "semantics": PRIORITY_FALLBACK_SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "query_route": route,
        "query_count": int(image_ids.size),
        "global_support": {
            "path": str(support_path), "file_sha256": support_file_sha,
            "content_sha256": global_metadata["content_sha256"],
        },
        "global_position_count": int(global_arrays["cell_indices_world"].shape[0]),
        "parent_order": {
            "path": str(order_path), "file_sha256": file_sha256(order_path),
            "content_sha256": parent_metadata["content_sha256"],
        },
        "global_order": {
            "path": str(global_order_path),
            "file_sha256": file_sha256(global_order_path),
            "content_sha256": global_order_metadata["content_sha256"],
        },
        "retrieval_summaries": summaries,
        "retrieval_artifact_inventory": artifacts,
        "parent_priority_prefix": 64,
        "fallback_fractions": [
            {"numerator": value[0], "denominator": value[1]}
            for value in FALLBACK_FRACTIONS
        ],
        "total_position_budgets": list(PRIORITY_FALLBACK_POSITION_BUDGETS),
        "interleave_schedule": "exact_cumulative_floor_rational_word",
        "schedule_exhaustion_policy": "fail_closed_mark_larger_budget_invalid",
        "fallback_queue_excludes_complete_top64_parent_union": True,
        "fallback_fraction_is_outside_parent_union_fraction": True,
        "candidate_cells_strictly_from_global_v2": True,
        "candidate_prefixes_unique": True,
        "position_budget_prefixes_nested_within_fraction": True,
        "parent_retrieval_prioritizes_but_global_fallback_preserves_support": True,
        "implicit_full_completion_tail": (
            "after_stored_exact_schedule_append_remaining_parent_queue_then_"
            "remaining_outside_parent_union_global_queue"
        ),
        "uses_query_image_features": True,
        "uses_query_rgb_directly": False,
        "uses_query_retrieval": True,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_mapping_camera_position_seed": False,
        "orientation_source": "frozen_global_v2_analytic60",
        "position_orientation_cartesian_product_materialized": False,
        "generation_seconds": elapsed,
        "phase2_labels_opened": False,
    }
    _atomic_save(output, arrays, metadata)
    loaded, loaded_metadata = load_priority_fallback_query_proposal(output)
    if (
        loaded_metadata["content_sha256"] != metadata["content_sha256"]
        or arrays_sha256(loaded) != metadata["content_sha256"]
    ):
        raise AssertionError("priority-fallback proposal round-trip differs")
    counts = arrays["candidate_count_by_fallback_fraction"]
    report = {
        "artifact_type": "goal_maplet_priority_fallback_position_proposal_build_v1",
        "proposal": str(output),
        "proposal_file_sha256": file_sha256(output),
        "proposal_content_sha256": metadata["content_sha256"],
        "query_route": route,
        "query_count": int(image_ids.size),
        "candidate_count_by_fallback_fraction": [
            {
                "fallback_fraction": numerator / denominator,
                "minimum": int(np.min(counts[:, index])),
                "median": float(np.median(counts[:, index])),
                "maximum": int(np.max(counts[:, index])),
            }
            for index, (numerator, denominator) in enumerate(FALLBACK_FRACTIONS)
        ],
        "valid_query_count_by_fraction_budget": [
            [
                int(np.sum(counts[:, fraction] >= budget))
                for budget in PRIORITY_FALLBACK_POSITION_BUDGETS
            ]
            for fraction in range(len(FALLBACK_FRACTIONS))
        ],
        "generation_seconds": elapsed,
        "phase1_score_before_label": True,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
