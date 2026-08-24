"""Freeze complete Top4/8/16/32/64 parent-box unions before seq10 labels."""

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
    PARENT_CEILING_SCHEMA,
    PARENT_CEILING_SEMANTICS,
    build_parent_cropped_ceiling_arrays,
    load_parent_cropped_ceiling,
    load_parent_progressive_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_support", required=True)
    parser.add_argument("--parent_order", required=True)
    parser.add_argument("--retrieval_summary", action="append", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz).resolve()
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent-cropped ceiling")
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    order_path = Path(args.parent_order).resolve()
    order_arrays, order_metadata = load_parent_progressive_order(order_path)
    if (
        order_metadata.get("global_support", {}).get("file_sha256")
        != file_sha256(support_path)
        or order_metadata.get("global_support", {}).get("content_sha256")
        != global_metadata["content_sha256"]
    ):
        raise ValueError("parent order is not bound to global-v2")
    summary_paths = [Path(value).resolve() for value in args.retrieval_summary]
    image_ids, parent_ids, parent_scores, summaries, artifacts = (
        _load_layout_parent_inventory(
            summary_paths, "seq10",
            str(global_metadata["physical_map"]["content_sha256"]),
            maximum_parent_prefix=64,
        )
    )
    started = time.perf_counter()
    arrays = build_parent_cropped_ceiling_arrays(
        image_ids, parent_ids, parent_scores, order_arrays,
        global_position_count=int(global_arrays["cell_indices_world"].shape[0]),
    )
    elapsed = float(time.perf_counter() - started)
    metadata: dict[str, object] = {
        "artifact_type": PARENT_CEILING_SCHEMA,
        "semantics": PARENT_CEILING_SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "query_route": "seq10",
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
        "parent_prefix_budgets": [4, 8, 16, 32, 64],
        "complete_parent_box_union_no_position_budget_truncation": True,
        "candidate_cells_strictly_from_global_v2": True,
        "uses_query_image_features": True,
        "uses_query_rgb_directly": False,
        "uses_query_retrieval": True,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_mapping_camera_position_seed": False,
        "phase2_labels_opened": False,
        "generation_seconds": elapsed,
    }
    _atomic_save(output, arrays, metadata)
    loaded, loaded_metadata = load_parent_cropped_ceiling(output)
    if (
        loaded_metadata["content_sha256"] != metadata["content_sha256"]
        or arrays_sha256(loaded) != metadata["content_sha256"]
    ):
        raise AssertionError("parent-cropped ceiling round-trip differs")
    counts = arrays["candidate_count_by_parent_prefix"]
    report = {
        "artifact_type": "goal_maplet_parent_cropped_position_ceiling_build_v1",
        "candidate_inventory": str(output),
        "candidate_inventory_file_sha256": file_sha256(output),
        "candidate_inventory_content_sha256": metadata["content_sha256"],
        "query_count": int(image_ids.size),
        "candidate_count_by_parent_prefix": [
            {
                "parent_prefix_budget": budget,
                "minimum": int(np.min(counts[:, index])),
                "median": float(np.median(counts[:, index])),
                "maximum": int(np.max(counts[:, index])),
            }
            for index, budget in enumerate([4, 8, 16, 32, 64])
        ],
        "generation_seconds": elapsed,
        "phase1_score_before_label": True,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
