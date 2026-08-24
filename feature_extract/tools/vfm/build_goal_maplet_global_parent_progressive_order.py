"""Precompute deterministic per-parent nested orders inside global-v2 cells."""

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
    PARENT_ORDER_SCHEMA,
    PARENT_ORDER_SEMANTICS,
    build_parent_progressive_order_arrays,
    load_parent_progressive_order,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_support", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz).resolve()
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite progressive parent order")
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    started = time.perf_counter()
    arrays = build_parent_progressive_order_arrays(global_arrays)
    elapsed = float(time.perf_counter() - started)
    counts = np.diff(arrays["parent_cell_offsets"])
    metadata: dict[str, object] = {
        "artifact_type": PARENT_ORDER_SCHEMA,
        "semantics": PARENT_ORDER_SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "global_support": {
            "path": str(support_path),
            "file_sha256": file_sha256(support_path),
            "content_sha256": global_metadata["content_sha256"],
            "position_count": int(global_arrays["cell_indices_world"].shape[0]),
        },
        "parent_count": int(arrays["maplet_ids"].size),
        "parent_cell_entry_count_with_cross_parent_duplicates": int(
            arrays["parent_order_cell_rows"].size
        ),
        "parent_cell_count_minimum": int(np.min(counts)),
        "parent_cell_count_median": float(np.median(counts)),
        "parent_cell_count_maximum": int(np.max(counts)),
        "ordering": {
            "levels": "dyadic_axis_partitions_coarse_to_fine",
            "representative": "integer_cell_nearest_each_subbox_center",
            "cross_level_policy": "one_representative_per_current_dyadic_bin",
            "partial_level_dispersion": "bit_reversed_morton_bin_key",
            "duplicate_policy": "first_occurrence",
            "nested_prefix": True,
            "complete_parent_box_inventory": True,
        },
        "uses_query_image": False,
        "uses_query_retrieval": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_mapping_camera_position_seed": False,
        "adds_positions_outside_global_v2": False,
        "generation_seconds": elapsed,
    }
    _atomic_save(output, arrays, metadata)
    loaded, loaded_metadata = load_parent_progressive_order(output)
    if (
        loaded_metadata["content_sha256"] != metadata["content_sha256"]
        or arrays_sha256(loaded) != metadata["content_sha256"]
    ):
        raise AssertionError("progressive parent order round-trip differs")
    report = {
        "artifact_type": "goal_maplet_global_parent_progressive_order_build_v1",
        "parent_order": str(output),
        "parent_order_file_sha256": file_sha256(output),
        "parent_order_content_sha256": metadata["content_sha256"],
        "global_support_file_sha256": metadata["global_support"]["file_sha256"],
        "global_support_content_sha256": metadata["global_support"]["content_sha256"],
        "parent_count": metadata["parent_count"],
        "parent_cell_entry_count_with_cross_parent_duplicates": metadata[
            "parent_cell_entry_count_with_cross_parent_duplicates"
        ],
        "generation_seconds": elapsed,
        "phase1_geometry_only_no_query_or_label": True,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
