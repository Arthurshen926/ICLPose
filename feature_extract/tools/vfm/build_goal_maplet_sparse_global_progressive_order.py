"""Freeze a query-independent occupied-bin progressive order of global-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_progressive_global_position_proposal import (
    _atomic_save,
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
    GLOBAL_ORDER_SCHEMA,
    GLOBAL_ORDER_SEMANTICS,
    build_sparse_global_progressive_order_arrays,
    load_sparse_global_progressive_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_support", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz).resolve()
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite sparse global order")
    support_path = Path(args.global_support).resolve()
    global_arrays, global_metadata = load_all_parent_union_support(support_path)
    started = time.perf_counter()
    arrays = build_sparse_global_progressive_order_arrays(
        global_arrays["cell_indices_world"],
    )
    elapsed = float(time.perf_counter() - started)
    metadata = {
        "artifact_type": GLOBAL_ORDER_SCHEMA,
        "semantics": GLOBAL_ORDER_SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "global_support": {
            "path": str(support_path), "file_sha256": file_sha256(support_path),
            "content_sha256": global_metadata["content_sha256"],
        },
        "global_position_count": int(global_arrays["cell_indices_world"].shape[0]),
        "dyadic_level_count": int(arrays["dyadic_level_bins"].shape[0]),
        "one_inherited_or_new_representative_per_nonempty_bin": True,
        "newly_uncovered_bins_choose_nearest_center_cell": True,
        "complete_permutation_of_global_v2": True,
        "uses_query_image": False,
        "uses_query_retrieval": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "generation_seconds": elapsed,
    }
    _atomic_save(output, arrays, metadata)
    loaded, loaded_metadata = load_sparse_global_progressive_order(output)
    if (
        loaded_metadata["content_sha256"] != metadata["content_sha256"]
        or arrays_sha256(loaded) != metadata["content_sha256"]
    ):
        raise AssertionError("sparse global progressive order round-trip differs")
    report = {
        "artifact_type": "goal_maplet_sparse_global_progressive_order_build_v1",
        "global_order": str(output),
        "global_order_file_sha256": file_sha256(output),
        "global_order_content_sha256": metadata["content_sha256"],
        "global_position_count": metadata["global_position_count"],
        "dyadic_level_bins": arrays["dyadic_level_bins"].tolist(),
        "dyadic_level_nonempty_bin_counts": (
            arrays["dyadic_level_nonempty_bin_counts"].tolist()
        ),
        "generation_seconds": elapsed,
        "phase1_query_independent_no_label": True,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
