"""Freeze a query-independent union of all expanded physical-parent boxes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    ALL_PARENT_UNION_SCHEMA,
    ALL_PARENT_UNION_SEMANTICS,
    ORIENTATION_COUNT,
    all_parent_union_public_arrays,
    build_all_parent_union_support_arrays,
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.parent_geometry_pose_proposal import (
    LATTICE_SPACING_M,
    MAXIMUM_UNIQUE_CELLS_PER_QUERY,
    PARENT_AABB_EXPANSION_LINF_M,
    orientation_cover_certificate,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)


def _atomic_save_npz(
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


def _load_single_aabb_kill(path: Path, physical: GoalMapletPhysicalMap) -> dict:
    report = json.loads(path.read_text())
    unhashed = {key: value for key, value in report.items() if key != "content_sha256"}
    if (
        report.get("artifact_type")
        != "goal_maplet_query_independent_global_physical_pose_support_audit_v1"
        or report.get("content_sha256") != canonical_json_sha256(unhashed)
        or report.get("structural_gate", {}).get("decision") != "KILL"
        or report.get("structural_gate", {}).get("seq10_pose_labels_read") is not False
        or report.get("physical_map", {}).get("content_sha256")
        != physical.content_sha256
    ):
        raise ValueError("single-global-AABB structural KILL audit differs")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--single_aabb_kill_audit", required=True)
    parser.add_argument(
        "--maximum_position_count", type=int,
        default=MAXIMUM_UNIQUE_CELLS_PER_QUERY,
    )
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz).resolve()
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite all-parent support proposal")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    if physical.metadata.get("primitive_membership_coverage_fraction") != 1.0:
        raise ValueError("physical map lacks complete primitive membership")
    kill_path = Path(args.single_aabb_kill_audit).resolve()
    kill_report = _load_single_aabb_kill(kill_path, physical)
    started = time.perf_counter()
    internal = build_all_parent_union_support_arrays(
        physical, maximum_position_count=int(args.maximum_position_count),
    )
    raw_parent_cell_count = int(internal["_raw_parent_cell_count"])
    arrays = all_parent_union_public_arrays(internal)
    elapsed = float(time.perf_counter() - started)
    position_count = int(arrays["cell_indices_world"].shape[0])
    factor_count = int(arrays["implicit_pose_factor_count"])
    metadata: dict[str, object] = {
        "artifact_type": ALL_PARENT_UNION_SCHEMA,
        "semantics": ALL_PARENT_UNION_SEMANTICS,
        "phase": "phase1_score_before_label_physical_geometry_only",
        "content_sha256": arrays_sha256(arrays),
        "physical_map": {
            "path": str(physical_path),
            "file_sha256": file_sha256(physical_path),
            "content_sha256": physical.content_sha256,
            "maplet_count": int(physical.maplet_ids.size),
            "primitive_count": int(physical.primitive_centers.shape[0]),
        },
        "predecessor_single_global_aabb_kill_audit": {
            "path": str(kill_path),
            "file_sha256": file_sha256(kill_path),
            "content_sha256": str(kill_report["content_sha256"]),
        },
        "uses_query_image": False,
        "uses_query_retrieval": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_mapping_camera_position_seed": False,
        "query_independent_positions_shared": True,
        "all_physical_parents_included": True,
        "parent_retrieval_cuts_domain": False,
        "parent_retrieval_future_role": "priority_only_not_support_pruning",
        "primitive_rectangle_axis_extent": (
            "abs(tangent1_axis)*scale1+abs(tangent2_axis)*scale2"
        ),
        "support_sigma_multiplier": 1.0,
        "parent_aabb_expansion_linf_m": PARENT_AABB_EXPANSION_LINF_M,
        "lattice_spacing_m": LATTICE_SPACING_M,
        "cell_cover_radius_m": float(np.sqrt(3.0)),
        "cell_cover_radius_strictly_below_2m": True,
        "raw_parent_cell_count_before_cross_parent_dedup": raw_parent_cell_count,
        "position_count": position_count,
        "position_hard_cap": int(args.maximum_position_count),
        "position_count_within_cap": True,
        "orientation_count": ORIENTATION_COUNT,
        "orientation_cover_certificate": orientation_cover_certificate(),
        "implicit_pose_factor_count": factor_count,
        "cartesian_product_materialized": False,
        "generation_seconds": elapsed,
        "support_claim": (
            "nonphysical_reconstruction_surface_neighborhood_raw_support_upper_bound_only"
        ),
        "free_space_or_clearance_certificate": False,
        "collision_certificate": False,
        "ranking_performed": False,
        "phase2_labels_opened": False,
    }
    _atomic_save_npz(output, arrays, metadata)
    loaded_arrays, loaded_metadata = load_all_parent_union_support(output)
    if (
        loaded_metadata["content_sha256"] != metadata["content_sha256"]
        or arrays_sha256(loaded_arrays) != metadata["content_sha256"]
    ):
        raise AssertionError("all-parent support proposal round-trip differs")
    sidecar_report = {
        "artifact_type": "goal_maplet_all_parent_geometry_support_build_report_v1",
        "proposal": str(output),
        "proposal_file_sha256": file_sha256(output),
        "proposal_content_sha256": metadata["content_sha256"],
        "position_count": position_count,
        "implicit_pose_factor_count": factor_count,
        "generation_seconds": elapsed,
        "phase1_only_no_query_or_label_input": True,
    }
    sidecar_report["content_sha256"] = canonical_json_sha256(sidecar_report)
    sidecar.write_text(
        json.dumps(sidecar_report, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    print(json.dumps(sidecar_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
