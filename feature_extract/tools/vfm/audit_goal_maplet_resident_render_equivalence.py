"""Prove resident-GPU full-map rendering is decision-equivalent to legacy rendering.

The audit opens no query pose or ground truth.  It requires byte-bound input
lineage to agree, compares every per-query render diagnostic exactly, and then
replays the frozen two-geometry consensus decision.  Runtime-only report fields
are deliberately excluded from the semantic comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus import (
    _load_plane_geometry,
    _load_render,
    _load_selected,
    _normal_good_ray_score,
    _select_geometry_consensus,
)
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


OUTPUT_ARTIFACT = "goal_maplet_resident_full2dgs_render_equivalence_v1"
_IMMUTABLE_RENDER_FIELDS = (
    "artifact_type",
    "query_count",
    "query_pose_or_ground_truth_read",
    "query_depth_or_scale_used_by_pose_solver",
    "query_moge3_role",
    "depth_scale_fit",
    "affine_log_depth_fit",
    "raw_metric_depth_retained",
    "query_depth_changes_frozen_pose",
    "frozen_pose_inventory_file_sha256",
    "frozen_pose_inventory_content_sha256",
    "physical_map_file_sha256",
    "physical_map_content_sha256",
    "query_camera_inventory_file_sha256",
    "query_camera_inventory_content_sha256",
    "frozen_correspondence_file_sha256",
    "frozen_correspondence_content_sha256",
    "moge3_manifest_file_sha256_in_order",
    "moge3_manifest_content_sha256_in_order",
    "minimum_front_incidence",
)


def _validate_render_equivalence(
    legacy: dict[str, object], optimized: dict[str, object],
) -> None:
    for key in _IMMUTABLE_RENDER_FIELDS:
        if legacy.get(key) != optimized.get(key):
            raise ValueError(f"render lineage differs at {key}")
    if legacy.get("rows") != optimized.get("rows"):
        raise ValueError("resident renderer changes per-query diagnostics")
    if (
        legacy.get("renderer")
        != "clean_2dgs_disks_alpha_transmittance_dominant_depth"
        or optimized.get("renderer")
        != "resident_batched_clean_2dgs_disks_alpha_transmittance_dominant_depth"
        or int(optimized.get("resident_batch_size", 0)) <= 1
        or optimized.get("resident_gpu_composite") is not True
    ):
        raise ValueError("render implementation roles differ")
    legacy_time = float(legacy.get("elapsed_seconds", np.nan))
    optimized_time = float(optimized.get("elapsed_seconds", np.nan))
    if not (np.isfinite(legacy_time) and np.isfinite(optimized_time)):
        raise ValueError("render runtime is not finite")
    if legacy_time <= 0.0 or optimized_time <= 0.0:
        raise ValueError("render runtime must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose", type=Path, required=True)
    parser.add_argument("--alternate_pose", type=Path, required=True)
    parser.add_argument("--plane_geometry", type=Path, required=True)
    parser.add_argument("--legacy_primary_render", type=Path, required=True)
    parser.add_argument("--legacy_alternate_render", type=Path, required=True)
    parser.add_argument("--optimized_primary_render", type=Path, required=True)
    parser.add_argument("--optimized_alternate_render", type=Path, required=True)
    parser.add_argument("--frozen_consensus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite resident-render audit")

    primary, _primary_meta = _load_pose_candidate(args.primary_pose)
    alternate, _alternate_meta = _load_pose_candidate(args.alternate_pose)
    plane, plane_meta = _load_plane_geometry(args.plane_geometry)
    frozen, frozen_meta = _load_selected(args.frozen_consensus)
    legacy_primary = _load_render(args.legacy_primary_render)
    legacy_alternate = _load_render(args.legacy_alternate_render)
    optimized_primary = _load_render(args.optimized_primary_render)
    optimized_alternate = _load_render(args.optimized_alternate_render)
    _validate_render_equivalence(legacy_primary, optimized_primary)
    _validate_render_equivalence(legacy_alternate, optimized_alternate)

    names = primary["names"].astype(str)
    if not (
        np.array_equal(names, alternate["names"].astype(str))
        and np.array_equal(names, plane["names"].astype(str))
        and np.array_equal(names, frozen["names"].astype(str))
        and plane_meta.get("point_pose_file_sha256") == file_sha256(args.primary_pose)
        and plane_meta.get("surface_pose_file_sha256") == file_sha256(args.alternate_pose)
        and frozen_meta.get("primary_pose_file_sha256") == file_sha256(args.primary_pose)
        and frozen_meta.get("alternate_pose_file_sha256") == file_sha256(args.alternate_pose)
    ):
        raise ValueError("pose/plane/consensus inventory differs")
    for report in (
        legacy_primary, legacy_alternate, optimized_primary, optimized_alternate,
    ):
        if [str(row.get("name")) for row in report["rows"]] != names.tolist():
            raise ValueError("render query order differs")

    usable = np.stack([primary["usable"], alternate["usable"]], axis=1).astype(bool)
    objective = np.asarray(plane["moge3_plane_geometry_objective"], np.float64)

    def replay(primary_report: dict[str, object], alternate_report: dict[str, object]):
        score = np.stack([
            np.asarray([_normal_good_ray_score(row) for row in primary_report["rows"]]),
            np.asarray([_normal_good_ray_score(row) for row in alternate_report["rows"]]),
        ], axis=1)
        return score, _select_geometry_consensus(objective, score, usable)

    legacy_score, legacy_branch = replay(legacy_primary, legacy_alternate)
    optimized_score, optimized_branch = replay(optimized_primary, optimized_alternate)
    if not np.array_equal(legacy_score, optimized_score):
        raise ValueError("resident renderer changes the dense normal score")
    if not np.array_equal(legacy_branch, optimized_branch):
        raise ValueError("resident renderer changes the consensus branch")
    if not np.array_equal(optimized_branch, frozen["selected_branch"]):
        raise ValueError("resident renderer does not replay frozen consensus")

    legacy_seconds = np.asarray([
        legacy_primary["elapsed_seconds"], legacy_alternate["elapsed_seconds"],
    ], np.float64)
    optimized_seconds = np.asarray([
        optimized_primary["elapsed_seconds"], optimized_alternate["elapsed_seconds"],
    ], np.float64)
    report: dict[str, object] = {
        "artifact_type": OUTPUT_ARTIFACT,
        "query_count": int(len(names)),
        "query_pose_or_ground_truth_read": False,
        "per_query_render_rows_exact": True,
        "dense_normal_scores_exact": True,
        "consensus_branches_exact": True,
        "frozen_consensus_replayed": True,
        "selected_alternate_count": int(np.sum(optimized_branch == 1)),
        "resident_batch_size": int(optimized_primary["resident_batch_size"]),
        "legacy_elapsed_seconds_primary_alternate": legacy_seconds.tolist(),
        "optimized_elapsed_seconds_primary_alternate": optimized_seconds.tolist(),
        "speedup_primary_alternate": (legacy_seconds / optimized_seconds).tolist(),
        "primary_pose_file_sha256": file_sha256(args.primary_pose),
        "alternate_pose_file_sha256": file_sha256(args.alternate_pose),
        "plane_geometry_file_sha256": file_sha256(args.plane_geometry),
        "frozen_consensus_file_sha256": file_sha256(args.frozen_consensus),
        "legacy_render_file_sha256_primary_alternate": [
            file_sha256(args.legacy_primary_render), file_sha256(args.legacy_alternate_render),
        ],
        "optimized_render_file_sha256_primary_alternate": [
            file_sha256(args.optimized_primary_render), file_sha256(args.optimized_alternate_render),
        ],
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
