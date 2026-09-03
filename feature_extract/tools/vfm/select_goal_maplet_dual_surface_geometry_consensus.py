"""Select ideal/local plane geometry poses by symmetric pose-free consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load,
    _score,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _select_by_cross_geometry_inlier_ratio(ratio: np.ndarray) -> np.ndarray:
    """Return the stable best branch after grading every pose on every geometry."""
    value = np.asarray(ratio, np.float64)
    if value.ndim != 3 or value.shape[1:] != (2, 2) or not np.all(np.isfinite(value)):
        raise ValueError("cross-geometry inlier ratios must have shape (query,2,2)")
    return np.argmax(np.mean(value, axis=2), axis=1).astype(np.int8)


def _poses(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        complete = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    required = {
        "names", "pose_w2c", "usable", "candidate_correspondence_count",
        "pnp_inlier_count",
    }
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_moge3_plane_scale_surface_refinement_v1",
            "goal_maplet_uncertainty_weighted_plane_pose_refinement_v1",
        )
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
        or not required.issubset(complete)
    ):
        raise ValueError("surface pose inventory contract differs")
    return complete, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_inventory", type=Path, nargs=2, required=True)
    parser.add_argument("--frozen_correspondences", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite dual-surface selection")

    poses_and_meta = [_poses(path) for path in args.pose_inventory]
    corr_and_meta = [_load(path) for path in args.frozen_correspondences]
    names = poses_and_meta[0][0]["names"].astype(str)
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in poses_and_meta[1:]):
        raise ValueError("surface pose query order differs")
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in corr_and_meta):
        raise ValueError("surface correspondence query order differs")

    count = len(names)
    ratio = np.zeros((count, 2, 2), np.float64)
    median = np.full((count, 2, 2), np.inf, np.float64)
    inliers = np.zeros((count, 2, 2), np.int64)
    for query in range(count):
        for geometry, (corr, _) in enumerate(corr_and_meta):
            lo, hi = map(int, corr["correspondence_offsets"][query : query + 2])
            for candidate, (pose, _) in enumerate(poses_and_meta):
                if not bool(pose["usable"][query]):
                    continue
                score = _score(
                    pose["pose_w2c"][query], corr["world_points"][lo:hi],
                    corr["query_tokens"][lo:hi],
                    corr["provenance_region_plane_atlas_row"][lo:hi],
                    corr["camera_matrices"][query], float(corr["radial_k1"][query]),
                    corr["query_measurements_xy"][lo:hi]
                    if "query_measurements_xy" in corr else None,
                )
                ratio[query, candidate, geometry] = float(score["inlier_ratio"])
                inliers[query, candidate, geometry] = int(score["inlier_count"])
                value = score["reprojection_median_px"]
                median[query, candidate, geometry] = np.inf if value is None else float(value)

    mean_ratio = np.mean(ratio, axis=2)
    # Stable tie-break keeps branch zero.  Both candidates are evaluated on
    # both geometry inventories, preventing either representation from grading
    # only its own fitted pose.
    selected = _select_by_cross_geometry_inlier_ratio(ratio)
    row = np.arange(count)
    usable = np.asarray([
        bool(poses_and_meta[int(branch)][0]["usable"][query])
        for query, branch in enumerate(selected)
    ])
    arrays = {
        "names": names,
        "pose_w2c": np.stack([
            poses_and_meta[int(branch)][0]["pose_w2c"][query]
            for query, branch in enumerate(selected)
        ]),
        "usable": usable,
        "selected_branch": selected,
        "selected_inlier_ratio": mean_ratio[row, selected],
        "selected_candidate_correspondence_count": np.asarray([
            poses_and_meta[int(branch)][0]["candidate_correspondence_count"][query]
            for query, branch in enumerate(selected)
        ], np.int64),
        "selected_pnp_inlier_count": np.rint(np.mean(inliers[row, selected], axis=1)).astype(np.int64),
        "cross_geometry_inlier_ratio": ratio,
        "cross_geometry_reprojection_median_px": median,
    }
    metadata = {
        "artifact_type": "goal_maplet_dual_surface_geometry_consensus_selected_v1",
        "selection_rule": "maximum_mean_unique_token_inlier_ratio_across_both_frozen_geometries",
        "stable_tie_break": "branch_zero",
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "selected_confidence_semantics": "mean_cross_geometry_unique_token_inlier_ratio_uncalibrated",
        "source_rgb_stored_or_consumed_at_runtime": False,
        "pose_inventory_file_sha256": [file_sha256(path) for path in args.pose_inventory],
        "pose_inventory_content_sha256": [item[1].get("content_sha256") for item in poses_and_meta],
        "correspondence_file_sha256": [file_sha256(path) for path in args.frozen_correspondences],
        "correspondence_content_sha256": [item[1].get("content_sha256") for item in corr_and_meta],
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({**metadata, "selected_branch_counts": np.bincount(selected, minlength=2).tolist()}, indent=2))


if __name__ == "__main__":
    main()
