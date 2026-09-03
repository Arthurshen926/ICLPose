"""Select between two source-free plane atlases by cross-atlas geometric consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load, _score
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def _load_selected(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    required = {"names", "pose_w2c", "usable", "selected_inlier_ratio"}
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_dual_surface_geometry_consensus_selected_v1",
            "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
            "goal_maplet_probabilistic_surface_pose_refinement_v1",
        )
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("source_rgb_stored_or_consumed_at_runtime") is not False
        or not required.issubset(arrays)
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("dual-surface pose inventory contract differs")
    return arrays, metadata


def _select(scores: np.ndarray) -> np.ndarray:
    value = np.asarray(scores, np.float64)
    if value.ndim != 3 or value.shape[1] != 2 or value.shape[2] < 2:
        raise ValueError("cross-atlas scores must have shape (query,2,>=2)")
    if not np.all(np.isfinite(value)):
        raise ValueError("cross-atlas scores must be finite")
    return np.argmax(np.mean(value, axis=2), axis=1).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_inventory", type=Path, nargs=2, required=True)
    parser.add_argument("--frozen_correspondences", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite cross-atlas consensus")
    pose_items = [_load_selected(path) for path in args.pose_inventory]
    corr_items = [_load(path) for path in args.frozen_correspondences]
    names = pose_items[0][0]["names"].astype(str)
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in pose_items[1:]):
        raise ValueError("pose query order differs")
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in corr_items):
        raise ValueError("correspondence query order differs")
    # The four inventories must be two thresholds for each of two distinct
    # anonymous atlases.  This prevents accidentally counting duplicate files.
    contents = [item[1].get("content_sha256") for item in corr_items]
    atlas_contents = [item[1].get("plane_uv_atlas_content_sha256") for item in corr_items]
    thresholds = [float(item[1].get("homography_threshold_m", -1.0)) for item in corr_items]
    if len(set(contents)) != 4 or atlas_contents[0] != atlas_contents[1] or atlas_contents[2] != atlas_contents[3]:
        raise ValueError("four distinct correspondence inventories must form two atlas pairs")
    if atlas_contents[0] == atlas_contents[2] or thresholds[:2] != thresholds[2:] or len(set(thresholds[:2])) != 2:
        raise ValueError("cross-atlas threshold pairing differs")

    scores = np.zeros((len(names), 2, 4), np.float64)
    inliers = np.zeros((len(names), 2, 4), np.int64)
    for query in range(len(names)):
        for geometry, (corr, _) in enumerate(corr_items):
            lo, hi = map(int, corr["correspondence_offsets"][query : query + 2])
            for candidate, (pose, _) in enumerate(pose_items):
                if not bool(pose["usable"][query]):
                    continue
                result = _score(
                    pose["pose_w2c"][query], corr["world_points"][lo:hi],
                    corr["query_tokens"][lo:hi], corr["provenance_region_plane_atlas_row"][lo:hi],
                    corr["camera_matrices"][query], float(corr["radial_k1"][query]),
                    corr["query_measurements_xy"][lo:hi]
                    if "query_measurements_xy" in corr else None,
                )
                scores[query, candidate, geometry] = float(result["inlier_ratio"])
                inliers[query, candidate, geometry] = int(result["inlier_count"])
    selected = _select(scores)
    row = np.arange(len(names))
    arrays = {
        "names": names,
        "pose_w2c": np.stack([pose_items[int(branch)][0]["pose_w2c"][query] for query, branch in enumerate(selected)]),
        "usable": np.asarray([bool(pose_items[int(branch)][0]["usable"][query]) for query, branch in enumerate(selected)]),
        "selected_branch": selected,
        "selected_inlier_ratio": np.mean(scores, axis=2)[row, selected],
        "selected_candidate_correspondence_count": np.rint(np.mean(inliers[row, selected], axis=1)).astype(np.int64),
        "selected_pnp_inlier_count": np.rint(np.mean(inliers[row, selected], axis=1)).astype(np.int64),
        "cross_atlas_geometry_inlier_ratio": scores,
        "cross_atlas_geometry_inlier_count": inliers,
    }
    metadata = {
        "artifact_type": "goal_maplet_cross_atlas_geometry_consensus_selected_v1",
        "selection_rule": "maximum_mean_unique_token_inlier_ratio_across_two_thresholds_and_two_anonymous_atlases",
        "stable_tie_break": "legacy_atlas_candidate_zero",
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "selected_confidence_semantics": "mean_four_inventory_unique_token_inlier_ratio_uncalibrated",
        "pose_inventory_file_sha256": [file_sha256(path) for path in args.pose_inventory],
        "pose_inventory_content_sha256": [item[1].get("content_sha256") for item in pose_items],
        "correspondence_file_sha256": [file_sha256(path) for path in args.frozen_correspondences],
        "correspondence_content_sha256": contents,
        "atlas_content_sha256_in_pair_order": atlas_contents,
        "homography_threshold_m_in_pair_order": thresholds,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({**metadata, "selected_branch_counts": np.bincount(selected, minlength=2).tolist()}, indent=2))


if __name__ == "__main__":
    main()
