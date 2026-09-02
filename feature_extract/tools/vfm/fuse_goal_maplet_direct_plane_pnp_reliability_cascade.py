"""Cascade two frozen plane-PnP refinements without reopening correspondences.

The conservative branch is the default.  The spatially balanced refinement is
used only where its own upstream bounded-motion gate actually applied.  This
prevents a failed optional refinement from silently reverting a query to the
older pre-refinement pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


COMMON_KEYS = (
    "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
    "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
    "reliability_candidate_pose_distance", "reliability_candidate_pnp_inlier_count",
    "reliability_weight_minimum", "reliability_weight_maximum",
)


def _load(path: Path, expected_type: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        complete = {
            key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
        }
    if (
        metadata.get("artifact_type") != expected_type
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("strict_runtime_phase_separation_eligible") is not True
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
        or any(key not in complete for key in COMMON_KEYS)
    ):
        raise ValueError("reliability-cascade input is not strict and frozen")
    return complete, metadata


def _spatial_refinement_applied(branch: np.ndarray) -> np.ndarray:
    value = np.asarray(branch, np.int64)
    if value.ndim != 1 or np.any(~np.isin(value, (70, 71, 72))):
        raise ValueError("spatial refinement branch semantics differ")
    return value != 70


def _select_spatial_candidate(
    applied: np.ndarray,
    candidate_inliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(applied, bool)
    inliers = np.asarray(candidate_inliers, np.int64)
    if valid.ndim != 2 or inliers.shape != valid.shape or valid.shape[1] < 1:
        raise ValueError("spatial candidate selection arrays differ")
    ranked = np.where(valid, inliers, -1)
    return np.argmax(ranked, axis=1), np.any(valid, axis=1)


def _strict_inlier_improvement(
    has_spatial: np.ndarray,
    spatial_inliers: np.ndarray,
    conservative_inliers: np.ndarray,
) -> np.ndarray:
    available = np.asarray(has_spatial, bool)
    spatial = np.asarray(spatial_inliers, np.int64)
    conservative = np.asarray(conservative_inliers, np.int64)
    if available.shape != spatial.shape or spatial.shape != conservative.shape:
        raise ValueError("cascade inlier comparison arrays differ")
    return available & (spatial > conservative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conservative_pose_inventory", type=Path, required=True)
    parser.add_argument("--spatial_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite reliability cascade")

    conservative, conservative_meta = _load(
        args.conservative_pose_inventory,
        "goal_maplet_direct_plane_pnp_reliability_tiered_damped_refinement_v2",
    )
    loaded_spatial = [
        _load(
            path,
            "goal_maplet_direct_plane_pnp_spatial_plane_reliability_tiered_damped_refinement_v4",
        )
        for path in args.spatial_pose_inventory
    ]
    spatial_inventories = [item[0] for item in loaded_spatial]
    spatial_metadata = [item[1] for item in loaded_spatial]
    names = conservative["names"].astype(str)
    if len(set(names.tolist())) != len(names):
        raise ValueError("reliability-cascade query names are duplicated")
    if any(
        not np.array_equal(names, spatial["names"].astype(str))
        or not np.array_equal(conservative["usable"], spatial["usable"])
        or conservative_meta.get("query_camera_only_inventory_file_sha256")
        != spatial_meta.get("query_camera_only_inventory_file_sha256")
        or conservative_meta.get("primary_file_sha256")
        != spatial_meta.get("primary_file_sha256")
        or conservative_meta.get("primary_content_sha256")
        != spatial_meta.get("primary_content_sha256")
        or conservative_meta.get("correspondence_file_sha256_by_branch")
        != spatial_meta.get("correspondence_file_sha256_by_branch")
        for spatial, spatial_meta in loaded_spatial
    ):
        raise ValueError("reliability-cascade inventories or lineage differ")

    applied = np.stack([
        _spatial_refinement_applied(spatial["selected_branch"])
        for spatial in spatial_inventories
    ], axis=1)
    candidate_inliers = np.stack([
        np.asarray(spatial["reliability_candidate_pnp_inlier_count"], np.int64)
        for spatial in spatial_inventories
    ], axis=1)
    # Invalid candidates can never win. np.argmax gives a deterministic,
    # input-order tie break among equal unique-token inlier counts.
    selected_spatial_index, has_spatial = _select_spatial_candidate(
        applied, candidate_inliers,
    )
    selected_pose = np.stack([
        spatial_inventories[int(which)]["pose_w2c"][row]
        for row, which in enumerate(selected_spatial_index.tolist())
    ])
    selected_distance = np.asarray([
        spatial_inventories[int(which)]["reliability_candidate_pose_distance"][row]
        for row, which in enumerate(selected_spatial_index.tolist())
    ], np.float64)
    selected_candidate_inliers = np.asarray([
        candidate_inliers[row, int(which)]
        for row, which in enumerate(selected_spatial_index.tolist())
    ], np.int64)
    choose_spatial = _strict_inlier_improvement(
        has_spatial,
        selected_candidate_inliers,
        np.asarray(conservative["reliability_candidate_pnp_inlier_count"], np.int64),
    )
    arrays = {
        "names": names,
        "pose_w2c": np.where(
            choose_spatial[:, None, None], selected_pose, conservative["pose_w2c"],
        ),
        "usable": np.asarray(conservative["usable"], bool),
        "selected_branch": np.where(choose_spatial, 81, 80).astype(np.int16),
        "selected_inlier_ratio": np.asarray(
            conservative["selected_inlier_ratio"], np.float64,
        ),
        "selected_candidate_correspondence_count": np.asarray(
            conservative["selected_candidate_correspondence_count"], np.int64,
        ),
        "selected_pnp_inlier_count": np.asarray(
            conservative["selected_pnp_inlier_count"], np.int64,
        ),
        "cascade_selected_spatial": choose_spatial,
        "selected_spatial_inventory_index": np.where(
            choose_spatial, selected_spatial_index, -1,
        ).astype(np.int16),
        "spatial_candidate_pose_distance": np.asarray(
            selected_distance, np.float64,
        ),
        "spatial_candidate_pnp_inlier_count": np.asarray(
            selected_candidate_inliers, np.int64,
        ),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_multiscale_reliability_cascade_v7",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": (
            "conservative_plane_reliability_default_then_maximum_unique_token_inlier_"
            "spatial_candidate_among_frozen_bounded_motion_gate_survivors_only_if_"
            "strictly_more_inliers_than_conservative_candidate"
        ),
        "spatial_selected_count": int(np.sum(choose_spatial)),
        "spatial_inventory_count": int(len(spatial_inventories)),
        "strict_unique_token_inlier_improvement_required": True,
        "selected_spatial_inventory_counts": {
            str(index): int(np.sum(choose_spatial & (selected_spatial_index == index)))
            for index in range(len(spatial_inventories))
        },
        "source_branch_codes": {"80": "conservative", "81": "spatial"},
        "conservative_file_sha256": file_sha256(args.conservative_pose_inventory),
        "conservative_content_sha256": conservative_meta.get("content_sha256"),
        "spatial_file_sha256_in_order": [
            file_sha256(path) for path in args.spatial_pose_inventory
        ],
        "spatial_content_sha256_in_order": [
            metadata.get("content_sha256") for metadata in spatial_metadata
        ],
        "primary_file_sha256": conservative_meta.get("primary_file_sha256"),
        "primary_content_sha256": conservative_meta.get("primary_content_sha256"),
        "query_camera_only_inventory_file_sha256": conservative_meta.get(
            "query_camera_only_inventory_file_sha256"
        ),
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "query_depth_or_scale_used_by_pose_solver": False,
        "configuration_role": "historical_validation_ablation_not_pristine_blind",
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
