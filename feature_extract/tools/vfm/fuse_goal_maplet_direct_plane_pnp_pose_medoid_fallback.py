"""Apply a conservative SE(3)-medoid fallback to frozen plane-PnP poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


COMMON_KEYS = (
    "names", "pose_w2c", "usable", "selected_inlier_ratio",
    "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
)


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        complete = {
            key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
        }
        arrays = {key: complete[key] for key in COMMON_KEYS}
    if (
        metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("strict_runtime_phase_separation_eligible") is not True
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
    ):
        raise ValueError("pose inventory is not a strict frozen runtime artifact")
    return arrays, metadata


def _pose_distance(
    left: np.ndarray,
    right: np.ndarray,
    translation_scale_m: float = 0.5,
    rotation_scale_deg: float = 5.0,
) -> float:
    if translation_scale_m <= 0.0 or rotation_scale_deg <= 0.0:
        raise ValueError("pose-distance scales must be positive")
    left_center = -left[:3, :3].T @ left[:3, 3]
    right_center = -right[:3, :3].T @ right[:3, 3]
    translation = float(np.linalg.norm(left_center - right_center))
    rotation = float(
        Rotation.from_matrix(left[:3, :3] @ right[:3, :3].T).magnitude()
        * 180.0 / np.pi
    )
    return translation / translation_scale_m + rotation / rotation_scale_deg


def _medoid_index(poses: list[np.ndarray]) -> tuple[int, np.ndarray]:
    if not poses:
        raise ValueError("medoid requires at least one pose")
    distances = np.asarray([
        [_pose_distance(left, right) for right in poses] for left in poses
    ], np.float64)
    return int(np.argmin(np.sum(distances, axis=1))), distances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose_inventory", type=Path, required=True)
    parser.add_argument("--lm_pose_inventory", type=Path, required=True)
    parser.add_argument("--point_consensus_pose_inventory", type=Path, required=True)
    parser.add_argument("--override_distance_threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite pose-medoid fallback")
    if args.override_distance_threshold <= 0.0:
        raise ValueError("override distance threshold must be positive")

    paths = (
        args.primary_pose_inventory,
        args.lm_pose_inventory,
        args.point_consensus_pose_inventory,
    )
    loaded = [_load(path) for path in paths]
    inventories = [item[0] for item in loaded]
    metadata = [item[1] for item in loaded]
    names = inventories[0]["names"].astype(str)
    camera_hash = metadata[0].get("query_camera_only_inventory_file_sha256")
    if len(set(names.tolist())) != len(names):
        raise ValueError("query names are duplicated")
    for arrays, meta in zip(inventories[1:], metadata[1:]):
        if (
            not np.array_equal(names, arrays["names"].astype(str))
            or meta.get("query_camera_only_inventory_file_sha256") != camera_hash
        ):
            raise ValueError("pose-medoid query inventories differ")

    output_pose, output_usable, output_source = [], [], []
    output_ratio, output_candidates, output_inliers = [], [], []
    override_distance = []
    override_count = 0
    for query_index in range(len(names)):
        primary_valid = bool(inventories[0]["usable"][query_index])
        all_valid = all(bool(arrays["usable"][query_index]) for arrays in inventories)
        selected_index = 0
        distance_from_primary = 0.0
        if primary_valid and all_valid:
            poses = [np.asarray(arrays["pose_w2c"][query_index], np.float64) for arrays in inventories]
            medoid_index, distances = _medoid_index(poses)
            distance_from_primary = float(distances[0, medoid_index])
            if medoid_index != 0 and distance_from_primary > args.override_distance_threshold:
                selected_index = medoid_index
                override_count += 1
        selected = inventories[selected_index]
        valid = primary_valid
        output_pose.append(
            np.asarray(selected["pose_w2c"][query_index], np.float64)
            if valid else np.full((4, 4), np.nan)
        )
        output_usable.append(valid)
        output_source.append(50 + selected_index)
        output_ratio.append(float(selected["selected_inlier_ratio"][query_index]))
        output_candidates.append(int(selected["selected_candidate_correspondence_count"][query_index]))
        output_inliers.append(int(selected["selected_pnp_inlier_count"][query_index]))
        override_distance.append(distance_from_primary)

    arrays = {
        "names": names,
        "pose_w2c": np.asarray(output_pose, np.float64),
        "usable": np.asarray(output_usable, bool),
        "selected_branch": np.asarray(output_source, np.int16),
        "selected_inlier_ratio": np.asarray(output_ratio, np.float64),
        "selected_candidate_correspondence_count": np.asarray(output_candidates, np.int64),
        "selected_pnp_inlier_count": np.asarray(output_inliers, np.int64),
        "override_distance_from_primary": np.asarray(override_distance, np.float64),
    }
    report = {
        "artifact_type": "goal_maplet_direct_plane_pnp_pose_medoid_fallback_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": "primary_unless_three_pose_se3_medoid_differs_beyond_one_gate",
        "pose_distance": "translation_m/0.5 + rotation_deg/5",
        "override_distance_threshold": float(args.override_distance_threshold),
        "override_count": int(override_count),
        "source_branch_codes": {"50": "primary", "51": "lm", "52": "point_consensus"},
        "source_file_sha256_in_order": [file_sha256(path) for path in paths],
        "source_content_sha256_in_order": [meta.get("content_sha256") for meta in metadata],
        "query_camera_only_inventory_file_sha256": camera_hash,
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "query_depth_or_scale_used_by_pose_solver": False,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(report, sort_keys=True)),
    )
    temporary.replace(args.output)
    print(json.dumps({**report, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
