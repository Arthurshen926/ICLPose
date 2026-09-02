"""Damped final pose closure on frozen multi-budget plane correspondences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_budget_consensus import (
    _interpolate_pose,
)
from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_map_density import (
    _candidate,
    _merge_correspondence,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


PRIMARY_KEYS = (
    "names", "pose_w2c", "usable", "selected_inlier_ratio",
    "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
)


def _load_primary(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        complete = {
            key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
        }
        arrays = {key: complete[key] for key in PRIMARY_KEYS}
    if (
        metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("strict_runtime_phase_separation_eligible") is not True
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
    ):
        raise ValueError("primary pose inventory is not strict and frozen")
    return arrays, metadata


def _normalized_pose_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_center = -left[:3, :3].T @ left[:3, 3]
    right_center = -right[:3, :3].T @ right[:3, 3]
    translation = float(np.linalg.norm(left_center - right_center))
    rotation = float(
        Rotation.from_matrix(left[:3, :3] @ right[:3, :3].T).magnitude()
        * 180.0 / np.pi
    )
    return translation / 0.5 + rotation / 5.0


def _closure_enabled(
    primary_inlier_ratio: float,
    normalized_pose_distance: float,
    *,
    maximum_primary_ratio: float = 0.6,
    maximum_pose_distance: float = 2.0,
) -> bool:
    return bool(
        np.isfinite(primary_inlier_ratio)
        and np.isfinite(normalized_pose_distance)
        and primary_inlier_ratio < maximum_primary_ratio
        and normalized_pose_distance < maximum_pose_distance
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose_inventory", type=Path, required=True)
    parser.add_argument("--top5_sparse_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument("--top5_dense_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_sparse_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_dense_correspondence", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum_primary_ratio", type=float, default=0.6)
    parser.add_argument("--maximum_pose_distance", type=float, default=2.0)
    parser.add_argument("--closure_fraction", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite union closure")
    if not 0.0 < args.maximum_primary_ratio < 1.0:
        raise ValueError("maximum primary ratio must be in (0, 1)")
    if args.maximum_pose_distance <= 0.0:
        raise ValueError("maximum pose distance must be positive")
    if not 0.0 < args.closure_fraction < 1.0:
        raise ValueError("closure fraction must be in (0, 1)")

    primary, primary_meta = _load_primary(args.primary_pose_inventory)
    branch_paths = (
        args.top5_sparse_correspondence,
        args.top5_dense_correspondence,
        args.top10_sparse_correspondence,
        args.top10_dense_correspondence,
    )
    loaded = [_merge_correspondence(paths) for paths in branch_paths]
    branches = [item[0] for item in loaded]
    branch_meta = [item[1] for item in loaded]
    names = primary["names"].astype(str)
    for rows in branches:
        if not np.array_equal(names, np.asarray([str(row["name"].item()) for row in rows])):
            raise ValueError("union-closure query inventories differ")
    camera_hashes = {
        str(meta.get("query_camera_only_inventory_file_sha256"))
        for metas in branch_meta for meta in metas
    }
    if camera_hashes != {str(primary_meta.get("query_camera_only_inventory_file_sha256"))}:
        raise ValueError("union-closure camera lineage differs")
    token_grids = {
        tuple(map(int, meta.get("token_grid", ()))) for metas in branch_meta for meta in metas
    }
    if len(token_grids) != 1:
        raise ValueError("union-closure token grids differ")
    token_grid = next(iter(token_grids))

    output_pose, output_usable, output_branch = [], [], []
    closure_ratio, closure_distance, closure_inliers = [], [], []
    applied_count = 0
    for index, name in enumerate(names.tolist()):
        pose = np.asarray(primary["pose_w2c"][index], np.float64)
        usable = bool(primary["usable"][index])
        if not usable:
            output_pose.append(np.full((4, 4), np.nan))
            output_usable.append(False)
            output_branch.append(60)
            closure_ratio.append(0.0)
            closure_distance.append(np.inf)
            closure_inliers.append(0)
            continue
        rows = [branch[index] for branch in branches]
        first = rows[0]
        if any(
            not np.array_equal(first["camera_matrix"], row["camera_matrix"])
            or float(first["radial_k1"]) != float(row["radial_k1"])
            for row in rows[1:]
        ):
            raise ValueError(f"union-closure cameras differ for {name}")
        world = np.concatenate([row["world_points"] for row in rows], axis=0)
        tokens = np.concatenate([row["query_tokens"] for row in rows], axis=0)
        candidate = _candidate(
            pose, world, tokens, first["camera_matrix"], float(first["radial_k1"]),
            token_grid, "robust_huber",
        )
        candidate_pose = np.asarray(candidate["pose"], np.float64)
        distance = _normalized_pose_distance(pose, candidate_pose)
        denominator = max(int(len(np.unique(tokens))), 1)
        ratio = float(candidate["inlier_count"]) / denominator
        apply = _closure_enabled(
            float(primary["selected_inlier_ratio"][index]), distance,
            maximum_primary_ratio=float(args.maximum_primary_ratio),
            maximum_pose_distance=float(args.maximum_pose_distance),
        )
        if apply:
            pose = _interpolate_pose(pose, candidate_pose, float(args.closure_fraction))
            applied_count += 1
        output_pose.append(pose)
        output_usable.append(True)
        output_branch.append(61 if apply else 60)
        closure_ratio.append(ratio)
        closure_distance.append(distance)
        closure_inliers.append(int(candidate["inlier_count"]))

    arrays = {
        "names": names,
        "pose_w2c": np.asarray(output_pose, np.float64),
        "usable": np.asarray(output_usable, bool),
        "selected_branch": np.asarray(output_branch, np.int16),
        "selected_inlier_ratio": np.asarray(primary["selected_inlier_ratio"], np.float64),
        "selected_candidate_correspondence_count": np.asarray(
            primary["selected_candidate_correspondence_count"], np.int64,
        ),
        "selected_pnp_inlier_count": np.asarray(primary["selected_pnp_inlier_count"], np.int64),
        "closure_inlier_ratio": np.asarray(closure_ratio, np.float64),
        "closure_normalized_pose_distance": np.asarray(closure_distance, np.float64),
        "closure_pnp_inlier_count": np.asarray(closure_inliers, np.int64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_damped_union_closure_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": "low_primary_support_and_bounded_closure_then_damped_se3",
        "maximum_primary_ratio": float(args.maximum_primary_ratio),
        "maximum_pose_distance": float(args.maximum_pose_distance),
        "pose_distance": "translation_m/0.5 + rotation_deg/5",
        "closure_fraction": float(args.closure_fraction),
        "closure_applied_count": int(applied_count),
        "primary_file_sha256": file_sha256(args.primary_pose_inventory),
        "primary_content_sha256": primary_meta.get("content_sha256"),
        "correspondence_file_sha256_by_branch": [
            [file_sha256(path) for path in paths] for paths in branch_paths
        ],
        "query_camera_only_inventory_file_sha256": primary_meta.get(
            "query_camera_only_inventory_file_sha256"
        ),
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "query_depth_or_scale_used_by_pose_solver": False,
        "configuration_role": "historical_validation_adaptive_ablation_not_pristine_blind",
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
