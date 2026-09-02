"""Conservatively fuse frozen Top-5 and Top-10 plane-PnP pose budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_map_density import (
    _midpoint_pose,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


KEYS = (
    "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
    "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
    "sparse_inlier_ratio", "dense_inlier_ratio",
)


def _interpolate_pose(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate camera centre and rotation without mixing w2c translations."""
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("pose interpolation fraction must be in [0, 1]")
    rotations = Rotation.from_matrix(np.asarray([left[:3, :3], right[:3, :3]]))
    rotation = Slerp([0.0, 1.0], rotations)([float(fraction)]).as_matrix()[0]
    left_center = -left[:3, :3].T @ left[:3, 3]
    right_center = -right[:3, :3].T @ right[:3, 3]
    center = (1.0 - float(fraction)) * left_center + float(fraction) * right_center
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = -rotation @ center
    return output


def _agreement_shrunk_fraction(
    translation_disagreement_m: float,
    rotation_disagreement_deg: float,
    translation_limit_m: float = 0.5,
    rotation_limit_deg: float = 5.0,
) -> float:
    """Shrink the auxiliary Top-10 contribution as pose disagreement grows."""
    if translation_limit_m <= 0.0 or rotation_limit_deg <= 0.0:
        raise ValueError("agreement limits must be positive")
    normalized = max(
        float(translation_disagreement_m) / float(translation_limit_m),
        float(rotation_disagreement_deg) / float(rotation_limit_deg),
    )
    if not np.isfinite(normalized) or normalized > 1.0:
        return 0.0
    return float(0.5 * np.exp(-max(0.0, normalized)))


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in KEYS}
    if (
        metadata.get("artifact_type")
        != "goal_maplet_direct_plane_pnp_map_density_inlier_selected_v1"
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("strict_runtime_phase_separation_eligible") is not True
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("map-density union pose inventory differs")
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5_pose_inventory", type=Path, required=True)
    parser.add_argument("--top10_pose_inventory", type=Path, required=True)
    parser.add_argument(
        "--consensus_mode", choices=("midpoint", "agreement_shrunk"),
        default="midpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite budget-consensus poses")
    top5, top5_meta = _load(args.top5_pose_inventory)
    top10, top10_meta = _load(args.top10_pose_inventory)
    names = top5["names"].astype(str)
    if (
        not np.array_equal(names, top10["names"].astype(str))
        or len(set(names.tolist())) != len(names)
        or top5_meta.get("query_camera_only_inventory_file_sha256")
        != top10_meta.get("query_camera_only_inventory_file_sha256")
    ):
        raise ValueError("Top-5 and Top-10 query inventories differ")

    pose = []
    usable = []
    selected_branch = []
    selected_ratio = []
    selected_candidates = []
    selected_inliers = []
    midpoint_count = agreement_shrunk_count = top10_fallback_count = 0
    interpolation_fractions = []
    for index in range(len(names)):
        use5 = bool(top5["usable"][index])
        use10 = bool(top10["usable"][index])
        branch = 5
        selected = top5
        selected_pose = np.asarray(top5["pose_w2c"][index], np.float64)
        if not use5 and use10:
            branch = 10
            selected = top10
            selected_pose = np.asarray(top10["pose_w2c"][index], np.float64)
            top10_fallback_count += 1
        elif use5 and use10:
            pose5 = np.asarray(top5["pose_w2c"][index], np.float64)
            pose10 = np.asarray(top10["pose_w2c"][index], np.float64)
            center5 = -pose5[:3, :3].T @ pose5[:3, 3]
            center10 = -pose10[:3, :3].T @ pose10[:3, 3]
            translation = float(np.linalg.norm(center5 - center10))
            rotation = float(
                Rotation.from_matrix(pose5[:3, :3] @ pose10[:3, :3].T).magnitude()
                * 180.0 / np.pi
            )
            if translation <= 0.5 and rotation <= 5.0:
                branch = 15
                if args.consensus_mode == "midpoint":
                    fraction = 0.5
                    selected_pose = _midpoint_pose(pose5, pose10)
                    midpoint_count += 1
                else:
                    fraction = _agreement_shrunk_fraction(translation, rotation)
                    selected_pose = _interpolate_pose(pose5, pose10, fraction)
                    agreement_shrunk_count += 1
                interpolation_fractions.append(fraction)
        valid = use5 or (not use5 and use10)
        pose.append(selected_pose if valid else np.full((4, 4), np.nan))
        usable.append(valid)
        selected_branch.append(branch)
        if branch == 15:
            selected_ratio.append(float(min(
                top5["selected_inlier_ratio"][index],
                top10["selected_inlier_ratio"][index],
            )))
            selected_candidates.append(int(max(
                top5["selected_candidate_correspondence_count"][index],
                top10["selected_candidate_correspondence_count"][index],
            )))
            selected_inliers.append(int(min(
                top5["selected_pnp_inlier_count"][index],
                top10["selected_pnp_inlier_count"][index],
            )))
        else:
            selected_ratio.append(float(selected["selected_inlier_ratio"][index]))
            selected_candidates.append(int(
                selected["selected_candidate_correspondence_count"][index]
            ))
            selected_inliers.append(int(selected["selected_pnp_inlier_count"][index]))

    arrays = {
        "names": names,
        "pose_w2c": np.asarray(pose, np.float64),
        "usable": np.asarray(usable, bool),
        "selected_branch": np.asarray(selected_branch, np.int16),
        "selected_inlier_ratio": np.asarray(selected_ratio, np.float64),
        "selected_candidate_correspondence_count": np.asarray(selected_candidates, np.int64),
        "selected_pnp_inlier_count": np.asarray(selected_inliers, np.int64),
        "top5_inlier_ratio": np.asarray(top5["selected_inlier_ratio"], np.float64),
        "top10_inlier_ratio": np.asarray(top10["selected_inlier_ratio"], np.float64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_top5_top10_pose_consensus_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": (
            "top5_default_top10_fallback_same_basin_se3_midpoint"
            if args.consensus_mode == "midpoint"
            else "top5_default_top10_fallback_same_basin_agreement_shrunk_se3"
        ),
        "consensus_mode": str(args.consensus_mode),
        "same_basin_translation_limit_m": 0.5,
        "same_basin_rotation_limit_deg": 5.0,
        "midpoint_count": int(midpoint_count),
        "agreement_shrunk_count": int(agreement_shrunk_count),
        "interpolation_fraction_summary": {
            "count": int(len(interpolation_fractions)),
            "minimum": float(np.min(interpolation_fractions)) if interpolation_fractions else None,
            "median": float(np.median(interpolation_fractions)) if interpolation_fractions else None,
            "maximum": float(np.max(interpolation_fractions)) if interpolation_fractions else None,
        },
        "top10_fallback_count": int(top10_fallback_count),
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_camera_only_inventory_file_sha256": top5_meta.get(
            "query_camera_only_inventory_file_sha256"
        ),
        "top5_source_file_sha256": file_sha256(args.top5_pose_inventory),
        "top5_source_content_sha256": top5_meta.get("content_sha256"),
        "top10_source_file_sha256": file_sha256(args.top10_pose_inventory),
        "top10_source_content_sha256": top10_meta.get("content_sha256"),
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
