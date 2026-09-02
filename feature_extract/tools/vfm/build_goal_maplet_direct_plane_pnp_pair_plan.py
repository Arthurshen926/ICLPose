"""Freeze a Top5/Top10 direct-plane PnP candidate comparison plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _pose_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    a = np.asarray(left, np.float64).reshape(4, 4)
    b = np.asarray(right, np.float64).reshape(4, 4)
    center_a = -a[:3, :3].T @ a[:3, 3]
    center_b = -b[:3, :3].T @ b[:3, 3]
    translation = float(np.linalg.norm(center_a - center_b))
    rotation = float(
        Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()
        * 180.0 / np.pi
    )
    return translation, rotation


def _save_subset(
    path: Path,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    source_paths: list[Path],
    camera_file_sha256: str,
    branch: str,
) -> dict[str, object]:
    subset = {key: np.asarray(value)[indices] for key, value in arrays.items()}
    metadata = {
        "artifact_type": "goal_maplet_frozen_direct_plane_pnp_pose_inventory_v1",
        "arrays_sha256": arrays_sha256(subset),
        "query_count": int(len(indices)),
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_moge3_role": "plane_segmentation_only",
        "candidate_branch": str(branch),
        "query_camera_only_inventory_file_sha256": str(camera_file_sha256),
        "source_pose_inventory_file_sha256_in_order": [file_sha256(p) for p in source_paths],
        "subset_role": "divergent_candidate_render_verification_only",
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    if path.exists():
        raise FileExistsError("refusing to overwrite candidate subset")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary.npz")
    np.savez_compressed(
        temporary,
        **subset,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum_same_basin_translation_m", type=float, default=0.5)
    parser.add_argument("--maximum_same_basin_rotation_deg", type=float, default=5.0)
    parser.add_argument("--output_top5_divergent", type=Path, required=True)
    parser.add_argument("--output_top10_divergent", type=Path, required=True)
    parser.add_argument("--output_plan", type=Path, required=True)
    args = parser.parse_args()
    if len(args.top5_pose_inventory) != len(args.top10_pose_inventory):
        raise ValueError("Top5/Top10 inventory shard counts differ")
    if args.output_plan.exists():
        raise FileExistsError("refusing to overwrite PnP pair plan")

    merged: dict[int, dict[str, list[np.ndarray]]] = {
        5: {}, 10: {},
    }
    metas: dict[int, list[dict[str, object]]] = {5: [], 10: []}
    for branch, paths in ((5, args.top5_pose_inventory), (10, args.top10_pose_inventory)):
        for path in paths:
            arrays, metadata = _load_frozen_poses(path)
            metas[branch].append(metadata)
            for key, value in arrays.items():
                merged[branch].setdefault(key, []).append(np.asarray(value))
        merged[branch] = {
            key: np.concatenate(value, axis=0) for key, value in merged[branch].items()
        }
    names5 = merged[5]["names"].astype(str)
    names10 = merged[10]["names"].astype(str)
    if (
        not np.array_equal(names5, names10)
        or len(set(names5.tolist())) != len(names5)
    ):
        raise ValueError("paired candidate inventories differ in name/order")
    camera_hashes = {
        str(meta.get("query_camera_only_inventory_file_sha256"))
        for branch in (5, 10) for meta in metas[branch]
    }
    if len(camera_hashes) != 1:
        raise ValueError("paired candidate camera inventories differ")
    camera_hash = next(iter(camera_hashes))

    rows = []
    divergent = []
    for index, name in enumerate(names5.tolist()):
        usable5 = bool(merged[5]["usable"][index])
        usable10 = bool(merged[10]["usable"][index])
        ratio5 = (
            float(merged[5]["pnp_inlier_count"][index])
            / max(1, int(merged[5]["candidate_correspondence_count"][index]))
            if usable5 else -1.0
        )
        ratio10 = (
            float(merged[10]["pnp_inlier_count"][index])
            / max(1, int(merged[10]["candidate_correspondence_count"][index]))
            if usable10 else -1.0
        )
        if usable5 and usable10:
            translation, rotation = _pose_distance(
                merged[5]["pose_w2c"][index], merged[10]["pose_w2c"][index],
            )
            requires_render = (
                translation > float(args.maximum_same_basin_translation_m)
                or rotation > float(args.maximum_same_basin_rotation_deg)
            )
        else:
            translation = rotation = None
            requires_render = False
        if requires_render:
            divergent.append(index)
            preliminary = None
        else:
            preliminary = 5 if ratio5 >= ratio10 else 10
        rows.append({
            "name": name,
            "top5_usable": usable5,
            "top10_usable": usable10,
            "top5_inlier_ratio": ratio5,
            "top10_inlier_ratio": ratio10,
            "candidate_translation_difference_m": translation,
            "candidate_rotation_difference_deg": rotation,
            "requires_render_comparison": bool(requires_render),
            "preliminary_selected_branch": preliminary,
        })
    indices = np.asarray(divergent, np.int64)
    meta5 = _save_subset(
        args.output_top5_divergent, merged[5], indices,
        source_paths=args.top5_pose_inventory, camera_file_sha256=camera_hash,
        branch="top5",
    )
    meta10 = _save_subset(
        args.output_top10_divergent, merged[10], indices,
        source_paths=args.top10_pose_inventory, camera_file_sha256=camera_hash,
        branch="top10",
    )
    report = {
        "artifact_type": "goal_maplet_direct_plane_pnp_top5_top10_pair_plan_v1",
        "query_count": int(len(rows)),
        "divergent_pair_count": int(len(indices)),
        "same_basin_translation_m": float(args.maximum_same_basin_translation_m),
        "same_basin_rotation_deg": float(args.maximum_same_basin_rotation_deg),
        "same_basin_selection": "maximum_pnp_inlier_ratio_tie_top5",
        "divergent_selection": "minimum_scale_fitted_absolute_log_depth_median",
        "query_pose_or_ground_truth_read": False,
        "threshold_semantics": "frozen_from_seq10_development_accuracy_basin",
        "top5_source_file_sha256_in_order": [file_sha256(p) for p in args.top5_pose_inventory],
        "top10_source_file_sha256_in_order": [file_sha256(p) for p in args.top10_pose_inventory],
        "top5_divergent_file_sha256": file_sha256(args.output_top5_divergent),
        "top5_divergent_content_sha256": meta5["content_sha256"],
        "top10_divergent_file_sha256": file_sha256(args.output_top10_divergent),
        "top10_divergent_content_sha256": meta10["content_sha256"],
        "production_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
