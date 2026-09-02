"""Select Top5/Top10 frozen plane-PnP candidates without pose labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _merge(paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    values: dict[str, list[np.ndarray]] = {}
    metadata = []
    for path in paths:
        arrays, meta = _load_frozen_poses(path)
        metadata.append(meta)
        for key, value in arrays.items():
            values.setdefault(key, []).append(np.asarray(value))
    return {key: np.concatenate(rows, axis=0) for key, rows in values.items()}, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.top5_pose_inventory) != len(args.top10_pose_inventory):
        raise ValueError("Top5/Top10 shard counts differ")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite selected PnP inventory")
    top5, meta5 = _merge(args.top5_pose_inventory)
    top10, meta10 = _merge(args.top10_pose_inventory)
    names = top5["names"].astype(str)
    if (
        not np.array_equal(names, top10["names"].astype(str))
        or len(set(names.tolist())) != len(names)
    ):
        raise ValueError("Top5/Top10 names differ")
    camera_hashes = {
        str(meta.get("query_camera_only_inventory_file_sha256"))
        for meta in meta5 + meta10
    }
    if not camera_hashes or "None" in camera_hashes:
        raise ValueError("Top5/Top10 camera lineage is missing")
    usable5 = np.asarray(top5["usable"], bool)
    usable10 = np.asarray(top10["usable"], bool)
    ratio5 = np.where(
        usable5,
        top5["pnp_inlier_count"] / np.maximum(top5["candidate_correspondence_count"], 1),
        -1.0,
    )
    ratio10 = np.where(
        usable10,
        top10["pnp_inlier_count"] / np.maximum(top10["candidate_correspondence_count"], 1),
        -1.0,
    )
    choose10 = ratio10 > ratio5
    indices = np.arange(len(names))
    arrays = {
        "names": names,
        "pose_w2c": np.where(
            choose10[:, None, None], top10["pose_w2c"], top5["pose_w2c"],
        ),
        "usable": np.where(choose10, usable10, usable5),
        "selected_branch": np.where(choose10, 10, 5).astype(np.int8),
        "selected_inlier_ratio": np.where(choose10, ratio10, ratio5).astype(np.float64),
        "selected_candidate_correspondence_count": np.where(
            choose10, top10["candidate_correspondence_count"], top5["candidate_correspondence_count"],
        ).astype(np.int64),
        "selected_pnp_inlier_count": np.where(
            choose10, top10["pnp_inlier_count"], top5["pnp_inlier_count"],
        ).astype(np.int64),
        "top5_inlier_ratio": ratio5.astype(np.float64),
        "top10_inlier_ratio": ratio10.astype(np.float64),
    }
    del indices
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_top5_top10_inlier_selected_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": "maximum_pnp_inlier_ratio_tie_top5",
        "selection_rule_frozen_on": "seq10_development_route",
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "render_consistency_used_for_selection": False,
        "render_consistency_ablation_result": "KILL_on_official_seq13_complement",
        "query_camera_only_inventory_file_sha256_sorted": sorted(camera_hashes),
        "top5_source_file_sha256_in_order": [file_sha256(path) for path in args.top5_pose_inventory],
        "top10_source_file_sha256_in_order": [file_sha256(path) for path in args.top10_pose_inventory],
        "selected_top10_count": int(np.sum(choose10)),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    print(json.dumps({
        **metadata,
        "output_file_sha256": file_sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
