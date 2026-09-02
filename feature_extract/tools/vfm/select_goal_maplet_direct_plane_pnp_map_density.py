"""Select between frozen map-density PnP branches without pose labels."""

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
    metadata: list[dict[str, object]] = []
    for path in paths:
        arrays, meta = _load_frozen_poses(path)
        metadata.append(meta)
        for key, value in arrays.items():
            values.setdefault(key, []).append(np.asarray(value))
    return {key: np.concatenate(rows, axis=0) for key, rows in values.items()}, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--dense_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--sparse_view_count", type=int, required=True)
    parser.add_argument("--dense_view_count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite map-density PnP selection")
    if len(args.sparse_pose_inventory) != len(args.dense_pose_inventory):
        raise ValueError("map-density shard counts differ")
    if not 0 < int(args.sparse_view_count) < int(args.dense_view_count):
        raise ValueError("map-density view counts differ")

    sparse, sparse_meta = _merge(args.sparse_pose_inventory)
    dense, dense_meta = _merge(args.dense_pose_inventory)
    names = sparse["names"].astype(str)
    if (
        not np.array_equal(names, dense["names"].astype(str))
        or len(set(names.tolist())) != len(names)
    ):
        raise ValueError("map-density query inventories differ")
    camera_hashes = {
        str(meta.get("query_camera_only_inventory_file_sha256"))
        for meta in sparse_meta + dense_meta
    }
    if len(camera_hashes) != 1 or "None" in camera_hashes:
        raise ValueError("map-density camera lineage differs")

    sparse_usable = np.asarray(sparse["usable"], bool)
    dense_usable = np.asarray(dense["usable"], bool)
    sparse_ratio = np.where(
        sparse_usable,
        sparse["pnp_inlier_count"]
        / np.maximum(sparse["candidate_correspondence_count"], 1),
        -1.0,
    )
    dense_ratio = np.where(
        dense_usable,
        dense["pnp_inlier_count"]
        / np.maximum(dense["candidate_correspondence_count"], 1),
        -1.0,
    )
    # The denser map must strictly improve pose-free support; ties retain the
    # cheaper and historically stronger sparse-map branch.
    choose_dense = dense_ratio > sparse_ratio
    arrays = {
        "names": names,
        "pose_w2c": np.where(
            choose_dense[:, None, None], dense["pose_w2c"], sparse["pose_w2c"],
        ).astype(np.float64),
        "usable": np.where(choose_dense, dense_usable, sparse_usable),
        "selected_branch": np.where(
            choose_dense, int(args.dense_view_count), int(args.sparse_view_count),
        ).astype(np.int16),
        "selected_inlier_ratio": np.where(
            choose_dense, dense_ratio, sparse_ratio,
        ).astype(np.float64),
        "selected_candidate_correspondence_count": np.where(
            choose_dense,
            dense["candidate_correspondence_count"],
            sparse["candidate_correspondence_count"],
        ).astype(np.int64),
        "selected_pnp_inlier_count": np.where(
            choose_dense, dense["pnp_inlier_count"], sparse["pnp_inlier_count"],
        ).astype(np.int64),
        "sparse_inlier_ratio": sparse_ratio.astype(np.float64),
        "dense_inlier_ratio": dense_ratio.astype(np.float64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_map_density_inlier_selected_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "sparse_view_count": int(args.sparse_view_count),
        "dense_view_count": int(args.dense_view_count),
        "selection_rule": "maximum_pnp_inlier_ratio_tie_lower_map_density",
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_camera_only_inventory_file_sha256": next(iter(camera_hashes)),
        "sparse_source_file_sha256_in_order": [
            file_sha256(path) for path in args.sparse_pose_inventory
        ],
        "dense_source_file_sha256_in_order": [
            file_sha256(path) for path in args.dense_pose_inventory
        ],
        "sparse_source_content_sha256_in_order": [
            meta.get("content_sha256") for meta in sparse_meta
        ],
        "dense_source_content_sha256_in_order": [
            meta.get("content_sha256") for meta in dense_meta
        ],
        "selected_dense_count": int(np.sum(choose_dense)),
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
