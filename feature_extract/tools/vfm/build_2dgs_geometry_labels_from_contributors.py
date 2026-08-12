"""Build geometry-head labels from exact clean-2DGS contributor buffers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256
from feature_extract.vfm.vfm_2dgs_geometry_labels import _camera_facing_normals
from feature_extract.vfm.vfm_depth_head import RADIO_TOKEN_DEPTH_INVALID


def _downsample_front_surface(
    dominant_ids: np.ndarray,
    dominant_depth: np.ndarray,
    alpha: np.ndarray,
    *,
    output_height: int,
    output_width: int,
    minimum_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(dominant_ids, dtype=np.int64)
    depth = np.asarray(dominant_depth, dtype=np.float32)
    opacity = np.asarray(alpha, dtype=np.float32)
    if ids.shape != depth.shape or ids.shape != opacity.shape:
        raise ValueError("contributor geometry arrays differ")
    height, width = ids.shape
    if height % int(output_height) or width % int(output_width):
        raise ValueError("output geometry size must evenly divide contributor size")
    fy, fx = height // int(output_height), width // int(output_width)
    block_ids = ids.reshape(output_height, fy, output_width, fx).transpose(0, 2, 1, 3)
    block_depth = depth.reshape(output_height, fy, output_width, fx).transpose(0, 2, 1, 3)
    block_alpha = opacity.reshape(output_height, fy, output_width, fx).transpose(0, 2, 1, 3)
    valid = (
        (block_ids >= 0) & np.isfinite(block_depth) & (block_depth > 0.0)
        & (block_alpha >= float(minimum_alpha))
    )
    flat_depth = np.where(valid, block_depth, np.inf).reshape(
        output_height, output_width, -1
    )
    flat_ids = block_ids.reshape(output_height, output_width, -1)
    front_index = np.argmin(flat_depth, axis=2)
    front_depth = np.take_along_axis(flat_depth, front_index[..., None], axis=2)[..., 0]
    front_ids = np.take_along_axis(flat_ids, front_index[..., None], axis=2)[..., 0]
    support = np.sum(valid, axis=(2, 3)).astype(np.int32)
    output_alpha = np.mean(block_alpha, axis=(2, 3)).astype(np.float32)
    output_valid = support > 0
    front_ids[~output_valid] = -1
    front_depth[~output_valid] = float(RADIO_TOKEN_DEPTH_INVALID)
    return (
        front_ids.astype(np.int64),
        front_depth.astype(np.float32),
        output_alpha,
        support,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_width", type=int, default=128)
    parser.add_argument("--output_height", type=int, default=72)
    parser.add_argument("--minimum_alpha", type=float, default=0.05)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if int(args.shard_count) < 1 or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index/count")
    source_path = Path(args.gaussian_ply)
    source = load_gaussian_vfm_source_from_ply(source_path)
    if source.normal is None:
        raise ValueError("geometry labels require source 2DGS normals")
    paths = sorted(Path(args.contributors).glob("*.npz"))[
        int(args.shard_index) :: int(args.shard_count)
    ]
    output = Path(args.output_dir)
    label_dir = output / "geometry_npz"
    label_dir.mkdir(parents=True, exist_ok=True)
    records = []
    valid_ratios = []
    contributor_geometry_hashes: set[str] = set()
    contributor_clean_geometry_hashes: set[str] = set()
    contributor_clean_index_hashes: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            ids = np.asarray(data["topk_ids"], dtype=np.int64)
            weights = np.asarray(data["topk_weights"], dtype=np.float32)
            dominant_depth = np.asarray(data["dominant_depth"], dtype=np.float32)
            pose_w2c = np.asarray(data["pose_w2c"], dtype=np.float64)
        if not bool(metadata.get("uses_declared_clean_2dgs_for_occlusion", False)):
            raise ValueError(f"non-clean contributor cache: {path}")
        contributor_geometry_hashes.add(str(metadata["geometry_source_sha256"]))
        contributor_clean_geometry_hashes.add(
            str(metadata["clean_geometry_source_sha256"])
        )
        contributor_clean_index_hashes.add(str(metadata["clean_source_index_sha256"]))
        alpha = np.clip(np.sum(weights, axis=2), 0.0, 1.0)
        front_ids, depth, output_alpha, support = _downsample_front_surface(
            ids[..., 0], dominant_depth, alpha,
            output_height=int(args.output_height),
            output_width=int(args.output_width),
            minimum_alpha=float(args.minimum_alpha),
        )
        valid = front_ids >= 0
        normal_cam = np.zeros((*front_ids.shape, 3), dtype=np.float32)
        selected_ids = front_ids[valid]
        if selected_ids.size:
            if np.any(selected_ids >= source.xyz.shape[0]):
                raise ValueError("contributor ID exceeds source 2DGS")
            normal_cam[valid] = _camera_facing_normals(
                np.asarray(source.xyz)[selected_ids],
                np.asarray(source.normal)[selected_ids],
                pose_w2c,
                True,
            )
        destination = label_dir / f"{str(metadata['image_id']).replace('/', '__')}.npz"
        np.savez_compressed(
            destination,
            image_id=np.asarray(str(metadata["image_id"])),
            token_path=np.asarray(str(metadata["token_path"])),
            depth=depth,
            normal_cam=normal_cam,
            alpha=output_alpha,
            valid=valid,
            support_count=support,
        )
        valid_ratio = float(np.mean(valid))
        valid_ratios.append(valid_ratio)
        records.append({
            "image_id": str(metadata["image_id"]),
            "token_path": str(metadata["token_path"]),
            "geometry_path": str(destination),
            "split": "official_train",
            "scene": "StMarysChurch",
            "resolution": [int(args.output_width), int(args.output_height)],
            "valid_ratio": valid_ratio,
            "mean_alpha": float(np.mean(output_alpha[valid])) if np.any(valid) else 0.0,
            "median_depth_m": float(np.median(depth[valid])) if np.any(valid) else 0.0,
        })
    payload = {
        "stage": "exact_contributor_2dgs_geometry_labels_v1",
        "records": records,
        "inputs": {
            "contributors": str(args.contributors),
            "gaussian_ply": str(source_path),
            "gaussian_ply_sha256": file_sha256(source_path),
            "contributor_geometry_source_sha256": sorted(contributor_geometry_hashes),
            "contributor_clean_geometry_source_sha256": sorted(
                contributor_clean_geometry_hashes
            ),
            "contributor_clean_source_index_sha256": sorted(
                contributor_clean_index_hashes
            ),
            "contributor_image_ids_sha256": ordered_id_sha256(
                str(record["image_id"]) for record in records
            ),
        },
        "render": {
            "source": "gsplat_top4_clean_2dgs_contributor_buffer",
            "front_surface": "minimum_dominant_depth_in_downsample_block",
            "normal": "camera_facing_source_2dgs_normal",
            "alpha": "mean_clipped_top4_contributor_mass",
            "width": int(args.output_width),
            "height": int(args.output_height),
            "minimum_alpha": float(args.minimum_alpha),
        },
        "shard": {
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
        },
        "summary": {
            "record_count": len(records),
            "mean_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output / "geometry_label_summary.json").write_text(
        json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
