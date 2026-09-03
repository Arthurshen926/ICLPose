"""Pack exact finite-plane source tokens, RADIO features, and metric 3D points."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    MINIMUM_PLANE_TOKEN_PIXELS,
    _radio,
    _records,
    _scaled_intrinsics,
    _token_world_points,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import inverse_simple_radial
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


def _load_contributor_geometry(contributor: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(contributor, allow_pickle=False) as data:
        depth = np.asarray(data["dominant_depth"], np.float64)
        pose = np.asarray(data["pose_w2c"], np.float64)
        model_id = int(data["camera_model_id"]); width = int(data["camera_width"]); height = int(data["camera_height"])
        params = np.asarray(data["camera_params"], np.float64)
    K, k1 = _scaled_intrinsics(model_id, params, width, height)
    return depth, pose, K, k1


def _plane_token_world_statistics(
    contributor: Path | tuple[np.ndarray, np.ndarray, np.ndarray, float],
    token_ids: np.ndarray,
    plane_mask: np.ndarray,
    *,
    token_grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate metric geometry only from pixels assigned to this plane."""
    depth, pose, K, k1 = (
        _load_contributor_geometry(contributor) if isinstance(contributor, Path) else contributor
    )
    mask = np.asarray(plane_mask, bool)
    if depth.shape != (144, 256) or mask.shape != depth.shape:
        raise ValueError("plane mask and mapping depth grid differ")
    token_height, token_width = map(int, token_grid)
    valid_depth = np.isfinite(depth) & (depth > 0)
    rotation, translation = pose[:3, :3], pose[:3, 3]; center = -rotation.T @ translation
    points=[]; covariance=[]; purity=[]; dispersion=[]; kept=[]
    for token in np.asarray(token_ids, np.int64):
        ty, tx = divmod(int(token), token_width)
        y0, y1 = ty * depth.shape[0] // token_height, (ty + 1) * depth.shape[0] // token_height
        x0, x1 = tx * depth.shape[1] // token_width, (tx + 1) * depth.shape[1] // token_width
        block_valid = valid_depth[y0:y1, x0:x1]
        selected = block_valid & mask[y0:y1, x0:x1]; denominator = int(np.sum(block_valid))
        yy, xx = np.nonzero(selected); yy += y0; xx += x0
        if not len(xx): continue
        z = depth[yy, xx]; pixel = np.c_[xx, yy].astype(np.float64)
        distorted = np.c_[(pixel[:, 0] - K[0, 2]) / K[0, 0], (pixel[:, 1] - K[1, 2]) / K[1, 1]]
        ideal = inverse_simple_radial(distorted, k1); camera = np.c_[ideal * z[:, None], z]
        world = camera @ rotation + center; location = np.median(world, axis=0)
        delta = world - location; cov = delta.T @ delta / max(len(world), 1)
        points.append(location); covariance.append(cov); purity.append(len(world) / max(denominator, 1))
        dispersion.append(float(np.sqrt(np.mean((z - np.median(z)) ** 2)))); kept.append(int(token))
    return (
        np.asarray(points, np.float64).reshape(-1, 3),
        np.asarray(covariance, np.float64).reshape(-1, 3, 3),
        np.asarray(purity, np.float32), np.asarray(dispersion, np.float32), np.asarray(kept, np.int64),
    )


def _observation_lookup(directory: Path) -> dict[int, tuple[Path, int]]:
    lookup: dict[int, tuple[Path, int]] = {}; row = 0
    for path in sorted(directory.glob("*.planes.npz")):
        with np.load(path, allow_pickle=False) as data:
            count = len(data["normals_world"])
        for local in range(count): lookup[row] = (path, local); row += 1
    return lookup


def _tree_sha256(directory: Path) -> str:
    rows = [(str(path.relative_to(directory)), file_sha256(path)) for path in sorted(directory.glob("*.planes.npz"))]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--mapping_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planar_observation_dir", type=Path)
    parser.add_argument("--radio_dtype", choices=("float16", "float32"), default="float32")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite plane PnP observation bank")
    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    plane_specific = args.planar_observation_dir is not None
    observation_lookup = _observation_lookup(args.planar_observation_dir) if plane_specific else {}
    token_grid = tuple(map(int, atlas.token_pixel_counts.shape[1:]))
    records = _records([args.radio_manifest])
    by_view: dict[str, list[int]] = defaultdict(list)
    for row, name in enumerate(atlas.view_names.astype(str).tolist()):
        by_view[name].append(row)
    token_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    point_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    feature_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    covariance_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    purity_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    dispersion_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    for index, (name, observation_rows) in enumerate(sorted(by_view.items())):
        feature = _radio(name, records)
        label_cache: dict[Path, np.ndarray] = {}
        contributor_geometry = _load_contributor_geometry(args.mapping_contributors / name) if plane_specific else None
        if feature.shape[0] != int(np.prod(token_grid)):
            raise ValueError("RADIO and visibility token grids differ")
        requested = [
            np.flatnonzero(
                atlas.token_pixel_counts[row].reshape(-1) >= MINIMUM_PLANE_TOKEN_PIXELS
            ).astype(np.int64)
            for row in observation_rows
        ]
        union = np.unique(np.concatenate(requested)) if requested else np.zeros(0, np.int64)
        for atlas_row, tokens in zip(observation_rows, requested):
            if plane_specific:
                global_row = int(atlas.plane_observation_rows[atlas_row])
                if global_row not in observation_lookup: raise ValueError("visibility row lacks source plane observation")
                observation_path, local_plane = observation_lookup[global_row]
                if observation_path not in label_cache:
                    with np.load(observation_path, allow_pickle=False) as data:
                        label_cache[observation_path] = np.asarray(data["labels"], np.int32)
                plane_mask = label_cache[observation_path] == local_plane
                points, covariance, purity, dispersion, kept = _plane_token_world_statistics(
                    contributor_geometry, tokens, plane_mask, token_grid=token_grid,
                )
            else:
                points, kept = _token_world_points(args.mapping_contributors / name, tokens, token_grid=token_grid)
                covariance = np.zeros((len(kept), 3, 3), np.float64); purity = np.ones(len(kept), np.float32); dispersion = np.zeros(len(kept), np.float32)
            lookup = {int(token): row for row, token in enumerate(kept.tolist())}
            selected = np.asarray([lookup[int(token)] for token in tokens if int(token) in lookup], np.int64)
            valid_tokens = np.asarray([token for token in tokens if int(token) in lookup], np.int16)
            token_rows[atlas_row] = valid_tokens
            point_rows[atlas_row] = points[selected].astype(np.float64)
            feature_rows[atlas_row] = feature[valid_tokens.astype(np.int64)].astype(args.radio_dtype)
            covariance_rows[atlas_row] = covariance[selected]; purity_rows[atlas_row] = purity[selected]; dispersion_rows[atlas_row] = dispersion[selected]
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{len(by_view)} source views", flush=True)
    if any(row is None for row in token_rows + point_rows + feature_rows + covariance_rows + purity_rows + dispersion_rows):
        raise ValueError("observation bank construction left an unfilled row")
    counts = np.asarray([len(row) for row in token_rows], np.int64)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    arrays = {
        "observation_offsets": offsets,
        "token_ids": np.concatenate(token_rows).astype(np.int16),
        "world_points": np.concatenate(point_rows).astype(np.float64),
        "radio_features": np.concatenate(feature_rows).astype(args.radio_dtype),
        "world_point_covariance_m2": np.concatenate(covariance_rows).astype(np.float32),
        "plane_pixel_purity": np.concatenate(purity_rows).astype(np.float32),
        "plane_depth_dispersion_m": np.concatenate(dispersion_rows).astype(np.float32),
    }
    metadata = {
        "artifact_type": "goal_maplet_plane_pnp_observation_bank_v2",
        "observation_count": int(len(token_rows)),
        "token_point_count": int(offsets[-1]),
        "radio_dtype": str(args.radio_dtype),
        "world_point_dtype": "float64",
        "minimum_plane_token_pixels": MINIMUM_PLANE_TOKEN_PIXELS,
        "token_grid": list(token_grid),
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "uses_query_pose_or_ground_truth": False,
        "plane_specific_geometry": plane_specific,
        "plane_pixel_geometry_semantics": "only_pixels_with_exact_rendered_plane_observation_label",
        "planar_observation_dir_tree_note": "sorted plane observation inventory replayed by global observation row",
        "planar_observation_tree_sha256": _tree_sha256(args.planar_observation_dir) if plane_specific else None,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately uncompressed: this is a deployment/read-speed bank, not an archive.
    np.savez(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
