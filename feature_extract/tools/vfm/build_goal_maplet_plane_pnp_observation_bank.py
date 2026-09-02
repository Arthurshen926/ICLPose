"""Pack exact finite-plane source tokens, RADIO features, and metric 3D points."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    MINIMUM_PLANE_TOKEN_PIXELS,
    _radio,
    _records,
    _token_world_points,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--mapping_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radio_dtype", choices=("float16", "float32"), default="float32")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite plane PnP observation bank")
    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    token_grid = tuple(map(int, atlas.token_pixel_counts.shape[1:]))
    records = _records([args.radio_manifest])
    by_view: dict[str, list[int]] = defaultdict(list)
    for row, name in enumerate(atlas.view_names.astype(str).tolist()):
        by_view[name].append(row)
    token_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    point_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    feature_rows: list[np.ndarray | None] = [None] * len(atlas.view_names)
    for index, (name, observation_rows) in enumerate(sorted(by_view.items())):
        feature = _radio(name, records)
        if feature.shape[0] != int(np.prod(token_grid)):
            raise ValueError("RADIO and visibility token grids differ")
        requested = [
            np.flatnonzero(
                atlas.token_pixel_counts[row].reshape(-1) >= MINIMUM_PLANE_TOKEN_PIXELS
            ).astype(np.int64)
            for row in observation_rows
        ]
        union = np.unique(np.concatenate(requested)) if requested else np.zeros(0, np.int64)
        points, kept = _token_world_points(
            args.mapping_contributors / name, union, token_grid=token_grid,
        )
        lookup = {int(token): row for row, token in enumerate(kept.tolist())}
        for atlas_row, tokens in zip(observation_rows, requested):
            selected = np.asarray([lookup[int(token)] for token in tokens if int(token) in lookup], np.int64)
            valid_tokens = np.asarray([token for token in tokens if int(token) in lookup], np.int16)
            token_rows[atlas_row] = valid_tokens
            point_rows[atlas_row] = points[selected].astype(np.float64)
            feature_rows[atlas_row] = feature[valid_tokens.astype(np.int64)].astype(args.radio_dtype)
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{len(by_view)} source views", flush=True)
    if any(row is None for row in token_rows + point_rows + feature_rows):
        raise ValueError("observation bank construction left an unfilled row")
    counts = np.asarray([len(row) for row in token_rows], np.int64)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    arrays = {
        "observation_offsets": offsets,
        "token_ids": np.concatenate(token_rows).astype(np.int16),
        "world_points": np.concatenate(point_rows).astype(np.float64),
        "radio_features": np.concatenate(feature_rows).astype(args.radio_dtype),
    }
    metadata = {
        "artifact_type": "goal_maplet_plane_pnp_observation_bank_v1",
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
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately uncompressed: this is a deployment/read-speed bank, not an archive.
    np.savez(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
