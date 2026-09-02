"""Build a pose/label-free RADIO descriptor for every finite-plane observation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _radio, _records
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum_token_pixels", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite direct plane RADIO field")
    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    token_grid = tuple(map(int, atlas.token_pixel_counts.shape[1:]))
    records = _records([args.radio_manifest])
    by_view: dict[str, list[int]] = defaultdict(list)
    for row, name in enumerate(atlas.view_names.astype(str).tolist()):
        by_view[name].append(row)
    descriptors = np.zeros((len(atlas.view_names), 1280), np.float32)
    for index, (name, rows) in enumerate(sorted(by_view.items())):
        feature = _radio(name, records)
        if feature.shape[0] != int(np.prod(token_grid)):
            raise ValueError("RADIO and visibility token grids differ")
        for row in rows:
            tokens = np.flatnonzero(
                atlas.token_pixel_counts[row].reshape(-1) >= int(args.minimum_token_pixels)
            )
            if len(tokens):
                value = np.mean(feature[tokens], axis=0)
                descriptors[row] = value / max(float(np.linalg.norm(value)), 1e-8)
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{len(by_view)} source views", flush=True)
    observation_plane_rows = np.repeat(
        np.arange(len(atlas.plane_offsets) - 1, dtype=np.int32),
        np.diff(atlas.plane_offsets),
    )
    arrays = {
        "plane_offsets": np.asarray(atlas.plane_offsets, np.int64),
        "observation_plane_rows": observation_plane_rows,
        "observation_descriptors": descriptors,
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_radio_plane_observation_field_v1",
        "plane_count": int(len(atlas.plane_offsets) - 1),
        "observation_count": int(len(descriptors)),
        "source_view_count": int(len(by_view)),
        "minimum_token_pixels": int(args.minimum_token_pixels),
        "token_grid": list(token_grid),
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "uses_pose_or_ground_truth": False,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
