"""Build a pose-free source-view planar context descriptor per observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_field", type=Path, required=True)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite plane context field")
    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    with np.load(args.plane_field, allow_pickle=False) as data:
        field_meta = json.loads(str(data["metadata_json"].item()))
        field_arrays = {name: np.asarray(data[name]) for name in (
            "plane_offsets", "observation_plane_rows", "observation_descriptors",
        )}
    if (
        field_meta.get("artifact_type") != "goal_maplet_direct_radio_plane_observation_field_v1"
        or arrays_sha256(field_arrays) != field_meta.get("arrays_sha256")
        or field_meta.get("visibility_atlas_content_sha256") != atlas_meta.get("content_sha256")
        or not np.array_equal(field_arrays["plane_offsets"], atlas.plane_offsets)
        or len(field_arrays["observation_descriptors"]) != len(atlas.view_names)
    ):
        raise ValueError("plane field and visibility atlas contracts differ")

    descriptors = np.asarray(field_arrays["observation_descriptors"], np.float64)
    context = np.zeros_like(descriptors, dtype=np.float32)
    names = atlas.view_names.astype(str)
    for name in sorted(set(names.tolist())):
        rows = np.flatnonzero(names == name)
        weights = np.asarray(atlas.token_pixel_counts[rows].sum((1, 2)), np.float64)
        value = np.sum(descriptors[rows] * weights[:, None], axis=0)
        value /= max(float(np.linalg.norm(value)), 1e-12)
        context[rows] = value.astype(np.float32)
    arrays = {
        "plane_offsets": np.asarray(atlas.plane_offsets, np.int64),
        "observation_context_descriptors": context,
    }
    metadata = {
        "artifact_type": "goal_maplet_plane_observation_context_field_v1",
        "context_semantics": "view-level token-mass-weighted mean of finite-plane observation RADIO descriptors",
        "plane_count": int(len(atlas.plane_offsets) - 1),
        "observation_count": int(len(context)),
        "source_view_count": int(len(set(names.tolist()))),
        "uses_pose_or_ground_truth": False,
        "plane_field_file_sha256": file_sha256(args.plane_field),
        "plane_field_content_sha256": field_meta.get("content_sha256"),
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
