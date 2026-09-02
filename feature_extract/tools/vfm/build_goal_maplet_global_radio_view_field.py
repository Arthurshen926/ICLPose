"""Build mapping-view global RADIO descriptors and pose geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_pnp_pose_conditioned_view_context import (
    _pose_center_forward,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _radio, _records
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--mapping_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite global RADIO view field")
    atlas, atlas_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    names = np.asarray(sorted(set(atlas.view_names.astype(str).tolist())))
    records = _records(args.radio_manifest)
    descriptors = []
    centers = []
    forwards = []
    contributor_hashes = []
    for index, name in enumerate(names.tolist()):
        feature = _radio(name, records)
        descriptor = np.mean(feature, axis=0)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)
        contributor = args.mapping_contributors / name
        with np.load(contributor, allow_pickle=False) as data:
            pose = np.asarray(data["pose_w2c"], np.float64)
        center, forward = _pose_center_forward(pose)
        descriptors.append(descriptor)
        centers.append(center)
        forwards.append(forward)
        contributor_hashes.append(file_sha256(contributor))
        if (index + 1) % 100 == 0:
            print(json.dumps({"completed": index + 1, "total": len(names)}))
    arrays = {
        "names": names,
        "global_radio_descriptors": np.asarray(descriptors, np.float32),
        "camera_centers_world": np.asarray(centers, np.float64),
        "camera_forwards_world": np.asarray(forwards, np.float64),
        "contributor_file_sha256": np.asarray(contributor_hashes),
    }
    metadata = {
        "artifact_type": "goal_maplet_global_radio_mapping_view_field_v1",
        "view_count": int(len(names)),
        "descriptor_semantics": "unit_normalized_mean_of_all_unit_RADIO_final_tokens",
        "mapping_pose_role": "candidate_conditioned_neighborhood_only",
        "uses_query_pose_or_ground_truth": False,
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": atlas_meta.get("content_sha256"),
        "radio_manifest_file_sha256_in_order": [file_sha256(path) for path in args.radio_manifest],
        "arrays_sha256": arrays_sha256(arrays),
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
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
