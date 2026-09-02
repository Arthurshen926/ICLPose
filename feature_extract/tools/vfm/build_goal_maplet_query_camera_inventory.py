"""Extract a camera-only query inventory without reading pose/GT members."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite camera-only inventory")
    names, model, width, height, params, hashes = [], [], [], [], [], []
    for plane_path in sorted(args.query_plane_dir.glob("*.npz")):
        path = args.contributors / plane_path.name
        with np.load(path, allow_pickle=False) as data:
            # Intentionally never index pose_w2c or any label/error member.
            names.append(plane_path.name)
            model.append(int(data["camera_model_id"]))
            width.append(int(data["camera_width"]))
            height.append(int(data["camera_height"]))
            value = np.asarray(data["camera_params"], np.float64).reshape(-1)
        if value.shape != (4,):
            raise ValueError("camera parameter shape differs")
        params.append(value); hashes.append(file_sha256(path))
    arrays = {
        "names": np.asarray(names),
        "camera_model_id": np.asarray(model, np.int32),
        "camera_width": np.asarray(width, np.int32),
        "camera_height": np.asarray(height, np.int32),
        "camera_params": np.asarray(params, np.float64).reshape(-1, 4),
        "source_contributor_file_sha256": np.asarray(hashes),
    }
    metadata = {
        "artifact_type": "goal_maplet_query_camera_only_inventory_v1",
        "query_count": len(names),
        "pose_or_ground_truth_member_read": False,
        "source_archives_are_pose_bearing": True,
        "consumer_may_use_before_pose_freeze": True,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
