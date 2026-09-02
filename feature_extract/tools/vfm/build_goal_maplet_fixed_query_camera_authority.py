"""Build a query camera-only authority without parsing any pose/GT file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256, canonical_json_sha256, file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--calibration_manifest", type=Path, required=True)
    parser.add_argument("--pose_archive_root", type=Path, required=True)
    parser.add_argument("--output_camera_manifest", type=Path, required=True)
    parser.add_argument("--output_camera_inventory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_camera_manifest.exists() or args.output_camera_inventory.exists():
        raise FileExistsError("refusing to overwrite camera-only authority")
    radio = json.loads(args.radio_manifest.read_text())
    names = sorted(str(row["image_id"]) for row in radio["records"])
    if len(names) != len(set(names)) or not names:
        raise ValueError("RADIO query names are empty or duplicated")
    calibration = json.loads(args.calibration_manifest.read_text())
    rows = list(calibration["cameras"].values())
    first = rows[0]
    if any(row != first for row in rows[1:]):
        raise ValueError("fixed calibration is not constant")
    model = int(first["model_id"]); width = int(first["width"]); height = int(first["height"])
    params = np.asarray(first["params"], np.float64)
    cameras = {name: {
        "model_id": model, "width": width, "height": height, "params": params.tolist(),
    } for name in names}
    payload = {
        "format": "per_query_colmap_calibration_only_v1", "query_count": len(names),
        "cameras": cameras,
        "production_contract": {
            "contains_camera_pose": False, "contains_sfm_points": False,
            "contains_sfm_tracks": False, "pose_or_ground_truth_file_parsed": False,
        },
        "lineage": {
            "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
            "calibration_manifest_file_sha256": file_sha256(args.calibration_manifest),
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output_camera_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_camera_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    archive_names = [name.replace("/", "__") + ".npz" for name in names]
    archive_hashes = []
    for name in archive_names:
        path = args.pose_archive_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        # Bind bytes only.  No NPZ member, including pose_w2c, is opened.
        archive_hashes.append(file_sha256(path))
    arrays = {
        "names": np.asarray(archive_names),
        "camera_model_id": np.full(len(names), model, np.int32),
        "camera_width": np.full(len(names), width, np.int32),
        "camera_height": np.full(len(names), height, np.int32),
        "camera_params": np.repeat(params[None], len(names), axis=0),
        "source_contributor_file_sha256": np.asarray(archive_hashes),
    }
    metadata = {
        "artifact_type": "goal_maplet_query_camera_only_inventory_v1",
        "query_count": len(names), "pose_or_ground_truth_member_read": False,
        "pose_or_ground_truth_file_parsed": False,
        "source_archives_are_pose_bearing": True,
        "consumer_may_use_before_pose_freeze": True,
        "camera_manifest_file_sha256": file_sha256(args.output_camera_manifest),
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    np.savez_compressed(
        args.output_camera_inventory, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        "query_count": len(names), "pose_or_ground_truth_file_parsed": False,
        "camera_manifest_file_sha256": file_sha256(args.output_camera_manifest),
        "camera_inventory_file_sha256": file_sha256(args.output_camera_inventory),
        "camera_inventory_content_sha256": metadata["content_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
