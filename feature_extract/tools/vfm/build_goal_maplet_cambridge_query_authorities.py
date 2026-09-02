"""Build separated camera-only and pose-bearing Cambridge query authorities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_file", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--calibration_manifest", type=Path, required=True)
    parser.add_argument("--output_camera_manifest", type=Path, required=True)
    parser.add_argument("--output_camera_inventory", type=Path, required=True)
    parser.add_argument("--output_pose_archives", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_camera_manifest, args.output_camera_inventory, args.output_pose_archives)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite query authorities")

    radio_payload = json.loads(args.radio_manifest.read_text())
    records = {str(row["image_id"]): row for row in radio_payload["records"]}
    poses = {row.image_id: np.asarray(row.pose_w2c, np.float64) for row in parse_cambridge_pose_file(args.pose_file)}
    names = sorted(set(records) & set(poses))
    if not names or len(names) != len(records):
        raise ValueError("RADIO query inventory and Cambridge poses differ")
    calibration = json.loads(args.calibration_manifest.read_text())
    calibration_rows = list(calibration["cameras"].values())
    first = calibration_rows[0]
    if any(row != first for row in calibration_rows[1:]):
        raise ValueError("fixed-calibration authority is not constant")
    model_id = int(first["model_id"])
    width, height = int(first["width"]), int(first["height"])
    params = np.asarray(first["params"], np.float64)
    if params.shape != (4,):
        raise ValueError("query calibration parameter shape differs")

    cameras = {}
    source_hashes = []
    args.output_pose_archives.mkdir(parents=True)
    for name in names:
        image_path = args.image_root / name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        cameras[name] = {
            "model_id": model_id, "width": width, "height": height,
            "params": params.tolist(),
        }
        destination = args.output_pose_archives / (name.replace("/", "__") + ".npz")
        np.savez(
            destination,
            pose_w2c=poses[name], camera_model_id=np.asarray(model_id, np.int32),
            camera_width=np.asarray(width, np.int32), camera_height=np.asarray(height, np.int32),
            camera_params=params,
        )
        source_hashes.append(file_sha256(destination))

    camera_payload = {
        "format": "per_query_colmap_calibration_only_v1",
        "query_count": len(names), "cameras": cameras,
        "production_contract": {
            "contains_camera_pose": False, "contains_sfm_points": False,
            "contains_sfm_tracks": False, "query_only": True,
        },
        "lineage": {
            "calibration_manifest_file_sha256": file_sha256(args.calibration_manifest),
            "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
            "pose_file_file_sha256": file_sha256(args.pose_file),
            "pose_values_materialized": False,
        },
    }
    camera_payload["content_sha256"] = canonical_json_sha256(camera_payload)
    args.output_camera_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_camera_manifest.write_text(json.dumps(camera_payload, indent=2, sort_keys=True) + "\n")

    arrays = {
        "names": np.asarray([name.replace("/", "__") + ".npz" for name in names]),
        "camera_model_id": np.full(len(names), model_id, np.int32),
        "camera_width": np.full(len(names), width, np.int32),
        "camera_height": np.full(len(names), height, np.int32),
        "camera_params": np.repeat(params[None], len(names), axis=0),
        "source_contributor_file_sha256": np.asarray(source_hashes),
    }
    metadata = {
        "artifact_type": "goal_maplet_query_camera_only_inventory_v1",
        "query_count": len(names), "pose_or_ground_truth_member_read": False,
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
        "query_count": len(names),
        "camera_manifest_file_sha256": file_sha256(args.output_camera_manifest),
        "camera_inventory_file_sha256": file_sha256(args.output_camera_inventory),
        "pose_archive_count": len(source_hashes),
    }, indent=2))


if __name__ == "__main__":
    main()
