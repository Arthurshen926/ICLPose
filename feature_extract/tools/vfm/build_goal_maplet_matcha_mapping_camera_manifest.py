"""Build a pose-audited mapping-camera manifest from a MAtCha 2DGS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matcha_cameras", type=Path, required=True)
    parser.add_argument("--mapping_pose_file", type=Path, required=True)
    parser.add_argument("--mapping_radio_manifest", type=Path, required=True)
    parser.add_argument("--canvas_width", type=int, default=1024)
    parser.add_argument("--canvas_height", type=int, default=576)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output_image_ids", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_image_ids.exists():
        raise FileExistsError("refusing to overwrite mapping camera authority")

    camera_rows = json.loads(args.matcha_cameras.read_text())
    radio_payload = json.loads(args.mapping_radio_manifest.read_text())
    radio_ids = {str(row["image_id"]) for row in radio_payload["records"]}
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(args.mapping_pose_file)
    }
    cameras: dict[str, dict[str, object]] = {}
    selected_ids = []
    center_errors, rotation_errors = [], []
    for row in camera_rows:
        image_id = str(row["img_name"]).replace("__", "/") + ".png"
        if image_id not in radio_ids or image_id not in pose_by_image:
            raise ValueError("MAtCha camera is absent from mapping RADIO/pose authority")
        width, height = int(row["width"]), int(row["height"])
        output_width, output_height = int(args.canvas_width), int(args.canvas_height)
        fx = float(row["fx"]) * output_width / width
        fy = float(row["fy"]) * output_height / height
        pose = np.asarray(pose_by_image[image_id], np.float64)
        center = -pose[:3, :3].T @ pose[:3, 3]
        center_errors.append(float(np.linalg.norm(center - np.asarray(row["position"]))))
        camera_c2w = np.asarray(row["rotation"], np.float64)
        rotation_errors.append(float(
            Rotation.from_matrix(camera_c2w @ pose[:3, :3]).magnitude() * 180.0 / np.pi
        ))
        cameras[image_id] = {
            "model_id": 1,
            "width": output_width,
            "height": output_height,
            "params": [fx, fy, output_width / 2.0, output_height / 2.0],
        }
        selected_ids.append(image_id)
    if len(cameras) == 0 or len(cameras) != len(camera_rows):
        raise ValueError("MAtCha mapping camera inventory is empty or duplicated")
    if max(center_errors) > 1e-4 or max(rotation_errors) > 1e-3:
        raise ValueError("MAtCha cameras do not replay Cambridge mapping poses")

    payload = {
        "format": "per_query_colmap_calibration_only_v1",
        "query_count": len(cameras),
        "cameras": dict(sorted(cameras.items())),
        "intrinsic_audit": {
            "source": "MAtCha_2DGS_training_cameras_rescaled_to_RADIO_canvas",
            "maximum_mapping_center_replay_error_m": max(center_errors),
            "maximum_mapping_rotation_replay_error_deg": max(rotation_errors),
            "principal_point_convention": "centered_projection_cx=W/2_cy=H/2",
        },
        "production_contract": {
            "contains_camera_pose": False,
            "contains_sfm_points": False,
            "contains_sfm_tracks": False,
            "mapping_only": True,
        },
        "lineage": {
            "matcha_cameras_file_sha256": file_sha256(args.matcha_cameras),
            "mapping_pose_file_sha256": file_sha256(args.mapping_pose_file),
            "mapping_radio_manifest_file_sha256": file_sha256(args.mapping_radio_manifest),
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_image_ids.write_text("\n".join(selected_ids) + "\n")
    print(json.dumps({
        "camera_count": len(cameras),
        "maximum_center_error_m": max(center_errors),
        "maximum_rotation_error_deg": max(rotation_errors),
        "content_sha256": payload["content_sha256"],
        "output_file_sha256": file_sha256(args.output),
        "image_ids_file_sha256": file_sha256(args.output_image_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
