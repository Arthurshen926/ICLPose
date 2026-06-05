"""Stage F2: triangulate VFM patch tracks into a provisional sparse landmark bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import TrackObservation, aggregate_selected_tracks, save_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_aware_landmarks import (
    flatten_feature_map,
    load_token_feature_map,
    triangulate_two_view_dlt,
    triangulation_angles_deg,
)


def _camera_depths(points_xyz: np.ndarray, pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return (pose[:3, :3] @ points_xyz.T + pose[:3, 3:4]).T[:, 2]


def _load_patch_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record_type") != "vfm_patch_pair_track":
            continue
        rows.append(item)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage F2 VFM-native landmark bank builder")
    parser.add_argument("--patch_tracks_jsonl", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--max_reprojection_error_px", type=float, default=4.0)
    parser.add_argument("--min_triangulation_angle_deg", type=float, default=0.25)
    parser.add_argument("--track_id_offset", type=int, default=200_000_000)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--output_track_observations", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    manifest.validate(verify_checksums=False)
    records_by_image = {record.image_id: record for record in manifest.records}
    poses = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.reference_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    intrinsic, _distortion = camera_matrix_and_distortion(camera)
    patch_rows = _load_patch_rows(Path(args.patch_tracks_jsonl))
    feature_cache: dict[str, np.ndarray] = {}

    feature_observations: list[TrackObservation] = []
    track_observation_rows: list[dict[str, object]] = []
    accepted_reprojection_errors: list[float] = []
    accepted_angles: list[float] = []
    reject_counts = {
        "missing_pose_or_tokens": 0,
        "negative_depth": 0,
        "high_reprojection_error": 0,
        "low_triangulation_angle": 0,
    }
    for item in patch_rows:
        source_id = str(item["source_image_id"])
        target_id = str(item["target_image_id"])
        if source_id not in poses or target_id not in poses or source_id not in records_by_image or target_id not in records_by_image:
            reject_counts["missing_pose_or_tokens"] += 1
            continue
        source_xy = np.asarray([item["source_xy"][0], item["source_xy"][1]], dtype=np.float64).reshape(1, 2)
        target_xy = np.asarray([item["target_xy"][0], item["target_xy"][1]], dtype=np.float64).reshape(1, 2)
        source_pose = poses[source_id]
        target_pose = poses[target_id]
        xyz, source_error, target_error = triangulate_two_view_dlt(
            source_xy,
            target_xy,
            source_pose,
            target_pose,
            intrinsic,
            intrinsic,
        )
        if float(_camera_depths(xyz, source_pose)[0]) <= 1e-6 or float(_camera_depths(xyz, target_pose)[0]) <= 1e-6:
            reject_counts["negative_depth"] += 1
            continue
        max_error = max(float(source_error[0]), float(target_error[0]))
        if max_error > float(args.max_reprojection_error_px):
            reject_counts["high_reprojection_error"] += 1
            continue
        angle = float(triangulation_angles_deg(xyz, source_pose, target_pose)[0])
        if angle < float(args.min_triangulation_angle_deg):
            reject_counts["low_triangulation_angle"] += 1
            continue
        source_token_index = int(item["source_token_index"])
        target_token_index = int(item["target_token_index"])
        if source_id not in feature_cache:
            feature_cache[source_id] = flatten_feature_map(load_token_feature_map(records_by_image[source_id].token_path, args.layer_name))
        if target_id not in feature_cache:
            feature_cache[target_id] = flatten_feature_map(load_token_feature_map(records_by_image[target_id].token_path, args.layer_name))
        if source_token_index >= feature_cache[source_id].shape[0] or target_token_index >= feature_cache[target_id].shape[0]:
            reject_counts["missing_pose_or_tokens"] += 1
            continue
        output_track_id = int(args.track_id_offset) + int(item["track_id"])
        for image_id, token_index, xy, error in (
            (source_id, source_token_index, source_xy[0], float(source_error[0])),
            (target_id, target_token_index, target_xy[0], float(target_error[0])),
        ):
            feature = feature_cache[image_id][int(token_index)]
            feature_observations.append(
                TrackObservation(
                    track_id=output_track_id,
                    image_id=image_id,
                    feature=feature,
                    visible=True,
                    geometry_valid=True,
                    utility=1.0 / max(error, 1e-3),
                )
            )
            track_observation_rows.append(
                {
                    "track_id": int(output_track_id),
                    "source_pair_track_id": int(item["track_id"]),
                    "image_id": image_id,
                    "point2d_idx": int(token_index),
                    "xy": [float(xy[0]), float(xy[1])],
                    "xyz": [float(xyz[0, 0]), float(xyz[0, 1]), float(xyz[0, 2])],
                    "track_length": 2,
                    "reprojection_error": float(error),
                    "camera_id": int(camera.camera_id),
                    "image_width": int(camera.width),
                    "image_height": int(camera.height),
                    "triangulation_angle_deg": float(angle),
                    "pair_similarity": float(item.get("similarity", 0.0)),
                    "pair_epipolar_error_px": item.get("epipolar_error_px"),
                }
            )
        accepted_reprojection_errors.append(max_error)
        accepted_angles.append(angle)

    bank = aggregate_selected_tracks(feature_observations, min_observations=2)
    output_bank = Path(args.output_bank)
    save_selected_track_bank_npz(bank, output_bank)
    output_obs = Path(args.output_track_observations)
    output_obs.parent.mkdir(parents=True, exist_ok=True)
    with output_obs.open("w") as handle:
        for row in track_observation_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "stage": "stage_f2_vfm_native_landmark_bank",
        "camera_source": camera_source,
        "config": {
            "max_reprojection_error_px": float(args.max_reprojection_error_px),
            "min_triangulation_angle_deg": float(args.min_triangulation_angle_deg),
            "track_id_offset": int(args.track_id_offset),
        },
        "inputs": {
            "patch_tracks_jsonl": str(args.patch_tracks_jsonl),
            "token_manifest": str(args.token_manifest),
            "reference_pose_file": str(args.reference_pose_file),
        },
        "outputs": {
            "bank": str(output_bank),
            "track_observations": str(output_obs),
        },
        "input_pair_track_count": int(len(patch_rows)),
        "accepted_track_count": int(len(bank.tracks)),
        "feature_dim": int(bank.feature_dim),
        "reject_counts": reject_counts,
        "mean_max_reprojection_error_px": None
        if not accepted_reprojection_errors
        else float(np.mean(accepted_reprojection_errors)),
        "median_max_reprojection_error_px": None
        if not accepted_reprojection_errors
        else float(np.median(accepted_reprojection_errors)),
        "mean_triangulation_angle_deg": None if not accepted_angles else float(np.mean(accepted_angles)),
        "median_triangulation_angle_deg": None if not accepted_angles else float(np.median(accepted_angles)),
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
