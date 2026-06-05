"""Stage F4: link pairwise VFM patch tracks into multi-view 3D landmarks."""

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
from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c, parse_cambridge_pose_file
from feature_extract.vfm.map_lifting import TrackObservation, aggregate_selected_tracks, save_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_aware_landmarks import (
    flatten_feature_map,
    link_vfm_patch_pair_rows,
    load_token_feature_map,
    triangulate_multiview_dlt,
)


def _load_pair_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record_type") == "vfm_patch_pair_track":
            rows.append(item)
    return rows


def _camera_depth(point_xyz: np.ndarray, pose_w2c: np.ndarray) -> float:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return float((pose[:3, :3] @ np.asarray(point_xyz, dtype=np.float64).reshape(3) + pose[:3, 3])[2])


def _max_triangulation_angle_deg(point_xyz: np.ndarray, poses_w2c: Sequence[np.ndarray]) -> float:
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    rays = []
    for pose_w2c in poses_w2c:
        center = camera_center_from_pose_w2c(np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4))
        ray = point - center
        norm = float(np.linalg.norm(ray))
        if norm <= 1e-12:
            continue
        rays.append(ray / norm)
    if len(rays) < 2:
        return 0.0
    max_angle = 0.0
    for idx in range(len(rays)):
        for jdx in range(idx + 1, len(rays)):
            cosine = float(np.clip(np.dot(rays[idx], rays[jdx]), -1.0, 1.0))
            max_angle = max(max_angle, float(np.degrees(np.arccos(cosine))))
    return max_angle


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build multi-view VFM-native landmarks from pairwise patch tracks")
    parser.add_argument("--patch_tracks_jsonl", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--min_observations", type=int, default=3)
    parser.add_argument("--max_observations", type=int, default=12)
    parser.add_argument("--min_pair_count", type=int, default=2)
    parser.add_argument("--max_reprojection_error_px", type=float, default=4.0)
    parser.add_argument("--min_triangulation_angle_deg", type=float, default=0.5)
    parser.add_argument("--track_id_offset", type=int, default=300_000_000)
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
    pair_rows = _load_pair_rows(Path(args.patch_tracks_jsonl))
    linked_tracks = link_vfm_patch_pair_rows(
        pair_rows,
        min_observations=int(args.min_observations),
        max_observations=None if int(args.max_observations) <= 0 else int(args.max_observations),
    )

    feature_cache: dict[str, np.ndarray] = {}
    feature_observations: list[TrackObservation] = []
    track_rows: list[dict[str, object]] = []
    accepted_errors: list[float] = []
    accepted_angles: list[float] = []
    accepted_lengths: list[int] = []
    reject_counts = {
        "low_pair_count": 0,
        "missing_pose_or_tokens": 0,
        "negative_depth": 0,
        "high_reprojection_error": 0,
        "low_triangulation_angle": 0,
    }

    for linked_idx, linked in enumerate(linked_tracks):
        if int(linked.pair_count) < int(args.min_pair_count):
            reject_counts["low_pair_count"] += 1
            continue
        if any(obs.image_id not in poses or obs.image_id not in records_by_image for obs in linked.observations):
            reject_counts["missing_pose_or_tokens"] += 1
            continue
        observations_xy = [np.asarray(obs.xy, dtype=np.float64).reshape(2) for obs in linked.observations]
        poses_w2c = [poses[obs.image_id] for obs in linked.observations]
        intrinsics = [intrinsic for _obs in linked.observations]
        try:
            xyz, errors = triangulate_multiview_dlt(observations_xy, poses_w2c, intrinsics)
        except np.linalg.LinAlgError:
            reject_counts["high_reprojection_error"] += 1
            continue
        if any(_camera_depth(xyz, pose_w2c) <= 1e-6 for pose_w2c in poses_w2c):
            reject_counts["negative_depth"] += 1
            continue
        max_error = float(np.max(errors))
        if max_error > float(args.max_reprojection_error_px):
            reject_counts["high_reprojection_error"] += 1
            continue
        angle = _max_triangulation_angle_deg(xyz, poses_w2c)
        if angle < float(args.min_triangulation_angle_deg):
            reject_counts["low_triangulation_angle"] += 1
            continue
        output_track_id = int(args.track_id_offset) + int(linked_idx)
        for obs, error in zip(linked.observations, errors):
            if obs.image_id not in feature_cache:
                feature_cache[obs.image_id] = flatten_feature_map(
                    load_token_feature_map(records_by_image[obs.image_id].token_path, args.layer_name)
                )
            features = feature_cache[obs.image_id]
            if int(obs.token_index) >= features.shape[0]:
                reject_counts["missing_pose_or_tokens"] += 1
                continue
            utility = 1.0 / max(float(error), 1e-3)
            feature_observations.append(
                TrackObservation(
                    track_id=output_track_id,
                    image_id=obs.image_id,
                    feature=features[int(obs.token_index)],
                    visible=True,
                    geometry_valid=True,
                    utility=utility,
                )
            )
            track_rows.append(
                {
                    "track_id": int(output_track_id),
                    "linked_track_index": int(linked_idx),
                    "image_id": obs.image_id,
                    "point2d_idx": int(obs.token_index),
                    "xy": [float(obs.xy[0]), float(obs.xy[1])],
                    "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "track_length": int(len(linked.observations)),
                    "pair_count": int(linked.pair_count),
                    "reprojection_error": float(error),
                    "camera_id": int(camera.camera_id),
                    "image_width": int(camera.width),
                    "image_height": int(camera.height),
                    "triangulation_angle_deg": float(angle),
                    "mean_pair_similarity": float(linked.mean_similarity),
                    "stage_f_source": "vfm_multiview",
                }
            )
        accepted_errors.append(max_error)
        accepted_angles.append(angle)
        accepted_lengths.append(int(len(linked.observations)))

    bank = aggregate_selected_tracks(feature_observations, min_observations=int(args.min_observations))
    output_bank = Path(args.output_bank)
    save_selected_track_bank_npz(bank, output_bank)
    output_obs = Path(args.output_track_observations)
    output_obs.parent.mkdir(parents=True, exist_ok=True)
    with output_obs.open("w") as handle:
        for row in track_rows:
            if int(row["track_id"]) in bank.tracks:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "stage": "stage_f4_vfm_multiview_landmark_bank",
        "camera_source": camera_source,
        "config": {
            "min_observations": int(args.min_observations),
            "max_observations": int(args.max_observations),
            "min_pair_count": int(args.min_pair_count),
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
        "pair_track_count": int(len(pair_rows)),
        "linked_track_count": int(len(linked_tracks)),
        "accepted_track_count": int(len(bank.tracks)),
        "feature_dim": int(bank.feature_dim),
        "reject_counts": reject_counts,
        "mean_max_reprojection_error_px": None if not accepted_errors else float(np.mean(accepted_errors)),
        "median_max_reprojection_error_px": None if not accepted_errors else float(np.median(accepted_errors)),
        "mean_triangulation_angle_deg": None if not accepted_angles else float(np.mean(accepted_angles)),
        "median_triangulation_angle_deg": None if not accepted_angles else float(np.median(accepted_angles)),
        "mean_track_length": None if not accepted_lengths else float(np.mean(accepted_lengths)),
        "median_track_length": None if not accepted_lengths else float(np.median(accepted_lengths)),
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
