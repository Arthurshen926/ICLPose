"""Evaluate raw query-token to 3D VFM landmark matching with PnP-RANSAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary
from feature_extract.vfm.landmark_visibility import (
    LandmarkVisibilityIndex,
    count_projected_landmarks,
    filter_landmarks_by_visibility,
)
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    QueryTo3DMatchingConfig,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_query_tokens_to_landmarks,
    pnp_pose_error,
    reprojection_precision,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _parse_default_camera(text: str) -> ColmapCamera:
    values = [float(item) for item in text.split(",")]
    if len(values) < 7:
        raise ValueError("--default_camera must be 'model_id,width,height,param0,param1,...'")
    model_id = int(values[0])
    width = int(values[1])
    height = int(values[2])
    params = tuple(float(item) for item in values[3:])
    return ColmapCamera(camera_id=-1, model_id=model_id, width=width, height=height, params=params)


def _load_default_camera(camera_model_dir: str, fallback: ColmapCamera) -> ColmapCamera:
    if not camera_model_dir:
        return fallback
    cameras = read_colmap_cameras_binary(Path(camera_model_dir) / "cameras.bin")
    if not cameras:
        return fallback
    ordered = sorted(cameras.values(), key=lambda camera: camera.camera_id)
    return ordered[len(ordered) // 2]


def _load_xyz_by_track(track_observations: Path) -> dict[int, np.ndarray]:
    xyz_by_track: dict[int, np.ndarray] = {}
    for obs in load_colmap_track_observations_jsonl(Path(track_observations)):
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
    return xyz_by_track


def _load_reference_submaps(candidate_bank: str, submap_top_n: int) -> dict[str, list[str]]:
    if not candidate_bank:
        return {}
    if submap_top_n <= 0:
        raise ValueError("submap_top_n must be positive")
    by_query: dict[str, list[tuple[int, int, str]]] = {}
    order = 0
    for line in Path(candidate_bank).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record_type") == "header":
            continue
        if item.get("record_type", "candidate") != "candidate":
            continue
        query_id = item.get("query_id")
        reference_image = item.get("reference_image")
        if query_id is None or reference_image is None:
            continue
        metadata = dict(item.get("metadata") or {})
        rank = int(metadata.get("retrieval_rank", len(by_query.get(str(query_id), [])) + 1))
        by_query.setdefault(str(query_id), []).append((rank, order, str(reference_image)))
        order += 1
    result: dict[str, list[str]] = {}
    for query_id, rows in by_query.items():
        refs = []
        for _rank, _order, reference in sorted(rows)[:submap_top_n]:
            if reference not in refs:
                refs.append(reference)
        result[query_id] = refs
    return result


def _limit_submap(index: LandmarkMapIndex, max_landmarks: int) -> LandmarkMapIndex:
    if max_landmarks <= 0 or len(index) <= max_landmarks:
        return index
    order = np.lexsort((-index.observation_counts, index.mean_variances))
    return index.subset(order[:max_landmarks])


def _load_query_feature(path: Path, layer_name: str) -> np.ndarray:
    with np.load(path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {path}")
        feature_map = np.asarray(data[layer_name], dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("query token feature map must have shape (C, H, W)")
    return feature_map


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate query VFM token to 3D landmark VFM matching")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--query_pose_file", default="")
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--submap_mode", default="reference_visibility", choices=("reference_visibility", "none"))
    parser.add_argument("--submap_top_n", type=int, default=5)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--query_token_step", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--max_landmark_variance", type=float, default=None)
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--match_block_size", type=int, default=256)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--precision_reprojection_threshold_px", type=float, default=16.0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    bank = load_selected_track_bank_npz(Path(args.landmark_bank))
    landmark_index = LandmarkMapIndex.from_track_bank(bank, _load_xyz_by_track(Path(args.track_observations)))
    visibility_index = None
    if args.visibility_index:
        visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index))
    camera = _load_default_camera(args.camera_model_dir, _parse_default_camera(args.default_camera))
    reference_submaps = _load_reference_submaps(args.candidate_bank, args.submap_top_n)
    gt_by_query = {}
    if args.query_pose_file:
        gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}

    config = QueryTo3DMatchingConfig(
        top_k=args.top_k,
        ratio_threshold=None if args.disable_ratio_test else args.ratio_threshold,
        min_similarity=args.min_similarity,
        mutual=bool(args.mutual),
        max_landmark_variance=args.max_landmark_variance,
        min_observation_count=args.min_observation_count,
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
        block_size=args.match_block_size,
    )

    rows = []
    records = list(manifest.records)
    if args.max_queries > 0:
        records = records[: args.max_queries]
    for record in records:
        query_id = record.image_id
        submap = landmark_index
        references = reference_submaps.get(query_id, [])
        visibility_gate = {
            "full_visible_tracks": None,
            "bank_visible_tracks": None,
            "bank_visibility_coverage": None,
        }
        if args.submap_mode == "reference_visibility":
            if visibility_index is None:
                submap = filter_landmarks_by_reference_images(landmark_index, references)
                visibility_gate = {
                    "full_visible_tracks": len(submap),
                    "bank_visible_tracks": len(submap),
                    "bank_visibility_coverage": 1.0 if len(submap) else 0.0,
                }
            else:
                submap, visibility_gate = filter_landmarks_by_visibility(landmark_index, visibility_index, references)
        pre_limit_submap_count = len(submap)
        submap = _limit_submap(submap, int(args.max_submap_landmarks))
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        matches = match_query_tokens_to_landmarks(
            query_feature,
            submap,
            config,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=args.pnp_reprojection_error_px,
            iterations=args.pnp_iterations,
        )
        gt_pose = gt_by_query.get(query_id)
        pose_error = pnp_pose_error(pnp.pose_w2c, gt_pose.pose_w2c) if gt_pose is not None else None
        projected_landmarks = (
            count_projected_landmarks(submap, gt_pose.pose_w2c, camera) if gt_pose is not None else None
        )
        precision = None
        false_match_rate = None
        if gt_pose is not None and matches:
            precision, false_match_rate = reprojection_precision(
                matches,
                gt_pose.pose_w2c,
                camera,
                threshold_px=args.precision_reprojection_threshold_px,
            )
        translation_error = None if pose_error is None else float(pose_error.translation_m)
        rotation_error = None if pose_error is None else float(pose_error.rotation_deg)
        row = {
            "query_id": query_id,
            "submap_mode": args.submap_mode,
            "submap_reference_count": len(references),
            "full_visible_tracks": visibility_gate["full_visible_tracks"],
            "bank_visible_tracks": visibility_gate["bank_visible_tracks"],
            "bank_visibility_coverage": visibility_gate["bank_visibility_coverage"],
            "pre_limit_submap_landmark_count": pre_limit_submap_count,
            "projected_landmarks": projected_landmarks,
            "submap_landmark_count": len(submap),
            "match_count": len(matches),
            "mean_similarity": _mean([match.similarity for match in matches]),
            "feature_precision_at_px": precision,
            "hard_false_match_rate": false_match_rate,
            "pnp_success": bool(pnp.success),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": float(pnp.inlier_ratio),
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "success_10cm_5deg": bool(
                pnp.success
                and translation_error is not None
                and translation_error <= 0.10
                and rotation_error is not None
                and rotation_error <= 5.0
            ),
            "success_25cm_10deg": bool(
                pnp.success
                and translation_error is not None
                and translation_error <= 0.25
                and rotation_error is not None
                and rotation_error <= 10.0
            ),
        }
        rows.append(row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))

    labeled_rows = [row for row in rows if row["translation_error_m"] is not None]
    summary = {
        "stage": "query_to_3d_vfm_matching_baseline",
        "query_count": len(rows),
        "labeled_query_count": len(labeled_rows),
        "landmark_count": len(landmark_index),
        "matching_config": {
            "top_k": args.top_k,
            "ratio_threshold": None if args.disable_ratio_test else args.ratio_threshold,
            "min_similarity": args.min_similarity,
            "mutual": bool(args.mutual),
            "max_landmark_variance": args.max_landmark_variance,
            "min_observation_count": args.min_observation_count,
            "query_token_step": args.query_token_step,
            "max_matches": args.max_matches,
        },
        "submap": {
            "mode": args.submap_mode,
            "top_n": args.submap_top_n,
            "max_landmarks": args.max_submap_landmarks,
            "mean_landmark_count": _mean([float(row["submap_landmark_count"]) for row in rows]),
            "mean_full_visible_tracks": _mean(
                [float(row["full_visible_tracks"]) for row in rows if row["full_visible_tracks"] is not None]
            ),
            "mean_bank_visible_tracks": _mean(
                [float(row["bank_visible_tracks"]) for row in rows if row["bank_visible_tracks"] is not None]
            ),
            "mean_bank_visibility_coverage": _mean(
                [float(row["bank_visibility_coverage"]) for row in rows if row["bank_visibility_coverage"] is not None]
            ),
            "mean_projected_landmarks": _mean(
                [float(row["projected_landmarks"]) for row in rows if row["projected_landmarks"] is not None]
            ),
        },
        "mean_match_count": _mean([float(row["match_count"]) for row in rows]),
        "mean_feature_precision": _mean(
            [float(row["feature_precision_at_px"]) for row in rows if row["feature_precision_at_px"] is not None]
        ),
        "mean_hard_false_match_rate": _mean(
            [float(row["hard_false_match_rate"]) for row in rows if row["hard_false_match_rate"] is not None]
        ),
        "pnp_success_rate": _mean([1.0 if row["pnp_success"] else 0.0 for row in rows]),
        "success_10cm_5deg": _mean([1.0 if row["success_10cm_5deg"] else 0.0 for row in labeled_rows]),
        "success_25cm_10deg": _mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in labeled_rows]),
        "median_translation_error_m": None
        if not labeled_rows
        else float(np.median([float(row["translation_error_m"]) for row in labeled_rows])),
        "median_rotation_error_deg": None
        if not labeled_rows
        else float(np.median([float(row["rotation_error_deg"]) for row in labeled_rows])),
        "inputs": {
            "query_manifest": args.query_manifest,
            "landmark_bank": args.landmark_bank,
            "track_observations": args.track_observations,
            "visibility_index": args.visibility_index,
            "query_pose_file": args.query_pose_file,
            "candidate_bank": args.candidate_bank,
        },
        "outputs": {
            "rows": str(output_jsonl),
        },
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
