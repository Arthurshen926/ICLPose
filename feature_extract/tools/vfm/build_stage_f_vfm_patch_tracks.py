"""Stage F1: build smoke-test VFM patch tracks between posed reference images."""

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
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion, token_grid_xy
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_aware_landmarks import (
    candidate_reference_image_order,
    flatten_feature_map,
    fundamental_from_w2c_poses,
    load_token_feature_map,
    observation_records_from_matches,
    reciprocal_token_matches,
    sampson_epipolar_errors_px,
    select_distinctive_tokens,
    token_distinctiveness_scores,
)


def _pose_neighbor_pairs(image_ids: list[str], centers: dict[str, np.ndarray], neighbors_per_image: int) -> list[tuple[str, str, float]]:
    pairs: dict[tuple[str, str], float] = {}
    for image_id in image_ids:
        center = centers[image_id]
        distances = []
        for other_id in image_ids:
            if other_id == image_id:
                continue
            distance = float(np.linalg.norm(center - centers[other_id]))
            distances.append((distance, other_id))
        for distance, other_id in sorted(distances)[: int(neighbors_per_image)]:
            key = tuple(sorted((image_id, other_id)))
            pairs[key] = min(float(distance), pairs.get(key, float("inf")))
    return [(a, b, distance) for (a, b), distance in sorted(pairs.items(), key=lambda item: (item[1], item[0][0], item[0][1]))]


def _load_feature_cache(records_by_image: dict[str, object], image_ids: set[str], layer_name: str) -> dict[str, np.ndarray]:
    cache = {}
    for image_id in image_ids:
        record = records_by_image[image_id]
        cache[image_id] = load_token_feature_map(record.token_path, layer_name)
    return cache


def _load_candidate_rows(candidate_bank: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(candidate_bank).read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage F1 VFM patch track smoke builder")
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--max_images", type=int, default=40)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--candidate_top_n", type=int, default=0)
    parser.add_argument("--candidate_max_queries", type=int, default=0)
    parser.add_argument("--max_candidate_images", type=int, default=0)
    parser.add_argument("--neighbors_per_image", type=int, default=2)
    parser.add_argument("--tokens_per_image", type=int, default=384)
    parser.add_argument("--min_distance_to_boundary_px", type=float, default=16.0)
    parser.add_argument("--min_similarity", type=float, default=0.35)
    parser.add_argument("--max_epipolar_error_px", type=float, default=0.0)
    parser.add_argument("--distinctiveness_block_size", type=int, default=1024)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    manifest.validate(verify_checksums=False)
    records_by_image = {record.image_id: record for record in manifest.records}
    all_pose_records = [record for record in parse_cambridge_pose_file(Path(args.reference_pose_file)) if record.image_id in records_by_image]
    pose_record_by_image = {record.image_id: record for record in all_pose_records}
    candidate_reference_count = 0
    missing_candidate_reference_count = 0
    if args.candidate_bank:
        if int(args.candidate_top_n) <= 0:
            raise ValueError("candidate_top_n must be positive when candidate_bank is set")
        candidate_rows = _load_candidate_rows(args.candidate_bank)
        candidate_references = candidate_reference_image_order(
            candidate_rows,
            top_n=int(args.candidate_top_n),
            max_queries=int(args.candidate_max_queries),
        )
        candidate_reference_count = int(len(candidate_references))
        limit = int(args.max_candidate_images) if int(args.max_candidate_images) > 0 else int(args.max_images)
        if limit > 0:
            candidate_references = candidate_references[:limit]
        pose_records = [pose_record_by_image[image_id] for image_id in candidate_references if image_id in pose_record_by_image]
        missing_candidate_reference_count = int(len(candidate_references) - len(pose_records))
    else:
        pose_records = all_pose_records
        if args.max_images:
            pose_records = pose_records[: int(args.max_images)]
    pose_by_image = {record.image_id: record.pose_w2c for record in pose_records}
    image_ids = [record.image_id for record in pose_records]
    centers = {record.image_id: camera_center_from_pose_w2c(record.pose_w2c) for record in pose_records}
    pairs = _pose_neighbor_pairs(image_ids, centers, int(args.neighbors_per_image))
    if args.max_pairs:
        pairs = pairs[: int(args.max_pairs)]
    needed_images = {image_id for pair in pairs for image_id in pair[:2]}
    feature_cache = _load_feature_cache(records_by_image, needed_images, args.layer_name)
    camera_model_dir = _infer_camera_model_dir(args.reference_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    intrinsic, _distortion = camera_matrix_and_distortion(camera)

    selected_tokens: dict[str, np.ndarray] = {}
    selected_features: dict[str, np.ndarray] = {}
    token_xy: dict[str, np.ndarray] = {}
    saliency_stats: dict[str, dict[str, float | int]] = {}
    for image_id, feature_map in feature_cache.items():
        _channels, token_height, token_width = feature_map.shape
        features = flatten_feature_map(feature_map)
        xy = token_grid_xy(
            token_width,
            token_height,
            int(camera.width),
            int(camera.height),
        )
        saliency = token_distinctiveness_scores(features, block_size=int(args.distinctiveness_block_size))
        selected = select_distinctive_tokens(
            saliency,
            xy,
            image_width=int(camera.width),
            image_height=int(camera.height),
            top_count=int(args.tokens_per_image),
            min_distance_to_boundary_px=float(args.min_distance_to_boundary_px),
        )
        selected_tokens[image_id] = selected
        selected_features[image_id] = features[selected]
        token_xy[image_id] = xy
        saliency_stats[image_id] = {
            "selected_count": int(selected.size),
            "mean_selected_saliency": 0.0 if selected.size == 0 else float(np.mean(saliency[selected])),
            "mean_all_saliency": float(np.mean(saliency)) if saliency.size else 0.0,
            "token_width": int(token_width),
            "token_height": int(token_height),
        }

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    total_matches = 0
    similarities: list[float] = []
    with output_jsonl.open("w") as handle:
        for pair_index, (source_id, target_id, center_distance_m) in enumerate(pairs):
            matches = reciprocal_token_matches(
                selected_features[source_id],
                selected_features[target_id],
                source_token_indices=selected_tokens[source_id],
                target_token_indices=selected_tokens[target_id],
                min_similarity=float(args.min_similarity),
            )
            epipolar_errors = np.zeros((len(matches),), dtype=np.float64)
            if matches:
                source_xy = np.asarray([token_xy[source_id][int(match.source_token_index)] for match in matches], dtype=np.float64)
                target_xy = np.asarray([token_xy[target_id][int(match.target_token_index)] for match in matches], dtype=np.float64)
                fundamental = fundamental_from_w2c_poses(
                    pose_by_image[source_id],
                    pose_by_image[target_id],
                    intrinsic,
                    intrinsic,
                )
                epipolar_errors = sampson_epipolar_errors_px(source_xy, target_xy, fundamental)
                if float(args.max_epipolar_error_px) > 0.0:
                    keep = epipolar_errors <= float(args.max_epipolar_error_px)
                    matches = [match for match, keep_match in zip(matches, keep) if bool(keep_match)]
                    epipolar_errors = epipolar_errors[keep]
            rows = observation_records_from_matches(
                matches,
                source_id,
                target_id,
                token_xy[source_id],
                token_xy[target_id],
            )
            for match_index, row in enumerate(rows):
                payload = {
                    "record_type": "vfm_patch_pair_track",
                    "track_id": int(pair_index * 1_000_000 + match_index),
                    "pair_index": int(pair_index),
                    "center_distance_m": float(center_distance_m),
                    "epipolar_error_px": float(epipolar_errors[match_index]) if match_index < len(epipolar_errors) else None,
                    **row,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                similarities.append(float(payload["similarity"]))
            total_matches += len(rows)
            pair_rows.append(
                {
                    "source_image_id": source_id,
                    "target_image_id": target_id,
                    "center_distance_m": float(center_distance_m),
                    "match_count": int(len(rows)),
                    "mean_similarity": 0.0 if not rows else float(np.mean([row["similarity"] for row in rows])),
                    "median_epipolar_error_px": None if len(epipolar_errors) == 0 else float(np.median(epipolar_errors)),
                }
            )

    summary = {
        "stage": "stage_f1_vfm_patch_track_smoke",
        "config": {
            "max_images": int(args.max_images),
            "candidate_bank": str(args.candidate_bank),
            "candidate_top_n": int(args.candidate_top_n),
            "candidate_max_queries": int(args.candidate_max_queries),
            "max_candidate_images": int(args.max_candidate_images),
            "neighbors_per_image": int(args.neighbors_per_image),
            "tokens_per_image": int(args.tokens_per_image),
            "min_distance_to_boundary_px": float(args.min_distance_to_boundary_px),
            "min_similarity": float(args.min_similarity),
            "max_epipolar_error_px": float(args.max_epipolar_error_px),
            "image_width": int(camera.width),
            "image_height": int(camera.height),
        },
        "inputs": {
            "token_manifest": str(args.token_manifest),
            "reference_pose_file": str(args.reference_pose_file),
        },
        "outputs": {
            "tracks_jsonl": str(output_jsonl),
        },
        "image_count": int(len(image_ids)),
        "candidate_reference_count_before_limit": int(candidate_reference_count),
        "missing_candidate_reference_count": int(missing_candidate_reference_count),
        "camera_source": camera_source,
        "pair_count": int(len(pairs)),
        "match_count": int(total_matches),
        "mean_matches_per_pair": 0.0 if not pairs else float(total_matches / max(len(pairs), 1)),
        "mean_similarity": 0.0 if not similarities else float(np.mean(similarities)),
        "median_similarity": None if not similarities else float(np.median(similarities)),
        "image_saliency_stats": saliency_stats,
        "pairs": pair_rows,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
