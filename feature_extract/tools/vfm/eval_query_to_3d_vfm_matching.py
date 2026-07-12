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
    LandmarkQualityConfig,
    LandmarkMapIndex,
    QueryTo3DMatchingConfig,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_reprojection_errors,
    match_spatial_distribution_stats,
    match_query_tokens_to_landmarks,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    reprojection_error_stats,
    reprojection_precision,
    with_landmark_ambiguity_scores,
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


def _infer_camera_model_dir(query_pose_file: str, explicit: str) -> str:
    if explicit:
        return explicit
    if not query_pose_file:
        return ""
    scene_dir = Path(query_pose_file).resolve(strict=False).parent
    candidate = scene_dir / "sparse" / "0"
    if (candidate / "cameras.bin").exists():
        return str(candidate)
    return ""


def _load_camera_with_source(camera_model_dir: str, fallback: ColmapCamera) -> tuple[ColmapCamera, str]:
    if camera_model_dir and (Path(camera_model_dir) / "cameras.bin").exists():
        return _load_default_camera(camera_model_dir, fallback), str(Path(camera_model_dir) / "cameras.bin")
    return fallback, "fallback_default_camera"


def _load_track_stats(
    track_observations: Path,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    errors_by_track: dict[int, list[float]] = {}
    for obs in load_colmap_track_observations_jsonl(Path(track_observations)):
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        errors_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    reprojection_error_by_track = {
        int(track_id): float(np.mean(errors)) for track_id, errors in errors_by_track.items() if errors
    }
    return xyz_by_track, reprojection_error_by_track


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
    parser.add_argument("--min_similarity_margin", type=float, default=None)
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--max_landmark_variance", type=float, default=None)
    parser.add_argument("--max_landmark_reprojection_error", type=float, default=None)
    parser.add_argument("--max_landmark_ambiguity", type=float, default=None)
    parser.add_argument("--min_distance_to_boundary_px", type=float, default=None)
    parser.add_argument("--min_quality_weighted_similarity", type=float, default=None)
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--enable_landmark_quality", action="store_true")
    parser.add_argument("--quality_min_score", type=float, default=None)
    parser.add_argument("--quality_min_track_length", type=int, default=None)
    parser.add_argument("--quality_track_weight", type=float, default=1.0)
    parser.add_argument("--quality_variance_weight", type=float, default=1.0)
    parser.add_argument("--quality_reprojection_weight", type=float, default=1.0)
    parser.add_argument("--quality_idf_weight", type=float, default=0.0)
    parser.add_argument("--quality_ambiguity_weight", type=float, default=0.5)
    parser.add_argument("--quality_ambiguity_reference_size", type=int, default=4096)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--match_block_size", type=int, default=256)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--precision_reprojection_threshold_px", type=float, default=16.0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_matches_jsonl", default="")
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    bank = load_selected_track_bank_npz(Path(args.landmark_bank))
    xyz_by_track, reprojection_error_by_track = _load_track_stats(Path(args.track_observations))
    landmark_index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
    if args.max_landmark_ambiguity is not None or args.enable_landmark_quality:
        landmark_index = with_landmark_ambiguity_scores(
            landmark_index,
            reference_size=args.quality_ambiguity_reference_size,
            block_size=args.match_block_size,
        )
    visibility_index = None
    if args.visibility_index:
        visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index))
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    reference_submaps = _load_reference_submaps(args.candidate_bank, args.submap_top_n)
    gt_by_query = {}
    if args.query_pose_file:
        gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}

    config = QueryTo3DMatchingConfig(
        top_k=args.top_k,
        ratio_threshold=None if args.disable_ratio_test else args.ratio_threshold,
        min_similarity_margin=args.min_similarity_margin,
        min_similarity=args.min_similarity,
        mutual=bool(args.mutual),
        max_landmark_variance=args.max_landmark_variance,
        max_landmark_reprojection_error=args.max_landmark_reprojection_error,
        max_landmark_ambiguity=args.max_landmark_ambiguity,
        min_distance_to_boundary_px=args.min_distance_to_boundary_px,
        min_quality_weighted_similarity=args.min_quality_weighted_similarity,
        min_observation_count=args.min_observation_count,
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
        block_size=args.match_block_size,
        landmark_quality=LandmarkQualityConfig(
            enabled=bool(args.enable_landmark_quality),
            track_weight=args.quality_track_weight,
            variance_weight=args.quality_variance_weight,
            reprojection_weight=args.quality_reprojection_weight,
            idf_weight=args.quality_idf_weight,
            ambiguity_weight=args.quality_ambiguity_weight,
            ambiguity_reference_size=args.quality_ambiguity_reference_size,
            min_score=args.quality_min_score,
            min_track_length=args.quality_min_track_length,
        ),
    )

    rows = []
    match_rows = []
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
        _channels, token_height, token_width = query_feature.shape
        token_scale_x = 0.0 if token_width <= 1 else float(camera.width - 1) / float(token_width - 1)
        token_scale_y = 0.0 if token_height <= 1 else float(camera.height - 1) / float(token_height - 1)
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
        geometry_stats = {}
        if gt_pose is not None and matches:
            precision, false_match_rate = reprojection_precision(
                matches,
                gt_pose.pose_w2c,
                camera,
                threshold_px=args.precision_reprojection_threshold_px,
            )
            geometry_stats = reprojection_error_stats(
                matches,
                gt_pose.pose_w2c,
                camera,
                thresholds_px=(5.0, 10.0, 16.0, 32.0),
                pnp_inlier_mask=pnp.inlier_mask,
            )
            gt_errors = match_reprojection_errors(matches, gt_pose.pose_w2c, camera)
        else:
            gt_errors = np.zeros((len(matches),), dtype=np.float64)
        pnp_residual_stats = pnp_reprojection_residual_stats(
            matches,
            pnp.pose_w2c,
            camera,
            inlier_mask=pnp.inlier_mask,
        )
        all_spatial_stats = match_spatial_distribution_stats(
            matches, int(camera.width), int(camera.height), pose_w2c=pnp.pose_w2c
        )
        inlier_spatial_stats = match_spatial_distribution_stats(
            matches,
            int(camera.width),
            int(camera.height),
            mask=pnp.inlier_mask,
            pose_w2c=pnp.pose_w2c,
        )
        translation_error = None if pose_error is None else float(pose_error.translation_m)
        rotation_error = None if pose_error is None else float(pose_error.rotation_deg)
        for match_idx, match in enumerate(matches):
            match_rows.append(
                {
                    "query_id": query_id,
                    "match_index": int(match_idx),
                    "source": match.source,
                    "token_index": int(match.token_index),
                    "track_id": int(match.track_id),
                    "xy": [float(match.xy[0]), float(match.xy[1])],
                    "xyz": [float(value) for value in match.xyz.tolist()],
                    "similarity": float(match.similarity),
                    "ratio": float(match.ratio),
                    "similarity_margin": match.similarity_margin,
                    "distance_to_boundary_px": match.distance_to_boundary_px,
                    "landmark_variance": float(match.landmark_variance),
                    "landmark_reprojection_error": match.landmark_reprojection_error,
                    "landmark_quality": match.landmark_quality,
                    "landmark_ambiguity": match.landmark_ambiguity,
                    "quality_weighted_similarity": match.quality_weighted_similarity,
                    "track_length": match.observation_count,
                    "visibility_count": match.visibility_count,
                    "gt_reproj_error_px": None if gt_errors.shape[0] <= match_idx else float(gt_errors[match_idx]),
                    "gt_inlier_5px": bool(gt_errors.shape[0] > match_idx and gt_errors[match_idx] <= 5.0),
                    "gt_inlier_16px": bool(gt_errors.shape[0] > match_idx and gt_errors[match_idx] <= 16.0),
                    "gt_inlier_32px": bool(gt_errors.shape[0] > match_idx and gt_errors[match_idx] <= 32.0),
                    "pnp_inlier": bool(pnp.inlier_mask.shape[0] > match_idx and bool(pnp.inlier_mask[match_idx])),
                }
            )
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
            "coordinate_audit": {
                "query_feature_shape_chw": [int(v) for v in query_feature.shape],
                "camera_model_id": int(camera.model_id),
                "camera_width": int(camera.width),
                "camera_height": int(camera.height),
                "token_grid_scale_x_px": token_scale_x,
                "token_grid_scale_y_px": token_scale_y,
                "query_token_step": int(args.query_token_step),
            },
            "match_count": len(matches),
            "mean_similarity": _mean([match.similarity for match in matches]),
            "feature_precision_at_px": precision,
            "hard_false_match_rate": false_match_rate,
            "match_geometry": geometry_stats,
            "match_sources": {
                "source": "sparse_landmark",
                "mean_track_length": _mean(
                    [float(match.observation_count) for match in matches if match.observation_count is not None]
                ),
                "median_track_length": None
                if not [match.observation_count for match in matches if match.observation_count is not None]
                else float(
                    np.median(
                        [float(match.observation_count) for match in matches if match.observation_count is not None]
                    )
                ),
                "mean_landmark_variance": _mean([float(match.landmark_variance) for match in matches]),
                "mean_landmark_reprojection_error": _mean(
                    [
                        float(match.landmark_reprojection_error)
                        for match in matches
                        if match.landmark_reprojection_error is not None
                    ]
                ),
                "mean_landmark_quality": _mean(
                    [float(match.landmark_quality) for match in matches if match.landmark_quality is not None]
                ),
                "mean_landmark_ambiguity": _mean(
                    [float(match.landmark_ambiguity) for match in matches if match.landmark_ambiguity is not None]
                ),
                "mean_quality_weighted_similarity": _mean(
                    [
                        float(match.quality_weighted_similarity)
                        for match in matches
                        if match.quality_weighted_similarity is not None
                    ]
                ),
                "mean_similarity_margin": _mean(
                    [float(match.similarity_margin) for match in matches if match.similarity_margin is not None]
                ),
                "median_distance_to_boundary_px": None
                if not [match.distance_to_boundary_px for match in matches if match.distance_to_boundary_px is not None]
                else float(
                    np.median(
                        [
                            float(match.distance_to_boundary_px)
                            for match in matches
                            if match.distance_to_boundary_px is not None
                        ]
                    )
                ),
            },
            "all_match_spatial": all_spatial_stats,
            "pnp_inlier_spatial": inlier_spatial_stats,
            "pnp_reprojection": pnp_residual_stats,
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
    if args.output_matches_jsonl:
        matches_path = Path(args.output_matches_jsonl)
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        matches_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in match_rows) + ("\n" if match_rows else "")
        )

    labeled_rows = [row for row in rows if row["translation_error_m"] is not None]
    summary = {
        "stage": "query_to_3d_vfm_matching_baseline",
        "query_count": len(rows),
        "labeled_query_count": len(labeled_rows),
        "landmark_count": len(landmark_index),
        "matching_config": {
            "top_k": args.top_k,
            "ratio_threshold": None if args.disable_ratio_test else args.ratio_threshold,
            "min_similarity_margin": args.min_similarity_margin,
            "min_similarity": args.min_similarity,
            "mutual": bool(args.mutual),
            "max_landmark_variance": args.max_landmark_variance,
            "max_landmark_reprojection_error": args.max_landmark_reprojection_error,
            "max_landmark_ambiguity": args.max_landmark_ambiguity,
            "min_distance_to_boundary_px": args.min_distance_to_boundary_px,
            "min_quality_weighted_similarity": args.min_quality_weighted_similarity,
            "min_observation_count": args.min_observation_count,
            "query_token_step": args.query_token_step,
            "max_matches": args.max_matches,
            "landmark_quality": {
                "enabled": bool(args.enable_landmark_quality),
                "min_score": args.quality_min_score,
                "min_track_length": args.quality_min_track_length,
                "track_weight": args.quality_track_weight,
                "variance_weight": args.quality_variance_weight,
                "reprojection_weight": args.quality_reprojection_weight,
                "idf_weight": args.quality_idf_weight,
                "ambiguity_weight": args.quality_ambiguity_weight,
                "ambiguity_reference_size": args.quality_ambiguity_reference_size,
            },
        },
        "camera": {
            "source": camera_source,
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
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
        "mean_gt_precision_5px": _mean(
            [
                float(row["match_geometry"]["gt_precision_5px"])
                for row in rows
                if row["match_geometry"].get("gt_precision_5px") is not None
            ]
        ),
        "mean_gt_precision_16px": _mean(
            [
                float(row["match_geometry"]["gt_precision_16px"])
                for row in rows
                if row["match_geometry"].get("gt_precision_16px") is not None
            ]
        ),
        "mean_gt_precision_32px": _mean(
            [
                float(row["match_geometry"]["gt_precision_32px"])
                for row in rows
                if row["match_geometry"].get("gt_precision_32px") is not None
            ]
        ),
        "median_gt_reproj_median_px": None
        if not [row for row in rows if row["match_geometry"].get("gt_reproj_median_px") is not None]
        else float(
            np.median(
                [
                    float(row["match_geometry"]["gt_reproj_median_px"])
                    for row in rows
                    if row["match_geometry"].get("gt_reproj_median_px") is not None
                ]
            )
        ),
        "mean_pnp_inlier_gt_precision_5px": _mean(
            [
                float(row["match_geometry"]["pnp_inlier_gt_precision_5px"])
                for row in rows
                if row["match_geometry"].get("pnp_inlier_gt_precision_5px") is not None
            ]
        ),
        "mean_pnp_inlier_gt_precision_16px": _mean(
            [
                float(row["match_geometry"]["pnp_inlier_gt_precision_16px"])
                for row in rows
                if row["match_geometry"].get("pnp_inlier_gt_precision_16px") is not None
            ]
        ),
        "mean_pnp_inlier_gt_precision_32px": _mean(
            [
                float(row["match_geometry"]["pnp_inlier_gt_precision_32px"])
                for row in rows
                if row["match_geometry"].get("pnp_inlier_gt_precision_32px") is not None
            ]
        ),
        "mean_pnp_reproj_inlier_median_px": _mean(
            [
                float(row["pnp_reprojection"]["pnp_reproj_inlier_median_px"])
                for row in rows
                if row["pnp_reprojection"].get("pnp_reproj_inlier_median_px") is not None
            ]
        ),
        "mean_pnp_inlier_bbox_area_frac": _mean(
            [
                float(row["pnp_inlier_spatial"]["bbox_area_frac"])
                for row in rows
                if row["pnp_inlier_spatial"].get("bbox_area_frac") is not None
            ]
        ),
        "mean_pnp_inlier_grid_4x4_occupancy_frac": _mean(
            [
                float(row["pnp_inlier_spatial"]["grid_4x4_occupancy_frac"])
                for row in rows
                if row["pnp_inlier_spatial"].get("grid_4x4_occupancy_frac") is not None
            ]
        ),
        "mean_pnp_inlier_xy_pca_minor_major_ratio": _mean(
            [
                float(row["pnp_inlier_spatial"]["xy_pca_minor_major_ratio"])
                for row in rows
                if row["pnp_inlier_spatial"].get("xy_pca_minor_major_ratio") is not None
            ]
        ),
        "mean_pnp_inlier_xyz_planarity_ratio": _mean(
            [
                float(row["pnp_inlier_spatial"]["xyz_planarity_ratio"])
                for row in rows
                if row["pnp_inlier_spatial"].get("xyz_planarity_ratio") is not None
            ]
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
            "matches": args.output_matches_jsonl,
        },
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
