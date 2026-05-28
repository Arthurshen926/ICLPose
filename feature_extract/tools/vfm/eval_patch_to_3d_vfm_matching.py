"""Evaluate patch-level query-token to sparse 3D VFM landmark-set matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _limit_submap,
    _load_camera_with_source,
    _load_query_feature,
    _load_reference_submaps,
    _load_track_stats,
    _mean,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, count_projected_landmarks, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    PatchTo3DMatchingConfig,
    build_patch_positive_sets,
    evaluate_patch_matches,
    match_query_patches_to_landmarks,
    patch_uncertainty_pnp_threshold,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    with_landmark_ambiguity_scores,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _none_to_float(value):
    return None if value is None else float(value)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate patch-level query VFM token to sparse 3D landmark matching")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--submap_mode", default="reference_visibility", choices=("reference_visibility", "none"))
    parser.add_argument("--submap_top_n", type=int, default=5)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mutual_top_k", type=int, default=5)
    parser.add_argument("--match_mode", default="soft_mutual", choices=("nn", "mnn", "soft_mutual"))
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--min_similarity_margin", type=float, default=None)
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--max_landmark_variance", type=float, default=None)
    parser.add_argument("--max_landmark_reprojection_error", type=float, default=None)
    parser.add_argument("--max_landmark_ambiguity", type=float, default=None)
    parser.add_argument("--min_distance_to_boundary_px", type=float, default=None)
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--quality_ambiguity_reference_size", type=int, default=4096)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--match_block_size", type=int, default=256)
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--patch_at_k", type=int, default=5)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=1.5)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
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
    if args.max_landmark_ambiguity is not None:
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
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    config = PatchTo3DMatchingConfig(
        top_k=args.top_k,
        mutual_top_k=args.mutual_top_k,
        match_mode=args.match_mode,
        ratio_threshold=None if args.disable_ratio_test else args.ratio_threshold,
        min_similarity_margin=args.min_similarity_margin,
        min_similarity=args.min_similarity,
        max_landmark_variance=args.max_landmark_variance,
        max_landmark_reprojection_error=args.max_landmark_reprojection_error,
        max_landmark_ambiguity=args.max_landmark_ambiguity,
        min_distance_to_boundary_px=args.min_distance_to_boundary_px,
        min_observation_count=args.min_observation_count,
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
        block_size=args.match_block_size,
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
        visibility_gate = {"full_visible_tracks": None, "bank_visible_tracks": None, "bank_visibility_coverage": None}
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
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        stride = float(max(stride_x, stride_y))
        matches = match_query_patches_to_landmarks(query_feature, submap, config, int(camera.width), int(camera.height))
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=patch_uncertainty_pnp_threshold(stride, args.pnp_threshold_stride_multiplier),
            iterations=args.pnp_iterations,
        )
        gt_pose = gt_by_query.get(query_id)
        if gt_pose is None:
            raise ValueError(f"query pose not found for {query_id}")
        positives = build_patch_positive_sets(
            submap,
            gt_pose.pose_w2c,
            camera,
            token_width,
            token_height,
            patch_scale=args.patch_scale,
        )
        patch_stats = evaluate_patch_matches(
            matches,
            positives,
            gt_pose.pose_w2c,
            camera,
            stride_px=stride,
            pnp_inlier_mask=pnp.inlier_mask,
            top_k=args.patch_at_k,
        )
        pose_error = pnp_pose_error(pnp.pose_w2c, gt_pose.pose_w2c)
        all_spatial_stats = match_spatial_distribution_stats(matches, int(camera.width), int(camera.height))
        inlier_spatial_stats = match_spatial_distribution_stats(matches, int(camera.width), int(camera.height), pnp.inlier_mask)
        pnp_residual_stats = pnp_reprojection_residual_stats(matches, pnp.pose_w2c, camera, inlier_mask=pnp.inlier_mask)
        projected_landmarks = count_projected_landmarks(submap, gt_pose.pose_w2c, camera)
        if args.output_matches_jsonl:
            positive_by_token = positives.by_token
            for match_idx, match in enumerate(matches):
                positive = positive_by_token.get(int(match.token_index))
                match_rows.append(
                    {
                        "query_id": query_id,
                        "match_index": int(match_idx),
                        "token_index": int(match.token_index),
                        "track_id": int(match.track_id),
                        "source": match.source,
                        "xy": [float(match.xy[0]), float(match.xy[1])],
                        "similarity": float(match.similarity),
                        "similarity_margin": match.similarity_margin,
                        "landmark_variance": float(match.landmark_variance),
                        "landmark_reprojection_error": match.landmark_reprojection_error,
                        "landmark_ambiguity": match.landmark_ambiguity,
                        "patch_correct": bool(positive is not None and int(match.track_id) in positive.track_ids),
                        "positive_count": 0 if positive is None else int(positive.count),
                        "pnp_inlier": bool(pnp.inlier_mask.shape[0] > match_idx and pnp.inlier_mask[match_idx]),
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
            "submap_landmark_count": len(submap),
            "projected_landmarks": projected_landmarks,
            "coordinate_audit": {
                "query_feature_shape_chw": [int(v) for v in query_feature.shape],
                "camera_model_id": int(camera.model_id),
                "camera_width": int(camera.width),
                "camera_height": int(camera.height),
                "token_grid_scale_x_px": stride_x,
                "token_grid_scale_y_px": stride_y,
                "query_token_step": int(args.query_token_step),
                "patch_scale": float(args.patch_scale),
            },
            "match_count": len(matches),
            "mean_similarity": _mean([float(match.similarity) for match in matches]),
            "mean_positive_count": _mean([float(item.count) for item in positives.by_token.values()]),
            "nonempty_patch_fraction": _mean([1.0 if item.count > 0 else 0.0 for item in positives.by_token.values()]),
            "patch_geometry": patch_stats,
            "all_match_spatial": all_spatial_stats,
            "pnp_inlier_spatial": inlier_spatial_stats,
            "pnp_reprojection": pnp_residual_stats,
            "pnp_success": bool(pnp.success),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": float(pnp.inlier_ratio),
            "translation_error_m": None if not np.isfinite(pose_error.translation_m) else float(pose_error.translation_m),
            "rotation_error_deg": None if not np.isfinite(pose_error.rotation_deg) else float(pose_error.rotation_deg),
        }
        rows.append(row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    if args.output_matches_jsonl:
        matches_path = Path(args.output_matches_jsonl)
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        matches_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in match_rows) + ("\n" if match_rows else ""))

    labeled_rows = [row for row in rows if row["translation_error_m"] is not None]
    summary = {
        "stage": "patch_to_3d_vfm_matching_baseline",
        "query_count": len(rows),
        "labeled_query_count": len(labeled_rows),
        "landmark_count": len(landmark_index),
        "camera": {
            "source": camera_source,
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        },
        "matching_config": {
            **config.__dict__,
            "pnp_threshold_stride_multiplier": float(args.pnp_threshold_stride_multiplier),
            "patch_scale": float(args.patch_scale),
            "patch_at_k": int(args.patch_at_k),
        },
        "submap": {
            "mode": args.submap_mode,
            "top_n": args.submap_top_n,
            "max_landmarks": args.max_submap_landmarks,
            "mean_landmark_count": _mean([float(row["submap_landmark_count"]) for row in rows]),
            "mean_projected_landmarks": _mean([float(row["projected_landmarks"]) for row in rows]),
        },
        "mean_match_count": _mean([float(row["match_count"]) for row in rows]),
        "mean_patch_at_1": _mean([float(row["patch_geometry"]["patch_at_1"]) for row in rows]),
        f"mean_patch_at_{int(args.patch_at_k)}": _mean([float(row["patch_geometry"][f"patch_at_{int(args.patch_at_k)}"]) for row in rows]),
        "mean_gt_precision_5px": _mean([float(row["patch_geometry"]["gt_precision_5px"]) for row in rows]),
        "mean_gt_precision_16px": _mean([float(row["patch_geometry"]["gt_precision_16px"]) for row in rows]),
        "mean_gt_precision_stride": _mean([float(row["patch_geometry"]["gt_precision_stride"]) for row in rows]),
        "mean_gt_precision_2stride": _mean([float(row["patch_geometry"]["gt_precision_2stride"]) for row in rows]),
        "median_gt_reproj_median_px": None
        if not rows
        else float(np.median([float(row["patch_geometry"]["gt_reproj_median_px"]) for row in rows if row["patch_geometry"]["gt_reproj_median_px"] is not None])),
        "mean_pnp_inlier_patch_at_1": _mean(
            [
                float(row["patch_geometry"]["pnp_inlier_patch_at_1"])
                for row in rows
                if row["patch_geometry"]["pnp_inlier_patch_at_1"] is not None
            ]
        ),
        "mean_pnp_inlier_gt_precision_stride": _mean(
            [
                float(row["patch_geometry"]["pnp_inlier_gt_precision_stride"])
                for row in rows
                if row["patch_geometry"]["pnp_inlier_gt_precision_stride"] is not None
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
        "pnp_success_rate": _mean([1.0 if row["pnp_success"] else 0.0 for row in rows]),
        "success_25cm_10deg": _mean(
            [
                1.0
                if row["translation_error_m"] is not None
                and row["translation_error_m"] <= 0.25
                and row["rotation_error_deg"] is not None
                and row["rotation_error_deg"] <= 10.0
                else 0.0
                for row in labeled_rows
            ]
        ),
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
        "outputs": {"rows": str(output_jsonl), "matches": args.output_matches_jsonl},
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
