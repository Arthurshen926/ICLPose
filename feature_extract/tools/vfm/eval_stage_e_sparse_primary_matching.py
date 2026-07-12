"""Evaluate Stage E sparse-primary semi-dense matching ablations."""

from __future__ import annotations

import argparse
import json
import time
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
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    PatchTo3DMatchingConfig,
    build_patch_positive_sets,
    evaluate_patch_matches,
    patch_positive_set_stats,
    patch_uncertainty_pnp_threshold,
    match_query_patches_to_landmarks,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    estimate_pose_pnp_ransac,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
)
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.semidense_stage_e import (
    SemiDensePruningConfig,
    SparsePrimaryFillConfig,
    apply_semidense_context_to_sparse_matches,
    concatenate_landmark_indices,
    duplicate_anchor_stats,
    prune_semidense_anchors,
    semidense_gaussian_only,
    source_breakdown_stats,
    sparse_primary_fill_matches,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _semidense_visible_subset(
    anchor_map: SemiDenseAnchorMap,
    visibility_index: LandmarkVisibilityIndex,
    references: Sequence[str],
) -> SemiDenseAnchorMap:
    visible = visibility_index.visible_tracks(references)
    if not visible:
        return anchor_map.subset([])
    return anchor_map.subset(np.asarray([int(track_id) in visible for track_id in anchor_map.source_track_ids], dtype=bool))


def _full_inlier_mask(matches, pnp_matches, pnp_mask: np.ndarray) -> np.ndarray:
    full = np.zeros((len(matches),), dtype=bool)
    positions = {id(match): idx for idx, match in enumerate(matches)}
    for local_idx, match in enumerate(pnp_matches):
        if local_idx < pnp_mask.shape[0] and bool(pnp_mask[local_idx]):
            idx = positions.get(id(match))
            if idx is not None:
                full[idx] = True
    return full


def _success(t: float | None, r: float | None, t_thr: float, r_thr: float) -> bool:
    return bool(t is not None and r is not None and float(t) <= float(t_thr) and float(r) <= float(r_thr))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage E sparse-primary semi-dense matching ablation")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--sparse_landmark_bank", required=True)
    parser.add_argument("--semidense_npz", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--visibility_index", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--mutual_top_k", type=int, default=1)
    parser.add_argument("--match_mode", default="mnn", choices=("nn", "mnn", "soft_mutual"))
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--candidate_max_matches", type=int, default=1000)
    parser.add_argument("--match_block_size", type=int, default=1024)
    parser.add_argument("--similarity_device", default="cpu")
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--patch_at_k", type=int, default=5)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=0.75)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--fill_mode", default="no_sparse", choices=("sparse_only", "all", "no_sparse", "low_margin", "low_coverage", "context_only"))
    parser.add_argument("--semi_quota_fraction", type=float, default=0.2)
    parser.add_argument("--low_margin_threshold", type=float, default=0.02)
    parser.add_argument("--fill_grid_rows", type=int, default=4)
    parser.add_argument("--fill_grid_cols", type=int, default=4)
    parser.add_argument("--fill_min_sparse_per_cell", type=int, default=4)
    parser.add_argument("--fill_max_semidense_per_cell", type=int, default=0)
    parser.add_argument("--semi_min_support", type=int, default=1)
    parser.add_argument("--semi_min_quality", type=float, default=None)
    parser.add_argument("--semi_max_distance", type=float, default=None)
    parser.add_argument("--semi_max_feature_variance", type=float, default=None)
    parser.add_argument("--semi_max_per_source_track", type=int, default=0)
    parser.add_argument("--context_radius_m", type=float, default=0.10)
    parser.add_argument("--context_weight", type=float, default=0.10)
    parser.add_argument("--context_keep_fraction", type=float, default=None)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    xyz_by_track, reprojection_error_by_track = _load_track_stats(Path(args.track_observations))
    sparse_index = LandmarkMapIndex.from_track_bank(
        load_selected_track_bank_npz(Path(args.sparse_landmark_bank)),
        xyz_by_track,
        reprojection_error_by_track,
    )
    semidense_map = SemiDenseAnchorMap.load_npz(Path(args.semidense_npz))
    semidense_gaussian = semidense_gaussian_only(semidense_map)
    semidense_gaussian = prune_semidense_anchors(
        semidense_gaussian,
        SemiDensePruningConfig(
            min_support=int(args.semi_min_support),
            min_quality=args.semi_min_quality,
            max_distance=args.semi_max_distance,
            max_feature_variance=args.semi_max_feature_variance,
            max_per_source_track=None if int(args.semi_max_per_source_track) <= 0 else int(args.semi_max_per_source_track),
        ),
    )
    visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index))
    reference_submaps = _load_reference_submaps(args.candidate_bank, int(args.submap_top_n))
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    config = PatchTo3DMatchingConfig(
        top_k=int(args.top_k),
        mutual_top_k=int(args.mutual_top_k),
        match_mode=args.match_mode,
        ratio_threshold=None if args.disable_ratio_test else 0.95,
        min_similarity=float(args.min_similarity),
        min_observation_count=1,
        query_token_step=int(args.query_token_step),
        max_matches=int(args.candidate_max_matches),
        block_size=int(args.match_block_size),
        similarity_device=args.similarity_device,
        match_score_mode="similarity_quality",
    )

    rows = []
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    for record in records:
        pose = poses.get(record.image_id)
        if pose is None:
            continue
        references = reference_submaps.get(record.image_id, [])
        sparse_submap, sparse_gate = filter_landmarks_by_visibility(sparse_index, visibility_index, references)
        sparse_submap = _limit_submap(sparse_submap, int(args.max_submap_landmarks))
        semi_visible = _semidense_visible_subset(semidense_gaussian, visibility_index, references)
        semi_index = _limit_submap(semi_visible.to_landmark_index(), int(args.max_submap_landmarks))
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_height, token_width = query_feature.shape
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        stride = max(stride_x, stride_y)
        sparse_matches = match_query_patches_to_landmarks(
            query_feature,
            sparse_submap,
            config,
            int(camera.width),
            int(camera.height),
        )
        if args.fill_mode == "sparse_only":
            matches = sparse_matches[: int(args.max_matches)]
        elif args.fill_mode == "context_only":
            matches = apply_semidense_context_to_sparse_matches(
                sparse_matches,
                query_feature,
                semi_index,
                radius_m=float(args.context_radius_m),
                context_weight=float(args.context_weight),
                keep_fraction=args.context_keep_fraction,
            )[: int(args.max_matches)]
        else:
            semi_matches = match_query_patches_to_landmarks(
                query_feature,
                semi_index,
                config,
                int(camera.width),
                int(camera.height),
            )
            matches = sparse_primary_fill_matches(
                sparse_matches,
                semi_matches,
                SparsePrimaryFillConfig(
                    mode=args.fill_mode,
                    max_semidense_fraction=float(args.semi_quota_fraction),
                    low_margin_threshold=float(args.low_margin_threshold),
                    grid_rows=int(args.fill_grid_rows),
                    grid_cols=int(args.fill_grid_cols),
                    min_sparse_per_cell=int(args.fill_min_sparse_per_cell),
                    max_semidense_per_cell=None
                    if int(args.fill_max_semidense_per_cell) <= 0
                    else int(args.fill_max_semidense_per_cell),
                ),
                int(camera.width),
                int(camera.height),
            )[: int(args.max_matches)]
        combined_index = concatenate_landmark_indices([sparse_submap, semi_index])
        positives = build_patch_positive_sets(
            combined_index,
            pose.pose_w2c,
            camera,
            token_width,
            token_height,
            patch_scale=float(args.patch_scale),
        )
        pnp_threshold = patch_uncertainty_pnp_threshold(stride, float(args.pnp_threshold_stride_multiplier))
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=pnp_threshold,
            iterations=int(args.pnp_iterations),
            confidence=float(args.pnp_confidence),
        )
        pose_error = pnp_pose_error(pnp.pose_w2c, pose.pose_w2c) if pnp.pose_w2c is not None else None
        translation = None if pose_error is None else float(pose_error.translation_m)
        rotation = None if pose_error is None else float(pose_error.rotation_deg)
        patch_geometry = evaluate_patch_matches(
            matches,
            positives,
            pose.pose_w2c,
            camera,
            stride,
            pnp_inlier_mask=pnp.inlier_mask,
            top_k=int(args.patch_at_k),
        )
        source_breakdown = source_breakdown_stats(
            matches,
            positives,
            pose.pose_w2c,
            camera,
            stride,
            pnp_inlier_mask=pnp.inlier_mask,
        )
        rows.append(
            {
                "query_id": record.image_id,
                "fill_mode": args.fill_mode,
                "reference_count": len(references),
                "sparse_submap_landmarks": int(len(sparse_submap)),
                "semidense_submap_landmarks": int(len(semi_index)),
                "sparse_full_visible_tracks": sparse_gate["full_visible_tracks"],
                "sparse_bank_visible_tracks": sparse_gate["bank_visible_tracks"],
                "positive_set_stats": patch_positive_set_stats(positives),
                "match_count": int(len(matches)),
                "source_breakdown": source_breakdown,
                "patch_geometry": patch_geometry,
                "all_match_spatial": match_spatial_distribution_stats(
                    matches,
                    int(camera.width),
                    int(camera.height),
                    pose_w2c=pnp.pose_w2c,
                ),
                "pnp_inlier_spatial": match_spatial_distribution_stats(
                    matches,
                    int(camera.width),
                    int(camera.height),
                    mask=pnp.inlier_mask,
                    pose_w2c=pnp.pose_w2c,
                ),
                "pnp_reprojection": pnp_reprojection_residual_stats(matches, pnp.pose_w2c, camera, pnp.inlier_mask),
                "pnp_solve": bool(pnp.success),
                "pnp_inlier_count": int(pnp.inlier_count),
                "pnp_inlier_ratio": float(pnp.inlier_ratio),
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
                "success_10cm_5deg": _success(translation, rotation, 0.10, 5.0),
                "success_25cm_10deg": _success(translation, rotation, 0.25, 10.0),
                "success_50cm_10deg": _success(translation, rotation, 0.50, 10.0),
                "success_1m_10deg": _success(translation, rotation, 1.0, 10.0),
            }
        )

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    labeled = [row for row in rows if row["translation_error_m"] is not None]
    summary = {
        "stage": "stage_e_sparse_primary_semidense_matching",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": len(rows),
        "labeled_query_count": len(labeled),
        "camera_source": camera_source,
        "fill_mode": args.fill_mode,
        "semidense_pruning": {
            "input_gaussian_anchor_count": int(np.sum(np.asarray(semidense_map.source_types, dtype=str) != "sfm")),
            "kept_gaussian_anchor_count": int(len(semidense_gaussian)),
            "duplicate_stats": duplicate_anchor_stats(semidense_gaussian),
            "min_support": int(args.semi_min_support),
            "min_quality": args.semi_min_quality,
            "max_distance": args.semi_max_distance,
            "max_feature_variance": args.semi_max_feature_variance,
            "max_per_source_track": None
            if int(args.semi_max_per_source_track) <= 0
            else int(args.semi_max_per_source_track),
        },
        "fill_config": {
            "semi_quota_fraction": float(args.semi_quota_fraction),
            "low_margin_threshold": float(args.low_margin_threshold),
            "grid_rows": int(args.fill_grid_rows),
            "grid_cols": int(args.fill_grid_cols),
            "min_sparse_per_cell": int(args.fill_min_sparse_per_cell),
            "max_semidense_per_cell": None
            if int(args.fill_max_semidense_per_cell) <= 0
            else int(args.fill_max_semidense_per_cell),
            "context_radius_m": float(args.context_radius_m),
            "context_weight": float(args.context_weight),
            "context_keep_fraction": args.context_keep_fraction,
        },
        "mean_sparse_submap_landmarks": _mean([float(row["sparse_submap_landmarks"]) for row in rows]),
        "mean_semidense_submap_landmarks": _mean([float(row["semidense_submap_landmarks"]) for row in rows]),
        "mean_match_count": _mean([float(row["match_count"]) for row in rows]),
        "mean_semidense_match_fraction": _mean(
            [float(row["source_breakdown"]["semidense"]["match_fraction"]) for row in rows]
        ),
        "mean_semidense_inlier_fraction": _mean(
            [float(row["source_breakdown"]["semidense"]["pnp_inlier_fraction"]) for row in rows]
        ),
        "mean_patch_at_1": _mean([float(row["patch_geometry"]["patch_at_1"]) for row in rows]),
        "mean_gt_precision_stride": _mean([float(row["patch_geometry"]["gt_precision_stride"]) for row in rows]),
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
        "mean_pnp_inlier_count": _mean([float(row["pnp_inlier_count"]) for row in rows]),
        "success_10cm_5deg": _mean([1.0 if row["success_10cm_5deg"] else 0.0 for row in labeled]),
        "success_25cm_10deg": _mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in labeled]),
        "success_50cm_10deg": _mean([1.0 if row["success_50cm_10deg"] else 0.0 for row in labeled]),
        "success_1m_10deg": _mean([1.0 if row["success_1m_10deg"] else 0.0 for row in labeled]),
        "median_translation_error_m": _median([float(row["translation_error_m"]) for row in labeled]),
        "median_rotation_error_deg": _median([float(row["rotation_error_deg"]) for row in labeled]),
        "inputs": {
            "query_manifest": args.query_manifest,
            "sparse_landmark_bank": args.sparse_landmark_bank,
            "semidense_npz": args.semidense_npz,
            "track_observations": args.track_observations,
            "visibility_index": args.visibility_index,
            "query_pose_file": args.query_pose_file,
            "candidate_bank": args.candidate_bank,
        },
        "outputs": {"rows": str(output)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
