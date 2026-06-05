"""Visualize patch-level query-token to sparse 3D VFM landmark matches."""

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
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    PatchTo3DMatchingConfig,
    build_patch_positive_sets,
    evaluate_patch_matches,
    filter_landmarks_by_projected_visibility,
    match_query_patches_to_landmarks,
    patch_positive_set_stats,
    patch_uncertainty_pnp_threshold,
)
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkAmbiguityPruningConfig,
    LandmarkQualityConfig,
    LandmarkMapIndex,
    LocalGeometricConsistencyConfig,
    MapReliabilityConfig,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    pnp_pose_error,
    with_landmark_ambiguity_scores,
)
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap


def _matching_config_dict(config: PatchTo3DMatchingConfig) -> dict[str, object]:
    values = dict(config.__dict__)
    quality = values.get("landmark_quality")
    if isinstance(quality, LandmarkQualityConfig):
        values["landmark_quality"] = dict(quality.__dict__)
    reliability = values.get("map_reliability")
    if isinstance(reliability, MapReliabilityConfig):
        values["map_reliability"] = dict(reliability.__dict__)
    local = values.get("local_geometric_consistency")
    if isinstance(local, LocalGeometricConsistencyConfig):
        values["local_geometric_consistency"] = dict(local.__dict__)
    ambiguity = values.get("landmark_ambiguity_pruning")
    if isinstance(ambiguity, LandmarkAmbiguityPruningConfig):
        values["landmark_ambiguity_pruning"] = dict(ambiguity.__dict__)
    return values


from feature_extract.vfm.query_to_3d_visualization import (
    _match_patch_correct_mask,
    render_patch_to_3d_match_overlay,
    render_query_to_projected_map_correspondence,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _safe_query_name(query_id: str) -> str:
    return query_id.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _read_image_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"failed to write image: {path}")


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _quality_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.max_landmark_ambiguity is not None
        or (args.enable_landmark_quality and (args.quality_idf_weight > 0.0 or args.quality_ambiguity_weight > 0.0))
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize patch-level query VFM token to sparse 3D landmark matches")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", default="")
    parser.add_argument("--semidense_anchor_npz", default="")
    parser.add_argument("--track_observations", default="")
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--query_id", action="append", required=True)
    parser.add_argument("--submap_mode", default="reference_visibility", choices=("reference_visibility", "gt_visible", "none"))
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=20000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--mutual_top_k", type=int, default=1)
    parser.add_argument("--match_mode", default="mnn", choices=("nn", "mnn", "soft_mutual"))
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--min_similarity_margin", type=float, default=None)
    parser.add_argument("--min_similarity", type=float, default=0.2)
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
    parser.add_argument("--deduplicate_track_matches", action="store_true")
    parser.add_argument("--match_block_size", type=int, default=256)
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--patch_at_k", type=int, default=5)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=1.5)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--precision_reprojection_threshold_px", type=float, default=16.0)
    parser.add_argument("--max_draw_matches", type=int, default=160)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = {record.image_id: record for record in manifest.records}
    missing_queries = sorted(set(args.query_id) - set(records))
    if missing_queries:
        raise ValueError(f"query_id not found in manifest: {missing_queries}")
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    if not args.landmark_bank and not args.semidense_anchor_npz:
        raise ValueError("either --landmark_bank or --semidense_anchor_npz is required")
    if args.semidense_anchor_npz:
        semidense_map = SemiDenseAnchorMap.load_npz(Path(args.semidense_anchor_npz))
        landmark_index = semidense_map.to_landmark_index()
        map_source = "semidense_anchor_npz"
    else:
        if not args.track_observations:
            raise ValueError("--track_observations is required when loading --landmark_bank")
        bank = load_selected_track_bank_npz(Path(args.landmark_bank))
        xyz_by_track, reprojection_error_by_track = _load_track_stats(Path(args.track_observations))
        landmark_index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
        map_source = "landmark_bank"
    if _quality_enabled(args):
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
        min_quality_weighted_similarity=args.min_quality_weighted_similarity,
        min_observation_count=args.min_observation_count,
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
        block_size=args.match_block_size,
        deduplicate_track_matches=bool(args.deduplicate_track_matches),
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

    output_dir = Path(args.output_dir)
    summaries = []
    for query_id in args.query_id:
        record = records[query_id]
        gt = gt_by_query.get(query_id)
        if gt is None:
            raise ValueError(f"query pose not found for {query_id}")
        references = reference_submaps.get(query_id, [])
        submap = landmark_index
        if args.submap_mode == "gt_visible":
            submap = filter_landmarks_by_projected_visibility(landmark_index, gt.pose_w2c, camera)
        elif args.submap_mode == "reference_visibility":
            if visibility_index is None:
                submap = filter_landmarks_by_reference_images(landmark_index, references)
            else:
                submap, _visibility_gate = filter_landmarks_by_visibility(landmark_index, visibility_index, references)
        submap = _limit_submap(submap, args.max_submap_landmarks)
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_height, token_width = query_feature.shape
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        stride = float(max(stride_x, stride_y))
        matches = match_query_patches_to_landmarks(
            query_feature,
            submap,
            config,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=patch_uncertainty_pnp_threshold(stride, args.pnp_threshold_stride_multiplier),
            iterations=args.pnp_iterations,
        )
        positives = build_patch_positive_sets(
            submap,
            gt.pose_w2c,
            camera,
            token_width,
            token_height,
            patch_scale=args.patch_scale,
        )
        patch_stats = evaluate_patch_matches(
            matches,
            positives,
            gt.pose_w2c,
            camera,
            stride_px=stride,
            pnp_inlier_mask=pnp.inlier_mask,
            top_k=args.patch_at_k,
        )
        pose_error = pnp_pose_error(pnp.pose_w2c, gt.pose_w2c)
        image_rgb = _read_image_rgb(Path(args.image_root) / query_id)
        safe_name = _safe_query_name(query_id)
        patch_correct = _match_patch_correct_mask(matches, positives)
        overlay, overlay_summary = render_patch_to_3d_match_overlay(
            image_rgb,
            matches,
            positives,
            pose_w2c=gt.pose_w2c,
            camera=camera,
            inlier_mask=pnp.inlier_mask,
            max_draw=args.max_draw_matches,
        )
        projection_rgb, projection_rgb_summary = render_query_to_projected_map_correspondence(
            image_rgb,
            query_feature,
            submap,
            matches,
            pose_w2c=gt.pose_w2c,
            camera=camera,
            mode="rgb",
            inlier_mask=pnp.inlier_mask,
            match_correct_mask=patch_correct,
            correct_label="Patch precision",
            reprojection_threshold_px=args.precision_reprojection_threshold_px,
            max_draw=args.max_draw_matches,
        )
        projection_feature, projection_feature_summary = render_query_to_projected_map_correspondence(
            image_rgb,
            query_feature,
            submap,
            matches,
            pose_w2c=gt.pose_w2c,
            camera=camera,
            mode="feature",
            inlier_mask=pnp.inlier_mask,
            match_correct_mask=patch_correct,
            correct_label="Patch precision",
            reprojection_threshold_px=args.precision_reprojection_threshold_px,
            max_draw=args.max_draw_matches,
        )
        overlay_path = output_dir / f"{safe_name}_patch_overlay.png"
        projection_rgb_path = output_dir / f"{safe_name}_patch_projection_rgb.png"
        projection_feature_path = output_dir / f"{safe_name}_patch_projection_feature.png"
        _write_image_rgb(overlay_path, overlay)
        _write_image_rgb(projection_rgb_path, projection_rgb)
        _write_image_rgb(projection_feature_path, projection_feature)
        item = {
            "query_id": query_id,
            "map_source": map_source,
            "overlay_png": str(overlay_path),
            "projection_rgb_png": str(projection_rgb_path),
            "projection_feature_png": str(projection_feature_path),
            "submap_reference_count": len(references),
            "submap_landmark_count": len(submap),
            "camera_source": camera_source,
            "pnp_success": bool(pnp.success),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": float(pnp.inlier_ratio),
            "translation_error_m": _finite_or_none(pose_error.translation_m),
            "rotation_error_deg": _finite_or_none(pose_error.rotation_deg),
            "positive_set_stats": patch_positive_set_stats(positives),
            "patch_geometry": patch_stats,
            "overlay": overlay_summary,
            "projection_rgb": projection_rgb_summary,
            "projection_feature": projection_feature_summary,
        }
        summary_path = output_dir / f"{safe_name}_patch_matches_summary.json"
        summary_path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
        summaries.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(
            {
                "stage": "patch_to_3d_vfm_match_visualization",
                "map_source": map_source,
                "query_count": len(summaries),
                "queries": summaries,
                "matching_config": {
                    **_matching_config_dict(config),
                    "pnp_threshold_stride_multiplier": float(args.pnp_threshold_stride_multiplier),
                    "patch_scale": float(args.patch_scale),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
