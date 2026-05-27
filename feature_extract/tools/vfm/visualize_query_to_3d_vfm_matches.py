"""Visualize query-token to 3D VFM landmark matches for Stage B diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _limit_submap,
    _load_default_camera,
    _load_query_feature,
    _load_reference_submaps,
    _load_xyz_by_track,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex, filter_landmarks_by_visibility
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    QueryTo3DMatchingConfig,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_query_tokens_to_landmarks,
    pnp_pose_error,
)
from feature_extract.vfm.query_to_3d_visualization import (
    render_query_to_3d_match_overlay,
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize query-to-3D raw VFM matches")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--query_id", action="append", required=True)
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
    parser.add_argument("--match_block_size", type=int, default=128)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--precision_reprojection_threshold_px", type=float, default=16.0)
    parser.add_argument("--max_draw_matches", type=int, default=250)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = {record.image_id: record for record in manifest.records}
    missing_queries = sorted(set(args.query_id) - set(records))
    if missing_queries:
        raise ValueError(f"query_id not found in manifest: {missing_queries}")
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    bank = load_selected_track_bank_npz(Path(args.landmark_bank))
    landmark_index = LandmarkMapIndex.from_track_bank(bank, _load_xyz_by_track(Path(args.track_observations)))
    visibility_index = None
    if args.visibility_index:
        visibility_index = LandmarkVisibilityIndex.load_npz(Path(args.visibility_index))
    camera = _load_default_camera(args.camera_model_dir, _parse_default_camera(args.default_camera))
    reference_submaps = _load_reference_submaps(args.candidate_bank, args.submap_top_n)
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

    output_dir = Path(args.output_dir)
    summaries = []
    for query_id in args.query_id:
        record = records[query_id]
        references = reference_submaps.get(query_id, [])
        submap = landmark_index
        if args.submap_mode == "reference_visibility":
            if visibility_index is None:
                submap = filter_landmarks_by_reference_images(landmark_index, references)
            else:
                submap, _visibility_gate = filter_landmarks_by_visibility(landmark_index, visibility_index, references)
        submap = _limit_submap(submap, args.max_submap_landmarks)
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
        gt = gt_by_query.get(query_id)
        if gt is None:
            raise ValueError(f"query pose not found for {query_id}")
        pose_error = pnp_pose_error(pnp.pose_w2c, gt.pose_w2c)
        image_rgb = _read_image_rgb(Path(args.image_root) / query_id)
        overlay, overlay_summary = render_query_to_3d_match_overlay(
            image_rgb,
            matches,
            pose_w2c=gt.pose_w2c,
            camera=camera,
            inlier_mask=pnp.inlier_mask,
            reprojection_threshold_px=args.precision_reprojection_threshold_px,
            max_draw=args.max_draw_matches,
        )
        safe_name = _safe_query_name(query_id)
        png_path = output_dir / f"{safe_name}_matches.png"
        _write_image_rgb(png_path, overlay)
        projection_rgb, projection_rgb_summary = render_query_to_projected_map_correspondence(
            image_rgb,
            query_feature,
            submap,
            matches,
            pose_w2c=gt.pose_w2c,
            camera=camera,
            mode="rgb",
            inlier_mask=pnp.inlier_mask,
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
            reprojection_threshold_px=args.precision_reprojection_threshold_px,
            max_draw=args.max_draw_matches,
        )
        projection_rgb_path = output_dir / f"{safe_name}_projection_rgb.png"
        projection_feature_path = output_dir / f"{safe_name}_projection_feature.png"
        _write_image_rgb(projection_rgb_path, projection_rgb)
        _write_image_rgb(projection_feature_path, projection_feature)
        item = {
            "query_id": query_id,
            "output_png": str(png_path),
            "projection_rgb_png": str(projection_rgb_path),
            "projection_feature_png": str(projection_feature_path),
            "submap_reference_count": len(references),
            "submap_landmark_count": len(submap),
            "pnp_success": bool(pnp.success),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": float(pnp.inlier_ratio),
            "translation_error_m": _finite_or_none(pose_error.translation_m),
            "rotation_error_deg": _finite_or_none(pose_error.rotation_deg),
            "overlay": overlay_summary,
            "projection_rgb": projection_rgb_summary,
            "projection_feature": projection_feature_summary,
        }
        summary_path = output_dir / f"{safe_name}_matches_summary.json"
        summary_path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
        summaries.append(item)

    (output_dir / "visualization_summary.json").write_text(
        json.dumps(
            {
                "stage": "query_to_3d_vfm_match_visualization",
                "query_count": len(summaries),
                "queries": summaries,
                "matching_config": config.__dict__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
