"""Visualize query-token to rendered Gaussian VFM map matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_render_vfm_matching import (
    _load_candidate_poses,
    _load_default_camera,
    _load_query_feature,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMField,
    GaussianVFMRenderConfig,
    load_gaussian_rgb_source_from_ply,
    render_gaussian_rgb_image_gsplat,
    render_gaussian_vfm_feature_map_gsplat,
)
from feature_extract.vfm.query_to_3d_matching import (
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)
from feature_extract.vfm.query_to_3d_visualization import _fit_pca_projection, _pca_colors, project_match_landmarks
from feature_extract.vfm.query_to_render_matching import (
    QueryToRenderMatchingConfig,
    match_query_tokens_to_rendered_map,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _read_image_rgb(path: Path) -> np.ndarray:
    import cv2

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"failed to write image: {path}")


def _clip_xy(xy: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(round(float(np.clip(xy[0], 0.0, max(width - 1, 0)))))
    y = int(round(float(np.clip(xy[1], 0.0, max(height - 1, 0)))))
    return x, y


def _safe_query_name(query_id: str) -> str:
    return query_id.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _render_feature_pca(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    channels, height, width = feature_map.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    flat = feature_map.reshape(channels, -1).T
    visible = flat[mask.reshape(-1)]
    if visible.shape[0] == 0:
        return rgb
    mean, components, low, high = _fit_pca_projection(visible)
    rgb.reshape(-1, 3)[mask.reshape(-1)] = _pca_colors(visible, mean, components, low, high)
    return rgb


def _render_overlay(
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    right_label: str,
    matches,
    gt_pose_w2c: np.ndarray,
    camera,
    inlier_mask: np.ndarray,
    reprojection_threshold_px: float,
    max_draw: int,
) -> tuple[np.ndarray, dict[str, object]]:
    import cv2

    query_rgb = np.asarray(query_rgb, dtype=np.uint8)
    render_rgb = np.asarray(render_rgb, dtype=np.uint8)
    qh, qw = query_rgb.shape[:2]
    render_panel = cv2.resize(render_rgb, (qw, qh), interpolation=cv2.INTER_NEAREST)
    gap = 48
    canvas = np.zeros((qh, qw * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :qw] = query_rgb
    canvas[:, qw + gap :] = render_panel
    canvas[:, qw : qw + gap] = 18
    if not matches:
        return canvas, {
            "match_count": 0,
            "drawn_match_count": 0,
            "gt_precision": 0.0,
            "pnp_inlier_count": 0,
        }
    projected = project_match_landmarks(matches, pose_w2c=gt_pose_w2c, camera=camera)
    query_xy = np.stack([match.xy for match in matches], axis=0).astype(np.float64)
    errors = np.linalg.norm(query_xy - projected, axis=1)
    gt_mask = errors <= float(reprojection_threshold_px)
    pnp_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    draw_count = min(int(max_draw), len(matches))
    for idx in range(draw_count):
        qx, qy = _clip_xy(query_xy[idx], qw, qh)
        projected_xy = projected[idx]
        rx = int(round(float(match_pixel_x(matches[idx].track_id, render_rgb.shape[1]) / max(render_rgb.shape[1] - 1, 1) * (qw - 1))))
        ry = int(round(float(match_pixel_y(matches[idx].track_id, render_rgb.shape[1]) / max(render_rgb.shape[0] - 1, 1) * (qh - 1))))
        color = (30, 210, 70) if bool(gt_mask[idx]) else (235, 45, 45)
        target = (qw + gap + rx, ry)
        cv2.line(canvas, (qx, qy), target, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (qx, qy), 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, target, 4, color, -1, lineType=cv2.LINE_AA)
        px, py = _clip_xy(projected_xy, qw, qh)
        cv2.circle(canvas, (px, py), 5, (70, 160, 255), 1, lineType=cv2.LINE_AA)
        if idx < pnp_mask.shape[0] and bool(pnp_mask[idx]):
            cv2.circle(canvas, (qx, qy), 7, (255, 220, 40), 1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, target, 7, (255, 220, 40), 1, lineType=cv2.LINE_AA)
    panel = canvas.copy()
    cv2.rectangle(panel, (8, 8), (650, 108), (0, 0, 0), -1)
    canvas = cv2.addWeighted(panel, 0.42, canvas, 0.58, 0.0)
    lines = [
        f"matches: {len(matches)} drawn: {draw_count}",
        f"GT precision@{reprojection_threshold_px:g}px: {float(np.mean(gt_mask)):.3f}",
        f"mean reproj error: {float(np.mean(errors)):.1f}px",
        f"yellow rings: PnP inliers {int(np.sum(pnp_mask))}",
        f"left: query RGB   right: {right_label}",
    ]
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return canvas, {
        "match_count": int(len(matches)),
        "drawn_match_count": int(draw_count),
        "gt_precision": float(np.mean(gt_mask)),
        "gt_inlier_count": int(np.sum(gt_mask)),
        "pnp_inlier_count": int(np.sum(pnp_mask)),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "median_reprojection_error_px": float(np.median(errors)),
    }


def match_pixel_x(track_id: int, width: int) -> int:
    return int(track_id) % int(width)


def match_pixel_y(track_id: int, width: int) -> int:
    return int(track_id) // int(width)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize query-to-rendered VFM matches")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--rgb_gaussian_ply", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--query_id", action="append", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--candidate_top_n", type=int, default=1)
    parser.add_argument("--render_width", type=int, default=160)
    parser.add_argument("--render_height", type=int, default=90)
    parser.add_argument("--render_radius_px", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channel_chunk", type=int, default=32)
    parser.add_argument("--query_token_step", type=int, default=8)
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--precision_reprojection_threshold_px", type=float, default=16.0)
    parser.add_argument("--max_draw_matches", type=int, default=120)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = {record.image_id: record for record in manifest.records}
    field = GaussianVFMField.load_npz(Path(args.field))
    rgb_source = load_gaussian_rgb_source_from_ply(Path(args.rgb_gaussian_ply)) if args.rgb_gaussian_ply else None
    camera = _load_default_camera(args.camera_model_dir, _parse_default_camera(args.default_camera))
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    candidates_by_query = _load_candidate_poses(args.candidate_bank, args.candidate_top_n)
    render_config = GaussianVFMRenderConfig(
        width=args.render_width,
        height=args.render_height,
        radius_px=args.render_radius_px,
    )
    match_config = QueryToRenderMatchingConfig(
        top_k=2,
        ratio_threshold=args.ratio_threshold,
        min_similarity=args.min_similarity,
        mutual=bool(args.mutual),
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
    )
    output_dir = Path(args.output_dir)
    summaries = []
    for query_id in args.query_id:
        if query_id not in records:
            raise ValueError(f"query_id not in manifest: {query_id}")
        if query_id not in gt_by_query:
            raise ValueError(f"query_id not in pose file: {query_id}")
        candidates = candidates_by_query.get(query_id, [])
        if not candidates:
            raise ValueError(f"no candidates found for query_id: {query_id}")
        candidate = candidates[0]
        rendered = render_gaussian_vfm_feature_map_gsplat(
            field,
            pose_w2c=np.asarray(candidate["pose_w2c"], dtype=np.float64),
            camera=camera,
            config=render_config,
            device=args.device,
            channel_chunk=args.channel_chunk,
        )
        query_feature = _load_query_feature(records[query_id].token_path, args.layer_name)
        matches = match_query_tokens_to_rendered_map(
            query_feature,
            rendered.feature_map,
            rendered.xyz_map,
            rendered.visibility_mask,
            match_config,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=args.pnp_reprojection_error_px,
            iterations=args.pnp_iterations,
        )
        gt = gt_by_query[query_id]
        pose_error = pnp_pose_error(pnp.pose_w2c, gt.pose_w2c)
        query_rgb = _read_image_rgb(Path(args.image_root) / query_id)
        if rgb_source is None:
            render_rgb = _render_feature_pca(rendered.feature_map, rendered.visibility_mask)
            right_panel_kind = "rendered_hybrid_vfm_pca"
        else:
            render_rgb_float, _alpha = render_gaussian_rgb_image_gsplat(
                rgb_source,
                pose_w2c=np.asarray(candidate["pose_w2c"], dtype=np.float64),
                camera=camera,
                config=render_config,
                device=args.device,
            )
            render_rgb = np.asarray(np.clip(render_rgb_float, 0.0, 1.0) * 255.0, dtype=np.uint8)
            right_panel_kind = "rendered_gaussian_rgb"
        overlay, overlay_summary = _render_overlay(
            query_rgb,
            render_rgb,
            "rendered Gaussian RGB" if rgb_source is not None else "rendered hybrid VFM PCA",
            matches,
            gt_pose_w2c=gt.pose_w2c,
            camera=camera,
            inlier_mask=pnp.inlier_mask,
            reprojection_threshold_px=args.precision_reprojection_threshold_px,
            max_draw=args.max_draw_matches,
        )
        safe = _safe_query_name(query_id)
        png_path = output_dir / f"{safe}_query_to_hybrid_render.png"
        _write_image_rgb(png_path, overlay)
        summary = {
            "query_id": query_id,
            "candidate_id": candidate["candidate_id"],
            "reference_image": candidate["reference_image"],
            "selected_candidate_rank": candidate["rank"],
            "output_png": str(png_path),
            "right_panel_kind": right_panel_kind,
            "render_visible_fraction": float(np.mean(rendered.visibility_mask)),
            "pnp_success": bool(pnp.success),
            "pnp_inlier_count": int(pnp.inlier_count),
            "pnp_inlier_ratio": float(pnp.inlier_ratio),
            "translation_error_m": None if not np.isfinite(pose_error.translation_m) else float(pose_error.translation_m),
            "rotation_error_deg": None if not np.isfinite(pose_error.rotation_deg) else float(pose_error.rotation_deg),
            "overlay": overlay_summary,
        }
        (output_dir / f"{safe}_query_to_hybrid_render_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        summaries.append(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(
            {
                "stage": "query_to_render_vfm_match_visualization",
                "query_count": len(summaries),
                "queries": summaries,
                "matching_config": match_config.__dict__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
