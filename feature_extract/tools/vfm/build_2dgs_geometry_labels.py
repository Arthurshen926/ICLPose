"""Render high-resolution depth/normal/alpha labels from trained 2DGS PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file, safe_image_id_key
from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_geometry_labels import TwoDgsGeometryRenderConfig, render_2dgs_geometry_label
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, _normalize_vectors, _surface_tangent_axes_and_scales


def _render_surface_from_ply(
    gaussian_ply: Path,
    clean_membership_ply: Path | None = None,
    max_gaussians: int = 0,
    min_opacity: float = 0.0,
    max_scale: float = 0.0,
) -> SurfaceElementMap:
    # Geometry must come from the full oriented 2DGS.  A cleaned PLY may only
    # provide membership via source_index; it is not allowed to replace the
    # missing rotations/scales with isotropic points.
    source = load_gaussian_vfm_source_from_ply(Path(gaussian_ply), max_gaussians=0)
    if source.rotation is None or source.normal is None:
        raise ValueError(
            "geometry labels require an oriented 2DGS PLY with rotation/normal; "
            "pass a clean point PLY through --clean_membership_ply instead"
        )
    if clean_membership_ply is not None:
        membership = load_gaussian_vfm_source_from_ply(
            Path(clean_membership_ply), max_gaussians=int(max_gaussians),
        )
        rows = np.asarray(membership.gaussian_indices, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= source.xyz.shape[0]):
            raise ValueError("clean membership source_index is outside the oriented 2DGS")
        if not np.allclose(membership.xyz, source.xyz[rows], rtol=0.0, atol=1e-6):
            raise ValueError("clean membership xyz does not match oriented 2DGS source_index")
    else:
        rows = np.arange(source.xyz.shape[0], dtype=np.int64)
        if int(max_gaussians) > 0:
            rows = rows[: int(max_gaussians)]
    keep = np.asarray(source.opacity, dtype=np.float32)[rows] >= float(min_opacity)
    if float(max_scale) > 0.0:
        keep &= np.asarray(source.scale, dtype=np.float32)[rows] <= float(max_scale)
    rows = rows[keep]
    centers = np.asarray(source.xyz, dtype=np.float64)[rows]
    normals = (
        np.asarray(source.normal, dtype=np.float32)[rows]
        if source.normal is not None
        else np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (rows.size, 1))
    )
    normals = _normalize_vectors(normals)
    tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(source, rows, normals)
    area = (np.pi * np.maximum(scale1, 1e-8) * np.maximum(scale2, 1e-8)).astype(np.float32, copy=False)
    return SurfaceElementMap(
        element_ids=np.arange(rows.size, dtype=np.int64),
        parent_gaussian_indices=np.asarray(source.gaussian_indices, dtype=np.int64)[rows],
        centers=centers,
        tangent1=tangent1.astype(np.float32, copy=False),
        tangent2=tangent2.astype(np.float32, copy=False),
        normals=normals.astype(np.float32, copy=False),
        scale1=scale1.astype(np.float32, copy=False),
        scale2=scale2.astype(np.float32, copy=False),
        opacity=np.asarray(source.opacity, dtype=np.float32)[rows],
        area=area,
        adjacency=(),
        metadata={
            "gaussian_ply": str(gaussian_ply),
            "clean_membership_ply": (
                None if clean_membership_ply is None else str(clean_membership_ply)
            ),
            "source_gaussian_count": int(source.xyz.shape[0]),
            "surface_element_count": int(rows.size),
            "min_opacity": float(min_opacity),
            "max_scale": None if float(max_scale) <= 0.0 else float(max_scale),
            "render_only": True,
        },
    )


def _render_size(camera, max_long_edge: int, render_width: int, render_height: int) -> tuple[int, int]:
    width = int(render_width) if int(render_width) > 0 else int(camera.width)
    height = int(render_height) if int(render_height) > 0 else int(camera.height)
    if int(max_long_edge) > 0:
        scale = float(max_long_edge) / max(float(max(width, height)), 1.0)
        if scale < 1.0:
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
    return width, height


def _heatmap(values: np.ndarray, valid: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometry label visualization") from exc
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if vmin is None:
        vmin = float(np.percentile(values[valid], 5.0)) if np.any(valid) else 0.0
    if vmax is None:
        vmax = float(np.percentile(values[valid], 95.0)) if np.any(valid) else 1.0
    if float(vmax) <= float(vmin):
        vmax = float(vmin) + 1.0
    normalized = np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    normalized[valid] = np.asarray(scaled[valid] * 255.0, dtype=np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _label_panel(image: np.ndarray, title: str) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometry label visualization") from exc
    image = np.asarray(image, dtype=np.uint8)
    h0, w0 = image.shape[:2]
    scale = max(1, int(np.ceil(320.0 / max(float(w0), 1.0))), int(np.ceil(180.0 / max(float(h0), 1.0))))
    if scale > 1:
        image = cv2.resize(image, (w0 * scale, h0 * scale), interpolation=cv2.INTER_NEAREST)
    h, w = image.shape[:2]
    canvas = np.zeros((h + 34, w, 3), dtype=np.uint8)
    canvas[34:] = image
    cv2.putText(canvas, title, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _write_visualization(output_path: Path, depth: np.ndarray, normal_cam: np.ndarray, alpha: np.ndarray, valid: np.ndarray) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometry label visualization") from exc
    valid = np.asarray(valid, dtype=bool)
    normal_rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    mapped = np.clip((np.asarray(normal_cam, dtype=np.float32) + 1.0) * 0.5 * 255.0, 0.0, 255.0).astype(np.uint8)
    normal_rgb[valid] = mapped[valid]
    alpha_rgb = _heatmap(alpha, valid, 0.0, 1.0)
    valid_rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    valid_rgb[valid] = 255
    panel = np.concatenate(
        [
            _label_panel(_heatmap(depth, valid), "depth: blue near, red far"),
            _label_panel(normal_rgb, "camera normal: xyz mapped to rgb"),
            _label_panel(alpha_rgb, "alpha/confidence"),
            _label_panel(valid_rgb, "valid mask"),
        ],
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--clean_membership_ply", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--surface_min_opacity", type=float, default=0.05)
    parser.add_argument("--surface_max_scale", type=float, default=0.0)
    parser.add_argument("--render_width", type=int, default=0)
    parser.add_argument("--render_height", type=int, default=0)
    parser.add_argument("--max_long_edge", type=int, default=0)
    parser.add_argument("--sigma_scale", type=float, default=2.0)
    parser.add_argument("--max_radius_px", type=float, default=8.0)
    parser.add_argument("--depth_epsilon", type=float, default=0.05)
    parser.add_argument("--min_support", type=int, default=1)
    parser.add_argument("--max_depth_m", type=float, default=0.0)
    parser.add_argument("--visualize", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    manifest.validate(verify_checksums=False)
    records = list(manifest.records)
    if int(args.shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    records = records[int(args.shard_index) :: int(args.shard_count)]
    if int(args.max_records) > 0:
        records = records[: int(args.max_records)]
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    width, height = _render_size(camera, int(args.max_long_edge), int(args.render_width), int(args.render_height))
    surface = _render_surface_from_ply(
        Path(args.gaussian_ply),
        clean_membership_ply=(
            Path(args.clean_membership_ply) if args.clean_membership_ply else None
        ),
        max_gaussians=int(args.max_gaussians),
        min_opacity=float(args.surface_min_opacity),
        max_scale=float(args.surface_max_scale),
    )
    cfg = TwoDgsGeometryRenderConfig(
        sigma_scale=float(args.sigma_scale),
        max_radius_px=float(args.max_radius_px),
        min_opacity=float(args.surface_min_opacity),
        depth_epsilon=float(args.depth_epsilon),
        min_support=int(args.min_support),
        max_depth_m=None if float(args.max_depth_m) <= 0.0 else float(args.max_depth_m),
    )
    output_dir = Path(args.output_dir)
    label_dir = output_dir / "geometry_npz"
    vis_dir = output_dir / "visualizations"
    label_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    valid_ratios = []
    alpha_means = []
    depth_medians = []
    for record in records:
        pose = pose_by_image.get(record.image_id)
        if pose is None:
            continue
        label = render_2dgs_geometry_label(surface, pose.pose_w2c, camera, width, height, cfg)
        key = safe_image_id_key(record.image_id)
        label_path = label_dir / f"{key}.npz"
        np.savez_compressed(
            label_path,
            image_id=np.asarray(record.image_id),
            token_path=np.asarray(str(record.token_path)),
            depth=label.depth.astype(np.float32, copy=False),
            normal_cam=label.normal_cam.astype(np.float32, copy=False),
            alpha=label.alpha.astype(np.float32, copy=False),
            valid=label.valid.astype(bool, copy=False),
            support_count=label.support_count.astype(np.int32, copy=False),
        )
        valid_ratio = float(np.mean(label.valid))
        alpha_mean = float(np.mean(label.alpha[label.valid])) if np.any(label.valid) else 0.0
        depth_median = float(np.median(label.depth[label.valid])) if np.any(label.valid) else 0.0
        valid_ratios.append(valid_ratio)
        alpha_means.append(alpha_mean)
        depth_medians.append(depth_median)
        if len(output_records) < int(args.visualize):
            _write_visualization(vis_dir / f"{len(output_records):03d}_{key}_geometry_label.png", label.depth, label.normal_cam, label.alpha, label.valid)
        output_records.append(
            {
                "image_id": record.image_id,
                "token_path": str(record.token_path),
                "geometry_path": str(label_path),
                "split": record.split,
                "scene": record.scene,
                "resolution": [int(width), int(height)],
                "valid_ratio": valid_ratio,
                "mean_alpha": alpha_mean,
                "median_depth_m": depth_median,
            }
        )
    payload = {
        "stage": "2dgs_highres_geometry_labels",
        "records": output_records,
        "inputs": {
            "token_manifest": str(args.token_manifest),
            "pose_file": str(args.pose_file),
            "gaussian_ply": str(args.gaussian_ply),
            "camera_source": camera_source,
        },
        "camera": {
            "width": int(camera.width),
            "height": int(camera.height),
            "model_id": int(camera.model_id),
            "params": [float(value) for value in camera.params],
        },
        "render": {"width": int(width), "height": int(height), **cfg.to_dict()},
        "shard": {
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
        },
        "surface": dict(surface.metadata or {}),
        "summary": {
            "record_count": int(len(output_records)),
            "mean_valid_ratio": 0.0 if not valid_ratios else float(np.mean(valid_ratios)),
            "median_valid_ratio": 0.0 if not valid_ratios else float(np.median(valid_ratios)),
            "mean_alpha": 0.0 if not alpha_means else float(np.mean(alpha_means)),
            "median_depth_m": 0.0 if not depth_medians else float(np.median(depth_medians)),
        },
        "outputs": {
            "manifest": str(output_dir / "geometry_manifest.json"),
            "summary": str(output_dir / "geometry_label_summary.json"),
            "visualizations": str(vis_dir),
        },
    }
    (output_dir / "geometry_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "geometry_label_summary.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
