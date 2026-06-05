"""Build token-grid pseudo-depth labels from 2DGS surface geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file, safe_image_id_key
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap
from feature_extract.vfm.vfm_depth_head import (
    TokenDepthRasterConfig,
    rasterize_surface_token_depth,
    render_surface_token_depth_label,
)


def _heatmap(values: np.ndarray, valid: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for depth label visualization") from exc
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


def _mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = np.asarray(color, dtype=np.uint8)
    return rgb


def _coverage_reason_rgb(valid: np.ndarray, center_valid: np.ndarray, filled_by_splat: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    center_valid = np.asarray(center_valid, dtype=bool)
    filled_by_splat = np.asarray(filled_by_splat, dtype=bool)
    rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    rgb[valid & center_valid] = np.asarray([0, 220, 0], dtype=np.uint8)
    rgb[filled_by_splat] = np.asarray([255, 180, 0], dtype=np.uint8)
    rgb[~valid] = np.asarray([0, 0, 0], dtype=np.uint8)
    return rgb


def _label_panel(image: np.ndarray, title: str) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for depth label visualization") from exc
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


def _write_label_visualization(
    output_path: Path,
    depth: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    center_valid: np.ndarray,
    filled_by_splat: np.ndarray,
) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for depth label visualization") from exc
    valid = np.asarray(valid, dtype=bool)
    depth_panel = _label_panel(_heatmap(depth, valid), "target depth: blue near, red far")
    valid_panel = _label_panel(_mask_rgb(valid, (255, 255, 255)), "valid mask: white supervised")
    conf_panel = _label_panel(_heatmap(confidence, valid, 0.0, 1.0), "confidence: blue low, red high")
    reason_panel = _label_panel(
        _coverage_reason_rgb(valid, center_valid, filled_by_splat),
        "green=center orange=splat black=invalid",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate([depth_panel, valid_panel, conf_panel, reason_panel], axis=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--surface_elements_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--depth_epsilon", type=float, default=0.10)
    parser.add_argument("--min_points_per_token", type=int, default=1)
    parser.add_argument("--max_depth_m", type=float, default=0.0)
    parser.add_argument("--raster_mode", default="splat", choices=("splat", "center"))
    parser.add_argument("--splat_sigma_scale", type=float, default=2.0)
    parser.add_argument("--max_splat_radius_tokens", type=float, default=3.0)
    parser.add_argument("--min_opacity", type=float, default=0.0)
    parser.add_argument("--visualize", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    manifest.validate(verify_checksums=False)
    records = list(manifest.records)
    if int(args.max_records) > 0:
        records = records[: int(args.max_records)]
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    surface = SurfaceElementMap.load_npz(Path(args.surface_elements_npz))
    cfg = TokenDepthRasterConfig(
        depth_epsilon=float(args.depth_epsilon),
        min_points_per_token=int(args.min_points_per_token),
        max_depth_m=None if float(args.max_depth_m) <= 0.0 else float(args.max_depth_m),
    )
    output_dir = Path(args.output_dir)
    depth_dir = output_dir / "depth_npz"
    depth_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    output_records = []
    valid_counts = []
    valid_ratios = []
    center_valid_ratios = []
    filled_by_splat_ratios = []
    confidence_means = []
    for record in records:
        pose = pose_by_image.get(record.image_id)
        if pose is None:
            continue
        token = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_h, token_w = token.shape
        if args.raster_mode == "center":
            depth, valid = rasterize_surface_token_depth(surface, pose.pose_w2c, camera, token_h, token_w, cfg)
            confidence = valid.astype(np.float32)
            support_count = valid.astype(np.int32)
            center_valid = valid.copy()
            filled_by_splat = np.zeros_like(valid)
        else:
            label = render_surface_token_depth_label(
                surface,
                pose.pose_w2c,
                camera,
                token_h,
                token_w,
                cfg,
                sigma_scale=float(args.splat_sigma_scale),
                min_opacity=float(args.min_opacity),
                max_radius_tokens=float(args.max_splat_radius_tokens),
            )
            depth = label.depth
            valid = label.valid
            confidence = label.confidence
            support_count = label.support_count
            center_valid = label.center_valid
            filled_by_splat = label.filled_by_splat
        key = safe_image_id_key(record.image_id)
        depth_path = depth_dir / f"{key}.npz"
        np.savez_compressed(
            depth_path,
            image_id=np.asarray(record.image_id),
            depth=depth.astype(np.float32, copy=False),
            valid=valid.astype(bool, copy=False),
            confidence=confidence.astype(np.float32, copy=False),
            support_count=support_count.astype(np.int32, copy=False),
            center_valid=center_valid.astype(bool, copy=False),
            filled_by_splat=filled_by_splat.astype(bool, copy=False),
            token_path=np.asarray(str(record.token_path)),
        )
        valid_count = int(np.sum(valid))
        valid_ratio = float(valid_count / max(int(valid.size), 1))
        center_valid_ratio = float(np.sum(valid & center_valid) / max(int(valid.size), 1))
        filled_by_splat_ratio = float(np.sum(filled_by_splat) / max(int(valid.size), 1))
        confidence_mean = float(np.mean(confidence[valid])) if np.any(valid) else 0.0
        valid_counts.append(valid_count)
        valid_ratios.append(valid_ratio)
        center_valid_ratios.append(center_valid_ratio)
        filled_by_splat_ratios.append(filled_by_splat_ratio)
        confidence_means.append(confidence_mean)
        if len(output_records) < int(args.visualize):
            _write_label_visualization(
                visualization_dir / f"{len(output_records):03d}_{safe_image_id_key(record.image_id)}_depth_label.png",
                depth,
                valid,
                confidence,
                center_valid,
                filled_by_splat,
            )
        output_records.append(
            {
                "image_id": record.image_id,
                "token_path": str(record.token_path),
                "depth_path": str(depth_path),
                "split": record.split,
                "scene": record.scene,
                "token_shape": [int(token.shape[0]), int(token_h), int(token_w)],
                "valid_count": valid_count,
                "valid_ratio": valid_ratio,
                "center_valid_ratio": center_valid_ratio,
                "filled_by_splat_ratio": filled_by_splat_ratio,
                "mean_confidence": confidence_mean,
            }
        )
    payload = {
        "stage": "vfm_token_depth_labels",
        "records": output_records,
        "inputs": {
            "token_manifest": str(args.token_manifest),
            "pose_file": str(args.pose_file),
            "surface_elements_npz": str(args.surface_elements_npz),
            "camera_source": camera_source,
        },
        "camera": {
            "width": int(camera.width),
            "height": int(camera.height),
            "model_id": int(camera.model_id),
            "params": [float(value) for value in camera.params],
        },
        "config": {
            **cfg.to_dict(),
            "raster_mode": str(args.raster_mode),
            "splat_sigma_scale": float(args.splat_sigma_scale),
            "max_splat_radius_tokens": float(args.max_splat_radius_tokens),
            "min_opacity": float(args.min_opacity),
        },
        "summary": {
            "record_count": int(len(output_records)),
            "mean_valid_count": 0.0 if not valid_counts else float(np.mean(valid_counts)),
            "mean_valid_ratio": 0.0 if not valid_ratios else float(np.mean(valid_ratios)),
            "median_valid_ratio": 0.0 if not valid_ratios else float(np.median(valid_ratios)),
            "mean_center_valid_ratio": 0.0 if not center_valid_ratios else float(np.mean(center_valid_ratios)),
            "mean_filled_by_splat_ratio": 0.0 if not filled_by_splat_ratios else float(np.mean(filled_by_splat_ratios)),
            "mean_confidence": 0.0 if not confidence_means else float(np.mean(confidence_means)),
        },
        "outputs": {
            "manifest": str(output_dir / "depth_manifest.json"),
            "summary": str(output_dir / "depth_label_summary.json"),
            "visualizations": str(visualization_dir),
        },
    }
    manifest_path = output_dir / "depth_manifest.json"
    summary_path = output_dir / "depth_label_summary.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
