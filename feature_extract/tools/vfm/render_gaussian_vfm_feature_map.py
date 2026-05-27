"""Render a dense feature map from a Gaussian VFM field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMField,
    GaussianVFMRenderConfig,
    render_gaussian_vfm_feature_map,
    render_gaussian_vfm_feature_map_gsplat,
)


def _parse_default_camera(text: str) -> ColmapCamera:
    values = [float(item) for item in text.split(",") if item.strip()]
    if len(values) < 6:
        raise ValueError("--default_camera must be 'model_id,width,height,param0,param1,...'")
    return ColmapCamera(
        camera_id=-1,
        model_id=int(values[0]),
        width=int(values[1]),
        height=int(values[2]),
        params=tuple(float(item) for item in values[3:]),
    )


def _load_camera(model_dir: str, image_id: str, fallback: ColmapCamera) -> ColmapCamera:
    if not model_dir:
        return fallback
    model_path = Path(model_dir)
    cameras = read_colmap_cameras_binary(model_path / "cameras.bin")
    images = read_colmap_images_binary(model_path / "images.bin")
    for image in images.values():
        if image.image_name == image_id and image.camera_id in cameras:
            return cameras[image.camera_id]
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Render dense VFM features from a Gaussian VFM field")
    parser.add_argument("--field", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883,512,288,0")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--radius_px", type=float, default=2.0)
    parser.add_argument("--depth_epsilon", type=float, default=0.02)
    parser.add_argument("--no_l2_normalize_pixels", action="store_true")
    parser.add_argument("--renderer", default="soft", choices=("soft", "gsplat"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel_chunk", type=int, default=32)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    field = GaussianVFMField.load_npz(Path(args.field))
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.pose_file))}
    if args.image_id not in pose_by_image:
        raise ValueError(f"image_id not found in pose file: {args.image_id}")
    fallback = _parse_default_camera(args.default_camera)
    camera = _load_camera(args.camera_model_dir, args.image_id, fallback)
    width = int(args.width) if args.width > 0 else int(camera.width)
    height = int(args.height) if args.height > 0 else int(camera.height)
    config = GaussianVFMRenderConfig(
        width=width,
        height=height,
        radius_px=args.radius_px,
        depth_epsilon=args.depth_epsilon,
        l2_normalize_pixels=not args.no_l2_normalize_pixels,
    )
    if args.renderer == "gsplat":
        result = render_gaussian_vfm_feature_map_gsplat(
            field,
            pose_w2c=pose_by_image[args.image_id].pose_w2c,
            camera=camera,
            config=config,
            device=args.device,
            channel_chunk=args.channel_chunk,
        )
    else:
        result = render_gaussian_vfm_feature_map(
            field,
            pose_w2c=pose_by_image[args.image_id].pose_w2c,
            camera=camera,
            config=config,
        )
    metadata = {
        "field": str(args.field),
        "pose_file": str(args.pose_file),
        "image_id": args.image_id,
        "camera_model_dir": args.camera_model_dir,
        "config": config.to_dict(),
        "renderer": args.renderer,
        "device": args.device,
    }
    result.save_npz(Path(args.output), metadata=metadata)
    visible_count = int(np.sum(result.visibility_mask))
    summary = {
        "stage": "gaussian_vfm_feature_render",
        "image_id": args.image_id,
        "feature_dim": int(result.feature_map.shape[0]),
        "height": int(result.feature_map.shape[1]),
        "width": int(result.feature_map.shape[2]),
        "visible_pixel_count": visible_count,
        "visible_fraction": float(visible_count / max(width * height, 1)),
        "mean_weight_sum": 0.0
        if visible_count == 0
        else float(np.mean(result.weight_sum[result.visibility_mask])),
        "config": config.to_dict(),
        "renderer": args.renderer,
        "device": args.device,
        "outputs": {"render": str(args.output)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
