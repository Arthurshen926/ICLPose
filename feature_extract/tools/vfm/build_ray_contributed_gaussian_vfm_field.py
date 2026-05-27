"""Build a Gaussian VFM field by ray-contribution token aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMFeatureView,
    GaussianVFMRayContributionConfig,
    aggregate_ray_contributed_gaussian_vfm_features,
    load_gaussian_vfm_source_from_ply,
)
from feature_extract.vfm.tokens import TokenBankManifest


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


def _load_camera_by_image(model_dir: str) -> dict[str, ColmapCamera]:
    if not model_dir:
        return {}
    model_path = Path(model_dir)
    cameras = read_colmap_cameras_binary(model_path / "cameras.bin")
    images = read_colmap_images_binary(model_path / "images.bin")
    return {
        image.image_name: cameras[image.camera_id]
        for image in images.values()
        if image.camera_id in cameras
    }


def _load_feature(path: Path, layer_name: str) -> np.ndarray:
    with np.load(path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {path}")
        return np.asarray(data[layer_name], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dense token features onto visible Gaussians")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--radius_px", type=float, default=1.0)
    parser.add_argument("--depth_epsilon", type=float, default=0.02)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--opacity_threshold", type=float, default=0.0)
    parser.add_argument("--no_l2_normalize_observations", action="store_true")
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    parser.add_argument("--default_camera", default="2,1024,576,883,512,288,0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    camera_by_image = _load_camera_by_image(args.camera_model_dir)
    fallback_camera = _parse_default_camera(args.default_camera)
    views: list[GaussianVFMFeatureView] = []
    records = list(manifest.records)
    if args.max_views > 0 and args.view_selection == "uniform" and len(records) > args.max_views:
        indices = np.linspace(0, len(records) - 1, int(args.max_views), dtype=np.int64)
        records = [records[int(idx)] for idx in indices]
    for record in records:
        if args.max_views > 0 and args.view_selection == "prefix" and len(views) >= args.max_views:
            break
        if record.image_id not in pose_by_image:
            continue
        views.append(
            GaussianVFMFeatureView(
                image_id=record.image_id,
                feature_map=_load_feature(record.token_path, args.layer_name),
                pose_w2c=pose_by_image[record.image_id].pose_w2c,
                camera=camera_by_image.get(record.image_id, fallback_camera),
            )
        )
    if not views:
        raise ValueError("no reference views with both tokens and poses")
    config = GaussianVFMRayContributionConfig(
        radius_px=args.radius_px,
        depth_epsilon=args.depth_epsilon,
        min_samples=args.min_samples,
        opacity_threshold=args.opacity_threshold,
        l2_normalize_observations=not args.no_l2_normalize_observations,
        l2_normalize_features=not args.no_l2_normalize_features,
    )
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply), max_gaussians=args.max_gaussians)
    field = aggregate_ray_contributed_gaussian_vfm_features(source, views, config)
    field.save_npz(Path(args.output))
    summary = {
        "stage": "ray_contributed_gaussian_vfm_field",
        "source_gaussian_count": int(source.xyz.shape[0]),
        "feature_bearing_gaussian_count": int(len(field)),
        "coverage_fraction": 0.0 if source.xyz.shape[0] == 0 else float(len(field) / source.xyz.shape[0]),
        "feature_dim": int(field.feature_dim),
        "view_count": int(len(views)),
        "mean_samples": 0.0 if len(field) == 0 else float(np.mean(field.support_counts)),
        "mean_pixel_distance": 0.0 if len(field) == 0 else float(np.mean(field.mean_distances)),
        "config": config.to_dict(),
        "inputs": {
            "gaussian_ply": args.gaussian_ply,
            "reference_manifest": args.reference_manifest,
            "reference_pose_file": args.reference_pose_file,
            "camera_model_dir": args.camera_model_dir,
            "layer_name": args.layer_name,
            "view_selection": args.view_selection,
        },
        "outputs": {"field": str(args.output)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
