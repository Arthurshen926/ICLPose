#!/usr/bin/env python3
"""Evaluate feature mapability from COLMAP 2D-3D track observations."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import colmap_to_w2c, read_colmap_cameras, read_colmap_images  # noqa: E402
from feature_extract.localizability.mapability import observation_track_feature_variance  # noqa: E402
from feature_extract.localizability.reference_pose_scoring import find_feature_path  # noqa: E402


@dataclass
class ImageTrackObservations:
    image_id: int
    camera_id: int
    name: str
    xy: torch.Tensor
    point3d_ids: torch.Tensor


def read_colmap_points3d_xyz(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read point ids and XYZ coordinates from COLMAP points3D.bin."""
    point_ids: list[int] = []
    xyzs: list[tuple[float, float, float]] = []
    with Path(path).open("rb") as handle:
        num_points = struct.unpack("Q", handle.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("Q", handle.read(8))[0]
            xyz = struct.unpack("ddd", handle.read(24))
            handle.read(3)  # rgb
            handle.read(8)  # reprojection error
            track_len = struct.unpack("Q", handle.read(8))[0]
            handle.read(int(track_len) * 8)  # image_id:uint32, point2D_idx:uint32
            point_ids.append(int(point_id))
            xyzs.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return torch.as_tensor(point_ids, dtype=torch.long), torch.as_tensor(xyzs, dtype=torch.float32)


def read_colmap_image_track_observations(path: str | Path) -> dict[int, ImageTrackObservations]:
    """Read COLMAP images.bin including 2D keypoints and point3D ids."""
    images: dict[int, ImageTrackObservations] = {}
    with Path(path).open("rb") as handle:
        num_images = struct.unpack("Q", handle.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("I", handle.read(4))[0]
            handle.read(32)  # qvec
            handle.read(24)  # tvec
            camera_id = struct.unpack("I", handle.read(4))[0]
            name_bytes = b""
            while True:
                char = handle.read(1)
                if char == b"\x00":
                    break
                name_bytes += char
            name = name_bytes.decode()
            num_points = struct.unpack("Q", handle.read(8))[0]
            xy = torch.empty((num_points, 2), dtype=torch.float32)
            point_ids = torch.empty((num_points,), dtype=torch.long)
            for idx in range(num_points):
                x, y, point3d_id = struct.unpack("ddq", handle.read(24))
                xy[idx, 0] = float(x)
                xy[idx, 1] = float(y)
                point_ids[idx] = int(point3d_id)
            images[image_id] = ImageTrackObservations(
                image_id=int(image_id),
                camera_id=int(camera_id),
                name=name,
                xy=xy,
                point3d_ids=point_ids,
            )
    return images


def sample_feature_at_pixels(
    feature: torch.Tensor,
    xy: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Nearest-neighbor sample a CxHxW dense feature at original-image pixels."""
    if feature.ndim != 3:
        raise ValueError("feature must have shape (C,H,W)")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N,2)")
    channels, height, width = feature.shape
    if int(image_width) <= 1 or int(image_height) <= 1:
        raise ValueError("image_width and image_height must be > 1")
    x = xy[:, 0].float().clamp(0, float(image_width - 1))
    y = xy[:, 1].float().clamp(0, float(image_height - 1))
    fx = torch.round(x / float(image_width - 1) * float(width - 1)).long().clamp(0, width - 1)
    fy = torch.round(y / float(image_height - 1) * float(height - 1)).long().clamp(0, height - 1)
    return feature.float().reshape(channels, height, width)[:, fy, fx].transpose(0, 1).contiguous()


def project_points_to_image(
    xyz_world: torch.Tensor,
    w2c: torch.Tensor,
    *,
    camera_model: str,
    camera_params: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points into one COLMAP camera with basic distortion support."""
    if xyz_world.ndim != 2 or xyz_world.shape[1] != 3:
        raise ValueError("xyz_world must have shape (N,3)")
    if w2c.shape != (4, 4):
        raise ValueError("w2c must have shape (4,4)")
    xyz_h = torch.cat([xyz_world.float(), torch.ones((xyz_world.shape[0], 1), dtype=torch.float32)], dim=1)
    cam = (w2c.float() @ xyz_h.t()).t()[:, :3]
    z = cam[:, 2]
    xn = cam[:, 0] / z.clamp_min(1.0e-6)
    yn = cam[:, 1] / z.clamp_min(1.0e-6)
    params = camera_params.float().view(-1)
    model = str(camera_model).upper()
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        x = fx * xn + cx
        y = fy * yn + cy
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        if model == "SIMPLE_RADIAL" and params.numel() >= 4:
            r2 = xn * xn + yn * yn
            radial = 1.0 + params[3] * r2
            xn = xn * radial
            yn = yn * radial
        elif model == "RADIAL" and params.numel() >= 5:
            r2 = xn * xn + yn * yn
            radial = 1.0 + params[3] * r2 + params[4] * r2 * r2
            xn = xn * radial
            yn = yn * radial
        x = f * xn + cx
        y = f * yn + cy
    elif model == "OPENCV":
        fx, fy, cx, cy = params[:4]
        x = fx * xn + cx
        y = fy * yn + cy
    else:
        raise ValueError(f"Unsupported camera model for projection: {camera_model}")
    valid = (z > 1.0e-6) & (x >= 0) & (y >= 0) & (x <= float(image_width - 1)) & (y <= float(image_height - 1))
    return torch.stack([x, y], dim=1), valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--feature-subdir", default="fine_geo")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--max-points-per-image", type=int, default=512)
    parser.add_argument("--min-track-observations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    colmap_dir = Path(args.colmap_dir)
    cameras = read_colmap_cameras(str(colmap_dir / "cameras.bin"))
    observations = read_colmap_image_track_observations(colmap_dir / "images.bin")
    images_meta = read_colmap_images(str(colmap_dir / "images.bin"))
    point_ids, point_xyz = read_colmap_points3d_xyz(colmap_dir / "points3D.bin")
    use_projection_fallback = all(obs.xy.shape[0] == 0 for obs in observations.values())
    all_features: list[torch.Tensor] = []
    all_track_ids: list[torch.Tensor] = []
    used_images = 0
    missing_features = 0
    for image_id, image_obs in sorted(observations.items()):
        if used_images >= int(args.max_images):
            break
        feature_path = find_feature_path(args.feature_root, image_id, subdir=args.feature_subdir)
        if feature_path is None:
            missing_features += 1
            continue
        try:
            feature = torch.load(feature_path, map_location="cpu", weights_only=True)
        except TypeError:
            feature = torch.load(feature_path, map_location="cpu")
        if use_projection_fallback:
            meta = images_meta[int(image_id)]
            w2c = torch.as_tensor(colmap_to_w2c(meta.qvec, meta.tvec), dtype=torch.float32)
            camera = cameras[int(image_obs.camera_id)]
            xy_all, valid = project_points_to_image(
                point_xyz,
                w2c,
                camera_model=str(camera.model),
                camera_params=torch.as_tensor(camera.params, dtype=torch.float32),
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
            if not valid.any():
                continue
            xy = xy_all[valid]
            obs_point_ids = point_ids[valid]
        else:
            valid = image_obs.point3d_ids >= 0
            if not valid.any():
                continue
            xy = image_obs.xy[valid]
            obs_point_ids = image_obs.point3d_ids[valid]
        if xy.shape[0] > int(args.max_points_per_image):
            order = torch.linspace(0, xy.shape[0] - 1, steps=int(args.max_points_per_image)).round().long()
            xy = xy[order]
            obs_point_ids = obs_point_ids[order]
        camera = cameras[int(image_obs.camera_id)]
        sampled = sample_feature_at_pixels(
            feature,
            xy,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        all_features.append(sampled)
        all_track_ids.append(obs_point_ids)
        used_images += 1

    if all_features:
        features = torch.cat(all_features, dim=0)
        track_ids = torch.cat(all_track_ids, dim=0)
        unique_ids, counts = torch.unique(track_ids, return_counts=True)
        keep_ids = unique_ids[counts >= int(args.min_track_observations)]
        keep = torch.isin(track_ids, keep_ids)
        variance, stats = observation_track_feature_variance(features, track_ids, valid_mask=keep)
        feature_dim = int(features.shape[1])
    else:
        variance = torch.tensor(0.0)
        stats = {"num_tracks": torch.tensor(0.0), "num_observations": torch.tensor(0.0)}
        feature_dim = 0
        keep = torch.zeros(0, dtype=torch.bool)
    summary = {
        "colmap_dir": str(colmap_dir),
        "feature_root": str(args.feature_root),
        "feature_subdir": str(args.feature_subdir),
        "used_images": int(used_images),
        "missing_features": int(missing_features),
        "projection_fallback": bool(use_projection_fallback),
        "feature_dim": feature_dim,
        "storage_dim": feature_dim,
        "num_observations": int(keep.sum().item()) if all_features else 0,
        "num_tracks": float(stats["num_tracks"].item()),
        "track_variance": float(variance.item()),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
