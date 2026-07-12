"""Build a complete aspect-preserving canonical-resolution COLMAP model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
    scale_colmap_camera,
    write_colmap_cameras_binary,
    write_colmap_images_binary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--target_width", type=int, required=True)
    parser.add_argument("--target_height", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.input_model_dir)
    output = Path(args.output_model_dir)
    target_width = int(args.target_width)
    target_height = int(args.target_height)
    source_paths = {
        name: source / name
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    expected = {
        "format": "scaled_colmap_model_v1",
        "source_model_dir": str(source),
        "source_cameras_sha256": file_sha256_short(source_paths["cameras.bin"]),
        "source_images_sha256": file_sha256_short(source_paths["images.bin"]),
        "source_points3d_sha256": file_sha256_short(source_paths["points3D.bin"]),
        "target_width": target_width,
        "target_height": target_height,
        "coordinate_scaling": "width_ratio_height_ratio_v1",
    }
    metadata_path = output / "scaled_model_manifest.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        output_hash_keys = {
            "output_cameras_sha256": output / "cameras.bin",
            "output_images_sha256": output / "images.bin",
            "output_points3d_sha256": output / "points3D.bin",
        }
        for key, path in output_hash_keys.items():
            actual = None if not path.exists() else file_sha256_short(path)
            if metadata.get(key) != actual:
                mismatches[key] = {
                    "expected": metadata.get(key),
                    "actual": actual,
                }
        if mismatches:
            raise ValueError(
                f"stale scaled COLMAP model: {json.dumps(mismatches, sort_keys=True)}"
            )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    cameras = read_colmap_cameras_binary(source_paths["cameras.bin"])
    images = read_colmap_images_binary(source_paths["images.bin"])
    scaled = {
        camera_id: scale_colmap_camera(
            camera,
            width=target_width,
            height=target_height,
        )
        for camera_id, camera in cameras.items()
    }
    scales = {
        camera_id: (
            float(scaled[camera_id].width) / float(camera.width),
            float(scaled[camera_id].height) / float(camera.height),
        )
        for camera_id, camera in cameras.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    write_colmap_cameras_binary(scaled, output / "cameras.bin")
    write_colmap_images_binary(
        images,
        output / "images.bin",
        xy_scale_by_camera_id=scales,
    )
    shutil.copy2(source_paths["points3D.bin"], output / "points3D.bin")
    metadata = {
        **expected,
        "camera_count": int(len(cameras)),
        "image_count": int(len(images)),
        "output_cameras_sha256": file_sha256_short(output / "cameras.bin"),
        "output_images_sha256": file_sha256_short(output / "images.bin"),
        "output_points3d_sha256": file_sha256_short(output / "points3D.bin"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
