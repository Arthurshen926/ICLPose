"""Export per-query calibration only, excluding COLMAP poses and tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model_dir", required=True)
    parser.add_argument("--query_list", required=True)
    parser.add_argument("--target_width", type=int, required=True)
    parser.add_argument("--target_height", type=int, required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args(argv)


def _query_ids(path: Path) -> list[str]:
    output = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        image_id = line.split()[0]
        if "/" not in image_id or not image_id.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue
        output.append(image_id)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    model_dir = Path(args.source_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    image_by_name = {image.image_name: image for image in images.values()}
    output: dict[str, object] = {}
    for image_id in _query_ids(Path(args.query_list)):
        image = image_by_name.get(image_id)
        if image is None:
            raise ValueError(f"query calibration is absent: {image_id}")
        camera = cameras[int(image.camera_id)]
        scale_x = float(args.target_width) / float(camera.width)
        scale_y = float(args.target_height) / float(camera.height)
        if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=1e-8):
            raise ValueError("query resize changes aspect ratio")
        parameters = np.asarray(camera.params, dtype=np.float64).copy()
        if int(camera.model_id) == 0:
            parameters[:3] *= np.asarray([scale_x, scale_x, scale_y])
        elif int(camera.model_id) == 1:
            parameters[:4] *= np.asarray(
                [scale_x, scale_y, scale_x, scale_y]
            )
        elif int(camera.model_id) in {2, 3}:
            parameters[:3] *= np.asarray([scale_x, scale_x, scale_y])
        else:
            raise ValueError(
                f"unsupported query calibration model ID: {camera.model_id}"
            )
        output[image_id] = {
            "model_id": int(camera.model_id),
            "width": int(args.target_width),
            "height": int(args.target_height),
            "params": parameters.tolist(),
        }
    parameter_matrix = np.asarray(
        [record["params"] for record in output.values()], dtype=np.float64
    )
    payload = {
        "format": "per_query_colmap_calibration_only_v1",
        "query_count": len(output),
        "cameras": output,
        "intrinsic_audit": {
            "parameter_min": np.min(parameter_matrix, axis=0).tolist(),
            "parameter_max": np.max(parameter_matrix, axis=0).tolist(),
            "parameter_std": np.std(parameter_matrix, axis=0).tolist(),
            "uses_per_query_calibration": True,
        },
        "production_contract": {
            "contains_camera_calibration": True,
            "contains_camera_pose": False,
            "contains_sfm_points": False,
            "contains_sfm_tracks": False,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["intrinsic_audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
