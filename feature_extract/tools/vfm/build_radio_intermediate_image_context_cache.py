"""Build pose-free per-image RADIO-intermediate spatial grids from real RGB."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
from feature_extract.tools.vfm.build_radio_image_context_cache import (
    _parse_spatial_grid_sizes,
    spatial_grid_descriptors,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    save_spatial_image_context_cache,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import image_root_manifest


ARTIFACT_FORMAT = "radio_intermediate_image_spatial_context_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--spatial_grid_sizes", default="16")
    parser.add_argument("--intermediate_index", type=int, default=-6)
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument(
        "--radio_checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument(
        "--cache_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_rgb_batch(image_root: Path, image_ids: Sequence[str]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    shape: tuple[int, ...] | None = None
    for image_id in image_ids:
        with Image.open(image_root / str(image_id)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        value = torch.from_numpy(rgb).permute(2, 0, 1)
        if shape is None:
            shape = tuple(value.shape)
        if tuple(value.shape) != shape:
            raise ValueError("RADIO intermediate image batching requires common RGB sizes")
        rows.append(value)
    return torch.stack(rows, dim=0)


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must contain unique non-empty device names")
    if any(torch.device(device).type == "cuda" and not torch.cuda.is_available() for device in devices):
        raise RuntimeError("a requested CUDA device is unavailable")
    return devices


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    grid_sizes = _parse_spatial_grid_sizes(args.spatial_grid_sizes)
    output = Path(args.output)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output}")
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images_path = model_dir / "images.bin"
    images = read_colmap_images_binary(images_path)
    image_records = sorted(images.values(), key=lambda image: str(image.image_name))
    image_ids = np.asarray([str(image.image_name) for image in image_records], dtype=np.str_)
    image_sizes = np.asarray(
        [
            [
                cameras[int(image.camera_id)].width,
                cameras[int(image.camera_id)].height,
            ]
            for image in image_records
        ],
        dtype=np.int64,
    )
    image_root = Path(args.image_root)
    missing = [image_id for image_id in image_ids.tolist() if not (image_root / image_id).is_file()]
    if missing:
        raise FileNotFoundError(f"missing real RGB images: {missing[:10]}")
    checkpoint = Path(args.radio_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    devices = _parse_devices(args.devices)
    grids = {
        int(size): np.empty((len(image_ids), int(size) ** 2, 1280), dtype=np.float16)
        for size in grid_sizes
    }
    started = time.time()

    def worker(worker_index: int) -> dict[str, object]:
        device = devices[worker_index]
        extractor = RADIOFeatureExtractor(
            version=str(checkpoint.resolve()),
            device=device,
            radio_repo=str(args.radio_repo),
        )
        rows = np.arange(worker_index, len(image_ids), len(devices), dtype=np.int64)
        completed = 0
        for start in range(0, len(rows), int(args.batch_size)):
            batch_rows = rows[start : start + int(args.batch_size)]
            batch = _load_rgb_batch(image_root, image_ids[batch_rows].tolist())
            intermediate = extractor.extract_intermediate_batch(
                batch,
                intermediate_index=int(args.intermediate_index),
                norm_intermediates=True,
                aggregation="sparse",
            )
            if int(intermediate.shape[1]) != 1280:
                raise ValueError("RADIO intermediate descriptor dimension drifted")
            for size, target in grids.items():
                target[batch_rows] = (
                    spatial_grid_descriptors(intermediate, grid_size=int(size))
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            completed += len(batch_rows)
            if completed % 32 == 0 or completed == len(rows):
                print(
                    json.dumps(
                        {
                            "stage": "radio_intermediate_image_context",
                            "device": device,
                            "completed": completed,
                            "assigned": int(len(rows)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return {"device": device, "image_count": int(len(rows))}

    if len(devices) == 1:
        workers = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            workers = list(executor.map(worker, range(len(devices))))
    for size, values in grids.items():
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"RADIO intermediate grid{size} extraction is incomplete")
    # Keep the former full-file hash only as a diagnostic. All production
    # consumers compare the canonical source contract below.
    legacy_source_manifest = "\n".join(
        f"{image_id}:{file_sha256_short(image_root / image_id)}"
        for image_id in image_ids.tolist()
    )
    image_source_contract = image_root_manifest(image_root, image_ids.tolist())
    metadata = {
        "format": ARTIFACT_FORMAT,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "image_count": int(len(image_ids)),
        "spatial_grid_sizes": [int(size) for size in grid_sizes],
        "radio_version": str(args.radio_version),
        "radio_checkpoint_sha256": file_sha256_short(checkpoint),
        "radio_model_load_spec": "explicit_checkpoint_path_v1",
        "intermediate_index": int(args.intermediate_index),
        "norm_intermediates": True,
        "aggregation": "sparse",
        "coordinate_convention": "colmap_pixel_relative_to_adaptive_grid_of_processed_rgb_v1",
        "colmap_images_sha256": file_sha256_short(images_path),
        "source_image_manifest_sha256": image_source_contract[
            "sampled_content_manifest_sha256"
        ],
        "legacy_source_image_manifest_sha256": hashlib.sha256(
            legacy_source_manifest.encode()
        ).hexdigest()[:16],
        "image_source_contract": image_source_contract,
        "cache_dtype": str(args.cache_dtype),
        "devices": list(devices),
        "batch_size_per_device": int(args.batch_size),
        "workers": workers,
        "elapsed_seconds": float(time.time() - started),
    }
    cache = SpatialImageContextCache(
        image_ids=image_ids,
        image_sizes=image_sizes,
        grids=grids,
        metadata=metadata,
    )
    save_spatial_image_context_cache(cache, output, cache_dtype=str(args.cache_dtype))
    print(
        json.dumps(
            {"output": str(output), "metadata": metadata}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
