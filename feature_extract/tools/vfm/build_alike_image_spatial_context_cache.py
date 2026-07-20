"""Build pose-free ALIKE dense-FPN spatial grids from real RGB images."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import cv2
import numpy as np
import torch

from feature_extract.tools.vfm.build_radio_image_context_cache import (
    _parse_spatial_grid_sizes,
    spatial_grid_descriptors,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    save_spatial_image_context_cache,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import image_root_manifest


ARTIFACT_FORMAT = "alike_image_spatial_context_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--spatial_grid_sizes", default="16,32")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=1,
        help="Batch same-resolution real RGB images for ALIKE dense-map inference.",
    )
    parser.add_argument(
        "--cache_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must contain unique non-empty device names")
    if any(torch.device(device).type == "cuda" and not torch.cuda.is_available() for device in devices):
        raise RuntimeError("a requested CUDA device is unavailable")
    return devices


def _descriptor_dim(matcha_repo: Path, model_name: str) -> int:
    repo = Path(matcha_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from third_party.alike.alike import configs  # type: ignore

    config = configs.get(str(model_name))
    if config is None:
        raise ValueError(f"unknown ALIKE model {model_name!r}")
    return int(config["dim"])


def _load_alike_input(path: Path, *, width: int, height: int) -> tuple[torch.Tensor, str]:
    payload = Path(path).read_bytes()
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode real RGB image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (int(height), int(width)):
        rgb = cv2.resize(rgb, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    value = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return value, hashlib.sha256(payload).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    grid_sizes = _parse_spatial_grid_sizes(args.spatial_grid_sizes)
    if int(args.batch_size_per_device) <= 0:
        raise ValueError("batch_size_per_device must be positive")
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
    devices = _parse_devices(args.devices)
    descriptor_dim = _descriptor_dim(Path(args.matcha_repo), str(args.alike_model_name))
    grids = {
        int(size): np.empty(
            (len(image_ids), int(size) ** 2, descriptor_dim), dtype=np.float16
        )
        for size in grid_sizes
    }
    started = time.time()

    def worker(worker_index: int) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        device = devices[worker_index]
        extractor = AlikeDenseObservationExtractor(
            device=device,
            matcha_repo=Path(args.matcha_repo),
            model_name=str(args.alike_model_name),
        )
        rows = np.arange(worker_index, len(image_ids), len(devices), dtype=np.int64)
        image_hashes: dict[str, str] = {}
        completed = 0
        for offset in range(0, len(rows), int(args.batch_size_per_device)):
            batch_rows = rows[offset : offset + int(args.batch_size_per_device)]
            # Keep the exporter usable for datasets with mixed image sizes:
            # only equally shaped images share a forward pass.
            batches: dict[tuple[int, int], list[tuple[int, torch.Tensor, str]]] = {}
            for row in batch_rows.tolist():
                width, height = image_sizes[int(row)].tolist()
                input_tensor, image_hash = _load_alike_input(
                    image_root / str(image_ids[int(row)]),
                    width=int(width),
                    height=int(height),
                )
                batches.setdefault((int(height), int(width)), []).append(
                    (int(row), input_tensor, image_hash)
                )
            for entries in batches.values():
                entry_rows = np.asarray([row for row, _image, _hash in entries], dtype=np.int64)
                image_batch = torch.stack([image for _row, image, _hash in entries], dim=0).to(
                    extractor.device
                )
                with torch.no_grad():
                    descriptor_map, _score_map = extractor.model.extract_dense_map(image_batch)
                if int(descriptor_map.shape[1]) != descriptor_dim:
                    raise ValueError("ALIKE dense descriptor dimension drifted")
                for size, target in grids.items():
                    target[entry_rows] = (
                        spatial_grid_descriptors(descriptor_map, grid_size=int(size))
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                image_hashes.update(
                    {
                        str(image_ids[row]): image_hash
                        for row, _image, image_hash in entries
                    }
                )
            completed += len(batch_rows)
            if completed % 32 == 0 or completed == len(rows):
                print(
                    json.dumps(
                        {
                            "stage": "alike_image_spatial_context",
                            "device": device,
                            "completed": completed,
                            "assigned": int(len(rows)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return (
            {"device": device, "image_count": int(len(rows))},
            image_hashes,
            extractor.metadata,
        )

    if len(devices) == 1:
        results = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(executor.map(worker, range(len(devices))))
    for size, values in grids.items():
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"ALIKE grid{size} extraction is incomplete")
    image_hashes = {key: value for _worker, hashes, _metadata in results for key, value in hashes.items()}
    if set(image_hashes) != set(image_ids.tolist()):
        raise RuntimeError("ALIKE image workers did not cover every image")
    extractor_metadata = dict(results[0][2])
    if any(metadata != extractor_metadata for _worker, _hashes, metadata in results[1:]):
        raise RuntimeError("ALIKE workers used inconsistent extractor metadata")
    # Preserve the historical per-file digest for diagnostics only. The shared
    # cache contract below is the authoritative cross-model source identity.
    legacy_source_manifest = "\n".join(
        f"{image_id}:{image_hashes[image_id]}" for image_id in sorted(image_hashes)
    )
    image_source_contract = image_root_manifest(image_root, image_ids.tolist())
    metadata = {
        "format": ARTIFACT_FORMAT,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "image_count": int(len(image_ids)),
        "spatial_grid_sizes": [int(size) for size in grid_sizes],
        "alike_model_name": str(args.alike_model_name),
        "alike_checkpoint_sha256": extractor_metadata["model_checkpoint_sha256"],
        "extractor": extractor_metadata,
        "coordinate_convention": "processed_rgb_resized_to_colmap_pixels_then_adaptive_fpn_grid_v1",
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
        "batch_size_per_device": int(args.batch_size_per_device),
        "workers": [worker for worker, _hashes, _metadata in results],
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
