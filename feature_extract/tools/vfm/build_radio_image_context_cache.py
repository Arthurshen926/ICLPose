"""Build pose-free multiscale RADIO-final context descriptors per real image."""

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
import torch.nn.functional as F
from PIL import Image

from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary


ARTIFACT_FORMAT = "radio_image_multiscale_context_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument(
        "--radio_checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _normalize_last(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), p=2, dim=-1, eps=1e-8)


def multiscale_context_descriptors(
    summary: torch.Tensor,
    local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized summary, global, 2x2, and 4x4 descriptors."""

    if summary.ndim != 2 or local.ndim != 4 or len(summary) != len(local):
        raise ValueError("RADIO summary/local batch shapes differ")
    global_local = _normalize_last(local.mean(dim=(2, 3)))
    grid2 = F.adaptive_avg_pool2d(local.float(), (2, 2)).permute(0, 2, 3, 1)
    grid4 = F.adaptive_avg_pool2d(local.float(), (4, 4)).permute(0, 2, 3, 1)
    return (
        _normalize_last(summary),
        global_local,
        _normalize_last(grid2).reshape(len(local), 4, local.shape[1]),
        _normalize_last(grid4).reshape(len(local), 16, local.shape[1]),
    )


def _load_rgb_batch(image_root: Path, image_ids: Sequence[str]) -> torch.Tensor:
    tensors = []
    shape = None
    for image_id in image_ids:
        path = Path(image_root) / str(image_id)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        if shape is None:
            shape = tuple(tensor.shape)
        if tuple(tensor.shape) != shape:
            raise ValueError("image-context batching requires a common image shape")
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    devices = tuple(value.strip() for value in str(args.devices).split(",") if value.strip())
    if not devices:
        raise ValueError("at least one device is required")
    model_dir = Path(args.colmap_model_dir)
    images_path = model_dir / "images.bin"
    images = read_colmap_images_binary(images_path)
    image_ids = np.asarray(
        sorted(str(image.image_name) for image in images.values()), dtype=np.str_
    )
    image_root = Path(args.image_root)
    missing = [value for value in image_ids.tolist() if not (image_root / value).exists()]
    if missing:
        raise FileNotFoundError(f"missing context RGB images: {missing[:10]}")
    checkpoint_path = Path(args.radio_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    count = len(image_ids)
    summaries = np.empty((count, 2560), dtype=np.float16)
    global_local = np.empty((count, 1280), dtype=np.float16)
    grid2 = np.empty((count, 4, 1280), dtype=np.float16)
    grid4 = np.empty((count, 16, 1280), dtype=np.float16)
    started = time.time()

    def worker(worker_index: int) -> dict[str, object]:
        device = devices[worker_index]
        extractor = RADIOFeatureExtractor(
            version=str(checkpoint_path.resolve()),
            device=device,
            radio_repo=str(args.radio_repo),
        )
        rows = np.arange(worker_index, count, len(devices), dtype=np.int64)
        completed = 0
        for start in range(0, len(rows), int(args.batch_size)):
            batch_rows = rows[start : start + int(args.batch_size)]
            batch_ids = image_ids[batch_rows].tolist()
            batch = _load_rgb_batch(image_root, batch_ids)
            output = extractor.extract_batch(batch)
            values = multiscale_context_descriptors(
                output["summary"], output["local"]
            )
            for target, value in zip(
                (summaries, global_local, grid2, grid4), values
            ):
                target[batch_rows] = value.cpu().numpy().astype(np.float16)
            completed += len(batch_rows)
            if completed % 32 == 0 or completed == len(rows):
                print(
                    json.dumps(
                        {
                            "stage": "radio_image_context",
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
    for name, values in (
        ("summary", summaries),
        ("global_local", global_local),
        ("grid2", grid2),
        ("grid4", grid4),
    ):
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"RADIO context extraction left invalid {name}")
    manifest = "\n".join(
        f"{value}:{file_sha256_short(image_root / value)}"
        for value in image_ids.tolist()
    )
    metadata = {
        "format": ARTIFACT_FORMAT,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "image_count": int(count),
        "radio_version": str(args.radio_version),
        "radio_checkpoint_sha256": file_sha256_short(checkpoint_path),
        "radio_model_load_spec": "explicit_checkpoint_path_v1",
        "colmap_images_sha256": file_sha256_short(images_path),
        "image_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()[:16],
        "descriptor_semantics": {
            "summary": "l2_normalized_radio_final_summary",
            "global_local": "l2_normalized_mean_radio_final_patch_tokens",
            "grid2": "per_cell_l2_normalized_radio_final_2x2_pool",
            "grid4": "per_cell_l2_normalized_radio_final_4x4_pool",
        },
        "devices": list(devices),
        "batch_size_per_device": int(args.batch_size),
        "workers": workers,
        "elapsed_seconds": float(time.time() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_ids=image_ids,
        summary_descriptors=summaries,
        global_local_descriptors=global_local,
        grid2_descriptors=grid2,
        grid4_descriptors=grid4,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({"output": str(output_path), "metadata": metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
