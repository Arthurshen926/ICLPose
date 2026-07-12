"""Build a dense, image-referenced ALIKE query-context detector cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
    DetectedImageFeatures,
    SPATIAL_DETECTION_SELECTION_VERSION,
)
from feature_extract.vfm.tokens import TokenBankManifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--candidate_top_k", type=int, default=8192)
    parser.add_argument("--nms_radius_px", type=float, default=4.0)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--cache_dtype", default="float16", choices=("float16", "float32"))
    return parser.parse_args(argv)


def _alike_checkpoint(matcha_repo: Path, model_name: str) -> Path:
    suffix = str(model_name).split("-")[-1]
    path = Path(matcha_repo) / "third_party" / "alike" / "models" / f"alike-{suffix}.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.top_k) <= 0 or int(args.candidate_top_k) < int(args.top_k):
        raise ValueError("candidate_top_k must be at least top_k > 0")
    start = time.time()
    manifest_path = Path(args.query_manifest)
    records = TokenBankManifest.from_json(manifest_path).records
    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    devices = tuple(value.strip() for value in str(args.devices).split(",") if value.strip())
    if not devices:
        raise ValueError("at least one device is required")
    partitions = [list(range(index, len(records), len(devices))) for index in range(len(devices))]

    def worker(worker_index: int):
        extractor = AlikeDenseObservationExtractor(
            device=devices[worker_index],
            matcha_repo=Path(args.matcha_repo),
            model_name=str(args.alike_model_name),
        )
        output: list[tuple[int, DetectedImageFeatures]] = []
        for record_index in partitions[worker_index]:
            record = records[int(record_index)]
            image = images_by_name.get(str(record.image_id))
            if image is None:
                raise KeyError(f"query image missing from COLMAP model: {record.image_id}")
            camera = cameras[int(image.camera_id)]
            detected = extractor.detect(
                Path(args.image_root) / str(record.image_id),
                image_width=int(camera.width),
                image_height=int(camera.height),
                top_k=int(args.top_k),
                candidate_top_k=int(args.candidate_top_k),
                nms_radius_px=float(args.nms_radius_px),
                grid_rows=int(args.grid_rows),
                grid_cols=int(args.grid_cols),
                min_score=args.min_score,
                sub_pixel=True,
            )
            output.append((int(record_index), detected))
        return output, extractor.metadata

    if len(devices) == 1:
        worker_outputs = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            worker_outputs = list(executor.map(worker, range(len(devices))))
    detections: list[DetectedImageFeatures | None] = [None] * len(records)
    for values, _metadata in worker_outputs:
        for index, detected in values:
            detections[int(index)] = detected
    if any(value is None for value in detections):
        raise RuntimeError("detector workers did not return every query image")
    resolved = [value for value in detections if value is not None]
    offsets = np.concatenate(
        [
            np.zeros((1,), dtype=np.int64),
            np.cumsum([len(value.xy) for value in resolved], dtype=np.int64),
        ]
    )
    image_hashes = {
        str(record.image_id): str(detected.image_sha256)
        for record, detected in zip(records, resolved)
    }
    source_manifest = "\n".join(
        f"{image_id}:{image_hashes[image_id]}" for image_id in sorted(image_hashes)
    )
    checkpoint = _alike_checkpoint(Path(args.matcha_repo), str(args.alike_model_name))
    metadata = {
        "format": "alike_dense_query_context_cache_v1",
        "query_manifest": str(manifest_path),
        "query_manifest_sha256": file_sha256_short(manifest_path),
        "source_image_manifest_sha256": hashlib.sha256(source_manifest.encode("utf8")).hexdigest()[:16],
        "image_sha256_by_id": image_hashes,
        "query_count": int(len(records)),
        "point_count": int(offsets[-1]),
        "alike_model_name": str(args.alike_model_name),
        "alike_checkpoint_sha256": file_sha256_short(checkpoint),
        "top_k": int(args.top_k),
        "candidate_top_k": int(args.candidate_top_k),
        "nms_radius_px": float(args.nms_radius_px),
        "grid_rows": int(args.grid_rows),
        "grid_cols": int(args.grid_cols),
        "min_score": args.min_score,
        "detector_selection_version": SPATIAL_DETECTION_SELECTION_VERSION,
        "coordinate_convention": "sfm_pixel_endpoint_subpixel_v1",
        "cache_dtype": str(args.cache_dtype),
        "devices": list(devices),
        "partition_image_counts": [int(len(partition)) for partition in partitions],
        "extractor": dict(worker_outputs[0][1]),
    }
    cache_dtype = np.float16 if str(args.cache_dtype) == "float16" else np.float32
    output_path = Path(args.output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        image_ids=np.asarray([str(record.image_id) for record in records], dtype=np.str_),
        offsets=offsets,
        xy=np.concatenate([value.xy for value in resolved], axis=0).astype(np.float32),
        local_descriptors=np.concatenate([value.descriptors for value in resolved], axis=0).astype(cache_dtype),
        detector_scores=np.concatenate([value.scores for value in resolved], axis=0).astype(np.float32),
        detector_dispersions=np.concatenate([value.dispersions for value in resolved], axis=0).astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    summary = {
        "stage": "alike_dense_query_context_cache",
        "metadata": metadata,
        "runtime_seconds": float(time.time() - start),
        "outputs": {
            "cache": str(output_path),
            "summary": str(args.summary_json),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
