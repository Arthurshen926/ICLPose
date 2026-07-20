"""Cache full-bank, target-free LoFTR correspondences for one query image.

This is deliberately not image retrieval: every image in the fixed mapping
support manifest is paired with the query exactly once, and no image-level
score, shortlist, submap, pose, candidate track, or target is accepted.  The
output can later provide independent pairwise evidence only at predeclared
landmark observation anchors.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _load_maplet_support_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    FROZEN_LOFTR_PAIR_CACHE_FORMAT,
    restore_pixel_centers_half_pixel,
    split_batched_loftr_matches,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import image_root_manifest


_LOFTR_RUNTIME_CACHE: dict[
    tuple[str, str, str, str, str], tuple[torch.nn.Module, dict[str, Any]]
] = {}
# Freeze this at module import. Long-running shard workers must report the code
# they actually loaded, not the contents of this path after a later edit.
_LOFTR_BUILDER_SOURCE_SHA256 = file_sha256_short(Path(__file__))


def kornia_pretrained_checkpoint_path(
    *, weights: str, hub_dir: Path, urls: Mapping[str, str]
) -> Path:
    """Return the exact torch-hub file Kornia will load for a preset name."""

    url = urls.get(str(weights))
    parsed = None if url is None else urlparse(str(url))
    path = "" if parsed is None else str(parsed.path)
    filename = Path(path).name if path and not path.endswith("/") else ""
    if not filename or Path(filename).name != filename:
        raise ValueError("Kornia LoFTR preset URL does not define a safe checkpoint filename")
    return Path(hub_dir).resolve() / "checkpoints" / filename


def _validate_declared_kornia_checkpoint(*, checkpoint: Path, weights: str) -> Path:
    """Bind provenance to the file actually consumed by Kornia's preset loader."""

    try:
        from kornia.feature.loftr.loftr import urls
    except ImportError as error:  # pragma: no cover - LoFTR itself imports Kornia.
        raise RuntimeError("LoFTR requires Kornia") from error
    expected = kornia_pretrained_checkpoint_path(
        weights=str(weights), hub_dir=Path(torch.hub.get_dir()), urls=urls
    ).resolve(strict=True)
    declared = Path(checkpoint).resolve(strict=True)
    if declared != expected:
        raise ValueError(
            "declared LoFTR checkpoint must be Kornia's exact pretrained cache file: "
            f"declared={declared}, expected={expected}"
        )
    return expected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--mapping-support-manifest", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--hloc-root", default="third_party/Hierarchical-Localization")
    parser.add_argument("--loftr-checkpoint", required=True)
    parser.add_argument("--loftr-weights", default="outdoor", choices=("outdoor", "indoor"))
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--resize-height", type=int, default=540)
    parser.add_argument("--pair-batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _safe_image_path(*, image_root: Path, image_id: str) -> Path:
    root = Path(image_root).resolve(strict=True)
    relative = Path(str(image_id))
    if not str(image_id) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"image id escapes image root: {image_id!r}")
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError(f"image id escapes image root: {image_id!r}")
    return path


def _load_mapping_support_manifest(
    *, path: Path, maplet_support_index: Path, query_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "maplet_support_image_manifest_v1"
        or not isinstance(metadata, Mapping)
        or not isinstance(records, list)
        or str(metadata.get("maplet_support_index_sha256", ""))
        != file_sha256_short(Path(maplet_support_index))
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or int(metadata.get("support_query_overlap_count", -1)) != 0
    ):
        raise ValueError("mapping support manifest violates the fixed full-bank contract")
    image_ids = np.asarray(
        [str(record.get("image_id", "")).strip() for record in records if isinstance(record, Mapping)],
        dtype=np.str_,
    )
    if (
        len(image_ids) == 0
        or len(image_ids) != len(records)
        or len(set(image_ids.tolist())) != len(image_ids)
        or np.any(image_ids == "")
        or str(query_id) in set(image_ids.tolist())
        or not np.array_equal(image_ids, np.sort(image_ids))
    ):
        raise ValueError("mapping support manifest image ids are invalid")
    _tracks, maplet_image_ids, _indices, _coverage, _metadata = _load_maplet_support_fields(
        Path(maplet_support_index)
    )
    if tuple(image_ids.tolist()) != tuple(sorted(maplet_image_ids)):
        raise ValueError("mapping support manifest does not contain exactly the current maplet images")
    return image_ids, dict(metadata)


def _load_gray_resized(
    *, image_root: Path, image_id: str, resize_width: int, resize_height: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    path = _safe_image_path(image_root=Path(image_root), image_id=str(image_id))
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"failed to read real grayscale image: {path}")
    source_height, source_width = (int(image.shape[0]), int(image.shape[1]))
    if source_width * int(resize_height) != source_height * int(resize_width):
        raise ValueError(
            f"{image_id}: LoFTR cache only permits aspect-preserving resize, "
            f"got {source_width}x{source_height} -> {resize_width}x{resize_height}"
        )
    resized = cv2.resize(
        image, (int(resize_width), int(resize_height)), interpolation=cv2.INTER_LINEAR
    )
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).to(dtype=torch.float32)
    return tensor[None, None] / 255.0, (source_width, source_height)


def _load_loftr_runtime(
    *, hloc_root: Path, weights: str, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    root = Path(hloc_root).resolve(strict=True)
    wrapper_path = root / "hloc" / "matchers" / "loftr.py"
    if not wrapper_path.is_file() or not Path(checkpoint).is_file():
        raise FileNotFoundError("LoFTR wrapper or checkpoint is absent")
    actual_loader_checkpoint = _validate_declared_kornia_checkpoint(
        checkpoint=Path(checkpoint), weights=str(weights)
    )
    checkpoint_hash = file_sha256_short(Path(checkpoint))
    wrapper_hash = file_sha256_short(wrapper_path)
    runtime_key = (
        str(root),
        str(weights),
        str(checkpoint_hash),
        str(wrapper_hash),
        str(device),
    )
    cached = _LOFTR_RUNTIME_CACHE.get(runtime_key)
    if cached is not None:
        return cached
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("hloc.matchers.loftr")
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != wrapper_path.resolve():
        raise RuntimeError("imported LoFTR wrapper differs from the declared hloc root")
    model = module.LoFTR(
        {"weights": str(weights), "match_threshold": 0.2, "max_num_matches": None}
    ).eval().to(device)
    try:
        import kornia
    except ImportError as error:  # pragma: no cover - LoFTR itself imports Kornia.
        raise RuntimeError("LoFTR requires Kornia") from error
    metadata = {
        "name": "hloc_kornia_loftr",
        "weights": str(weights),
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_sha256": checkpoint_hash,
        "kornia_pretrained_loader_checkpoint": str(actual_loader_checkpoint),
        "kornia_pretrained_loader_checkpoint_sha256": file_sha256_short(
            actual_loader_checkpoint
        ),
        "hloc_wrapper": str(wrapper_path),
        "hloc_wrapper_sha256": wrapper_hash,
        "kornia_version": str(getattr(kornia, "__version__", "unknown")),
        "torch_version": str(torch.__version__),
        "dtype": "float32",
        "max_num_matches": None,
        "match_threshold": 0.2,
    }
    _LOFTR_RUNTIME_CACHE[runtime_key] = (model, metadata)
    return model, metadata


def _validate_args(args: argparse.Namespace) -> None:
    if (
        not str(args.query_id).strip()
        or int(args.resize_width) <= 0
        or int(args.resize_height) <= 0
        or int(args.pair_batch_size) <= 0
        or not 0.0 < float(args.match_threshold) < 1.0
        or abs(float(args.match_threshold) - 0.2) > 1e-12
    ):
        raise ValueError("frozen LoFTR pair cache arguments are invalid")


def build_frozen_loftr_pair_cache(
    *,
    query_id: str,
    mapping_support_manifest: Path,
    maplet_support_index: Path,
    image_root: Path,
    hloc_root: Path,
    loftr_checkpoint: Path,
    loftr_weights: str,
    match_threshold: float,
    resize_width: int,
    resize_height: int,
    pair_batch_size: int,
    device: str,
    output: Path,
) -> dict[str, Any]:
    """Materialize the complete mapping-bank LoFTR cache for one query."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if (
        not str(query_id).strip()
        or int(resize_width) <= 0
        or int(resize_height) <= 0
        or int(pair_batch_size) <= 0
        or abs(float(match_threshold) - 0.2) > 1e-12
    ):
        raise ValueError("frozen LoFTR cache configuration is invalid")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {torch_device}")
    support_ids, support_manifest_metadata = _load_mapping_support_manifest(
        path=Path(mapping_support_manifest),
        maplet_support_index=Path(maplet_support_index),
        query_id=str(query_id),
    )
    query_tensor, query_size = _load_gray_resized(
        image_root=Path(image_root),
        image_id=str(query_id),
        resize_width=int(resize_width),
        resize_height=int(resize_height),
    )
    model, matcher_metadata = _load_loftr_runtime(
        hloc_root=Path(hloc_root),
        weights=str(loftr_weights),
        checkpoint=Path(loftr_checkpoint),
        device=torch_device,
    )
    query_tensor = query_tensor.to(torch_device)
    query_match_parts: list[np.ndarray] = []
    support_match_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    offsets = [0]
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(support_ids), int(pair_batch_size)):
            batch_ids = support_ids[start : start + int(pair_batch_size)]
            loaded = [
                _load_gray_resized(
                    image_root=Path(image_root),
                    image_id=str(image_id),
                    resize_width=int(resize_width),
                    resize_height=int(resize_height),
                )
                for image_id in batch_ids.tolist()
            ]
            support_tensor = torch.cat([item[0] for item in loaded], dim=0).to(torch_device)
            support_sizes = [item[1] for item in loaded]
            query_batch = query_tensor.expand(len(batch_ids), -1, -1, -1).contiguous()
            prediction = model({"image0": query_batch, "image1": support_tensor})
            split = split_batched_loftr_matches(
                query_xy=prediction["keypoints0"].detach().cpu().numpy(),
                support_xy=prediction["keypoints1"].detach().cpu().numpy(),
                confidence=prediction["scores"].detach().cpu().numpy(),
                batch_indices=prediction["batch_indexes"].detach().cpu().numpy(),
                batch_size=len(batch_ids),
            )
            for (query_xy, support_xy, confidence), support_size in zip(split, support_sizes):
                if len(confidence) and np.any(confidence < float(match_threshold) - 1e-6):
                    raise RuntimeError("LoFTR emitted a match below its fixed threshold")
                original_query_xy = restore_pixel_centers_half_pixel(
                    query_xy,
                    source_size=query_size,
                    resized_size=(int(resize_width), int(resize_height)),
                )
                original_support_xy = restore_pixel_centers_half_pixel(
                    support_xy,
                    source_size=support_size,
                    resized_size=(int(resize_width), int(resize_height)),
                )
                query_match_parts.append(original_query_xy)
                support_match_parts.append(original_support_xy)
                confidence_parts.append(np.asarray(confidence, dtype=np.float32))
                offsets.append(offsets[-1] + len(confidence))
    query_matches = (
        np.concatenate(query_match_parts, axis=0)
        if query_match_parts
        else np.zeros((0, 2), dtype=np.float32)
    )
    support_matches = (
        np.concatenate(support_match_parts, axis=0)
        if support_match_parts
        else np.zeros((0, 2), dtype=np.float32)
    )
    confidences = (
        np.concatenate(confidence_parts, axis=0)
        if confidence_parts
        else np.zeros((0,), dtype=np.float32)
    )
    if len(offsets) != len(support_ids) + 1 or len(confidences) != int(offsets[-1]):
        raise RuntimeError("LoFTR pair cache offsets do not cover all fixed mapping images")
    source_contract = {
        "query": image_root_manifest(Path(image_root), [str(query_id)]),
        "mapping_support": image_root_manifest(Path(image_root), support_ids.tolist()),
    }
    metadata: dict[str, Any] = {
        "format": FROZEN_LOFTR_PAIR_CACHE_FORMAT,
        "version": 1,
        "query_id": str(query_id),
        "contains_target_fields": False,
        "contains_target_errors": False,
        "supervision_arrays_loaded": False,
        "strict_global_pair_contract": {
            "all_mapping_manifest_images_processed": True,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "max_num_matches": None,
            "batch_indexes_length_checked": True,
            "coordinate_space": "source_pixel_centers_half_pixel_resize_inverse_v1",
            "support_images": "frozen_maplet_mapping_manifest_complete_v1",
        },
        "mapping_support_manifest": {
            "path": str(Path(mapping_support_manifest)),
            "sha256": file_sha256_short(Path(mapping_support_manifest)),
            "metadata": support_manifest_metadata,
        },
        "maplet_support_index": {
            "path": str(Path(maplet_support_index)),
            "sha256": file_sha256_short(Path(maplet_support_index)),
        },
        "image_source_contract": source_contract,
        "matcher": matcher_metadata,
        "preprocessing": {
            "input": "real_grayscale_cv2_linear_resize_v1",
            "resize_width": int(resize_width),
            "resize_height": int(resize_height),
            "aspect_ratio_policy": "exact_preserving_no_crop_no_pad_v1",
        },
        "runtime": {
            "device": str(torch_device),
            "pair_batch_size": int(pair_batch_size),
            "support_image_count": int(len(support_ids)),
            "match_count": int(len(confidences)),
            "elapsed_seconds": float(time.monotonic() - started),
        },
        "implementation": {
            "source_path": str(Path(__file__)),
            "source_sha256": _LOFTR_BUILDER_SOURCE_SHA256,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_id=np.asarray([str(query_id)], dtype=np.str_),
            support_image_ids=support_ids,
            match_offsets=np.asarray(offsets, dtype=np.int64),
            query_match_xy=query_matches.astype(np.float32, copy=False),
            support_match_xy=support_matches.astype(np.float32, copy=False),
            match_confidence=confidences.astype(np.float32, copy=False),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    summary = {
        "stage": "build_frozen_loftr_global_pair_cache",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": str(query_id),
        "support_image_count": int(len(support_ids)),
        "match_count": int(len(confidences)),
        "matches_per_support_image": {
            "min": int(np.min(counts)),
            "median": float(np.median(counts)),
            "p90": float(np.quantile(counts, 0.90)),
            "max": int(np.max(counts)),
        },
        "elapsed_seconds": metadata["runtime"]["elapsed_seconds"],
        "protocol": {
            "target_free": True,
            "full_mapping_image_manifest": True,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    summary = build_frozen_loftr_pair_cache(
        query_id=str(args.query_id),
        mapping_support_manifest=Path(args.mapping_support_manifest),
        maplet_support_index=Path(args.maplet_support_index),
        image_root=Path(args.image_root),
        hloc_root=Path(args.hloc_root),
        loftr_checkpoint=Path(args.loftr_checkpoint),
        loftr_weights=str(args.loftr_weights),
        match_threshold=float(args.match_threshold),
        resize_width=int(args.resize_width),
        resize_height=int(args.resize_height),
        pair_batch_size=int(args.pair_batch_size),
        device=str(args.device),
        output=Path(args.output),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
