"""Strict, target-free storage for global LoFTR query-to-mapping pairs.

The cache stores every accepted sparse LoFTR correspondence for one query image
against a fixed complete mapping-image manifest.  It intentionally has no
candidate, pose, target, residual, or image-ranking field.  A later candidate
evidence stage may only *read* correspondences at already-frozen SfM
observation anchors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FROZEN_LOFTR_PAIR_CACHE_FORMAT = "frozen_loftr_global_pair_cache_v1"


def _pixel_sizes(value: Sequence[int | float], *, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must be width,height")
    width, height = (float(item) for item in value)
    if not np.isfinite([width, height]).all() or width <= 0.0 or height <= 0.0:
        raise ValueError(f"{name} is invalid")
    return width, height


def resize_pixel_centers_half_pixel(
    xy: np.ndarray,
    *,
    source_size: Sequence[int | float],
    resized_size: Sequence[int | float],
) -> np.ndarray:
    """Map source-image pixel centers through an align-corners=False resize."""

    values = np.asarray(xy, dtype=np.float32)
    source_width, source_height = _pixel_sizes(source_size, name="source size")
    resized_width, resized_height = _pixel_sizes(resized_size, name="resized size")
    if values.ndim < 1 or values.shape[-1] != 2 or np.any(~np.isfinite(values)):
        raise ValueError("pixel coordinates are invalid")
    scale = np.asarray(
        [resized_width / source_width, resized_height / source_height], dtype=np.float32
    )
    return ((values + 0.5) * scale - 0.5).astype(np.float32, copy=False)


def restore_pixel_centers_half_pixel(
    xy: np.ndarray,
    *,
    source_size: Sequence[int | float],
    resized_size: Sequence[int | float],
) -> np.ndarray:
    """Map resized-image pixel centers back into the source image coordinates."""

    values = np.asarray(xy, dtype=np.float32)
    source_width, source_height = _pixel_sizes(source_size, name="source size")
    resized_width, resized_height = _pixel_sizes(resized_size, name="resized size")
    if values.ndim < 1 or values.shape[-1] != 2 or np.any(~np.isfinite(values)):
        raise ValueError("pixel coordinates are invalid")
    scale = np.asarray(
        [resized_width / source_width, resized_height / source_height], dtype=np.float32
    )
    return ((values + 0.5) / scale - 0.5).astype(np.float32, copy=False)


def split_batched_loftr_matches(
    *,
    query_xy: np.ndarray,
    support_xy: np.ndarray,
    confidence: np.ndarray,
    batch_indices: np.ndarray,
    batch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Split matcher output by input pair while rejecting silent truncation.

    HLoc's optional global ``max_num_matches`` truncation historically did not
    truncate ``batch_indexes`` with the keypoint arrays.  The global cache
    therefore forbids a cap and checks every returned array length here.
    """

    query = np.asarray(query_xy, dtype=np.float32)
    support = np.asarray(support_xy, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32).reshape(-1)
    batches = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
    count = len(scores)
    if (
        int(batch_size) <= 0
        or query.shape != (count, 2)
        or support.shape != (count, 2)
        or len(batches) != count
        or np.any(~np.isfinite(query))
        or np.any(~np.isfinite(support))
        or np.any(~np.isfinite(scores))
        or np.any((batches < 0) | (batches >= int(batch_size)))
    ):
        raise ValueError("batched LoFTR matches are incompatible or were truncated")
    return tuple(
        (
            query[batches == index].copy(),
            support[batches == index].copy(),
            scores[batches == index].copy(),
        )
        for index in range(int(batch_size))
    )


@dataclass(frozen=True)
class FrozenLoFTRPairCache:
    """CSR-like correspondences for one query and its full mapping image bank."""

    path: Path
    query_id: str
    support_image_ids: np.ndarray
    match_offsets: np.ndarray
    query_match_xy: np.ndarray
    support_match_xy: np.ndarray
    match_confidence: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        support_ids = np.asarray(self.support_image_ids).astype(str).reshape(-1)
        offsets = np.asarray(self.match_offsets, dtype=np.int64).reshape(-1)
        query_xy = np.asarray(self.query_match_xy, dtype=np.float32).reshape(-1, 2)
        support_xy = np.asarray(self.support_match_xy, dtype=np.float32).reshape(-1, 2)
        confidence = np.asarray(self.match_confidence, dtype=np.float32).reshape(-1)
        metadata = dict(self.metadata)
        if (
            not str(self.query_id)
            or len(support_ids) == 0
            or len(set(support_ids.tolist())) != len(support_ids)
            or np.any(support_ids == "")
            or offsets.shape != (len(support_ids) + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(confidence)
            or support_xy.shape != query_xy.shape
            or len(query_xy) != len(confidence)
            or np.any(~np.isfinite(query_xy))
            or np.any(~np.isfinite(support_xy))
            or np.any(~np.isfinite(confidence))
            or np.any(confidence < 0.0)
            or np.any(confidence > 1.0)
        ):
            raise ValueError("frozen LoFTR pair cache arrays are invalid")
        required_contract = {
            "all_mapping_manifest_images_processed": True,
            "image_level_selection": False,
            "max_num_matches": None,
            "pose_or_ground_truth_used": False,
            "render": False,
        }
        contract = metadata.get("strict_global_pair_contract")
        if (
            metadata.get("format") != FROZEN_LOFTR_PAIR_CACHE_FORMAT
            or metadata.get("contains_target_fields") is not False
            or metadata.get("supervision_arrays_loaded") is not False
            or not isinstance(contract, Mapping)
            or any(contract.get(key) != value for key, value in required_contract.items())
        ):
            raise ValueError("frozen LoFTR pair cache metadata is invalid")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "support_image_ids", support_ids)
        object.__setattr__(self, "match_offsets", offsets)
        object.__setattr__(self, "query_match_xy", query_xy)
        object.__setattr__(self, "support_match_xy", support_xy)
        object.__setattr__(self, "match_confidence", confidence)
        object.__setattr__(self, "metadata", metadata)

    def image_index(self, image_id: str) -> int:
        positions = np.flatnonzero(self.support_image_ids == str(image_id))
        if len(positions) != 1:
            raise KeyError(f"support image is absent from frozen LoFTR cache: {image_id!r}")
        return int(positions[0])

    def matches_for_index(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = int(index)
        if position < 0 or position >= len(self.support_image_ids):
            raise IndexError("frozen LoFTR cache support image index is out of range")
        start, stop = (int(self.match_offsets[position]), int(self.match_offsets[position + 1]))
        return (
            self.query_match_xy[start:stop],
            self.support_match_xy[start:stop],
            self.match_confidence[start:stop],
        )


def load_frozen_loftr_pair_cache(path: Path) -> FrozenLoFTRPairCache:
    """Load a cache without accepting optional labels or score artifacts."""

    required = (
        "query_id",
        "support_image_ids",
        "match_offsets",
        "query_match_xy",
        "support_match_xy",
        "match_confidence",
        "metadata_json",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(required).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: frozen LoFTR cache lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required}
    query_ids = np.asarray(arrays["query_id"]).astype(str).reshape(-1)
    if len(query_ids) != 1:
        raise ValueError(f"{path}: frozen LoFTR cache must contain exactly one query id")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if not isinstance(metadata, dict) or str(metadata.get("query_id", "")) != str(query_ids[0]):
        raise ValueError(f"{path}: frozen LoFTR cache query metadata differs from arrays")
    return FrozenLoFTRPairCache(
        path=Path(path),
        query_id=str(query_ids[0]),
        support_image_ids=np.asarray(arrays["support_image_ids"]),
        match_offsets=np.asarray(arrays["match_offsets"], dtype=np.int64),
        query_match_xy=np.asarray(arrays["query_match_xy"], dtype=np.float32),
        support_match_xy=np.asarray(arrays["support_match_xy"], dtype=np.float32),
        match_confidence=np.asarray(arrays["match_confidence"], dtype=np.float32),
        metadata=metadata,
    )
