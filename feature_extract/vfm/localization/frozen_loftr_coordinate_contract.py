"""Fail-closed coordinate contract between cached LoFTR and COLMAP grids.

LoFTR pair caches retain endpoints in the original real-image pixel grid.  The
landmark, maplet, and PnP pipeline uses the resized COLMAP model grid.  Both
spaces use pixel-center coordinates, but their values are not interchangeable.
This module binds every future evidence artifact to the exact half-pixel resize
transform and camera ownership used to make that conversion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    COLMAP_MODEL_COORDINATE_SPACE,
    LOFTR_SOURCE_COORDINATE_SPACE,
    LOFTR_SOURCE_TO_MODEL_COORDINATE_TRANSFORM,
    MODEL_TO_LOFTR_SOURCE_COORDINATE_TRANSFORM,
    loftr_source_to_model_pixel_centers_by_image,
    model_to_loftr_source_pixel_centers_by_image,
    shared_source_size_from_loftr_cache_metadata,
)


FROZEN_LOFTR_COLMAP_COORDINATE_CONTRACT_VERSION = 1


def _short_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _size_key(size: Sequence[int | float]) -> str:
    if len(size) != 2:
        raise ValueError("image size must contain width,height")
    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    return f"{width}x{height}"


@dataclass(frozen=True)
class FrozenLoFTRColmapCoordinateContract:
    """One immutable source-to-COLMAP coordinate lineage for a pair cache."""

    source_size: tuple[int, int]
    query_id: str
    query_model_size: tuple[int, int]
    support_image_ids: tuple[str, ...]
    model_sizes_by_image: Mapping[str, tuple[int, int]]
    colmap_model_dir: Path
    cameras_sha256: str
    images_camera_ownership_sha256: str

    def __post_init__(self) -> None:
        source_width, source_height = (int(value) for value in self.source_size)
        query_size = tuple(int(value) for value in self.query_model_size)
        support_ids = tuple(str(value) for value in self.support_image_ids)
        sizes = {
            str(image_id): tuple(int(value) for value in size)
            for image_id, size in dict(self.model_sizes_by_image).items()
        }
        required = (str(self.query_id), *support_ids)
        if (
            not str(self.query_id)
            or source_width <= 0
            or source_height <= 0
            or len(query_size) != 2
            or any(value <= 0 for value in query_size)
            or len(set(support_ids)) != len(support_ids)
            or any(not image_id for image_id in support_ids)
            or set(required) != set(sizes)
            or tuple(sizes[str(self.query_id)]) != query_size
        ):
            raise ValueError("LoFTR/COLMAP coordinate contract inputs are invalid")
        for model_width, model_height in sizes.values():
            if (
                model_width <= 0
                or model_height <= 0
                or model_width * source_height != model_height * source_width
            ):
                raise ValueError(
                    "COLMAP model grid is not an exact aspect-preserving resize of source RGB"
                )
        object.__setattr__(self, "source_size", (source_width, source_height))
        object.__setattr__(self, "query_model_size", query_size)
        object.__setattr__(self, "support_image_ids", support_ids)
        object.__setattr__(self, "model_sizes_by_image", sizes)
        object.__setattr__(self, "colmap_model_dir", Path(self.colmap_model_dir))

    def model_to_source(self, xy: np.ndarray, *, image_ids: Sequence[str]) -> np.ndarray:
        return model_to_loftr_source_pixel_centers_by_image(
            xy,
            image_ids=image_ids,
            model_sizes_by_image=self.model_sizes_by_image,
            source_size=self.source_size,
        )

    def source_to_model(self, xy: np.ndarray, *, image_ids: Sequence[str]) -> np.ndarray:
        return loftr_source_to_model_pixel_centers_by_image(
            xy,
            image_ids=image_ids,
            model_sizes_by_image=self.model_sizes_by_image,
            source_size=self.source_size,
        )

    def metadata(self) -> dict[str, Any]:
        histogram: dict[str, int] = {}
        for image_id in self.support_image_ids:
            key = _size_key(self.model_sizes_by_image[image_id])
            histogram[key] = histogram.get(key, 0) + 1
        sizes_payload = {
            image_id: list(self.model_sizes_by_image[image_id])
            for image_id in sorted(self.model_sizes_by_image)
        }
        return {
            "version": FROZEN_LOFTR_COLMAP_COORDINATE_CONTRACT_VERSION,
            "cache_coordinate_space": LOFTR_SOURCE_COORDINATE_SPACE,
            "sfm_coordinate_space": COLMAP_MODEL_COORDINATE_SPACE,
            "model_to_cache_transform": MODEL_TO_LOFTR_SOURCE_COORDINATE_TRANSFORM,
            "cache_to_model_transform": LOFTR_SOURCE_TO_MODEL_COORDINATE_TRANSFORM,
            "source_image_size": list(self.source_size),
            "query": {
                "image_id": self.query_id,
                "model_image_size": list(self.query_model_size),
            },
            "support": {
                "image_count": len(self.support_image_ids),
                "image_ids_sha256": _short_json_hash(list(self.support_image_ids)),
                "model_image_size_histogram": histogram,
                "model_sizes_by_image_sha256": _short_json_hash(sizes_payload),
            },
            "colmap_model": {
                "path": str(self.colmap_model_dir),
                "cameras_sha256": self.cameras_sha256,
                "images_camera_ownership_sha256": self.images_camera_ownership_sha256,
            },
        }


def build_frozen_loftr_colmap_coordinate_contract(
    *,
    cache_metadata: Mapping[str, Any],
    query_id: str,
    support_image_ids: Sequence[str],
    colmap_model_dir: Path,
) -> FrozenLoFTRColmapCoordinateContract:
    """Load only camera ownership/dimensions and bind the coordinate mapping.

    The ownership-only COLMAP parser intentionally does not retain qvec/tvec,
    so this remains a target-free inference-side contract.
    """

    source_size = shared_source_size_from_loftr_cache_metadata(cache_metadata)
    model_dir = Path(colmap_model_dir).resolve(strict=True)
    cameras_path = model_dir / "cameras.bin"
    images_path = model_dir / "images.bin"
    cameras = read_colmap_cameras_binary(cameras_path)
    owners = read_colmap_image_camera_ids_binary(images_path)
    support_ids = tuple(str(value) for value in support_image_ids)
    required = (str(query_id), *support_ids)
    if (
        not str(query_id)
        or not support_ids
        or len(set(support_ids)) != len(support_ids)
        or str(query_id) in set(support_ids)
        or any(not image_id for image_id in required)
    ):
        raise ValueError("LoFTR coordinate contract image ids are invalid")
    missing = sorted(image_id for image_id in required if image_id not in owners)
    if missing:
        raise ValueError(
            "COLMAP camera ownership is missing LoFTR images: " + ", ".join(missing[:3])
        )
    sizes: dict[str, tuple[int, int]] = {}
    for image_id in required:
        camera = cameras.get(int(owners[image_id]))
        if camera is None:
            raise ValueError(f"COLMAP camera is missing for image: {image_id!r}")
        sizes[image_id] = (int(camera.width), int(camera.height))
    return FrozenLoFTRColmapCoordinateContract(
        source_size=source_size,
        query_id=str(query_id),
        query_model_size=sizes[str(query_id)],
        support_image_ids=support_ids,
        model_sizes_by_image=sizes,
        colmap_model_dir=model_dir,
        cameras_sha256=file_sha256_short(cameras_path),
        images_camera_ownership_sha256=file_sha256_short(images_path),
    )


def validate_frozen_loftr_colmap_coordinate_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject an evidence artifact that lacks the explicit v2 coordinate contract."""

    contract = metadata.get("coordinate_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("LoFTR evidence has no explicit source-to-COLMAP coordinate contract")
    expected = {
        "version": FROZEN_LOFTR_COLMAP_COORDINATE_CONTRACT_VERSION,
        "cache_coordinate_space": LOFTR_SOURCE_COORDINATE_SPACE,
        "sfm_coordinate_space": COLMAP_MODEL_COORDINATE_SPACE,
        "model_to_cache_transform": MODEL_TO_LOFTR_SOURCE_COORDINATE_TRANSFORM,
        "cache_to_model_transform": LOFTR_SOURCE_TO_MODEL_COORDINATE_TRANSFORM,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("LoFTR evidence coordinate contract is invalid or stale")
    source_size = contract.get("source_image_size")
    query = contract.get("query")
    support = contract.get("support")
    colmap = contract.get("colmap_model")
    if (
        not isinstance(source_size, list)
        or len(source_size) != 2
        or any(int(value) <= 0 for value in source_size)
        or not isinstance(query, Mapping)
        or not isinstance(query.get("image_id"), str)
        or not isinstance(query.get("model_image_size"), list)
        or not isinstance(support, Mapping)
        or int(support.get("image_count", 0)) <= 0
        or not isinstance(support.get("image_ids_sha256"), str)
        or not isinstance(support.get("model_sizes_by_image_sha256"), str)
        or not isinstance(colmap, Mapping)
        or not isinstance(colmap.get("cameras_sha256"), str)
        or not isinstance(colmap.get("images_camera_ownership_sha256"), str)
    ):
        raise ValueError("LoFTR evidence coordinate contract fields are invalid")
