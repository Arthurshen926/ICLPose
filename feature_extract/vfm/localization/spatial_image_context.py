"""Validated per-image spatial descriptor caches for local appearance probes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np


_GRID_KEY = re.compile(r"grid([1-9][0-9]*)_descriptors")


@dataclass(frozen=True)
class SpatialImageContextCache:
    """Pose-free, L2-normalized image grids in one descriptor space."""

    image_ids: np.ndarray
    image_sizes: np.ndarray
    grids: Mapping[int, np.ndarray]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.image_ids).astype(str).reshape(-1)
        image_sizes = np.asarray(self.image_sizes, dtype=np.int64).reshape(-1, 2)
        grids = {
            int(size): np.asarray(values, dtype=np.float32)
            for size, values in dict(self.grids).items()
        }
        count = len(image_ids)
        if not count or len(set(image_ids.tolist())) != count:
            raise ValueError("spatial image context cache image IDs are invalid")
        if image_sizes.shape != (count, 2) or np.any(image_sizes <= 0):
            raise ValueError("spatial image context cache image sizes are invalid")
        if not grids or any(size <= 0 for size in grids):
            raise ValueError("spatial image context cache lacks grids")
        descriptor_dims: set[int] = set()
        for size, values in grids.items():
            if values.ndim != 3 or values.shape[:2] != (count, int(size) ** 2):
                raise ValueError(f"spatial image context grid{size} is not aligned")
            if np.any(~np.isfinite(values)):
                raise ValueError(f"spatial image context grid{size} is non-finite")
            norms = np.linalg.norm(values, axis=-1)
            if np.max(np.abs(norms - 1.0)) > 5e-3:
                raise ValueError(f"spatial image context grid{size} is not L2 normalized")
            descriptor_dims.add(int(values.shape[2]))
        if len(descriptor_dims) != 1:
            raise ValueError("spatial image context grids have different descriptor dimensions")
        metadata = dict(self.metadata)
        if not str(metadata.get("format", "")):
            raise ValueError("spatial image context cache lacks a format")
        if bool(metadata.get("pose_or_ground_truth_used", True)):
            raise ValueError("spatial image context cache must be pose/GT free")
        declared_sizes = tuple(sorted(int(value) for value in metadata.get("spatial_grid_sizes", ())))
        if declared_sizes != tuple(sorted(grids)):
            raise ValueError("spatial image context cache grid metadata is stale")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_sizes", image_sizes)
        object.__setattr__(self, "grids", grids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def descriptor_dim(self) -> int:
        return int(next(iter(self.grids.values())).shape[2])

    def grid_descriptors(self, grid_size: int) -> np.ndarray:
        values = self.grids.get(int(grid_size))
        if values is None:
            raise ValueError(f"spatial image context cache has no grid{int(grid_size)}")
        return values

    def image_grid_descriptors(
        self, image_id: str, *, grid_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = getattr(self, "_position_by_image", None)
        if positions is None:
            positions = {
                image_id: row for row, image_id in enumerate(self.image_ids.tolist())
            }
            object.__setattr__(self, "_position_by_image", positions)
        row = positions.get(str(image_id))
        if row is None:
            raise KeyError(f"spatial image context cache has no image {image_id!r}")
        grid_size = int(grid_size)
        values = self.grid_descriptors(grid_size)
        return (
            values[int(row)].reshape(grid_size, grid_size, self.descriptor_dim),
            self.image_sizes[int(row)].astype(np.int64),
        )


def save_spatial_image_context_cache(
    cache: SpatialImageContextCache,
    path: Path,
    *,
    cache_dtype: str = "float16",
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Write a validated cache without pickled data or implicit fields."""

    if str(cache_dtype) not in {"float16", "float32"}:
        raise ValueError("spatial image context cache_dtype must be float16 or float32")
    dtype = np.float16 if str(cache_dtype) == "float16" else np.float32
    extras = {} if extra_arrays is None else {
        str(name): np.asarray(values) for name, values in extra_arrays.items()
    }
    reserved = {"image_ids", "image_sizes", "metadata_json"} | {
        f"grid{int(size)}_descriptors" for size in cache.grids
    }
    if set(extras) & reserved:
        raise ValueError("spatial image context extra arrays collide with cache fields")
    for name, values in extras.items():
        if values.dtype.hasobject:
            raise ValueError(f"spatial image context extra array {name!r} has object dtype")
        if np.issubdtype(values.dtype, np.inexact) and np.any(~np.isfinite(values)):
            raise ValueError(f"spatial image context extra array {name!r} is non-finite")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_ids=np.asarray(cache.image_ids, dtype=np.str_),
        image_sizes=np.asarray(cache.image_sizes, dtype=np.int64),
        **{
            f"grid{int(size)}_descriptors": np.asarray(values, dtype=dtype)
            for size, values in sorted(cache.grids.items())
        },
        **extras,
        metadata_json=np.asarray(json.dumps(cache.metadata, sort_keys=True)),
    )


def load_spatial_image_context_cache(
    path: Path,
    *,
    expected_format: str,
    expected_metadata: Mapping[str, object] | None = None,
) -> SpatialImageContextCache:
    """Load one cache and reject stale format or explicit lineage mismatches."""

    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if str(metadata.get("format", "")) != str(expected_format):
            raise ValueError("unsupported spatial image context cache format")
        if expected_metadata:
            mismatches = {
                key: {"expected": expected, "actual": metadata.get(key)}
                for key, expected in expected_metadata.items()
                if metadata.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    "stale spatial image context cache: "
                    f"{json.dumps(mismatches, sort_keys=True)}"
                )
        grids = {
            int(match.group(1)): np.asarray(data[key])
            for key in data.files
            for match in [_GRID_KEY.fullmatch(str(key))]
            if match is not None
        }
        return SpatialImageContextCache(
            image_ids=np.asarray(data["image_ids"]),
            image_sizes=np.asarray(data["image_sizes"]),
            grids=grids,
            metadata=metadata,
        )
