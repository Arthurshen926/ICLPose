"""Persistent mapping-view depth rendered from a track-free 2DGS surface."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


TWO_DGS_SURFACE_DEPTH_BANK_FORMAT = "track_free_2dgs_surface_depth_bank_v1"


@dataclass
class TwoDGSSurfaceDepthBank:
    """Lazy, bounded-memory access to immutable mapping-view depth maps."""

    manifest_path: Path
    paths_by_image: Mapping[str, Path]
    sha256_by_image: Mapping[str, str]
    width: int
    height: int
    metadata: Mapping[str, object]
    maximum_cached_views: int = 16
    _cache: OrderedDict[str, np.ndarray] = field(default_factory=OrderedDict)

    @classmethod
    def load_json(
        cls,
        path: Path,
        *,
        maximum_cached_views: int = 16,
    ) -> "TwoDGSSurfaceDepthBank":
        manifest_path = Path(path).resolve(strict=True)
        payload = json.loads(manifest_path.read_text())
        records = payload.get("records")
        metadata = payload.get("metadata")
        if (
            payload.get("format") != TWO_DGS_SURFACE_DEPTH_BANK_FORMAT
            or not isinstance(records, list)
            or not isinstance(metadata, dict)
            or int(payload.get("width", 0)) <= 0
            or int(payload.get("height", 0)) <= 0
            or int(maximum_cached_views) <= 0
        ):
            raise ValueError("2DGS surface depth-bank manifest is invalid")
        paths: dict[str, Path] = {}
        checksums: dict[str, str] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("2DGS depth-bank record is invalid")
            image_id = str(record.get("image_id", ""))
            value = Path(str(record.get("path", "")))
            depth_path = (
                value
                if value.is_absolute()
                else (manifest_path.parent / value)
            ).resolve(strict=True)
            if (
                not image_id
                or image_id in paths
                or depth_path.suffix != ".npy"
                or int(record.get("byte_size", -1)) != depth_path.stat().st_size
                or len(str(record.get("sha256", ""))) != 16
            ):
                raise ValueError("2DGS depth-bank record lineage is invalid")
            paths[image_id] = depth_path
            checksums[image_id] = str(record["sha256"])
        if not paths:
            raise ValueError("2DGS surface depth bank is empty")
        for forbidden in ("uses_sfm_points", "uses_sfm_tracks", "uses_radio_intermediate"):
            if metadata.get(forbidden) is not False:
                raise ValueError(f"2DGS depth bank must declare {forbidden}=false")
        return cls(
            manifest_path=manifest_path,
            paths_by_image=paths,
            sha256_by_image=checksums,
            width=int(payload["width"]),
            height=int(payload["height"]),
            metadata=dict(metadata),
            maximum_cached_views=int(maximum_cached_views),
        )

    def depth_for_image(self, image_id: str) -> np.ndarray:
        key = str(image_id)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        path = self.paths_by_image.get(key)
        if path is None:
            raise KeyError(f"mapping view is absent from 2DGS depth bank: {key}")
        if file_sha256_short(path) != str(self.sha256_by_image[key]):
            raise ValueError(f"2DGS depth-map checksum mismatch: {path}")
        depth = np.load(path, mmap_mode="r")
        if depth.shape != (self.height, self.width) or depth.dtype != np.float32:
            raise ValueError(f"2DGS depth map has incompatible shape/dtype: {path}")
        self._cache[key] = depth
        while len(self._cache) > int(self.maximum_cached_views):
            self._cache.popitem(last=False)
        return depth
