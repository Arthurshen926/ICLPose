"""Sample dense token features at COLMAP track observations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.tokens import TokenBankManifest


def _manifest_index(manifest: TokenBankManifest):
    manifest.validate(verify_checksums=False)
    return {record.image_id: record for record in manifest.records}


@lru_cache(maxsize=16)
def _load_layer(path: str, layer_name: str) -> np.ndarray:
    with np.load(path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {path}")
        return np.asarray(data[layer_name], dtype=np.float32)


def _nearest_token_xy(
    xy: tuple[float, float],
    image_width: int,
    image_height: int,
    token_width: int,
    token_height: int,
) -> tuple[int, int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x_norm = float(xy[0]) / max(float(image_width - 1), 1.0)
    y_norm = float(xy[1]) / max(float(image_height - 1), 1.0)
    x_idx = int(round(np.clip(x_norm, 0.0, 1.0) * max(token_width - 1, 0)))
    y_idx = int(round(np.clip(y_norm, 0.0, 1.0) * max(token_height - 1, 0)))
    return x_idx, y_idx


def sample_token_track_observations(
    track_observations: Iterable[ColmapTrackObservation],
    token_manifest: TokenBankManifest,
    layer_name: str,
    missing: str = "skip",
) -> list[TrackObservation]:
    """Sample token vectors at COLMAP observation coordinates.

    Coordinates are mapped by relative image position, so COLMAP models built
    at a resized resolution can still sample token grids extracted from the
    corresponding full-resolution image.
    """

    if missing not in {"skip", "error"}:
        raise ValueError("missing must be 'skip' or 'error'")
    index = _manifest_index(token_manifest)
    observations_by_image: dict[str, list[ColmapTrackObservation]] = {}
    for obs in track_observations:
        observations_by_image.setdefault(obs.image_id, []).append(obs)

    sampled: list[TrackObservation] = []
    for image_id, image_observations in observations_by_image.items():
        record = index.get(image_id)
        if record is None:
            if missing == "error":
                raise ValueError(f"token record not found for image {image_id}")
            continue
        feature_map = _load_layer(str(record.token_path), layer_name)
        if feature_map.ndim != 3:
            raise ValueError("token feature map must have shape (C, H, W)")
        _channels, token_height, token_width = feature_map.shape
        for obs in image_observations:
            if obs.image_width is None or obs.image_height is None:
                if missing == "error":
                    raise ValueError(f"image dimensions missing for {obs.image_id}")
                continue
            x_idx, y_idx = _nearest_token_xy(
                obs.xy,
                int(obs.image_width),
                int(obs.image_height),
                token_width,
                token_height,
            )
            sampled.append(
                TrackObservation(
                    track_id=obs.track_id,
                    image_id=obs.image_id,
                    feature=np.array(feature_map[:, y_idx, x_idx], dtype=np.float32, copy=True),
                    visible=True,
                    geometry_valid=True,
                    utility=1.0 / max(float(obs.reprojection_error), 1e-3),
                )
            )
    return sampled


def load_colmap_track_observations_jsonl(path: Path) -> list[ColmapTrackObservation]:
    observations: list[ColmapTrackObservation] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        observations.append(
            ColmapTrackObservation(
                track_id=int(item["track_id"]),
                image_id=str(item["image_id"]),
                point2d_idx=int(item["point2d_idx"]),
                xy=(float(item["xy"][0]), float(item["xy"][1])),
                xyz=np.asarray(item["xyz"], dtype=np.float64),
                track_length=int(item["track_length"]),
                reprojection_error=float(item["reprojection_error"]),
                camera_id=None if item.get("camera_id") is None else int(item["camera_id"]),
                image_width=None if item.get("image_width") is None else int(item["image_width"]),
                image_height=None if item.get("image_height") is None else int(item["image_height"]),
            )
        )
    return observations
