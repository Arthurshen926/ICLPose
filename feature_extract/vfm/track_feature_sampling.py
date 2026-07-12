"""Sample dense token features at COLMAP track observations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.tokens import TokenBankManifest


_SAMPLED_TRACK_OBSERVATION_CACHE_FORMAT = "vfm_sampled_track_observations_v1"


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


def _token_xy_float(
    xy: tuple[float, float],
    image_width: int,
    image_height: int,
    token_width: int,
    token_height: int,
) -> tuple[float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x_norm = float(xy[0]) / max(float(image_width - 1), 1.0)
    y_norm = float(xy[1]) / max(float(image_height - 1), 1.0)
    x_pos = float(np.clip(x_norm, 0.0, 1.0) * max(token_width - 1, 0))
    y_pos = float(np.clip(y_norm, 0.0, 1.0) * max(token_height - 1, 0))
    return x_pos, y_pos


def _sample_feature_vector(
    feature_map: np.ndarray,
    xy: tuple[float, float],
    image_width: int,
    image_height: int,
    sample_mode: str,
) -> np.ndarray:
    _channels, token_height, token_width = feature_map.shape
    if sample_mode == "nearest":
        x_idx, y_idx = _nearest_token_xy(xy, image_width, image_height, token_width, token_height)
        return np.array(feature_map[:, y_idx, x_idx], dtype=np.float32, copy=True)
    if sample_mode != "bilinear":
        raise ValueError("sample_mode must be 'nearest' or 'bilinear'")
    x_pos, y_pos = _token_xy_float(xy, image_width, image_height, token_width, token_height)
    x0 = int(np.floor(x_pos))
    y0 = int(np.floor(y_pos))
    x1 = min(x0 + 1, token_width - 1)
    y1 = min(y0 + 1, token_height - 1)
    wx = float(x_pos - x0)
    wy = float(y_pos - y0)
    top = (1.0 - wx) * feature_map[:, y0, x0] + wx * feature_map[:, y0, x1]
    bottom = (1.0 - wx) * feature_map[:, y1, x0] + wx * feature_map[:, y1, x1]
    return np.asarray((1.0 - wy) * top + wy * bottom, dtype=np.float32)


def _sample_feature_vectors(
    feature_map: np.ndarray,
    xy: np.ndarray,
    image_widths: np.ndarray,
    image_heights: np.ndarray,
    sample_mode: str,
) -> np.ndarray:
    """Vectorized counterpart of :func:`_sample_feature_vector`."""

    values = np.asarray(feature_map, dtype=np.float32)
    coordinates = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    widths = np.asarray(image_widths, dtype=np.float64).reshape(-1)
    heights = np.asarray(image_heights, dtype=np.float64).reshape(-1)
    if values.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    if widths.shape[0] != coordinates.shape[0] or heights.shape[0] != coordinates.shape[0]:
        raise ValueError("image dimensions must contain one value per coordinate")
    if np.any(widths <= 0.0) or np.any(heights <= 0.0):
        raise ValueError("image dimensions must be positive")
    channels, token_height, token_width = values.shape
    if coordinates.shape[0] == 0:
        return np.zeros((0, int(channels)), dtype=np.float32)
    x_pos = np.clip(coordinates[:, 0] / np.maximum(widths - 1.0, 1.0), 0.0, 1.0) * max(
        int(token_width) - 1,
        0,
    )
    y_pos = np.clip(coordinates[:, 1] / np.maximum(heights - 1.0, 1.0), 0.0, 1.0) * max(
        int(token_height) - 1,
        0,
    )
    if str(sample_mode) == "nearest":
        x_idx = np.rint(x_pos).astype(np.int64)
        y_idx = np.rint(y_pos).astype(np.int64)
        return values[:, y_idx, x_idx].T.astype(np.float32, copy=True)
    if str(sample_mode) != "bilinear":
        raise ValueError("sample_mode must be 'nearest' or 'bilinear'")
    x0 = np.floor(x_pos).astype(np.int64)
    y0 = np.floor(y_pos).astype(np.int64)
    x1 = np.minimum(x0 + 1, int(token_width) - 1)
    y1 = np.minimum(y0 + 1, int(token_height) - 1)
    wx = (x_pos - x0).astype(np.float32)[:, None]
    wy = (y_pos - y0).astype(np.float32)[:, None]
    top = (1.0 - wx) * values[:, y0, x0].T + wx * values[:, y0, x1].T
    bottom = (1.0 - wx) * values[:, y1, x0].T + wx * values[:, y1, x1].T
    return ((1.0 - wy) * top + wy * bottom).astype(np.float32, copy=False)


def _track_view_consistency_weights(
    track_observations: Iterable[ColmapTrackObservation],
    weight_floor: float,
) -> dict[tuple[int, str, int], float]:
    grouped: dict[int, list[ColmapTrackObservation]] = {}
    for obs in track_observations:
        if obs.viewing_ray is None:
            continue
        grouped.setdefault(int(obs.track_id), []).append(obs)
    weights: dict[tuple[int, str, int], float] = {}
    for track_id, observations in grouped.items():
        rays = np.stack([np.asarray(obs.viewing_ray, dtype=np.float64).reshape(3) for obs in observations], axis=0)
        rays = rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        mean_ray = rays.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_ray))
        if mean_norm < 1e-12:
            values = np.ones((len(observations),), dtype=np.float64)
        else:
            mean_ray = mean_ray / mean_norm
            values = np.clip(rays @ mean_ray, 0.0, 1.0)
        for obs, value in zip(observations, values):
            weights[(track_id, obs.image_id, int(obs.point2d_idx))] = max(float(value), weight_floor)
    return weights


def _observation_utility(
    obs: ColmapTrackObservation,
    utility_mode: str,
    weight_floor: float,
    view_consistency_weight: float | None = None,
) -> float:
    if weight_floor <= 0.0:
        raise ValueError("weight_floor must be positive")
    inverse_error = 1.0 / max(float(obs.reprojection_error), 1e-3)
    if utility_mode == "inverse_reprojection":
        return inverse_error
    view_weight = 1.0 if view_consistency_weight is None else max(float(view_consistency_weight), weight_floor)
    if utility_mode == "view_consistency":
        return view_weight
    if utility_mode in {"center", "inverse_reprojection_center"}:
        if obs.image_width is None or obs.image_height is None:
            center_weight = 1.0
        else:
            x_centered = (float(obs.xy[0]) / max(float(obs.image_width - 1), 1.0)) * 2.0 - 1.0
            y_centered = (float(obs.xy[1]) / max(float(obs.image_height - 1), 1.0)) * 2.0 - 1.0
            radial = np.sqrt(x_centered * x_centered + y_centered * y_centered) / np.sqrt(2.0)
            center_weight = max(float(weight_floor), 1.0 - float(np.clip(radial, 0.0, 1.0)))
        if utility_mode == "center":
            return center_weight
        return inverse_error * center_weight
    if utility_mode == "inverse_reprojection_center_view":
        if obs.image_width is None or obs.image_height is None:
            center_weight = 1.0
        else:
            x_centered = (float(obs.xy[0]) / max(float(obs.image_width - 1), 1.0)) * 2.0 - 1.0
            y_centered = (float(obs.xy[1]) / max(float(obs.image_height - 1), 1.0)) * 2.0 - 1.0
            radial = np.sqrt(x_centered * x_centered + y_centered * y_centered) / np.sqrt(2.0)
            center_weight = max(float(weight_floor), 1.0 - float(np.clip(radial, 0.0, 1.0)))
        return inverse_error * center_weight * view_weight
    raise ValueError(
        "utility_mode must be one of: inverse_reprojection, center, inverse_reprojection_center, "
        "view_consistency, inverse_reprojection_center_view"
    )


def sample_token_track_observations(
    track_observations: Iterable[ColmapTrackObservation],
    token_manifest: TokenBankManifest,
    layer_name: str,
    missing: str = "skip",
    utility_mode: str = "inverse_reprojection",
    weight_floor: float = 1e-3,
    sample_mode: str = "nearest",
) -> list[TrackObservation]:
    """Sample token vectors at COLMAP observation coordinates.

    Coordinates are mapped by relative image position, so COLMAP models built
    at a resized resolution can still sample token grids extracted from the
    corresponding full-resolution image.
    """

    if missing not in {"skip", "error"}:
        raise ValueError("missing must be 'skip' or 'error'")
    if utility_mode not in {
        "inverse_reprojection",
        "center",
        "inverse_reprojection_center",
        "view_consistency",
        "inverse_reprojection_center_view",
    }:
        raise ValueError(
            "utility_mode must be one of: inverse_reprojection, center, inverse_reprojection_center, "
            "view_consistency, inverse_reprojection_center_view"
        )
    if sample_mode not in {"nearest", "bilinear"}:
        raise ValueError("sample_mode must be 'nearest' or 'bilinear'")
    track_observation_list = list(track_observations)
    view_weights = _track_view_consistency_weights(track_observation_list, weight_floor=weight_floor)
    index = _manifest_index(token_manifest)
    observations_by_image: dict[str, list[ColmapTrackObservation]] = {}
    for obs in track_observation_list:
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
            feature = _sample_feature_vector(
                feature_map,
                obs.xy,
                int(obs.image_width),
                int(obs.image_height),
                sample_mode=sample_mode,
            )
            view_weight = view_weights.get((int(obs.track_id), obs.image_id, int(obs.point2d_idx)))
            sampled.append(
                TrackObservation(
                    track_id=obs.track_id,
                    image_id=obs.image_id,
                    feature=feature,
                    visible=True,
                    geometry_valid=True,
                    utility=_observation_utility(
                        obs,
                        utility_mode=utility_mode,
                        weight_floor=weight_floor,
                        view_consistency_weight=view_weight,
                    ),
                )
            )
    return sampled


def deduplicate_track_image_observations(
    observations: Iterable[ColmapTrackObservation],
) -> list[ColmapTrackObservation]:
    """Keep one deterministic SfM observation for each physical track/image pair."""

    selected: list[ColmapTrackObservation] = []
    position_by_key: dict[tuple[int, str], int] = {}
    for observation in observations:
        key = (int(observation.track_id), str(observation.image_id))
        position = position_by_key.get(key)
        if position is None:
            position_by_key[key] = len(selected)
            selected.append(observation)
            continue
        current = selected[position]
        current_key = (
            float(current.reprojection_error),
            int(current.point2d_idx),
            float(current.xy[0]),
            float(current.xy[1]),
        )
        candidate_key = (
            float(observation.reprojection_error),
            int(observation.point2d_idx),
            float(observation.xy[0]),
            float(observation.xy[1]),
        )
        if candidate_key < current_key:
            selected[position] = observation
    return selected


def load_colmap_track_observations_jsonl(
    path: Path,
    *,
    image_ids: set[str] | None = None,
    track_ids: set[int] | None = None,
    deduplicate_track_images: bool = True,
) -> list[ColmapTrackObservation]:
    observations: list[ColmapTrackObservation] = []
    selected_images = None if image_ids is None else {str(image_id) for image_id in image_ids}
    selected_tracks = None if track_ids is None else {int(track_id) for track_id in track_ids}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_id = str(item["image_id"])
            if selected_images is not None and image_id not in selected_images:
                continue
            track_id = int(item["track_id"])
            if selected_tracks is not None and track_id not in selected_tracks:
                continue
            observations.append(
                ColmapTrackObservation(
                    track_id=track_id,
                    image_id=image_id,
                    point2d_idx=int(item["point2d_idx"]),
                    xy=(float(item["xy"][0]), float(item["xy"][1])),
                    xyz=np.asarray(item["xyz"], dtype=np.float64),
                    track_length=int(item["track_length"]),
                    reprojection_error=float(item["reprojection_error"]),
                    camera_id=None if item.get("camera_id") is None else int(item["camera_id"]),
                    image_width=None if item.get("image_width") is None else int(item["image_width"]),
                    image_height=None if item.get("image_height") is None else int(item["image_height"]),
                    camera_center=None
                    if item.get("camera_center") is None
                    else np.asarray(item["camera_center"], dtype=np.float64),
                    viewing_ray=None
                    if item.get("viewing_ray") is None
                    else np.asarray(item["viewing_ray"], dtype=np.float64),
                )
            )
    if not bool(deduplicate_track_images):
        return observations
    return deduplicate_track_image_observations(observations)


def save_sampled_track_observations_npz(
    observations: Iterable[TrackObservation],
    path: Path,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Persist sampled raw VFM track observations for reuse across aggregators."""

    obs_list = list(observations)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if obs_list:
        features = np.stack([np.asarray(obs.feature, dtype=np.float32).reshape(-1) for obs in obs_list], axis=0)
        track_ids = np.asarray([int(obs.track_id) for obs in obs_list], dtype=np.int64)
        image_ids = np.asarray([str(obs.image_id) for obs in obs_list], dtype=str)
        utilities = np.asarray([float(obs.utility) for obs in obs_list], dtype=np.float32)
        visible = np.asarray([bool(obs.visible) for obs in obs_list], dtype=bool)
        geometry_valid = np.asarray([bool(obs.geometry_valid) for obs in obs_list], dtype=bool)
        feature_dim = int(features.shape[1])
    else:
        features = np.zeros((0, 0), dtype=np.float32)
        track_ids = np.zeros((0,), dtype=np.int64)
        image_ids = np.zeros((0,), dtype=str)
        utilities = np.zeros((0,), dtype=np.float32)
        visible = np.zeros((0,), dtype=bool)
        geometry_valid = np.zeros((0,), dtype=bool)
        feature_dim = 0
    payload = {
        "format": _SAMPLED_TRACK_OBSERVATION_CACHE_FORMAT,
        "observation_count": len(obs_list),
        "feature_dim": feature_dim,
        "metadata": dict(metadata or {}),
    }
    np.savez_compressed(
        output,
        metadata=np.asarray(json.dumps(payload, sort_keys=True)),
        track_ids=track_ids,
        image_ids=image_ids,
        features=features.astype(np.float32, copy=False),
        utilities=utilities,
        visible=visible,
        geometry_valid=geometry_valid,
    )


def load_sampled_track_observations_npz(path: Path) -> tuple[list[TrackObservation], dict[str, object]]:
    """Load sampled raw VFM track observations written by `save_sampled_track_observations_npz`."""

    cache_path = Path(path)
    with np.load(cache_path) as data:
        if "metadata" not in data:
            raise ValueError(f"sampled observation cache {cache_path} is missing metadata")
        payload = json.loads(str(data["metadata"].item()))
        if payload.get("format") != _SAMPLED_TRACK_OBSERVATION_CACHE_FORMAT:
            raise ValueError(f"unsupported sampled observation cache format in {cache_path}")
        track_ids = data["track_ids"].astype(np.int64)
        image_ids = [str(item) for item in data["image_ids"].tolist()]
        features = data["features"].astype(np.float32)
        utilities = data["utilities"].astype(np.float32)
        visible = data["visible"].astype(bool)
        geometry_valid = data["geometry_valid"].astype(bool)
    if features.shape[0] != track_ids.shape[0]:
        raise ValueError("sampled observation cache has inconsistent feature and track counts")
    observations = [
        TrackObservation(
            track_id=int(track_ids[idx]),
            image_id=image_ids[idx],
            feature=features[idx].astype(np.float32, copy=True),
            visible=bool(visible[idx]),
            geometry_valid=bool(geometry_valid[idx]),
            utility=float(utilities[idx]),
        )
        for idx in range(track_ids.shape[0])
    ]
    return observations, dict(payload.get("metadata", {}))
