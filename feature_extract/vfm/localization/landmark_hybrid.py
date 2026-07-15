"""Landmark-referenced real RADIO localization evaluation helpers."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord
from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapImageObservation, ColmapTrackObservation
from feature_extract.vfm.landmark_feature_aggregation import LandmarkAggregationConfig, TrackPrototypeBuilder
from feature_extract.vfm.localization.pipeline import _load_feature_map, _load_rgb_chw
from feature_extract.vfm.localization.schemas import CoarseProposal, MeasurementResult
from feature_extract.vfm.localization.measurement_calibration import GeometryProbabilityModel
from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.measurement_v1.rgb_patch_pose_proxy import scaled_colmap_camera
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    QueryTo3DMatch,
    QueryTo3DMatchingConfig,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_query_tokens_to_landmarks,
    match_reprojection_errors,
    normalize_rows,
    token_grid_xy,
    )
from feature_extract.vfm.localization.pose_eval import (
    evaluate_query_poses,
    summarize_pose_rows,
    write_mapping_rows_csv,
    write_mapping_rows_jsonl,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord
from feature_extract.vfm.track_feature_sampling import (
    _observation_utility,
    _sample_feature_vectors,
    _track_view_consistency_weights,
    deduplicate_track_image_observations,
)


@dataclass(frozen=True)
class LandmarkRetrievalConfig:
    """Fast query-token to landmark retrieval settings for the hybrid evaluator."""

    backend: str = "auto"
    top_k: int = 2
    nn_search_k_for_ratio: int | None = None
    proposal_top_l: int = 1
    ratio_threshold: float | None = 0.95
    min_similarity: float = 0.2
    min_similarity_margin: float | None = None
    min_observation_count: int = 2
    query_token_step: int = 4
    max_matches: int | None = 1000
    block_size: int = 512
    deduplicate_tracks: bool = True
    query_token_selection: str = "uniform"
    query_heatmap_top_k: int = 0
    query_heatmap_nms_radius: int = 0
    query_heatmap_grid_rows: int = 4
    query_heatmap_grid_cols: int = 4
    query_heatmap_min_score: float | None = None
    query_xy_coordinate_mode: str = "edge_legacy"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "exact", "faiss", "torch_cuda"}:
            raise ValueError("backend must be one of: auto, exact, faiss, torch_cuda")
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if self.nn_search_k_for_ratio is not None and int(self.nn_search_k_for_ratio) <= 0:
            raise ValueError("nn_search_k_for_ratio must be positive")
        if int(self.proposal_top_l) <= 0:
            raise ValueError("proposal_top_l must be positive")
        if self.ratio_threshold is not None and not 0.0 < float(self.ratio_threshold) <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")
        if self.min_similarity_margin is not None and float(self.min_similarity_margin) < 0.0:
            raise ValueError("min_similarity_margin must be non-negative")
        if int(self.min_observation_count) <= 0:
            raise ValueError("min_observation_count must be positive")
        if int(self.query_token_step) <= 0:
            raise ValueError("query_token_step must be positive")
        if self.max_matches is not None and int(self.max_matches) <= 0:
            raise ValueError("max_matches must be positive")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if str(self.query_token_selection) not in {"uniform", "heatmap"}:
            raise ValueError("query_token_selection must be 'uniform' or 'heatmap'")
        if int(self.query_heatmap_top_k) < 0:
            raise ValueError("query_heatmap_top_k must be non-negative")
        if int(self.query_heatmap_nms_radius) < 0:
            raise ValueError("query_heatmap_nms_radius must be non-negative")
        if int(self.query_heatmap_grid_rows) <= 0 or int(self.query_heatmap_grid_cols) <= 0:
            raise ValueError("query_heatmap grid dimensions must be positive")
        if self.query_heatmap_min_score is not None and not 0.0 <= float(self.query_heatmap_min_score) <= 1.0:
            raise ValueError("query_heatmap_min_score must be in [0, 1]")
        if str(self.query_xy_coordinate_mode) not in {"edge_legacy", "cell_center"}:
            raise ValueError("query_xy_coordinate_mode must be 'edge_legacy' or 'cell_center'")


@dataclass(frozen=True)
class _PreparedLandmarkSearchIndex:
    index: LandmarkMapIndex
    landmark_features: np.ndarray
    backend_name: str
    faiss_index: Any | None = None
    torch_features: torch.Tensor | None = None
    max_prototypes_per_track: int = 1


class LandmarkSearchIndexCache:
    """LRU cache for normalized landmark search features and optional FAISS indices."""

    def __init__(self, max_entries: int = 64) -> None:
        if int(max_entries) < 0:
            raise ValueError("max_entries must be non-negative")
        self.max_entries = int(max_entries)
        self._items: OrderedDict[Hashable, _PreparedLandmarkSearchIndex] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        landmark_index: LandmarkMapIndex,
        config: LandmarkRetrievalConfig,
        *,
        cache_key: Hashable | None = None,
    ) -> tuple[_PreparedLandmarkSearchIndex, bool]:
        if self.max_entries == 0:
            return _prepare_landmark_search_index(landmark_index, config), False
        key = self._entry_key(landmark_index, config, cache_key=cache_key)
        cached = self._items.get(key)
        if cached is not None:
            self._hits += 1
            self._items.move_to_end(key)
            return cached, True
        self._misses += 1
        prepared = _prepare_landmark_search_index(landmark_index, config)
        self._items[key] = prepared
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
        return prepared, False

    def stats(self) -> dict[str, int]:
        return {
            "max_entries": int(self.max_entries),
            "entries": int(len(self._items)),
            "hits": int(self._hits),
            "misses": int(self._misses),
        }

    def _entry_key(
        self,
        landmark_index: LandmarkMapIndex,
        config: LandmarkRetrievalConfig,
        *,
        cache_key: Hashable | None,
    ) -> Hashable:
        track_ids = np.asarray(landmark_index.track_ids, dtype=np.int64).reshape(-1)
        if track_ids.size:
            fingerprint = (
                int(track_ids.size),
                int(track_ids[0]),
                int(track_ids[-1]),
                int(np.sum(track_ids, dtype=np.int64) % np.int64(2_147_483_647)),
            )
        else:
            fingerprint = (0, 0, 0, 0)
        user_key = cache_key if cache_key is not None else ("track_ids", fingerprint)
        return (
            user_key,
            str(config.backend),
            int(config.min_observation_count),
            int(landmark_index.feature_dim),
            fingerprint,
        )


@dataclass(frozen=True)
class LandmarkOwnerObservation:
    track_id: int
    image_id: str
    xy: np.ndarray
    image_size: tuple[int, int]
    observation: ColmapTrackObservation

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(xy)):
            raise ValueError("owner observation xy must be finite")
        width, height = int(self.image_size[0]), int(self.image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("owner observation image_size must be positive")
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "image_size", (width, height))


def _scaled_observation_xy(
    observation: ColmapTrackObservation,
    target_image_sizes: Mapping[str, tuple[int, int]],
) -> tuple[np.ndarray, tuple[int, int]]:
    target = target_image_sizes.get(str(observation.image_id))
    source_w = int(observation.image_width or (target[0] if target is not None else 0))
    source_h = int(observation.image_height or (target[1] if target is not None else 0))
    if target is None:
        target = (source_w, source_h)
    if source_w <= 0 or source_h <= 0 or int(target[0]) <= 0 or int(target[1]) <= 0:
        raise ValueError(f"invalid image size for owner observation {observation.image_id}")
    xy = np.asarray(observation.xy, dtype=np.float64).reshape(2)
    scaled = np.asarray(
        [xy[0] * float(target[0]) / float(source_w), xy[1] * float(target[1]) / float(source_h)],
        dtype=np.float32,
    )
    return scaled, (int(target[0]), int(target[1]))


class LandmarkOwnerObservationIndex:
    """Best real owner-view observation lookup for landmark tracks."""

    def __init__(
        self,
        observations: Sequence[ColmapTrackObservation],
        *,
        target_image_sizes: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        target_sizes = {str(key): (int(value[0]), int(value[1])) for key, value in (target_image_sizes or {}).items()}
        by_track: dict[int, list[LandmarkOwnerObservation]] = {}
        for observation in observations:
            xy, image_size = _scaled_observation_xy(observation, target_sizes)
            owner = LandmarkOwnerObservation(
                track_id=int(observation.track_id),
                image_id=str(observation.image_id),
                xy=xy,
                image_size=image_size,
                observation=observation,
            )
            by_track.setdefault(int(observation.track_id), []).append(owner)
        self._by_track = {
            track_id: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        float(item.observation.reprojection_error),
                        -int(item.observation.track_length),
                        item.image_id,
                    ),
                )
            )
            for track_id, values in by_track.items()
        }

    def select(self, track_id: int, *, exclude_image_id: str | None = None) -> LandmarkOwnerObservation | None:
        values = self._by_track.get(int(track_id), ())
        if not values:
            return None
        if exclude_image_id is not None:
            excluded = str(exclude_image_id)
            for item in values:
                if item.image_id != excluded:
                    return item
        return values[0]


def project_landmark_index_features(
    index: LandmarkMapIndex,
    projector: Callable[[np.ndarray], np.ndarray],
) -> LandmarkMapIndex:
    """Return a copy of a landmark index with projected descriptor features."""

    projected = np.asarray(projector(np.asarray(index.features, dtype=np.float32)), dtype=np.float32)
    if projected.ndim != 2 or int(projected.shape[0]) != len(index):
        raise ValueError("projector must return an array with shape (N, C)")
    return LandmarkMapIndex(
        track_ids=index.track_ids,
        xyz=index.xyz,
        features=projected,
        mean_variances=index.mean_variances,
        observation_counts=index.observation_counts,
        observation_image_ids=index.observation_image_ids,
        reprojection_errors=index.reprojection_errors,
        feature_ambiguities=index.feature_ambiguities,
        prototype_ids=index.prototype_ids,
    )


def _records_by_image_id(manifest: TokenBankManifest) -> dict[str, TokenBankRecord]:
    manifest.validate(verify_checksums=False)
    return {str(record.image_id): record for record in manifest.records}


def _track_xyz_and_reprojection_stats(
    observations: Sequence[ColmapTrackObservation],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    errors_by_track: dict[int, list[float]] = {}
    for observation in observations:
        track_id = int(observation.track_id)
        xyz_by_track.setdefault(track_id, np.asarray(observation.xyz, dtype=np.float64).reshape(3))
        errors_by_track.setdefault(track_id, []).append(float(observation.reprojection_error))
    reprojection_error_by_track = {
        int(track_id): float(np.mean(values)) for track_id, values in errors_by_track.items() if values
    }
    return xyz_by_track, reprojection_error_by_track


def sample_projected_track_observations(
    observations: Sequence[ColmapTrackObservation],
    token_manifest: TokenBankManifest,
    feature_mapper,
    *,
    feature_key: str = "radio_final",
    missing: str = "skip",
    utility_mode: str = "inverse_reprojection",
    weight_floor: float = 1e-6,
    sample_mode: str = "bilinear",
    projection_image_batch_size: int = 8,
    projection_load_workers: int = 4,
) -> tuple[list[TrackObservation], dict[str, Any]]:
    """Project each source image as a full map, then sample mapped descriptors at SfM observations."""

    if str(missing) not in {"skip", "error"}:
        raise ValueError("missing must be 'skip' or 'error'")
    if str(sample_mode) not in {"nearest", "bilinear"}:
        raise ValueError("sample_mode must be 'nearest' or 'bilinear'")
    if float(weight_floor) <= 0.0:
        raise ValueError("weight_floor must be positive")
    if int(projection_image_batch_size) <= 0:
        raise ValueError("projection_image_batch_size must be positive")
    if int(projection_load_workers) < 0:
        raise ValueError("projection_load_workers must be non-negative")

    raw_values = list(observations)
    values = deduplicate_track_image_observations(raw_values)
    records = _records_by_image_id(token_manifest)
    view_weights = (
        _track_view_consistency_weights(values, weight_floor=float(weight_floor))
        if str(utility_mode) in {"view_consistency", "inverse_reprojection_center_view"}
        else {}
    )
    by_image: dict[str, list[ColmapTrackObservation]] = {}
    for observation in values:
        by_image.setdefault(str(observation.image_id), []).append(observation)

    sampled: list[TrackObservation] = []
    missing_images: list[str] = []
    projected_image_count = 0
    image_items = sorted(by_image.items())
    loader = ThreadPoolExecutor(max_workers=int(projection_load_workers)) if int(projection_load_workers) > 1 else None
    try:
        for batch_start in range(0, len(image_items), int(projection_image_batch_size)):
            load_specs: list[tuple[str, list[ColmapTrackObservation], Path]] = []
            for image_id, image_observations in image_items[
                batch_start : batch_start + int(projection_image_batch_size)
            ]:
                record = records.get(str(image_id))
                if record is None:
                    if str(missing) == "error":
                        raise ValueError(f"token record not found for image {image_id}")
                    missing_images.append(str(image_id))
                    continue
                load_specs.append((str(image_id), image_observations, Path(record.token_path)))
            if not load_specs:
                continue
            if loader is None:
                raw_maps = [_load_feature_map(path, key=str(feature_key)) for _image, _obs, path in load_specs]
            else:
                raw_maps = list(
                    loader.map(
                        lambda path: _load_feature_map(path, key=str(feature_key)),
                        [path for _image, _obs, path in load_specs],
                    )
                )
            available = [
                (image_id, image_observations, raw_map)
                for (image_id, image_observations, _path), raw_map in zip(load_specs, raw_maps)
            ]
            project_batch = getattr(feature_mapper, "project_batch", None)
            if callable(project_batch) and len({tuple(raw_map.shape) for raw_map in raw_maps}) == 1:
                mapped_batch = project_batch(np.stack(raw_maps, axis=0))
                projected_maps = [np.asarray(item.coarse_descriptors, dtype=np.float32) for item in mapped_batch]
            else:
                projected_maps = [
                    np.asarray(feature_mapper.project(raw_map).coarse_descriptors, dtype=np.float32)
                    for raw_map in raw_maps
                ]
            if len(projected_maps) != len(available):
                raise ValueError("feature mapper returned the wrong number of projected maps")
            projected_image_count += int(len(projected_maps))
            for (_image_id, image_observations, _raw_map), projected_map in zip(available, projected_maps):
                if projected_map.ndim != 3:
                    raise ValueError("mapped feature map must have shape (C, H, W)")
                valid_observations = [
                    observation
                    for observation in image_observations
                    if observation.image_width is not None and observation.image_height is not None
                ]
                if len(valid_observations) != len(image_observations) and str(missing) == "error":
                    missing_observation = next(
                        observation
                        for observation in image_observations
                        if observation.image_width is None or observation.image_height is None
                    )
                    raise ValueError(f"image dimensions missing for {missing_observation.image_id}")
                features = _sample_feature_vectors(
                    projected_map,
                    np.asarray([observation.xy for observation in valid_observations], dtype=np.float64),
                    np.asarray([observation.image_width for observation in valid_observations], dtype=np.int64),
                    np.asarray([observation.image_height for observation in valid_observations], dtype=np.int64),
                    str(sample_mode),
                )
                for observation, feature in zip(valid_observations, features):
                    view_weight = view_weights.get(
                        (int(observation.track_id), str(observation.image_id), int(observation.point2d_idx))
                    )
                    sampled.append(
                        TrackObservation(
                            track_id=int(observation.track_id),
                            image_id=str(observation.image_id),
                            feature=feature,
                            visible=True,
                            geometry_valid=True,
                            utility=_observation_utility(
                                observation,
                                str(utility_mode),
                                float(weight_floor),
                                view_consistency_weight=view_weight,
                            ),
                            camera_center=(
                                None
                                if observation.camera_center is None
                                else np.asarray(
                                    observation.camera_center, dtype=np.float64
                                ).reshape(3)
                            ),
                            viewing_ray=(
                                None
                                if observation.viewing_ray is None
                                else np.asarray(
                                    observation.viewing_ray, dtype=np.float64
                                ).reshape(3)
                            ),
                        )
                    )
    finally:
        if loader is not None:
            loader.shutdown(wait=True)

    metadata = {
        "projection_mode": "full_map_projected_observations",
        "input_observation_count": int(len(raw_values)),
        "effective_observation_count": int(len(values)),
        "duplicate_track_image_observation_count": int(len(raw_values) - len(values)),
        "sampled_observation_count": int(len(sampled)),
        "projected_image_count": int(projected_image_count),
        "missing_image_count": int(len(missing_images)),
        "missing_images": missing_images[:20],
        "feature_key": str(feature_key),
        "sample_mode": str(sample_mode),
        "utility_mode": str(utility_mode),
        "projection_image_batch_size": int(projection_image_batch_size),
        "projection_load_workers": int(projection_load_workers),
    }
    return sampled, metadata


def build_projected_observation_landmark_index(
    observations: Sequence[ColmapTrackObservation],
    token_manifest: TokenBankManifest,
    feature_mapper,
    *,
    feature_key: str = "radio_final",
    aggregation_method: str = "mean",
    min_observations: int = 2,
    missing: str = "skip",
    utility_mode: str = "inverse_reprojection",
    weight_floor: float = 1e-6,
    sample_mode: str = "bilinear",
    seed: int = 0,
    trim_fraction: float = 0.2,
    view_consistent_keep: int = 4,
    geometric_median_iterations: int = 32,
    l2_normalize_observations: bool = False,
    projection_image_batch_size: int = 8,
    projection_load_workers: int = 4,
) -> tuple[LandmarkMapIndex, dict[str, Any]]:
    """Build a 3D landmark index by aggregating per-observation full-map projected descriptors."""

    sampled, sampling_metadata = sample_projected_track_observations(
        observations,
        token_manifest,
        feature_mapper,
        feature_key=str(feature_key),
        missing=str(missing),
        utility_mode=str(utility_mode),
        weight_floor=float(weight_floor),
        sample_mode=str(sample_mode),
        projection_image_batch_size=int(projection_image_batch_size),
        projection_load_workers=int(projection_load_workers),
    )
    aggregation = LandmarkAggregationConfig(
        method=str(aggregation_method),
        min_observations=int(min_observations),
        seed=int(seed),
        trim_fraction=float(trim_fraction),
        view_consistent_keep=int(view_consistent_keep),
        geometric_median_iterations=int(geometric_median_iterations),
        l2_normalize_observations=bool(l2_normalize_observations),
        weight_floor=float(weight_floor),
    )
    prototype_builder = TrackPrototypeBuilder(
        aggregation=aggregation,
        normalize_final_prototypes=True,
    )
    aggregation_device = str(getattr(feature_mapper, "device", "cpu"))
    bank = prototype_builder.build_bank_torch(sampled, device=aggregation_device)
    xyz_by_track, reprojection_error_by_track = _track_xyz_and_reprojection_stats(list(observations))
    index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
    metadata = {
        **sampling_metadata,
        "aggregation": aggregation.to_dict(),
        "prototype_builder": prototype_builder.to_dict(),
        "aggregation_device": aggregation_device,
        "landmark_count": int(len(index)),
        "feature_dim": int(index.feature_dim),
    }
    return index, metadata


def save_landmark_index_npz(
    index: LandmarkMapIndex,
    path: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist a projected landmark index without Python pickle payloads."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "landmark_map_index_npz",
        "format_version": 2,
        **dict(metadata or {}),
    }
    np.savez(
        output,
        track_ids=np.asarray(index.track_ids, dtype=np.int64),
        xyz=np.asarray(index.xyz, dtype=np.float64),
        features=np.asarray(index.features, dtype=np.float32),
        mean_variances=np.asarray(index.mean_variances, dtype=np.float32),
        observation_counts=np.asarray(index.observation_counts, dtype=np.int64),
        reprojection_errors=np.asarray(index.reprojection_errors, dtype=np.float32),
        feature_ambiguities=np.asarray(index.feature_ambiguities, dtype=np.float32),
        prototype_ids=np.asarray(index.prototype_ids, dtype=np.int64),
        observation_image_ids_json=np.asarray(json.dumps(index.observation_image_ids), dtype=np.str_),
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True), dtype=np.str_),
    )


def load_landmark_index_npz(path: Path) -> tuple[LandmarkMapIndex, dict[str, Any]]:
    """Load a projected landmark index written by :func:`save_landmark_index_npz`."""

    with np.load(Path(path), allow_pickle=False) as data:
        observation_image_ids = tuple(tuple(str(item) for item in ids) for ids in json.loads(str(data["observation_image_ids_json"].item())))
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        index = LandmarkMapIndex(
            track_ids=np.asarray(data["track_ids"], dtype=np.int64),
            xyz=np.asarray(data["xyz"], dtype=np.float64),
            features=np.asarray(data["features"], dtype=np.float32),
            mean_variances=np.asarray(data["mean_variances"], dtype=np.float32),
            observation_counts=np.asarray(data["observation_counts"], dtype=np.int64),
            observation_image_ids=observation_image_ids,
            reprojection_errors=np.asarray(data["reprojection_errors"], dtype=np.float32),
            feature_ambiguities=np.asarray(data["feature_ambiguities"], dtype=np.float32),
            prototype_ids=(
                np.asarray(data["prototype_ids"], dtype=np.int64)
                if "prototype_ids" in data
                else np.zeros_like(np.asarray(data["track_ids"], dtype=np.int64))
            ),
        )
    return index, dict(metadata)


def project_landmark_features_with_joint_model(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 8192,
) -> np.ndarray:
    """Project raw landmark vectors as one-cell maps through a joint model."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), max(1, int(batch_size))):
            batch = torch.as_tensor(values[start : start + int(batch_size), :, None, None], device=torch_device)
            descriptors, _heatmap, _offsets = model.forward_feature_map(batch)
            projected = descriptors[:, :, 0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            outputs.append(projected)
    if was_training:
        model.train()
    if not outputs:
        output_dim = int(getattr(model, "output_dim", 0))
        return np.zeros((0, output_dim), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _flatten_query_feature_map(
    feature_map: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    step: int,
    coordinate_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("query feature map must have shape (C, H, W)")
    channels, token_height, token_width = values.shape
    rows: list[np.ndarray] = []
    token_indices: list[int] = []
    for y_idx in range(0, token_height, int(step)):
        for x_idx in range(0, token_width, int(step)):
            rows.append(values[:, y_idx, x_idx])
            token_indices.append(int(y_idx * token_width + x_idx))
    if not rows:
        return (
            np.zeros((0, channels), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    xy = token_grid_xy(
        token_width,
        token_height,
        image_width,
        image_height,
        step=int(step),
        coordinate_mode="edge" if str(coordinate_mode) == "edge_legacy" else "center",
    )
    return np.stack(rows, axis=0).astype(np.float32, copy=False), xy, np.asarray(token_indices, dtype=np.int64)


def _heatmap_grid_cell(row: int, col: int, *, height: int, width: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    grid_row = int(np.clip(np.floor(float(row) / max(float(height), 1.0) * int(grid_rows)), 0, int(grid_rows) - 1))
    grid_col = int(np.clip(np.floor(float(col) / max(float(width), 1.0) * int(grid_cols)), 0, int(grid_cols) - 1))
    return grid_row, grid_col


def _select_heatmap_token_indices(
    heatmap: np.ndarray,
    *,
    top_k: int,
    nms_radius: int,
    grid_rows: int,
    grid_cols: int,
    min_score: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.asarray(heatmap, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("query heatmap must have shape (H, W)")
    height, width = int(scores.shape[0]), int(scores.shape[1])
    finite = np.isfinite(scores)
    if min_score is not None:
        finite &= scores >= float(min_score)
    candidate_indices = np.flatnonzero(finite.reshape(-1))
    if candidate_indices.size == 0:
        return np.zeros((0,), dtype=np.int64), {
            "query_token_selection": "heatmap",
            "candidate_query_token_count": 0,
            "selected_query_token_count": 0,
        }
    candidate_scores = scores.reshape(-1)[candidate_indices]
    order = np.lexsort((candidate_indices, -candidate_scores))
    ranked = candidate_indices[order].astype(np.int64, copy=False)
    limit = int(top_k) if int(top_k) > 0 else int(ranked.size)
    limit = min(limit, int(ranked.size))
    radius = int(nms_radius)
    selected: list[int] = []
    selected_set: set[int] = set()
    occupied: list[tuple[int, int]] = []
    counts: dict[tuple[int, int], int] = {}
    quota = max(1, int(np.ceil(float(limit) / float(max(1, int(grid_rows) * int(grid_cols))))))

    def accepted(index: int, *, respect_quota: bool) -> bool:
        row, col = divmod(int(index), width)
        if radius > 0:
            for used_row, used_col in occupied:
                if max(abs(int(row) - int(used_row)), abs(int(col) - int(used_col))) <= radius:
                    return False
        if respect_quota:
            cell = _heatmap_grid_cell(row, col, height=height, width=width, grid_rows=int(grid_rows), grid_cols=int(grid_cols))
            if counts.get(cell, 0) >= quota:
                return False
        return True

    def add(index: int) -> None:
        row, col = divmod(int(index), width)
        cell = _heatmap_grid_cell(row, col, height=height, width=width, grid_rows=int(grid_rows), grid_cols=int(grid_cols))
        selected.append(int(index))
        selected_set.add(int(index))
        occupied.append((int(row), int(col)))
        counts[cell] = counts.get(cell, 0) + 1

    for index in ranked:
        if len(selected) >= limit:
            break
        if accepted(int(index), respect_quota=True):
            add(int(index))
    for index in ranked:
        if len(selected) >= limit:
            break
        if int(index) in selected_set:
            continue
        if accepted(int(index), respect_quota=False):
            add(int(index))

    selected_array = np.asarray(selected, dtype=np.int64)
    metadata = {
        "query_token_selection": "heatmap",
        "candidate_query_token_count": int(candidate_indices.size),
        "selected_query_token_count": int(selected_array.size),
        "query_heatmap_top_k": int(top_k),
        "query_heatmap_nms_radius": int(nms_radius),
        "query_heatmap_grid_rows": int(grid_rows),
        "query_heatmap_grid_cols": int(grid_cols),
        "query_heatmap_min_score": None if min_score is None else float(min_score),
        "query_heatmap_selected_min": float(np.min(scores.reshape(-1)[selected_array])) if selected_array.size else None,
        "query_heatmap_selected_max": float(np.max(scores.reshape(-1)[selected_array])) if selected_array.size else None,
    }
    return selected_array, metadata


def select_query_tokens_for_landmark_retrieval(
    feature_map: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    config: LandmarkRetrievalConfig,
    query_heatmap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Select query token descriptors for landmark retrieval, preserving the uniform baseline by default."""

    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("query feature map must have shape (C, H, W)")
    channels, token_height, token_width = int(values.shape[0]), int(values.shape[1]), int(values.shape[2])
    if str(config.query_token_selection) == "uniform":
        features, xy, token_indices = _flatten_query_feature_map(
            values,
            image_width=int(image_width),
            image_height=int(image_height),
            step=int(config.query_token_step),
            coordinate_mode=str(config.query_xy_coordinate_mode),
        )
        return features, xy, token_indices, {
            "query_token_selection": "uniform",
            "query_token_step": int(config.query_token_step),
            "selected_query_token_count": int(token_indices.size),
            "query_xy_coordinate_mode": str(config.query_xy_coordinate_mode),
        }
    if query_heatmap is None:
        raise ValueError("query_heatmap is required when query_token_selection='heatmap'")
    heatmap = np.asarray(query_heatmap, dtype=np.float32)
    if heatmap.shape != (token_height, token_width):
        raise ValueError("query_heatmap shape must match query feature map spatial shape")
    selected, metadata = _select_heatmap_token_indices(
        heatmap,
        top_k=int(config.query_heatmap_top_k or (config.max_matches or token_height * token_width)),
        nms_radius=int(config.query_heatmap_nms_radius),
        grid_rows=int(config.query_heatmap_grid_rows),
        grid_cols=int(config.query_heatmap_grid_cols),
        min_score=config.query_heatmap_min_score,
    )
    flat = values.reshape(channels, token_height * token_width).T
    features = flat[selected].astype(np.float32, copy=False) if selected.size else np.zeros((0, channels), dtype=np.float32)
    all_xy = token_grid_xy(
        token_width,
        token_height,
        int(image_width),
        int(image_height),
        step=1,
        coordinate_mode="edge" if str(config.query_xy_coordinate_mode) == "edge_legacy" else "center",
    )
    xy = all_xy[selected].astype(np.float64, copy=False) if selected.size else np.zeros((0, 2), dtype=np.float64)
    metadata["query_xy_coordinate_mode"] = str(config.query_xy_coordinate_mode)
    return features, xy, selected, metadata


def _valid_retrieval_landmark_subset(index: LandmarkMapIndex, config: LandmarkRetrievalConfig) -> LandmarkMapIndex:
    mask = np.asarray(index.observation_counts >= int(config.min_observation_count), dtype=bool)
    return index.subset(mask)


def _exact_topk(
    query_features: np.ndarray,
    landmark_features: np.ndarray,
    *,
    top_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_count = int(query_features.shape[0])
    landmark_count = int(landmark_features.shape[0])
    effective_top_k = min(int(top_k), landmark_count)
    top_indices = np.full((query_count, effective_top_k), -1, dtype=np.int64)
    top_scores = np.full((query_count, effective_top_k), -np.inf, dtype=np.float32)
    for start in range(0, query_count, max(1, int(block_size))):
        end = min(start + max(1, int(block_size)), query_count)
        scores = query_features[start:end] @ landmark_features.T
        if effective_top_k == 1:
            local_indices = np.argmax(scores, axis=1)[:, None]
        else:
            local_indices = np.argpartition(-scores, kth=effective_top_k - 1, axis=1)[:, :effective_top_k]
            local_scores = np.take_along_axis(scores, local_indices, axis=1)
            order = np.argsort(-local_scores, axis=1)
            local_indices = np.take_along_axis(local_indices, order, axis=1)
        local_scores = np.take_along_axis(scores, local_indices, axis=1)
        top_indices[start:end] = local_indices.astype(np.int64)
        top_scores[start:end] = local_scores.astype(np.float32)
    return top_indices, top_scores


def _faiss_topk(
    query_features: np.ndarray,
    landmark_features: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError("faiss backend requested but faiss is not importable") from exc

    index = faiss.IndexFlatIP(int(landmark_features.shape[1]))
    index.add(np.ascontiguousarray(landmark_features.astype(np.float32, copy=False)))
    scores, indices = index.search(np.ascontiguousarray(query_features.astype(np.float32, copy=False)), int(top_k))
    return indices.astype(np.int64, copy=False), scores.astype(np.float32, copy=False)


def _retrieval_backend_available(backend: str) -> bool:
    if backend == "exact":
        return True
    if backend == "torch_cuda":
        return bool(torch.cuda.is_available())
    if backend != "faiss":
        return False
    try:
        import faiss  # noqa: F401  # type: ignore
    except Exception:
        return False
    return True


def _resolved_retrieval_backend(config: LandmarkRetrievalConfig) -> str:
    requested = str(config.backend)
    if requested == "auto":
        return "faiss" if _retrieval_backend_available("faiss") else "exact"
    return requested


def _prepare_landmark_search_index(
    landmark_index: LandmarkMapIndex,
    config: LandmarkRetrievalConfig,
) -> _PreparedLandmarkSearchIndex:
    index = _valid_retrieval_landmark_subset(landmark_index, config)
    landmark_features, valid_landmarks = normalize_rows(index.features)
    if not np.all(valid_landmarks):
        index = index.subset(valid_landmarks)
        landmark_features = landmark_features[valid_landmarks]
    backend = _resolved_retrieval_backend(config)
    if len(index) > 0:
        _track_ids, prototype_counts = np.unique(index.track_ids, return_counts=True)
        max_prototypes_per_track = int(np.max(prototype_counts))
    else:
        max_prototypes_per_track = 1
    if backend == "faiss":
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError("faiss backend requested but faiss is not importable") from exc
        faiss_index = faiss.IndexFlatIP(int(landmark_features.shape[1]))
        if landmark_features.shape[0] > 0:
            faiss_index.add(np.ascontiguousarray(landmark_features.astype(np.float32, copy=False)))
        return _PreparedLandmarkSearchIndex(
            index=index,
            landmark_features=landmark_features,
            backend_name="faiss_flat_ip",
            faiss_index=faiss_index,
            torch_features=None,
            max_prototypes_per_track=max_prototypes_per_track,
        )
    if backend == "torch_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda landmark search requires CUDA")
        return _PreparedLandmarkSearchIndex(
            index=index,
            landmark_features=landmark_features,
            backend_name="torch_cuda_exact_ip",
            faiss_index=None,
            torch_features=torch.as_tensor(landmark_features, dtype=torch.float32, device="cuda"),
            max_prototypes_per_track=max_prototypes_per_track,
        )
    return _PreparedLandmarkSearchIndex(
        index=index,
        landmark_features=landmark_features,
        backend_name="exact",
        faiss_index=None,
        torch_features=None,
        max_prototypes_per_track=max_prototypes_per_track,
    )


def _prepared_topk(
    prepared: _PreparedLandmarkSearchIndex,
    query_features: np.ndarray,
    *,
    top_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if prepared.faiss_index is not None:
        scores, indices = prepared.faiss_index.search(
            np.ascontiguousarray(query_features.astype(np.float32, copy=False)),
            int(top_k),
        )
        return indices.astype(np.int64, copy=False), scores.astype(np.float32, copy=False)
    if prepared.torch_features is not None:
        query = torch.as_tensor(query_features, dtype=torch.float32, device=prepared.torch_features.device)
        output_indices: list[np.ndarray] = []
        output_scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(query.shape[0]), max(1, int(block_size))):
                scores = query[start : start + int(block_size)] @ prepared.torch_features.T
                top_scores, top_indices = torch.topk(scores, k=int(top_k), dim=1, largest=True, sorted=True)
                output_indices.append(top_indices.cpu().numpy().astype(np.int64, copy=False))
                output_scores.append(top_scores.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(output_indices, axis=0), np.concatenate(output_scores, axis=0)
    return _exact_topk(
        query_features,
        prepared.landmark_features,
        top_k=int(top_k),
        block_size=int(block_size),
    )


def match_query_tokens_to_landmarks_ann(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    config: LandmarkRetrievalConfig | None = None,
    *,
    image_width: int = 1024,
    image_height: int = 576,
    query_heatmap: np.ndarray | None = None,
    index_cache: LandmarkSearchIndexCache | None = None,
    cache_key: Hashable | None = None,
) -> tuple[list[QueryTo3DMatch], dict[str, Any]]:
    """Retrieve landmark matches with a FAISS/exact backend and the standard match schema."""

    config = config or LandmarkRetrievalConfig()
    if index_cache is None:
        prepared = _prepare_landmark_search_index(landmark_index, config)
        index_cache_hit = False
    else:
        prepared, index_cache_hit = index_cache.get(landmark_index, config, cache_key=cache_key)
    index = prepared.index
    landmark_features = prepared.landmark_features
    metadata: dict[str, Any] = {
        "backend": str(prepared.backend_name),
        "input_landmark_count": int(len(landmark_index)),
        "valid_landmark_count": int(len(index)),
        "query_token_step": int(config.query_token_step),
        "index_cache_enabled": bool(index_cache is not None and index_cache.max_entries > 0),
        "index_cache_hit": bool(index_cache_hit),
    }
    if len(index) == 0:
        metadata["output_match_count"] = 0
        return [], metadata

    query_features, query_xy, token_indices, selection_metadata = select_query_tokens_for_landmark_retrieval(
        query_feature_map,
        image_width=int(image_width),
        image_height=int(image_height),
        config=config,
        query_heatmap=query_heatmap,
    )
    query_features, valid_query = normalize_rows(query_features)
    valid_query_indices = np.flatnonzero(valid_query)
    if valid_query_indices.size == 0 or len(index) == 0:
        metadata.update(selection_metadata)
        metadata.update({"valid_query_token_count": int(valid_query_indices.size), "output_match_count": 0})
        return [], metadata

    query_features = query_features[valid_query_indices]
    query_xy = query_xy[valid_query_indices]
    token_indices = token_indices[valid_query_indices]
    heatmap_values = None
    if query_heatmap is not None:
        heatmap_flat = np.asarray(query_heatmap, dtype=np.float32).reshape(-1)
        heatmap_values = heatmap_flat[token_indices]
    ratio_k = int(config.nn_search_k_for_ratio or config.top_k)
    unique_search_top_k = max(
        int(ratio_k),
        int(config.proposal_top_l),
        2 if config.ratio_threshold is not None else 1,
    )
    search_top_k = min(
        int(unique_search_top_k) * int(prepared.max_prototypes_per_track),
        len(index),
    )
    top_indices, top_scores = _prepared_topk(
        prepared,
        query_features,
        top_k=search_top_k,
        block_size=int(config.block_size),
    )

    matches: list[QueryTo3DMatch] = []
    source = (
        "landmark_faiss"
        if str(prepared.backend_name).startswith("faiss")
        else "landmark_torch_cuda"
        if str(prepared.backend_name).startswith("torch_cuda")
        else "landmark_exact"
    )
    for query_idx in range(query_features.shape[0]):
        candidate_indices = np.asarray(top_indices[query_idx], dtype=np.int64)
        candidate_scores = np.asarray(top_scores[query_idx], dtype=np.float32)
        valid_candidates = candidate_indices >= 0
        if not np.any(valid_candidates):
            continue
        candidate_indices = candidate_indices[valid_candidates]
        candidate_scores = candidate_scores[valid_candidates]
        raw_order = np.argsort(-candidate_scores, kind="mergesort")
        order_values: list[int] = []
        seen_candidate_tracks: set[int] = set()
        for rank_pos in raw_order.tolist():
            landmark_idx = int(candidate_indices[int(rank_pos)])
            track_id = int(index.track_ids[landmark_idx])
            if track_id in seen_candidate_tracks:
                continue
            seen_candidate_tracks.add(track_id)
            order_values.append(int(rank_pos))
            if len(order_values) >= int(unique_search_top_k):
                break
        order = np.asarray(order_values, dtype=np.int64)
        if order.size == 0:
            continue
        best_pos = int(order[0])
        best_similarity = float(candidate_scores[best_pos])
        if best_similarity < float(config.min_similarity):
            continue
        other_scores = candidate_scores[order[1:]]
        ratio = 0.0
        second_similarity = None
        if other_scores.size >= 1:
            second_similarity = float(np.max(other_scores))
            best_distance = max(0.0, 1.0 - best_similarity)
            second_distance = max(1e-6, 1.0 - second_similarity)
            ratio = float(best_distance / second_distance)
            if config.ratio_threshold is not None and ratio > float(config.ratio_threshold):
                continue
        best_similarity_margin = None if second_similarity is None else float(best_similarity - second_similarity)
        if config.min_similarity_margin is not None:
            if best_similarity_margin is None or best_similarity_margin < float(config.min_similarity_margin):
                continue
        xy = query_xy[query_idx].astype(np.float64, copy=True)
        boundary = min(
            float(xy[0]),
            float(xy[1]),
            float(image_width - 1) - float(xy[0]),
            float(image_height - 1) - float(xy[1]),
        )
        for proposal_rank, rank_pos in enumerate(order[: int(config.proposal_top_l)], start=1):
            landmark_idx = int(candidate_indices[int(rank_pos)])
            similarity = float(candidate_scores[int(rank_pos)])
            if similarity < float(config.min_similarity):
                continue
            next_similarity = None
            if proposal_rank < len(order):
                next_similarity = float(candidate_scores[int(order[proposal_rank])])
            similarity_margin = (
                best_similarity_margin
                if proposal_rank == 1
                else None if next_similarity is None else float(similarity - next_similarity)
            )
            matches.append(
                QueryTo3DMatch(
                    token_index=int(token_indices[query_idx]),
                    xy=xy.copy(),
                    track_id=int(index.track_ids[landmark_idx]),
                    xyz=index.xyz[landmark_idx].astype(np.float64, copy=True),
                    similarity=similarity,
                    ratio=ratio if proposal_rank == 1 else 0.0,
                    landmark_variance=float(index.mean_variances[landmark_idx]),
                    source=source,
                    observation_count=int(index.observation_counts[landmark_idx]),
                    visibility_count=len(index.observation_image_ids[landmark_idx]),
                    landmark_reprojection_error=float(index.reprojection_errors[landmark_idx]),
                    landmark_ambiguity=float(index.feature_ambiguities[landmark_idx]),
                    quality_weighted_similarity=similarity,
                    query_heatmap_score=None if heatmap_values is None else float(heatmap_values[query_idx]),
                    similarity_margin=similarity_margin,
                    distance_to_boundary_px=float(boundary),
                    coarse_rank=int(proposal_rank),
                    coarse_score=similarity,
                    coarse_score_gap=similarity_margin,
                    prototype_id=int(index.prototype_ids[landmark_idx]),
                )
            )
    matches.sort(key=lambda item: item.similarity, reverse=True)
    if bool(config.deduplicate_tracks):
        deduped: list[QueryTo3DMatch] = []
        seen_tracks: set[int] = set()
        for match in matches:
            track_id = int(match.track_id)
            if track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
            deduped.append(match)
        matches = deduped
    if config.max_matches is not None:
        matches = matches[: int(config.max_matches)]
    metadata.update(
        {
            "backend": str(prepared.backend_name),
            **selection_metadata,
            "valid_query_token_count": int(query_features.shape[0]),
            "search_top_k": int(search_top_k),
            "unique_track_search_top_k": int(unique_search_top_k),
            "max_prototypes_per_track": int(prepared.max_prototypes_per_track),
            "nn_search_k_for_ratio": int(ratio_k),
            "proposal_top_l": int(config.proposal_top_l),
            "deduplicate_tracks": bool(config.deduplicate_tracks),
            "output_match_count": int(len(matches)),
        }
    )
    return matches, metadata


def _normalized_inverse_penalty(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.ones_like(arr, dtype=np.float32)
    valid = np.maximum(arr[finite], 0.0)
    scale = float(np.percentile(valid, 95.0))
    if scale <= 1e-12:
        scale = float(np.max(valid))
    if scale <= 1e-12:
        out = np.ones_like(arr, dtype=np.float32)
        out[~finite] = 0.0
        return out
    out = 1.0 - np.clip(np.maximum(arr, 0.0) / scale, 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def landmark_submap_quality_scores(index: LandmarkMapIndex) -> np.ndarray:
    """Heuristic landmark quality score for submap truncation."""

    count = len(index)
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    obs = np.log1p(np.maximum(index.observation_counts.astype(np.float32), 0.0))
    obs_scale = float(np.percentile(obs, 95.0))
    if obs_scale <= 1e-12:
        obs_quality = np.ones((count,), dtype=np.float32)
    else:
        obs_quality = np.clip(obs / obs_scale, 0.0, 1.0).astype(np.float32, copy=False)
    variance_quality = _normalized_inverse_penalty(index.mean_variances)
    reprojection_quality = _normalized_inverse_penalty(index.reprojection_errors)
    ambiguity_quality = 1.0 - np.clip(np.asarray(index.feature_ambiguities, dtype=np.float32), 0.0, 1.0)
    score = (
        0.40 * obs_quality
        + 0.25 * variance_quality
        + 0.25 * reprojection_quality
        + 0.10 * ambiguity_quality
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)


def select_quality_spatial_landmark_submap(
    index: LandmarkMapIndex,
    *,
    max_landmarks: int | None,
    grid_rows: int = 8,
    grid_cols: int = 8,
) -> tuple[LandmarkMapIndex, dict[str, Any]]:
    """Limit a submap by landmark quality while keeping spatial coverage in X-Z space."""

    input_count = len(index)
    metadata: dict[str, Any] = {
        "submap_selection_mode": "quality_spatial",
        "input_landmark_count": int(input_count),
        "max_landmarks": None if max_landmarks is None else int(max_landmarks),
        "spatial_grid_rows": int(grid_rows),
        "spatial_grid_cols": int(grid_cols),
    }
    if max_landmarks is None or int(max_landmarks) <= 0 or input_count <= int(max_landmarks):
        metadata.update(
            {
                "output_landmark_count": int(input_count),
                "spatial_cell_count": 0,
                "mean_selected_quality": float(np.mean(landmark_submap_quality_scores(index))) if input_count else 0.0,
            }
        )
        return index, metadata
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")

    scores = landmark_submap_quality_scores(index)
    xyz = np.asarray(index.xyz, dtype=np.float64)
    x = xyz[:, 0]
    z = xyz[:, 2]
    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-6)
    z_span = max(float(np.nanmax(z) - np.nanmin(z)), 1e-6)
    cols = np.clip(np.floor((x - float(np.nanmin(x))) / x_span * int(grid_cols)), 0, int(grid_cols) - 1).astype(int)
    rows = np.clip(np.floor((z - float(np.nanmin(z))) / z_span * int(grid_rows)), 0, int(grid_rows) - 1).astype(int)
    cell_ids = list(zip(rows.tolist(), cols.tolist()))
    ranked = sorted(range(input_count), key=lambda idx: (float(scores[idx]), -idx), reverse=True)
    cell_count = len(set(cell_ids))
    selected: list[int] = []
    selected_set: set[int] = set()
    covered_cells: set[tuple[int, int]] = set()
    for idx in ranked:
        if len(selected) >= int(max_landmarks):
            break
        cell = cell_ids[int(idx)]
        if cell in covered_cells:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        covered_cells.add(cell)
    for idx in ranked:
        if len(selected) >= int(max_landmarks):
            break
        if int(idx) in selected_set:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
    selected_array = np.asarray(selected, dtype=np.int64)
    metadata.update(
        {
            "output_landmark_count": int(selected_array.size),
            "spatial_cell_count": int(cell_count),
            "selected_spatial_cell_count": int(len({cell_ids[int(idx)] for idx in selected})),
            "mean_input_quality": float(np.mean(scores)) if scores.size else 0.0,
            "mean_selected_quality": float(np.mean(scores[selected_array])) if selected_array.size else 0.0,
        }
    )
    return index.subset(selected_array), metadata


def _spatial_cell_count(
    index: LandmarkMapIndex,
    *,
    grid_rows: int,
    grid_cols: int,
    bounds_index: LandmarkMapIndex | None = None,
) -> int:
    if len(index) == 0:
        return 0
    bounds = bounds_index if bounds_index is not None and len(bounds_index) > 0 else index
    xyz = np.asarray(index.xyz, dtype=np.float64)
    bounds_xyz = np.asarray(bounds.xyz, dtype=np.float64)
    x = xyz[:, 0]
    z = xyz[:, 2]
    min_x = float(np.nanmin(bounds_xyz[:, 0]))
    min_z = float(np.nanmin(bounds_xyz[:, 2]))
    x_span = max(float(np.nanmax(bounds_xyz[:, 0]) - min_x), 1e-6)
    z_span = max(float(np.nanmax(bounds_xyz[:, 2]) - min_z), 1e-6)
    cols = np.clip(np.floor((x - min_x) / x_span * int(grid_cols)), 0, int(grid_cols) - 1).astype(int)
    rows = np.clip(np.floor((z - min_z) / z_span * int(grid_rows)), 0, int(grid_rows) - 1).astype(int)
    return int(len(set(zip(rows.tolist(), cols.tolist()))))


def _spatial_cell_ids(
    index: LandmarkMapIndex,
    *,
    grid_rows: int,
    grid_cols: int,
    bounds_index: LandmarkMapIndex | None = None,
) -> list[tuple[int, int]]:
    if len(index) == 0:
        return []
    bounds = bounds_index if bounds_index is not None and len(bounds_index) > 0 else index
    xyz = np.asarray(index.xyz, dtype=np.float64)
    bounds_xyz = np.asarray(bounds.xyz, dtype=np.float64)
    min_x = float(np.nanmin(bounds_xyz[:, 0]))
    min_z = float(np.nanmin(bounds_xyz[:, 2]))
    x_span = max(float(np.nanmax(bounds_xyz[:, 0]) - min_x), 1e-6)
    z_span = max(float(np.nanmax(bounds_xyz[:, 2]) - min_z), 1e-6)
    cols = np.clip(np.floor((xyz[:, 0] - min_x) / x_span * int(grid_cols)), 0, int(grid_cols) - 1).astype(int)
    rows = np.clip(np.floor((xyz[:, 2] - min_z) / z_span * int(grid_rows)), 0, int(grid_rows) - 1).astype(int)
    return list(zip(rows.tolist(), cols.tolist()))


def _concat_landmark_indices(first: LandmarkMapIndex, second: LandmarkMapIndex) -> LandmarkMapIndex:
    if len(first) == 0:
        return second
    if len(second) == 0:
        return first
    return LandmarkMapIndex(
        track_ids=np.concatenate([first.track_ids, second.track_ids], axis=0),
        xyz=np.concatenate([first.xyz, second.xyz], axis=0),
        features=np.concatenate([first.features, second.features], axis=0),
        mean_variances=np.concatenate([first.mean_variances, second.mean_variances], axis=0),
        observation_counts=np.concatenate([first.observation_counts, second.observation_counts], axis=0),
        observation_image_ids=tuple(first.observation_image_ids) + tuple(second.observation_image_ids),
        reprojection_errors=np.concatenate([first.reprojection_errors, second.reprojection_errors], axis=0),
        feature_ambiguities=np.concatenate([first.feature_ambiguities, second.feature_ambiguities], axis=0),
        prototype_ids=np.concatenate([first.prototype_ids, second.prototype_ids], axis=0),
    )


def _select_landmark_submap(
    index: LandmarkMapIndex,
    *,
    max_landmarks: int | None,
    selection_mode: str,
    grid_rows: int,
    grid_cols: int,
) -> tuple[LandmarkMapIndex, dict[str, Any]]:
    if str(selection_mode) == "quality_spatial":
        return select_quality_spatial_landmark_submap(
            index,
            max_landmarks=max_landmarks,
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
        )
    if str(selection_mode) == "lexsort":
        selected = _limit_landmark_index_for_eval(index, max_landmarks)
        return selected, {
            "submap_selection_mode": "lexsort",
            "input_landmark_count": int(len(index)),
            "output_landmark_count": int(len(selected)),
            "max_landmarks": None if max_landmarks is None else int(max_landmarks),
        }
    raise ValueError("submap_selection_mode must be one of: lexsort, quality_spatial")


def select_submap_with_global_fallback(
    candidate_index: LandmarkMapIndex,
    full_index: LandmarkMapIndex,
    *,
    max_landmarks: int | None,
    selection_mode: str = "quality_spatial",
    grid_rows: int = 8,
    grid_cols: int = 8,
    min_landmarks: int = 0,
    min_spatial_cells: int = 0,
    fallback_fraction: float = 0.25,
) -> tuple[LandmarkMapIndex, dict[str, Any]]:
    """Select a retrieval submap and reserve global high-quality landmarks when coverage is weak."""

    selected, metadata = _select_landmark_submap(
        candidate_index,
        max_landmarks=max_landmarks,
        selection_mode=str(selection_mode),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )
    selected_cells = _spatial_cell_count(
        selected,
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        bounds_index=full_index,
    )
    needs_fallback = (
        (int(min_landmarks) > 0 and len(selected) < int(min_landmarks))
        or (int(min_spatial_cells) > 0 and selected_cells < int(min_spatial_cells))
    )
    metadata.update(
        {
            "fallback_applied": False,
            "fallback_added_landmark_count": 0,
            "selected_spatial_cell_count": int(selected_cells),
            "min_submap_landmarks": int(min_landmarks),
            "min_submap_spatial_cells": int(min_spatial_cells),
            "submap_fallback_fraction": float(fallback_fraction),
        }
    )
    if not needs_fallback or len(full_index) == 0:
        return selected, metadata

    limit = len(full_index) if max_landmarks is None or int(max_landmarks) <= 0 else int(max_landmarks)
    if limit <= 0:
        return selected, metadata
    reserve = int(np.ceil(float(limit) * max(0.0, float(fallback_fraction))))
    reserve = max(1, min(reserve, limit)) if float(fallback_fraction) > 0.0 else max(0, limit - len(selected))
    keep_limit = max(0, limit - reserve)
    base = selected
    if len(base) > keep_limit:
        base, _base_meta = _select_landmark_submap(
            candidate_index,
            max_landmarks=keep_limit,
            selection_mode=str(selection_mode),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
        )
    remaining = max(0, limit - len(base))
    if remaining <= 0:
        metadata["fallback_applied"] = True
        metadata["output_landmark_count"] = int(len(base))
        metadata["selected_spatial_cell_count"] = _spatial_cell_count(
            base,
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            bounds_index=full_index,
        )
        return base, metadata

    existing = {int(track_id) for track_id in base.track_ids.tolist()}
    fallback_mask = np.asarray([int(track_id) not in existing for track_id in full_index.track_ids.tolist()], dtype=bool)
    fallback_source = full_index.subset(fallback_mask)
    fallback_scores = landmark_submap_quality_scores(fallback_source)
    fallback_cells = _spatial_cell_ids(
        fallback_source,
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        bounds_index=full_index,
    )
    covered_cells = set(
        _spatial_cell_ids(
            base,
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            bounds_index=full_index,
        )
    )
    ranked_fallback = sorted(range(len(fallback_source)), key=lambda idx: (float(fallback_scores[idx]), -idx), reverse=True)
    fallback_indices: list[int] = []
    fallback_index_set: set[int] = set()
    for idx in ranked_fallback:
        if len(fallback_indices) >= remaining:
            break
        cell = fallback_cells[int(idx)]
        if cell in covered_cells:
            continue
        fallback_indices.append(int(idx))
        fallback_index_set.add(int(idx))
        covered_cells.add(cell)
    for idx in ranked_fallback:
        if len(fallback_indices) >= remaining:
            break
        if int(idx) in fallback_index_set:
            continue
        fallback_indices.append(int(idx))
        fallback_index_set.add(int(idx))
    fallback = fallback_source.subset(np.asarray(fallback_indices, dtype=np.int64))
    merged = _concat_landmark_indices(base, fallback)
    metadata.update(
        {
            "fallback_applied": True,
            "fallback_added_landmark_count": int(len(fallback)),
            "output_landmark_count": int(len(merged)),
            "selected_spatial_cell_count": _spatial_cell_count(
                merged,
                grid_rows=int(grid_rows),
                grid_cols=int(grid_cols),
                bounds_index=full_index,
            ),
        }
    )
    return merged, metadata


def _score_for_match(match: QueryTo3DMatch) -> float:
    if match.pnp_soft_score is not None:
        return float(match.pnp_soft_score)
    return float(match.similarity)


def _bounded_score(value: float | None, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(score):
        return float(default)
    return float(np.clip(score, 0.0, 1.0))


def _track_quality_for_match(match: QueryTo3DMatch) -> float:
    observation_count = 1 if match.observation_count is None else max(int(match.observation_count), 1)
    observation_quality = float(np.clip(np.log1p(float(observation_count)) / np.log1p(20.0), 0.0, 1.0))
    variance_quality = 1.0 / (1.0 + max(float(match.landmark_variance), 0.0))
    reprojection = 0.0 if match.landmark_reprojection_error is None else max(float(match.landmark_reprojection_error), 0.0)
    reprojection_quality = 1.0 / (1.0 + reprojection)
    ambiguity_quality = 1.0 - _bounded_score(match.landmark_ambiguity, default=0.0)
    return float(
        np.clip(
            0.35 * observation_quality
            + 0.25 * variance_quality
            + 0.25 * reprojection_quality
            + 0.15 * ambiguity_quality,
            0.0,
            1.0,
        )
    )


def landmark_match_measurement_quality(match: QueryTo3DMatch) -> float:
    """Score a match for measurement/PnP using retrieval, selector, landmark and measurement cues."""

    if match.geometry_probability is not None:
        return _bounded_score(match.geometry_probability, default=0.0)
    components: list[tuple[float, float]] = [
        (0.45, _bounded_score(match.similarity, default=0.0)),
        (0.25, _bounded_score(match.query_heatmap_score, default=0.5)),
        (0.30, _track_quality_for_match(match)),
    ]
    if match.patch_offset_confidence is not None:
        components.append((0.30, _bounded_score(match.patch_offset_confidence, default=0.0)))
    if match.measurement_sigma_px is not None:
        sigma = max(float(match.measurement_sigma_px), 0.0)
        components.append((0.15, float(1.0 / (1.0 + sigma / 4.0))))
    total_weight = max(float(sum(weight for weight, _score in components)), 1e-6)
    score = sum(float(weight) * float(value) for weight, value in components) / total_weight
    return float(np.clip(score, 0.0, 1.0))


def rescore_landmark_matches_for_measurement(matches: Sequence[QueryTo3DMatch]) -> list[QueryTo3DMatch]:
    """Attach combined confidence scores used by measurement ranking and weighted PnP."""

    rescored = []
    for match in matches:
        score = landmark_match_measurement_quality(match)
        rescored.append(
            replace(
                match,
                pnp_soft_score=score,
                quality_weighted_similarity=score,
            )
        )
    rescored.sort(key=lambda item: float(item.pnp_soft_score or item.similarity), reverse=True)
    return rescored


def _with_measurement_quality_score(match: QueryTo3DMatch) -> QueryTo3DMatch:
    score = landmark_match_measurement_quality(match)
    return replace(match, pnp_soft_score=score, quality_weighted_similarity=score)


def select_measurement_candidates_spatially(
    matches: Sequence[QueryTo3DMatch],
    *,
    image_width: int,
    image_height: int,
    max_count: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    score_mode: str = "match",
) -> tuple[list[QueryTo3DMatch], list[QueryTo3DMatch]]:
    """Pick high-score measurement candidates while covering query image cells."""

    values = list(matches)
    mode = str(score_mode)
    if mode not in {"match", "measurement_quality"}:
        raise ValueError("score_mode must be 'match' or 'measurement_quality'")
    if int(max_count) <= 0 or len(values) <= int(max_count):
        return values, []
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image size must be positive")

    def candidate_score(match: QueryTo3DMatch) -> float:
        if mode == "measurement_quality":
            return landmark_match_measurement_quality(match)
        return _score_for_match(match)

    ranked = sorted(enumerate(values), key=lambda item: (candidate_score(item[1]), -item[0]), reverse=True)
    cell_count = int(grid_rows) * int(grid_cols)
    per_cell_quota = max(1, int(np.ceil(float(max_count) / float(cell_count))))
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    counts: dict[tuple[int, int], int] = {}

    def cell_for(match: QueryTo3DMatch) -> tuple[int, int]:
        xy = np.asarray(match.xy, dtype=np.float64).reshape(2)
        col = int(np.clip(np.floor(float(xy[0]) / max(1.0, float(image_width)) * int(grid_cols)), 0, int(grid_cols) - 1))
        row = int(np.clip(np.floor(float(xy[1]) / max(1.0, float(image_height)) * int(grid_rows)), 0, int(grid_rows) - 1))
        return row, col

    for original_idx, match in ranked:
        if len(selected_indices) >= int(max_count):
            break
        cell = cell_for(match)
        if counts.get(cell, 0) >= per_cell_quota:
            continue
        selected_indices.append(int(original_idx))
        selected_set.add(int(original_idx))
        counts[cell] = counts.get(cell, 0) + 1

    for original_idx, _match in ranked:
        if len(selected_indices) >= int(max_count):
            break
        if int(original_idx) in selected_set:
            continue
        selected_indices.append(int(original_idx))
        selected_set.add(int(original_idx))

    selected = [values[idx] for idx in selected_indices]
    skipped = [match for idx, match in enumerate(values) if idx not in selected_set]
    return selected, skipped


def select_measurement_candidates_from_inlier_mask(
    matches: Sequence[QueryTo3DMatch],
    *,
    inlier_mask: np.ndarray | Sequence[bool],
    image_width: int,
    image_height: int,
    max_count: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    score_mode: str = "match",
) -> tuple[list[QueryTo3DMatch], list[QueryTo3DMatch], dict[str, Any]]:
    values = list(matches)
    if int(max_count) <= 0 or len(values) <= int(max_count):
        return values, [], {
            "measurement_selection_strategy": "coarse_pnp_inliers",
            "coarse_pnp_inlier_count": int(np.count_nonzero(np.asarray(inlier_mask, dtype=bool))),
            "measurement_filled_from_non_inliers": 0,
        }
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(values):
        raise ValueError("inlier_mask length must match matches length")
    inliers = [match for match, keep in zip(values, mask) if bool(keep)]
    non_inliers = [match for match, keep in zip(values, mask) if not bool(keep)]
    selected: list[QueryTo3DMatch] = []
    selected_ids: set[int] = set()
    if inliers:
        inlier_selected, _inlier_skipped = select_measurement_candidates_spatially(
            inliers,
            image_width=int(image_width),
            image_height=int(image_height),
            max_count=min(int(max_count), len(inliers)),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            score_mode=str(score_mode),
        )
        selected.extend(inlier_selected)
        selected_ids.update(id(match) for match in inlier_selected)
    fill_count = max(0, int(max_count) - len(selected))
    filled = 0
    if fill_count:
        fill_selected, _fill_skipped = select_measurement_candidates_spatially(
            [match for match in non_inliers if id(match) not in selected_ids],
            image_width=int(image_width),
            image_height=int(image_height),
            max_count=fill_count,
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            score_mode=str(score_mode),
        )
        selected.extend(fill_selected)
        selected_ids.update(id(match) for match in fill_selected)
        filled = int(len(fill_selected))
    skipped = [match for match in values if id(match) not in selected_ids]
    metadata = {
        "measurement_selection_strategy": "coarse_pnp_inliers",
        "coarse_pnp_inlier_count": int(np.count_nonzero(mask)),
        "measurement_filled_from_non_inliers": int(filled),
    }
    return selected, skipped, metadata


def select_measurement_candidates_after_coarse_pnp(
    matches: Sequence[QueryTo3DMatch],
    *,
    camera: ColmapCamera,
    image_width: int,
    image_height: int,
    max_count: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    score_mode: str = "match",
    pnp_reprojection_error_px: float = 8.0,
    pnp_iterations: int = 1000,
    pnp_confidence: float = 0.999,
    pnp_min_inliers: int = 4,
) -> tuple[list[QueryTo3DMatch], list[QueryTo3DMatch], dict[str, Any]]:
    values = list(matches)
    if len(values) < max(4, int(pnp_min_inliers)):
        selected, skipped = select_measurement_candidates_spatially(
            values,
            image_width=int(image_width),
            image_height=int(image_height),
            max_count=int(max_count),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            score_mode=str(score_mode),
        )
        return selected, skipped, {
            "measurement_selection_strategy": "score_spatial_fallback",
            "coarse_pnp_success": False,
            "coarse_pnp_inlier_count": 0,
            "measurement_filled_from_non_inliers": 0,
        }
    pnp = estimate_pose_pnp_ransac(
        values,
        camera,
        reprojection_error_px=float(pnp_reprojection_error_px),
        confidence=float(pnp_confidence),
        iterations=int(pnp_iterations),
        min_inliers=int(pnp_min_inliers),
        refine_method="LM",
    )
    if not pnp.success:
        selected, skipped = select_measurement_candidates_spatially(
            values,
            image_width=int(image_width),
            image_height=int(image_height),
            max_count=int(max_count),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            score_mode=str(score_mode),
        )
        return selected, skipped, {
            "measurement_selection_strategy": "score_spatial_fallback",
            "coarse_pnp_success": False,
            "coarse_pnp_inlier_count": int(pnp.inlier_count),
            "measurement_filled_from_non_inliers": 0,
        }
    values_for_selection = values
    if pnp.pose_w2c is not None:
        residuals = match_reprojection_errors(values, pnp.pose_w2c, camera)
        threshold = max(float(pnp_reprojection_error_px), 1e-6)
        values_for_selection = [
            replace(
                match,
                local_consistency_score=float(1.0 / (1.0 + max(float(residual), 0.0) / threshold)),
                patch_offset_consistency_before_px=float(residual),
            )
            for match, residual in zip(values, residuals)
        ]
    selected, skipped, metadata = select_measurement_candidates_from_inlier_mask(
        values_for_selection,
        inlier_mask=pnp.inlier_mask,
        image_width=int(image_width),
        image_height=int(image_height),
        max_count=int(max_count),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        score_mode=str(score_mode),
    )
    metadata["coarse_pnp_success"] = True
    return selected, skipped, metadata


def _proposal_from_landmark_match(match: QueryTo3DMatch, owner: LandmarkOwnerObservation, rank: int) -> CoarseProposal:
    return CoarseProposal(
        query_index=int(match.token_index),
        reference_index=int(rank),
        query_xy=np.asarray(match.xy, dtype=np.float32),
        reference_xy=np.asarray(owner.xy, dtype=np.float32),
        score=float(match.similarity),
        confidence=float(_score_for_match(match)),
        rank=int(rank),
        metadata={"track_id": int(match.track_id), "owner_image_id": owner.image_id},
    )


def _match_with_measurement_verification(
    base_match: QueryTo3DMatch,
    measurement: MeasurementResult,
    *,
    min_measurement_confidence: float | None = None,
    max_measurement_uncertainty_px: float | None = None,
    measurement_geometry_model: GeometryProbabilityModel | None = None,
    min_measurement_geometry_probability: float | None = None,
    drop_rejected_measurements: bool = False,
) -> tuple[QueryTo3DMatch | None, str, bool, bool, float | None]:
    confidence = measurement.confidence if measurement.confidence is not None else _score_for_match(base_match)
    candidate = replace(
        base_match,
        patch_offset_confidence=measurement.confidence,
        measurement_sigma_px=measurement.uncertainty_px,
    )
    geometry_probability = None
    if measurement_geometry_model is not None:
        geometry_probability = float(measurement_geometry_model.predict_match(candidate))
        candidate = replace(
            candidate,
            geometry_probability=geometry_probability,
            pnp_soft_score=geometry_probability,
            quality_weighted_similarity=geometry_probability,
        )

    reject_reason = ""
    if min_measurement_confidence is not None and float(confidence) < float(min_measurement_confidence):
        reject_reason = "rejected_low_confidence"
    if (
        not reject_reason
        and max_measurement_uncertainty_px is not None
        and measurement.uncertainty_px is not None
        and float(measurement.uncertainty_px) > float(max_measurement_uncertainty_px)
    ):
        reject_reason = "rejected_high_uncertainty"
    if (
        not reject_reason
        and min_measurement_geometry_probability is not None
        and geometry_probability is not None
        and float(geometry_probability) < float(min_measurement_geometry_probability)
    ):
        reject_reason = "rejected_low_geometry_probability"

    if reject_reason:
        updated = candidate
        measurement_applied = False
    else:
        updated = replace(candidate, xy=np.asarray(measurement.measured_query_xy, dtype=np.float64).reshape(2))
        measurement_applied = True

    if measurement_geometry_model is None:
        updated = _with_measurement_quality_score(updated)

    measurement_kept = not (bool(reject_reason) and bool(drop_rejected_measurements))
    return (updated if measurement_kept else None), reject_reason, measurement_applied, measurement_kept, geometry_probability


def refine_landmark_matches_with_measurement(
    *,
    query_id: str,
    query_rgb: np.ndarray,
    matches: Sequence[QueryTo3DMatch],
    owner_index: LandmarkOwnerObservationIndex,
    measurement_adapter,
    reference_image_loader: Callable[[str], np.ndarray],
    min_measurement_confidence: float | None = None,
    max_measurement_uncertainty_px: float | None = None,
    measurement_geometry_model: GeometryProbabilityModel | None = None,
    min_measurement_geometry_probability: float | None = None,
    drop_rejected_measurements: bool = False,
) -> tuple[list[QueryTo3DMatch], list[dict[str, Any]]]:
    """Refine query coordinates for landmark matches using real owner-view RGB patches."""

    refined: list[QueryTo3DMatch] = []
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[int, QueryTo3DMatch, LandmarkOwnerObservation, CoarseProposal]]] = {}
    for rank, match in enumerate(matches):
        owner = owner_index.select(int(match.track_id), exclude_image_id=str(query_id))
        if owner is None:
            refined.append(match)
            rows.append(
                {
                    "query_id": str(query_id),
                    "track_id": int(match.track_id),
                    "measurement_status": "missing_owner_observation",
                    "query_x_before": float(np.asarray(match.xy).reshape(2)[0]),
                    "query_y_before": float(np.asarray(match.xy).reshape(2)[1]),
                }
            )
            continue
        proposal = _proposal_from_landmark_match(match, owner, rank)
        grouped.setdefault(owner.image_id, []).append((rank, match, owner, proposal))

    measured_by_rank: dict[int, tuple[QueryTo3DMatch, LandmarkOwnerObservation, MeasurementResult]] = {}
    measure_by_reference = getattr(measurement_adapter, "measure_by_reference", None)
    if callable(measure_by_reference):
        reference_rgb_by_id = {str(image_id): reference_image_loader(str(image_id)) for image_id in grouped}
        proposals_by_reference = {
            str(image_id): [item[3] for item in items]
            for image_id, items in grouped.items()
        }
        measurements_by_reference = measure_by_reference(query_rgb, reference_rgb_by_id, proposals_by_reference)
        for image_id, items in grouped.items():
            measurements = list(measurements_by_reference.get(str(image_id), []))
            proposals = proposals_by_reference[str(image_id)]
            if len(measurements) != len(proposals):
                raise ValueError("measurement adapter must return one result per proposal")
            for (rank, match, owner, _proposal), measurement in zip(items, measurements):
                measured_by_rank[int(rank)] = (match, owner, measurement)
    else:
        for image_id, items in grouped.items():
            reference_rgb = reference_image_loader(str(image_id))
            proposals = [item[3] for item in items]
            measurements = measurement_adapter.measure(query_rgb, reference_rgb, proposals)
            if len(measurements) != len(proposals):
                raise ValueError("measurement adapter must return one result per proposal")
            for (rank, match, owner, _proposal), measurement in zip(items, measurements):
                measured_by_rank[int(rank)] = (match, owner, measurement)

    for rank, match in enumerate(matches):
        measured = measured_by_rank.get(int(rank))
        if measured is None:
            if owner_index.select(int(match.track_id), exclude_image_id=str(query_id)) is not None:
                refined.append(match)
            continue
        base_match, owner, measurement = measured
        updated, reject_reason, measurement_applied, measurement_kept, geometry_probability = _match_with_measurement_verification(
            base_match,
            measurement,
            min_measurement_confidence=min_measurement_confidence,
            max_measurement_uncertainty_px=max_measurement_uncertainty_px,
            measurement_geometry_model=measurement_geometry_model,
            min_measurement_geometry_probability=min_measurement_geometry_probability,
            drop_rejected_measurements=drop_rejected_measurements,
        )
        if updated is not None:
            refined.append(updated)
        before = np.asarray(base_match.xy, dtype=np.float64).reshape(2)
        after = before if updated is None else np.asarray(updated.xy, dtype=np.float64).reshape(2)
        rows.append(
            {
                "query_id": str(query_id),
                "track_id": int(base_match.track_id),
                "owner_image_id": owner.image_id,
                "measurement_status": reject_reason or "measured",
                "query_x_before": float(before[0]),
                "query_y_before": float(before[1]),
                "query_x_after": float(after[0]),
                "query_y_after": float(after[1]),
                "owner_x": float(owner.xy[0]),
                "owner_y": float(owner.xy[1]),
                "confidence": None if measurement.confidence is None else float(measurement.confidence),
                "uncertainty_px": None if measurement.uncertainty_px is None else float(measurement.uncertainty_px),
                "geometry_probability": None if geometry_probability is None else float(geometry_probability),
                "measurement_applied": bool(measurement_applied),
                "measurement_kept": bool(measurement_kept),
            }
        )
    return refined, rows


def refine_landmark_match_batches_with_measurement(
    query_batches: Sequence[tuple[str, np.ndarray, Sequence[QueryTo3DMatch]]],
    *,
    owner_index: LandmarkOwnerObservationIndex,
    measurement_adapter,
    reference_image_loader: Callable[[str], np.ndarray],
    min_measurement_confidence: float | None = None,
    max_measurement_uncertainty_px: float | None = None,
    measurement_geometry_model: GeometryProbabilityModel | None = None,
    min_measurement_geometry_probability: float | None = None,
    drop_rejected_measurements: bool = False,
) -> dict[str, tuple[list[QueryTo3DMatch], list[dict[str, Any]]]]:
    """Refine multiple queries together when the adapter supports cross-query measurement batching."""

    measure_many = getattr(measurement_adapter, "measure_many_by_reference", None)
    if not callable(measure_many):
        return {
            str(query_id): refine_landmark_matches_with_measurement(
                query_id=str(query_id),
                query_rgb=query_rgb,
                matches=matches,
                owner_index=owner_index,
                measurement_adapter=measurement_adapter,
                reference_image_loader=reference_image_loader,
                min_measurement_confidence=min_measurement_confidence,
                max_measurement_uncertainty_px=max_measurement_uncertainty_px,
                measurement_geometry_model=measurement_geometry_model,
                min_measurement_geometry_probability=min_measurement_geometry_probability,
                drop_rejected_measurements=drop_rejected_measurements,
            )
            for query_id, query_rgb, matches in query_batches
        }

    query_rgb_by_id = {str(query_id): query_rgb for query_id, query_rgb, _matches in query_batches}
    rows_by_query: dict[str, list[dict[str, Any]]] = {str(query_id): [] for query_id, _query_rgb, _matches in query_batches}
    measured_by_query_rank: dict[tuple[str, int], tuple[QueryTo3DMatch, LandmarkOwnerObservation, MeasurementResult]] = {}
    proposals_by_pair: dict[tuple[str, str], list[CoarseProposal]] = {}
    entries_by_pair: dict[tuple[str, str], list[tuple[int, QueryTo3DMatch, LandmarkOwnerObservation, CoarseProposal]]] = {}
    reference_ids: set[str] = set()

    for query_id, _query_rgb, matches in query_batches:
        qid = str(query_id)
        for rank, match in enumerate(matches):
            owner = owner_index.select(int(match.track_id), exclude_image_id=qid)
            if owner is None:
                rows_by_query[qid].append(
                    {
                        "query_id": qid,
                        "track_id": int(match.track_id),
                        "measurement_status": "missing_owner_observation",
                        "query_x_before": float(np.asarray(match.xy).reshape(2)[0]),
                        "query_y_before": float(np.asarray(match.xy).reshape(2)[1]),
                    }
                )
                continue
            proposal = _proposal_from_landmark_match(match, owner, rank)
            pair = (qid, str(owner.image_id))
            proposals_by_pair.setdefault(pair, []).append(proposal)
            entries_by_pair.setdefault(pair, []).append((rank, match, owner, proposal))
            reference_ids.add(str(owner.image_id))

    reference_rgb_by_id = {image_id: reference_image_loader(image_id) for image_id in sorted(reference_ids)}
    measurements_by_pair = measure_many(query_rgb_by_id, reference_rgb_by_id, proposals_by_pair)
    for pair, entries in entries_by_pair.items():
        measurements = list(measurements_by_pair.get(pair, []))
        if len(measurements) != len(entries):
            raise ValueError("measurement adapter must return one result per proposal")
        for (rank, match, owner, _proposal), measurement in zip(entries, measurements):
            measured_by_query_rank[(str(pair[0]), int(rank))] = (match, owner, measurement)

    output: dict[str, tuple[list[QueryTo3DMatch], list[dict[str, Any]]]] = {}
    for query_id, _query_rgb, matches in query_batches:
        qid = str(query_id)
        refined: list[QueryTo3DMatch] = []
        rows = list(rows_by_query.get(qid, []))
        for rank, match in enumerate(matches):
            measured = measured_by_query_rank.get((qid, int(rank)))
            if measured is None:
                refined.append(match)
                continue
            base_match, owner, measurement = measured
            updated, reject_reason, measurement_applied, measurement_kept, geometry_probability = _match_with_measurement_verification(
                base_match,
                measurement,
                min_measurement_confidence=min_measurement_confidence,
                max_measurement_uncertainty_px=max_measurement_uncertainty_px,
                measurement_geometry_model=measurement_geometry_model,
                min_measurement_geometry_probability=min_measurement_geometry_probability,
                drop_rejected_measurements=drop_rejected_measurements,
            )
            if updated is not None:
                refined.append(updated)
            before = np.asarray(base_match.xy, dtype=np.float64).reshape(2)
            after = before if updated is None else np.asarray(updated.xy, dtype=np.float64).reshape(2)
            rows.append(
                {
                    "query_id": qid,
                    "track_id": int(base_match.track_id),
                    "owner_image_id": owner.image_id,
                    "measurement_status": reject_reason or "measured",
                    "query_x_before": float(before[0]),
                    "query_y_before": float(before[1]),
                    "query_x_after": float(after[0]),
                    "query_y_after": float(after[1]),
                    "owner_x": float(owner.xy[0]),
                    "owner_y": float(owner.xy[1]),
                    "confidence": None if measurement.confidence is None else float(measurement.confidence),
                    "uncertainty_px": None if measurement.uncertainty_px is None else float(measurement.uncertainty_px),
                    "geometry_probability": None if geometry_probability is None else float(geometry_probability),
                    "measurement_applied": bool(measurement_applied),
                    "measurement_kept": bool(measurement_kept),
                }
            )
        output[qid] = (refined, rows)
    return output


def _query_match_rows(query_id: str, matches: Sequence[QueryTo3DMatch]) -> list[dict[str, Any]]:
    rows = []
    for index, match in enumerate(matches):
        rows.append(
            {
                "query_id": str(query_id),
                "match_index": int(index),
                "source": str(match.source),
                "token_index": int(match.token_index),
                "track_id": int(match.track_id),
                "prototype_id": None if match.prototype_id is None else int(match.prototype_id),
                "x": float(np.asarray(match.xy).reshape(2)[0]),
                "y": float(np.asarray(match.xy).reshape(2)[1]),
                "xyz": [float(value) for value in np.asarray(match.xyz).reshape(3).tolist()],
                "similarity": float(match.similarity),
                "ratio": float(match.ratio),
                "similarity_margin": match.similarity_margin,
                "observation_count": match.observation_count,
                "visibility_count": match.visibility_count,
                "landmark_reprojection_error": match.landmark_reprojection_error,
                "landmark_ambiguity": match.landmark_ambiguity,
                "query_heatmap_score": None if match.query_heatmap_score is None else float(match.query_heatmap_score),
                "quality_weighted_similarity": None
                if match.quality_weighted_similarity is None
                else float(match.quality_weighted_similarity),
                "pnp_soft_score": None if match.pnp_soft_score is None else float(match.pnp_soft_score),
                "patch_offset_confidence": None
                if match.patch_offset_confidence is None
                else float(match.patch_offset_confidence),
                "measurement_sigma_px": None if match.measurement_sigma_px is None else float(match.measurement_sigma_px),
                "geometry_probability": None
                if match.geometry_probability is None
                else float(match.geometry_probability),
                "local_consistency_score": None
                if match.local_consistency_score is None
                else float(match.local_consistency_score),
                "patch_offset_consistency_before_px": None
                if match.patch_offset_consistency_before_px is None
                else float(match.patch_offset_consistency_before_px),
            }
        )
    return rows


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def cameras_by_image_name(
    *,
    cameras: Mapping[int, ColmapCamera],
    colmap_images: Mapping[int, ColmapImageObservation],
    image_root: Path,
) -> dict[str, ColmapCamera]:
    out: dict[str, ColmapCamera] = {}
    for image in colmap_images.values():
        camera = cameras.get(int(image.camera_id))
        if camera is None:
            continue
        image_path = Path(image_root) / image.image_name
        out[str(image.image_name)] = (
            scaled_colmap_camera(camera, image_width=_image_size(image_path)[0], image_height=_image_size(image_path)[1])
            if image_path.exists()
            else camera
        )
    return out


def target_image_sizes_from_observations(
    observations: Sequence[ColmapTrackObservation],
    *,
    image_root: Path,
) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for observation in observations:
        image_id = str(observation.image_id)
        if image_id in sizes:
            continue
        path = Path(image_root) / image_id
        if path.exists():
            sizes[image_id] = _image_size(path)
    return sizes


def _load_query_feature(record: TokenBankRecord, feature_key: str) -> np.ndarray:
    return _load_feature_map(Path(record.token_path), key=str(feature_key))


def _limit_landmark_index_for_eval(index: LandmarkMapIndex, max_landmarks: int | None) -> LandmarkMapIndex:
    if max_landmarks is None or int(max_landmarks) <= 0 or len(index) <= int(max_landmarks):
        return index
    order = np.lexsort((-index.observation_counts, index.mean_variances))
    return index.subset(order[: int(max_landmarks)])


def _submap_cache_key(
    *,
    query_id: str,
    references: Sequence[str],
    selection_mode: str,
    max_submap_landmarks: int | None,
    grid_rows: int,
    grid_cols: int,
    min_landmarks: int = 0,
    min_spatial_cells: int = 0,
    fallback_fraction: float = 0.0,
) -> Hashable:
    if references:
        base: Hashable = ("references", tuple(str(item) for item in references))
    else:
        base = ("full_bank",)
    return (
        base,
        str(selection_mode),
        None if max_submap_landmarks is None else int(max_submap_landmarks),
        int(grid_rows),
        int(grid_cols),
        int(min_landmarks),
        int(min_spatial_cells),
        round(float(fallback_fraction), 6),
    )


def run_real_radio_landmark_hybrid_eval(
    query_records: Sequence[TokenBankRecord],
    *,
    landmark_index: LandmarkMapIndex,
    output_dir: Path,
    feature_mapper,
    cameras_by_query: Mapping[str, ColmapCamera],
    gt_poses_by_query: Mapping[str, CambridgePoseRecord],
    image_root: Path,
    feature_key: str = "radio_final",
    matching_config: QueryTo3DMatchingConfig = QueryTo3DMatchingConfig(),
    retrieval_config: LandmarkRetrievalConfig | None = None,
    retrieval_index_cache_size: int = 64,
    reference_submaps_by_query: Mapping[str, Sequence[str]] | None = None,
    max_submap_landmarks: int | None = None,
    submap_selection_mode: str = "lexsort",
    submap_spatial_grid_rows: int = 8,
    submap_spatial_grid_cols: int = 8,
    submap_min_landmarks: int = 0,
    submap_min_spatial_cells: int = 0,
    submap_fallback_fraction: float = 0.25,
    measurement_adapter=None,
    owner_index: LandmarkOwnerObservationIndex | None = None,
    measurement_max_matches: int | None = None,
    measurement_selection_strategy: str = "score_spatial",
    measurement_query_batch_size: int = 1,
    measurement_grid_rows: int = 4,
    measurement_grid_cols: int = 4,
    measurement_score_mode: str = "match",
    measurement_final_match_policy: str = "keep_all",
    min_measurement_confidence: float | None = None,
    max_measurement_uncertainty_px: float | None = None,
    measurement_geometry_model: GeometryProbabilityModel | None = None,
    min_measurement_geometry_probability: float | None = None,
    drop_rejected_measurements: bool = False,
    enable_quality_rescore: bool = False,
    pnp_min_soft_score: float | None = None,
    reference_rgb_cache_size: int = 32,
    max_queries: int | None = None,
    pnp_reprojection_error_px: float = 8.0,
    pnp_iterations: int = 1000,
    pnp_confidence: float = 0.999,
    pnp_min_inliers: int = 4,
    pnp_weighted_refine: bool = False,
    pnp_weighted_loss: str = "huber",
    pnp_weighted_f_scale_px: float = 4.0,
    pnp_weighted_max_nfev: int = 50,
    progress_interval_queries: int = 0,
    evaluate_pose: bool = True,
) -> dict[str, Any]:
    records = list(query_records)
    if max_queries is not None:
        records = records[: int(max_queries)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if bool(evaluate_pose) and retrieval_config is not None and int(retrieval_config.proposal_top_l) > 1:
        raise ValueError(
            "unresolved top-L proposals cannot be passed to ordinary PnP; "
            "run proposal-only or resolve each query-token group first"
        )

    matches_by_query: dict[str, list[QueryTo3DMatch]] = {}
    match_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    measurement_selection_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    measurement_candidate_counts: list[int] = []
    submap_counts: list[int] = []
    submap_pre_limit_counts: list[int] = []
    submap_reference_counts: list[int] = []
    submap_empty_fallback_count = 0
    reference_rgb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    retrieval_index_cache = (
        LandmarkSearchIndexCache(max_entries=int(retrieval_index_cache_size))
        if retrieval_config is not None and int(retrieval_index_cache_size) >= 0
        else None
    )
    pending_measurement: list[dict[str, Any]] = []
    final_policy = str(measurement_final_match_policy)
    if final_policy not in {"keep_all", "measured_only"}:
        raise ValueError("measurement_final_match_policy must be 'keep_all' or 'measured_only'")

    def load_reference_rgb_cached(image_id: str) -> np.ndarray:
        key = str(image_id)
        cached = reference_rgb_cache.get(key)
        if cached is not None:
            reference_rgb_cache.move_to_end(key)
            return cached
        image = _load_rgb_chw(Path(image_root) / key)
        if int(reference_rgb_cache_size) > 0:
            reference_rgb_cache[key] = image
            reference_rgb_cache.move_to_end(key)
            while len(reference_rgb_cache) > int(reference_rgb_cache_size):
                reference_rgb_cache.popitem(last=False)
        return image

    def store_query_matches(query_id: str, matches: Sequence[QueryTo3DMatch]) -> None:
        values = list(matches)
        if pnp_min_soft_score is not None:
            threshold = float(pnp_min_soft_score)
            values = [
                match
                for match in values
                if match.pnp_soft_score is None or float(match.pnp_soft_score) >= threshold
            ]
        matches_by_query[str(query_id)] = list(values)
        match_rows.extend(_query_match_rows(str(query_id), values))

    def flush_pending_measurement() -> None:
        if not pending_measurement:
            return
        batch_inputs = [
            (str(item["query_id"]), item["query_rgb"], item["measurement_input"])
            for item in pending_measurement
        ]
        refined_by_query = refine_landmark_match_batches_with_measurement(
            batch_inputs,
            owner_index=owner_index,
            measurement_adapter=measurement_adapter,
            reference_image_loader=load_reference_rgb_cached,
            min_measurement_confidence=min_measurement_confidence,
            max_measurement_uncertainty_px=max_measurement_uncertainty_px,
            measurement_geometry_model=measurement_geometry_model,
            min_measurement_geometry_probability=min_measurement_geometry_probability,
            drop_rejected_measurements=drop_rejected_measurements,
        )
        for item in pending_measurement:
            query_id = str(item["query_id"])
            refined_measurement_matches, rows = refined_by_query.get(query_id, ([], []))
            if bool(enable_quality_rescore):
                refined_measurement_matches = rescore_landmark_matches_for_measurement(refined_measurement_matches)
            measurement_skipped = list(item["measurement_skipped"])
            if final_policy == "measured_only":
                matches = list(refined_measurement_matches)
            elif measurement_skipped:
                matches = sorted(
                    list(refined_measurement_matches) + measurement_skipped,
                    key=_score_for_match,
                    reverse=True,
                )
            else:
                matches = refined_measurement_matches
            for row in rows:
                row["measurement_candidate_count"] = int(item["measurement_candidate_count"])
                row["measurement_skipped_count"] = int(len(measurement_skipped))
                row["measurement_grid_rows"] = int(measurement_grid_rows)
                row["measurement_grid_cols"] = int(measurement_grid_cols)
                row["measurement_score_mode"] = str(measurement_score_mode)
            measurement_rows.extend(rows)
            store_query_matches(query_id, matches)
        pending_measurement.clear()

    start_time = time.time()
    for query_index, record in enumerate(records):
        query_feature = _load_query_feature(record, feature_key)
        mapped = feature_mapper.project(query_feature)
        mapped_query = mapped.coarse_descriptors
        query_heatmap = mapped.heatmap
        camera = cameras_by_query.get(str(record.image_id))
        if camera is None:
            continue
        query_id = str(record.image_id)
        query_landmarks = landmark_index
        references = list((reference_submaps_by_query or {}).get(query_id, ()))
        submap_reference_counts.append(int(len(references)))
        if reference_submaps_by_query is not None:
            if references:
                query_landmarks = filter_landmarks_by_reference_images(landmark_index, references)
            else:
                submap_empty_fallback_count += 1
        pre_limit_submap_count = int(len(query_landmarks))
        if reference_submaps_by_query is not None and (
            int(submap_min_landmarks) > 0 or int(submap_min_spatial_cells) > 0
        ):
            query_landmarks, submap_selection_metadata = select_submap_with_global_fallback(
                query_landmarks,
                landmark_index,
                max_landmarks=max_submap_landmarks,
                selection_mode=str(submap_selection_mode),
                grid_rows=int(submap_spatial_grid_rows),
                grid_cols=int(submap_spatial_grid_cols),
                min_landmarks=int(submap_min_landmarks),
                min_spatial_cells=int(submap_min_spatial_cells),
                fallback_fraction=float(submap_fallback_fraction),
            )
        else:
            query_landmarks, submap_selection_metadata = _select_landmark_submap(
                query_landmarks,
                max_landmarks=max_submap_landmarks,
                selection_mode=str(submap_selection_mode),
                grid_rows=int(submap_spatial_grid_rows),
                grid_cols=int(submap_spatial_grid_cols),
            )
        cache_key = _submap_cache_key(
            query_id=query_id,
            references=references,
            selection_mode=str(submap_selection_mode),
            max_submap_landmarks=max_submap_landmarks,
            grid_rows=int(submap_spatial_grid_rows),
            grid_cols=int(submap_spatial_grid_cols),
            min_landmarks=int(submap_min_landmarks),
            min_spatial_cells=int(submap_min_spatial_cells),
            fallback_fraction=float(submap_fallback_fraction),
        )
        submap_pre_limit_counts.append(pre_limit_submap_count)
        submap_counts.append(int(len(query_landmarks)))
        if retrieval_config is None:
            matches = match_query_tokens_to_landmarks(
                mapped_query,
                query_landmarks,
                matching_config,
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
            retrieval_rows.append(
                {
                    "query_id": query_id,
                    "backend": "legacy_exact",
                    "reference_count": int(len(references)),
                    "pre_limit_submap_landmark_count": pre_limit_submap_count,
                    "submap_landmark_count": int(len(query_landmarks)),
                    "output_match_count": int(len(matches)),
                    **submap_selection_metadata,
                }
            )
        else:
            matches, retrieval_metadata = match_query_tokens_to_landmarks_ann(
                mapped_query,
                query_landmarks,
                retrieval_config,
                image_width=int(camera.width),
                image_height=int(camera.height),
                query_heatmap=query_heatmap,
                index_cache=retrieval_index_cache,
                cache_key=cache_key,
            )
            retrieval_rows.append(
                {
                    "query_id": query_id,
                    "reference_count": int(len(references)),
                    "pre_limit_submap_landmark_count": pre_limit_submap_count,
                    "submap_landmark_count": int(len(query_landmarks)),
                    **submap_selection_metadata,
                    **retrieval_metadata,
                }
            )
        if bool(enable_quality_rescore):
            matches = rescore_landmark_matches_for_measurement(matches)
        if measurement_adapter is not None:
            if owner_index is None:
                raise ValueError("owner_index is required when measurement_adapter is set")
            query_rgb = _load_rgb_chw(Path(image_root) / record.image_id)
            candidate_limit = int(measurement_max_matches or 0)
            measurement_input = list(matches)
            measurement_skipped: list[QueryTo3DMatch] = []
            if candidate_limit > 0:
                if str(measurement_selection_strategy) == "coarse_pnp_inliers":
                    measurement_input, measurement_skipped, selection_metadata = select_measurement_candidates_after_coarse_pnp(
                        matches,
                        camera=camera,
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        max_count=candidate_limit,
                        grid_rows=int(measurement_grid_rows),
                        grid_cols=int(measurement_grid_cols),
                        score_mode=str(measurement_score_mode),
                        pnp_reprojection_error_px=float(pnp_reprojection_error_px),
                        pnp_iterations=int(pnp_iterations),
                        pnp_confidence=float(pnp_confidence),
                        pnp_min_inliers=int(pnp_min_inliers),
                    )
                elif str(measurement_selection_strategy) == "score_spatial":
                    measurement_input, measurement_skipped = select_measurement_candidates_spatially(
                        matches,
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        max_count=candidate_limit,
                        grid_rows=int(measurement_grid_rows),
                        grid_cols=int(measurement_grid_cols),
                        score_mode=str(measurement_score_mode),
                    )
                    selection_metadata = {
                        "measurement_selection_strategy": "score_spatial",
                        "coarse_pnp_success": None,
                        "coarse_pnp_inlier_count": None,
                        "measurement_filled_from_non_inliers": None,
                    }
                else:
                    raise ValueError("measurement_selection_strategy must be 'score_spatial' or 'coarse_pnp_inliers'")
                measurement_selection_rows.append(
                    {
                        "query_id": str(record.image_id),
                        "measurement_candidate_count": int(len(measurement_input)),
                        "measurement_skipped_count": int(len(measurement_skipped)),
                        **selection_metadata,
                    }
                )
            measurement_candidate_counts.append(int(len(measurement_input)))
            pending_measurement.append(
                {
                    "query_id": str(record.image_id),
                    "query_rgb": query_rgb,
                    "measurement_input": list(measurement_input),
                    "measurement_skipped": list(measurement_skipped),
                    "measurement_candidate_count": int(len(measurement_input)),
                }
            )
            if len(pending_measurement) >= max(1, int(measurement_query_batch_size)):
                flush_pending_measurement()
        else:
            store_query_matches(query_id, matches)
        if int(progress_interval_queries) > 0 and (
            (query_index + 1) % int(progress_interval_queries) == 0 or (query_index + 1) == len(records)
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                "[landmark-hybrid] "
                f"query {query_index + 1}/{len(records)} "
                f"elapsed={elapsed:.1f}s "
                f"submap={submap_counts[-1] if submap_counts else 0} "
                f"matches={len(matches)} "
                f"measurement_candidates={measurement_candidate_counts[-1] if measurement_candidate_counts else 0}",
                flush=True,
            )
    flush_pending_measurement()

    pose_rows = (
        evaluate_query_poses(
            matches_by_query,
            cameras_by_query=cameras_by_query,
            gt_poses_by_query=gt_poses_by_query,
            pnp_reprojection_error_px=float(pnp_reprojection_error_px),
            pnp_iterations=int(pnp_iterations),
            pnp_confidence=float(pnp_confidence),
            pnp_min_inliers=int(pnp_min_inliers),
            pnp_weighted_refine=bool(pnp_weighted_refine),
            pnp_weighted_loss=str(pnp_weighted_loss),
            pnp_weighted_f_scale_px=float(pnp_weighted_f_scale_px),
            pnp_weighted_max_nfev=int(pnp_weighted_max_nfev),
        )
        if bool(evaluate_pose)
        else []
    )

    write_mapping_rows_jsonl(output / "matches_2d3d.jsonl", match_rows)
    write_mapping_rows_csv(output / "matches_2d3d.csv", match_rows)
    write_mapping_rows_jsonl(output / "measurement_rows.jsonl", measurement_rows)
    write_mapping_rows_csv(output / "measurement_rows.csv", measurement_rows)
    write_mapping_rows_jsonl(output / "measurement_selection_rows.jsonl", measurement_selection_rows)
    write_mapping_rows_csv(output / "measurement_selection_rows.csv", measurement_selection_rows)
    write_mapping_rows_jsonl(output / "retrieval_rows.jsonl", retrieval_rows)
    write_mapping_rows_csv(output / "retrieval_rows.csv", retrieval_rows)
    write_mapping_rows_jsonl(output / "pose_rows.jsonl", pose_rows)
    write_mapping_rows_csv(output / "pose_rows.csv", pose_rows)
    match_counts = [len(values) for values in matches_by_query.values()]
    summary = {
        "stage": (
            "real_radio_landmark_hybrid_pose_localization"
            if bool(evaluate_pose)
            else "real_radio_landmark_proposal_generation"
        ),
        "proposal_only": not bool(evaluate_pose),
        "query_count": int(len(records)),
        "evaluated_query_count": int(len(matches_by_query)),
        "landmark_count": int(len(landmark_index)),
        "measurement_enabled": bool(measurement_adapter is not None),
        "measurement_max_matches": int(measurement_max_matches or 0),
        "measurement_query_batch_size": int(measurement_query_batch_size),
        "measurement_selection_strategy": str(measurement_selection_strategy),
        "measurement_grid_rows": int(measurement_grid_rows),
        "measurement_grid_cols": int(measurement_grid_cols),
        "measurement_score_mode": str(measurement_score_mode),
        "measurement_final_match_policy": final_policy,
        "measurement_adapter_config": {}
        if measurement_adapter is None
        else {
            "confidence_temperature": float(getattr(measurement_adapter, "confidence_temperature", 1.0)),
            "confidence_bias": float(getattr(measurement_adapter, "confidence_bias", 0.0)),
            "uncertainty_scale": float(getattr(measurement_adapter, "uncertainty_scale", 1.0)),
            "uncertainty_floor_px": float(getattr(measurement_adapter, "uncertainty_floor_px", 0.0)),
            "use_amp": bool(getattr(measurement_adapter, "use_amp", False)),
            "amp_dtype": str(getattr(measurement_adapter, "amp_dtype", "")),
            "tensor_cache_size": int(getattr(measurement_adapter, "tensor_cache_size", 0)),
        },
        "min_measurement_confidence": None
        if min_measurement_confidence is None
        else float(min_measurement_confidence),
        "max_measurement_uncertainty_px": None
        if max_measurement_uncertainty_px is None
        else float(max_measurement_uncertainty_px),
        "measurement_geometry_probability_model_enabled": bool(measurement_geometry_model is not None),
        "min_measurement_geometry_probability": None
        if min_measurement_geometry_probability is None
        else float(min_measurement_geometry_probability),
        "drop_rejected_measurements": bool(drop_rejected_measurements),
        "enable_quality_rescore": bool(enable_quality_rescore),
        "pnp_min_soft_score": None if pnp_min_soft_score is None else float(pnp_min_soft_score),
        "pnp_weighted_refine": bool(pnp_weighted_refine),
        "pnp_weighted_loss": str(pnp_weighted_loss),
        "pnp_weighted_f_scale_px": float(pnp_weighted_f_scale_px),
        "pnp_weighted_max_nfev": int(pnp_weighted_max_nfev),
        "mean_measurement_candidate_count": float(np.mean(measurement_candidate_counts))
        if measurement_candidate_counts
        else 0.0,
        "reference_rgb_cache_size": int(reference_rgb_cache_size),
        "mean_match_count": float(np.mean(match_counts)) if match_counts else 0.0,
        "mean_submap_reference_count": float(np.mean(submap_reference_counts)) if submap_reference_counts else 0.0,
        "mean_pre_limit_submap_landmark_count": float(np.mean(submap_pre_limit_counts)) if submap_pre_limit_counts else 0.0,
        "mean_submap_landmark_count": float(np.mean(submap_counts)) if submap_counts else 0.0,
        "submap_empty_fallback_count": int(submap_empty_fallback_count),
        "submap_selection_mode": str(submap_selection_mode),
        "submap_spatial_grid_rows": int(submap_spatial_grid_rows),
        "submap_spatial_grid_cols": int(submap_spatial_grid_cols),
        "submap_min_landmarks": int(submap_min_landmarks),
        "submap_min_spatial_cells": int(submap_min_spatial_cells),
        "submap_fallback_fraction": float(submap_fallback_fraction),
        "retrieval_index_cache": {}
        if retrieval_index_cache is None
        else retrieval_index_cache.stats(),
        "pose": summarize_pose_rows(pose_rows),
        "matching_config": {
            "top_k": int(matching_config.top_k),
            "ratio_threshold": matching_config.ratio_threshold,
            "min_similarity": float(matching_config.min_similarity),
            "query_token_step": int(matching_config.query_token_step),
            "max_matches": matching_config.max_matches,
            "mutual": bool(matching_config.mutual),
        },
        "retrieval_config": None
        if retrieval_config is None
        else {
            "backend": str(retrieval_config.backend),
            "top_k": int(retrieval_config.top_k),
            "top_k_legacy_alias_for_nn_search": int(retrieval_config.top_k),
            "nn_search_k_for_ratio": int(retrieval_config.nn_search_k_for_ratio or retrieval_config.top_k),
            "proposal_top_l": int(retrieval_config.proposal_top_l),
            "ratio_threshold": retrieval_config.ratio_threshold,
            "min_similarity": float(retrieval_config.min_similarity),
            "min_similarity_margin": retrieval_config.min_similarity_margin,
            "min_observation_count": int(retrieval_config.min_observation_count),
            "query_token_step": int(retrieval_config.query_token_step),
            "max_matches": retrieval_config.max_matches,
            "block_size": int(retrieval_config.block_size),
            "deduplicate_tracks": bool(retrieval_config.deduplicate_tracks),
            "query_token_selection": str(retrieval_config.query_token_selection),
            "query_heatmap_top_k": int(retrieval_config.query_heatmap_top_k),
            "query_heatmap_nms_radius": int(retrieval_config.query_heatmap_nms_radius),
            "query_heatmap_grid_rows": int(retrieval_config.query_heatmap_grid_rows),
            "query_heatmap_grid_cols": int(retrieval_config.query_heatmap_grid_cols),
            "query_heatmap_min_score": retrieval_config.query_heatmap_min_score,
            "query_xy_coordinate_mode": str(retrieval_config.query_xy_coordinate_mode),
        },
        "outputs": {
            "matches_2d3d_jsonl": str(output / "matches_2d3d.jsonl"),
            "matches_2d3d_csv": str(output / "matches_2d3d.csv"),
            "measurement_rows_jsonl": str(output / "measurement_rows.jsonl"),
            "measurement_rows_csv": str(output / "measurement_rows.csv"),
            "measurement_selection_rows_jsonl": str(output / "measurement_selection_rows.jsonl"),
            "measurement_selection_rows_csv": str(output / "measurement_selection_rows.csv"),
            "retrieval_rows_jsonl": str(output / "retrieval_rows.jsonl"),
            "retrieval_rows_csv": str(output / "retrieval_rows.csv"),
            "pose_rows_jsonl": str(output / "pose_rows.jsonl"),
            "pose_rows_csv": str(output / "pose_rows.csv"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
