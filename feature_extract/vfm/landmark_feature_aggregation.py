"""Aggregate raw high-dimensional VFM observations into 3D landmark features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from feature_extract.vfm.map_lifting import (
    SelectedTrackFeatureBank,
    TrackFeature,
    TrackObservation,
)


AGGREGATION_METHODS = {
    "mean",
    "cosine_weighted_mean",
    "random_observation",
    "geometry_weighted",
    "robust_trimmed_mean",
    "view_consistent",
    "ulf_geometry_weighted",
    "geometric_median",
    "medoid",
}

TRACK_PROTOTYPE_BUILDER_VERSION = "track_prototype_builder_v2"


@dataclass(frozen=True)
class LandmarkViewClusteringConfig:
    method: str = "descriptor_spherical_kmeans"
    max_prototypes_per_track: int = 2
    min_observations_per_prototype: int = 2
    iterations: int = 16

    def __post_init__(self) -> None:
        if str(self.method) != "descriptor_spherical_kmeans":
            raise ValueError("view clustering method must be 'descriptor_spherical_kmeans'")
        if int(self.max_prototypes_per_track) <= 0:
            raise ValueError("max_prototypes_per_track must be positive")
        if int(self.min_observations_per_prototype) <= 0:
            raise ValueError("min_observations_per_prototype must be positive")
        if int(self.iterations) <= 0:
            raise ValueError("view clustering iterations must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(int(self.max_prototypes_per_track) > 1),
            "method": str(self.method),
            "max_prototypes_per_track": int(self.max_prototypes_per_track),
            "min_observations_per_prototype": int(self.min_observations_per_prototype),
            "iterations": int(self.iterations),
        }


@dataclass(frozen=True)
class TrackPrototype:
    track_id: int
    prototype_id: int
    feature: np.ndarray
    variance: np.ndarray
    observation_count: int
    mean_utility: float
    observation_image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        feature = np.asarray(self.feature, dtype=np.float32).reshape(-1)
        variance = np.asarray(self.variance, dtype=np.float32).reshape(-1)
        if feature.shape != variance.shape:
            raise ValueError("prototype feature and variance must have the same shape")
        if int(self.prototype_id) < 0:
            raise ValueError("prototype_id must be non-negative")
        if int(self.observation_count) <= 0:
            raise ValueError("prototype observation_count must be positive")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "observation_image_ids", tuple(str(item) for item in self.observation_image_ids))


@dataclass(frozen=True)
class MultiPrototypeTrackBank:
    prototypes: tuple[TrackPrototype, ...]
    feature_dim: int
    clustering: LandmarkViewClusteringConfig

    def __post_init__(self) -> None:
        for prototype in self.prototypes:
            if int(prototype.feature.size) != int(self.feature_dim):
                raise ValueError("all prototypes must match feature_dim")
        keys = [(int(item.track_id), int(item.prototype_id)) for item in self.prototypes]
        if len(keys) != len(set(keys)):
            raise ValueError("prototype (track_id, prototype_id) keys must be unique")

    def __len__(self) -> int:
        return len(self.prototypes)

    @property
    def track_count(self) -> int:
        return len({int(item.track_id) for item in self.prototypes})


@dataclass(frozen=True)
class LandmarkAggregationConfig:
    method: str = "mean"
    min_observations: int = 2
    seed: int = 0
    trim_fraction: float = 0.2
    view_consistent_keep: int = 4
    geometric_median_iterations: int = 32
    l2_normalize_observations: bool = False
    weight_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.method not in AGGREGATION_METHODS:
            raise ValueError(f"unsupported landmark aggregation method: {self.method}")
        if self.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if not 0.0 <= float(self.trim_fraction) < 1.0:
            raise ValueError("trim_fraction must be in [0, 1)")
        if self.view_consistent_keep <= 0:
            raise ValueError("view_consistent_keep must be positive")
        if self.geometric_median_iterations <= 0:
            raise ValueError("geometric_median_iterations must be positive")
        if self.weight_floor <= 0.0:
            raise ValueError("weight_floor must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "min_observations": int(self.min_observations),
            "seed": int(self.seed),
            "trim_fraction": float(self.trim_fraction),
            "view_consistent_keep": int(self.view_consistent_keep),
            "geometric_median_iterations": int(self.geometric_median_iterations),
            "l2_normalize_observations": bool(self.l2_normalize_observations),
            "weight_floor": float(self.weight_floor),
        }


@dataclass(frozen=True)
class LandmarkSplitStabilityReport:
    common_track_count: int
    mean_cosine_similarity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "common_track_count": int(self.common_track_count),
            "mean_cosine_similarity": float(self.mean_cosine_similarity),
        }


@dataclass(frozen=True)
class LandmarkObservationRetrievalReport:
    query_count: int
    recall_at_k: Mapping[int, float]
    mean_rank: float

    def to_dict(self) -> dict[str, object]:
        return {
            "query_count": int(self.query_count),
            "recall_at_k": {str(key): float(value) for key, value in sorted(self.recall_at_k.items())},
            "mean_rank": float(self.mean_rank),
        }


def _valid_observations(observations: Iterable[TrackObservation]) -> list[TrackObservation]:
    valid: list[TrackObservation] = []
    feature_dim: int | None = None
    for obs in observations:
        if not obs.visible or not obs.geometry_valid:
            continue
        feature = np.asarray(obs.feature, dtype=np.float32).reshape(-1)
        if feature_dim is None:
            feature_dim = int(feature.size)
        elif feature.size != feature_dim:
            raise ValueError("all VFM observations must have the same feature dimension")
        valid.append(
            TrackObservation(
                track_id=int(obs.track_id),
                image_id=str(obs.image_id),
                feature=feature,
                visible=True,
                geometry_valid=True,
                utility=float(obs.utility),
            )
        )
    return valid


def _group_by_track(observations: Sequence[TrackObservation]) -> dict[int, list[TrackObservation]]:
    grouped: dict[int, list[TrackObservation]] = {}
    for obs in observations:
        grouped.setdefault(int(obs.track_id), []).append(obs)
    return grouped


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-6)


def _features_for_track(observations: Sequence[TrackObservation], config: LandmarkAggregationConfig) -> np.ndarray:
    features = np.stack([np.asarray(obs.feature, dtype=np.float32).reshape(-1) for obs in observations], axis=0)
    if config.l2_normalize_observations:
        features = _normalize_rows(features)
    return features.astype(np.float32, copy=False)


def _weighted_mean_and_variance(
    features: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        mean = features.mean(axis=0)
        variance = features.var(axis=0)
        return mean.astype(np.float32), variance.astype(np.float32)
    clean_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    clean_weights = np.maximum(clean_weights, 0.0)
    if float(clean_weights.sum()) <= 1e-12:
        clean_weights = np.ones_like(clean_weights, dtype=np.float32)
    normalized = clean_weights / np.maximum(float(clean_weights.sum()), 1e-12)
    mean = np.sum(features * normalized[:, None], axis=0)
    variance = np.sum(((features - mean[None, :]) ** 2) * normalized[:, None], axis=0)
    return mean.astype(np.float32), variance.astype(np.float32)


def _weighted_geometric_median(
    features: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    eps: float = 1e-6,
) -> np.ndarray:
    clean_weights = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), eps)
    current, _variance = _weighted_mean_and_variance(features, clean_weights.astype(np.float32))
    current = current.astype(np.float64)
    feature64 = features.astype(np.float64)
    for _ in range(iterations):
        distances = np.linalg.norm(feature64 - current[None, :], axis=1)
        if float(np.min(distances)) < eps:
            return feature64[int(np.argmin(distances))].astype(np.float32)
        step_weights = clean_weights / np.maximum(distances, eps)
        next_value = np.sum(feature64 * step_weights[:, None], axis=0) / np.maximum(float(np.sum(step_weights)), eps)
        if float(np.linalg.norm(next_value - current)) < eps:
            current = next_value
            break
        current = next_value
    return current.astype(np.float32)


def _weighted_medoid(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, int]:
    feature64 = features.astype(np.float64)
    clean_weights = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12)
    squared_norms = np.sum(feature64 * feature64, axis=1)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (feature64 @ feature64.T)
    distances = np.sqrt(np.maximum(squared_distances, 0.0))
    weighted_distances = distances @ clean_weights
    idx = int(np.argmin(weighted_distances))
    return features[idx].astype(np.float32), idx


def _seed_for_track(seed: int, track_id: int) -> int:
    return int((int(seed) * 1_000_003 + int(track_id) * 9_176 + 17) % (2**32))


def _aggregate_one_track(
    track_id: int,
    observations: Sequence[TrackObservation],
    config: LandmarkAggregationConfig,
) -> TrackFeature:
    features = _features_for_track(observations, config)
    utilities = np.asarray([float(obs.utility) for obs in observations], dtype=np.float32)
    selected_features = features
    selected_utilities = utilities
    output_observation_count = int(features.shape[0])

    if config.method == "mean":
        mean, variance = _weighted_mean_and_variance(features, None)
    elif config.method == "cosine_weighted_mean":
        normalized = _normalize_rows(features)
        mean, variance = _weighted_mean_and_variance(normalized, np.maximum(utilities, config.weight_floor))
        mean = (mean / max(float(np.linalg.norm(mean)), 1e-6)).astype(np.float32)
    elif config.method == "geometry_weighted":
        mean, variance = _weighted_mean_and_variance(features, np.maximum(utilities, config.weight_floor))
    elif config.method == "random_observation":
        rng = np.random.default_rng(_seed_for_track(config.seed, track_id))
        selected_idx = int(rng.integers(0, features.shape[0]))
        mean = features[selected_idx].astype(np.float32)
        variance = ((features - mean[None, :]) ** 2).mean(axis=0).astype(np.float32)
        selected_features = features[selected_idx : selected_idx + 1]
        selected_utilities = utilities[selected_idx : selected_idx + 1]
        output_observation_count = int(features.shape[0])
    elif config.method == "robust_trimmed_mean":
        center = features.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(features - center, axis=1)
        keep_count = int(np.ceil(features.shape[0] * (1.0 - float(config.trim_fraction))))
        keep_count = min(features.shape[0], max(config.min_observations, keep_count))
        keep_indices = np.sort(np.argsort(distances, kind="mergesort")[:keep_count])
        selected_features = features[keep_indices]
        selected_utilities = utilities[keep_indices]
        mean, variance = _weighted_mean_and_variance(selected_features, None)
    elif config.method == "view_consistent":
        normalized = _normalize_rows(features)
        similarity = normalized @ normalized.T
        medoid_idx = int(np.argmax(similarity.mean(axis=1)))
        keep_count = min(features.shape[0], max(config.min_observations, int(config.view_consistent_keep)))
        keep_indices = np.sort(np.argsort(-similarity[medoid_idx], kind="mergesort")[:keep_count])
        selected_features = features[keep_indices]
        selected_utilities = utilities[keep_indices]
        mean, variance = _weighted_mean_and_variance(selected_features, np.maximum(selected_utilities, config.weight_floor))
    elif config.method == "ulf_geometry_weighted":
        normalized = _normalize_rows(features)
        similarity = normalized @ normalized.T
        off_diagonal = similarity.copy()
        np.fill_diagonal(off_diagonal, -np.inf)
        neighbor_count = max(1, min(features.shape[0] - 1, int(config.min_observations) - 1))
        nearest = np.sort(off_diagonal, axis=1)[:, -neighbor_count:]
        consensus = np.maximum(nearest, 0.0).mean(axis=1)
        positive_indices = np.flatnonzero(consensus > float(config.weight_floor))
        candidate_indices = positive_indices if positive_indices.size >= int(config.min_observations) else np.arange(features.shape[0])
        keep_count = min(candidate_indices.size, max(config.min_observations, int(config.view_consistent_keep)))
        candidate_scores = consensus[candidate_indices] * np.maximum(utilities[candidate_indices], config.weight_floor)
        keep_indices = np.sort(candidate_indices[np.argsort(-candidate_scores, kind="mergesort")[:keep_count]])
        selected_features = normalized[keep_indices]
        selected_utilities = utilities[keep_indices] * np.maximum(consensus[keep_indices], config.weight_floor)
        mean, variance = _weighted_mean_and_variance(selected_features, np.maximum(selected_utilities, config.weight_floor))
        mean = (mean / max(float(np.linalg.norm(mean)), 1e-6)).astype(np.float32)
    elif config.method == "geometric_median":
        mean = _weighted_geometric_median(
            features,
            np.maximum(utilities, config.weight_floor),
            iterations=config.geometric_median_iterations,
        )
        variance = np.mean((features - mean[None, :]) ** 2, axis=0).astype(np.float32)
    elif config.method == "medoid":
        mean, selected_idx = _weighted_medoid(features, np.maximum(utilities, config.weight_floor))
        variance = np.mean((features - mean[None, :]) ** 2, axis=0).astype(np.float32)
        selected_features = features[selected_idx : selected_idx + 1]
        selected_utilities = utilities[selected_idx : selected_idx + 1]
        output_observation_count = int(features.shape[0])
    else:
        raise ValueError(f"unsupported landmark aggregation method: {config.method}")

    return TrackFeature(
        track_id=int(track_id),
        mean_feature=mean.astype(np.float32, copy=False),
        variance=variance.astype(np.float32, copy=False),
        observation_count=output_observation_count
        if config.method in {"random_observation", "medoid"}
        else int(selected_features.shape[0]),
        mean_utility=float(np.mean(selected_utilities)) if selected_utilities.size else 0.0,
        observation_image_ids=tuple(sorted({obs.image_id for obs in observations})),
    )


def aggregate_landmark_features(
    observations: Iterable[TrackObservation],
    config: LandmarkAggregationConfig | None = None,
) -> SelectedTrackFeatureBank:
    """Aggregate sampled raw VFM observations into one feature per 3D landmark."""

    cfg = config or LandmarkAggregationConfig()
    valid = _valid_observations(observations)
    feature_dim = int(valid[0].feature.size) if valid else 0
    tracks: dict[int, TrackFeature] = {}
    for track_id, group in _group_by_track(valid).items():
        if len(group) < cfg.min_observations:
            continue
        tracks[track_id] = _aggregate_one_track(track_id, group, cfg)
    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=feature_dim)


def _deterministic_spherical_kmeans_labels(
    features: np.ndarray,
    *,
    cluster_count: int,
    iterations: int,
) -> np.ndarray:
    values = _normalize_rows(np.asarray(features, dtype=np.float32))
    count = int(values.shape[0])
    clusters = min(max(1, int(cluster_count)), count)
    if clusters == 1:
        return np.zeros((count,), dtype=np.int64)

    mean_direction = values.mean(axis=0)
    mean_direction /= max(float(np.linalg.norm(mean_direction)), 1e-6)
    center_indices = [int(np.argmax(values @ mean_direction))]
    while len(center_indices) < clusters:
        similarity = values @ values[np.asarray(center_indices, dtype=np.int64)].T
        nearest_similarity = np.max(similarity, axis=1)
        nearest_similarity[np.asarray(center_indices, dtype=np.int64)] = np.inf
        center_indices.append(int(np.argmin(nearest_similarity)))
    centers = values[np.asarray(center_indices, dtype=np.int64)].copy()
    labels = np.full((count,), -1, dtype=np.int64)
    for _ in range(int(iterations)):
        similarities = values @ centers.T
        next_labels = np.argmax(similarities, axis=1).astype(np.int64)
        for cluster_id in range(clusters):
            if np.any(next_labels == cluster_id):
                continue
            assigned_similarity = similarities[np.arange(count), next_labels]
            move_index = int(np.argmin(assigned_similarity))
            next_labels[move_index] = int(cluster_id)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_id in range(clusters):
            members = values[labels == cluster_id]
            center = members.mean(axis=0)
            centers[cluster_id] = center / max(float(np.linalg.norm(center)), 1e-6)
    return labels


def build_multi_prototype_track_bank(
    observations: Iterable[TrackObservation],
    *,
    aggregation: LandmarkAggregationConfig | None = None,
    clustering: LandmarkViewClusteringConfig | None = None,
    normalize_final_prototypes: bool = True,
) -> MultiPrototypeTrackBank:
    """Build deterministic descriptor-clustered prototypes without changing track identity."""

    aggregation_config = aggregation or LandmarkAggregationConfig(
        method="mean",
        l2_normalize_observations=True,
    )
    clustering_config = clustering or LandmarkViewClusteringConfig()
    valid = _valid_observations(observations)
    feature_dim = int(valid[0].feature.size) if valid else 0
    output: list[TrackPrototype] = []
    for track_id, group in sorted(_group_by_track(valid).items()):
        if len(group) < int(aggregation_config.min_observations):
            continue
        max_by_support = max(1, len(group) // int(clustering_config.min_observations_per_prototype))
        cluster_count = min(int(clustering_config.max_prototypes_per_track), max_by_support)
        features = _features_for_track(group, aggregation_config)
        labels = _deterministic_spherical_kmeans_labels(
            features,
            cluster_count=int(cluster_count),
            iterations=int(clustering_config.iterations),
        )
        groups = [np.flatnonzero(labels == cluster_id) for cluster_id in range(int(cluster_count))]
        if any(indices.size < int(clustering_config.min_observations_per_prototype) for indices in groups):
            groups = [np.arange(len(group), dtype=np.int64)]
        groups.sort(key=lambda indices: int(np.min(indices)))
        for prototype_id, indices in enumerate(groups):
            selected = [group[int(index)] for index in indices.tolist()]
            aggregated = _aggregate_one_track(int(track_id), selected, aggregation_config)
            feature = np.asarray(aggregated.mean_feature, dtype=np.float32)
            if bool(normalize_final_prototypes):
                feature = feature / max(float(np.linalg.norm(feature)), 1e-6)
            output.append(
                TrackPrototype(
                    track_id=int(track_id),
                    prototype_id=int(prototype_id),
                    feature=feature.astype(np.float32, copy=False),
                    variance=np.asarray(aggregated.variance, dtype=np.float32),
                    observation_count=int(aggregated.observation_count),
                    mean_utility=float(aggregated.mean_utility),
                    observation_image_ids=tuple(aggregated.observation_image_ids),
                )
            )
    return MultiPrototypeTrackBank(
        prototypes=tuple(output),
        feature_dim=int(feature_dim),
        clustering=clustering_config,
    )


@dataclass(frozen=True)
class TrackPrototypeBuilder:
    """One prototype contract shared by offline bank building and training."""

    aggregation: LandmarkAggregationConfig = LandmarkAggregationConfig()
    normalize_final_prototypes: bool = True
    version: str = TRACK_PROTOTYPE_BUILDER_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": str(self.version),
            "aggregation": self.aggregation.to_dict(),
            "normalize_final_prototypes": bool(self.normalize_final_prototypes),
        }

    def _normalize_bank(self, bank: SelectedTrackFeatureBank) -> SelectedTrackFeatureBank:
        if not bool(self.normalize_final_prototypes):
            return bank
        tracks: dict[int, TrackFeature] = {}
        for track_id, track in bank.tracks.items():
            feature = np.asarray(track.mean_feature, dtype=np.float32)
            feature = feature / max(float(np.linalg.norm(feature)), 1e-6)
            tracks[int(track_id)] = TrackFeature(
                track_id=int(track.track_id),
                mean_feature=feature.astype(np.float32, copy=False),
                variance=np.asarray(track.variance, dtype=np.float32),
                observation_count=int(track.observation_count),
                mean_utility=float(track.mean_utility),
                observation_image_ids=tuple(track.observation_image_ids),
            )
        return SelectedTrackFeatureBank(tracks=tracks, feature_dim=int(bank.feature_dim))

    def build_bank(self, observations: Iterable[TrackObservation]) -> SelectedTrackFeatureBank:
        return self._normalize_bank(aggregate_landmark_features(observations, self.aggregation))

    def build_bank_torch(
        self,
        observations: Iterable[TrackObservation],
        *,
        device: str = "cuda",
    ) -> SelectedTrackFeatureBank:
        """Use the scatter fast path where it is exactly supported."""

        bank = aggregate_landmark_features_torch(observations, self.aggregation, device=str(device))
        return self._normalize_bank(bank)

    def aggregate_torch(
        self,
        descriptors,
        inverse,
        track_count: int,
        *,
        utilities=None,
    ):
        """Differentiably aggregate rows with the same semantics as ``build_bank``."""

        import torch
        from torch.nn import functional as torch_functional

        if descriptors.ndim != 2:
            raise ValueError("descriptors must have shape (N, C)")
        groups = inverse.to(device=descriptors.device, dtype=torch.long).reshape(-1)
        if groups.shape[0] != descriptors.shape[0]:
            raise ValueError("inverse must contain one group index per descriptor")
        if int(track_count) <= 0:
            raise ValueError("track_count must be positive")
        if groups.numel() and (torch.any(groups < 0) or torch.any(groups >= int(track_count))):
            raise ValueError("inverse contains an out-of-range track index")
        method = str(self.aggregation.method)
        if method not in {"mean", "cosine_weighted_mean", "geometry_weighted"}:
            raise ValueError(f"training-time prototype aggregation does not support method={method!r}")
        values = descriptors
        if bool(self.aggregation.l2_normalize_observations) or method == "cosine_weighted_mean":
            values = torch_functional.normalize(values, dim=1, eps=1e-6)
        if utilities is None or method == "mean":
            weights = torch.ones((values.shape[0],), dtype=values.dtype, device=values.device)
        else:
            weights = torch.as_tensor(utilities, dtype=values.dtype, device=values.device).reshape(-1)
            if weights.shape[0] != values.shape[0]:
                raise ValueError("utilities must contain one value per descriptor")
            weights = torch.clamp(weights, min=float(self.aggregation.weight_floor))
        sums = torch.zeros(
            (int(track_count), int(values.shape[1])),
            dtype=values.dtype,
            device=values.device,
        )
        weight_sums = torch.zeros((int(track_count),), dtype=values.dtype, device=values.device)
        sums.index_add_(0, groups, values * weights[:, None])
        weight_sums.index_add_(0, groups, weights)
        prototypes = sums / weight_sums.clamp_min(1e-12)[:, None]
        if bool(self.normalize_final_prototypes) or method == "cosine_weighted_mean":
            prototypes = torch_functional.normalize(prototypes, dim=1, eps=1e-6)
        counts = torch.bincount(groups, minlength=int(track_count)).to(dtype=values.dtype)
        return prototypes, counts


def aggregate_landmark_features_torch(
    observations: Iterable[TrackObservation],
    config: LandmarkAggregationConfig | None = None,
    device: str = "cuda",
) -> SelectedTrackFeatureBank:
    """Aggregate mean/cosine/geometry-weighted landmark features with torch scatter ops.

    This is a fast path for large raw VFM banks. Robust methods still use the
    NumPy path because their per-track control flow dominates and is harder to
    batch efficiently.
    """

    cfg = config or LandmarkAggregationConfig()
    if cfg.method not in {"mean", "cosine_weighted_mean", "geometry_weighted"}:
        return aggregate_landmark_features(observations, cfg)
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for aggregate_landmark_features_torch") from exc

    valid = _valid_observations(observations)
    if not valid:
        return SelectedTrackFeatureBank(tracks={}, feature_dim=0)
    feature_dim = int(valid[0].feature.size)
    track_ids = np.asarray([int(obs.track_id) for obs in valid], dtype=np.int64)
    unique_track_ids, inverse, counts = np.unique(track_ids, return_inverse=True, return_counts=True)
    eligible = counts >= int(cfg.min_observations)
    if not np.any(eligible):
        return SelectedTrackFeatureBank(tracks={}, feature_dim=feature_dim)
    group_index_np = np.full((unique_track_ids.shape[0],), -1, dtype=np.int64)
    group_index_np[np.flatnonzero(eligible)] = np.arange(int(np.sum(eligible)), dtype=np.int64)
    obs_group_np = group_index_np[inverse]
    keep_obs_np = obs_group_np >= 0
    if not np.any(keep_obs_np):
        return SelectedTrackFeatureBank(tracks={}, feature_dim=feature_dim)

    features_np = np.stack([obs.feature for obs in valid], axis=0).astype(np.float32, copy=False)[keep_obs_np]
    utilities_np = np.asarray([float(obs.utility) for obs in valid], dtype=np.float32)[keep_obs_np]
    obs_group_np = obs_group_np[keep_obs_np]
    selected_track_ids = unique_track_ids[eligible]
    group_count = int(selected_track_ids.shape[0])
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    features = torch.as_tensor(features_np, dtype=torch.float32, device=torch_device)
    if cfg.l2_normalize_observations or cfg.method == "cosine_weighted_mean":
        features = torch.nn.functional.normalize(features, dim=1, eps=1e-6)
    group_index = torch.as_tensor(obs_group_np, dtype=torch.long, device=torch_device)
    sums = torch.zeros((group_count, feature_dim), dtype=torch.float32, device=torch_device)
    weight_sums = torch.zeros((group_count,), dtype=torch.float32, device=torch_device)
    if cfg.method == "mean":
        weights = torch.ones((features.shape[0],), dtype=torch.float32, device=torch_device)
    else:
        weights = torch.clamp(torch.as_tensor(utilities_np, dtype=torch.float32, device=torch_device), min=cfg.weight_floor)
    sums.index_add_(0, group_index, features * weights[:, None])
    weight_sums.index_add_(0, group_index, weights)
    means = sums / torch.clamp(weight_sums[:, None], min=1e-12)
    if cfg.method == "cosine_weighted_mean":
        means = torch.nn.functional.normalize(means, dim=1, eps=1e-6)
    variance_sums = torch.zeros((group_count, feature_dim), dtype=torch.float32, device=torch_device)
    diff = features - means[group_index]
    variance_sums.index_add_(0, group_index, (diff * diff) * weights[:, None])
    variances = variance_sums / torch.clamp(weight_sums[:, None], min=1e-12)
    means_np = means.detach().cpu().numpy().astype(np.float32, copy=False)
    variances_np = variances.detach().cpu().numpy().astype(np.float32, copy=False)

    image_ids_by_track: dict[int, set[str]] = {int(track_id): set() for track_id in selected_track_ids.tolist()}
    utility_sum_by_track: dict[int, float] = {int(track_id): 0.0 for track_id in selected_track_ids.tolist()}
    selected_count_by_track: dict[int, int] = {int(track_id): 0 for track_id in selected_track_ids.tolist()}
    selected_track_set = set(int(track_id) for track_id in selected_track_ids.tolist())
    for obs in valid:
        track_id = int(obs.track_id)
        if track_id not in selected_track_set:
            continue
        image_ids_by_track[track_id].add(str(obs.image_id))
        utility_sum_by_track[track_id] += float(obs.utility)
        selected_count_by_track[track_id] += 1
    tracks: dict[int, TrackFeature] = {}
    for row, track_id_value in enumerate(selected_track_ids.tolist()):
        track_id = int(track_id_value)
        count = int(selected_count_by_track[track_id])
        tracks[track_id] = TrackFeature(
            track_id=track_id,
            mean_feature=means_np[row],
            variance=variances_np[row],
            observation_count=count,
            mean_utility=float(utility_sum_by_track[track_id] / max(count, 1)),
            observation_image_ids=tuple(sorted(image_ids_by_track[track_id])),
        )
    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=feature_dim)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6)
    return float(np.dot(a, b) / denom)


def evaluate_landmark_split_stability(
    observations: Sequence[TrackObservation],
    config: LandmarkAggregationConfig,
    seed: int = 0,
) -> LandmarkSplitStabilityReport:
    """Aggregate two deterministic observation subsets and compare common landmarks."""

    grouped = _group_by_track(_valid_observations(observations))
    rng = np.random.default_rng(seed)
    left: list[TrackObservation] = []
    right: list[TrackObservation] = []
    for track_id in sorted(grouped):
        group = list(grouped[track_id])
        if len(group) < max(2, config.min_observations * 2):
            continue
        order = rng.permutation(len(group))
        split = len(group) // 2
        left.extend(group[int(idx)] for idx in order[:split])
        right.extend(group[int(idx)] for idx in order[split:])
    left_bank = aggregate_landmark_features(left, config)
    right_bank = aggregate_landmark_features(right, config)
    common = sorted(set(left_bank.tracks).intersection(right_bank.tracks))
    if not common:
        return LandmarkSplitStabilityReport(common_track_count=0, mean_cosine_similarity=0.0)
    similarities = [
        _cosine(left_bank.tracks[track_id].mean_feature, right_bank.tracks[track_id].mean_feature)
        for track_id in common
    ]
    return LandmarkSplitStabilityReport(
        common_track_count=len(common),
        mean_cosine_similarity=float(np.mean(similarities)),
    )


def evaluate_landmark_observation_retrieval(
    observations: Sequence[TrackObservation],
    config: LandmarkAggregationConfig,
    top_k: Sequence[int] = (1, 5),
    seed: int = 0,
) -> LandmarkObservationRetrievalReport:
    """Hold out one observation per track and retrieve its 3D landmark by cosine similarity."""

    grouped = _group_by_track(_valid_observations(observations))
    rng = np.random.default_rng(seed)
    train: list[TrackObservation] = []
    queries: list[TrackObservation] = []
    for track_id in sorted(grouped):
        group = list(grouped[track_id])
        if len(group) < config.min_observations + 1:
            continue
        heldout = int(rng.integers(0, len(group)))
        queries.append(group[heldout])
        train.extend(obs for idx, obs in enumerate(group) if idx != heldout)
    bank = aggregate_landmark_features(train, config)
    if not bank.tracks or not queries:
        return LandmarkObservationRetrievalReport(
            query_count=0,
            recall_at_k={int(k): 0.0 for k in top_k},
            mean_rank=0.0,
        )
    track_ids = np.asarray(sorted(bank.tracks), dtype=np.int64)
    features = np.stack([bank.tracks[int(track_id)].mean_feature for track_id in track_ids], axis=0)
    features = _normalize_rows(features.astype(np.float32))
    ranks: list[int] = []
    for query in queries:
        if int(query.track_id) not in bank.tracks:
            continue
        query_feature = np.asarray(query.feature, dtype=np.float32).reshape(1, -1)
        if config.l2_normalize_observations:
            query_feature = _normalize_rows(query_feature)
        else:
            query_feature = _normalize_rows(query_feature)
        scores = (query_feature @ features.T).reshape(-1)
        order = np.argsort(-scores, kind="mergesort")
        target_index = int(np.where(track_ids[order] == int(query.track_id))[0][0])
        ranks.append(target_index + 1)
    if not ranks:
        return LandmarkObservationRetrievalReport(
            query_count=0,
            recall_at_k={int(k): 0.0 for k in top_k},
            mean_rank=0.0,
        )
    recall = {int(k): float(np.mean([rank <= int(k) for rank in ranks])) for k in top_k}
    return LandmarkObservationRetrievalReport(
        query_count=len(ranks),
        recall_at_k=recall,
        mean_rank=float(np.mean(ranks)),
    )
