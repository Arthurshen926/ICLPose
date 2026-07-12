"""Contextual patch-to-3D landmark matching with local maplets.

Stage C3.1 keeps the PnP observation as a single 2D patch center matched to a
single 3D landmark.  The maplet only provides local context for scoring, so it
can improve correspondence ranking without reintroducing sparse footprint
ambiguity into the solver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.patch_footprint_matching import FootprintObservation
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch, normalize_rows, token_grid_xy


@dataclass(frozen=True)
class LocalMapletBank:
    neighbor_indices: np.ndarray
    context_features: np.ndarray
    neighbor_counts: np.ndarray
    context_radius: np.ndarray
    context_feature_variance: np.ndarray
    covisibility_strength: np.ndarray
    xyz_cov_eigvals: np.ndarray
    neighbor_idf_mean: np.ndarray
    maplet_type: str
    maplet_k: int
    context_pool: str = "quality_mean"

    def __post_init__(self) -> None:
        neighbor_indices = np.asarray(self.neighbor_indices, dtype=np.int64)
        context_features = np.asarray(self.context_features, dtype=np.float32)
        neighbor_counts = np.asarray(self.neighbor_counts, dtype=np.int64).reshape(-1)
        if neighbor_indices.ndim != 2:
            raise ValueError("neighbor_indices must have shape (N, K)")
        if context_features.ndim != 2:
            raise ValueError("context_features must have shape (N, C)")
        if neighbor_indices.shape[0] != context_features.shape[0]:
            raise ValueError("neighbor_indices and context_features must have the same N")
        if neighbor_counts.shape[0] != context_features.shape[0]:
            raise ValueError("neighbor_counts must have shape (N,)")
        object.__setattr__(self, "neighbor_indices", neighbor_indices)
        object.__setattr__(self, "context_features", context_features)
        object.__setattr__(self, "neighbor_counts", neighbor_counts)
        object.__setattr__(self, "context_radius", np.asarray(self.context_radius, dtype=np.float32).reshape(-1))
        object.__setattr__(
            self,
            "context_feature_variance",
            np.asarray(self.context_feature_variance, dtype=np.float32).reshape(-1),
        )
        object.__setattr__(
            self,
            "covisibility_strength",
            np.asarray(self.covisibility_strength, dtype=np.float32).reshape(-1),
        )
        object.__setattr__(self, "xyz_cov_eigvals", np.asarray(self.xyz_cov_eigvals, dtype=np.float32).reshape(-1, 3))
        object.__setattr__(self, "neighbor_idf_mean", np.asarray(self.neighbor_idf_mean, dtype=np.float32).reshape(-1))

    def __len__(self) -> int:
        return int(self.context_features.shape[0])


@dataclass(frozen=True)
class LocalMapletSupportIndex:
    maplets: LocalMapletBank
    anchor_track_ids: np.ndarray
    neighbor_track_ids: np.ndarray
    support_image_ids: tuple[str, ...]
    support_image_indices: np.ndarray
    support_coverage_counts: np.ndarray
    candidate_k: int
    version: str = "local_maplet_support_v1"

    def __post_init__(self) -> None:
        anchor_track_ids = np.asarray(self.anchor_track_ids, dtype=np.int64).reshape(-1)
        neighbor_track_ids = np.asarray(self.neighbor_track_ids, dtype=np.int64)
        support_image_indices = np.asarray(self.support_image_indices, dtype=np.int64)
        support_coverage_counts = np.asarray(self.support_coverage_counts, dtype=np.int64)
        count = int(anchor_track_ids.size)
        if len(self.maplets) != count:
            raise ValueError("maplet count must match anchor_track_ids")
        if neighbor_track_ids.shape != self.maplets.neighbor_indices.shape:
            raise ValueError("neighbor_track_ids must match maplet neighbor shape")
        if support_image_indices.ndim != 2 or support_image_indices.shape[0] != count:
            raise ValueError("support_image_indices must have shape (N, V)")
        if support_coverage_counts.shape != support_image_indices.shape:
            raise ValueError("support_coverage_counts must match support_image_indices")
        if np.any((support_image_indices < -1) | (support_image_indices >= len(self.support_image_ids))):
            raise ValueError("support_image_indices contains an out-of-range image index")
        if int(self.candidate_k) < int(self.maplets.maplet_k):
            raise ValueError("candidate_k must be at least maplet_k")
        object.__setattr__(self, "anchor_track_ids", anchor_track_ids)
        object.__setattr__(self, "neighbor_track_ids", neighbor_track_ids)
        object.__setattr__(self, "support_image_ids", tuple(str(item) for item in self.support_image_ids))
        object.__setattr__(self, "support_image_indices", support_image_indices)
        object.__setattr__(self, "support_coverage_counts", support_coverage_counts)

    def __len__(self) -> int:
        return int(self.anchor_track_ids.size)

    def support_views(self, row: int) -> tuple[str, ...]:
        indices = self.support_image_indices[int(row)]
        return tuple(self.support_image_ids[int(index)] for index in indices if int(index) >= 0)


@dataclass(frozen=True)
class ContextualLandmarkMatchingConfig:
    top_k: int = 1
    mutual_top_k: int = 1
    match_mode: str = "mnn"
    query_context: str = "3x3"
    context_weight: float = 0.25
    anchor_weight: float = 1.0
    quality_weight: float = 0.0
    score_mode: str = "anchor_context"
    context_pool: str = "quality_mean"
    min_similarity: float = 0.0
    query_token_step: int = 1
    max_matches: int | None = 1000
    block_size: int = 512
    similarity_device: str = "cpu"
    top_m_anchor: int = 0
    gate_mode: str = "none"
    anchor_margin_tau: float = 0.08
    anchor_margin_slope: float = 20.0
    context_margin_tau: float = 0.05
    context_margin_slope: float = 20.0

    def __post_init__(self) -> None:
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if int(self.mutual_top_k) <= 0:
            raise ValueError("mutual_top_k must be positive")
        if self.match_mode not in {"nn", "mnn", "soft_mutual"}:
            raise ValueError("match_mode must be one of: nn, mnn, soft_mutual")
        if self.query_context not in {"1x1", "3x3", "5x5"}:
            raise ValueError("query_context must be one of: 1x1, 3x3, 5x5")
        if self.context_pool not in {"mean", "quality_mean", "topk"}:
            raise ValueError("context_pool must be one of: mean, quality_mean, topk")
        if self.score_mode not in {"anchor_only", "context_only", "anchor_context", "anchor_context_quality"}:
            raise ValueError("score_mode must be one of: anchor_only, context_only, anchor_context, anchor_context_quality")
        if self.gate_mode not in {"none", "anchor_margin", "maplet", "anchor_maplet", "anchor_context_maplet", "full"}:
            raise ValueError(
                "gate_mode must be one of: none, anchor_margin, maplet, anchor_maplet, anchor_context_maplet, full"
            )
        if int(self.query_token_step) <= 0:
            raise ValueError("query_token_step must be positive")
        if self.max_matches is not None and int(self.max_matches) <= 0:
            raise ValueError("max_matches must be positive")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if int(self.top_m_anchor) < 0:
            raise ValueError("top_m_anchor must be non-negative")


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32, copy=False)


def _token_index_from_xy(
    xy: np.ndarray,
    image_width: int,
    image_height: int,
    token_width: int,
    token_height: int,
) -> int:
    point = np.asarray(xy, dtype=np.float64).reshape(2)
    x_norm = float(point[0]) / max(float(image_width - 1), 1.0)
    y_norm = float(point[1]) / max(float(image_height - 1), 1.0)
    x_idx = int(round(np.clip(x_norm, 0.0, 1.0) * max(token_width - 1, 0)))
    y_idx = int(round(np.clip(y_norm, 0.0, 1.0) * max(token_height - 1, 0)))
    return int(y_idx * token_width + x_idx)


def _flatten_token_features(feature_map: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature map must have shape (C, H, W)")
    _channels, height, width = values.shape
    features = []
    token_indices = []
    for y_idx in range(0, height, int(step)):
        for x_idx in range(0, width, int(step)):
            features.append(values[:, y_idx, x_idx])
            token_indices.append(y_idx * width + x_idx)
    if not features:
        return np.zeros((0, values.shape[0]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(features, axis=0).astype(np.float32), np.asarray(token_indices, dtype=np.int64)


def _landmark_quality_weights(index: LandmarkMapIndex) -> np.ndarray:
    if len(index) == 0:
        return np.zeros((0,), dtype=np.float32)
    counts = np.log1p(np.maximum(np.asarray(index.observation_counts, dtype=np.float32), 0.0))
    if counts.size and float(np.max(counts)) > 0.0:
        counts = counts / float(np.max(counts))
    variance = np.asarray(index.mean_variances, dtype=np.float32)
    variance_scale = float(np.percentile(variance, 95.0)) if variance.size else 1.0
    if variance_scale <= 1e-12:
        variance_scale = 1.0
    variance_score = 1.0 / (1.0 + np.maximum(variance, 0.0) / variance_scale)
    reproj = np.asarray(index.reprojection_errors, dtype=np.float32)
    reproj_scale = float(np.percentile(reproj, 95.0)) if reproj.size else 1.0
    if reproj_scale <= 1e-12:
        reproj_scale = 1.0
    reproj_score = 1.0 / (1.0 + np.maximum(reproj, 0.0) / reproj_scale)
    quality = np.clip(counts * variance_score * reproj_score, 0.0, 1.0)
    return quality.astype(np.float32, copy=False)


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(array, -60.0, 60.0)))).astype(np.float32, copy=False)


def maplet_reliability_scores(bank: LocalMapletBank) -> np.ndarray:
    """Compute an interpretable [0, 1] reliability prior for each local maplet."""

    count = len(bank)
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    neighbor_ratio = np.clip(
        np.asarray(bank.neighbor_counts, dtype=np.float32) / max(float(bank.maplet_k), 1.0),
        0.0,
        1.0,
    )
    variance = np.maximum(np.asarray(bank.context_feature_variance, dtype=np.float32), 0.0)
    finite = variance[np.isfinite(variance)]
    scale = float(np.percentile(finite, 75.0)) if finite.size else 1.0
    if scale <= 1e-8:
        scale = 1.0
    variance_score = 1.0 / (1.0 + variance / scale)
    covisibility = np.asarray(bank.covisibility_strength, dtype=np.float32)
    if np.max(covisibility) > 0.0:
        covisibility_score = np.clip(covisibility / max(float(np.percentile(covisibility, 90.0)), 1e-6), 0.0, 1.0)
    else:
        covisibility_score = np.ones((count,), dtype=np.float32)
    reliability = np.clip(neighbor_ratio * variance_score * covisibility_score, 0.0, 1.0)
    return reliability.astype(np.float32, copy=False)


def compute_query_context_descriptors(feature_map: np.ndarray, context: str = "3x3") -> np.ndarray:
    """Average neighboring token descriptors and L2-normalize each context token."""

    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature map must have shape (C, H, W)")
    if context not in {"1x1", "3x3", "5x5"}:
        raise ValueError("context must be one of: 1x1, 3x3, 5x5")
    if context == "1x1":
        flat, valid = normalize_rows(values.reshape(values.shape[0], -1).T)
        flat[~valid] = 0.0
        return flat.T.reshape(values.shape).astype(np.float32, copy=False)

    radius = 1 if context == "3x3" else 2
    channels, height, width = values.shape
    output = np.zeros_like(values, dtype=np.float32)
    for y_idx in range(height):
        y0 = max(0, y_idx - radius)
        y1 = min(height, y_idx + radius + 1)
        for x_idx in range(width):
            x0 = max(0, x_idx - radius)
            x1 = min(width, x_idx + radius + 1)
            output[:, y_idx, x_idx] = np.mean(values[:, y0:y1, x0:x1].reshape(channels, -1), axis=1)
    flat, valid = normalize_rows(output.reshape(channels, -1).T)
    flat[~valid] = 0.0
    return flat.T.reshape(output.shape).astype(np.float32, copy=False)


def _maplet_context_features(
    index: LandmarkMapIndex,
    neighbor_indices: np.ndarray,
    context_pool: str,
    covisibility_counts: np.ndarray | None = None,
) -> LocalMapletBank:
    features, _valid = normalize_rows(index.features)
    quality = _landmark_quality_weights(index)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    count, maplet_k = neighbor_indices.shape
    context_features = np.zeros((count, index.feature_dim), dtype=np.float32)
    neighbor_counts = np.zeros((count,), dtype=np.int64)
    context_radius = np.zeros((count,), dtype=np.float32)
    context_feature_variance = np.zeros((count,), dtype=np.float32)
    covisibility_strength = np.zeros((count,), dtype=np.float32)
    xyz_cov_eigvals = np.zeros((count, 3), dtype=np.float32)
    neighbor_idf_mean = np.zeros((count,), dtype=np.float32)
    for row in range(count):
        neighbors = neighbor_indices[row]
        neighbors = neighbors[neighbors >= 0]
        if neighbors.size == 0:
            context_features[row] = features[row]
            continue
        neighbor_counts[row] = int(neighbors.size)
        if context_pool == "mean":
            weights = np.ones((neighbors.size,), dtype=np.float32)
        else:
            weights = quality[neighbors].astype(np.float32, copy=True)
            if context_pool == "topk" and neighbors.size > 4:
                order = np.argsort(-weights)[:4]
                neighbors = neighbors[order]
                weights = weights[order]
        if not np.any(weights > 0.0):
            weights = np.ones((neighbors.size,), dtype=np.float32)
        weighted = np.sum(features[neighbors] * weights[:, None], axis=0) / max(float(np.sum(weights)), 1e-6)
        context_features[row] = _normalize_vector(weighted)
        deltas = index.xyz[neighbors] - index.xyz[row]
        distances = np.linalg.norm(deltas, axis=1)
        context_radius[row] = float(np.mean(distances)) if distances.size else 0.0
        if neighbors.size > 1:
            context_feature_variance[row] = float(np.mean(np.var(features[neighbors], axis=0)))
            cov = np.cov(index.xyz[neighbors].T)
            eigvals = np.linalg.eigvalsh(cov)
            xyz_cov_eigvals[row] = np.sort(np.maximum(eigvals, 0.0)).astype(np.float32)
        neighbor_idf_mean[row] = float(np.mean(1.0 / (1.0 + np.asarray(index.observation_counts[neighbors], dtype=np.float32))))
        if covisibility_counts is not None:
            counts = covisibility_counts[row, : neighbors.size]
            covisibility_strength[row] = float(np.mean(counts)) if counts.size else 0.0
    context_features, _valid_context = normalize_rows(context_features)
    return LocalMapletBank(
        neighbor_indices=neighbor_indices,
        context_features=context_features,
        neighbor_counts=neighbor_counts,
        context_radius=context_radius,
        context_feature_variance=context_feature_variance,
        covisibility_strength=covisibility_strength,
        xyz_cov_eigvals=xyz_cov_eigvals,
        neighbor_idf_mean=neighbor_idf_mean,
        maplet_type="unknown",
        maplet_k=int(maplet_k),
        context_pool=context_pool,
    )


def _replace_bank_type(bank: LocalMapletBank, maplet_type: str) -> LocalMapletBank:
    return LocalMapletBank(
        neighbor_indices=bank.neighbor_indices,
        context_features=bank.context_features,
        neighbor_counts=bank.neighbor_counts,
        context_radius=bank.context_radius,
        context_feature_variance=bank.context_feature_variance,
        covisibility_strength=bank.covisibility_strength,
        xyz_cov_eigvals=bank.xyz_cov_eigvals,
        neighbor_idf_mean=bank.neighbor_idf_mean,
        maplet_type=maplet_type,
        maplet_k=bank.maplet_k,
        context_pool=bank.context_pool,
    )


def build_knn_maplets(
    index: LandmarkMapIndex,
    maplet_k: int = 8,
    radius: float | None = None,
    context_pool: str = "quality_mean",
) -> LocalMapletBank:
    """Build local 3D-neighborhood maplets around each anchor landmark."""

    k = int(maplet_k)
    if k <= 0:
        raise ValueError("maplet_k must be positive")
    if len(index) == 0:
        empty = np.zeros((0, k), dtype=np.int64)
        return LocalMapletBank(
            empty,
            np.zeros((0, index.feature_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            "knn",
            k,
            context_pool,
        )
    neighbor_indices = np.full((len(index), k), -1, dtype=np.int64)
    effective = min(k + 1, len(index))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(index.xyz.astype(np.float64))
        distances, indices = tree.query(index.xyz.astype(np.float64), k=effective)
        if effective == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for row in range(len(index)):
            keep = []
            for distance, idx in zip(distances[row].tolist(), indices[row].tolist()):
                if int(idx) == row:
                    continue
                if radius is not None and float(distance) > float(radius):
                    continue
                keep.append(int(idx))
                if len(keep) >= k:
                    break
            if keep:
                neighbor_indices[row, : len(keep)] = np.asarray(keep, dtype=np.int64)
    except Exception:
        for start in range(0, len(index), 512):
            end = min(start + 512, len(index))
            deltas = index.xyz[start:end, None, :] - index.xyz[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            for local_row, row in enumerate(range(start, end)):
                distances[local_row, row] = np.inf
                if radius is not None:
                    distances[local_row, distances[local_row] > float(radius)] = np.inf
                order = np.argsort(distances[local_row])[:k]
                order = order[np.isfinite(distances[local_row, order])]
                if order.size:
                    neighbor_indices[row, : order.size] = order.astype(np.int64)
    return _replace_bank_type(_maplet_context_features(index, neighbor_indices, context_pool), "knn")


def build_covisibility_maplets(
    index: LandmarkMapIndex,
    reference_image_ids: Sequence[str] | None = None,
    maplet_k: int = 8,
    context_pool: str = "quality_mean",
) -> LocalMapletBank:
    """Build maplets from landmarks that are frequently co-visible with anchor."""

    k = int(maplet_k)
    if k <= 0:
        raise ValueError("maplet_k must be positive")
    references = None if reference_image_ids is None else {str(item) for item in reference_image_ids}
    image_to_indices: dict[str, list[int]] = {}
    for idx, image_ids in enumerate(index.observation_image_ids):
        for image_id in image_ids:
            if references is not None and str(image_id) not in references:
                continue
            image_to_indices.setdefault(str(image_id), []).append(int(idx))
    neighbor_indices = np.full((len(index), k), -1, dtype=np.int64)
    covis_counts = np.zeros((len(index), k), dtype=np.float32)
    for row, image_ids in enumerate(index.observation_image_ids):
        counter: dict[int, int] = {}
        for image_id in image_ids:
            if references is not None and str(image_id) not in references:
                continue
            for candidate in image_to_indices.get(str(image_id), ()):
                if int(candidate) == row:
                    continue
                counter[int(candidate)] = counter.get(int(candidate), 0) + 1
        if not counter:
            continue
        ordered = sorted(
            counter.items(),
            key=lambda item: (item[1], int(index.observation_counts[int(item[0])]), -int(item[0])),
            reverse=True,
        )[:k]
        neighbor_indices[row, : len(ordered)] = np.asarray([item[0] for item in ordered], dtype=np.int64)
        covis_counts[row, : len(ordered)] = np.asarray([item[1] for item in ordered], dtype=np.float32)
    return _replace_bank_type(_maplet_context_features(index, neighbor_indices, context_pool, covis_counts), "covis")


def build_hybrid_maplet_support_index(
    index: LandmarkMapIndex,
    *,
    maplet_k: int = 32,
    candidate_k: int = 128,
    max_support_views: int = 8,
    radius: float | None = None,
    context_pool: str = "quality_mean",
) -> LocalMapletSupportIndex:
    """Build bounded 3D-neighbor maplets reranked by true SfM covisibility."""

    if len(np.unique(index.track_ids)) != len(index):
        raise ValueError("maplet construction requires one row per unique track")
    if int(maplet_k) <= 0:
        raise ValueError("maplet_k must be positive")
    if int(candidate_k) < int(maplet_k):
        raise ValueError("candidate_k must be at least maplet_k")
    if int(max_support_views) <= 0:
        raise ValueError("max_support_views must be positive")
    if radius is not None and float(radius) <= 0.0:
        raise ValueError("radius must be positive when provided")
    count = len(index)
    maplet_size = int(maplet_k)
    candidate_count = int(candidate_k)
    image_vocab = tuple(sorted({str(image_id) for values in index.observation_image_ids for image_id in values}))
    image_to_index = {image_id: row for row, image_id in enumerate(image_vocab)}
    neighbor_indices = np.full((count, maplet_size), -1, dtype=np.int64)
    neighbor_track_ids = np.full((count, maplet_size), -1, dtype=np.int64)
    covisibility_counts = np.zeros((count, maplet_size), dtype=np.float32)
    support_image_indices = np.full((count, int(max_support_views)), -1, dtype=np.int64)
    support_coverage_counts = np.zeros((count, int(max_support_views)), dtype=np.int64)
    if count == 0:
        empty_maplets = _replace_bank_type(
            _maplet_context_features(index, neighbor_indices, context_pool, covisibility_counts),
            "knn_covisibility_hybrid",
        )
        return LocalMapletSupportIndex(
            maplets=empty_maplets,
            anchor_track_ids=index.track_ids,
            neighbor_track_ids=neighbor_track_ids,
            support_image_ids=image_vocab,
            support_image_indices=support_image_indices,
            support_coverage_counts=support_coverage_counts,
            candidate_k=candidate_count,
        )

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("scipy is required for full-bank hybrid maplet construction") from exc
    tree = cKDTree(np.asarray(index.xyz, dtype=np.float64))
    effective = min(count, candidate_count + 1)
    try:
        distances, candidates = tree.query(index.xyz, k=effective, workers=-1)
    except TypeError:  # pragma: no cover - old scipy compatibility
        distances, candidates = tree.query(index.xyz, k=effective)
    if effective == 1:
        distances = distances[:, None]
        candidates = candidates[:, None]
    image_sets = tuple(frozenset(str(item) for item in values) for values in index.observation_image_ids)
    observation_counts = np.asarray(index.observation_counts, dtype=np.int64)

    for row in range(count):
        anchor_images = image_sets[row]
        scored: list[tuple[int, float, int, int]] = []
        for distance, candidate in zip(distances[row].tolist(), candidates[row].tolist()):
            candidate_row = int(candidate)
            if candidate_row == row or not np.isfinite(float(distance)):
                continue
            if radius is not None and float(distance) > float(radius):
                continue
            shared = len(anchor_images.intersection(image_sets[candidate_row]))
            scored.append((candidate_row, float(distance), int(shared), int(observation_counts[candidate_row])))
        scored.sort(key=lambda item: (-item[2], item[1], -item[3], int(index.track_ids[item[0]])))
        selected = scored[:maplet_size]
        if selected:
            selected_rows = np.asarray([item[0] for item in selected], dtype=np.int64)
            neighbor_indices[row, : len(selected)] = selected_rows
            neighbor_track_ids[row, : len(selected)] = index.track_ids[selected_rows]
            covisibility_counts[row, : len(selected)] = np.asarray([item[2] for item in selected], dtype=np.float32)

        maplet_rows = [row] + [item[0] for item in selected]
        support_scores = [
            (
                str(image_id),
                sum(1 for maplet_row in maplet_rows if str(image_id) in image_sets[int(maplet_row)]),
            )
            for image_id in anchor_images
        ]
        support_scores.sort(key=lambda item: (-item[1], item[0]))
        for support_rank, (image_id, coverage) in enumerate(support_scores[: int(max_support_views)]):
            support_image_indices[row, support_rank] = int(image_to_index[image_id])
            support_coverage_counts[row, support_rank] = int(coverage)

    maplet_bank = _replace_bank_type(
        _maplet_context_features(index, neighbor_indices, context_pool, covisibility_counts),
        "knn_covisibility_hybrid",
    )
    return LocalMapletSupportIndex(
        maplets=maplet_bank,
        anchor_track_ids=index.track_ids,
        neighbor_track_ids=neighbor_track_ids,
        support_image_ids=image_vocab,
        support_image_indices=support_image_indices,
        support_coverage_counts=support_coverage_counts,
        candidate_k=candidate_count,
    )


def save_local_maplet_support_index_npz(
    index: LocalMapletSupportIndex,
    path: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bank = index.maplets
    payload = {
        "format": "local_maplet_support_index_npz",
        "format_version": 1,
        "version": str(index.version),
        **dict(metadata or {}),
    }
    np.savez(
        output,
        anchor_track_ids=index.anchor_track_ids,
        neighbor_indices=bank.neighbor_indices,
        neighbor_track_ids=index.neighbor_track_ids,
        context_features=bank.context_features,
        neighbor_counts=bank.neighbor_counts,
        context_radius=bank.context_radius,
        context_feature_variance=bank.context_feature_variance,
        covisibility_strength=bank.covisibility_strength,
        xyz_cov_eigvals=bank.xyz_cov_eigvals,
        neighbor_idf_mean=bank.neighbor_idf_mean,
        support_image_ids=np.asarray(index.support_image_ids, dtype=np.str_),
        support_image_indices=index.support_image_indices,
        support_coverage_counts=index.support_coverage_counts,
        maplet_type=np.asarray(bank.maplet_type, dtype=np.str_),
        maplet_k=np.asarray(bank.maplet_k, dtype=np.int64),
        context_pool=np.asarray(bank.context_pool, dtype=np.str_),
        candidate_k=np.asarray(index.candidate_k, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True), dtype=np.str_),
    )


def load_local_maplet_support_index_npz(path: Path) -> tuple[LocalMapletSupportIndex, dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        maplets = LocalMapletBank(
            neighbor_indices=np.asarray(data["neighbor_indices"], dtype=np.int64),
            context_features=np.asarray(data["context_features"], dtype=np.float32),
            neighbor_counts=np.asarray(data["neighbor_counts"], dtype=np.int64),
            context_radius=np.asarray(data["context_radius"], dtype=np.float32),
            context_feature_variance=np.asarray(data["context_feature_variance"], dtype=np.float32),
            covisibility_strength=np.asarray(data["covisibility_strength"], dtype=np.float32),
            xyz_cov_eigvals=np.asarray(data["xyz_cov_eigvals"], dtype=np.float32),
            neighbor_idf_mean=np.asarray(data["neighbor_idf_mean"], dtype=np.float32),
            maplet_type=str(data["maplet_type"].item()),
            maplet_k=int(data["maplet_k"].item()),
            context_pool=str(data["context_pool"].item()),
        )
        metadata = json.loads(str(data["metadata_json"].item()))
        output = LocalMapletSupportIndex(
            maplets=maplets,
            anchor_track_ids=np.asarray(data["anchor_track_ids"], dtype=np.int64),
            neighbor_track_ids=np.asarray(data["neighbor_track_ids"], dtype=np.int64),
            support_image_ids=tuple(str(item) for item in data["support_image_ids"].tolist()),
            support_image_indices=np.asarray(data["support_image_indices"], dtype=np.int64),
            support_coverage_counts=np.asarray(data["support_coverage_counts"], dtype=np.int64),
            candidate_k=int(data["candidate_k"].item()),
            version=str(metadata.get("version", "local_maplet_support_v1")),
        )
    return output, dict(metadata)


def build_ref_neighborhood_maplets(
    index: LandmarkMapIndex,
    observations_by_image: Mapping[str, Sequence[FootprintObservation]],
    reference_image_ids: Sequence[str],
    token_width: int,
    token_height: int,
    maplet_k: int = 8,
    cell_radius: int = 1,
    context_pool: str = "quality_mean",
) -> LocalMapletBank:
    """Build maplets from landmarks falling in nearby reference-view token cells."""

    k = int(maplet_k)
    if k <= 0:
        raise ValueError("maplet_k must be positive")
    if int(cell_radius) < 0:
        raise ValueError("cell_radius must be non-negative")
    track_to_idx = {int(track_id): idx for idx, track_id in enumerate(index.track_ids.tolist())}
    reference_set = {str(item) for item in reference_image_ids}
    cell_to_indices: dict[tuple[str, int], list[int]] = {}
    anchor_cells: dict[int, list[tuple[str, int]]] = {}
    for image_id in reference_set:
        for obs in observations_by_image.get(str(image_id), ()):
            idx = track_to_idx.get(int(obs.track_id))
            if idx is None:
                continue
            token = _token_index_from_xy(obs.xy, obs.image_width, obs.image_height, token_width, token_height)
            token_x = int(token) % int(token_width)
            token_y = int(token) // int(token_width)
            anchor_cells.setdefault(int(idx), []).append((str(image_id), int(token)))
            for yy in range(max(0, token_y - int(cell_radius)), min(int(token_height), token_y + int(cell_radius) + 1)):
                for xx in range(max(0, token_x - int(cell_radius)), min(int(token_width), token_x + int(cell_radius) + 1)):
                    cell_to_indices.setdefault((str(image_id), int(yy * int(token_width) + xx)), []).append(int(idx))
    neighbor_indices = np.full((len(index), k), -1, dtype=np.int64)
    support_counts = np.zeros((len(index), k), dtype=np.float32)
    for row, cells in anchor_cells.items():
        counter: dict[int, int] = {}
        for key in cells:
            for candidate in cell_to_indices.get(key, ()):
                if int(candidate) == int(row):
                    continue
                counter[int(candidate)] = counter.get(int(candidate), 0) + 1
        if not counter:
            continue
        ordered = sorted(
            counter.items(),
            key=lambda item: (item[1], int(index.observation_counts[int(item[0])]), -int(item[0])),
            reverse=True,
        )[:k]
        neighbor_indices[row, : len(ordered)] = np.asarray([item[0] for item in ordered], dtype=np.int64)
        support_counts[row, : len(ordered)] = np.asarray([item[1] for item in ordered], dtype=np.float32)
    return _replace_bank_type(_maplet_context_features(index, neighbor_indices, context_pool, support_counts), "ref_neighbor")


def local_maplet_bank_stats(bank: LocalMapletBank) -> dict[str, float | int | str]:
    counts = np.asarray(bank.neighbor_counts, dtype=np.float32)
    reliability = maplet_reliability_scores(bank)
    if counts.size == 0:
        return {
            "maplet_type": bank.maplet_type,
            "maplet_k": int(bank.maplet_k),
            "landmark_count": 0,
            "mean_neighbors": 0.0,
            "median_neighbors": 0.0,
            "zero_neighbor_ratio": 1.0,
            "mean_context_radius": 0.0,
            "mean_context_feature_variance": 0.0,
            "mean_covisibility_strength": 0.0,
            "mean_maplet_reliability": 0.0,
        }
    return {
        "maplet_type": bank.maplet_type,
        "maplet_k": int(bank.maplet_k),
        "landmark_count": int(counts.size),
        "mean_neighbors": float(np.mean(counts)),
        "median_neighbors": float(np.median(counts)),
        "zero_neighbor_ratio": float(np.mean(counts == 0.0)),
        "mean_context_radius": float(np.mean(bank.context_radius)),
        "mean_context_feature_variance": float(np.mean(bank.context_feature_variance)),
        "mean_covisibility_strength": float(np.mean(bank.covisibility_strength)),
        "mean_maplet_reliability": float(np.mean(reliability)) if reliability.size else 0.0,
    }


def local_maplet_support_index_stats(index: LocalMapletSupportIndex) -> dict[str, float | int | str]:
    base = local_maplet_bank_stats(index.maplets)
    valid_supports = index.support_image_indices >= 0
    support_counts = np.sum(valid_supports, axis=1) if len(index) else np.zeros((0,), dtype=np.int64)
    valid_coverages = index.support_coverage_counts[valid_supports]
    return {
        **base,
        "version": str(index.version),
        "candidate_k": int(index.candidate_k),
        "support_image_count": int(len(index.support_image_ids)),
        "mean_support_view_count": float(np.mean(support_counts)) if support_counts.size else 0.0,
        "min_support_view_count": int(np.min(support_counts)) if support_counts.size else 0,
        "mean_support_coverage_count": float(np.mean(valid_coverages)) if valid_coverages.size else 0.0,
    }


def _score_weights(config: ContextualLandmarkMatchingConfig) -> tuple[float, float, float]:
    if config.score_mode == "anchor_only":
        return float(config.anchor_weight), 0.0, 0.0
    if config.score_mode == "context_only":
        return 0.0, float(config.context_weight), 0.0
    if config.score_mode == "anchor_context_quality":
        return float(config.anchor_weight), float(config.context_weight), float(config.quality_weight)
    return float(config.anchor_weight), float(config.context_weight), 0.0


def _combined_topk(
    query_anchor: np.ndarray,
    query_context: np.ndarray,
    landmark_anchor: np.ndarray,
    landmark_context: np.ndarray,
    anchor_weight: float,
    context_weight: float,
    quality_bias: np.ndarray | None,
    top_k: int,
    block_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    query_count = int(query_anchor.shape[0])
    landmark_count = int(landmark_anchor.shape[0])
    effective_top_k = min(int(top_k), landmark_count)
    top_indices = np.full((query_count, effective_top_k), -1, dtype=np.int64)
    top_scores = np.full((query_count, effective_top_k), -np.inf, dtype=np.float32)
    if landmark_count == 0 or query_count == 0 or effective_top_k == 0:
        return top_indices, top_scores
    if str(device).lower() != "cpu":
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Torch is required for non-CPU contextual matching similarity") from exc
        requested = str(device).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested != "cpu":
            torch_device = torch.device(requested)
            anchor_t = torch.as_tensor(landmark_anchor, dtype=torch.float32, device=torch_device).T.contiguous()
            context_t = torch.as_tensor(landmark_context, dtype=torch.float32, device=torch_device).T.contiguous()
            bias_t = None if quality_bias is None else torch.as_tensor(quality_bias, dtype=torch.float32, device=torch_device)
            for start in range(0, query_count, int(block_size)):
                end = min(start + int(block_size), query_count)
                qa = torch.as_tensor(query_anchor[start:end], dtype=torch.float32, device=torch_device)
                qc = torch.as_tensor(query_context[start:end], dtype=torch.float32, device=torch_device)
                scores = float(anchor_weight) * torch.matmul(qa, anchor_t)
                if float(context_weight) != 0.0:
                    scores = scores + float(context_weight) * torch.matmul(qc, context_t)
                if bias_t is not None:
                    scores = scores + bias_t.reshape(1, -1)
                local_scores, local_indices = torch.topk(scores, k=effective_top_k, dim=1, largest=True, sorted=True)
                top_indices[start:end] = local_indices.detach().cpu().numpy().astype(np.int64)
                top_scores[start:end] = local_scores.detach().cpu().numpy().astype(np.float32)
            return top_indices, top_scores
    bias = None if quality_bias is None else np.asarray(quality_bias, dtype=np.float32).reshape(1, -1)
    for start in range(0, query_count, int(block_size)):
        end = min(start + int(block_size), query_count)
        scores = float(anchor_weight) * (query_anchor[start:end] @ landmark_anchor.T)
        if float(context_weight) != 0.0:
            scores = scores + float(context_weight) * (query_context[start:end] @ landmark_context.T)
        if bias is not None:
            scores = scores + bias
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


def _gated_context_lambdas(
    anchor_margin: np.ndarray,
    context_margin: np.ndarray,
    maplet_reliability: np.ndarray,
    config: ContextualLandmarkMatchingConfig,
) -> np.ndarray:
    mode = str(config.gate_mode)
    base = np.full_like(context_margin, float(config.context_weight), dtype=np.float32)
    if mode == "none":
        return base
    anchor_factor = _sigmoid(float(config.anchor_margin_slope) * (float(config.anchor_margin_tau) - anchor_margin))
    context_factor = _sigmoid(float(config.context_margin_slope) * (context_margin - float(config.context_margin_tau)))
    maplet_factor = np.asarray(maplet_reliability, dtype=np.float32)
    if mode == "anchor_margin":
        return (base * anchor_factor[:, None]).astype(np.float32, copy=False)
    if mode == "maplet":
        return (base * maplet_factor).astype(np.float32, copy=False)
    if mode == "anchor_maplet":
        return (base * anchor_factor[:, None] * maplet_factor).astype(np.float32, copy=False)
    return (base * anchor_factor[:, None] * context_factor * maplet_factor).astype(np.float32, copy=False)


def _gated_anchor_topm_rerank(
    query_anchor: np.ndarray,
    query_context: np.ndarray,
    landmark_anchor: np.ndarray,
    landmark_context: np.ndarray,
    quality_bias: np.ndarray | None,
    maplet_reliability: np.ndarray,
    config: ContextualLandmarkMatchingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    top_m = max(int(config.top_m_anchor), int(config.top_k), 2)
    anchor_indices, anchor_scores = _combined_topk(
        query_anchor,
        query_context,
        landmark_anchor,
        landmark_context,
        anchor_weight=1.0,
        context_weight=0.0,
        quality_bias=None,
        top_k=top_m,
        block_size=int(config.block_size),
        device=config.similarity_device,
    )
    if anchor_indices.size == 0:
        return anchor_indices, anchor_scores
    candidate_context_scores = np.full_like(anchor_scores, -np.inf, dtype=np.float32)
    for row in range(anchor_indices.shape[0]):
        valid = anchor_indices[row] >= 0
        if not np.any(valid):
            continue
        candidates = anchor_indices[row, valid]
        candidate_context_scores[row, valid] = query_context[row] @ landmark_context[candidates].T
    if anchor_scores.shape[1] > 1:
        anchor_margin = anchor_scores[:, 0] - anchor_scores[:, 1]
    else:
        anchor_margin = np.zeros((anchor_scores.shape[0],), dtype=np.float32)
    context_margin = np.zeros_like(candidate_context_scores, dtype=np.float32)
    for row in range(candidate_context_scores.shape[0]):
        for col in range(candidate_context_scores.shape[1]):
            if anchor_indices[row, col] < 0:
                continue
            others = np.delete(candidate_context_scores[row], col)
            others = others[np.isfinite(others)]
            best_other = float(np.max(others)) if others.size else 0.0
            context_margin[row, col] = float(candidate_context_scores[row, col] - best_other)
    candidate_reliability = np.zeros_like(candidate_context_scores, dtype=np.float32)
    for row in range(anchor_indices.shape[0]):
        valid = anchor_indices[row] >= 0
        if np.any(valid):
            candidate_reliability[row, valid] = maplet_reliability[anchor_indices[row, valid]]
    lambdas = _gated_context_lambdas(anchor_margin, context_margin, candidate_reliability, config)
    final_scores = float(config.anchor_weight) * anchor_scores + lambdas * candidate_context_scores
    if quality_bias is not None:
        for row in range(anchor_indices.shape[0]):
            valid = anchor_indices[row] >= 0
            if np.any(valid):
                final_scores[row, valid] += quality_bias[anchor_indices[row, valid]]
    final_scores[anchor_indices < 0] = -np.inf
    keep_k = min(int(config.top_k), final_scores.shape[1])
    final_indices = np.full((anchor_indices.shape[0], keep_k), -1, dtype=np.int64)
    sorted_scores = np.full((anchor_indices.shape[0], keep_k), -np.inf, dtype=np.float32)
    for row in range(anchor_indices.shape[0]):
        order = np.argsort(-final_scores[row])[:keep_k]
        final_indices[row, : order.size] = anchor_indices[row, order]
        sorted_scores[row, : order.size] = final_scores[row, order]
    return final_indices, sorted_scores


def match_query_patches_to_contextual_landmarks(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    maplet_bank: LocalMapletBank,
    config: ContextualLandmarkMatchingConfig | None = None,
    image_width: int = 1024,
    image_height: int = 576,
) -> list[QueryTo3DMatch]:
    """Match query patches to single 3D anchors using local context for scoring."""

    config = config or ContextualLandmarkMatchingConfig()
    if len(landmark_index) == 0 or len(maplet_bank) == 0:
        return []
    if len(landmark_index) != len(maplet_bank):
        raise ValueError("landmark_index and maplet_bank must contain the same number of landmarks")
    query_anchor_map = np.asarray(query_feature_map, dtype=np.float32)
    query_context_map = compute_query_context_descriptors(query_anchor_map, config.query_context)
    query_anchor, token_indices = _flatten_token_features(query_anchor_map, config.query_token_step)
    query_context, context_token_indices = _flatten_token_features(query_context_map, config.query_token_step)
    if not np.array_equal(token_indices, context_token_indices):
        raise RuntimeError("query anchor/context token indexing mismatch")
    query_anchor, valid_query = normalize_rows(query_anchor)
    query_context, valid_context = normalize_rows(query_context)
    valid_rows = np.flatnonzero(valid_query & valid_context)
    if valid_rows.size == 0:
        return []
    query_anchor = query_anchor[valid_rows]
    query_context = query_context[valid_rows]
    token_indices = token_indices[valid_rows]

    landmark_anchor, valid_landmarks = normalize_rows(landmark_index.features)
    landmark_context, valid_context_landmarks = normalize_rows(maplet_bank.context_features)
    valid_map = valid_landmarks & valid_context_landmarks
    all_maplet_reliability = maplet_reliability_scores(maplet_bank)
    if not np.any(valid_map):
        return []
    if not np.all(valid_map):
        index = landmark_index.subset(valid_map)
        landmark_anchor = landmark_anchor[valid_map]
        landmark_context = landmark_context[valid_map]
        maplet_neighbor_counts = maplet_bank.neighbor_counts[valid_map]
        maplet_reliability = all_maplet_reliability[valid_map]
    else:
        index = landmark_index
        maplet_neighbor_counts = maplet_bank.neighbor_counts
        maplet_reliability = all_maplet_reliability
    quality = _landmark_quality_weights(index)
    anchor_weight, context_weight, quality_weight = _score_weights(config)
    quality_bias = None
    if quality_weight != 0.0:
        quality_bias = float(quality_weight) * quality
    use_gated_topm = int(config.top_m_anchor) > 0
    if use_gated_topm:
        query_top_indices, query_top_scores = _gated_anchor_topm_rerank(
            query_anchor,
            query_context,
            landmark_anchor,
            landmark_context,
            quality_bias,
            maplet_reliability,
            config,
        )
    else:
        query_top_indices, query_top_scores = _combined_topk(
            query_anchor,
            query_context,
            landmark_anchor,
            landmark_context,
            anchor_weight,
            context_weight,
            quality_bias,
            top_k=int(config.top_k),
            block_size=int(config.block_size),
            device=config.similarity_device,
        )
    if config.match_mode in {"mnn", "soft_mutual"}:
        if use_gated_topm:
            landmark_top_indices, _landmark_top_scores = _combined_topk(
                landmark_anchor,
                landmark_context,
                query_anchor,
                query_context,
                anchor_weight=1.0,
                context_weight=0.0,
                quality_bias=None,
                top_k=int(config.mutual_top_k),
                block_size=int(config.block_size),
                device=config.similarity_device,
            )
        else:
            landmark_top_indices, _landmark_top_scores = _combined_topk(
                landmark_anchor,
                landmark_context,
                query_anchor,
                query_context,
                anchor_weight,
                context_weight,
                None,
                top_k=int(config.mutual_top_k),
                block_size=int(config.block_size),
                device=config.similarity_device,
            )
        reciprocal = [set(row.tolist()) for row in landmark_top_indices]
    else:
        reciprocal = []

    token_height = int(query_feature_map.shape[1])
    token_width = int(query_feature_map.shape[2])
    centers = token_grid_xy(token_width, token_height, image_width, image_height, step=1)
    matches: list[QueryTo3DMatch] = []
    for query_row in range(query_anchor.shape[0]):
        for rank in range(query_top_indices.shape[1]):
            landmark_idx = int(query_top_indices[query_row, rank])
            if landmark_idx < 0:
                continue
            if config.match_mode == "nn" and rank > 0:
                continue
            if config.match_mode == "mnn" and (rank > 0 or query_row not in reciprocal[landmark_idx]):
                continue
            if config.match_mode == "soft_mutual" and query_row not in reciprocal[landmark_idx]:
                continue
            score = float(query_top_scores[query_row, rank])
            if score < float(config.min_similarity):
                continue
            second_similarity = None
            if query_top_scores.shape[1] > 1:
                other = np.delete(query_top_scores[query_row], rank)
                if other.size:
                    second_similarity = float(np.max(other))
            margin = None if second_similarity is None else float(score - second_similarity)
            token_index = int(token_indices[query_row])
            xy = centers[token_index].astype(np.float64, copy=True)
            boundary = min(
                float(xy[0]),
                float(xy[1]),
                float(image_width - 1) - float(xy[0]),
                float(image_height - 1) - float(xy[1]),
            )
            matches.append(
                QueryTo3DMatch(
                    token_index=token_index,
                    xy=xy,
                    track_id=int(index.track_ids[landmark_idx]),
                    xyz=index.xyz[landmark_idx].astype(np.float64, copy=True),
                    similarity=score,
                    ratio=0.0,
                    landmark_variance=float(index.mean_variances[landmark_idx]),
                    source=(
                        f"contextual_landmark_{maplet_bank.maplet_type}_k{int(maplet_bank.maplet_k)}"
                        + ("_gated" if use_gated_topm and config.gate_mode != "none" else "")
                    ),
                    observation_count=int(index.observation_counts[landmark_idx]),
                    visibility_count=len(index.observation_image_ids[landmark_idx]),
                    landmark_reprojection_error=float(index.reprojection_errors[landmark_idx]),
                    landmark_quality=float(quality[landmark_idx]),
                    quality_weighted_similarity=score,
                    similarity_margin=margin,
                    distance_to_boundary_px=float(boundary),
                    local_consistency_support=int(maplet_neighbor_counts[landmark_idx]),
                    local_consistency_score=float(maplet_neighbor_counts[landmark_idx]) / max(float(maplet_bank.maplet_k), 1.0),
                )
            )
    matches.sort(key=lambda item: float(item.quality_weighted_similarity or item.similarity), reverse=True)
    if config.max_matches is not None:
        matches = matches[: int(config.max_matches)]
    return matches
