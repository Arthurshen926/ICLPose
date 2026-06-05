"""Dense patch context around sparse 3D anchors.

This module keeps sparse SfM landmarks as the only geometric PnP anchors.
Nearby semi-dense Gaussian descriptors are aggregated into an anchor-local
context descriptor used for diagnostics or top-M sparse-anchor reranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.patch_to_3d_matching import _flatten_query_features, _query_landmark_topk
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch, normalize_rows, token_grid_xy
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap


@dataclass(frozen=True)
class DensePatchContextConfig:
    max_radius_m: float = 0.10
    max_support: int = 16
    min_support: int = 1
    prototype_count: int = 4
    quality_power: float = 1.0
    distance_power: float = 1.0
    include_radius_fallback: bool = True
    l2_normalize: bool = True

    def __post_init__(self) -> None:
        if float(self.max_radius_m) <= 0.0:
            raise ValueError("max_radius_m must be positive")
        if int(self.max_support) <= 0:
            raise ValueError("max_support must be positive")
        if int(self.min_support) <= 0:
            raise ValueError("min_support must be positive")
        if int(self.prototype_count) <= 0:
            raise ValueError("prototype_count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_radius_m": float(self.max_radius_m),
            "max_support": int(self.max_support),
            "min_support": int(self.min_support),
            "prototype_count": int(self.prototype_count),
            "quality_power": float(self.quality_power),
            "distance_power": float(self.distance_power),
            "include_radius_fallback": bool(self.include_radius_fallback),
            "l2_normalize": bool(self.l2_normalize),
        }


@dataclass(frozen=True)
class DensePatchContextBank:
    track_ids: np.ndarray
    anchor_xyz: np.ndarray
    anchor_features: np.ndarray
    support_indices: np.ndarray
    support_counts: np.ndarray
    mean_features: np.ndarray
    medoid_features: np.ndarray
    prototype_features: np.ndarray
    prototype_counts: np.ndarray
    feature_variance: np.ndarray
    support_radius: np.ndarray
    support_quality: np.ndarray
    reliability_scores: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        count = int(track_ids.shape[0])
        anchor_xyz = np.asarray(self.anchor_xyz, dtype=np.float64)
        anchor_features = np.asarray(self.anchor_features, dtype=np.float32)
        mean_features = np.asarray(self.mean_features, dtype=np.float32)
        medoid_features = np.asarray(self.medoid_features, dtype=np.float32)
        prototype_features = np.asarray(self.prototype_features, dtype=np.float32)
        if anchor_xyz.shape != (count, 3):
            raise ValueError("anchor_xyz must have shape (N, 3)")
        if anchor_features.ndim != 2 or anchor_features.shape[0] != count:
            raise ValueError("anchor_features must have shape (N, C)")
        if mean_features.shape != anchor_features.shape:
            raise ValueError("mean_features must have shape (N, C)")
        if medoid_features.shape != anchor_features.shape:
            raise ValueError("medoid_features must have shape (N, C)")
        if prototype_features.ndim != 3 or prototype_features.shape[0] != count:
            raise ValueError("prototype_features must have shape (N, K, C)")
        if prototype_features.shape[2] != anchor_features.shape[1]:
            raise ValueError("prototype feature dimension must match anchor_features")
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "anchor_xyz", anchor_xyz)
        object.__setattr__(self, "anchor_features", anchor_features)
        object.__setattr__(self, "mean_features", mean_features)
        object.__setattr__(self, "medoid_features", medoid_features)
        object.__setattr__(self, "prototype_features", prototype_features)
        for name in ("support_indices",):
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if value.ndim != 2 or value.shape[0] != count:
                raise ValueError(f"{name} must have shape (N, S)")
            object.__setattr__(self, name, value)
        for name in ("support_counts", "prototype_counts"):
            value = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if value.shape[0] != count:
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        for name in ("feature_variance", "support_radius", "support_quality", "reliability_scores"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape[0] != count:
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __len__(self) -> int:
        return int(self.track_ids.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.anchor_features.shape[1]) if self.anchor_features.ndim == 2 else 0

    @property
    def prototype_count(self) -> int:
        return int(self.prototype_features.shape[1]) if self.prototype_features.ndim == 3 else 0

    def track_to_row(self) -> dict[int, int]:
        return {int(track_id): int(row) for row, track_id in enumerate(self.track_ids.tolist())}

    def subset(self, indices: Sequence[int] | np.ndarray) -> "DensePatchContextBank":
        idx = np.asarray(indices)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64).reshape(-1)
        return DensePatchContextBank(
            track_ids=self.track_ids[idx],
            anchor_xyz=self.anchor_xyz[idx],
            anchor_features=self.anchor_features[idx],
            support_indices=self.support_indices[idx],
            support_counts=self.support_counts[idx],
            mean_features=self.mean_features[idx],
            medoid_features=self.medoid_features[idx],
            prototype_features=self.prototype_features[idx],
            prototype_counts=self.prototype_counts[idx],
            feature_variance=self.feature_variance[idx],
            support_radius=self.support_radius[idx],
            support_quality=self.support_quality[idx],
            reliability_scores=self.reliability_scores[idx],
            metadata=self.metadata,
        )

    def save_npz(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            track_ids=self.track_ids.astype(np.int64, copy=False),
            anchor_xyz=self.anchor_xyz.astype(np.float32, copy=False),
            anchor_features=self.anchor_features.astype(np.float32, copy=False),
            support_indices=self.support_indices.astype(np.int64, copy=False),
            support_counts=self.support_counts.astype(np.int64, copy=False),
            mean_features=self.mean_features.astype(np.float32, copy=False),
            medoid_features=self.medoid_features.astype(np.float32, copy=False),
            prototype_features=self.prototype_features.astype(np.float32, copy=False),
            prototype_counts=self.prototype_counts.astype(np.int64, copy=False),
            feature_variance=self.feature_variance.astype(np.float32, copy=False),
            support_radius=self.support_radius.astype(np.float32, copy=False),
            support_quality=self.support_quality.astype(np.float32, copy=False),
            reliability_scores=self.reliability_scores.astype(np.float32, copy=False),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "DensePatchContextBank":
        with np.load(Path(path), allow_pickle=True) as data:
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"].tolist()))
            return cls(
                track_ids=np.asarray(data["track_ids"], dtype=np.int64),
                anchor_xyz=np.asarray(data["anchor_xyz"], dtype=np.float64),
                anchor_features=np.asarray(data["anchor_features"], dtype=np.float32),
                support_indices=np.asarray(data["support_indices"], dtype=np.int64),
                support_counts=np.asarray(data["support_counts"], dtype=np.int64),
                mean_features=np.asarray(data["mean_features"], dtype=np.float32),
                medoid_features=np.asarray(data["medoid_features"], dtype=np.float32),
                prototype_features=np.asarray(data["prototype_features"], dtype=np.float32),
                prototype_counts=np.asarray(data["prototype_counts"], dtype=np.int64),
                feature_variance=np.asarray(data["feature_variance"], dtype=np.float32),
                support_radius=np.asarray(data["support_radius"], dtype=np.float32),
                support_quality=np.asarray(data["support_quality"], dtype=np.float32),
                reliability_scores=np.asarray(data["reliability_scores"], dtype=np.float32),
                metadata=metadata,
            )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32, copy=False)


def _weighted_mean(features: np.ndarray, weights: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return _normalize_vector(fallback)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if not np.any(weights > 0.0):
        weights = np.ones((features.shape[0],), dtype=np.float32)
    mean = np.sum(np.asarray(features, dtype=np.float32) * weights[:, None], axis=0) / max(float(np.sum(weights)), 1e-6)
    return _normalize_vector(mean)


def _medoid(features: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return _normalize_vector(fallback)
    normalized, _valid = normalize_rows(features)
    sims = normalized @ normalized.T
    return normalized[int(np.argmax(np.mean(sims, axis=1)))].astype(np.float32, copy=False)


def _prototype_features(features: np.ndarray, weights: np.ndarray, prototype_count: int, fallback: np.ndarray) -> tuple[np.ndarray, int]:
    count = int(prototype_count)
    output = np.zeros((count, np.asarray(fallback).reshape(-1).shape[0]), dtype=np.float32)
    if features.shape[0] == 0:
        output[0] = _normalize_vector(fallback)
        return output, 1
    normalized, _valid = normalize_rows(features)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    order = np.argsort(-weights)
    chosen: list[int] = []
    for idx in order.tolist():
        if len(chosen) >= count:
            break
        if not chosen:
            chosen.append(int(idx))
            continue
        sims = normalized[int(idx)] @ normalized[np.asarray(chosen, dtype=np.int64)].T
        if float(np.max(sims)) < 0.98 or len(chosen) + 1 == min(count, features.shape[0]):
            chosen.append(int(idx))
    if not chosen:
        chosen = [int(order[0])]
    for col, idx in enumerate(chosen[:count]):
        output[col] = normalized[int(idx)]
    return output, int(min(len(chosen), count))


def _reliability(
    support_count: int,
    max_support: int,
    feature_variance: float,
    support_radius: float,
    max_radius: float,
    support_quality: float,
) -> float:
    count_score = min(float(support_count) / max(float(max_support), 1.0), 1.0)
    variance_score = 1.0 / (1.0 + max(float(feature_variance), 0.0) * 10.0)
    radius_score = 1.0 - min(max(float(support_radius) / max(float(max_radius), 1e-6), 0.0), 1.0)
    return float(np.clip(count_score * variance_score * radius_score * max(float(support_quality), 0.0), 0.0, 1.0))


def build_dense_patch_context_bank(
    sparse_index: LandmarkMapIndex,
    semidense_map: SemiDenseAnchorMap,
    config: DensePatchContextConfig | None = None,
) -> DensePatchContextBank:
    """Aggregate Gaussian support descriptors around each sparse landmark."""

    config = config or DensePatchContextConfig()
    if sparse_index.feature_dim != semidense_map.feature_dim:
        raise ValueError("sparse_index and semidense_map feature dimensions must match")
    gaussian_mask = np.asarray(semidense_map.source_types, dtype=str) != "sfm"
    gaussian_rows = np.flatnonzero(gaussian_mask)
    support_indices = np.full((len(sparse_index), int(config.max_support)), -1, dtype=np.int64)
    mean_features = np.zeros((len(sparse_index), sparse_index.feature_dim), dtype=np.float32)
    medoid_features = np.zeros_like(mean_features)
    prototype_features = np.zeros((len(sparse_index), int(config.prototype_count), sparse_index.feature_dim), dtype=np.float32)
    support_counts = np.zeros((len(sparse_index),), dtype=np.int64)
    prototype_counts = np.zeros((len(sparse_index),), dtype=np.int64)
    feature_variance = np.zeros((len(sparse_index),), dtype=np.float32)
    support_radius = np.zeros((len(sparse_index),), dtype=np.float32)
    support_quality = np.zeros((len(sparse_index),), dtype=np.float32)
    reliability_scores = np.zeros((len(sparse_index),), dtype=np.float32)
    rows_by_source_track: dict[int, list[int]] = {}
    for row in gaussian_rows.tolist():
        rows_by_source_track.setdefault(int(semidense_map.source_track_ids[int(row)]), []).append(int(row))
    gaussian_tree = None
    if bool(config.include_radius_fallback) and gaussian_rows.size:
        gaussian_tree = cKDTree(np.asarray(semidense_map.xyz[gaussian_rows], dtype=np.float64))

    for row, track_id in enumerate(sparse_index.track_ids.tolist()):
        candidate_rows = list(rows_by_source_track.get(int(track_id), ()))
        if gaussian_tree is not None:
            local_gaussian_offsets = gaussian_tree.query_ball_point(
                np.asarray(sparse_index.xyz[row], dtype=np.float64).reshape(3),
                r=float(config.max_radius_m),
            )
            if local_gaussian_offsets:
                candidate_rows.extend(gaussian_rows[np.asarray(local_gaussian_offsets, dtype=np.int64)].astype(np.int64).tolist())
        if candidate_rows:
            local_rows = np.asarray(sorted(set(int(item) for item in candidate_rows)), dtype=np.int64)
            local_distances = np.linalg.norm(semidense_map.xyz[local_rows] - sparse_index.xyz[row], axis=1)
            keep = local_distances <= float(config.max_radius_m)
            local_rows = local_rows[keep]
            local_distances = local_distances[keep]
        else:
            local_rows = np.zeros((0,), dtype=np.int64)
            local_distances = np.zeros((0,), dtype=np.float32)
        if local_rows.size < int(config.min_support):
            local_rows = np.zeros((0,), dtype=np.int64)
            local_distances = np.zeros((0,), dtype=np.float32)
        if local_rows.size > 0:
            qualities = np.asarray(semidense_map.quality_scores[local_rows], dtype=np.float32)
            distance_weight = 1.0 / np.maximum(local_distances.astype(np.float32), 1e-4)
            weights = np.power(np.clip(qualities, 0.0, 1.0), float(config.quality_power)) * np.power(
                distance_weight,
                float(config.distance_power),
            )
            order = np.lexsort((local_rows, -weights))[: int(config.max_support)]
            local_rows = local_rows[order]
            local_distances = local_distances[order]
            weights = weights[order]
            features = np.asarray(semidense_map.features[local_rows], dtype=np.float32)
            support_indices[row, : local_rows.size] = local_rows.astype(np.int64)
            support_counts[row] = int(local_rows.size)
            mean_features[row] = _weighted_mean(features, weights, sparse_index.features[row])
            medoid_features[row] = _medoid(features, sparse_index.features[row])
            prototype_features[row], prototype_counts[row] = _prototype_features(
                features,
                weights,
                int(config.prototype_count),
                sparse_index.features[row],
            )
            if features.shape[0] > 1:
                feature_variance[row] = float(np.mean(np.var(normalize_rows(features)[0], axis=0)))
            support_radius[row] = float(np.mean(local_distances)) if local_distances.size else 0.0
            support_quality[row] = float(np.mean(qualities[order])) if qualities.size else 0.0
        else:
            mean_features[row] = _normalize_vector(sparse_index.features[row])
            medoid_features[row] = mean_features[row]
            prototype_features[row, 0] = mean_features[row]
            prototype_counts[row] = 1
            support_quality[row] = 0.0
        reliability_scores[row] = _reliability(
            int(support_counts[row]),
            int(config.max_support),
            float(feature_variance[row]),
            float(support_radius[row]),
            float(config.max_radius_m),
            float(support_quality[row]),
        )
    if bool(config.l2_normalize):
        anchor_features, _ = normalize_rows(sparse_index.features)
        mean_features, _ = normalize_rows(mean_features)
        medoid_features, _ = normalize_rows(medoid_features)
        flat_proto, _ = normalize_rows(prototype_features.reshape(-1, sparse_index.feature_dim))
        prototype_features = flat_proto.reshape(prototype_features.shape)
    else:
        anchor_features = np.asarray(sparse_index.features, dtype=np.float32)
    return DensePatchContextBank(
        track_ids=sparse_index.track_ids,
        anchor_xyz=sparse_index.xyz,
        anchor_features=anchor_features,
        support_indices=support_indices,
        support_counts=support_counts,
        mean_features=mean_features,
        medoid_features=medoid_features,
        prototype_features=prototype_features,
        prototype_counts=prototype_counts,
        feature_variance=feature_variance,
        support_radius=support_radius,
        support_quality=support_quality,
        reliability_scores=reliability_scores,
        metadata={"config": config.to_dict()},
    )


def align_dense_context_to_landmarks(
    landmark_index: LandmarkMapIndex,
    context_bank: DensePatchContextBank,
) -> tuple[LandmarkMapIndex, DensePatchContextBank]:
    """Keep only landmarks that have a dense context row and align row order."""

    track_to_context = context_bank.track_to_row()
    landmark_rows = []
    context_rows = []
    for row, track_id in enumerate(landmark_index.track_ids.tolist()):
        context_row = track_to_context.get(int(track_id))
        if context_row is None:
            continue
        landmark_rows.append(int(row))
        context_rows.append(int(context_row))
    if not landmark_rows:
        return landmark_index.subset([]), context_bank.subset([])
    return landmark_index.subset(np.asarray(landmark_rows, dtype=np.int64)), context_bank.subset(
        np.asarray(context_rows, dtype=np.int64)
    )


def dense_context_scores(
    query_features: np.ndarray,
    context_bank: DensePatchContextBank,
    anchor_rows: np.ndarray,
    mode: str = "max_proto",
    topk_average_k: int = 3,
    apply_reliability: bool = False,
    require_support: bool = True,
) -> np.ndarray:
    """Score query descriptors against dense context rows."""

    if mode not in {"mean", "medoid", "max_proto", "topk_avg"}:
        raise ValueError("mode must be one of: mean, medoid, max_proto, topk_avg")
    query, _valid = normalize_rows(np.asarray(query_features, dtype=np.float32).reshape(-1, context_bank.feature_dim))
    rows = np.asarray(anchor_rows, dtype=np.int64).reshape(-1)
    if query.shape[0] != rows.shape[0]:
        raise ValueError("query_features and anchor_rows must have the same row count")
    scores = np.zeros((rows.shape[0],), dtype=np.float32)
    valid = (rows >= 0) & (rows < len(context_bank))
    scores[~valid] = -np.inf
    if bool(require_support):
        supported = np.zeros_like(valid, dtype=bool)
        supported[valid] = context_bank.support_counts[rows[valid]] > 0
        active = valid & supported
    else:
        active = valid
    if not np.any(active):
        return scores.astype(np.float32, copy=False)

    active_rows = rows[active]
    active_query = query[active]
    if mode == "mean":
        active_scores = np.sum(active_query * context_bank.mean_features[active_rows], axis=1)
    elif mode == "medoid":
        active_scores = np.sum(active_query * context_bank.medoid_features[active_rows], axis=1)
    else:
        prototypes = context_bank.prototype_features[active_rows]
        proto_scores = np.einsum("nkc,nc->nk", prototypes, active_query, optimize=True)
        counts = np.maximum(context_bank.prototype_counts[active_rows], 1)
        valid_proto = np.arange(prototypes.shape[1])[None, :] < counts[:, None]
        proto_scores = np.where(valid_proto, proto_scores, -np.inf)
        if mode == "topk_avg":
            k = max(1, int(topk_average_k))
            k = min(k, proto_scores.shape[1])
            topk = np.sort(proto_scores, axis=1)[:, -k:]
            finite = np.isfinite(topk)
            active_scores = np.sum(np.where(finite, topk, 0.0), axis=1) / np.maximum(
                np.sum(finite, axis=1),
                1,
            )
        else:
            active_scores = np.max(proto_scores, axis=1)
    if bool(apply_reliability):
        active_scores = active_scores * context_bank.reliability_scores[active_rows]
    scores[active] = np.asarray(active_scores, dtype=np.float32)
    return scores.astype(np.float32, copy=False)


def dense_context_topm_candidates(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    context_bank: DensePatchContextBank,
    top_m: int = 5,
    query_token_step: int = 1,
    context_mode: str = "max_proto",
    topk_average_k: int = 3,
    image_width: int = 1024,
    image_height: int = 576,
    block_size: int = 1024,
    similarity_device: str = "cpu",
    min_anchor_similarity: float = -1.0,
    apply_reliability: bool = False,
    require_support: bool = True,
) -> list[tuple[QueryTo3DMatch, float, float, int]]:
    """Return anchor-topM candidates with separate anchor and dense-context scores.

    The returned tuple is `(match, anchor_score, context_score, anchor_row)`.  The
    `QueryTo3DMatch.similarity` field stores the anchor-only score so consumers
    can perform controlled reranking without changing the geometric 3D point.
    """

    if len(landmark_index) != len(context_bank):
        raise ValueError("landmark_index and context_bank must be aligned")
    if len(landmark_index) == 0:
        return []
    query_features, token_indices = _flatten_query_features(query_feature_map, int(query_token_step))
    query_features, valid_query = normalize_rows(query_features)
    valid_rows = np.flatnonzero(valid_query)
    if valid_rows.size == 0:
        return []
    query_features = query_features[valid_rows]
    token_indices = token_indices[valid_rows]
    landmark_features, valid_landmarks = normalize_rows(landmark_index.features)
    if not np.all(valid_landmarks):
        landmark_index = landmark_index.subset(valid_landmarks)
        context_bank = context_bank.subset(valid_landmarks)
        landmark_features = landmark_features[valid_landmarks]
    if len(landmark_index) == 0:
        return []
    top_indices, top_scores = _query_landmark_topk(
        query_features,
        landmark_features,
        top_k=int(top_m),
        block_size=int(block_size),
        device=similarity_device,
    )
    token_height = int(query_feature_map.shape[1])
    token_width = int(query_feature_map.shape[2])
    centers = token_grid_xy(token_width, token_height, int(image_width), int(image_height), step=1)
    query_rows = np.repeat(np.arange(top_indices.shape[0], dtype=np.int64), top_indices.shape[1])
    landmark_rows = top_indices.reshape(-1).astype(np.int64, copy=False)
    anchor_scores = top_scores.reshape(-1).astype(np.float32, copy=False)
    keep = (landmark_rows >= 0) & (anchor_scores >= float(min_anchor_similarity))
    query_rows = query_rows[keep]
    landmark_rows = landmark_rows[keep]
    anchor_scores = anchor_scores[keep]
    if landmark_rows.size == 0:
        return []
    context_scores = dense_context_scores(
        query_features[query_rows],
        context_bank,
        landmark_rows,
        mode=context_mode,
        topk_average_k=topk_average_k,
        apply_reliability=apply_reliability,
        require_support=require_support,
    )
    output: list[tuple[QueryTo3DMatch, float, float, int]] = []
    for flat_idx, (query_row, landmark_row, anchor_score, context_score) in enumerate(
        zip(query_rows.tolist(), landmark_rows.tolist(), anchor_scores.tolist(), context_scores.tolist())
    ):
        token = int(token_indices[int(query_row)])
        second = None
        if top_scores.shape[1] > 1:
            other = top_scores[int(query_row)][top_indices[int(query_row)] != int(landmark_row)]
            if other.size:
                second = float(np.max(other))
        margin = None if second is None else float(anchor_score - second)
        xy = centers[token].astype(np.float64, copy=True)
        match = QueryTo3DMatch(
            token_index=token,
            xy=xy,
            track_id=int(landmark_index.track_ids[int(landmark_row)]),
            xyz=landmark_index.xyz[int(landmark_row)].astype(np.float64, copy=True),
            similarity=float(anchor_score),
            ratio=0.0,
            landmark_variance=float(landmark_index.mean_variances[int(landmark_row)]),
            source="dense_patch_context_candidate",
            observation_count=int(landmark_index.observation_counts[int(landmark_row)]),
            visibility_count=len(landmark_index.observation_image_ids[int(landmark_row)]),
            landmark_reprojection_error=float(landmark_index.reprojection_errors[int(landmark_row)]),
            similarity_margin=margin,
            local_consistency_support=int(context_bank.support_counts[int(landmark_row)]),
            local_consistency_score=float(context_score),
        )
        output.append((match, float(anchor_score), float(context_score), int(landmark_row)))
    return output


def rerank_anchor_candidates_with_dense_context(
    candidates: Sequence[QueryTo3DMatch],
    query_features_by_token: np.ndarray,
    context_bank: DensePatchContextBank,
    context_weight: float = 0.10,
    context_mode: str = "max_proto",
    top_m_per_token: int = 5,
    apply_reliability: bool = True,
    require_support: bool = True,
) -> list[QueryTo3DMatch]:
    """Rerank sparse-anchor candidates within each token using dense context."""

    if not candidates:
        return []
    track_to_row = context_bank.track_to_row()
    query_features = np.asarray(query_features_by_token, dtype=np.float32)
    by_token: dict[int, list[QueryTo3DMatch]] = {}
    for match in candidates:
        by_token.setdefault(int(match.token_index), []).append(match)
    output: list[QueryTo3DMatch] = []
    for token, token_matches in by_token.items():
        ordered = sorted(token_matches, key=lambda item: float(item.similarity), reverse=True)
        top_m = ordered if int(top_m_per_token) <= 0 else ordered[: int(top_m_per_token)]
        remainder = [] if int(top_m_per_token) <= 0 else ordered[int(top_m_per_token) :]
        valid_matches = [match for match in top_m if int(match.track_id) in track_to_row and 0 <= token < query_features.shape[0]]
        if valid_matches:
            rows = np.asarray([track_to_row[int(match.track_id)] for match in valid_matches], dtype=np.int64)
            query = np.stack([query_features[token] for _match in valid_matches], axis=0)
            context_scores = dense_context_scores(
                query,
                context_bank,
                rows,
                mode=context_mode,
                apply_reliability=apply_reliability,
                require_support=require_support,
            )
            rescored: list[QueryTo3DMatch] = []
            for match, dense_score, row in zip(valid_matches, context_scores.tolist(), rows.tolist()):
                final_score = float(match.similarity) + float(context_weight) * float(dense_score)
                rescored.append(
                    replace(
                        match,
                        similarity=final_score,
                        quality_weighted_similarity=final_score,
                        pnp_soft_score=final_score,
                        local_consistency_support=int(context_bank.support_counts[int(row)]),
                        local_consistency_score=float(dense_score),
                    )
                )
            valid_ids = {id(match) for match in valid_matches}
            invalid = [match for match in top_m if id(match) not in valid_ids]
            top_m_rescored = sorted(rescored + invalid, key=lambda item: float(item.similarity), reverse=True)
        else:
            top_m_rescored = top_m
        output.extend(top_m_rescored + remainder)
    output.sort(key=lambda item: float(item.quality_weighted_similarity or item.similarity), reverse=True)
    return output
