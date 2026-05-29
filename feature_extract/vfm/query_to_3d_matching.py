"""Query dense VFM token to 3D landmark VFM feature matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c, rotation_angle_deg
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank


@dataclass(frozen=True)
class LandmarkMapIndex:
    track_ids: np.ndarray
    xyz: np.ndarray
    features: np.ndarray
    mean_variances: np.ndarray
    observation_counts: np.ndarray
    observation_image_ids: tuple[tuple[str, ...], ...]
    reprojection_errors: np.ndarray | None = None
    feature_ambiguities: np.ndarray | None = None

    def __post_init__(self) -> None:
        track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.xyz, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float32)
        mean_variances = np.asarray(self.mean_variances, dtype=np.float32).reshape(-1)
        observation_counts = np.asarray(self.observation_counts, dtype=np.int64).reshape(-1)
        if self.reprojection_errors is None:
            reprojection_errors = np.zeros((track_ids.size,), dtype=np.float32)
        else:
            reprojection_errors = np.asarray(self.reprojection_errors, dtype=np.float32).reshape(-1)
        if self.feature_ambiguities is None:
            feature_ambiguities = np.zeros((track_ids.size,), dtype=np.float32)
        else:
            feature_ambiguities = np.asarray(self.feature_ambiguities, dtype=np.float32).reshape(-1)
        if xyz.shape != (track_ids.size, 3):
            raise ValueError("xyz must have shape (N, 3)")
        if features.ndim != 2 or features.shape[0] != track_ids.size:
            raise ValueError("features must have shape (N, C)")
        if mean_variances.shape[0] != track_ids.size:
            raise ValueError("mean_variances must have shape (N,)")
        if observation_counts.shape[0] != track_ids.size:
            raise ValueError("observation_counts must have shape (N,)")
        if reprojection_errors.shape[0] != track_ids.size:
            raise ValueError("reprojection_errors must have shape (N,)")
        if feature_ambiguities.shape[0] != track_ids.size:
            raise ValueError("feature_ambiguities must have shape (N,)")
        if len(self.observation_image_ids) != track_ids.size:
            raise ValueError("observation_image_ids must have length N")
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mean_variances", mean_variances)
        object.__setattr__(self, "observation_counts", observation_counts)
        object.__setattr__(self, "reprojection_errors", reprojection_errors)
        object.__setattr__(self, "feature_ambiguities", np.clip(feature_ambiguities, 0.0, 1.0))
        object.__setattr__(
            self,
            "observation_image_ids",
            tuple(tuple(str(item) for item in ids) for ids in self.observation_image_ids),
        )

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    def __len__(self) -> int:
        return int(self.track_ids.size)

    @classmethod
    def from_track_bank(
        cls,
        bank: SelectedTrackFeatureBank,
        xyz_by_track: Mapping[int, np.ndarray],
        reprojection_error_by_track: Mapping[int, float] | None = None,
    ) -> "LandmarkMapIndex":
        track_ids = []
        xyz = []
        features = []
        variances = []
        counts = []
        image_ids = []
        reprojection_errors = []
        reprojection_error_by_track = reprojection_error_by_track or {}
        for track_id in sorted(bank.tracks):
            if int(track_id) not in xyz_by_track:
                continue
            track = bank.tracks[int(track_id)]
            track_ids.append(int(track_id))
            xyz.append(np.asarray(xyz_by_track[int(track_id)], dtype=np.float64).reshape(3))
            features.append(np.asarray(track.mean_feature, dtype=np.float32).reshape(-1))
            variances.append(float(np.mean(track.variance)))
            counts.append(int(track.observation_count))
            image_ids.append(tuple(str(item) for item in track.observation_image_ids))
            reprojection_errors.append(float(reprojection_error_by_track.get(int(track_id), 0.0)))
        feature_dim = int(bank.feature_dim)
        if not track_ids:
            return cls(
                track_ids=np.zeros((0,), dtype=np.int64),
                xyz=np.zeros((0, 3), dtype=np.float64),
                features=np.zeros((0, feature_dim), dtype=np.float32),
                mean_variances=np.zeros((0,), dtype=np.float32),
                observation_counts=np.zeros((0,), dtype=np.int64),
                observation_image_ids=(),
                reprojection_errors=np.zeros((0,), dtype=np.float32),
                feature_ambiguities=np.zeros((0,), dtype=np.float32),
            )
        return cls(
            track_ids=np.asarray(track_ids, dtype=np.int64),
            xyz=np.stack(xyz, axis=0),
            features=np.stack(features, axis=0),
            mean_variances=np.asarray(variances, dtype=np.float32),
            observation_counts=np.asarray(counts, dtype=np.int64),
            observation_image_ids=tuple(image_ids),
            reprojection_errors=np.asarray(reprojection_errors, dtype=np.float32),
            feature_ambiguities=np.zeros((len(track_ids),), dtype=np.float32),
        )

    def subset(self, indices: Sequence[int] | np.ndarray) -> "LandmarkMapIndex":
        idx = np.asarray(indices)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64).reshape(-1)
        return LandmarkMapIndex(
            track_ids=self.track_ids[idx],
            xyz=self.xyz[idx],
            features=self.features[idx],
            mean_variances=self.mean_variances[idx],
            observation_counts=self.observation_counts[idx],
            observation_image_ids=tuple(self.observation_image_ids[int(i)] for i in idx),
            reprojection_errors=self.reprojection_errors[idx],
            feature_ambiguities=self.feature_ambiguities[idx],
        )


@dataclass(frozen=True)
class LandmarkQualityConfig:
    enabled: bool = False
    track_weight: float = 1.0
    variance_weight: float = 1.0
    reprojection_weight: float = 1.0
    idf_weight: float = 0.0
    ambiguity_weight: float = 0.5
    ambiguity_reference_size: int = 4096
    min_score: float | None = None
    min_track_length: int | None = None

    def __post_init__(self) -> None:
        weights = (
            self.track_weight,
            self.variance_weight,
            self.reprojection_weight,
            self.idf_weight,
            self.ambiguity_weight,
        )
        if any(float(weight) < 0.0 for weight in weights):
            raise ValueError("landmark quality weights must be non-negative")
        if self.min_score is not None and not 0.0 <= float(self.min_score) <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if self.min_track_length is not None and self.min_track_length <= 0:
            raise ValueError("min_track_length must be positive")
        if self.ambiguity_reference_size <= 1:
            raise ValueError("ambiguity_reference_size must be greater than 1")


@dataclass(frozen=True)
class QueryTo3DMatchingConfig:
    top_k: int = 2
    ratio_threshold: float | None = 0.9
    min_similarity_margin: float | None = None
    min_similarity: float = 0.0
    mutual: bool = False
    max_landmark_variance: float | None = None
    max_landmark_reprojection_error: float | None = None
    max_landmark_ambiguity: float | None = None
    min_distance_to_boundary_px: float | None = None
    min_quality_weighted_similarity: float | None = None
    min_observation_count: int = 1
    query_token_step: int = 1
    max_matches: int | None = None
    block_size: int = 512
    landmark_quality: LandmarkQualityConfig = LandmarkQualityConfig()

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.ratio_threshold is not None and not 0.0 < float(self.ratio_threshold) <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")
        if self.min_similarity_margin is not None and float(self.min_similarity_margin) < 0.0:
            raise ValueError("min_similarity_margin must be non-negative")
        if self.max_landmark_reprojection_error is not None and float(self.max_landmark_reprojection_error) < 0.0:
            raise ValueError("max_landmark_reprojection_error must be non-negative")
        if self.max_landmark_ambiguity is not None and not 0.0 <= float(self.max_landmark_ambiguity) <= 1.0:
            raise ValueError("max_landmark_ambiguity must be in [0, 1]")
        if self.min_distance_to_boundary_px is not None and float(self.min_distance_to_boundary_px) < 0.0:
            raise ValueError("min_distance_to_boundary_px must be non-negative")
        if self.min_quality_weighted_similarity is not None and float(self.min_quality_weighted_similarity) < -1.0:
            raise ValueError("min_quality_weighted_similarity must be >= -1")
        if self.query_token_step <= 0:
            raise ValueError("query_token_step must be positive")
        if self.min_observation_count <= 0:
            raise ValueError("min_observation_count must be positive")
        if self.max_matches is not None and self.max_matches <= 0:
            raise ValueError("max_matches must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


@dataclass(frozen=True)
class QueryTo3DMatch:
    token_index: int
    xy: np.ndarray
    track_id: int
    xyz: np.ndarray
    similarity: float
    ratio: float
    landmark_variance: float
    source: str = "sparse_landmark"
    observation_count: int | None = None
    visibility_count: int | None = None
    landmark_reprojection_error: float | None = None
    landmark_quality: float | None = None
    landmark_ambiguity: float | None = None
    quality_weighted_similarity: float | None = None
    pairwise_inlier_logit: float | None = None
    pairwise_inlier_logprob: float | None = None
    pairwise_weighted_similarity: float | None = None
    similarity_margin: float | None = None
    distance_to_boundary_px: float | None = None
    render_alpha: float | None = None
    render_depth: float | None = None
    alpha_entropy: float | None = None
    top1_alpha_contribution: float | None = None
    depth_variance_along_ray: float | None = None


@dataclass(frozen=True)
class PnPResult:
    success: bool
    pose_w2c: np.ndarray | None
    inlier_mask: np.ndarray
    match_count: int
    inlier_count: int

    @property
    def inlier_ratio(self) -> float:
        if self.match_count <= 0:
            return 0.0
        return float(self.inlier_count) / float(self.match_count)


@dataclass(frozen=True)
class PoseError:
    translation_m: float
    rotation_deg: float


def normalize_rows(matrix: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms.reshape(-1) > eps
    normalized = values / np.maximum(norms, eps)
    return normalized.astype(np.float32, copy=False), valid


def token_grid_xy(
    token_width: int,
    token_height: int,
    image_width: int,
    image_height: int,
    step: int = 1,
) -> np.ndarray:
    if token_width <= 0 or token_height <= 0:
        raise ValueError("token grid dimensions must be positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    xs = np.linspace(0.0, float(image_width - 1), token_width, dtype=np.float64)
    ys = np.linspace(0.0, float(image_height - 1), token_height, dtype=np.float64)
    coords = []
    for y_idx in range(0, token_height, step):
        for x_idx in range(0, token_width, step):
            coords.append((xs[x_idx], ys[y_idx]))
    return np.asarray(coords, dtype=np.float64)


def _flatten_query_features(
    feature_map: np.ndarray,
    image_width: int,
    image_height: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_map = np.asarray(feature_map, dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("query feature map must have shape (C, H, W)")
    channels, token_height, token_width = feature_map.shape
    selected = []
    token_indices = []
    for y_idx in range(0, token_height, step):
        for x_idx in range(0, token_width, step):
            selected.append(feature_map[:, y_idx, x_idx])
            token_indices.append(y_idx * token_width + x_idx)
    if not selected:
        return (
            np.zeros((0, channels), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    features = np.stack(selected, axis=0).astype(np.float32, copy=False)
    xy = token_grid_xy(token_width, token_height, image_width, image_height, step=step)
    return features, xy, np.asarray(token_indices, dtype=np.int64)


def _valid_landmark_subset(index: LandmarkMapIndex, config: QueryTo3DMatchingConfig) -> LandmarkMapIndex:
    mask = index.observation_counts >= int(config.min_observation_count)
    if config.max_landmark_variance is not None:
        mask = mask & (index.mean_variances <= float(config.max_landmark_variance))
    if config.max_landmark_reprojection_error is not None:
        mask = mask & (index.reprojection_errors <= float(config.max_landmark_reprojection_error))
    if config.landmark_quality.enabled and config.landmark_quality.min_track_length is not None:
        mask = mask & (index.observation_counts >= int(config.landmark_quality.min_track_length))
    return index.subset(mask)


def _normalize_penalty(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    valid = np.maximum(values[finite], 0.0)
    scale = float(np.percentile(valid, 95.0))
    if scale <= 1e-12:
        scale = float(np.max(valid))
    if scale <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    normalized = np.clip(np.maximum(values, 0.0) / scale, 0.0, 1.0)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32, copy=False)


def _landmark_nearest_neighbor_ambiguity(
    landmark_features: np.ndarray,
    block_size: int,
    reference_size: int,
) -> np.ndarray:
    count = int(landmark_features.shape[0])
    if count <= 1:
        return np.zeros((count,), dtype=np.float32)
    if count > int(reference_size):
        reference_indices = np.linspace(0, count - 1, int(reference_size), dtype=np.int64)
    else:
        reference_indices = np.arange(count, dtype=np.int64)
    reference_features = landmark_features[reference_indices]
    reference_position_by_index = {int(index): pos for pos, index in enumerate(reference_indices.tolist())}
    ambiguity = np.full((count,), -1.0, dtype=np.float32)
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        scores = landmark_features[start:end] @ reference_features.T
        for local_row, global_row in enumerate(range(start, end)):
            ref_pos = reference_position_by_index.get(int(global_row))
            if ref_pos is not None:
                scores[local_row, ref_pos] = -np.inf
        ambiguity[start:end] = np.max(scores, axis=1).astype(np.float32)
    return np.clip((ambiguity + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)


def landmark_quality_scores(
    index: LandmarkMapIndex,
    landmark_features: np.ndarray,
    config: LandmarkQualityConfig,
    block_size: int = 512,
    force_ambiguity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute interpretable 0-1 mapability scores for sparse VFM landmarks."""

    count = len(index)
    if count == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    needs_ambiguity = bool(force_ambiguity or config.idf_weight > 0.0 or config.ambiguity_weight > 0.0)
    if needs_ambiguity:
        if np.any(index.feature_ambiguities):
            ambiguity = index.feature_ambiguities.astype(np.float32, copy=True)
        else:
            ambiguity = _landmark_nearest_neighbor_ambiguity(
                landmark_features,
                block_size=block_size,
                reference_size=config.ambiguity_reference_size,
            )
    else:
        ambiguity = np.zeros((count,), dtype=np.float32)
    if not config.enabled:
        return np.ones((count,), dtype=np.float32), ambiguity

    weights = {
        "track": float(config.track_weight),
        "variance": float(config.variance_weight),
        "reprojection": float(config.reprojection_weight),
        "idf": float(config.idf_weight),
        "ambiguity": float(config.ambiguity_weight),
    }
    denominator = sum(weights.values())
    if denominator <= 1e-12:
        return np.ones((count,), dtype=np.float32), np.zeros((count,), dtype=np.float32)

    track = np.asarray(index.observation_counts, dtype=np.float32)
    max_track = max(float(np.max(np.log1p(np.maximum(track, 0.0)))), 1e-12)
    track_quality = np.log1p(np.maximum(track, 0.0)) / max_track
    variance_quality = 1.0 - _normalize_penalty(index.mean_variances)
    reprojection_quality = 1.0 - _normalize_penalty(index.reprojection_errors)
    idf_quality = 1.0 - ambiguity
    ambiguity_quality = 1.0 - ambiguity
    score = (
        weights["track"] * track_quality
        + weights["variance"] * variance_quality
        + weights["reprojection"] * reprojection_quality
        + weights["idf"] * idf_quality
        + weights["ambiguity"] * ambiguity_quality
    ) / denominator
    return np.clip(score, 0.0, 1.0).astype(np.float32, copy=False), ambiguity.astype(np.float32, copy=False)


def with_landmark_ambiguity_scores(
    index: LandmarkMapIndex,
    reference_size: int = 4096,
    block_size: int = 512,
) -> LandmarkMapIndex:
    features, valid = normalize_rows(index.features)
    if not np.all(valid):
        features = features.copy()
        features[~valid] = 0.0
    ambiguity = _landmark_nearest_neighbor_ambiguity(
        features,
        block_size=block_size,
        reference_size=reference_size,
    )
    return LandmarkMapIndex(
        track_ids=index.track_ids,
        xyz=index.xyz,
        features=index.features,
        mean_variances=index.mean_variances,
        observation_counts=index.observation_counts,
        observation_image_ids=index.observation_image_ids,
        reprojection_errors=index.reprojection_errors,
        feature_ambiguities=ambiguity,
    )


def _query_topk_and_landmark_best(
    query_features: np.ndarray,
    landmark_features: np.ndarray,
    top_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_count = int(query_features.shape[0])
    landmark_count = int(landmark_features.shape[0])
    effective_top_k = min(int(top_k), landmark_count)
    top_indices = np.full((query_count, effective_top_k), -1, dtype=np.int64)
    top_scores = np.full((query_count, effective_top_k), -np.inf, dtype=np.float32)
    landmark_best_query = np.full((landmark_count,), -1, dtype=np.int64)
    landmark_best_score = np.full((landmark_count,), -np.inf, dtype=np.float32)
    for start in range(0, query_count, block_size):
        end = min(start + block_size, query_count)
        scores = query_features[start:end] @ landmark_features.T
        block_best_query = np.argmax(scores, axis=0)
        block_best_score = scores[block_best_query, np.arange(landmark_count)]
        improve = block_best_score > landmark_best_score
        landmark_best_score[improve] = block_best_score[improve]
        landmark_best_query[improve] = start + block_best_query[improve]

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
    return top_indices, top_scores, landmark_best_query


def match_query_tokens_to_landmarks(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    config: QueryTo3DMatchingConfig | None = None,
    image_width: int = 1024,
    image_height: int = 576,
) -> list[QueryTo3DMatch]:
    config = config or QueryTo3DMatchingConfig()
    index = _valid_landmark_subset(landmark_index, config)
    if len(index) == 0:
        return []
    query_features, query_xy, token_indices = _flatten_query_features(
        query_feature_map,
        image_width=image_width,
        image_height=image_height,
        step=config.query_token_step,
    )
    query_features, valid_query = normalize_rows(query_features)
    landmark_features, valid_landmarks = normalize_rows(index.features)
    if not np.all(valid_landmarks):
        index = index.subset(valid_landmarks)
        landmark_features = landmark_features[valid_landmarks]
    valid_indices = np.flatnonzero(valid_query)
    if valid_indices.size == 0 or len(index) == 0:
        return []
    quality_scores, landmark_ambiguity = landmark_quality_scores(
        index,
        landmark_features,
        config.landmark_quality,
        block_size=config.block_size,
        force_ambiguity=config.max_landmark_ambiguity is not None,
    )
    if config.landmark_quality.enabled and config.landmark_quality.min_score is not None:
        keep_quality = quality_scores >= float(config.landmark_quality.min_score)
        if not np.any(keep_quality):
            return []
        index = index.subset(keep_quality)
        landmark_features = landmark_features[keep_quality]
        quality_scores = quality_scores[keep_quality]
        landmark_ambiguity = landmark_ambiguity[keep_quality]
    if config.max_landmark_ambiguity is not None:
        keep_ambiguity = landmark_ambiguity <= float(config.max_landmark_ambiguity)
        if not np.any(keep_ambiguity):
            return []
        index = index.subset(keep_ambiguity)
        landmark_features = landmark_features[keep_ambiguity]
        quality_scores = quality_scores[keep_ambiguity]
        landmark_ambiguity = landmark_ambiguity[keep_ambiguity]
    query_features = query_features[valid_indices]
    query_xy = query_xy[valid_indices]
    token_indices = token_indices[valid_indices]

    top_indices, top_scores, landmark_best_query = _query_topk_and_landmark_best(
        query_features,
        landmark_features,
        top_k=config.top_k,
        block_size=config.block_size,
    )
    matches: list[QueryTo3DMatch] = []
    for query_idx in range(query_features.shape[0]):
        candidate_indices = top_indices[query_idx]
        candidate_scores = top_scores[query_idx]
        valid_candidate_mask = candidate_indices >= 0
        if not np.any(valid_candidate_mask):
            continue
        candidate_indices = candidate_indices[valid_candidate_mask]
        candidate_scores = candidate_scores[valid_candidate_mask]
        if config.landmark_quality.enabled:
            weighted_scores = candidate_scores * quality_scores[candidate_indices]
            best_local_idx = int(np.argmax(weighted_scores))
        else:
            weighted_scores = candidate_scores
            best_local_idx = 0
        landmark_idx = int(candidate_indices[best_local_idx])
        if landmark_idx < 0:
            continue
        similarity = float(candidate_scores[best_local_idx])
        if similarity < float(config.min_similarity):
            continue
        quality_weighted_similarity = float(weighted_scores[best_local_idx])
        if (
            config.min_quality_weighted_similarity is not None
            and quality_weighted_similarity < float(config.min_quality_weighted_similarity)
        ):
            continue
        ratio = 0.0
        other_scores = np.delete(candidate_scores, best_local_idx)
        if other_scores.size >= 1:
            second_similarity = float(np.max(other_scores))
            best_distance = max(0.0, 1.0 - similarity)
            second_distance = max(1e-6, 1.0 - second_similarity)
            ratio = float(best_distance / second_distance)
            if config.ratio_threshold is not None and ratio > float(config.ratio_threshold):
                continue
        else:
            second_similarity = None
        similarity_margin = None if second_similarity is None else float(similarity - second_similarity)
        if config.min_similarity_margin is not None:
            if similarity_margin is None or similarity_margin < float(config.min_similarity_margin):
                continue
        xy = query_xy[query_idx].astype(np.float64, copy=True)
        boundary = min(
            float(xy[0]),
            float(xy[1]),
            float(image_width - 1) - float(xy[0]),
            float(image_height - 1) - float(xy[1]),
        )
        if config.min_distance_to_boundary_px is not None and boundary < float(config.min_distance_to_boundary_px):
            continue
        if config.mutual and int(landmark_best_query[landmark_idx]) != query_idx:
            continue
        matches.append(
            QueryTo3DMatch(
                token_index=int(token_indices[query_idx]),
                xy=xy,
                track_id=int(index.track_ids[landmark_idx]),
                xyz=index.xyz[landmark_idx].astype(np.float64, copy=True),
                similarity=similarity,
                ratio=ratio,
                landmark_variance=float(index.mean_variances[landmark_idx]),
                source="sparse_landmark",
                observation_count=int(index.observation_counts[landmark_idx]),
                visibility_count=len(index.observation_image_ids[landmark_idx]),
                landmark_reprojection_error=float(index.reprojection_errors[landmark_idx]),
                landmark_quality=float(quality_scores[landmark_idx]),
                landmark_ambiguity=float(landmark_ambiguity[landmark_idx]),
                quality_weighted_similarity=quality_weighted_similarity,
                similarity_margin=similarity_margin,
                distance_to_boundary_px=float(boundary),
            )
        )
    matches.sort(
        key=lambda item: (
            item.quality_weighted_similarity if item.quality_weighted_similarity is not None else item.similarity
        ),
        reverse=True,
    )
    if config.max_matches is not None:
        matches = matches[: int(config.max_matches)]
    return matches


def filter_landmarks_by_reference_images(
    landmark_index: LandmarkMapIndex,
    reference_images: set[str] | Sequence[str],
) -> LandmarkMapIndex:
    references = {str(item) for item in reference_images}
    if not references:
        return landmark_index.subset([])
    mask = np.asarray(
        [bool(references.intersection(set(image_ids))) for image_ids in landmark_index.observation_image_ids],
        dtype=bool,
    )
    return landmark_index.subset(mask)


def camera_matrix_and_distortion(camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    if camera.model_id == 0:
        f, cx, cy = camera.params[:3]
        fx = fy = f
        dist = np.zeros((4, 1), dtype=np.float64)
    elif camera.model_id == 1:
        fx, fy, cx, cy = camera.params[:4]
        dist = np.zeros((4, 1), dtype=np.float64)
    elif camera.model_id == 2:
        f, cx, cy, k = camera.params[:4]
        fx = fy = f
        dist = np.asarray([k, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4, 1)
    else:
        raise ValueError(f"unsupported camera model id for PnP: {camera.model_id}")
    matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return matrix, dist


def estimate_pose_pnp_ransac(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    reprojection_error_px: float = 8.0,
    confidence: float = 0.999,
    iterations: int = 1000,
) -> PnPResult:
    if len(matches) < 4:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - exercised only when OpenCV is absent.
        raise RuntimeError("OpenCV is required for PnP-RANSAC") from exc

    object_points = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)
    image_points = np.stack([match.xy for match in matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        iterationsCount=int(iterations),
        reprojectionError=float(reprojection_error_px),
        confidence=float(confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or rvec is None or tvec is None or inliers is None:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    rotation, _jacobian = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.astype(np.float64)
    pose[:3, 3] = tvec.reshape(3).astype(np.float64)
    mask = np.zeros((len(matches),), dtype=bool)
    mask[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    return PnPResult(
        success=True,
        pose_w2c=pose,
        inlier_mask=mask,
        match_count=len(matches),
        inlier_count=int(mask.sum()),
    )


def pnp_pose_error(pose_w2c: np.ndarray | None, gt_pose_w2c: np.ndarray) -> PoseError:
    if pose_w2c is None:
        return PoseError(translation_m=float("inf"), rotation_deg=float("inf"))
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    gt = np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    translation = float(np.linalg.norm(camera_center_from_pose_w2c(pose) - camera_center_from_pose_w2c(gt)))
    rotation = float(rotation_angle_deg(pose[:3, :3], gt[:3, :3]))
    return PoseError(translation_m=translation, rotation_deg=rotation)


def reprojection_precision(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    threshold_px: float = 16.0,
) -> tuple[float, float]:
    if not matches:
        return 0.0, 1.0
    points = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)
    xy = np.stack([match.xy for match in matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for reprojection precision") from exc
    rotation = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)[:3, :3]
    translation = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)[:3, 3]
    rvec, _jacobian = cv2.Rodrigues(rotation)
    projected, _jacobian = cv2.projectPoints(points, rvec, translation, camera_matrix, distortion)
    errors = np.linalg.norm(projected.reshape(-1, 2) - xy, axis=1)
    precision = float(np.mean(errors <= float(threshold_px)))
    return precision, float(1.0 - precision)


def match_reprojection_errors(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> np.ndarray:
    if not matches:
        return np.zeros((0,), dtype=np.float64)
    points = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)
    xy = np.stack([match.xy for match in matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for reprojection error stats") from exc
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(points, rvec, pose[:3, 3], camera_matrix, distortion)
    return np.linalg.norm(projected.reshape(-1, 2) - xy, axis=1).astype(np.float64)


def reprojection_error_stats(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    thresholds_px: Sequence[float] = (5.0, 10.0, 16.0, 32.0),
    pnp_inlier_mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Measure whether accepted 2D-3D correspondences are geometrically correct."""

    stats: dict[str, float | int | None] = {"match_count": int(len(matches))}
    for threshold in thresholds_px:
        stats[f"gt_precision_{float(threshold):g}px"] = 0.0
    stats.update(
        {
            "gt_reproj_mean_px": None,
            "gt_reproj_median_px": None,
            "gt_reproj_p90_px": None,
            "gt_reproj_p95_px": None,
            "pnp_inlier_count": 0,
            "pnp_inlier_gt_reproj_median_px": None,
        }
    )
    for threshold in thresholds_px:
        stats[f"pnp_inlier_gt_precision_{float(threshold):g}px"] = None
    if not matches:
        return stats
    errors = match_reprojection_errors(matches, pose_w2c, camera)
    for threshold in thresholds_px:
        stats[f"gt_precision_{float(threshold):g}px"] = float(np.mean(errors <= float(threshold)))
    stats["gt_reproj_mean_px"] = float(np.mean(errors))
    stats["gt_reproj_median_px"] = float(np.median(errors))
    stats["gt_reproj_p90_px"] = float(np.percentile(errors, 90.0))
    stats["gt_reproj_p95_px"] = float(np.percentile(errors, 95.0))
    if pnp_inlier_mask is not None:
        mask = np.asarray(pnp_inlier_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != len(matches):
            raise ValueError("pnp_inlier_mask must have one value per match")
        stats["pnp_inlier_count"] = int(np.sum(mask))
        if np.any(mask):
            inlier_errors = errors[mask]
            for threshold in thresholds_px:
                stats[f"pnp_inlier_gt_precision_{float(threshold):g}px"] = float(
                    np.mean(inlier_errors <= float(threshold))
                )
            stats["pnp_inlier_gt_reproj_median_px"] = float(np.median(inlier_errors))
    return stats


def pnp_reprojection_residual_stats(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray | None,
    camera: ColmapCamera,
    inlier_mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    stats: dict[str, float | int | None] = {
        "pnp_reproj_match_count": int(len(matches)),
        "pnp_reproj_mean_px": None,
        "pnp_reproj_median_px": None,
        "pnp_reproj_p90_px": None,
        "pnp_reproj_inlier_count": 0,
        "pnp_reproj_inlier_mean_px": None,
        "pnp_reproj_inlier_median_px": None,
        "pnp_reproj_inlier_p90_px": None,
    }
    if pose_w2c is None or not matches:
        return stats
    errors = match_reprojection_errors(matches, pose_w2c, camera)
    stats["pnp_reproj_mean_px"] = float(np.mean(errors))
    stats["pnp_reproj_median_px"] = float(np.median(errors))
    stats["pnp_reproj_p90_px"] = float(np.percentile(errors, 90.0))
    if inlier_mask is not None:
        mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != len(matches):
            raise ValueError("inlier_mask must have one value per match")
        stats["pnp_reproj_inlier_count"] = int(np.sum(mask))
        if np.any(mask):
            inlier_errors = errors[mask]
            stats["pnp_reproj_inlier_mean_px"] = float(np.mean(inlier_errors))
            stats["pnp_reproj_inlier_median_px"] = float(np.median(inlier_errors))
            stats["pnp_reproj_inlier_p90_px"] = float(np.percentile(inlier_errors, 90.0))
    return stats


def match_spatial_distribution_stats(
    matches: Sequence[QueryTo3DMatch],
    image_width: int,
    image_height: int,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    if mask is None:
        selected = np.ones((len(matches),), dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if selected.shape[0] != len(matches):
            raise ValueError("mask must have one value per match")
    stats: dict[str, float | int | None] = {
        "count": int(np.sum(selected)),
        "bbox_area_frac": None,
        "convex_hull_area_frac": None,
        "xy_std_px": None,
        "xy_pca_minor_major_ratio": None,
        "grid_4x4_occupancy_frac": None,
        "boundary_median_px": None,
        "xyz_planarity_ratio": None,
        "xyz_linearity_ratio": None,
        "depth_range_m": None,
    }
    if not matches or not np.any(selected):
        return stats
    xy = np.stack([match.xy for match in matches], axis=0).astype(np.float64)[selected]
    xyz = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)[selected]
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    stats["bbox_area_frac"] = float(max(x_max - x_min, 0.0) * max(y_max - y_min, 0.0) / (width * height))
    stats["xy_std_px"] = float(np.sqrt(np.sum(np.var(xy, axis=0)))) if xy.shape[0] >= 2 else 0.0
    boundary = np.minimum.reduce([xy[:, 0], xy[:, 1], width - 1.0 - xy[:, 0], height - 1.0 - xy[:, 1]])
    stats["boundary_median_px"] = float(np.median(boundary))
    cells_x = np.clip(np.floor(xy[:, 0] / width * 4.0).astype(np.int64), 0, 3)
    cells_y = np.clip(np.floor(xy[:, 1] / height * 4.0).astype(np.int64), 0, 3)
    stats["grid_4x4_occupancy_frac"] = float(len(set((int(x), int(y)) for x, y in zip(cells_x, cells_y))) / 16.0)
    if xy.shape[0] >= 3:
        try:
            import cv2
            hull = cv2.convexHull(xy.astype(np.float32))
            stats["convex_hull_area_frac"] = float(cv2.contourArea(hull) / (width * height))
        except Exception:
            stats["convex_hull_area_frac"] = None
    else:
        stats["convex_hull_area_frac"] = 0.0
    if xy.shape[0] >= 2:
        cov = np.cov((xy - np.mean(xy, axis=0)).T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))
        stats["xy_pca_minor_major_ratio"] = float(eigvals[0] / max(eigvals[-1], 1e-12))
    else:
        stats["xy_pca_minor_major_ratio"] = 0.0
    if xyz.shape[0] >= 3:
        xyz_cov = np.cov((xyz - np.mean(xyz, axis=0)).T)
        eigvals3 = np.sort(np.maximum(np.linalg.eigvalsh(xyz_cov), 0.0))
        stats["xyz_planarity_ratio"] = float(eigvals3[0] / max(float(np.sum(eigvals3)), 1e-12))
        stats["xyz_linearity_ratio"] = float(eigvals3[1] / max(eigvals3[2], 1e-12))
    else:
        stats["xyz_planarity_ratio"] = 0.0
        stats["xyz_linearity_ratio"] = 0.0
    stats["depth_range_m"] = float(np.max(xyz[:, 2]) - np.min(xyz[:, 2])) if xyz.shape[0] else None
    return stats
