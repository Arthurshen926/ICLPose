"""Query dense VFM token to 3D landmark VFM feature matching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c, rotation_angle_deg
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.landmark_feature_aggregation import MultiPrototypeTrackBank
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
    prototype_ids: np.ndarray | None = None

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
        if self.prototype_ids is None:
            prototype_ids = np.zeros((track_ids.size,), dtype=np.int64)
        else:
            prototype_ids = np.asarray(self.prototype_ids, dtype=np.int64).reshape(-1)
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
        if prototype_ids.shape[0] != track_ids.size:
            raise ValueError("prototype_ids must have shape (N,)")
        if np.any(prototype_ids < 0):
            raise ValueError("prototype_ids must be non-negative")
        if len(self.observation_image_ids) != track_ids.size:
            raise ValueError("observation_image_ids must have length N")
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mean_variances", mean_variances)
        object.__setattr__(self, "observation_counts", observation_counts)
        object.__setattr__(self, "reprojection_errors", reprojection_errors)
        object.__setattr__(self, "feature_ambiguities", np.clip(feature_ambiguities, 0.0, 1.0))
        object.__setattr__(self, "prototype_ids", prototype_ids)
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
                prototype_ids=np.zeros((0,), dtype=np.int64),
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
            prototype_ids=np.zeros((len(track_ids),), dtype=np.int64),
        )

    @classmethod
    def from_multi_prototype_bank(
        cls,
        bank: MultiPrototypeTrackBank,
        xyz_by_track: Mapping[int, np.ndarray],
        reprojection_error_by_track: Mapping[int, float] | None = None,
    ) -> "LandmarkMapIndex":
        reprojection_error_by_track = reprojection_error_by_track or {}
        prototypes = [item for item in bank.prototypes if int(item.track_id) in xyz_by_track]
        if not prototypes:
            return cls(
                track_ids=np.zeros((0,), dtype=np.int64),
                xyz=np.zeros((0, 3), dtype=np.float64),
                features=np.zeros((0, int(bank.feature_dim)), dtype=np.float32),
                mean_variances=np.zeros((0,), dtype=np.float32),
                observation_counts=np.zeros((0,), dtype=np.int64),
                observation_image_ids=(),
                reprojection_errors=np.zeros((0,), dtype=np.float32),
                feature_ambiguities=np.zeros((0,), dtype=np.float32),
                prototype_ids=np.zeros((0,), dtype=np.int64),
            )
        return cls(
            track_ids=np.asarray([int(item.track_id) for item in prototypes], dtype=np.int64),
            xyz=np.stack(
                [np.asarray(xyz_by_track[int(item.track_id)], dtype=np.float64).reshape(3) for item in prototypes],
                axis=0,
            ),
            features=np.stack([np.asarray(item.feature, dtype=np.float32) for item in prototypes], axis=0),
            mean_variances=np.asarray(
                [float(np.mean(np.asarray(item.variance, dtype=np.float32))) for item in prototypes],
                dtype=np.float32,
            ),
            observation_counts=np.asarray([int(item.observation_count) for item in prototypes], dtype=np.int64),
            observation_image_ids=tuple(tuple(item.observation_image_ids) for item in prototypes),
            reprojection_errors=np.asarray(
                [float(reprojection_error_by_track.get(int(item.track_id), 0.0)) for item in prototypes],
                dtype=np.float32,
            ),
            feature_ambiguities=np.zeros((len(prototypes),), dtype=np.float32),
            prototype_ids=np.asarray([int(item.prototype_id) for item in prototypes], dtype=np.int64),
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
            prototype_ids=self.prototype_ids[idx],
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
class LandmarkAmbiguityPruningConfig:
    enabled: bool = False
    drop_fraction: float | None = None
    max_score: float | None = None
    close_similarity_threshold: float = 0.9
    reference_size: int = 4096
    block_size: int = 512

    def __post_init__(self) -> None:
        if self.drop_fraction is not None and not 0.0 < float(self.drop_fraction) < 1.0:
            raise ValueError("drop_fraction must be in (0, 1)")
        if self.max_score is not None and not 0.0 <= float(self.max_score) <= 1.0:
            raise ValueError("max_score must be in [0, 1]")
        if not -1.0 <= float(self.close_similarity_threshold) <= 1.0:
            raise ValueError("close_similarity_threshold must be in [-1, 1]")
        if int(self.reference_size) <= 1:
            raise ValueError("reference_size must be greater than 1")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")


@dataclass(frozen=True)
class MapReliabilityConfig:
    enabled: bool = False
    track_weight: float = 1.0
    variance_weight: float = 1.0
    reprojection_weight: float = 1.0
    idf_weight: float = 0.0
    ambiguity_weight: float = 0.5
    view_angle_weight: float = 0.0
    ambiguity_reference_size: int = 4096
    min_score: float | None = None
    filter_keep_fraction: float | None = None
    pnp_keep_fraction: float | None = None
    uncertainty_min_scale: float = 0.75
    uncertainty_max_scale: float = 2.0

    def __post_init__(self) -> None:
        weights = (
            self.track_weight,
            self.variance_weight,
            self.reprojection_weight,
            self.idf_weight,
            self.ambiguity_weight,
            self.view_angle_weight,
        )
        if any(float(weight) < 0.0 for weight in weights):
            raise ValueError("map reliability weights must be non-negative")
        if self.min_score is not None and not 0.0 <= float(self.min_score) <= 1.0:
            raise ValueError("map reliability min_score must be in [0, 1]")
        if self.filter_keep_fraction is not None and not 0.0 < float(self.filter_keep_fraction) <= 1.0:
            raise ValueError("map reliability filter_keep_fraction must be in (0, 1]")
        if self.pnp_keep_fraction is not None and not 0.0 < float(self.pnp_keep_fraction) <= 1.0:
            raise ValueError("map reliability pnp_keep_fraction must be in (0, 1]")
        if self.ambiguity_reference_size <= 1:
            raise ValueError("ambiguity_reference_size must be greater than 1")
        if float(self.uncertainty_min_scale) <= 0.0 or float(self.uncertainty_max_scale) <= 0.0:
            raise ValueError("uncertainty scales must be positive")
        if float(self.uncertainty_min_scale) > float(self.uncertainty_max_scale):
            raise ValueError("uncertainty_min_scale must be <= uncertainty_max_scale")


@dataclass(frozen=True)
class LocalGeometricConsistencyConfig:
    enabled: bool = False
    image_radius_px: float = 96.0
    xyz_radius_m: float = 2.0
    min_support: int | None = 1
    min_score: float | None = None
    keep_fraction: float | None = None
    max_input_matches: int | None = None

    def __post_init__(self) -> None:
        if float(self.image_radius_px) <= 0.0:
            raise ValueError("image_radius_px must be positive")
        if float(self.xyz_radius_m) <= 0.0:
            raise ValueError("xyz_radius_m must be positive")
        if self.min_support is not None and int(self.min_support) < 0:
            raise ValueError("min_support must be non-negative")
        if self.min_score is not None and not 0.0 <= float(self.min_score) <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if self.keep_fraction is not None and not 0.0 < float(self.keep_fraction) <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.max_input_matches is not None and int(self.max_input_matches) <= 0:
            raise ValueError("max_input_matches must be positive")


@dataclass(frozen=True)
class SpatialDiversityPnPConfig:
    enabled: bool = False
    grid_rows: int = 4
    grid_cols: int = 4
    max_per_cell: int = 2
    max_matches: int | None = None
    score_mode: str = "margin"
    min_world_z_range_m: float | None = None
    min_planarity_ratio: float | None = None

    def __post_init__(self) -> None:
        if int(self.grid_rows) <= 0 or int(self.grid_cols) <= 0:
            raise ValueError("grid rows/cols must be positive")
        if int(self.max_per_cell) <= 0:
            raise ValueError("max_per_cell must be positive")
        if self.max_matches is not None and int(self.max_matches) <= 0:
            raise ValueError("max_matches must be positive")
        if self.score_mode not in {"margin", "pairwise", "reliability", "similarity"}:
            raise ValueError("score_mode must be one of: margin, pairwise, reliability, similarity")
        if self.min_world_z_range_m is not None and float(self.min_world_z_range_m) < 0.0:
            raise ValueError("min_world_z_range_m must be non-negative")
        if self.min_planarity_ratio is not None and float(self.min_planarity_ratio) < 0.0:
            raise ValueError("min_planarity_ratio must be non-negative")


@dataclass(frozen=True)
class PoseRiskConfig:
    min_inlier_count: float = 64.0
    min_inlier_ratio: float = 0.25
    min_inlier_patch_at_1: float = 0.5
    max_reprojection_median_px: float = 16.0
    min_grid_coverage: float = 0.25
    min_depth_range_m: float = 1.0
    min_planarity_ratio: float = 0.02
    min_map_reliability: float = 0.5


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
    map_reliability: float | None = None
    pnp_uncertainty_scale: float | None = None
    local_consistency_support: int | None = None
    local_consistency_score: float | None = None
    pnp_soft_score: float | None = None
    patch_offset_confidence: float | None = None
    patch_offset_sigma: float | None = None
    patch_offset_applied: bool | None = None
    patch_offset_norm_px: float | None = None
    anchor_xyz_change_m: float | None = None
    render_depth_change_m: float | None = None
    surface_switch_flag: bool | None = None
    render_depth_gradient: float | None = None
    patch_offset_consistency_before_px: float | None = None
    patch_offset_consistency_after_px: float | None = None
    token_match_rank: int | None = None
    measurement_sigma_px: float | None = None
    query_heatmap_score: float | None = None
    geometry_probability: float | None = None
    measurement_refined_xy: np.ndarray | None = None
    render_xy: np.ndarray | None = None
    base_render_index: int | None = None
    candidate_render_index: int | None = None
    candidate_id: int | None = None
    coarse_rank: int | None = None
    coarse_score: float | None = None
    coarse_score_gap: float | None = None
    prototype_id: int | None = None
    mutual_rank: int | None = None
    cell_delta_x: int | None = None
    cell_delta_y: int | None = None


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
    coordinate_mode: str = "edge",
) -> np.ndarray:
    if token_width <= 0 or token_height <= 0:
        raise ValueError("token grid dimensions must be positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if coordinate_mode == "edge":
        xs = np.linspace(0.0, float(image_width - 1), token_width, dtype=np.float64)
        ys = np.linspace(0.0, float(image_height - 1), token_height, dtype=np.float64)
    elif coordinate_mode == "center":
        stride_x = float(image_width) / float(token_width)
        stride_y = float(image_height) / float(token_height)
        xs = (np.arange(token_width, dtype=np.float64) + 0.5) * stride_x - 0.5
        ys = (np.arange(token_height, dtype=np.float64) + 0.5) * stride_y - 0.5
    else:
        raise ValueError("coordinate_mode must be one of: edge, center")
    coords = []
    for y_idx in range(0, token_height, step):
        for x_idx in range(0, token_width, step):
            coords.append((xs[x_idx], ys[y_idx]))
    return np.asarray(coords, dtype=np.float64)


def deduplicate_pnp_matches(matches: Sequence[QueryTo3DMatch]) -> tuple[list[QueryTo3DMatch], np.ndarray]:
    """Keep one 2D observation per 3D track before PnP.

    Callers choose the conflict policy through input order. Keeping the first
    occurrence avoids repeated 3D points with conflicting 2D observations.
    """

    unique: list[QueryTo3DMatch] = []
    original_indices: list[int] = []
    seen_tracks: set[int] = set()
    for idx, match in enumerate(matches):
        track_id = int(match.track_id)
        if track_id in seen_tracks:
            continue
        seen_tracks.add(track_id)
        unique.append(match)
        original_indices.append(int(idx))
    return unique, np.asarray(original_indices, dtype=np.int64)


def canonicalize_pnp_solver_order(
    matches: Sequence[QueryTo3DMatch], original_indices: np.ndarray
) -> tuple[list[QueryTo3DMatch], np.ndarray]:
    """Canonicalize solver input after confidence-based conflict resolution."""

    indices = np.asarray(original_indices, dtype=np.int64).reshape(-1)
    if len(matches) != len(indices):
        raise ValueError("PnP matches and original indices must have equal length")
    order = sorted(
        range(len(matches)),
        key=lambda index: (
            int(matches[index].token_index),
            int(matches[index].track_id),
            -1
            if matches[index].prototype_id is None
            else int(matches[index].prototype_id),
        ),
    )
    return [matches[index] for index in order], indices[np.asarray(order, dtype=np.int64)]


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


def landmark_submap_ambiguity_scores(
    index: LandmarkMapIndex,
    close_similarity_threshold: float = 0.9,
    reference_size: int = 4096,
    block_size: int = 512,
) -> np.ndarray:
    """Score descriptor ambiguity inside the current landmark pool."""

    count = len(index)
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    features, valid = normalize_rows(index.features)
    if count <= 1:
        return np.zeros((count,), dtype=np.float32)
    if count > int(reference_size):
        reference_indices = np.linspace(0, count - 1, int(reference_size), dtype=np.int64)
    else:
        reference_indices = np.arange(count, dtype=np.int64)
    reference = features[reference_indices]
    reference_position_by_index = {int(index_value): pos for pos, index_value in enumerate(reference_indices.tolist())}
    top1 = np.zeros((count,), dtype=np.float32)
    top2 = np.zeros((count,), dtype=np.float32)
    close_counts = np.zeros((count,), dtype=np.float32)
    threshold = float(close_similarity_threshold)
    for start in range(0, count, int(block_size)):
        end = min(start + int(block_size), count)
        scores = features[start:end] @ reference.T
        for local_row, global_row in enumerate(range(start, end)):
            if not bool(valid[global_row]):
                scores[local_row, :] = -np.inf
                continue
            ref_pos = reference_position_by_index.get(int(global_row))
            if ref_pos is not None:
                scores[local_row, ref_pos] = -np.inf
        finite_scores = np.where(np.isfinite(scores), scores, -1.0)
        close_counts[start:end] = np.sum(finite_scores >= threshold, axis=1).astype(np.float32)
        if finite_scores.shape[1] == 1:
            top1[start:end] = finite_scores[:, 0]
            top2[start:end] = -1.0
        else:
            part = np.partition(finite_scores, kth=max(finite_scores.shape[1] - 2, 0), axis=1)
            top2[start:end] = part[:, -2]
            top1[start:end] = part[:, -1]
    top1_norm = np.clip((top1 + 1.0) * 0.5, 0.0, 1.0)
    margin = np.clip(top1 - top2, 0.0, 2.0)
    margin_ambiguity = 1.0 - np.clip(margin / 2.0, 0.0, 1.0)
    close_norm = close_counts / max(float(np.max(close_counts)), 1.0)
    ambiguity = 0.6 * top1_norm + 0.25 * margin_ambiguity + 0.15 * close_norm
    ambiguity[~valid] = 1.0
    return np.clip(ambiguity, 0.0, 1.0).astype(np.float32, copy=False)


def prune_ambiguous_landmarks(
    index: LandmarkMapIndex,
    config: LandmarkAmbiguityPruningConfig,
) -> LandmarkMapIndex:
    if not config.enabled or len(index) == 0:
        return index
    scores = landmark_submap_ambiguity_scores(
        index,
        close_similarity_threshold=float(config.close_similarity_threshold),
        reference_size=int(config.reference_size),
        block_size=int(config.block_size),
    )
    keep = np.ones((len(index),), dtype=bool)
    if config.max_score is not None:
        keep &= scores <= float(config.max_score)
    if config.drop_fraction is not None:
        drop_count = int(np.floor(len(index) * float(config.drop_fraction)))
        if drop_count > 0:
            order = np.argsort(-scores, kind="mergesort")
            keep[order[:drop_count]] = False
    return index.subset(keep)


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


def map_reliability_scores(
    index: LandmarkMapIndex,
    landmark_features: np.ndarray,
    config: MapReliabilityConfig,
    block_size: int = 512,
) -> np.ndarray:
    """Compute a map-side reliability prior without changing descriptor similarity."""

    if len(index) == 0:
        return np.zeros((0,), dtype=np.float32)
    if not config.enabled:
        return np.ones((len(index),), dtype=np.float32)
    quality_config = LandmarkQualityConfig(
        enabled=True,
        track_weight=float(config.track_weight),
        variance_weight=float(config.variance_weight),
        reprojection_weight=float(config.reprojection_weight),
        idf_weight=float(config.idf_weight),
        ambiguity_weight=float(config.ambiguity_weight),
        ambiguity_reference_size=int(config.ambiguity_reference_size),
    )
    scores, _ambiguity = landmark_quality_scores(
        index,
        landmark_features,
        quality_config,
        block_size=block_size,
        force_ambiguity=bool(config.idf_weight > 0.0 or config.ambiguity_weight > 0.0),
    )
    return scores.astype(np.float32, copy=False)


def map_reliability_uncertainty_scale(reliability: float, config: MapReliabilityConfig) -> float:
    clipped = float(np.clip(float(reliability), 0.0, 1.0))
    min_scale = float(config.uncertainty_min_scale)
    max_scale = float(config.uncertainty_max_scale)
    return float(max_scale - clipped * (max_scale - min_scale))


def select_pnp_matches_by_map_reliability(
    matches: Sequence[QueryTo3DMatch],
    keep_fraction: float | None = None,
    min_score: float | None = None,
) -> list[QueryTo3DMatch]:
    """Select high-reliability PnP inputs while preserving match order."""

    values = list(matches)
    if not values:
        return values
    if keep_fraction is None and min_score is None:
        return values
    scored = [match for match in values if match.map_reliability is not None]
    if not scored:
        return values
    keep_ids: set[int] = set()
    if keep_fraction is not None:
        keep_count = max(1, int(np.ceil(len(scored) * float(keep_fraction))))
        order = np.argsort([-float(match.map_reliability) for match in scored], kind="mergesort")
        keep_ids.update(id(scored[int(idx)]) for idx in order[:keep_count])
    else:
        keep_ids.update(id(match) for match in scored)
    selected = []
    for match in values:
        if match.map_reliability is None:
            continue
        if id(match) not in keep_ids:
            continue
        if min_score is not None and float(match.map_reliability) < float(min_score):
            continue
        selected.append(match)
    return selected


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = np.exp(-float(value))
        return float(1.0 / (1.0 + z))
    z = np.exp(float(value))
    return float(z / (1.0 + z))


def _soft_pnp_score(match: QueryTo3DMatch, mode: str) -> float:
    similarity = float(np.clip((float(match.similarity) + 1.0) * 0.5, 0.0, 1.0))
    margin = 0.0 if match.similarity_margin is None else float(np.clip(match.similarity_margin, 0.0, 1.0))
    reliability = 1.0 if match.map_reliability is None else float(np.clip(match.map_reliability, 0.0, 1.0))
    quality = 1.0 if match.landmark_quality is None else float(np.clip(match.landmark_quality, 0.0, 1.0))
    local = 1.0 if match.local_consistency_score is None else float(np.clip(match.local_consistency_score, 0.0, 1.0))
    if match.pairwise_inlier_logit is not None:
        confidence = _sigmoid(float(match.pairwise_inlier_logit))
    elif match.pairwise_inlier_logprob is not None:
        confidence = float(np.clip(np.exp(float(match.pairwise_inlier_logprob)), 0.0, 1.0))
    elif match.pnp_soft_score is not None:
        confidence = float(np.clip(float(match.pnp_soft_score), 0.0, 1.0))
    else:
        confidence = 1.0
    sigma = 1.0
    if match.measurement_sigma_px is not None and np.isfinite(float(match.measurement_sigma_px)):
        sigma = max(float(match.measurement_sigma_px), 1.0)
    if match.pnp_uncertainty_scale is not None and np.isfinite(float(match.pnp_uncertainty_scale)):
        sigma *= max(float(match.pnp_uncertainty_scale), 1e-6)
    uncertainty = float(1.0 / (1.0 + max(0.0, sigma - 1.0) / 8.0))

    if mode == "similarity":
        return similarity
    if mode == "margin":
        return 0.75 * margin + 0.25 * similarity
    if mode == "reliability":
        return 0.5 * reliability + 0.25 * margin + 0.25 * similarity
    if mode == "confidence":
        return 0.5 * confidence + 0.25 * margin + 0.25 * similarity
    if mode == "uncertainty":
        return float((0.60 * similarity + 0.25 * margin + 0.15 * confidence) * uncertainty)
    if mode == "composite":
        return float(
            similarity
            * (0.5 + 0.5 * reliability)
            * (0.5 + 0.5 * quality)
            * (0.5 + 0.5 * confidence)
            * (0.5 + 0.5 * local)
            * (0.5 + 0.5 * uncertainty)
            * (1.0 + margin)
        )
    raise ValueError(f"unsupported soft PnP ordering mode: {mode}")


def soft_order_pnp_matches(
    matches: Sequence[QueryTo3DMatch],
    mode: str = "none",
    max_matches: int | None = None,
) -> list[QueryTo3DMatch]:
    """Order PnP inputs by a soft confidence score without grid-style deletion."""

    values = list(matches)
    if mode == "none" or not values:
        return values if max_matches is None else values[: int(max_matches)]
    if max_matches is not None and int(max_matches) <= 0:
        raise ValueError("max_matches must be positive when provided")
    annotated = [replace(match, pnp_soft_score=_soft_pnp_score(match, mode)) for match in values]
    ordered = sorted(
        annotated,
        key=lambda match: (
            float(match.pnp_soft_score) if match.pnp_soft_score is not None else float("-inf"),
            float(match.similarity),
        ),
        reverse=True,
    )
    if max_matches is not None:
        ordered = ordered[: int(max_matches)]
    return ordered


def local_geometric_consistency_scores(
    matches: Sequence[QueryTo3DMatch],
    config: LocalGeometricConsistencyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(matches)
    if count == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    xy = np.stack([match.xy for match in matches], axis=0).astype(np.float64)
    xyz = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)
    token_indices = np.asarray([int(match.token_index) for match in matches], dtype=np.int64)
    supports = np.zeros((count,), dtype=np.int64)
    scores = np.zeros((count,), dtype=np.float32)
    image_radius = float(config.image_radius_px)
    xyz_radius = float(config.xyz_radius_m)
    for idx in range(count):
        image_distance = np.linalg.norm(xy - xy[idx], axis=1)
        xyz_distance = np.linalg.norm(xyz - xyz[idx], axis=1)
        neighbor = image_distance <= image_radius
        neighbor &= token_indices != token_indices[idx]
        support = neighbor & (xyz_distance <= xyz_radius)
        supports[idx] = int(np.sum(support))
        denominator = max(int(np.sum(neighbor)), 1)
        scores[idx] = float(supports[idx]) / float(denominator)
    return supports, scores


def filter_matches_by_local_geometric_consistency(
    matches: Sequence[QueryTo3DMatch],
    config: LocalGeometricConsistencyConfig,
) -> list[QueryTo3DMatch]:
    values = list(matches)
    if not config.enabled or not values:
        return values
    if config.max_input_matches is not None and len(values) > int(config.max_input_matches):
        values = values[: int(config.max_input_matches)]
    supports, scores = local_geometric_consistency_scores(values, config)
    annotated = [
        replace(
            match,
            local_consistency_support=int(supports[idx]),
            local_consistency_score=float(scores[idx]),
        )
        for idx, match in enumerate(values)
    ]
    keep = np.ones((len(annotated),), dtype=bool)
    if config.min_support is not None:
        keep &= supports >= int(config.min_support)
    if config.min_score is not None:
        keep &= scores >= float(config.min_score)
    if config.keep_fraction is not None:
        keep_count = max(1, int(np.ceil(len(annotated) * float(config.keep_fraction))))
        order = sorted(
            range(len(annotated)),
            key=lambda idx: (float(scores[idx]), int(supports[idx])),
            reverse=True,
        )
        top_keep = np.zeros((len(annotated),), dtype=bool)
        top_keep[order[:keep_count]] = True
        keep &= top_keep
    return [match for idx, match in enumerate(annotated) if bool(keep[idx])]


def _spatial_diversity_match_score(match: QueryTo3DMatch, mode: str) -> float:
    if mode == "margin":
        if match.similarity_margin is not None:
            return float(match.similarity_margin)
        return float(match.similarity)
    if mode == "pairwise":
        if match.pairwise_inlier_logit is not None:
            return float(match.pairwise_inlier_logit)
        return float(match.similarity)
    if mode == "reliability":
        if match.map_reliability is not None:
            return float(match.map_reliability)
        return float(match.similarity)
    return float(match.similarity)


def _selected_world_z_range_m(matches: Sequence[QueryTo3DMatch]) -> float:
    if not matches:
        return 0.0
    world_z = np.asarray([float(match.xyz[2]) for match in matches], dtype=np.float64)
    return float(np.max(world_z) - np.min(world_z)) if world_z.size else 0.0


def _selected_planarity_ratio(matches: Sequence[QueryTo3DMatch]) -> float:
    if len(matches) < 3:
        return 0.0
    xyz = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)
    cov = np.cov((xyz - np.mean(xyz, axis=0)).T)
    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
    return float(eigvals[0] / max(float(np.sum(eigvals)), 1e-12))


def _spatial_diversity_targets_met(
    matches: Sequence[QueryTo3DMatch],
    config: SpatialDiversityPnPConfig,
) -> bool:
    if config.min_world_z_range_m is not None and _selected_world_z_range_m(matches) < float(config.min_world_z_range_m):
        return False
    if config.min_planarity_ratio is not None and _selected_planarity_ratio(matches) < float(config.min_planarity_ratio):
        return False
    return True


def _spatial_diversity_target_progress(
    matches: Sequence[QueryTo3DMatch],
    config: SpatialDiversityPnPConfig,
) -> float:
    progress = 0.0
    if config.min_world_z_range_m is not None:
        progress += float(np.clip(_selected_world_z_range_m(matches) / max(float(config.min_world_z_range_m), 1e-12), 0.0, 1.0))
    if config.min_planarity_ratio is not None:
        progress += float(np.clip(_selected_planarity_ratio(matches) / max(float(config.min_planarity_ratio), 1e-12), 0.0, 1.0))
    return progress


def select_pnp_matches_by_spatial_diversity(
    matches: Sequence[QueryTo3DMatch],
    image_width: int,
    image_height: int,
    config: SpatialDiversityPnPConfig,
) -> list[QueryTo3DMatch]:
    values = list(matches)
    if not config.enabled or not values:
        return values
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    cells: dict[tuple[int, int], list[QueryTo3DMatch]] = {}
    for match in values:
        x_cell = int(np.clip(np.floor(float(match.xy[0]) / width * int(config.grid_cols)), 0, int(config.grid_cols) - 1))
        y_cell = int(np.clip(np.floor(float(match.xy[1]) / height * int(config.grid_rows)), 0, int(config.grid_rows) - 1))
        cells.setdefault((y_cell, x_cell), []).append(match)
    keep_ids: set[int] = set()
    for cell_matches in cells.values():
        ordered = sorted(
            cell_matches,
            key=lambda item: _spatial_diversity_match_score(item, config.score_mode),
            reverse=True,
        )
        keep_ids.update(id(match) for match in ordered[: int(config.max_per_cell)])
    selected = [match for match in values if id(match) in keep_ids]
    if config.max_matches is not None and len(selected) > int(config.max_matches):
        ordered_ids = {
            id(match)
            for match in sorted(
                selected,
                key=lambda item: _spatial_diversity_match_score(item, config.score_mode),
                reverse=True,
            )[: int(config.max_matches)]
        }
        selected = [match for match in selected if id(match) in ordered_ids]
    if (config.min_world_z_range_m is not None or config.min_planarity_ratio is not None) and not _spatial_diversity_targets_met(
        selected,
        config,
    ):
        selected_ids = {id(match) for match in selected}
        remaining = [match for match in values if id(match) not in selected_ids]
        max_count = int(config.max_matches) if config.max_matches is not None else len(values)
        while remaining:
            if len(selected) >= max_count:
                break
            current_progress = _spatial_diversity_target_progress(selected, config)
            best_index = max(
                range(len(remaining)),
                key=lambda idx: (
                    _spatial_diversity_target_progress([*selected, remaining[idx]], config) - current_progress,
                    _spatial_diversity_match_score(remaining[idx], config.score_mode),
                ),
            )
            match = remaining.pop(int(best_index))
            selected.append(match)
            selected_ids.add(id(match))
            if _spatial_diversity_targets_met(selected, config):
                break
    return selected


def _risk_component_low(value: float | None, target: float) -> float:
    if value is None or not np.isfinite(float(value)):
        return 1.0
    return float(np.clip(1.0 - float(value) / max(float(target), 1e-6), 0.0, 1.0))


def _risk_component_high(value: float | None, target: float) -> float:
    if value is None or not np.isfinite(float(value)):
        return 1.0
    return float(np.clip(float(value) / max(float(target), 1e-6), 0.0, 1.0))


def pose_risk_score(row: Mapping[str, object], config: PoseRiskConfig) -> float:
    patch = dict(row.get("patch_geometry") or {})
    reproj = dict(row.get("pnp_reprojection") or {})
    spatial = dict(row.get("pnp_inlier_spatial") or {})
    reliability = dict(dict(row.get("map_reliability") or {}).get("pnp_inliers") or {})
    components = [
        _risk_component_low(row.get("pnp_inlier_count"), config.min_inlier_count),
        _risk_component_low(row.get("pnp_inlier_ratio"), config.min_inlier_ratio),
        _risk_component_low(patch.get("pnp_inlier_patch_at_1"), config.min_inlier_patch_at_1),
        _risk_component_high(reproj.get("pnp_reproj_inlier_median_px"), config.max_reprojection_median_px),
        _risk_component_low(spatial.get("grid_4x4_occupancy_frac"), config.min_grid_coverage),
        _risk_component_low(spatial.get("depth_range_m"), config.min_depth_range_m),
        _risk_component_low(spatial.get("xyz_planarity_ratio"), config.min_planarity_ratio),
        _risk_component_low(reliability.get("mean"), config.min_map_reliability),
    ]
    return float(np.clip(np.mean(components), 0.0, 1.0))


def selective_localization_summary(
    rows: Sequence[Mapping[str, object]],
    coverages: Sequence[float] = (0.8, 0.9, 1.0),
    success_key: str = "success_25cm_10deg",
) -> dict[str, dict[str, float | int | None]]:
    usable = [dict(row) for row in rows if row.get("pose_risk") is not None]
    usable.sort(key=lambda row: float(row["pose_risk"]))
    output: dict[str, dict[str, float | int | None]] = {}
    if not usable:
        for coverage in coverages:
            output[f"coverage_{float(coverage):.3f}"] = {
                "query_count": 0,
                "success_rate": None,
                "median_translation_error_m": None,
                "median_rotation_error_deg": None,
                "risk_threshold": None,
            }
        return output
    for coverage in coverages:
        count = max(1, int(np.ceil(len(usable) * float(coverage))))
        selected = usable[: min(count, len(usable))]
        translations = [
            float(row["translation_error_m"])
            for row in selected
            if row.get("translation_error_m") is not None and np.isfinite(float(row["translation_error_m"]))
        ]
        rotations = [
            float(row["rotation_error_deg"])
            for row in selected
            if row.get("rotation_error_deg") is not None and np.isfinite(float(row["rotation_error_deg"]))
        ]
        output[f"coverage_{float(coverage):.3f}"] = {
            "query_count": int(len(selected)),
            "success_rate": float(np.mean([1.0 if row.get(success_key) else 0.0 for row in selected])),
            "median_translation_error_m": None if not translations else float(np.median(translations)),
            "median_rotation_error_deg": None if not rotations else float(np.median(rotations)),
            "risk_threshold": float(selected[-1]["pose_risk"]),
        }
    return output


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
    min_inliers: int = 0,
    refine_lm: bool = False,
    pnp_method: str = "EPNP",
    refine_method: str = "none",
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

    method_name = str(pnp_method).upper()
    method_attr = {
        "AP3P": "SOLVEPNP_AP3P",
        "EPNP": "SOLVEPNP_EPNP",
        "ITERATIVE": "SOLVEPNP_ITERATIVE",
        "P3P": "SOLVEPNP_P3P",
        "SQPNP": "SOLVEPNP_SQPNP",
    }.get(method_name)
    if method_attr is None or not hasattr(cv2, method_attr):
        raise ValueError(f"unsupported PnP method: {pnp_method}")
    refine_name = str(refine_method).upper()
    if bool(refine_lm) and refine_name == "NONE":
        refine_name = "LM"
    if refine_name not in {"NONE", "LM", "VVS"}:
        raise ValueError(f"unsupported PnP refine method: {refine_method}")

    unique_matches, original_indices = deduplicate_pnp_matches(matches)
    unique_matches, original_indices = canonicalize_pnp_solver_order(
        unique_matches, original_indices
    )
    if len(unique_matches) < 4:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    object_points = np.stack([match.xyz for match in unique_matches], axis=0).astype(np.float64)
    image_points = np.stack([match.xy for match in unique_matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        iterationsCount=int(iterations),
        reprojectionError=float(reprojection_error_px),
        confidence=float(confidence),
        flags=getattr(cv2, method_attr),
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
    if refine_name == "LM" and hasattr(cv2, "solvePnPRefineLM"):
        try:
            inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
            if inlier_indices.size >= 4:
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    camera_matrix,
                    distortion,
                    rvec,
                    tvec,
                )
                rotation, _jacobian = cv2.Rodrigues(rvec)
        except Exception:
            rotation, _jacobian = cv2.Rodrigues(rvec)
    elif refine_name == "VVS" and hasattr(cv2, "solvePnPRefineVVS"):
        try:
            inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
            if inlier_indices.size >= 4:
                rvec, tvec = cv2.solvePnPRefineVVS(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    camera_matrix,
                    distortion,
                    rvec,
                    tvec,
                )
                rotation, _jacobian = cv2.Rodrigues(rvec)
        except Exception:
            rotation, _jacobian = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.astype(np.float64)
    pose[:3, 3] = tvec.reshape(3).astype(np.float64)
    mask = np.zeros((len(matches),), dtype=bool)
    unique_inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
    mask[original_indices[unique_inlier_indices]] = True
    if int(min_inliers) > 0 and int(mask.sum()) < int(min_inliers):
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    return PnPResult(
        success=True,
        pose_w2c=pose,
        inlier_mask=mask,
        match_count=len(matches),
        inlier_count=int(mask.sum()),
    )


def estimate_pose_pnp_fixed(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    min_inliers: int = 4,
    pnp_method: str = "EPNP",
    refine_method: str = "none",
) -> PnPResult:
    """Estimate pose from a fixed correspondence set without RANSAC reselection."""

    if len(matches) < 4 or len(matches) < int(min_inliers):
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
        raise RuntimeError("OpenCV is required for fixed PnP") from exc

    method_name = str(pnp_method).upper()
    method_attr = {
        "AP3P": "SOLVEPNP_AP3P",
        "EPNP": "SOLVEPNP_EPNP",
        "ITERATIVE": "SOLVEPNP_ITERATIVE",
        "P3P": "SOLVEPNP_P3P",
        "SQPNP": "SOLVEPNP_SQPNP",
    }.get(method_name)
    if method_attr is None or not hasattr(cv2, method_attr):
        raise ValueError(f"unsupported PnP method: {pnp_method}")
    refine_name = str(refine_method).upper()
    if refine_name not in {"NONE", "LM", "VVS"}:
        raise ValueError(f"unsupported PnP refine method: {refine_method}")

    unique_matches, original_indices = deduplicate_pnp_matches(matches)
    unique_matches, original_indices = canonicalize_pnp_solver_order(
        unique_matches, original_indices
    )
    if len(unique_matches) < 4 or len(unique_matches) < int(min_inliers):
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    object_points = np.stack([match.xyz for match in unique_matches], axis=0).astype(np.float64)
    image_points = np.stack([match.xy for match in unique_matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=getattr(cv2, method_attr),
    )
    if not success or rvec is None or tvec is None:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    if refine_name == "LM" and hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, distortion, rvec, tvec)
        except Exception:
            pass
    elif refine_name == "VVS" and hasattr(cv2, "solvePnPRefineVVS"):
        try:
            rvec, tvec = cv2.solvePnPRefineVVS(object_points, image_points, camera_matrix, distortion, rvec, tvec)
        except Exception:
            pass
    rotation, _jacobian = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.astype(np.float64)
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    mask = np.zeros((len(matches),), dtype=bool)
    mask[original_indices] = True
    return PnPResult(
        success=True,
        pose_w2c=pose,
        inlier_mask=mask,
        match_count=len(matches),
        inlier_count=int(mask.sum()),
    )


def estimate_pose_pnp_fixed_robust(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    weights: np.ndarray | Sequence[float] | None = None,
    min_inliers: int = 4,
    initial_pose_w2c: np.ndarray | None = None,
    pnp_method: str = "EPNP",
    loss: str = "huber",
    f_scale_px: float = 4.0,
    max_nfev: int = 50,
) -> PnPResult:
    """Refine a fixed correspondence set with robust weighted reprojection LM.

    The correspondence set is fixed: all unique tracks are kept in the returned
    inlier mask. Weights only scale residuals inside the optimizer and never
    remove matches.
    """

    if len(matches) < 4 or len(matches) < int(min_inliers):
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    try:
        import cv2
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV and SciPy are required for robust fixed PnP refinement") from exc

    robust_loss = str(loss).lower()
    if robust_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise ValueError(f"unsupported robust PnP loss: {loss}")
    unique_matches, original_indices = deduplicate_pnp_matches(matches)
    if len(unique_matches) < 4 or len(unique_matches) < int(min_inliers):
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    object_points = np.stack([match.xyz for match in unique_matches], axis=0).astype(np.float64)
    image_points = np.stack([match.xy for match in unique_matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    if initial_pose_w2c is None:
        initial = estimate_pose_pnp_fixed(
            matches,
            camera,
            min_inliers=min_inliers,
            pnp_method=pnp_method,
            refine_method="none",
        )
        if not initial.success or initial.pose_w2c is None:
            return initial
        pose0 = np.asarray(initial.pose_w2c, dtype=np.float64).reshape(4, 4)
    else:
        pose0 = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    rvec0, _jacobian = cv2.Rodrigues(pose0[:3, :3])
    tvec0 = pose0[:3, 3].reshape(3, 1)
    params0 = np.concatenate([rvec0.reshape(3), tvec0.reshape(3)]).astype(np.float64)
    if weights is None:
        weight_values = np.ones((len(unique_matches),), dtype=np.float64)
    else:
        all_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if all_weights.shape[0] != len(matches):
            raise ValueError("weights must have one value per input match")
        weight_values = all_weights[np.asarray(original_indices, dtype=np.int64)]
        weight_values = np.where(np.isfinite(weight_values), weight_values, 0.0)
        weight_values = np.clip(weight_values, 0.0, None)
        if float(np.sum(weight_values)) <= 1e-12:
            weight_values = np.ones((len(unique_matches),), dtype=np.float64)
    weight_values = weight_values / max(float(np.mean(weight_values)), 1e-12)
    sqrt_weights = np.sqrt(weight_values).astype(np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = np.asarray(params[:3], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(params[3:6], dtype=np.float64).reshape(3, 1)
        projected, _jacobian = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        errors = projected.reshape(-1, 2) - image_points
        return (errors * sqrt_weights[:, None]).reshape(-1)

    try:
        result = least_squares(
            residuals,
            params0,
            loss=robust_loss,
            f_scale=max(float(f_scale_px), 1e-6),
            max_nfev=int(max_nfev),
            method="trf",
        )
    except Exception:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    if not bool(result.success) or not np.all(np.isfinite(result.x)):
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            match_count=len(matches),
            inlier_count=0,
        )
    rotation, _jacobian = cv2.Rodrigues(np.asarray(result.x[:3], dtype=np.float64).reshape(3, 1))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.astype(np.float64)
    pose[:3, 3] = np.asarray(result.x[3:6], dtype=np.float64).reshape(3)
    mask = np.zeros((len(matches),), dtype=bool)
    mask[original_indices] = True
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


def _pnp_match_confidence(match: QueryTo3DMatch) -> float:
    for value in (
        match.pnp_soft_score,
        match.pairwise_weighted_similarity,
        match.quality_weighted_similarity,
        match.pairwise_inlier_logprob,
        match.similarity,
        match.landmark_quality,
    ):
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            return score
    return 0.0


def select_unique_query_inlier_mask(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray | None,
    camera: ColmapCamera,
    inlier_mask: np.ndarray | Sequence[bool] | None,
    *,
    residual_tie_px: float = 1e-3,
) -> np.ndarray:
    """Keep at most one inlier candidate per query token under a pose.

    Render-side local expansion can create multiple possible 3D anchors for one
    query measurement. A valid PnP inlier set should not accept several of them
    at once, because they share the same 2D observation but represent different
    render-depth backprojections.
    """

    values = list(matches)
    if not values:
        return np.zeros((0,), dtype=bool)
    if pose_w2c is None:
        return np.zeros((len(values),), dtype=bool)
    if inlier_mask is None:
        base_mask = np.ones((len(values),), dtype=bool)
    else:
        base_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if base_mask.shape[0] != len(values):
            raise ValueError("inlier_mask must contain one value per match")
    errors = match_reprojection_errors(values, np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4), camera)
    selected = np.zeros((len(values),), dtype=bool)
    best_by_query: dict[int, tuple[int, float, float]] = {}
    tie = max(float(residual_tie_px), 0.0)
    for idx, (match, is_inlier) in enumerate(zip(values, base_mask)):
        if not bool(is_inlier) or not np.isfinite(errors[idx]):
            continue
        query_id = int(match.token_index)
        residual = float(errors[idx])
        confidence = _pnp_match_confidence(match)
        previous = best_by_query.get(query_id)
        if previous is None:
            best_by_query[query_id] = (idx, residual, confidence)
            continue
        _prev_idx, prev_residual, prev_confidence = previous
        if residual < prev_residual - tie or (abs(residual - prev_residual) <= tie and confidence > prev_confidence):
            best_by_query[query_id] = (idx, residual, confidence)
    for idx, _residual, _confidence in best_by_query.values():
        selected[int(idx)] = True
    return selected


def refit_pose_with_unique_query_inliers(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    initial_pose_w2c: np.ndarray | None,
    initial_inlier_mask: np.ndarray | Sequence[bool] | None,
    *,
    min_inliers: int = 4,
    pnp_method: str = "EPNP",
    refine_method: str = "LM",
) -> PnPResult:
    """Refit PnP after enforcing at most one accepted 3D candidate per query token."""

    values = list(matches)
    if len(values) < 4 or initial_pose_w2c is None:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(values),), dtype=bool),
            match_count=len(values),
            inlier_count=0,
        )
    unique_mask = select_unique_query_inlier_mask(values, initial_pose_w2c, camera, initial_inlier_mask)
    selected_indices = np.flatnonzero(unique_mask)
    if selected_indices.shape[0] < 4 or selected_indices.shape[0] < int(min_inliers):
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(values),), dtype=bool),
            match_count=len(values),
            inlier_count=0,
        )
    selected_matches = [values[int(idx)] for idx in selected_indices]
    refit = estimate_pose_pnp_fixed(
        selected_matches,
        camera,
        min_inliers=int(min_inliers),
        pnp_method=pnp_method,
        refine_method=refine_method,
    )
    if not refit.success or refit.pose_w2c is None:
        return PnPResult(
            success=False,
            pose_w2c=None,
            inlier_mask=np.zeros((len(values),), dtype=bool),
            match_count=len(values),
            inlier_count=0,
        )
    mask = np.zeros((len(values),), dtype=bool)
    refit_mask = np.asarray(refit.inlier_mask, dtype=bool).reshape(-1)
    if refit_mask.shape[0] != selected_indices.shape[0]:
        refit_mask = np.ones((selected_indices.shape[0],), dtype=bool)
    mask[selected_indices[refit_mask]] = True
    return PnPResult(
        success=True,
        pose_w2c=np.asarray(refit.pose_w2c, dtype=np.float64).reshape(4, 4),
        inlier_mask=mask,
        match_count=len(values),
        inlier_count=int(mask.sum()),
    )


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
    *,
    pose_w2c: np.ndarray | None = None,
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
        "world_z_range_m": None,
        "depth_range_m": None,
        "positive_depth_fraction": None,
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
    stats["world_z_range_m"] = float(np.max(xyz[:, 2]) - np.min(xyz[:, 2])) if xyz.shape[0] else None
    if pose_w2c is not None and xyz.shape[0]:
        pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        camera_depths = (xyz @ pose[:3, :3].T + pose[:3, 3])[:, 2]
        finite = np.isfinite(camera_depths)
        positive = finite & (camera_depths > 1e-6)
        stats["positive_depth_fraction"] = float(np.sum(positive) / max(len(camera_depths), 1))
        if np.any(positive):
            stats["depth_range_m"] = float(
                np.max(camera_depths[positive]) - np.min(camera_depths[positive])
            )
    return stats
