"""Patch-to-3D-footprint matching for VFM local localization.

The footprint formulation treats a VFM token as a regional observation.  A
query patch is matched against a reference-token 3D landmark set instead of a
single landmark, then converted back to sparse 2D-3D correspondences for the
existing PnP backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.patch_to_3d_matching import (
    PatchPositiveSet,
    PatchPositiveSets,
    TokenPatchBox,
    _query_landmark_topk,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch, normalize_rows, token_grid_xy


@dataclass(frozen=True)
class FootprintObservation:
    image_id: str
    track_id: int
    xy: np.ndarray
    image_width: int
    image_height: int
    reprojection_error: float | None = None


@dataclass(frozen=True)
class FootprintUnit:
    unit_id: int
    reference_image_id: str
    token_index: int
    landmark_indices: np.ndarray
    track_ids: tuple[int, ...]
    xyz_mean: np.ndarray
    feature_mean: np.ndarray
    feature_medoid: np.ndarray
    feature_var: float
    num_landmarks: int
    mean_observation_count: float
    mean_landmark_variance: float
    mean_reprojection_error: float | None = None


@dataclass(frozen=True)
class FootprintBank:
    units: tuple[FootprintUnit, ...]
    track_to_units: Mapping[int, tuple[int, ...]]
    landmark_index_to_units: Mapping[int, tuple[int, ...]]
    feature_dim: int

    def __len__(self) -> int:
        return len(self.units)

    @property
    def mean_features(self) -> np.ndarray:
        if not self.units:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack([unit.feature_mean for unit in self.units], axis=0).astype(np.float32, copy=False)

    @property
    def medoid_features(self) -> np.ndarray:
        if not self.units:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack([unit.feature_medoid for unit in self.units], axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class PatchToFootprintMatchingConfig:
    footprint_top_k: int = 5
    footprint_score_mode: str = "max"
    topk_average_k: int = 3
    conversion_mode: str = "top_landmark"
    min_similarity: float = 0.0
    min_landmarks_per_footprint: int = 1
    max_landmarks_per_footprint: int | None = 64
    footprint_cell_radius: int = 0
    query_token_step: int = 1
    landmark_candidate_k: int = 64
    max_matches: int | None = 1000
    block_size: int = 512
    similarity_device: str = "cpu"
    min_positive_overlap: int = 1
    strong_positive_overlap: int = 3
    strong_positive_ratio: float = 0.2

    def __post_init__(self) -> None:
        if int(self.footprint_top_k) <= 0:
            raise ValueError("footprint_top_k must be positive")
        if self.footprint_score_mode not in {"mean", "medoid", "max", "topk_avg"}:
            raise ValueError("footprint_score_mode must be one of: mean, medoid, max, topk_avg")
        if int(self.topk_average_k) <= 0:
            raise ValueError("topk_average_k must be positive")
        if self.conversion_mode not in {"top_landmark", "quality_landmark", "centroid"}:
            raise ValueError("conversion_mode must be one of: top_landmark, quality_landmark, centroid")
        if int(self.min_landmarks_per_footprint) <= 0:
            raise ValueError("min_landmarks_per_footprint must be positive")
        if self.max_landmarks_per_footprint is not None and int(self.max_landmarks_per_footprint) <= 0:
            raise ValueError("max_landmarks_per_footprint must be positive")
        if int(self.footprint_cell_radius) < 0:
            raise ValueError("footprint_cell_radius must be non-negative")
        if int(self.query_token_step) <= 0:
            raise ValueError("query_token_step must be positive")
        if int(self.landmark_candidate_k) <= 0:
            raise ValueError("landmark_candidate_k must be positive")
        if self.max_matches is not None and int(self.max_matches) <= 0:
            raise ValueError("max_matches must be positive")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if int(self.min_positive_overlap) <= 0:
            raise ValueError("min_positive_overlap must be positive")
        if int(self.strong_positive_overlap) <= 0:
            raise ValueError("strong_positive_overlap must be positive")
        if not 0.0 <= float(self.strong_positive_ratio) <= 1.0:
            raise ValueError("strong_positive_ratio must be in [0, 1]")


@dataclass(frozen=True)
class FootprintMatch:
    token_index: int
    xy: np.ndarray
    unit_id: int
    reference_image_id: str
    footprint_token_index: int
    footprint_track_ids: tuple[int, ...]
    selected_track_id: int
    selected_xyz: np.ndarray
    similarity: float
    selected_landmark_similarity: float
    score_mode: str
    source: str = "patch_footprint"
    footprint_num_landmarks: int = 0
    footprint_feature_var: float | None = None
    footprint_mean_observation_count: float | None = None
    footprint_mean_landmark_variance: float | None = None
    footprint_mean_reprojection_error: float | None = None


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


def _flatten_query_features(feature_map: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("query feature map must have shape (C, H, W)")
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


def _unit_feature_medoid(features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 1:
        return features[0].astype(np.float32, copy=True)
    scores = features @ features.T
    medoid_idx = int(np.argmax(np.sum(scores, axis=1)))
    return features[medoid_idx].astype(np.float32, copy=True)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32, copy=False)


def build_reference_token_footprint_bank(
    landmark_index: LandmarkMapIndex,
    observations_by_image: Mapping[str, Sequence[FootprintObservation]],
    reference_image_ids: Sequence[str],
    token_width: int,
    token_height: int,
    min_landmarks_per_footprint: int = 1,
    max_landmarks_per_footprint: int | None = 64,
    footprint_cell_radius: int = 0,
) -> FootprintBank:
    """Build reference-token 3D footprint units from COLMAP 2D observations."""

    if token_width <= 0 or token_height <= 0:
        raise ValueError("token grid dimensions must be positive")
    min_landmarks = int(min_landmarks_per_footprint)
    if min_landmarks <= 0:
        raise ValueError("min_landmarks_per_footprint must be positive")
    cell_radius = int(footprint_cell_radius)
    if cell_radius < 0:
        raise ValueError("footprint_cell_radius must be non-negative")
    if len(landmark_index) == 0:
        return FootprintBank(units=(), track_to_units={}, landmark_index_to_units={}, feature_dim=landmark_index.feature_dim)

    track_to_landmark = {int(track_id): idx for idx, track_id in enumerate(landmark_index.track_ids.tolist())}
    cells: dict[tuple[str, int], set[int]] = {}
    references = []
    seen_refs: set[str] = set()
    for image_id in reference_image_ids:
        image_key = str(image_id)
        if image_key in seen_refs:
            continue
        seen_refs.add(image_key)
        references.append(image_key)
    for image_id in references:
        for obs in observations_by_image.get(str(image_id), ()):
            landmark_idx = track_to_landmark.get(int(obs.track_id))
            if landmark_idx is None:
                continue
            token_index = _token_index_from_xy(
                np.asarray(obs.xy, dtype=np.float64),
                int(obs.image_width),
                int(obs.image_height),
                int(token_width),
                int(token_height),
            )
            token_x = int(token_index) % int(token_width)
            token_y = int(token_index) // int(token_width)
            for yy in range(max(0, token_y - cell_radius), min(int(token_height), token_y + cell_radius + 1)):
                for xx in range(max(0, token_x - cell_radius), min(int(token_width), token_x + cell_radius + 1)):
                    local_token = int(yy * int(token_width) + xx)
                    cells.setdefault((str(image_id), local_token), set()).add(int(landmark_idx))

    landmark_features, valid = normalize_rows(landmark_index.features)
    units = []
    track_to_units: dict[int, list[int]] = {}
    landmark_index_to_units: dict[int, list[int]] = {}
    for unit_id, ((image_id, token_index), landmark_set) in enumerate(sorted(cells.items())):
        landmark_indices = np.asarray(sorted(landmark_set), dtype=np.int64)
        if landmark_indices.size < min_landmarks:
            continue
        if max_landmarks_per_footprint is not None and landmark_indices.size > int(max_landmarks_per_footprint):
            order = np.lexsort(
                (
                    -landmark_index.observation_counts[landmark_indices],
                    landmark_index.mean_variances[landmark_indices],
                )
            )
            landmark_indices = landmark_indices[order[: int(max_landmarks_per_footprint)]]
        landmark_indices = landmark_indices[valid[landmark_indices]]
        if landmark_indices.size < min_landmarks:
            continue
        features = landmark_features[landmark_indices]
        mean_feature = _normalize_vector(np.mean(features, axis=0))
        medoid_feature = _unit_feature_medoid(features)
        tracks = tuple(int(landmark_index.track_ids[idx]) for idx in landmark_indices.tolist())
        reproj = landmark_index.reprojection_errors[landmark_indices]
        unit = FootprintUnit(
            unit_id=len(units),
            reference_image_id=str(image_id),
            token_index=int(token_index),
            landmark_indices=landmark_indices.astype(np.int64, copy=True),
            track_ids=tracks,
            xyz_mean=np.mean(landmark_index.xyz[landmark_indices], axis=0).astype(np.float64),
            feature_mean=mean_feature,
            feature_medoid=medoid_feature,
            feature_var=float(np.mean(np.var(features, axis=0))) if features.shape[0] > 1 else 0.0,
            num_landmarks=int(landmark_indices.size),
            mean_observation_count=float(np.mean(landmark_index.observation_counts[landmark_indices])),
            mean_landmark_variance=float(np.mean(landmark_index.mean_variances[landmark_indices])),
            mean_reprojection_error=float(np.mean(reproj)) if reproj is not None and reproj.size else None,
        )
        units.append(unit)
        for landmark_idx, track_id in zip(landmark_indices.tolist(), tracks):
            track_to_units.setdefault(int(track_id), []).append(int(unit.unit_id))
            landmark_index_to_units.setdefault(int(landmark_idx), []).append(int(unit.unit_id))

    return FootprintBank(
        units=tuple(units),
        track_to_units={key: tuple(values) for key, values in track_to_units.items()},
        landmark_index_to_units={key: tuple(values) for key, values in landmark_index_to_units.items()},
        feature_dim=landmark_index.feature_dim,
    )


def footprint_bank_stats(bank: FootprintBank) -> dict[str, float | int]:
    counts = np.asarray([unit.num_landmarks for unit in bank.units], dtype=np.float64)
    references = {unit.reference_image_id for unit in bank.units}
    covered_tracks = set(bank.track_to_units.keys())
    if counts.size == 0:
        return {
            "footprint_count": 0,
            "reference_image_count": 0,
            "covered_track_count": 0,
            "mean_landmarks_per_footprint": 0.0,
            "median_landmarks_per_footprint": 0.0,
            "max_landmarks_per_footprint": 0,
            "single_landmark_footprint_ratio": 0.0,
        }
    return {
        "footprint_count": int(counts.size),
        "reference_image_count": int(len(references)),
        "covered_track_count": int(len(covered_tracks)),
        "mean_landmarks_per_footprint": float(np.mean(counts)),
        "median_landmarks_per_footprint": float(np.median(counts)),
        "max_landmarks_per_footprint": int(np.max(counts)),
        "single_landmark_footprint_ratio": float(np.mean(counts == 1.0)),
    }


def _select_landmark_from_unit(
    query_feature: np.ndarray,
    landmark_features: np.ndarray,
    landmark_index: LandmarkMapIndex,
    unit: FootprintUnit,
    conversion_mode: str,
) -> tuple[int, np.ndarray, float]:
    if conversion_mode == "centroid":
        return -int(unit.unit_id) - 1, np.asarray(unit.xyz_mean, dtype=np.float64).reshape(3), float("nan")
    local_features = landmark_features[unit.landmark_indices]
    similarities = np.asarray(local_features @ query_feature.reshape(-1), dtype=np.float32)
    if conversion_mode == "quality_landmark":
        quality = np.asarray(landmark_index.observation_counts[unit.landmark_indices], dtype=np.float32)
        quality = np.log1p(np.maximum(quality, 0.0))
        if np.max(quality) > 0.0:
            quality = quality / np.max(quality)
        score = similarities * (0.5 + 0.5 * quality) - np.asarray(
            landmark_index.mean_variances[unit.landmark_indices],
            dtype=np.float32,
        )
    else:
        score = similarities
    selected_local = int(np.argmax(score))
    landmark_idx = int(unit.landmark_indices[selected_local])
    return (
        int(landmark_index.track_ids[landmark_idx]),
        np.asarray(landmark_index.xyz[landmark_idx], dtype=np.float64).reshape(3),
        float(similarities[selected_local]),
    )


def _make_match(
    token_index: int,
    xy: np.ndarray,
    unit: FootprintUnit,
    query_feature: np.ndarray,
    landmark_features: np.ndarray,
    landmark_index: LandmarkMapIndex,
    similarity: float,
    config: PatchToFootprintMatchingConfig,
) -> FootprintMatch:
    selected_track_id, selected_xyz, selected_similarity = _select_landmark_from_unit(
        query_feature,
        landmark_features,
        landmark_index,
        unit,
        config.conversion_mode,
    )
    return FootprintMatch(
        token_index=int(token_index),
        xy=np.asarray(xy, dtype=np.float64).reshape(2),
        unit_id=int(unit.unit_id),
        reference_image_id=unit.reference_image_id,
        footprint_token_index=int(unit.token_index),
        footprint_track_ids=tuple(int(track_id) for track_id in unit.track_ids),
        selected_track_id=int(selected_track_id),
        selected_xyz=np.asarray(selected_xyz, dtype=np.float64).reshape(3),
        similarity=float(similarity),
        selected_landmark_similarity=float(selected_similarity),
        score_mode=str(config.footprint_score_mode),
        source=f"patch_footprint_{config.footprint_score_mode}",
        footprint_num_landmarks=int(unit.num_landmarks),
        footprint_feature_var=float(unit.feature_var),
        footprint_mean_observation_count=float(unit.mean_observation_count),
        footprint_mean_landmark_variance=float(unit.mean_landmark_variance),
        footprint_mean_reprojection_error=unit.mean_reprojection_error,
    )


def _match_query_to_aggregate_footprints(
    query_features: np.ndarray,
    token_indices: np.ndarray,
    centers: np.ndarray,
    landmark_features: np.ndarray,
    landmark_index: LandmarkMapIndex,
    bank: FootprintBank,
    unit_features: np.ndarray,
    config: PatchToFootprintMatchingConfig,
) -> list[FootprintMatch]:
    top_indices, top_scores = _query_landmark_topk(
        query_features,
        unit_features,
        top_k=config.footprint_top_k,
        block_size=config.block_size,
        device=config.similarity_device,
    )
    matches: list[FootprintMatch] = []
    for query_row in range(query_features.shape[0]):
        for rank in range(top_indices.shape[1]):
            unit_idx = int(top_indices[query_row, rank])
            if unit_idx < 0:
                continue
            similarity = float(top_scores[query_row, rank])
            if similarity < float(config.min_similarity):
                continue
            unit = bank.units[unit_idx]
            token_index = int(token_indices[query_row])
            matches.append(
                _make_match(
                    token_index,
                    centers[token_index],
                    unit,
                    query_features[query_row],
                    landmark_features,
                    landmark_index,
                    similarity,
                    config,
                )
            )
    return matches


def _match_query_to_member_scored_footprints(
    query_features: np.ndarray,
    token_indices: np.ndarray,
    centers: np.ndarray,
    landmark_features: np.ndarray,
    landmark_index: LandmarkMapIndex,
    bank: FootprintBank,
    config: PatchToFootprintMatchingConfig,
) -> list[FootprintMatch]:
    top_indices, top_scores = _query_landmark_topk(
        query_features,
        landmark_features,
        top_k=max(config.landmark_candidate_k, config.topk_average_k),
        block_size=config.block_size,
        device=config.similarity_device,
    )
    matches: list[FootprintMatch] = []
    for query_row in range(query_features.shape[0]):
        scores_by_unit: dict[int, list[float]] = {}
        for landmark_idx, score in zip(top_indices[query_row].tolist(), top_scores[query_row].tolist()):
            if int(landmark_idx) < 0:
                continue
            for unit_id in bank.landmark_index_to_units.get(int(landmark_idx), ()):
                scores_by_unit.setdefault(int(unit_id), []).append(float(score))
        if not scores_by_unit:
            continue
        scored_units: list[tuple[float, int]] = []
        for unit_id, scores in scores_by_unit.items():
            ordered = sorted(scores, reverse=True)
            if config.footprint_score_mode == "topk_avg":
                value = float(np.mean(ordered[: int(config.topk_average_k)]))
            else:
                value = float(ordered[0])
            if value >= float(config.min_similarity):
                scored_units.append((value, int(unit_id)))
        if not scored_units:
            continue
        scored_units.sort(key=lambda item: item[0], reverse=True)
        token_index = int(token_indices[query_row])
        for similarity, unit_id in scored_units[: int(config.footprint_top_k)]:
            matches.append(
                _make_match(
                    token_index,
                    centers[token_index],
                    bank.units[int(unit_id)],
                    query_features[query_row],
                    landmark_features,
                    landmark_index,
                    similarity,
                    config,
                )
            )
    return matches


def match_query_patches_to_footprints(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    footprint_bank: FootprintBank,
    config: PatchToFootprintMatchingConfig | None = None,
    image_width: int = 1024,
    image_height: int = 576,
) -> list[FootprintMatch]:
    config = config or PatchToFootprintMatchingConfig()
    if len(landmark_index) == 0 or len(footprint_bank) == 0:
        return []
    query_features, token_indices = _flatten_query_features(query_feature_map, config.query_token_step)
    query_features, valid_query = normalize_rows(query_features)
    landmark_features, valid_landmarks = normalize_rows(landmark_index.features)
    if not np.all(valid_landmarks):
        # Footprint units were built against the original submap.  Invalid map
        # descriptors should already be rare; keeping array sizes stable avoids
        # changing unit landmark indices after construction.
        landmark_features[~valid_landmarks] = 0.0
    valid_query_indices = np.flatnonzero(valid_query)
    if valid_query_indices.size == 0:
        return []
    query_features = query_features[valid_query_indices]
    token_indices = token_indices[valid_query_indices]
    token_height = int(query_feature_map.shape[1])
    token_width = int(query_feature_map.shape[2])
    centers = token_grid_xy(token_width, token_height, image_width, image_height, step=1)

    if config.footprint_score_mode == "mean":
        unit_features = footprint_bank.mean_features
        unit_features, _valid = normalize_rows(unit_features)
        matches = _match_query_to_aggregate_footprints(
            query_features,
            token_indices,
            centers,
            landmark_features,
            landmark_index,
            footprint_bank,
            unit_features,
            config,
        )
    elif config.footprint_score_mode == "medoid":
        unit_features = footprint_bank.medoid_features
        unit_features, _valid = normalize_rows(unit_features)
        matches = _match_query_to_aggregate_footprints(
            query_features,
            token_indices,
            centers,
            landmark_features,
            landmark_index,
            footprint_bank,
            unit_features,
            config,
        )
    else:
        matches = _match_query_to_member_scored_footprints(
            query_features,
            token_indices,
            centers,
            landmark_features,
            landmark_index,
            footprint_bank,
            config,
        )
    matches.sort(key=lambda item: item.similarity, reverse=True)
    if config.max_matches is not None:
        matches = matches[: int(config.max_matches)]
    return matches


def footprint_matches_to_pnp_matches(matches: Sequence[FootprintMatch]) -> list[QueryTo3DMatch]:
    output: list[QueryTo3DMatch] = []
    for match in matches:
        output.append(
            QueryTo3DMatch(
                token_index=int(match.token_index),
                xy=np.asarray(match.xy, dtype=np.float64).reshape(2),
                track_id=int(match.selected_track_id),
                xyz=np.asarray(match.selected_xyz, dtype=np.float64).reshape(3),
                similarity=float(match.similarity),
                ratio=0.0,
                landmark_variance=0.0
                if match.footprint_mean_landmark_variance is None
                else float(match.footprint_mean_landmark_variance),
                source=str(match.source),
                observation_count=None
                if match.footprint_mean_observation_count is None
                else int(round(float(match.footprint_mean_observation_count))),
                visibility_count=None,
                landmark_reprojection_error=match.footprint_mean_reprojection_error,
                similarity_margin=None,
            )
        )
    return output


def _footprint_overlap(match: FootprintMatch, positives: PatchPositiveSets) -> tuple[int, int, int]:
    positive = positives.by_token.get(int(match.token_index))
    if positive is None:
        return 0, 0, len(match.footprint_track_ids)
    positive_tracks = positive.track_ids
    overlap = len(positive_tracks.intersection(int(track_id) for track_id in match.footprint_track_ids))
    return int(overlap), int(len(positive_tracks)), int(len(match.footprint_track_ids))


def evaluate_footprint_matches(
    matches: Sequence[FootprintMatch],
    positives: PatchPositiveSets,
    pnp_inlier_mask: np.ndarray | None = None,
    top_k: int = 5,
    min_overlap: int = 1,
    strong_overlap: int = 3,
    strong_overlap_ratio: float = 0.2,
) -> dict[str, float | int | None]:
    stats: dict[str, float | int | None] = {
        "footprint_match_count": int(len(matches)),
        "footprint_at_1": 0.0,
        f"footprint_at_{int(top_k)}": 0.0,
        "strong_footprint_at_1": 0.0,
        f"strong_footprint_at_{int(top_k)}": 0.0,
        f"positive_landmark_recall_at_{int(top_k)}": 0.0,
        "mean_overlap_count": 0.0,
        "mean_overlap_ratio": 0.0,
        "pnp_inlier_footprint_at_1": None,
        "pnp_inlier_strong_footprint_at_1": None,
    }
    if not matches:
        return stats
    overlaps = []
    overlap_ratios = []
    footprint_correct = []
    strong_correct = []
    for match in matches:
        overlap, positive_count, footprint_count = _footprint_overlap(match, positives)
        denom = max(min(positive_count, footprint_count), 1)
        ratio = float(overlap / denom)
        overlaps.append(float(overlap))
        overlap_ratios.append(ratio)
        footprint_correct.append(overlap >= int(min_overlap))
        strong_correct.append(overlap >= int(strong_overlap) or ratio >= float(strong_overlap_ratio))
    footprint_correct_arr = np.asarray(footprint_correct, dtype=bool)
    strong_correct_arr = np.asarray(strong_correct, dtype=bool)
    stats["footprint_at_1"] = float(np.mean(footprint_correct_arr))
    stats["strong_footprint_at_1"] = float(np.mean(strong_correct_arr))
    by_token: dict[int, list[int]] = {}
    for match_idx, match in enumerate(matches):
        by_token.setdefault(int(match.token_index), []).append(int(match_idx))
    if by_token:
        stats[f"footprint_at_{int(top_k)}"] = float(
            np.mean([np.any(footprint_correct_arr[indices[: int(top_k)]]) for indices in by_token.values()])
        )
        stats[f"strong_footprint_at_{int(top_k)}"] = float(
            np.mean([np.any(strong_correct_arr[indices[: int(top_k)]]) for indices in by_token.values()])
        )
    recalls = []
    for token_index, positive in positives.by_token.items():
        if not positive.track_ids:
            continue
        indices = by_token.get(int(token_index), [])[: int(top_k)]
        if not indices:
            recalls.append(0.0)
            continue
        covered: set[int] = set()
        for idx in indices:
            covered.update(int(track_id) for track_id in matches[int(idx)].footprint_track_ids)
        recalls.append(float(len(positive.track_ids.intersection(covered)) / max(len(positive.track_ids), 1)))
    stats[f"positive_landmark_recall_at_{int(top_k)}"] = float(np.mean(recalls)) if recalls else 0.0
    stats["mean_overlap_count"] = float(np.mean(overlaps))
    stats["mean_overlap_ratio"] = float(np.mean(overlap_ratios))
    if pnp_inlier_mask is not None:
        mask = np.asarray(pnp_inlier_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != len(matches):
            raise ValueError("pnp_inlier_mask must have one value per footprint match")
        if np.any(mask):
            stats["pnp_inlier_footprint_at_1"] = float(np.mean(footprint_correct_arr[mask]))
            stats["pnp_inlier_strong_footprint_at_1"] = float(np.mean(strong_correct_arr[mask]))
    return stats


def query_footprint_positive_stats(
    positives: PatchPositiveSets,
    bank: FootprintBank,
    min_overlap: int = 1,
    strong_overlap: int = 3,
    strong_overlap_ratio: float = 0.2,
) -> dict[str, float | int]:
    positive_counts = []
    strong_counts = []
    for positive in positives.by_token.values():
        unit_to_overlap: dict[int, int] = {}
        for track_id in positive.track_ids:
            for unit_id in bank.track_to_units.get(int(track_id), ()):
                unit_to_overlap[int(unit_id)] = unit_to_overlap.get(int(unit_id), 0) + 1
        positive_count = 0
        strong_count = 0
        for unit_id, overlap in unit_to_overlap.items():
            unit = bank.units[int(unit_id)]
            ratio = float(overlap / max(min(len(positive.track_ids), unit.num_landmarks), 1))
            if overlap >= int(min_overlap):
                positive_count += 1
            if overlap >= int(strong_overlap) or ratio >= float(strong_overlap_ratio):
                strong_count += 1
        positive_counts.append(float(positive_count))
        strong_counts.append(float(strong_count))
    counts = np.asarray(positive_counts, dtype=np.float64)
    strong = np.asarray(strong_counts, dtype=np.float64)
    if counts.size == 0:
        return {
            "token_count": 0,
            "mean_positive_footprints_per_token": 0.0,
            "median_positive_footprints_per_token": 0.0,
            "zero_positive_footprint_token_ratio": 1.0,
            "mean_strong_positive_footprints_per_token": 0.0,
        }
    return {
        "token_count": int(counts.size),
        "mean_positive_footprints_per_token": float(np.mean(counts)),
        "median_positive_footprints_per_token": float(np.median(counts)),
        "zero_positive_footprint_token_ratio": float(np.mean(counts == 0.0)),
        "mean_strong_positive_footprints_per_token": float(np.mean(strong)),
    }


def observation_index_from_colmap_observations(observations: Sequence[object]) -> dict[str, list[FootprintObservation]]:
    by_image: dict[str, list[FootprintObservation]] = {}
    for obs in observations:
        image_width = getattr(obs, "image_width", None)
        image_height = getattr(obs, "image_height", None)
        if image_width is None or image_height is None:
            continue
        item = FootprintObservation(
            image_id=str(getattr(obs, "image_id")),
            track_id=int(getattr(obs, "track_id")),
            xy=np.asarray(getattr(obs, "xy"), dtype=np.float64).reshape(2),
            image_width=int(image_width),
            image_height=int(image_height),
            reprojection_error=None
            if getattr(obs, "reprojection_error", None) is None
            else float(getattr(obs, "reprojection_error")),
        )
        by_image.setdefault(item.image_id, []).append(item)
    return by_image
