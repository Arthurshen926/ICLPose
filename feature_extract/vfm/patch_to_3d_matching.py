"""Patch-level query VFM token to sparse 3D landmark matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkQualityConfig,
    LandmarkMapIndex,
    QueryTo3DMatchingConfig,
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_ransac,
    landmark_quality_scores,
    match_reprojection_errors,
    match_query_tokens_to_landmarks,
    normalize_rows,
    token_grid_xy,
)


@dataclass(frozen=True)
class TokenPatchBox:
    token_index: int
    center: np.ndarray
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, xy: np.ndarray) -> bool:
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        return bool(self.x0 <= point[0] <= self.x1 and self.y0 <= point[1] <= self.y1)


@dataclass(frozen=True)
class PatchPositiveSet:
    token_index: int
    patch_box: TokenPatchBox
    track_ids: set[int]

    @property
    def count(self) -> int:
        return len(self.track_ids)


@dataclass(frozen=True)
class PatchPositiveSets:
    by_token: Mapping[int, PatchPositiveSet]
    stride_x_px: float
    stride_y_px: float
    visible_track_ids: set[int] | None = None


@dataclass(frozen=True)
class PatchTo3DMatchingConfig:
    top_k: int = 5
    mutual_top_k: int = 5
    match_mode: str = "soft_mutual"
    min_similarity: float = 0.0
    ratio_threshold: float | None = 0.95
    min_similarity_margin: float | None = None
    min_observation_count: int = 1
    max_landmark_variance: float | None = None
    max_landmark_reprojection_error: float | None = None
    max_landmark_ambiguity: float | None = None
    min_distance_to_boundary_px: float | None = None
    min_quality_weighted_similarity: float | None = None
    query_token_step: int = 1
    max_matches: int | None = None
    block_size: int = 512
    landmark_quality: LandmarkQualityConfig = LandmarkQualityConfig()

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.mutual_top_k <= 0:
            raise ValueError("mutual_top_k must be positive")
        if self.match_mode not in {"nn", "mnn", "soft_mutual"}:
            raise ValueError("match_mode must be one of: nn, mnn, soft_mutual")
        if self.ratio_threshold is not None and not 0.0 < float(self.ratio_threshold) <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")
        if self.min_quality_weighted_similarity is not None and float(self.min_quality_weighted_similarity) < -1.0:
            raise ValueError("min_quality_weighted_similarity must be >= -1")


def token_patch_boxes(
    token_width: int,
    token_height: int,
    image_width: int,
    image_height: int,
    scale: float = 1.0,
) -> list[TokenPatchBox]:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    centers = token_grid_xy(token_width, token_height, image_width, image_height, step=1)
    stride_x = float(image_width - 1) / max(float(token_width - 1), 1.0)
    stride_y = float(image_height - 1) / max(float(token_height - 1), 1.0)
    half_x = 0.5 * stride_x * float(scale)
    half_y = 0.5 * stride_y * float(scale)
    boxes = []
    for token_index, center in enumerate(centers):
        boxes.append(
            TokenPatchBox(
                token_index=int(token_index),
                center=np.asarray(center, dtype=np.float64),
                x0=max(0.0, float(center[0]) - half_x),
                y0=max(0.0, float(center[1]) - half_y),
                x1=min(float(image_width - 1), float(center[0]) + half_x),
                y1=min(float(image_height - 1), float(center[1]) + half_y),
            )
        )
    return boxes


def _project_landmarks(index: LandmarkMapIndex, pose_w2c: np.ndarray, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    if len(index) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for patch positive sets") from exc
    matrix, distortion = camera_matrix_and_distortion(camera)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points = index.xyz.astype(np.float64)
    camera_points = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    visible = camera_points[:, 2] > 1e-6
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(points, rvec, pose[:3, 3], matrix, distortion)
    xy = projected.reshape(-1, 2).astype(np.float64)
    visible &= (xy[:, 0] >= 0.0) & (xy[:, 0] <= float(camera.width - 1))
    visible &= (xy[:, 1] >= 0.0) & (xy[:, 1] <= float(camera.height - 1))
    return xy, visible


def build_patch_positive_sets(
    index: LandmarkMapIndex,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    token_width: int,
    token_height: int,
    patch_scale: float = 1.0,
) -> PatchPositiveSets:
    boxes = token_patch_boxes(token_width, token_height, camera.width, camera.height, scale=patch_scale)
    stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
    stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
    projected_xy, visible = _project_landmarks(index, pose_w2c, camera)
    by_token = {
        int(box.token_index): PatchPositiveSet(token_index=int(box.token_index), patch_box=box, track_ids=set())
        for box in boxes
    }
    visible_track_ids: set[int] = set()
    if len(index) == 0:
        return PatchPositiveSets(
            by_token=by_token,
            stride_x_px=stride_x,
            stride_y_px=stride_y,
            visible_track_ids=visible_track_ids,
        )
    for landmark_idx, is_visible in enumerate(visible):
        if not bool(is_visible):
            continue
        visible_track_ids.add(int(index.track_ids[landmark_idx]))
        xy = projected_xy[landmark_idx]
        x_idx = int(round(np.clip(xy[0] / max(float(camera.width - 1), 1.0), 0.0, 1.0) * max(token_width - 1, 0)))
        y_idx = int(round(np.clip(xy[1] / max(float(camera.height - 1), 1.0), 0.0, 1.0) * max(token_height - 1, 0)))
        for yy in range(max(0, y_idx - 1), min(token_height, y_idx + 2)):
            for xx in range(max(0, x_idx - 1), min(token_width, x_idx + 2)):
                token_index = yy * token_width + xx
                positive = by_token[token_index]
                if positive.patch_box.contains(xy):
                    positive.track_ids.add(int(index.track_ids[landmark_idx]))
    return PatchPositiveSets(
        by_token=by_token,
        stride_x_px=stride_x,
        stride_y_px=stride_y,
        visible_track_ids=visible_track_ids,
    )


def patch_positive_set_stats(positives: PatchPositiveSets) -> dict[str, float | int]:
    counts = np.asarray([item.count for item in positives.by_token.values()], dtype=np.float64)
    if counts.size == 0:
        return {
            "token_count": 0,
            "visible_landmark_count": 0,
            "positive_landmark_count": 0,
            "mean_positives_per_token": 0.0,
            "median_positives_per_token": 0.0,
            "mean_positives_per_nonempty_token": 0.0,
            "max_positives_per_token": 0,
            "zero_positive_token_ratio": 1.0,
            "nonempty_patch_fraction": 0.0,
            "positive_landmark_density_per_token": 0.0,
        }
    positive_ids: set[int] = set()
    for item in positives.by_token.values():
        positive_ids.update(int(track_id) for track_id in item.track_ids)
    nonempty = counts[counts > 0.0]
    token_count = int(counts.size)
    return {
        "token_count": token_count,
        "visible_landmark_count": 0 if positives.visible_track_ids is None else int(len(positives.visible_track_ids)),
        "positive_landmark_count": int(len(positive_ids)),
        "mean_positives_per_token": float(np.mean(counts)),
        "median_positives_per_token": float(np.median(counts)),
        "mean_positives_per_nonempty_token": 0.0 if nonempty.size == 0 else float(np.mean(nonempty)),
        "max_positives_per_token": int(np.max(counts)),
        "zero_positive_token_ratio": float(np.mean(counts == 0.0)),
        "nonempty_patch_fraction": float(np.mean(counts > 0.0)),
        "positive_landmark_density_per_token": float(len(positive_ids) / max(token_count, 1)),
    }


def filter_landmarks_by_projected_visibility(
    index: LandmarkMapIndex,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> LandmarkMapIndex:
    _projected_xy, visible = _project_landmarks(index, pose_w2c, camera)
    return index.subset(visible)


def _query_landmark_topk(
    query_features: np.ndarray,
    landmark_features: np.ndarray,
    top_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_count = int(query_features.shape[0])
    landmark_count = int(landmark_features.shape[0])
    effective_top_k = min(int(top_k), landmark_count)
    top_indices = np.full((query_count, effective_top_k), -1, dtype=np.int64)
    top_scores = np.full((query_count, effective_top_k), -np.inf, dtype=np.float32)
    for start in range(0, query_count, block_size):
        end = min(start + block_size, query_count)
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


def _flatten_query_features(feature_map: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("query feature map must have shape (C, H, W)")
    _channels, height, width = values.shape
    features = []
    token_indices = []
    for y_idx in range(0, height, step):
        for x_idx in range(0, width, step):
            features.append(values[:, y_idx, x_idx])
            token_indices.append(y_idx * width + x_idx)
    if not features:
        return np.zeros((0, values.shape[0]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(features, axis=0).astype(np.float32), np.asarray(token_indices, dtype=np.int64)


def _valid_subset(index: LandmarkMapIndex, config: PatchTo3DMatchingConfig) -> LandmarkMapIndex:
    mask = index.observation_counts >= int(config.min_observation_count)
    if config.max_landmark_variance is not None:
        mask &= index.mean_variances <= float(config.max_landmark_variance)
    if config.max_landmark_reprojection_error is not None:
        mask &= index.reprojection_errors <= float(config.max_landmark_reprojection_error)
    if config.max_landmark_ambiguity is not None:
        mask &= index.feature_ambiguities <= float(config.max_landmark_ambiguity)
    return index.subset(mask)


def match_query_patches_to_landmarks(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    config: PatchTo3DMatchingConfig | None = None,
    image_width: int = 1024,
    image_height: int = 576,
) -> list[QueryTo3DMatch]:
    config = config or PatchTo3DMatchingConfig()
    index = _valid_subset(landmark_index, config)
    if len(index) == 0:
        return []
    query_features, token_indices = _flatten_query_features(query_feature_map, config.query_token_step)
    query_features, valid_query = normalize_rows(query_features)
    landmark_features, valid_landmarks = normalize_rows(index.features)
    if not np.all(valid_landmarks):
        index = index.subset(valid_landmarks)
        landmark_features = landmark_features[valid_landmarks]
    valid_query_indices = np.flatnonzero(valid_query)
    if valid_query_indices.size == 0 or len(index) == 0:
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
    query_features = query_features[valid_query_indices]
    token_indices = token_indices[valid_query_indices]

    token_height = int(query_feature_map.shape[1])
    token_width = int(query_feature_map.shape[2])
    centers = token_grid_xy(token_width, token_height, image_width, image_height, step=1)
    query_top_indices, query_top_scores = _query_landmark_topk(
        query_features,
        landmark_features,
        top_k=config.top_k,
        block_size=config.block_size,
    )
    if config.match_mode in {"mnn", "soft_mutual"}:
        landmark_top_indices, _landmark_top_scores = _query_landmark_topk(
            landmark_features,
            query_features,
            top_k=config.mutual_top_k,
            block_size=config.block_size,
        )
        reciprocal = [set(row.tolist()) for row in landmark_top_indices]
    else:
        reciprocal = []

    matches: list[QueryTo3DMatch] = []
    for query_row in range(query_features.shape[0]):
        candidate_matches: list[QueryTo3DMatch] = []
        for rank in range(query_top_indices.shape[1]):
            landmark_idx = int(query_top_indices[query_row, rank])
            if landmark_idx < 0:
                continue
            if config.match_mode == "nn" and rank > 0 and not config.landmark_quality.enabled:
                continue
            if config.match_mode == "mnn" and (rank > 0 or query_row not in reciprocal[landmark_idx]):
                continue
            if config.match_mode == "soft_mutual" and query_row not in reciprocal[landmark_idx]:
                continue
            similarity = float(query_top_scores[query_row, rank])
            if similarity < float(config.min_similarity):
                continue
            second_similarity = None
            if query_top_scores.shape[1] > 1:
                other = np.delete(query_top_scores[query_row], rank)
                if other.size:
                    second_similarity = float(np.max(other))
                    best_distance = max(0.0, 1.0 - similarity)
                    second_distance = max(1e-6, 1.0 - second_similarity)
                    if config.ratio_threshold is not None and best_distance / second_distance > float(config.ratio_threshold):
                        continue
            margin = None if second_similarity is None else float(similarity - second_similarity)
            if config.min_similarity_margin is not None and (margin is None or margin < float(config.min_similarity_margin)):
                continue
            quality = float(quality_scores[landmark_idx])
            quality_weighted_similarity = float(similarity * quality)
            if (
                config.min_quality_weighted_similarity is not None
                and quality_weighted_similarity < float(config.min_quality_weighted_similarity)
            ):
                continue
            token_index = int(token_indices[query_row])
            xy = centers[token_index].astype(np.float64, copy=True)
            boundary = min(float(xy[0]), float(xy[1]), float(image_width - 1) - float(xy[0]), float(image_height - 1) - float(xy[1]))
            if config.min_distance_to_boundary_px is not None and boundary < float(config.min_distance_to_boundary_px):
                continue
            candidate_matches.append(
                QueryTo3DMatch(
                    token_index=token_index,
                    xy=xy,
                    track_id=int(index.track_ids[landmark_idx]),
                    xyz=index.xyz[landmark_idx].astype(np.float64, copy=True),
                    similarity=similarity,
                    ratio=0.0,
                    landmark_variance=float(index.mean_variances[landmark_idx]),
                    source="sparse_patch",
                    observation_count=int(index.observation_counts[landmark_idx]),
                    visibility_count=len(index.observation_image_ids[landmark_idx]),
                    landmark_reprojection_error=float(index.reprojection_errors[landmark_idx]),
                    landmark_quality=quality,
                    landmark_ambiguity=float(landmark_ambiguity[landmark_idx]),
                    quality_weighted_similarity=quality_weighted_similarity,
                    similarity_margin=margin,
                    distance_to_boundary_px=float(boundary),
                )
            )
        if config.match_mode == "nn" and config.landmark_quality.enabled and candidate_matches:
            matches.append(
                max(
                    candidate_matches,
                    key=lambda item: item.quality_weighted_similarity
                    if item.quality_weighted_similarity is not None
                    else item.similarity,
                )
            )
        else:
            matches.extend(candidate_matches)
    matches.sort(
        key=lambda item: item.quality_weighted_similarity
        if item.quality_weighted_similarity is not None
        else item.similarity,
        reverse=True,
    )
    if config.max_matches is not None:
        matches = matches[: int(config.max_matches)]
    return matches


def evaluate_patch_matches(
    matches: Sequence[QueryTo3DMatch],
    positives: PatchPositiveSets,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    stride_px: float,
    pnp_inlier_mask: np.ndarray | None = None,
    top_k: int = 5,
) -> dict[str, float | int | None]:
    stats: dict[str, float | int | None] = {
        "match_count": int(len(matches)),
        "patch_at_1": 0.0,
        f"patch_at_{int(top_k)}": 0.0,
        "gt_precision_5px": 0.0,
        "gt_precision_16px": 0.0,
        "gt_precision_stride": 0.0,
        "gt_precision_2stride": 0.0,
        "gt_reproj_median_px": None,
        "gt_reproj_median_stride": None,
        "pnp_inlier_patch_at_1": None,
        f"pnp_inlier_patch_at_{int(top_k)}": None,
        "pnp_inlier_gt_precision_stride": None,
    }
    if not matches:
        return stats
    errors = match_reprojection_errors(matches, pose_w2c, camera)
    patch_correct = np.asarray(
        [int(match.track_id) in positives.by_token.get(int(match.token_index), PatchPositiveSet(int(match.token_index), TokenPatchBox(int(match.token_index), np.zeros(2), 0, 0, 0, 0), set())).track_ids for match in matches],
        dtype=bool,
    )
    stats["patch_at_1"] = float(np.mean(patch_correct))
    by_token: dict[int, list[bool]] = {}
    for match, correct in zip(matches, patch_correct):
        by_token.setdefault(int(match.token_index), []).append(bool(correct))
    if by_token:
        stats[f"patch_at_{int(top_k)}"] = float(np.mean([any(values[: int(top_k)]) for values in by_token.values()]))
    stats["gt_precision_5px"] = float(np.mean(errors <= 5.0))
    stats["gt_precision_16px"] = float(np.mean(errors <= 16.0))
    stats["gt_precision_stride"] = float(np.mean(errors <= float(stride_px)))
    stats["gt_precision_2stride"] = float(np.mean(errors <= 2.0 * float(stride_px)))
    stats["gt_reproj_median_px"] = float(np.median(errors))
    stats["gt_reproj_median_stride"] = float(np.median(errors) / max(float(stride_px), 1e-6))
    if pnp_inlier_mask is not None:
        mask = np.asarray(pnp_inlier_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != len(matches):
            raise ValueError("pnp_inlier_mask must have one value per match")
        if np.any(mask):
            stats["pnp_inlier_patch_at_1"] = float(np.mean(patch_correct[mask]))
            inlier_by_token: dict[int, list[bool]] = {}
            for match, correct, is_inlier in zip(matches, patch_correct, mask):
                if bool(is_inlier):
                    inlier_by_token.setdefault(int(match.token_index), []).append(bool(correct))
            stats[f"pnp_inlier_patch_at_{int(top_k)}"] = (
                0.0
                if not inlier_by_token
                else float(np.mean([any(values[: int(top_k)]) for values in inlier_by_token.values()]))
            )
            stats["pnp_inlier_gt_precision_stride"] = float(np.mean(errors[mask] <= float(stride_px)))
    return stats


def patch_uncertainty_pnp_threshold(stride_px: float, multiplier: float = 1.5) -> float:
    return float(stride_px) * float(multiplier)
