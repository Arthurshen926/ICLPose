"""Patch-level query VFM token to sparse 3D landmark matching."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkAmbiguityPruningConfig,
    LandmarkQualityConfig,
    LandmarkMapIndex,
    LocalGeometricConsistencyConfig,
    MapReliabilityConfig,
    QueryTo3DMatchingConfig,
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_ransac,
    filter_matches_by_local_geometric_consistency,
    landmark_quality_scores,
    map_reliability_scores,
    map_reliability_uncertainty_scale,
    match_reprojection_errors,
    match_query_tokens_to_landmarks,
    normalize_rows,
    prune_ambiguous_landmarks,
    select_pnp_matches_by_map_reliability,
    token_grid_xy,
)


class PairwiseInlierScorer(Protocol):
    def score_pairs(self, query_descriptors: np.ndarray, landmark_descriptors: np.ndarray) -> np.ndarray:
        ...


CandidateMatchScorer = Callable[[Sequence[QueryTo3DMatch]], Sequence[QueryTo3DMatch]]


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
    projected_xy_by_track: Mapping[int, np.ndarray] | None = None


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
    similarity_device: str = "cpu"
    match_score_mode: str = "similarity_quality"
    landmark_quality: LandmarkQualityConfig = LandmarkQualityConfig()
    pairwise_filter_keep_fraction: float | None = None
    pairwise_filter_min_logit: float | None = None
    pairwise_filter_min_logprob: float | None = None
    map_reliability: MapReliabilityConfig = MapReliabilityConfig()
    local_geometric_consistency: LocalGeometricConsistencyConfig = LocalGeometricConsistencyConfig()
    landmark_ambiguity_pruning: LandmarkAmbiguityPruningConfig = LandmarkAmbiguityPruningConfig()
    return_all_topk_candidates: bool = False
    deduplicate_track_matches: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.mutual_top_k <= 0:
            raise ValueError("mutual_top_k must be positive")
        if self.match_mode not in {"nn", "mnn", "soft_mutual"}:
            raise ValueError("match_mode must be one of: nn, mnn, soft_mutual")
        if self.match_score_mode not in {"similarity", "landmark_quality", "similarity_quality", "similarity_pairwise"}:
            raise ValueError(
                "match_score_mode must be one of: similarity, landmark_quality, similarity_quality, similarity_pairwise"
            )
        if self.ratio_threshold is not None and not 0.0 < float(self.ratio_threshold) <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")
        if self.min_quality_weighted_similarity is not None and float(self.min_quality_weighted_similarity) < -1.0:
            raise ValueError("min_quality_weighted_similarity must be >= -1")
        if self.pairwise_filter_keep_fraction is not None and not 0.0 < float(self.pairwise_filter_keep_fraction) <= 1.0:
            raise ValueError("pairwise_filter_keep_fraction must be in (0, 1]")


def token_patch_boxes(
    token_width: int,
    token_height: int,
    image_width: int,
    image_height: int,
    scale: float = 1.0,
    coordinate_mode: str = "edge",
) -> list[TokenPatchBox]:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    centers = token_grid_xy(token_width, token_height, image_width, image_height, step=1, coordinate_mode=coordinate_mode)
    if coordinate_mode == "edge":
        stride_x = float(image_width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(image_height - 1) / max(float(token_height - 1), 1.0)
    elif coordinate_mode == "center":
        stride_x = float(image_width) / float(token_width)
        stride_y = float(image_height) / float(token_height)
    else:
        raise ValueError("coordinate_mode must be one of: edge, center")
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
            projected_xy_by_track={},
        )
    projected_xy_by_track: dict[int, np.ndarray] = {}
    for landmark_idx, is_visible in enumerate(visible):
        if not bool(is_visible):
            continue
        track_id = int(index.track_ids[landmark_idx])
        visible_track_ids.add(track_id)
        xy = projected_xy[landmark_idx]
        projected_xy_by_track[track_id] = xy.astype(np.float64, copy=True)
        x_idx = int(round(np.clip(xy[0] / max(float(camera.width - 1), 1.0), 0.0, 1.0) * max(token_width - 1, 0)))
        y_idx = int(round(np.clip(xy[1] / max(float(camera.height - 1), 1.0), 0.0, 1.0) * max(token_height - 1, 0)))
        for yy in range(max(0, y_idx - 1), min(token_height, y_idx + 2)):
            for xx in range(max(0, x_idx - 1), min(token_width, x_idx + 2)):
                token_index = yy * token_width + xx
                positive = by_token[token_index]
                if positive.patch_box.contains(xy):
                    positive.track_ids.add(track_id)
    return PatchPositiveSets(
        by_token=by_token,
        stride_x_px=stride_x,
        stride_y_px=stride_y,
        visible_track_ids=visible_track_ids,
        projected_xy_by_track=projected_xy_by_track,
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


def oracle_patch_positive_matches(
    index: LandmarkMapIndex,
    positives: PatchPositiveSets,
    max_per_token: int = 1,
) -> list[QueryTo3DMatch]:
    """Build GT-correct patch-center to landmark matches for upper-bound PnP."""

    if int(max_per_token) <= 0:
        raise ValueError("max_per_token must be positive")
    if len(index) == 0:
        return []
    landmark_index_by_track = {int(track_id): idx for idx, track_id in enumerate(index.track_ids.tolist())}
    matches: list[QueryTo3DMatch] = []
    projected = positives.projected_xy_by_track or {}
    for token_index in sorted(positives.by_token):
        positive = positives.by_token[int(token_index)]
        if not positive.track_ids:
            continue

        def distance_to_center(track_id: int) -> float:
            xy = projected.get(int(track_id))
            if xy is None:
                return 0.0
            return float(np.linalg.norm(np.asarray(xy, dtype=np.float64).reshape(2) - positive.patch_box.center))

        ordered_tracks = sorted((int(track_id) for track_id in positive.track_ids), key=distance_to_center)
        for track_id in ordered_tracks[: int(max_per_token)]:
            landmark_idx = landmark_index_by_track.get(int(track_id))
            if landmark_idx is None:
                continue
            matches.append(
                QueryTo3DMatch(
                    token_index=int(token_index),
                    xy=np.asarray(positive.patch_box.center, dtype=np.float64).reshape(2),
                    track_id=int(track_id),
                    xyz=np.asarray(index.xyz[landmark_idx], dtype=np.float64).reshape(3),
                    similarity=1.0,
                    ratio=0.0,
                    landmark_variance=float(index.mean_variances[landmark_idx]),
                    source="oracle_patch_positive",
                    observation_count=int(index.observation_counts[landmark_idx]),
                    landmark_reprojection_error=None
                    if index.reprojection_errors is None
                    else float(index.reprojection_errors[landmark_idx]),
                    landmark_ambiguity=None
                    if index.feature_ambiguities is None
                    else float(index.feature_ambiguities[landmark_idx]),
                    similarity_margin=1.0,
                )
            )
    return matches


def filter_patch_correct_matches(
    matches: Sequence[QueryTo3DMatch],
    positives: PatchPositiveSets,
) -> list[QueryTo3DMatch]:
    """Keep current matches whose landmark is in that token's GT patch-positive set."""

    filtered: list[QueryTo3DMatch] = []
    for match in matches:
        positive = positives.by_token.get(int(match.token_index))
        if positive is None:
            continue
        if int(match.track_id) in positive.track_ids:
            filtered.append(match)
    return filtered


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
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    query_count = int(query_features.shape[0])
    landmark_count = int(landmark_features.shape[0])
    effective_top_k = min(int(top_k), landmark_count)
    top_indices = np.full((query_count, effective_top_k), -1, dtype=np.int64)
    top_scores = np.full((query_count, effective_top_k), -np.inf, dtype=np.float32)
    if str(device).lower() != "cpu":
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Torch is required for non-CPU patch matching similarity") from exc
        requested = str(device).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested != "cpu":
            torch_device = torch.device(requested)
            landmark_tensor = torch.as_tensor(landmark_features, dtype=torch.float32, device=torch_device).T.contiguous()
            for start in range(0, query_count, block_size):
                end = min(start + block_size, query_count)
                query_tensor = torch.as_tensor(query_features[start:end], dtype=torch.float32, device=torch_device)
                scores = torch.matmul(query_tensor, landmark_tensor)
                local_scores, local_indices = torch.topk(scores, k=effective_top_k, dim=1, largest=True, sorted=True)
                top_indices[start:end] = local_indices.detach().cpu().numpy().astype(np.int64)
                top_scores[start:end] = local_scores.detach().cpu().numpy().astype(np.float32)
            return top_indices, top_scores
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
    subset = index.subset(mask)
    return prune_ambiguous_landmarks(subset, config.landmark_ambiguity_pruning)


def _match_score(match: QueryTo3DMatch, mode: str) -> float:
    if mode == "similarity":
        return float(match.similarity)
    if mode == "landmark_quality":
        return float(match.landmark_quality if match.landmark_quality is not None else 1.0)
    if mode == "similarity_quality":
        return float(match.quality_weighted_similarity if match.quality_weighted_similarity is not None else match.similarity)
    if mode == "similarity_pairwise":
        return float(match.pairwise_weighted_similarity if match.pairwise_weighted_similarity is not None else match.similarity)
    raise ValueError("unsupported match_score_mode")


def _log_sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32)
    return (-np.logaddexp(0.0, -logits)).astype(np.float32)


def _pairwise_inlier_logits_for_topk(
    scorer: PairwiseInlierScorer | None,
    query_features: np.ndarray,
    landmark_features: np.ndarray,
    top_indices: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if scorer is None:
        return None, None
    if top_indices.size == 0:
        return (
            np.zeros(top_indices.shape, dtype=np.float32),
            np.zeros(top_indices.shape, dtype=np.float32),
        )
    query_rows = []
    landmark_rows = []
    pair_positions = []
    for query_row in range(top_indices.shape[0]):
        for rank in range(top_indices.shape[1]):
            landmark_idx = int(top_indices[query_row, rank])
            if landmark_idx < 0:
                continue
            query_rows.append(query_features[query_row])
            landmark_rows.append(landmark_features[landmark_idx])
            pair_positions.append((query_row, rank))
    logits = np.full(top_indices.shape, -np.inf, dtype=np.float32)
    logprobs = np.full(top_indices.shape, -np.inf, dtype=np.float32)
    if not pair_positions:
        return logits, logprobs
    pair_logits = np.asarray(
        scorer.score_pairs(
            np.stack(query_rows, axis=0).astype(np.float32, copy=False),
            np.stack(landmark_rows, axis=0).astype(np.float32, copy=False),
        ),
        dtype=np.float32,
    ).reshape(-1)
    if pair_logits.shape[0] != len(pair_positions):
        raise ValueError("pairwise scorer must return one logit per query-landmark pair")
    pair_logprobs = _log_sigmoid(pair_logits)
    for (query_row, rank), logit, logprob in zip(pair_positions, pair_logits, pair_logprobs):
        logits[query_row, rank] = float(logit)
        logprobs[query_row, rank] = float(logprob)
    return logits, logprobs


def _filter_matches_by_pairwise_score(
    matches: list[QueryTo3DMatch],
    keep_fraction: float | None,
    min_logit: float | None,
    min_logprob: float | None,
) -> list[QueryTo3DMatch]:
    if not matches:
        return matches
    if keep_fraction is None and min_logit is None and min_logprob is None:
        return matches
    scored = [match for match in matches if match.pairwise_inlier_logit is not None]
    if not scored:
        return matches
    keep_ids: set[int] = set()
    if keep_fraction is not None:
        keep_count = max(1, int(np.ceil(len(scored) * float(keep_fraction))))
        order = np.argsort([-float(match.pairwise_inlier_logit) for match in scored], kind="mergesort")
        keep_ids.update(id(scored[int(idx)]) for idx in order[:keep_count])
    else:
        keep_ids.update(id(match) for match in scored)
    output = []
    for match in matches:
        if match.pairwise_inlier_logit is None:
            continue
        if id(match) not in keep_ids:
            continue
        if min_logit is not None and float(match.pairwise_inlier_logit) < float(min_logit):
            continue
        if min_logprob is not None and (
            match.pairwise_inlier_logprob is None or float(match.pairwise_inlier_logprob) < float(min_logprob)
        ):
            continue
        output.append(match)
    return output


def _with_pairwise_weighted_similarity(
    matches: Sequence[QueryTo3DMatch],
    pairwise_inlier_weight: float,
) -> list[QueryTo3DMatch]:
    output: list[QueryTo3DMatch] = []
    for match in matches:
        if match.pairwise_inlier_logprob is None:
            output.append(match)
            continue
        output.append(
            replace(
                match,
                pairwise_weighted_similarity=float(
                    match.similarity + float(pairwise_inlier_weight) * float(match.pairwise_inlier_logprob)
                ),
            )
        )
    return output


def _filter_matches_by_map_reliability(
    matches: list[QueryTo3DMatch],
    config: MapReliabilityConfig,
) -> list[QueryTo3DMatch]:
    if not config.enabled:
        return matches
    return select_pnp_matches_by_map_reliability(
        matches,
        keep_fraction=config.filter_keep_fraction,
        min_score=config.min_score,
    )


def _deduplicate_token_track_matches(
    matches: Sequence[QueryTo3DMatch],
    score_mode: str,
) -> list[QueryTo3DMatch]:
    best_by_key: dict[tuple[int, int], QueryTo3DMatch] = {}
    for match in matches:
        key = (int(match.token_index), int(match.track_id))
        current = best_by_key.get(key)
        if current is None or _match_score(match, score_mode) > _match_score(current, score_mode):
            best_by_key[key] = match
    output = list(best_by_key.values())
    output.sort(key=lambda item: _match_score(item, score_mode), reverse=True)
    return output


def match_query_patches_to_landmarks(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    config: PatchTo3DMatchingConfig | None = None,
    image_width: int = 1024,
    image_height: int = 576,
    pairwise_inlier_scorer: PairwiseInlierScorer | None = None,
    pairwise_inlier_weight: float = 0.0,
    candidate_match_scorer: CandidateMatchScorer | None = None,
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
    reliability_scores = map_reliability_scores(
        index,
        landmark_features,
        config.map_reliability,
        block_size=config.block_size,
    )
    if config.landmark_quality.enabled and config.landmark_quality.min_score is not None:
        keep_quality = quality_scores >= float(config.landmark_quality.min_score)
        if not np.any(keep_quality):
            return []
        index = index.subset(keep_quality)
        landmark_features = landmark_features[keep_quality]
        quality_scores = quality_scores[keep_quality]
        landmark_ambiguity = landmark_ambiguity[keep_quality]
        reliability_scores = reliability_scores[keep_quality]
    if config.max_landmark_ambiguity is not None:
        keep_ambiguity = landmark_ambiguity <= float(config.max_landmark_ambiguity)
        if not np.any(keep_ambiguity):
            return []
        index = index.subset(keep_ambiguity)
        landmark_features = landmark_features[keep_ambiguity]
        quality_scores = quality_scores[keep_ambiguity]
        landmark_ambiguity = landmark_ambiguity[keep_ambiguity]
        reliability_scores = reliability_scores[keep_ambiguity]
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
        device=config.similarity_device,
    )
    pairwise_logits, pairwise_logprobs = _pairwise_inlier_logits_for_topk(
        pairwise_inlier_scorer,
        query_features,
        landmark_features,
        query_top_indices,
    )
    if config.match_mode in {"mnn", "soft_mutual"}:
        landmark_top_indices, _landmark_top_scores = _query_landmark_topk(
            landmark_features,
            query_features,
            top_k=config.mutual_top_k,
            block_size=config.block_size,
            device=config.similarity_device,
        )
        reciprocal = [set(row.tolist()) for row in landmark_top_indices]
    else:
        reciprocal = []

    candidate_groups: list[tuple[list[QueryTo3DMatch], bool]] = []
    for query_row in range(query_features.shape[0]):
        candidate_matches: list[QueryTo3DMatch] = []
        allow_multi_candidate_nn = config.landmark_quality.enabled or config.match_score_mode in {"similarity_pairwise"}
        for rank in range(query_top_indices.shape[1]):
            landmark_idx = int(query_top_indices[query_row, rank])
            if landmark_idx < 0:
                continue
            if config.match_mode == "nn" and rank > 0 and not (allow_multi_candidate_nn or config.return_all_topk_candidates):
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
            map_reliability = None
            pnp_uncertainty_scale = None
            if config.map_reliability.enabled:
                map_reliability = float(reliability_scores[landmark_idx])
                pnp_uncertainty_scale = map_reliability_uncertainty_scale(map_reliability, config.map_reliability)
            quality_weighted_similarity = float(similarity * quality)
            pairwise_logit = None if pairwise_logits is None else float(pairwise_logits[query_row, rank])
            pairwise_logprob = None if pairwise_logprobs is None else float(pairwise_logprobs[query_row, rank])
            pairwise_weighted_similarity = None
            if pairwise_logprob is not None:
                pairwise_weighted_similarity = float(similarity + float(pairwise_inlier_weight) * pairwise_logprob)
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
                    pairwise_inlier_logit=pairwise_logit,
                    pairwise_inlier_logprob=pairwise_logprob,
                    pairwise_weighted_similarity=pairwise_weighted_similarity,
                    similarity_margin=margin,
                    distance_to_boundary_px=float(boundary),
                    map_reliability=map_reliability,
                    pnp_uncertainty_scale=pnp_uncertainty_scale,
                    token_match_rank=int(rank),
                )
            )
        candidate_groups.append((candidate_matches, allow_multi_candidate_nn))
    if candidate_match_scorer is not None and candidate_groups:
        flat_candidates = [match for group, _allow in candidate_groups for match in group]
        rescored_flat = list(candidate_match_scorer(flat_candidates))
        if len(rescored_flat) != len(flat_candidates):
            raise ValueError("candidate_match_scorer must return one match per input candidate")
        rescored_flat = _with_pairwise_weighted_similarity(rescored_flat, pairwise_inlier_weight)
        cursor = 0
        rescored_groups: list[tuple[list[QueryTo3DMatch], bool]] = []
        for group, allow_multi_candidate_nn in candidate_groups:
            next_cursor = cursor + len(group)
            rescored_groups.append((rescored_flat[cursor:next_cursor], allow_multi_candidate_nn))
            cursor = next_cursor
        candidate_groups = rescored_groups

    matches: list[QueryTo3DMatch] = []
    for candidate_matches, allow_multi_candidate_nn in candidate_groups:
        if config.return_all_topk_candidates:
            matches.extend(candidate_matches)
        elif config.match_mode == "nn" and allow_multi_candidate_nn and candidate_matches:
            matches.append(max(candidate_matches, key=lambda item: _match_score(item, config.match_score_mode)))
        else:
            matches.extend(candidate_matches)
    matches.sort(key=lambda item: _match_score(item, config.match_score_mode), reverse=True)
    matches = _filter_matches_by_pairwise_score(
        matches,
        keep_fraction=config.pairwise_filter_keep_fraction,
        min_logit=config.pairwise_filter_min_logit,
        min_logprob=config.pairwise_filter_min_logprob,
    )
    matches = _filter_matches_by_map_reliability(matches, config.map_reliability)
    matches = filter_matches_by_local_geometric_consistency(matches, config.local_geometric_consistency)
    if bool(config.deduplicate_track_matches):
        matches = _deduplicate_token_track_matches(matches, config.match_score_mode)
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
