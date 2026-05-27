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

    def __post_init__(self) -> None:
        track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.xyz, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float32)
        mean_variances = np.asarray(self.mean_variances, dtype=np.float32).reshape(-1)
        observation_counts = np.asarray(self.observation_counts, dtype=np.int64).reshape(-1)
        if xyz.shape != (track_ids.size, 3):
            raise ValueError("xyz must have shape (N, 3)")
        if features.ndim != 2 or features.shape[0] != track_ids.size:
            raise ValueError("features must have shape (N, C)")
        if mean_variances.shape[0] != track_ids.size:
            raise ValueError("mean_variances must have shape (N,)")
        if observation_counts.shape[0] != track_ids.size:
            raise ValueError("observation_counts must have shape (N,)")
        if len(self.observation_image_ids) != track_ids.size:
            raise ValueError("observation_image_ids must have length N")
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mean_variances", mean_variances)
        object.__setattr__(self, "observation_counts", observation_counts)
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
    ) -> "LandmarkMapIndex":
        track_ids = []
        xyz = []
        features = []
        variances = []
        counts = []
        image_ids = []
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
        feature_dim = int(bank.feature_dim)
        if not track_ids:
            return cls(
                track_ids=np.zeros((0,), dtype=np.int64),
                xyz=np.zeros((0, 3), dtype=np.float64),
                features=np.zeros((0, feature_dim), dtype=np.float32),
                mean_variances=np.zeros((0,), dtype=np.float32),
                observation_counts=np.zeros((0,), dtype=np.int64),
                observation_image_ids=(),
            )
        return cls(
            track_ids=np.asarray(track_ids, dtype=np.int64),
            xyz=np.stack(xyz, axis=0),
            features=np.stack(features, axis=0),
            mean_variances=np.asarray(variances, dtype=np.float32),
            observation_counts=np.asarray(counts, dtype=np.int64),
            observation_image_ids=tuple(image_ids),
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
        )


@dataclass(frozen=True)
class QueryTo3DMatchingConfig:
    top_k: int = 2
    ratio_threshold: float | None = 0.9
    min_similarity: float = 0.0
    mutual: bool = False
    max_landmark_variance: float | None = None
    min_observation_count: int = 1
    query_token_step: int = 1
    max_matches: int | None = None
    block_size: int = 512

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.ratio_threshold is not None and not 0.0 < float(self.ratio_threshold) <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")
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
    return index.subset(mask)


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
        landmark_idx = int(top_indices[query_idx, 0])
        if landmark_idx < 0:
            continue
        similarity = float(top_scores[query_idx, 0])
        if similarity < float(config.min_similarity):
            continue
        ratio = 0.0
        if top_scores.shape[1] >= 2:
            best_distance = max(0.0, 1.0 - similarity)
            second_distance = max(1e-6, 1.0 - float(top_scores[query_idx, 1]))
            ratio = float(best_distance / second_distance)
            if config.ratio_threshold is not None and ratio > float(config.ratio_threshold):
                continue
        if config.mutual and int(landmark_best_query[landmark_idx]) != query_idx:
            continue
        matches.append(
            QueryTo3DMatch(
                token_index=int(token_indices[query_idx]),
                xy=query_xy[query_idx].astype(np.float64, copy=True),
                track_id=int(index.track_ids[landmark_idx]),
                xyz=index.xyz[landmark_idx].astype(np.float64, copy=True),
                similarity=similarity,
                ratio=ratio,
                landmark_variance=float(index.mean_variances[landmark_idx]),
            )
        )
    matches.sort(key=lambda item: item.similarity, reverse=True)
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
