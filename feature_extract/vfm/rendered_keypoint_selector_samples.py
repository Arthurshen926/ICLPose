"""Build selector training samples from query-to-render keypoint geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.patch_selector_training import PatchSelectorTrainingSet
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion, normalize_rows
from feature_extract.vfm.rendered_keypoint_matching import backproject_depth_to_world, bilinear_sample_feature_map


@dataclass(frozen=True)
class RenderedKeypointSelectorSampleConfig:
    positive_threshold_px: float = 16.0
    negative_threshold_px: float = 32.0
    max_positives_per_keypoint: int = 1
    hard_negatives_per_keypoint: int = 16
    hard_negative_pool: int = 128
    max_samples: int = 0
    keypoint_stride_px: float = 16.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.positive_threshold_px <= 0.0:
            raise ValueError("positive_threshold_px must be positive")
        if self.negative_threshold_px <= self.positive_threshold_px:
            raise ValueError("negative_threshold_px must be greater than positive_threshold_px")
        if self.max_positives_per_keypoint <= 0:
            raise ValueError("max_positives_per_keypoint must be positive")
        if self.hard_negatives_per_keypoint <= 0:
            raise ValueError("hard_negatives_per_keypoint must be positive")
        if self.hard_negative_pool <= 0:
            raise ValueError("hard_negative_pool must be positive")
        if self.max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        if self.keypoint_stride_px <= 0.0:
            raise ValueError("keypoint_stride_px must be positive")


def render_keypoints_to_world(
    render_xy: np.ndarray,
    rendered_depth: np.ndarray,
    render_camera: ColmapCamera,
    render_pose_w2c: np.ndarray,
    *,
    render_width: int,
    render_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_values, depth_valid = bilinear_sample_feature_map(
        np.asarray(rendered_depth, dtype=np.float32).reshape(1, *np.asarray(rendered_depth).shape[-2:]),
        render_xy,
        image_width=int(render_width),
        image_height=int(render_height),
    )
    xyz, xyz_valid = backproject_depth_to_world(
        np.asarray(render_xy, dtype=np.float64).reshape(-1, 2),
        depth_values[:, 0],
        render_camera,
        render_pose_w2c,
    )
    return xyz, depth_valid & xyz_valid


def _project_world_to_image(
    xyz: np.ndarray,
    camera: ColmapCamera,
    pose_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for keypoint reprojection labels") from exc
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jac = cv2.Rodrigues(pose[:3, :3])
    projected, _jac = cv2.projectPoints(points, rvec, pose[:3, 3], camera_matrix, distortion)
    cam_xyz = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    valid = np.isfinite(projected).all(axis=(1, 2)) & (cam_xyz[:, 2] > 1e-6)
    return projected.reshape(-1, 2).astype(np.float64), valid


def render_keypoint_reprojection_errors(
    query_xy: np.ndarray,
    render_xy: np.ndarray,
    *,
    rendered_depth: np.ndarray,
    render_camera: ColmapCamera,
    query_camera: ColmapCamera,
    render_pose_w2c: np.ndarray,
    query_pose_w2c: np.ndarray,
    render_width: int,
    render_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return QxR GT reprojection errors for render keypoints in the query image."""

    qxy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    rxy = np.asarray(render_xy, dtype=np.float64).reshape(-1, 2)
    if qxy.shape[0] == 0 or rxy.shape[0] == 0:
        return np.zeros((qxy.shape[0], rxy.shape[0]), dtype=np.float32), np.zeros((rxy.shape[0],), dtype=bool)
    xyz, depth_valid = render_keypoints_to_world(
        rxy,
        rendered_depth,
        render_camera,
        render_pose_w2c,
        render_width=int(render_width),
        render_height=int(render_height),
    )
    projected, proj_valid = _project_world_to_image(xyz, query_camera, query_pose_w2c)
    render_valid = depth_valid & proj_valid
    errors = np.linalg.norm(qxy[:, None, :] - projected[None, :, :], axis=2).astype(np.float32)
    errors[:, ~render_valid] = np.inf
    return errors, render_valid


def _empty_samples(input_dim: int, config: RenderedKeypointSelectorSampleConfig, metadata: dict[str, object]) -> PatchSelectorTrainingSet:
    return PatchSelectorTrainingSet(
        query_features=np.zeros((0, input_dim), dtype=np.float32),
        positive_features=np.zeros((0, config.max_positives_per_keypoint, input_dim), dtype=np.float32),
        positive_mask=np.zeros((0, config.max_positives_per_keypoint), dtype=bool),
        negative_features=np.zeros((0, config.hard_negatives_per_keypoint, input_dim), dtype=np.float32),
        positive_reprojection_distances=np.zeros((0, config.max_positives_per_keypoint), dtype=np.float32),
        negative_reprojection_distances=np.zeros((0, config.hard_negatives_per_keypoint), dtype=np.float32),
        metadata=metadata,
    )


def build_rendered_keypoint_selector_samples(
    query_descriptors: np.ndarray,
    render_descriptors: np.ndarray,
    reprojection_errors_px: np.ndarray,
    config: RenderedKeypointSelectorSampleConfig | None = None,
) -> PatchSelectorTrainingSet:
    """Mine positive/hard-negative descriptor samples from query/render keypoints."""

    cfg = config or RenderedKeypointSelectorSampleConfig()
    query = np.asarray(query_descriptors, dtype=np.float32)
    render = np.asarray(render_descriptors, dtype=np.float32)
    errors = np.asarray(reprojection_errors_px, dtype=np.float32)
    if query.ndim != 2 or render.ndim != 2:
        raise ValueError("query_descriptors and render_descriptors must have shape (N, C)")
    if query.shape[1] != render.shape[1]:
        raise ValueError("query and render descriptor dimensions must match")
    if errors.shape != (query.shape[0], render.shape[0]):
        raise ValueError("reprojection_errors_px must have shape (Q, R)")
    if query.shape[0] == 0 or render.shape[0] == 0:
        return _empty_samples(query.shape[1] if query.ndim == 2 else 0, cfg, {"sample_count": 0})

    query_norm, query_valid = normalize_rows(query)
    render_norm, render_valid = normalize_rows(render)
    scores = (query_norm @ render_norm.T).astype(np.float32, copy=False)
    scores[:, ~render_valid] = -np.inf
    scores[~query_valid, :] = -np.inf
    rng = np.random.default_rng(int(cfg.seed))
    rows: list[tuple[int, list[int], list[int]]] = []
    for q_idx in range(query.shape[0]):
        if not bool(query_valid[q_idx]):
            continue
        positives = np.flatnonzero(np.isfinite(errors[q_idx]) & (errors[q_idx] <= float(cfg.positive_threshold_px)))
        negatives = np.flatnonzero(np.isfinite(errors[q_idx]) & (errors[q_idx] >= float(cfg.negative_threshold_px)) & render_valid)
        if positives.size == 0 or negatives.size < int(cfg.hard_negatives_per_keypoint):
            continue
        pos_order = positives[np.argsort(errors[q_idx, positives])]
        pos = [int(idx) for idx in pos_order[: int(cfg.max_positives_per_keypoint)].tolist()]
        neg_scores = scores[q_idx, negatives]
        finite_neg = np.isfinite(neg_scores)
        negatives = negatives[finite_neg]
        neg_scores = neg_scores[finite_neg]
        if negatives.size < int(cfg.hard_negatives_per_keypoint):
            continue
        pool = min(int(cfg.hard_negative_pool), int(negatives.size))
        if pool == int(negatives.size):
            hard_order = negatives[np.argsort(-neg_scores)]
        else:
            partial = np.argpartition(-neg_scores, kth=pool - 1)[:pool]
            hard_order = negatives[partial[np.argsort(-neg_scores[partial])]]
        neg = [int(idx) for idx in hard_order[: int(cfg.hard_negatives_per_keypoint)].tolist()]
        rows.append((int(q_idx), pos, neg))
    if cfg.max_samples > 0 and len(rows) > int(cfg.max_samples):
        keep = np.sort(rng.choice(len(rows), size=int(cfg.max_samples), replace=False))
        rows = [rows[int(idx)] for idx in keep.tolist()]
    if not rows:
        return _empty_samples(
            query.shape[1],
            cfg,
            {
                "sample_count": 0,
                "query_keypoint_count": int(query.shape[0]),
                "render_keypoint_count": int(render.shape[0]),
            },
        )

    positive_features = np.zeros((len(rows), cfg.max_positives_per_keypoint, query.shape[1]), dtype=np.float32)
    positive_mask = np.zeros((len(rows), cfg.max_positives_per_keypoint), dtype=bool)
    positive_distances = np.zeros((len(rows), cfg.max_positives_per_keypoint), dtype=np.float32)
    negative_features = np.zeros((len(rows), cfg.hard_negatives_per_keypoint, query.shape[1]), dtype=np.float32)
    negative_distances = np.zeros((len(rows), cfg.hard_negatives_per_keypoint), dtype=np.float32)
    query_features = np.zeros((len(rows), query.shape[1]), dtype=np.float32)
    for row_idx, (q_idx, pos, neg) in enumerate(rows):
        query_features[row_idx] = query[q_idx]
        for pos_idx, r_idx in enumerate(pos):
            positive_features[row_idx, pos_idx] = render[r_idx]
            positive_mask[row_idx, pos_idx] = True
            positive_distances[row_idx, pos_idx] = float(errors[q_idx, r_idx]) / float(cfg.keypoint_stride_px)
        negative_features[row_idx] = render[np.asarray(neg, dtype=np.int64)]
        negative_distances[row_idx] = errors[q_idx, np.asarray(neg, dtype=np.int64)] / float(cfg.keypoint_stride_px)
    return PatchSelectorTrainingSet(
        query_features=query_features,
        positive_features=positive_features,
        positive_mask=positive_mask,
        negative_features=negative_features,
        positive_reprojection_distances=positive_distances,
        negative_reprojection_distances=negative_distances,
        metadata={
            "sample_count": int(len(rows)),
            "query_keypoint_count": int(query.shape[0]),
            "render_keypoint_count": int(render.shape[0]),
            "positive_threshold_px": float(cfg.positive_threshold_px),
            "negative_threshold_px": float(cfg.negative_threshold_px),
            "hard_negatives_per_keypoint": int(cfg.hard_negatives_per_keypoint),
            "max_positives_per_keypoint": int(cfg.max_positives_per_keypoint),
            "keypoint_stride_px": float(cfg.keypoint_stride_px),
        },
    )
