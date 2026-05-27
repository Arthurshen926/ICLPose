"""Visualization helpers for query-token to 3D landmark VFM matches."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    QueryTo3DMatch,
    camera_matrix_and_distortion,
)


def project_match_landmarks(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> np.ndarray:
    if not matches:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    object_points = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(object_points, rvec, pose[:3, 3], camera_matrix, distortion)
    return projected.reshape(-1, 2).astype(np.float64)


def _draw_cross(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], radius: int = 5) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    cv2.line(image, (x - radius, y), (x + radius, y), color, 1, lineType=cv2.LINE_AA)
    cv2.line(image, (x, y - radius), (x, y + radius), color, 1, lineType=cv2.LINE_AA)


def _clip_xy(xy: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(round(float(np.clip(xy[0], 0.0, max(width - 1, 0)))))
    y = int(round(float(np.clip(xy[1], 0.0, max(height - 1, 0)))))
    return x, y


def _project_xyz_array(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_points = (pose @ np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1).T).T
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(points, rvec, pose[:3, 3], camera_matrix, distortion)
    projected = projected.reshape(-1, 2).astype(np.float64)
    visible = (
        (camera_points[:, 2] > 1e-6)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(width - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(height - 1))
    )
    return projected, visible


def _fit_pca_projection(features: np.ndarray, sample_size: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    if values.shape[0] == 0:
        return (
            np.zeros((values.shape[1],), dtype=np.float32),
            np.zeros((values.shape[1], 3), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
            np.ones((3,), dtype=np.float32),
        )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-6)
    if values.shape[0] > sample_size:
        sample_idx = np.linspace(0, values.shape[0] - 1, sample_size).round().astype(np.int64)
        sample = values[sample_idx]
    else:
        sample = values
    mean = sample.mean(axis=0).astype(np.float32)
    centered = sample - mean[None, :]
    _u, _s, vt = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    components = np.zeros((values.shape[1], 3), dtype=np.float32)
    keep = min(3, vt.shape[0])
    if keep:
        components[:, :keep] = vt[:keep].T.astype(np.float32)
    projected = (values - mean[None, :]) @ components
    low = np.percentile(projected, 1.0, axis=0).astype(np.float32)
    high = np.percentile(projected, 99.0, axis=0).astype(np.float32)
    high = np.where(np.abs(high - low) < 1e-6, low + 1.0, high).astype(np.float32)
    return mean, components, low, high


def _pca_colors(
    features: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    if values.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-6)
    projected = (values - mean[None, :]) @ components
    scaled = (projected - low[None, :]) / np.maximum(high[None, :] - low[None, :], 1e-6)
    return np.asarray(np.clip(scaled, 0.0, 1.0) * 255.0, dtype=np.uint8)


def _query_feature_pca_image(
    query_feature_map: np.ndarray,
    image_shape: tuple[int, int],
    mean: np.ndarray,
    components: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C, H, W)")
    channels, token_height, token_width = feature_map.shape
    flat = feature_map.reshape(channels, token_height * token_width).T
    colors = _pca_colors(flat, mean, components, low, high).reshape(token_height, token_width, 3)
    height, width = image_shape
    return cv2.resize(colors, (width, height), interpolation=cv2.INTER_NEAREST)


def _draw_label(image: np.ndarray, label: str) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 520), 32), (0, 0, 0), -1)
    cv2.putText(image, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def render_query_to_projected_map_correspondence(
    image_rgb: np.ndarray,
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    mode: str = "rgb",
    inlier_mask: np.ndarray | None = None,
    reprojection_threshold_px: float = 16.0,
    max_draw: int = 160,
) -> tuple[np.ndarray, dict[str, object]]:
    """Render query view beside a projected 3D VFM landmark map.

    `mode="rgb"` shows the query RGB on the left and a sparse 3D projection on
    the right. `mode="feature"` shows query feature-PCA on the left and 3D
    landmark feature-PCA on the right, using one PCA basis for both sides.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if mode not in {"rgb", "feature"}:
        raise ValueError("mode must be 'rgb' or 'feature'")
    height, width = image.shape[:2]
    query_feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if query_feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C, H, W)")

    query_flat = query_feature_map.reshape(query_feature_map.shape[0], -1).T
    combined = np.concatenate([query_flat, landmark_index.features], axis=0)
    mean, components, low, high = _fit_pca_projection(combined)
    projected_xy, visible = _project_xyz_array(landmark_index.xyz, pose_w2c, camera, width, height)
    projected_count = int(np.sum(visible))
    map_canvas = np.zeros_like(image)
    map_colors = _pca_colors(landmark_index.features, mean, components, low, high)
    for idx in np.flatnonzero(visible):
        px, py = _clip_xy(projected_xy[idx], width, height)
        color = tuple(int(v) for v in map_colors[idx].tolist())
        cv2.circle(map_canvas, (px, py), 2, color, -1, lineType=cv2.LINE_AA)

    if mode == "rgb":
        left = image.copy()
    else:
        left = _query_feature_pca_image(query_feature_map, (height, width), mean, components, low, high)
    right = map_canvas
    _draw_label(left, "query RGB" if mode == "rgb" else "query VFM feature PCA")
    _draw_label(right, "projected 3D VFM landmark map")

    gap = 48
    canvas = np.zeros((height, width * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :width] = left
    canvas[:, width + gap :] = right
    canvas[:, width : width + gap] = 18

    if not matches:
        return canvas, {
            "mode": mode,
            "match_count": 0,
            "drawn_match_count": 0,
            "projected_landmark_count": projected_count,
            "gt_precision": 0.0,
            "pnp_inlier_count": 0,
        }

    track_to_index = {int(track_id): idx for idx, track_id in enumerate(landmark_index.track_ids.tolist())}
    match_projected = []
    valid_matches = []
    for match in matches:
        idx = track_to_index.get(int(match.track_id))
        if idx is None:
            continue
        match_projected.append(projected_xy[idx])
        valid_matches.append(match)
    if not valid_matches:
        return canvas, {
            "mode": mode,
            "match_count": len(matches),
            "drawn_match_count": 0,
            "projected_landmark_count": projected_count,
            "gt_precision": 0.0,
            "pnp_inlier_count": 0,
        }
    match_projected_array = np.stack(match_projected, axis=0)
    query_xy = np.stack([match.xy for match in valid_matches], axis=0).astype(np.float64)
    errors = np.linalg.norm(query_xy - match_projected_array, axis=1)
    gt_inlier_mask = errors <= float(reprojection_threshold_px)
    draw_count = min(int(max_draw), len(valid_matches))
    pnp_mask = None
    if inlier_mask is not None:
        source_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if source_mask.shape[0] == len(matches):
            pnp_mask = source_mask[: len(valid_matches)]
        elif source_mask.shape[0] == len(valid_matches):
            pnp_mask = source_mask
        else:
            raise ValueError("inlier_mask must have one value per match")
    for local_idx in range(draw_count):
        qx, qy = _clip_xy(query_xy[local_idx], width, height)
        px, py = _clip_xy(match_projected_array[local_idx], width, height)
        is_gt_inlier = bool(gt_inlier_mask[local_idx])
        color = (30, 210, 70) if is_gt_inlier else (235, 45, 45)
        right_point = (width + gap + px, py)
        cv2.line(canvas, (qx, qy), right_point, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (qx, qy), 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, right_point, 4, color, -1, lineType=cv2.LINE_AA)
        if pnp_mask is not None and bool(pnp_mask[local_idx]):
            cv2.circle(canvas, (qx, qy), 7, (255, 220, 40), 1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, right_point, 7, (255, 220, 40), 1, lineType=cv2.LINE_AA)

    panel = canvas.copy()
    cv2.rectangle(panel, (width - 260, 8), (width + gap + 420, 112), (0, 0, 0), -1)
    canvas = cv2.addWeighted(panel, 0.35, canvas, 0.65, 0.0)
    pnp_inliers = 0 if pnp_mask is None else int(np.sum(pnp_mask))
    info_lines = [
        f"matches: {len(matches)} drawn: {draw_count} projected landmarks: {projected_count}",
        f"GT precision@{reprojection_threshold_px:g}px: {float(np.mean(gt_inlier_mask)):.3f}",
        f"mean reproj error: {float(np.mean(errors)):.1f}px",
        f"yellow rings: PnP inliers {pnp_inliers}",
    ]
    y = 30
    for line in info_lines:
        cv2.putText(canvas, line, (width - 246, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 19

    return canvas, {
        "mode": mode,
        "match_count": int(len(matches)),
        "drawn_match_count": int(draw_count),
        "projected_landmark_count": int(projected_count),
        "gt_inlier_count": int(np.sum(gt_inlier_mask)),
        "gt_precision": float(np.mean(gt_inlier_mask)),
        "pnp_inlier_count": int(pnp_inliers),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "median_reprojection_error_px": float(np.median(errors)),
    }


def render_query_to_3d_match_overlay(
    image_rgb: np.ndarray,
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    inlier_mask: np.ndarray | None = None,
    reprojection_threshold_px: float = 16.0,
    max_draw: int = 250,
) -> tuple[np.ndarray, dict[str, object]]:
    """Render query-token positions linked to GT-projected 3D landmarks.

    Green lines are geometrically correct under `pose_w2c`; red lines are false
    matches under the same threshold. PnP inliers are additionally marked with a
    yellow ring around the query token.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for match visualization") from exc
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    overlay = image.copy()
    if not matches:
        return overlay, {
            "match_count": 0,
            "drawn_match_count": 0,
            "gt_inlier_count": 0,
            "gt_precision": 0.0,
            "pnp_inlier_count": 0,
            "mean_reprojection_error_px": None,
        }

    projected = project_match_landmarks(matches, pose_w2c=pose_w2c, camera=camera)
    query_xy = np.stack([match.xy for match in matches], axis=0).astype(np.float64)
    errors = np.linalg.norm(query_xy - projected, axis=1)
    gt_inlier_mask = errors <= float(reprojection_threshold_px)
    draw_count = min(int(max_draw), len(matches))
    # Matches are already ordered by descriptor confidence. Draw that order so
    # the visualization exposes the matcher's accepted high-confidence evidence
    # instead of cherry-picking the geometrically best correspondences.
    draw_indices = np.arange(draw_count, dtype=np.int64)
    pnp_mask = None
    if inlier_mask is not None:
        pnp_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if pnp_mask.shape[0] != len(matches):
            raise ValueError("inlier_mask must have one value per match")

    height, width = overlay.shape[:2]
    for idx in draw_indices:
        qx, qy = _clip_xy(query_xy[idx], width, height)
        px, py = _clip_xy(projected[idx], width, height)
        is_gt_inlier = bool(gt_inlier_mask[idx])
        color = (30, 180, 60) if is_gt_inlier else (220, 40, 40)
        cv2.line(overlay, (qx, qy), (px, py), color, 1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (qx, qy), 3, color, -1, lineType=cv2.LINE_AA)
        _draw_cross(overlay, px, py, color, radius=4)
        if pnp_mask is not None and bool(pnp_mask[idx]):
            cv2.circle(overlay, (qx, qy), 6, (255, 210, 40), 1, lineType=cv2.LINE_AA)

    panel = overlay.copy()
    cv2.rectangle(panel, (8, 8), (430, 112), (0, 0, 0), -1)
    overlay = cv2.addWeighted(panel, 0.45, overlay, 0.55, 0.0)
    pnp_inliers = 0 if pnp_mask is None else int(np.sum(pnp_mask))
    lines = [
        f"matches: {len(matches)}   drawn: {draw_count}",
        f"GT precision@{reprojection_threshold_px:g}px: {float(np.mean(gt_inlier_mask)):.3f}",
        f"mean reproj error: {float(np.mean(errors)):.1f}px",
        f"PnP inliers: {pnp_inliers}",
        "green: GT-consistent   red: false   yellow ring: PnP inlier",
    ]
    y = 28
    for line in lines:
        cv2.putText(overlay, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    summary = {
        "match_count": int(len(matches)),
        "drawn_match_count": int(draw_count),
        "gt_inlier_count": int(np.sum(gt_inlier_mask)),
        "gt_precision": float(np.mean(gt_inlier_mask)),
        "pnp_inlier_count": int(pnp_inliers),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "median_reprojection_error_px": float(np.median(errors)),
    }
    return overlay, summary
