"""Sparse keypoint matching with rendered dense VFM features and depth."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch, camera_matrix_and_distortion, normalize_rows


@dataclass(frozen=True)
class KeypointFeatureMatch:
    query_index: int
    render_index: int
    query_xy: np.ndarray
    render_xy: np.ndarray
    similarity: float
    ratio: float
    similarity_margin: float | None = None
    dual_softmax_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_xy", np.asarray(self.query_xy, dtype=np.float64).reshape(2))
        object.__setattr__(self, "render_xy", np.asarray(self.render_xy, dtype=np.float64).reshape(2))


def bilinear_sample_feature_map(
    feature_map: np.ndarray,
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a CHW feature map at image-coordinate keypoints."""

    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    channels, height, width = fmap.shape
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image_width and image_height must be positive")
    valid = (
        np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
        & (coords[:, 0] >= 0.0)
        & (coords[:, 0] <= float(image_width - 1))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] <= float(image_height - 1))
    )
    if coords.shape[0] == 0:
        return np.zeros((0, channels), dtype=np.float32), valid
    gx = coords[:, 0] / max(float(image_width - 1), 1.0) * max(float(width - 1), 0.0)
    gy = coords[:, 1] / max(float(image_height - 1), 1.0) * max(float(height - 1), 0.0)
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, max(width - 1, 0))
    y1 = np.clip(y0 + 1, 0, max(height - 1, 0))
    x0 = np.clip(x0, 0, max(width - 1, 0))
    y0 = np.clip(y0, 0, max(height - 1, 0))
    wx = (gx - x0.astype(np.float64)).astype(np.float32)
    wy = (gy - y0.astype(np.float64)).astype(np.float32)
    flat = fmap.reshape(channels, -1).T
    idx00 = y0 * width + x0
    idx10 = y0 * width + x1
    idx01 = y1 * width + x0
    idx11 = y1 * width + x1
    samples = (
        flat[idx00] * ((1.0 - wx) * (1.0 - wy))[:, None]
        + flat[idx10] * (wx * (1.0 - wy))[:, None]
        + flat[idx01] * ((1.0 - wx) * wy)[:, None]
        + flat[idx11] * (wx * wy)[:, None]
    ).astype(np.float32, copy=False)
    samples[~valid] = 0.0
    finite = np.isfinite(samples).all(axis=1)
    return samples, valid & finite


def _top2(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape[1] == 1:
        return np.zeros((scores.shape[0], 1), dtype=np.int64), scores
    order = np.argpartition(-scores, kth=1, axis=1)[:, :2]
    local = np.take_along_axis(scores, order, axis=1)
    sort_order = np.argsort(-local, axis=1)
    return np.take_along_axis(order, sort_order, axis=1), np.take_along_axis(local, sort_order, axis=1)


def _softmax_stable(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-12)


def _dual_softmax_confidence(scores: np.ndarray, logit_scale: float) -> np.ndarray:
    logits = np.asarray(scores, dtype=np.float32) * float(logit_scale)
    return _softmax_stable(logits, axis=1) * _softmax_stable(logits, axis=0)


def dual_softmax_keypoint_matches(
    query_xy: np.ndarray,
    query_descriptors: np.ndarray,
    render_xy: np.ndarray,
    render_descriptors: np.ndarray,
    *,
    logit_scale: float = 10.0,
    min_confidence: float = 0.0,
    min_similarity: float = 0.0,
    max_matches: int | None = None,
) -> list[KeypointFeatureMatch]:
    """Match descriptors using MATCHA-style row/column softmax confidence."""

    qxy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    rxy = np.asarray(render_xy, dtype=np.float64).reshape(-1, 2)
    qdesc = np.asarray(query_descriptors, dtype=np.float32)
    rdesc = np.asarray(render_descriptors, dtype=np.float32)
    if qdesc.ndim != 2 or rdesc.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    if qdesc.shape[0] != qxy.shape[0] or rdesc.shape[0] != rxy.shape[0]:
        raise ValueError("descriptor and keypoint counts must match")
    if qdesc.shape[1] != rdesc.shape[1]:
        raise ValueError("query and render descriptor dimensions must match")
    if qdesc.shape[0] == 0 or rdesc.shape[0] == 0:
        return []
    qdesc, qvalid = normalize_rows(qdesc)
    rdesc, rvalid = normalize_rows(rdesc)
    qrows = np.flatnonzero(qvalid)
    rrows = np.flatnonzero(rvalid)
    if qrows.size == 0 or rrows.size == 0:
        return []
    qdesc = qdesc[qrows]
    rdesc = rdesc[rrows]
    scores = qdesc @ rdesc.T
    confidence = _dual_softmax_confidence(scores, float(logit_scale))
    query_best_render = np.argmax(confidence, axis=1)
    render_best_query = np.argmax(confidence, axis=0)
    top_indices, top_scores = _top2(scores)
    matches: list[KeypointFeatureMatch] = []
    for local_q in range(qdesc.shape[0]):
        local_r = int(query_best_render[local_q])
        if int(render_best_query[local_r]) != local_q:
            continue
        conf = float(confidence[local_q, local_r])
        if conf < float(min_confidence):
            continue
        similarity = float(scores[local_q, local_r])
        if similarity < float(min_similarity):
            continue
        ratio = 0.0
        margin = None
        if top_scores.shape[1] >= 2:
            second_similarity = float(top_scores[local_q, 1])
            best_distance = max(0.0, 1.0 - similarity)
            second_distance = max(1e-6, 1.0 - second_similarity)
            ratio = float(best_distance / second_distance)
            margin = float(similarity - second_similarity)
        matches.append(
            KeypointFeatureMatch(
                query_index=int(qrows[local_q]),
                render_index=int(rrows[local_r]),
                query_xy=qxy[int(qrows[local_q])],
                render_xy=rxy[int(rrows[local_r])],
                similarity=similarity,
                ratio=ratio,
                similarity_margin=margin,
                dual_softmax_confidence=conf,
            )
        )
    matches.sort(
        key=lambda item: (
            float(item.dual_softmax_confidence or 0.0),
            float(item.similarity),
        ),
        reverse=True,
    )
    if max_matches is not None:
        matches = matches[: int(max_matches)]
    return matches


def refine_render_keypoint_matches_by_local_correlation(
    matches: Sequence[KeypointFeatureMatch],
    query_descriptors: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    search_radius_px: float = 4.0,
    step_px: float = 1.0,
) -> list[KeypointFeatureMatch]:
    """Refine render-side keypoint measurements by local descriptor correlation."""

    if not matches:
        return []
    radius = float(search_radius_px)
    step = float(step_px)
    if radius <= 0.0:
        return list(matches)
    if step <= 0.0:
        raise ValueError("step_px must be positive")
    qdesc = np.asarray(query_descriptors, dtype=np.float32)
    if qdesc.ndim != 2:
        raise ValueError("query_descriptors must have shape (N, C)")
    qdesc_norm, qvalid = normalize_rows(qdesc)
    offsets_1d = np.arange(-radius, radius + step * 0.5, step, dtype=np.float64)
    dx, dy = np.meshgrid(offsets_1d, offsets_1d, indexing="xy")
    offsets = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=1)
    refined: list[KeypointFeatureMatch] = []
    for match in matches:
        qidx = int(match.query_index)
        if qidx < 0 or qidx >= qdesc_norm.shape[0] or not bool(qvalid[qidx]):
            refined.append(match)
            continue
        candidates = np.asarray(match.render_xy, dtype=np.float64).reshape(1, 2) + offsets
        samples, valid = bilinear_sample_feature_map(
            render_feature_map,
            candidates,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        if samples.shape[0] == 0 or not np.any(valid):
            refined.append(match)
            continue
        samples_norm, sample_valid = normalize_rows(samples)
        valid = valid & sample_valid
        if not np.any(valid):
            refined.append(match)
            continue
        scores = samples_norm[valid] @ qdesc_norm[qidx]
        valid_candidates = candidates[valid]
        best_score = float(np.max(scores))
        tie = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-6, atol=1e-8))
        if tie.size > 1:
            distances = np.linalg.norm(valid_candidates[tie] - np.asarray(match.render_xy, dtype=np.float64).reshape(1, 2), axis=1)
            best = int(tie[int(np.argmin(distances))])
        else:
            best = int(tie[0])
        refined.append(
            replace(
                match,
                render_xy=valid_candidates[best].astype(np.float64, copy=False),
                similarity=float(scores[best]),
            )
        )
    return refined


def mutual_nn_keypoint_matches(
    query_xy: np.ndarray,
    query_descriptors: np.ndarray,
    render_xy: np.ndarray,
    render_descriptors: np.ndarray,
    *,
    ratio_threshold: float | None = 0.9,
    min_similarity: float = 0.0,
    dual_softmax_logit_scale: float | None = None,
    min_dual_softmax_confidence: float = 0.0,
    max_matches: int | None = None,
) -> list[KeypointFeatureMatch]:
    """Match keypoint descriptors by cosine mutual nearest neighbor."""

    qxy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    rxy = np.asarray(render_xy, dtype=np.float64).reshape(-1, 2)
    qdesc = np.asarray(query_descriptors, dtype=np.float32)
    rdesc = np.asarray(render_descriptors, dtype=np.float32)
    if qdesc.ndim != 2 or rdesc.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    if qdesc.shape[0] != qxy.shape[0] or rdesc.shape[0] != rxy.shape[0]:
        raise ValueError("descriptor and keypoint counts must match")
    if qdesc.shape[1] != rdesc.shape[1]:
        raise ValueError("query and render descriptor dimensions must match")
    if qdesc.shape[0] == 0 or rdesc.shape[0] == 0:
        return []
    qdesc, qvalid = normalize_rows(qdesc)
    rdesc, rvalid = normalize_rows(rdesc)
    qrows = np.flatnonzero(qvalid)
    rrows = np.flatnonzero(rvalid)
    if qrows.size == 0 or rrows.size == 0:
        return []
    qdesc = qdesc[qrows]
    rdesc = rdesc[rrows]
    scores = qdesc @ rdesc.T
    confidence = (
        _dual_softmax_confidence(scores, float(dual_softmax_logit_scale))
        if dual_softmax_logit_scale is not None
        else None
    )
    top_indices, top_scores = _top2(scores)
    render_best_query = np.argmax(scores, axis=0)
    matches: list[KeypointFeatureMatch] = []
    for local_q in range(qdesc.shape[0]):
        local_r = int(top_indices[local_q, 0])
        similarity = float(top_scores[local_q, 0])
        if similarity < float(min_similarity):
            continue
        if int(render_best_query[local_r]) != local_q:
            continue
        conf = None if confidence is None else float(confidence[local_q, local_r])
        if conf is not None and conf < float(min_dual_softmax_confidence):
            continue
        ratio = 0.0
        margin = None
        if top_scores.shape[1] >= 2:
            second_similarity = float(top_scores[local_q, 1])
            best_distance = max(0.0, 1.0 - similarity)
            second_distance = max(1e-6, 1.0 - second_similarity)
            ratio = float(best_distance / second_distance)
            margin = float(similarity - second_similarity)
            if ratio_threshold is not None and ratio > float(ratio_threshold):
                continue
        matches.append(
            KeypointFeatureMatch(
                query_index=int(qrows[local_q]),
                render_index=int(rrows[local_r]),
                query_xy=qxy[int(qrows[local_q])],
                render_xy=rxy[int(rrows[local_r])],
                similarity=similarity,
                ratio=ratio,
                similarity_margin=margin,
                dual_softmax_confidence=conf,
            )
        )
    matches.sort(
        key=lambda item: (
            float(item.dual_softmax_confidence if item.dual_softmax_confidence is not None else item.similarity),
            float(item.similarity),
        ),
        reverse=True,
    )
    if max_matches is not None:
        matches = matches[: int(max_matches)]
    return matches


def _sample_scalar_map(values: np.ndarray, xy: np.ndarray, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    sampled, valid = bilinear_sample_feature_map(
        np.asarray(values, dtype=np.float32).reshape(1, *np.asarray(values).shape[-2:]),
        xy,
        image_width=image_width,
        image_height=image_height,
    )
    return sampled[:, 0].astype(np.float64), valid


def _coarse_cell_centers_from_indices(
    indices: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    grid_w = int(grid_width)
    grid_h = int(grid_height)
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("render grid dimensions must be positive")
    col = np.clip(idx % grid_w, 0, grid_w - 1)
    row = np.clip(idx // grid_w, 0, grid_h - 1)
    x = (col.astype(np.float64) + 0.5) * float(image_width) / float(grid_w)
    y = (row.astype(np.float64) + 0.5) * float(image_height) / float(grid_h)
    return np.stack([x, y], axis=1)


def backproject_depth_to_world(
    xy: np.ndarray,
    depth: np.ndarray,
    camera: ColmapCamera,
    pose_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject image pixels and metric camera depth into world coordinates."""

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    depth_values = np.asarray(depth, dtype=np.float64).reshape(-1)
    if depth_values.shape[0] != coords.shape[0]:
        raise ValueError("depth must contain one value per keypoint")
    matrix, distortion = camera_matrix_and_distortion(camera)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for depth backprojection") from exc
    points = cv2.undistortPoints(coords.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
    cam_xyz = np.stack((points[:, 0] * depth_values, points[:, 1] * depth_values, depth_values), axis=1)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    world = (pose[:3, :3].T @ (cam_xyz - pose[:3, 3]).T).T
    valid = np.isfinite(world).all(axis=1) & np.isfinite(depth_values) & (depth_values > 1e-6)
    return world.astype(np.float64), valid.astype(bool)


def keypoint_feature_matches_to_pnp_matches(
    matches: Sequence[KeypointFeatureMatch],
    rendered_depth: np.ndarray,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    rendered_alpha: np.ndarray | None = None,
    render_grid_width: int | None = None,
    render_grid_height: int | None = None,
    min_render_alpha: float = 0.0,
    max_render_depth_delta_m: float | None = None,
    fallback_to_cell_center: bool = False,
) -> list[QueryTo3DMatch]:
    """Convert render-keypoint matches into query 2D to world 3D PnP matches."""

    if not matches:
        return []
    depth_map = np.asarray(rendered_depth, dtype=np.float32)
    if depth_map.ndim != 2:
        raise ValueError("rendered_depth must have shape (H, W)")
    width = int(image_width) if image_width is not None else int(depth_map.shape[1])
    height = int(image_height) if image_height is not None else int(depth_map.shape[0])
    render_xy = np.stack([match.render_xy for match in matches], axis=0).astype(np.float64)
    depth_values, depth_valid = _sample_scalar_map(depth_map, render_xy, width, height)
    alpha_values = np.ones((render_xy.shape[0],), dtype=np.float64)
    alpha_valid = np.ones((render_xy.shape[0],), dtype=bool)
    if rendered_alpha is not None:
        alpha_values, alpha_valid = _sample_scalar_map(np.asarray(rendered_alpha, dtype=np.float32), render_xy, width, height)
    guard_valid = depth_valid & alpha_valid & (alpha_values >= float(min_render_alpha))
    center_xy = None
    center_depth = None
    center_alpha = None
    center_valid = None
    has_grid = render_grid_width is not None and render_grid_height is not None
    if max_render_depth_delta_m is not None and float(max_render_depth_delta_m) >= 0.0 and not has_grid:
        raise ValueError("max_render_depth_delta_m requires render_grid_width and render_grid_height")
    if has_grid:
        center_xy = _coarse_cell_centers_from_indices(
            np.asarray([match.render_index for match in matches], dtype=np.int64),
            image_width=width,
            image_height=height,
            grid_width=int(render_grid_width),
            grid_height=int(render_grid_height),
        )
        center_depth, center_depth_valid = _sample_scalar_map(depth_map, center_xy, width, height)
        center_alpha = np.ones((render_xy.shape[0],), dtype=np.float64)
        center_alpha_valid = np.ones((render_xy.shape[0],), dtype=bool)
        if rendered_alpha is not None:
            center_alpha, center_alpha_valid = _sample_scalar_map(np.asarray(rendered_alpha, dtype=np.float32), center_xy, width, height)
        center_valid = center_depth_valid & center_alpha_valid & (center_alpha >= float(min_render_alpha))
        if max_render_depth_delta_m is not None and float(max_render_depth_delta_m) >= 0.0:
            guard_valid &= np.isfinite(center_depth) & (np.abs(depth_values - center_depth) <= float(max_render_depth_delta_m))
    used_xy = render_xy.copy()
    used_depth = depth_values.copy()
    used_alpha = alpha_values.copy()
    offset_applied = np.full((render_xy.shape[0],), None, dtype=object)
    offset_norm = np.full((render_xy.shape[0],), np.nan, dtype=np.float64)
    if has_grid and center_xy is not None:
        offset_norm = np.linalg.norm(render_xy - center_xy, axis=1).astype(np.float64)
        offset_applied[:] = True
    valid = guard_valid.copy()
    if bool(fallback_to_cell_center) and has_grid and center_xy is not None and center_valid is not None:
        fallback = ~valid & center_valid
        used_xy[fallback] = center_xy[fallback]
        used_depth[fallback] = center_depth[fallback]
        used_alpha[fallback] = center_alpha[fallback] if center_alpha is not None else 1.0
        valid[fallback] = True
        offset_applied[fallback] = False
        offset_norm[fallback] = 0.0
    xyz, xyz_valid = backproject_depth_to_world(used_xy, used_depth, camera, render_pose_w2c)
    valid = valid & xyz_valid
    pnp_matches: list[QueryTo3DMatch] = []
    for idx, (match, is_valid) in enumerate(zip(matches, valid)):
        if not bool(is_valid):
            continue
        pnp_matches.append(
            QueryTo3DMatch(
                token_index=int(match.query_index),
                xy=np.asarray(match.query_xy, dtype=np.float64).reshape(2),
                track_id=int(match.render_index),
                xyz=xyz[idx].astype(np.float64, copy=True),
                similarity=float(match.similarity),
                ratio=float(match.ratio),
                landmark_variance=0.0,
                source="rendered_keypoint_feature",
                observation_count=None,
                similarity_margin=match.similarity_margin,
                render_alpha=float(used_alpha[idx]),
                render_depth=float(used_depth[idx]),
                pnp_soft_score=match.dual_softmax_confidence,
                patch_offset_applied=None if offset_applied[idx] is None else bool(offset_applied[idx]),
                patch_offset_norm_px=None if not np.isfinite(offset_norm[idx]) else float(offset_norm[idx]),
            )
        )
    return pnp_matches
