"""Detector-only high-resolution support with RADIO-only map matching.

ALIKE contributes query coordinates and scores only.  Every match descriptor is
derived from RADIO-final through the frozen surface mapper, and the map reads
the single canonical primitive code already stored by Goal-Maplet.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

from .canonical_field import CanonicalSurfaceField
from .child_retrieval import ChildTilePosterior
from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap


@dataclass(frozen=True)
class DetectorRadioRefinement:
    pose_w2c: np.ndarray
    success: bool
    selected_child_count: int
    candidate_primitive_count: int
    match_count: int
    inlier_count: int
    median_reprojection_px: float
    mean_inlier_similarity: float


def sample_radio_at_pixels(
    feature_map: np.ndarray,
    xy_px: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Bilinearly sample token-center RADIO codes at detector coordinates."""

    feature = np.asarray(feature_map, dtype=np.float32)
    xy = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    if feature.ndim != 3:
        raise ValueError("feature_map must have shape (C,H,W)")
    channels, height, width = feature.shape
    token_x = (xy[:, 0] + 0.5) * width / float(image_width) - 0.5
    token_y = (xy[:, 1] + 0.5) * height / float(image_height) - 0.5
    x0 = np.floor(token_x).astype(np.int64)
    y0 = np.floor(token_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = token_x - x0, token_y - y0
    x0, x1 = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0, y1 = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    value = (
        (1.0 - wx)[:, None] * (1.0 - wy)[:, None] * feature[:, y0, x0].T
        + wx[:, None] * (1.0 - wy)[:, None] * feature[:, y0, x1].T
        + (1.0 - wx)[:, None] * wy[:, None] * feature[:, y1, x0].T
        + wx[:, None] * wy[:, None] * feature[:, y1, x1].T
    )
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def _project(points: np.ndarray, pose: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation, _ = cv2.Rodrigues(pose[:3, :3])
    xy, _ = cv2.projectPoints(points.astype(np.float64), rotation, pose[:3, 3], matrix, distortion)
    camera_xyz = points @ pose[:3, :3].T + pose[:3, 3]
    return xy.reshape(-1, 2), camera_xyz[:, 2]


def _pose_conditioned_children(
    pose: np.ndarray,
    support_xy_px: np.ndarray,
    support_extent_px: np.ndarray,
    posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    camera,
    *,
    maximum_per_support: int = 2,
) -> np.ndarray:
    rows = posterior.candidate_child_rows
    probability = posterior.candidate_probabilities.astype(np.float64)
    unique = np.unique(rows[rows >= 0])
    if unique.size == 0:
        return unique
    projected, depth = _project(physical.child_centers[unique], pose, camera)
    projection_by_child = {int(child): projected[index] for index, child in enumerate(unique.tolist())}
    depth_by_child = {int(child): float(depth[index]) for index, child in enumerate(unique.tolist())}
    selected: set[int] = set()
    for support in range(rows.shape[0]):
        valid = (rows[support] >= 0) & (probability[support] > 0.0)
        candidate = rows[support, valid]
        if candidate.size == 0:
            continue
        point = np.asarray([projection_by_child[int(child)] for child in candidate])
        child_depth = np.asarray([depth_by_child[int(child)] for child in candidate])
        residual = np.linalg.norm(point - support_xy_px[support], axis=1)
        radius = max(24.0, 1.25 * float(np.linalg.norm(support_extent_px[support])))
        score = np.log(np.maximum(probability[support, valid], 1e-12)) - 0.5 * np.square(residual / radius)
        score[child_depth <= 0.05] = -np.inf
        for local in np.argsort(-score, kind="stable")[: int(maximum_per_support)].tolist():
            if np.isfinite(score[local]) and residual[local] <= 2.0 * radius:
                selected.add(int(candidate[local]))
    return np.asarray(sorted(selected), dtype=np.int64)


def refine_pose_with_detector_radio(
    query_feature_map: np.ndarray,
    detector_xy_px: np.ndarray,
    detector_scores: np.ndarray,
    support_xy_px: np.ndarray,
    support_extent_px: np.ndarray,
    child_posterior: ChildTilePosterior,
    initial_pose_w2c: np.ndarray,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    camera,
    *,
    local_radius_px: float = 40.0,
    neighbors_per_detection: int = 32,
    maximum_detections: int = 512,
    maximum_matches: int = 384,
    minimum_similarity: float = 0.35,
    minimum_margin: float = 0.005,
    ransac_reprojection_px: float = 10.0,
    ransac_iterations: int = 10000,
) -> DetectorRadioRefinement:
    """Refine one coarse mode in a retrieved child union, never globally."""

    pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    child_rows = _pose_conditioned_children(
        pose, np.asarray(support_xy_px), np.asarray(support_extent_px), child_posterior,
        physical, camera,
    )
    field_row_by_scene = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_row_by_scene[field.primitive_rows] = np.arange(field.primitive_rows.size, dtype=np.int64)
    primitive_parts = []
    for child in child_rows.tolist():
        start, end = int(physical.child_member_offsets[child]), int(physical.child_member_offsets[child + 1])
        primitive_parts.append(physical.child_member_primitive_rows[start:end])
    primitive_rows = np.unique(np.concatenate(primitive_parts)) if primitive_parts else np.zeros((0,), dtype=np.int64)
    field_rows = field_row_by_scene[primitive_rows]
    valid_field = field_rows >= 0
    primitive_rows, field_rows = primitive_rows[valid_field], field_rows[valid_field]
    if primitive_rows.size:
        projected, depth = _project(physical.primitive_centers[primitive_rows], pose, camera)
        camera_center = -pose[:3, :3].T @ pose[:3, 3]
        view = camera_center[None] - physical.primitive_centers[primitive_rows]
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
        incidence = np.sum(physical.primitive_normals[primitive_rows] * view, axis=1)
        front = (physical.primitive_sidedness[primitive_rows] == DOUBLE_SIDED) | (incidence >= 0.02)
        visible = (
            front & (depth > 0.05)
            & (projected[:, 0] >= -local_radius_px) & (projected[:, 0] < camera.width + local_radius_px)
            & (projected[:, 1] >= -local_radius_px) & (projected[:, 1] < camera.height + local_radius_px)
        )
        primitive_rows, field_rows, projected = primitive_rows[visible], field_rows[visible], projected[visible]
    detection_xy = np.asarray(detector_xy_px, dtype=np.float64).reshape(-1, 2)
    detection_score = np.asarray(detector_scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-detection_score, kind="stable")[: int(maximum_detections)]
    detection_xy, detection_score = detection_xy[order], detection_score[order]
    query_descriptor = sample_radio_at_pixels(
        query_feature_map, detection_xy,
        image_width=int(camera.width), image_height=int(camera.height),
    )
    if primitive_rows.size < 6 or detection_xy.shape[0] < 6:
        return DetectorRadioRefinement(pose, False, int(child_rows.size), int(primitive_rows.size), 0, 0, np.inf, 0.0)
    tree = cKDTree(projected)
    match_candidates = []
    for query_row, point in enumerate(detection_xy):
        nearby = tree.query_ball_point(point, r=float(local_radius_px))
        if not nearby:
            continue
        if len(nearby) > int(neighbors_per_detection):
            distance = np.linalg.norm(projected[nearby] - point, axis=1)
            nearby = np.asarray(nearby, dtype=np.int64)[np.argsort(distance)[: int(neighbors_per_detection)]].tolist()
        nearby = np.asarray(nearby, dtype=np.int64)
        similarity = field.codes[field_rows[nearby]] @ query_descriptor[query_row]
        ranked = np.argsort(-similarity, kind="stable")
        best = int(ranked[0])
        second = float(similarity[ranked[1]]) if ranked.size > 1 else -1.0
        if float(similarity[best]) >= float(minimum_similarity) and float(similarity[best] - second) >= float(minimum_margin):
            map_local = int(nearby[best])
            match_candidates.append((
                float(similarity[best]), float(detection_score[query_row]), query_row, map_local
            ))
    match_candidates.sort(key=lambda value: (-(value[0] + 0.05 * value[1]), value[2], value[3]))
    selected_matches = []
    used_query, used_primitive = set(), set()
    for value in match_candidates:
        primitive = int(primitive_rows[value[3]])
        if value[2] in used_query or primitive in used_primitive:
            continue
        used_query.add(value[2])
        used_primitive.add(primitive)
        selected_matches.append(value)
        if len(selected_matches) >= int(maximum_matches):
            break
    if len(selected_matches) < 6:
        return DetectorRadioRefinement(pose, False, int(child_rows.size), int(primitive_rows.size), len(selected_matches), 0, np.inf, 0.0)
    xy = np.asarray([detection_xy[value[2]] for value in selected_matches], dtype=np.float64)
    xyz = np.asarray([physical.primitive_centers[primitive_rows[value[3]]] for value in selected_matches], dtype=np.float64)
    similarity = np.asarray([value[0] for value in selected_matches], dtype=np.float64)
    matrix, distortion = camera_matrix_and_distortion(camera)
    cv2.setRNGSeed(194917)
    try:
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            xyz, xy, matrix, distortion,
            iterationsCount=int(ransac_iterations), reprojectionError=float(ransac_reprojection_px),
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        success, inliers = False, None
    if not success or inliers is None or len(inliers) < 6:
        return DetectorRadioRefinement(pose, False, int(child_rows.size), int(primitive_rows.size), len(selected_matches), 0, np.inf, 0.0)
    inlier = np.asarray(inliers, dtype=np.int64).reshape(-1)
    try:
        rotation, translation = cv2.solvePnPRefineLM(xyz[inlier], xy[inlier], matrix, distortion, rotation, translation)
    except cv2.error:
        pass
    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = cv2.Rodrigues(rotation)[0]
    refined[:3, 3] = np.asarray(translation).reshape(3)
    projected_inlier, _ = _project(xyz[inlier], refined, camera)
    residual = np.linalg.norm(projected_inlier - xy[inlier], axis=1)
    return DetectorRadioRefinement(
        refined, True, int(child_rows.size), int(primitive_rows.size), len(selected_matches),
        int(inlier.size), float(np.median(residual)), float(np.mean(similarity[inlier])),
    )
