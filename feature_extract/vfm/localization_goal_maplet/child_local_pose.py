"""Joint pose proposal from multi-modal child-local surface factors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


@dataclass(frozen=True)
class JointChildLocalPose:
    success: bool
    pose_w2c: np.ndarray
    score: float
    support_count: int
    selected_modes: np.ndarray


def solve_joint_child_local_pose(
    query_xy_px: np.ndarray,
    query_scale_px: np.ndarray,
    mode_points: np.ndarray,
    mode_probabilities: np.ndarray,
    mode_valid: np.ndarray,
    camera,
    *,
    initial_pose_w2c: np.ndarray,
    random_key: str,
    trials: int = 512,
    minimal_group_count: int = 6,
) -> JointChildLocalPose:
    """Preserve local ambiguity and solve one globally coherent surface mode."""

    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    scale = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
    points = np.asarray(mode_points, dtype=np.float64)
    probability = np.asarray(mode_probabilities, dtype=np.float64)
    valid = np.asarray(mode_valid, dtype=bool)
    count = xy.shape[0]
    if (
        points.ndim != 3 or points.shape[0] != count or points.shape[2] != 3
        or probability.shape != points.shape[:2] or valid.shape != probability.shape
        or scale.shape != (count,)
    ):
        raise ValueError("joint child-local pose inputs differ")
    probability = np.where(valid, np.maximum(probability, 1e-8), 0.0)
    probability /= np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-12)
    eligible = np.flatnonzero(np.sum(valid, axis=1) > 0)
    empty_pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    if eligible.size < int(minimal_group_count):
        return JointChildLocalPose(False, empty_pose, float("-inf"), 0, np.full(count, -1, dtype=np.int64))
    matrix, distortion = camera_matrix_and_distortion(camera)

    def evaluate(pose: np.ndarray):
        rotation_vector, _ = cv2.Rodrigues(pose[:3, :3])
        projected, _ = cv2.projectPoints(
            points.reshape(-1, 3), rotation_vector, pose[:3, 3], matrix, distortion
        )
        projected = projected.reshape(points.shape[:2] + (2,))
        camera_xyz = points.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        depth = camera_xyz[:, 2].reshape(points.shape[:2])
        residual = np.linalg.norm(projected - xy[:, None], axis=2)
        local_valid = valid & (depth > 0.05)
        log_factor = np.log(np.maximum(probability, 1e-12))
        log_factor -= 0.5 * np.square(residual / np.maximum(scale[:, None], 8.0))
        log_factor[~local_valid] = -1e6
        selected = np.argmax(log_factor, axis=1)
        selected_residual = residual[np.arange(count), selected]
        maximum = np.max(log_factor, axis=1)
        log_mixture = maximum + np.log(np.maximum(
            np.sum(np.exp(np.clip(log_factor - maximum[:, None], -60.0, 0.0)), axis=1), 1e-12
        ))
        usable = np.any(local_valid, axis=1)
        score = float(np.mean(log_mixture[usable]))
        support = int(np.sum(usable & (selected_residual <= np.maximum(2.0 * scale, 24.0))))
        return score, support, selected, selected_residual

    initial = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    best_score, best_support, best_selected, best_residual = evaluate(initial)
    best_pose = initial.copy()
    seed = int.from_bytes(hashlib.sha256(str(random_key).encode("utf8")).digest()[:4], "little") & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    cv2.setRNGSeed(seed)
    leverage = 0.5 + np.linalg.norm(
        (xy - np.asarray([0.5 * camera.width, 0.5 * camera.height]))
        / np.asarray([camera.width, camera.height]), axis=1
    )
    group_probability = leverage[eligible]
    group_probability /= np.sum(group_probability)
    for _ in range(int(trials)):
        groups = rng.choice(
            eligible, size=int(minimal_group_count), replace=False, p=group_probability
        )
        modes = np.asarray([
            rng.choice(probability.shape[1], p=probability[group]) for group in groups
        ], dtype=np.int64)
        xyz = points[groups, modes]
        if np.unique(np.round(xyz, 5), axis=0).shape[0] < int(minimal_group_count):
            continue
        try:
            success, rotation, translation = cv2.solvePnP(
                xyz, xy[groups], matrix, distortion, flags=cv2.SOLVEPNP_EPNP
            )
        except cv2.error:
            continue
        if not success:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = np.asarray(translation).reshape(3)
        score, support, selected, residual = evaluate(pose)
        if (score, support) > (best_score, best_support):
            best_score, best_support = score, support
            best_pose, best_selected, best_residual = pose, selected, residual
    selected_xyz = points[np.arange(count), best_selected]
    refine_rows = np.flatnonzero(
        np.any(valid, axis=1)
        & (best_residual <= np.maximum(2.0 * scale, 24.0))
    )
    if (
        refine_rows.size >= int(minimal_group_count)
        and np.unique(np.round(selected_xyz[refine_rows], 5), axis=0).shape[0] >= int(minimal_group_count)
    ):
        rotation, _ = cv2.Rodrigues(best_pose[:3, :3])
        translation = best_pose[:3, 3].reshape(3, 1)
        try:
            rotation, translation = cv2.solvePnPRefineLM(
                selected_xyz[refine_rows], xy[refine_rows], matrix, distortion,
                rotation, translation,
            )
            refined = np.eye(4, dtype=np.float64)
            refined[:3, :3] = cv2.Rodrigues(rotation)[0]
            refined[:3, 3] = np.asarray(translation).reshape(3)
            score, support, selected, residual = evaluate(refined)
            if (score, support) >= (best_score, best_support):
                best_pose, best_score, best_support = refined, score, support
                best_selected = selected
        except cv2.error:
            pass
    return JointChildLocalPose(True, best_pose, best_score, best_support, best_selected)
