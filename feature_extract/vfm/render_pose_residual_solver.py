"""Weighted SE(3) residual solver initialized from a rendered pose."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + omega_hat + 0.5 * (omega_hat @ omega_hat)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * omega_hat + b * (omega_hat @ omega_hat)


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega = xi[:3]
    velocity = xi[3:]
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    rotation = _so3_exp(omega)
    if theta < 1e-12:
        v_matrix = np.eye(3, dtype=np.float64) + 0.5 * omega_hat + (1.0 / 6.0) * (omega_hat @ omega_hat)
    else:
        theta2 = theta * theta
        theta3 = theta2 * theta
        v_matrix = (
            np.eye(3, dtype=np.float64)
            + ((1.0 - np.cos(theta)) / theta2) * omega_hat
            + ((theta - np.sin(theta)) / theta3) * (omega_hat @ omega_hat)
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = v_matrix @ velocity
    return transform


def _camera_params(camera: ColmapCamera) -> tuple[float, float, float, float, float | None]:
    if camera.model_id == 0:
        f, cx, cy = camera.params[:3]
        return float(f), float(f), float(cx), float(cy), None
    if camera.model_id == 1:
        fx, fy, cx, cy = camera.params[:4]
        return float(fx), float(fy), float(cx), float(cy), None
    if camera.model_id == 2:
        f, cx, cy, k = camera.params[:4]
        return float(f), float(f), float(cx), float(cy), float(k)
    raise ValueError(f"unsupported camera model id for residual solver: {camera.model_id}")


def _project_points(points_xyz: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy, radial_k = _camera_params(camera)
    points_cam = (pose_w2c[:3, :3] @ points_xyz.T + pose_w2c[:3, 3:4]).T
    z = points_cam[:, 2]
    safe_z = np.where(np.abs(z) > 1e-12, z, np.sign(z) * 1e-12 + (z == 0.0) * 1e-12)
    x = points_cam[:, 0] / safe_z
    y = points_cam[:, 1] / safe_z
    if radial_k is not None:
        scale = 1.0 + radial_k * (x * x + y * y)
        x = x * scale
        y = y * scale
    projected = np.column_stack([fx * x + cx, fy * y + cy]).astype(np.float64)
    return projected, z.astype(np.float64)


def _match_weight(match: QueryTo3DMatch) -> float:
    weight = 1.0
    for attr in (
        "pnp_soft_score",
        "pairwise_weighted_similarity",
        "quality_weighted_similarity",
        "patch_offset_confidence",
        "landmark_quality",
        "map_reliability",
    ):
        value = getattr(match, attr, None)
        if value is not None and np.isfinite(float(value)):
            weight *= max(float(value), 1e-6)
            break
    sigma = match.measurement_sigma_px
    if sigma is not None and np.isfinite(float(sigma)) and float(sigma) > 0.0:
        weight /= float(sigma) * float(sigma)
    return max(float(weight), 1e-12)


def _residual_vector(
    pose_w2c: np.ndarray,
    points_xyz: np.ndarray,
    image_xy: np.ndarray,
    camera: ColmapCamera,
    sqrt_weights: np.ndarray,
) -> np.ndarray:
    projected, depth = _project_points(points_xyz, pose_w2c, camera)
    residuals = (projected - image_xy).reshape(-1)
    invalid = ~np.isfinite(residuals)
    if np.any(depth <= 1e-9):
        invalid = invalid | np.repeat(depth <= 1e-9, 2)
    if np.any(invalid):
        residuals = residuals.copy()
        residuals[invalid] = 1e6
    return residuals * np.repeat(sqrt_weights, 2)


def _numeric_jacobian(
    pose_w2c: np.ndarray,
    points_xyz: np.ndarray,
    image_xy: np.ndarray,
    camera: ColmapCamera,
    sqrt_weights: np.ndarray,
) -> np.ndarray:
    base_residual = _residual_vector(pose_w2c, points_xyz, image_xy, camera, sqrt_weights)
    jacobian = np.zeros((base_residual.size, 6), dtype=np.float64)
    for col in range(6):
        eps = 1e-6
        delta = np.zeros((6,), dtype=np.float64)
        delta[col] = eps
        plus = _residual_vector(_se3_exp(delta) @ pose_w2c, points_xyz, image_xy, camera, sqrt_weights)
        delta[col] = -eps
        minus = _residual_vector(_se3_exp(delta) @ pose_w2c, points_xyz, image_xy, camera, sqrt_weights)
        jacobian[:, col] = (plus - minus) / (2.0 * eps)
    return jacobian


def solve_render_pose_delta_from_matches(
    matches: Sequence[QueryTo3DMatch],
    render_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    max_iterations: int = 10,
    damping: float = 1e-4,
) -> np.ndarray:
    """Solve a small SE(3) update initialized at render_pose_w2c.

    The returned pose is global query pose, not a local-frame pose.
    """

    pose = np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    if len(matches) < 3:
        return pose
    points_xyz = np.stack([np.asarray(match.xyz, dtype=np.float64).reshape(3) for match in matches], axis=0)
    image_xy = np.stack([np.asarray(match.xy, dtype=np.float64).reshape(2) for match in matches], axis=0)
    weights = np.asarray([_match_weight(match) for match in matches], dtype=np.float64)
    sqrt_weights = np.sqrt(weights)
    damping_value = max(float(damping), 0.0)

    previous_error = np.inf
    for _iteration in range(max(0, int(max_iterations))):
        residual = _residual_vector(pose, points_xyz, image_xy, camera, sqrt_weights)
        if not np.all(np.isfinite(residual)):
            break
        error = float(residual @ residual)
        jacobian = _numeric_jacobian(pose, points_xyz, image_xy, camera, sqrt_weights)
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        diagonal = np.maximum(np.diag(normal_matrix), 1.0)
        system = normal_matrix + damping_value * np.diag(diagonal)
        try:
            step = -np.linalg.solve(system, gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(system, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate_pose = _se3_exp(step * scale) @ pose
            candidate_residual = _residual_vector(candidate_pose, points_xyz, image_xy, camera, sqrt_weights)
            candidate_error = float(candidate_residual @ candidate_residual)
            if np.isfinite(candidate_error) and candidate_error <= error:
                pose = candidate_pose
                error = candidate_error
                accepted = True
                break
        if not accepted:
            break
        if float(np.linalg.norm(step)) < 1e-10 or abs(previous_error - error) < 1e-12:
            break
        previous_error = error
    return pose
