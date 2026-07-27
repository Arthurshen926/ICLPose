"""Analytic geometry from correlation displacement to a joint SE(3) update."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_v6.local_correlation import (
    CorrelationDistribution,
)
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


@dataclass(frozen=True)
class SE3UpdateResult:
    delta: np.ndarray
    updated_pose_w2c: np.ndarray
    covariance: np.ndarray
    used_point_count: int
    used_maplet_count: int
    condition_number: float
    residual_rms_px: float
    success: bool


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def projection_jacobian(
    world_xyz: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected pixels and d(pixel)/d(left-SE3)."""

    world = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_xyz = world @ pose[:3, :3].T + pose[:3, 3]
    z = camera_xyz[:, 2]
    safe_z = np.maximum(z, 1e-8)
    x = camera_xyz[:, 0] / safe_z
    y = camera_xyz[:, 1] / safe_z
    matrix, distortion = camera_matrix_and_distortion(camera)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    k = float(distortion.reshape(-1)[0]) if distortion.size else 0.0
    radial = 1.0 + k * (x * x + y * y)
    pixels = np.stack(
        [fx * x * radial + cx, fy * y * radial + cy], axis=1
    )
    jacobian = np.zeros((world.shape[0], 2, 6), dtype=np.float64)
    for row, point in enumerate(camera_xyz):
        normalized_jacobian = np.asarray(
            [
                [1.0 / safe_z[row], 0.0, -point[0] / safe_z[row] ** 2],
                [0.0, 1.0 / safe_z[row], -point[1] / safe_z[row] ** 2],
            ]
        )
        distortion_jacobian = np.asarray(
            [
                [
                    fx * (radial[row] + 2.0 * k * x[row] * x[row]),
                    fx * 2.0 * k * x[row] * y[row],
                ],
                [
                    fy * 2.0 * k * x[row] * y[row],
                    fy * (radial[row] + 2.0 * k * y[row] * y[row]),
                ],
            ]
        )
        motion_jacobian = np.concatenate(
            [-_skew(point), np.eye(3, dtype=np.float64)], axis=1
        )
        jacobian[row] = (
            distortion_jacobian @ normalized_jacobian @ motion_jacobian
        )
    return pixels, jacobian


def se3_exp(delta: np.ndarray) -> np.ndarray:
    value = np.asarray(delta, dtype=np.float64).reshape(6)
    omega, translation = value[:3], value[3:]
    theta = float(np.linalg.norm(omega))
    skew = _skew(omega)
    if theta < 1e-8:
        rotation = np.eye(3) + skew
        left_jacobian = np.eye(3) + 0.5 * skew
    else:
        rotation = (
            np.eye(3)
            + np.sin(theta) / theta * skew
            + (1.0 - np.cos(theta)) / theta**2 * (skew @ skew)
        )
        left_jacobian = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta**2 * skew
            + (theta - np.sin(theta)) / theta**3 * (skew @ skew)
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = left_jacobian @ translation
    return transform


def solve_correlation_se3_update(
    correlation: CorrelationDistribution,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    fit_maplet_ids: np.ndarray | None = None,
    maximum_null_probability: float = 0.50,
    maximum_entropy: float = 3.0,
    minimum_matchability: float = 0.10,
    minimum_variance_px2: float = 0.25,
    damping: float = 1e-3,
    robust_delta: float = 2.5,
    iterations: int = 4,
    minimum_points: int = 8,
    displacement_scale_xy: tuple[float, float] = (1.0, 1.0),
    maximum_translation_step_m: float = 0.10,
    maximum_rotation_step_deg: float = 2.0,
    dominant_mode_conditioning: bool = True,
    mode_radius_cells: float = 1.5,
    balance_maplet_weights: bool = True,
) -> SE3UpdateResult:
    count = int(correlation.xyz.shape[0])
    keep = (
        (correlation.null_probability <= float(maximum_null_probability))
        & (correlation.entropy <= float(maximum_entropy))
        & (correlation.matchability >= float(minimum_matchability))
        & np.isfinite(correlation.mean_displacement).all(axis=1)
    )
    if fit_maplet_ids is not None:
        keep &= np.isin(
            correlation.maplet_ids,
            np.asarray(fit_maplet_ids, dtype=np.int64),
        )
    rows = np.flatnonzero(keep)
    failure = SE3UpdateResult(
        delta=np.zeros((6,), dtype=np.float64),
        updated_pose_w2c=np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
        covariance=np.full((6, 6), np.inf, dtype=np.float64),
        used_point_count=int(rows.size),
        used_maplet_count=int(np.unique(correlation.maplet_ids[rows]).size),
        condition_number=float("inf"),
        residual_rms_px=float("inf"),
        success=False,
    )
    if rows.size < int(minimum_points):
        return failure
    projected_pixels, jacobian = projection_jacobian(
        correlation.xyz[rows], pose_w2c, camera
    )
    mean_displacement = np.asarray(
        correlation.mean_displacement[rows], dtype=np.float64
    )
    covariance2 = np.asarray(correlation.covariance[rows], dtype=np.float64)
    if bool(dominant_mode_conditioning):
        probability = np.asarray(
            correlation.probabilities[rows], dtype=np.float64
        )
        offsets = np.asarray(correlation.offsets_xy, dtype=np.float64)
        mode = offsets[np.argmax(probability, axis=1)]
        neighborhood = (
            np.linalg.norm(offsets[None] - mode[:, None], axis=2)
            <= float(mode_radius_cells)
        )
        local_probability = probability * neighborhood
        local_mass = np.sum(local_probability, axis=1)
        visible_mass = np.maximum(np.sum(probability, axis=1), 1e-12)
        local_probability /= np.maximum(
            np.sum(local_probability, axis=1, keepdims=True), 1e-12
        )
        mode_mean = local_probability @ offsets
        confident_mode = local_mass / visible_mass >= 0.50
        mean_displacement = np.where(
            confident_mode[:, None], mode_mean, mean_displacement
        )
    scale = np.asarray(displacement_scale_xy, dtype=np.float64).reshape(2)
    # Correlation offsets originate at discrete raster pixel centers, while
    # the interpolated surface XYZ generally reprojects at a subpixel
    # location (especially for conservative subpixel triangles). Convert the
    # matched query grid coordinate to original-image pixels, then subtract
    # the actual current projection used by the analytic Jacobian.
    render_pixel = np.asarray(
        correlation.pixel_xy[rows], dtype=np.float64
    )
    matched_query_pixel = (
        (render_pixel + 0.5) * scale[None]
        - 0.5
        + mean_displacement * scale[None]
    )
    target = matched_query_pixel - projected_pixels
    covariance2 = covariance2 * (
        scale[None, :, None] * scale[None, None, :]
    )
    covariance2 += np.eye(2)[None] * float(minimum_variance_px2)
    inverse_covariance = np.linalg.inv(covariance2)
    confidence = (
        (1.0 - correlation.null_probability[rows])
        * correlation.matchability[rows]
    ).astype(np.float64)
    if bool(balance_maplet_weights):
        # Equalise total evidence per maplet before robust reweighting.  Pixel
        # area must not let one large repetitive facade suppress several
        # smaller, geometrically independent maplets.
        row_maplets = correlation.maplet_ids[rows]
        unique_maplets = np.unique(row_maplets)
        target_mass = float(np.sum(confidence)) / max(unique_maplets.size, 1)
        for maplet_id in unique_maplets:
            local = row_maplets == maplet_id
            confidence[local] *= target_mass / max(
                float(np.sum(confidence[local])), 1e-8
            )
    delta = np.zeros((6,), dtype=np.float64)
    normal_matrix = np.eye(6, dtype=np.float64)
    for _ in range(max(int(iterations), 1)):
        residual = target - np.einsum("nij,j->ni", jacobian, delta)
        mahalanobis2 = np.einsum(
            "ni,nij,nj->n", residual, inverse_covariance, residual
        )
        mahalanobis = np.sqrt(np.maximum(mahalanobis2, 1e-12))
        robust_weight = np.minimum(
            1.0, float(robust_delta) / np.maximum(mahalanobis, 1e-8)
        )
        weights = confidence * robust_weight
        normal_matrix = np.eye(6, dtype=np.float64) * float(damping)
        right_hand = np.zeros((6,), dtype=np.float64)
        for row in range(rows.size):
            information = inverse_covariance[row] * weights[row]
            normal_matrix += jacobian[row].T @ information @ jacobian[row]
            right_hand += jacobian[row].T @ information @ target[row]
        try:
            delta = np.linalg.solve(normal_matrix, right_hand)
        except np.linalg.LinAlgError:
            return failure
    residual = target - np.einsum("nij,j->ni", jacobian, delta)
    rotation_norm = float(np.linalg.norm(delta[:3]))
    maximum_rotation = np.deg2rad(float(maximum_rotation_step_deg))
    if rotation_norm > maximum_rotation > 0.0:
        delta[:3] *= maximum_rotation / rotation_norm
    translation_norm = float(np.linalg.norm(delta[3:]))
    if translation_norm > float(maximum_translation_step_m) > 0.0:
        delta[3:] *= float(maximum_translation_step_m) / translation_norm
    residual = target - np.einsum("nij,j->ni", jacobian, delta)
    condition = float(np.linalg.cond(normal_matrix))
    success = bool(np.all(np.isfinite(delta)) and condition < 1e12)
    try:
        pose_covariance = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError:
        pose_covariance = np.full((6, 6), np.inf, dtype=np.float64)
        success = False
    updated = se3_exp(delta) @ np.asarray(pose_w2c, dtype=np.float64).reshape(
        4, 4
    )
    return SE3UpdateResult(
        delta=delta,
        updated_pose_w2c=updated,
        covariance=pose_covariance,
        used_point_count=int(rows.size),
        used_maplet_count=int(np.unique(correlation.maplet_ids[rows]).size),
        condition_number=condition,
        residual_rms_px=float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        success=success,
    )


def _mode_conditioned_correlation(
    correlation: CorrelationDistribution,
    *,
    mode_rank: int,
    radius_cells: float = 1.5,
) -> CorrelationDistribution:
    probability = np.asarray(correlation.probabilities, dtype=np.float64)
    if probability.shape[1] <= int(mode_rank):
        return correlation
    order = np.argsort(-probability, axis=1, kind="stable")
    centres = correlation.offsets_xy[order[:, int(mode_rank)]]
    local = (
        np.linalg.norm(
            correlation.offsets_xy[None] - centres[:, None], axis=2
        )
        <= float(radius_cells)
    )
    conditioned = probability * local
    conditioned /= np.maximum(np.sum(conditioned, axis=1, keepdims=True), 1e-12)
    mean = conditioned @ np.asarray(correlation.offsets_xy, dtype=np.float64)
    residual = correlation.offsets_xy[None] - mean[:, None]
    covariance = np.einsum(
        "nk,nki,nkj->nij", conditioned, residual, residual
    )
    return replace(
        correlation,
        mean_displacement=mean.astype(np.float32),
        covariance=covariance.astype(np.float32),
    )


def solve_correlation_se3_hypotheses(
    correlation: CorrelationDistribution,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    include_second_mode: bool = True,
    **solver_kwargs,
) -> tuple[SE3UpdateResult, ...]:
    """Preserve posterior-mean/top-mode alternatives until verification."""

    variants = [
        correlation,
        _mode_conditioned_correlation(correlation, mode_rank=0),
    ]
    if bool(include_second_mode):
        variants.append(_mode_conditioned_correlation(correlation, mode_rank=1))
    results = []
    for variant in variants:
        result = solve_correlation_se3_update(
            variant,
            pose_w2c,
            camera,
            dominant_mode_conditioning=False,
            **solver_kwargs,
        )
        if not result.success:
            continue
        if any(
            np.linalg.norm(result.delta - previous.delta) < 1e-5
            for previous in results
        ):
            continue
        results.append(result)
    return tuple(results)
