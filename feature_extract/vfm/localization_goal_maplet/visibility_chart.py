"""Continuous local chart updates for mapping-view geometry proposals.

The mapping-view node supplies a set-valued visibility condition, not a pose
lookup.  Given an existing geometry proposal, this module uses that condition
to softly marginalize the query support's existing child alternatives and
re-estimate one bounded Sim(3) chart coordinate.  The update is a proposal: the
original state remains available and the common surface verifier decides
between them later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


@dataclass(frozen=True)
class VisibilityChartUpdate:
    pose_w2c: np.ndarray
    scale: float
    initial_objective: float
    final_objective: float
    accepted_iterations: int
    effective_support_count: int
    translation_update_m: float
    rotation_update_deg: float


def _weighted_similarity_pose(
    world_xyz: np.ndarray,
    query_xyz: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Fit ``query = scale * (R world + t)`` with positive weights."""

    world = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    query = np.asarray(query_xyz, dtype=np.float64).reshape(-1, 3)
    value = np.asarray(weight, dtype=np.float64).reshape(-1)
    valid = (
        np.all(np.isfinite(world), axis=1)
        & np.all(np.isfinite(query), axis=1)
        & np.isfinite(value) & (value > 0.0)
    )
    world, query, value = world[valid], query[valid], value[valid]
    if world.shape[0] < 3 or np.unique(np.round(world, 6), axis=0).shape[0] < 3:
        return None
    value /= max(float(np.sum(value)), 1e-12)
    world_center = np.sum(value[:, None] * world, axis=0)
    query_center = np.sum(value[:, None] * query, axis=0)
    world_zero = world - world_center
    query_zero = query - query_center
    covariance = (value[:, None] * query_zero).T @ world_zero
    variance = float(np.sum(value * np.sum(np.square(world_zero), axis=1)))
    if variance <= 1e-10:
        return None
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    sign = np.ones((3,), dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular * sign) / variance)
    if not np.isfinite(scale) or scale <= 1e-8:
        return None
    offset = query_center - scale * (rotation @ world_center)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = offset / scale
    if not np.all(np.isfinite(pose)):
        return None
    return pose, scale


def _soft_chart_measurement(
    pose_w2c: np.ndarray,
    scale: float,
    query_xyz: np.ndarray,
    candidate_child_rows: np.ndarray,
    candidate_parent_rows: np.ndarray,
    candidate_mass: np.ndarray,
    candidate_valid: np.ndarray,
    child_centers: np.ndarray,
    parent_visibility: np.ndarray,
    support_weight: np.ndarray,
    *,
    residual_sigma: float,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Return chart evidence and soft expected world points per support."""

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    query = np.asarray(query_xyz, dtype=np.float64).reshape(-1, 3)
    child = np.asarray(candidate_child_rows, dtype=np.int64)
    parent = np.asarray(candidate_parent_rows, dtype=np.int64)
    mass = np.asarray(candidate_mass, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool).copy()
    visibility = np.asarray(parent_visibility, dtype=np.float64).reshape(-1)
    support = np.asarray(support_weight, dtype=np.float64).reshape(-1)
    if (
        child.ndim != 2 or parent.shape != child.shape or mass.shape != child.shape
        or valid.shape != child.shape or query.shape[0] != child.shape[0]
        or support.shape != (child.shape[0],)
    ):
        raise ValueError("visibility-chart candidate arrays differ")
    valid &= (
        (child >= 0) & (child < int(np.asarray(child_centers).shape[0]))
        & (parent >= 0) & (parent < visibility.size)
        & np.isfinite(mass) & (mass > 0.0)
    )
    safe_child = np.maximum(child, 0)
    safe_parent = np.maximum(parent, 0)
    world = np.asarray(child_centers, dtype=np.float64)[safe_child]
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    predicted = float(scale) * camera
    query_radius = np.maximum(np.linalg.norm(query, axis=1), 0.25)
    residual = np.linalg.norm(predicted - query[:, None, :], axis=2) / query_radius[:, None]
    query_ray = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    camera_ray = camera / np.maximum(np.linalg.norm(camera, axis=2, keepdims=True), 1e-8)
    ray_cosine = np.sum(camera_ray * query_ray[:, None, :], axis=2)
    compatible = 0.02 + 0.98 * np.clip(visibility[safe_parent], 0.0, 1.0)
    valid &= (camera[:, :, 2] > 0.05) & np.isfinite(residual)
    prior = np.where(valid, mass * compatible, 0.0)
    prior /= np.maximum(np.sum(prior, axis=1, keepdims=True), 1e-12)
    kernel = np.exp(
        -0.5 * np.square(residual / max(float(residual_sigma), 1e-6))
        + 2.0 * np.clip(ray_cosine - 1.0, -1.0, 0.0)
    )
    evidence = np.sum(prior * kernel, axis=1)
    posterior = prior * kernel
    posterior /= np.maximum(np.sum(posterior, axis=1, keepdims=True), 1e-12)
    expected_world = np.sum(posterior[:, :, None] * world, axis=1)
    expected_residual = np.sum(posterior * residual, axis=1)
    entropy = -np.sum(
        np.where(posterior > 0.0, posterior * np.log(np.maximum(posterior, 1e-12)), 0.0),
        axis=1,
    )
    alternative_count = np.sum(valid, axis=1)
    maximum_entropy = np.log(np.maximum(alternative_count, 2))
    certainty = np.clip(1.0 - entropy / maximum_entropy, 0.0, 1.0)
    fit_weight = (
        np.maximum(support, 0.0)
        * (0.10 + 0.90 * certainty)
        * np.exp(-0.5 * np.square(expected_residual / 0.35))
    )
    supported = (alternative_count > 0) & (evidence > 1e-6) & np.isfinite(fit_weight)
    fit_weight[~supported] = 0.0
    objective_weight = np.where(supported, np.maximum(support, 0.0), 0.0)
    objective = float(
        np.sum(objective_weight * np.log(np.maximum(evidence, 1e-12)))
        / max(float(np.sum(objective_weight)), 1e-12)
    )
    return objective, expected_world, fit_weight, int(np.sum(fit_weight > 1e-4))


def refine_pose_in_visibility_chart(
    initial_pose_w2c: np.ndarray,
    initial_scale: float,
    query_xyz: np.ndarray,
    candidate_child_rows: np.ndarray,
    candidate_parent_rows: np.ndarray,
    candidate_mass: np.ndarray,
    candidate_valid: np.ndarray,
    child_centers: np.ndarray,
    parent_visibility: np.ndarray,
    support_weight: np.ndarray,
    *,
    iterations: int = 2,
    residual_sigma: float = 0.20,
    maximum_translation_update_m: float = 2.0,
    maximum_rotation_update_deg: float = 15.0,
    minimum_objective_improvement: float = 1e-4,
) -> VisibilityChartUpdate:
    """Estimate a bounded continuous chart coordinate by soft assignment."""

    initial = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    pose = initial.copy()
    scale = float(initial_scale)
    if not np.isfinite(scale) or scale <= 0.0 or int(iterations) <= 0:
        raise ValueError("invalid visibility-chart initialization")

    def measure(current_pose: np.ndarray, current_scale: float):
        return _soft_chart_measurement(
            current_pose, current_scale, query_xyz,
            candidate_child_rows, candidate_parent_rows,
            candidate_mass, candidate_valid, child_centers,
            parent_visibility, support_weight,
            residual_sigma=float(residual_sigma),
        )

    objective, expected_world, fit_weight, effective = measure(pose, scale)
    initial_objective = float(objective)
    accepted = 0
    for _ in range(int(iterations)):
        estimated = _weighted_similarity_pose(expected_world, query_xyz, fit_weight)
        if estimated is None:
            break
        candidate_pose, candidate_scale = estimated
        update_error = pnp_pose_error(candidate_pose, initial)
        if (
            float(update_error.translation_m) > float(maximum_translation_update_m)
            or float(update_error.rotation_deg) > float(maximum_rotation_update_deg)
            or candidate_scale / float(initial_scale) < 0.5
            or candidate_scale / float(initial_scale) > 2.0
        ):
            break
        candidate_measurement = measure(candidate_pose, candidate_scale)
        candidate_objective = float(candidate_measurement[0])
        if candidate_objective < objective + float(minimum_objective_improvement):
            break
        pose, scale = candidate_pose, float(candidate_scale)
        objective, expected_world, fit_weight, effective = candidate_measurement
        accepted += 1
    update = pnp_pose_error(pose, initial)
    return VisibilityChartUpdate(
        pose_w2c=pose,
        scale=float(scale),
        initial_objective=float(initial_objective),
        final_objective=float(objective),
        accepted_iterations=int(accepted),
        effective_support_count=int(effective),
        translation_update_m=float(update.translation_m),
        rotation_update_deg=float(update.rotation_deg),
    )
