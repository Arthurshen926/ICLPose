"""Latent top-L correspondence refinement for grouped 2D-3D localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.candidate_pose_evidence import (
    candidate_pose_evidence,
    project_candidate_xyz,
)
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


@dataclass(frozen=True)
class LatentEMConfig:
    iterations: int = 3
    identity_temperature: float = 1.0
    identity_temperature_floor: float = 0.75
    null_mass_floor: float = 0.02
    null_likelihood: float = 1e-3
    outlier_likelihood: float = 1e-3
    residual_sigma_px: float = 3.0
    spatial_evidence_weight: float | None = None
    coordinate_update_policy: str = "posterior_mean"
    minimum_coordinate_mode_probability: float = 0.0
    max_responsibility_change: float = 0.25
    min_candidate_weight: float = 2e-3
    min_effective_group_mass: float = 0.05
    min_effective_groups: int = 8
    enforce_track_capacity: bool = True
    robust_loss: str = "huber"
    robust_f_scale_px: float = 2.0
    max_nfev: int = 50
    max_translation_step_m: float = 1.0
    max_rotation_step_deg: float = 10.0
    objective_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not 1 <= int(self.iterations) <= 10:
            raise ValueError("latent EM iterations must be in [1, 10]")
        if float(self.identity_temperature_floor) <= 0.0:
            raise ValueError("identity temperature floor must be positive")
        if float(self.identity_temperature) < float(
            self.identity_temperature_floor
        ):
            raise ValueError("identity temperature is below its safety floor")
        if not 0.0 <= float(self.null_mass_floor) < 1.0:
            raise ValueError("null mass floor must be in [0, 1)")
        if not 0.0 < float(self.null_likelihood) <= 1.0:
            raise ValueError("null likelihood must be in (0, 1]")
        if not 0.0 < float(self.outlier_likelihood) <= 1.0:
            raise ValueError("outlier likelihood must be in (0, 1]")
        if float(self.residual_sigma_px) <= 0.0:
            raise ValueError("latent EM residual sigma must be positive")
        if self.spatial_evidence_weight is not None and not 0.0 <= float(
            self.spatial_evidence_weight
        ) <= 1.0:
            raise ValueError("spatial evidence weight must be in [0, 1]")
        if str(self.coordinate_update_policy) not in {
            "posterior_mean",
            "concentrated_map",
            "calibrated_mixture_map",
        }:
            raise ValueError("unsupported latent coordinate update policy")
        if not 0.0 <= float(self.minimum_coordinate_mode_probability) <= 1.0:
            raise ValueError(
                "minimum latent coordinate mode probability must be in [0, 1]"
            )
        if not 0.0 < float(self.max_responsibility_change) <= 1.0:
            raise ValueError("maximum responsibility change must be in (0, 1]")
        if float(self.min_candidate_weight) < 0.0:
            raise ValueError("minimum candidate weight must be non-negative")
        if not 0.0 <= float(self.min_effective_group_mass) <= 1.0:
            raise ValueError("effective group mass must be in [0, 1]")
        if int(self.min_effective_groups) < 4:
            raise ValueError("latent EM requires at least four effective groups")
        if str(self.robust_loss).lower() not in {
            "linear",
            "soft_l1",
            "huber",
            "cauchy",
            "arctan",
        }:
            raise ValueError("unsupported latent EM robust loss")
        if float(self.robust_f_scale_px) <= 0.0:
            raise ValueError("robust scale must be positive")
        if int(self.max_nfev) <= 0:
            raise ValueError("max_nfev must be positive")
        if float(self.max_translation_step_m) <= 0.0:
            raise ValueError("maximum translation step must be positive")
        if float(self.max_rotation_step_deg) <= 0.0:
            raise ValueError("maximum rotation step must be positive")
        if float(self.objective_tolerance) < 0.0:
            raise ValueError("objective tolerance must be non-negative")


@dataclass(frozen=True)
class LatentResponsibilityState:
    candidate_responsibilities: np.ndarray
    candidate_inlier_probabilities: np.ndarray
    null_responsibilities: np.ndarray
    candidate_xy: np.ndarray
    candidate_likelihoods: np.ndarray
    log_likelihood_sum: float
    effective_group_count: int
    mean_group_entropy: float
    max_track_mass: float


@dataclass(frozen=True)
class LatentEMResult:
    success: bool
    pose_w2c: np.ndarray
    state: LatentResponsibilityState
    accepted_iterations: int
    initial_log_likelihood_sum: float
    final_log_likelihood_sum: float
    failure_reason: str | None = None


def _project_candidates(
    pool: Any,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    return project_candidate_xyz(
        np.asarray(pool.xyz, dtype=np.float64),
        np.asarray(pool.valid_mask, dtype=bool),
        pose_w2c,
        camera,
    )


def _tempered_identity_prior(
    pool: Any, config: LatentEMConfig
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(pool.valid_mask, dtype=bool)
    scores = np.where(
        valid, np.maximum(np.asarray(pool.descriptor_scores, dtype=np.float64), 0.0), 0.0
    )
    null = np.maximum(np.asarray(pool.null_scores, dtype=np.float64), 0.0)
    total = np.sum(scores, axis=1) + null
    if np.any(total <= 1e-12):
        raise ValueError("latent correspondence prior has zero probability mass")
    scores /= total[:, None]
    null /= total
    prior_null = np.maximum(null, float(config.null_mass_floor))
    prior_null = np.minimum(prior_null, 1.0)
    prior = np.zeros_like(scores)
    inverse_temperature = 1.0 / float(config.identity_temperature)
    for row in range(len(prior)):
        columns = np.flatnonzero(valid[row] & (scores[row] > 0.0))
        if len(columns) == 0 or prior_null[row] >= 1.0:
            prior_null[row] = 1.0
            continue
        logits = np.log(np.maximum(scores[row, columns], 1e-12))
        logits *= inverse_temperature
        logits -= float(np.max(logits))
        conditional = np.exp(logits)
        conditional /= max(float(np.sum(conditional)), 1e-12)
        prior[row, columns] = (1.0 - prior_null[row]) * conditional
    return prior, prior_null


def _candidate_likelihoods_and_xy(
    pool: Any,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: LatentEMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evidence = candidate_pose_evidence(
        pool,
        pose_w2c,
        camera,
        residual_sigma_px=float(config.residual_sigma_px),
        outlier_likelihood=float(config.outlier_likelihood),
        spatial_evidence_weight=config.spatial_evidence_weight,
        coordinate_update_policy=str(config.coordinate_update_policy),
        minimum_coordinate_mode_probability=float(
            config.minimum_coordinate_mode_probability
        ),
    )
    return (
        evidence.candidate_likelihoods,
        evidence.candidate_xy,
        evidence.candidate_inlier_probabilities,
    )


def _enforce_track_capacity(
    responsibilities: np.ndarray,
    null_responsibilities: np.ndarray,
    track_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    output = np.asarray(responsibilities, dtype=np.float64).copy()
    null = np.asarray(null_responsibilities, dtype=np.float64).copy()
    valid_tracks = np.asarray(track_ids, dtype=np.int64)[valid_mask]
    for track_id in np.unique(valid_tracks):
        mask = valid_mask & (track_ids == int(track_id))
        mass = float(np.sum(output[mask]))
        if mass <= 1.0:
            continue
        scaled = output[mask] / mass
        removed_by_row = np.zeros((output.shape[0],), dtype=np.float64)
        rows, _columns = np.nonzero(mask)
        np.add.at(removed_by_row, rows, output[mask] - scaled)
        output[mask] = scaled
        null += removed_by_row
    row_total = np.sum(output, axis=1) + null
    output /= np.maximum(row_total[:, None], 1e-12)
    null /= np.maximum(row_total, 1e-12)
    max_track_mass = 0.0
    for track_id in np.unique(valid_tracks):
        max_track_mass = max(
            max_track_mass,
            float(np.sum(output[valid_mask & (track_ids == int(track_id))])),
        )
    return output, null, float(max_track_mass)


def _enforce_null_floor(
    responsibilities: np.ndarray,
    null_responsibilities: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(responsibilities, dtype=np.float64).copy()
    null = np.asarray(null_responsibilities, dtype=np.float64).copy()
    target_null = np.maximum(null, float(floor))
    candidate_mass = np.sum(candidate, axis=1)
    target_candidate_mass = np.maximum(1.0 - target_null, 0.0)
    scale = np.divide(
        target_candidate_mass,
        candidate_mass,
        out=np.zeros_like(target_candidate_mass),
        where=candidate_mass > 1e-12,
    )
    candidate *= scale[:, None]
    null = 1.0 - np.sum(candidate, axis=1)
    return candidate, null


def _maximum_track_mass(
    responsibilities: np.ndarray,
    track_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    tracks = np.asarray(track_ids, dtype=np.int64)
    valid_tracks = tracks[valid_mask]
    if valid_tracks.size == 0:
        return 0.0
    return float(
        max(
            np.sum(responsibilities[valid_mask & (tracks == int(track_id))])
            for track_id in np.unique(valid_tracks)
        )
    )


def _bounded_joint_update(
    previous_candidate: np.ndarray,
    previous_null: np.ndarray,
    proposed_candidate: np.ndarray,
    proposed_null: np.ndarray,
    max_change: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Damp a coupled posterior with one global convex step.

    A different step per query row can violate a track-capacity constraint that
    couples rows, even when both endpoint distributions are feasible.
    """

    previous_joint = np.concatenate(
        [
            np.asarray(previous_candidate, dtype=np.float64),
            np.asarray(previous_null, dtype=np.float64)[:, None],
        ],
        axis=1,
    )
    proposed_joint = np.concatenate(
        [
            np.asarray(proposed_candidate, dtype=np.float64),
            np.asarray(proposed_null, dtype=np.float64)[:, None],
        ],
        axis=1,
    )
    if previous_joint.shape != proposed_joint.shape:
        raise ValueError("latent responsibility states are not aligned")
    delta = proposed_joint - previous_joint
    largest_change = float(np.max(np.abs(delta), initial=0.0))
    scale = min(1.0, float(max_change) / max(largest_change, 1e-12))
    bounded = previous_joint + scale * delta
    bounded /= np.maximum(np.sum(bounded, axis=1, keepdims=True), 1e-12)
    return bounded[:, :-1], bounded[:, -1]


def latent_correspondence_responsibilities(
    pool: Any,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: LatentEMConfig = LatentEMConfig(),
    previous: LatentResponsibilityState | None = None,
) -> LatentResponsibilityState:
    """E-step with explicit null and inequality-constrained track capacity."""

    if not bool(pool.has_explicit_null):
        raise ValueError("latent correspondence refinement requires explicit null")
    prior, prior_null = _tempered_identity_prior(pool, config)
    likelihoods, candidate_xy, inlier_probabilities = _candidate_likelihoods_and_xy(
        pool, pose_w2c, camera, config
    )
    numerator = prior * likelihoods
    null_numerator = prior_null * float(config.null_likelihood)
    denominator = np.sum(numerator, axis=1) + null_numerator
    candidate = numerator / np.maximum(denominator[:, None], 1e-12)
    null = null_numerator / np.maximum(denominator, 1e-12)
    valid = np.asarray(pool.valid_mask, dtype=bool)
    candidate[~valid] = 0.0

    candidate, null = _enforce_null_floor(
        candidate, null, float(config.null_mass_floor)
    )
    max_track_mass = 0.0
    if bool(config.enforce_track_capacity):
        candidate, null, max_track_mass = _enforce_track_capacity(
            candidate,
            null,
            np.asarray(pool.track_ids, dtype=np.int64),
            valid,
        )

    if previous is not None:
        candidate, null = _bounded_joint_update(
            previous.candidate_responsibilities,
            previous.null_responsibilities,
            candidate,
            null,
            float(config.max_responsibility_change),
        )
        max_track_mass = _maximum_track_mass(
            candidate,
            np.asarray(pool.track_ids, dtype=np.int64),
            valid,
        )
    effective_candidate = candidate * inlier_probabilities
    group_mass = np.sum(effective_candidate, axis=1)
    joint = np.concatenate([candidate, null[:, None]], axis=1)
    entropy = -np.sum(joint * np.log(np.maximum(joint, 1e-12)), axis=1)
    return LatentResponsibilityState(
        candidate_responsibilities=candidate,
        candidate_inlier_probabilities=inlier_probabilities,
        null_responsibilities=null,
        candidate_xy=candidate_xy,
        candidate_likelihoods=likelihoods,
        log_likelihood_sum=float(np.sum(np.log(np.maximum(denominator, 1e-12)))),
        effective_group_count=int(
            np.sum(group_mass >= float(config.min_effective_group_mass))
        ),
        mean_group_entropy=float(np.mean(entropy)),
        max_track_mass=float(max_track_mass),
    )


def _pose_step(
    previous_pose: np.ndarray, proposed_pose: np.ndarray
) -> tuple[float, float]:
    previous = np.asarray(previous_pose, dtype=np.float64).reshape(4, 4)
    proposed = np.asarray(proposed_pose, dtype=np.float64).reshape(4, 4)
    previous_center = -previous[:3, :3].T @ previous[:3, 3]
    proposed_center = -proposed[:3, :3].T @ proposed[:3, 3]
    translation = float(np.linalg.norm(previous_center - proposed_center))
    relative = proposed[:3, :3] @ previous[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation = float(np.degrees(np.arccos(cosine)))
    return translation, rotation


def _fractional_latent_edges(
    pool: Any,
    state: LatentResponsibilityState,
    config: LatentEMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the constrained soft-assignment edges used by the M-step."""

    effective_weights = (
        np.asarray(state.candidate_responsibilities, dtype=np.float64)
        * np.asarray(state.candidate_inlier_probabilities, dtype=np.float64)
    )
    rows, columns = np.nonzero(
        np.asarray(pool.valid_mask, dtype=bool)
        & (effective_weights >= float(config.min_candidate_weight))
    )
    order = np.argsort(
        -effective_weights[rows, columns], kind="mergesort"
    )
    rows = rows[order].astype(np.int64, copy=False)
    columns = columns[order].astype(np.int64, copy=False)
    return (
        np.asarray(pool.xyz, dtype=np.float64)[rows, columns],
        np.asarray(state.candidate_xy, dtype=np.float64)[rows, columns],
        effective_weights[rows, columns].astype(np.float64, copy=False),
        rows,
        np.asarray(pool.track_ids, dtype=np.int64)[rows, columns],
    )


def _refine_fractional_latent_pose(
    pool: Any,
    state: LatentResponsibilityState,
    initial_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: LatentEMConfig,
) -> tuple[np.ndarray | None, str | None]:
    """Optimize the expected reprojection objective without hard identity collapse."""

    try:
        import cv2
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV and SciPy are required for latent EM") from exc

    xyz, xy, weights, rows, track_ids = _fractional_latent_edges(
        pool, state, config
    )
    if (
        len(weights) < int(config.min_effective_groups)
        or len(np.unique(rows)) < int(config.min_effective_groups)
        or len(np.unique(track_ids)) < int(config.min_effective_groups)
    ):
        return None, "insufficient_unique_groups_or_tracks"
    finite = (
        np.all(np.isfinite(xyz), axis=1)
        & np.all(np.isfinite(xy), axis=1)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    xyz = xyz[finite]
    xy = xy[finite]
    weights = weights[finite]
    rows = rows[finite]
    track_ids = track_ids[finite]
    if (
        len(weights) < int(config.min_effective_groups)
        or len(np.unique(rows)) < int(config.min_effective_groups)
        or len(np.unique(track_ids)) < int(config.min_effective_groups)
    ):
        return None, "insufficient_unique_groups_or_tracks"

    pose0 = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    rvec0, _jacobian = cv2.Rodrigues(pose0[:3, :3])
    params0 = np.concatenate([rvec0.reshape(3), pose0[:3, 3]]).astype(np.float64)
    matrix, distortion = camera_matrix_and_distortion(camera)
    normalized_weights = weights / max(float(np.mean(weights)), 1e-12)
    sqrt_weights = np.sqrt(normalized_weights)

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = np.asarray(params[:3], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(params[3:6], dtype=np.float64).reshape(3, 1)
        projected, _jacobian = cv2.projectPoints(
            xyz, rvec, tvec, matrix, distortion
        )
        errors = projected.reshape(-1, 2) - xy
        return (errors * sqrt_weights[:, None]).reshape(-1)

    try:
        result = least_squares(
            residuals,
            params0,
            loss=str(config.robust_loss).lower(),
            f_scale=float(config.robust_f_scale_px),
            max_nfev=int(config.max_nfev),
            method="trf",
        )
    except Exception:
        return None, "fractional_weighted_pnp_failure"
    if not bool(result.success) or not np.all(np.isfinite(result.x)):
        return None, "fractional_weighted_pnp_failure"
    rotation, _jacobian = cv2.Rodrigues(
        np.asarray(result.x[:3], dtype=np.float64).reshape(3, 1)
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(result.x[3:6], dtype=np.float64)
    return pose, None


def refine_pose_latent_em(
    pool: Any,
    initial_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: LatentEMConfig = LatentEMConfig(),
) -> LatentEMResult:
    """Run bounded EM while preserving the caller's initial hypothesis."""

    pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    state = latent_correspondence_responsibilities(pool, pose, camera, config)
    initial_objective = float(state.log_likelihood_sum)
    accepted_iterations = 0
    failure_reason: str | None = None
    for _iteration in range(int(config.iterations)):
        if state.effective_group_count < int(config.min_effective_groups):
            failure_reason = "insufficient_effective_groups"
            break
        proposed_pose, refine_failure = _refine_fractional_latent_pose(
            pool, state, pose, camera, config
        )
        if proposed_pose is None:
            failure_reason = refine_failure
            break
        translation_step, rotation_step = _pose_step(pose, proposed_pose)
        if (
            translation_step > float(config.max_translation_step_m)
            or rotation_step > float(config.max_rotation_step_deg)
        ):
            failure_reason = "pose_step_safety_limit"
            break
        proposed_state = latent_correspondence_responsibilities(
            pool,
            proposed_pose,
            camera,
            config,
            previous=state,
        )
        if proposed_state.log_likelihood_sum + float(
            config.objective_tolerance
        ) < state.log_likelihood_sum:
            failure_reason = "non_monotonic_objective"
            break
        pose = proposed_pose
        state = proposed_state
        accepted_iterations += 1
    return LatentEMResult(
        success=bool(accepted_iterations > 0),
        pose_w2c=pose,
        state=state,
        accepted_iterations=int(accepted_iterations),
        initial_log_likelihood_sum=initial_objective,
        final_log_likelihood_sum=float(state.log_likelihood_sum),
        failure_reason=failure_reason,
    )
