"""Shared candidate-specific pose evidence for latent fitting and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


CANDIDATE_POSE_EVIDENCE_VERSION = "spatial_kernel_mixture_v8_action_separated"
POSE_CONDITIONED_VIEW_GEOMETRY_VERSION = (
    "track_to_camera_direction_gaussian_v1"
)


@dataclass(frozen=True)
class CandidatePoseEvidence:
    projected_xy: np.ndarray
    projection_valid: np.ndarray
    candidate_likelihoods: np.ndarray
    base_candidate_likelihoods: np.ndarray
    candidate_xy: np.ndarray
    candidate_inlier_probabilities: np.ndarray
    spatial_candidate_mask: np.ndarray
    spatial_calibrated_mask: np.ndarray


def project_candidate_xyz(
    xyz: np.ndarray,
    valid_mask: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a candidate matrix with one shared visibility contract."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for candidate projection") from exc

    points = np.asarray(xyz, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.shape != (*valid.shape, 3):
        raise ValueError("candidate XYZ and validity mask are not aligned")
    projected = np.full((*valid.shape, 2), np.nan, dtype=np.float64)
    projection_valid = np.zeros(valid.shape, dtype=bool)
    rows, columns = np.nonzero(valid)
    if len(rows) == 0:
        return projected, projection_valid

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    selected = points[rows, columns]
    camera_points = selected @ pose[:3, :3].T + pose[:3, 3]
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    pixels, _jacobian = cv2.projectPoints(
        selected,
        rvec,
        pose[:3, 3],
        matrix,
        distortion,
    )
    pixels = pixels.reshape(-1, 2).astype(np.float64)
    finite = np.all(np.isfinite(pixels), axis=1)
    visible = (
        finite
        & (camera_points[:, 2] > 1e-8)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(camera.width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(camera.height))
    )
    projected[rows[visible], columns[visible]] = pixels[visible]
    projection_valid[rows[visible], columns[visible]] = True
    return projected, projection_valid


def measurement_reliability(
    raw_reliability: np.ndarray,
    geometry_probability: float,
    calibration_weight: float,
) -> np.ndarray:
    reliability = np.asarray(raw_reliability, dtype=np.float64)
    weight = float(calibration_weight)
    if weight > 0.0 and np.isfinite(geometry_probability):
        reliability = (
            (1.0 - weight) * reliability + weight * float(geometry_probability)
        )
    return np.clip(reliability, 0.0, 1.0)


def measurement_utility_action_gate(
    utility_probability: float,
    update_threshold: float,
    gate_weight: float,
) -> float:
    """Return an interpolation for a coordinate-update action only.

    The calibrated utility is thresholded into a decision. ``gate_weight`` is
    only an ablation interpolation between disabled and fully selective use.
    The result must never scale identity mass or pose likelihood.
    """

    weight = float(gate_weight)
    threshold = float(update_threshold)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("measurement utility gate weight must be in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("measurement utility threshold must be in [0, 1]")
    if weight <= 0.0:
        return 1.0
    probability = float(utility_probability)
    approved = bool(np.isfinite(probability) and probability >= threshold)
    return float((1.0 - weight) + weight * float(approved))


def pose_conditioned_view_probabilities(
    probabilities: np.ndarray,
    support_camera_centers: np.ndarray,
    *,
    track_xyz: np.ndarray,
    query_camera_center: np.ndarray,
    sigma_deg: float,
) -> np.ndarray:
    """Reweight a frozen view posterior by hypothesized viewing direction.

    The operation preserves the available support-view mass. It does not add
    probability to missing views and never changes candidate identity or null
    mass.
    """

    prior = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    centers = np.asarray(support_camera_centers, dtype=np.float64).reshape(-1, 3)
    xyz = np.asarray(track_xyz, dtype=np.float64).reshape(3)
    query_center = np.asarray(query_camera_center, dtype=np.float64).reshape(3)
    sigma = float(sigma_deg)
    if len(prior) != len(centers):
        raise ValueError("view probabilities and support camera centers differ")
    if np.any(~np.isfinite(prior)) or np.any(prior < 0.0):
        raise ValueError("view probabilities must be finite and non-negative")
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(xyz)) or np.any(
        ~np.isfinite(query_center)
    ):
        raise ValueError("pose-conditioned view geometry must be finite")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("pose-conditioned view geometry sigma must be positive")
    available_mass = float(np.sum(prior))
    if available_mass <= 0.0:
        return np.zeros_like(prior)

    query_direction = query_center - xyz
    query_norm = float(np.linalg.norm(query_direction))
    support_directions = centers - xyz[None, :]
    support_norms = np.linalg.norm(support_directions, axis=1)
    if query_norm <= 1e-12 or np.any(support_norms <= 1e-12):
        return prior.copy()
    query_direction /= query_norm
    support_directions /= support_norms[:, None]
    angles_deg = np.degrees(
        np.arccos(
            np.clip(support_directions @ query_direction, -1.0, 1.0)
        )
    )
    positive = prior > 0.0
    if not np.any(positive):
        return np.zeros_like(prior)
    log_weight = np.full_like(prior, -np.inf)
    log_weight[positive] = (
        np.log(prior[positive])
        - 0.5 * np.square(angles_deg[positive] / sigma)
    )
    log_weight[positive] -= float(np.max(log_weight[positive]))
    weight = np.zeros_like(prior)
    weight[positive] = np.exp(log_weight[positive])
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0.0:
        return prior.copy()
    return available_mass * weight / weight_sum


def candidate_pose_evidence(
    pool: Any,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float,
    outlier_likelihood: float,
    spatial_evidence_weight: float | None = None,
) -> CandidatePoseEvidence:
    """Evaluate mutually exclusive candidate evidence under a pose.

    Identity and null mass remain immutable. Dustbin and optional geometry
    calibration control whether the candidate-specific RGB density replaces
    the base query-coordinate likelihood. Measurement-update utility is not
    consumed here: it is an action posterior used only after an identity has
    been selected for an optional coordinate-refinement step.
    """

    sigma = float(residual_sigma_px)
    outlier = float(outlier_likelihood)
    if sigma <= 0.0:
        raise ValueError("residual_sigma_px must be positive")
    if not 0.0 < outlier <= 1.0:
        raise ValueError("outlier_likelihood must be in (0, 1]")

    valid = np.asarray(pool.valid_mask, dtype=bool)
    base_xy = np.asarray(pool.xy, dtype=np.float64).reshape(-1, 2)
    if valid.shape[0] != len(base_xy):
        raise ValueError("candidate pool and query coordinates are not aligned")
    projected, projection_valid = project_candidate_xyz(
        np.asarray(pool.xyz, dtype=np.float64),
        valid,
        pose_w2c,
        camera,
    )
    candidate_xy = np.broadcast_to(
        base_xy[:, None, :], (*valid.shape, 2)
    ).copy()
    residuals = np.linalg.norm(projected - candidate_xy, axis=2)
    gaussian = np.zeros(valid.shape, dtype=np.float64)
    gaussian[projection_valid] = np.exp(
        -0.5 * np.square(residuals[projection_valid] / sigma)
    )
    base_likelihood = outlier + (1.0 - outlier) * gaussian
    base_likelihood[~valid] = 0.0
    base_inlier = np.zeros(valid.shape, dtype=np.float64)
    base_inlier[projection_valid] = (
        (1.0 - outlier) * gaussian[projection_valid]
    ) / np.maximum(base_likelihood[projection_valid], 1e-12)

    spatial = getattr(pool, "spatial_likelihood", None)
    spatial_candidate_mask = np.zeros(valid.shape, dtype=bool)
    spatial_calibrated_mask = np.zeros(valid.shape, dtype=bool)
    if spatial is None:
        return CandidatePoseEvidence(
            projected,
            projection_valid,
            base_likelihood,
            base_likelihood,
            candidate_xy,
            base_inlier,
            spatial_candidate_mask,
            spatial_calibrated_mask,
        )
    weight = (
        float(spatial.log_evidence_weight)
        if spatial_evidence_weight is None
        else float(spatial_evidence_weight)
    )
    if not 0.0 <= weight <= 1.0:
        raise ValueError("spatial evidence weight must be in [0, 1]")
    spatial_candidate_mask = (
        np.any(np.asarray(spatial.valid_mask, dtype=bool), axis=2) & valid
    )
    if weight <= 0.0 or not np.any(spatial_candidate_mask):
        return CandidatePoseEvidence(
            projected,
            projection_valid,
            base_likelihood,
            base_likelihood,
            candidate_xy,
            base_inlier,
            spatial_candidate_mask,
            spatial_calibrated_mask,
        )

    offsets_xy = np.asarray(spatial.offsets_xy, dtype=np.float64)
    offset_min = np.min(offsets_xy, axis=0)
    offset_max = np.max(offsets_xy, axis=0)
    geometry_probability = np.asarray(
        pool.measurement_geometry_probabilities, dtype=np.float64
    )
    calibration_weight = float(pool.spatial_geometry_calibration_weight)
    spatial_calibrated_mask = (
        spatial_candidate_mask
        & np.isfinite(geometry_probability)
        & (calibration_weight > 0.0)
    )

    likelihoods = base_likelihood.copy()
    inlier_probabilities = base_inlier.copy()
    pose_view_geometry_sigma_deg = float(
        getattr(spatial, "pose_view_geometry_sigma_deg", 0.0)
    )
    support_camera_centers = getattr(
        spatial, "support_camera_centers", None
    )
    query_camera_center = None
    if pose_view_geometry_sigma_deg > 0.0:
        if support_camera_centers is None:
            raise ValueError(
                "pose-conditioned view geometry requires support camera centers"
            )
        pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        query_camera_center = -pose[:3, :3].T @ pose[:3, 3]
    for row, column in np.argwhere(spatial_candidate_mask).tolist():
        view_mask = np.asarray(spatial.valid_mask[row, column], dtype=bool)
        view_indices = np.flatnonzero(view_mask)
        view_prior = np.maximum(
            np.asarray(
                spatial.view_probabilities[row, column, view_indices],
                dtype=np.float64,
            ),
            0.0,
        )
        if query_camera_center is not None:
            view_prior = pose_conditioned_view_probabilities(
                view_prior,
                np.asarray(support_camera_centers, dtype=np.float64)[
                    row, column, view_indices
                ],
                track_xyz=np.asarray(pool.xyz, dtype=np.float64)[row, column],
                query_camera_center=query_camera_center,
                sigma_deg=pose_view_geometry_sigma_deg,
            )
        available_view_mass = float(np.sum(view_prior))
        if available_view_mass > 1.0 + 2e-5:
            raise ValueError("available support-view probability mass exceeds one")
        missing_view_mass = max(1.0 - available_view_mass, 0.0)
        raw_reliability = 1.0 - np.asarray(
            spatial.dustbin_probabilities[row, column, view_indices],
            dtype=np.float64,
        )
        reliability = measurement_reliability(
            raw_reliability,
            float(geometry_probability[row, column]),
            calibration_weight,
        )
        spatial_offset = projected[row, column] - base_xy[row]
        inside = bool(
            projection_valid[row, column]
            and np.all(spatial_offset >= offset_min)
            and np.all(spatial_offset <= offset_max)
        )
        base_value = float(base_likelihood[row, column])
        base_inlier_numerator = (1.0 - outlier) * float(
            gaussian[row, column]
        )
        mode_xy = base_xy[row][None, :] + offsets_xy
        mode_residuals = np.linalg.norm(
            mode_xy - projected[row, column][None, :], axis=1
        )
        geometric = (
            np.exp(-0.5 * np.square(mode_residuals / sigma))
            if inside
            else np.zeros_like(mode_residuals)
        )
        # Keep all three quantities under one mixture model. In particular,
        # view/dustbin weights must multiply unnormalised inlier mass before
        # deriving either the inlier posterior or the expected measurement.
        # The base branch is a point mass at zero offset convolved with the
        # same Gaussian kernel used to evaluate the categorical RGB offset
        # distribution. Dividing only the RGB branch by uniform-map evidence
        # would turn it into a likelihood ratio and make it incompatible with
        # the base likelihood in this mixture.
        spatial_likelihood = missing_view_mass * base_value
        spatial_inlier_numerator = (
            missing_view_mass * base_inlier_numerator
        )
        spatial_coordinate_numerator = (
            missing_view_mass * base_inlier_numerator * base_xy[row]
        )
        for local_view, view in enumerate(view_indices.tolist()):
            log_probability = np.array(
                spatial.local_log_probabilities[row, column, view],
                dtype=np.float64,
                copy=True,
            )
            log_probability -= float(np.max(log_probability))
            mode_probability = np.exp(log_probability)
            mode_probability /= max(float(np.sum(mode_probability)), 1e-12)
            mode_evidence = float(np.sum(mode_probability * geometric))
            mode_inlier_numerator = (1.0 - outlier) * mode_evidence
            mode_likelihood = outlier + mode_inlier_numerator
            local_reliability = float(reliability[local_view])
            local_view_prior = float(view_prior[local_view])
            local_likelihood = (
                (1.0 - local_reliability) * base_value
                + local_reliability * mode_likelihood
            )
            local_base_inlier_mass = (
                (1.0 - local_reliability) * base_inlier_numerator
            )
            local_mode_inlier_mass = (
                local_reliability * mode_inlier_numerator
            )
            spatial_likelihood += local_view_prior * local_likelihood
            spatial_inlier_numerator += local_view_prior * (
                local_base_inlier_mass + local_mode_inlier_mass
            )
            spatial_coordinate_numerator += (
                local_view_prior * local_base_inlier_mass * base_xy[row]
            )
            if mode_evidence > 1e-12 and local_mode_inlier_mass > 0.0:
                mode_mean_xy = np.sum(
                    (mode_probability * geometric)[:, None] * mode_xy, axis=0
                ) / mode_evidence
                spatial_coordinate_numerator += (
                    local_view_prior * local_mode_inlier_mass * mode_mean_xy
                )

        combined_likelihood = (
            (1.0 - weight) * base_value + weight * spatial_likelihood
        )
        combined_inlier_numerator = (
            (1.0 - weight) * base_inlier_numerator
            + weight * spatial_inlier_numerator
        )
        combined_coordinate_numerator = (
            (1.0 - weight) * base_inlier_numerator * base_xy[row]
            + weight * spatial_coordinate_numerator
        )
        likelihoods[row, column] = combined_likelihood
        if combined_inlier_numerator > 1e-12:
            candidate_xy[row, column] = (
                combined_coordinate_numerator / combined_inlier_numerator
            )
        inlier_probabilities[row, column] = (
            combined_inlier_numerator / max(combined_likelihood, 1e-12)
        )

    likelihoods[~valid] = 0.0
    inlier_probabilities = np.clip(inlier_probabilities, 0.0, 1.0)
    inlier_probabilities[~valid] = 0.0
    return CandidatePoseEvidence(
        projected,
        projection_valid,
        likelihoods,
        base_likelihood,
        candidate_xy,
        inlier_probabilities,
        spatial_candidate_mask,
        spatial_calibrated_mask,
    )
