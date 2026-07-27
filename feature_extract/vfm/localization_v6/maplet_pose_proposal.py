"""Coarse pose modes from probabilistic query-region/maplet groups."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.se3_update import (
    projection_jacobian,
    se3_exp,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    QueryMapletGroup,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


@dataclass(frozen=True)
class MapletPoseHypothesis:
    pose_w2c: np.ndarray
    score: float
    supporting_group_count: int
    sampled_maplet_ids: np.ndarray
    source: str = "stochastic_surface_component"


def _group_candidate_geometry(
    group: QueryMapletGroup,
    bank: SurfaceRetrievalMapletBank,
    row_by_id: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve query-conditioned regional 3D moments for one retrieval group."""

    rows = np.asarray(
        [row_by_id.get(int(value), -1) for value in group.maplet_ids],
        dtype=np.int64,
    )
    valid = rows >= 0
    valid_rows = rows[valid]
    if group.candidate_centers is None:
        centers = np.asarray(bank.centers[valid_rows], dtype=np.float64)
    else:
        all_centers = np.asarray(group.candidate_centers, dtype=np.float64)
        if all_centers.shape != (group.maplet_ids.size, 3):
            raise ValueError("candidate_centers differ from group candidates")
        centers = all_centers[valid]
    if group.candidate_covariances is None:
        radius = np.max(bank.extents[valid_rows], axis=1)
        covariances = (
            np.eye(3, dtype=np.float64)[None]
            * np.maximum(radius, 1e-3)[:, None, None] ** 2
        )
    else:
        all_covariances = np.asarray(
            group.candidate_covariances, dtype=np.float64
        )
        if all_covariances.shape != (group.maplet_ids.size, 3, 3):
            raise ValueError(
                "candidate_covariances differ from group candidates"
            )
        covariances = all_covariances[valid]
    return valid_rows, centers, covariances, valid


def _group_component_geometry(
    group: QueryMapletGroup,
    bank: SurfaceRetrievalMapletBank,
    row_by_id: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand a region's maplet posterior into real surface-location modes."""

    if (
        group.component_offsets is None
        or group.component_centers is None
        or group.component_covariances is None
        or group.component_probabilities is None
    ):
        rows, centers, covariances, valid = _group_candidate_geometry(
            group, bank, row_by_id
        )
        return (
            np.asarray(group.maplet_ids, dtype=np.int64)[valid],
            centers,
            covariances,
            np.asarray(group.probabilities, dtype=np.float64)[valid],
        )
    offsets = np.asarray(group.component_offsets, dtype=np.int64)
    centers = np.asarray(group.component_centers, dtype=np.float64)
    covariances = np.asarray(group.component_covariances, dtype=np.float64)
    conditional = np.asarray(
        group.component_probabilities, dtype=np.float64
    )
    if (
        offsets.shape != (group.maplet_ids.size + 1,)
        or offsets[0] != 0
        or offsets[-1] != centers.shape[0]
        or covariances.shape != (centers.shape[0], 3, 3)
        or conditional.shape != (centers.shape[0],)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("invalid query-conditioned component geometry")
    maplet_ids = []
    joint_probability = []
    for candidate, maplet_id in enumerate(group.maplet_ids.tolist()):
        if int(maplet_id) not in row_by_id:
            continue
        begin, end = int(offsets[candidate]), int(offsets[candidate + 1])
        local = np.maximum(conditional[begin:end], 0.0)
        local /= max(float(np.sum(local)), 1e-12)
        maplet_ids.extend([int(maplet_id)] * (end - begin))
        joint_probability.extend(
            (float(group.probabilities[candidate]) * local).tolist()
        )
    keep = np.asarray(
        [
            component
            for candidate, maplet_id in enumerate(group.maplet_ids.tolist())
            if int(maplet_id) in row_by_id
            for component in range(
                int(offsets[candidate]), int(offsets[candidate + 1])
            )
        ],
        dtype=np.int64,
    )
    return (
        np.asarray(maplet_ids, dtype=np.int64),
        centers[keep],
        covariances[keep],
        np.asarray(joint_probability, dtype=np.float64),
    )


def _sample_group_component_center(
    group: QueryMapletGroup,
    candidate: int,
    bank: SurfaceRetrievalMapletBank,
    row_by_id: dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    if (
        group.component_offsets is not None
        and group.component_centers is not None
        and group.component_probabilities is not None
    ):
        offsets = np.asarray(group.component_offsets, dtype=np.int64)
        begin, end = int(offsets[candidate]), int(offsets[candidate + 1])
        probability = np.maximum(
            np.asarray(
                group.component_probabilities[begin:end], dtype=np.float64
            ),
            0.0,
        )
        probability /= max(float(np.sum(probability)), 1e-12)
        component = int(rng.choice(end - begin, p=probability)) + begin
        return np.asarray(
            group.component_centers[component], dtype=np.float64
        )
    if group.candidate_centers is not None:
        return np.asarray(
            group.candidate_centers[candidate], dtype=np.float64
        )
    row = row_by_id[int(group.maplet_ids[candidate])]
    return np.asarray(bank.centers[row], dtype=np.float64)


def _projected_candidate_radius(
    covariances: np.ndarray,
    depth: np.ndarray,
    camera: ColmapCamera,
) -> np.ndarray:
    world_sigma = np.sqrt(
        np.maximum(
            np.max(
                np.linalg.eigvalsh(
                    np.asarray(covariances, dtype=np.float64)
                ),
                axis=1,
            ),
            1e-8,
        )
    )
    output = np.zeros_like(np.asarray(depth, dtype=np.float64))
    in_front = np.isfinite(depth) & (depth > 0.10)
    output[in_front] = (
        float(camera.params[0])
        * world_sigma[in_front]
        / depth[in_front]
    )
    return output


def _refine_hypothesis_em(
    hypothesis: MapletPoseHypothesis,
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
    *,
    iterations: int = 5,
    damping: float = 1e-3,
    maximum_translation_step_m: float = 0.50,
    maximum_rotation_step_deg: float = 5.0,
) -> MapletPoseHypothesis:
    """Latent region→maplet EM with covariance-aware joint SE(3) updates."""

    row_by_id = {
        int(value): int(row)
        for row, value in enumerate(bank.maplet_ids.tolist())
    }
    pose = np.asarray(hypothesis.pose_w2c, dtype=np.float64).copy()
    for _iteration in range(max(int(iterations), 0)):
        normal = np.eye(6, dtype=np.float64) * float(damping)
        rhs = np.zeros((6,), dtype=np.float64)
        effective_groups = 0
        for group in groups:
            (
                component_maplet_ids,
                candidate_centers,
                candidate_covariances,
                prior_probability,
            ) = (
                _group_component_geometry(group, bank, row_by_id)
            )
            if component_maplet_ids.size == 0:
                continue
            projected, jacobian = projection_jacobian(
                candidate_centers, pose, camera
            )
            _pixels, depth = project_world_points(
                candidate_centers, pose, camera
            )
            residual = (
                np.asarray(group.query_region_xy, dtype=np.float64)[None]
                - projected
            )
            in_front = np.isfinite(depth) & (depth > 0.10)
            projected_radius = _projected_candidate_radius(
                candidate_covariances, depth, camera
            )
            region_radius = float(np.linalg.norm(group.query_region_extent))
            sigma = np.maximum(projected_radius + region_radius, 2.0)
            finite = (
                in_front
                & np.isfinite(projected).all(axis=1)
                & np.isfinite(jacobian).all(axis=(1, 2))
                & np.isfinite(sigma)
                & (
                    projected_radius
                    <= 2.0 * max(camera.width, camera.height)
                )
            )
            squared = np.sum(residual * residual, axis=1)
            spatial = np.zeros(component_maplet_ids.size, dtype=np.float64)
            spatial[finite] = np.exp(
                np.maximum(
                    -0.5
                    * squared[finite]
                    / np.maximum(sigma[finite] * sigma[finite], 1e-8),
                    -80.0,
                )
            )
            likelihood = prior_probability * spatial
            denominator = float(np.sum(likelihood)) + 0.1 * float(
                group.null_probability
            )
            responsibility = likelihood / max(denominator, 1e-12)
            keep = finite & (responsibility > 1e-6)
            if not np.any(keep):
                continue
            # Normalize evidence within a query region.  A region with many
            # nearly identical maplet modes must not outweigh independent
            # regions simply through candidate multiplicity.
            responsibility = responsibility[keep]
            residual_keep = residual[keep]
            sigma_keep = sigma[keep]
            jacobian_keep = jacobian[keep]
            normalized_residual = (
                np.linalg.norm(residual_keep, axis=1)
                / np.maximum(sigma_keep, 1e-6)
            )
            huber = np.minimum(
                1.0, 2.5 / np.maximum(normalized_residual, 1e-8)
            )
            weight = responsibility * huber / np.maximum(
                sigma_keep * sigma_keep, 4.0
            )
            group_mass = min(float(np.sum(responsibility)), 1.0)
            weight *= group_mass / max(float(np.sum(weight)), 1e-12)
            normal += np.einsum(
                "n,nai,naj->ij", weight, jacobian_keep, jacobian_keep
            )
            rhs += np.einsum(
                "n,nai,na->i", weight, jacobian_keep, residual_keep
            )
            effective_groups += 1
        if effective_groups < 4:
            break
        try:
            delta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            break
        rotation_norm = float(np.linalg.norm(delta[:3]))
        maximum_rotation = np.deg2rad(float(maximum_rotation_step_deg))
        if rotation_norm > maximum_rotation:
            delta[:3] *= maximum_rotation / rotation_norm
        translation_norm = float(np.linalg.norm(delta[3:]))
        if translation_norm > float(maximum_translation_step_m):
            delta[3:] *= float(maximum_translation_step_m) / translation_norm
        if not np.all(np.isfinite(delta)):
            break
        pose = se3_exp(delta) @ pose
        if np.linalg.norm(delta) < 1e-5:
            break
    score, support = _score_hypothesis(pose, groups, bank, camera)
    return MapletPoseHypothesis(
        pose_w2c=pose,
        score=score,
        supporting_group_count=support,
        sampled_maplet_ids=hypothesis.sampled_maplet_ids,
        source=hypothesis.source,
    )


def _score_hypothesis(
    pose: np.ndarray,
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
) -> tuple[float, int]:
    row_by_id = {
        int(value): int(row) for row, value in enumerate(bank.maplet_ids.tolist())
    }
    total = 0.0
    support = 0
    for group in groups:
        (
            component_maplet_ids,
            candidate_centers,
            candidate_covariances,
            prior_probability,
        ) = (
            _group_component_geometry(group, bank, row_by_id)
        )
        if component_maplet_ids.size == 0:
            total += np.log(max(float(group.null_probability), 1e-8))
            continue
        pixels, depth = project_world_points(
            candidate_centers, pose, camera
        )
        residual = pixels - np.asarray(group.query_region_xy)[None]
        in_front = np.isfinite(depth) & (depth > 0.10)
        projected_radius = _projected_candidate_radius(
            candidate_covariances, depth, camera
        )
        region_radius = float(np.linalg.norm(group.query_region_extent))
        sigma = np.maximum(projected_radius + region_radius, 2.0)
        finite = (
            in_front
            & np.isfinite(pixels).all(axis=1)
            & np.isfinite(sigma)
            # A near-camera/behind-camera solution must not gain likelihood
            # merely because its projected maplet covariance explodes.
            & (projected_radius <= 2.0 * max(camera.width, camera.height))
        )
        squared_residual = np.sum(residual * residual, axis=1)
        log_spatial = np.full(depth.shape, -np.inf, dtype=np.float64)
        log_spatial[finite] = (
            -0.5
            * squared_residual[finite]
            / np.maximum(sigma[finite] * sigma[finite], 1e-8)
        )
        likelihood = prior_probability * np.exp(
            np.maximum(log_spatial, -80.0)
        )
        likelihood[~finite] = 0.0
        mass = float(np.sum(likelihood))
        combined = mass + float(group.null_probability) * 0.1
        total += np.log(max(combined, 1e-8))
        if mass > float(group.null_probability) * 0.1:
            support += 1
    return total, support


def _regional_map_correspondences(
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One deployable MAP surface component per query region.

    A physical surface position may be proposed by several overlapping RADIO
    regions. It is retained only once, using the observation with the largest
    joint maplet/component probability.
    """

    row_by_id = {
        int(value): int(row)
        for row, value in enumerate(bank.maplet_ids.tolist())
    }
    by_surface: dict[
        tuple[int, float, float, float],
        tuple[float, np.ndarray, np.ndarray, int, float],
    ] = {}
    for group in groups:
        (
            component_maplet_ids,
            centers,
            _covariances,
            joint_probability,
        ) = _group_component_geometry(group, bank, row_by_id)
        if component_maplet_ids.size == 0:
            continue
        candidate = int(np.argmax(joint_probability))
        probability = float(joint_probability[candidate])
        if not np.isfinite(probability) or probability <= 0.0:
            continue
        maplet_id = int(component_maplet_ids[candidate])
        center = np.asarray(centers[candidate], dtype=np.float64)
        key = (
            maplet_id,
            *np.round(center, decimals=4).tolist(),
        )
        # Confidence remains an absolute posterior mass; null-heavy regions
        # therefore rank below visually supported regions.
        confidence = probability
        value = (
            confidence,
            center,
            np.asarray(group.query_region_xy, dtype=np.float64),
            maplet_id,
            float(np.linalg.norm(group.query_region_extent)),
        )
        previous = by_surface.get(key)
        if previous is None or confidence > previous[0]:
            by_surface[key] = value
    ordered = sorted(by_surface.values(), key=lambda row: row[0], reverse=True)
    if not ordered:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.stack([row[1] for row in ordered]),
        np.stack([row[2] for row in ordered]),
        np.asarray([row[0] for row in ordered], dtype=np.float64),
        np.asarray([row[3] for row in ordered], dtype=np.int64),
        np.asarray([row[4] for row in ordered], dtype=np.float64),
    )


def _deterministic_regional_ransac_hypotheses(
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
) -> list[MapletPoseHypothesis]:
    """Generate coarse modes from confidence-ranked regional MAP components."""

    xyz, xy, _confidence, maplet_ids, region_radii = (
        _regional_map_correspondences(groups, bank)
    )
    if xyz.shape[0] < 6:
        return []
    matrix, distortion = camera_matrix_and_distortion(camera)
    maximum = int(xyz.shape[0])
    prefixes = sorted(
        {
            min(maximum, value)
            for value in (12, 16, 24, 32, 48, 64, 96, maximum)
            if min(maximum, value) >= 6
        }
    )
    hypotheses = []
    for prefix in prefixes:
        # RADIO-final is a regional feature, not a keypoint. Derive the
        # inlier gate from its image support instead of pretending subpixel
        # measurement precision.
        reprojection_threshold = max(
            6.0, 1.5 * float(np.median(region_radii[:prefix]))
        )
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            xyz[:prefix],
            xy[:prefix],
            matrix,
            distortion,
            iterationsCount=5000,
            reprojectionError=float(reprojection_threshold),
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < 6:
            continue
        keep = np.asarray(inliers, dtype=np.int64).reshape(-1)
        rotation, translation = cv2.solvePnPRefineLM(
            xyz[:prefix][keep],
            xy[:prefix][keep],
            matrix,
            distortion,
            rotation,
            translation,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = np.asarray(translation).reshape(3)
        score, support = _score_hypothesis(pose, groups, bank, camera)
        hypotheses.append(
            MapletPoseHypothesis(
                pose_w2c=pose,
                score=score,
                supporting_group_count=support,
                sampled_maplet_ids=np.unique(maplet_ids[:prefix][keep]),
                source=f"regional_map_ransac_prefix_{prefix}",
            )
        )
    return hypotheses


def propose_maplet_surface_mode_poses(
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
    *,
    trials: int = 0,
    sample_size: int = 4,
    maximum_modes: int = 16,
    refinement_candidates: int = 64,
    em_iterations: int = 0,
    seed: int = 73,
) -> tuple[MapletPoseHypothesis, ...]:
    """Grouped probabilistic surface-mode PnP for basin entry only.

    Each sampled 3D observation is a real query-conditioned atlas location.
    A maplet centre is used only by legacy banks that do not carry spatial
    component geometry.
    """

    usable = tuple(group for group in groups if group.maplet_ids.size)
    if len(usable) < max(int(sample_size), 4):
        return tuple()
    matrix, distortion = camera_matrix_and_distortion(camera)
    row_by_id = {
        int(value): int(row) for row, value in enumerate(bank.maplet_ids.tolist())
    }
    rng = np.random.default_rng(int(seed))
    group_sampling_probability = np.asarray(
        [
            max(
                float(np.max(group.probabilities, initial=0.0)),
                1e-8,
            )
            for group in usable
        ],
        dtype=np.float64,
    )
    group_sampling_probability /= np.sum(group_sampling_probability)
    deterministic_hypotheses = (
        _deterministic_regional_ransac_hypotheses(
            usable, bank, camera
        )
    )
    hypotheses: list[MapletPoseHypothesis] = deterministic_hypotheses
    for _ in range(int(trials)):
        selected_groups = rng.choice(
            len(usable),
            size=int(sample_size),
            replace=False,
            p=group_sampling_probability,
        )
        object_points = []
        image_points = []
        sampled_ids = []
        sampled_surface_keys = set()
        for group_row in selected_groups.tolist():
            group = usable[group_row]
            probability = np.asarray(group.probabilities, dtype=np.float64)
            probability /= max(float(np.sum(probability)), 1e-12)
            choice = int(rng.choice(group.maplet_ids.size, p=probability))
            maplet_id = int(group.maplet_ids[choice])
            row = row_by_id.get(maplet_id)
            if row is None:
                break
            center = _sample_group_component_center(
                group, choice, bank, row_by_id, rng
            )
            surface_key = (
                maplet_id,
                *np.round(center, decimals=4).tolist(),
            )
            if surface_key in sampled_surface_keys:
                break
            sampled_surface_keys.add(surface_key)
            sampled_ids.append(maplet_id)
            object_points.append(center)
            image_points.append(group.query_region_xy)
        if len(object_points) < int(sample_size):
            continue
        centered = np.asarray(object_points) - np.mean(
            object_points, axis=0, keepdims=True
        )
        if np.linalg.matrix_rank(centered, tol=1e-4) < 2:
            continue
        success, rotation, translation = cv2.solvePnP(
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = translation.reshape(3)
        score, support = _score_hypothesis(pose, usable, bank, camera)
        hypotheses.append(
            MapletPoseHypothesis(
                pose_w2c=pose,
                score=score,
                supporting_group_count=support,
                sampled_maplet_ids=np.asarray(sampled_ids, dtype=np.int64),
                source="stochastic_surface_component",
            )
        )
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    if int(em_iterations) > 0 and hypotheses:
        refined_hypotheses = []
        for hypothesis in hypotheses[
            : max(int(refinement_candidates), int(maximum_modes))
        ]:
            refined = _refine_hypothesis_em(
                hypothesis,
                usable,
                bank,
                camera,
                iterations=int(em_iterations),
            )
            # Refinement is an optimizer proposal, not permission to discard
            # a valid basin. Keep the original whenever the deployment
            # likelihood does not improve.
            refined_hypotheses.append(
                refined if refined.score >= hypothesis.score else hypothesis
            )
        hypotheses = refined_hypotheses + hypotheses[
            max(int(refinement_candidates), int(maximum_modes)) :
        ]
        hypotheses.sort(key=lambda item: item.score, reverse=True)
    modes: list[MapletPoseHypothesis] = []
    for candidate in hypotheses:
        center = -candidate.pose_w2c[:3, :3].T @ candidate.pose_w2c[:3, 3]
        duplicate = False
        for kept in modes:
            kept_center = -kept.pose_w2c[:3, :3].T @ kept.pose_w2c[:3, 3]
            rotation = candidate.pose_w2c[:3, :3] @ kept.pose_w2c[:3, :3].T
            angle = np.degrees(
                np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
            )
            if np.linalg.norm(center - kept_center) < 0.20 and angle < 2.0:
                duplicate = True
                break
        if not duplicate:
            modes.append(candidate)
        if len(modes) >= int(maximum_modes):
            break
    return tuple(modes)


# Compatibility for frozen experiments.  The active evaluator imports the
# surface-mode name above, so production reports cannot silently describe this
# as maplet-centre PnP.
propose_maplet_center_poses = propose_maplet_surface_mode_poses
