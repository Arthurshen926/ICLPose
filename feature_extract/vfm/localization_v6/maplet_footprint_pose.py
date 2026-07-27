"""Region-level coarse pose from projected maplet footprint likelihood.

This module never interprets a RADIO region centre as the measurement of a
particular 3D point.  A pose explains a region when the projected *area* of
one or more identity candidates overlaps that region.  Identity candidates
are marginalized, and omitted/identity-null mass is an uninformative outcome.
"""

from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_pose_proposal import (
    MapletPoseHypothesis,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    QueryMapletGroup,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left) @ np.asarray(right).T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def score_maplet_footprint_pose(
    pose_w2c: np.ndarray,
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
    *,
    candidates_per_region: int = 16,
) -> tuple[float, int]:
    """Marginal footprint likelihood ratio for one pose.

    The unmatched event has likelihood ratio one by definition.  Therefore
    identity-null and candidates omitted from the footprint evaluation enter
    without a heuristic multiplier.
    """

    if not bool(
        (bank.metadata or {}).get(
            "has_canonical_tangent_frames", False
        )
    ):
        raise ValueError(
            "maplet footprint likelihood requires stored canonical "
            "tangent frames; a normal-only inferred frame is not sufficient"
        )
    row_by_id = {
        int(value): int(row)
        for row, value in enumerate(bank.maplet_ids.tolist())
    }
    total = 0.0
    supporting_regions = 0
    limit = max(int(candidates_per_region), 1)
    for group in groups:
        candidate_ids = np.asarray(group.maplet_ids, dtype=np.int64)[:limit]
        candidate_probability = np.asarray(
            group.probabilities, dtype=np.float64
        )[: candidate_ids.size]
        rows = np.asarray(
            [row_by_id.get(int(value), -1) for value in candidate_ids],
            dtype=np.int64,
        )
        valid_identity = rows >= 0
        omitted = float(group.null_probability) + float(
            np.sum(np.asarray(group.probabilities)[candidate_ids.size :])
        )
        # An identity missing from the geometry bank is unavailable to this
        # scorer, not evidence against the pose.  Keep its probability mass in
        # the unit-likelihood unmatched branch.
        omitted += float(np.sum(candidate_probability[~valid_identity]))
        if not np.any(valid_identity):
            total += np.log(max(omitted, 1e-12))
            continue
        rows = rows[valid_identity]
        probability = candidate_probability[valid_identity]
        centers = np.asarray(bank.centers[rows], dtype=np.float64)
        pixels, depth = project_world_points(centers, pose_w2c, camera)
        frames = np.asarray(
            bank.tangent_frames[rows], dtype=np.float64
        )
        extents = np.asarray(bank.extents[rows], dtype=np.float64)
        corner_sign = np.asarray(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            dtype=np.float64,
        )
        corners = (
            centers[:, None]
            + corner_sign[None, :, 0, None]
            * extents[:, None, 0, None]
            * frames[:, None, 0]
            + corner_sign[None, :, 1, None]
            * extents[:, None, 1, None]
            * frames[:, None, 1]
        )
        corner_pixels, corner_depth = project_world_points(
            corners.reshape(-1, 3), pose_w2c, camera
        )
        corner_pixels = corner_pixels.reshape(-1, 4, 2)
        corner_depth = corner_depth.reshape(-1, 4)
        camera_points = (
            np.asarray(pose_w2c, dtype=np.float64)[:3, :3]
            @ centers.T
            + np.asarray(pose_w2c, dtype=np.float64)[:3, 3:4]
        ).T
        normal_camera = (
            np.asarray(pose_w2c, dtype=np.float64)[:3, :3]
            @ np.asarray(bank.normals[rows], dtype=np.float64).T
        ).T
        view_direction = -camera_points / np.maximum(
            np.linalg.norm(camera_points, axis=1, keepdims=True), 1e-8
        )
        incidence = np.abs(np.sum(normal_camera * view_direction, axis=1))
        edges = np.roll(corner_pixels, -1, axis=1) - corner_pixels
        signed_cross = (
            edges[:, :, 0]
            * (
                np.asarray(group.query_region_xy, dtype=np.float64)[1]
                - corner_pixels[:, :, 1]
            )
            - edges[:, :, 1]
            * (
                np.asarray(group.query_region_xy, dtype=np.float64)[0]
                - corner_pixels[:, :, 0]
            )
        )
        inside = np.all(signed_cross >= -1e-6, axis=1) | np.all(
            signed_cross <= 1e-6, axis=1
        )
        point_residual = (
            np.asarray(group.query_region_xy, dtype=np.float64)[None, None]
            - corner_pixels
        )
        edge_norm2 = np.sum(edges * edges, axis=2)
        projection = np.clip(
            np.sum(point_residual * edges, axis=2)
            / np.maximum(edge_norm2, 1e-8),
            0.0,
            1.0,
        )
        nearest = corner_pixels + projection[:, :, None] * edges
        boundary_distance = np.min(
            np.linalg.norm(
                nearest
                - np.asarray(
                    group.query_region_xy, dtype=np.float64
                )[None, None],
                axis=2,
            ),
            axis=1,
        )
        outside_distance = np.where(inside, 0.0, boundary_distance)
        polygon_area = 0.5 * np.abs(
            np.sum(
                corner_pixels[:, :, 0]
                * np.roll(corner_pixels[:, :, 1], -1, axis=1)
                - corner_pixels[:, :, 1]
                * np.roll(corner_pixels[:, :, 0], -1, axis=1),
                axis=1,
            )
        )
        perimeter = np.sum(np.sqrt(np.maximum(edge_norm2, 0.0)), axis=1)
        # The RADIO descriptor centre is an area observation.  Its pooled
        # image support dilates the projected canonical chart rather than
        # becoming a point-to-centre residual.  Steiner's formula gives the
        # area of the quad dilated by an isotropic support disk.
        region_radius = max(
            0.5
            * float(
                np.linalg.norm(
                    np.asarray(group.query_region_extent, dtype=np.float64)
                )
            ),
            2.0,
        )
        expanded_area = (
            polygon_area
            + perimeter * region_radius
            + np.pi * region_radius * region_radius
        )
        finite = (
            np.isfinite(pixels).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0.10)
            & np.isfinite(corner_pixels).all(axis=(1, 2))
            & np.isfinite(corner_depth).all(axis=1)
            & np.all(corner_depth > 0.10, axis=1)
            & np.isfinite(polygon_area)
            & (polygon_area > 1e-3)
            & (
                polygon_area
                <= 16.0 * max(float(camera.width * camera.height), 1.0)
            )
        )
        footprint_likelihood_ratio = np.zeros(
            rows.size, dtype=np.float64
        )
        # This is a foreground/background *density ratio*.  The previous
        # implementation omitted the Gaussian normalizer and consequently
        # bounded every candidate likelihood by one.  In a mixture with the
        # unmatched branch (whose ratio is one), a match could therefore
        # never provide positive evidence.  Normalizing by the uniform image
        # density restores the intended probability semantics.
        image_area = max(float(camera.width * camera.height), 1.0)
        footprint_likelihood_ratio[finite] = (
            image_area / np.maximum(expanded_area[finite], 1.0)
        ) * np.exp(
            np.maximum(
                -0.5
                * (
                    outside_distance[finite] / region_radius
                )
                ** 2,
                -80.0,
            )
        )
        # A two-sided surface remains possible, but an edge-on footprint
        # carries less area evidence than a fronto-parallel one.
        footprint_likelihood_ratio *= (
            0.25 + 0.75 * np.clip(incidence, 0.0, 1.0)
        )
        explained = float(
            np.sum(probability * footprint_likelihood_ratio)
        )
        evidence = omitted + explained
        total += np.log(max(evidence, 1e-12))
        # Under an uninformative pose all retained identities also have
        # likelihood ratio one.  Count support only when this region's full
        # mixture beats that unit baseline.
        if evidence > omitted + float(np.sum(probability)):
            supporting_regions += 1
    return total, supporting_regions


def propose_maplet_footprint_poses(
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
    initial_hypotheses: tuple[MapletPoseHypothesis, ...],
    *,
    maximum_modes: int = 16,
    candidates_per_region: int = 16,
    cem_iterations: int = 3,
    particles_per_mode: int = 64,
    elite_fraction: float = 0.125,
    initial_translation_sigma_m: float = 0.75,
    initial_rotation_sigma_deg: float = 8.0,
    footprint_log_likelihood_weight: float = 1.0,
    view_vote_prior_weight: float = 1.0,
    seed: int = 911,
) -> tuple[MapletPoseHypothesis, ...]:
    """Refine anonymous view-mode seeds using region footprint evidence.

    RADIO regions overlap heavily, so their likelihood terms are a correlated
    composite likelihood.  We average, rather than multiply, their log
    evidence and retain the anonymous view-mode density as an explicit prior.
    This prevents 128 overlapping regions from overwhelming a good pose seed
    and driving CEM toward an unrelated but visually repetitive facade.
    """

    if not initial_hypotheses:
        return tuple()
    rng = np.random.default_rng(int(seed))
    proposed: list[MapletPoseHypothesis] = []
    rotation_sigma = np.deg2rad(float(initial_rotation_sigma_deg))
    initial_sigma = np.asarray(
        [rotation_sigma] * 3 + [float(initial_translation_sigma_m)] * 3,
        dtype=np.float64,
    )
    inverse_prior_variance = 1.0 / np.maximum(
        initial_sigma * initial_sigma, 1e-12
    )
    particle_count = max(int(particles_per_mode), 8)
    elite_count = max(
        2, int(np.ceil(particle_count * float(elite_fraction)))
    )
    seeds = initial_hypotheses[: max(int(maximum_modes), 1)]
    seed_mass = np.asarray(
        [max(float(item.score), 0.0) for item in seeds],
        dtype=np.float64,
    )
    if float(np.sum(seed_mass)) <= 0.0:
        seed_mass = np.ones_like(seed_mass)
    seed_probability = seed_mass / np.sum(seed_mass)
    for seed_index, initial in enumerate(seeds):
        mean = np.zeros(6, dtype=np.float64)
        covariance = np.diag(initial_sigma * initial_sigma)
        best_pose = np.asarray(initial.pose_w2c, dtype=np.float64)
        initial_footprint_score, best_support = score_maplet_footprint_pose(
            best_pose,
            groups,
            bank,
            camera,
            candidates_per_region=int(candidates_per_region),
        )
        # Pose-vote scores are unnormalised kernel densities.  Convert the
        # retained modes into a discrete prior before combining them with a
        # footprint likelihood; a raw density is not a probability.
        seed_log_prior = float(
            np.log(max(float(seed_probability[seed_index]), 1e-12))
        )

        def posterior_score(
            footprint_score: float, delta_value: np.ndarray
        ) -> float:
            return (
                float(footprint_log_likelihood_weight)
                * float(footprint_score)
                / max(len(groups), 1)
                + float(view_vote_prior_weight) * seed_log_prior
                - 0.5
                * float(
                    np.sum(
                        np.asarray(delta_value, dtype=np.float64) ** 2
                        * inverse_prior_variance
                    )
                )
            )

        best_score = posterior_score(
            initial_footprint_score, np.zeros(6, dtype=np.float64)
        )
        for _iteration in range(max(int(cem_iterations), 0)):
            delta = rng.multivariate_normal(
                mean, covariance, size=particle_count
            )
            # Always retain the current mode as one particle.
            delta[0] = mean
            scores = np.empty(particle_count, dtype=np.float64)
            supports = np.empty(particle_count, dtype=np.int64)
            poses = []
            for row in range(particle_count):
                pose = se3_exp(delta[row]) @ initial.pose_w2c
                footprint_score, support = score_maplet_footprint_pose(
                    pose,
                    groups,
                    bank,
                    camera,
                    candidates_per_region=int(candidates_per_region),
                )
                poses.append(pose)
                scores[row] = posterior_score(
                    footprint_score, delta[row]
                )
                supports[row] = support
            order = np.argsort(-scores, kind="stable")
            elite = delta[order[:elite_count]]
            mean = np.mean(elite, axis=0)
            centered = elite - mean[None]
            covariance = (
                centered.T @ centered / max(elite_count - 1, 1)
                + np.diag(initial_sigma * initial_sigma) * 1e-3
            )
            winner = int(order[0])
            if scores[winner] > best_score:
                best_score = float(scores[winner])
                best_support = int(supports[winner])
                best_pose = poses[winner]
        proposed.append(
            MapletPoseHypothesis(
                pose_w2c=best_pose,
                score=float(best_score),
                supporting_group_count=int(best_support),
                sampled_maplet_ids=initial.sampled_maplet_ids,
                source="maplet_footprint_likelihood_cem",
            )
        )
    proposed.sort(key=lambda item: item.score, reverse=True)
    output: list[MapletPoseHypothesis] = []
    for candidate in proposed:
        candidate_center = (
            -candidate.pose_w2c[:3, :3].T @ candidate.pose_w2c[:3, 3]
        )
        duplicate = False
        for retained in output:
            retained_center = (
                -retained.pose_w2c[:3, :3].T @ retained.pose_w2c[:3, 3]
            )
            if (
                np.linalg.norm(candidate_center - retained_center) < 0.20
                and _rotation_distance_deg(
                    candidate.pose_w2c[:3, :3],
                    retained.pose_w2c[:3, :3],
                )
                < 2.0
            ):
                duplicate = True
                break
        if not duplicate:
            output.append(candidate)
        if len(output) >= int(maximum_modes):
            break
    return tuple(output)
