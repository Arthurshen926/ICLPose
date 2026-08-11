"""VFM-geometry-guided latent configuration and SE(3) proposal.

The query never commits to point identities.  Two region hypotheses define a
candidate metric relation, RADIO-predicted query geometry deterministically
extends it to a third region, and a three-region Sim(3) alignment proposes the
camera pose.  Metric scale is marginalized as a nuisance variable.  The final
ranking is still the fixed-support projected 2DGS surface/query-graph
likelihood, so predicted depth is a proposal mechanism rather than pose truth.
"""

from __future__ import annotations

import heapq

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
    pnp_pose_error,
)

from .child_retrieval import ChildTilePosterior
from .joint_pose_proposal import _projected_surface_assignment, _weighted_choice
from .physical_map import GoalMapletPhysicalMap
from .pose_proposal import CoarsePoseModes, _rotation_distance_degrees
from .query_edge_factor import (
    build_query_edge_graph,
    score_pose_conditioned_query_edges,
)
from .soft_maplet_pose_likelihood import pose_conditioned_soft_maplet_evidence
from .sparse_vfm_pose_likelihood import (
    score_pose_conditioned_sparse_primitives,
    score_pose_conditioned_sparse_vfm,
)
from .typed_graph import GEOMETRY_ADJACENCY, SURFACE_CONTINUITY, TypedParentGraph


def unproject_query_depth(
    query_xy_px: np.ndarray,
    depth_z: np.ndarray,
    camera: ColmapCamera,
) -> np.ndarray:
    """Unproject distorted image coordinates using camera-z depth."""

    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth_z, dtype=np.float64).reshape(-1)
    if depth.shape != (xy.shape[0],):
        raise ValueError("depth_z must have one value per query coordinate")
    matrix, distortion = camera_matrix_and_distortion(camera)
    normalized = cv2.undistortPoints(
        xy.reshape(-1, 1, 2), matrix, distortion,
    ).reshape(-1, 2)
    return np.column_stack((normalized, np.ones((xy.shape[0],), dtype=np.float64))) * depth[:, None]


def estimate_pose_from_scaled_query_geometry(
    world_xyz: np.ndarray,
    query_xyz: np.ndarray,
    world_normal: np.ndarray | None = None,
    query_normal: np.ndarray | None = None,
    normal_weight: float = 0.0,
) -> tuple[np.ndarray, float] | None:
    """Estimate world-to-camera pose while marginalizing one depth scale.

    The fitted similarity is ``query = scale * R * world + offset``.  Dividing
    the offset by scale recovers the metric SE(3) translation.
    """

    world = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    query = np.asarray(query_xyz, dtype=np.float64).reshape(-1, 3)
    if world.shape != query.shape or world.shape[0] < 3:
        raise ValueError("world_xyz and query_xyz must have matching (N, 3), N >= 3")
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(query)):
        return None
    world_center = np.mean(world, axis=0)
    query_center = np.mean(query, axis=0)
    world_zero = world - world_center
    query_zero = query - query_center
    world_variance = float(np.mean(np.sum(np.square(world_zero), axis=1)))
    if world_variance <= 1e-10:
        return None
    covariance = query_zero.T @ world_zero / float(world.shape[0])
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    sign = np.ones((3,), dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    if float(normal_weight) > 0.0:
        if world_normal is None or query_normal is None:
            raise ValueError("normal-guided alignment requires world_normal and query_normal")
        world_n = np.asarray(world_normal, dtype=np.float64).reshape(-1, 3).copy()
        query_n = np.asarray(query_normal, dtype=np.float64).reshape(-1, 3).copy()
        if world_n.shape != world.shape or query_n.shape != query.shape:
            raise ValueError("normal arrays must match geometry arrays")
        world_n /= np.maximum(np.linalg.norm(world_n, axis=1, keepdims=True), 1e-8)
        query_n /= np.maximum(np.linalg.norm(query_n, axis=1, keepdims=True), 1e-8)
        # 2DGS tangent planes are unoriented while query labels face the
        # camera.  Resolve each sign against the point-only initialization,
        # then solve one point+normal Wahba problem.
        transformed_n = world_n @ rotation.T
        world_n[np.sum(transformed_n * query_n, axis=1) < 0.0] *= -1.0
        point_scale = max(
            float(np.sqrt(np.mean(np.sum(np.square(world_zero), axis=1))))
            * float(np.sqrt(np.mean(np.sum(np.square(query_zero), axis=1)))),
            1e-8,
        )
        joint_covariance = covariance / point_scale + float(normal_weight) * (
            query_n.T @ world_n / float(world.shape[0])
        )
        try:
            u, _singular, vt = np.linalg.svd(joint_covariance)
        except np.linalg.LinAlgError:
            return None
        sign = np.ones((3,), dtype=np.float64)
        if np.linalg.det(u @ vt) < 0.0:
            sign[-1] = -1.0
        rotation = u @ np.diag(sign) @ vt
        rotated_world = world_zero @ rotation.T
        scale = float(
            np.sum(query_zero * rotated_world)
            / max(np.sum(np.square(world_zero)), 1e-10)
        )
    else:
        scale = float(np.sum(singular * sign) / world_variance)
    if not np.isfinite(scale) or scale <= 1e-8:
        return None
    offset = query_center - scale * (rotation @ world_center)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = offset / scale
    if not np.all(np.isfinite(pose)):
        return None
    return pose, scale


def _triangle_quality(points: np.ndarray) -> float:
    value = np.asarray(points, dtype=np.float64).reshape(3, 3)
    a = value[1] - value[0]
    b = value[2] - value[0]
    return float(np.linalg.norm(np.cross(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _normal_consistency(
    pose: np.ndarray,
    world_normals: np.ndarray,
    query_normals: np.ndarray,
    query_xyz: np.ndarray,
) -> float:
    predicted = np.asarray(world_normals, dtype=np.float64) @ pose[:3, :3].T
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-8)
    query = np.asarray(query_normals, dtype=np.float64)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    # Geometry labels orient every 2DGS normal toward the camera.
    flip = np.sum(predicted * (-np.asarray(query_xyz, dtype=np.float64)), axis=1) < 0.0
    predicted[flip] *= -1.0
    return float(np.mean(np.clip(np.sum(predicted * query, axis=1), -1.0, 1.0)))


def scale_invariant_configuration_geometry(
    pose_w2c: np.ndarray,
    query_xyz: np.ndarray,
    query_normal_cam: np.ndarray,
    geometry_confidence: np.ndarray,
    assigned_child_rows: np.ndarray,
    child_centers: np.ndarray,
    child_normals: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure global query/map shape without treating VFM depth as metric.

    Query depth has one nuisance scale, so the factor first aggregates all
    query regions assigned to the same physical child and then removes one
    robust global log scale.  The remaining point, pair-distance, ray and
    unoriented-normal agreement are pose/configuration evidence.  In
    particular, this is not a 2D--3D correspondence solver and cannot create a
    pose from the observations it later scores.
    """

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    query = np.asarray(query_xyz, dtype=np.float64).reshape(-1, 3)
    query_normal = np.asarray(query_normal_cam, dtype=np.float64).reshape(-1, 3)
    confidence = np.asarray(geometry_confidence, dtype=np.float64).reshape(-1)
    assigned = np.asarray(assigned_child_rows, dtype=np.int64).reshape(-1)
    if not (
        query.shape == query_normal.shape
        and confidence.shape == assigned.shape == (query.shape[0],)
    ):
        raise ValueError("configuration geometry arrays differ")
    valid = (
        (assigned >= 0) & np.isfinite(confidence) & (confidence > 0.0)
        & np.all(np.isfinite(query), axis=1)
        & np.all(np.isfinite(query_normal), axis=1)
    )
    unique_child = np.unique(assigned[valid])
    if unique_child.size < 3:
        return {
            "score": None,
            "unique_child_count": int(unique_child.size),
            "scale": None,
            "point_score": None,
            "pair_score": None,
            "ray_score": None,
            "normal_score": None,
        }

    aggregated_query = []
    aggregated_normal = []
    aggregated_weight = []
    for child in unique_child.tolist():
        rows = np.flatnonzero(valid & (assigned == int(child)))
        weight = np.maximum(confidence[rows], 1e-6)
        total = float(np.sum(weight))
        aggregated_query.append(np.sum(query[rows] * weight[:, None], axis=0) / total)
        normal = np.sum(query_normal[rows] * weight[:, None], axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        aggregated_normal.append(normal)
        aggregated_weight.append(float(np.sqrt(total)))
    query_value = np.asarray(aggregated_query, dtype=np.float64)
    query_normal_value = np.asarray(aggregated_normal, dtype=np.float64)
    weight = np.asarray(aggregated_weight, dtype=np.float64)
    weight /= max(float(np.sum(weight)), 1e-8)
    world = np.asarray(child_centers, dtype=np.float64)[unique_child]
    world_normal = np.asarray(child_normals, dtype=np.float64)[unique_child]
    map_camera = world @ pose[:3, :3].T + pose[:3, 3]
    positive = (
        np.all(np.isfinite(map_camera), axis=1)
        & (map_camera[:, 2] > 0.05)
        & (query_value[:, 2] > 0.05)
    )
    if int(np.sum(positive)) < 3:
        return {
            "score": None,
            "unique_child_count": int(np.sum(positive)),
            "scale": None,
            "point_score": None,
            "pair_score": None,
            "ray_score": None,
            "normal_score": None,
        }
    query_value = query_value[positive]
    query_normal_value = query_normal_value[positive]
    weight = weight[positive]
    weight /= max(float(np.sum(weight)), 1e-8)
    map_camera = map_camera[positive]
    world_normal = world_normal[positive]

    # Median log-radius ratio removes the sole query-depth scale without
    # granting the candidate an arbitrary 3D translation.
    query_radius = np.linalg.norm(query_value, axis=1)
    map_radius = np.linalg.norm(map_camera, axis=1)
    log_ratio = np.log(np.maximum(query_radius, 1e-8) / np.maximum(map_radius, 1e-8))
    scale = float(np.exp(np.median(log_ratio)))
    point_residual = np.linalg.norm(query_value - scale * map_camera, axis=1) / np.maximum(
        query_radius, 0.25,
    )
    point_score = float(np.sum(weight * np.exp(-0.5 * np.square(point_residual / 0.20))))

    query_direction = query_value / np.maximum(query_radius[:, None], 1e-8)
    map_direction = map_camera / np.maximum(map_radius[:, None], 1e-8)
    ray_score = float(np.sum(weight * np.clip(np.sum(
        query_direction * map_direction, axis=1,
    ), 0.0, 1.0)))

    first, second = np.triu_indices(query_value.shape[0], k=1)
    query_distance = np.linalg.norm(query_value[first] - query_value[second], axis=1)
    map_distance = np.linalg.norm(map_camera[first] - map_camera[second], axis=1)
    pair_valid = (query_distance > 0.10) & (map_distance > 0.10)
    if np.any(pair_valid):
        pair_log_ratio = np.log(
            query_distance[pair_valid] / np.maximum(map_distance[pair_valid], 1e-8)
        )
        pair_center = float(np.median(pair_log_ratio))
        pair_residual = np.abs(pair_log_ratio - pair_center)
        pair_weight = np.sqrt(weight[first[pair_valid]] * weight[second[pair_valid]])
        pair_weight /= max(float(np.sum(pair_weight)), 1e-8)
        pair_score = float(np.sum(
            pair_weight * np.exp(-0.5 * np.square(pair_residual / 0.20))
        ))
    else:
        pair_score = 0.0

    predicted_normal = world_normal @ pose[:3, :3].T
    predicted_normal /= np.maximum(np.linalg.norm(predicted_normal, axis=1, keepdims=True), 1e-8)
    query_normal_value /= np.maximum(
        np.linalg.norm(query_normal_value, axis=1, keepdims=True), 1e-8,
    )
    # A 2DGS tangent plane has no stable sign, hence absolute cosine.
    normal_score = float(np.sum(weight * np.abs(np.sum(
        predicted_normal * query_normal_value, axis=1,
    ))))
    score = float(
        0.45 * point_score
        + 0.35 * pair_score
        + 0.10 * ray_score
        + 0.10 * normal_score
    )
    return {
        "score": score,
        "unique_child_count": int(query_value.shape[0]),
        "scale": scale,
        "point_score": point_score,
        "pair_score": pair_score,
        "ray_score": ray_score,
        "normal_score": normal_score,
    }


def _empty_modes() -> CoarsePoseModes:
    return CoarsePoseModes(
        np.zeros((0, 4, 4), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.int64),
    )


def _pose_stage_diagnostic(
    poses: list[np.ndarray] | np.ndarray,
    gt_pose_w2c: np.ndarray | None,
) -> dict[str, object]:
    values = list(poses)
    result: dict[str, object] = {"pose_count": int(len(values))}
    if gt_pose_w2c is None or not values:
        return result
    errors = [pnp_pose_error(pose, gt_pose_w2c) for pose in values]
    translation = np.asarray([value.translation_m for value in errors], dtype=np.float64)
    rotation = np.asarray([value.rotation_deg for value in errors], dtype=np.float64)
    best = int(np.argmin(translation / 0.5 + rotation / 5.0))
    result.update({
        "top1_translation_m": float(translation[0]),
        "top1_rotation_deg": float(rotation[0]),
        "best_translation_m": float(translation[best]),
        "best_rotation_deg": float(rotation[best]),
        "within_1m_10deg_count": int(np.sum((translation <= 1.0) & (rotation <= 10.0))),
        "within_0_5m_5deg_count": int(np.sum((translation <= 0.5) & (rotation <= 5.0))),
        "first_within_1m_10deg_rank": (
            int(np.flatnonzero((translation <= 1.0) & (rotation <= 10.0))[0]) + 1
            if np.any((translation <= 1.0) & (rotation <= 10.0)) else None
        ),
        "first_within_0_5m_5deg_rank": (
            int(np.flatnonzero((translation <= 0.5) & (rotation <= 5.0))[0]) + 1
            if np.any((translation <= 0.5) & (rotation <= 5.0)) else None
        ),
    })
    return result


def _mapping_view_seed_diagnostic(
    seeds: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]],
    anchor_labels: np.ndarray,
    gt_pose_w2c: np.ndarray | None,
) -> dict[str, object]:
    """Audit the exact raw seed that best covers the held-out query pose.

    Aggregate stage metrics were insufficient here: they established that a
    good pose existed, but not which view-hyperedge, identity triple, or
    within-hyperedge proposal rank produced it.  This diagnostic deliberately
    records map/query row indices and geometry only; it never serializes an
    image or image descriptor.
    """

    result = _pose_stage_diagnostic([value[1] for value in seeds], gt_pose_w2c)
    labels = np.asarray(anchor_labels, dtype=np.int64).reshape(-1)
    if gt_pose_w2c is None or not seeds:
        return result
    if labels.size != len(seeds):
        raise ValueError("mapping-view seed labels differ from seeds")
    errors = [pnp_pose_error(value[1], gt_pose_w2c) for value in seeds]
    translation = np.asarray([value.translation_m for value in errors], dtype=np.float64)
    rotation = np.asarray([value.rotation_deg for value in errors], dtype=np.float64)
    joint = translation / 0.5 + rotation / 5.0
    best = int(np.argmin(joint))
    label = int(labels[best])
    same_anchor = np.flatnonzero(labels == label)
    proposal_scores = np.asarray([value[0] for value in seeds], dtype=np.float64)
    global_order = np.argsort(-proposal_scores, kind="stable")
    local_order = same_anchor[np.argsort(-proposal_scores[same_anchor], kind="stable")]
    result["best_joint_seed"] = {
        "raw_index": int(best),
        "anchor_local_label": label,
        "pose_w2c": np.asarray(seeds[best][1], dtype=np.float64).tolist(),
        "translation_m": float(translation[best]),
        "rotation_deg": float(rotation[best]),
        "proposal_score": float(proposal_scores[best]),
        "global_proposal_rank": int(np.flatnonzero(global_order == best)[0]) + 1,
        "anchor_local_proposal_rank": int(np.flatnonzero(local_order == best)[0]) + 1,
        "anchor_seed_count": int(same_anchor.size),
        "seed_parent_rows": np.asarray(seeds[best][2], dtype=np.int64).tolist(),
        "seed_support_rows": np.asarray(seeds[best][3], dtype=np.int64).tolist(),
        "fitted_scale": float(seeds[best][4]),
    }
    return result


def _diverse_support_subset(
    order: np.ndarray,
    xy_normalized: np.ndarray,
    maximum_count: int,
    minimum_separation: float = 0.025,
) -> np.ndarray:
    selected: list[int] = []
    for value in np.asarray(order, dtype=np.int64).tolist():
        if selected and np.min(np.linalg.norm(
            np.asarray(xy_normalized, dtype=np.float64)[np.asarray(selected)]
            - np.asarray(xy_normalized, dtype=np.float64)[value], axis=1,
        )) < float(minimum_separation):
            continue
        selected.append(int(value))
        if len(selected) >= int(maximum_count):
            break
    return np.asarray(selected, dtype=np.int64)


def _coverage_support_pairs(
    supports: np.ndarray,
    xy_normalized: np.ndarray,
    support_weight: np.ndarray,
    maximum_pairs: int,
) -> set[tuple[int, int]]:
    """Bound pair cost while giving every selected image region a chance."""

    count = int(np.asarray(supports).size)
    if count < 2:
        return set()
    all_rows = []
    xy = np.asarray(xy_normalized, dtype=np.float64)[supports]
    weight = np.asarray(support_weight, dtype=np.float64)[supports]
    for first in range(count):
        for second in range(first + 1, count):
            score = float(
                np.linalg.norm(xy[first] - xy[second])
                * np.sqrt(max(weight[first] * weight[second], 1e-12))
            )
            all_rows.append((score, first, second))
    if int(maximum_pairs) <= 0 or len(all_rows) <= int(maximum_pairs):
        return {(first, second) for _score, first, second in all_rows}
    per_support = max(1, int(maximum_pairs) // max(count, 1))
    selected: set[tuple[int, int]] = set()
    for support in range(count):
        incident = [row for row in all_rows if row[1] == support or row[2] == support]
        incident.sort(key=lambda row: (-row[0], row[1], row[2]))
        for _score, first, second in incident[:per_support]:
            selected.add((first, second))
    for _score, first, second in sorted(all_rows, key=lambda row: (-row[0], row[1], row[2])):
        if len(selected) >= int(maximum_pairs):
            break
        selected.add((first, second))
    return selected


def _disconnected_query_token_masks(
    seed_support_rows: np.ndarray,
    query_xy_normalized: np.ndarray,
    query_extent_normalized: np.ndarray,
    token_height: int,
    token_width: int,
) -> np.ndarray:
    """Rasterize fixed query regions for disconnected-maplet likelihoods."""

    seed = np.asarray(seed_support_rows, dtype=np.int64)
    xy = np.asarray(query_xy_normalized, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(query_extent_normalized, dtype=np.float64).reshape(-1, 2)
    if seed.ndim != 2 or xy.shape != extent.shape:
        raise ValueError("disconnected query mask arrays differ")
    grid_y, grid_x = np.meshgrid(
        (np.arange(int(token_height), dtype=np.float64) + 0.5) / float(token_height),
        (np.arange(int(token_width), dtype=np.float64) + 0.5) / float(token_width),
        indexing="ij",
    )
    grid = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1)
    result = np.zeros((seed.shape[0], grid.shape[0]), dtype=bool)
    token_half = np.asarray(
        [0.5 / float(token_width), 0.5 / float(token_height)], dtype=np.float64,
    )
    for pose in range(seed.shape[0]):
        valid = seed[pose][(seed[pose] >= 0) & (seed[pose] < xy.shape[0])]
        for support in np.unique(valid).tolist():
            half = np.maximum(extent[int(support)], token_half)
            result[pose] |= np.all(
                np.abs(grid - xy[int(support)]) <= half[None], axis=1,
            )
        if not np.any(result[pose]) and valid.size:
            nearest = int(np.argmin(np.linalg.norm(grid - xy[int(valid[0])], axis=1)))
            result[pose, nearest] = True
    return result


def _deterministic_pair_beam_seeds(
    *,
    beam_support: np.ndarray,
    extension_support: np.ndarray,
    xy_normalized: np.ndarray,
    query_xyz: np.ndarray,
    query_normal: np.ndarray,
    support_weight: np.ndarray,
    candidate_mass: np.ndarray,
    valid_candidate: np.ndarray,
    child_rows: np.ndarray,
    parent_rows: np.ndarray,
    child_centers: np.ndarray,
    child_normals: np.ndarray,
    covisibility: np.ndarray,
    pair_candidate_count: int,
    extension_candidate_count: int,
    pair_hypotheses_per_support_pair: int,
    maximum_support_pairs: int,
    extensions_per_pair: int,
    minimum_pair_image_separation: float,
    minimum_triangle_quality: float,
    minimum_scale: float,
    maximum_scale: float,
    triangle_log_ratio_sigma: float,
    normal_proposal_weight: float,
    orientation_normal_weight: float,
    marginal_pair_consensus: bool,
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Enumerate a bounded support/candidate pair beam and score global consensus."""

    result: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    pair_k = min(int(pair_candidate_count), int(candidate_mass.shape[1]))
    extension_k = min(int(extension_candidate_count), int(candidate_mass.shape[1]))
    ext_support = np.asarray(extension_support, dtype=np.int64)
    ext_children = child_rows[ext_support, :extension_k]
    ext_parents = parent_rows[ext_support, :extension_k]
    ext_mass = candidate_mass[ext_support, :extension_k]
    ext_valid_base = valid_candidate[ext_support, :extension_k]
    ext_world = child_centers[np.maximum(ext_children, 0)]
    ext_query = query_xyz[ext_support]
    ext_weight = np.asarray(support_weight[ext_support], dtype=np.float64)
    sigma = max(float(triangle_log_ratio_sigma), 1e-6)
    allowed_pairs = _coverage_support_pairs(
        beam_support, xy_normalized, support_weight, int(maximum_support_pairs),
    )
    for first_index in range(int(beam_support.size)):
        first_support = int(beam_support[first_index])
        for second_index in range(first_index + 1, int(beam_support.size)):
            if (first_index, second_index) not in allowed_pairs:
                continue
            second_support = int(beam_support[second_index])
            if np.linalg.norm(
                xy_normalized[first_support] - xy_normalized[second_support]
            ) < float(minimum_pair_image_separation):
                continue
            first_slots = np.flatnonzero(valid_candidate[first_support, :pair_k])
            second_slots = np.flatnonzero(valid_candidate[second_support, :pair_k])
            if first_slots.size == 0 or second_slots.size == 0:
                continue
            slot0, slot1 = np.meshgrid(first_slots, second_slots, indexing="ij")
            slot0 = slot0.reshape(-1)
            slot1 = slot1.reshape(-1)
            child0 = child_rows[first_support, slot0]
            child1 = child_rows[second_support, slot1]
            parent0 = parent_rows[first_support, slot0]
            parent1 = parent_rows[second_support, slot1]
            world0 = child_centers[child0]
            world1 = child_centers[child1]
            world_distance = np.linalg.norm(world1 - world0, axis=1)
            query_distance = float(np.linalg.norm(
                query_xyz[second_support] - query_xyz[first_support]
            ))
            scale = query_distance / np.maximum(world_distance, 1e-12)
            pair_valid = (
                (child0 != child1) & (world_distance > 0.20)
                & (scale >= float(minimum_scale)) & (scale <= float(maximum_scale))
            )
            if not np.any(pair_valid):
                continue
            world_d0 = np.linalg.norm(
                ext_world[None] - world0[:, None, None, :], axis=3,
            )
            world_d1 = np.linalg.norm(
                ext_world[None] - world1[:, None, None, :], axis=3,
            )
            query_d0 = np.linalg.norm(
                ext_query - query_xyz[first_support], axis=1,
            )[None, :, None]
            query_d1 = np.linalg.norm(
                ext_query - query_xyz[second_support], axis=1,
            )[None, :, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_error = (
                    np.abs(np.log((query_d0 / world_d0) / scale[:, None, None]))
                    + np.abs(np.log((query_d1 / world_d1) / scale[:, None, None]))
                )
            extension_valid = (
                ext_valid_base[None]
                & (ext_children[None] != child0[:, None, None])
                & (ext_children[None] != child1[:, None, None])
                & (world_d0 > 0.10) & (world_d1 > 0.10)
                & (query_d0 > 0.10) & (query_d1 > 0.10)
                & np.isfinite(ratio_error)
            )
            extension_valid[:, np.isin(ext_support, [first_support, second_support]), :] = False
            safe_extension_parent = np.maximum(ext_parents, 0)
            compatibility0 = 0.02 + 0.98 * covisibility[
                safe_extension_parent[None], parent0[:, None, None]
            ]
            compatibility1 = 0.02 + 0.98 * covisibility[
                safe_extension_parent[None], parent1[:, None, None]
            ]
            extension_compatibility = np.sqrt(compatibility0 * compatibility1)
            extension_factor = (
                -ratio_error / sigma
                + 0.15 * np.log(np.maximum(ext_mass[None], 1e-12))
                + 0.10 * np.log(np.maximum(extension_compatibility, 1e-12))
            )
            extension_factor[~extension_valid] = -np.inf
            best_slot = np.argmax(extension_factor, axis=2)
            best_factor = np.max(extension_factor, axis=2)
            best_error = np.take_along_axis(
                ratio_error, best_slot[:, :, None], axis=2,
            )[:, :, 0]
            finite_support = np.isfinite(best_factor)
            if bool(marginal_pair_consensus):
                # Marginalize extension identity.  Taking the best residual
                # over K ambiguous candidates rewards the wrong phase merely
                # for having more chances (look-elsewhere bias).  MAP is
                # retained only to instantiate a numerical third point.
                extension_prior = np.where(
                    extension_valid,
                    ext_mass[None] * extension_compatibility,
                    0.0,
                )
                extension_kernel = np.exp(
                    -0.5 * np.square(np.minimum(ratio_error / sigma, 8.0))
                )
                extension_kernel = np.where(
                    extension_valid & np.isfinite(extension_kernel),
                    extension_kernel,
                    0.0,
                )
                explained_mass = np.sum(extension_prior * extension_kernel, axis=2)
                retained_mass = np.sum(
                    np.where(extension_valid, ext_mass[None], 0.0), axis=2,
                )
                consensus_support = explained_mass / np.maximum(
                    retained_mass, 1e-12,
                )
            else:
                safe_best_error = np.where(finite_support, best_error, 8.0 * sigma)
                consensus_support = np.exp(
                    -0.5 * np.square(np.minimum(safe_best_error / sigma, 8.0))
                )
            consensus_numerator = np.sum(
                ext_weight[None] * finite_support * consensus_support,
                axis=1,
            )
            consensus_denominator = np.sum(ext_weight[None] * finite_support, axis=1)
            consensus = consensus_numerator / np.maximum(consensus_denominator, 1e-8)
            pair_mass = (
                candidate_mass[first_support, slot0]
                * candidate_mass[second_support, slot1]
            )
            compatibility = np.maximum(
                0.05 + 0.95 * covisibility[parent0, parent1], 1e-8,
            )
            pair_score = (
                4.0 * consensus
                + 0.10 * np.log(np.maximum(pair_mass, 1e-12))
                + 0.10 * np.log(compatibility)
            )
            pair_score[~pair_valid | ~(consensus_denominator > 0.0)] = -np.inf
            finite_pair = np.flatnonzero(np.isfinite(pair_score))
            if finite_pair.size == 0:
                continue
            pair_take = min(int(pair_hypotheses_per_support_pair), int(finite_pair.size))
            if pair_take < finite_pair.size:
                local = np.argpartition(-pair_score[finite_pair], kth=pair_take - 1)[:pair_take]
                finite_pair = finite_pair[local]
            finite_pair = finite_pair[np.argsort(-pair_score[finite_pair], kind="stable")]
            for pair_row in finite_pair.tolist():
                extension_rows = np.flatnonzero(np.isfinite(best_factor[pair_row]))
                if extension_rows.size == 0:
                    continue
                extension_take = min(int(extensions_per_pair), int(extension_rows.size))
                if extension_take < extension_rows.size:
                    local = np.argpartition(
                        -best_factor[pair_row, extension_rows], kth=extension_take - 1,
                    )[:extension_take]
                    extension_rows = extension_rows[local]
                extension_rows = extension_rows[np.argsort(
                    -best_factor[pair_row, extension_rows], kind="stable",
                )]
                for extension_row in extension_rows.tolist():
                    third_support = int(ext_support[extension_row])
                    third_slot = int(best_slot[pair_row, extension_row])
                    third_child = int(ext_children[extension_row, third_slot])
                    selected_support = np.asarray(
                        [first_support, second_support, third_support], dtype=np.int64,
                    )
                    selected_child = np.asarray(
                        [child0[pair_row], child1[pair_row], third_child], dtype=np.int64,
                    )
                    selected_world = child_centers[selected_child]
                    selected_query = query_xyz[selected_support]
                    if (
                        _triangle_quality(selected_world) < float(minimum_triangle_quality)
                        or _triangle_quality(selected_query) < float(minimum_triangle_quality)
                    ):
                        continue
                    estimated = estimate_pose_from_scaled_query_geometry(
                        selected_world,
                        selected_query,
                        child_normals[selected_child],
                        query_normal[selected_support],
                        normal_weight=float(orientation_normal_weight),
                    )
                    if estimated is None:
                        continue
                    pose, fitted_scale = estimated
                    if fitted_scale < float(minimum_scale) or fitted_scale > float(maximum_scale):
                        continue
                    normal_score = _normal_consistency(
                        pose, child_normals[selected_child],
                        query_normal[selected_support], selected_query,
                    )
                    selected_parent = np.asarray([
                        parent0[pair_row], parent1[pair_row],
                        ext_parents[extension_row, third_slot],
                    ], dtype=np.int64)
                    score = (
                        float(pair_score[pair_row])
                        + 0.10 * float(best_factor[pair_row, extension_row])
                        + float(normal_proposal_weight) * normal_score
                    )
                    result.append((
                        score, pose, selected_parent, selected_support, fitted_scale,
                    ))
    return result


def _phase_anchor_pair_beam_seeds(
    *,
    parent_rows: np.ndarray,
    candidate_mass: np.ndarray,
    valid_candidate: np.ndarray,
    child_rows: np.ndarray,
    phase_compatibility: np.ndarray,
    anchor_count: int,
    anchor_message_power: float,
    anchor_support_pairs: int,
    anchor_candidate_count: int,
    anchor_extension_count: int,
    anchor_hypotheses: int,
    common_arguments: dict[str, object],
) -> tuple[
    list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]],
    np.ndarray,
    np.ndarray,
]:
    """Enumerate coherent physical phases without committing query supports.

    Aggregate retrieval mass proposes anonymous physical-parent anchors.  Each
    anchor only reorders every support's existing alternatives through map
    co-visibility; no query-to-map feature is recomputed and no assignment is
    fixed.  Hard triples remain a numerical Sim(3) proposal primitive, while
    the final pose score marginalizes all identities.
    """

    parent_count = int(phase_compatibility.shape[0])
    global_mass = np.zeros((parent_count,), dtype=np.float64)
    valid = np.asarray(valid_candidate, dtype=bool)
    rows = np.asarray(parent_rows, dtype=np.int64)
    mass = np.asarray(candidate_mass, dtype=np.float64)
    np.add.at(global_mass, rows[valid], mass[valid])
    available = np.flatnonzero(global_mass > 0.0)
    if available.size == 0 or int(anchor_count) <= 0:
        empty = np.zeros((0,), dtype=np.int64)
        return [], empty, empty
    order = available[np.argsort(-global_mass[available], kind="stable")]
    anchors = order[: min(int(anchor_count), int(order.size))]
    result: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    result_anchor: list[int] = []
    safe_rows = np.maximum(rows, 0)
    for anchor in anchors.tolist():
        compatibility = 0.02 + 0.98 * phase_compatibility[safe_rows, int(anchor)]
        compatibility[~valid] = 0.0
        conditioned = mass * np.power(
            np.maximum(compatibility, 1.0e-12), float(anchor_message_power),
        )
        candidate_order = np.argsort(-conditioned, axis=1, kind="stable")
        reordered_mass = np.take_along_axis(conditioned, candidate_order, axis=1)
        reordered_valid = np.take_along_axis(valid, candidate_order, axis=1)
        reordered_child = np.take_along_axis(child_rows, candidate_order, axis=1)
        reordered_parent = np.take_along_axis(rows, candidate_order, axis=1)
        seeds = _deterministic_pair_beam_seeds(
            candidate_mass=reordered_mass,
            valid_candidate=reordered_valid,
            child_rows=reordered_child,
            parent_rows=reordered_parent,
            covisibility=phase_compatibility,
            pair_candidate_count=int(anchor_candidate_count),
            extension_candidate_count=int(anchor_extension_count),
            pair_hypotheses_per_support_pair=int(anchor_hypotheses),
            maximum_support_pairs=int(anchor_support_pairs),
            **common_arguments,
        )
        anchor_prior = 0.05 * float(np.log(max(global_mass[int(anchor)], 1.0e-12)))
        result.extend(
            (float(score) + anchor_prior, pose, seed_parent, seed_support, scale)
            for score, pose, seed_parent, seed_support, scale in seeds
        )
        result_anchor.extend([int(anchor)] * len(seeds))
    return result, anchors, np.asarray(result_anchor, dtype=np.int64)


def _mapping_view_pair_beam_seeds(
    *,
    parent_rows: np.ndarray,
    candidate_mass: np.ndarray,
    valid_candidate: np.ndarray,
    child_rows: np.ndarray,
    view_parent_weights: np.ndarray,
    view_anchor_scores: np.ndarray,
    anchor_message_power: float,
    anchor_support_pairs: int,
    anchor_candidate_count: int,
    anchor_extension_count: int,
    anchor_hypotheses: int,
    common_arguments: dict[str, object],
) -> tuple[
    list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]], np.ndarray,
]:
    """Propose configurations from pose-bearing co-visibility hyperedges.

    A mapping-view node is used only as a set-valued physical identity factor;
    its stored pose is not copied into the result.  This distinction matters:
    trajectories need not revisit the exact query camera, while the maplets
    jointly visible from a mapping camera still define a useful repeated-
    structure equivalence class.
    """

    view_weight = np.asarray(view_parent_weights, dtype=np.float64)
    anchor_score = np.asarray(view_anchor_scores, dtype=np.float64).reshape(-1)
    rows = np.asarray(parent_rows, dtype=np.int64)
    mass = np.asarray(candidate_mass, dtype=np.float64)
    valid = np.asarray(valid_candidate, dtype=bool)
    if (
        view_weight.ndim != 2
        or view_weight.shape[0] != anchor_score.size
        or view_weight.shape[1] == 0
    ):
        raise ValueError("mapping-view anchor arrays differ")
    safe_rows = np.maximum(rows, 0)
    result: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    result_anchor: list[int] = []
    for anchor in range(view_weight.shape[0]):
        # Presence, not contributor pixel count, is the identity factor.  A
        # large facade must not suppress a small but decisive maplet merely
        # because it occupies fewer mapping pixels.
        present = view_weight[anchor, safe_rows] > 0.0
        compatibility = np.where(present, 1.0, 0.02)
        compatibility[~valid] = 0.0
        conditioned = mass * np.power(
            np.maximum(compatibility, 1.0e-12), float(anchor_message_power),
        )
        candidate_order = np.argsort(-conditioned, axis=1, kind="stable")
        seeds = _deterministic_pair_beam_seeds(
            candidate_mass=np.take_along_axis(conditioned, candidate_order, axis=1),
            valid_candidate=np.take_along_axis(valid, candidate_order, axis=1),
            child_rows=np.take_along_axis(child_rows, candidate_order, axis=1),
            parent_rows=np.take_along_axis(rows, candidate_order, axis=1),
            # Within one view-node hyperedge, absence of a pairwise mapping
            # edge must not erase the already observed higher-order relation.
            covisibility=np.maximum(
                np.asarray(common_arguments["covisibility"], dtype=np.float64),
                np.outer(view_weight[anchor] > 0.0, view_weight[anchor] > 0.0),
            ),
            pair_candidate_count=int(anchor_candidate_count),
            extension_candidate_count=int(anchor_extension_count),
            pair_hypotheses_per_support_pair=int(anchor_hypotheses),
            maximum_support_pairs=int(anchor_support_pairs),
            **{
                key: value for key, value in common_arguments.items()
                if key != "covisibility"
            },
        )
        # View scores are proposal priors only.  Their deliberately small
        # coefficient cannot remove a geometrically coherent low-rank view.
        prior = 0.02 * float(anchor_score[anchor])
        result.extend(
            (float(score) + prior, pose, seed_parent, seed_support, scale)
            for score, pose, seed_parent, seed_support, scale in seeds
        )
        result_anchor.extend([int(anchor)] * len(seeds))
    return result, np.asarray(result_anchor, dtype=np.int64)


def _physical_phase_compatibility(
    graph: TypedParentGraph,
    physical: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, float]:
    """Map-only compatibility for a common camera-visible physical phase.

    Mapping co-visibility is a useful positive relation, but its absence is
    sampling-dependent: two nearby surface maplets may simply not occur in the
    same selected mapping frame.  Physical adjacency supplies that missing
    support.  Its scale is derived from the map's own adjacency/continuity
    edges, so this factor introduces neither query labels nor a scene-specific
    metric threshold.
    """

    covisibility = graph.covisibility_matrix().astype(np.float64)
    local_edge = np.isin(
        graph.edge_type,
        np.asarray([GEOMETRY_ADJACENCY, SURFACE_CONTINUITY], dtype=np.uint8),
    )
    local_distance = np.asarray(graph.edge_features[local_edge, 0], dtype=np.float64)
    local_distance = local_distance[np.isfinite(local_distance) & (local_distance > 0.0)]
    if hasattr(physical, "maplet_centers"):
        center = np.asarray(physical.maplet_centers, dtype=np.float64)
    else:
        # Minimal synthetic fixtures may only carry child geometry.
        parent_count = int(graph.parent_view_count.size)
        center = np.zeros((parent_count, 3), dtype=np.float64)
        for parent in range(parent_count):
            rows = np.flatnonzero(physical.child_parent_rows == parent)
            if rows.size:
                center[parent] = np.mean(physical.child_centers[rows], axis=0)
    if local_distance.size:
        radius = float(np.percentile(local_distance, 90.0))
    else:
        if center.shape[0] > 1:
            distance = np.linalg.norm(center[:, None] - center[None], axis=2)
            distance[distance <= 0.0] = np.inf
            radius = float(np.median(np.min(distance, axis=1))) * 3.0
        else:
            radius = 1.0
    radius = max(radius, 0.25)
    distance = np.linalg.norm(center[:, None] - center[None], axis=2)
    spatial = np.exp(-0.5 * np.square(distance / radius))
    compatibility = np.maximum(covisibility, spatial)
    np.fill_diagonal(compatibility, 1.0)
    return compatibility, radius


def generate_geometry_guided_configuration_pose_modes(
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    query_context_descriptors: np.ndarray,
    query_depth_z: np.ndarray,
    query_normal_cam: np.ndarray,
    geometry_confidence: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_out_of_map_probabilities: np.ndarray,
    parent_unresolved_probabilities: np.ndarray,
    child_posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    graph: TypedParentGraph,
    camera: ColmapCamera,
    *,
    maximum_modes: int = 32,
    proposal_trials: int = 8192,
    maximum_parent_candidates: int = 32,
    extension_support_count: int = 48,
    extensions_per_pair: int = 2,
    deterministic_pair_beam: bool = True,
    pair_beam_support_count: int = 64,
    maximum_support_pairs: int = 512,
    pair_candidate_count: int = 8,
    extension_candidate_count: int = 16,
    pair_hypotheses_per_support_pair: int = 2,
    seed_pool_multiplier: int = 4,
    preliminary_pose_count: int = 768,
    preliminary_support_count: int = 64,
    minimum_geometry_confidence: float = 0.05,
    minimum_pair_image_separation: float = 0.08,
    minimum_triangle_quality: float = 0.03,
    minimum_scale: float = 0.20,
    maximum_scale: float = 5.0,
    triangle_log_ratio_sigma: float = 0.20,
    normal_proposal_weight: float = 0.25,
    orientation_normal_weight: float = 0.0,
    query_edge_weight: float = 0.5,
    phase_anchor_count: int = 0,
    phase_anchor_message_power: float = 1.0,
    phase_anchor_support_pairs: int = 64,
    phase_anchor_candidate_count: int = 8,
    phase_anchor_extension_count: int = 16,
    phase_anchor_hypotheses: int = 2,
    phase_anchor_keep_per_anchor: int = 0,
    mapping_view_parent_weights: np.ndarray | None = None,
    mapping_view_anchor_scores: np.ndarray | None = None,
    mapping_view_message_power: float = 1.0,
    mapping_view_support_pairs: int = 64,
    mapping_view_candidate_count: int = 8,
    mapping_view_extension_count: int = 16,
    mapping_view_hypotheses: int = 4,
    seed_identity_vfm_alignment: bool = False,
    seed_identity_canonical_primitives: bool = False,
    seed_identity_full_map_primitives: bool = False,
    seed_identity_prescore_per_anchor: int = 0,
    seed_identity_exact_verify_count: int = 0,
    soft_identity_marginalization: bool = False,
    marginal_pair_consensus: bool = False,
    soft_parent_candidate_count: int = 8,
    soft_edge_candidate_count: int = 4,
    soft_maximum_edges: int = 128,
    pose_conditioned_vfm_alignment: bool = False,
    pose_conditioned_primitive_vfm_alignment: bool = False,
    query_token_context_descriptors: np.ndarray | None = None,
    query_token_local_descriptors: np.ndarray | None = None,
    map_parent_descriptors: np.ndarray | None = None,
    map_child_descriptors: np.ndarray | None = None,
    parent_descriptor_valid: np.ndarray | None = None,
    child_descriptor_valid: np.ndarray | None = None,
    token_height: int | None = None,
    token_width: int | None = None,
    sparse_vfm_temperature: float = 0.07,
    sparse_vfm_batch_size: int = 16,
    sparse_vfm_maximum_splat_radius_tokens: int = 2,
    sparse_vfm_score_semantics: str = "marginal_centered",
    query_token_canonical_descriptors: np.ndarray | None = None,
    canonical_field_primitive_rows: np.ndarray | None = None,
    canonical_field_codes: np.ndarray | None = None,
    canonical_field_confidence: np.ndarray | None = None,
    sparse_vfm_primitives_per_child: int = 8,
    sparse_primitive_score_semantics: str = "visible_sample_mean",
    sparse_vfm_device: str = "cuda",
    translation_nms_m: float = 0.20,
    rotation_nms_deg: float = 3.0,
    random_seed: int = 194917,
    diagnostic_pose_w2c: np.ndarray | None = None,
    diagnostics_out: dict[str, object] | None = None,
) -> CoarsePoseModes:
    """Propose poses from pair hypotheses plus geometry-determined extension."""

    if graph.physical_map_sha256 != physical.content_sha256:
        raise ValueError("geometry-guided proposal graph and physical map differ")
    xy_px = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent_px = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    descriptor = np.asarray(query_context_descriptors, dtype=np.float64)
    depth = np.asarray(query_depth_z, dtype=np.float64).reshape(-1)
    query_normal = np.asarray(query_normal_cam, dtype=np.float64).reshape(-1, 3)
    geometry_conf = np.asarray(geometry_confidence, dtype=np.float64).reshape(-1)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    out_of_map = np.asarray(parent_out_of_map_probabilities, dtype=np.float64).reshape(-1)
    unresolved = np.asarray(parent_unresolved_probabilities, dtype=np.float64).reshape(-1)
    child_rows = np.asarray(child_posterior.best_child_rows_by_parent, dtype=np.int64)
    child_probability = np.asarray(
        child_posterior.best_child_probabilities_by_parent, dtype=np.float64,
    )
    support_count = xy_px.shape[0]
    if (
        xy_px.shape != extent_px.shape
        or descriptor.ndim != 2 or descriptor.shape[0] != support_count
        or depth.shape != (support_count,)
        or query_normal.shape != (support_count, 3)
        or geometry_conf.shape != (support_count,)
        or parent_ids.shape != parent_probability.shape
        or parent_ids.shape[0] != support_count
        or out_of_map.shape != (support_count,)
        or unresolved.shape != (support_count,)
        or child_posterior.conditional_parent_ids is None
        or child_posterior.best_child_rows_by_parent is None
        or child_posterior.best_child_probabilities_by_parent is None
        or not np.array_equal(child_posterior.conditional_parent_ids, parent_ids)
    ):
        raise ValueError("geometry-guided proposal arrays or conditional children differ")
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    parent_rows = np.full(parent_ids.shape, -1, dtype=np.int64)
    for support in range(parent_ids.shape[0]):
        for slot in range(parent_ids.shape[1]):
            parent_rows[support, slot] = row_by_id.get(int(parent_ids[support, slot]), -1)
    slot_count = min(int(maximum_parent_candidates), int(parent_ids.shape[1]))
    valid_candidate = (
        (parent_rows[:, :slot_count] >= 0)
        & (child_rows[:, :slot_count] >= 0)
        & (parent_probability[:, :slot_count] > 0.0)
        & (child_probability[:, :slot_count] > 0.0)
    )
    eligible = np.flatnonzero(
        np.any(valid_candidate, axis=1)
        & np.isfinite(depth) & (depth > 0.05)
        & np.isfinite(geometry_conf) & (geometry_conf >= float(minimum_geometry_confidence))
        & (out_of_map < 0.98)
    )
    if eligible.size < 3:
        return _empty_modes()
    query_xyz = unproject_query_depth(xy_px, depth, camera)
    image_size = np.asarray([float(camera.width), float(camera.height)], dtype=np.float64)
    xy_normalized = xy_px / image_size
    extent_normalized = extent_px / image_size
    leverage = 0.5 + np.linalg.norm(xy_normalized - 0.5, axis=1)
    support_weight = (
        np.clip(1.0 - out_of_map, 1e-4, 1.0)
        * np.sqrt(np.clip(geometry_conf, 1e-4, 1.0))
        * leverage
    )
    sampling = support_weight[eligible]
    sampling /= np.sum(sampling)
    extension_order = eligible[np.argsort(-support_weight[eligible], kind="stable")]
    extension_order = extension_order[: min(int(extension_support_count), extension_order.size)]
    candidate_mass = (
        parent_probability[:, :slot_count]
        * child_probability[:, :slot_count]
        * valid_candidate
    )
    extension_children = child_rows[extension_order, :slot_count]
    extension_parents = parent_rows[extension_order, :slot_count]
    extension_mass = candidate_mass[extension_order, :slot_count]
    extension_valid = valid_candidate[extension_order, :slot_count]
    extension_world = physical.child_centers[np.maximum(extension_children, 0)]
    extension_query = query_xyz[extension_order]
    covisibility = graph.covisibility_matrix().astype(np.float64)
    phase_compatibility = covisibility
    phase_compatibility_radius = None
    if int(phase_anchor_count) > 0:
        phase_compatibility, phase_compatibility_radius = _physical_phase_compatibility(
            graph, physical,
        )
    rng = np.random.default_rng(int(random_seed))
    seed_capacity = max(int(maximum_modes), int(preliminary_pose_count) * int(seed_pool_multiplier))
    seed_heap: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    phase_anchor_seed_items: list[
        tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]
    ] = []
    phase_anchor_by_serial: dict[int, int] = {}
    mapping_anchor_label_offset: int | None = None
    phase_anchors = np.zeros((0,), dtype=np.int64)
    serial = 0
    if bool(deterministic_pair_beam):
        beam_support = _diverse_support_subset(
            eligible[np.argsort(-support_weight[eligible], kind="stable")],
            xy_normalized,
            maximum_count=int(pair_beam_support_count),
        )
        deterministic_seeds = _deterministic_pair_beam_seeds(
            beam_support=beam_support,
            extension_support=extension_order,
            xy_normalized=xy_normalized,
            query_xyz=query_xyz,
            query_normal=query_normal,
            support_weight=support_weight,
            candidate_mass=candidate_mass,
            valid_candidate=valid_candidate,
            child_rows=child_rows,
            parent_rows=parent_rows,
            child_centers=physical.child_centers,
            child_normals=physical.child_normals,
            covisibility=covisibility,
            pair_candidate_count=int(pair_candidate_count),
            extension_candidate_count=int(extension_candidate_count),
            pair_hypotheses_per_support_pair=int(pair_hypotheses_per_support_pair),
            maximum_support_pairs=int(maximum_support_pairs),
            extensions_per_pair=int(extensions_per_pair),
            minimum_pair_image_separation=float(minimum_pair_image_separation),
            minimum_triangle_quality=float(minimum_triangle_quality),
            minimum_scale=float(minimum_scale),
            maximum_scale=float(maximum_scale),
            triangle_log_ratio_sigma=float(triangle_log_ratio_sigma),
            normal_proposal_weight=float(normal_proposal_weight),
            orientation_normal_weight=float(orientation_normal_weight),
            marginal_pair_consensus=bool(marginal_pair_consensus),
        )
        if diagnostics_out is not None:
            diagnostics_out["pair_beam_support_count"] = int(beam_support.size)
            diagnostics_out["pair_beam_maximum_support_pairs"] = int(maximum_support_pairs)
            diagnostics_out["deterministic_pair_beam"] = _pose_stage_diagnostic(
                [value[1] for value in deterministic_seeds], diagnostic_pose_w2c,
            )
        for seed_score, pose, seed_parent, seed_support, fitted_scale in deterministic_seeds:
            item = (
                float(seed_score), serial, pose, seed_parent, seed_support,
                float(fitted_scale),
            )
            serial += 1
            if len(seed_heap) < seed_capacity:
                heapq.heappush(seed_heap, item)
            elif seed_score > seed_heap[0][0]:
                heapq.heapreplace(seed_heap, item)
        anchor_common: dict[str, object] = {
            "beam_support": beam_support,
            "extension_support": extension_order,
            "xy_normalized": xy_normalized,
            "query_xyz": query_xyz,
            "query_normal": query_normal,
            "support_weight": support_weight,
            "child_centers": physical.child_centers,
            "child_normals": physical.child_normals,
            "extensions_per_pair": int(extensions_per_pair),
            "minimum_pair_image_separation": float(minimum_pair_image_separation),
            "minimum_triangle_quality": float(minimum_triangle_quality),
            "minimum_scale": float(minimum_scale),
            "maximum_scale": float(maximum_scale),
            "triangle_log_ratio_sigma": float(triangle_log_ratio_sigma),
            "normal_proposal_weight": float(normal_proposal_weight),
            "orientation_normal_weight": float(orientation_normal_weight),
            "marginal_pair_consensus": True,
            "covisibility": phase_compatibility,
        }
        anchor_seeds, phase_anchors, anchor_seed_labels = _phase_anchor_pair_beam_seeds(
            parent_rows=parent_rows[:, :slot_count],
            candidate_mass=candidate_mass,
            valid_candidate=valid_candidate,
            child_rows=child_rows[:, :slot_count],
            phase_compatibility=phase_compatibility,
            anchor_count=int(phase_anchor_count),
            anchor_message_power=float(phase_anchor_message_power),
            anchor_support_pairs=int(phase_anchor_support_pairs),
            anchor_candidate_count=int(phase_anchor_candidate_count),
            anchor_extension_count=int(phase_anchor_extension_count),
            anchor_hypotheses=int(phase_anchor_hypotheses),
            common_arguments={
                key: value for key, value in anchor_common.items()
                if key != "covisibility"
            },
        )
        if diagnostics_out is not None:
            if phase_compatibility_radius is not None:
                diagnostics_out["phase_compatibility_radius_m"] = float(
                    phase_compatibility_radius
                )
            diagnostics_out["phase_anchor_parent_rows"] = phase_anchors.tolist()
            diagnostics_out["phase_anchor_seed_count"] = int(len(anchor_seeds))
            diagnostics_out["phase_anchor_beam"] = _pose_stage_diagnostic(
                [value[1] for value in anchor_seeds], diagnostic_pose_w2c,
            )
        for anchor_label, (
            seed_score, pose, seed_parent, seed_support, fitted_scale,
        ) in zip(anchor_seed_labels.tolist(), anchor_seeds):
            item = (
                float(seed_score), serial, pose, seed_parent, seed_support,
                float(fitted_scale),
            )
            phase_anchor_by_serial[serial] = int(anchor_label)
            serial += 1
            # Anchor scores are calibrated only within a phase.  Comparing
            # them in one global heap silently deletes lower-mass but valid
            # physical phases.  Preserve the bounded per-anchor beam until a
            # pose-conditioned surface likelihood provides a common score.
            phase_anchor_seed_items.append(item)
        if mapping_view_parent_weights is not None:
            if mapping_view_anchor_scores is None:
                raise ValueError("mapping-view anchors require anchor scores")
            view_seeds, view_seed_labels = _mapping_view_pair_beam_seeds(
                parent_rows=parent_rows[:, :slot_count],
                candidate_mass=candidate_mass,
                valid_candidate=valid_candidate,
                child_rows=child_rows[:, :slot_count],
                view_parent_weights=np.asarray(mapping_view_parent_weights),
                view_anchor_scores=np.asarray(mapping_view_anchor_scores),
                anchor_message_power=float(mapping_view_message_power),
                anchor_support_pairs=int(mapping_view_support_pairs),
                anchor_candidate_count=int(mapping_view_candidate_count),
                anchor_extension_count=int(mapping_view_extension_count),
                anchor_hypotheses=int(mapping_view_hypotheses),
                common_arguments=anchor_common,
            )
            if diagnostics_out is not None:
                diagnostics_out["mapping_view_anchor_count"] = int(
                    np.asarray(mapping_view_parent_weights).shape[0]
                )
                diagnostics_out["mapping_view_anchor_seed_count"] = int(len(view_seeds))
                diagnostics_out["mapping_view_anchor_beam"] = _mapping_view_seed_diagnostic(
                    view_seeds, view_seed_labels, diagnostic_pose_w2c,
                )
            label_offset = int(phase_anchors.size)
            mapping_anchor_label_offset = label_offset
            for view_label, (
                seed_score, pose, seed_parent, seed_support, fitted_scale,
            ) in zip(view_seed_labels.tolist(), view_seeds):
                item = (
                    float(seed_score), serial, pose, seed_parent, seed_support,
                    float(fitted_scale),
                )
                phase_anchor_by_serial[serial] = label_offset + int(view_label)
                serial += 1
                phase_anchor_seed_items.append(item)
    for _ in range(0 if bool(deterministic_pair_beam) else int(proposal_trials)):
        pair = rng.choice(eligible, size=2, replace=False, p=sampling)
        if np.linalg.norm(xy_normalized[pair[0]] - xy_normalized[pair[1]]) < float(minimum_pair_image_separation):
            continue
        first_slot = _weighted_choice(rng, np.sqrt(np.maximum(candidate_mass[pair[0]], 0.0)))
        if first_slot < 0:
            continue
        first_parent = int(parent_rows[pair[0], first_slot])
        second_weight = np.sqrt(np.maximum(candidate_mass[pair[1]], 0.0))
        safe_parent = np.maximum(parent_rows[pair[1], :slot_count], 0)
        second_weight *= np.power(0.05 + 0.95 * covisibility[safe_parent, first_parent], 0.75)
        second_weight[parent_rows[pair[1], :slot_count] < 0] = 0.0
        second_slot = _weighted_choice(rng, second_weight)
        if second_slot < 0:
            continue
        seed_children = [
            int(child_rows[pair[0], first_slot]),
            int(child_rows[pair[1], second_slot]),
        ]
        if seed_children[0] == seed_children[1]:
            continue
        world_pair = physical.child_centers[np.asarray(seed_children, dtype=np.int64)]
        world_distance = float(np.linalg.norm(world_pair[1] - world_pair[0]))
        query_distance = float(np.linalg.norm(query_xyz[pair[1]] - query_xyz[pair[0]]))
        if world_distance <= 0.20 or query_distance <= 0.10:
            continue
        pair_scale = query_distance / world_distance
        if pair_scale < float(minimum_scale) or pair_scale > float(maximum_scale):
            continue
        world_d0 = np.linalg.norm(extension_world - world_pair[0], axis=2)
        world_d1 = np.linalg.norm(extension_world - world_pair[1], axis=2)
        query_d0 = np.linalg.norm(extension_query - query_xyz[pair[0]], axis=1)[:, None]
        query_d1 = np.linalg.norm(extension_query - query_xyz[pair[1]], axis=1)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_error = (
                np.abs(np.log((query_d0 / world_d0) / pair_scale))
                + np.abs(np.log((query_d1 / world_d1) / pair_scale))
            )
        safe_parent = np.maximum(extension_parents, 0)
        compatibility = np.maximum(
            0.05 + 0.95 * covisibility[safe_parent, first_parent], 1e-8,
        )
        extension_score = (
            -log_error / max(float(triangle_log_ratio_sigma), 1e-6)
            + 0.15 * np.log(np.maximum(extension_mass, 1e-12))
            + 0.10 * np.log(compatibility)
        )
        extension_mask = (
            extension_valid
            & (extension_children != seed_children[0])
            & (extension_children != seed_children[1])
            & (world_d0 > 0.10) & (world_d1 > 0.10)
            & (query_d0 > 0.10) & (query_d1 > 0.10)
            & np.isfinite(extension_score)
        )
        extension_mask[np.isin(extension_order, pair), :] = False
        flat_valid = np.flatnonzero(extension_mask.reshape(-1))
        if flat_valid.size == 0:
            continue
        take = min(int(extensions_per_pair), int(flat_valid.size))
        score_valid = extension_score.reshape(-1)[flat_valid]
        if take < flat_valid.size:
            chosen = np.argpartition(-score_valid, kth=take - 1)[:take]
            flat_valid = flat_valid[chosen]
        flat_valid = flat_valid[np.argsort(-extension_score.reshape(-1)[flat_valid], kind="stable")]
        for flat_index in flat_valid.tolist():
            extension_row, third_slot = np.unravel_index(flat_index, extension_score.shape)
            third_support = int(extension_order[extension_row])
            third_child = int(extension_children[extension_row, third_slot])
            candidate_extension_score = float(extension_score[extension_row, third_slot])
            selected_support = np.asarray([pair[0], pair[1], third_support], dtype=np.int64)
            selected_child = np.asarray([seed_children[0], seed_children[1], third_child], dtype=np.int64)
            selected_world = physical.child_centers[selected_child]
            selected_query = query_xyz[selected_support]
            if (
                _triangle_quality(selected_world) < float(minimum_triangle_quality)
                or _triangle_quality(selected_query) < float(minimum_triangle_quality)
            ):
                continue
            estimated = estimate_pose_from_scaled_query_geometry(
                selected_world,
                selected_query,
                physical.child_normals[selected_child],
                query_normal[selected_support],
                normal_weight=float(orientation_normal_weight),
            )
            if estimated is None:
                continue
            pose, fitted_scale = estimated
            if fitted_scale < float(minimum_scale) or fitted_scale > float(maximum_scale):
                continue
            normal_score = _normal_consistency(
                pose,
                physical.child_normals[selected_child],
                query_normal[selected_support],
                selected_query,
            )
            selected_slots = np.asarray([first_slot, second_slot, third_slot], dtype=np.int64)
            selected_parent = parent_rows[selected_support, selected_slots]
            selected_mass = candidate_mass[selected_support, selected_slots]
            seed_score = (
                candidate_extension_score
                + 0.10 * float(np.sum(np.log(np.maximum(selected_mass, 1e-12))))
                + float(normal_proposal_weight) * normal_score
                - abs(np.log(fitted_scale / pair_scale))
            )
            item = (
                seed_score, serial, pose, selected_parent.astype(np.int64),
                selected_support, fitted_scale,
            )
            serial += 1
            if len(seed_heap) < seed_capacity:
                heapq.heappush(seed_heap, item)
            elif seed_score > seed_heap[0][0]:
                heapq.heapreplace(seed_heap, item)
    seed_pool = list(seed_heap) + phase_anchor_seed_items
    if not seed_pool:
        return _empty_modes()
    if diagnostics_out is not None:
        diagnostics_out["retained_seed_heap"] = _pose_stage_diagnostic(
            [value[2] for value in seed_pool], diagnostic_pose_w2c,
        )

    seed_identity_vfm = None
    seed_identity_score_by_serial: dict[int, float] = {}
    if bool(seed_identity_vfm_alignment):
        required = (
            (
                query_token_canonical_descriptors,
                canonical_field_primitive_rows,
                canonical_field_codes,
                canonical_field_confidence,
                token_height,
                token_width,
            )
            if bool(seed_identity_canonical_primitives)
            else (
                query_token_context_descriptors,
                query_token_local_descriptors,
                map_parent_descriptors,
                map_child_descriptors,
                parent_descriptor_valid,
                child_descriptor_valid,
                token_height,
                token_width,
            )
        )
        if any(value is None for value in required):
            raise ValueError("seed identity VFM alignment requires complete canonical readouts")
        if bool(seed_identity_full_map_primitives) and not bool(
            seed_identity_canonical_primitives
        ):
            raise ValueError("full-map seed evidence requires canonical primitive codes")
        prescore_per_anchor = max(int(seed_identity_prescore_per_anchor), 0)
        if prescore_per_anchor > 0 and phase_anchor_by_serial:
            # Proposal scores have meaning only inside their structural
            # equivalence class.  Keep a bounded class-conditional beam before
            # asking the shared VFM field to compare poses across classes.
            anchor_heaps: dict[
                int, list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]]
            ] = {}
            global_items = []
            for value in seed_pool:
                anchor = phase_anchor_by_serial.get(int(value[1]))
                if anchor is None:
                    global_items.append(value)
                    continue
                heap = anchor_heaps.setdefault(int(anchor), [])
                if len(heap) < prescore_per_anchor:
                    heapq.heappush(heap, value)
                elif value[0] > heap[0][0]:
                    heapq.heapreplace(heap, value)
            global_items = sorted(
                global_items, key=lambda value: (-value[0], value[1]),
            )[: int(preliminary_pose_count)]
            seed_pool = global_items + [
                value for anchor in sorted(anchor_heaps) for value in anchor_heaps[anchor]
            ]
            if diagnostics_out is not None:
                diagnostics_out["structural_equivalence_prescreen"] = {
                    **_pose_stage_diagnostic(
                        [value[2] for value in seed_pool], diagnostic_pose_w2c,
                    ),
                    "per_anchor_capacity": int(prescore_per_anchor),
                    "anchor_count": int(len(anchor_heaps)),
                    "unanchored_capacity": int(preliminary_pose_count),
                    "score_semantics": "proposal_score_compared_within_anchor_only",
                }
        seed_poses = np.asarray([value[2] for value in seed_pool], dtype=np.float64)
        seed_parents = np.asarray([value[3] for value in seed_pool], dtype=np.int64)
        seed_query_mask = (
            None
            if bool(seed_identity_full_map_primitives)
            else _disconnected_query_token_masks(
                np.asarray([value[4] for value in seed_pool], dtype=np.int64),
                xy_normalized,
                extent_normalized,
                int(token_height),
                int(token_width),
            )
        )
        seed_allowed_parents = (
            None if bool(seed_identity_full_map_primitives) else seed_parents
        )
        if bool(seed_identity_canonical_primitives):
            seed_identity_vfm = score_pose_conditioned_sparse_primitives(
                seed_poses,
                np.asarray(query_token_canonical_descriptors),
                np.asarray(canonical_field_primitive_rows),
                np.asarray(canonical_field_codes),
                np.asarray(canonical_field_confidence),
                physical,
                camera,
                token_height=int(token_height),
                token_width=int(token_width),
                primitives_per_child=int(sparse_vfm_primitives_per_child),
                batch_size=int(sparse_vfm_batch_size),
                maximum_splat_radius_tokens=int(sparse_vfm_maximum_splat_radius_tokens),
                score_semantics="fixed_grid",
                allowed_parent_rows=seed_allowed_parents,
                query_token_mask=seed_query_mask,
                device=str(sparse_vfm_device),
            )
        else:
            seed_identity_vfm = score_pose_conditioned_sparse_vfm(
                seed_poses,
                np.asarray(query_token_context_descriptors),
                np.asarray(query_token_local_descriptors),
                np.asarray(map_parent_descriptors),
                np.asarray(map_child_descriptors),
                np.asarray(parent_descriptor_valid),
                np.asarray(child_descriptor_valid),
                physical,
                camera,
                token_height=int(token_height),
                token_width=int(token_width),
                temperature=float(sparse_vfm_temperature),
                batch_size=int(sparse_vfm_batch_size),
                maximum_splat_radius_tokens=int(sparse_vfm_maximum_splat_radius_tokens),
                score_semantics=str(sparse_vfm_score_semantics),
                allowed_parent_rows=seed_allowed_parents,
                query_token_mask=seed_query_mask,
                device=str(sparse_vfm_device),
            )
        sparse_seed_identity_vfm = seed_identity_vfm
        sparse_seed_pool = seed_pool
        exact_verify_count = min(
            max(int(seed_identity_exact_verify_count), 0), len(seed_pool),
        )
        if exact_verify_count > 0:
            if not bool(seed_identity_full_map_primitives):
                raise ValueError("exact seed verification requires full-map primitive evidence")
            sparse_order = np.argsort(-seed_identity_vfm.scores, kind="stable")
            exact_rows = sparse_order[:exact_verify_count]
            seed_pool = [seed_pool[int(row)] for row in exact_rows.tolist()]
            seed_identity_vfm = score_pose_conditioned_sparse_primitives(
                np.asarray([value[2] for value in seed_pool], dtype=np.float64),
                np.asarray(query_token_canonical_descriptors),
                np.asarray(canonical_field_primitive_rows),
                np.asarray(canonical_field_codes),
                np.asarray(canonical_field_confidence),
                physical,
                camera,
                token_height=int(token_height),
                token_width=int(token_width),
                primitives_per_child=0,
                batch_size=max(1, min(int(sparse_vfm_batch_size), 2)),
                maximum_splat_radius_tokens=int(sparse_vfm_maximum_splat_radius_tokens),
                score_semantics=str(sparse_primitive_score_semantics),
                device=str(sparse_vfm_device),
            )
        seed_identity_score_by_serial = {
            int(value[1]): float(seed_identity_vfm.scores[index])
            for index, value in enumerate(seed_pool)
        }
        if diagnostics_out is not None:
            sparse_order = np.argsort(-sparse_seed_identity_vfm.scores, kind="stable")
            diagnostics_out["full_map_sparse_seed_vfm_likelihood"] = {
                **_pose_stage_diagnostic(
                    [sparse_seed_pool[int(row)][2] for row in sparse_order.tolist()],
                    diagnostic_pose_w2c,
                ),
                "pose_count": int(len(sparse_seed_pool)),
                "full_map_evidence": bool(seed_identity_full_map_primitives),
                "primitives_per_child": int(sparse_vfm_primitives_per_child),
                "selected_score_semantics": str(sparse_primitive_score_semantics),
            }
            order = np.argsort(-seed_identity_vfm.scores, kind="stable")
            ranked_seed_snapshots = []
            for rank, row in enumerate(order[: min(16, order.size)].tolist(), start=1):
                value = seed_pool[int(row)]
                anchor_label = phase_anchor_by_serial.get(int(value[1]))
                mapping_score = None
                if (
                    anchor_label is not None
                    and mapping_anchor_label_offset is not None
                    and mapping_view_anchor_scores is not None
                    and int(anchor_label) >= int(mapping_anchor_label_offset)
                ):
                    local_anchor = int(anchor_label) - int(mapping_anchor_label_offset)
                    anchor_scores = np.asarray(mapping_view_anchor_scores).reshape(-1)
                    if 0 <= local_anchor < anchor_scores.size:
                        mapping_score = float(anchor_scores[local_anchor])
                snapshot: dict[str, object] = {
                    "rank": int(rank),
                    "pose_w2c": np.asarray(value[2], dtype=np.float64).tolist(),
                    "score": float(seed_identity_vfm.scores[int(row)]),
                    "structural_anchor_label": (
                        int(anchor_label) if anchor_label is not None else None
                    ),
                    "mapping_view_score": mapping_score,
                    "seed_parent_rows": np.asarray(value[3], dtype=np.int64).tolist(),
                    "seed_support_rows": np.asarray(value[4], dtype=np.int64).tolist(),
                }
                if diagnostic_pose_w2c is not None:
                    error = pnp_pose_error(value[2], diagnostic_pose_w2c)
                    snapshot.update({
                        "translation_m": float(error.translation_m),
                        "rotation_deg": float(error.rotation_deg),
                    })
                ranked_seed_snapshots.append(snapshot)
            diagnostics_out["disconnected_seed_vfm_likelihood"] = {
                **_pose_stage_diagnostic(
                    [seed_pool[int(row)][2] for row in order.tolist()],
                    diagnostic_pose_w2c,
                ),
                "score_min": float(np.min(seed_identity_vfm.scores)),
                "score_max": float(np.max(seed_identity_vfm.scores)),
                "rendered_coverage_mean": float(np.mean(seed_identity_vfm.rendered_coverage)),
                "feature_coverage_mean": float(np.mean(seed_identity_vfm.feature_coverage)),
                "fixed_denominator_token_count": int(token_height) * int(token_width),
                "denominator_semantics": (
                    "full_query_grid"
                    if bool(seed_identity_full_map_primitives)
                    else "fixed_union_of_seed_query_regions_per_candidate"
                ),
                "typed_missing_semantics": (
                    "missing_canonical_surface_zero_similarity"
                    if bool(seed_identity_full_map_primitives)
                    else "non_hypothesized_or_missing_canonical_surface_zero_llr"
                ),
                "canonical_primitive_codes": bool(seed_identity_canonical_primitives),
                "exact_all_primitive_verification": bool(exact_verify_count > 0),
                "exact_verify_count": int(exact_verify_count),
                "top_ranked_seed_snapshots": ranked_seed_snapshots,
            }

    preliminary_order = eligible[np.argsort(-support_weight[eligible], kind="stable")]
    preliminary_order = preliminary_order[: min(int(preliminary_support_count), preliminary_order.size)]
    preliminary_heap: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    anchor_count_total = len(set(phase_anchor_by_serial.values()))
    per_anchor_preliminary = (
        int(phase_anchor_keep_per_anchor)
        if int(phase_anchor_keep_per_anchor) > 0
        else max(
            8,
            min(
                64,
                int(np.ceil(
                    float(preliminary_pose_count) / max(int(anchor_count_total), 1)
                )),
            ),
        )
    )
    preliminary_anchor_heaps: dict[
        int, list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]]
    ] = {}
    for seed_score, item_serial, pose, seed_parent, seed_support, fitted_scale in seed_pool:
        surface_score, _parent, _child, support = _projected_surface_assignment(
            pose,
            xy_normalized[preliminary_order],
            extent_normalized[preliminary_order],
            parent_probability[preliminary_order],
            unresolved[preliminary_order],
            child_rows[preliminary_order],
            child_probability[preliminary_order],
            physical,
            camera,
            maximum_parent_candidates=min(8, slot_count),
        )
        score = (
            float(seed_identity_score_by_serial[int(item_serial)])
            if seed_identity_vfm is not None
            else float(surface_score + 0.05 * seed_score)
        )
        item = (score, item_serial, pose, seed_parent, seed_support, fitted_scale)
        if len(preliminary_heap) < int(preliminary_pose_count):
            heapq.heappush(preliminary_heap, item)
        elif score > preliminary_heap[0][0]:
            heapq.heapreplace(preliminary_heap, item)
        anchor = phase_anchor_by_serial.get(int(item_serial))
        if anchor is not None:
            anchor_heap = preliminary_anchor_heaps.setdefault(int(anchor), [])
            if len(anchor_heap) < int(per_anchor_preliminary):
                heapq.heappush(anchor_heap, item)
            elif score > anchor_heap[0][0]:
                heapq.heapreplace(anchor_heap, item)

    query_graph = build_query_edge_graph(xy_normalized, descriptor)
    if diagnostics_out is not None:
        diagnostics_out["global_preliminary_surface_heap"] = _pose_stage_diagnostic(
            [value[2] for value in preliminary_heap], diagnostic_pose_w2c,
        )
    preliminary_by_serial = {int(value[1]): value for value in preliminary_heap}
    for anchor_heap in preliminary_anchor_heaps.values():
        for value in anchor_heap:
            preliminary_by_serial.setdefault(int(value[1]), value)
    # Preserve an independent per-phase proposal channel as well.  The unary
    # surface score is precisely the evidence under test and cannot be the
    # sole gate deciding which structural hypotheses reach the joint score.
    proposal_anchor_heaps: dict[
        int, list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]]
    ] = {}
    for value in phase_anchor_seed_items:
        if seed_identity_vfm is not None and int(value[1]) not in seed_identity_score_by_serial:
            continue
        anchor = phase_anchor_by_serial[int(value[1])]
        anchor_heap = proposal_anchor_heaps.setdefault(int(anchor), [])
        if len(anchor_heap) < int(per_anchor_preliminary):
            heapq.heappush(anchor_heap, value)
        elif value[0] > anchor_heap[0][0]:
            heapq.heapreplace(anchor_heap, value)
    for anchor_heap in proposal_anchor_heaps.values():
        for value in anchor_heap:
            preliminary_by_serial.setdefault(int(value[1]), value)
    preliminary_pool = list(preliminary_by_serial.values())
    if diagnostics_out is not None:
        diagnostics_out["phase_anchor_preliminary_per_anchor"] = int(
            per_anchor_preliminary
        )
        diagnostics_out["preliminary_surface_heap"] = _pose_stage_diagnostic(
            [value[2] for value in preliminary_pool], diagnostic_pose_w2c,
        )
    evaluated = []
    ordered_preliminary = sorted(
        preliminary_pool, key=lambda value: (-value[0], value[1]),
    )
    sparse_vfm = None
    sparse_primitive_vfm = None
    if bool(pose_conditioned_vfm_alignment) and bool(pose_conditioned_primitive_vfm_alignment):
        raise ValueError("child and primitive sparse VFM alignment are mutually exclusive")
    if bool(pose_conditioned_primitive_vfm_alignment):
        required = (
            query_token_canonical_descriptors,
            canonical_field_primitive_rows,
            canonical_field_codes,
            canonical_field_confidence,
            token_height,
            token_width,
        )
        if any(value is None for value in required):
            raise ValueError("primitive VFM alignment requires the sole canonical field")
        sparse_primitive_vfm = score_pose_conditioned_sparse_primitives(
            np.asarray([value[2] for value in ordered_preliminary], dtype=np.float64),
            np.asarray(query_token_canonical_descriptors),
            np.asarray(canonical_field_primitive_rows),
            np.asarray(canonical_field_codes),
            np.asarray(canonical_field_confidence),
            physical,
            camera,
            token_height=int(token_height),
            token_width=int(token_width),
            primitives_per_child=int(sparse_vfm_primitives_per_child),
            batch_size=int(sparse_vfm_batch_size),
            maximum_splat_radius_tokens=int(sparse_vfm_maximum_splat_radius_tokens),
            score_semantics=str(sparse_primitive_score_semantics),
            device=str(sparse_vfm_device),
        )
        primitive_order = np.argsort(-sparse_primitive_vfm.scores, kind="stable")
        if diagnostics_out is not None:
            diagnostics_out["sparse_primitive_vfm_likelihood"] = {
                **_pose_stage_diagnostic(
                    [ordered_preliminary[int(row)][2] for row in primitive_order.tolist()],
                    diagnostic_pose_w2c,
                ),
                "score_min": float(np.min(sparse_primitive_vfm.scores)),
                "score_max": float(np.max(sparse_primitive_vfm.scores)),
                "rendered_coverage_mean": float(np.mean(
                    sparse_primitive_vfm.rendered_coverage
                )),
                "feature_coverage_mean": float(np.mean(
                    sparse_primitive_vfm.feature_coverage
                )),
                "sample_count": int(sparse_primitive_vfm.sample_count),
                "primitives_per_child": int(sparse_primitive_vfm.primitives_per_child),
                "fixed_denominator_token_count": int(token_height) * int(token_width),
                "map_feature_semantics": (
                    "all_physical_primitive_centers_zbuffer_plus_single_canonical_field_codes"
                    if int(sparse_vfm_primitives_per_child) <= 0
                    else "spatially_stratified_real_codes_from_single_canonical_primitive_field"
                ),
                "selected_score_semantics": str(sparse_primitive_score_semantics),
                "fixed_grid_ranking": _pose_stage_diagnostic(
                    [
                        ordered_preliminary[int(row)][2]
                        for row in np.argsort(
                            -sparse_primitive_vfm.fixed_grid_scores, kind="stable"
                        ).tolist()
                    ],
                    diagnostic_pose_w2c,
                ),
                "visible_sample_mean_ranking": _pose_stage_diagnostic(
                    [
                        ordered_preliminary[int(row)][2]
                        for row in np.argsort(
                            -sparse_primitive_vfm.visible_sample_mean_scores,
                            kind="stable",
                        ).tolist()
                    ],
                    diagnostic_pose_w2c,
                ),
            }
    elif bool(pose_conditioned_vfm_alignment):
        required = (
            query_token_context_descriptors,
            query_token_local_descriptors,
            map_parent_descriptors,
            map_child_descriptors,
            parent_descriptor_valid,
            child_descriptor_valid,
            token_height,
            token_width,
        )
        if any(value is None for value in required):
            raise ValueError("pose-conditioned VFM alignment requires complete query/map readouts")
        sparse_vfm = score_pose_conditioned_sparse_vfm(
            np.asarray([value[2] for value in ordered_preliminary], dtype=np.float64),
            np.asarray(query_token_context_descriptors),
            np.asarray(query_token_local_descriptors),
            np.asarray(map_parent_descriptors),
            np.asarray(map_child_descriptors),
            np.asarray(parent_descriptor_valid),
            np.asarray(child_descriptor_valid),
            physical,
            camera,
            token_height=int(token_height),
            token_width=int(token_width),
            temperature=float(sparse_vfm_temperature),
            batch_size=int(sparse_vfm_batch_size),
            maximum_splat_radius_tokens=int(sparse_vfm_maximum_splat_radius_tokens),
            score_semantics=str(sparse_vfm_score_semantics),
            device=str(sparse_vfm_device),
        )
        sparse_order = np.argsort(-sparse_vfm.scores, kind="stable")
        if diagnostics_out is not None:
            best_candidate_snapshot = None
            if diagnostic_pose_w2c is not None and ordered_preliminary:
                candidate_error = [
                    pnp_pose_error(value[2], diagnostic_pose_w2c)
                    for value in ordered_preliminary
                ]
                candidate_joint = np.asarray([
                    value.translation_m / 0.5 + value.rotation_deg / 5.0
                    for value in candidate_error
                ], dtype=np.float64)
                best_row = int(np.argmin(candidate_joint))
                best_value = candidate_error[best_row]
                best_candidate_snapshot = {
                    "pose_w2c": np.asarray(
                        ordered_preliminary[best_row][2], dtype=np.float64,
                    ).tolist(),
                    "translation_m": float(best_value.translation_m),
                    "rotation_deg": float(best_value.rotation_deg),
                    "selected_score": float(sparse_vfm.scores[best_row]),
                    "raw_cosine_score": float(sparse_vfm.raw_cosine_scores[best_row]),
                    "marginal_centered_score": float(
                        sparse_vfm.marginal_centered_scores[best_row]
                    ),
                    "log_partition_llr_score": float(
                        sparse_vfm.log_partition_llr_scores[best_row]
                    ),
                    "rendered_coverage": float(sparse_vfm.rendered_coverage[best_row]),
                    "feature_coverage": float(sparse_vfm.feature_coverage[best_row]),
                }
            diagnostics_out["sparse_vfm_likelihood"] = {
                **_pose_stage_diagnostic(
                    [ordered_preliminary[int(row)][2] for row in sparse_order.tolist()],
                    diagnostic_pose_w2c,
                ),
                "score_min": float(np.min(sparse_vfm.scores)),
                "score_max": float(np.max(sparse_vfm.scores)),
                "context_score_min": float(np.min(sparse_vfm.context_scores)),
                "context_score_max": float(np.max(sparse_vfm.context_scores)),
                "local_score_min": float(np.min(sparse_vfm.local_scores)),
                "local_score_max": float(np.max(sparse_vfm.local_scores)),
                "rendered_coverage_mean": float(np.mean(sparse_vfm.rendered_coverage)),
                "feature_coverage_mean": float(np.mean(sparse_vfm.feature_coverage)),
                "fixed_denominator_token_count": int(token_height) * int(token_width),
                "null_semantics": "per_query_token_marginal_map_feature_log_partition",
                "selected_score_semantics": str(sparse_vfm_score_semantics),
                "diagnostic_best_joint_candidate": best_candidate_snapshot,
                "raw_cosine_ranking": _pose_stage_diagnostic(
                    [
                        ordered_preliminary[int(row)][2]
                        for row in np.argsort(-sparse_vfm.raw_cosine_scores, kind="stable").tolist()
                    ],
                    diagnostic_pose_w2c,
                ),
                "marginal_centered_ranking": _pose_stage_diagnostic(
                    [
                        ordered_preliminary[int(row)][2]
                        for row in np.argsort(
                            -sparse_vfm.marginal_centered_scores, kind="stable"
                        ).tolist()
                    ],
                    diagnostic_pose_w2c,
                ),
                "log_partition_llr_ranking": _pose_stage_diagnostic(
                    [
                        ordered_preliminary[int(row)][2]
                        for row in np.argsort(
                            -sparse_vfm.log_partition_llr_scores, kind="stable"
                        ).tolist()
                    ],
                    diagnostic_pose_w2c,
                ),
            }
    for preliminary_index, (
        _score, item_serial, pose, seed_parent, seed_support, fitted_scale,
    ) in enumerate(ordered_preliminary):
        if sparse_primitive_vfm is not None:
            score = float(sparse_primitive_vfm.scores[preliminary_index])
            support = int(round(
                float(sparse_primitive_vfm.rendered_coverage[preliminary_index]) * support_count
            ))
            assigned_parent = np.full((support_count,), -1, dtype=np.int64)
            assigned_child = np.full((support_count,), -1, dtype=np.int64)
        elif sparse_vfm is not None:
            score = float(sparse_vfm.scores[preliminary_index])
            support = int(round(float(sparse_vfm.feature_coverage[preliminary_index]) * support_count))
            assigned_parent = np.full((support_count,), -1, dtype=np.int64)
            assigned_child = np.full((support_count,), -1, dtype=np.int64)
        elif seed_identity_vfm is not None:
            score = float(seed_identity_score_by_serial[int(item_serial)])
            surface_score, assigned_parent, assigned_child, support = _projected_surface_assignment(
                pose,
                xy_normalized,
                extent_normalized,
                parent_probability,
                unresolved,
                child_rows,
                child_probability,
                physical,
                camera,
                maximum_parent_candidates=slot_count,
            )
        elif bool(soft_identity_marginalization):
            soft_child_rows = (
                child_posterior.child_rows_by_parent
                if child_posterior.child_rows_by_parent is not None
                else child_rows
            )
            soft_child_probability = (
                child_posterior.child_probabilities_by_parent
                if child_posterior.child_probabilities_by_parent is not None
                else child_probability
            )
            soft = pose_conditioned_soft_maplet_evidence(
                pose,
                xy_normalized,
                extent_normalized,
                query_graph,
                parent_probability,
                out_of_map,
                unresolved,
                soft_child_rows,
                soft_child_probability,
                physical,
                camera,
                maximum_parent_candidates=min(
                    int(soft_parent_candidate_count), slot_count,
                ),
                edge_candidate_count=int(soft_edge_candidate_count),
                maximum_edges=int(soft_maximum_edges),
                edge_weight=float(query_edge_weight),
            )
            score = float(soft.score)
            assigned_parent = soft.chosen_parent_rows
            assigned_child = soft.chosen_child_rows
            support = int(soft.supporting_region_count)
        else:
            surface_score, assigned_parent, assigned_child, support = _projected_surface_assignment(
                pose,
                xy_normalized,
                extent_normalized,
                parent_probability,
                unresolved,
                child_rows,
                child_probability,
                physical,
                camera,
                maximum_parent_candidates=slot_count,
            )
            edge = score_pose_conditioned_query_edges(
                query_graph,
                xy_normalized,
                extent_normalized,
                assigned_child,
                pose,
                physical,
                camera,
            )["all"]
            edge_score = edge["fixed_vector_score"]
            if edge_score is None:
                edge_score = -2.0
            score = float(surface_score + float(query_edge_weight) * float(edge_score))
        padded_parent = np.full((4,), -1, dtype=np.int64)
        padded_support = np.full((4,), -1, dtype=np.int64)
        padded_parent[:3] = seed_parent
        padded_support[:3] = seed_support
        evaluated.append((
            score, support, pose, assigned_parent, assigned_child,
            padded_parent, padded_support, fitted_scale, item_serial,
        ))
    evaluated.sort(key=lambda value: (-value[0], -value[1], value[8]))
    if diagnostics_out is not None:
        diagnostics_out["full_surface_ranked"] = _pose_stage_diagnostic(
            [value[2] for value in evaluated], diagnostic_pose_w2c,
        )
    retained = []
    for candidate in evaluated:
        center = -candidate[2][:3, :3].T @ candidate[2][:3, 3]
        if any(
            np.linalg.norm(center - (-other[2][:3, :3].T @ other[2][:3, 3]))
            < float(translation_nms_m)
            and _rotation_distance_degrees(candidate[2], other[2]) < float(rotation_nms_deg)
            for other in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= int(maximum_modes):
            break
    if not retained:
        return _empty_modes()
    if diagnostics_out is not None:
        diagnostics_out["topn_after_nms"] = _pose_stage_diagnostic(
            [value[2] for value in retained], diagnostic_pose_w2c,
        )
    configuration_parent = (
        None
        if bool(pose_conditioned_vfm_alignment) or bool(pose_conditioned_primitive_vfm_alignment)
        else np.asarray([value[3] for value in retained], dtype=np.int64).reshape(-1, support_count)
    )
    configuration_child = (
        None
        if bool(pose_conditioned_vfm_alignment) or bool(pose_conditioned_primitive_vfm_alignment)
        else np.asarray([value[4] for value in retained], dtype=np.int64).reshape(-1, support_count)
    )
    mapping_labels = None
    mapping_scores = None
    if mapping_view_anchor_scores is not None and mapping_anchor_label_offset is not None:
        mapping_labels = np.full((len(retained),), -1, dtype=np.int64)
        mapping_scores = np.full((len(retained),), np.nan, dtype=np.float64)
        anchor_scores = np.asarray(mapping_view_anchor_scores, dtype=np.float64).reshape(-1)
        for index, value in enumerate(retained):
            anchor = phase_anchor_by_serial.get(int(value[8]))
            if anchor is None or int(anchor) < int(mapping_anchor_label_offset):
                continue
            local_anchor = int(anchor) - int(mapping_anchor_label_offset)
            if 0 <= local_anchor < anchor_scores.size:
                mapping_labels[index] = local_anchor
                mapping_scores[index] = float(anchor_scores[local_anchor])
    return CoarsePoseModes(
        np.asarray([value[2] for value in retained], dtype=np.float64).reshape(-1, 4, 4),
        np.asarray([value[0] for value in retained], dtype=np.float64),
        np.asarray([value[1] for value in retained], dtype=np.int64),
        configuration_parent,
        configuration_child,
        np.asarray([value[5] for value in retained], dtype=np.int64).reshape(-1, 4),
        np.asarray([value[6] for value in retained], dtype=np.int64).reshape(-1, 4),
        mapping_labels,
        mapping_scores,
    )
