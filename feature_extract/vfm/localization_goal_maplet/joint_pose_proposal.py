"""Joint latent physical-configuration and coarse SE(3) proposal.

The VFM posterior remains multi-modal until a minimal physical configuration
has produced a pose.  Only then are parent-conditioned child surfaces projected
and selected.  AP3P is an internal numerical proposal primitive; candidates
are accepted and ranked by fixed-support region/surface and query-edge factors,
not by the four center correspondences that generated them.
"""

from __future__ import annotations

import heapq

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

from .child_retrieval import ChildTilePosterior
from .physical_map import GoalMapletPhysicalMap
from .pose_proposal import CoarsePoseModes, _rotation_distance_degrees
from .query_edge_factor import (
    build_query_edge_graph,
    project_child_surfaces,
    score_pose_conditioned_query_edges,
)
from .typed_graph import TypedParentGraph


def _weighted_choice(rng: np.random.Generator, weight: np.ndarray) -> int:
    value = np.asarray(weight, dtype=np.float64).reshape(-1)
    total = float(np.sum(value))
    if not np.isfinite(total) or total <= 0.0:
        return -1
    return int(rng.choice(value.size, p=value / total))


def _projected_surface_assignment(
    pose: np.ndarray,
    query_xy_normalized: np.ndarray,
    query_extent_normalized: np.ndarray,
    parent_probability: np.ndarray,
    parent_unresolved: np.ndarray,
    child_rows_by_parent: np.ndarray,
    child_probability_by_parent: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera: ColmapCamera,
    *,
    maximum_parent_candidates: int,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Marginalize parent/child alternatives under one pose, then read MAP."""

    count = min(int(maximum_parent_candidates), int(parent_probability.shape[1]))
    children = np.asarray(child_rows_by_parent[:, :count], dtype=np.int64)
    probability = (
        np.asarray(parent_probability[:, :count], dtype=np.float64)
        * np.asarray(child_probability_by_parent[:, :count], dtype=np.float64)
    )
    flat_center, flat_extent, flat_visible = project_child_surfaces(
        children.reshape(-1), pose, physical, camera,
    )
    center = flat_center.reshape(children.shape + (2,))
    extent = flat_extent.reshape(children.shape + (2,))
    visible = flat_visible.reshape(children.shape) & (children >= 0) & (probability > 0.0)
    query_xy = np.asarray(query_xy_normalized, dtype=np.float64)
    query_extent = np.asarray(query_extent_normalized, dtype=np.float64)
    # A VFM region denotes an area, not an exact keypoint.  The observation is
    # compatible when its center falls inside (or close to) the projected tile.
    # Only a small fraction of the receptive-field extent expands the tile;
    # using the full support would make every repeated facade tile compatible.
    expanded_extent = extent + 0.10 * query_extent[:, None]
    outside = np.maximum(np.abs(query_xy[:, None] - center) - expanded_extent, 0.0)
    sigma = np.maximum(0.008, 0.20 * np.linalg.norm(query_extent, axis=1))
    normalized = np.linalg.norm(outside, axis=2) / sigma[:, None]
    likelihood = probability * np.exp(-0.5 * np.square(np.minimum(normalized, 8.0))) * visible
    surface_mass = np.sum(likelihood, axis=1)
    floor = np.maximum(0.02 * np.asarray(parent_unresolved, dtype=np.float64), 1e-8)
    in_map = np.clip(1.0 - np.asarray(parent_unresolved, dtype=np.float64), 0.0, 1.0)
    weight = np.maximum(in_map, 0.05)
    score = float(np.sum(weight * np.log(np.maximum(surface_mass, floor))) / np.sum(weight))
    slot = np.argmax(likelihood, axis=1)
    chosen_child = children[np.arange(children.shape[0]), slot]
    chosen_child[surface_mass <= floor] = -1
    chosen_parent = np.full(chosen_child.shape, -1, dtype=np.int64)
    valid = chosen_child >= 0
    chosen_parent[valid] = physical.child_parent_rows[chosen_child[valid]]
    supporting = int(np.sum(surface_mass > np.maximum(floor, 1e-4)))
    return score, chosen_parent, chosen_child, supporting


def generate_joint_configuration_pose_modes(
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    query_context_descriptors: np.ndarray,
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
    proposal_trials: int = 32768,
    maximum_parent_candidates: int = 32,
    preliminary_support_count: int = 96,
    preliminary_pose_count: int = 768,
    anchor_conditioned_fraction: float = 0.75,
    parent_probability_power: float = 0.5,
    covisibility_power: float = 0.75,
    query_edge_weight: float = 0.5,
    translation_nms_m: float = 0.20,
    rotation_nms_deg: float = 3.0,
    random_seed: int = 194917,
) -> CoarsePoseModes:
    """Generate Top-N modes without hardening a full identity assignment first."""

    if graph.physical_map_sha256 != physical.content_sha256:
        raise ValueError("joint proposal graph and physical map differ")
    xy_px = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent_px = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    descriptor = np.asarray(query_context_descriptors, dtype=np.float64)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    out_of_map = np.asarray(parent_out_of_map_probabilities, dtype=np.float64).reshape(-1)
    unresolved = np.asarray(parent_unresolved_probabilities, dtype=np.float64).reshape(-1)
    if (
        xy_px.shape != extent_px.shape
        or descriptor.ndim != 2
        or descriptor.shape[0] != xy_px.shape[0]
        or parent_ids.shape != parent_probability.shape
        or parent_ids.shape[0] != xy_px.shape[0]
        or out_of_map.shape != (xy_px.shape[0],)
        or unresolved.shape != out_of_map.shape
        or child_posterior.conditional_parent_ids is None
        or child_posterior.best_child_rows_by_parent is None
        or child_posterior.best_child_probabilities_by_parent is None
        or not np.array_equal(child_posterior.conditional_parent_ids, parent_ids)
    ):
        raise ValueError("joint proposal arrays or parent-conditioned children differ")
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    parent_rows = np.full(parent_ids.shape, -1, dtype=np.int64)
    for support in range(parent_ids.shape[0]):
        for slot in range(parent_ids.shape[1]):
            parent_rows[support, slot] = row_by_id.get(int(parent_ids[support, slot]), -1)
    child_rows = np.asarray(child_posterior.best_child_rows_by_parent, dtype=np.int64)
    child_probability = np.asarray(
        child_posterior.best_child_probabilities_by_parent, dtype=np.float64,
    )
    slot_count = min(int(maximum_parent_candidates), parent_ids.shape[1])
    valid = (
        (parent_rows[:, :slot_count] >= 0)
        & (child_rows[:, :slot_count] >= 0)
        & (parent_probability[:, :slot_count] > 0.0)
        & (child_probability[:, :slot_count] > 0.0)
    )
    eligible = np.flatnonzero(np.any(valid, axis=1) & (out_of_map < 0.98))
    if eligible.size < 4:
        return CoarsePoseModes(np.zeros((0, 4, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64))

    image_size = np.asarray([float(camera.width), float(camera.height)])
    xy_normalized = xy_px / image_size
    extent_normalized = extent_px / image_size
    image_center = 0.5 * image_size
    leverage = 0.5 + np.linalg.norm((xy_px - image_center) / image_size, axis=1)
    support_weight = np.clip(1.0 - out_of_map, 1e-4, 1.0) * leverage
    support_order = eligible[np.argsort(-support_weight[eligible], kind="stable")]
    preliminary_support = support_order[: min(int(preliminary_support_count), support_order.size)]
    preliminary_candidate_count = min(8, slot_count)
    preliminary_child = child_rows[preliminary_support, :preliminary_candidate_count]
    preliminary_probability = (
        parent_probability[preliminary_support, :preliminary_candidate_count]
        * child_probability[preliminary_support, :preliminary_candidate_count]
        * valid[preliminary_support, :preliminary_candidate_count]
    )
    support_sampling = support_weight[eligible]
    support_sampling /= np.sum(support_sampling)
    global_parent_mass = np.zeros((physical.maplet_ids.size,), dtype=np.float64)
    valid_parent = (parent_rows[:, :slot_count] >= 0) & (parent_probability[:, :slot_count] > 0.0)
    np.add.at(
        global_parent_mass,
        parent_rows[:, :slot_count][valid_parent],
        parent_probability[:, :slot_count][valid_parent],
    )
    anchor_rows = np.flatnonzero(global_parent_mass > 0.0)
    anchor_weight = np.power(global_parent_mass[anchor_rows], 0.5)
    anchor_weight /= np.sum(anchor_weight)
    covisibility = graph.covisibility_matrix().astype(np.float64)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rng = np.random.default_rng(int(random_seed))
    # Min-heap over the best provisional proposals.  The serial counter makes
    # equal-score entries deterministic without comparing numpy arrays.
    heap: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    serial = 0
    for _ in range(int(proposal_trials)):
        selected = rng.choice(eligible, size=4, replace=False, p=support_sampling)
        normalized_selected = xy_normalized[selected]
        if (
            np.ptp(normalized_selected[:, 0]) < 0.12
            or np.ptp(normalized_selected[:, 1]) < 0.12
        ):
            continue
        anchor = -1
        if anchor_rows.size and rng.random() < float(anchor_conditioned_fraction):
            anchor = int(rng.choice(anchor_rows, p=anchor_weight))
        selected_children = []
        selected_parents = []
        for support in selected.tolist():
            weight = np.power(
                np.maximum(parent_probability[support, :slot_count], 0.0),
                float(parent_probability_power),
            ) * valid[support]
            if anchor >= 0:
                safe = np.maximum(parent_rows[support, :slot_count], 0)
                compatibility = 0.05 + 0.95 * covisibility[safe, anchor]
                compatibility[parent_rows[support, :slot_count] < 0] = 0.0
                weight *= np.power(compatibility, float(covisibility_power))
            slot = _weighted_choice(rng, weight)
            if slot < 0:
                selected_children = []
                break
            selected_children.append(int(child_rows[support, slot]))
            selected_parents.append(int(parent_rows[support, slot]))
        if len(selected_children) != 4 or len(set(selected_children)) < 4:
            continue
        xyz = physical.child_centers[np.asarray(selected_children, dtype=np.int64)]
        try:
            success, rotations, translations, _ = cv2.solvePnPGeneric(
                xyz.astype(np.float64),
                xy_px[selected].astype(np.float64),
                matrix,
                distortion,
                flags=cv2.SOLVEPNP_AP3P,
            )
        except cv2.error:
            continue
        if not success:
            continue
        for rotation, translation in zip(rotations, translations):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = cv2.Rodrigues(rotation)[0]
            pose[:3, 3] = np.asarray(translation).reshape(3)
            if not np.all(np.isfinite(pose)):
                continue
            projected, _ = cv2.projectPoints(
                physical.child_centers[np.maximum(preliminary_child, 0).reshape(-1)].astype(np.float64),
                rotation,
                translation,
                matrix,
                distortion,
            )
            camera_xyz = (
                physical.child_centers[np.maximum(preliminary_child, 0)] @ pose[:3, :3].T
                + pose[:3, 3]
            )
            projected = projected.reshape(preliminary_child.shape + (2,))
            residual = np.linalg.norm(
                projected - xy_px[preliminary_support, None], axis=2,
            )
            sigma = np.maximum(10.0, 0.75 * np.linalg.norm(extent_px[preliminary_support], axis=1))
            valid_depth = (camera_xyz[..., 2] > 0.05) & (preliminary_child >= 0)
            likelihood = preliminary_probability * np.exp(
                -0.5 * np.square(np.minimum(residual / sigma[:, None], 8.0))
            ) * valid_depth
            likelihood = np.sum(likelihood, axis=1)
            floor = np.maximum(0.02 * unresolved[preliminary_support], 1e-8)
            provisional = float(np.mean(np.log(np.maximum(likelihood, floor))))
            item = (
                provisional,
                serial,
                pose,
                np.asarray(selected_parents, dtype=np.int64),
                np.asarray(selected, dtype=np.int64),
            )
            serial += 1
            if len(heap) < int(preliminary_pose_count):
                heapq.heappush(heap, item)
            elif provisional > heap[0][0]:
                heapq.heapreplace(heap, item)

    if not heap:
        return CoarsePoseModes(np.zeros((0, 4, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64))
    query_graph = build_query_edge_graph(xy_normalized, descriptor)
    evaluated = []
    for _preliminary, _serial, pose, seed_parent, seed_support in sorted(heap, key=lambda value: (-value[0], value[1])):
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
        padded_seed = np.full((4,), -1, dtype=np.int64)
        padded_support = np.full((4,), -1, dtype=np.int64)
        padded_seed[: seed_parent.size] = seed_parent
        padded_support[: seed_support.size] = seed_support
        evaluated.append((
            score, support, pose, assigned_parent, assigned_child,
            padded_seed, padded_support,
        ))
    evaluated.sort(key=lambda value: (-value[0], -value[1]))
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
    return CoarsePoseModes(
        np.asarray([value[2] for value in retained], dtype=np.float64).reshape(-1, 4, 4),
        np.asarray([value[0] for value in retained], dtype=np.float64),
        np.asarray([value[1] for value in retained], dtype=np.int64),
        np.asarray([value[3] for value in retained], dtype=np.int64).reshape(-1, xy_px.shape[0]),
        np.asarray([value[4] for value in retained], dtype=np.int64).reshape(-1, xy_px.shape[0]),
        np.asarray([value[5] for value in retained], dtype=np.int64).reshape(-1, 4),
        np.asarray([value[6] for value in retained], dtype=np.int64).reshape(-1, 4),
    )
