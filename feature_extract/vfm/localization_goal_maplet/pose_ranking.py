"""Pose ranking with exact rendered region identity and full-scene occlusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

from .canonical_field import CanonicalSurfaceField
from .child_retrieval import ChildTilePosterior
from .physical_map import GoalMapletPhysicalMap
from .pose_proposal import CoarsePoseModes
from .surface_renderer import dominant_child_owner, render_surface_identity


def _conditional_child_probability(joint: float, parent: float, floor: float) -> float:
    return float(np.clip(float(joint) / max(float(parent), float(floor)), 0.0, 1.0))


def _render_child_splats(
    physical: GoalMapletPhysicalMap,
    pose: np.ndarray,
    camera,
    *,
    token_height: int,
    token_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cheap region renderer used to pre-rank pose modes before exact audit."""

    import cv2

    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation, _ = cv2.Rodrigues(np.asarray(pose[:3, :3], dtype=np.float64))
    projected, _ = cv2.projectPoints(
        physical.child_centers.astype(np.float64), rotation, pose[:3, 3], matrix, distortion
    )
    projected = projected.reshape(-1, 2)
    camera_xyz = physical.child_centers @ pose[:3, :3].T + pose[:3, 3]
    depth = camera_xyz[:, 2]
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    view = camera_center[None] - physical.child_centers
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
    incidence = np.sum(physical.child_normals * view, axis=1)
    parent = physical.child_parent_rows
    front = (physical.maplet_sidedness[parent] == 2) | (incidence >= 0.02)
    xy = projected * np.asarray([token_width / camera.width, token_height / camera.height])
    focal = 0.5 * float(matrix[0, 0] + matrix[1, 1])
    radius_world = np.maximum(np.linalg.norm(physical.child_extents[:, :2], axis=1), 0.02)
    radius_px = focal * radius_world / np.maximum(depth, 1e-4)
    radius = radius_px * 0.5 * (token_width / camera.width + token_height / camera.height)
    radius = np.clip(radius, 0.6, 5.0)
    valid = (
        front & (depth > 0.05)
        & (xy[:, 0] >= -radius) & (xy[:, 0] < token_width + radius)
        & (xy[:, 1] >= -radius) & (xy[:, 1] < token_height + radius)
    )
    zbuffer = np.full((token_height, token_width), np.inf, dtype=np.float64)
    child_image = np.full((token_height, token_width), -1, dtype=np.int64)
    # Near surfaces win, making this a conservative child-level z-buffer.
    for child in np.flatnonzero(valid)[np.argsort(depth[valid], kind="stable")].tolist():
        x, y, value = float(xy[child, 0]), float(xy[child, 1]), float(radius[child])
        x0, x1 = max(0, int(np.floor(x - value))), min(token_width, int(np.ceil(x + value + 1)))
        y0, y1 = max(0, int(np.floor(y - value))), min(token_height, int(np.ceil(y + value + 1)))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = np.square((xx + 0.5 - x) / value) + np.square((yy + 0.5 - y) / value) <= 1.0
        replace = inside & (depth[child] < zbuffer[y0:y1, x0:x1])
        zbuffer[y0:y1, x0:x1][replace] = depth[child]
        child_image[y0:y1, x0:x1][replace] = int(child)
    return child_image, child_image >= 0


@dataclass(frozen=True)
class RenderIdentityRanking:
    modes: CoarsePoseModes
    identity_scores: np.ndarray
    parent_log_likelihood: np.ndarray
    child_log_likelihood: np.ndarray
    rendered_coverage: np.ndarray
    proposal_scores: np.ndarray
    original_indices: np.ndarray


@dataclass(frozen=True)
class CascadeIdentityRanking:
    modes: CoarsePoseModes
    cheap_identity_scores: np.ndarray
    cheap_parent_log_likelihood: np.ndarray
    cheap_child_log_likelihood: np.ndarray
    cheap_rendered_coverage: np.ndarray
    proposal_scores: np.ndarray
    original_indices: np.ndarray
    exact_evaluated: np.ndarray
    exact_identity_scores: np.ndarray
    exact_parent_log_likelihood: np.ndarray
    exact_child_log_likelihood: np.ndarray
    exact_rendered_coverage: np.ndarray


def merge_cascade_identity_rankings(
    cheap: RenderIdentityRanking,
    exact: RenderIdentityRanking,
    *,
    exact_candidate_count: int,
) -> CascadeIdentityRanking:
    """Merge a cheap Top-N screen and exact Top-K rerank without losing alignment."""

    count = int(cheap.modes.poses_w2c.shape[0])
    take = min(int(exact_candidate_count), count)
    if exact.modes.poses_w2c.shape[0] != take:
        raise ValueError("exact cascade result differs from requested candidate count")
    local_order = np.asarray(exact.original_indices, dtype=np.int64)
    if sorted(local_order.tolist()) != list(range(take)):
        raise ValueError("exact cascade ordering is not a permutation of its input")
    final_order = np.concatenate([local_order, np.arange(take, count, dtype=np.int64)])
    score = np.asarray(cheap.identity_scores, dtype=np.float64)[final_order]
    score[:take] = np.asarray(exact.identity_scores, dtype=np.float64)
    modes = CoarsePoseModes(
        np.asarray(cheap.modes.poses_w2c)[final_order],
        score,
        np.asarray(cheap.modes.supporting_region_count)[final_order],
        (
            np.asarray(cheap.modes.configuration_parent_rows)[final_order]
            if cheap.modes.configuration_parent_rows is not None else None
        ),
        (
            np.asarray(cheap.modes.configuration_child_rows)[final_order]
            if cheap.modes.configuration_child_rows is not None else None
        ),
        (
            np.asarray(cheap.modes.proposal_seed_parent_rows)[final_order]
            if cheap.modes.proposal_seed_parent_rows is not None else None
        ),
        (
            np.asarray(cheap.modes.proposal_seed_support_rows)[final_order]
            if cheap.modes.proposal_seed_support_rows is not None else None
        ),
        (
            np.asarray(cheap.modes.mapping_view_anchor_labels)[final_order]
            if cheap.modes.mapping_view_anchor_labels is not None else None
        ),
        (
            np.asarray(cheap.modes.mapping_view_prior_scores)[final_order]
            if cheap.modes.mapping_view_prior_scores is not None else None
        ),
    )

    def exact_array(values: np.ndarray) -> np.ndarray:
        # Unevaluated values are explicit zeros and must only be consumed with
        # exact_evaluated; this keeps the serialized artifact strict JSON.
        output = np.zeros((count,), dtype=np.float64)
        output[:take] = np.asarray(values, dtype=np.float64)
        return output

    return CascadeIdentityRanking(
        modes=modes,
        cheap_identity_scores=np.asarray(cheap.identity_scores)[final_order],
        cheap_parent_log_likelihood=np.asarray(cheap.parent_log_likelihood)[final_order],
        cheap_child_log_likelihood=np.asarray(cheap.child_log_likelihood)[final_order],
        cheap_rendered_coverage=np.asarray(cheap.rendered_coverage)[final_order],
        proposal_scores=np.asarray(cheap.proposal_scores)[final_order],
        original_indices=np.asarray(cheap.original_indices)[final_order],
        exact_evaluated=np.arange(count) < take,
        exact_identity_scores=exact_array(exact.identity_scores),
        exact_parent_log_likelihood=exact_array(exact.parent_log_likelihood),
        exact_child_log_likelihood=exact_array(exact.child_log_likelihood),
        exact_rendered_coverage=exact_array(exact.rendered_coverage),
    )


def rerank_modes_with_rendered_identity(
    modes: CoarsePoseModes,
    member_offsets: np.ndarray,
    member_token_indices: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_null_probabilities: np.ndarray,
    child_posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    camera,
    *,
    token_height: int,
    token_width: int,
    parent_weight: float = 0.65,
    probability_floor: float = 1e-4,
    render_mode: str = "child_splat",
    device: str = "cuda",
) -> RenderIdentityRanking:
    """Rank complete pose modes; losing rendered coverage never improves score."""

    offsets = np.asarray(member_offsets, dtype=np.int64).reshape(-1)
    members = np.asarray(member_token_indices, dtype=np.int64).reshape(-1)
    group_count = offsets.size - 1
    token_count = int(token_height) * int(token_width)
    if offsets[0] != 0 or offsets[-1] != members.size or members.size != token_count:
        raise ValueError("rendered identity ranking requires an all-token grouping")
    token_group = np.full((token_count,), -1, dtype=np.int64)
    for group in range(group_count):
        token_group[members[offsets[group] : offsets[group + 1]]] = group
    if np.any(token_group < 0):
        raise ValueError("grouping does not cover every query token")
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    parent_null = np.asarray(parent_null_probabilities, dtype=np.float64).reshape(-1)
    if parent_ids.shape != parent_probability.shape or parent_ids.shape[0] != group_count:
        raise ValueError("parent posterior and grouping differ")
    maplet_row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    parent_candidate_rows = np.full(parent_ids.shape, -1, dtype=np.int64)
    for group in range(group_count):
        for slot in range(parent_ids.shape[1]):
            parent_candidate_rows[group, slot] = maplet_row_by_id.get(int(parent_ids[group, slot]), -1)
    child_owner = dominant_child_owner(physical)
    parent_scores, child_scores, coverages = [], [], []
    for pose in modes.poses_w2c:
        if str(render_mode) == "full_2dgs":
            rendered = render_surface_identity(
                physical, pose, camera,
                width=int(token_width), height=int(token_height), device=str(device),
            )
            primitive = np.asarray(rendered.primitive_rows, dtype=np.int64).reshape(-1)
            visible = np.asarray(rendered.mask, dtype=bool).reshape(-1) & (primitive >= 0)
            rendered_child = np.full((token_count,), -1, dtype=np.int64)
            rendered_child[visible] = child_owner[primitive[visible]]
        elif str(render_mode) == "child_splat":
            child_image, child_mask = _render_child_splats(
                physical, pose, camera,
                token_height=int(token_height), token_width=int(token_width),
            )
            rendered_child = child_image.reshape(-1)
            visible = child_mask.reshape(-1)
        else:
            raise ValueError(f"unknown identity render mode: {render_mode}")
        rendered_parent = np.full((token_count,), -1, dtype=np.int64)
        child_valid = rendered_child >= 0
        rendered_parent[child_valid] = physical.child_parent_rows[rendered_child[child_valid]]
        parent_value = np.zeros((token_count,), dtype=np.float64)
        child_value = np.zeros((token_count,), dtype=np.float64)
        for token in range(token_count):
            group = int(token_group[token])
            if rendered_parent[token] >= 0:
                selected = parent_candidate_rows[group] == rendered_parent[token]
                parent_value[token] = float(np.sum(parent_probability[group, selected]))
            else:
                parent_value[token] = float(parent_null[group])
            if rendered_child[token] >= 0:
                selected = child_posterior.candidate_child_rows[group] == rendered_child[token]
                joint = float(np.sum(child_posterior.candidate_probabilities[group, selected]))
                # Child retrieval stores P(parent|context)P(child|parent,local).
                # The parent factor is scored separately above, so divide it
                # out instead of counting the same context evidence twice.
                parent_mass = float(parent_value[token])
                child_value[token] = _conditional_child_probability(
                    joint, parent_mass, float(probability_floor)
                )
            else:
                child_value[token] = float(child_posterior.null_probabilities[group])
        # Every query token participates.  Background and missing field regions
        # are explicit null observations rather than disappearing denominators.
        parent_scores.append(float(np.mean(np.log(np.maximum(parent_value, float(probability_floor))))))
        child_scores.append(float(np.mean(np.log(np.maximum(child_value, float(probability_floor))))))
        coverages.append(float(np.mean(visible)))
    parent_array = np.asarray(parent_scores, dtype=np.float64)
    child_array = np.asarray(child_scores, dtype=np.float64)
    identity = float(parent_weight) * parent_array + (1.0 - float(parent_weight)) * child_array
    # Region RANSAC support is used only as a tie-breaker.  The calibrated
    # rendered categorical likelihood defines the ranking semantics.
    order = np.lexsort((-modes.supporting_region_count, -identity))
    reranked = CoarsePoseModes(
        modes.poses_w2c[order], identity[order], modes.supporting_region_count[order],
        (
            modes.configuration_parent_rows[order]
            if modes.configuration_parent_rows is not None else None
        ),
        (
            modes.configuration_child_rows[order]
            if modes.configuration_child_rows is not None else None
        ),
        (
            modes.proposal_seed_parent_rows[order]
            if modes.proposal_seed_parent_rows is not None else None
        ),
        (
            modes.proposal_seed_support_rows[order]
            if modes.proposal_seed_support_rows is not None else None
        ),
        (
            modes.mapping_view_anchor_labels[order]
            if modes.mapping_view_anchor_labels is not None else None
        ),
        (
            modes.mapping_view_prior_scores[order]
            if modes.mapping_view_prior_scores is not None else None
        ),
    )
    return RenderIdentityRanking(
        modes=reranked,
        identity_scores=identity[order],
        parent_log_likelihood=parent_array[order],
        child_log_likelihood=child_array[order],
        rendered_coverage=np.asarray(coverages, dtype=np.float64)[order],
        proposal_scores=np.asarray(modes.scores, dtype=np.float64)[order],
        original_indices=np.asarray(order, dtype=np.int64),
    )
