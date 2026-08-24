"""Differentiable sparse source-substochastic pose transport.

The NumPy implementation in :mod:`candidate_conditioned_pose_attribution`
is the semantic authority.  This module separates the discrete, frozen edge
construction from the trainable compatibility values.  It never regresses an
absolute pose and never introduces point correspondences or PnP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
    _STAGE,
    pose_transport_hierarchy_content_sha256,
)
from .trainable_pose_transport import MinimalPoseTransportReadout, QueryPoseHeadOutput


TORCH_TRANSPORT_SEMANTICS = "differentiable_sparse_substochastic_pose_transport_v1"
FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS = (
    "differentiable_fixed_local_kernel_source_target_capacity_pose_transport_v2"
)
QUERY_TOKEN_CAPACITY_TRANSPORT_SEMANTICS = (
    "query_token_to_candidate_slot_fixed_kernel_capacity_pose_transport_v1"
)
CATEGORICAL_TOKEN_CAPACITY_TRANSPORT_SEMANTICS = (
    "categorical_token_layout_fixed_kernel_source_target_capacity_transport_v1"
)


@dataclass(frozen=True)
class FrozenSparseTransportEdges:
    source_index: np.ndarray
    target_index: np.ndarray
    hierarchy_score: np.ndarray
    layout_score: np.ndarray
    source_count: int
    target_count: int
    stage: str


@dataclass(frozen=True)
class DifferentiablePoseTransportResult:
    combined_score: torch.Tensor
    matched_source_probability: torch.Tensor
    unmatched_source_probability: torch.Tensor
    source_probability: torch.Tensor
    edge_probability: torch.Tensor
    edge_source_index: torch.Tensor
    matched_target_probability: torch.Tensor | None = None
    target_probability: torch.Tensor | None = None
    transport_semantics: str = TORCH_TRANSPORT_SEMANTICS
    production_eligible: bool = False


@dataclass(frozen=True)
class QueryTokenCapacityPoseEnergy:
    combined_score: torch.Tensor
    token_score: torch.Tensor
    matched_source_probability: torch.Tensor
    unmatched_source_probability: torch.Tensor
    matched_target_probability: torch.Tensor
    target_probability: torch.Tensor
    local_radius_tokens: int
    minimum_cosine_evidence: float
    transport_semantics: str = QUERY_TOKEN_CAPACITY_TRANSPORT_SEMANTICS
    production_eligible: bool = False

    @property
    def score(self) -> torch.Tensor:
        return self.combined_score


def categorical_token_capacity_pose_transport(
    source_category_ids: torch.Tensor,
    source_category_probability: torch.Tensor,
    query_reliability: torch.Tensor,
    target_category_ids: torch.Tensor,
    target_category_probability: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    height: int = 36,
    width: int = 64,
    local_radius_tokens: int = 1,
) -> QueryTokenCapacityPoseEnergy:
    """Transport parent/support probability by local categorical layout.

    This is the coarse/medium identity channel.  It deliberately uses stable
    physical parent/support categories instead of exact children or visual
    appearance.  Duplicate target slots with the same category are harmless:
    their masses remain separate capacities and their per-token sum is still
    bounded by one.
    """

    source_ids = torch.as_tensor(source_category_ids, dtype=torch.long)
    source = torch.as_tensor(source_category_probability)
    if source.ndim != 2 or source_ids.shape != source.shape or not torch.is_floating_point(source):
        raise ValueError("source categorical posterior must be aligned floating matrices")
    device, dtype = source.device, source.dtype
    source_ids = source_ids.to(device=device)
    token_count, source_slots = int(source.shape[0]), int(source.shape[1])
    target_ids = torch.as_tensor(target_category_ids, device=device, dtype=torch.long)
    target = torch.as_tensor(target_category_probability, device=device, dtype=dtype)
    valid = torch.as_tensor(target_valid, device=device, dtype=torch.bool)
    reliability = torch.as_tensor(query_reliability, device=device, dtype=dtype).reshape(-1)
    if (
        token_count != int(height) * int(width)
        or target.ndim != 2
        or target_ids.shape != target.shape
        or valid.shape != target.shape
        or target.shape[0] != token_count
        or reliability.shape != (token_count,)
    ):
        raise ValueError("categorical token transport shapes differ")
    if any(not torch.isfinite(value).all() for value in (source, target, reliability)):
        raise ValueError("categorical token transport inputs must be finite")
    tolerance = 2.0e-5
    if (
        torch.any(source < 0.0)
        or torch.any(target < 0.0)
        or torch.any(reliability < 0.0)
        or torch.any(source.sum(dim=1) > 1.0 + tolerance)
        or torch.any(target.sum(dim=1) > 1.0 + tolerance)
    ):
        raise ValueError("categorical token transport masses are invalid")
    radius = int(local_radius_tokens)
    if radius < 0 or radius > 4:
        raise ValueError("local_radius_tokens must lie in [0,4]")
    target_slots = int(target.shape[1])
    source_id_grid = source_ids.reshape(int(height), int(width), source_slots)
    source_grid = source.reshape(int(height), int(width), source_slots)
    target_id_grid = target_ids.reshape(int(height), int(width), target_slots)
    target_grid = target.reshape(int(height), int(width), target_slots)
    target_valid_grid = valid.reshape(int(height), int(width), target_slots)
    matched_source_grid = torch.zeros_like(source_grid)
    matched_target_grid = torch.zeros_like(target_grid)
    spatial_kernel = 1.0 / float((2 * radius + 1) ** 2)
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            sy0, sy1 = max(0, -shift_y), min(int(height), int(height) - shift_y)
            sx0, sx1 = max(0, -shift_x), min(int(width), int(width) - shift_x)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            ty0, ty1 = sy0 + shift_y, sy1 + shift_y
            tx0, tx1 = sx0 + shift_x, sx1 + shift_x
            source_id = source_id_grid[sy0:sy1, sx0:sx1]
            target_id = target_id_grid[ty0:ty1, tx0:tx1]
            active = (
                (source_id[:, :, :, None] >= 0)
                & (target_id[:, :, None, :] >= 0)
                & target_valid_grid[ty0:ty1, tx0:tx1, None, :]
                & (source_id[:, :, :, None] == target_id[:, :, None, :])
            )
            allocation = (
                source_grid[sy0:sy1, sx0:sx1, :, None]
                * target_grid[ty0:ty1, tx0:tx1, None, :]
                * float(spatial_kernel)
                * active
            )
            matched_source_grid[sy0:sy1, sx0:sx1] += allocation.sum(dim=3)
            matched_target_grid[ty0:ty1, tx0:tx1] += allocation.sum(dim=2)
    capacity_tolerance = 3.0e-6
    if torch.any(matched_source_grid > source_grid + capacity_tolerance):
        raise AssertionError("categorical transport exceeds a source capacity")
    if torch.any(matched_target_grid > target_grid + capacity_tolerance):
        raise AssertionError("categorical transport exceeds a target capacity")
    unmatched = torch.clamp_min(source_grid - matched_source_grid, 0.0)
    token_matched = matched_source_grid.sum(dim=2)
    token_score = -1.0 + 2.0 * token_matched
    reliability_sum = reliability.sum()
    combined = torch.where(
        reliability_sum > 1.0e-12,
        torch.sum(reliability * token_score.reshape(-1))
        / reliability_sum.clamp_min(1.0e-12),
        torch.full((), -1.0, device=device, dtype=dtype),
    )
    return QueryTokenCapacityPoseEnergy(
        combined_score=combined,
        token_score=token_score.reshape(-1),
        matched_source_probability=matched_source_grid.reshape(-1),
        unmatched_source_probability=unmatched.reshape(-1),
        matched_target_probability=matched_target_grid.reshape_as(target),
        target_probability=target,
        local_radius_tokens=radius,
        minimum_cosine_evidence=-1.0,
        transport_semantics=CATEGORICAL_TOKEN_CAPACITY_TRANSPORT_SEMANTICS,
    )


def query_token_capacity_pose_transport(
    query_feature: torch.Tensor,
    source_token_probability: torch.Tensor,
    query_reliability: torch.Tensor,
    target_feature: torch.Tensor,
    target_probability: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    height: int = 36,
    width: int = 64,
    local_radius_tokens: int = 1,
    minimum_cosine_evidence: float = -1.0,
) -> QueryTokenCapacityPoseEnergy:
    """Conservative candidate-conditioned token-to-slot re-attribution.

    Global child retrieval is used to find a physical basin, but it is not
    trusted as an exact pose-stage child label.  Each query token instead
    transports its retained sub-probability directly to rendered target slots
    in a fixed local window.  The translation-invariant kernel sums to one on
    the infinite grid and is never renormalized at image boundaries.  Thus
    both source rows and target columns are sub-stochastic:

    ``allocation = source_mass * target_mass * K * nonnegative_similarity``.

    Removing a target slot, lowering its mass, or invalidating its feature can
    only delete allocation.  No keypoint, point correspondence, PnP solve, or
    absolute-pose regression is introduced.
    """

    query = torch.as_tensor(query_feature)
    if query.ndim != 2 or not torch.is_floating_point(query):
        raise ValueError("query token feature must be a floating matrix")
    device, dtype = query.device, query.dtype
    token_count, channels = int(query.shape[0]), int(query.shape[1])
    if token_count != int(height) * int(width):
        raise ValueError("query token feature differs from the declared grid")
    source = torch.as_tensor(
        source_token_probability, device=device, dtype=dtype
    ).reshape(-1)
    reliability = torch.as_tensor(
        query_reliability, device=device, dtype=dtype
    ).reshape(-1)
    target = torch.as_tensor(target_feature, device=device, dtype=dtype)
    target_mass = torch.as_tensor(target_probability, device=device, dtype=dtype)
    valid = torch.as_tensor(target_valid, device=device, dtype=torch.bool)
    if (
        source.shape != (token_count,)
        or reliability.shape != (token_count,)
        or target.ndim != 3
        or target.shape[0] != token_count
        or target.shape[2] != channels
        or target_mass.shape != target.shape[:2]
        or valid.shape != target_mass.shape
    ):
        raise ValueError("query-token capacity transport shapes differ")
    if any(
        not torch.isfinite(value).all()
        for value in (query, source, reliability, target, target_mass)
    ):
        raise ValueError("query-token capacity transport inputs must be finite")
    tolerance = 2.0e-5
    if (
        torch.any(source < 0.0)
        or torch.any(source > 1.0 + tolerance)
        or torch.any(reliability < 0.0)
        or torch.any(target_mass < 0.0)
        or torch.any(target_mass.sum(dim=1) > 1.0 + tolerance)
    ):
        raise ValueError("query-token capacity transport masses are invalid")
    radius = int(local_radius_tokens)
    if radius < 0 or radius > 4:
        raise ValueError("local_radius_tokens must lie in [0,4]")
    cosine_floor = float(minimum_cosine_evidence)
    if not np.isfinite(cosine_floor) or not -1.0 <= cosine_floor < 1.0:
        raise ValueError("minimum_cosine_evidence must lie in [-1,1)")

    query_norm = torch.linalg.vector_norm(query, dim=1)
    query_valid = query_norm >= 1.0e-8
    query_unit = query / query_norm[:, None].clamp_min(1.0e-8)
    target_norm = torch.linalg.vector_norm(target, dim=2)
    target_slot_valid = valid & (target_norm >= 1.0e-8)
    target_unit = target / target_norm[:, :, None].clamp_min(1.0e-8)
    slots = int(target.shape[1])
    q_grid = query_unit.reshape(int(height), int(width), channels)
    q_valid_grid = query_valid.reshape(int(height), int(width))
    source_grid = source.reshape(int(height), int(width))
    target_grid = target_unit.reshape(int(height), int(width), slots, channels)
    target_mass_grid = target_mass.reshape(int(height), int(width), slots)
    target_valid_grid = target_slot_valid.reshape(int(height), int(width), slots)
    matched_source_grid = torch.zeros_like(source_grid)
    matched_target_grid = torch.zeros_like(target_mass_grid)
    spatial_kernel = 1.0 / float((2 * radius + 1) ** 2)
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            sy0, sy1 = max(0, -shift_y), min(int(height), int(height) - shift_y)
            sx0, sx1 = max(0, -shift_x), min(int(width), int(width) - shift_x)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            ty0, ty1 = sy0 + shift_y, sy1 + shift_y
            tx0, tx1 = sx0 + shift_x, sx1 + shift_x
            q_value = q_grid[sy0:sy1, sx0:sx1]
            target_value = target_grid[ty0:ty1, tx0:tx1]
            cosine = torch.sum(q_value[:, :, None, :] * target_value, dim=3).clamp(
                -1.0, 1.0
            )
            similarity = ((cosine - cosine_floor) / (1.0 - cosine_floor)).clamp(
                0.0, 1.0
            )
            active = (
                q_valid_grid[sy0:sy1, sx0:sx1, None]
                & target_valid_grid[ty0:ty1, tx0:tx1]
            )
            allocation = (
                source_grid[sy0:sy1, sx0:sx1, None]
                * target_mass_grid[ty0:ty1, tx0:tx1]
                * float(spatial_kernel)
                * torch.where(active, similarity, torch.zeros_like(similarity))
            )
            matched_source_grid[sy0:sy1, sx0:sx1] += allocation.sum(dim=2)
            matched_target_grid[ty0:ty1, tx0:tx1] += allocation

    capacity_tolerance = 3.0e-6
    if torch.any(matched_source_grid > source_grid + capacity_tolerance):
        raise AssertionError("query-token transport exceeds a source capacity")
    if torch.any(matched_target_grid > target_mass_grid + capacity_tolerance):
        raise AssertionError("query-token transport exceeds a target capacity")
    unmatched = torch.clamp_min(source_grid - matched_source_grid, 0.0)
    token_score = -1.0 + 2.0 * matched_source_grid
    reliability_sum = reliability.sum()
    combined = torch.where(
        reliability_sum > 1.0e-12,
        torch.sum(reliability * token_score.reshape(-1))
        / reliability_sum.clamp_min(1.0e-12),
        torch.full((), -1.0, device=device, dtype=dtype),
    )
    return QueryTokenCapacityPoseEnergy(
        combined_score=combined,
        token_score=token_score.reshape(-1),
        matched_source_probability=matched_source_grid.reshape(-1),
        unmatched_source_probability=unmatched.reshape(-1),
        matched_target_probability=matched_target_grid.reshape_as(target_mass),
        target_probability=target_mass,
        local_radius_tokens=radius,
        minimum_cosine_evidence=cosine_floor,
    )


def _validated_hierarchy(hierarchy: PoseTransportHierarchy) -> tuple[np.ndarray, tuple[set[int], ...]]:
    parent = np.asarray(hierarchy.child_parent_ids, dtype=np.int64).reshape(-1)
    support = np.asarray(hierarchy.child_support_ids, dtype=np.int64).reshape(-1)
    offsets = np.asarray(hierarchy.adjacency_offsets, dtype=np.int64).reshape(-1)
    rows = np.asarray(hierarchy.adjacency_child_rows, dtype=np.int64).reshape(-1)
    if (
        support.shape != parent.shape or offsets.shape != (parent.size + 1,)
        or offsets[0] != 0 or offsets[-1] != rows.size
        or np.any(np.diff(offsets) < 0) or np.any(parent < 0)
        or np.any(support < -1) or np.any(rows < 0) or np.any(rows >= parent.size)
    ):
        raise ValueError("invalid sparse transport hierarchy")
    adjacency = tuple(set(rows[offsets[i] : offsets[i + 1]].tolist()) for i in range(parent.size))
    if any(i in values for i, values in enumerate(adjacency)):
        raise ValueError("sparse transport adjacency contains self edges")
    if any(i not in adjacency[j] for i, values in enumerate(adjacency) for j in values):
        raise ValueError("sparse transport adjacency must be symmetric")
    if str(hierarchy.content_sha256) != pose_transport_hierarchy_content_sha256(
        parent, support, offsets, rows
    ):
        raise ValueError("sparse transport hierarchy content hash differs")
    return support, adjacency


def build_frozen_sparse_transport_edges(
    source_child_rows: np.ndarray,
    source_child_probabilities: np.ndarray,
    token_xy: np.ndarray,
    target_child_rows: np.ndarray,
    target_child_weights: np.ndarray,
    hierarchy: PoseTransportHierarchy,
    *,
    stage: str,
    minimum_source_probability: float = 1.0e-6,
    minimum_target_weight: float = 1.0e-6,
) -> FrozenSparseTransportEdges:
    """Build a deterministic local edge set without looking at descriptors."""

    if str(stage) not in _STAGE:
        raise ValueError("pose transport stage must be coarse, medium, or fine")
    source_rows = np.asarray(source_child_rows, dtype=np.int64)
    source_mass = np.asarray(source_child_probabilities, dtype=np.float64)
    target_rows = np.asarray(target_child_rows, dtype=np.int64)
    target_mass = np.asarray(target_child_weights, dtype=np.float64)
    xy = np.asarray(token_xy, dtype=np.int64)
    if (
        source_rows.ndim != 2 or source_mass.shape != source_rows.shape
        or target_rows.ndim != 2 or target_mass.shape != target_rows.shape
        or target_rows.shape[0] != source_rows.shape[0]
        or xy.shape != (source_rows.shape[0], 2)
        or np.any(~np.isfinite(source_mass)) or np.any(source_mass < 0.0)
        or np.any(~np.isfinite(target_mass)) or np.any(target_mass < 0.0)
        or np.unique(xy, axis=0).shape[0] != xy.shape[0]
    ):
        raise ValueError("invalid frozen sparse transport masses")
    if not np.isfinite(minimum_source_probability) or minimum_source_probability <= 0.0:
        raise ValueError("minimum_source_probability must be positive")
    if not np.isfinite(minimum_target_weight) or minimum_target_weight <= 0.0:
        raise ValueError("minimum_target_weight must be positive")
    parent = np.asarray(hierarchy.child_parent_ids, dtype=np.int64).reshape(-1)
    support, adjacency = _validated_hierarchy(hierarchy)
    if np.any(source_rows >= parent.size) or np.any(target_rows >= parent.size):
        raise ValueError("transport child row exceeds hierarchy")
    config = _STAGE[str(stage)]
    relations = set(config["relations"])
    radius = int(config["radius"])
    # Per-token child uniqueness prevents duplicate evidence from inflating a
    # source's outgoing mass.  The actual edge join below is vectorized over a
    # bounded token neighbourhood; a Python loop over all 64 retrieval slots
    # was measured to dominate the real 36x64 experiment.
    for token in range(target_rows.shape[0]):
        active = target_rows[token]
        active = active[active >= 0]
        if np.unique(active).size != active.size:
            raise ValueError("target contains duplicate child at one token")
    for token in range(source_rows.shape[0]):
        active = source_rows[token]
        active = active[active >= 0]
        if np.unique(active).size != active.size:
            raise ValueError("source contains duplicate child at one token")

    source_slots = source_rows.shape[1]
    target_slots = target_rows.shape[1]
    coordinate_to_token = {tuple(value.tolist()): row for row, value in enumerate(xy)}
    neighbour_count = (2 * radius + 1) ** 2
    neighbour_token = np.full((xy.shape[0], neighbour_count), -1, dtype=np.int64)
    neighbour_distance = np.zeros((xy.shape[0], neighbour_count), dtype=np.int64)
    for token, (x, y) in enumerate(xy.tolist()):
        position = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbour_token[token, position] = coordinate_to_token.get((x + dx, y + dy), -1)
                neighbour_distance[token, position] = max(abs(dx), abs(dy))
                position += 1
    target_index = (
        neighbour_token[:, :, None] * target_slots
        + np.arange(target_slots, dtype=np.int64)[None, None, :]
    ).reshape(xy.shape[0], -1)
    target_present = target_index >= 0
    target_safe = np.maximum(target_index, 0)
    target_child_flat = target_rows.reshape(-1)
    target_mass_flat = target_mass.reshape(-1)
    local_target_child = target_child_flat[target_safe]
    local_target_active = (
        target_present & (local_target_child >= 0)
        & (target_mass_flat[target_safe] >= float(minimum_target_weight))
    )
    local_layout = np.repeat(
        1.0 - neighbour_distance / float(radius + 1), target_slots, axis=1
    ).astype(np.float32)
    adjacency_pairs = np.asarray(sorted(
        child * parent.size + other
        for child, values in enumerate(adjacency) for other in values
    ), dtype=np.int64)

    edge_source_parts: list[np.ndarray] = []
    edge_target_parts: list[np.ndarray] = []
    hierarchy_parts: list[np.ndarray] = []
    layout_parts: list[np.ndarray] = []
    # 128 tokens bounds temporary pair arrays to roughly one million entries
    # for the real 64-source/4-target medium stage.
    for begin in range(0, xy.shape[0], 128):
        end = min(begin + 128, xy.shape[0])
        source_child = source_rows[begin:end, :, None]
        target_child = local_target_child[begin:end, None, :]
        valid = (
            (source_child >= 0)
            & (source_mass[begin:end, :, None] >= float(minimum_source_probability))
            & local_target_active[begin:end, None, :]
        )
        hierarchy_score = np.zeros(valid.shape, dtype=np.float32)
        exact = valid & (source_child == target_child)
        if "exact" in relations:
            hierarchy_score[exact] = 1.0
        if "support" in relations:
            source_support = support[np.maximum(source_child, 0)]
            target_support = support[np.maximum(target_child, 0)]
            match = valid & ~exact & (source_support >= 0) & (source_support == target_support)
            hierarchy_score[match] = 0.7
        if "adjacent" in relations and adjacency_pairs.size:
            pair_key = (
                np.maximum(source_child, 0) * parent.size + np.maximum(target_child, 0)
            )
            position = np.searchsorted(adjacency_pairs, pair_key)
            adjacent_match = position < adjacency_pairs.size
            adjacent_match[adjacent_match] &= (
                adjacency_pairs[position[adjacent_match]] == pair_key[adjacent_match]
            )
            match = valid & (hierarchy_score == 0.0) & adjacent_match
            hierarchy_score[match] = 0.45
        if "parent" in relations:
            source_parent = parent[np.maximum(source_child, 0)]
            target_parent = parent[np.maximum(target_child, 0)]
            match = valid & (hierarchy_score == 0.0) & (source_parent == target_parent)
            hierarchy_score[match] = 0.25
        keep = valid & (hierarchy_score > 0.0)
        local_token, source_slot, local_target = np.nonzero(keep)
        if local_token.size == 0:
            continue
        absolute_token = begin + local_token
        edge_source_parts.append(absolute_token * source_slots + source_slot)
        edge_target_parts.append(target_index[absolute_token, local_target])
        hierarchy_parts.append(hierarchy_score[local_token, source_slot, local_target])
        layout_parts.append(local_layout[absolute_token, local_target])
    edge_source = np.concatenate(edge_source_parts) if edge_source_parts else np.zeros(0, dtype=np.int64)
    edge_target = np.concatenate(edge_target_parts) if edge_target_parts else np.zeros(0, dtype=np.int64)
    hierarchy_value = np.concatenate(hierarchy_parts) if hierarchy_parts else np.zeros(0, dtype=np.float32)
    layout_value = np.concatenate(layout_parts) if layout_parts else np.zeros(0, dtype=np.float32)
    order = np.lexsort((edge_target, edge_source)) if edge_source.size else np.zeros(0, dtype=np.int64)
    return FrozenSparseTransportEdges(
        source_index=edge_source[order], target_index=edge_target[order],
        hierarchy_score=hierarchy_value[order], layout_score=layout_value[order],
        source_count=int(source_rows.size), target_count=int(target_rows.size),
        stage=str(stage),
    )


def differentiable_sparse_pose_transport(
    model: MinimalPoseTransportReadout,
    query: QueryPoseHeadOutput,
    source_probability: torch.Tensor,
    query_reliability: torch.Tensor,
    target_weight: torch.Tensor,
    target_canonical_feature: torch.Tensor,
    target_normal_camera: torch.Tensor,
    target_double_sided: torch.Tensor,
    target_relative_depth: torch.Tensor,
    target_boundary: torch.Tensor,
    target_validity: torch.Tensor,
    target_confidence: torch.Tensor,
    edges: FrozenSparseTransportEdges,
    *,
    depth_scale: float = 0.25,
) -> DifferentiablePoseTransportResult:
    """Evaluate trainable edge logits with an explicit per-source sink."""

    if str(edges.stage) not in _STAGE:
        raise ValueError("unknown frozen edge stage")
    if query.normal_frame != "camera":
        raise ValueError("query normal must be expressed in camera frame")
    if query.depth_semantics != _STAGE[str(edges.stage)]["depth"]:
        raise ValueError("query depth semantics differ from the transport stage")
    source = torch.as_tensor(source_probability)
    target_mass = torch.as_tensor(target_weight, device=source.device, dtype=source.dtype).reshape(-1)
    if source.ndim != 2 or int(source.numel()) != int(edges.source_count):
        raise ValueError("source probability differs from frozen edges")
    token_count = int(source.shape[0])
    reliability = torch.as_tensor(query_reliability, device=source.device, dtype=source.dtype).reshape(-1)
    if reliability.shape != (token_count,) or torch.any(reliability < 0.0):
        raise ValueError("query reliability must align with tokens")
    query_code = query.pose_code.permute(0, 2, 3, 1).reshape(-1, query.pose_code.shape[1])
    if query_code.shape[0] != token_count:
        raise ValueError("query head output must contain one batch item aligned to tokens")
    query_normal = query.normal_camera.permute(0, 2, 3, 1).reshape(-1, 3)
    query_depth = query.relative_depth.reshape(-1)
    query_boundary = query.boundary.reshape(-1)
    query_valid = torch.stack([
        query.pose_code_valid.reshape(-1), query.normal_valid.reshape(-1),
        query.depth_valid.reshape(-1), query.boundary_valid.reshape(-1),
    ], dim=1)
    query_conf = query.confidence.permute(0, 2, 3, 1).reshape(-1, 4)
    map_code, map_code_valid = model.project_map_code(target_canonical_feature)
    map_code = map_code.reshape(-1, map_code.shape[-1])
    map_normal = torch.as_tensor(target_normal_camera, device=source.device, dtype=source.dtype).reshape(-1, 3)
    map_depth = torch.as_tensor(target_relative_depth, device=source.device, dtype=source.dtype).reshape(-1)
    map_boundary = torch.as_tensor(target_boundary, device=source.device, dtype=source.dtype).reshape(-1)
    map_double = torch.as_tensor(target_double_sided, device=source.device, dtype=torch.bool).reshape(-1)
    map_valid = torch.as_tensor(target_validity, device=source.device, dtype=torch.bool).reshape(-1, 4).clone()
    map_conf = torch.as_tensor(target_confidence, device=source.device, dtype=source.dtype).reshape(-1, 4)
    map_valid[:, 0] &= map_code_valid.reshape(-1)
    map_valid[:, 1] &= torch.linalg.vector_norm(map_normal, dim=1) >= float(
        model.config.zero_norm_threshold
    )
    if target_mass.numel() != int(edges.target_count) or map_code.shape[0] != int(edges.target_count):
        raise ValueError("target observation differs from frozen edges")
    finite_inputs = {
        "source": source, "reliability": reliability,
        "target_mass": target_mass, "map_code": map_code,
        "map_normal": map_normal, "map_depth": map_depth,
        "map_boundary": map_boundary, "query_code": query_code,
        "query_normal": query_normal, "query_depth": query_depth,
        "query_boundary": query_boundary, "query_confidence": query_conf,
        "map_confidence": map_conf,
    }
    nonfinite = [name for name, value in finite_inputs.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(
            "differentiable transport inputs must be finite: " + ",".join(nonfinite)
        )
    if torch.any(source < 0.0) or torch.any(target_mass < 0.0):
        raise ValueError("transport masses must be nonnegative")
    if torch.any((query_conf < 0.0) | (query_conf > 1.0)) or torch.any((map_conf < 0.0) | (map_conf > 1.0)):
        raise ValueError("transport confidence must lie in [0,1]")

    edge_source = torch.as_tensor(edges.source_index, dtype=torch.long, device=source.device)
    edge_target = torch.as_tensor(edges.target_index, dtype=torch.long, device=source.device)
    source_token = torch.div(edge_source, source.shape[1], rounding_mode="floor")
    weights = model.edge_weights().to(dtype=source.dtype)
    config = _STAGE[str(edges.stage)]
    edge_logit = model.source_bias.to(dtype=source.dtype) + torch.log(target_mass[edge_target].clamp_min(1.0e-12))
    modality_similarity: list[torch.Tensor] = []
    feature_cos = torch.sum(query_code[source_token] * map_code[edge_target], dim=1)
    modality_similarity.append(0.5 * (1.0 + feature_cos.clamp(-1.0, 1.0)))
    normal_cos = torch.sum(query_normal[source_token] * map_normal[edge_target], dim=1).clamp(-1.0, 1.0)
    modality_similarity.append(torch.where(map_double[edge_target], torch.abs(normal_cos), 0.5 * (1.0 + normal_cos)))
    modality_similarity.append(1.0 - torch.clamp(
        torch.abs(query_depth[source_token] - map_depth[edge_target]) / float(depth_scale), 0.0, 1.0
    ))
    modality_similarity.append(1.0 - torch.abs(query_boundary[source_token] - map_boundary[edge_target]).clamp(0.0, 1.0))
    for modality in range(4):
        active = query_valid[source_token, modality] & map_valid[edge_target, modality]
        confidence_product = (
            query_conf[source_token, modality] * map_conf[edge_target, modality]
        )
        # The exact geometric mean has an infinite derivative at zero. Real
        # typed observations intentionally contain zero confidence, so a
        # naive sqrt yields finite forward scores but NaN gradients through
        # ``0 * inf``. This zero-preserving smooth form has bounded gradient,
        # remains monotone, and differs from sqrt(x) by at most sqrt(eps).
        confidence_epsilon = torch.as_tensor(
            1.0e-12, device=source.device, dtype=source.dtype
        )
        confidence = torch.sqrt(confidence_product + confidence_epsilon) - torch.sqrt(
            confidence_epsilon
        )
        edge_logit = edge_logit + weights[modality] * torch.where(
            active, confidence * modality_similarity[modality].clamp(0.0, 1.0), 0.0
        )
    hierarchy_value = torch.as_tensor(edges.hierarchy_score, device=source.device, dtype=source.dtype)
    layout_value = torch.as_tensor(edges.layout_score, device=source.device, dtype=source.dtype)
    edge_logit = edge_logit + weights[4] * hierarchy_value + weights[5] * layout_value

    source_count = int(edges.source_count)
    sink = model.unmatched_sink_logit.to(dtype=source.dtype)
    maximum = torch.full((source_count,), -torch.inf, device=source.device, dtype=source.dtype)
    if edge_source.numel():
        maximum.scatter_reduce_(0, edge_source, edge_logit, reduce="amax", include_self=True)
    maximum = torch.maximum(maximum, sink.expand_as(maximum))
    # This maximum is only a numerical log-sum-exp shift. Its derivative
    # cancels analytically. Keeping the scatter/max autograd graph is also
    # unsafe for unused source slots: PyTorch 2.0 may backpropagate NaNs from
    # the ``-inf`` scatter initializer even though the final forward values
    # are finite. Detaching is the standard stable-softmax construction and
    # leaves every forward probability exactly unchanged.
    maximum = maximum.detach()
    edge_exp = torch.exp(edge_logit - maximum[edge_source])
    edge_sum = torch.zeros((source_count,), device=source.device, dtype=source.dtype)
    if edge_source.numel():
        edge_sum.scatter_add_(0, edge_source, edge_exp)
    sink_exp = torch.exp(sink - maximum)
    denominator = sink_exp + edge_sum
    matched_fraction = edge_sum / denominator
    source_flat = source.reshape(-1)
    matched = source_flat * matched_fraction
    unmatched = source_flat - matched
    token_matched = matched.reshape_as(source).sum(dim=1)
    token_score = 2.0 * token_matched - 1.0
    reliability_sum = reliability.sum()
    combined = torch.where(
        reliability_sum > 1.0e-12,
        torch.sum(reliability * token_score) / reliability_sum.clamp_min(1.0e-12),
        torch.full((), -1.0, dtype=source.dtype, device=source.device),
    )
    edge_probability = edge_exp / denominator[edge_source]
    if not torch.allclose(matched + unmatched, source_flat, atol=2e-6, rtol=0.0):
        raise AssertionError("differentiable source transport does not conserve mass")
    return DifferentiablePoseTransportResult(
        combined_score=combined,
        matched_source_probability=matched,
        unmatched_source_probability=unmatched,
        source_probability=source_flat,
        edge_probability=edge_probability,
        edge_source_index=edge_source,
    )


def differentiable_fixed_kernel_capacity_pose_transport(
    model: MinimalPoseTransportReadout,
    query: QueryPoseHeadOutput,
    source_child_probability: torch.Tensor,
    query_reliability: torch.Tensor,
    target_child_weight: torch.Tensor,
    target_canonical_feature: torch.Tensor,
    target_normal_camera: torch.Tensor,
    target_double_sided: torch.Tensor,
    target_relative_depth: torch.Tensor,
    target_boundary: torch.Tensor,
    target_validity: torch.Tensor,
    target_confidence: torch.Tensor,
    edges: FrozenSparseTransportEdges,
    *,
    depth_scale: float = 0.25,
) -> DifferentiablePoseTransportResult:
    """Score a candidate with a bounded local source--target coupling.

    The legacy source-softmax transport normalizes every source over the
    *candidate's* available edges.  Consequently, adding many mediocre edges
    can pull mass away from the sink and a single rendered target can accept
    mass from arbitrarily many query sources.  That is useful as a learned
    attention mechanism, but it is not a conservative physical transport.

    This v2 path instead uses a fixed translation-invariant spatial kernel
    ``1 / (2r+1)^2``.  It is not renormalized at image boundaries, so both its
    row and column sums are at most one.  Query child probabilities and
    rendered child weights are themselves sub-probabilities at every token.
    Therefore the edge allocation

    ``q(source) * m(target) * K(token,target_token) * compatibility``

    simultaneously obeys source and target capacities.  Compatibility is a
    fixed-denominator nonnegative sum of typed evidence.  Removing a target,
    invalidating a modality, or lowering any compatibility term can only
    remove allocation; it cannot improve the reported score.  The unmatched
    sink is the exact residual source mass.
    """

    if str(edges.stage) not in _STAGE:
        raise ValueError("pose transport stage must be coarse, medium, or fine")
    if query.normal_frame != "camera":
        raise ValueError("query normal must be expressed in camera frame")
    if query.depth_semantics != _STAGE[str(edges.stage)]["depth"]:
        raise ValueError("query depth semantics differ from the transport stage")
    scale = float(depth_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth_scale must be positive")

    source = torch.as_tensor(source_child_probability)
    if source.ndim != 2 or not torch.is_floating_point(source):
        raise ValueError("source child probability must be a floating matrix")
    device, dtype = source.device, source.dtype
    reliability = torch.as_tensor(query_reliability, device=device, dtype=dtype).reshape(-1)
    target_mass_matrix = torch.as_tensor(
        target_child_weight, device=device, dtype=dtype
    )
    if target_mass_matrix.ndim != 2 or reliability.shape != (source.shape[0],):
        raise ValueError("fixed-kernel transport mass/layout shapes differ")
    if (
        int(edges.source_count) != int(source.numel())
        or int(edges.target_count) != int(target_mass_matrix.numel())
    ):
        raise ValueError("fixed-kernel transport observations differ from frozen edges")
    if any(not torch.isfinite(value).all() for value in (source, target_mass_matrix, reliability)):
        raise ValueError("fixed-kernel transport masses must be finite")
    tolerance = 2.0e-5
    if (
        torch.any(source < 0.0)
        or torch.any(target_mass_matrix < 0.0)
        or torch.any(reliability < 0.0)
        or torch.any(source.sum(dim=1) > 1.0 + tolerance)
        or torch.any(target_mass_matrix.sum(dim=1) > 1.0 + tolerance)
    ):
        raise ValueError("fixed-kernel source/target masses must be token sub-probabilities")

    target_mass = target_mass_matrix.reshape(-1)
    target_slots = int(target_mass_matrix.shape[1])
    query_code = query.pose_code[0].permute(1, 2, 0).reshape(-1, query.pose_code.shape[1])
    query_normal = query.normal_camera[0].permute(1, 2, 0).reshape(-1, 3)
    query_depth = query.relative_depth[0].reshape(-1)
    query_boundary_value = query.boundary[0].reshape(-1)
    query_conf = query.confidence[0].permute(1, 2, 0).reshape(-1, 4)
    query_valid = torch.stack(
        [
            query.pose_code_valid[0].reshape(-1),
            query.normal_valid[0].reshape(-1),
            query.depth_valid[0].reshape(-1),
            query.boundary_valid[0].reshape(-1),
        ],
        dim=1,
    )
    map_code, map_code_valid = model.project_map_code(
        torch.as_tensor(target_canonical_feature, device=device, dtype=dtype)
    )
    map_code = map_code.reshape(-1, map_code.shape[-1])
    map_normal = torch.as_tensor(
        target_normal_camera, device=device, dtype=dtype
    ).reshape(-1, 3)
    map_depth = torch.as_tensor(
        target_relative_depth, device=device, dtype=dtype
    ).reshape(-1)
    map_boundary = torch.as_tensor(
        target_boundary, device=device, dtype=dtype
    ).reshape(-1)
    map_double = torch.as_tensor(
        target_double_sided, device=device, dtype=torch.bool
    ).reshape(-1)
    map_valid = torch.as_tensor(
        target_validity, device=device, dtype=torch.bool
    ).reshape(-1, 4).clone()
    map_conf = torch.as_tensor(
        target_confidence, device=device, dtype=dtype
    ).reshape(-1, 4)
    map_valid[:, 0] &= map_code_valid.reshape(-1)
    map_valid[:, 1] &= torch.linalg.vector_norm(map_normal, dim=1) >= float(
        model.config.zero_norm_threshold
    )
    finite_inputs = {
        "query_code": query_code,
        "query_normal": query_normal,
        "query_depth": query_depth,
        "query_boundary": query_boundary_value,
        "query_confidence": query_conf,
        "map_code": map_code,
        "map_normal": map_normal,
        "map_depth": map_depth,
        "map_boundary": map_boundary,
        "map_confidence": map_conf,
    }
    nonfinite = [name for name, value in finite_inputs.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(
            "fixed-kernel transport inputs must be finite: " + ",".join(nonfinite)
        )
    if torch.any((query_conf < 0.0) | (query_conf > 1.0)) or torch.any(
        (map_conf < 0.0) | (map_conf > 1.0)
    ):
        raise ValueError("fixed-kernel transport confidence must lie in [0,1]")

    edge_source = torch.as_tensor(edges.source_index, dtype=torch.long, device=device)
    edge_target = torch.as_tensor(edges.target_index, dtype=torch.long, device=device)
    if edge_source.numel() != edge_target.numel():
        raise ValueError("fixed-kernel edge arrays differ")
    source_token = torch.div(edge_source, source.shape[1], rounding_mode="floor")
    target_token = torch.div(edge_target, target_slots, rounding_mode="floor")
    source_flat = source.reshape(-1)

    feature_cos = torch.sum(query_code[source_token] * map_code[edge_target], dim=1)
    feature_similarity = 0.5 * (1.0 + feature_cos.clamp(-1.0, 1.0))
    normal_cos = torch.sum(
        query_normal[source_token] * map_normal[edge_target], dim=1
    ).clamp(-1.0, 1.0)
    normal_similarity = torch.where(
        map_double[edge_target], torch.abs(normal_cos), 0.5 * (1.0 + normal_cos)
    )
    depth_similarity = 1.0 - torch.clamp(
        torch.abs(query_depth[source_token] - map_depth[edge_target]) / scale,
        0.0,
        1.0,
    )
    boundary_similarity = 1.0 - torch.abs(
        query_boundary_value[source_token] - map_boundary[edge_target]
    ).clamp(0.0, 1.0)
    similarities = (
        feature_similarity,
        normal_similarity,
        depth_similarity,
        boundary_similarity,
    )
    components: list[torch.Tensor] = []
    confidence_epsilon = torch.as_tensor(1.0e-12, device=device, dtype=dtype)
    for modality, similarity in enumerate(similarities):
        active = query_valid[source_token, modality] & map_valid[edge_target, modality]
        confidence_product = (
            query_conf[source_token, modality] * map_conf[edge_target, modality]
        )
        confidence = torch.sqrt(confidence_product + confidence_epsilon) - torch.sqrt(
            confidence_epsilon
        )
        components.append(
            torch.where(active, confidence * similarity.clamp(0.0, 1.0), 0.0)
        )
    components.append(
        torch.as_tensor(edges.hierarchy_score, device=device, dtype=dtype).clamp(0.0, 1.0)
    )
    components.append(
        torch.as_tensor(edges.layout_score, device=device, dtype=dtype).clamp(0.0, 1.0)
    )
    weights = model.edge_weights().to(device=device, dtype=dtype)
    compatibility = sum(
        weights[index] * component for index, component in enumerate(components)
    ) / weights.sum().clamp_min(torch.finfo(dtype).tiny)

    radius = int(_STAGE[str(edges.stage)]["radius"])
    spatial_kernel = 1.0 / float((2 * radius + 1) ** 2)
    edge_allocation = (
        source_flat[edge_source]
        * target_mass[edge_target]
        * float(spatial_kernel)
        * compatibility
    )
    matched_source = torch.zeros_like(source_flat)
    matched_target = torch.zeros_like(target_mass)
    if edge_source.numel():
        matched_source.scatter_add_(0, edge_source, edge_allocation)
        matched_target.scatter_add_(0, edge_target, edge_allocation)
    source_slack = source_flat - matched_source
    source_capacity_tolerance = 3.0e-6
    if torch.any(source_slack < -source_capacity_tolerance):
        raise AssertionError("fixed-kernel transport exceeds a source capacity")
    if torch.any(matched_target > target_mass + source_capacity_tolerance):
        raise AssertionError("fixed-kernel transport exceeds a target capacity")
    unmatched = torch.clamp_min(source_slack, 0.0)
    token_matched = matched_source.reshape_as(source).sum(dim=1)
    token_score = 2.0 * token_matched - 1.0
    reliability_sum = reliability.sum()
    combined = torch.where(
        reliability_sum > 1.0e-12,
        torch.sum(reliability * token_score) / reliability_sum.clamp_min(1.0e-12),
        torch.full((), -1.0, dtype=dtype, device=device),
    )
    edge_fraction = edge_allocation / source_flat[edge_source].clamp_min(1.0e-12)
    return DifferentiablePoseTransportResult(
        combined_score=combined,
        matched_source_probability=matched_source,
        unmatched_source_probability=unmatched,
        source_probability=source_flat,
        edge_probability=edge_fraction,
        edge_source_index=edge_source,
        matched_target_probability=matched_target,
        target_probability=target_mass,
        transport_semantics=FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
    )
