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
    production_eligible: bool = False


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
        active = target_rows[token][target_mass[token] >= float(minimum_target_weight)]
        active = active[active >= 0]
        if np.unique(active).size != active.size:
            raise ValueError("target contains duplicate child at one token")
    for token in range(source_rows.shape[0]):
        active = source_rows[token][source_mass[token] >= float(minimum_source_probability)]
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
        torch.ones_like(query.relative_depth, dtype=torch.bool).reshape(-1),
        torch.ones_like(query.boundary, dtype=torch.bool).reshape(-1),
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
