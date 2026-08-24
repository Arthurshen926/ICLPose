"""Batched dense replay of the fixed-kernel identity pose transport.

The scientific authority for pose transport remains
``differentiable_fixed_kernel_capacity_pose_transport``.  That implementation
first materializes a CPU sparse edge list and is useful for training arbitrary
typed readouts.  The currently frozen identity readout, however, observes only
the shared RADIO feature plus hierarchy and layout.  Its complete edge graph
is a small, regular token neighbourhood and can be evaluated directly on the
GPU without changing the objective.

This module deliberately supports only that narrow, auditable case.  It does
not estimate a pose, add correspondences, or consume pose-error labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .candidate_conditioned_pose_attribution import PoseTransportHierarchy, _STAGE
from .trainable_pose_transport import MinimalPoseTransportReadout, QueryPoseHeadOutput


DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS = (
    "batched_dense_exact_fixed_kernel_identity_feature_hierarchy_layout_v1"
)


@dataclass(frozen=True)
class DenseFixedIdentityTransportResult:
    """Scores and exact linear sufficient statistics for one candidate batch."""

    scores: torch.Tensor
    component_statistics: torch.Tensor
    maximum_source_capacity_excess: float
    maximum_target_capacity_excess: float
    stage: str
    transport_semantics: str = DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS


class DensePoseTransportHierarchyGPU:
    """Immutable hierarchy tensors shared by all streamed candidate batches."""

    def __init__(
        self,
        hierarchy: PoseTransportHierarchy,
        *,
        device: torch.device | str,
    ) -> None:
        target_device = torch.device(device)
        parent = np.asarray(hierarchy.child_parent_ids, dtype=np.int64).reshape(-1)
        support = np.asarray(hierarchy.child_support_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(hierarchy.adjacency_offsets, dtype=np.int64).reshape(-1)
        adjacency = np.asarray(hierarchy.adjacency_child_rows, dtype=np.int64).reshape(-1)
        if (
            parent.size == 0
            or support.shape != parent.shape
            or offsets.shape != (parent.size + 1,)
            or offsets[0] != 0
            or offsets[-1] != adjacency.size
            or np.any(offsets[1:] < offsets[:-1])
            or np.any(parent < 0)
            or np.any((support < -1) | (support >= parent.size))
            or np.any((adjacency < 0) | (adjacency >= parent.size))
        ):
            raise ValueError("invalid dense pose-transport hierarchy")
        source = np.repeat(np.arange(parent.size, dtype=np.int64), np.diff(offsets))
        adjacency_keys = source * int(parent.size) + adjacency
        # The authority constructs a sorted set.  Do the same here rather than
        # depending on CSR neighbour ordering or uniqueness.
        adjacency_keys = np.unique(adjacency_keys)
        self.parent = torch.as_tensor(parent, device=target_device, dtype=torch.long)
        self.support = torch.as_tensor(support, device=target_device, dtype=torch.long)
        self.adjacency_keys = torch.as_tensor(
            adjacency_keys, device=target_device, dtype=torch.long
        )
        self.child_count = int(parent.size)
        self.content_sha256 = str(hierarchy.content_sha256)
        self.device = target_device

    def adjacent(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return directed CSR adjacency for broadcast-compatible child rows."""

        if source.device != self.device or target.device != self.device:
            raise ValueError("dense hierarchy and child rows must share a device")
        if self.adjacency_keys.numel() == 0:
            return torch.zeros(torch.broadcast_shapes(source.shape, target.shape), device=self.device, dtype=torch.bool)
        key = source * int(self.child_count) + target
        position = torch.searchsorted(self.adjacency_keys, key)
        safe = position.clamp_max(int(self.adjacency_keys.numel()) - 1)
        return (position < int(self.adjacency_keys.numel())) & (
            self.adjacency_keys[safe] == key
        )


def _as_tensor(
    value: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def dense_fixed_identity_transport(
    model: MinimalPoseTransportReadout,
    query: QueryPoseHeadOutput,
    source_child_rows: np.ndarray | torch.Tensor,
    source_child_probability: np.ndarray | torch.Tensor,
    query_reliability: np.ndarray | torch.Tensor,
    token_xy: np.ndarray | torch.Tensor,
    target_child_rows: np.ndarray | torch.Tensor,
    target_child_weight: np.ndarray | torch.Tensor,
    target_canonical_feature: np.ndarray | torch.Tensor,
    target_feature_confidence: np.ndarray | torch.Tensor,
    target_feature_valid: np.ndarray | torch.Tensor,
    hierarchy: DensePoseTransportHierarchyGPU,
    *,
    stage: str,
    height: int = 36,
    width: int = 64,
    minimum_source_probability: float = 1.0e-6,
    minimum_target_weight: float = 1.0e-6,
) -> DenseFixedIdentityTransportResult:
    """Evaluate the exact feature-only fixed-kernel objective for a batch.

    ``target_*`` arrays have shape ``[batch, token, slot, ...]``.  No pose,
    candidate rank, or error label is accepted by this API, which makes it
    suitable for the score-before-label half of a natural-candidate audit.
    """

    stage_value = str(stage)
    if stage_value not in _STAGE:
        raise ValueError("dense identity transport stage must be coarse, medium, or fine")
    if query.normal_frame != "camera" or query.depth_semantics != _STAGE[stage_value]["depth"]:
        raise ValueError("query observation semantics differ from the transport stage")
    device = hierarchy.device
    dtype = next(model.parameters()).dtype
    if next(model.parameters()).device != device:
        raise ValueError("dense hierarchy and pose readout must share a device")
    h, w = int(height), int(width)
    if h <= 0 or w <= 0:
        raise ValueError("dense token grid dimensions must be positive")
    token_count = h * w

    source_rows = _as_tensor(source_child_rows, device=device, dtype=torch.long)
    source = _as_tensor(source_child_probability, device=device, dtype=dtype)
    reliability = _as_tensor(query_reliability, device=device, dtype=dtype).reshape(-1)
    xy = _as_tensor(token_xy, device=device, dtype=torch.long)
    target_rows = _as_tensor(target_child_rows, device=device, dtype=torch.long)
    target_mass = _as_tensor(target_child_weight, device=device, dtype=dtype)
    target_feature = _as_tensor(target_canonical_feature, device=device, dtype=dtype)
    target_confidence = _as_tensor(
        target_feature_confidence, device=device, dtype=dtype
    )
    feature_valid = _as_tensor(target_feature_valid, device=device, dtype=torch.bool)
    if source_rows.ndim != 2 or source.shape != source_rows.shape:
        raise ValueError("dense source child rows and probabilities differ")
    if source_rows.shape[0] != token_count or reliability.shape != (token_count,):
        raise ValueError("dense source observations differ from the token grid")
    batch_size = int(target_rows.shape[0]) if target_rows.ndim == 3 else 0
    if (
        batch_size <= 0
        or target_mass.shape != target_rows.shape
        or target_rows.shape[1] != token_count
        or target_feature.shape[:-1] != target_rows.shape
        or int(target_feature.shape[-1]) != int(model.config.map_feature_dim)
        or target_confidence.shape != target_rows.shape
        or feature_valid.shape != target_rows.shape
    ):
        raise ValueError("dense target candidate observations differ")
    expected_xy = torch.stack(
        [
            torch.arange(w, device=device, dtype=torch.long).repeat(h),
            torch.arange(h, device=device, dtype=torch.long).repeat_interleave(w),
        ],
        dim=1,
    )
    if xy.shape != expected_xy.shape or not torch.equal(xy, expected_xy):
        raise ValueError("dense transport requires the canonical row-major token grid")
    finite = (source, reliability, target_mass, target_feature, target_confidence)
    if any(not torch.isfinite(value).all() for value in finite):
        raise ValueError("dense fixed-kernel inputs must be finite")
    tolerance = 2.0e-5
    if (
        torch.any(source < 0.0)
        or torch.any(reliability < 0.0)
        or torch.any(target_mass < 0.0)
        or torch.any((target_confidence < 0.0) | (target_confidence > 1.0))
        or torch.any(source.sum(dim=1) > 1.0 + tolerance)
        or torch.any(target_mass.sum(dim=2) > 1.0 + tolerance)
    ):
        raise ValueError("dense fixed-kernel masses must be token sub-probabilities")
    child_count = int(hierarchy.child_count)
    if torch.any(source_rows >= child_count) or torch.any(target_rows >= child_count):
        raise ValueError("dense child row exceeds the transport hierarchy")
    for rows, name in ((source_rows, "source"), (target_rows, "target")):
        sorted_rows = torch.sort(rows, dim=-1).values
        duplicate = (sorted_rows[..., 1:] == sorted_rows[..., :-1]) & (
            sorted_rows[..., 1:] >= 0
        )
        if torch.any(duplicate):
            raise ValueError(f"dense {name} contains a duplicate child at one token")

    query_code = query.pose_code[0].permute(1, 2, 0).reshape(h, w, -1)
    query_valid = query.pose_code_valid[0].reshape(h, w)
    query_feature_confidence = query.confidence[0, 0].reshape(h, w)
    if (
        query.pose_code.shape[0] != 1
        or query_code.shape[-1] != int(model.config.pose_code_dim)
        or query_valid.shape != (h, w)
        or query_code.device != device
        or query_code.dtype != dtype
        or not torch.isfinite(query_code).all()
        or not torch.isfinite(query_feature_confidence).all()
        or torch.any(
            (query_feature_confidence < 0.0) | (query_feature_confidence > 1.0)
        )
    ):
        raise ValueError("dense query feature differs from the token grid/readout")
    map_code, map_code_valid = model.project_map_code(target_feature)
    source_rows = source_rows.reshape(h, w, -1)
    source = source.reshape(h, w, -1)
    target_rows = target_rows.reshape(batch_size, h, w, -1)
    target_mass = target_mass.reshape(batch_size, h, w, -1)
    target_confidence = target_confidence.reshape(batch_size, h, w, -1)
    feature_valid = feature_valid.reshape_as(target_rows) & map_code_valid.reshape_as(target_rows)
    map_code = map_code.reshape(batch_size, h, w, target_rows.shape[-1], -1)
    source_slots = int(source.shape[-1])
    target_slots = int(target_mass.shape[-1])
    matched_source = torch.zeros(
        (batch_size, h, w, source_slots), device=device, dtype=dtype
    )
    matched_target = torch.zeros_like(target_mass)
    # Only [feature, hierarchy, layout] are observed in the identity readout.
    component_token_mass = torch.zeros(
        (batch_size, h, w, 3), device=device, dtype=dtype
    )
    weights = model.edge_weights().to(device=device, dtype=dtype)
    weight_denominator = weights.sum().clamp_min(torch.finfo(dtype).tiny)
    radius = int(_STAGE[stage_value]["radius"])
    relations = set(_STAGE[stage_value]["relations"])
    kernel = 1.0 / float((2 * radius + 1) ** 2)
    confidence_epsilon = torch.as_tensor(1.0e-12, device=device, dtype=dtype)

    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            sy0, sy1 = max(0, -shift_y), min(h, h - shift_y)
            sx0, sx1 = max(0, -shift_x), min(w, w - shift_x)
            ty0, ty1 = sy0 + shift_y, sy1 + shift_y
            tx0, tx1 = sx0 + shift_x, sx1 + shift_x
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            source_id = source_rows[sy0:sy1, sx0:sx1][None, ..., :, None]
            target_id = target_rows[:, ty0:ty1, tx0:tx1][..., None, :]
            source_mass = source[sy0:sy1, sx0:sx1][None, ..., :, None]
            target_mass_local = target_mass[:, ty0:ty1, tx0:tx1][..., None, :]
            valid = (
                (source_id >= 0)
                & (target_id >= 0)
                & (source_mass >= float(minimum_source_probability))
                & (target_mass_local >= float(minimum_target_weight))
            )
            safe_source = source_id.clamp_min(0)
            safe_target = target_id.clamp_min(0)
            hierarchy_score = torch.zeros(
                torch.broadcast_shapes(source_id.shape, target_id.shape),
                device=device,
                dtype=dtype,
            )
            exact = valid & (source_id == target_id)
            if "exact" in relations:
                hierarchy_score = torch.where(exact, 1.0, hierarchy_score)
            if "support" in relations:
                source_support = hierarchy.support[safe_source]
                target_support = hierarchy.support[safe_target]
                match = (
                    valid
                    & ~exact
                    & (source_support >= 0)
                    & (source_support == target_support)
                )
                hierarchy_score = torch.where(match, 0.7, hierarchy_score)
            if "adjacent" in relations:
                match = valid & (hierarchy_score == 0.0) & hierarchy.adjacent(
                    safe_source, safe_target
                )
                hierarchy_score = torch.where(match, 0.45, hierarchy_score)
            if "parent" in relations:
                source_parent = hierarchy.parent[safe_source]
                target_parent = hierarchy.parent[safe_target]
                match = (
                    valid
                    & (hierarchy_score == 0.0)
                    & (source_parent == target_parent)
                )
                hierarchy_score = torch.where(match, 0.25, hierarchy_score)
            keep = valid & (hierarchy_score > 0.0)

            query_local = query_code[sy0:sy1, sx0:sx1]
            map_local = map_code[:, ty0:ty1, tx0:tx1]
            feature_cosine = torch.einsum("hwc,bhwtc->bhwt", query_local, map_local)
            feature_similarity = 0.5 * (1.0 + feature_cosine.clamp(-1.0, 1.0))
            local_feature_valid = (
                query_valid[sy0:sy1, sx0:sx1][None, ..., None]
                & feature_valid[:, ty0:ty1, tx0:tx1]
            )
            confidence_product = (
                query_feature_confidence[sy0:sy1, sx0:sx1][None, ..., None]
                * target_confidence[:, ty0:ty1, tx0:tx1]
            )
            feature_confidence = torch.sqrt(
                confidence_product + confidence_epsilon
            ) - torch.sqrt(confidence_epsilon)
            feature_component = torch.where(
                local_feature_valid,
                feature_confidence * feature_similarity,
                0.0,
            )[..., None, :]
            layout_component = 1.0 - float(max(abs(shift_x), abs(shift_y))) / float(
                radius + 1
            )
            compatibility = (
                weights[0] * feature_component
                + weights[4] * hierarchy_score
                + weights[5] * float(layout_component)
            ) / weight_denominator
            base = source_mass * target_mass_local * float(kernel) * keep
            allocation = base * compatibility
            matched_source[:, sy0:sy1, sx0:sx1] += allocation.sum(dim=-1)
            matched_target[:, ty0:ty1, tx0:tx1] += allocation.sum(dim=-2)
            component_token_mass[:, sy0:sy1, sx0:sx1, 0] += (
                base * feature_component
            ).sum(dim=(-2, -1))
            component_token_mass[:, sy0:sy1, sx0:sx1, 1] += (
                base * hierarchy_score
            ).sum(dim=(-2, -1))
            component_token_mass[:, sy0:sy1, sx0:sx1, 2] += (
                base * float(layout_component)
            ).sum(dim=(-2, -1))

    source_excess = matched_source - source[None]
    target_excess = matched_target - target_mass
    maximum_source_excess = float(torch.clamp_min(source_excess.max(), 0.0).detach().cpu())
    maximum_target_excess = float(torch.clamp_min(target_excess.max(), 0.0).detach().cpu())
    capacity_tolerance = 3.0e-6
    if maximum_source_excess > capacity_tolerance:
        raise AssertionError("dense fixed-kernel transport exceeds a source capacity")
    if maximum_target_excess > capacity_tolerance:
        raise AssertionError("dense fixed-kernel transport exceeds a target capacity")
    reliability_sum = reliability.sum().clamp_min(1.0e-12)
    reduced = 2.0 * torch.sum(
        reliability[None, :, None] * component_token_mass.reshape(batch_size, token_count, 3),
        dim=1,
    ) / reliability_sum
    statistics = torch.zeros((batch_size, 6), device=device, dtype=dtype)
    statistics[:, 0] = reduced[:, 0]
    statistics[:, 4] = reduced[:, 1]
    statistics[:, 5] = reduced[:, 2]
    scores = -1.0 + torch.sum(statistics * weights[None], dim=1) / weight_denominator
    direct_scores = -1.0 + 2.0 * torch.sum(
        reliability[None] * matched_source.sum(dim=-1).reshape(batch_size, token_count),
        dim=1,
    ) / reliability_sum
    if not torch.allclose(scores, direct_scores, atol=2.0e-6, rtol=2.0e-6):
        raise AssertionError("dense sufficient statistics and capacity score differ")
    return DenseFixedIdentityTransportResult(
        scores=scores,
        component_statistics=statistics,
        maximum_source_capacity_excess=maximum_source_excess,
        maximum_target_capacity_excess=maximum_target_excess,
        stage=stage_value,
    )
