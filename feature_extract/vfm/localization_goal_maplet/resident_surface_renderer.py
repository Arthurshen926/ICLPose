"""Resident-GPU batched full-geometry renderer for soft surface pose energy.

This is the first production-candidate batching seam.  Geometry and canonical
payload are materialized once; camera projection and tile intersection are
batched.  The deterministic token/child reducer is intentionally shared with
the scalar authority until a GPU reducer passes the same equivalence gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np

from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    _composite_sorted_packed_hits,
    _intrinsic_matrix,
    _surface_element_quaternions_and_scales,
)

from .canonical_field import CanonicalSurfaceField
from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap
from .surface_renderer import (
    RenderedSoftChildMixture,
    dominant_child_owner,
    _primitive_parent_memberships,
    _raw_to_ideal_token_warp,
    _reduce_soft_child_token_hits,
)
from .resident_exact_reducer import reduce_soft_child_token_hits_torch
from .view_conditioned_field import (
    ViewConditionedPrimitiveField,
    _camera_focal_pixels,
    condition_canonical_codes_for_pose,
)


def _composite_packed_hits_torch(
    global_pixel,
    depth,
    primitive_row,
    alpha,
):
    """GPU/CPU Torch equivalent of the deterministic NumPy compositor.

    The lexicographic order is exactly ``(pixel, depth, primitive_row)`` and
    the 1e-4 early-stop hit remains inclusive.  Float64 prefix arithmetic is
    intentional: this is the first production-candidate GPU seam and keeps
    compositing error below the CPU authority tolerance while avoiding D2H of
    every raw raster hit.
    """

    import torch

    values = (global_pixel, depth, primitive_row, alpha)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("packed compositor inputs must be Torch tensors")
    pixel, depth_value, row, opacity = values
    if any(value.ndim != 1 for value in values) or len({int(value.numel()) for value in values}) != 1:
        raise ValueError("packed compositor arrays must be aligned one-dimensional tensors")
    if pixel.dtype != torch.int64 or row.dtype != torch.int64:
        raise ValueError("packed compositor identity arrays must be int64")
    if depth_value.dtype != torch.float32 or opacity.dtype != torch.float32:
        raise ValueError("packed compositor numeric arrays must be float32")
    if len({value.device for value in values}) != 1:
        raise ValueError("packed compositor arrays must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("packed compositor arrays must be contiguous")
    if not torch.isfinite(depth_value).all() or not torch.isfinite(opacity).all():
        raise ValueError("packed compositor numeric arrays must be finite")
    if torch.any(pixel < 0) or torch.any(row < 0):
        raise ValueError("packed compositor identities must be nonnegative")
    if pixel.numel() == 0:
        return pixel.clone(), row.clone(), opacity.clone()

    # Stable least-significant to most-significant sorts reproduce np.lexsort.
    order = torch.argsort(row, stable=True)
    order = order[torch.argsort(depth_value[order], stable=True)]
    order = order[torch.argsort(pixel[order], stable=True)]
    pixel = pixel[order]
    row = row[order]
    opacity = torch.clamp(opacity[order].to(torch.float64), 0.0, 0.999)
    starts = torch.empty_like(pixel, dtype=torch.bool)
    starts[0] = True
    starts[1:] = pixel[1:] != pixel[:-1]
    start_indices = torch.nonzero(starts, as_tuple=False).reshape(-1)
    counts = torch.diff(torch.cat([
        start_indices,
        torch.as_tensor([pixel.numel()], dtype=torch.int64, device=pixel.device),
    ]))
    log_survival = torch.log1p(-opacity)
    prefix = torch.cumsum(log_survival, dim=0)
    group_base = torch.cat([
        torch.zeros((1,), dtype=torch.float64, device=pixel.device),
        prefix[start_indices[1:] - 1],
    ])
    base = torch.repeat_interleave(group_base, counts)
    log_transmittance = prefix - log_survival - base
    transmittance = torch.exp(torch.clamp(log_transmittance, min=-745.0, max=0.0))
    weight = transmittance * opacity
    transmittance_after = torch.exp(torch.clamp(prefix - base, min=-745.0, max=0.0))
    group = torch.cumsum(starts.to(torch.int64), dim=0) - 1
    index = torch.arange(pixel.numel(), dtype=torch.int64, device=pixel.device)
    stop = transmittance_after <= 1.0e-4
    first_stop = torch.full(
        (start_indices.numel(),), pixel.numel(), dtype=torch.int64, device=pixel.device
    )
    stop_index = index[stop]
    stop_group = group[stop]
    if stop_index.numel():
        first = torch.empty_like(stop_group, dtype=torch.bool)
        first[0] = True
        first[1:] = stop_group[1:] != stop_group[:-1]
        first_stop[stop_group[first]] = stop_index[first]
    keep = (index <= first_stop[group]) & (weight > 1.0e-12)
    return pixel[keep], row[keep], weight[keep].to(torch.float32)


def _reduce_direct_canonical_token_hits_torch(
    token_ids,
    primitive_rows,
    contribution,
    field_row_by_primitive,
    canonical_codes,
    *,
    token_count: int,
    minimum_feature_alpha: float = 1.0e-4,
    accumulation_chunk_rows: int = 65536,
    override_field_rows=None,
    override_codes=None,
):
    """Reduce all canonical primitive payload directly to token features.

    Occlusion has already been resolved by the full-scene compositor.  This
    reducer intentionally does not construct child or parent Top-L identity;
    it is the minimal sufficient target representation for full-token
    candidate-conditioned appearance/phase scoring.
    """

    import torch

    values = (token_ids, primitive_rows, contribution, field_row_by_primitive, canonical_codes)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("direct canonical reducer inputs must be Torch tensors")
    token, primitive, weight, field_row, codes = values
    if (override_field_rows is None) != (override_codes is None):
        raise ValueError("direct canonical overrides must be supplied together")
    override_row = override_value = None
    if override_field_rows is not None:
        override_row = override_field_rows
        override_value = override_codes
        if not isinstance(override_row, torch.Tensor) or not isinstance(
            override_value, torch.Tensor
        ):
            raise TypeError("direct canonical overrides must be Torch tensors")
        if (
            override_row.ndim != 1 or override_row.dtype != torch.int64
            or override_value.shape != (override_row.numel(), codes.shape[1])
            or override_value.dtype != torch.float32
            or override_row.device != token.device or override_value.device != token.device
            or not override_row.is_contiguous() or not override_value.is_contiguous()
            or (override_row.numel() and (
                torch.any(override_row < 0) or torch.any(override_row >= codes.shape[0])
                or torch.any(override_row[1:] <= override_row[:-1])
            ))
            or not torch.isfinite(override_value).all()
        ):
            raise ValueError("direct canonical sparse overrides differ")
    if token.ndim != 1 or primitive.ndim != 1 or weight.ndim != 1:
        raise ValueError("direct canonical hit arrays must be one-dimensional")
    if token.shape != primitive.shape or token.shape != weight.shape:
        raise ValueError("direct canonical hit arrays differ")
    if token.dtype != torch.int64 or primitive.dtype != torch.int64 or field_row.dtype != torch.int64:
        raise ValueError("direct canonical identity arrays must be int64")
    if weight.dtype != torch.float32 or codes.dtype != torch.float32:
        raise ValueError("direct canonical numeric arrays must be float32")
    if field_row.ndim != 1 or codes.ndim != 2:
        raise ValueError("direct canonical field arrays differ")
    if len({value.device for value in values}) != 1:
        raise ValueError("direct canonical reducer arrays must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("direct canonical reducer arrays must be contiguous")
    count = int(token_count)
    if count <= 0 or int(accumulation_chunk_rows) <= 0:
        raise ValueError("direct canonical reducer dimensions must be positive")
    if primitive.numel() and (
        torch.any(token < 0)
        or torch.any(token >= count)
        or torch.any(primitive < 0)
        or torch.any(primitive >= field_row.numel())
    ):
        raise ValueError("direct canonical hit identity is out of bounds")
    if not torch.isfinite(weight).all() or not torch.isfinite(codes).all() or torch.any(weight < 0.0):
        raise ValueError("direct canonical evidence must be finite and nonnegative")
    feature_dim = int(codes.shape[1])
    mass = torch.zeros((count,), dtype=torch.float32, device=token.device)
    feature_sum = torch.zeros((count, feature_dim), dtype=torch.float32, device=token.device)
    if token.numel():
        rows = field_row[primitive]
        keep = (rows >= 0) & (rows < codes.shape[0]) & (weight > 0.0)
        selected = torch.nonzero(keep, as_tuple=False).reshape(-1)
        for begin in range(0, int(selected.numel()), int(accumulation_chunk_rows)):
            chosen = selected[begin:begin + int(accumulation_chunk_rows)]
            chosen_token = token[chosen]
            chosen_weight = weight[chosen]
            chosen_row = rows[chosen]
            chosen_code = codes[chosen_row]
            if override_row is not None and override_row.numel():
                location = torch.searchsorted(override_row, chosen_row)
                bounded = location.clamp_max(override_row.numel() - 1)
                replaced = override_row[bounded] == chosen_row
                chosen_code = torch.where(
                    replaced[:, None], override_value[bounded], chosen_code,
                )
            mass.index_add_(0, chosen_token, chosen_weight)
            feature_sum.index_add_(
                0, chosen_token, chosen_code * chosen_weight[:, None]
            )
    feature = feature_sum / mass[:, None].clamp_min(1.0e-8)
    norm = torch.linalg.vector_norm(feature, dim=1)
    valid = (mass >= float(minimum_feature_alpha)) & (norm >= 1.0e-8)
    feature = torch.where(
        valid[:, None], feature / norm[:, None].clamp_min(1.0e-8),
        torch.zeros_like(feature),
    )
    return feature, mass, valid


def _condition_canonical_codes_for_pose_torch(
    canonical_codes,
    field_indices,
    field_centers_world,
    field_tangent1_world,
    field_tangent2_world,
    field_normals_world,
    field_radius,
    residual_basis,
    coefficients,
    observation_count,
    mean_local_direction,
    direction_concentration,
    minimum_direction_cosine,
    mean_log_projected_scale,
    minimum_log_projected_scale,
    maximum_log_projected_scale,
    pose_w2c,
    *,
    focal_pixels: float,
    minimum_views: int,
    direction_cosine_margin: float,
    log_scale_margin: float,
):
    """Torch-native sparse evaluation of the frozen low-rank view field.

    All geometry and field statistics are aligned to canonical-field rows.
    This avoids copying tens of thousands of hit rows to NumPy for every pose.
    """

    import torch

    values = (
        canonical_codes, field_indices, field_centers_world,
        field_tangent1_world, field_tangent2_world, field_normals_world,
        field_radius, residual_basis, coefficients, observation_count,
        mean_local_direction, direction_concentration,
        minimum_direction_cosine, mean_log_projected_scale,
        minimum_log_projected_scale, maximum_log_projected_scale, pose_w2c,
    )
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("view-conditioned Torch inputs must be tensors")
    index = field_indices
    count, feature_dim = canonical_codes.shape
    rank = int(residual_basis.shape[0])
    if (
        canonical_codes.ndim != 2 or canonical_codes.dtype != torch.float32
        or index.ndim != 1 or index.dtype != torch.int64
        or field_centers_world.shape != (count, 3)
        or field_tangent1_world.shape != (count, 3)
        or field_tangent2_world.shape != (count, 3)
        or field_normals_world.shape != (count, 3)
        or field_radius.shape != (count,)
        or residual_basis.shape != (rank, feature_dim)
        or coefficients.shape != (count, 5, rank)
        or observation_count.shape != (count,)
        or mean_local_direction.shape != (count, 3)
        or direction_concentration.shape != (count,)
        or minimum_direction_cosine.shape != (count,)
        or mean_log_projected_scale.shape != (count,)
        or minimum_log_projected_scale.shape != (count,)
        or maximum_log_projected_scale.shape != (count,)
        or pose_w2c.shape != (4, 4)
    ):
        raise ValueError("view-conditioned Torch arrays differ")
    device = canonical_codes.device
    if any(value.device != device for value in values):
        raise ValueError("view-conditioned Torch arrays must share one device")
    if index.numel() and (torch.any(index < 0) or torch.any(index >= count)):
        raise ValueError("view-conditioned field index is out of bounds")
    if not np.isfinite(float(focal_pixels)) or float(focal_pixels) <= 0.0:
        raise ValueError("view-conditioned focal must be finite and positive")

    center = field_centers_world[index]
    rotation = pose_w2c[:3, :3]
    translation = pose_w2c[:3, 3]
    camera_center = -(rotation.transpose(0, 1) @ translation)
    view_world = camera_center[None] - center
    view_world = view_world / torch.linalg.vector_norm(
        view_world, dim=1, keepdim=True,
    ).clamp_min(1.0e-8)
    local_direction = torch.stack((
        torch.sum(view_world * field_tangent1_world[index], dim=1),
        torch.sum(view_world * field_tangent2_world[index], dim=1),
        torch.sum(view_world * field_normals_world[index], dim=1),
    ), dim=1)
    local_direction = local_direction / torch.linalg.vector_norm(
        local_direction, dim=1, keepdim=True,
    ).clamp_min(1.0e-8)
    mean_direction = mean_local_direction[index]
    direction_cosine = torch.sum(local_direction * mean_direction, dim=1)
    camera_xyz = center @ rotation.transpose(0, 1) + translation[None]
    log_scale = torch.log(torch.clamp(
        float(focal_pixels) * field_radius[index]
        / camera_xyz[:, 2].clamp_min(1.0e-4),
        min=1.0e-6,
    ))
    active = (
        (observation_count[index] >= int(minimum_views))
        & (direction_cosine >= minimum_direction_cosine[index] - float(direction_cosine_margin))
        & (log_scale >= minimum_log_projected_scale[index] - float(log_scale_margin))
        & (log_scale <= maximum_log_projected_scale[index] + float(log_scale_margin))
    )
    predictor = torch.cat((
        torch.ones((index.numel(), 1), dtype=torch.float32, device=device),
        local_direction
        - mean_direction * direction_concentration[index, None],
        (log_scale - mean_log_projected_scale[index])[:, None],
    ), dim=1)
    latent = torch.einsum(
        "np,npr->nr", predictor, coefficients[index].to(torch.float32),
    )
    residual = latent @ residual_basis
    code = canonical_codes[index] + residual * active[:, None]
    code = code / torch.linalg.vector_norm(code, dim=1, keepdim=True).clamp_min(1.0e-8)
    return code.contiguous(), active


def _reduce_direct_typed_geometry_token_hits_torch(
    token_ids,
    primitive_rows,
    contribution,
    field_row_by_primitive,
    primitive_centers_world,
    primitive_normals_world,
    poses_w2c,
    *,
    token_count_per_pose: int,
):
    """Reduce sign-invariant normal axes and relative log depth per token.

    Geometry uses exactly the same feature-valid, occlusion-resolved hit set as
    the direct canonical reducer.  The six normal second moments are invariant
    to the arbitrary sign of a two-sided 2DGS normal.
    """

    import torch

    values = (
        token_ids, primitive_rows, contribution, field_row_by_primitive,
        primitive_centers_world, primitive_normals_world, poses_w2c,
    )
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("direct typed reducer inputs must be Torch tensors")
    token, primitive, weight, field_row, centers, normals, poses = values
    if (
        token.ndim != 1 or primitive.ndim != 1 or weight.ndim != 1
        or token.shape != primitive.shape or token.shape != weight.shape
        or field_row.ndim != 1 or centers.ndim != 2 or centers.shape[1:] != (3,)
        or normals.shape != centers.shape or poses.ndim != 3 or poses.shape[1:] != (4, 4)
    ):
        raise ValueError("direct typed reducer arrays differ")
    if token.dtype != torch.int64 or primitive.dtype != torch.int64 or field_row.dtype != torch.int64:
        raise ValueError("direct typed identity arrays must be int64")
    if any(value.dtype != torch.float32 for value in (weight, centers, normals, poses)):
        raise ValueError("direct typed numeric arrays must be float32")
    if len({value.device for value in values}) != 1 or any(not value.is_contiguous() for value in values):
        raise ValueError("direct typed arrays must share one device and be contiguous")
    tokens_per_pose = int(token_count_per_pose)
    total_tokens = int(poses.shape[0]) * tokens_per_pose
    if tokens_per_pose <= 0 or centers.shape[0] != field_row.shape[0]:
        raise ValueError("direct typed reducer dimensions are invalid")
    if any(not torch.isfinite(value).all() for value in (weight, centers, normals, poses)):
        raise ValueError("direct typed reducer inputs must be finite")
    if primitive.numel() and (
        torch.any(token < 0) or torch.any(token >= total_tokens)
        or torch.any(primitive < 0) or torch.any(primitive >= centers.shape[0])
        or torch.any(weight < 0.0)
    ):
        raise ValueError("direct typed hit identity or mass is invalid")
    expected_last_row = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=poses.device,
    )
    if not torch.all(poses[:, 3] == expected_last_row):
        raise ValueError("direct typed poses must be homogeneous w2c matrices")

    mass = torch.zeros((total_tokens,), dtype=torch.float32, device=token.device)
    axis_sum = torch.zeros((total_tokens, 6), dtype=torch.float32, device=token.device)
    log_depth_sum = torch.zeros_like(mass)
    log_depth_square_sum = torch.zeros_like(mass)
    if token.numel():
        rows = field_row[primitive]
        batch = torch.div(token, tokens_per_pose, rounding_mode="floor")
        rotation = poses[batch, :3, :3]
        translation = poses[batch, :3, 3]
        center_camera = torch.bmm(
            rotation, centers[primitive, :, None],
        )[:, :, 0] + translation
        normal_camera = torch.bmm(
            rotation, normals[primitive, :, None],
        )[:, :, 0]
        normal_camera = normal_camera / torch.linalg.vector_norm(
            normal_camera, dim=1, keepdim=True,
        ).clamp_min(1.0e-8)
        depth = center_camera[:, 2]
        keep = (
            (rows >= 0) & (weight > 0.0) & (depth > 1.0e-6)
            & torch.isfinite(depth)
        )
        selected = torch.nonzero(keep, as_tuple=False).reshape(-1)
        if selected.numel():
            chosen_token = token[selected]
            chosen_weight = weight[selected]
            chosen_normal = normal_camera[selected]
            x, y, z = chosen_normal.unbind(dim=1)
            axis = torch.stack((x * x, y * y, z * z, x * y, x * z, y * z), dim=1)
            log_depth = torch.log(depth[selected])
            mass.index_add_(0, chosen_token, chosen_weight)
            axis_sum.index_add_(0, chosen_token, axis * chosen_weight[:, None])
            log_depth_sum.index_add_(0, chosen_token, log_depth * chosen_weight)
            log_depth_square_sum.index_add_(
                0, chosen_token, log_depth.square() * chosen_weight,
            )
    axis_moment = axis_sum / mass[:, None].clamp_min(1.0e-8)
    mean_log_depth = log_depth_sum / mass.clamp_min(1.0e-8)
    variance = torch.clamp_min(
        log_depth_square_sum / mass.clamp_min(1.0e-8) - mean_log_depth.square(), 0.0,
    )
    pose_index = torch.arange(total_tokens, device=token.device) // tokens_per_pose
    pose_mass = torch.zeros((poses.shape[0],), dtype=torch.float32, device=token.device)
    pose_log_sum = torch.zeros_like(pose_mass)
    pose_mass.index_add_(0, pose_index, mass)
    pose_log_sum.index_add_(0, pose_index, log_depth_sum)
    pose_mean_log = pose_log_sum / pose_mass.clamp_min(1.0e-8)
    relative_log_depth = mean_log_depth - pose_mean_log[pose_index]
    valid = mass > 0.0
    axis_moment = torch.where(valid[:, None], axis_moment, torch.zeros_like(axis_moment))
    relative_log_depth = torch.where(valid, relative_log_depth, torch.zeros_like(relative_log_depth))
    log_depth_std = torch.where(valid, torch.sqrt(variance), torch.zeros_like(variance))
    boundary = torch.where(valid, torch.clamp(1.0 - mass, 0.0, 1.0), torch.zeros_like(mass))
    return axis_moment, relative_log_depth, log_depth_std, boundary, mass, valid


@dataclass(frozen=True)
class ResidentRendererBatchAudit:
    batch_size: int
    projection_tile_raster_seconds: float
    device_to_host_seconds: float
    depth_sort_composite_seconds: float
    raw_token_gather_seconds: float
    child_identity_reduction_seconds: float
    feature_reduction_seconds: float
    typed_finalize_seconds: float
    direct_parent_reduction_seconds: float
    child_and_feature_reduction_seconds: float
    raster_seconds: float
    host_reduction_seconds: float
    total_seconds: float
    packed_hit_count: int
    remapped_hit_count: int
    resident_geometry_bytes: int
    gpu_compositor_implemented: bool
    gpu_child_reducer_implemented: bool
    production_speed_gate_passed: bool


@dataclass(frozen=True)
class ResidentSoftSurfaceBatch:
    rendered: tuple[RenderedSoftChildMixture, ...]
    audit: ResidentRendererBatchAudit


@dataclass(frozen=True)
class ResidentCanonicalTokenGridBatch:
    feature: np.ndarray
    mass: np.ndarray
    valid: np.ndarray
    batch_size: int
    raster_seconds: float
    token_remap_seconds: float
    direct_reduction_seconds: float
    total_seconds: float
    packed_hit_count: int
    remapped_hit_count: int
    semantics: str = "full_scene_occluded_direct_canonical_token_feature_grid_v1"
    production_eligible: bool = False


@dataclass(frozen=True)
class ResidentTypedCanonicalTokenGridBatch:
    feature: np.ndarray
    mass: np.ndarray
    valid: np.ndarray
    normal_axis_moment: np.ndarray
    relative_log_depth: np.ndarray
    log_depth_std: np.ndarray
    boundary: np.ndarray
    batch_size: int
    total_seconds: float
    packed_hit_count: int
    remapped_hit_count: int
    semantics: str = "full_scene_direct_canonical_plus_typed_geometry_grid_v1"
    normal_semantics: str = "camera_frame_unsigned_axis_second_moment_xx_yy_zz_xy_xz_yz_v1"
    depth_semantics: str = "candidate_view_mass_centered_log_primitive_center_depth_v1"
    production_eligible: bool = False


class FrozenSoftSurfaceSceneGPU:
    """Hash-bound static scene with batched camera projection on one GPU."""

    def __init__(
        self,
        physical: GoalMapletPhysicalMap,
        field: CanonicalSurfaceField,
        *,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for resident surface rendering") from exc
        self._torch = torch
        self.device = torch.device(str(device))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FrozenSoftSurfaceSceneGPU requires CUDA")
        if field.physical_map_sha256 != physical.content_sha256:
            raise ValueError("canonical field and physical map lineage differ")
        self.physical = physical
        self.field = field
        elements = SurfaceElementMap(
            element_ids=physical.primitive_ids,
            parent_gaussian_indices=physical.primitive_ids,
            centers=physical.primitive_centers,
            tangent1=physical.primitive_tangent1,
            tangent2=physical.primitive_tangent2,
            normals=physical.primitive_normals,
            scale1=physical.primitive_scale1,
            scale2=physical.primitive_scale2,
            opacity=physical.primitive_opacity,
            area=np.pi * physical.primitive_scale1 * physical.primitive_scale2,
            adjacency=tuple(),
            metadata={"representation": "goal_maplet_resident_full_clean_scene_v1"},
        )
        quats, scales = _surface_element_quaternions_and_scales(elements)
        self.means = torch.as_tensor(
            physical.primitive_centers, dtype=torch.float32, device=self.device
        ).contiguous()
        self.quats = torch.as_tensor(
            quats, dtype=torch.float32, device=self.device
        ).contiguous()
        self.scales = torch.as_tensor(
            scales, dtype=torch.float32, device=self.device
        ).contiguous()
        self.opacities = torch.as_tensor(
            physical.primitive_opacity, dtype=torch.float32, device=self.device
        ).reshape(-1).clamp(0.0, 1.0).contiguous()
        self.normals = torch.as_tensor(
            physical.primitive_normals, dtype=torch.float32, device=self.device
        ).contiguous()
        self.double_sided = torch.as_tensor(
            np.asarray(physical.primitive_sidedness, dtype=np.uint8) == DOUBLE_SIDED,
            dtype=torch.bool, device=self.device,
        ).contiguous()
        child_owner = np.array(dominant_child_owner(physical), dtype=np.int64, copy=True)
        parent_owner = np.full(child_owner.shape, -1, dtype=np.int64)
        owned = child_owner >= 0
        parent_owner[owned] = np.asarray(
            physical.child_parent_rows, dtype=np.int64
        )[child_owner[owned]]
        field_row = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
        field_row[field.primitive_rows] = np.arange(field.primitive_rows.size, dtype=np.int64)
        self.field_row_by_primitive_numpy = field_row.copy()
        self.stable_primitive_ids = torch.as_tensor(
            physical.primitive_ids, dtype=torch.int64, device=self.device
        ).contiguous()
        self.child_owner = torch.as_tensor(
            child_owner, dtype=torch.int64, device=self.device
        ).contiguous()
        self.parent_owner = torch.as_tensor(
            parent_owner, dtype=torch.int64, device=self.device
        ).contiguous()
        parent_offsets, parent_rows, parent_weights = _primitive_parent_memberships(
            physical
        )
        self.parent_membership_offsets = torch.as_tensor(
            np.array(parent_offsets, dtype=np.int64, copy=True),
            dtype=torch.int64, device=self.device,
        ).contiguous()
        self.parent_membership_rows = torch.as_tensor(
            np.array(parent_rows, dtype=np.int64, copy=True),
            dtype=torch.int64, device=self.device,
        ).contiguous()
        self.parent_membership_weights = torch.as_tensor(
            np.array(parent_weights, dtype=np.float32, copy=True),
            dtype=torch.float32, device=self.device,
        ).contiguous()
        self.stable_parent_ids = torch.as_tensor(
            np.array(physical.maplet_ids, dtype=np.int64, copy=True),
            dtype=torch.int64, device=self.device,
        ).contiguous()
        self.field_row_by_primitive = torch.as_tensor(
            field_row, dtype=torch.int64, device=self.device
        ).contiguous()
        codes = np.asarray(field.codes, dtype=np.float32)
        self.normalized_codes = codes / np.maximum(
            np.linalg.norm(codes, axis=1, keepdims=True), 1e-8
        )
        self.canonical_codes = torch.as_tensor(
            self.normalized_codes, dtype=torch.float32, device=self.device
        ).contiguous()
        self.field_confidence = torch.as_tensor(
            field.confidence, dtype=torch.float32, device=self.device
        ).contiguous()
        self.field_uncertainty = torch.as_tensor(
            field.uncertainty, dtype=torch.float32, device=self.device
        ).contiguous()
        self._view_conditioned_torch_cache: dict[str, dict[str, object]] = {}
        self._resident_geometry_bytes = int(sum(
            value.numel() * value.element_size()
            for value in (
                self.means, self.quats, self.scales, self.opacities,
                self.normals, self.double_sided,
                self.stable_primitive_ids, self.child_owner, self.parent_owner,
                self.parent_membership_offsets, self.parent_membership_rows,
                self.parent_membership_weights, self.stable_parent_ids,
                self.field_row_by_primitive, self.canonical_codes,
                self.field_confidence, self.field_uncertainty,
            )
        ))

    def _view_conditioned_cache(
        self, view_field: ViewConditionedPrimitiveField,
    ) -> dict[str, object]:
        """Materialize one hash-bound field-aligned view model on the GPU."""

        view_field.validate_alignment(
            physical_map_sha256=self.physical.content_sha256,
            canonical_field_sha256=self.field.content_sha256,
            canonical_primitive_rows=self.field.primitive_rows,
            canonical_feature_dim=self.field.feature_dim,
        )
        key = str(view_field.content_sha256)
        cached = self._view_conditioned_torch_cache.get(key)
        if cached is not None:
            return cached
        torch = self._torch
        primitive = np.asarray(self.field.primitive_rows, dtype=np.int64)
        tensor = lambda value, dtype: torch.as_tensor(
            np.asarray(value), dtype=dtype, device=self.device,
        ).contiguous()
        cached = {
            "centers": tensor(self.physical.primitive_centers[primitive], torch.float32),
            "tangent1": tensor(self.physical.primitive_tangent1[primitive], torch.float32),
            "tangent2": tensor(self.physical.primitive_tangent2[primitive], torch.float32),
            "normals": tensor(self.physical.primitive_normals[primitive], torch.float32),
            "radius": tensor(np.sqrt(np.maximum(
                self.physical.primitive_scale1[primitive]
                * self.physical.primitive_scale2[primitive], 1.0e-12,
            )), torch.float32),
            "basis": tensor(view_field.residual_basis, torch.float32),
            "coefficients": tensor(view_field.coefficients, torch.float16),
            "observation_count": tensor(view_field.observation_count, torch.int32),
            "mean_direction": tensor(view_field.mean_local_direction, torch.float32),
            "direction_concentration": tensor(
                view_field.direction_concentration, torch.float32,
            ),
            "minimum_direction_cosine": tensor(
                view_field.minimum_direction_cosine, torch.float32,
            ),
            "mean_log_scale": tensor(
                view_field.mean_log_projected_scale, torch.float32,
            ),
            "minimum_log_scale": tensor(
                view_field.minimum_log_projected_scale, torch.float32,
            ),
            "maximum_log_scale": tensor(
                view_field.maximum_log_projected_scale, torch.float32,
            ),
            "minimum_views": int(view_field.minimum_views),
            "direction_margin": float(
                dict(view_field.metadata or {}).get("direction_cosine_margin", 0.05)
            ),
            "scale_margin": float(
                dict(view_field.metadata or {}).get("log_scale_margin", 0.25)
            ),
        }
        self._view_conditioned_torch_cache[key] = cached
        return cached

    def _batch_ideal_hits(
        self,
        poses_w2c: np.ndarray,
        camera,
        *,
        render_width: int,
        render_height: int,
        minimum_incidence: float,
        return_device_tensors: bool = False,
    ):
        torch = self._torch
        try:
            from gsplat.cuda._wrapper import (
                fully_fused_projection_2dgs,
                isect_offset_encode,
                isect_tiles,
                rasterize_to_indices_in_range_2dgs,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("gsplat 2DGS CUDA wrappers are required") from exc
        pose = np.asarray(poses_w2c, dtype=np.float32)
        if pose.ndim != 3 or pose.shape[1:] != (4, 4) or pose.shape[0] == 0:
            raise ValueError("poses_w2c must have shape [batch,4,4]")
        if np.any(~np.isfinite(pose)):
            raise ValueError("poses_w2c must be finite")
        batch = int(pose.shape[0])
        viewmats = torch.as_tensor(pose, dtype=torch.float32, device=self.device).contiguous()
        k = _intrinsic_matrix(camera, int(render_width), int(render_height))
        ks = torch.as_tensor(
            np.broadcast_to(k, (batch, 3, 3)).copy(),
            dtype=torch.float32, device=self.device,
        ).contiguous()
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        radii, means2d, depths, transforms, _camera_normals = fully_fused_projection_2dgs(
            self.means, self.quats, self.scales, viewmats, ks,
            int(render_width), int(render_height), packed=False,
        )
        rotation = viewmats[:, :3, :3]
        translation = viewmats[:, :3, 3]
        camera_center = -torch.bmm(rotation.transpose(1, 2), translation[..., None])[..., 0]
        view = camera_center[:, None, :] - self.means[None, :, :]
        view = view / torch.linalg.vector_norm(view, dim=2, keepdim=True).clamp_min(1e-12)
        signed = torch.sum(self.normals[None, :, :] * view, dim=2)
        incidence = torch.where(
            self.double_sided[None, :], torch.abs(signed), torch.clamp_min(signed, 0.0)
        )
        front = incidence >= float(minimum_incidence)
        radii = torch.where(front, radii, torch.zeros_like(radii)).contiguous()
        per_camera_opacity = (
            self.opacities[None, :].expand(batch, -1) * front.to(torch.float32)
        ).contiguous()
        tile_size = 16
        tile_width = math.ceil(int(render_width) / tile_size)
        tile_height = math.ceil(int(render_height) / tile_size)
        _tiles, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=False, n_cameras=batch,
        )
        offsets = isect_offset_encode(isect_ids, batch, tile_width, tile_height)
        transmittance = torch.ones(
            (batch, int(render_height), int(render_width)),
            dtype=torch.float32, device=self.device,
        )
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_2dgs(
            0, 1_000_000_000, transmittance, means2d, transforms,
            per_camera_opacity, int(render_width), int(render_height), tile_size,
            offsets, flatten_ids,
        )
        packed_hit_count = int(gs_ids.numel())
        if packed_hit_count == 0:
            torch.cuda.synchronize(self.device)
            if bool(return_device_tensors):
                empty_identity = torch.zeros((0,), dtype=torch.int64, device=self.device)
                empty_weight = torch.zeros((0,), dtype=torch.float32, device=self.device)
                return (
                    empty_identity, empty_identity.clone(), empty_weight, 0,
                    {
                        "projection_tile_raster_seconds": time.perf_counter() - started,
                        "device_to_host_seconds": 0.0,
                        "depth_sort_composite_seconds": 0.0,
                    },
                )
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32), 0,
                {
                    "projection_tile_raster_seconds": time.perf_counter() - started,
                    "device_to_host_seconds": 0.0,
                    "depth_sort_composite_seconds": 0.0,
                },
            )
        px = (pixel_ids % int(render_width)).to(torch.float32) + 0.5
        py = (pixel_ids // int(render_width)).to(torch.float32) + 0.5
        delta = torch.stack([px, py], dim=-1) - means2d[camera_ids, gs_ids]
        transform = transforms[camera_ids, gs_ids]
        h_u = -transform[..., 0, :3] + transform[..., 2, :3] * px[..., None]
        h_v = -transform[..., 1, :3] + transform[..., 2, :3] * py[..., None]
        tmp = torch.cross(h_u, h_v, dim=-1)
        denominator = tmp[..., 2]
        denominator = torch.where(
            torch.abs(denominator) < 1e-12,
            torch.where(denominator >= 0.0, 1e-12, -1e-12), denominator,
        )
        u = tmp[..., 0] / denominator
        v = tmp[..., 1] / denominator
        sigma3 = u * u + v * v
        sigma2 = 2.0 * torch.sum(delta * delta, dim=1)
        sigma = 0.5 * torch.minimum(sigma3, sigma2)
        alpha = torch.clamp(
            per_camera_opacity[camera_ids, gs_ids] * torch.exp(-sigma), max=0.999
        )
        hit_depth = depths[camera_ids, gs_ids]
        global_pixel = camera_ids.to(torch.int64) * (
            int(render_width) * int(render_height)
        ) + pixel_ids.to(torch.int64)
        torch.cuda.synchronize(self.device)
        projection_seconds = time.perf_counter() - started
        composite_started = time.perf_counter()
        out_pixel, out_row, out_weight = _composite_packed_hits_torch(
            global_pixel.contiguous(), hit_depth.contiguous(),
            gs_ids.to(torch.int64).contiguous(), alpha.contiguous(),
        )
        torch.cuda.synchronize(self.device)
        composite_seconds = time.perf_counter() - composite_started
        if bool(return_device_tensors):
            return out_pixel, out_row, out_weight, packed_hit_count, {
                "projection_tile_raster_seconds": float(projection_seconds),
                "device_to_host_seconds": 0.0,
                "depth_sort_composite_seconds": float(composite_seconds),
            }
        transfer_started = time.perf_counter()
        arrays = [
            out_pixel.detach().cpu().numpy(),
            out_row.detach().cpu().numpy(),
            out_weight.detach().cpu().numpy(),
        ]
        torch.cuda.synchronize(self.device)
        transfer_seconds = time.perf_counter() - transfer_started
        out_pixel_np, out_row_np, out_weight_np = arrays
        return out_pixel_np, out_row_np, out_weight_np, packed_hit_count, {
            "projection_tile_raster_seconds": float(projection_seconds),
            "device_to_host_seconds": float(transfer_seconds),
            "depth_sort_composite_seconds": float(composite_seconds),
        }

    @staticmethod
    def _batch_token_remap_torch(
        global_ideal_pixels,
        primitive_rows,
        contribution,
        camera,
        *,
        batch_size: int,
        token_width: int,
        token_height: int,
        supersample_factor: int,
    ):
        """Device-resident equivalent of :meth:`_batch_token_remap`."""

        import torch

        values = (global_ideal_pixels, primitive_rows, contribution)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("device token-remap inputs must be Torch tensors")
        hits, primitive, weight = values
        if any(value.ndim != 1 for value in values) or not (
            hits.shape == primitive.shape == weight.shape
        ):
            raise ValueError("device token-remap hit arrays differ")
        if hits.dtype != torch.int64 or primitive.dtype != torch.int64 or weight.dtype != torch.float32:
            raise ValueError("device token-remap dtypes differ")
        if len({value.device for value in values}) != 1 or any(not value.is_contiguous() for value in values):
            raise ValueError("device token-remap arrays must be contiguous on one device")
        if hits.numel() > 1 and torch.any(hits[1:] < hits[:-1]):
            raise ValueError("device token remap requires globally sorted pixel hits")
        factor = int(supersample_factor)
        render_pixels = int(token_width) * factor * int(token_height) * factor
        token_pixels = int(token_width) * int(token_height)
        warp = _raw_to_ideal_token_warp(
            camera, token_width=int(token_width), token_height=int(token_height),
            supersample_factor=factor,
        )
        device = hits.device
        source_local = torch.tensor(
            warp.source_ideal_pixel_ids, dtype=torch.int64, device=device,
        )
        destination_local = torch.tensor(
            warp.destination_token_pixel_ids, dtype=torch.int64, device=device,
        )
        batch_rows = torch.arange(int(batch_size), dtype=torch.int64, device=device)[:, None]
        sources = (batch_rows * render_pixels + source_local[None]).reshape(-1)
        destinations = (batch_rows * token_pixels + destination_local[None]).reshape(-1)
        left = torch.searchsorted(hits, sources, right=False)
        right = torch.searchsorted(hits, sources, right=True)
        keep = right > left
        left, right, destinations = left[keep], right[keep], destinations[keep]
        if destinations.numel() == 0:
            return (
                torch.zeros((0,), dtype=torch.int64, device=device),
                torch.zeros((0,), dtype=torch.int64, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
            )
        counts = right - left
        expanded_left = torch.repeat_interleave(left, counts)
        group_origin = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        selected = expanded_left + (
            torch.arange(int(torch.sum(counts).item()), dtype=torch.int64, device=device)
            - group_origin
        )
        return (
            torch.repeat_interleave(destinations, counts).contiguous(),
            primitive[selected].contiguous(),
            (weight[selected] / float(factor * factor)).contiguous(),
        )

    @staticmethod
    def _batch_token_remap(
        global_ideal_pixels: np.ndarray,
        primitive_rows: np.ndarray,
        contribution: np.ndarray,
        camera,
        *,
        batch_size: int,
        token_width: int,
        token_height: int,
        supersample_factor: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        factor = int(supersample_factor)
        render_pixels = int(token_width) * factor * int(token_height) * factor
        token_pixels = int(token_width) * int(token_height)
        warp = _raw_to_ideal_token_warp(
            camera, token_width=int(token_width), token_height=int(token_height),
            supersample_factor=factor,
        )
        sources = (
            np.arange(int(batch_size), dtype=np.int64)[:, None] * render_pixels
            + warp.source_ideal_pixel_ids[None, :]
        ).reshape(-1)
        destinations = (
            np.arange(int(batch_size), dtype=np.int64)[:, None] * token_pixels
            + warp.destination_token_pixel_ids[None, :]
        ).reshape(-1)
        hits = np.asarray(global_ideal_pixels, dtype=np.int64).reshape(-1)
        if hits.shape != np.asarray(primitive_rows).reshape(-1).shape or hits.shape != np.asarray(contribution).reshape(-1).shape:
            raise ValueError("batched token-remap hit arrays differ")
        if np.any(hits[1:] < hits[:-1]):
            raise ValueError("batched token remap requires globally sorted pixel hits")
        left = np.searchsorted(hits, sources, side="left")
        right = np.searchsorted(hits, sources, side="right")
        keep = right > left
        left, right, destinations = left[keep], right[keep], destinations[keep]
        if destinations.size == 0:
            return (
                np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )
        counts = right - left
        expanded_left = np.repeat(left, counts)
        group_origin = np.repeat(np.cumsum(counts) - counts, counts)
        selected = expanded_left + (
            np.arange(int(np.sum(counts)), dtype=np.int64) - group_origin
        )
        return (
            np.repeat(destinations, counts),
            np.asarray(primitive_rows, dtype=np.int64)[selected],
            (
                np.asarray(contribution, dtype=np.float32)[selected]
                / float(factor * factor)
            ).astype(np.float32),
        )

    def _render_exact_batch_device_reduced(
        self,
        pose: np.ndarray,
        camera,
        *,
        width: int,
        height: int,
        selected_child_rows: np.ndarray | None,
        top_l: int,
        coordinate_supersample_factor: int,
        minimum_incidence: float,
        minimum_feature_alpha: float,
        alpha_conservation_tolerance: float,
    ) -> ResidentSoftSurfaceBatch:
        """Exact resident fast path for the view-independent canonical field."""

        torch = self._torch
        batch = int(pose.shape[0])
        started = time.perf_counter()
        factor = int(coordinate_supersample_factor)
        ideal_pixel, primitive, weight, packed_count, timings = self._batch_ideal_hits(
            pose, camera,
            render_width=int(width) * factor,
            render_height=int(height) * factor,
            minimum_incidence=float(minimum_incidence),
            return_device_tensors=True,
        )
        remap_started = time.perf_counter()
        token, primitive, weight = self._batch_token_remap_torch(
            ideal_pixel, primitive, weight, camera, batch_size=batch,
            token_width=int(width), token_height=int(height),
            supersample_factor=factor,
        )
        torch.cuda.synchronize(self.device)
        remap_seconds = time.perf_counter() - remap_started
        reduction_started = time.perf_counter()
        selected = None
        if selected_child_rows is not None:
            selected = torch.as_tensor(
                np.asarray(selected_child_rows, dtype=np.int64),
                dtype=torch.int64, device=self.device,
            ).contiguous()
        token_count = int(width) * int(height)
        reduced_rows = []
        # Keep dynamic reduction memory bounded by one pose.  Raster batching
        # remains unchanged; only the already-occluded hit stream is sliced.
        # This avoids making the builder's safe batch size depend on the number
        # of retained feature channels or hierarchy memberships.
        for row in range(batch):
            lower, upper = row * token_count, (row + 1) * token_count
            mask = (token >= lower) & (token < upper)
            reduced_rows.append(reduce_soft_child_token_hits_torch(
                token_ids=(token[mask] - lower).contiguous(),
                primitive_rows=primitive[mask].contiguous(),
                contribution=weight[mask].contiguous(),
                stable_primitive_ids=self.stable_primitive_ids,
                child_owner_by_primitive=self.child_owner,
                field_row_by_primitive=self.field_row_by_primitive,
                canonical_codes=self.canonical_codes,
                parent_membership_offsets=self.parent_membership_offsets,
                parent_membership_rows=self.parent_membership_rows,
                parent_membership_weights=self.parent_membership_weights,
                stable_parent_ids=self.stable_parent_ids,
                token_count=token_count,
                child_count=int(self.physical.child_parent_rows.size),
                top_l=int(top_l),
                minimum_feature_alpha=float(minimum_feature_alpha),
                alpha_conservation_tolerance=float(alpha_conservation_tolerance),
                selected_child_rows=selected,
            ))
        torch.cuda.synchronize(self.device)
        reduction_seconds = time.perf_counter() - reduction_started
        transfer_started = time.perf_counter()
        names = (
            "child_rows", "child_weights", "child_features", "child_feature_valid",
            "parent_rows", "parent_weights", "parent_tail_weight",
            "child_tail_weight", "unassigned_geometry_weight", "background_weight",
            "canonical_field_missing_weight", "payload_excluded_weight", "null_weight",
            "total_alpha", "alpha_overflow",
        )
        host = {
            name: torch.cat([
                getattr(reduced, name) for reduced in reduced_rows
            ], dim=0).detach().cpu().numpy()
            for name in names
        }
        torch.cuda.synchronize(self.device)
        transfer_seconds = time.perf_counter() - transfer_started
        shape = (int(height), int(width))
        rendered: list[RenderedSoftChildMixture] = []
        materialize_started = time.perf_counter()
        for row in range(batch):
            lower, upper = row * token_count, (row + 1) * token_count
            overflow = host["alpha_overflow"][lower:upper]
            rendered.append(RenderedSoftChildMixture(
                child_rows=host["child_rows"][lower:upper].reshape(
                    *shape, int(top_l)
                ),
                child_weights=host["child_weights"][lower:upper].reshape(
                    *shape, int(top_l)
                ),
                child_features=host["child_features"][lower:upper].reshape(
                    *shape, int(top_l), self.field.feature_dim
                ),
                child_feature_valid=host["child_feature_valid"][lower:upper].reshape(
                    *shape, int(top_l)
                ),
                parent_rows=host["parent_rows"][lower:upper].reshape(
                    *shape, int(top_l)
                ),
                parent_weights=host["parent_weights"][lower:upper].reshape(
                    *shape, int(top_l)
                ),
                parent_tail_weight=host["parent_tail_weight"][lower:upper].reshape(shape),
                child_tail_weight=host["child_tail_weight"][lower:upper].reshape(shape),
                unassigned_geometry_weight=host[
                    "unassigned_geometry_weight"
                ][lower:upper].reshape(shape),
                background_weight=host["background_weight"][lower:upper].reshape(shape),
                canonical_field_missing_weight=host[
                    "canonical_field_missing_weight"
                ][lower:upper].reshape(shape),
                payload_excluded_weight=host[
                    "payload_excluded_weight"
                ][lower:upper].reshape(shape),
                null_weight=host["null_weight"][lower:upper].reshape(shape),
                total_alpha=host["total_alpha"][lower:upper].reshape(shape),
                maximum_alpha_overflow=float(np.max(overflow, initial=0.0)),
                overflow_token_fraction=float(np.mean(overflow > 0.0)),
            ))
        materialize_seconds = time.perf_counter() - materialize_started
        total_seconds = time.perf_counter() - started
        device_to_host_seconds = (
            float(timings["device_to_host_seconds"]) + float(transfer_seconds)
        )
        raster_seconds = (
            float(timings["projection_tile_raster_seconds"])
            + float(timings["depth_sort_composite_seconds"])
            + device_to_host_seconds
        )
        return ResidentSoftSurfaceBatch(
            rendered=tuple(rendered),
            audit=ResidentRendererBatchAudit(
                batch_size=batch,
                projection_tile_raster_seconds=float(
                    timings["projection_tile_raster_seconds"]
                ),
                device_to_host_seconds=device_to_host_seconds,
                depth_sort_composite_seconds=float(
                    timings["depth_sort_composite_seconds"]
                ),
                raw_token_gather_seconds=float(remap_seconds),
                # The device reducer is a fused child/feature/parent operation;
                # record its wall time once instead of double-counting it across
                # the legacy CPU substage fields.
                child_identity_reduction_seconds=float(reduction_seconds),
                feature_reduction_seconds=0.0,
                typed_finalize_seconds=0.0,
                direct_parent_reduction_seconds=0.0,
                child_and_feature_reduction_seconds=float(reduction_seconds),
                raster_seconds=float(raster_seconds),
                host_reduction_seconds=float(materialize_seconds),
                total_seconds=float(total_seconds),
                packed_hit_count=int(packed_count),
                remapped_hit_count=int(weight.numel()),
                resident_geometry_bytes=self._resident_geometry_bytes,
                gpu_compositor_implemented=True,
                gpu_child_reducer_implemented=True,
                # Promotion remains gated by the independent Top-8 audit.
                production_speed_gate_passed=False,
            ),
        )

    def render_exact_batch(
        self,
        poses_w2c: np.ndarray,
        camera,
        *,
        width: int = 64,
        height: int = 36,
        selected_child_rows: np.ndarray | None = None,
        top_l: int = 4,
        coordinate_supersample_factor: int = 4,
        minimum_incidence: float = 0.05,
        minimum_feature_alpha: float = 1e-4,
        alpha_conservation_tolerance: float = 2e-5,
        view_conditioned_field: ViewConditionedPrimitiveField | None = None,
    ) -> ResidentSoftSurfaceBatch:
        pose = np.asarray(poses_w2c, dtype=np.float64)
        batch = int(pose.shape[0]) if pose.ndim else 0
        if batch <= 0:
            raise ValueError("exact resident batch cannot be empty")
        if pose.shape != (batch, 4, 4) or np.any(~np.isfinite(pose)):
            raise ValueError("exact resident batch poses must have shape [B,4,4]")
        if view_conditioned_field is None:
            return self._render_exact_batch_device_reduced(
                pose, camera, width=int(width), height=int(height),
                selected_child_rows=selected_child_rows, top_l=int(top_l),
                coordinate_supersample_factor=int(coordinate_supersample_factor),
                minimum_incidence=float(minimum_incidence),
                minimum_feature_alpha=float(minimum_feature_alpha),
                alpha_conservation_tolerance=float(alpha_conservation_tolerance),
            )
        started = time.perf_counter()
        factor = int(coordinate_supersample_factor)
        ideal_pixel, primitive, weight, packed_count, timings = self._batch_ideal_hits(
            pose, camera, render_width=int(width) * factor,
            render_height=int(height) * factor,
            minimum_incidence=float(minimum_incidence),
        )
        gather_started = time.perf_counter()
        token, primitive, weight = self._batch_token_remap(
            ideal_pixel, primitive, weight, camera, batch_size=batch,
            token_width=int(width), token_height=int(height), supersample_factor=factor,
        )
        gather_seconds = time.perf_counter() - gather_started
        reduction_started = time.perf_counter()
        token_count = int(width) * int(height)
        rendered: list[RenderedSoftChildMixture] = []
        reduction_timings: list[dict[str, float]] = []
        for row in range(batch):
            mask = (token >= row * token_count) & (token < (row + 1) * token_count)
            row_timing: dict[str, float] = {}
            conditioned_index = np.zeros((0,), dtype=np.int64)
            conditioned_code = np.zeros(
                (0, self.normalized_codes.shape[1]), dtype=np.float32
            )
            if view_conditioned_field is not None:
                hit_field_rows = self.field_row_by_primitive_numpy[primitive[mask]]
                hit_field_rows = np.unique(hit_field_rows[hit_field_rows >= 0])
                if hit_field_rows.size:
                    conditioned_index, conditioned_code, _conditioned_active = (
                        condition_canonical_codes_for_pose(
                            view_conditioned_field, self.field, self.physical,
                            pose[row], camera, field_indices=hit_field_rows,
                        )
                    )
            rendered.append(_reduce_soft_child_token_hits(
                self.physical, self.field,
                token_pixel_ids=token[mask] - row * token_count,
                primitive_rows=primitive[mask], contribution=weight[mask],
                width=int(width), height=int(height),
                normalized_codes=self.normalized_codes,
                normalized_code_override_rows=conditioned_index,
                normalized_code_override_values=conditioned_code,
                selected_child_rows=selected_child_rows, top_l=int(top_l),
                minimum_feature_alpha=float(minimum_feature_alpha),
                alpha_conservation_tolerance=float(alpha_conservation_tolerance),
                timing_sink=row_timing,
            ))
            reduction_timings.append(row_timing)
        host_seconds = time.perf_counter() - reduction_started
        total_seconds = time.perf_counter() - started
        raster_seconds = (
            timings["projection_tile_raster_seconds"]
            + timings["device_to_host_seconds"]
            + timings["depth_sort_composite_seconds"]
        )
        return ResidentSoftSurfaceBatch(
            rendered=tuple(rendered),
            audit=ResidentRendererBatchAudit(
                batch_size=batch, raster_seconds=float(raster_seconds),
                projection_tile_raster_seconds=float(timings["projection_tile_raster_seconds"]),
                device_to_host_seconds=float(timings["device_to_host_seconds"]),
                depth_sort_composite_seconds=float(timings["depth_sort_composite_seconds"]),
                raw_token_gather_seconds=float(gather_seconds),
                child_identity_reduction_seconds=float(sum(
                    row["child_identity_reduction_seconds"] for row in reduction_timings
                )),
                feature_reduction_seconds=float(sum(
                    row["feature_reduction_seconds"] for row in reduction_timings
                )),
                typed_finalize_seconds=float(sum(
                    row["typed_finalize_seconds"] for row in reduction_timings
                )),
                direct_parent_reduction_seconds=float(sum(
                    row["direct_parent_reduction_seconds"] for row in reduction_timings
                )),
                child_and_feature_reduction_seconds=float(host_seconds),
                host_reduction_seconds=float(host_seconds), total_seconds=float(total_seconds),
                packed_hit_count=int(packed_count), remapped_hit_count=int(weight.size),
                resident_geometry_bytes=self._resident_geometry_bytes,
                gpu_compositor_implemented=True,
                gpu_child_reducer_implemented=False,
                production_speed_gate_passed=False,
            ),
        )

    def render_direct_canonical_grid_batch(
        self,
        poses_w2c: np.ndarray,
        camera,
        *,
        width: int = 64,
        height: int = 36,
        coordinate_supersample_factor: int = 4,
        minimum_incidence: float = 0.05,
        minimum_feature_alpha: float = 1.0e-4,
        accumulation_chunk_rows: int = 65536,
        view_conditioned_field: ViewConditionedPrimitiveField | None = None,
    ) -> ResidentCanonicalTokenGridBatch:
        """Render the minimal full-token canonical field without identity Top-L."""

        torch = self._torch
        pose = np.asarray(poses_w2c, dtype=np.float64)
        batch = int(pose.shape[0]) if pose.ndim else 0
        if pose.shape != (batch, 4, 4) or batch <= 0:
            raise ValueError("direct canonical batch poses must have shape [B,4,4]")
        started = time.perf_counter()
        factor = int(coordinate_supersample_factor)
        ideal_pixel, primitive, weight, packed_count, timings = self._batch_ideal_hits(
            pose,
            camera,
            render_width=int(width) * factor,
            render_height=int(height) * factor,
            minimum_incidence=float(minimum_incidence),
            return_device_tensors=True,
        )
        remap_started = time.perf_counter()
        token_t, primitive_t, weight_t = self._batch_token_remap_torch(
            ideal_pixel,
            primitive,
            weight,
            camera,
            batch_size=batch,
            token_width=int(width),
            token_height=int(height),
            supersample_factor=factor,
        )
        torch.cuda.synchronize(self.device)
        remap_seconds = time.perf_counter() - remap_started
        reduce_started = time.perf_counter()
        token_count_per_pose = int(width) * int(height)
        if view_conditioned_field is None:
            feature, mass, valid = _reduce_direct_canonical_token_hits_torch(
                token_t,
                primitive_t,
                weight_t,
                self.field_row_by_primitive,
                self.canonical_codes,
                token_count=batch * token_count_per_pose,
                minimum_feature_alpha=float(minimum_feature_alpha),
                accumulation_chunk_rows=int(accumulation_chunk_rows),
            )
        else:
            cache = self._view_conditioned_cache(view_conditioned_field)
            focal_pixels = _camera_focal_pixels(camera)
            feature_rows, mass_rows, valid_rows = [], [], []
            for row in range(batch):
                lower = row * token_count_per_pose
                upper = (row + 1) * token_count_per_pose
                mask = (token_t >= lower) & (token_t < upper)
                hit_field = self.field_row_by_primitive[primitive_t[mask]]
                hit_field = torch.unique(hit_field[hit_field >= 0], sorted=True)
                if hit_field.numel():
                    override_row = hit_field.contiguous()
                    override_value, _ = _condition_canonical_codes_for_pose_torch(
                        self.canonical_codes,
                        override_row,
                        cache["centers"],
                        cache["tangent1"],
                        cache["tangent2"],
                        cache["normals"],
                        cache["radius"],
                        cache["basis"],
                        cache["coefficients"],
                        cache["observation_count"],
                        cache["mean_direction"],
                        cache["direction_concentration"],
                        cache["minimum_direction_cosine"],
                        cache["mean_log_scale"],
                        cache["minimum_log_scale"],
                        cache["maximum_log_scale"],
                        torch.as_tensor(
                            pose[row], dtype=torch.float32, device=self.device,
                        ).contiguous(),
                        focal_pixels=focal_pixels,
                        minimum_views=cache["minimum_views"],
                        direction_cosine_margin=cache["direction_margin"],
                        log_scale_margin=cache["scale_margin"],
                    )
                else:
                    override_row = torch.zeros(
                        (0,), dtype=torch.int64, device=self.device,
                    )
                    override_value = torch.zeros(
                        (0, self.canonical_codes.shape[1]),
                        dtype=torch.float32, device=self.device,
                    )
                row_feature, row_mass, row_valid = _reduce_direct_canonical_token_hits_torch(
                    token_t[mask] - lower,
                    primitive_t[mask],
                    weight_t[mask],
                    self.field_row_by_primitive,
                    self.canonical_codes,
                    token_count=token_count_per_pose,
                    minimum_feature_alpha=float(minimum_feature_alpha),
                    accumulation_chunk_rows=int(accumulation_chunk_rows),
                    override_field_rows=override_row,
                    override_codes=override_value,
                )
                feature_rows.append(row_feature)
                mass_rows.append(row_mass)
                valid_rows.append(row_valid)
            feature = torch.cat(feature_rows, dim=0)
            mass = torch.cat(mass_rows, dim=0)
            valid = torch.cat(valid_rows, dim=0)
        torch.cuda.synchronize(self.device)
        feature_np = feature.reshape(batch, int(height), int(width), -1).cpu().numpy()
        mass_np = mass.reshape(batch, int(height), int(width)).cpu().numpy()
        valid_np = valid.reshape(batch, int(height), int(width)).cpu().numpy()
        torch.cuda.synchronize(self.device)
        reduction_seconds = time.perf_counter() - reduce_started
        if np.any(mass_np > 1.0 + 2.0e-5):
            raise ValueError("direct canonical token mass exceeds the compositing contract")
        return ResidentCanonicalTokenGridBatch(
            feature=feature_np,
            mass=mass_np,
            valid=valid_np,
            batch_size=batch,
            raster_seconds=float(
                timings["projection_tile_raster_seconds"]
                + timings["device_to_host_seconds"]
                + timings["depth_sort_composite_seconds"]
            ),
            token_remap_seconds=float(remap_seconds),
            direct_reduction_seconds=float(reduction_seconds),
            total_seconds=float(time.perf_counter() - started),
            packed_hit_count=int(packed_count),
            remapped_hit_count=int(weight_t.numel()),
        )

    def render_direct_typed_canonical_grid_batch(
        self,
        poses_w2c: np.ndarray,
        camera,
        *,
        width: int = 64,
        height: int = 36,
        coordinate_supersample_factor: int = 4,
        minimum_incidence: float = 0.05,
        minimum_feature_alpha: float = 1.0e-4,
        accumulation_chunk_rows: int = 65536,
    ) -> ResidentTypedCanonicalTokenGridBatch:
        """Render canonical appearance and typed geometry from one hit stream."""

        torch = self._torch
        pose = np.asarray(poses_w2c, dtype=np.float64)
        batch = int(pose.shape[0]) if pose.ndim else 0
        if pose.shape != (batch, 4, 4) or batch <= 0 or np.any(~np.isfinite(pose)):
            raise ValueError("direct typed canonical poses must have shape [B,4,4]")
        started = time.perf_counter()
        factor = int(coordinate_supersample_factor)
        ideal_pixel, primitive, weight, packed_count, _ = self._batch_ideal_hits(
            pose, camera,
            render_width=int(width) * factor,
            render_height=int(height) * factor,
            minimum_incidence=float(minimum_incidence),
            return_device_tensors=True,
        )
        token_t, primitive_t, weight_t = self._batch_token_remap_torch(
            ideal_pixel, primitive, weight, camera, batch_size=batch,
            token_width=int(width), token_height=int(height), supersample_factor=factor,
        )
        token_count = batch * int(width) * int(height)
        feature, feature_mass, feature_valid = _reduce_direct_canonical_token_hits_torch(
            token_t, primitive_t, weight_t,
            self.field_row_by_primitive, self.canonical_codes,
            token_count=token_count,
            minimum_feature_alpha=float(minimum_feature_alpha),
            accumulation_chunk_rows=int(accumulation_chunk_rows),
        )
        typed = _reduce_direct_typed_geometry_token_hits_torch(
            token_t, primitive_t, weight_t,
            self.field_row_by_primitive,
            self.means, self.normals,
            torch.as_tensor(pose, dtype=torch.float32, device=self.device).contiguous(),
            token_count_per_pose=int(width) * int(height),
        )
        axis, relative_depth, depth_std, boundary, typed_mass, typed_valid = typed
        torch.cuda.synchronize(self.device)
        shape = (batch, int(height), int(width))
        arrays = {
            "feature": feature.reshape(*shape, -1).cpu().numpy(),
            "mass": feature_mass.reshape(shape).cpu().numpy(),
            "valid": feature_valid.reshape(shape).cpu().numpy(),
            "normal_axis_moment": axis.reshape(*shape, 6).cpu().numpy(),
            "relative_log_depth": relative_depth.reshape(shape).cpu().numpy(),
            "log_depth_std": depth_std.reshape(shape).cpu().numpy(),
            "boundary": boundary.reshape(shape).cpu().numpy(),
        }
        if not torch.allclose(feature_mass, typed_mass, atol=2.0e-6, rtol=2.0e-6):
            raise AssertionError("typed geometry and canonical feature mass differ")
        if torch.any(feature_valid & ~typed_valid):
            raise AssertionError("feature-valid token lacks typed geometry")
        return ResidentTypedCanonicalTokenGridBatch(
            **arrays,
            batch_size=batch,
            total_seconds=float(time.perf_counter() - started),
            packed_hit_count=int(packed_count),
            remapped_hit_count=int(weight_t.numel()),
        )
