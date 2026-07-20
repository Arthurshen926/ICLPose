"""Target-free absolute image-phase features for frozen full-track edges.

The local translation-mode probe recentres both query and support around their
anchors.  That is useful local evidence, but it cannot distinguish two
repeated facade cells that have the same local neighbourhood.  This module
keeps a small, fixed lattice in *image coordinates* instead.  It compares
query and support image regions for every already-frozen real SfM support
observation and never receives a pose, identity target, or retrieval result.

The visual features deliberately exclude region coverage/count fields.  A
separate position-only control exposes those fields so any apparent gain from
the anchor masks can be detected in the frozen audit rather than silently
becoming a visual identity cue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F


FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_absolute_phase_v1"
)
FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS = (
    "absolute_image_region_transport_per_real_sfm_observation_v1"
)
FULLTRACK_ABSOLUTE_PHASE_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_absolute_image_region_transport_v1"
)
ABSOLUTE_PHASE_REGION_GRID_SIZE = 3
ABSOLUTE_PHASE_CENTER_MASK_RADIUS = 1
ABSOLUTE_PHASE_TEMPERATURE = 0.07


@dataclass(frozen=True)
class AbsolutePhaseProfile:
    """One native descriptor space in the absolute-region probe."""

    name: str
    source_name: str
    grid_size: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or int(self.grid_size) <= 2 * ABSOLUTE_PHASE_CENTER_MASK_RADIUS + 1
        ):
            raise ValueError("absolute-phase profile is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "grid_size", int(self.grid_size))


ABSOLUTE_PHASE_VISUAL_PROFILES: tuple[AbsolutePhaseProfile, ...] = (
    AbsolutePhaseProfile("radio_final_absolute_phase", "radio_final", 16),
    AbsolutePhaseProfile(
        "radio_intermediate_pca256_absolute_phase", "radio_intermediate_pca256", 16
    ),
    AbsolutePhaseProfile("alike_fpn_absolute_phase", "alike_fpn", 32),
)


def _region_name(index: int, *, region_grid_size: int) -> str:
    return f"r{int(index) // int(region_grid_size)}_c{int(index) % int(region_grid_size)}"


def absolute_phase_visual_feature_names(
    profile_name: str,
    *,
    region_grid_size: int = ABSOLUTE_PHASE_REGION_GRID_SIZE,
) -> tuple[str, ...]:
    """Stable names for appearance-only absolute-region transport fields."""

    name = str(profile_name)
    size = int(region_grid_size)
    if not name or size <= 1:
        raise ValueError("absolute-phase visual feature naming inputs are invalid")
    region_count = size * size
    fields: list[str] = []
    for query_region in range(region_count):
        for support_region in range(region_count):
            fields.append(
                f"{name}_qregion_{_region_name(query_region, region_grid_size=size)}"
                f"_sregion_{_region_name(support_region, region_grid_size=size)}"
                "_attention_mass"
            )
    for statistic in (
        "same_region_cosine",
        "best_cosine",
        "best_minus_second",
        "normalized_entropy",
        "expected_dx",
        "expected_dy",
    ):
        for query_region in range(region_count):
            fields.append(
                f"{name}_qregion_{_region_name(query_region, region_grid_size=size)}"
                f"_{statistic}"
            )
    return tuple(fields)


def absolute_phase_position_feature_names(
    profile_name: str,
    *,
    region_grid_size: int = ABSOLUTE_PHASE_REGION_GRID_SIZE,
) -> tuple[str, ...]:
    """Matched mask-geometry control names, kept out of visual families."""

    name = str(profile_name)
    size = int(region_grid_size)
    if not name or size <= 1:
        raise ValueError("absolute-phase position feature naming inputs are invalid")
    region_count = size * size
    fields: list[str] = []
    for side in ("query", "support"):
        for region in range(region_count):
            fields.append(
                f"{name}_{side}_region_{_region_name(region, region_grid_size=size)}"
                "_coverage_control"
            )
    return tuple(fields)


ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE = {
    profile.name: absolute_phase_visual_feature_names(profile.name)
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
}
ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE = {
    profile.name: absolute_phase_position_feature_names(profile.name)
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
}
FULLTRACK_ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES = tuple(
    field
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
    for field in ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE[profile.name]
)
FULLTRACK_ABSOLUTE_PHASE_POSITION_FEATURE_NAMES = tuple(
    field
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
    for field in ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE[profile.name]
)
FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES = (
    *FULLTRACK_ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES,
    *FULLTRACK_ABSOLUTE_PHASE_POSITION_FEATURE_NAMES,
)


def absolute_phase_anchor_mask(
    *,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    grid_size: int,
    center_mask_radius: int = ABSOLUTE_PHASE_CENTER_MASK_RADIUS,
) -> torch.Tensor:
    """Return a full-image mask with only the local anchor core excluded."""

    sizes = torch.as_tensor(image_sizes, dtype=torch.float32, device=xy.device)
    indices = torch.as_tensor(image_indices, dtype=torch.long, device=xy.device)
    coordinates = torch.as_tensor(xy, dtype=torch.float32, device=xy.device)
    size = int(grid_size)
    radius = int(center_mask_radius)
    if (
        sizes.ndim != 2
        or sizes.shape[1] != 2
        or indices.ndim != 1
        or coordinates.shape != (len(indices), 2)
        or len(indices) == 0
        or torch.any(indices < 0)
        or torch.any(indices >= len(sizes))
        or torch.any(sizes <= 1.0)
        or size <= 2 * radius + 1
        or radius < 0
    ):
        raise ValueError("absolute-phase anchor mask inputs are invalid")
    selected_sizes = sizes.index_select(0, indices)
    columns = torch.round(
        coordinates[:, 0] * float(size - 1) / selected_sizes[:, 0].sub(1.0)
    ).long().clamp(0, size - 1)
    rows = torch.round(
        coordinates[:, 1] * float(size - 1) / selected_sizes[:, 1].sub(1.0)
    ).long().clamp(0, size - 1)
    lattice = torch.arange(size, device=coordinates.device)
    lattice_rows, lattice_columns = torch.meshgrid(lattice, lattice, indexing="ij")
    excluded = (
        (lattice_rows[None] - rows[:, None, None]).abs() <= radius
    ) & ((lattice_columns[None] - columns[:, None, None]).abs() <= radius)
    return ~excluded


def _region_membership(
    *, grid_size: int, region_grid_size: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(grid_size)
    regions = int(region_grid_size)
    if size <= 0 or regions <= 1 or regions > size:
        raise ValueError("absolute-phase region lattice is invalid")
    rows, columns = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing="ij",
    )
    region_rows = torch.div(rows * regions, size, rounding_mode="floor")
    region_columns = torch.div(columns * regions, size, rounding_mode="floor")
    indices = (region_rows * regions + region_columns).reshape(-1)
    membership = F.one_hot(indices, num_classes=regions * regions).to(dtype=dtype)
    cells_per_region = membership.sum(dim=0).clamp_min(1.0)
    return membership, cells_per_region


def _validate_grid_inputs(
    *, grid: torch.Tensor, valid: torch.Tensor, region_grid_size: int
) -> tuple[int, int]:
    values = torch.as_tensor(grid)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    if (
        values.ndim != 4
        or values.shape[1] != values.shape[2]
        or values.shape[:3] != mask.shape
        or values.shape[0] == 0
        or values.shape[3] == 0
        or int(region_grid_size) <= 1
        or int(region_grid_size) > int(values.shape[1])
    ):
        raise ValueError("absolute-phase region transport inputs are invalid")
    return int(values.shape[0]), int(values.shape[1])


def _pooled_regions(
    *, grid: torch.Tensor, valid: torch.Tensor, region_grid_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool native grid descriptors into fixed image-coordinate regions."""

    batch, grid_size = _validate_grid_inputs(
        grid=grid, valid=valid, region_grid_size=region_grid_size
    )
    values = torch.as_tensor(grid)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    region_count = int(region_grid_size) ** 2
    membership, cells_per_region = _region_membership(
        grid_size=grid_size,
        region_grid_size=int(region_grid_size),
        device=values.device,
        dtype=torch.float32,
    )
    tokens = F.normalize(
        values.reshape(batch, grid_size * grid_size, values.shape[-1]).to(torch.float32),
        p=2,
        dim=2,
        eps=1e-8,
    )
    weights = mask.reshape(batch, grid_size * grid_size).to(torch.float32)[..., None]
    weights = weights * membership[None]
    counts = weights.sum(dim=1)
    pooled = torch.bmm(weights.transpose(1, 2), tokens)
    pooled = pooled / counts[..., None].clamp_min(1.0)
    region_valid = counts > 0.0
    pooled = torch.where(region_valid[..., None], pooled, torch.zeros_like(pooled))
    pooled = F.normalize(pooled, p=2, dim=2, eps=1e-8)
    coverage = counts / cells_per_region[None]
    if pooled.shape != (batch, region_count, values.shape[-1]):
        raise RuntimeError("absolute-phase pooled region layout drifted")
    return pooled, region_valid, coverage


def batched_absolute_phase_region_transport_features(
    query_grid: torch.Tensor,
    query_valid: torch.Tensor,
    support_grid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    region_grid_size: int = ABSOLUTE_PHASE_REGION_GRID_SIZE,
    temperature: float = ABSOLUTE_PHASE_TEMPERATURE,
) -> torch.Tensor:
    """Return appearance-only absolute-region transport for a CSR edge batch."""

    if float(temperature) <= 0.0:
        raise ValueError("absolute-phase temperature must be positive")
    query = torch.as_tensor(query_grid)
    support = torch.as_tensor(support_grid, device=query.device)
    q_mask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device)
    s_mask = torch.as_tensor(support_valid, dtype=torch.bool, device=query.device)
    if query.shape != support.shape or q_mask.shape != s_mask.shape:
        raise ValueError("absolute-phase query/support tensors are incompatible")
    batch, _grid_size = _validate_grid_inputs(
        grid=query, valid=q_mask, region_grid_size=region_grid_size
    )
    _validate_grid_inputs(grid=support, valid=s_mask, region_grid_size=region_grid_size)
    query_regions, query_region_valid, _query_coverage = _pooled_regions(
        grid=query, valid=q_mask, region_grid_size=region_grid_size
    )
    support_regions, support_region_valid, _support_coverage = _pooled_regions(
        grid=support, valid=s_mask, region_grid_size=region_grid_size
    )
    region_count = int(region_grid_size) ** 2
    cosine = torch.bmm(query_regions, support_regions.transpose(1, 2))
    pair_valid = query_region_valid[:, :, None] & support_region_valid[:, None, :]
    row_valid = query_region_valid & support_region_valid.any(dim=1, keepdim=True)
    masked_cosine = torch.where(pair_valid, cosine, torch.full_like(cosine, -1e4))
    attention = torch.softmax(masked_cosine / float(temperature), dim=2)
    attention = torch.where(row_valid[..., None], attention, torch.zeros_like(attention))
    diagonal = torch.diagonal(cosine, dim1=1, dim2=2)
    diagonal_valid = query_region_valid & support_region_valid
    same_region = torch.where(diagonal_valid, diagonal, torch.zeros_like(diagonal))
    sorted_cosine, _ = torch.sort(masked_cosine, dim=2, descending=True)
    best = torch.where(row_valid, sorted_cosine[:, :, 0], torch.zeros_like(same_region))
    second = torch.where(
        row_valid & (support_region_valid.sum(dim=1, keepdim=True) >= 2),
        sorted_cosine[:, :, 1],
        best,
    )
    entropy = -torch.sum(
        attention * torch.log(attention.clamp_min(torch.finfo(attention.dtype).tiny)), dim=2
    )
    support_count = support_region_valid.sum(dim=1, keepdim=True).to(entropy.dtype)
    entropy = torch.where(
        support_count > 1.0,
        entropy / torch.log(support_count.clamp_min(2.0)),
        torch.zeros_like(entropy),
    )
    entropy = torch.where(row_valid, entropy, torch.zeros_like(entropy))
    coordinate = (
        torch.arange(int(region_grid_size), device=query.device, dtype=torch.float32) + 0.5
    ) / float(region_grid_size) * 2.0 - 1.0
    rows, columns = torch.meshgrid(coordinate, coordinate, indexing="ij")
    xy = torch.stack((columns, rows), dim=-1).reshape(region_count, 2)
    expected = torch.matmul(attention, xy)
    delta = expected - xy[None]
    output = torch.cat(
        (
            attention.reshape(batch, region_count * region_count),
            same_region,
            best,
            best - second,
            entropy,
            delta[:, :, 0],
            delta[:, :, 1],
        ),
        dim=1,
    )
    expected_width = len(
        absolute_phase_visual_feature_names(
            "profile", region_grid_size=int(region_grid_size)
        )
    )
    if output.shape != (batch, expected_width) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("absolute-phase visual transport output is invalid")
    return output


def batched_absolute_phase_position_control_features(
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    region_grid_size: int = ABSOLUTE_PHASE_REGION_GRID_SIZE,
) -> torch.Tensor:
    """Return only anchor-mask geometry for the matched negative control."""

    q_mask = torch.as_tensor(query_valid, dtype=torch.bool)
    s_mask = torch.as_tensor(support_valid, dtype=torch.bool, device=q_mask.device)
    if q_mask.shape != s_mask.shape or q_mask.ndim != 3:
        raise ValueError("absolute-phase position controls are incompatible")
    batch, grid_size, _ = q_mask.shape
    zeros = torch.zeros((batch, grid_size, grid_size, 1), device=q_mask.device)
    _query, _query_valid, query_coverage = _pooled_regions(
        grid=zeros, valid=q_mask, region_grid_size=region_grid_size
    )
    _support, _support_valid, support_coverage = _pooled_regions(
        grid=zeros, valid=s_mask, region_grid_size=region_grid_size
    )
    output = torch.cat((query_coverage, support_coverage), dim=1)
    expected_width = len(
        absolute_phase_position_feature_names(
            "profile", region_grid_size=int(region_grid_size)
        )
    )
    if output.shape != (batch, expected_width) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("absolute-phase position-control output is invalid")
    return output


def profile_feature_slices(
    profiles: Sequence[AbsolutePhaseProfile] = ABSOLUTE_PHASE_VISUAL_PROFILES,
) -> dict[str, slice]:
    """Return fixed visual and control column slices for all source profiles."""

    source = tuple(profiles)
    if source != ABSOLUTE_PHASE_VISUAL_PROFILES:
        raise ValueError("absolute-phase profile slices require the fixed profile set")
    offset = 0
    result: dict[str, slice] = {}
    for profile in source:
        width = len(ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE[profile.name])
        result[f"{profile.name}:visual"] = slice(offset, offset + width)
        offset += width
    for profile in source:
        width = len(ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE[profile.name])
        result[f"{profile.name}:position"] = slice(offset, offset + width)
        offset += width
    if offset != len(FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES):
        raise RuntimeError("absolute-phase feature slice layout drifted")
    return result
