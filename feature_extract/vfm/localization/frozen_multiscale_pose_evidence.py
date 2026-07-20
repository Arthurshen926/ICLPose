"""Target-free primitives for fixed-candidate multiscale pose evidence.

The S1 probe keeps three probability layers separate:

* each support observation owns a normalized local image-position map;
* support views are marginalized without averaging their descriptors; and
* the frozen candidate posterior is mixed with its explicit null state.

The module intentionally contains no target, pose-error, or candidate-label
loading.  A caller may project a frozen hypothesis into these fixed maps, but
it must not alter maps, candidate identities, support views, or denominators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ContextPositionLikelihoodMaps,
)


FROZEN_MULTISCALE_POSE_EVIDENCE_VERSION = (
    "fixed_candidate_per_view_local_context_ratio_v1"
)


@dataclass(frozen=True)
class LocalContextModeRatios:
    """One discrete normalized spatial posterior for every fixed support view.

    ``log_ratios`` are log likelihood ratios against the uniform distribution
    over the *same* local valid cells.  Invalid/missing support cells are
    represented separately by ``valid_cells`` rather than a fabricated score.
    """

    log_ratios: torch.Tensor
    valid_cells: torch.Tensor
    grid_rows: torch.Tensor
    grid_columns: torch.Tensor
    template_usable: torch.Tensor

    def __post_init__(self) -> None:
        ratios = torch.as_tensor(self.log_ratios)
        valid = torch.as_tensor(self.valid_cells, dtype=torch.bool, device=ratios.device)
        rows = torch.as_tensor(self.grid_rows, dtype=torch.long, device=ratios.device)
        columns = torch.as_tensor(
            self.grid_columns, dtype=torch.long, device=ratios.device
        )
        usable = torch.as_tensor(
            self.template_usable, dtype=torch.bool, device=ratios.device
        )
        if (
            ratios.ndim != 2
            or valid.shape != ratios.shape
            or rows.shape != ratios.shape
            or columns.shape != ratios.shape
            or usable.shape != (ratios.shape[0],)
            or not bool(torch.isfinite(ratios).all())
            or torch.any(valid & ~usable[:, None])
            or torch.any(valid & ((rows < 0) | (columns < 0)))
        ):
            raise ValueError("local context mode-ratio tensors are invalid")
        object.__setattr__(self, "log_ratios", ratios)
        object.__setattr__(self, "valid_cells", valid)
        object.__setattr__(self, "grid_rows", rows)
        object.__setattr__(self, "grid_columns", columns)
        object.__setattr__(self, "template_usable", usable)


@dataclass(frozen=True)
class DenseLocalContextModeMaps:
    """Dense grid form of local modes, materialized once per profile/query."""

    log_ratios: torch.Tensor
    valid_cells: torch.Tensor
    template_usable: torch.Tensor

    def __post_init__(self) -> None:
        ratios = torch.as_tensor(self.log_ratios)
        valid = torch.as_tensor(self.valid_cells, dtype=torch.bool, device=ratios.device)
        usable = torch.as_tensor(
            self.template_usable, dtype=torch.bool, device=ratios.device
        )
        if (
            ratios.ndim != 3
            or valid.shape != ratios.shape
            or usable.shape != (ratios.shape[0],)
            or not bool(torch.isfinite(ratios).all())
            or torch.any(valid & ~usable[:, None, None])
        ):
            raise ValueError("dense local context-mode maps are invalid")
        object.__setattr__(self, "log_ratios", ratios)
        object.__setattr__(self, "valid_cells", valid)
        object.__setattr__(self, "template_usable", usable)


def image_xy_to_grid_indices_torch(
    xy: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map image pixels to the cache's stable floor-based grid cell contract."""

    coordinates = torch.as_tensor(xy)
    width = int(image_width)
    height = int(image_height)
    size = int(grid_size)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or width <= 1
        or height <= 1
        or size <= 0
        or not bool(torch.isfinite(coordinates).all())
    ):
        raise ValueError("image-to-grid coordinates are invalid")
    columns = torch.floor(coordinates[:, 0] / float(width) * float(size)).to(torch.long)
    rows = torch.floor(coordinates[:, 1] / float(height) * float(size)).to(torch.long)
    return rows.clamp(0, size - 1), columns.clamp(0, size - 1)


def local_context_mode_ratios(
    *,
    maps: ContextPositionLikelihoodMaps,
    anchor_xy: torch.Tensor,
    image_width: int,
    image_height: int,
    local_window_size: int,
) -> LocalContextModeRatios:
    """Restrict full-image NCC logits to a fixed anchor-centred local grid.

    The returned map is normalized only over its own valid local cells.  That
    yields a proper discrete spatial posterior and a grid-uniform likelihood
    ratio without filtering, clipping, or a pose-dependent fallback.
    """

    logits = torch.as_tensor(maps.logits)
    valid_maps = torch.as_tensor(
        maps.valid_cells, dtype=torch.bool, device=logits.device
    )
    usable = torch.as_tensor(
        maps.template_usable, dtype=torch.bool, device=logits.device
    )
    anchors = torch.as_tensor(anchor_xy, dtype=logits.dtype, device=logits.device)
    window = int(local_window_size)
    if (
        logits.ndim != 3
        or valid_maps.shape != logits.shape
        or usable.shape != (logits.shape[0],)
        or anchors.shape != (logits.shape[0], 2)
        or window <= 0
        or window % 2 != 1
        or window > min(int(logits.shape[1]), int(logits.shape[2]))
    ):
        raise ValueError("local context-mode inputs are incompatible")
    if not bool(torch.isfinite(anchors).all()):
        raise ValueError("local context anchors must be finite")

    height, width = int(logits.shape[1]), int(logits.shape[2])
    center_rows, center_columns = image_xy_to_grid_indices_torch(
        anchors,
        image_width=int(image_width),
        image_height=int(image_height),
        grid_size=height,
    )
    if height != width:
        raise ValueError("local context maps must have square descriptor grids")
    radius = window // 2
    offsets = torch.arange(-radius, radius + 1, device=logits.device)
    row_offsets, column_offsets = torch.meshgrid(offsets, offsets, indexing="ij")
    raw_rows = center_rows[:, None, None] + row_offsets[None]
    raw_columns = center_columns[:, None, None] + column_offsets[None]
    in_bounds = (
        (raw_rows >= 0)
        & (raw_rows < height)
        & (raw_columns >= 0)
        & (raw_columns < width)
    )
    safe_rows = raw_rows.clamp(0, height - 1)
    safe_columns = raw_columns.clamp(0, width - 1)
    template_rows = torch.arange(len(logits), device=logits.device)[:, None, None]
    selected_logits = logits[template_rows, safe_rows, safe_columns].reshape(len(logits), -1)
    selected_valid = (
        in_bounds
        & valid_maps[template_rows, safe_rows, safe_columns]
        & usable[:, None, None]
    ).reshape(len(logits), -1)
    masked_logits = torch.where(
        selected_valid,
        selected_logits,
        torch.full_like(selected_logits, -torch.inf),
    )
    normalizer = torch.logsumexp(masked_logits, dim=1)
    counts = selected_valid.sum(dim=1).to(dtype=logits.dtype)
    has_cells = counts > 0.0
    normalizer = torch.where(has_cells, normalizer, torch.zeros_like(normalizer))
    log_ratios = selected_logits - normalizer[:, None] + torch.log(
        counts.clamp_min(1.0)
    )[:, None]
    log_ratios = torch.where(selected_valid, log_ratios, torch.zeros_like(log_ratios))
    rows = torch.where(
        selected_valid,
        raw_rows.reshape(len(logits), -1),
        torch.full_like(raw_rows.reshape(len(logits), -1), -1),
    )
    columns = torch.where(
        selected_valid,
        raw_columns.reshape(len(logits), -1),
        torch.full_like(raw_columns.reshape(len(logits), -1), -1),
    )
    return LocalContextModeRatios(
        log_ratios=log_ratios,
        valid_cells=selected_valid,
        grid_rows=rows,
        grid_columns=columns,
        template_usable=usable & has_cells,
    )


def sample_local_context_mode_ratios(
    *,
    modes: LocalContextModeRatios,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    image_width: int,
    image_height: int,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample per-view local maps for a batch of projected hypotheses.

    Returns ``(log_ratio, visual_available, geometric_in_window)``.  A support
    template unavailable because of crop/image coverage is neutral unknown;
    an otherwise usable template whose projected landmark is out of image or
    out of its declared local search window is a geometric zero-likelihood.
    """

    ratios = torch.as_tensor(modes.log_ratios)
    projected = torch.as_tensor(
        projected_xy, dtype=ratios.dtype, device=ratios.device
    )
    visible = torch.as_tensor(
        projection_valid, dtype=torch.bool, device=ratios.device
    )
    if (
        projected.ndim != 3
        or projected.shape[2] != 2
        or visible.shape != projected.shape[:2]
        or ratios.ndim != 2
        or projected.shape[1] != ratios.shape[0]
        or int(grid_size) <= 0
    ):
        raise ValueError("local context-mode sampling inputs are incompatible")
    if not bool(torch.isfinite(torch.nan_to_num(projected)).all()):
        raise ValueError("local context projections are invalid")
    dense = materialize_dense_local_context_mode_maps(modes=modes, grid_size=int(grid_size))
    return sample_dense_local_context_mode_ratios(
        maps=dense,
        projected_xy=projected,
        projection_valid=visible,
        image_width=int(image_width),
        image_height=int(image_height),
    )


def materialize_dense_local_context_mode_maps(
    *, modes: LocalContextModeRatios, grid_size: int
) -> DenseLocalContextModeMaps:
    """Scatter sparse local cells once, avoiding a batch×mode comparison."""

    ratios = torch.as_tensor(modes.log_ratios)
    size = int(grid_size)
    if size <= 0:
        raise ValueError("dense local context grid size must be positive")
    template_count, mode_count = ratios.shape
    valid_modes = torch.as_tensor(modes.valid_cells, dtype=torch.bool, device=ratios.device)
    if torch.any(
        valid_modes
        & (
            (modes.grid_rows < 0)
            | (modes.grid_rows >= size)
            | (modes.grid_columns < 0)
            | (modes.grid_columns >= size)
        )
    ):
        raise ValueError("local context modes exceed the declared descriptor grid")
    dense_values = torch.zeros(
        (template_count, size, size),
        dtype=ratios.dtype,
        device=ratios.device,
    )
    dense_valid = torch.zeros(
        (template_count, size, size),
        dtype=torch.bool,
        device=ratios.device,
    )
    template_rows = torch.arange(template_count, device=ratios.device)[:, None].expand(
        -1, mode_count
    )
    flat_indices = (
        template_rows[valid_modes] * size * size
        + modes.grid_rows[valid_modes] * size
        + modes.grid_columns[valid_modes]
    )
    if int(torch.unique(flat_indices).numel()) != int(flat_indices.numel()):
        raise RuntimeError("a local context mode repeats a descriptor-grid cell")
    dense_values.reshape(-1)[flat_indices] = ratios[valid_modes]
    dense_valid.reshape(-1)[flat_indices] = True
    return DenseLocalContextModeMaps(
        log_ratios=dense_values,
        valid_cells=dense_valid,
        template_usable=modes.template_usable,
    )


def sample_dense_local_context_mode_ratios(
    *,
    maps: DenseLocalContextModeMaps,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-cell sample a pre-materialized local discrete posterior."""

    ratios = torch.as_tensor(maps.log_ratios)
    projected = torch.as_tensor(
        projected_xy, dtype=ratios.dtype, device=ratios.device
    )
    visible = torch.as_tensor(
        projection_valid, dtype=torch.bool, device=ratios.device
    )
    if (
        projected.ndim != 3
        or projected.shape[2] != 2
        or visible.shape != projected.shape[:2]
        or projected.shape[1] != ratios.shape[0]
        or ratios.shape[1] != ratios.shape[2]
    ):
        raise ValueError("dense local context-mode sampling inputs are incompatible")
    safe_projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    rows, columns = image_xy_to_grid_indices_torch(
        safe_projected.reshape(-1, 2),
        image_width=int(image_width),
        image_height=int(image_height),
        grid_size=int(ratios.shape[1]),
    )
    rows = rows.reshape(projected.shape[:2])
    columns = columns.reshape(projected.shape[:2])
    templates = torch.arange(ratios.shape[0], device=ratios.device)[None].expand(
        projected.shape[0], -1
    )
    values = ratios[templates, rows, columns]
    local_hit = maps.valid_cells[templates, rows, columns]
    template_usable = maps.template_usable[None]
    geometric_in_window = visible & template_usable & local_hit
    visual_available = template_usable.expand_as(visible)
    return torch.where(geometric_in_window, values, torch.zeros_like(values)), visual_available, geometric_in_window


def fixed_candidate_point_log_ratios(
    *,
    view_log_ratios: torch.Tensor,
    view_available: torch.Tensor,
    view_geometric_in_window: torch.Tensor,
    candidate_view_weights: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Marginalize per-view evidence then mix immutable candidate/null mass.

    ``view_available=False`` is an unknown observation and contributes ratio
    one.  If a view is usable but a visible pose projects outside its declared
    local likelihood support, it contributes ratio zero.  No branch is allowed
    to substitute a query-centred geometric Gaussian.
    """

    logs = torch.as_tensor(view_log_ratios)
    available = torch.as_tensor(view_available, dtype=torch.bool, device=logs.device)
    in_window = torch.as_tensor(
        view_geometric_in_window, dtype=torch.bool, device=logs.device
    )
    weights = torch.as_tensor(candidate_view_weights, dtype=logs.dtype, device=logs.device)
    probabilities = torch.as_tensor(
        candidate_probabilities, dtype=logs.dtype, device=logs.device
    )
    null = torch.as_tensor(null_probabilities, dtype=logs.dtype, device=logs.device)
    if (
        logs.ndim != 4
        or available.shape != logs.shape
        or in_window.shape != logs.shape
        or weights.shape != logs.shape[1:]
        or probabilities.shape != logs.shape[1:3]
        or null.shape != (logs.shape[1],)
        or torch.any(weights < 0.0)
        or torch.any(probabilities < 0.0)
        or torch.any(null < 0.0)
        or not bool(torch.isfinite(logs).all())
        or not bool(torch.isfinite(weights).all())
        or not bool(torch.isfinite(probabilities).all())
        or not bool(torch.isfinite(null).all())
    ):
        raise ValueError("fixed candidate view-mixture tensors are invalid")
    if torch.any(torch.abs(weights.sum(dim=2) - (probabilities > 0.0).to(weights.dtype)) > 2e-5):
        raise ValueError("candidate support-view weights do not preserve one-view mass")
    if torch.any(torch.abs(probabilities.sum(dim=1) + null - 1.0) > 2e-5):
        raise ValueError("candidate/null posterior mass is not conserved")

    # Missing visual templates are unknown.  A usable template that cannot
    # explain this pose's projection has exactly zero spatial likelihood.
    view_ratios = torch.where(
        available,
        torch.where(in_window, torch.exp(logs), torch.zeros_like(logs)),
        torch.ones_like(logs),
    )
    candidate_ratios = torch.sum(view_ratios * weights[None], dim=3)
    point_ratios = null[None] + torch.sum(
        candidate_ratios * probabilities[None], dim=2
    )
    point_logs = torch.log(point_ratios.clamp_min(torch.finfo(logs.dtype).tiny))
    contributed = torch.sum(
        (available & in_window).to(dtype=logs.dtype) * weights[None], dim=3
    )
    return point_logs, candidate_ratios, contributed
