"""Frozen multi-scale RGB phase evidence for candidate-specific pose scoring.

The local FPN cost volume already provides sub-pixel measurement evidence, but
its small physical template can be periodic on repeated facades.  This module
keeps the same target-free query/support contract while evaluating a second,
wider physical context at a coarser sampling rate.  It deliberately has no
learned scalar head, candidate prior, pose, target, residual, track ID, or
rank input:

``real query/support RGB -> FPN -> per-view normalized spatial density``.

Only a caller *after the visual forward* may evaluate the density at a
train-only or runtime pose projection.  Missing or out-of-window projections
remain a fixed neutral likelihood; they must never be converted into a learned
dustbin reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
    local_offset_grid,
    template_search_cost_volume_logits,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    bounded_log_likelihood_ratio,
)


CANDIDATE_MULTISCALE_RGB_PHASE_FORMAT = "candidate_multiscale_rgb_phase_density_v2"


@dataclass(frozen=True)
class RGBPhaseScale:
    """One fixed physical search/template geometry in aligned image pixels."""

    name: str
    search_radius_px: float
    context_radius_px: float
    step_px: float

    def __post_init__(self) -> None:
        name = str(self.name)
        search = float(self.search_radius_px)
        context = float(self.context_radius_px)
        step = float(self.step_px)
        if (
            not name
            or not all(math.isfinite(value) for value in (search, context, step))
            or search < step
            or context <= 0.0
            or step <= 0.0
        ):
            raise ValueError("RGB phase scale is invalid")
        for radius in (search, context, search + context):
            steps = radius / step
            if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("RGB phase scale radii must lie on the sampling grid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "search_radius_px", search)
        object.__setattr__(self, "context_radius_px", context)
        object.__setattr__(self, "step_px", step)

    @property
    def patch_side(self) -> int:
        return int(round(2.0 * (self.search_radius_px + self.context_radius_px) / self.step_px)) + 1

    @property
    def offset_count(self) -> int:
        side = int(round(2.0 * self.search_radius_px / self.step_px)) + 1
        return side * side

    def offsets_xy(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return local_offset_grid(
            search_radius_px=float(self.search_radius_px),
            step_px=float(self.step_px),
            device=device,
            dtype=torch.float32,
        )


DEFAULT_RGB_PHASE_SCALES: tuple[RGBPhaseScale, ...] = (
    # Exact local measurement geometry of the gate-checked RGB FPN branch.
    RGBPhaseScale("local_8px", search_radius_px=8.0, context_radius_px=12.0, step_px=0.5),
    # Same target-free center and per-view support observation, but 3.25x
    # wider physical template.  The coarser grid keeps cost-volume work bounded
    # while retaining a 65x65 FPN template for absolute facade phase.
    RGBPhaseScale("wide_phase_32px", search_radius_px=8.0, context_radius_px=32.0, step_px=1.0),
)


def resolve_rgb_phase_scales(
    scales: Sequence[RGBPhaseScale] | None = None,
) -> tuple[RGBPhaseScale, ...]:
    values = DEFAULT_RGB_PHASE_SCALES if scales is None else tuple(scales)
    if not values or any(not isinstance(value, RGBPhaseScale) for value in values):
        raise ValueError("RGB phase scales are invalid")
    names = [value.name for value in values]
    if len(set(names)) != len(names):
        raise ValueError("RGB phase scale names must be unique")
    return tuple(values)


@dataclass(frozen=True)
class RGBPhaseDensity:
    """One target-free per-edge normalized spatial density grid."""

    scale: RGBPhaseScale
    log_probabilities: torch.Tensor
    edge_usable: torch.Tensor

    def __post_init__(self) -> None:
        log_probabilities = torch.as_tensor(self.log_probabilities, dtype=torch.float32)
        usable = torch.as_tensor(
            self.edge_usable, dtype=torch.bool, device=log_probabilities.device
        )
        if (
            not isinstance(self.scale, RGBPhaseScale)
            or log_probabilities.ndim != 4
            or log_probabilities.shape[:3] != usable.shape
            or log_probabilities.shape[0] == 0
            or log_probabilities.shape[1] == 0
            or log_probabilities.shape[2] == 0
            or log_probabilities.shape[3] != self.scale.offset_count
            or not torch.isfinite(log_probabilities).all()
        ):
            raise ValueError("RGB phase density is invalid")
        normalizer = torch.logsumexp(log_probabilities, dim=-1)
        if torch.max(torch.abs(normalizer)).item() > 2e-4:
            raise ValueError("RGB phase density is not normalized")
        object.__setattr__(self, "log_probabilities", log_probabilities)
        object.__setattr__(self, "edge_usable", usable)

    @property
    def point_count(self) -> int:
        return int(self.log_probabilities.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.log_probabilities.shape[1])

    @property
    def support_view_count(self) -> int:
        return int(self.log_probabilities.shape[2])

    def to(self, device: torch.device | str) -> "RGBPhaseDensity":
        return RGBPhaseDensity(
            scale=self.scale,
            log_probabilities=self.log_probabilities.to(device),
            edge_usable=self.edge_usable.to(device),
        )


def _validate_patch_inputs(
    *,
    scale: RGBPhaseScale,
    query_patches: torch.Tensor,
    support_patches: torch.Tensor,
    edge_usable: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.as_tensor(query_patches, dtype=torch.float32)
    support = torch.as_tensor(support_patches, dtype=torch.float32, device=query.device)
    usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=query.device)
    side = int(scale.patch_side)
    if (
        query.ndim != 4
        or query.shape[1:] != (3, side, side)
        or support.ndim != 6
        or support.shape[:1] != query.shape[:1]
        or support.shape[3:] != (3, side, side)
        or usable.shape != support.shape[:3]
        or not torch.isfinite(query).all()
        or not torch.isfinite(support).all()
    ):
        raise ValueError("RGB phase patch inputs are invalid")
    return query, support, usable


@torch.inference_mode()
def extract_rgb_phase_density(
    *,
    texture_encoder: TexturePatchEncoder,
    scale: RGBPhaseScale,
    query_patches: torch.Tensor,
    support_patches: torch.Tensor,
    edge_usable: torch.Tensor,
    edge_chunk_size: int,
    temperature: float = 10.0,
    amp_enabled: bool = False,
) -> RGBPhaseDensity:
    """Encode real RGB once and return target-free per-view density grids.

    The function has no projection or supervision arguments.  A caller must
    sample this density only after visual inference is complete.
    """

    if not isinstance(texture_encoder, TexturePatchEncoder):
        raise TypeError("RGB phase extraction requires a TexturePatchEncoder")
    chunk = int(edge_chunk_size)
    heat = float(temperature)
    if chunk <= 0 or not math.isfinite(heat) or heat < 1.0:
        raise ValueError("RGB phase extraction configuration is invalid")
    query, support, usable = _validate_patch_inputs(
        scale=scale,
        query_patches=query_patches,
        support_patches=support_patches,
        edge_usable=edge_usable,
    )
    device = query.device
    if next(texture_encoder.parameters()).device != device:
        raise ValueError("RGB phase encoder and patches are on different devices")
    point_count, candidate_count, view_count = usable.shape
    flat_support = support.reshape(
        point_count * candidate_count * view_count, *support.shape[3:]
    )
    owner = torch.arange(point_count, device=device).repeat_interleave(
        candidate_count * view_count
    )
    autocast = (
        torch.cuda.amp.autocast(enabled=True)
        if bool(amp_enabled) and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with autocast:
        query_features = texture_encoder(query)
    chunks: list[torch.Tensor] = []
    expected_offsets = scale.offsets_xy(device=device)
    for start in range(0, len(flat_support), chunk):
        stop = min(len(flat_support), start + chunk)
        with autocast:
            support_features = texture_encoder(flat_support[start:stop])
            logits, offsets = template_search_cost_volume_logits(
                query_features.index_select(0, owner[start:stop]),
                support_features,
                search_radius_px=float(scale.search_radius_px),
                context_radius_px=float(scale.context_radius_px),
                step_px=float(scale.step_px),
                temperature=heat,
            )
        offsets = offsets.to(device=device, dtype=torch.float32)
        if not torch.allclose(offsets, expected_offsets, atol=1e-5, rtol=1e-5):
            raise RuntimeError("RGB phase cost volume emitted an unexpected offset grid")
        chunks.append(F.log_softmax(logits.float(), dim=-1))
    log_probabilities = torch.cat(chunks, dim=0).reshape(
        point_count, candidate_count, view_count, scale.offset_count
    )
    return RGBPhaseDensity(
        scale=scale,
        log_probabilities=log_probabilities,
        edge_usable=usable,
    )


def extract_multiscale_rgb_phase_densities(
    *,
    texture_encoder: TexturePatchEncoder,
    patches_by_scale: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    scales: Sequence[RGBPhaseScale] | None = None,
    edge_chunk_size: int,
    temperature: float = 10.0,
    amp_enabled: bool = False,
) -> dict[str, RGBPhaseDensity]:
    """Extract several fixed physical contexts without mixing their scores."""

    resolved = resolve_rgb_phase_scales(scales)
    if set(patches_by_scale) != {scale.name for scale in resolved}:
        raise ValueError("RGB phase patch scale set is incomplete")
    output: dict[str, RGBPhaseDensity] = {}
    for scale in resolved:
        query, support, usable = patches_by_scale[scale.name]
        output[scale.name] = extract_rgb_phase_density(
            texture_encoder=texture_encoder,
            scale=scale,
            query_patches=query,
            support_patches=support,
            edge_usable=usable,
            edge_chunk_size=int(edge_chunk_size),
            temperature=float(temperature),
            amp_enabled=bool(amp_enabled),
        )
    return output


def _offset_grid_geometry(scale: RGBPhaseScale) -> tuple[int, float, float, float, float, float]:
    offsets = scale.offsets_xy()
    side = int(round(math.sqrt(len(offsets))))
    if side * side != len(offsets):
        raise RuntimeError("RGB phase offset grid is not square")
    x_values = offsets[:, 0].reshape(side, side)
    y_values = offsets[:, 1].reshape(side, side)
    if (
        not torch.allclose(x_values, x_values[0:1].expand_as(x_values))
        or not torch.allclose(y_values, y_values[:, 0:1].expand_as(y_values))
    ):
        raise RuntimeError("RGB phase offset grid ordering is invalid")
    return (
        side,
        float(x_values[0, 0].item()),
        float(y_values[0, 0].item()),
        float(x_values[0, -1].item()),
        float(y_values[-1, 0].item()),
        float(scale.step_px),
    )


def _continuous_selected_log_probability(
    *, density: RGBPhaseDensity, points: torch.Tensor, candidates: torch.Tensor, offsets_xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample probability, then return its logarithm and bounds."""

    values = density.log_probabilities[points, candidates]
    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32, device=values.device)
    if values.ndim != 3 or offsets.shape != (len(values), 2) or not torch.isfinite(offsets).all():
        raise ValueError("RGB phase continuous sampling inputs are invalid")
    side, min_x, min_y, max_x, max_y, step = _offset_grid_geometry(density.scale)
    epsilon = max(1e-5, step * 1e-5)
    in_window = (
        (offsets[:, 0] >= min_x - epsilon)
        & (offsets[:, 0] <= max_x + epsilon)
        & (offsets[:, 1] >= min_y - epsilon)
        & (offsets[:, 1] <= max_y + epsilon)
    )
    column = ((offsets[:, 0] - min_x) / step).clamp(0.0, float(side - 1))
    row = ((offsets[:, 1] - min_y) / step).clamp(0.0, float(side - 1))
    col0 = torch.floor(column).to(torch.long)
    row0 = torch.floor(row).to(torch.long)
    col1 = (col0 + 1).clamp_max(side - 1)
    row1 = (row0 + 1).clamp_max(side - 1)
    fx = torch.where(col0 == col1, torch.zeros_like(column), column - col0)
    fy = torch.where(row0 == row1, torch.zeros_like(row), row - row0)
    probabilities = values.exp().reshape(len(values), values.shape[1], side, side)
    batch = torch.arange(len(values), device=values.device)
    p00 = probabilities[batch, :, row0, col0]
    p10 = probabilities[batch, :, row0, col1]
    p01 = probabilities[batch, :, row1, col0]
    p11 = probabilities[batch, :, row1, col1]
    probability = (
        (1.0 - fx)[:, None] * (1.0 - fy)[:, None] * p00
        + fx[:, None] * (1.0 - fy)[:, None] * p10
        + (1.0 - fx)[:, None] * fy[:, None] * p01
        + fx[:, None] * fy[:, None] * p11
    )
    return torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny)), in_window


def _selected_candidate_phase_inputs(
    *,
    density: RGBPhaseDensity,
    candidate_view_weights: torch.Tensor,
    point_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    offsets_xy: torch.Tensor,
    projection_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate one selected-candidate request and return fixed view mass."""

    active = density
    weights = torch.as_tensor(
        candidate_view_weights,
        dtype=torch.float32,
        device=active.log_probabilities.device,
    )
    points = torch.as_tensor(point_indices, dtype=torch.long, device=weights.device).reshape(-1)
    candidates = torch.as_tensor(
        candidate_indices, dtype=torch.long, device=weights.device
    ).reshape(-1)
    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32, device=weights.device)
    if projection_valid is None:
        valid = torch.ones((len(points),), dtype=torch.bool, device=weights.device)
    else:
        valid = torch.as_tensor(
            projection_valid, dtype=torch.bool, device=weights.device
        ).reshape(-1)
    if (
        weights.shape
        != (active.point_count, active.candidate_count, active.support_view_count)
        or points.shape != candidates.shape
        or offsets.shape != (len(points), 2)
        or valid.shape != points.shape
        or len(points) == 0
        or torch.any(points < 0)
        or torch.any(points >= active.point_count)
        or torch.any(candidates < 0)
        or torch.any(candidates >= active.candidate_count)
        or torch.any(weights < 0.0)
        or not torch.isfinite(weights).all()
    ):
        raise ValueError("RGB phase selected candidate inputs are invalid")
    selected_weights = weights[points, candidates]
    if torch.any(torch.abs(selected_weights.sum(dim=1) - 1.0) > 1e-4):
        raise ValueError("RGB phase selected candidate view weights must sum to one")
    return points, candidates, offsets, valid, selected_weights


def _selected_centered_log_probability(
    *,
    density: RGBPhaseDensity,
    points: torch.Tensor,
    candidates: torch.Tensor,
    offsets_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate conditional spatial evidence relative to its fixed uniform prior."""

    local, in_window = _continuous_selected_log_probability(
        density=density,
        points=points,
        candidates=candidates,
        offsets_xy=offsets_xy,
    )
    return local + math.log(float(density.scale.offset_count)), in_window


def _fixed_view_mixture_score(
    *,
    edge_log_likelihood_ratios: torch.Tensor,
    edge_valid: torch.Tensor,
    selected_weights: torch.Tensor,
    missing_edge_log_likelihood_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize fixed support-view mass without reassigning missing mass."""

    values = torch.as_tensor(edge_log_likelihood_ratios, dtype=torch.float32)
    usable = torch.as_tensor(edge_valid, dtype=torch.bool, device=values.device)
    weights = torch.as_tensor(selected_weights, dtype=torch.float32, device=values.device)
    missing = float(missing_edge_log_likelihood_ratio)
    if (
        values.ndim != 2
        or usable.shape != values.shape
        or weights.shape != values.shape
        or not math.isfinite(missing)
        or not torch.isfinite(values).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
    ):
        raise ValueError("RGB phase fixed view mixture inputs are invalid")
    log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    effective = torch.where(usable, values, torch.full_like(values, missing))
    score = torch.logsumexp(log_weights + effective, dim=1)
    candidate_usable = torch.any(usable & (weights > 0.0), dim=1)
    return score, candidate_usable


def selected_candidate_rgb_phase_log_likelihood_ratio(
    *,
    density: RGBPhaseDensity,
    candidate_view_weights: torch.Tensor,
    point_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    offsets_xy: torch.Tensor,
    projection_valid: torch.Tensor | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one selected candidate with fixed per-view mixture mass.

    The returned value is centered at the uniform spatial density, making a
    missing support edge exactly neutral.  Crucially, fixed mass for missing
    views is *not* reassigned to a surviving support view.

    ``RGBPhaseDensity`` stores the conditional finite-offset distribution so
    this branch has no learned dustbin path.  To preserve the gate-checked
    local RGB-only scorer semantics, its fixed non-dustbin logit is zero: the
    finite local support therefore has one half of the joint mass.
    Subtracting ``-log(2K)`` is consequently equivalent to
    ``log_softmax(offset_logits) + log(K)``.  Apply the same bounded LLR
    before fixed-view marginalization, so ``local_8px`` is numerically
    equivalent to the original raw RGB cost-volume component instead of
    silently changing its evidence scale while adding the wide phase probe.
    """

    missing = float(missing_edge_log_likelihood_ratio)
    cap = float(max_abs_log_likelihood_ratio)
    if not math.isfinite(missing) or not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("RGB phase missing-edge likelihood is invalid")
    points, candidates, offsets, valid, selected_weights = _selected_candidate_phase_inputs(
        density=density,
        candidate_view_weights=candidate_view_weights,
        point_indices=point_indices,
        candidate_indices=candidate_indices,
        offsets_xy=offsets_xy,
        projection_valid=projection_valid,
    )
    centered_raw, in_window = _selected_centered_log_probability(
        density=density,
        points=points,
        candidates=candidates,
        offsets_xy=offsets,
    )
    edge_valid = valid[:, None] & in_window[:, None] & density.edge_usable[points, candidates]
    centered = bounded_log_likelihood_ratio(
        centered_raw,
        max_abs_log_ratio=cap,
    )
    return _fixed_view_mixture_score(
        edge_log_likelihood_ratios=centered,
        edge_valid=edge_valid,
        selected_weights=selected_weights,
        missing_edge_log_likelihood_ratio=missing,
    )


def selected_candidate_rgb_phase_interaction_log_likelihood_ratio(
    *,
    pair_density: RGBPhaseDensity,
    query_zero_support_density: RGBPhaseDensity,
    query_support_zero_density: RGBPhaseDensity,
    zero_density: RGBPhaseDensity,
    candidate_view_weights: torch.Tensor,
    point_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    offsets_xy: torch.Tensor,
    projection_valid: torch.Tensor | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score only RGB information jointly dependent on query and support.

    This frozen PMI-style residual removes cost-volume modes that can be
    generated from either appearance alone:

    ``pair - query_zero/support - query/support_zero + zero/zero``.

    It is an auditable log-likelihood-ratio feature, not a replacement for a
    normalized spatial density.  All four visual forwards remain target-free;
    pose projections are sampled only after they have been fixed.  Any absent
    edge or out-of-window projection receives the same fixed neutral value as
    the base scorer.
    """

    missing = float(missing_edge_log_likelihood_ratio)
    cap = float(max_abs_log_likelihood_ratio)
    if not math.isfinite(missing) or not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("RGB phase interaction constants are invalid")
    densities = (
        pair_density,
        query_zero_support_density,
        query_support_zero_density,
        zero_density,
    )
    reference = pair_density
    if any(
        density.scale != reference.scale
        or density.log_probabilities.shape != reference.log_probabilities.shape
        or density.log_probabilities.device != reference.log_probabilities.device
        for density in densities[1:]
    ):
        raise ValueError("RGB phase interaction densities are incompatible")
    points, candidates, offsets, valid, selected_weights = _selected_candidate_phase_inputs(
        density=reference,
        candidate_view_weights=candidate_view_weights,
        point_indices=point_indices,
        candidate_indices=candidate_indices,
        offsets_xy=offsets_xy,
        projection_valid=projection_valid,
    )
    centered: list[torch.Tensor] = []
    in_windows: list[torch.Tensor] = []
    for density in densities:
        value, in_window = _selected_centered_log_probability(
            density=density,
            points=points,
            candidates=candidates,
            offsets_xy=offsets,
        )
        centered.append(value)
        in_windows.append(in_window)
    raw_interaction = centered[0] - centered[1] - centered[2] + centered[3]
    edge_valid = valid[:, None]
    for density, in_window in zip(densities, in_windows):
        edge_valid = edge_valid & in_window[:, None] & density.edge_usable[points, candidates]
    bounded = bounded_log_likelihood_ratio(raw_interaction, max_abs_log_ratio=cap)
    return _fixed_view_mixture_score(
        edge_log_likelihood_ratios=bounded,
        edge_valid=edge_valid,
        selected_weights=selected_weights,
        missing_edge_log_likelihood_ratio=missing,
    )
