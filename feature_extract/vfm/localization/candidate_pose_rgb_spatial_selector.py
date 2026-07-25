"""Strict target-free point selection for candidate RGB spatial evidence.

The raw-RGB likelihood is evaluated on a fixed pool of P1 query anchors.  A
runtime selector may reduce that pool for throughput, but it must never use a
registered track, pose projection, residual, dustbin target, or any other
train-only field.  This module keeps that boundary explicit and provides
deterministic spatial-quota selectors for audits and inference.

This is deliberately a *point-budget* selector, not a candidate identity
head.  It decides which already-frozen query anchors receive pose scoring; it
does not alter the top-L candidate posterior or use a hypothesis-dependent
active-point normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialEdgePrediction,
    CandidatePoseRGBSpatialRuntime,
)


CANDIDATE_POSE_RGB_SPATIAL_SELECTOR_FORMAT = "candidate_pose_rgb_spatial_target_free_selector_v1"

TARGET_FREE_SELECTOR_POLICIES = (
    "uniform",
    "coarse_max_probability",
    "coarse_margin",
    "support_coverage",
    "rgb_peakiness",
    "rgb_center_probability",
    "coarse_margin_rgb_peakiness",
    "coarse_margin_support_coverage",
)


@dataclass(frozen=True)
class CandidatePoseRGBSpatialSelectorInput:
    """Target-free per-point fields sufficient for a fixed point budget.

    The type intentionally contains no candidate IDs, pose projections, or
    supervision.  Candidate probabilities are a frozen runtime prior and are
    used only to summarize uncertainty, never to choose a candidate identity.
    """

    source_point_ids: np.ndarray
    xy: np.ndarray
    point_sources: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    support_view_valid: np.ndarray
    support_view_weights: np.ndarray
    support_coverage_counts: np.ndarray
    support_view_rgb_usable: np.ndarray | None = None

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32)
        sources = np.asarray(self.point_sources).astype(str).reshape(-1)
        candidate = np.asarray(self.candidate_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        valid = np.asarray(self.support_view_valid, dtype=bool)
        weights = np.asarray(self.support_view_weights, dtype=np.float32)
        coverage = np.asarray(self.support_coverage_counts, dtype=np.int32)
        rgb_usable = (
            valid.copy()
            if self.support_view_rgb_usable is None
            else np.asarray(self.support_view_rgb_usable, dtype=bool)
        )
        count = len(source_ids)
        if (
            count == 0
            or len(np.unique(source_ids)) != count
            or xy.shape != (count, 2)
            or sources.shape != (count,)
            or np.any(sources == "")
            or candidate.ndim != 2
            or candidate.shape[0] != count
            or candidate.shape[1] == 0
            or null.shape != (count,)
            or valid.ndim != 3
            or valid.shape[:2] != candidate.shape
            or valid.shape[2] == 0
            or weights.shape != valid.shape
            or coverage.shape != valid.shape
            or rgb_usable.shape != valid.shape
            or not np.isfinite(xy).all()
            or not np.isfinite(candidate).all()
            or not np.isfinite(null).all()
            or not np.isfinite(weights).all()
            or np.any(candidate < 0.0)
            or np.any(null < 0.0)
            or np.any(weights < 0.0)
            or np.any(coverage < 0)
            or np.any(rgb_usable & ~valid)
        ):
            raise ValueError("target-free RGB spatial selector inputs are invalid")
        if np.any(np.abs(candidate.sum(axis=1) + null - 1.0) > 1e-4):
            raise ValueError("target-free RGB spatial selector prior mass is invalid")
        positive = candidate > 0.0
        view_mass = weights.sum(axis=2)
        if (
            np.any(np.abs(view_mass[positive] - 1.0) > 1e-4)
            or np.any(view_mass[~positive] > 1e-6)
            or np.any(positive & ~np.any(valid, axis=2))
            or np.any(weights[~valid] > 1e-6)
            or np.any(coverage[~valid] != 0)
        ):
            raise ValueError("target-free RGB spatial selector support fields are invalid")
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "point_sources", sources)
        object.__setattr__(self, "candidate_probabilities", candidate)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "support_view_valid", valid)
        object.__setattr__(self, "support_view_weights", weights)
        object.__setattr__(self, "support_coverage_counts", coverage)
        object.__setattr__(self, "support_view_rgb_usable", rgb_usable)

    @property
    def point_count(self) -> int:
        return int(len(self.source_point_ids))

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_probabilities.shape[1])


def selector_input_from_target_free_layout(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    rows: Sequence[int] | np.ndarray,
    rgb_context_radius_px: float | None = None,
    coordinate_image_size: tuple[int, int] | None = None,
) -> CandidatePoseRGBSpatialSelectorInput:
    """Extract a target-free selector input from an immutable RGB layout.

    When a context radius and coordinate image size are supplied, the input
    also marks support edges whose query or support crop would fall outside the
    real RGB image.  This is static availability, not a pose-dependent
    residual: it prevents the selector from spending its fixed point budget on
    support views that the downstream FPN must deterministically neutralize.
    """

    if not isinstance(layout, CandidatePoseRGBSpatialLayout):
        raise ValueError("target-free selector requires a candidate RGB spatial layout")
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    if (
        len(selected) == 0
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= layout.row_count)
    ):
        raise ValueError("target-free selector rows are invalid")
    if (rgb_context_radius_px is None) != (coordinate_image_size is None):
        raise ValueError("RGB crop availability requires both radius and image size")
    valid = np.asarray(layout.support_view_valid[selected], dtype=bool)
    if rgb_context_radius_px is None:
        rgb_usable = valid.copy()
    else:
        radius = float(rgb_context_radius_px)
        width, height = (int(value) for value in coordinate_image_size)
        if (
            not math.isfinite(radius)
            or radius < 0.0
            or width <= 1
            or height <= 1
            or radius > min(width - 1.0, height - 1.0) / 2.0
        ):
            raise ValueError("target-free selector RGB crop geometry is invalid")
        query_xy = np.asarray(layout.xy[selected], dtype=np.float32)
        support_xy = np.asarray(layout.support_xy[selected], dtype=np.float32)
        query_in_bounds = (
            (query_xy[:, 0] >= radius)
            & (query_xy[:, 0] <= float(width - 1) - radius)
            & (query_xy[:, 1] >= radius)
            & (query_xy[:, 1] <= float(height - 1) - radius)
        )
        support_in_bounds = (
            (support_xy[..., 0] >= radius)
            & (support_xy[..., 0] <= float(width - 1) - radius)
            & (support_xy[..., 1] >= radius)
            & (support_xy[..., 1] <= float(height - 1) - radius)
        )
        rgb_usable = valid & query_in_bounds[:, None, None] & support_in_bounds
    return CandidatePoseRGBSpatialSelectorInput(
        source_point_ids=np.asarray(layout.source_point_ids[selected], dtype=np.int64),
        xy=np.asarray(layout.xy[selected], dtype=np.float32),
        point_sources=np.asarray(layout.point_sources[selected]).astype(str),
        candidate_probabilities=np.asarray(
            layout.candidate_prior_probabilities[selected], dtype=np.float32
        ),
        null_probabilities=np.asarray(layout.null_probabilities[selected], dtype=np.float32),
        support_view_valid=valid,
        support_view_weights=np.asarray(layout.support_view_weights[selected], dtype=np.float32),
        support_coverage_counts=np.asarray(layout.support_coverage_counts[selected], dtype=np.int32),
        support_view_rgb_usable=rgb_usable,
    )


def _rank_unit_interval(values: np.ndarray, *, tie_breaker: np.ndarray) -> np.ndarray:
    """Return deterministic within-query ranks in ``[0, 1]``.

    A selector only consumes relative quality.  Ranking makes heterogeneous
    coarse and RGB quality scales composable without fitting a calibration on
    a target-bearing split.
    """

    score = np.asarray(values, dtype=np.float64).reshape(-1)
    ids = np.asarray(tie_breaker, dtype=np.int64).reshape(-1)
    if score.shape != ids.shape or len(score) == 0 or not np.isfinite(score).all():
        raise ValueError("target-free selector ranks are invalid")
    if len(score) == 1:
        return np.ones((1,), dtype=np.float32)
    order = np.lexsort((ids, score))
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(order), dtype=np.float64)
    return (rank / float(len(order) - 1)).astype(np.float32)


def target_free_layout_point_quality(
    selector_input: CandidatePoseRGBSpatialSelectorInput,
) -> Mapping[str, np.ndarray]:
    """Compute target-free coarse/support quality summaries per query point."""

    if not isinstance(selector_input, CandidatePoseRGBSpatialSelectorInput):
        raise ValueError("target-free selector quality requires selector inputs")
    candidate = selector_input.candidate_probabilities.astype(np.float64, copy=False)
    order = np.sort(candidate, axis=1)
    maximum = order[:, -1]
    margin = maximum if candidate.shape[1] == 1 else maximum - order[:, -2]
    positive = candidate > 0.0
    normalized = candidate / np.maximum(candidate.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(
        np.where(positive, normalized * np.log(np.maximum(normalized, 1e-12)), 0.0), axis=1
    )
    usable = selector_input.support_view_valid & selector_input.support_view_rgb_usable
    support_count = np.sum(usable, axis=2, dtype=np.float64)
    coverage = np.sum(
        selector_input.support_view_weights.astype(np.float64)
        * np.log1p(selector_input.support_coverage_counts.astype(np.float64)),
        where=usable,
        axis=2,
    )
    candidate_mass = np.maximum(candidate.sum(axis=1), 1e-12)
    weighted_coverage = np.sum(candidate * coverage, axis=1) / candidate_mass
    weighted_view_count = np.sum(candidate * support_count, axis=1) / candidate_mass
    weighted_rgb_usable_fraction = np.sum(
        candidate * np.mean(usable, axis=2, dtype=np.float64), axis=1
    ) / candidate_mass
    if not all(
        np.isfinite(value).all()
        for value in (
            maximum,
            margin,
            entropy,
            weighted_coverage,
            weighted_view_count,
            weighted_rgb_usable_fraction,
        )
    ):
        raise ValueError("target-free selector layout quality is non-finite")
    return {
        "coarse_max_probability": maximum.astype(np.float32),
        "coarse_margin": margin.astype(np.float32),
        "coarse_concentration": (-entropy).astype(np.float32),
        "support_coverage": weighted_coverage.astype(np.float32),
        "support_view_count": weighted_view_count.astype(np.float32),
        "rgb_context_usable_fraction": weighted_rgb_usable_fraction.astype(np.float32),
    }


def target_free_rgb_point_quality(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
) -> Mapping[str, np.ndarray]:
    """Summarize a pose-independent RGB density without target projections.

    The quantities are computed before a hypothesis supplies any projected
    offset.  In particular, this function does not score a center as a known
    correspondence: center probability is merely a static local registration
    sharpness statistic and must pass an independent hard-pose audit.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime) or not isinstance(
        prediction, CandidatePoseRGBSpatialEdgePrediction
    ):
        raise ValueError("target-free RGB selector quality requires runtime and prediction")
    device = prediction.joint_log_probabilities.device
    active = runtime.to(device)
    spatial = prediction.joint_log_probabilities[..., :-1]
    if spatial.shape[:3] != active.support_image_indices.shape:
        raise ValueError("target-free RGB selector prediction/runtime mismatch")
    category_count = int(spatial.shape[-1])
    if category_count < 4:
        raise ValueError("target-free RGB selector spatial grid is invalid")
    probabilities = torch.exp(spatial)
    spatial_mass = probabilities.sum(dim=-1).clamp_min(torch.finfo(probabilities.dtype).tiny)
    conditional = probabilities / spatial_mass.unsqueeze(-1)
    entropy = -torch.sum(
        conditional * torch.log(conditional.clamp_min(torch.finfo(conditional.dtype).tiny)), dim=-1
    )
    peakiness = 1.0 - entropy / math.log(float(category_count))
    offsets = prediction.offsets_xy.to(device=device, dtype=torch.float32)
    center_matches = torch.nonzero(
        torch.linalg.vector_norm(offsets, dim=1) <= 1e-6, as_tuple=False
    ).reshape(-1)
    if len(center_matches) != 1:
        raise ValueError("target-free RGB selector requires exactly one zero offset")
    center_probability = probabilities[..., int(center_matches.item())]
    edge_valid = prediction.edge_usable & active.support_view_valid
    mixture = (
        active.candidate_probabilities.unsqueeze(-1)
        * active.candidate_view_weights
        * edge_valid.to(dtype=probabilities.dtype)
    )
    mass = mixture.sum(dim=(1, 2)).clamp_min(torch.finfo(probabilities.dtype).tiny)
    weighted_peakiness = (mixture * peakiness).sum(dim=(1, 2)) / mass
    weighted_center_probability = (mixture * center_probability).sum(dim=(1, 2)) / mass
    weighted_spatial_mass = (mixture * spatial_mass).sum(dim=(1, 2)) / mass
    usable_fraction = edge_valid.to(dtype=probabilities.dtype).mean(dim=(1, 2))
    output = {
        "rgb_peakiness": weighted_peakiness.detach().cpu().numpy().astype(np.float32),
        "rgb_center_probability": weighted_center_probability.detach().cpu().numpy().astype(np.float32),
        "rgb_spatial_mass": weighted_spatial_mass.detach().cpu().numpy().astype(np.float32),
        "rgb_usable_edge_fraction": usable_fraction.detach().cpu().numpy().astype(np.float32),
    }
    if not all(np.isfinite(value).all() for value in output.values()):
        raise ValueError("target-free RGB selector quality is non-finite")
    return output


def target_free_selector_scores(
    *,
    selector_input: CandidatePoseRGBSpatialSelectorInput,
    policy: str,
    rgb_quality: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Return one target-free scalar score per point for a named policy."""

    if not isinstance(selector_input, CandidatePoseRGBSpatialSelectorInput):
        raise ValueError("target-free selector score requires selector inputs")
    name = str(policy).strip().lower()
    if name not in TARGET_FREE_SELECTOR_POLICIES:
        raise ValueError(f"unknown target-free selector policy: {policy!r}")
    layout_quality = target_free_layout_point_quality(selector_input)
    count = selector_input.point_count
    if name == "uniform":
        return np.zeros((count,), dtype=np.float32)
    if name in layout_quality:
        return np.asarray(layout_quality[name], dtype=np.float32)
    ids = selector_input.source_point_ids
    # This composition is entirely layout-derived.  Resolve it before the
    # RGB-only branch so a static selector never spuriously requires a learned
    # RGB prediction at checkpoint-selection time.
    if name == "coarse_margin_support_coverage":
        return 0.5 * (
            _rank_unit_interval(layout_quality["coarse_margin"], tie_breaker=ids)
            + _rank_unit_interval(layout_quality["support_coverage"], tie_breaker=ids)
        )
    if rgb_quality is None:
        raise ValueError(f"target-free selector policy {name} requires RGB quality")
    required = {"rgb_peakiness", "rgb_center_probability"}
    if not required.issubset(rgb_quality):
        raise ValueError("target-free RGB selector quality is incomplete")
    rgb = {key: np.asarray(value, dtype=np.float32).reshape(-1) for key, value in rgb_quality.items()}
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in rgb.values()):
        raise ValueError("target-free RGB selector quality shape is invalid")
    if name in rgb:
        return rgb[name]
    if name == "coarse_margin_rgb_peakiness":
        return 0.5 * (
            _rank_unit_interval(layout_quality["coarse_margin"], tie_breaker=ids)
            + _rank_unit_interval(rgb["rgb_peakiness"], tie_breaker=ids)
        )
    raise RuntimeError("target-free selector policy resolution is incomplete")


def slice_target_free_edge_prediction(
    *,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    positions: Sequence[int] | np.ndarray,
) -> CandidatePoseRGBSpatialEdgePrediction:
    """Subset a post-encoder prediction without adding target information."""

    if not isinstance(prediction, CandidatePoseRGBSpatialEdgePrediction):
        raise ValueError("target-free prediction slice requires an edge prediction")
    selected = torch.as_tensor(
        np.asarray(positions, dtype=np.int64).reshape(-1),
        dtype=torch.long,
        device=prediction.joint_log_probabilities.device,
    )
    count = int(prediction.spatial_logits.shape[0])
    if (
        len(selected) == 0
        or torch.any(selected < 0)
        or torch.any(selected >= count)
        or len(torch.unique(selected)) != len(selected)
    ):
        raise ValueError("target-free prediction slice positions are invalid")
    raw = None if prediction.raw_spatial_logits is None else prediction.raw_spatial_logits.index_select(0, selected)
    residual = (
        None
        if prediction.spatial_residual_logits is None
        else prediction.spatial_residual_logits.index_select(0, selected)
    )
    return CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=prediction.spatial_logits.index_select(0, selected),
        non_dustbin_logits=prediction.non_dustbin_logits.index_select(0, selected),
        joint_log_probabilities=prediction.joint_log_probabilities.index_select(0, selected),
        offsets_xy=prediction.offsets_xy,
        context_log_likelihood_ratios=prediction.context_log_likelihood_ratios.index_select(0, selected),
        edge_usable=prediction.edge_usable.index_select(0, selected),
        raw_spatial_logits=raw,
        spatial_residual_logits=residual,
        rgb_edge_usable=(
            None
            if prediction.rgb_edge_usable is None
            else prediction.rgb_edge_usable.index_select(0, selected)
        ),
        context_edge_usable=(
            None
            if prediction.context_edge_usable is None
            else prediction.context_edge_usable.index_select(0, selected)
        ),
    )


def select_target_free_spatial_quota(
    *,
    selector_input: CandidatePoseRGBSpatialSelectorInput,
    quality_scores: np.ndarray,
    point_budget: int,
    grid_rows: int,
    grid_columns: int,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Choose a deterministic spatially diverse subset using only static quality.

    The function deliberately keeps a fixed denominator after selection.  It
    returns local point positions; callers must not re-normalize the final pose
    score by pose-dependent usable-edge masks.
    """

    if not isinstance(selector_input, CandidatePoseRGBSpatialSelectorInput):
        raise ValueError("target-free selector requires selector inputs")
    scores = np.asarray(quality_scores, dtype=np.float64).reshape(-1)
    count = selector_input.point_count
    width, height = (int(value) for value in image_size)
    budget = int(point_budget)
    rows = int(grid_rows)
    columns = int(grid_columns)
    if (
        scores.shape != (count,)
        or not np.isfinite(scores).all()
        or budget <= 0
        or budget > count
        or rows <= 0
        or columns <= 0
        or width <= 1
        or height <= 1
        or np.any(selector_input.xy[:, 0] < 0.0)
        or np.any(selector_input.xy[:, 1] < 0.0)
    ):
        raise ValueError("target-free spatial selector quota inputs are invalid")
    if budget == count:
        return np.arange(count, dtype=np.int64)
    x = np.minimum((selector_input.xy[:, 0] * columns / float(width)).astype(np.int64), columns - 1)
    y = np.minimum((selector_input.xy[:, 1] * rows / float(height)).astype(np.int64), rows - 1)
    x = np.maximum(x, 0)
    y = np.maximum(y, 0)
    ids = selector_input.source_point_ids
    # Higher quality first; stable IDs resolve ties without depending on array
    # insertion order or train-only point annotations.
    def ordered(indices: np.ndarray) -> np.ndarray:
        local = np.asarray(indices, dtype=np.int64).reshape(-1)
        return local[np.lexsort((ids[local], -scores[local]))]

    cell_count = rows * columns
    base, remainder = divmod(budget, cell_count)
    selected: list[int] = []
    selected_set: set[int] = set()
    for cell in range(cell_count):
        quota = base + (1 if cell < remainder else 0)
        if quota <= 0:
            continue
        row, column = divmod(cell, columns)
        candidates = np.flatnonzero((y == row) & (x == column))
        for local in ordered(candidates)[:quota].tolist():
            selected.append(int(local))
            selected_set.add(int(local))
    if len(selected) < budget:
        for local in ordered(np.arange(count, dtype=np.int64)).tolist():
            if int(local) in selected_set:
                continue
            selected.append(int(local))
            selected_set.add(int(local))
            if len(selected) == budget:
                break
    result = np.asarray(sorted(selected), dtype=np.int64)
    if result.shape != (budget,) or len(np.unique(result)) != budget:
        raise RuntimeError("target-free spatial selector failed to fill its point budget")
    return result


def summarize_target_free_point_selection(
    *,
    selector_input: CandidatePoseRGBSpatialSelectorInput,
    selected_positions: Sequence[int] | np.ndarray,
    quality_scores: np.ndarray,
    grid_rows: int,
    grid_columns: int,
    image_size: tuple[int, int],
) -> Mapping[str, object]:
    """Return target-free selection diagnostics for an audit artifact."""

    selected = np.asarray(selected_positions, dtype=np.int64).reshape(-1)
    scores = np.asarray(quality_scores, dtype=np.float32).reshape(-1)
    count = selector_input.point_count
    if (
        len(selected) == 0
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= count)
        or scores.shape != (count,)
    ):
        raise ValueError("target-free selector summary inputs are invalid")
    width, height = (int(value) for value in image_size)
    columns = int(grid_columns)
    rows = int(grid_rows)
    if width <= 1 or height <= 1 or rows <= 0 or columns <= 0:
        raise ValueError("target-free selector summary image geometry is invalid")
    x = np.minimum((selector_input.xy[selected, 0] * columns / float(width)).astype(np.int64), columns - 1)
    y = np.minimum((selector_input.xy[selected, 1] * rows / float(height)).astype(np.int64), rows - 1)
    x = np.maximum(x, 0)
    y = np.maximum(y, 0)
    source_counts: dict[str, int] = {}
    for source in selector_input.point_sources[selected].tolist():
        source_counts[str(source)] = source_counts.get(str(source), 0) + 1
    occupied = len(set((int(row), int(column)) for row, column in zip(y.tolist(), x.tolist())))
    return {
        "format": CANDIDATE_POSE_RGB_SPATIAL_SELECTOR_FORMAT,
        "point_count": int(count),
        "selected_point_count": int(len(selected)),
        "spatial_grid_rows": rows,
        "spatial_grid_columns": columns,
        "occupied_grid_cells": int(occupied),
        "source_counts": dict(sorted(source_counts.items())),
        "selected_quality_mean": float(scores[selected].mean()),
        "selected_quality_min": float(scores[selected].min()),
        "selected_quality_max": float(scores[selected].max()),
    }
