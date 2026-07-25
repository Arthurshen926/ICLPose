"""Independent multiscale candidate-context likelihoods for fixed top-L poses.

This module deliberately sits between global landmark retrieval and local RGB
measurement.  It answers a narrower question than either of them:

``Does this query observation visually support this fixed candidate/view?``

RADIO-final, RADIO-intermediate, and ALIKE remain *separate* evidence
families.  Each uses a full two-dimensional query/support crop and an explicit
translation-phase summary; no absolute image coordinates, track ID, candidate
rank, coarse score, pose, residual, or target enters the visual encoder.  The
only learned part is one small, independently calibratable scalar LLR head per
source.  Support observations are marginalized with the immutable maplet
weights, not a learned best-view selector, so a single accidental repeated
patch cannot silently acquire all candidate mass.

Pose projections are accepted only by :func:`score_fixed_global_topl_phase_pose`
*after* the target-free visual forward.  Out-of-window candidates receive the
fixed neutral LLR rather than a learned dustbin reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_pose_llr import (
    _crop_subpixel_grid_tokens,
    bounded_log_likelihood_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)


CANDIDATE_MULTISCALE_CONTEXT_LLR_FORMAT = "candidate_multiscale_context_llr_v1"
CANDIDATE_MULTISCALE_CONTEXT_SOURCES = (
    "radio_final",
    "radio_intermediate",
    "alike",
)

# These are deliberately different physical roles rather than one concatenated
# crop: final captures broad facade context, intermediate captures structure,
# and ALIKE captures the local spatial phase.
DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    "radio_final": 5,
    "radio_intermediate": 9,
    "alike": 13,
}
DEFAULT_PHASE_SHIFT_RADII: dict[str, int] = {
    "radio_final": 1,
    "radio_intermediate": 2,
    "alike": 3,
}


def resolve_context_windows(values: Mapping[str, int] | None = None) -> dict[str, int]:
    """Validate one odd two-dimensional crop size per visual source."""

    source = DEFAULT_CONTEXT_WINDOWS if values is None else dict(values)
    if set(source) != set(CANDIDATE_MULTISCALE_CONTEXT_SOURCES):
        raise ValueError("multiscale context window source set is incomplete")
    result = {str(name): int(value) for name, value in source.items()}
    if any(value < 3 or value % 2 == 0 for value in result.values()):
        raise ValueError("multiscale context windows must be odd and at least three")
    return result


def resolve_phase_shift_radii(values: Mapping[str, int] | None = None) -> dict[str, int]:
    """Validate the fixed translation range used to summarize each crop."""

    source = DEFAULT_PHASE_SHIFT_RADII if values is None else dict(values)
    if set(source) != set(CANDIDATE_MULTISCALE_CONTEXT_SOURCES):
        raise ValueError("multiscale phase-radius source set is incomplete")
    result = {str(name): int(value) for name, value in source.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError("multiscale phase radii must be non-negative")
    return result


def resolve_visual_source_scales(
    scales: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return target-free appearance scales used only by visual controls."""

    source = {} if scales is None else dict(scales)
    if not set(source).issubset(CANDIDATE_MULTISCALE_CONTEXT_SOURCES):
        raise ValueError("multiscale visual source-scale set is invalid")
    result = {
        name: float(source.get(name, 1.0)) for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("multiscale visual source scales must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class CandidateMultiscaleContextLLRPrediction:
    """Target-free per-edge LLRs emitted separately for every visual source."""

    source_edge_log_likelihood_ratios: Mapping[str, torch.Tensor]
    source_edge_usable: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        llrs = {
            str(name): torch.as_tensor(value, dtype=torch.float32)
            for name, value in self.source_edge_log_likelihood_ratios.items()
        }
        usable = {
            str(name): torch.as_tensor(value, dtype=torch.bool)
            for name, value in self.source_edge_usable.items()
        }
        if (
            set(llrs) != set(CANDIDATE_MULTISCALE_CONTEXT_SOURCES)
            or set(usable) != set(CANDIDATE_MULTISCALE_CONTEXT_SOURCES)
        ):
            raise ValueError("multiscale context prediction source set is incomplete")
        reference_shape: tuple[int, ...] | None = None
        for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
            values = llrs[name]
            mask = usable[name].to(device=values.device)
            if (
                values.ndim != 3
                or values.shape[0] == 0
                or values.shape[1] == 0
                or values.shape[2] == 0
                or mask.shape != values.shape
                or not torch.isfinite(values).all()
            ):
                raise ValueError("multiscale context prediction tensor is invalid")
            if reference_shape is None:
                reference_shape = tuple(int(value) for value in values.shape)
            elif tuple(int(value) for value in values.shape) != reference_shape:
                raise ValueError("multiscale context prediction shapes differ by source")
            usable[name] = mask
        object.__setattr__(self, "source_edge_log_likelihood_ratios", llrs)
        object.__setattr__(self, "source_edge_usable", usable)

    @property
    def shape(self) -> tuple[int, int, int]:
        values = self.source_edge_log_likelihood_ratios["radio_final"]
        return tuple(int(value) for value in values.shape)


@dataclass(frozen=True)
class CandidateMultiscaleContextPoseScore:
    """Fixed-mixture target-free visual score for a caller supplied pose set."""

    pose_log_likelihood_ratios: torch.Tensor
    point_log_likelihood_ratios: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    candidate_projection_compatible: torch.Tensor
    source_candidate_log_likelihood_ratios: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        pose = torch.as_tensor(self.pose_log_likelihood_ratios, dtype=torch.float32)
        point = torch.as_tensor(self.point_log_likelihood_ratios, dtype=torch.float32)
        candidate = torch.as_tensor(self.candidate_log_likelihood_ratios, dtype=torch.float32)
        compatible = torch.as_tensor(self.candidate_projection_compatible, dtype=torch.bool)
        source = {
            str(name): torch.as_tensor(value, dtype=torch.float32)
            for name, value in self.source_candidate_log_likelihood_ratios.items()
        }
        if (
            pose.ndim != 1
            or point.ndim != 2
            or candidate.ndim != 3
            or pose.shape != (point.shape[0],)
            or candidate.shape[:2] != point.shape
            or compatible.shape != candidate.shape
            or set(source) != set(CANDIDATE_MULTISCALE_CONTEXT_SOURCES)
            or any(value.shape != candidate.shape[1:] for value in source.values())
            or not torch.isfinite(pose).all()
            or not torch.isfinite(point).all()
            or not torch.isfinite(candidate).all()
            or any(not torch.isfinite(value).all() for value in source.values())
        ):
            raise ValueError("multiscale context pose score is invalid")
        object.__setattr__(self, "pose_log_likelihood_ratios", pose)
        object.__setattr__(self, "point_log_likelihood_ratios", point)
        object.__setattr__(self, "candidate_log_likelihood_ratios", candidate)
        object.__setattr__(self, "candidate_projection_compatible", compatible)
        object.__setattr__(self, "source_candidate_log_likelihood_ratios", source)


def _phase_feature_dimension(*, shift_radius: int) -> int:
    shift_count = (2 * int(shift_radius) + 1) ** 2
    # Per translation: mean cosine, maximum cosine, valid mass.  The suffix
    # adds global/peak summaries plus nine spatial-block triplets.
    return 3 * shift_count + 12 + 3 * 9


def _phase_features(
    *,
    query_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    window_size: int,
    shift_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summarize full 2-D query/support correlation without coordinate leakage.

    The feature vector retains the complete small translation phase landscape,
    its peak/ambiguity statistics, and aligned 3x3 regional agreement.  It is
    intentionally deterministic: only the source-specific final calibration
    head is learned from system hard negatives.
    """

    query = torch.as_tensor(query_tokens, dtype=torch.float32)
    support = torch.as_tensor(support_tokens, dtype=torch.float32, device=query.device)
    query_mask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device)
    support_mask = torch.as_tensor(support_valid, dtype=torch.bool, device=query.device)
    width = int(window_size)
    radius = int(shift_radius)
    if (
        query.ndim != 3
        or support.shape != query.shape
        or query.shape[1] != width * width
        or query_mask.shape != query.shape[:2]
        or support_mask.shape != query.shape[:2]
        or width < 3
        or radius < 0
        or not torch.isfinite(query).all()
        or not torch.isfinite(support).all()
    ):
        raise ValueError("multiscale phase feature inputs are invalid")
    query = F.normalize(query, dim=2)
    support = F.normalize(support, dim=2)
    query_grid = query.reshape(len(query), width, width, -1)
    support_grid = support.reshape(len(support), width, width, -1)
    query_mask_grid = query_mask.reshape(len(query), width, width)
    support_mask_grid = support_mask.reshape(len(query), width, width)
    shift_means: list[torch.Tensor] = []
    shift_maxima: list[torch.Tensor] = []
    shift_coverage: list[torch.Tensor] = []
    shift_x: list[float] = []
    shift_y: list[float] = []
    for row_shift in range(-radius, radius + 1):
        for column_shift in range(-radius, radius + 1):
            query_rows = slice(max(row_shift, 0), width + min(row_shift, 0))
            support_rows = slice(max(-row_shift, 0), width - max(row_shift, 0))
            query_columns = slice(max(column_shift, 0), width + min(column_shift, 0))
            support_columns = slice(max(-column_shift, 0), width - max(column_shift, 0))
            cosine = torch.sum(
                query_grid[:, query_rows, query_columns]
                * support_grid[:, support_rows, support_columns],
                dim=3,
            )
            valid = (
                query_mask_grid[:, query_rows, query_columns]
                & support_mask_grid[:, support_rows, support_columns]
            )
            mass = valid.sum(dim=(1, 2))
            valid_float = valid.to(dtype=cosine.dtype)
            mean = (cosine * valid_float).sum(dim=(1, 2)) / mass.to(dtype=cosine.dtype).clamp_min(1.0)
            maximum = cosine.masked_fill(~valid, -1.0).amax(dim=(1, 2))
            maximum = torch.where(mass > 0, maximum, torch.zeros_like(maximum))
            overlap = float((width - abs(row_shift)) * (width - abs(column_shift)))
            shift_means.append(torch.where(mass > 0, mean, torch.zeros_like(mean)))
            shift_maxima.append(maximum)
            shift_coverage.append(mass.to(dtype=cosine.dtype) / overlap)
            shift_x.append(float(column_shift))
            shift_y.append(float(row_shift))
    means = torch.stack(shift_means, dim=1)
    maxima = torch.stack(shift_maxima, dim=1)
    coverage = torch.stack(shift_coverage, dim=1)
    valid_shift = coverage > 0.0
    masked_means = means.masked_fill(~valid_shift, -torch.inf)
    fallback = torch.zeros_like(masked_means)
    fallback[:, 0] = 0.0
    safe_means = torch.where(torch.any(valid_shift, dim=1, keepdim=True), masked_means, fallback)
    peak_values, peak_indices = safe_means.max(dim=1)
    sorted_values = torch.topk(safe_means, k=min(2, safe_means.shape[1]), dim=1).values
    second_values = (
        sorted_values[:, 1] if sorted_values.shape[1] == 2 else torch.zeros_like(peak_values)
    )
    center_index = len(shift_means) // 2
    center_values = means[:, center_index]
    weights = torch.softmax(safe_means * 4.0, dim=1)
    entropy = -(weights * torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))).sum(dim=1)
    entropy = entropy / math.log(float(max(2, means.shape[1])))
    coordinate_x = torch.as_tensor(shift_x, device=query.device, dtype=query.dtype)
    coordinate_y = torch.as_tensor(shift_y, device=query.device, dtype=query.dtype)
    denominator = float(max(1, radius))
    peak_x = coordinate_x.index_select(0, peak_indices) / denominator
    peak_y = coordinate_y.index_select(0, peak_indices) / denominator
    query_global = F.normalize(
        (query * query_mask.unsqueeze(2).to(dtype=query.dtype)).sum(dim=1)
        / query_mask.sum(dim=1, keepdim=True).to(dtype=query.dtype).clamp_min(1.0),
        dim=1,
    )
    support_global = F.normalize(
        (support * support_mask.unsqueeze(2).to(dtype=support.dtype)).sum(dim=1)
        / support_mask.sum(dim=1, keepdim=True).to(dtype=support.dtype).clamp_min(1.0),
        dim=1,
    )
    global_cosine = torch.sum(query_global * support_global, dim=1)
    block_features: list[torch.Tensor] = []
    boundaries = [round(index * width / 3.0) for index in range(4)]
    for row in range(3):
        for column in range(3):
            rows = slice(boundaries[row], boundaries[row + 1])
            columns = slice(boundaries[column], boundaries[column + 1])
            cosine = torch.sum(query_grid[:, rows, columns] * support_grid[:, rows, columns], dim=3)
            valid = query_mask_grid[:, rows, columns] & support_mask_grid[:, rows, columns]
            mass = valid.sum(dim=(1, 2))
            valid_float = valid.to(dtype=cosine.dtype)
            mean = (cosine * valid_float).sum(dim=(1, 2)) / mass.to(dtype=cosine.dtype).clamp_min(1.0)
            maximum = cosine.masked_fill(~valid, -1.0).amax(dim=(1, 2))
            maximum = torch.where(mass > 0, maximum, torch.zeros_like(maximum))
            block_mass = float((boundaries[row + 1] - boundaries[row]) * (boundaries[column + 1] - boundaries[column]))
            block_features.extend(
                (
                    torch.where(mass > 0, mean, torch.zeros_like(mean)),
                    maximum,
                    mass.to(dtype=cosine.dtype) / block_mass,
                )
            )
    summary = torch.stack(
        (
            means.mean(dim=1),
            means.std(dim=1, unbiased=False),
            maxima.mean(dim=1),
            peak_values,
            peak_values - second_values,
            peak_values - center_values,
            entropy,
            peak_x,
            peak_y,
            center_values,
            global_cosine,
            coverage.mean(dim=1),
        ),
        dim=1,
    )
    features = torch.cat(
        [means, maxima, coverage, summary, torch.stack(block_features, dim=1)], dim=1
    )
    expected = _phase_feature_dimension(shift_radius=radius)
    if features.shape != (len(query), expected) or not torch.isfinite(features).all():
        raise RuntimeError("multiscale phase feature construction drifted")
    paired = torch.any(query_mask & support_mask, dim=1)
    return features, paired


class _PhaseLLRHead(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).reshape(-1)


class CandidateMultiscaleContextLLR(nn.Module):
    """Full 2-D, per-source candidate/view LLR without learned view selection."""

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        context_windows: Mapping[str, int] | None = None,
        phase_shift_radii: Mapping[str, int] | None = None,
        hidden_dim: int = 32,
        max_abs_log_ratio: float = 4.0,
        edge_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if set(sources) != set(CANDIDATE_MULTISCALE_CONTEXT_SOURCES):
            raise ValueError("multiscale context LLR source set is incomplete")
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        if sizes.ndim != 2 or sizes.shape[1] != 2 or len(sizes) == 0 or torch.any(sizes <= 1.0):
            raise ValueError("multiscale context LLR image sizes are invalid")
        windows = resolve_context_windows(context_windows)
        radii = resolve_phase_shift_radii(phase_shift_radii)
        if (
            int(hidden_dim) < 4
            or int(edge_chunk_size) <= 0
            or not math.isfinite(float(max_abs_log_ratio))
            or float(max_abs_log_ratio) <= 0.0
        ):
            raise ValueError("multiscale context LLR configuration is invalid")
        heads: dict[str, nn.Module] = {}
        for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
            grid = torch.as_tensor(sources[name], dtype=torch.float32)
            if (
                grid.ndim != 4
                or grid.shape[0] != len(sizes)
                or grid.shape[1] != grid.shape[2]
                or grid.shape[1] < int(windows[name])
                or grid.shape[3] <= 0
                or not torch.isfinite(grid).all()
            ):
                raise ValueError(f"multiscale context {name} source grid is invalid")
            norms = torch.linalg.vector_norm(grid, dim=-1)
            if torch.max(torch.abs(norms - 1.0)) > 5e-3:
                raise ValueError(f"multiscale context {name} descriptors are not normalized")
            self.register_buffer(f"_{name}_grid", grid, persistent=False)
            heads[name] = _PhaseLLRHead(
                feature_dim=_phase_feature_dimension(shift_radius=int(radii[name])),
                hidden_dim=int(hidden_dim),
            )
        self.heads = nn.ModuleDict(heads)
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.context_windows = windows
        self.phase_shift_radii = radii
        self.max_abs_log_ratio = float(max_abs_log_ratio)
        self.edge_chunk_size = int(edge_chunk_size)

    @property
    def device(self) -> torch.device:
        return self._image_sizes.device

    def _validate_runtime(self, runtime: CandidatePoseRGBSpatialRuntime) -> CandidatePoseRGBSpatialRuntime:
        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("multiscale context LLR requires a target-free runtime")
        active = runtime.to(self.device)
        image_count = len(self._image_sizes)
        if (
            torch.any(active.query_image_indices >= image_count)
            or torch.any(active.support_image_indices >= image_count)
        ):
            raise ValueError("multiscale context LLR runtime image index is out of range")
        return active

    def forward(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        visual_source_scales: Mapping[str, float] | None = None,
    ) -> CandidateMultiscaleContextLLRPrediction:
        """Encode all fixed candidate/view edges once without a pose input."""

        active = self._validate_runtime(runtime)
        scales = resolve_visual_source_scales(visual_source_scales)
        point_count = active.point_count
        candidate_count = active.candidate_count
        view_count = active.support_view_count
        edge_count = point_count * candidate_count * view_count
        edge_to_point = torch.arange(point_count, device=self.device).repeat_interleave(
            candidate_count * view_count
        )
        query_indices = active.query_image_indices.index_select(0, edge_to_point)
        query_xy = active.query_xy.index_select(0, edge_to_point)
        support_indices = active.support_image_indices.reshape(-1)
        support_xy = active.support_xy.reshape(-1, 2)
        source_outputs: dict[str, torch.Tensor] = {}
        source_masks: dict[str, torch.Tensor] = {}
        for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
            grid = getattr(self, f"_{name}_grid")
            window = int(self.context_windows[name])
            radius = int(self.phase_shift_radii[name])
            parts: list[torch.Tensor] = []
            mask_parts: list[torch.Tensor] = []
            for begin in range(0, edge_count, self.edge_chunk_size):
                end = min(begin + self.edge_chunk_size, edge_count)
                query_crop, query_valid = _crop_subpixel_grid_tokens(
                    image_grids=grid,
                    image_sizes=self._image_sizes,
                    image_indices=query_indices[begin:end],
                    xy=query_xy[begin:end],
                    window_size=window,
                )
                support_crop, support_valid = _crop_subpixel_grid_tokens(
                    image_grids=grid,
                    image_sizes=self._image_sizes,
                    image_indices=support_indices[begin:end],
                    xy=support_xy[begin:end],
                    window_size=window,
                )
                features, paired = _phase_features(
                    query_tokens=query_crop * float(scales[name]),
                    support_tokens=support_crop * float(scales[name]),
                    query_valid=query_valid,
                    support_valid=support_valid,
                    window_size=window,
                    shift_radius=radius,
                )
                # A zero-content counterfactual is an exact neutral visual
                # control.  Do not let masks or MLP biases turn it into a
                # geometry/availability shortcut.
                if float(scales[name]) == 0.0:
                    values = torch.zeros((len(features),), dtype=features.dtype, device=features.device)
                else:
                    values = bounded_log_likelihood_ratio(
                        self.heads[name](features), max_abs_log_ratio=self.max_abs_log_ratio
                    )
                parts.append(values)
                mask_parts.append(paired)
            values = torch.cat(parts, dim=0).reshape(point_count, candidate_count, view_count)
            usable = torch.cat(mask_parts, dim=0).reshape(point_count, candidate_count, view_count)
            usable = usable & active.support_view_valid
            source_outputs[name] = torch.where(usable, values, torch.zeros_like(values))
            source_masks[name] = usable
        return CandidateMultiscaleContextLLRPrediction(
            source_edge_log_likelihood_ratios=source_outputs,
            source_edge_usable=source_masks,
        )


def fixed_support_view_mixture_log_likelihood_ratio(
    *,
    edge_log_likelihood_ratios: torch.Tensor,
    edge_usable: torch.Tensor,
    candidate_view_weights: torch.Tensor,
) -> torch.Tensor:
    """Marginalize views with immutable maplet mass and neutral missingness."""

    values = torch.as_tensor(edge_log_likelihood_ratios, dtype=torch.float32)
    usable = torch.as_tensor(edge_usable, dtype=torch.bool, device=values.device)
    weights = torch.as_tensor(candidate_view_weights, dtype=torch.float32, device=values.device)
    if (
        values.ndim != 3
        or usable.shape != values.shape
        or weights.shape != values.shape
        or not torch.isfinite(values).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
    ):
        raise ValueError("fixed support-view LLR mixture inputs are invalid")
    mass = weights.sum(dim=2)
    active = mass > 0.0
    if torch.any(torch.abs(mass[active] - 1.0) > 1e-4) or torch.any(mass[~active] > 1e-6):
        raise ValueError("fixed support-view LLR weights must sum to one")
    effective = torch.where(usable, values, torch.zeros_like(values))
    log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    mixture = torch.logsumexp(effective + log_weights, dim=2)
    return torch.where(active, mixture, torch.zeros_like(mixture))


def source_candidate_log_likelihood_ratios(
    *,
    prediction: CandidateMultiscaleContextLLRPrediction,
    runtime: CandidatePoseRGBSpatialRuntime,
) -> dict[str, torch.Tensor]:
    """Return one fixed-view candidate LLR table per independent source."""

    if not isinstance(prediction, CandidateMultiscaleContextLLRPrediction):
        raise ValueError("source candidate LLRs require a multiscale prediction")
    device = prediction.source_edge_log_likelihood_ratios["radio_final"].device
    active = runtime.to(device)
    expected = tuple(int(value) for value in active.support_image_indices.shape)
    if prediction.shape != expected:
        raise ValueError("multiscale prediction and target-free runtime differ")
    return {
        name: fixed_support_view_mixture_log_likelihood_ratio(
            edge_log_likelihood_ratios=prediction.source_edge_log_likelihood_ratios[name],
            edge_usable=prediction.source_edge_usable[name],
            candidate_view_weights=active.candidate_view_weights,
        )
        for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES
    }


def _resolve_source_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    source = {} if weights is None else dict(weights)
    if not set(source).issubset(CANDIDATE_MULTISCALE_CONTEXT_SOURCES):
        raise ValueError("multiscale source-weight set is invalid")
    result = {name: float(source.get(name, 1.0)) for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()) or not any(
        value > 0.0 for value in result.values()
    ):
        raise ValueError("multiscale source weights must be finite and non-negative")
    return result


def score_fixed_global_topl_phase_pose(
    *,
    prediction: CandidateMultiscaleContextLLRPrediction,
    runtime: CandidatePoseRGBSpatialRuntime,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    local_radius_px: float,
    source_weights: Mapping[str, float] | None = None,
) -> CandidateMultiscaleContextPoseScore:
    """Score supplied poses with fixed global top-L/null mass.

    This function deliberately never selects a new candidate or support view
    for a particular pose.  A candidate outside the local compatibility window
    gets a neutral visual LLR, leaving its immutable prior mass and the
    explicit null unchanged.
    """

    radius = float(local_radius_px)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("multiscale pose local radius is invalid")
    source_candidate = source_candidate_log_likelihood_ratios(
        prediction=prediction, runtime=runtime
    )
    device = next(iter(source_candidate.values())).device
    active = runtime.to(device)
    offsets = torch.as_tensor(candidate_projection_offsets_xy, dtype=torch.float32, device=device)
    valid = torch.as_tensor(candidate_projection_valid, dtype=torch.bool, device=device)
    if (
        offsets.ndim != 4
        or offsets.shape[-1] != 2
        or offsets.shape[1:] != active.candidate_probabilities.shape + (2,)
        or valid.shape != offsets.shape[:-1]
        or len(offsets) == 0
        or not torch.isfinite(offsets).all()
    ):
        raise ValueError("multiscale pose projection inputs are invalid")
    compatible = valid & (torch.amax(torch.abs(offsets), dim=3) <= radius)
    weights = _resolve_source_weights(source_weights)
    combined_static = sum(
        float(weights[name]) * source_candidate[name] for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES
    )
    candidate_llr = torch.where(
        compatible,
        combined_static.unsqueeze(0).expand(len(offsets), -1, -1),
        torch.zeros((len(offsets), *combined_static.shape), dtype=combined_static.dtype, device=device),
    )
    prior = active.candidate_probabilities
    null = active.null_probabilities
    candidate_terms = torch.where(
        prior.unsqueeze(0) > 0.0,
        torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny)).unsqueeze(0) + candidate_llr,
        torch.full_like(candidate_llr, -torch.inf),
    )
    null_terms = torch.where(
        null > 0.0,
        torch.log(null.clamp_min(torch.finfo(null.dtype).tiny)),
        torch.full_like(null, -torch.inf),
    ).reshape(1, len(null), 1).expand(len(offsets), -1, -1)
    point_llr = torch.logsumexp(torch.cat((candidate_terms, null_terms), dim=2), dim=2)
    return CandidateMultiscaleContextPoseScore(
        pose_log_likelihood_ratios=point_llr.mean(dim=1),
        point_log_likelihood_ratios=point_llr,
        candidate_log_likelihood_ratios=candidate_llr,
        candidate_projection_compatible=compatible,
        source_candidate_log_likelihood_ratios=source_candidate,
    )
