"""Target-free multiscale candidate identity likelihood from 2-D phase fields.

This module is deliberately separate from the RGB measurement density.  It
answers only whether a fixed query point and a fixed landmark/support-view
edge look like the same physical observation.  The visual forward consumes
three real-image descriptor grids:

* RADIO-final for broader facade context;
* RADIO-intermediate for structural layout;
* ALIKE for the finer spatial branch.

For every candidate/view edge it preserves the full bounded 2-D translation
field in each crop before a source-specific LLR head.  It never receives a
pose, projection offset, track ID, candidate rank, coarse score, residual, or
training label.  Those values may only be joined after :meth:`forward` by a
train-only loss or by a downstream pose scorer.

The output remains per support view.  Fixed maplet view mass is marginalized
outside the visual model, so a convenient support view cannot silently absorb
the mass of unavailable views.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_pose_llr import (
    bounded_log_likelihood_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    _crop_subpixel_grid_tokens,
)


CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT = (
    "candidate_multiscale_phase_identity_llr_v1"
)
CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES = (
    "radio_final",
    "radio_intermediate",
    "alike",
)


@dataclass(frozen=True)
class PhaseIdentitySourceConfig:
    """Fixed geometry for one descriptor-space phase field.

    ``window_size`` is the full two-dimensional context crop.  Every crop
    token participates in every bounded translation statistic; it is not a
    center-token feature with an enlarged receptive-field annotation.
    """

    name: str
    window_size: int
    shift_radius: int
    region_bins: int = 3

    def __post_init__(self) -> None:
        name = str(self.name)
        window = int(self.window_size)
        shift = int(self.shift_radius)
        bins = int(self.region_bins)
        if (
            name not in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
            or window < 3
            or window % 2 != 1
            or shift < 0
            or shift >= window
            or bins < 1
            or bins > window
        ):
            raise ValueError("phase identity source configuration is invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "window_size", window)
        object.__setattr__(self, "shift_radius", shift)
        object.__setattr__(self, "region_bins", bins)

    @property
    def shift_count(self) -> int:
        return (2 * int(self.shift_radius) + 1) ** 2

    @property
    def feature_dimension(self) -> int:
        # Per shift: mean/max cosine in every fixed query crop region, plus
        # global mean/max.  No validity fraction enters the learned head: crop
        # masks determine availability only and cannot become a geometry cue.
        return self.shift_count * (2 * int(self.region_bins) ** 2 + 2)


DEFAULT_PHASE_IDENTITY_SOURCE_CONFIGS = (
    # All windows must be fully real in both query and support images.  These
    # sizes were selected from the frozen P1 layout rather than padded to a
    # nominal larger receptive field: 3/3/5 retains 76%+ of registered and
    # current-hard candidate edges, whereas 7/9/13 leaves most supervision
    # unavailable at image boundaries.  The shift field still retains every
    # token in the crop; it is not a center-descriptor fallback.
    PhaseIdentitySourceConfig("radio_final", window_size=3, shift_radius=1, region_bins=3),
    PhaseIdentitySourceConfig("radio_intermediate", window_size=3, shift_radius=1, region_bins=3),
    PhaseIdentitySourceConfig("alike", window_size=5, shift_radius=2, region_bins=3),
)


def resolve_phase_identity_source_configs(
    configs: Mapping[str, PhaseIdentitySourceConfig] | None = None,
) -> dict[str, PhaseIdentitySourceConfig]:
    """Return one validated independent phase layout per visual source."""

    if configs is None:
        values = {item.name: item for item in DEFAULT_PHASE_IDENTITY_SOURCE_CONFIGS}
    else:
        values = {str(name): value for name, value in configs.items()}
    if set(values) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES) or any(
        not isinstance(value, PhaseIdentitySourceConfig) for value in values.values()
    ):
        raise ValueError("phase identity source configuration set is incomplete")
    if any(value.name != name for name, value in values.items()):
        raise ValueError("phase identity source configuration names differ from keys")
    return {name: values[name] for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES}


def resolve_phase_identity_source_weights(
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve a fixed conservative source blend.

    The three branches share a query image and are not asserted independent
    probability factors.  A convex blend keeps their calibration explicit and
    lets source-only audits reuse exactly the same runtime forward.
    """

    source = {} if weights is None else dict(weights)
    if not set(source).issubset(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES):
        raise ValueError("phase identity source weights contain an unknown source")
    values = {
        name: float(source.get(name, 1.0))
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("phase identity source weights are invalid")
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("phase identity source weights must retain one source")
    return {name: value / total for name, value in values.items()}


def phase_identity_point_block_derangement_shift(*, point_count: int, shift: int) -> int:
    """Return the shared distant point-block appearance-control shift.

    A flat-edge roll mainly swaps adjacent candidate/view slots of the same
    query point.  The production control instead preserves every candidate
    and support-view slot while moving support appearance from a distant point
    block.  RGB hybrid branches call this helper too, so their support-image
    control cannot silently drift from the RADIO phase control.
    """

    count = int(point_count)
    value = int(shift)
    if count < 2 or value == 0:
        raise ValueError("phase identity point-block derangement requires two points and a nonzero shift")
    amount = (count // 2 + value - 1) % count
    return 1 if amount == 0 else int(amount)


@dataclass(frozen=True)
class CandidateMultiscalePhaseIdentityPrediction:
    """Target-free candidate/support-view LLRs from independent phase fields."""

    source_edge_log_likelihood_ratios: Mapping[str, torch.Tensor]
    source_edge_usable: Mapping[str, torch.Tensor]
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor
    source_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        source_llrs = {
            str(name): torch.as_tensor(value, dtype=torch.float32)
            for name, value in self.source_edge_log_likelihood_ratios.items()
        }
        source_usable = {
            str(name): torch.as_tensor(value, dtype=torch.bool)
            for name, value in self.source_edge_usable.items()
        }
        edge_llr = torch.as_tensor(self.edge_log_likelihood_ratios, dtype=torch.float32)
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool, device=edge_llr.device)
        weights = resolve_phase_identity_source_weights(self.source_weights)
        if (
            set(source_llrs) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
            or set(source_usable) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
            or edge_llr.ndim != 3
            or edge_llr.shape[0] == 0
            or edge_llr.shape[1] == 0
            or edge_llr.shape[2] == 0
            or usable.shape != edge_llr.shape
            or not torch.isfinite(edge_llr).all()
        ):
            raise ValueError("phase identity prediction is invalid")
        combined = torch.zeros_like(edge_llr)
        combined_usable = torch.zeros_like(usable)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
            values = source_llrs[name].to(device=edge_llr.device)
            mask = source_usable[name].to(device=edge_llr.device)
            if values.shape != edge_llr.shape or mask.shape != edge_llr.shape or not torch.isfinite(values).all():
                raise ValueError("phase identity source prediction shape is invalid")
            combined = combined + float(weights[name]) * torch.where(mask, values, torch.zeros_like(values))
            combined_usable = combined_usable | mask
        if not torch.allclose(combined, edge_llr, atol=2e-5, rtol=2e-5) or not torch.equal(
            combined_usable, usable
        ):
            raise ValueError("phase identity fused prediction does not match source factors")
        object.__setattr__(self, "source_edge_log_likelihood_ratios", source_llrs)
        object.__setattr__(self, "source_edge_usable", source_usable)
        object.__setattr__(self, "edge_log_likelihood_ratios", edge_llr)
        object.__setattr__(self, "edge_usable", usable)
        object.__setattr__(self, "source_weights", weights)


def _edge_layout(runtime: CandidatePoseRGBSpatialRuntime) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten fixed P1 edges without adding candidate or pose metadata."""

    point_count = runtime.point_count
    candidate_count = runtime.candidate_count
    view_count = runtime.support_view_count
    edge_to_point = torch.arange(point_count, device=runtime.query_xy.device).repeat_interleave(
        candidate_count * view_count
    )
    query_image_indices = runtime.query_image_indices.index_select(0, edge_to_point)
    query_xy = runtime.query_xy.index_select(0, edge_to_point)
    return (
        edge_to_point,
        query_image_indices,
        query_xy,
        runtime.support_image_indices.reshape(-1),
    )


def _phase_field_features(
    *,
    query_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    config: PhaseIdentitySourceConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode an entire crop's bounded 2-D translation field.

    A source is usable only when its full zero-relative-phase crop is real in
    both images.  This deliberately prevents padded crop masks or image-bound
    geometry from becoming a shortcut for identity classification.  All
    per-shift statistics are then calculated from valid descriptor values only.
    """

    query = torch.as_tensor(query_tokens)
    support = torch.as_tensor(support_tokens, device=query.device)
    query_mask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device)
    support_mask = torch.as_tensor(support_valid, dtype=torch.bool, device=query.device)
    width = int(config.window_size)
    if (
        query.ndim != 3
        or support.shape != query.shape
        or query.shape[1] != width * width
        or query_mask.shape != query.shape[:2]
        or support_mask.shape != query.shape[:2]
        or not torch.isfinite(query).all()
        or not torch.isfinite(support).all()
    ):
        raise ValueError("phase identity descriptor crop inputs are invalid")
    # FP32 cosine correlation is stable for the 1280-D RADIO descriptor maps
    # even when the cached grid itself is held in FP16 on the GPU.
    query_grid = F.normalize(query.float(), dim=2).reshape(len(query), width, width, -1)
    support_grid = F.normalize(support.float(), dim=2).reshape(len(support), width, width, -1)
    query_mask_grid = query_mask.reshape(len(query), width, width)
    support_mask_grid = support_mask.reshape(len(query), width, width)
    # The pinned PyTorch version accepts a single reduction dimension only for
    # ``torch.all``.  Flattening also makes the full-crop availability contract
    # explicit: an edge is usable only when every zero-phase crop token is real.
    usable = (query_mask_grid & support_mask_grid).reshape(len(query), -1).all(dim=1)
    bins = int(config.region_bins)
    feature_parts: list[torch.Tensor] = []
    for row_shift in range(-int(config.shift_radius), int(config.shift_radius) + 1):
        for column_shift in range(-int(config.shift_radius), int(config.shift_radius) + 1):
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
            # Whole-crop availability above makes every surviving map full, but
            # keep this guard for malformed/corner diagnostic inputs.
            mass = valid.sum(dim=(1, 2))
            global_mean = torch.where(
                mass > 0,
                (cosine * valid.to(dtype=cosine.dtype)).sum(dim=(1, 2))
                / mass.to(dtype=cosine.dtype).clamp_min(1.0),
                torch.zeros_like(mass, dtype=cosine.dtype),
            )
            global_max = torch.where(
                mass > 0,
                cosine.masked_fill(~valid, -1.0).amax(dim=(1, 2)),
                torch.zeros_like(global_mean),
            )
            padded = torch.zeros(
                (len(query), 1, width, width), dtype=cosine.dtype, device=cosine.device
            )
            padded_mask = torch.zeros(
                (len(query), 1, width, width), dtype=torch.bool, device=cosine.device
            )
            padded[:, 0, query_rows, query_columns] = cosine
            padded_mask[:, 0, query_rows, query_columns] = valid
            regional_sum = F.adaptive_avg_pool2d(
                padded * padded_mask.to(dtype=padded.dtype), (bins, bins)
            )
            regional_mass = F.adaptive_avg_pool2d(
                padded_mask.to(dtype=padded.dtype), (bins, bins)
            )
            regional_mean = torch.where(
                regional_mass > 0.0,
                regional_sum / regional_mass.clamp_min(torch.finfo(padded.dtype).tiny),
                torch.zeros_like(regional_sum),
            )
            regional_max = F.adaptive_max_pool2d(
                padded.masked_fill(~padded_mask, -1.0), (bins, bins)
            )
            regional_max = torch.where(regional_mass > 0.0, regional_max, torch.zeros_like(regional_max))
            feature_parts.append(
                torch.cat(
                    [
                        regional_mean.reshape(len(query), -1),
                        regional_max.reshape(len(query), -1),
                        global_mean[:, None],
                        global_max[:, None],
                    ],
                    dim=1,
                )
            )
    features = torch.cat(feature_parts, dim=1)
    if features.shape != (len(query), int(config.feature_dimension)) or not torch.isfinite(features).all():
        raise RuntimeError("phase identity field feature geometry drifted")
    return features, usable


class _PhaseIdentityHead(nn.Module):
    """One independently calibrated source LLR head over a full phase field."""

    def __init__(self, *, feature_dim: int, hidden_dim: int, max_abs_log_ratio: float) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim < 4 or max_abs_log_ratio <= 0.0:
            raise ValueError("phase identity source head configuration is invalid")
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.max_abs_log_ratio = float(max_abs_log_ratio)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(features, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] == 0 or not torch.isfinite(values).all():
            raise ValueError("phase identity source head inputs are invalid")
        return bounded_log_likelihood_ratio(
            self.network(values).reshape(-1),
            max_abs_log_ratio=self.max_abs_log_ratio,
        )


class CandidateMultiscalePhaseIdentityLLR(nn.Module):
    """Candidate/view identity LLR from retained RADIO/ALIKE phase fields."""

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        source_configs: Mapping[str, PhaseIdentitySourceConfig] | None = None,
        source_weights: Mapping[str, float] | None = None,
        hidden_dim: int = 96,
        max_abs_log_ratio: float = 4.0,
        source_storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        configs = resolve_phase_identity_source_configs(source_configs)
        weights = resolve_phase_identity_source_weights(source_weights)
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        source_map = {str(name): value for name, value in sources.items()}
        if (
            not set(source_map).issubset(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
            or any(
                name not in source_map and float(weights[name]) > 0.0
                for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
            )
            or sizes.ndim != 2
            or sizes.shape[0] == 0
            or sizes.shape[1] != 2
            or torch.any(sizes <= 1.0)
            or int(hidden_dim) < 4
            or not math.isfinite(float(max_abs_log_ratio))
            or float(max_abs_log_ratio) <= 0.0
            or source_storage_dtype not in {torch.float16, torch.float32}
        ):
            raise ValueError("phase identity model configuration is invalid")
        heads: dict[str, nn.Module] = {}
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
            config = configs[name]
            if name not in source_map:
                # Zero-mass sources must remain structurally explicit without
                # forcing a production RADIO-final-only experiment to load a
                # large inactive descriptor cache.  This normalized one-token
                # descriptor grid is never read by ``forward`` because the
                # zero-weight branch below emits neutral evidence directly.
                grid = torch.ones(
                    (len(sizes), int(config.window_size), int(config.window_size), 1),
                    dtype=source_storage_dtype,
                )
            else:
                grid = torch.as_tensor(source_map[name], dtype=source_storage_dtype)
            if (
                grid.ndim != 4
                or grid.shape[0] != len(sizes)
                or grid.shape[1] != grid.shape[2]
                or grid.shape[1] < int(config.window_size)
                or grid.shape[3] <= 0
                or not torch.isfinite(grid).all()
            ):
                raise ValueError(f"phase identity {name} source grid is invalid")
            # Values are normalized again during crop correlation, but reject
            # arbitrary feature tensors here so the descriptor-space contract
            # cannot silently drift into a learned image encoder.
            norms = torch.linalg.vector_norm(grid.float(), dim=-1)
            if not torch.isfinite(norms).all() or torch.max(torch.abs(norms - 1.0)).item() > 1e-2:
                raise ValueError(f"phase identity {name} descriptors are not normalized")
            self.register_buffer(f"_{name}_grid", grid, persistent=False)
            heads[name] = _PhaseIdentityHead(
                feature_dim=int(config.feature_dimension),
                hidden_dim=int(hidden_dim),
                max_abs_log_ratio=float(max_abs_log_ratio),
            )
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.source_heads = nn.ModuleDict(heads)
        self.source_configs = configs
        self.source_weights = weights
        self.max_abs_log_ratio = float(max_abs_log_ratio)

    @property
    def device(self) -> torch.device:
        return self._image_sizes.device

    def _validate_runtime(self, runtime: CandidatePoseRGBSpatialRuntime) -> CandidatePoseRGBSpatialRuntime:
        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("phase identity likelihood requires a target-free runtime")
        active = runtime.to(self.device)
        if (
            torch.any(active.query_image_indices < 0)
            or torch.any(active.query_image_indices >= len(self._image_sizes))
            or torch.any(active.support_image_indices < 0)
            or torch.any(active.support_image_indices >= len(self._image_sizes))
        ):
            raise ValueError("phase identity runtime image index is invalid")
        return active

    def _neutral_prediction(
        self, *, runtime: CandidatePoseRGBSpatialRuntime
    ) -> CandidateMultiscalePhaseIdentityPrediction:
        shape = tuple(int(value) for value in runtime.support_view_valid.shape)
        source_llrs = {
            name: torch.zeros(shape, dtype=torch.float32, device=self.device)
            for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        }
        source_usable = {
            name: runtime.support_view_valid.to(device=self.device)
            for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        }
        return CandidateMultiscalePhaseIdentityPrediction(
            source_edge_log_likelihood_ratios=source_llrs,
            source_edge_usable=source_usable,
            edge_log_likelihood_ratios=torch.zeros(shape, dtype=torch.float32, device=self.device),
            edge_usable=runtime.support_view_valid.to(device=self.device),
            source_weights=self.source_weights,
        )

    def _source_prediction(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        source_name: str,
        edge_to_point: torch.Tensor,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        support_permutation_shift: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.source_configs[source_name]
        grid = getattr(self, f"_{source_name}_grid")
        query_crop, query_valid = _crop_subpixel_grid_tokens(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=runtime.query_image_indices,
            xy=runtime.query_xy,
            window_size=int(config.window_size),
        )
        support_crop, support_valid = _crop_subpixel_grid_tokens(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=support_image_indices,
            xy=support_xy,
            window_size=int(config.window_size),
        )
        query_crop = query_crop.index_select(0, edge_to_point)
        query_valid = query_valid.index_select(0, edge_to_point)
        shift = int(support_permutation_shift)
        if shift:
            point_count = runtime.point_count
            candidate_count = runtime.candidate_count
            view_count = runtime.support_view_count
            if point_count >= 2:
                # A one-edge roll mostly substitutes an adjacent candidate/view
                # from the same query point, which is far too weak as an
                # appearance control on a repetitive facade.  Keep each
                # candidate/view slot fixed, but cyclically derange complete
                # support crops across distant point blocks.  The amount is
                # deliberately near half a batch for ``shift=1``; other
                # positive shifts remain deterministic audit variants.
                amount = phase_identity_point_block_derangement_shift(
                    point_count=point_count, shift=shift
                )
                crop_shape = (point_count, candidate_count, view_count, *support_crop.shape[1:])
                valid_shape = (point_count, candidate_count, view_count, support_valid.shape[1])
                support_crop = torch.roll(
                    support_crop.reshape(crop_shape), shifts=int(amount), dims=0
                ).reshape_as(support_crop)
                support_valid = torch.roll(
                    support_valid.reshape(valid_shape), shifts=int(amount), dims=0
                ).reshape_as(support_valid)
            else:
                # Unit-scale diagnostics can contain a single query point. In
                # that degenerate case only a candidate/view edge derangement
                # is possible; production gates always use point blocks.
                edge_count = len(support_crop)
                if edge_count < 2 or shift % edge_count == 0:
                    raise ValueError("phase identity support permutation is not nontrivial")
                permutation = torch.roll(
                    torch.arange(edge_count, device=self.device), shifts=shift, dims=0
                )
                support_crop = support_crop.index_select(0, permutation)
                support_valid = support_valid.index_select(0, permutation)
        features, crop_usable = _phase_field_features(
            query_tokens=query_crop,
            support_tokens=support_crop,
            query_valid=query_valid,
            support_valid=support_valid,
            config=config,
        )
        llr = self.source_heads[source_name](features)
        point_count = runtime.point_count
        candidate_count = runtime.candidate_count
        view_count = runtime.support_view_count
        shape = (point_count, candidate_count, view_count)
        edge_usable = crop_usable.reshape(shape) & runtime.support_view_valid
        return llr.reshape(shape), edge_usable

    def forward(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        support_permutation_shift: int = 0,
        zero_appearance: bool = False,
    ) -> CandidateMultiscalePhaseIdentityPrediction:
        """Emit candidate/view LLRs before any pose or target is supplied.

        ``support_permutation_shift`` is an audit-only appearance control. It
        swaps complete support crops across distant query-point blocks while
        retaining each query anchor and candidate/view slot; it is
        intentionally not a training target or runtime input.
        """

        active = self._validate_runtime(runtime)
        if bool(zero_appearance):
            return self._neutral_prediction(runtime=active)
        edge_to_point, query_image_indices, query_xy, support_image_indices = _edge_layout(active)
        del query_image_indices, query_xy  # ownership is carried by ``active`` for query crops.
        support_xy = active.support_xy.reshape(-1, 2)
        source_llrs: dict[str, torch.Tensor] = {}
        source_usable: dict[str, torch.Tensor] = {}
        combined = torch.zeros_like(active.candidate_view_weights, dtype=torch.float32)
        combined_usable = torch.zeros_like(active.support_view_valid)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
            # A source with zero fixed convex mass must remain an explicit
            # neutral factor.  Besides preventing an inactive descriptor
            # space from affecting availability diagnostics, this avoids
            # evaluating large ALIKE/intermediate phase fields in the
            # RADIO-final-only production ablation.
            if float(self.source_weights[name]) == 0.0:
                source_llrs[name] = torch.zeros_like(combined)
                source_usable[name] = torch.zeros_like(active.support_view_valid)
                continue
            llr, usable = self._source_prediction(
                runtime=active,
                source_name=name,
                edge_to_point=edge_to_point,
                query_image_indices=active.query_image_indices,
                query_xy=active.query_xy,
                support_image_indices=support_image_indices,
                support_xy=support_xy,
                support_permutation_shift=int(support_permutation_shift),
            )
            source_llrs[name] = llr
            source_usable[name] = usable
            combined = combined + float(self.source_weights[name]) * torch.where(
                usable, llr, torch.zeros_like(llr)
            )
            combined_usable = combined_usable | usable
        return CandidateMultiscalePhaseIdentityPrediction(
            source_edge_log_likelihood_ratios=source_llrs,
            source_edge_usable=source_usable,
            edge_log_likelihood_ratios=combined,
            edge_usable=combined_usable,
            source_weights=self.source_weights,
        )


def candidate_phase_identity_log_likelihood_ratios(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscalePhaseIdentityPrediction,
    source_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize fixed support-view mass without reassigning unavailable mass."""

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime) or not isinstance(
        prediction, CandidateMultiscalePhaseIdentityPrediction
    ):
        raise ValueError("phase identity candidate scoring requires runtime and prediction")
    active = runtime.to(prediction.edge_log_likelihood_ratios.device)
    if active.candidate_view_weights.shape != prediction.edge_log_likelihood_ratios.shape:
        raise ValueError("phase identity runtime and prediction layouts differ")
    if source_name is None:
        values = prediction.edge_log_likelihood_ratios
        usable = prediction.edge_usable
    else:
        name = str(source_name)
        if name not in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
            raise ValueError("phase identity source name is invalid")
        values = prediction.source_edge_log_likelihood_ratios[name].to(values_device := prediction.edge_log_likelihood_ratios.device)
        usable = prediction.source_edge_usable[name].to(values_device)
    weights = active.candidate_view_weights.to(dtype=values.dtype)
    if torch.any(weights < 0.0) or not torch.isfinite(weights).all():
        raise ValueError("phase identity fixed view weights are invalid")
    log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    # Missing views retain their immutable mass at neutral LLR=0.  Removing
    # them then renormalizing would allow a pose to discard contrary evidence.
    effective = torch.where(usable, values, torch.zeros_like(values))
    candidate_llr = torch.logsumexp(log_weights + effective, dim=2)
    candidate_usable = torch.any(usable & (weights > 0.0), dim=2)
    if not torch.isfinite(candidate_llr).all():
        raise RuntimeError("phase identity candidate mixture is non-finite")
    return candidate_llr, candidate_usable


def candidate_phase_identity_plus_null_logits(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscalePhaseIdentityPrediction,
    candidate_prior_logit_weight: float = 1.0,
    source_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add fixed top-L/null mass only after target-free visual inference."""

    weight = float(candidate_prior_logit_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("phase identity candidate prior weight is invalid")
    candidate_llr, usable = candidate_phase_identity_log_likelihood_ratios(
        runtime=runtime, prediction=prediction, source_name=source_name
    )
    active = runtime.to(candidate_llr.device)
    prior = active.candidate_probabilities.to(dtype=candidate_llr.dtype)
    null = active.null_probabilities.to(dtype=candidate_llr.dtype)
    if (
        prior.shape != candidate_llr.shape
        or null.shape != (len(candidate_llr),)
        or torch.any(prior < 0.0)
        or torch.any(null < 0.0)
        or torch.any(torch.abs(prior.sum(dim=1) + null - 1.0) > 1e-4)
    ):
        raise ValueError("phase identity fixed candidate/null mass is invalid")
    if weight == 0.0:
        candidate_logits = candidate_llr
        null_logits = torch.zeros_like(null)
    else:
        candidate_logits = candidate_llr + weight * torch.log(
            prior.clamp_min(torch.finfo(prior.dtype).tiny)
        )
        null_logits = weight * torch.log(null.clamp_min(torch.finfo(null.dtype).tiny))
    return torch.cat([candidate_logits, null_logits[:, None]], dim=1), usable


def canonical_registered_identity_or_null_targets(
    *,
    observed_candidate_mask: torch.Tensor,
    candidate_dustbin_mask: torch.Tensor,
    candidate_supervised_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse the registered exact-track contract to one class per point.

    The target artifact is candidate-shaped because it is also consumed by
    local-density branches.  Candidate identity is a different task: a fully
    supervised row must contain exactly one observed track and dustbin labels
    for every other fixed candidate, or no observed track and dustbin labels
    for all candidates.  Partial geometric visibility is not an identity
    label, so this function rejects it instead of silently treating it as a
    null class.
    """

    observed = torch.as_tensor(observed_candidate_mask, dtype=torch.bool)
    dustbin = torch.as_tensor(candidate_dustbin_mask, dtype=torch.bool, device=observed.device)
    supervised = torch.as_tensor(
        candidate_supervised_mask, dtype=torch.bool, device=observed.device
    )
    if (
        observed.ndim != 2
        or dustbin.shape != observed.shape
        or supervised.shape != observed.shape
        or observed.shape[0] == 0
        or observed.shape[1] < 2
        or torch.any(observed & ~supervised)
        or torch.any(dustbin & ~supervised)
        or torch.any(observed.sum(dim=1) > 1)
    ):
        raise ValueError("registered exact identity candidate targets are invalid")
    fully_supervised = supervised.all(dim=1)
    fully_unsupervised = (~supervised).all(dim=1)
    if torch.any(~(fully_supervised | fully_unsupervised)):
        raise ValueError("registered exact identity targets must supervise a complete candidate row")
    has_observed = observed.any(dim=1)
    target_dustbin = dustbin.all(dim=1)
    expected_dustbin = torch.where(
        has_observed[:, None], ~observed, torch.ones_like(dustbin)
    )
    if torch.any(fully_supervised[:, None] & (dustbin != expected_dustbin)) or torch.any(
        fully_unsupervised[:, None] & (observed | dustbin)
    ):
        raise ValueError("registered exact identity targets have mixed geometric semantics")
    row_supervised = fully_supervised
    return observed, target_dustbin, row_supervised


def exact_identity_or_null_cross_entropy(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscalePhaseIdentityPrediction,
    observed_candidate_mask: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor,
    candidate_prior_logit_weight: float = 1.0,
    source_name: str | None = None,
    balance_observed_and_null: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train-only exact-track-or-null NLL after the visual forward.

    ``observed_candidate_mask`` must be the registered exact-track target, not
    a geometry projection visibility mask.  The stricter shape contract keeps
    that historically easy-to-miss semantic substitution from recurring.
    """

    observed = torch.as_tensor(observed_candidate_mask, dtype=torch.bool)
    dustbin = torch.as_tensor(target_dustbin, dtype=torch.bool, device=observed.device)
    supervised = torch.as_tensor(target_supervised, dtype=torch.bool, device=observed.device)
    logits, candidate_usable = candidate_phase_identity_plus_null_logits(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
        source_name=source_name,
    )
    observed = observed.to(device=logits.device)
    dustbin = dustbin.to(device=logits.device)
    supervised = supervised.to(device=logits.device)
    if (
        observed.shape != (len(logits), logits.shape[1] - 1)
        or dustbin.shape != (len(logits),)
        or supervised.shape != dustbin.shape
        or torch.any(observed & ~supervised[:, None])
        or torch.any(dustbin & ~supervised)
        or torch.any(observed.sum(dim=1) > 1)
        or torch.any(observed.any(dim=1) & dustbin)
    ):
        raise ValueError("phase identity exact target contract is invalid")
    labels = torch.where(
        dustbin,
        torch.full((len(logits),), logits.shape[1] - 1, dtype=torch.long, device=logits.device),
        observed.to(dtype=torch.long).argmax(dim=1),
    )
    observed_rows = observed.any(dim=1)
    target_candidate_usable = candidate_usable.gather(1, labels[:, None].clamp_max(
        candidate_usable.shape[1] - 1
    )).squeeze(1)
    # A null target needs at least one visual candidate in order to provide a
    # gradient.  A positive target must be visually available and have a real
    # competing candidate; otherwise CE can be satisfied by an untestable
    # one-edge-versus-fixed-null shortcut.
    active = supervised & torch.where(
        observed_rows,
        target_candidate_usable & (candidate_usable.sum(dim=1) >= 2),
        candidate_usable.any(dim=1),
    )
    if not bool(active.any()):
        return logits.sum() * 0.0, {
            "identity_active": 0.0,
            "identity_observed_active": 0.0,
            "identity_null_active": 0.0,
            "identity_top1": 0.0,
            "identity_nll": 0.0,
        }
    individual = F.cross_entropy(logits[active], labels[active], reduction="none")
    active_observed = observed_rows[active]
    if bool(balance_observed_and_null) and bool(active_observed.any()) and bool((~active_observed).any()):
        loss = 0.5 * (
            individual[active_observed].mean() + individual[~active_observed].mean()
        )
    else:
        loss = individual.mean()
    top1 = logits.argmax(dim=1)
    return loss, {
        "identity_active": float(active.sum().item()),
        "identity_observed_active": float((active & observed_rows).sum().item()),
        "identity_null_active": float((active & ~observed_rows).sum().item()),
        "identity_top1": float((top1[active] == labels[active]).float().mean().item()),
        "identity_nll": float(loss.detach().item()),
    }


def current_hard_repeat_identity_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscalePhaseIdentityPrediction,
    point_indices: torch.Tensor,
    positive_candidate_indices: torch.Tensor,
    negative_candidate_indices: torch.Tensor,
    margin: float,
    source_name: str | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train exact candidate identity against current coherent-repeat errors."""

    value = float(margin)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("phase identity hard-repeat margin is invalid")
    point = torch.as_tensor(point_indices, dtype=torch.long).reshape(-1)
    positive = torch.as_tensor(positive_candidate_indices, dtype=torch.long).reshape(-1)
    negative = torch.as_tensor(negative_candidate_indices, dtype=torch.long).reshape(-1)
    candidate_llr, candidate_usable = candidate_phase_identity_log_likelihood_ratios(
        runtime=runtime, prediction=prediction, source_name=source_name
    )
    point = point.to(device=candidate_llr.device)
    positive = positive.to(device=candidate_llr.device)
    negative = negative.to(device=candidate_llr.device)
    if (
        len(point) == 0
        or positive.shape != point.shape
        or negative.shape != point.shape
        or torch.any(point < 0)
        or torch.any(point >= len(candidate_llr))
        or torch.any(positive < 0)
        or torch.any(positive >= candidate_llr.shape[1])
        or torch.any(negative < 0)
        or torch.any(negative >= candidate_llr.shape[1])
        or torch.any(positive == negative)
    ):
        raise ValueError("phase identity hard-repeat indices are invalid")
    positive_values = candidate_llr[point, positive]
    negative_values = candidate_llr[point, negative]
    active = candidate_usable[point, positive] & candidate_usable[point, negative]
    if not bool(active.any()):
        return candidate_llr.sum() * 0.0, {
            "hard_repeat_active": 0.0,
            "hard_repeat_mean_gap": 0.0,
            "hard_repeat_win_fraction": 0.0,
            "hard_repeat_margin_loss": 0.0,
        }
    gaps = positive_values[active] - negative_values[active]
    loss = F.softplus(torch.as_tensor(value, device=gaps.device) - gaps).mean()
    return loss, {
        "hard_repeat_active": float(active.sum().item()),
        "hard_repeat_mean_gap": float(gaps.detach().mean().item()),
        "hard_repeat_win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
        "hard_repeat_margin_loss": float(loss.detach().item()),
    }
