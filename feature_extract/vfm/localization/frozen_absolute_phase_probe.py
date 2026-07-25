"""Target-free, candidate-specific multiscale absolute-phase evidence.

This module deliberately contains no learned matcher, pose target, or
candidate-label input.  It crops real-image descriptor grids at a candidate's
projected query position and at its fixed SfM support observation, retains a
bounded two-dimensional translation cost volume, and measures how much visual
mass remains at the declared zero-relative-phase mode.  The result is useful
only as a frozen P1 diagnostic until a later train-only coherent-wrong-pose
calibration has passed its own gate.

The important distinction from the V4/V5 raw-layout heads is that the score is
computed directly from the local translation field.  No learned linear head is
allowed to collapse the cost volume before the target-side pose-rank audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ContextAttentionRuntimeArrays,
    crop_anchor_aligned_grid_tokens,
)


FROZEN_ABSOLUTE_PHASE_PROBE_VERSION = "frozen_multiscale_absolute_phase_probe_v1"
FROZEN_ABSOLUTE_CONTEXT_CORRELATION_PROBE_VERSION = (
    "frozen_multiscale_absolute_context_correlation_probe_v2"
)

# ``radio_final_grid4`` and ``radio_final_grid8`` are pooled views of the
# same mapping-side RADIO-final cache.  They are deliberately separate source
# names: a 3x3 crop on grid4 represents a much larger absolute image context
# than a 3x3 crop on grid16.  Existing phase-v1 artifacts use only the last
# three names and remain fully reproducible.
PHASE_PROBE_SOURCE_NAMES = (
    "radio_final_grid4",
    "radio_final_grid8",
    "radio_final",
    "radio_intermediate",
    "alike",
)


@dataclass(frozen=True)
class AbsolutePhaseProfile:
    """One immutable source/window/cost-volume specification for P1.

    ``center_cosine`` is the descriptor-only baseline.  ``translation_phase``
    computes a local two-dimensional translation distribution and returns the
    log ratio of its zero-phase neighbourhood mass against the same valid
    translation lattice under a uniform distribution.  It is not a learned
    identity posterior.
    """

    name: str
    source_name: str
    window_size: int
    score_kind: str
    maximum_translation_cells: int
    temperature: float
    alignment_sigma_cells: float
    minimum_zero_shift_coverage: float = 0.55
    log_ratio_clip: float = 4.0

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or str(self.source_name) not in PHASE_PROBE_SOURCE_NAMES
            or int(self.window_size) <= 0
            or int(self.window_size) % 2 != 1
            or str(self.score_kind)
            not in {"center_cosine", "translation_phase", "local_bipartite"}
            or float(self.temperature) <= 0.0
            or not 0.0 < float(self.minimum_zero_shift_coverage) <= 1.0
            or float(self.log_ratio_clip) <= 0.0
        ):
            raise ValueError("absolute-phase profile is invalid")
        radius = int(self.window_size) // 2
        if str(self.score_kind) == "center_cosine":
            if int(self.maximum_translation_cells) != 0 or int(self.window_size) != 1:
                raise ValueError("center-cosine phase profile must use one token")
        elif (
            int(self.maximum_translation_cells) <= 0
            or int(self.maximum_translation_cells) > radius
            or float(self.alignment_sigma_cells) <= 0.0
        ):
            raise ValueError("translation-phase profile has invalid local geometry")


# These values are declared before any validation/test pose target is read.
# RADIO final/intermediate provide identity/context evidence.  ALIKE is kept
# only as a local-detail reference profile; it is not a promotion candidate
# unless a later independent audit demonstrates complementary signal.
FROZEN_ABSOLUTE_PHASE_PROFILES = (
    AbsolutePhaseProfile(
        "radio_final_center",
        "radio_final",
        1,
        "center_cosine",
        0,
        0.10,
        0.0,
    ),
    AbsolutePhaseProfile(
        "radio_final_phase3",
        "radio_final",
        3,
        "translation_phase",
        1,
        0.10,
        0.75,
    ),
    AbsolutePhaseProfile(
        "radio_final_phase5",
        "radio_final",
        5,
        "translation_phase",
        2,
        0.10,
        0.90,
    ),
    AbsolutePhaseProfile(
        "radio_intermediate_center",
        "radio_intermediate",
        1,
        "center_cosine",
        0,
        0.10,
        0.0,
    ),
    AbsolutePhaseProfile(
        "radio_intermediate_phase5",
        "radio_intermediate",
        5,
        "translation_phase",
        2,
        0.10,
        0.90,
    ),
    AbsolutePhaseProfile(
        "radio_intermediate_phase9",
        "radio_intermediate",
        9,
        "translation_phase",
        2,
        0.10,
        0.90,
    ),
    AbsolutePhaseProfile(
        "alike_center",
        "alike",
        1,
        "center_cosine",
        0,
        0.07,
        0.0,
    ),
)


# This is a separate P1 profile set, not a learned replacement for phase-v1.
# Its local score keeps a separate shift distribution for every crop token and
# then averages both query-to-support and support-to-query evidence.  A global
# shift can score an entire repeated facade highly even when individual tokens
# admit several plausible phases; the local bidirectional field explicitly
# exposes that ambiguity.  Grid4/grid8 RADIO-final features add the large
# absolute context that a grid16-only crop cannot see.
FROZEN_ABSOLUTE_CONTEXT_CORRELATION_PROFILES = (
    AbsolutePhaseProfile(
        "radio_final_grid4_local_bipartite3",
        "radio_final_grid4",
        3,
        "local_bipartite",
        1,
        0.10,
        0.75,
        minimum_zero_shift_coverage=0.35,
    ),
    AbsolutePhaseProfile(
        "radio_final_grid8_local_bipartite5",
        "radio_final_grid8",
        5,
        "local_bipartite",
        1,
        0.10,
        0.85,
        minimum_zero_shift_coverage=0.45,
    ),
    AbsolutePhaseProfile(
        "radio_final_grid16_local_bipartite5",
        "radio_final",
        5,
        "local_bipartite",
        2,
        0.10,
        0.90,
    ),
    AbsolutePhaseProfile(
        "radio_intermediate_local_bipartite7",
        "radio_intermediate",
        7,
        "local_bipartite",
        2,
        0.10,
        1.00,
    ),
    AbsolutePhaseProfile(
        "radio_intermediate_local_bipartite9",
        "radio_intermediate",
        9,
        "local_bipartite",
        2,
        0.10,
        1.05,
    ),
    AbsolutePhaseProfile(
        "alike_local_bipartite5_diagnostic",
        "alike",
        5,
        "local_bipartite",
        2,
        0.07,
        0.80,
    ),
)


def profiles_by_source(
    profiles: Sequence[AbsolutePhaseProfile] = FROZEN_ABSOLUTE_PHASE_PROFILES,
) -> dict[str, tuple[AbsolutePhaseProfile, ...]]:
    """Group immutable profiles while rejecting duplicate or empty layouts."""

    values = tuple(profiles)
    if not values or len({item.name for item in values}) != len(values):
        raise ValueError("absolute-phase profiles must have unique non-empty names")
    grouped: dict[str, list[AbsolutePhaseProfile]] = {}
    for profile in values:
        grouped.setdefault(str(profile.source_name), []).append(profile)
    return {name: tuple(items) for name, items in grouped.items()}


@dataclass(frozen=True)
class PairwiseDescriptorCrops:
    """Descriptor crops for independent candidate/support-view edges."""

    query_descriptors: torch.Tensor
    query_valid: torch.Tensor
    support_descriptors: torch.Tensor
    support_valid: torch.Tensor
    edge_valid: torch.Tensor

    def __post_init__(self) -> None:
        query = torch.as_tensor(self.query_descriptors)
        support = torch.as_tensor(self.support_descriptors, device=query.device)
        query_valid = torch.as_tensor(self.query_valid, dtype=torch.bool, device=query.device)
        support_valid = torch.as_tensor(self.support_valid, dtype=torch.bool, device=query.device)
        edge_valid = torch.as_tensor(self.edge_valid, dtype=torch.bool, device=query.device)
        if (
            query.ndim != 3
            or support.shape != query.shape
            or query_valid.shape != query.shape[:2]
            or support_valid.shape != query.shape[:2]
            or edge_valid.shape != (query.shape[0],)
            or query.shape[1] == 0
            or int(round(float(query.shape[1]) ** 0.5)) ** 2 != query.shape[1]
        ):
            raise ValueError("pairwise absolute-phase descriptor crops are incompatible")
        object.__setattr__(self, "query_descriptors", query)
        object.__setattr__(self, "query_valid", query_valid)
        object.__setattr__(self, "support_descriptors", support)
        object.__setattr__(self, "support_valid", support_valid)
        object.__setattr__(self, "edge_valid", edge_valid)

    @property
    def window_size(self) -> int:
        return int(round(float(self.query_descriptors.shape[1]) ** 0.5))


@dataclass(frozen=True)
class AbsolutePhaseEvidence:
    """Raw target-free feature evidence for a candidate/support-view edge."""

    log_ratios: torch.Tensor
    available: torch.Tensor
    zero_shift_coverage: torch.Tensor

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.log_ratios)
        available = torch.as_tensor(self.available, dtype=torch.bool, device=values.device)
        coverage = torch.as_tensor(
            self.zero_shift_coverage, dtype=values.dtype, device=values.device
        )
        if (
            values.ndim != 1
            or available.shape != values.shape
            or coverage.shape != values.shape
            or not bool(torch.isfinite(values).all())
            or not bool(torch.isfinite(coverage).all())
            or torch.any(coverage < 0.0)
            or torch.any(coverage > 1.0 + 1e-6)
        ):
            raise ValueError("absolute-phase evidence is invalid")
        object.__setattr__(self, "log_ratios", values)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "zero_shift_coverage", coverage)


class FrozenCandidateMultiscaleCropRuntime(nn.Module):
    """Fixed real-image grids and support observations for a P1 pose probe.

    The runtime intentionally owns only descriptor grids and frozen SfM support
    coordinates.  It has no model parameters, candidate posterior, or original
    query-anchor descriptor.  Calling :meth:`pairwise_crops_at_query_xy` is
    therefore target-free and candidate/view permutation equivariant.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        runtime: ContextAttentionRuntimeArrays,
    ) -> None:
        super().__init__()
        source_names = tuple(str(name) for name in sources)
        if (
            not source_names
            or len(source_names) != len(set(source_names))
            or not set(source_names).issubset(PHASE_PROBE_SOURCE_NAMES)
        ):
            raise ValueError("absolute-phase source set is invalid")
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        query_indices = torch.as_tensor(runtime.query_image_indices, dtype=torch.long)
        support_indices = torch.as_tensor(runtime.support_image_indices, dtype=torch.long)
        support_xy = torch.as_tensor(runtime.support_xy, dtype=torch.float32)
        view_valid = torch.as_tensor(runtime.view_valid, dtype=torch.bool)
        row_count, candidate_count, view_count = support_indices.shape
        if (
            sizes.ndim != 2
            or sizes.shape[1] != 2
            or torch.any(sizes <= 1.0)
            or query_indices.shape != (row_count,)
            or support_xy.shape != (row_count, candidate_count, view_count, 2)
            or view_valid.shape != support_indices.shape
            or torch.any(query_indices < 0)
            or torch.any(support_indices < 0)
            or torch.any(query_indices >= len(sizes))
            or torch.any(support_indices >= len(sizes))
            or not bool(torch.isfinite(support_xy).all())
        ):
            raise ValueError("absolute-phase runtime arrays are incompatible")
        for name in source_names:
            grid = torch.as_tensor(sources[name])
            if (
                grid.ndim != 4
                or grid.shape[0] != len(sizes)
                or grid.shape[1] != grid.shape[2]
                or grid.shape[1] <= 0
                or grid.shape[-1] < 2
                or not bool(torch.isfinite(grid).all())
            ):
                raise ValueError(f"absolute-phase source {name} is invalid")
            self.register_buffer(f"_grid_{name}", grid.to(dtype=torch.float16), persistent=False)
        self._source_names = source_names
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.register_buffer("_query_image_indices", query_indices, persistent=False)
        self.register_buffer("_support_image_indices", support_indices, persistent=False)
        self.register_buffer("_support_xy", support_xy, persistent=False)
        self.register_buffer("_view_valid", view_valid, persistent=False)

    @property
    def row_count(self) -> int:
        return int(self._query_image_indices.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self._support_image_indices.shape[1])

    @property
    def view_count(self) -> int:
        return int(self._support_image_indices.shape[2])

    def grid_size(self, source_name: str) -> int:
        name = str(source_name)
        if name not in self._source_names:
            raise ValueError("absolute-phase source name is unknown")
        return int(getattr(self, f"_grid_{name}").shape[1])

    def pairwise_crops_at_query_xy(
        self,
        rows: torch.Tensor,
        query_xy_by_candidate: torch.Tensor,
        *,
        source_windows: Mapping[str, int],
    ) -> tuple[dict[str, PairwiseDescriptorCrops], torch.Tensor, torch.Tensor]:
        """Crop fixed support views and dynamic candidate-projected query sites.

        ``query_xy_by_candidate`` has no access to target data.  It may be
        supplied once per candidate and is expanded identically across that
        candidate's fixed support views.  Support views are never averaged.
        """

        if not source_windows or set(source_windows) - set(self._source_names):
            raise ValueError("absolute-phase crop source windows are invalid")
        selected_rows = torch.as_tensor(rows, dtype=torch.long, device=self._image_sizes.device)
        if (
            selected_rows.ndim != 1
            or selected_rows.numel() == 0
            or torch.any(selected_rows < 0)
            or torch.any(selected_rows >= self.row_count)
        ):
            raise ValueError("absolute-phase crop row indices are invalid")
        batch = int(len(selected_rows))
        candidate_count = self.candidate_count
        view_count = self.view_count
        coordinates = torch.as_tensor(
            query_xy_by_candidate, dtype=torch.float32, device=self._image_sizes.device
        )
        if coordinates.ndim == 3 and coordinates.shape == (batch, candidate_count, 2):
            coordinates = coordinates[:, :, None, :].expand(-1, -1, view_count, -1)
        if (
            coordinates.shape != (batch, candidate_count, view_count, 2)
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise ValueError("absolute-phase dynamic query coordinates are incompatible")
        query_indices = self._query_image_indices.index_select(0, selected_rows)
        support_indices = self._support_image_indices.index_select(0, selected_rows)
        support_xy = self._support_xy.index_select(0, selected_rows)
        view_valid = self._view_valid.index_select(0, selected_rows)
        edge_count = int(batch * candidate_count * view_count)
        edge_query_indices = (
            query_indices[:, None, None].expand(-1, candidate_count, view_count).reshape(-1)
        )
        edge_sizes = self._image_sizes.index_select(0, edge_query_indices).reshape(
            batch, candidate_count, view_count, 2
        )
        projection_in_image = (
            (coordinates[..., 0] >= 0.0)
            & (coordinates[..., 0] <= edge_sizes[..., 0] - 1.0)
            & (coordinates[..., 1] >= 0.0)
            & (coordinates[..., 1] <= edge_sizes[..., 1] - 1.0)
        )
        flat_coordinates = coordinates.reshape(edge_count, 2)
        flat_support_indices = support_indices.reshape(-1)
        flat_support_xy = support_xy.reshape(-1, 2)
        flat_edge_valid = view_valid.reshape(-1)
        output: dict[str, PairwiseDescriptorCrops] = {}
        for source_name, raw_window in source_windows.items():
            name = str(source_name)
            window_size = int(raw_window)
            grid = getattr(self, f"_grid_{name}")
            if (
                window_size <= 0
                or window_size % 2 != 1
                or window_size > int(grid.shape[1])
            ):
                raise ValueError(f"absolute-phase window for {name} is invalid")
            query, query_valid = crop_anchor_aligned_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=edge_query_indices,
                xy=flat_coordinates,
                window_size=window_size,
            )
            support, support_valid = crop_anchor_aligned_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=flat_support_indices,
                xy=flat_support_xy,
                window_size=window_size,
            )
            output[name] = PairwiseDescriptorCrops(
                query_descriptors=query,
                query_valid=query_valid,
                support_descriptors=support,
                support_valid=support_valid & flat_edge_valid[:, None],
                edge_valid=flat_edge_valid,
            )
        return output, view_valid, projection_in_image


def deterministic_support_channel_permutation(
    *, source_name: str, descriptor_dim: int, device: torch.device
) -> torch.Tensor:
    """Return one non-identity, target-free support-channel control permutation."""

    if str(source_name) not in PHASE_PROBE_SOURCE_NAMES or int(descriptor_dim) < 2:
        raise ValueError("support-channel permutation dimensions are invalid")
    digest = hashlib.sha256(
        f"{FROZEN_ABSOLUTE_PHASE_PROBE_VERSION}:{source_name}:{int(descriptor_dim)}".encode()
    ).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(int(descriptor_dim), generator=generator)
    if torch.equal(permutation, torch.arange(int(descriptor_dim))):
        permutation = torch.roll(permutation, shifts=1)
    return permutation.to(device=device)


def _central_subcrop(crops: PairwiseDescriptorCrops, *, window_size: int) -> PairwiseDescriptorCrops:
    """Select a centered square subcrop without changing descriptor sampling."""

    requested = int(window_size)
    original = crops.window_size
    if requested <= 0 or requested % 2 != 1 or requested > original:
        raise ValueError("absolute-phase subcrop window is invalid")
    if requested == original:
        return crops
    start = (original - requested) // 2
    end = start + requested
    count = len(crops.query_descriptors)
    dimension = int(crops.query_descriptors.shape[-1])
    query = crops.query_descriptors.reshape(count, original, original, dimension)[:, start:end, start:end]
    support = crops.support_descriptors.reshape(count, original, original, dimension)[:, start:end, start:end]
    query_valid = crops.query_valid.reshape(count, original, original)[:, start:end, start:end]
    support_valid = crops.support_valid.reshape(count, original, original)[:, start:end, start:end]
    return PairwiseDescriptorCrops(
        query_descriptors=query.reshape(count, requested * requested, dimension),
        query_valid=query_valid.reshape(count, requested * requested),
        support_descriptors=support.reshape(count, requested * requested, dimension),
        support_valid=support_valid.reshape(count, requested * requested),
        edge_valid=crops.edge_valid,
    )


def _apply_support_channel_permutation(
    crops: PairwiseDescriptorCrops, *, permutation: torch.Tensor | None
) -> PairwiseDescriptorCrops:
    if permutation is None:
        return crops
    indices = torch.as_tensor(permutation, dtype=torch.long, device=crops.support_descriptors.device)
    if (
        indices.ndim != 1
        or len(indices) != crops.support_descriptors.shape[-1]
        or len(torch.unique(indices)) != len(indices)
        or torch.any(indices < 0)
        or torch.any(indices >= len(indices))
    ):
        raise ValueError("support-channel permutation is incompatible with descriptor crops")
    return PairwiseDescriptorCrops(
        query_descriptors=crops.query_descriptors,
        query_valid=crops.query_valid,
        support_descriptors=crops.support_descriptors.index_select(2, indices),
        support_valid=crops.support_valid,
        edge_valid=crops.edge_valid,
    )


def _center_cosine_evidence(
    crops: PairwiseDescriptorCrops, *, profile: AbsolutePhaseProfile
) -> AbsolutePhaseEvidence:
    center = int(crops.query_descriptors.shape[1]) // 2
    query = F.normalize(crops.query_descriptors[:, center].float(), dim=1)
    support = F.normalize(crops.support_descriptors[:, center].float(), dim=1)
    available = (
        crops.edge_valid
        & crops.query_valid[:, center]
        & crops.support_valid[:, center]
    )
    values = torch.sum(query * support, dim=1) / float(profile.temperature)
    values = values.clamp(-float(profile.log_ratio_clip), float(profile.log_ratio_clip))
    values = torch.where(available, values, torch.zeros_like(values))
    coverage = available.to(dtype=values.dtype)
    return AbsolutePhaseEvidence(
        log_ratios=values.to(dtype=crops.query_descriptors.dtype),
        available=available,
        zero_shift_coverage=coverage.to(dtype=crops.query_descriptors.dtype),
    )


def _translation_phase_evidence(
    crops: PairwiseDescriptorCrops, *, profile: AbsolutePhaseProfile
) -> AbsolutePhaseEvidence:
    """Score local zero-relative-phase mass from a bounded 2-D cost volume."""

    size = crops.window_size
    maximum_shift = int(profile.maximum_translation_cells)
    if maximum_shift <= 0 or maximum_shift > size // 2:
        raise ValueError("translation-phase profile exceeds its crop")
    count = len(crops.query_descriptors)
    dimension = int(crops.query_descriptors.shape[-1])
    query = F.normalize(crops.query_descriptors.float(), dim=2).reshape(
        count, size, size, dimension
    )
    support = F.normalize(crops.support_descriptors.float(), dim=2).reshape(
        count, size, size, dimension
    )
    query_valid = crops.query_valid.reshape(count, size, size)
    support_valid = crops.support_valid.reshape(count, size, size)
    scores: list[torch.Tensor] = []
    valid_shifts: list[torch.Tensor] = []
    coverages: list[torch.Tensor] = []
    squared_distances: list[float] = []
    zero_index = -1
    for shift_row in range(-maximum_shift, maximum_shift + 1):
        for shift_column in range(-maximum_shift, maximum_shift + 1):
            query_row_start = max(0, -shift_row)
            query_row_end = min(size, size - shift_row)
            query_column_start = max(0, -shift_column)
            query_column_end = min(size, size - shift_column)
            support_row_start = query_row_start + shift_row
            support_row_end = query_row_end + shift_row
            support_column_start = query_column_start + shift_column
            support_column_end = query_column_end + shift_column
            query_cells = query[
                :, query_row_start:query_row_end, query_column_start:query_column_end
            ]
            support_cells = support[
                :, support_row_start:support_row_end, support_column_start:support_column_end
            ]
            valid = (
                query_valid[:, query_row_start:query_row_end, query_column_start:query_column_end]
                & support_valid[
                    :, support_row_start:support_row_end, support_column_start:support_column_end
                ]
            )
            pair_count = valid.sum(dim=(1, 2)).to(dtype=query.dtype)
            cosine = torch.sum(query_cells * support_cells, dim=3)
            score = torch.sum(cosine * valid.to(dtype=cosine.dtype), dim=(1, 2)) / pair_count.clamp_min(1.0)
            scores.append(score)
            valid_shifts.append(pair_count > 0.0)
            coverages.append(pair_count / float(size * size))
            squared_distances.append(float(shift_row * shift_row + shift_column * shift_column))
            if shift_row == 0 and shift_column == 0:
                zero_index = len(scores) - 1
    if zero_index < 0:
        raise RuntimeError("translation-phase cost volume has no zero shift")
    score_values = torch.stack(scores, dim=1)
    shift_valid = torch.stack(valid_shifts, dim=1)
    zero_coverage = torch.stack(coverages, dim=1)[:, zero_index]
    masked_scores = torch.where(
        shift_valid,
        score_values,
        torch.full_like(score_values, -torch.inf),
    )
    probabilities = torch.softmax(masked_scores / float(profile.temperature), dim=1)
    probabilities = torch.where(shift_valid, probabilities, torch.zeros_like(probabilities))
    distances = torch.as_tensor(
        squared_distances, dtype=probabilities.dtype, device=probabilities.device
    )
    alignment = torch.exp(
        -0.5 * distances / float(profile.alignment_sigma_cells) ** 2
    )
    alignment_mass = torch.sum(probabilities * alignment[None], dim=1)
    valid_count = shift_valid.sum(dim=1).to(dtype=probabilities.dtype)
    uniform_alignment_mass = torch.sum(
        shift_valid.to(dtype=probabilities.dtype) * alignment[None], dim=1
    ) / valid_count.clamp_min(1.0)
    available = (
        crops.edge_valid
        & shift_valid[:, zero_index]
        & (zero_coverage >= float(profile.minimum_zero_shift_coverage))
    )
    values = torch.log(alignment_mass.clamp_min(torch.finfo(probabilities.dtype).tiny)) - torch.log(
        uniform_alignment_mass.clamp_min(torch.finfo(probabilities.dtype).tiny)
    )
    values = values.clamp(-float(profile.log_ratio_clip), float(profile.log_ratio_clip))
    values = torch.where(available, values, torch.zeros_like(values))
    return AbsolutePhaseEvidence(
        log_ratios=values.to(dtype=crops.query_descriptors.dtype),
        available=available,
        zero_shift_coverage=zero_coverage.to(dtype=crops.query_descriptors.dtype),
    )


def _directional_local_bipartite_log_ratio(
    *,
    query: torch.Tensor,
    support: torch.Tensor,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    profile: AbsolutePhaseProfile,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate position-sensitive local correspondence in one direction.

    Every query crop token owns a separate bounded displacement distribution.
    This differs from :func:`_translation_phase_evidence`, which first pools a
    whole crop for each global shift.  It therefore represents a small local
    affine/deformation field without collapsing repeated local modes into one
    crop-level average.  The returned log ratio compares visual mass around
    zero relative phase to the same valid displacement lattice under a
    uniform match distribution.
    """

    if (
        query.ndim != 4
        or support.shape != query.shape
        or query_valid.shape != query.shape[:3]
        or support_valid.shape != query.shape[:3]
    ):
        raise ValueError("local bipartite descriptor tensors are incompatible")
    count, size, _, _dimension = query.shape
    maximum_shift = int(profile.maximum_translation_cells)
    if maximum_shift <= 0 or maximum_shift > size // 2:
        raise ValueError("local bipartite profile exceeds its crop")

    shift_scores: list[torch.Tensor] = []
    shift_valid: list[torch.Tensor] = []
    squared_distances: list[float] = []
    zero_index = -1
    for shift_row in range(-maximum_shift, maximum_shift + 1):
        for shift_column in range(-maximum_shift, maximum_shift + 1):
            query_row_start = max(0, -shift_row)
            query_row_end = min(size, size - shift_row)
            query_column_start = max(0, -shift_column)
            query_column_end = min(size, size - shift_column)
            support_row_start = query_row_start + shift_row
            support_row_end = query_row_end + shift_row
            support_column_start = query_column_start + shift_column
            support_column_end = query_column_end + shift_column
            pair_query = query[
                :, query_row_start:query_row_end, query_column_start:query_column_end
            ]
            pair_support = support[
                :, support_row_start:support_row_end, support_column_start:support_column_end
            ]
            pair_valid = (
                query_valid[
                    :, query_row_start:query_row_end, query_column_start:query_column_end
                ]
                & support_valid[
                    :, support_row_start:support_row_end, support_column_start:support_column_end
                ]
            )
            full_scores = torch.zeros((count, size, size), dtype=query.dtype, device=query.device)
            full_valid = torch.zeros((count, size, size), dtype=torch.bool, device=query.device)
            full_scores[
                :, query_row_start:query_row_end, query_column_start:query_column_end
            ] = torch.sum(pair_query * pair_support, dim=3)
            full_valid[
                :, query_row_start:query_row_end, query_column_start:query_column_end
            ] = pair_valid
            shift_scores.append(full_scores)
            shift_valid.append(full_valid)
            squared_distances.append(float(shift_row * shift_row + shift_column * shift_column))
            if shift_row == 0 and shift_column == 0:
                zero_index = len(shift_scores) - 1
    if zero_index < 0:
        raise RuntimeError("local bipartite field has no zero shift")

    scores = torch.stack(shift_scores, dim=1)
    valid = torch.stack(shift_valid, dim=1)
    has_any = valid.any(dim=1)
    masked_scores = torch.where(valid, scores, torch.full_like(scores, -torch.inf))
    # A location outside both crops has no evidence.  Replace its all-inf row
    # before softmax, then mask it out below rather than emitting a NaN.
    safe_scores = torch.where(has_any[:, None], masked_scores, torch.zeros_like(masked_scores))
    probabilities = torch.softmax(safe_scores / float(profile.temperature), dim=1)
    probabilities = torch.where(valid, probabilities, torch.zeros_like(probabilities))
    distances = torch.as_tensor(
        squared_distances, dtype=probabilities.dtype, device=probabilities.device
    )
    alignment = torch.exp(
        -0.5 * distances / float(profile.alignment_sigma_cells) ** 2
    )[:, None, None]
    alignment_mass = torch.sum(probabilities * alignment[None], dim=1)
    valid_count = valid.sum(dim=1).to(dtype=probabilities.dtype)
    uniform_alignment_mass = torch.sum(
        valid.to(dtype=probabilities.dtype) * alignment[None], dim=1
    ) / valid_count.clamp_min(1.0)
    token_values = torch.log(alignment_mass.clamp_min(torch.finfo(probabilities.dtype).tiny)) - torch.log(
        uniform_alignment_mass.clamp_min(torch.finfo(probabilities.dtype).tiny)
    )
    zero_valid = valid[:, zero_index]
    token_active = has_any & zero_valid
    active_count = token_active.sum(dim=(1, 2))
    values = (token_values * token_active.to(dtype=token_values.dtype)).sum(dim=(1, 2)) / active_count.clamp_min(
        1
    ).to(dtype=token_values.dtype)
    coverage = active_count.to(dtype=token_values.dtype) / float(size * size)
    return values, coverage, active_count > 0


def _local_bipartite_evidence(
    crops: PairwiseDescriptorCrops, *, profile: AbsolutePhaseProfile
) -> AbsolutePhaseEvidence:
    """Bidirectionally score a bounded per-token 2-D correspondence field."""

    size = crops.window_size
    count = len(crops.query_descriptors)
    dimension = int(crops.query_descriptors.shape[-1])
    query = F.normalize(crops.query_descriptors.float(), dim=2).reshape(
        count, size, size, dimension
    )
    support = F.normalize(crops.support_descriptors.float(), dim=2).reshape(
        count, size, size, dimension
    )
    query_valid = crops.query_valid.reshape(count, size, size)
    support_valid = crops.support_valid.reshape(count, size, size)
    forward, forward_coverage, forward_available = _directional_local_bipartite_log_ratio(
        query=query,
        support=support,
        query_valid=query_valid,
        support_valid=support_valid,
        profile=profile,
    )
    reverse, reverse_coverage, reverse_available = _directional_local_bipartite_log_ratio(
        query=support,
        support=query,
        query_valid=support_valid,
        support_valid=query_valid,
        profile=profile,
    )
    coverage = torch.minimum(forward_coverage, reverse_coverage)
    available = (
        crops.edge_valid
        & forward_available
        & reverse_available
        & (coverage >= float(profile.minimum_zero_shift_coverage))
    )
    values = 0.5 * (forward + reverse)
    values = values.clamp(-float(profile.log_ratio_clip), float(profile.log_ratio_clip))
    values = torch.where(available, values, torch.zeros_like(values))
    return AbsolutePhaseEvidence(
        log_ratios=values.to(dtype=crops.query_descriptors.dtype),
        available=available,
        zero_shift_coverage=coverage.to(dtype=crops.query_descriptors.dtype),
    )


def evaluate_absolute_phase_profile(
    crops: PairwiseDescriptorCrops,
    *,
    profile: AbsolutePhaseProfile,
    support_channel_permutation: torch.Tensor | None = None,
) -> AbsolutePhaseEvidence:
    """Evaluate one profile directly from frozen descriptor crops.

    Passing a permutation produces the paired descriptor-destroying control.
    It changes no candidate, view, query crop, support position, prior, or
    geometric visibility condition.
    """

    selected = _central_subcrop(crops, window_size=int(profile.window_size))
    selected = _apply_support_channel_permutation(
        selected, permutation=support_channel_permutation
    )
    if str(profile.score_kind) == "center_cosine":
        return _center_cosine_evidence(selected, profile=profile)
    if str(profile.score_kind) == "translation_phase":
        return _translation_phase_evidence(selected, profile=profile)
    if str(profile.score_kind) == "local_bipartite":
        return _local_bipartite_evidence(selected, profile=profile)
    raise RuntimeError("unreachable absolute-phase profile kind")
