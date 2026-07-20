"""Frozen multiscale, candidate-specific context-attention probe utilities.

This module deliberately sits before pose scoring.  It consumes only a fixed
query token, fixed top-L landmark candidates, their fixed SfM support views,
and real-image descriptor grids.  It never receives a query pose, residual,
or a target label at inference time.

The context-only family masks the central 3x3 token neighbourhood in both
images.  Consequently, any gain from that family must come from the larger
candidate-specific image layout rather than a recalibration of the original
anchor descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)
from feature_extract.vfm.localization.radio_final_context import (
    RADIO_FINAL_CONTEXT_PCA_FORMAT,
    load_radio_final_context_pca_cache,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    load_spatial_image_context_cache,
)


CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT = (
    "multiscale_context_attention_candidate_probe_contract_v1"
)
CONTEXT_ATTENTION_FAMILIES = (
    "context_attention_multiscale_context_only",
    "context_attention_multiscale_with_anchor",
)
ABSOLUTE_PHASE_CONTEXT_ATTENTION_FAMILIES = (
    "absolute_phase_attention_position_only",
    "absolute_phase_attention_visual_context_only",
    "absolute_phase_attention_with_anchor",
)
SUPPORTED_CONTEXT_ATTENTION_FAMILIES = frozenset(
    {*CONTEXT_ATTENTION_FAMILIES, *ABSOLUTE_PHASE_CONTEXT_ATTENTION_FAMILIES}
)
ANCHOR_RELATIVE_POSITION_ENCODING = "anchor_relative_v1"
ABSOLUTE_DUAL_FRAME_POSITION_ENCODING = "absolute_dual_frame_v1"
SUPPORTED_CONTEXT_POSITION_ENCODINGS = frozenset(
    {ANCHOR_RELATIVE_POSITION_ENCODING, ABSOLUTE_DUAL_FRAME_POSITION_ENCODING}
)
CONTEXT_ATTENTION_CENTER_MASK_RADIUS = 1
_RADIO_INTERMEDIATE_CONTEXT_PCA_FORMAT = "radio_intermediate_image_context_pca_v1"
_ALIKE_SPATIAL_CONTEXT_FORMAT = "alike_image_spatial_context_v1"
# Both scopes are fitted entirely on mapping-side images.  The latter is the
# stricter operational form: it explicitly excludes every query split rather
# than relying on the historical mapping-train alias.
_SAFE_PCA_FIT_SCOPES = frozenset(
    {
        "mapping_train_images_only",
        "mapping_support_images_excluding_all_query_splits_v1",
    }
)


def load_context_attention_frozen_layout(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load only the frozen layout fields needed by direct token attention.

    Some diagnostic layout artifacts contain multi-gigabyte correlation tensors.
    This probe intentionally does not consume those already-compressed features,
    so materializing them merely to validate candidate lineage would prevent
    multi-process fitting.  The selected arrays below fully define the fixed
    candidate/support-view contract and are independently hashed by the caller.
    """

    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "xy",
        "candidate_track_ids",
        "candidate_canonical_rows",
        "candidate_view_valid",
        "candidate_support_image_ids",
        "candidate_support_coverage_counts",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"context-attention frozen layout lacks {sorted(missing)}")
        if "labels" in data.files:
            raise ValueError("context-attention frozen layout unexpectedly contains labels")
        try:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("context-attention frozen layout metadata is invalid") from error
        arrays = {
            "source_row_indices": np.asarray(data["source_row_indices"], dtype=np.int64),
            "query_ids": np.asarray(data["query_ids"]).astype(str),
            "split_names": np.asarray(data["split_names"]).astype(str),
            "xy": np.asarray(data["xy"], dtype=np.float32),
            "candidate_track_ids": np.asarray(data["candidate_track_ids"], dtype=np.int64),
            "candidate_canonical_rows": np.asarray(data["candidate_canonical_rows"], dtype=np.int64),
            "candidate_view_valid": np.asarray(data["candidate_view_valid"], dtype=bool),
            "candidate_support_image_ids": np.asarray(data["candidate_support_image_ids"]).astype(str),
            "candidate_support_coverage_counts": np.asarray(
                data["candidate_support_coverage_counts"], dtype=np.int32
            ),
        }
    if not isinstance(metadata, dict):
        raise ValueError("context-attention frozen layout metadata is not an object")
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("whole_image_summary_or_global_used", True))
        or bool(metadata.get("render", False))
        or metadata.get("is_complete_frozen_layout") is not True
    ):
        raise ValueError("context-attention frozen layout violates the fixed local protocol")
    rows = arrays["source_row_indices"].reshape(-1)
    query_ids = arrays["query_ids"].reshape(-1)
    split_names = arrays["split_names"].reshape(-1)
    xy = arrays["xy"]
    tracks = arrays["candidate_track_ids"]
    canonical = arrays["candidate_canonical_rows"]
    views = arrays["candidate_view_valid"]
    support_ids = arrays["candidate_support_image_ids"]
    coverage = arrays["candidate_support_coverage_counts"]
    if (
        len(rows) == 0
        or np.unique(rows).size != len(rows)
        or query_ids.shape != split_names.shape != (len(rows),)
        or xy.shape != (len(rows), 2)
        or tracks.ndim != 2
        or canonical.shape != tracks.shape
        or views.ndim != 3
        or views.shape[:2] != tracks.shape
        or support_ids.shape != views.shape
        or coverage.shape != views.shape
        or np.any(~np.isfinite(xy))
        or set(split_names.tolist()) - {"train", "validation", "test"}
    ):
        raise ValueError("context-attention frozen layout arrays are invalid")
    valid = tracks >= 0
    if (
        np.any(canonical[valid] < 0)
        or np.any(canonical[~valid] >= 0)
        or np.any(valid & ~np.any(views, axis=2))
        or np.any(coverage < 0)
        or np.any((~views) & (coverage != 0))
        or np.any((~views) & (support_ids != ""))
    ):
        raise ValueError("context-attention frozen candidate/support layout is invalid")
    if not str(metadata.get("proposals_sha256", "")):
        raise ValueError("context-attention frozen layout lacks proposal lineage")
    return {
        "source_row_indices": rows,
        "query_ids": query_ids,
        "split_names": split_names,
        "xy": xy,
        "candidate_track_ids": tracks,
        "candidate_canonical_rows": canonical,
        "candidate_view_valid": views,
        "candidate_support_image_ids": support_ids,
        "candidate_support_coverage_counts": coverage,
    }, metadata


@dataclass(frozen=True)
class ContextAttentionScale:
    """One predeclared descriptor crop used by the frozen attention probe."""

    name: str
    grid_size: int
    window_size: int

    def __post_init__(self) -> None:
        if not str(self.name) or int(self.grid_size) <= 0 or int(self.window_size) <= 0:
            raise ValueError("context-attention scale is invalid")
        if int(self.window_size) % 2 != 1 or int(self.window_size) > int(self.grid_size):
            raise ValueError("context-attention window must be odd and fit its source grid")


# These scales are fixed before validation/test labels are inspected.  RADIO
# final/intermediate preserve broad facade phase; ALIKE contributes a finer
# local branch but is not allowed to replace the broad context branches.
CONTEXT_ATTENTION_SCALES = (
    ContextAttentionScale("radio_final", grid_size=16, window_size=15),
    ContextAttentionScale("radio_intermediate", grid_size=16, window_size=15),
    ContextAttentionScale("alike", grid_size=32, window_size=13),
)


@dataclass(frozen=True)
class ContextAttentionSource:
    """Validated real-image descriptor grid for one context branch."""

    name: str
    path: Path
    image_ids: np.ndarray
    image_sizes: np.ndarray
    grid: np.ndarray
    metadata: Mapping[str, object]
    grid_size: int | None = None

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.image_ids).astype(str).reshape(-1)
        sizes = np.asarray(self.image_sizes, dtype=np.int64).reshape(-1, 2)
        grid = np.asarray(self.grid)
        scale = next((item for item in CONTEXT_ATTENTION_SCALES if item.name == self.name), None)
        if scale is None and self.grid_size is None:
            raise ValueError("unknown context-attention source")
        expected_grid_size = int(self.grid_size) if self.grid_size is not None else int(scale.grid_size)
        if (
            len(image_ids) == 0
            or len(set(image_ids.tolist())) != len(image_ids)
            or sizes.shape != (len(image_ids), 2)
            or np.any(sizes <= 0)
            or grid.ndim != 4
            or grid.shape[:3] != (len(image_ids), expected_grid_size, expected_grid_size)
            or grid.shape[3] <= 0
            or np.any(~np.isfinite(grid))
        ):
            raise ValueError("context-attention source arrays are invalid")
        norms = np.linalg.norm(grid.astype(np.float32, copy=False), axis=-1)
        if np.max(np.abs(norms - 1.0)) > 5e-3:
            raise ValueError("context-attention source descriptors are not normalized")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_sizes", sizes)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def descriptor_dim(self) -> int:
        return int(np.asarray(self.grid).shape[-1])

    @property
    def spatial_grid_size(self) -> int:
        return int(np.asarray(self.grid).shape[1])


def _reject_non_target_free_cache(metadata: Mapping[str, object], *, context: str) -> None:
    if bool(metadata.get("pose_or_ground_truth_used", True)):
        raise ValueError(f"{context} is not pose/GT free")
    if bool(metadata.get("image_retrieval_or_submap_used", True)):
        raise ValueError(f"{context} violates the no-retrieval protocol")
    if bool(metadata.get("render", False)):
        raise ValueError(f"{context} uses rendering")


def load_context_attention_sources(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    expected_radio_checkpoint: str,
    require_equal_descriptor_dimensions: bool = True,
) -> tuple[ContextAttentionSource, ...]:
    """Load all three aligned real-image spaces and enforce their lineage.

    Context-attention fusion requires a shared descriptor width.  Independent
    per-source probes can safely compare each source in its own space, so they
    may explicitly relax that one representation constraint while preserving
    all image, checkpoint, and target-free lineage checks.
    """

    final = load_radio_final_context_pca_cache(Path(radio_final_context_cache))
    final_metadata = final.metadata
    if final_metadata.get("format") != RADIO_FINAL_CONTEXT_PCA_FORMAT:
        raise ValueError("unsupported RADIO-final context cache")
    _reject_non_target_free_cache(final_metadata, context="RADIO-final context cache")
    if final_metadata.get("pca_fit_scope") not in _SAFE_PCA_FIT_SCOPES:
        raise ValueError("RADIO-final context PCA was not fit only on mapping-side images")
    if expected_radio_checkpoint and str(final_metadata.get("radio_checkpoint_sha256", "")) != str(
        expected_radio_checkpoint
    ):
        raise ValueError("frozen layout and RADIO-final cache checkpoints differ")
    if final.grid16_descriptors is None:
        raise ValueError("RADIO-final context cache lacks grid16 descriptors")

    intermediate = load_spatial_image_context_cache(
        Path(radio_intermediate_context_cache),
        expected_format=_RADIO_INTERMEDIATE_CONTEXT_PCA_FORMAT,
    )
    intermediate_metadata = intermediate.metadata
    _reject_non_target_free_cache(intermediate_metadata, context="RADIO-intermediate context cache")
    if intermediate_metadata.get("pca_fit_scope") not in _SAFE_PCA_FIT_SCOPES:
        raise ValueError(
            "RADIO-intermediate context PCA was not fit only on mapping-side images"
        )
    if int(intermediate_metadata.get("intermediate_index", 0)) != -6:
        raise ValueError("RADIO-intermediate context cache uses an unexpected layer")
    if str(intermediate_metadata.get("radio_checkpoint_sha256", "")) != str(
        final_metadata.get("radio_checkpoint_sha256", "")
    ):
        raise ValueError("RADIO final/intermediate checkpoints differ")

    alike = load_spatial_image_context_cache(
        Path(alike_spatial_context_cache), expected_format=_ALIKE_SPATIAL_CONTEXT_FORMAT
    )
    alike_metadata = alike.metadata
    _reject_non_target_free_cache(alike_metadata, context="ALIKE context cache")
    if not str(alike_metadata.get("alike_checkpoint_sha256", "")):
        raise ValueError("ALIKE context cache lacks checkpoint lineage")

    final_ids = np.asarray(final.image_ids).astype(str)
    final_sizes = np.asarray(final.image_sizes, dtype=np.int64)
    manifest = str(final_metadata.get("source_image_manifest_sha256", ""))
    if not manifest:
        raise ValueError("RADIO-final context cache lacks a source image manifest")
    for name, cache in (("RADIO-intermediate", intermediate), ("ALIKE", alike)):
        if not np.array_equal(cache.image_ids.astype(str), final_ids) or not np.array_equal(
            cache.image_sizes, final_sizes
        ):
            raise ValueError(f"{name} context cache does not align with RADIO-final images")
        if str(cache.metadata.get("source_image_manifest_sha256", "")) != manifest:
            raise ValueError(f"{name} context cache source image manifest differs")

    sources = (
        ContextAttentionSource(
            name="radio_final",
            path=Path(radio_final_context_cache),
            image_ids=final_ids,
            image_sizes=final_sizes,
            grid=np.asarray(final.grid16_descriptors).reshape(
                len(final_ids), 16, 16, final.descriptor_dim
            ),
            metadata=final_metadata,
        ),
        ContextAttentionSource(
            name="radio_intermediate",
            path=Path(radio_intermediate_context_cache),
            image_ids=intermediate.image_ids,
            image_sizes=intermediate.image_sizes,
            grid=intermediate.grid_descriptors(16).reshape(
                len(intermediate.image_ids), 16, 16, intermediate.descriptor_dim
            ),
            metadata=intermediate_metadata,
        ),
        ContextAttentionSource(
            name="alike",
            path=Path(alike_spatial_context_cache),
            image_ids=alike.image_ids,
            image_sizes=alike.image_sizes,
            grid=alike.grid_descriptors(32).reshape(
                len(alike.image_ids), 32, 32, alike.descriptor_dim
            ),
            metadata=alike_metadata,
        ),
    )
    descriptor_dims = {source.descriptor_dim for source in sources}
    if bool(require_equal_descriptor_dimensions) and len(descriptor_dims) != 1:
        raise ValueError("context-attention source descriptor dimensions differ")
    return sources


def load_absolute_grid_context_sources(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    expected_radio_checkpoint: str,
) -> tuple[ContextAttentionSource, ...]:
    """Load the aligned 16x16 full-image grids for absolute-layout probing.

    The existing relative-crop probe uses ALIKE grid32.  This distinct probe
    deliberately uses a common 16x16 lattice across all three descriptor
    spaces so region-to-region transport has a stable absolute layout.
    """

    final, intermediate, _alike_grid32 = load_context_attention_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_context_cache=Path(radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
        expected_radio_checkpoint=expected_radio_checkpoint,
    )
    # The first loader above validates ALIKE's target-free lineage and image
    # manifest against RADIO.  Reload only the coarser grid required here.
    alike_cache = load_spatial_image_context_cache(
        Path(alike_spatial_context_cache), expected_format=_ALIKE_SPATIAL_CONTEXT_FORMAT
    )
    if int(16) not in alike_cache.grids:
        raise ValueError("ALIKE absolute-layout cache lacks grid16 descriptors")
    alike = ContextAttentionSource(
        name="alike",
        path=Path(alike_spatial_context_cache),
        image_ids=alike_cache.image_ids,
        image_sizes=alike_cache.image_sizes,
        grid=alike_cache.grid_descriptors(16).reshape(
            len(alike_cache.image_ids), 16, 16, alike_cache.descriptor_dim
        ),
        metadata=alike_cache.metadata,
        grid_size=16,
    )
    if not np.array_equal(alike.image_ids, final.image_ids) or not np.array_equal(
        alike.image_sizes, final.image_sizes
    ):
        raise ValueError("ALIKE absolute-layout grid does not align with RADIO-final")
    if alike.descriptor_dim != final.descriptor_dim or intermediate.descriptor_dim != final.descriptor_dim:
        raise ValueError("absolute-layout source descriptor dimensions differ")
    return final, intermediate, alike


@dataclass(frozen=True)
class ContextAttentionRuntimeArrays:
    """Frozen query/support indices used identically in fit and inference."""

    query_image_indices: np.ndarray
    support_image_indices: np.ndarray
    support_xy: np.ndarray
    view_valid: np.ndarray

    def __post_init__(self) -> None:
        query = np.asarray(self.query_image_indices, dtype=np.int64).reshape(-1)
        support = np.asarray(self.support_image_indices, dtype=np.int64)
        xy = np.asarray(self.support_xy, dtype=np.float32)
        valid = np.asarray(self.view_valid, dtype=bool)
        if (
            support.ndim != 3
            or xy.shape != (*support.shape, 2)
            or valid.shape != support.shape
            or query.shape != (support.shape[0],)
            or np.any(query < 0)
            or np.any(support < 0)
            or np.any(~np.isfinite(xy))
        ):
            raise ValueError("context-attention frozen runtime arrays are invalid")
        object.__setattr__(self, "query_image_indices", query)
        object.__setattr__(self, "support_image_indices", support)
        object.__setattr__(self, "support_xy", xy)
        object.__setattr__(self, "view_valid", valid)


def build_fixed_candidate_context_runtime(
    *,
    query_ids: Sequence[str] | np.ndarray,
    query_xy: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_support_image_ids: np.ndarray,
    candidate_view_valid: np.ndarray,
    cache_image_ids: Sequence[str] | np.ndarray,
    support_geometry: SupportObservationGeometryIndex,
) -> ContextAttentionRuntimeArrays:
    """Resolve fixed support observations without any pose or target input."""

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    support_ids = np.asarray(candidate_support_image_ids).astype(str)
    view_valid = np.asarray(candidate_view_valid, dtype=bool)
    cache_ids = np.asarray(cache_image_ids).astype(str).reshape(-1)
    if (
        tracks.ndim != 2
        or xy.shape != (len(ids), 2)
        or support_ids.shape != view_valid.shape
        or support_ids.shape[:2] != tracks.shape
        or len(cache_ids) == 0
        or len(set(cache_ids.tolist())) != len(cache_ids)
        or np.any(~np.isfinite(xy))
    ):
        raise ValueError("fixed context runtime inputs are incompatible")
    valid_candidate = tracks >= 0
    if np.any(valid_candidate & ~np.any(view_valid, axis=2)):
        raise ValueError("a valid frozen candidate lacks a fixed support view")
    if set(ids.tolist()) & set(support_ids[view_valid].tolist()):
        raise ValueError("query images leaked into fixed support evidence")
    cache_position = {value: index for index, value in enumerate(cache_ids.tolist())}
    try:
        query_indices = np.asarray([cache_position[value] for value in ids.tolist()], dtype=np.int64)
    except KeyError as error:
        raise KeyError(f"query image missing from spatial cache: {error.args[0]}") from error
    support_indices = np.zeros(support_ids.shape, dtype=np.int64)
    support_xy = np.zeros((*support_ids.shape, 2), dtype=np.float32)
    flat_valid = view_valid.reshape(-1)
    flat_support_ids = support_ids.reshape(-1)
    flat_tracks = np.repeat(tracks[:, :, None], support_ids.shape[2], axis=2).reshape(-1)
    flat_indices = support_indices.reshape(-1)
    flat_xy = support_xy.reshape(-1, 2)
    for image_id in np.unique(flat_support_ids[flat_valid]).tolist():
        positions = np.flatnonzero(flat_valid & (flat_support_ids == str(image_id)))
        image_position = cache_position.get(str(image_id))
        if image_position is None:
            raise KeyError(f"fixed support image missing from spatial cache: {image_id}")
        geometry_rows = support_geometry.geometry_rows_for_tracks(
            str(image_id), flat_tracks[positions]
        )
        if np.any(geometry_rows < 0):
            raise ValueError("fixed candidate support view lacks its SfM observation")
        flat_indices[positions] = int(image_position)
        flat_xy[positions] = support_geometry.xy[geometry_rows]
    if np.any(flat_support_ids[flat_valid] == ""):
        raise ValueError("fixed valid support view lacks an image ID")
    return ContextAttentionRuntimeArrays(
        query_image_indices=query_indices,
        support_image_indices=support_indices,
        support_xy=support_xy,
        view_valid=view_valid,
    )


def image_xy_to_grid_indices(
    *,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map image coordinates to clamped descriptor-grid rows and columns."""

    if (
        image_sizes.ndim != 2
        or image_sizes.shape[1] != 2
        or image_indices.ndim != 1
        or xy.shape != (len(image_indices), 2)
        or int(grid_size) <= 0
    ):
        raise ValueError("image-to-grid inputs are incompatible")
    selected_sizes = image_sizes.index_select(0, image_indices).to(dtype=torch.float32)
    columns = torch.floor(
        xy[:, 0] / selected_sizes[:, 0].clamp_min(1.0) * float(grid_size)
    ).to(dtype=torch.long).clamp(0, int(grid_size) - 1)
    rows = torch.floor(
        xy[:, 1] / selected_sizes[:, 1].clamp_min(1.0) * float(grid_size)
    ).to(dtype=torch.long).clamp(0, int(grid_size) - 1)
    return rows, columns


def crop_anchor_aligned_grid_tokens(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one masked, anchor-centred square descriptor crop per image."""

    if (
        image_grids.ndim != 4
        or image_grids.shape[1] != image_grids.shape[2]
        or image_sizes.shape != (image_grids.shape[0], 2)
        or int(window_size) <= 0
        or int(window_size) % 2 != 1
        or int(window_size) > int(image_grids.shape[1])
    ):
        raise ValueError("anchor-aligned crop inputs are invalid")
    grid_size = int(image_grids.shape[1])
    rows, columns = image_xy_to_grid_indices(
        image_sizes=image_sizes,
        image_indices=image_indices,
        xy=xy,
        grid_size=grid_size,
    )
    radius = int(window_size) // 2
    offsets = torch.arange(-radius, radius + 1, device=image_grids.device)
    raw_rows = rows[:, None] + offsets[None, :]
    raw_columns = columns[:, None] + offsets[None, :]
    valid = (
        (raw_rows[:, :, None] >= 0)
        & (raw_rows[:, :, None] < grid_size)
        & (raw_columns[:, None, :] >= 0)
        & (raw_columns[:, None, :] < grid_size)
    )
    safe_rows = raw_rows.clamp(0, grid_size - 1)
    safe_columns = raw_columns.clamp(0, grid_size - 1)
    selected = image_grids.index_select(0, image_indices)
    batch = torch.arange(len(image_indices), device=image_grids.device)[:, None, None]
    crop = selected[
        batch,
        safe_rows[:, :, None].expand(-1, int(window_size), int(window_size)),
        safe_columns[:, None, :].expand(-1, int(window_size), int(window_size)),
    ]
    return crop.reshape(len(image_indices), int(window_size) ** 2, crop.shape[-1]), valid.reshape(
        len(image_indices), int(window_size) ** 2
    )


def crop_anchor_aligned_grid_absolute_coordinates(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Return each cropped token's position in its full-image grid frame.

    This intentionally differs from :func:`context_relative_coordinates`: the
    values are normalized against the complete image lattice, so a repeated
    local pattern at a different facade phase remains distinguishable.  Invalid
    crop positions are clamped only for tensor safety and remain masked by the
    paired validity tensor from ``crop_anchor_aligned_grid_tokens``.
    """

    if (
        image_grids.ndim != 4
        or image_grids.shape[1] != image_grids.shape[2]
        or image_sizes.shape != (image_grids.shape[0], 2)
        or int(window_size) <= 0
        or int(window_size) % 2 != 1
        or int(window_size) > int(image_grids.shape[1])
    ):
        raise ValueError("absolute crop-coordinate inputs are invalid")
    grid_size = int(image_grids.shape[1])
    rows, columns = image_xy_to_grid_indices(
        image_sizes=image_sizes,
        image_indices=image_indices,
        xy=xy,
        grid_size=grid_size,
    )
    radius = int(window_size) // 2
    offsets = torch.arange(-radius, radius + 1, device=image_grids.device)
    safe_rows = (rows[:, None] + offsets[None, :]).clamp(0, grid_size - 1)
    safe_columns = (columns[:, None] + offsets[None, :]).clamp(0, grid_size - 1)
    denominator = float(max(grid_size - 1, 1))
    grid_rows = safe_rows[:, :, None].expand(-1, int(window_size), int(window_size))
    grid_columns = safe_columns[:, None, :].expand(-1, int(window_size), int(window_size))
    return torch.stack(
        [grid_columns.to(dtype=torch.float32) / denominator, grid_rows.to(dtype=torch.float32) / denominator],
        dim=-1,
    ).reshape(len(image_indices), int(window_size) ** 2, 2)


def normalized_image_coordinates(
    *, image_sizes: torch.Tensor, image_indices: torch.Tensor, xy: torch.Tensor
) -> torch.Tensor:
    """Normalize real image coordinates to a stable [0, 1] full-image frame."""

    if (
        image_sizes.ndim != 2
        or image_sizes.shape[1] != 2
        or image_indices.ndim != 1
        or xy.shape != (len(image_indices), 2)
    ):
        raise ValueError("normalized image-coordinate inputs are incompatible")
    sizes = image_sizes.index_select(0, image_indices).to(dtype=torch.float32).clamp_min(1.0)
    return (xy.to(dtype=torch.float32) / sizes).clamp(0.0, 1.0)


def anchor_grid_descriptors(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
) -> torch.Tensor:
    """Sample only the central descriptor used by the anchor+context branch."""

    rows, columns = image_xy_to_grid_indices(
        image_sizes=image_sizes,
        image_indices=image_indices,
        xy=xy,
        grid_size=int(image_grids.shape[1]),
    )
    selected = image_grids.index_select(0, image_indices)
    batch = torch.arange(len(image_indices), device=image_grids.device)
    return selected[batch, rows, columns]


def context_valid_mask(
    valid: torch.Tensor,
    *,
    window_size: int,
    center_mask_radius: int = CONTEXT_ATTENTION_CENTER_MASK_RADIUS,
) -> torch.Tensor:
    """Remove the central token neighbourhood from a valid crop mask."""

    values = valid.to(dtype=torch.bool)
    if values.ndim != 2 or values.shape[1] != int(window_size) ** 2:
        raise ValueError("context mask shape differs from its crop window")
    radius = int(center_mask_radius)
    if radius < 0 or int(window_size) <= 2 * radius + 1:
        raise ValueError("context centre mask leaves no spatial context")
    positions = torch.arange(int(window_size), device=values.device)
    rows, columns = torch.meshgrid(positions, positions, indexing="ij")
    centre = int(window_size) // 2
    excluded = (rows - centre).abs() <= radius
    excluded &= (columns - centre).abs() <= radius
    return values & ~excluded.reshape(1, -1)


def context_relative_coordinates(*, window_size: int, device: torch.device) -> torch.Tensor:
    """Anchor-relative token coordinates in the stable [-1, 1] crop frame."""

    if int(window_size) <= 0 or int(window_size) % 2 != 1:
        raise ValueError("context coordinate window must be positive and odd")
    radius = max(int(window_size) // 2, 1)
    values = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    rows, columns = torch.meshgrid(values, values, indexing="ij")
    return torch.stack([columns / float(radius), rows / float(radius)], dim=-1).reshape(-1, 2)


def masked_view_log_mean(view_logits: torch.Tensor, view_valid: torch.Tensor) -> torch.Tensor:
    """Marginalize fixed support views without pre-averaging their evidence."""

    logits = torch.as_tensor(view_logits)
    valid = torch.as_tensor(view_valid, dtype=torch.bool, device=logits.device)
    if logits.ndim != 3 or valid.shape != logits.shape:
        raise ValueError("per-view logits and support mask are incompatible")
    count = valid.sum(dim=2)
    masked = torch.where(
        valid, logits, torch.full_like(logits, torch.finfo(logits.dtype).min)
    )
    output = torch.logsumexp(masked, dim=2) - torch.log(
        count.clamp_min(1).to(dtype=logits.dtype)
    )
    # Invalid padded candidates have zero base probability and therefore never
    # enter the candidate softmax.  Keep their residual finite so a partial
    # top-L row remains a valid frozen inference artifact.
    return torch.where(count > 0, output, torch.zeros_like(output))


class _ContextScaleEncoder(nn.Module):
    """A small position-aware cross-attention encoder for one descriptor scale."""

    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        *,
        position_encoding: str = ANCHOR_RELATIVE_POSITION_ENCODING,
    ) -> None:
        super().__init__()
        if (
            int(descriptor_dim) <= 0
            or int(hidden_dim) <= 0
            or int(heads) <= 0
            or int(hidden_dim) % int(heads) != 0
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise ValueError("context scale encoder dimensions are invalid")
        if str(position_encoding) not in SUPPORTED_CONTEXT_POSITION_ENCODINGS:
            raise ValueError("unsupported context-attention position encoding")
        self.position_encoding = str(position_encoding)
        self.query_projection = nn.Linear(int(descriptor_dim), int(hidden_dim), bias=False)
        self.support_projection = nn.Linear(int(descriptor_dim), int(hidden_dim), bias=False)
        self.position_projection = nn.Sequential(
            nn.Linear(2, int(hidden_dim)), nn.Tanh(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.absolute_position_projections = nn.ModuleDict()
        if self.position_encoding == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING:
            for name in (
                "query_token",
                "support_token",
                "query_anchor",
                "support_anchor",
            ):
                self.absolute_position_projections[name] = nn.Sequential(
                    nn.Linear(2, int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
        self.cross_attention = nn.MultiheadAttention(
            int(hidden_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.normalizer = nn.LayerNorm(int(hidden_dim))
        self.pool_gate = nn.Linear(int(hidden_dim), 1)
        self.output = nn.Sequential(
            nn.LayerNorm(2 * int(hidden_dim) + 2),
            nn.Linear(2 * int(hidden_dim) + 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
        )

    def forward(
        self,
        *,
        query_tokens: torch.Tensor,
        query_valid: torch.Tensor,
        support_tokens: torch.Tensor,
        support_valid: torch.Tensor,
        coordinates: torch.Tensor,
        query_absolute_coordinates: torch.Tensor | None = None,
        support_absolute_coordinates: torch.Tensor | None = None,
        query_anchor_coordinates: torch.Tensor | None = None,
        support_anchor_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            query_tokens.ndim != 3
            or support_tokens.shape != query_tokens.shape
            or query_valid.shape != query_tokens.shape[:2]
            or support_valid.shape != query_tokens.shape[:2]
            or coordinates.shape != (query_tokens.shape[1], 2)
        ):
            raise ValueError("context-attention scale tensors are incompatible")
        if torch.any(~torch.any(query_valid, dim=1)) or torch.any(~torch.any(support_valid, dim=1)):
            raise ValueError("context-attention scale received an empty crop")
        absolute_coordinates = (
            query_absolute_coordinates,
            support_absolute_coordinates,
            query_anchor_coordinates,
            support_anchor_coordinates,
        )
        if self.position_encoding == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING:
            if (
                any(value is None for value in absolute_coordinates)
                or query_absolute_coordinates is None
                or support_absolute_coordinates is None
                or query_anchor_coordinates is None
                or support_anchor_coordinates is None
                or query_absolute_coordinates.shape != (*query_tokens.shape[:2], 2)
                or support_absolute_coordinates.shape != (*query_tokens.shape[:2], 2)
                or query_anchor_coordinates.shape != (query_tokens.shape[0], 2)
                or support_anchor_coordinates.shape != (query_tokens.shape[0], 2)
            ):
                raise ValueError("absolute dual-frame coordinates are incompatible")
        elif any(value is not None for value in absolute_coordinates):
            raise ValueError("relative context encoder received absolute coordinates")
        # Cache grids stay in fp16 on GPU, but CPU tests and non-autocast
        # inference require linear inputs to match parameter dtype.  CUDA AMP
        # still downcasts these operations where appropriate.
        query_tokens = query_tokens.to(dtype=self.query_projection.weight.dtype)
        support_tokens = support_tokens.to(dtype=self.support_projection.weight.dtype)
        position = self.position_projection(coordinates).unsqueeze(0)
        query = self.query_projection(query_tokens) + position
        support = self.support_projection(support_tokens) + position
        if self.position_encoding == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING:
            coordinate_dtype = query.dtype
            query = query + self.absolute_position_projections["query_token"](
                query_absolute_coordinates.to(dtype=coordinate_dtype)
            )
            support = support + self.absolute_position_projections["support_token"](
                support_absolute_coordinates.to(dtype=coordinate_dtype)
            )
            query = query + self.absolute_position_projections["query_anchor"](
                query_anchor_coordinates.to(dtype=coordinate_dtype)
            ).unsqueeze(1)
            support = support + self.absolute_position_projections["support_anchor"](
                support_anchor_coordinates.to(dtype=coordinate_dtype)
            ).unsqueeze(1)
        attended, _ = self.cross_attention(
            query,
            support,
            support,
            key_padding_mask=~support_valid,
            need_weights=False,
        )
        updated = self.normalizer(query + attended)
        valid_float = query_valid.to(dtype=updated.dtype)
        denominator = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = torch.sum(updated * valid_float[..., None], dim=1) / denominator
        gates = self.pool_gate(updated).squeeze(-1)
        gates = torch.where(
            query_valid, gates, torch.full_like(gates, torch.finfo(gates.dtype).min)
        )
        weights = torch.softmax(gates, dim=1)
        pooled = torch.sum(updated * weights[..., None], dim=1)
        cosine = F.cosine_similarity(query, attended, dim=2)
        alignment = torch.sum(cosine * valid_float, dim=1, keepdim=True) / denominator
        coverage = valid_float.mean(dim=1, keepdim=True)
        return self.output(torch.cat([mean, pooled, alignment, coverage], dim=1))


class CandidateContextAttentionProbe(nn.Module):
    """Fixed-candidate per-view context scorer with an explicit base residual."""

    def __init__(
        self,
        *,
        family: str,
        sources: Mapping[str, torch.Tensor],
        image_sizes: torch.Tensor,
        runtime: ContextAttentionRuntimeArrays,
        query_xy: np.ndarray | torch.Tensor,
        base_candidate_probabilities: np.ndarray | torch.Tensor,
        base_null_probabilities: np.ndarray | torch.Tensor,
        hidden_dim: int = 32,
        heads: int = 2,
        dropout: float = 0.1,
        position_encoding: str = ANCHOR_RELATIVE_POSITION_ENCODING,
    ) -> None:
        super().__init__()
        if str(family) not in SUPPORTED_CONTEXT_ATTENTION_FAMILIES:
            raise ValueError("unsupported context-attention probe family")
        if str(position_encoding) not in SUPPORTED_CONTEXT_POSITION_ENCODINGS:
            raise ValueError("unsupported context-attention position encoding")
        is_absolute_phase = str(family) in ABSOLUTE_PHASE_CONTEXT_ATTENTION_FAMILIES
        expected_position_encoding = (
            ABSOLUTE_DUAL_FRAME_POSITION_ENCODING
            if is_absolute_phase
            else ANCHOR_RELATIVE_POSITION_ENCODING
        )
        if str(position_encoding) != expected_position_encoding:
            raise ValueError("context-attention family and position encoding differ")
        if int(hidden_dim) < 4 or int(hidden_dim) % 2 != 0:
            raise ValueError("context-attention hidden dimension must be even and at least four")
        self.family = str(family)
        self.position_encoding = str(position_encoding)
        self._uses_position_only = self.family.endswith("position_only")
        self._uses_anchor = self.family.endswith("with_anchor")
        expected_names = {scale.name for scale in CONTEXT_ATTENTION_SCALES}
        if set(sources) != expected_names:
            raise ValueError("context-attention source set differs from the fixed profile")
        source_tensors = {name: torch.as_tensor(value) for name, value in sources.items()}
        descriptor_dims = {int(value.shape[-1]) for value in source_tensors.values()}
        if len(descriptor_dims) != 1:
            raise ValueError("context-attention source descriptor dimensions differ")
        descriptor_dim = next(iter(descriptor_dims))
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        query_image_indices = torch.as_tensor(runtime.query_image_indices, dtype=torch.long)
        support_image_indices = torch.as_tensor(runtime.support_image_indices, dtype=torch.long)
        support_xy = torch.as_tensor(runtime.support_xy, dtype=torch.float32)
        view_valid = torch.as_tensor(runtime.view_valid, dtype=torch.bool)
        query_coordinates = torch.as_tensor(query_xy, dtype=torch.float32)
        base_candidate = torch.as_tensor(base_candidate_probabilities, dtype=torch.float32)
        base_null = torch.as_tensor(base_null_probabilities, dtype=torch.float32).reshape(-1)
        row_count, candidate_count, view_count = support_image_indices.shape
        if (
            sizes.ndim != 2
            or sizes.shape[1] != 2
            or query_image_indices.shape != (row_count,)
            or query_coordinates.shape != (row_count, 2)
            or support_xy.shape != (row_count, candidate_count, view_count, 2)
            or view_valid.shape != support_image_indices.shape
            or base_candidate.shape != (row_count, candidate_count)
            or base_null.shape != (row_count,)
            or torch.any(base_candidate < 0.0)
            or torch.any(base_null < 0.0)
            or torch.any(torch.abs(base_candidate.sum(dim=1) + base_null - 1.0) > 1e-4)
            or torch.any((base_candidate > 0.0) & ~torch.any(view_valid, dim=2))
        ):
            raise ValueError("context-attention static arrays are incompatible")
        for scale in CONTEXT_ATTENTION_SCALES:
            values = source_tensors[scale.name]
            if values.ndim != 4 or values.shape[:3] != (
                len(sizes),
                scale.grid_size,
                scale.grid_size,
            ):
                raise ValueError("context-attention source grid shape differs from its profile")
            self.register_buffer(f"_{scale.name}_grid", values.to(dtype=torch.float16), persistent=False)
        self.register_buffer("_image_sizes", sizes, persistent=False)
        self.register_buffer("_query_image_indices", query_image_indices, persistent=False)
        self.register_buffer("_query_xy", query_coordinates, persistent=False)
        self.register_buffer("_support_image_indices", support_image_indices, persistent=False)
        self.register_buffer("_support_xy", support_xy, persistent=False)
        self.register_buffer("_view_valid", view_valid, persistent=False)
        self.register_buffer("_base_candidate_probabilities", base_candidate, persistent=False)
        self.register_buffer("_base_null_probabilities", base_null, persistent=False)
        self.encoders = nn.ModuleDict(
            {
                scale.name: _ContextScaleEncoder(
                    descriptor_dim,
                    int(hidden_dim),
                    int(heads),
                    float(dropout),
                    position_encoding=self.position_encoding,
                )
                for scale in CONTEXT_ATTENTION_SCALES
            }
        )
        branch_width = len(CONTEXT_ATTENTION_SCALES) * (int(hidden_dim) // 2)
        if self._uses_anchor:
            branch_width += len(CONTEXT_ATTENTION_SCALES)
        self.view_head = nn.Sequential(
            nn.LayerNorm(branch_width),
            nn.Linear(branch_width, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        # A zero residual is exactly the frozen base prior.  This makes the
        # training contract and failed-abstention fallback explicit.
        nn.init.zeros_(self.view_head[-1].weight)
        nn.init.zeros_(self.view_head[-1].bias)

    @property
    def row_count(self) -> int:
        return int(self._query_image_indices.shape[0])

    def _grid(self, name: str) -> torch.Tensor:
        return getattr(self, f"_{name}_grid")

    def _row_context_inputs(
        self, rows: torch.Tensor, scale: ContextAttentionScale
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        selected_rows = rows.to(dtype=torch.long)
        query_indices = self._query_image_indices.index_select(0, selected_rows)
        query_xy = self._query_xy.index_select(0, selected_rows)
        support_indices = self._support_image_indices.index_select(0, selected_rows)
        support_xy = self._support_xy.index_select(0, selected_rows)
        view_valid = self._view_valid.index_select(0, selected_rows)
        batch, candidate_count, view_count = support_indices.shape
        edge_count = int(batch * candidate_count * view_count)
        grid = self._grid(scale.name)
        query_tokens, query_valid = crop_anchor_aligned_grid_tokens(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=query_indices,
            xy=query_xy,
            window_size=scale.window_size,
        )
        query_tokens = (
            query_tokens[:, None, None]
            .expand(-1, candidate_count, view_count, -1, -1)
            .reshape(edge_count, query_tokens.shape[1], query_tokens.shape[2])
        )
        query_valid = (
            query_valid[:, None, None]
            .expand(-1, candidate_count, view_count, -1)
            .reshape(edge_count, query_valid.shape[1])
        )
        support_tokens, support_valid = crop_anchor_aligned_grid_tokens(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=support_indices.reshape(-1),
            xy=support_xy.reshape(-1, 2),
            window_size=scale.window_size,
        )
        edge_valid = view_valid.reshape(-1)
        query_context_valid = context_valid_mask(
            query_valid, window_size=scale.window_size
        )
        support_context_valid = context_valid_mask(
            support_valid, window_size=scale.window_size
        ) & edge_valid[:, None]
        # Invalid candidate/view slots must not become an all-masked attention
        # row.  They are zeroed after encoding and never participate in the
        # per-view marginalization.
        safe_support_valid = support_context_valid.clone()
        safe_support_valid[~edge_valid, 0] = True
        if torch.any(~torch.any(query_context_valid, dim=1)) or torch.any(
            ~torch.any(safe_support_valid, dim=1)
        ):
            raise RuntimeError("context crop lost every non-anchor descriptor")
        coordinates = context_relative_coordinates(
            window_size=scale.window_size, device=grid.device
        )
        query_absolute_coordinates: torch.Tensor | None = None
        support_absolute_coordinates: torch.Tensor | None = None
        query_anchor_coordinates: torch.Tensor | None = None
        support_anchor_coordinates: torch.Tensor | None = None
        if self.position_encoding == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING:
            query_absolute = crop_anchor_aligned_grid_absolute_coordinates(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=query_indices,
                xy=query_xy,
                window_size=scale.window_size,
            )
            query_absolute_coordinates = (
                query_absolute[:, None, None]
                .expand(-1, candidate_count, view_count, -1, -1)
                .reshape(edge_count, query_absolute.shape[1], 2)
            )
            support_absolute_coordinates = crop_anchor_aligned_grid_absolute_coordinates(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=support_indices.reshape(-1),
                xy=support_xy.reshape(-1, 2),
                window_size=scale.window_size,
            )
            query_anchor = normalized_image_coordinates(
                image_sizes=self._image_sizes,
                image_indices=query_indices,
                xy=query_xy,
            )
            query_anchor_coordinates = (
                query_anchor[:, None, None]
                .expand(-1, candidate_count, view_count, -1)
                .reshape(edge_count, 2)
            )
            support_anchor_coordinates = normalized_image_coordinates(
                image_sizes=self._image_sizes,
                image_indices=support_indices.reshape(-1),
                xy=support_xy.reshape(-1, 2),
            )
        if self._uses_position_only:
            query_tokens = torch.zeros_like(query_tokens)
            support_tokens = torch.zeros_like(support_tokens)
        return (
            query_tokens,
            query_context_valid,
            support_tokens,
            safe_support_valid,
            coordinates,
            query_absolute_coordinates,
            support_absolute_coordinates,
            query_anchor_coordinates,
            support_anchor_coordinates,
        )

    def _anchor_cosine(self, rows: torch.Tensor, scale: ContextAttentionScale) -> torch.Tensor:
        selected_rows = rows.to(dtype=torch.long)
        query_indices = self._query_image_indices.index_select(0, selected_rows)
        query_xy = self._query_xy.index_select(0, selected_rows)
        support_indices = self._support_image_indices.index_select(0, selected_rows)
        support_xy = self._support_xy.index_select(0, selected_rows)
        batch, candidate_count, view_count = support_indices.shape
        grid = self._grid(scale.name)
        query = anchor_grid_descriptors(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=query_indices,
            xy=query_xy,
        )
        query = (
            query[:, None, None]
            .expand(-1, candidate_count, view_count, -1)
            .reshape(-1, query.shape[-1])
        )
        support = anchor_grid_descriptors(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=support_indices.reshape(-1),
            xy=support_xy.reshape(-1, 2),
        )
        return F.cosine_similarity(
            query.to(dtype=torch.float32), support.to(dtype=torch.float32), dim=1
        )[:, None]

    def forward(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_rows = torch.as_tensor(rows, dtype=torch.long, device=self._query_xy.device)
        if selected_rows.ndim != 1 or selected_rows.numel() == 0:
            raise ValueError("context-attention forward needs non-empty row indices")
        if torch.any(selected_rows < 0) or torch.any(selected_rows >= self.row_count):
            raise ValueError("context-attention row index is out of range")
        view_features: list[torch.Tensor] = []
        anchor_features: list[torch.Tensor] = []
        for scale in CONTEXT_ATTENTION_SCALES:
            (
                query,
                query_valid,
                support,
                support_valid,
                coordinates,
                query_absolute_coordinates,
                support_absolute_coordinates,
                query_anchor_coordinates,
                support_anchor_coordinates,
            ) = self._row_context_inputs(selected_rows, scale)
            encoded = self.encoders[scale.name](
                query_tokens=query,
                query_valid=query_valid,
                support_tokens=support,
                support_valid=support_valid,
                coordinates=coordinates,
                query_absolute_coordinates=query_absolute_coordinates,
                support_absolute_coordinates=support_absolute_coordinates,
                query_anchor_coordinates=query_anchor_coordinates,
                support_anchor_coordinates=support_anchor_coordinates,
            )
            edge_valid = self._view_valid.index_select(0, selected_rows).reshape(-1)
            view_features.append(torch.where(edge_valid[:, None], encoded, torch.zeros_like(encoded)))
            if self._uses_anchor:
                anchor_features.append(self._anchor_cosine(selected_rows, scale))
        features = torch.cat([*view_features, *anchor_features], dim=1)
        raw_view_logits = self.view_head(features).reshape(
            len(selected_rows),
            self._support_image_indices.shape[1],
            self._support_image_indices.shape[2],
        )
        view_valid = self._view_valid.index_select(0, selected_rows)
        view_logits = torch.where(view_valid, raw_view_logits, torch.zeros_like(raw_view_logits))
        candidate_residual = masked_view_log_mean(view_logits, view_valid)
        base_candidate = self._base_candidate_probabilities.index_select(0, selected_rows)
        base_null = self._base_null_probabilities.index_select(0, selected_rows)
        candidate_logits = torch.where(
            base_candidate > 0.0,
            torch.log(base_candidate.clamp_min(1e-12)) + candidate_residual,
            torch.full_like(base_candidate, -1e9),
        )
        null_logits = torch.log(base_null.clamp_min(1e-12))
        logits = torch.cat([candidate_logits, null_logits[:, None]], dim=1)
        probability = torch.softmax(logits, dim=1)
        return probability[:, :-1], probability[:, -1], view_logits, logits
