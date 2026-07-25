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
import math
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

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
CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V2 = (
    "multiscale_context_attention_candidate_probe_contract_v2"
)
CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V3 = (
    "multiscale_context_attention_candidate_probe_contract_v3"
)
CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V4 = (
    "multiscale_context_attention_candidate_probe_contract_v4"
)
CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V5 = (
    "multiscale_context_attention_candidate_probe_contract_v5"
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
ABSOLUTE_CONTEXT_LIKELIHOOD_V2_FAMILIES = (
    "bidirectional_absolute_visual_v2",
    "bidirectional_absolute_position_control_v2",
)
ABSOLUTE_CONTEXT_LIKELIHOOD_V3_FAMILIES = (
    "bidirectional_absolute_raw_visual_v3",
    "bidirectional_absolute_raw_position_control_v3",
)
ABSOLUTE_CONTEXT_LIKELIHOOD_V4_FAMILIES = (
    "bidirectional_absolute_raw_layout_visual_v4",
    "bidirectional_absolute_raw_layout_position_control_v4",
)
ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES = (
    "bidirectional_absolute_dual_head_raw_layout_visual_v5",
    "bidirectional_absolute_dual_head_raw_layout_position_control_v5",
)
# RADIO final and intermediate carry the absolute contextual identity signal.
# ALIKE remains available to the geometric branch, but must not silently turn
# the strict-track identity posterior into another local-patch matcher.
_V5_IDENTITY_CONTEXT_SCALE_NAMES = frozenset({"radio_final", "radio_intermediate"})
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
_RAW_RADIO_FINAL_CONTEXT_FORMAT = "radio_image_multiscale_context_v1"
_RAW_RADIO_INTERMEDIATE_CONTEXT_FORMAT = "radio_intermediate_image_spatial_context_v1"
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

# The first probe used one-way attention over nearly full crops.  V2 instead
# gives each scale a deliberately different receptive field: final features
# retain a compact local neighbourhood plus image regions, intermediate
# features carry the structural neighbourhood, and ALIKE stays local.
BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES = (
    ContextAttentionScale("radio_final", grid_size=16, window_size=5),
    ContextAttentionScale("radio_intermediate", grid_size=16, window_size=9),
    ContextAttentionScale("alike", grid_size=32, window_size=13),
)
_BIDIRECTIONAL_GLOBAL_REGION_SIZE = {
    "radio_final": 4,
    "radio_intermediate": 4,
    "alike": 0,
}
_BIDIRECTIONAL_RAW_GLOBAL_REGION_SIZE = {
    # A 6x6 regional grid preserves facade phase substantially better than
    # V2's 4x4 pooling while keeping candidate-conditioned attention bounded.
    "radio_final": 6,
    "radio_intermediate": 6,
    "alike": 0,
}


def context_attention_global_region_sizes(architecture: str) -> dict[str, int]:
    """Return the frozen candidate-conditioned global-region resolution."""

    if str(architecture) == "bidirectional_absolute_v2":
        return dict(_BIDIRECTIONAL_GLOBAL_REGION_SIZE)
    if str(architecture) in {
        "bidirectional_absolute_raw_v3",
        "bidirectional_absolute_raw_layout_v4",
        "bidirectional_absolute_dual_head_raw_layout_v5",
    }:
        return dict(_BIDIRECTIONAL_RAW_GLOBAL_REGION_SIZE)
    if str(architecture) == "legacy_oneway_v1":
        return {scale.name: 0 for scale in CONTEXT_ATTENTION_SCALES}
    raise ValueError("unsupported context-attention global-region architecture")


def context_attention_profile(
    architecture: str,
) -> tuple[tuple[str, ...], tuple[ContextAttentionScale, ...], bool]:
    """Return the immutable family/source profile for a contract architecture.

    The profile is part of descriptor-space lineage.  In particular, v2 is
    not an old local-context contract with a different model flag: it has
    distinct receptive fields and a bounded, candidate-conditioned global
    image factor.
    """

    if str(architecture) == "legacy_oneway_v1":
        return CONTEXT_ATTENTION_FAMILIES, CONTEXT_ATTENTION_SCALES, False
    if str(architecture) == "bidirectional_absolute_v2":
        return (
            ABSOLUTE_CONTEXT_LIKELIHOOD_V2_FAMILIES,
            BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES,
            True,
        )
    if str(architecture) == "bidirectional_absolute_raw_v3":
        return (
            ABSOLUTE_CONTEXT_LIKELIHOOD_V3_FAMILIES,
            BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES,
            True,
        )
    if str(architecture) == "bidirectional_absolute_raw_layout_v4":
        return (
            ABSOLUTE_CONTEXT_LIKELIHOOD_V4_FAMILIES,
            BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES,
            True,
        )
    if str(architecture) == "bidirectional_absolute_dual_head_raw_layout_v5":
        return (
            ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES,
            BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES,
            True,
        )
    raise ValueError("unsupported context-attention contract architecture")


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


@dataclass(frozen=True)
class ContextAttentionSourceHeaders:
    """Lightweight, lineage-validated context contract without descriptor grids.

    The strict RGB-only likelihood never reads context descriptors at runtime,
    but it still needs the shared image ownership table, RGB coordinate bridge,
    and descriptor widths to reconstruct a checkpoint-compatible module tree.
    Reading only NPZ metadata, image tables, and NPY member headers avoids
    materializing multi-gigabyte RADIO/ALIKE grids in that mode.
    """

    image_ids: np.ndarray
    image_sizes: np.ndarray
    descriptor_dimensions: Mapping[str, int]
    metadata_by_name: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.image_ids).astype(str).reshape(-1)
        image_sizes = np.asarray(self.image_sizes, dtype=np.int64).reshape(-1, 2)
        dimensions = {str(name): int(value) for name, value in self.descriptor_dimensions.items()}
        metadata = {
            str(name): dict(value) for name, value in self.metadata_by_name.items()
        }
        if (
            len(image_ids) == 0
            or len(set(image_ids.tolist())) != len(image_ids)
            or image_sizes.shape != (len(image_ids), 2)
            or np.any(image_sizes <= 0)
            or set(dimensions) != {"radio_final", "radio_intermediate", "alike"}
            or any(value <= 0 for value in dimensions.values())
            or set(metadata) != set(dimensions)
        ):
            raise ValueError("lightweight context source headers are invalid")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_sizes", image_sizes)
        object.__setattr__(self, "descriptor_dimensions", dimensions)
        object.__setattr__(self, "metadata_by_name", metadata)


def _load_npz_header_fields(
    path: Path,
    *,
    required_fields: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Read small NPZ fields without materializing any descriptor arrays."""

    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(required_fields).difference(payload.files)
        if missing:
            raise ValueError(f"context cache lacks {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError, KeyError) as error:
            raise ValueError("context cache metadata is invalid") from error
        arrays = {
            str(name): np.asarray(payload[str(name)]) for name in required_fields if name != "metadata_json"
        }
    if not isinstance(metadata, dict):
        raise ValueError("context cache metadata is invalid")
    return arrays, dict(metadata)


def _npz_member_shape(path: Path, member_name: str) -> tuple[int, ...]:
    """Read an NPY shape from a compressed NPZ member header only."""

    archive_path = Path(path)
    member = f"{str(member_name)}.npy"
    try:
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY header version {version}")
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"context cache descriptor header is invalid: {archive_path}") from error
    result = tuple(int(value) for value in shape)
    if not result or any(value <= 0 for value in result):
        raise ValueError("context cache descriptor header shape is invalid")
    return result


def _grid_descriptor_dimension_from_header(
    *, path: Path, image_count: int, grid_size: int
) -> int:
    shape = _npz_member_shape(path, f"grid{int(grid_size)}_descriptors")
    if len(shape) == 3:
        expected = (int(image_count), int(grid_size) ** 2)
        if shape[:2] != expected:
            raise ValueError("context cache descriptor header does not align with image table")
        dimension = int(shape[2])
    elif len(shape) == 4:
        expected = (int(image_count), int(grid_size), int(grid_size))
        if shape[:3] != expected:
            raise ValueError("context cache descriptor header does not align with image table")
        dimension = int(shape[3])
    else:
        raise ValueError("context cache descriptor header rank is invalid")
    if dimension <= 0:
        raise ValueError("context cache descriptor dimension is invalid")
    return dimension


def _reject_non_target_free_cache(metadata: Mapping[str, object], *, context: str) -> None:
    if bool(metadata.get("pose_or_ground_truth_used", True)):
        raise ValueError(f"{context} is not pose/GT free")
    if bool(metadata.get("image_retrieval_or_submap_used", True)):
        raise ValueError(f"{context} violates the no-retrieval protocol")
    if bool(metadata.get("render", False)):
        raise ValueError(f"{context} uses rendering")


def _load_raw_radio_final_grid16_cache(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load current raw RADIO-final grids without assuming a PCA sidecar.

    Raw multiscale final caches predate the generic spatial-cache schema and do
    not serialize ``image_sizes``.  Their source-image contract still records
    the processed RGB dimensions; callers must validate that declaration
    against an aligned cache which carries the per-image size table.
    """

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {"image_ids", "grid16_descriptors", "metadata_json"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"raw RADIO-final context cache lacks {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("raw RADIO-final context metadata is invalid") from error
        image_ids = np.asarray(payload["image_ids"]).astype(str).reshape(-1)
        grid = np.asarray(payload["grid16_descriptors"], dtype=np.float32)
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != _RAW_RADIO_FINAL_CONTEXT_FORMAT
        or len(image_ids) == 0
        or len(set(image_ids.tolist())) != len(image_ids)
        or grid.ndim != 3
        or grid.shape[:2] != (len(image_ids), 16 * 16)
        or grid.shape[2] <= 0
        or np.any(~np.isfinite(grid))
    ):
        raise ValueError("raw RADIO-final context cache arrays are invalid")
    norms = np.linalg.norm(grid, axis=-1)
    if np.max(np.abs(norms - 1.0)) > 5e-3:
        raise ValueError("raw RADIO-final context descriptors are not normalized")
    dimensions = dict(metadata.get("image_source_contract", {})).get(
        "source_image_dimensions"
    )
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ValueError("raw RADIO-final context lacks a source image dimension contract")
    return image_ids, grid, dict(metadata)


def _validate_raw_final_size_contract(
    *, metadata: Mapping[str, object], image_sizes: np.ndarray
) -> dict[str, object]:
    """Build an explicit normalized-coordinate bridge for raw-final grids.

    Raw final grids were historically extracted on processed RGB, while SfM
    coordinates can live in an isotropically resized frame.  The adaptive grid
    itself is sampled in normalized image coordinates, so this is valid only
    when the bridge is explicit and isotropic.  Raw caches without per-image
    dimensions are intentionally limited to one resolution distribution.
    """

    sizes = np.asarray(image_sizes, dtype=np.int64)
    if sizes.ndim != 2 or sizes.shape[1] != 2 or len(sizes) == 0 or np.any(sizes <= 0):
        raise ValueError("aligned raw RADIO-final image sizes are invalid")
    declared_raw = dict(metadata.get("image_source_contract", {})).get(
        "source_image_dimensions"
    )
    if not isinstance(declared_raw, Mapping):
        raise ValueError("raw RADIO-final context lacks a source image dimension contract")
    declared: dict[str, int] = {}
    for key, value in declared_raw.items():
        try:
            width_text, height_text = str(key).split("x", 1)
            width, height, count = int(width_text), int(height_text), int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("raw RADIO-final source image dimension contract is invalid") from error
        if width <= 0 or height <= 0 or count <= 0:
            raise ValueError("raw RADIO-final source image dimension contract is invalid")
        declared[f"{width}x{height}"] = count
    if len(declared) != 1:
        raise ValueError("raw RADIO-final cache without image sizes must have one source resolution")
    aligned_unique, aligned_counts = np.unique(sizes, axis=0, return_counts=True)
    if len(aligned_unique) != 1 or int(aligned_counts[0]) != len(sizes):
        raise ValueError("raw RADIO-final cache without image sizes requires one aligned resolution")
    raw_key, raw_count = next(iter(declared.items()))
    raw_width, raw_height = (int(value) for value in raw_key.split("x", 1))
    aligned_width, aligned_height = (int(value) for value in aligned_unique[0].tolist())
    if int(raw_count) != len(sizes):
        raise ValueError("raw RADIO-final image count differs from aligned cache")
    scale_x = float(raw_width) / float(aligned_width)
    scale_y = float(raw_height) / float(aligned_height)
    if not math.isclose(scale_x, scale_y, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("raw RADIO-final and aligned cache resolutions are not isotropically related")
    return {
        "format": "normalized_adaptive_grid_coordinate_bridge_v1",
        "raw_rgb_size": [raw_width, raw_height],
        "aligned_coordinate_size": [aligned_width, aligned_height],
        "raw_pixels_per_aligned_pixel": scale_x,
    }


def _load_radio_intermediate_context(path: Path):
    """Load either the scoped PCA or current raw intermediate cache."""

    with np.load(Path(path), allow_pickle=False) as payload:
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("RADIO-intermediate context metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("RADIO-intermediate context metadata is invalid")
    cache_format = str(metadata.get("format", ""))
    if cache_format not in {
        _RADIO_INTERMEDIATE_CONTEXT_PCA_FORMAT,
        _RAW_RADIO_INTERMEDIATE_CONTEXT_FORMAT,
    }:
        raise ValueError("unsupported RADIO-intermediate context cache")
    cache = load_spatial_image_context_cache(Path(path), expected_format=cache_format)
    _reject_non_target_free_cache(cache.metadata, context="RADIO-intermediate context cache")
    if cache_format == _RADIO_INTERMEDIATE_CONTEXT_PCA_FORMAT and cache.metadata.get(
        "pca_fit_scope"
    ) not in _SAFE_PCA_FIT_SCOPES:
        raise ValueError("RADIO-intermediate context PCA was not fit only on mapping-side images")
    if int(cache.metadata.get("intermediate_index", 0)) != -6:
        raise ValueError("RADIO-intermediate context cache uses an unexpected layer")
    return cache


def load_context_attention_source_headers(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    expected_radio_checkpoint: str,
) -> ContextAttentionSourceHeaders:
    """Validate context lineage while loading only the RGB-only header contract.

    This is deliberately narrower than :func:`load_context_attention_sources`:
    descriptor values are not read or normalized because the caller has
    declared that it will execute the strict RGB-only cost-volume path.  The
    exact same image, checkpoint, grid-shape, target-free, and coordinate
    bridge contracts are still enforced.
    """

    final_arrays, final_metadata = _load_npz_header_fields(
        Path(radio_final_context_cache),
        required_fields=("image_ids", "metadata_json"),
    )
    final_format = str(final_metadata.get("format", ""))
    final_ids = np.asarray(final_arrays["image_ids"]).astype(str).reshape(-1)
    if final_format == RADIO_FINAL_CONTEXT_PCA_FORMAT:
        _reject_non_target_free_cache(final_metadata, context="RADIO-final context cache")
        if final_metadata.get("pca_fit_scope") not in _SAFE_PCA_FIT_SCOPES:
            raise ValueError("RADIO-final context PCA was not fit only on mapping-side images")
        pca_arrays, pca_metadata = _load_npz_header_fields(
            Path(radio_final_context_cache),
            required_fields=("image_ids", "image_sizes", "metadata_json"),
        )
        if pca_metadata != final_metadata or not np.array_equal(
            np.asarray(pca_arrays["image_ids"]).astype(str).reshape(-1), final_ids
        ):
            raise ValueError("RADIO-final context cache header is inconsistent")
        final_sizes: np.ndarray | None = np.asarray(pca_arrays["image_sizes"], dtype=np.int64)
    elif final_format == _RAW_RADIO_FINAL_CONTEXT_FORMAT:
        _reject_non_target_free_cache(final_metadata, context="raw RADIO-final context cache")
        final_sizes = None
    else:
        raise ValueError("unsupported RADIO-final context cache")
    if expected_radio_checkpoint and str(final_metadata.get("radio_checkpoint_sha256", "")) != str(
        expected_radio_checkpoint
    ):
        raise ValueError("frozen layout and RADIO-final cache checkpoints differ")
    final_dimension = _grid_descriptor_dimension_from_header(
        path=Path(radio_final_context_cache), image_count=len(final_ids), grid_size=16
    )

    intermediate_arrays, intermediate_metadata = _load_npz_header_fields(
        Path(radio_intermediate_context_cache),
        required_fields=("image_ids", "image_sizes", "metadata_json"),
    )
    intermediate_format = str(intermediate_metadata.get("format", ""))
    if intermediate_format not in {
        _RADIO_INTERMEDIATE_CONTEXT_PCA_FORMAT,
        _RAW_RADIO_INTERMEDIATE_CONTEXT_FORMAT,
    }:
        raise ValueError("unsupported RADIO-intermediate context cache")
    _reject_non_target_free_cache(intermediate_metadata, context="RADIO-intermediate context cache")
    if intermediate_format == _RADIO_INTERMEDIATE_CONTEXT_PCA_FORMAT and intermediate_metadata.get(
        "pca_fit_scope"
    ) not in _SAFE_PCA_FIT_SCOPES:
        raise ValueError("RADIO-intermediate context PCA was not fit only on mapping-side images")
    if int(intermediate_metadata.get("intermediate_index", 0)) != -6:
        raise ValueError("RADIO-intermediate context cache uses an unexpected layer")
    if str(intermediate_metadata.get("radio_checkpoint_sha256", "")) != str(
        final_metadata.get("radio_checkpoint_sha256", "")
    ):
        raise ValueError("RADIO final/intermediate checkpoints differ")
    intermediate_ids = np.asarray(intermediate_arrays["image_ids"]).astype(str).reshape(-1)
    intermediate_sizes = np.asarray(intermediate_arrays["image_sizes"], dtype=np.int64)
    intermediate_dimension = _grid_descriptor_dimension_from_header(
        path=Path(radio_intermediate_context_cache),
        image_count=len(intermediate_ids),
        grid_size=16,
    )

    alike_arrays, alike_metadata = _load_npz_header_fields(
        Path(alike_spatial_context_cache),
        required_fields=("image_ids", "image_sizes", "metadata_json"),
    )
    if str(alike_metadata.get("format", "")) != _ALIKE_SPATIAL_CONTEXT_FORMAT:
        raise ValueError("unsupported ALIKE spatial context cache")
    _reject_non_target_free_cache(alike_metadata, context="ALIKE context cache")
    if not str(alike_metadata.get("alike_checkpoint_sha256", "")):
        raise ValueError("ALIKE context cache lacks checkpoint lineage")
    alike_ids = np.asarray(alike_arrays["image_ids"]).astype(str).reshape(-1)
    alike_sizes = np.asarray(alike_arrays["image_sizes"], dtype=np.int64)
    alike_dimension = _grid_descriptor_dimension_from_header(
        path=Path(alike_spatial_context_cache), image_count=len(alike_ids), grid_size=32
    )

    manifest = str(final_metadata.get("source_image_manifest_sha256", ""))
    if not manifest:
        raise ValueError("RADIO-final context cache lacks a source image manifest")
    if final_sizes is None:
        if not np.array_equal(intermediate_ids, final_ids):
            raise ValueError("RADIO-intermediate context cache does not align with raw RADIO-final images")
        final_sizes = intermediate_sizes
        final_metadata = {
            **dict(final_metadata),
            "coordinate_bridge": _validate_raw_final_size_contract(
                metadata=final_metadata, image_sizes=final_sizes
            ),
        }
    for name, image_ids, image_sizes, metadata in (
        ("RADIO-intermediate", intermediate_ids, intermediate_sizes, intermediate_metadata),
        ("ALIKE", alike_ids, alike_sizes, alike_metadata),
    ):
        if not np.array_equal(image_ids, final_ids) or not np.array_equal(image_sizes, final_sizes):
            raise ValueError(f"{name} context cache does not align with RADIO-final images")
        if str(metadata.get("source_image_manifest_sha256", "")) != manifest:
            raise ValueError(f"{name} context cache source image manifest differs")
    return ContextAttentionSourceHeaders(
        image_ids=final_ids,
        image_sizes=np.asarray(final_sizes, dtype=np.int64),
        descriptor_dimensions={
            "radio_final": final_dimension,
            "radio_intermediate": intermediate_dimension,
            "alike": alike_dimension,
        },
        metadata_by_name={
            "radio_final": final_metadata,
            "radio_intermediate": intermediate_metadata,
            "alike": alike_metadata,
        },
    )


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

    with np.load(Path(radio_final_context_cache), allow_pickle=False) as payload:
        try:
            final_format = str(
                json.loads(str(np.asarray(payload["metadata_json"]).item())).get("format", "")
            )
        except (TypeError, ValueError, json.JSONDecodeError, KeyError) as error:
            raise ValueError("RADIO-final context metadata is invalid") from error
    final_is_raw = final_format == _RAW_RADIO_FINAL_CONTEXT_FORMAT
    if final_format == RADIO_FINAL_CONTEXT_PCA_FORMAT:
        final = load_radio_final_context_pca_cache(Path(radio_final_context_cache))
        final_metadata = final.metadata
        _reject_non_target_free_cache(final_metadata, context="RADIO-final context cache")
        if final_metadata.get("pca_fit_scope") not in _SAFE_PCA_FIT_SCOPES:
            raise ValueError("RADIO-final context PCA was not fit only on mapping-side images")
        if final.grid16_descriptors is None:
            raise ValueError("RADIO-final context cache lacks grid16 descriptors")
        final_ids = np.asarray(final.image_ids).astype(str)
        final_sizes: np.ndarray | None = np.asarray(final.image_sizes, dtype=np.int64)
        final_grid = np.asarray(final.grid16_descriptors)
    elif final_is_raw:
        final_ids, raw_final_grid, final_metadata = _load_raw_radio_final_grid16_cache(
            Path(radio_final_context_cache)
        )
        _reject_non_target_free_cache(final_metadata, context="raw RADIO-final context cache")
        final_sizes = None
        final_grid = raw_final_grid
    else:
        raise ValueError("unsupported RADIO-final context cache")
    if expected_radio_checkpoint and str(final_metadata.get("radio_checkpoint_sha256", "")) != str(
        expected_radio_checkpoint
    ):
        raise ValueError("frozen layout and RADIO-final cache checkpoints differ")

    intermediate = _load_radio_intermediate_context(Path(radio_intermediate_context_cache))
    intermediate_metadata = intermediate.metadata
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

    manifest = str(final_metadata.get("source_image_manifest_sha256", ""))
    if not manifest:
        raise ValueError("RADIO-final context cache lacks a source image manifest")
    if final_sizes is None:
        if not np.array_equal(intermediate.image_ids.astype(str), final_ids):
            raise ValueError("RADIO-intermediate context cache does not align with raw RADIO-final images")
        final_sizes = np.asarray(intermediate.image_sizes, dtype=np.int64)
        final_metadata = {
            **dict(final_metadata),
            "coordinate_bridge": _validate_raw_final_size_contract(
                metadata=final_metadata, image_sizes=final_sizes
            ),
        }
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
            grid=np.asarray(final_grid).reshape(len(final_ids), 16, 16, -1),
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
    # ``logsumexp`` over an all-masked candidate has a finite-looking forward
    # value only after a later ``where``.  Its backward path is still undefined
    # and can turn a zero upstream gradient into NaN.  Give empty padded
    # candidates one explicit zero evidence slot before marginalization; the
    # returned residual remains exactly zero for those candidates below.
    safe_valid = valid.clone()
    empty = count == 0
    safe_valid[..., 0] = safe_valid[..., 0] | empty
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    masked = torch.where(
        safe_valid,
        safe_logits,
        torch.full_like(logits, torch.finfo(logits.dtype).min),
    )
    output = torch.logsumexp(masked, dim=2) - torch.log(
        safe_valid.sum(dim=2).clamp_min(1).to(dtype=logits.dtype)
    )
    # Invalid padded candidates have zero base probability and therefore never
    # enter the candidate softmax.  Keep their residual finite so a partial
    # top-L row remains a valid frozen inference artifact.
    return torch.where(~empty, output, torch.zeros_like(output))


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


def _global_region_coordinates(*, region_size: int, device: torch.device) -> torch.Tensor:
    """Return stable normalized centres for a pooled full-image region grid."""

    size = int(region_size)
    if size <= 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    values = (torch.arange(size, dtype=torch.float32, device=device) + 0.5) / float(size)
    rows, columns = torch.meshgrid(values, values, indexing="ij")
    return torch.stack([columns, rows], dim=-1).reshape(-1, 2)


def _masked_token_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pool token features without allowing padded crop cells to contribute."""

    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("masked token pool inputs are incompatible")
    weights = valid.to(dtype=values.dtype)
    # ``-inf * 0`` is NaN.  Correlation summaries deliberately use -inf for
    # masked rows, so invalid values must be replaced before pooling instead
    # of relying on multiplication by a zero mask.
    selected = torch.where(valid[..., None], values, torch.zeros_like(values))
    return torch.sum(selected, dim=1) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)


def _candidate_set_visual_statistics(
    values: torch.Tensor, candidate_valid: torch.Tensor
) -> torch.Tensor:
    """Summarise candidate visual likelihoods without candidate-order leakage.

    The explicit null state must be able to learn from visual evidence too.
    It receives only permutation-invariant statistics of the per-candidate
    visual likelihoods, never a coarse prior, track ID, pose, or target field.
    """

    if (
        values.ndim != 2
        or candidate_valid.shape != values.shape
        or candidate_valid.dtype != torch.bool
    ):
        raise ValueError("candidate visual-statistic inputs are incompatible")
    count = candidate_valid.sum(dim=1, keepdim=True)
    selected = torch.where(candidate_valid, values, torch.zeros_like(values))
    mean = selected.sum(dim=1, keepdim=True) / count.clamp_min(1).to(dtype=values.dtype)
    variance = torch.where(
        candidate_valid, (values - mean).square(), torch.zeros_like(values)
    ).sum(dim=1, keepdim=True) / count.clamp_min(1).to(dtype=values.dtype)
    masked = torch.where(
        candidate_valid, values, torch.full_like(values, -torch.inf)
    )
    maximum = torch.max(masked, dim=1, keepdim=True).values
    log_mean = torch.logsumexp(masked, dim=1, keepdim=True) - torch.log(
        count.clamp_min(1).to(dtype=values.dtype)
    )
    zeros = torch.zeros_like(mean)
    return torch.cat(
        [
            torch.where(count > 0, mean, zeros),
            torch.where(count > 0, maximum, zeros),
            torch.where(count > 0, torch.sqrt(variance.clamp_min(0.0)), zeros),
            torch.where(count > 0, log_mean, zeros),
        ],
        dim=1,
    )


class _BidirectionalAbsoluteContextScaleEncoder(nn.Module):
    """Per-scale 2-D query/support matcher with explicit correlation summaries.

    The original probe used a single query-to-support attention pass and pooled
    only its updated query tokens.  This encoder keeps the two image grids
    separate through self-attention and both cross directions, then exposes a
    compact correlation summary to the likelihood head.  It is deliberately
    candidate/view local: no candidate axis is ever mixed here.
    """

    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        *,
        local_token_count: int,
        global_region_size: int,
        shared_descriptor_projection: bool = False,
        include_raw_descriptor_statistics: bool = False,
    ) -> None:
        super().__init__()
        if (
            int(descriptor_dim) <= 0
            or int(hidden_dim) < 8
            or int(heads) <= 0
            or int(hidden_dim) % int(heads) != 0
            or int(local_token_count) <= 0
            or int(global_region_size) < 0
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise ValueError("bidirectional context encoder dimensions are invalid")
        self.local_token_count = int(local_token_count)
        self.global_region_size = int(global_region_size)
        self.shared_descriptor_projection = bool(shared_descriptor_projection)
        self.include_raw_descriptor_statistics = bool(include_raw_descriptor_statistics)
        if self.shared_descriptor_projection:
            # RADIO/ALIKE grids on both sides already live in the same frozen
            # descriptor space.  V2 used independent random projections, so
            # its correlation branch first had to relearn comparability from
            # sparse labels.  V3 keeps a shared learned adaptation instead.
            self.descriptor_projection = nn.Linear(
                int(descriptor_dim), int(hidden_dim), bias=False
            )
            self.query_projection = None
            self.support_projection = None
        else:
            self.descriptor_projection = None
            self.query_projection = nn.Linear(int(descriptor_dim), int(hidden_dim), bias=False)
            self.support_projection = nn.Linear(int(descriptor_dim), int(hidden_dim), bias=False)
        self.query_local_position = nn.Sequential(
            nn.Linear(2, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.support_local_position = nn.Sequential(
            nn.Linear(2, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.query_absolute_position = nn.Sequential(
            nn.Linear(2, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.support_absolute_position = nn.Sequential(
            nn.Linear(2, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.query_self_attention = nn.MultiheadAttention(
            int(hidden_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.support_self_attention = nn.MultiheadAttention(
            int(hidden_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.query_to_support_attention = nn.MultiheadAttention(
            int(hidden_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.support_to_query_attention = nn.MultiheadAttention(
            int(hidden_dim), int(heads), dropout=float(dropout), batch_first=True
        )
        self.query_self_norm = nn.LayerNorm(int(hidden_dim))
        self.support_self_norm = nn.LayerNorm(int(hidden_dim))
        self.query_cross_norm = nn.LayerNorm(int(hidden_dim))
        self.support_cross_norm = nn.LayerNorm(int(hidden_dim))
        # q pool, support pool, q<-support pool, support<-q pool and six
        # learned-space correlation statistics. V3 additionally exposes six
        # raw frozen-descriptor cost-volume statistics to the calibration head.
        # The encoder itself never receives the coarse prior.
        statistic_width = 6 + (6 if self.include_raw_descriptor_statistics else 0)
        self.output = nn.Sequential(
            nn.LayerNorm(4 * int(hidden_dim) + statistic_width),
            nn.Linear(4 * int(hidden_dim) + statistic_width, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
        )

    def _tokens(
        self,
        *,
        descriptors: torch.Tensor,
        local_relative_coordinates: torch.Tensor,
        absolute_coordinates: torch.Tensor,
        global_descriptors: torch.Tensor,
        global_coordinates: torch.Tensor,
        projection: nn.Linear,
        local_position: nn.Module,
        absolute_position: nn.Module,
    ) -> torch.Tensor:
        if (
            descriptors.ndim != 3
            or descriptors.shape[1] != self.local_token_count
            or local_relative_coordinates.shape != (self.local_token_count, 2)
            or absolute_coordinates.shape != (*descriptors.shape[:2], 2)
            or global_descriptors.ndim != 3
            or global_descriptors.shape[0] != descriptors.shape[0]
            or global_descriptors.shape[2] != descriptors.shape[2]
            or global_coordinates.shape != (global_descriptors.shape[1], 2)
        ):
            raise ValueError("bidirectional context token inputs are incompatible")
        descriptor_dtype = projection.weight.dtype
        local = projection(descriptors.to(dtype=descriptor_dtype))
        local = local + local_position(
            local_relative_coordinates.to(dtype=local.dtype)
        ).unsqueeze(0)
        local = local + absolute_position(absolute_coordinates.to(dtype=local.dtype))
        if global_descriptors.shape[1] == 0:
            return local
        global_tokens = projection(global_descriptors.to(dtype=descriptor_dtype))
        global_tokens = global_tokens + absolute_position(
            global_coordinates.to(dtype=global_tokens.dtype)
        ).unsqueeze(0)
        return torch.cat([local, global_tokens], dim=1)

    @staticmethod
    def _correlation_statistics(
        *,
        query_tokens: torch.Tensor,
        query_valid: torch.Tensor,
        support_tokens: torch.Tensor,
        support_valid: torch.Tensor,
        local_token_count: int,
    ) -> torch.Tensor:
        """Summarise the full 2-D cost volume without discarding its modes."""

        if (
            query_tokens.ndim != 3
            or support_tokens.shape != query_tokens.shape
            or query_valid.shape != query_tokens.shape[:2]
            or support_valid.shape != query_tokens.shape[:2]
            or int(local_token_count) <= 0
            or int(local_token_count) > query_tokens.shape[1]
        ):
            raise ValueError("bidirectional correlation inputs are incompatible")
        query_normalized = F.normalize(query_tokens.float(), dim=-1)
        support_normalized = F.normalize(support_tokens.float(), dim=-1)
        correlation = torch.matmul(query_normalized, support_normalized.transpose(1, 2))
        pair_valid = query_valid[:, :, None] & support_valid[:, None, :]
        masked = torch.where(pair_valid, correlation, torch.full_like(correlation, -torch.inf))
        query_peak = torch.max(masked, dim=2).values
        support_peak = torch.max(masked, dim=1).values
        query_peak_mean = _masked_token_mean(query_peak[:, :, None], query_valid).squeeze(1)
        support_peak_mean = _masked_token_mean(support_peak[:, :, None], support_valid).squeeze(1)
        # The diagonal is a relative-layout cue, not a correspondence target.
        local_query_valid = query_valid[:, :local_token_count]
        local_support_valid = support_valid[:, :local_token_count]
        diagonal_valid = local_query_valid & local_support_valid
        diagonal = torch.diagonal(correlation[:, :local_token_count, :local_token_count], dim1=1, dim2=2)
        diagonal_mean = _masked_token_mean(diagonal[:, :, None], diagonal_valid).squeeze(1)
        temperature = 0.10
        query_probability = torch.softmax(masked / temperature, dim=2)
        support_probability = torch.softmax(masked / temperature, dim=1)
        query_probability = torch.where(pair_valid, query_probability, torch.zeros_like(query_probability))
        support_probability = torch.where(pair_valid, support_probability, torch.zeros_like(support_probability))
        query_entropy = -torch.sum(
            query_probability * torch.log(query_probability.clamp_min(1e-12)), dim=2
        )
        support_entropy = -torch.sum(
            support_probability * torch.log(support_probability.clamp_min(1e-12)), dim=1
        )
        query_entropy_mean = _masked_token_mean(query_entropy[:, :, None], query_valid).squeeze(1)
        support_entropy_mean = _masked_token_mean(support_entropy[:, :, None], support_valid).squeeze(1)
        # A soft mutual score stays finite even on repeated structure and makes
        # the cost-volume's ambiguity visible to the likelihood head.
        reciprocal = torch.sum(query_probability * support_probability, dim=(1, 2))
        reciprocal = reciprocal / pair_valid.to(dtype=correlation.dtype).sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        return torch.stack(
            [
                query_peak_mean,
                support_peak_mean,
                diagonal_mean,
                query_entropy_mean,
                support_entropy_mean,
                reciprocal,
            ],
            dim=1,
        ).to(dtype=query_tokens.dtype)

    def forward_with_statistics(
        self,
        *,
        query_local_descriptors: torch.Tensor,
        query_local_valid: torch.Tensor,
        query_local_relative_coordinates: torch.Tensor,
        query_local_absolute_coordinates: torch.Tensor,
        query_global_descriptors: torch.Tensor,
        support_local_descriptors: torch.Tensor,
        support_local_valid: torch.Tensor,
        support_local_relative_coordinates: torch.Tensor,
        support_local_absolute_coordinates: torch.Tensor,
        support_global_descriptors: torch.Tensor,
        global_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_projection = (
            self.descriptor_projection
            if self.shared_descriptor_projection
            else self.query_projection
        )
        support_projection = (
            self.descriptor_projection
            if self.shared_descriptor_projection
            else self.support_projection
        )
        if query_projection is None or support_projection is None:
            raise RuntimeError("bidirectional descriptor projection is missing")
        query = self._tokens(
            descriptors=query_local_descriptors,
            local_relative_coordinates=query_local_relative_coordinates,
            absolute_coordinates=query_local_absolute_coordinates,
            global_descriptors=query_global_descriptors,
            global_coordinates=global_coordinates,
            projection=query_projection,
            local_position=self.query_local_position,
            absolute_position=self.query_absolute_position,
        )
        support = self._tokens(
            descriptors=support_local_descriptors,
            local_relative_coordinates=support_local_relative_coordinates,
            absolute_coordinates=support_local_absolute_coordinates,
            global_descriptors=support_global_descriptors,
            global_coordinates=global_coordinates,
            projection=support_projection,
            local_position=self.support_local_position,
            absolute_position=self.support_absolute_position,
        )
        query_global_valid = torch.ones(
            (len(query), query.shape[1] - self.local_token_count),
            dtype=torch.bool,
            device=query.device,
        )
        support_global_valid = torch.ones(
            (len(support), support.shape[1] - self.local_token_count),
            dtype=torch.bool,
            device=support.device,
        )
        query_valid = torch.cat([query_local_valid, query_global_valid], dim=1)
        support_valid = torch.cat([support_local_valid, support_global_valid], dim=1)
        if torch.any(~torch.any(query_valid, dim=1)) or torch.any(~torch.any(support_valid, dim=1)):
            raise RuntimeError("bidirectional context encoder received an empty token set")
        query_self, _ = self.query_self_attention(
            query, query, query, key_padding_mask=~query_valid, need_weights=False
        )
        support_self, _ = self.support_self_attention(
            support, support, support, key_padding_mask=~support_valid, need_weights=False
        )
        query_self = self.query_self_norm(query + query_self)
        support_self = self.support_self_norm(support + support_self)
        query_cross, _ = self.query_to_support_attention(
            query_self,
            support_self,
            support_self,
            key_padding_mask=~support_valid,
            need_weights=False,
        )
        support_cross, _ = self.support_to_query_attention(
            support_self,
            query_self,
            query_self,
            key_padding_mask=~query_valid,
            need_weights=False,
        )
        query_updated = self.query_cross_norm(query_self + query_cross)
        support_updated = self.support_cross_norm(support_self + support_cross)
        statistics = self._correlation_statistics(
            query_tokens=query_self,
            query_valid=query_valid,
            support_tokens=support_self,
            support_valid=support_valid,
            local_token_count=self.local_token_count,
        )
        raw_statistics: torch.Tensor | None = None
        if self.include_raw_descriptor_statistics:
            raw_query = torch.cat([query_local_descriptors, query_global_descriptors], dim=1)
            raw_support = torch.cat([support_local_descriptors, support_global_descriptors], dim=1)
            raw_statistics = self._correlation_statistics(
                query_tokens=raw_query,
                query_valid=query_valid,
                support_tokens=raw_support,
                support_valid=support_valid,
                local_token_count=self.local_token_count,
            )
        output_inputs = [
            _masked_token_mean(query_self, query_valid),
            _masked_token_mean(support_self, support_valid),
            _masked_token_mean(query_updated, query_valid),
            _masked_token_mean(support_updated, support_valid),
            statistics,
        ]
        if raw_statistics is not None:
            output_inputs.append(raw_statistics.to(dtype=statistics.dtype))
        return self.output(torch.cat(output_inputs, dim=1)), raw_statistics

    def forward(
        self,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        encoded, _raw_statistics = self.forward_with_statistics(**kwargs)
        return encoded


class CandidateBidirectionalAbsoluteContextLikelihood(nn.Module):
    """Candidate-conditioned multi-scale absolute image likelihood, V2-V5.

    The forward pass has no pose, residual, landmark label, or hypothesis input.
    It only maps a fixed query row and its fixed top-L support observations to
    a residual over the supplied candidate posterior plus its explicit null.
    ``forward_with_details`` exposes per-scale and per-view outputs for later
    target-only audits without changing the inference result.  V3 keeps query
    and support descriptors in a shared projected space and exposes direct
    frozen-descriptor cost-volume statistics.  V4 removes the attention path
    entirely and calibrates only an explicit raw 6x6 layout cost volume plus
    matched anchor-position controls.  V5 separates its set-valued geometry
    posterior from an exact-track identity posterior, so their different SfM
    supervision semantics never conflict on a single final logit. Every
    returned residual remains a candidate-reranking factor, not an independent
    pose likelihood.
    """

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
        hidden_dim: int = 48,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if str(family) not in {
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V2_FAMILIES,
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V3_FAMILIES,
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V4_FAMILIES,
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES,
        }:
            raise ValueError("unsupported bidirectional absolute-context family")
        if int(hidden_dim) < 8 or int(hidden_dim) % int(heads) != 0 or int(hidden_dim) % 2:
            raise ValueError("bidirectional absolute-context hidden dimension is invalid")
        self.family = str(family)
        self._uses_dual_identity_head = self.family in ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES
        self._uses_raw_layout_cost_volume = self.family in {
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V4_FAMILIES,
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES,
        }
        self._uses_raw_descriptor_evidence = self.family in {
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V3_FAMILIES,
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V4_FAMILIES,
            *ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES,
        }
        self._uses_attention_context = not self._uses_raw_layout_cost_volume
        self._uses_position_only = self.family.endswith(
            (
                "position_control_v2",
                "position_control_v3",
                "position_control_v4",
                "position_control_v5",
            )
        )
        self._global_region_sizes = (
            _BIDIRECTIONAL_RAW_GLOBAL_REGION_SIZE
            if self._uses_raw_descriptor_evidence
            else _BIDIRECTIONAL_GLOBAL_REGION_SIZE
        )
        expected_names = {scale.name for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES}
        if set(sources) != expected_names:
            raise ValueError("bidirectional absolute-context source set differs from its fixed profile")
        source_tensors = {name: torch.as_tensor(value) for name, value in sources.items()}
        descriptor_dims = {int(values.shape[-1]) for values in source_tensors.values()}
        if len(descriptor_dims) != 1:
            raise ValueError("bidirectional absolute-context source descriptor dimensions differ")
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
            raise ValueError("bidirectional absolute-context static arrays are incompatible")
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            values = source_tensors[scale.name]
            if values.ndim != 4 or values.shape[:3] != (
                len(sizes),
                scale.grid_size,
                scale.grid_size,
            ):
                raise ValueError("bidirectional absolute-context source grid differs from its profile")
            grid = values.to(dtype=torch.float16)
            self.register_buffer(f"_{scale.name}_grid", grid, persistent=False)
            region_size = int(self._global_region_sizes[scale.name])
            if region_size:
                pooled = F.adaptive_avg_pool2d(
                    grid.float().permute(0, 3, 1, 2), (region_size, region_size)
                ).permute(0, 2, 3, 1).reshape(len(grid), region_size**2, grid.shape[-1])
                self.register_buffer(
                    f"_{scale.name}_global", pooled.to(dtype=torch.float16), persistent=False
                )
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
                scale.name: _BidirectionalAbsoluteContextScaleEncoder(
                    descriptor_dim,
                    int(hidden_dim),
                    int(heads),
                    float(dropout),
                    local_token_count=int(scale.window_size) ** 2,
                    global_region_size=int(self._global_region_sizes[scale.name]),
                    shared_descriptor_projection=self._uses_raw_descriptor_evidence,
                    include_raw_descriptor_statistics=self._uses_raw_descriptor_evidence,
                )
                for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
            }
            if self._uses_attention_context
            else {}
        )
        feature_width = len(BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES) * (int(hidden_dim) // 2)
        self.scale_heads = nn.ModuleDict(
            {
                scale.name: nn.Linear(int(hidden_dim) // 2, 1)
                for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
            }
            if self._uses_attention_context
            else {}
        )
        self.raw_scale_heads = (
            nn.ModuleDict(
                {scale.name: nn.Linear(6, 1) for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES}
            )
            if self._uses_raw_descriptor_evidence and self._uses_attention_context
            else None
        )
        self.raw_layout_heads = (
            nn.ModuleDict(
                {
                    scale.name: nn.Linear(
                        6
                        + 3 * int(self._global_region_sizes[scale.name]) ** 2
                        + 6,
                        1,
                    )
                    for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
                }
            )
            if self._uses_raw_layout_cost_volume
            else None
        )
        self.identity_raw_layout_heads = (
            nn.ModuleDict(
                {
                    scale.name: nn.Linear(
                        6
                        + 3 * int(self._global_region_sizes[scale.name]) ** 2
                        + 6,
                        1,
                    )
                    for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
                    if scale.name in _V5_IDENTITY_CONTEXT_SCALE_NAMES
                }
            )
            if self._uses_dual_identity_head
            else None
        )
        self.fusion_head = (
            nn.Sequential(
                nn.LayerNorm(feature_width),
                nn.Linear(feature_width, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
            if self._uses_attention_context
            else None
        )
        # This is deliberately separate from the candidate residual head.  It
        # learns whether the *set* of fixed candidates has any plausible visual
        # support, while remaining invariant to candidate ordering and blind to
        # the coarse posterior used below as an immutable prior.
        self.null_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.identity_null_head = (
            nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
            if self._uses_dual_identity_head
            else None
        )
        # A zero residual exactly replays the immutable candidate prior.
        for head in self.scale_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.raw_scale_heads is not None:
            for head in self.raw_scale_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.raw_layout_heads is not None:
            for head in self.raw_layout_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.identity_raw_layout_heads is not None:
            for head in self.identity_raw_layout_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.fusion_head is not None:
            nn.init.zeros_(self.fusion_head[-1].weight)
            nn.init.zeros_(self.fusion_head[-1].bias)
        nn.init.zeros_(self.null_head[-1].weight)
        nn.init.zeros_(self.null_head[-1].bias)
        if self.identity_null_head is not None:
            nn.init.zeros_(self.identity_null_head[-1].weight)
            nn.init.zeros_(self.identity_null_head[-1].bias)

    @property
    def row_count(self) -> int:
        return int(self._query_image_indices.shape[0])

    def _grid(self, name: str) -> torch.Tensor:
        return getattr(self, f"_{name}_grid")

    def _global(self, name: str) -> torch.Tensor:
        if int(self._global_region_sizes[str(name)]) == 0:
            # Do not register a [N, 0, C] buffer.  DDP's initial coalesced
            # broadcast cannot synchronize a zero-length inner dimension even
            # though this branch intentionally has no global tokens.
            grid = self._grid(name)
            return grid.new_empty((len(grid), 0, grid.shape[-1]))
        return getattr(self, f"_{name}_global")

    def _row_scale_inputs(
        self,
        rows: torch.Tensor,
        scale: ContextAttentionScale,
        *,
        query_xy_per_edge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Materialize one scale's fixed support and optional dynamic query crops.

        The normal candidate-reranking path has exactly one observed query
        coordinate per row.  The held-out pose probe instead projects each
        fixed candidate under a frozen hypothesis, so it needs a separate
        query crop for every candidate/support-view edge.  Only that query
        crop is dynamic: support observations, candidate identities, and
        global image descriptors remain fixed.
        """
        selected_rows = rows.to(dtype=torch.long)
        query_indices = self._query_image_indices.index_select(0, selected_rows)
        query_xy = self._query_xy.index_select(0, selected_rows)
        support_indices = self._support_image_indices.index_select(0, selected_rows)
        support_xy = self._support_xy.index_select(0, selected_rows)
        view_valid = self._view_valid.index_select(0, selected_rows)
        batch, candidate_count, view_count = support_indices.shape
        edge_count = int(batch * candidate_count * view_count)
        grid = self._grid(scale.name)
        safe_support_indices = support_indices.reshape(-1).clamp(0, len(self._image_sizes) - 1)
        dynamic_query = query_xy_per_edge is not None
        if not dynamic_query:
            query_local, query_local_valid = crop_anchor_aligned_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=query_indices,
                xy=query_xy,
                window_size=scale.window_size,
            )
            query_local = (
                query_local[:, None, None]
                .expand(-1, candidate_count, view_count, -1, -1)
                .reshape(edge_count, query_local.shape[1], query_local.shape[2])
            )
            query_local_valid = (
                context_valid_mask(query_local_valid, window_size=scale.window_size)[:, None, None]
                .expand(-1, candidate_count, view_count, -1)
                .reshape(edge_count, int(scale.window_size) ** 2)
            )
        else:
            edge_query_xy = torch.as_tensor(
                query_xy_per_edge, dtype=torch.float32, device=grid.device
            )
            if (
                edge_query_xy.shape != (batch, candidate_count, view_count, 2)
                or not bool(torch.isfinite(edge_query_xy).all())
            ):
                raise ValueError("dynamic query coordinates are incompatible with fixed edges")
            edge_query_xy = edge_query_xy.reshape(edge_count, 2)
            edge_query_indices = (
                query_indices[:, None, None]
                .expand(-1, candidate_count, view_count)
                .reshape(-1)
            )
            query_local, query_local_valid = crop_anchor_aligned_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=edge_query_indices,
                xy=edge_query_xy,
                window_size=scale.window_size,
            )
            query_local_valid = context_valid_mask(
                query_local_valid, window_size=scale.window_size
            )
        support_local, support_local_valid = crop_anchor_aligned_grid_tokens(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=safe_support_indices,
            xy=support_xy.reshape(-1, 2),
            window_size=scale.window_size,
        )
        edge_valid = view_valid.reshape(-1)
        support_local_valid = context_valid_mask(
            support_local_valid, window_size=scale.window_size
        ) & edge_valid[:, None]
        # Invalid padded edges never contribute after encoding, but ALIKE has
        # no global tokens.  Keeping one zero-valued local key prevents an
        # all-masked attention row from producing NaN gradients before that
        # edge is discarded.
        safe_support_local_valid = support_local_valid.clone()
        safe_support_local_valid[~edge_valid, 0] = True
        if not dynamic_query:
            query_absolute = crop_anchor_aligned_grid_absolute_coordinates(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=query_indices,
                xy=query_xy,
                window_size=scale.window_size,
            )
            query_absolute = (
                query_absolute[:, None, None]
                .expand(-1, candidate_count, view_count, -1, -1)
                .reshape(edge_count, int(scale.window_size) ** 2, 2)
            )
        else:
            query_absolute = crop_anchor_aligned_grid_absolute_coordinates(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=edge_query_indices,
                xy=edge_query_xy,
                window_size=scale.window_size,
            )
        support_absolute = crop_anchor_aligned_grid_absolute_coordinates(
            image_grids=grid,
            image_sizes=self._image_sizes,
            image_indices=safe_support_indices,
            xy=support_xy.reshape(-1, 2),
            window_size=scale.window_size,
        )
        global_grid = self._global(scale.name)
        if not dynamic_query:
            query_global = global_grid.index_select(0, query_indices)
            query_global = (
                query_global[:, None, None]
                .expand(-1, candidate_count, view_count, -1, -1)
                .reshape(edge_count, query_global.shape[1], query_global.shape[2])
            )
        else:
            query_global = global_grid.index_select(0, edge_query_indices)
        support_global = global_grid.index_select(0, safe_support_indices)
        if self._uses_position_only:
            query_local = torch.zeros_like(query_local)
            support_local = torch.zeros_like(support_local)
            query_global = torch.zeros_like(query_global)
            support_global = torch.zeros_like(support_global)
        return {
            "query_local_descriptors": query_local,
            "query_local_valid": query_local_valid,
            "query_local_relative_coordinates": context_relative_coordinates(
                window_size=scale.window_size, device=grid.device
            ),
            "query_local_absolute_coordinates": query_absolute,
            "query_global_descriptors": query_global,
            "support_local_descriptors": support_local,
            "support_local_valid": safe_support_local_valid,
            "support_local_relative_coordinates": context_relative_coordinates(
                window_size=scale.window_size, device=grid.device
            ),
            "support_local_absolute_coordinates": support_absolute,
            "support_global_descriptors": support_global,
            "global_coordinates": _global_region_coordinates(
                region_size=int(self._global_region_sizes[scale.name]), device=grid.device
            ),
            "edge_valid": edge_valid,
        }

    @staticmethod
    def _raw_layout_cost_volume_statistics(
        *,
        query_tokens: torch.Tensor,
        query_valid: torch.Tensor,
        support_tokens: torch.Tensor,
        support_valid: torch.Tensor,
        local_token_count: int,
        global_token_count: int,
    ) -> torch.Tensor:
        """Retain coarse 2-D phase instead of collapsing a raw cost volume.

        V3's six statistics were intentionally compact, but their global
        max/entropy aggregation discards *where* a distinctive region occurs.
        V4 keeps each query-global peak, support-global peak, and aligned
        global-region correlation.  These values are produced directly from
        frozen descriptors; no learned descriptor projection, attention, or
        candidate mixing can manufacture a local appearance cue.
        """

        if (
            query_tokens.ndim != 3
            or support_tokens.shape != query_tokens.shape
            or query_valid.shape != query_tokens.shape[:2]
            or support_valid.shape != query_tokens.shape[:2]
            or int(local_token_count) <= 0
            or int(global_token_count) < 0
            or query_tokens.shape[1] != int(local_token_count) + int(global_token_count)
        ):
            raise ValueError("raw layout cost-volume inputs are incompatible")
        basic = _BidirectionalAbsoluteContextScaleEncoder._correlation_statistics(
            query_tokens=query_tokens,
            query_valid=query_valid,
            support_tokens=support_tokens,
            support_valid=support_valid,
            local_token_count=int(local_token_count),
        )
        if int(global_token_count) == 0:
            return basic
        query_normalized = F.normalize(query_tokens.float(), dim=-1)
        support_normalized = F.normalize(support_tokens.float(), dim=-1)
        correlation = torch.matmul(query_normalized, support_normalized.transpose(1, 2))
        pair_valid = query_valid[:, :, None] & support_valid[:, None, :]
        masked = torch.where(pair_valid, correlation, torch.full_like(correlation, -torch.inf))
        start = int(local_token_count)
        # Global descriptors are dense pooled image regions, hence every valid
        # candidate/view has finite peak and diagonal values here.  Invalid
        # padded views are explicitly zeroed by the caller before their head.
        query_global_peak = torch.max(masked[:, start:, :], dim=2).values
        support_global_peak = torch.max(masked[:, :, start:], dim=1).values
        aligned_global = torch.diagonal(
            correlation[:, start:, start:], dim1=1, dim2=2
        )
        return torch.cat(
            [basic, query_global_peak, support_global_peak, aligned_global], dim=1
        ).to(dtype=query_tokens.dtype)

    @staticmethod
    def _raw_layout_position_features(
        *,
        query_absolute_coordinates: torch.Tensor,
        support_absolute_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Matched low-capacity absolute-coordinate control for V4.

        These six values give the paired control the same explicit anchor
        position channel as the visual branch.  The only additional V4 input
        is then the raw frozen descriptor cost volume above.
        """

        if (
            query_absolute_coordinates.ndim != 3
            or support_absolute_coordinates.shape != query_absolute_coordinates.shape
            or query_absolute_coordinates.shape[2] != 2
            or query_absolute_coordinates.shape[1] == 0
        ):
            raise ValueError("raw layout position inputs are incompatible")
        center = query_absolute_coordinates.shape[1] // 2
        query_center = query_absolute_coordinates[:, center]
        support_center = support_absolute_coordinates[:, center]
        return torch.cat([query_center, support_center, query_center - support_center], dim=1)

    def _raw_layout_scale_features(
        self,
        *,
        scale: ContextAttentionScale,
        inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build one V5 raw-layout edge representation without an output head."""

        if not self._uses_raw_layout_cost_volume or self.raw_layout_heads is None:
            raise RuntimeError("raw-layout geometry heads are unavailable")
        local_count = int(scale.window_size) ** 2
        query_global = inputs["query_global_descriptors"]
        support_global = inputs["support_global_descriptors"]
        query_descriptors = torch.cat(
            [inputs["query_local_descriptors"], query_global], dim=1
        )
        support_descriptors = torch.cat(
            [inputs["support_local_descriptors"], support_global], dim=1
        )
        global_count = int(query_global.shape[1])
        query_valid = torch.cat(
            [
                inputs["query_local_valid"],
                torch.ones(
                    (len(query_descriptors), global_count),
                    dtype=torch.bool,
                    device=query_descriptors.device,
                ),
            ],
            dim=1,
        )
        support_valid = torch.cat(
            [
                inputs["support_local_valid"],
                torch.ones(
                    (len(support_descriptors), global_count),
                    dtype=torch.bool,
                    device=support_descriptors.device,
                ),
            ],
            dim=1,
        )
        raw_statistics = self._raw_layout_cost_volume_statistics(
            query_tokens=query_descriptors,
            query_valid=query_valid,
            support_tokens=support_descriptors,
            support_valid=support_valid,
            local_token_count=local_count,
            global_token_count=global_count,
        )
        position_features = self._raw_layout_position_features(
            query_absolute_coordinates=inputs["query_local_absolute_coordinates"],
            support_absolute_coordinates=inputs["support_local_absolute_coordinates"],
        )
        raw_features = torch.cat([raw_statistics, position_features], dim=1)
        raw_features = torch.where(
            inputs["edge_valid"][:, None], raw_features, torch.zeros_like(raw_features)
        )
        return raw_features

    def _raw_layout_scale_logits(
        self,
        *,
        scale: ContextAttentionScale,
        inputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one raw-layout geometry head without candidate mixing."""

        raw_features = self._raw_layout_scale_features(scale=scale, inputs=inputs)
        logits = self.raw_layout_heads[scale.name](
            raw_features.to(dtype=self.raw_layout_heads[scale.name].weight.dtype)
        ).squeeze(1)
        return logits, raw_features

    def _pairwise_raw_layout_inputs_at_query_xy(
        self,
        rows: torch.Tensor,
        query_xy_by_candidate: torch.Tensor,
        *,
        scale_names: Sequence[str] | None = None,
    ) -> tuple[
        torch.Tensor,
        list[tuple[ContextAttentionScale, Mapping[str, torch.Tensor]]],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prepare one fixed dynamic query/support layout for either V5 head.

        Keeping coordinate validation and crop extraction in one path is
        important: the geometry and strict-identity diagnostics must differ
        only in their independently trained output head, never in the sampled
        query crop, support-view set, or in-image masking.
        """

        if not self._uses_raw_layout_cost_volume or self.raw_layout_heads is None:
            raise RuntimeError("pairwise dynamic scoring requires a raw-layout family")
        selected_rows = torch.as_tensor(rows, dtype=torch.long, device=self._query_xy.device)
        if selected_rows.ndim != 1 or selected_rows.numel() == 0:
            raise ValueError("pairwise dynamic scoring needs non-empty row indices")
        if torch.any(selected_rows < 0) or torch.any(selected_rows >= self.row_count):
            raise ValueError("pairwise dynamic scoring row index is out of range")
        batch = int(len(selected_rows))
        candidate_count = int(self._support_image_indices.shape[1])
        view_count = int(self._support_image_indices.shape[2])
        coordinates = torch.as_tensor(
            query_xy_by_candidate, dtype=torch.float32, device=self._query_xy.device
        )
        if coordinates.ndim == 3 and coordinates.shape == (batch, candidate_count, 2):
            coordinates = coordinates[:, :, None, :].expand(-1, -1, view_count, -1)
        if (
            coordinates.shape != (batch, candidate_count, view_count, 2)
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise ValueError("pairwise dynamic query coordinates are incompatible")
        query_indices = self._query_image_indices.index_select(0, selected_rows)
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
        available_scales = tuple(scale.name for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES)
        requested_scales = (
            available_scales
            if scale_names is None
            else tuple(str(name) for name in scale_names)
        )
        if (
            not requested_scales
            or len(set(requested_scales)) != len(requested_scales)
            or any(name not in available_scales for name in requested_scales)
        ):
            raise ValueError("pairwise dynamic score scale selection is invalid")
        selected_scales = [
            scale for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES if scale.name in requested_scales
        ]
        view_valid = self._view_valid.index_select(0, selected_rows)
        scale_inputs = [
            (
                scale,
                self._row_scale_inputs(
                    selected_rows, scale, query_xy_per_edge=coordinates
                ),
            )
            for scale in selected_scales
        ]
        return selected_rows, scale_inputs, view_valid, projection_in_image

    def forward_pairwise_raw_layout_at_query_xy(
        self,
        rows: torch.Tensor,
        query_xy_by_candidate: torch.Tensor,
        *,
        scale_names: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score fixed support edges with the uncalibrated V5 geometry head.

        This is deliberately lower level than :meth:`forward_with_details`:
        it does not form a candidate posterior, touch the null head, or expose
        V5's strict identity branch.  It only returns uncalibrated geometry
        raw scores for a frozen candidate/support layout.  A separate
        target-free pose scorer may combine them under a fixed denominator for
        a diagnostic probe; callers must not represent them as a calibrated
        pose likelihood or route them into PnP.
        """

        selected_rows, scale_inputs, view_valid, projection_in_image = (
            self._pairwise_raw_layout_inputs_at_query_xy(
                rows, query_xy_by_candidate, scale_names=scale_names
            )
        )
        batch = int(len(selected_rows))
        candidate_count = int(self._support_image_indices.shape[1])
        view_count = int(self._support_image_indices.shape[2])
        scale_logits = [
            self._raw_layout_scale_logits(scale=scale, inputs=inputs)[0]
            for scale, inputs in scale_inputs
        ]
        per_scale = torch.stack(scale_logits, dim=1).reshape(
            batch, candidate_count, view_count, len(scale_inputs)
        )
        per_scale = torch.where(view_valid[..., None], per_scale, torch.zeros_like(per_scale))
        raw = per_scale.sum(dim=-1)
        return {
            "view_raw_scores": raw,
            "per_scale_view_raw_scores": per_scale,
            "support_view_available": view_valid,
            "query_projection_in_image": projection_in_image,
        }

    def forward_pairwise_identity_raw_layout_at_query_xy(
        self,
        rows: torch.Tensor,
        query_xy_by_candidate: torch.Tensor,
        *,
        scale_names: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return V5 strict-track raw identity residuals at dynamic crops.

        The result is intentionally *not* an identity posterior or an
        independent pose likelihood.  It is an exact-track raw residual for a
        fixed candidate/support layout and may only be audited under the same
        fixed coarse-plus-null denominator as the geometry diagnostic.  This
        method neither invokes the identity null head nor mixes the result
        with the geometry branch, preventing an uncalibrated classification
        posterior from being silently used as pose evidence.
        """

        if not self._uses_dual_identity_head or self.identity_raw_layout_heads is None:
            raise RuntimeError("pairwise strict identity scoring requires the V5 dual-head family")
        selected_rows, scale_inputs, view_valid, projection_in_image = (
            self._pairwise_raw_layout_inputs_at_query_xy(
                rows, query_xy_by_candidate, scale_names=scale_names
            )
        )
        batch = int(len(selected_rows))
        candidate_count = int(self._support_image_indices.shape[1])
        view_count = int(self._support_image_indices.shape[2])
        identity_scale_logits: list[torch.Tensor] = []
        for scale, inputs in scale_inputs:
            raw_features = self._raw_layout_scale_features(
                scale=scale, inputs=inputs
            )
            if scale.name in self.identity_raw_layout_heads:
                head = self.identity_raw_layout_heads[scale.name]
                identity_scale_logits.append(
                    head(raw_features.to(dtype=head.weight.dtype)).squeeze(1)
                )
            else:
                # ALIKE remains spatial-only; retain a shape-stable zero
                # family so score audits cannot mistake omission for evidence.
                identity_scale_logits.append(raw_features.new_zeros((len(raw_features),)))
        per_scale = torch.stack(identity_scale_logits, dim=1).reshape(
            batch, candidate_count, view_count, len(scale_inputs)
        )
        per_scale = torch.where(view_valid[..., None], per_scale, torch.zeros_like(per_scale))
        return {
            "identity_view_raw_scores": per_scale.sum(dim=-1),
            "identity_per_scale_view_raw_scores": per_scale,
            "support_view_available": view_valid,
            "query_projection_in_image": projection_in_image,
        }

    def forward_with_details(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        selected_rows = torch.as_tensor(rows, dtype=torch.long, device=self._query_xy.device)
        if selected_rows.ndim != 1 or selected_rows.numel() == 0:
            raise ValueError("bidirectional absolute-context forward needs non-empty row indices")
        if torch.any(selected_rows < 0) or torch.any(selected_rows >= self.row_count):
            raise ValueError("bidirectional absolute-context row index is out of range")
        encoded_features: list[torch.Tensor] = []
        scale_logits: list[torch.Tensor] = []
        identity_scale_logits: list[torch.Tensor] = []
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            inputs = self._row_scale_inputs(selected_rows, scale)
            if self._uses_raw_layout_cost_volume:
                scale_logit, raw_features = self._raw_layout_scale_logits(
                    scale=scale, inputs=inputs
                )
                scale_logits.append(scale_logit)
                if self._uses_dual_identity_head:
                    if self.identity_raw_layout_heads is None:
                        raise RuntimeError("dual identity raw-layout heads are missing")
                    if scale.name in self.identity_raw_layout_heads:
                        identity_scale_logits.append(
                            self.identity_raw_layout_heads[scale.name](
                                raw_features.to(
                                    dtype=self.identity_raw_layout_heads[
                                        scale.name
                                    ].weight.dtype
                                )
                            ).squeeze(1)
                        )
                    else:
                        # Preserve the immutable three-scale audit tensor while
                        # making the ALIKE identity contribution exactly zero.
                        identity_scale_logits.append(torch.zeros_like(scale_logits[-1]))
                continue
            encoder_inputs = {
                name: value for name, value in inputs.items() if name != "edge_valid"
            }
            raw_statistics: torch.Tensor | None = None
            if self._uses_raw_descriptor_evidence:
                encoded, raw_statistics = self.encoders[scale.name].forward_with_statistics(
                    **encoder_inputs
                )
                if raw_statistics is None or self.raw_scale_heads is None:
                    raise RuntimeError("raw absolute-context encoder omitted raw statistics")
            else:
                encoded = self.encoders[scale.name](**encoder_inputs)
            encoded = torch.where(
                inputs["edge_valid"][:, None], encoded, torch.zeros_like(encoded)
            )
            encoded_features.append(encoded)
            scale_logit = self.scale_heads[scale.name](encoded).squeeze(1)
            if raw_statistics is not None:
                raw_statistics = torch.where(
                    inputs["edge_valid"][:, None], raw_statistics, torch.zeros_like(raw_statistics)
                )
                scale_logit = scale_logit + self.raw_scale_heads[scale.name](
                    raw_statistics.to(dtype=encoded.dtype)
                ).squeeze(1)
            scale_logits.append(scale_logit)
        if self._uses_raw_layout_cost_volume:
            raw = torch.stack(scale_logits, dim=1).sum(dim=1)
            identity_raw = (
                torch.stack(identity_scale_logits, dim=1).sum(dim=1)
                if self._uses_dual_identity_head
                else None
            )
            if self._uses_dual_identity_head and identity_raw is None:
                raise RuntimeError("dual identity raw-layout logits are missing")
        else:
            if self.fusion_head is None:
                raise RuntimeError("attention context fusion head is missing")
            feature_tensor = torch.cat(encoded_features, dim=1)
            raw = self.fusion_head(feature_tensor).squeeze(1)
            raw = raw + torch.stack(scale_logits, dim=1).sum(dim=1)
            identity_raw = None
        batch = len(selected_rows)
        candidate_count = int(self._support_image_indices.shape[1])
        view_count = int(self._support_image_indices.shape[2])
        raw_view_logits = raw.reshape(batch, candidate_count, view_count)
        per_scale_view_logits = torch.stack(scale_logits, dim=1).reshape(
            batch, candidate_count, view_count, len(BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES)
        )
        view_valid = self._view_valid.index_select(0, selected_rows)
        view_logits = torch.where(view_valid, raw_view_logits, torch.zeros_like(raw_view_logits))
        per_scale_view_logits = torch.where(
            view_valid[..., None], per_scale_view_logits, torch.zeros_like(per_scale_view_logits)
        )
        candidate_residual = masked_view_log_mean(view_logits, view_valid)
        base_candidate = self._base_candidate_probabilities.index_select(0, selected_rows)
        base_null = self._base_null_probabilities.index_select(0, selected_rows)
        candidate_valid = torch.any(view_valid, dim=2) & (base_candidate > 0.0)
        # The null branch consumes a permutation-invariant summary of current
        # candidate visual evidence, but it is a separate likelihood rather
        # than a second, numerically fragile training path into identity
        # scores.  Detaching here prevents zero upstream gradients through
        # max/log-mean statistics from contaminating the candidate matcher.
        null_log_likelihood_ratio = self.null_head(
            _candidate_set_visual_statistics(candidate_residual.detach(), candidate_valid)
        ).squeeze(1)
        candidate_logits = torch.where(
            base_candidate > 0.0,
            torch.log(base_candidate.clamp_min(1e-12)) + candidate_residual,
            torch.full_like(base_candidate, -1e9),
        )
        null_logits = torch.log(base_null.clamp_min(1e-12)) + null_log_likelihood_ratio
        logits = torch.cat([candidate_logits, null_logits[:, None]], dim=1)
        probability = torch.softmax(logits, dim=1)
        masked_view_logits = torch.where(
            view_valid, view_logits, torch.full_like(view_logits, -torch.inf)
        )
        view_log_probabilities = masked_view_logits - torch.logsumexp(
            masked_view_logits, dim=2, keepdim=True
        )
        view_log_probabilities = torch.where(
            view_valid, view_log_probabilities, torch.zeros_like(view_log_probabilities)
        )
        result = {
            "candidate_probabilities": probability[:, :-1],
            "null_probabilities": probability[:, -1],
            "view_logits": view_logits,
            "view_log_probabilities": view_log_probabilities,
            "per_scale_view_logits": per_scale_view_logits,
            "candidate_log_likelihood_ratios": candidate_residual,
            "null_log_likelihood_ratio": null_log_likelihood_ratio,
            "logits": logits,
        }

        if self._uses_dual_identity_head:
            if identity_raw is None or self.identity_null_head is None:
                raise RuntimeError("dual identity likelihood is incomplete")
            raw_identity_view_logits = identity_raw.reshape(batch, candidate_count, view_count)
            identity_per_scale_view_logits = torch.stack(identity_scale_logits, dim=1).reshape(
                batch,
                candidate_count,
                view_count,
                len(BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES),
            )
            identity_view_logits = torch.where(
                view_valid,
                raw_identity_view_logits,
                torch.zeros_like(raw_identity_view_logits),
            )
            identity_per_scale_view_logits = torch.where(
                view_valid[..., None],
                identity_per_scale_view_logits,
                torch.zeros_like(identity_per_scale_view_logits),
            )
            identity_candidate_residual = masked_view_log_mean(
                identity_view_logits, view_valid
            )
            identity_null_log_likelihood_ratio = self.identity_null_head(
                _candidate_set_visual_statistics(
                    identity_candidate_residual.detach(), candidate_valid
                )
            ).squeeze(1)
            identity_candidate_logits = torch.where(
                base_candidate > 0.0,
                torch.log(base_candidate.clamp_min(1e-12)) + identity_candidate_residual,
                torch.full_like(base_candidate, -1e9),
            )
            identity_null_logits = (
                torch.log(base_null.clamp_min(1e-12)) + identity_null_log_likelihood_ratio
            )
            identity_logits = torch.cat(
                [identity_candidate_logits, identity_null_logits[:, None]], dim=1
            )
            identity_probability = torch.softmax(identity_logits, dim=1)
            masked_identity_view_logits = torch.where(
                view_valid,
                identity_view_logits,
                torch.full_like(identity_view_logits, -torch.inf),
            )
            identity_view_log_probabilities = masked_identity_view_logits - torch.logsumexp(
                masked_identity_view_logits, dim=2, keepdim=True
            )
            identity_view_log_probabilities = torch.where(
                view_valid,
                identity_view_log_probabilities,
                torch.zeros_like(identity_view_log_probabilities),
            )
            result.update(
                {
                    "identity_candidate_probabilities": identity_probability[:, :-1],
                    "identity_null_probabilities": identity_probability[:, -1],
                    "identity_view_logits": identity_view_logits,
                    "identity_view_log_probabilities": identity_view_log_probabilities,
                    "identity_per_scale_view_logits": identity_per_scale_view_logits,
                    "identity_candidate_log_likelihood_ratios": identity_candidate_residual,
                    "identity_null_log_likelihood_ratio": identity_null_log_likelihood_ratio,
                    "identity_logits": identity_logits,
                }
            )
        return result

    def forward(
        self, rows: torch.Tensor, *, return_details: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        details = self.forward_with_details(rows)
        if bool(return_details):
            return details
        return (
            details["candidate_probabilities"],
            details["null_probabilities"],
            details["view_logits"],
            details["logits"],
        )
