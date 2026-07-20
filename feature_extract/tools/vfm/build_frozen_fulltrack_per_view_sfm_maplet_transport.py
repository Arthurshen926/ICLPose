"""Attach center-excluded SfM-maplet transport evidence to raw CSR edges.

For every already frozen ``(query point, candidate track, support
observation)`` edge, this exporter compares query context against a sparse
same-image support maplet.  The maplet contains only neighboring real SfM
observations of the support image; the candidate landmark centre is excluded
from both sides.  This makes the artifact a structural-context probe rather
than another central-descriptor or local-patch score.

Candidates, candidate mass, support observation order, and null mass are
copied bit-for-bit from the current raw full-track CSR source.  No pose,
target, image retrieval, rendering, candidate reselection, or early
support-view aggregation is permitted.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

from feature_extract.tools.vfm.build_frozen_fulltrack_global_context import (
    _array_sha256_short,
    _geometry_image_indices,
    _load_current_raw_per_view_source,
)
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    _cache_image_indices,
    _parse_devices,
)
from feature_extract.tools.vfm.build_multisource_landmark_region_prototype_candidate_probe_features import (
    _SourceArrays,
    _load_sources,
    _source_metadata,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_APPEARANCE_FORMAT,
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_sfm_maplet_transport import (
    SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
    SFM_MAPLET_TRANSPORT_PROFILES,
    batched_sfm_maplet_quadrant_transport_features,
    pool_center_excluded_query_quadrants,
    pool_sparse_support_quadrants,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)


ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_sfm_maplet_transport_v1"
_MAX_NEIGHBORS_PER_QUADRANT = 4
_MINIMUM_SUPPORT_NEIGHBORS_PER_QUADRANT = 1
_MINIMUM_QUERY_QUADRANT_FRACTION = 0.75
_TRANSPORT_TEMPERATURE = 0.07
# This fixed candidate buffer is deliberately much wider than the final
# 4-per-quadrant maplet while avoiding a dense all-observation pair matrix for
# a 4k-point support image.  The resulting neighbor availability is audited
# before any S1 fit is permitted.
_NEIGHBOR_QUERY_CANDIDATE_LIMIT = 128


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-per-view-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument(
        "--alike-spatial-context-cache",
        required=True,
        help=(
            "required only to validate the shared real-image cache contract; "
            "ALIKE is not a learned feature source in this structural probe"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _edge_offsets(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("candidate support observation counts are invalid")
    return np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(values.reshape(-1), dtype=np.int64),
        )
    )


def _source_by_name(sources: Sequence[_SourceArrays]) -> dict[str, _SourceArrays]:
    output = {str(source.profile.name): source for source in sources}
    required = {str(profile.source_name) for profile in SFM_MAPLET_TRANSPORT_PROFILES}
    if not required <= set(output):
        raise ValueError("SfM-maplet transport source caches are incomplete")
    for profile in SFM_MAPLET_TRANSPORT_PROFILES:
        source = output[str(profile.source_name)]
        if int(profile.window_size) > int(source.profile.grid_size):
            raise ValueError(
                f"{profile.name}: context window exceeds {source.profile.name} grid"
            )
    return output


def _source_image_rows(
    *, geometry: SupportObservationGeometryIndex, source: _SourceArrays
) -> np.ndarray:
    image_rows = _geometry_image_indices(geometry)
    geometry_ids = np.asarray(geometry.image_ids).astype(str)[image_rows]
    return _cache_image_indices(
        cache_image_ids=np.asarray(source.image_ids).astype(str),
        image_ids=geometry_ids,
        context="SfM-maplet support observation",
    )


def _maplet_neighbor_array(
    *,
    geometry: SupportObservationGeometryIndex,
    source: _SourceArrays,
    radius_cells: int,
    max_neighbors_per_quadrant: int,
) -> np.ndarray:
    """Build a target-free same-image, center-excluded neighbor maplet table."""

    radius = int(radius_cells)
    maximum = int(max_neighbors_per_quadrant)
    rows_by_image = _geometry_image_indices(geometry)
    coordinates = np.asarray(geometry.xy, dtype=np.float32)
    errors = np.asarray(geometry.reprojection_errors, dtype=np.float32)
    tracks = np.asarray(geometry.track_ids, dtype=np.int64)
    if (
        radius < 2
        or maximum <= 0
        or coordinates.shape != (len(rows_by_image), 2)
        or errors.shape != (len(rows_by_image),)
        or tracks.shape != (len(rows_by_image),)
        or np.any(~np.isfinite(coordinates))
        or np.any(~np.isfinite(errors))
        or np.any(errors < 0.0)
    ):
        raise ValueError("SfM-maplet neighbor topology inputs are invalid")
    source_rows = _source_image_rows(geometry=geometry, source=source)
    result = np.full(
        (len(coordinates), 4, maximum), -1, dtype=np.int64
    )
    # A dense facade can put many points in one quadrant.  Query a much wider
    # candidate buffer than the final retained 4 x 4 maplet slots, then select
    # every anchor/quadrant in one vectorized stable ordering below.
    requested_neighbors = max(
        int(_NEIGHBOR_QUERY_CANDIDATE_LIMIT), 8 * maximum * 4
    )
    for image_index in range(len(geometry.image_ids)):
        rows = np.flatnonzero(rows_by_image == int(image_index)).astype(np.int64)
        if len(rows) <= 1:
            continue
        cache_rows = np.unique(source_rows[rows])
        if len(cache_rows) != 1:
            raise RuntimeError("same support image maps to multiple source rows")
        width, height = (
            int(value)
            for value in np.asarray(source.image_sizes[int(cache_rows[0])], dtype=np.int64)
        )
        if width <= 1 or height <= 1:
            raise ValueError("SfM-maplet source image has invalid dimensions")
        grid = np.empty((len(rows), 2), dtype=np.float64)
        grid[:, 0] = coordinates[rows, 0] * float(source.profile.grid_size - 1) / float(width - 1)
        grid[:, 1] = coordinates[rows, 1] * float(source.profile.grid_size - 1) / float(height - 1)
        tree = cKDTree(grid)
        count = min(len(rows), int(requested_neighbors))
        distances, local_neighbors = tree.query(
            grid,
            k=count,
            distance_upper_bound=float(radius) * np.sqrt(2.0) + 1e-6,
            workers=-1,
        )
        distances = np.asarray(distances, dtype=np.float64)
        local_neighbors = np.asarray(local_neighbors, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            local_neighbors = local_neighbors[:, None]
        local_valid = (local_neighbors >= 0) & (local_neighbors < len(rows))
        safe_local_neighbors = np.clip(local_neighbors, 0, len(rows) - 1)
        neighbor_rows = rows[safe_local_neighbors]
        delta = grid[safe_local_neighbors] - grid[:, None, :]
        local_valid &= np.isfinite(distances)
        local_valid &= safe_local_neighbors != np.arange(len(rows))[:, None]
        local_valid &= np.max(np.abs(delta), axis=2) <= float(radius) + 1e-6
        # Exclude the landmark centre and its immediate local appearance.
        # Query quadrants use the same one-cell horizontal/vertical gap.
        local_valid &= np.abs(delta[:, :, 0]) >= 1.0
        local_valid &= np.abs(delta[:, :, 1]) >= 1.0
        quadrants = (2 * (delta[:, :, 1] > 0.0).astype(np.int64)) + (
            delta[:, :, 0] > 0.0
        ).astype(np.int64)
        neighbor_errors = errors[neighbor_rows]
        neighbor_tracks = tracks[neighbor_rows]
        for quadrant in range(4):
            valid = local_valid & (quadrants == quadrant)
            ranked_distance = np.where(valid, distances, np.inf)
            # Selecting four entries is much cheaper than sorting every
            # candidate buffer.  A stable lexicographic order is still used
            # on those retained slots for reproducible topology hashes.
            partial = np.argpartition(ranked_distance, kth=maximum - 1, axis=1)[
                :, :maximum
            ]
            selected = np.take_along_axis(neighbor_rows, partial, axis=1)
            selected_distance = np.take_along_axis(ranked_distance, partial, axis=1)
            selected_errors = np.take_along_axis(neighbor_errors, partial, axis=1)
            selected_tracks = np.take_along_axis(neighbor_tracks, partial, axis=1)
            order = np.lexsort(
                (selected, selected_tracks, selected_errors, selected_distance), axis=1
            )
            selected = np.take_along_axis(selected, order, axis=1)
            selected_distance = np.take_along_axis(selected_distance, order, axis=1)
            result[rows, quadrant, :] = np.where(
                np.isfinite(selected_distance), selected, -1
            )
    return result


def _build_maplet_neighbor_topologies(
    *, geometry: SupportObservationGeometryIndex, sources: Sequence[_SourceArrays]
) -> dict[str, np.ndarray]:
    by_source = _source_by_name(sources)
    shared: dict[tuple[int, int], np.ndarray] = {}
    output: dict[str, np.ndarray] = {}
    for profile in SFM_MAPLET_TRANSPORT_PROFILES:
        source = by_source[profile.source_name]
        key = (int(source.profile.grid_size), int(profile.radius_cells))
        if key not in shared:
            shared[key] = _maplet_neighbor_array(
                geometry=geometry,
                source=source,
                radius_cells=int(profile.radius_cells),
                max_neighbors_per_quadrant=_MAX_NEIGHBORS_PER_QUADRANT,
            )
        output[profile.name] = shared[key]
    return output


def _query_quadrants_torch(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: np.ndarray,
    xy: np.ndarray,
    radius_cells: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample all frozen query points once for one source/radius profile."""

    grids = torch.as_tensor(image_grids)
    sizes = torch.as_tensor(image_sizes, dtype=torch.float32, device=grids.device)
    indices = torch.as_tensor(image_indices, dtype=torch.long, device=grids.device)
    coordinates = torch.as_tensor(xy, dtype=torch.float32, device=grids.device)
    radius = int(radius_cells)
    if (
        grids.ndim != 4
        or grids.shape[1] != grids.shape[2]
        or sizes.shape != (grids.shape[0], 2)
        or indices.ndim != 1
        or coordinates.shape != (len(indices), 2)
        or radius < 2
    ):
        raise ValueError("query quadrant sampler inputs are invalid")
    grid_size = int(grids.shape[1])
    selected_sizes = sizes.index_select(0, indices)
    center_x = coordinates[:, 0] * float(grid_size - 1) / selected_sizes[:, 0].sub(1.0)
    center_y = coordinates[:, 1] * float(grid_size - 1) / selected_sizes[:, 1].sub(1.0)
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=grids.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    sample_x = center_x[:, None, None] + offset_x[None]
    sample_y = center_y[:, None, None] + offset_y[None]
    valid = (
        (sample_x >= 0.0)
        & (sample_x <= float(grid_size - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(grid_size - 1))
    )
    normalized = torch.stack(
        (
            2.0 * sample_x / float(grid_size - 1) - 1.0,
            2.0 * sample_y / float(grid_size - 1) - 1.0,
        ),
        dim=-1,
    )
    selected = grids.index_select(0, indices).permute(0, 3, 1, 2).contiguous()
    patches = F.grid_sample(
        selected,
        normalized,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    patches = F.normalize(patches, p=2, dim=1, eps=1e-8)
    return pool_center_excluded_query_quadrants(
        patches=patches,
        valid=valid,
        minimum_fraction=_MINIMUM_QUERY_QUADRANT_FRACTION,
    )


def _support_quadrants_torch(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    neighbor_geometry_rows: np.ndarray,
    geometry_xy: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample and pool fixed real support-maplet neighbors per CSR edge."""

    grids = torch.as_tensor(image_grids)
    sizes = torch.as_tensor(image_sizes, dtype=torch.float32, device=grids.device)
    indices = torch.as_tensor(image_indices, dtype=torch.long, device=grids.device)
    neighbors = torch.as_tensor(
        neighbor_geometry_rows, dtype=torch.long, device=grids.device
    )
    coordinates = torch.as_tensor(geometry_xy, dtype=torch.float32, device=grids.device)
    if (
        grids.ndim != 4
        or grids.shape[1] != grids.shape[2]
        or sizes.shape != (grids.shape[0], 2)
        or indices.ndim != 1
        or neighbors.ndim != 3
        or neighbors.shape[0] != len(indices)
        or neighbors.shape[1] != 4
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or torch.any(neighbors < -1)
    ):
        raise ValueError("support maplet quadrant sampler inputs are invalid")
    safe_neighbors = neighbors.clamp_min(0)
    present = neighbors >= 0
    neighbor_xy = coordinates[safe_neighbors]
    grid_size = int(grids.shape[1])
    selected_sizes = sizes.index_select(0, indices)
    normalized = torch.stack(
        (
            2.0 * neighbor_xy[..., 0] / selected_sizes[:, None, None, 0].sub(1.0) - 1.0,
            2.0 * neighbor_xy[..., 1] / selected_sizes[:, None, None, 1].sub(1.0) - 1.0,
        ),
        dim=-1,
    )
    on_image = (
        (normalized[..., 0] >= -1.0)
        & (normalized[..., 0] <= 1.0)
        & (normalized[..., 1] >= -1.0)
        & (normalized[..., 1] <= 1.0)
    )
    selected = grids.index_select(0, indices).permute(0, 3, 1, 2).contiguous()
    sample_grid = normalized.reshape(len(indices), -1, 1, 2)
    sampled = F.grid_sample(
        selected,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
        len(indices), neighbors.shape[1], neighbors.shape[2], grids.shape[3]
    )
    sampled = F.normalize(sampled, p=2, dim=3, eps=1e-8)
    return pool_sparse_support_quadrants(
        descriptors=sampled,
        present=present & on_image,
        minimum_neighbors=_MINIMUM_SUPPORT_NEIGHBORS_PER_QUADRANT,
    )


def _compute_partition(
    *,
    device_name: str,
    sources: tuple[_SourceArrays, ...],
    query_cache_rows: np.ndarray,
    query_xy: np.ndarray,
    edge_rows: np.ndarray,
    edge_support_cache_rows: np.ndarray,
    edge_geometry_rows: np.ndarray,
    geometry_xy: np.ndarray,
    neighbor_topologies: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Materialize one immutable edge range on exactly one device."""

    started = time.monotonic()
    device = torch.device(device_name)
    source_by_name = _source_by_name(sources)
    profile_count = len(SFM_MAPLET_TRANSPORT_PROFILES)
    feature_count = len(FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES)
    with torch.no_grad():
        tensors = {
            name: (
                torch.from_numpy(source.grids).to(device=device, dtype=torch.float32),
                torch.from_numpy(source.image_sizes).to(device=device, dtype=torch.float32),
            )
            for name, source in source_by_name.items()
        }
        query_context = {
            profile.name: _query_quadrants_torch(
                image_grids=tensors[profile.source_name][0],
                image_sizes=tensors[profile.source_name][1],
                image_indices=query_cache_rows,
                xy=query_xy,
                radius_cells=int(profile.radius_cells),
            )
            for profile in SFM_MAPLET_TRANSPORT_PROFILES
        }
        output = np.full((int(end) - int(begin), feature_count), np.nan, dtype=np.float16)
        profile_usable = np.zeros((int(end) - int(begin), profile_count), dtype=bool)
        support_counts = np.zeros(
            (int(end) - int(begin), profile_count, 4), dtype=np.uint8
        )
        feature_offset = 0
        profile_offsets: dict[str, slice] = {}
        for profile in SFM_MAPLET_TRANSPORT_PROFILES:
            width = len(SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name])
            profile_offsets[profile.name] = slice(feature_offset, feature_offset + width)
            feature_offset += width
        if feature_offset != feature_count:
            raise RuntimeError("SfM-maplet profile feature offsets drifted")
        for offset in range(int(begin), int(end), int(batch_size)):
            stop = min(offset + int(batch_size), int(end))
            point_rows = torch.from_numpy(edge_rows[offset:stop]).to(device=device)
            support_rows = torch.from_numpy(edge_support_cache_rows[offset:stop]).to(
                device=device
            )
            geometry_rows = np.asarray(edge_geometry_rows[offset:stop], dtype=np.int64)
            local_output = output[offset - int(begin) : stop - int(begin)]
            local_usable = profile_usable[offset - int(begin) : stop - int(begin)]
            local_counts = support_counts[offset - int(begin) : stop - int(begin)]
            for profile_index, profile in enumerate(SFM_MAPLET_TRANSPORT_PROFILES):
                query_values, query_valid, _query_counts = query_context[profile.name]
                support_values, support_valid, current_counts = _support_quadrants_torch(
                    image_grids=tensors[profile.source_name][0],
                    image_sizes=tensors[profile.source_name][1],
                    image_indices=support_rows,
                    neighbor_geometry_rows=np.asarray(
                        neighbor_topologies[profile.name][geometry_rows], dtype=np.int64
                    ),
                    geometry_xy=geometry_xy,
                )
                query_values = query_values.index_select(0, point_rows)
                query_valid = query_valid.index_select(0, point_rows)
                values = batched_sfm_maplet_quadrant_transport_features(
                    query_quadrants=query_values,
                    query_valid=query_valid,
                    support_quadrants=support_values,
                    support_valid=support_valid,
                    temperature=_TRANSPORT_TEMPERATURE,
                )
                usable = torch.all(query_valid, dim=1) & torch.all(support_valid, dim=1)
                usable_np = usable.cpu().numpy().astype(bool, copy=False)
                profile_slice = profile_offsets[profile.name]
                values_np = values.cpu().numpy().astype(np.float16, copy=False)
                local_output[usable_np, profile_slice] = values_np[usable_np]
                local_usable[:, profile_index] = usable_np
                local_counts[:, profile_index] = current_counts.cpu().numpy().astype(
                    np.uint8, copy=False
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if np.any(np.isinf(output)):
        raise RuntimeError("SfM-maplet transport exporter emitted infinity")
    return output, profile_usable, support_counts, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
        "query_quadrant_coverage": {
            profile.name: float(
                torch.mean(torch.all(query_context[profile.name][1], dim=1).to(torch.float32))
                .cpu()
                .item()
            )
            for profile in SFM_MAPLET_TRANSPORT_PROFILES
        },
    }


def _context_source_contract(sources: Sequence[_SourceArrays]) -> dict[str, Any]:
    by_name = _source_by_name(sources)
    return {
        "mode": "center_excluded_sfm_maplet_quadrant_transport_v1",
        "support_coordinate_source": "sfm_observation_xy",
        "support_neighbor_scope": "same_real_support_image_only",
        "view_aggregation": "none_before_learned_logsumexp_mixture_v1",
        "center_descriptor_in_features": False,
        "availability_is_a_learned_feature": False,
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "grid_size": int(by_name[profile.source_name].profile.grid_size),
                "radius_cells": int(profile.radius_cells),
                "window_size": int(profile.window_size),
                "feature_names": list(
                    SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]
                ),
            }
            for profile in SFM_MAPLET_TRANSPORT_PROFILES
        ],
        "sources": [_source_metadata(by_name[profile.source_name]) for profile in SFM_MAPLET_TRANSPORT_PROFILES],
    }


def build_frozen_fulltrack_per_view_sfm_maplet_transport(
    *,
    source_per_view_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    """Copy one raw CSR shard and attach frozen maplet transport only."""

    if int(batch_size) <= 0:
        raise ValueError("SfM-maplet transport batch size must be positive")
    selected_devices = tuple(str(value) for value in devices)
    if not selected_devices or len(set(selected_devices)) != len(selected_devices):
        raise ValueError("SfM-maplet transport devices must be unique and non-empty")
    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    final_path = Path(radio_final_context_cache)
    intermediate_path = Path(radio_intermediate_context_cache)
    alike_path = Path(alike_spatial_context_cache)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite SfM-maplet transport outputs")
    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("SfM-maplet transport requires real SfM observation xy")
    source, source_metadata, edges, source_maplet_counts, source_lineage = (
        _load_current_raw_per_view_source(
            source_path=source_path,
            support_geometry_index=geometry_path,
            geometry=geometry,
        )
    )
    source_context_hashes = source_metadata.get("context_cache_sha256")
    if (
        not isinstance(source_context_hashes, dict)
        or str(source_context_hashes.get("radio_final", ""))
        != file_sha256_short(final_path)
    ):
        raise ValueError("raw per-view source and RADIO-final spatial context differ")
    sources = _load_sources(
        radio_final_context_cache=final_path,
        radio_intermediate_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
    )
    source_by_name = _source_by_name(sources)
    cache_ids = np.asarray(source_by_name["radio_final"].image_ids).astype(str)
    query_cache_rows = _cache_image_indices(
        cache_image_ids=cache_ids,
        image_ids=np.asarray(source["verification_query_ids"]).astype(str),
        context="frozen query",
    )
    image_rows = _geometry_image_indices(geometry)
    support_ids = np.asarray(geometry.image_ids).astype(str)[
        image_rows[np.asarray(edges.geometry_rows, dtype=np.int64)]
    ]
    edge_support_cache_rows = _cache_image_indices(
        cache_image_ids=cache_ids,
        image_ids=support_ids,
        context="real full-track support observation",
    )
    candidate_shape = tuple(edges.candidate_shape)
    edge_rows = np.asarray(edges.edge_candidate_indices, dtype=np.int64) // int(
        candidate_shape[1]
    )
    edge_count = int(edges.edge_count)
    if edge_count <= 0 or edge_count != len(edge_rows):
        raise ValueError("SfM-maplet transport has no immutable CSR edges")
    neighbor_topologies = _build_maplet_neighbor_topologies(
        geometry=geometry, sources=sources
    )
    partitions = [
        (
            edge_count * index // len(selected_devices),
            edge_count * (index + 1) // len(selected_devices),
        )
        for index in range(len(selected_devices))
    ]
    if any(end <= begin for begin, end in partitions):
        raise ValueError("more devices than immutable full-track CSR edges")
    with ThreadPoolExecutor(max_workers=len(selected_devices)) as executor:
        futures = [
            executor.submit(
                _compute_partition,
                device_name=device,
                sources=sources,
                query_cache_rows=query_cache_rows,
                query_xy=np.asarray(source["verification_xy"], dtype=np.float32),
                edge_rows=edge_rows,
                edge_support_cache_rows=edge_support_cache_rows,
                edge_geometry_rows=np.asarray(edges.geometry_rows, dtype=np.int64),
                geometry_xy=np.asarray(geometry.xy, dtype=np.float32),
                neighbor_topologies=neighbor_topologies,
                begin=begin,
                end=end,
                batch_size=int(batch_size),
            )
            for device, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    scores = np.concatenate([value[0] for value in computed], axis=0)
    profile_usable = np.concatenate([value[1] for value in computed], axis=0)
    support_counts = np.concatenate([value[2] for value in computed], axis=0)
    expected_width = len(FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES)
    if (
        scores.shape != (edge_count, expected_width)
        or profile_usable.shape != (edge_count, len(SFM_MAPLET_TRANSPORT_PROFILES))
        or support_counts.shape != (
            edge_count,
            len(SFM_MAPLET_TRANSPORT_PROFILES),
            4,
        )
    ):
        raise RuntimeError("SfM-maplet transport output shape drifted")
    expected_valid = np.concatenate(
        [
            np.repeat(
                profile_usable[:, index : index + 1],
                len(SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]),
                axis=1,
            )
            for index, profile in enumerate(SFM_MAPLET_TRANSPORT_PROFILES)
        ],
        axis=1,
    )
    if (
        expected_valid.shape != scores.shape
        or np.any(np.isinf(scores))
        or np.any(~np.isfinite(scores[expected_valid]))
        or np.any(np.isfinite(scores[~expected_valid]))
    ):
        raise RuntimeError("SfM-maplet transport missingness semantics are invalid")
    offsets = _edge_offsets(np.asarray(edges.candidate_observation_counts, dtype=np.int64))
    if (
        offsets[-1] != edge_count
        or not np.array_equal(
            np.repeat(
                np.arange(edges.candidate_shape[0] * edges.candidate_shape[1]),
                np.diff(offsets),
            ),
            np.asarray(edges.edge_candidate_indices, dtype=np.int64),
        )
    ):
        raise RuntimeError("SfM-maplet transport changed CSR edge order")
    candidate = np.asarray(source["candidate_probabilities"], dtype=np.float32)
    if np.any((candidate > 0.0) & (edges.candidate_observation_counts <= 0)):
        raise RuntimeError("positive frozen candidate lost all real support observations")
    query_id = str(source["verification_query_ids"][0])
    source_contract = _context_source_contract(sources)
    topology_hashes = {
        profile.name: _array_sha256_short(neighbor_topologies[profile.name])
        for profile in SFM_MAPLET_TRANSPORT_PROFILES
    }
    metadata: dict[str, Any] = {
        "format": FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_APPEARANCE_FORMAT,
        "version": ARTIFACT_VERSION,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "query_count": 1,
        "source_frozen_appearance_artifact": source_lineage[
            "source_frozen_appearance_artifact"
        ],
        "source_frozen_appearance_artifact_sha256": source_lineage[
            "source_frozen_appearance_artifact_sha256"
        ],
        "source_fulltrack_per_view_artifact": source_lineage[
            "source_fulltrack_per_view_artifact"
        ],
        "source_fulltrack_per_view_artifact_sha256": source_lineage[
            "source_fulltrack_per_view_artifact_sha256"
        ],
        "source_edge_candidate_offsets_sha256": source_lineage[
            "source_edge_candidate_offsets_sha256"
        ],
        "source_edge_geometry_rows_sha256": source_lineage[
            "source_edge_geometry_rows_sha256"
        ],
        "source_edge_feature_semantics": source_lineage[
            "source_edge_feature_semantics"
        ],
        "source_fulltrack_edge_contract": source_lineage["source_kind"],
        "source_candidate_tracks_sha256": _array_sha256_short(
            source["candidate_track_ids"]
        ),
        "source_candidate_probabilities_sha256": _array_sha256_short(
            source["candidate_probabilities"]
        ),
        "source_null_probabilities_sha256": _array_sha256_short(
            source["null_probabilities"]
        ),
        "source_verification_rows_sha256": _array_sha256_short(
            source["verification_source_row_indices"]
        ),
        "support_geometry_index": str(geometry_path),
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "context_cache_sha256": {
            "radio_final": file_sha256_short(final_path),
            "radio_intermediate": file_sha256_short(intermediate_path),
            "alike": file_sha256_short(alike_path),
        },
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_preserve_source_csr_order_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": (
            FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": source_contract["profiles"],
        "sfm_maplet_transport_contract": {
            **source_contract,
            "maximum_neighbors_per_quadrant": _MAX_NEIGHBORS_PER_QUADRANT,
            "minimum_support_neighbors_per_quadrant": (
                _MINIMUM_SUPPORT_NEIGHBORS_PER_QUADRANT
            ),
            "minimum_query_quadrant_fraction": _MINIMUM_QUERY_QUADRANT_FRACTION,
            "transport_temperature": _TRANSPORT_TEMPERATURE,
            "neighbor_query_candidate_limit": _NEIGHBOR_QUERY_CANDIDATE_LIMIT,
            "neighbor_topology_sha256": topology_hashes,
        },
        "appearance_config": {
            "candidate_specific": True,
            "per_view": True,
            "feature_definition": (
                "center_excluded_same_image_sfm_maplet_quadrant_transport_v1"
            ),
            "support_view_marginalization": "not_aggregated_export_per_view_v1",
            "whole_image_retrieval_or_candidate_reselection": False,
        },
        "strict_fulltrack_appearance_contract": {
            "candidate_identity_fixed": True,
            "candidate_posterior_preserved": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "support_view_count_cap": None,
            "candidate_3d_projection_or_pose_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "heldout_s0_verification_rows": True,
            "source_fulltrack_csr_edges_preserved": True,
            "support_view_features_averaged_before_inference": False,
            "candidate_center_descriptor_excluded": True,
            "missing_or_partial_maplet_is_unknown": True,
            "availability_or_neighbor_count_is_not_a_learned_feature": True,
        },
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "transport_core_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm/localization/frozen_fulltrack_sfm_maplet_transport.py"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            verification_query_ids=source["verification_query_ids"],
            split_names=source["split_names"],
            verification_source_row_indices=source["verification_source_row_indices"],
            verification_xy=source["verification_xy"],
            candidate_track_ids=source["candidate_track_ids"],
            candidate_probabilities=candidate,
            null_probabilities=source["null_probabilities"],
            candidate_support_observation_counts=edges.candidate_observation_counts,
            source_maplet_support_view_counts=source_maplet_counts,
            profile_names=np.asarray(
                FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES, dtype=np.str_
            ),
            edge_candidate_offsets=offsets,
            edge_geometry_rows=np.asarray(edges.geometry_rows, dtype=np.int64),
            edge_profile_scores=scores.astype(np.float16, copy=False),
            edge_profile_valid=expected_valid,
            edge_maplet_profile_usable=profile_usable,
            edge_maplet_support_quadrant_counts=support_counts,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output_path)
    summary = {
        "stage": "build_frozen_fulltrack_per_view_sfm_maplet_transport",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "edge_count": edge_count,
        "feature_count": int(scores.shape[1]),
        "profile_edge_coverage": {
            profile.name: float(np.mean(profile_usable[:, index]))
            for index, profile in enumerate(SFM_MAPLET_TRANSPORT_PROFILES)
        },
        "devices": list(selected_devices),
        "workers": [worker for *_values, worker in computed],
        "elapsed_seconds": float(time.monotonic() - started),
        "protocol": {
            "fixed_global_topl": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "source_csr_edges_preserved": True,
            "all_real_sfm_track_observations": True,
            "support_view_features_averaged_before_inference": False,
            "candidate_center_descriptor_excluded": True,
            "hard_image_retrieval_or_submap": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "diagnostic_only": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frozen_fulltrack_per_view_sfm_maplet_transport(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        devices=_parse_devices(str(args.devices)),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
