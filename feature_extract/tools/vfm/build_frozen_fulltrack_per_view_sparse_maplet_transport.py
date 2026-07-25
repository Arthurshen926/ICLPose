"""Export v2 partial-coverage SfM-maplet transport and its topology control.

Every output edge is an already-frozen ``(query point, global-top20 track,
real SfM support observation)`` tuple.  The exporter receives no identity
target, query pose, pose hypothesis, retrieval result, rendering, candidate
reselection, or support-view reselection.  It only compares real-image
context around the query point with centre-excluded neighbouring SfM
observations in the support image.

Unlike v1, a maplet does not need one neighbour in each quadrant.  A
target-free minimum total-neighbour rule retains partial maplets, and a
separate artifact exposes exactly the non-visual topology/crop information
needed to audit density or border shortcuts.
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
from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_multiscale_translation_mode import (
    _TranslationSource,
    _crop_grid_torch,
    _load_translation_sources,
    _source_metadata,
)
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    _cache_image_indices,
    _parse_devices,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_sparse_maplet_transport import (
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT,
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT,
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
    SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS,
    SPARSE_MAPLET_TEMPERATURE,
    SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE,
    SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
    SPARSE_MAPLET_TRANSPORT_PROFILES,
    pool_sparse_maplet_query_quadrants,
    sparse_maplet_quadrant_transport_features,
    sparse_maplet_support_usable,
    sparse_maplet_topology_control_features,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)


ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_sparse_maplet_transport_v2"
TOPOLOGY_CACHE_FORMAT = "frozen_fulltrack_sparse_maplet_neighbor_topology_cache_v1"
_NEIGHBOR_QUERY_CANDIDATE_LIMIT = 128


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-per-view-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-pca256-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--topology-control-output", required=True)
    parser.add_argument("--topology-control-summary-json", required=True)
    parser.add_argument(
        "--neighbor-topology-cache",
        default=None,
        help="optional strict target-free sparse-maplet neighbour topology cache",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _edge_offsets(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("sparse-maplet candidate support counts are invalid")
    return np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(values.reshape(-1), dtype=np.int64))
    )


def _sources_by_name(sources: Sequence[_TranslationSource]) -> dict[str, _TranslationSource]:
    result = {str(source.name): source for source in sources}
    required = {str(profile.source_name) for profile in SPARSE_MAPLET_TRANSPORT_PROFILES}
    if set(result) != required:
        raise ValueError("sparse-maplet sources are incomplete or duplicated")
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES:
        source = result[profile.source_name]
        if int(source.grid_size) != int(profile.grid_size):
            raise ValueError(f"{profile.name}: source grid size differs from profile")
    return result


def _geometry_cache_rows(
    *, geometry: SupportObservationGeometryIndex, cache_image_ids: np.ndarray
) -> np.ndarray:
    image_rows = _geometry_image_indices(geometry)
    geometry_ids = np.asarray(geometry.image_ids).astype(str)[image_rows]
    return _cache_image_indices(
        cache_image_ids=np.asarray(cache_image_ids).astype(str),
        image_ids=geometry_ids,
        context="sparse-maplet real SfM observation",
    )


def _maplet_neighbor_array(
    *,
    geometry: SupportObservationGeometryIndex,
    cache_rows: np.ndarray,
    image_sizes: np.ndarray,
    grid_size: int,
    radius_cells: int,
    maximum_neighbors_per_quadrant: int,
) -> np.ndarray:
    """Build deterministic centre-excluded neighbours for every observation."""

    rows_by_image = _geometry_image_indices(geometry)
    xy = np.asarray(geometry.xy, dtype=np.float32)
    errors = np.asarray(geometry.reprojection_errors, dtype=np.float32)
    track_ids = np.asarray(geometry.track_ids, dtype=np.int64)
    cache = np.asarray(cache_rows, dtype=np.int64)
    sizes = np.asarray(image_sizes, dtype=np.int64)
    radius = int(radius_cells)
    maximum = int(maximum_neighbors_per_quadrant)
    if (
        xy.shape != (len(rows_by_image), 2)
        or errors.shape != (len(rows_by_image),)
        or track_ids.shape != (len(rows_by_image),)
        or cache.shape != (len(rows_by_image),)
        or sizes.ndim != 2
        or sizes.shape[1] != 2
        or radius < 2
        or maximum <= 0
        or np.any(cache < 0)
        or np.any(cache >= len(sizes))
        or np.any(sizes <= 1)
        or np.any(~np.isfinite(xy))
        or np.any(~np.isfinite(errors))
        or np.any(errors < 0.0)
    ):
        raise ValueError("sparse-maplet neighbour topology inputs are invalid")
    output = np.full(
        (len(xy), 4, maximum), -1, dtype=np.int64
    )
    requested = max(_NEIGHBOR_QUERY_CANDIDATE_LIMIT, 8 * maximum * 4)
    for image_index in range(len(geometry.image_ids)):
        rows = np.flatnonzero(rows_by_image == int(image_index)).astype(np.int64)
        if len(rows) <= 1:
            continue
        cache_for_image = np.unique(cache[rows])
        if len(cache_for_image) != 1:
            raise RuntimeError("one support image maps to multiple context-cache rows")
        width, height = (int(value) for value in sizes[int(cache_for_image[0])])
        grid = np.empty((len(rows), 2), dtype=np.float64)
        grid[:, 0] = xy[rows, 0] * float(int(grid_size) - 1) / float(width - 1)
        grid[:, 1] = xy[rows, 1] * float(int(grid_size) - 1) / float(height - 1)
        tree = cKDTree(grid)
        neighbor_count = min(len(rows), int(requested))
        distances, local = tree.query(
            grid,
            k=neighbor_count,
            distance_upper_bound=float(radius) * np.sqrt(2.0) + 1e-6,
            workers=-1,
        )
        distances = np.asarray(distances, dtype=np.float64)
        local = np.asarray(local, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            local = local[:, None]
        local_valid = (local >= 0) & (local < len(rows)) & np.isfinite(distances)
        safe_local = np.clip(local, 0, len(rows) - 1)
        neighbor_rows = rows[safe_local]
        delta = grid[safe_local] - grid[:, None, :]
        local_valid &= safe_local != np.arange(len(rows))[:, None]
        local_valid &= np.max(np.abs(delta), axis=2) <= float(radius) + 1e-6
        # The central observation is excluded from both maplet sides.  A
        # one-cell cross gap keeps its immediate local descriptor out too.
        local_valid &= np.abs(delta[:, :, 0]) >= 1.0
        local_valid &= np.abs(delta[:, :, 1]) >= 1.0
        quadrants = (2 * (delta[:, :, 1] > 0.0).astype(np.int64)) + (
            delta[:, :, 0] > 0.0
        ).astype(np.int64)
        neighbor_errors = errors[neighbor_rows]
        neighbor_tracks = track_ids[neighbor_rows]
        for quadrant in range(4):
            valid = local_valid & (quadrants == quadrant)
            ranking = np.where(valid, distances, np.inf)
            retained = min(maximum, ranking.shape[1])
            partial = np.argpartition(ranking, kth=retained - 1, axis=1)[:, :retained]
            selected_rows = np.take_along_axis(neighbor_rows, partial, axis=1)
            selected_distance = np.take_along_axis(ranking, partial, axis=1)
            selected_errors = np.take_along_axis(neighbor_errors, partial, axis=1)
            selected_tracks = np.take_along_axis(neighbor_tracks, partial, axis=1)
            order = np.lexsort(
                (selected_rows, selected_tracks, selected_errors, selected_distance), axis=1
            )
            selected_rows = np.take_along_axis(selected_rows, order, axis=1)
            selected_distance = np.take_along_axis(selected_distance, order, axis=1)
            output[rows, quadrant, :retained] = np.where(
                np.isfinite(selected_distance), selected_rows, -1
            )
    return output


def _maplet_neighbor_topologies(
    *, geometry: SupportObservationGeometryIndex, sources: Sequence[_TranslationSource]
) -> dict[str, np.ndarray]:
    by_name = _sources_by_name(sources)
    cache_rows = _geometry_cache_rows(
        geometry=geometry,
        cache_image_ids=by_name["radio_final"].image_ids,
    )
    shared: dict[tuple[int, int], np.ndarray] = {}
    output: dict[str, np.ndarray] = {}
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES:
        source = by_name[profile.source_name]
        key = (int(source.grid_size), int(profile.radius_cells))
        if key not in shared:
            shared[key] = _maplet_neighbor_array(
                geometry=geometry,
                cache_rows=cache_rows,
                image_sizes=source.image_sizes,
                grid_size=int(source.grid_size),
                radius_cells=int(profile.radius_cells),
                maximum_neighbors_per_quadrant=SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
            )
        output[profile.name] = shared[key]
    return output


def _topology_key(profile: object) -> str:
    grid_size = int(getattr(profile, "grid_size"))
    radius = int(getattr(profile, "radius_cells"))
    if grid_size <= 0 or radius < 2:
        raise ValueError("sparse-maplet topology key inputs are invalid")
    return f"grid{grid_size}_radius{radius}"


def _topology_cache_source_contract(
    *, sources: Sequence[_TranslationSource]
) -> dict[str, Any]:
    by_name = _sources_by_name(sources)
    final = by_name["radio_final"]
    return {
        "context_cache_sha256": {
            name: file_sha256_short(source.cache_path)
            for name, source in sorted(by_name.items())
        },
        "source_image_ids_sha256": _array_sha256_short(final.image_ids),
        "source_image_sizes_sha256": _array_sha256_short(final.image_sizes),
        "source_metadata": [_source_metadata(by_name[name]) for name in sorted(by_name)],
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "grid_size": int(profile.grid_size),
                "radius_cells": int(profile.radius_cells),
                "topology_key": _topology_key(profile),
            }
            for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
        ],
        "maximum_neighbors_per_quadrant": SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
        "minimum_total_neighbors": SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS,
    }


def _normalise_topology_cache_source_contract_paths(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize non-semantic cache paths before strict provenance compare.

    Older topology caches were often written from an absolute workspace path,
    while current callers may pass the identical files relative to the
    repository.  The file hash, source manifest, PCA/checkpoint fields, and
    geometry hash remain the identity checks; only this filesystem spelling is
    normalized.
    """

    output = dict(contract)
    metadata = output.get("source_metadata")
    if not isinstance(metadata, list):
        raise ValueError("sparse-maplet topology source contract lacks source metadata")
    normalized: list[dict[str, Any]] = []
    for item in metadata:
        if not isinstance(item, Mapping):
            raise ValueError("sparse-maplet topology source metadata is invalid")
        entry = dict(item)
        cache = entry.get("cache")
        if not isinstance(cache, str) or not cache:
            raise ValueError("sparse-maplet topology source cache path is invalid")
        entry["cache"] = str(Path(cache).expanduser().resolve())
        normalized.append(entry)
    output["source_metadata"] = normalized
    return output


def write_sparse_maplet_neighbor_topology_cache(
    *,
    output: Path,
    geometry_path: Path,
    geometry_metadata: Mapping[str, Any],
    sources: Sequence[_TranslationSource],
    neighbour_topologies: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Persist target-free same-image neighbour topology with strict lineage."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite sparse-maplet topology cache")
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("sparse-maplet topology cache needs real SfM observation xy")
    source_contract = _topology_cache_source_contract(sources=sources)
    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    expected_rows: int | None = None
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES:
        values = np.asarray(neighbour_topologies.get(profile.name), dtype=np.int64)
        if (
            values.ndim != 3
            or values.shape[1:] != (4, SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT)
            or np.any(values < -1)
        ):
            raise ValueError(f"{profile.name}: sparse-maplet topology is invalid")
        if expected_rows is None:
            expected_rows = int(values.shape[0])
        elif int(values.shape[0]) != expected_rows:
            raise ValueError("sparse-maplet topology profile row counts differ")
        key = _topology_key(profile)
        converted = values.astype(np.int32, copy=False)
        if key in arrays and not np.array_equal(arrays[key], converted):
            raise ValueError("shared sparse-maplet topology key has conflicting values")
        arrays[key] = converted
        hashes[profile.name] = _array_sha256_short(converted)
    if expected_rows is None or expected_rows <= 0:
        raise ValueError("sparse-maplet topology cache has no geometry rows")
    metadata = {
        "format": TOPOLOGY_CACHE_FORMAT,
        "version": 1,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "support_geometry_index": str(Path(geometry_path)),
        "support_geometry_index_sha256": file_sha256_short(Path(geometry_path)),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "geometry_row_count": int(expected_rows),
        "source_contract": source_contract,
        "profile_topology_sha256": hashes,
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "sparse_maplet_core_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm/localization/frozen_fulltrack_sparse_maplet_transport.py"
            ),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{f"topology_{key}": values for key, values in arrays.items()},
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(destination)
    return {
        "stage": "build_frozen_fulltrack_sparse_maplet_neighbor_topology_cache",
        "output": str(destination),
        "output_sha256": file_sha256_short(destination),
        "geometry_row_count": int(expected_rows),
        "topology_key_count": int(len(arrays)),
        "protocol": {
            "target_free": True,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }


def load_sparse_maplet_neighbor_topology_cache(
    *,
    path: Path,
    geometry_path: Path,
    geometry_metadata: Mapping[str, Any],
    geometry_row_count: int,
    sources: Sequence[_TranslationSource],
) -> dict[str, np.ndarray]:
    """Load only a cache whose geometry, source images, and profile config match."""

    source = Path(path)
    contract = _topology_cache_source_contract(sources=sources)
    with np.load(source, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError("sparse-maplet topology cache lacks metadata")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        cached_contract = _normalise_topology_cache_source_contract_paths(
            metadata.get("source_contract", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        expected_contract = _normalise_topology_cache_source_contract_paths(contract)
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("format") != TOPOLOGY_CACHE_FORMAT
            or metadata.get("version") != 1
            or metadata.get("contains_ground_truth") is not False
            or metadata.get("contains_target_errors") is not False
            or metadata.get("pose_or_ground_truth_used") is not False
            or metadata.get("supervision_arrays_loaded") is not False
            or metadata.get("support_geometry_index_sha256")
            != file_sha256_short(Path(geometry_path))
            or metadata.get("support_geometry_coordinate_source")
            != geometry_metadata.get("coordinate_source")
            or metadata.get("geometry_row_count") != int(geometry_row_count)
            or cached_contract != expected_contract
        ):
            raise ValueError("sparse-maplet topology cache lineage is stale or incompatible")
        hashes = metadata.get("profile_topology_sha256")
        if not isinstance(hashes, Mapping):
            raise ValueError("sparse-maplet topology cache lacks profile hashes")
        by_key: dict[str, np.ndarray] = {}
        for profile in SPARSE_MAPLET_TRANSPORT_PROFILES:
            key = _topology_key(profile)
            member = f"topology_{key}"
            if member not in payload.files:
                raise ValueError(f"sparse-maplet topology cache lacks {member}")
            values = np.asarray(payload[member], dtype=np.int32)
            if (
                values.shape
                != (
                    int(geometry_row_count),
                    4,
                    SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
                )
                or np.any(values < -1)
                or _array_sha256_short(values) != hashes.get(profile.name)
            ):
                raise ValueError(f"{profile.name}: cached sparse-maplet topology differs")
            if key in by_key and not np.array_equal(by_key[key], values):
                raise ValueError("sparse-maplet topology cache key is inconsistent")
            by_key[key] = values
    return {profile.name: by_key[_topology_key(profile)] for profile in SPARSE_MAPLET_TRANSPORT_PROFILES}


def _sample_support_quadrants(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    neighbor_geometry_rows: np.ndarray,
    geometry_xy: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool real sparse support observations into their available quadrants."""

    grids = torch.as_tensor(image_grids)
    sizes = torch.as_tensor(image_sizes, dtype=torch.float32, device=grids.device)
    indices = torch.as_tensor(image_indices, dtype=torch.long, device=grids.device)
    neighbours = torch.as_tensor(
        neighbor_geometry_rows, dtype=torch.long, device=grids.device
    )
    coordinates = torch.as_tensor(geometry_xy, dtype=torch.float32, device=grids.device)
    if (
        grids.ndim != 4
        or grids.shape[1] != grids.shape[2]
        or sizes.shape != (grids.shape[0], 2)
        or indices.ndim != 1
        or neighbours.ndim != 3
        or neighbours.shape[0] != len(indices)
        or neighbours.shape[1] != 4
        or neighbours.shape[2] != SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT
        or coordinates.shape != (len(geometry_xy), 2)
        or torch.any(neighbours < -1)
    ):
        raise ValueError("sparse-maplet support quadrant sampler inputs are invalid")
    safe = neighbours.clamp_min(0)
    present = neighbours >= 0
    neighbour_xy = coordinates[safe]
    selected_sizes = sizes.index_select(0, indices)
    normalized = torch.stack(
        (
            2.0 * neighbour_xy[..., 0] / selected_sizes[:, None, None, 0].sub(1.0)
            - 1.0,
            2.0 * neighbour_xy[..., 1] / selected_sizes[:, None, None, 1].sub(1.0)
            - 1.0,
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
    sampled = F.grid_sample(
        selected,
        normalized.reshape(len(indices), -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
        len(indices), 4, SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT, grids.shape[3]
    )
    present = present & on_image
    counts = present.sum(dim=2)
    pooled = (sampled * present[..., None].to(dtype=sampled.dtype)).sum(dim=2)
    pooled = pooled / counts.clamp_min(1).to(dtype=sampled.dtype)[..., None]
    pooled = F.normalize(pooled, p=2, dim=2, eps=1e-8)
    return pooled, counts


def _profile_slices(*, control: bool) -> dict[str, slice]:
    offset = 0
    output: dict[str, slice] = {}
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES:
        names = (
            SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE[profile.name]
            if bool(control)
            else SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]
        )
        output[profile.name] = slice(offset, offset + len(names))
        offset += len(names)
    expected = (
        len(FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES)
        if bool(control)
        else len(FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES)
    )
    if offset != expected:
        raise RuntimeError("sparse-maplet profile slices drifted")
    return output


def _compute_partition(
    *,
    device_name: str,
    sources: Sequence[_TranslationSource],
    query_cache_rows: np.ndarray,
    query_xy: np.ndarray,
    edge_rows: np.ndarray,
    edge_support_cache_rows: np.ndarray,
    edge_geometry_rows: np.ndarray,
    geometry_xy: np.ndarray,
    neighbour_topologies: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Export one disjoint immutable CSR range on a single GPU."""

    if int(end) <= int(begin) or int(batch_size) <= 0:
        raise ValueError("sparse-maplet partition is invalid")
    started = time.monotonic()
    device = torch.device(device_name)
    by_name = _sources_by_name(sources)
    visual_width = len(FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES)
    control_width = len(FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES)
    visual_slices = _profile_slices(control=False)
    control_slices = _profile_slices(control=True)
    profile_count = len(SPARSE_MAPLET_TRANSPORT_PROFILES)
    with torch.inference_mode():
        tensors = {
            name: (
                torch.from_numpy(source.grids).to(device=device, dtype=torch.float32),
                torch.from_numpy(source.image_sizes).to(device=device, dtype=torch.float32),
            )
            for name, source in by_name.items()
        }
        query_context = {
            profile.name: _crop_grid_torch(
                image_grids=tensors[profile.source_name][0],
                image_sizes=tensors[profile.source_name][1],
                image_indices=torch.as_tensor(query_cache_rows, device=device),
                xy=torch.as_tensor(query_xy, device=device, dtype=torch.float32),
                window_size=int(profile.window_size),
                padding_mode="reflection",
            )
            for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
        }
        query_visual: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        query_original_coverage: dict[str, torch.Tensor] = {}
        for profile in SPARSE_MAPLET_TRANSPORT_PROFILES:
            crop, original_mask = query_context[profile.name]
            visual_values, visual_valid, _ = pool_sparse_maplet_query_quadrants(
                patches=crop,
                valid=torch.ones_like(original_mask),
            )
            _unused, _unused_valid, original_coverage = pool_sparse_maplet_query_quadrants(
                patches=crop,
                valid=original_mask,
            )
            query_visual[profile.name] = (visual_values, visual_valid)
            query_original_coverage[profile.name] = original_coverage
        edge_count = int(end) - int(begin)
        visual = np.full((edge_count, visual_width), np.nan, dtype=np.float16)
        visual_valid = np.zeros(visual.shape, dtype=bool)
        control = np.full((edge_count, control_width), np.nan, dtype=np.float16)
        control_valid = np.zeros(control.shape, dtype=bool)
        profile_usable = np.zeros((edge_count, profile_count), dtype=bool)
        support_counts = np.zeros((edge_count, profile_count, 4), dtype=np.uint8)
        for offset in range(int(begin), int(end), int(batch_size)):
            stop = min(offset + int(batch_size), int(end))
            local_slice = slice(offset - int(begin), stop - int(begin))
            point_rows = torch.from_numpy(edge_rows[offset:stop]).to(device=device)
            support_rows = torch.from_numpy(edge_support_cache_rows[offset:stop]).to(
                device=device
            )
            geometry_rows = np.asarray(edge_geometry_rows[offset:stop], dtype=np.int64)
            for profile_index, profile in enumerate(SPARSE_MAPLET_TRANSPORT_PROFILES):
                source_grid, source_sizes = tensors[profile.source_name]
                support_values, current_counts = _sample_support_quadrants(
                    image_grids=source_grid,
                    image_sizes=source_sizes,
                    image_indices=support_rows,
                    neighbor_geometry_rows=np.asarray(
                        neighbour_topologies[profile.name][geometry_rows], dtype=np.int64
                    ),
                    geometry_xy=geometry_xy,
                )
                query_values, query_valid = query_visual[profile.name]
                query_values = query_values.index_select(0, point_rows)
                query_valid = query_valid.index_select(0, point_rows)
                support_valid = current_counts > 0
                values = sparse_maplet_quadrant_transport_features(
                    query_quadrants=query_values,
                    query_valid=query_valid,
                    support_quadrants=support_values,
                    support_valid=support_valid,
                    temperature=SPARSE_MAPLET_TEMPERATURE,
                )
                original_coverage = query_original_coverage[profile.name].index_select(
                    0, point_rows
                )
                control_values = sparse_maplet_topology_control_features(
                    query_original_coverage=original_coverage,
                    support_neighbor_counts=current_counts,
                )
                usable = sparse_maplet_support_usable(current_counts)
                usable_np = usable.cpu().numpy().astype(bool, copy=False)
                counts_np = current_counts.cpu().numpy().astype(np.uint8, copy=False)
                profile_usable[local_slice, profile_index] = usable_np
                support_counts[local_slice, profile_index] = counts_np
                visual_slice = visual_slices[profile.name]
                control_slice = control_slices[profile.name]
                local_visual = visual[local_slice]
                local_visual_valid = visual_valid[local_slice]
                local_control = control[local_slice]
                local_control_valid = control_valid[local_slice]
                local_visual[usable_np, visual_slice] = (
                    values[usable].cpu().numpy().astype(np.float16, copy=False)
                )
                local_visual_valid[usable_np, visual_slice] = True
                local_control[usable_np, control_slice] = (
                    control_values[usable].cpu().numpy().astype(np.float16, copy=False)
                )
                local_control_valid[usable_np, control_slice] = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    if (
        np.any(np.isinf(visual))
        or np.any(np.isinf(control))
        or np.any(~np.isfinite(visual[visual_valid]))
        or np.any(~np.isfinite(control[control_valid]))
        or np.any(np.isfinite(visual[~visual_valid]))
        or np.any(np.isfinite(control[~control_valid]))
    ):
        raise RuntimeError("sparse-maplet exporter emitted invalid missingness")
    return visual, visual_valid, control, control_valid, support_counts, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
        "profile_usable_fraction": {
            profile.name: float(profile_usable[:, index].mean())
            for index, profile in enumerate(SPARSE_MAPLET_TRANSPORT_PROFILES)
        },
        "profile_mean_total_neighbors": {
            profile.name: float(support_counts[:, index].sum(axis=1).mean())
            for index, profile in enumerate(SPARSE_MAPLET_TRANSPORT_PROFILES)
        },
    }


def _context_contract(sources: Sequence[_TranslationSource]) -> dict[str, Any]:
    by_name = _sources_by_name(sources)
    return {
        "mode": "partial_center_excluded_sfm_maplet_transport_v2",
        "support_coordinate_source": "sfm_observation_xy",
        "support_neighbor_scope": "same_real_support_image_only",
        "view_aggregation": "none_before_learned_logsumexp_mixture_v1",
        "center_descriptor_in_features": False,
        "partial_support_quadrants_retained": True,
        "minimum_total_neighbors": SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS,
        "maximum_neighbors_per_quadrant": SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
        "visual_border_padding": "reflection_from_real_feature_map_v1",
        "original_crop_topology_control_exported_separately": True,
        "availability_is_a_visual_feature": False,
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "grid_size": int(profile.grid_size),
                "radius_cells": int(profile.radius_cells),
                "window_size": int(profile.window_size),
                "visual_feature_names": list(
                    SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]
                ),
                "topology_control_feature_names": list(
                    SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE[
                        profile.name
                    ]
                ),
            }
            for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
        ],
        "sources": [_source_metadata(by_name[profile.source_name]) for profile in SPARSE_MAPLET_TRANSPORT_PROFILES],
    }


def _base_metadata(
    *,
    source: Mapping[str, np.ndarray],
    source_lineage: Mapping[str, str],
    geometry_path: Path,
    geometry_metadata: Mapping[str, Any],
    sources: Sequence[_TranslationSource],
    query_id: str,
    contract: Mapping[str, Any],
    topology_hashes: Mapping[str, str],
    control: bool,
) -> dict[str, Any]:
    by_name = _sources_by_name(sources)
    return {
        "format": (
            FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT
            if bool(control)
            else FULLTRACK_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT
        ),
        "version": ARTIFACT_VERSION,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "query_id": str(query_id),
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
        "source_edge_feature_semantics": source_lineage["source_edge_feature_semantics"],
        "source_fulltrack_edge_contract": source_lineage["source_kind"],
        "source_candidate_tracks_sha256": _array_sha256_short(source["candidate_track_ids"]),
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
            "radio_final": file_sha256_short(by_name["radio_final"].cache_path),
            "radio_intermediate_pca256": file_sha256_short(
                by_name["radio_intermediate_pca256"].cache_path
            ),
            "alike": file_sha256_short(by_name["alike_fpn"].cache_path),
        },
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_preserve_source_csr_order_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": (
            FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
            if bool(control)
            else FULLTRACK_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "sparse_maplet_transport_contract": {
            **dict(contract),
            "neighbor_topology_sha256": dict(topology_hashes),
            "transport_temperature": SPARSE_MAPLET_TEMPERATURE,
        },
        "artifact_role": (
            "topology_and_original_crop_control"
            if bool(control)
            else "visual_descriptor_transport"
        ),
        "appearance_config": {
            "candidate_specific": True,
            "per_view": True,
            "control_only": bool(control),
            "visual_descriptor_values_included": not bool(control),
            "topology_or_original_crop_values_included": bool(control),
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
            "partial_support_maplet_is_retained_when_total_neighbors_sufficient": True,
            "visual_descriptor_values_included": not bool(control),
            "topology_or_original_crop_values_included": bool(control),
            "paired_topology_control_artifact_required": True,
        },
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "sparse_maplet_core_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm/localization/frozen_fulltrack_sparse_maplet_transport.py"
            ),
        },
    }


def build_frozen_fulltrack_per_view_sparse_maplet_transport(
    *,
    source_per_view_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_pca256_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    topology_control_output: Path,
    topology_control_summary_json: Path,
    neighbor_topology_cache: Path | None,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    """Attach v2 sparse maplet appearance and control to one raw CSR shard."""

    selected_devices = tuple(str(value) for value in devices)
    if (
        int(batch_size) <= 0
        or not selected_devices
        or len(set(selected_devices)) != len(selected_devices)
    ):
        raise ValueError("sparse-maplet export configuration is invalid")
    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    final_path = Path(radio_final_context_cache)
    intermediate_path = Path(radio_intermediate_pca256_context_cache)
    alike_path = Path(alike_spatial_context_cache)
    output_path = Path(output)
    summary_path = Path(summary_json)
    control_path = Path(topology_control_output)
    control_summary_path = Path(topology_control_summary_json)
    destinations = (output_path, summary_path, control_path, control_summary_path)
    if any(path.exists() for path in destinations) and not bool(force):
        raise FileExistsError("refusing to overwrite sparse-maplet outputs")
    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("sparse-maplet transport requires real SfM observation xy")
    source, source_metadata, edges, source_maplet_counts, source_lineage = (
        _load_current_raw_per_view_source(
            source_path=source_path,
            support_geometry_index=geometry_path,
            geometry=geometry,
        )
    )
    raw_hashes = source_metadata.get("context_cache_sha256")
    if (
        not isinstance(raw_hashes, Mapping)
        or str(raw_hashes.get("radio_final", "")) != file_sha256_short(final_path)
    ):
        raise ValueError("raw full-track source and RADIO-final cache differ")
    sources = _load_translation_sources(
        radio_final_context_cache=final_path,
        radio_intermediate_pca256_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
    )
    by_name = _sources_by_name(sources)
    query_cache_rows = _cache_image_indices(
        cache_image_ids=by_name["radio_final"].image_ids,
        image_ids=np.asarray(source["verification_query_ids"]).astype(str),
        context="sparse-maplet frozen query",
    )
    geometry_cache_rows = _geometry_cache_rows(
        geometry=geometry, cache_image_ids=by_name["radio_final"].image_ids
    )
    geometry_rows = np.asarray(edges.geometry_rows, dtype=np.int64)
    edge_support_cache_rows = geometry_cache_rows[geometry_rows]
    edge_rows = np.asarray(edges.edge_candidate_indices, dtype=np.int64) // int(
        edges.candidate_shape[1]
    )
    edge_count = int(edges.edge_count)
    if edge_count <= 0 or edge_count != len(edge_rows):
        raise ValueError("sparse-maplet source has no immutable CSR edges")
    neighbour_topologies = (
        _maplet_neighbor_topologies(geometry=geometry, sources=sources)
        if neighbor_topology_cache is None
        else load_sparse_maplet_neighbor_topology_cache(
            path=Path(neighbor_topology_cache),
            geometry_path=geometry_path,
            geometry_metadata=geometry_metadata,
            geometry_row_count=len(geometry.track_ids),
            sources=sources,
        )
    )
    topology_hashes = {
        profile.name: _array_sha256_short(neighbour_topologies[profile.name])
        for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
    }
    partitions = tuple(
        (
            edge_count * index // len(selected_devices),
            edge_count * (index + 1) // len(selected_devices),
        )
        for index in range(len(selected_devices))
    )
    if any(end <= begin for begin, end in partitions):
        raise ValueError("more sparse-maplet devices than immutable CSR edges")
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
                edge_geometry_rows=geometry_rows,
                geometry_xy=np.asarray(geometry.xy, dtype=np.float32),
                neighbour_topologies=neighbour_topologies,
                begin=begin,
                end=end,
                batch_size=int(batch_size),
            )
            for device, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    visual = np.concatenate([entry[0] for entry in computed], axis=0)
    visual_valid = np.concatenate([entry[1] for entry in computed], axis=0)
    control = np.concatenate([entry[2] for entry in computed], axis=0)
    control_valid = np.concatenate([entry[3] for entry in computed], axis=0)
    support_counts = np.concatenate([entry[4] for entry in computed], axis=0)
    if (
        visual.shape != (edge_count, len(FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES))
        or visual_valid.shape != visual.shape
        or control.shape
        != (edge_count, len(FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES))
        or control_valid.shape != control.shape
        or support_counts.shape
        != (edge_count, len(SPARSE_MAPLET_TRANSPORT_PROFILES), 4)
    ):
        raise RuntimeError("sparse-maplet output shape drifted")
    offsets = _edge_offsets(np.asarray(edges.candidate_observation_counts, dtype=np.int64))
    if (
        offsets[-1] != edge_count
        or not np.array_equal(
            np.repeat(
                np.arange(int(edges.candidate_shape[0]) * int(edges.candidate_shape[1])),
                np.diff(offsets),
            ),
            np.asarray(edges.edge_candidate_indices, dtype=np.int64),
        )
    ):
        raise RuntimeError("sparse-maplet export changed immutable CSR ordering")
    candidate = np.asarray(source["candidate_probabilities"], dtype=np.float32)
    if np.any((candidate > 0.0) & (edges.candidate_observation_counts <= 0)):
        raise RuntimeError("positive frozen candidate lost all real support observations")
    query_id = str(source["verification_query_ids"][0])
    contract = _context_contract(sources)
    visual_metadata = _base_metadata(
        source=source,
        source_lineage=source_lineage,
        geometry_path=geometry_path,
        geometry_metadata=geometry_metadata,
        sources=sources,
        query_id=query_id,
        contract=contract,
        topology_hashes=topology_hashes,
        control=False,
    )
    control_metadata = _base_metadata(
        source=source,
        source_lineage=source_lineage,
        geometry_path=geometry_path,
        geometry_metadata=geometry_metadata,
        sources=sources,
        query_id=query_id,
        contract=contract,
        topology_hashes=topology_hashes,
        control=True,
    )
    common = {
        "verification_query_ids": source["verification_query_ids"],
        "split_names": source["split_names"],
        "verification_source_row_indices": source["verification_source_row_indices"],
        "verification_xy": source["verification_xy"],
        "candidate_track_ids": source["candidate_track_ids"],
        "candidate_probabilities": candidate,
        "null_probabilities": source["null_probabilities"],
        "candidate_support_observation_counts": edges.candidate_observation_counts,
        "source_maplet_support_view_counts": source_maplet_counts,
        "edge_candidate_offsets": offsets,
        "edge_geometry_rows": geometry_rows,
        "edge_sparse_maplet_support_quadrant_counts": support_counts,
    }
    for path in (output_path, summary_path, control_path, control_summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, names, scores, valid, metadata in (
        (
            output_path,
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES,
            visual,
            visual_valid,
            visual_metadata,
        ),
        (
            control_path,
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES,
            control,
            control_valid,
            control_metadata,
        ),
    ):
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **common,
                profile_names=np.asarray(names, dtype=np.str_),
                edge_profile_scores=scores.astype(np.float16, copy=False),
                edge_profile_valid=valid,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        temporary.replace(path)
    profile_slices = _profile_slices(control=False)
    summary = {
        "stage": "build_frozen_fulltrack_per_view_sparse_maplet_transport",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "topology_control_output": str(control_path),
        "topology_control_output_sha256": file_sha256_short(control_path),
        "neighbor_topology_cache": (
            None if neighbor_topology_cache is None else str(Path(neighbor_topology_cache))
        ),
        "neighbor_topology_cache_sha256": (
            None
            if neighbor_topology_cache is None
            else file_sha256_short(Path(neighbor_topology_cache))
        ),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "edge_count": edge_count,
        "visual_feature_count": int(visual.shape[1]),
        "topology_control_feature_count": int(control.shape[1]),
        "profile_edge_coverage": {
            profile.name: float(
                np.all(visual_valid[:, profile_slices[profile.name]], axis=1).mean()
            )
            for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
        },
        "profile_mean_total_neighbors": {
            profile.name: float(support_counts[:, index].sum(axis=1).mean())
            for index, profile in enumerate(SPARSE_MAPLET_TRANSPORT_PROFILES)
        },
        "devices": list(selected_devices),
        "workers": [entry[5] for entry in computed],
        "elapsed_seconds": float(time.monotonic() - started),
        "protocol": {
            "fixed_global_topl": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "source_csr_edges_preserved": True,
            "all_real_sfm_track_observations": True,
            "support_view_features_averaged_before_inference": False,
            "hard_image_retrieval_or_submap": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "partial_maplets_retained": True,
            "matched_topology_control_exported": True,
        },
    }
    control_summary = {
        **summary,
        "output": str(control_path),
        "output_sha256": file_sha256_short(control_path),
        "artifact_role": "topology_and_original_crop_control",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    control_summary_path.write_text(
        json.dumps(control_summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_frozen_fulltrack_per_view_sparse_maplet_transport(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            args.radio_intermediate_pca256_context_cache
        ),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        topology_control_output=Path(args.topology_control_output),
        topology_control_summary_json=Path(args.topology_control_summary_json),
        neighbor_topology_cache=(
            None
            if args.neighbor_topology_cache is None
            else Path(args.neighbor_topology_cache)
        ),
        devices=_parse_devices(args.devices),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
