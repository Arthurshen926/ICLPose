"""Export fixed support8 landmark-centred visual-region prototype features.

Each frozen global top-L landmark candidate keeps the deterministic maplet
support observations already attached to it.  A query token and every support
observation are cropped from the real RADIO-final spatial grid around their
respective image coordinates.  The exporter compares a pooled crop and its
anchor-relative 3x3 spatial regions, but never searches images, changes the
candidate pool, reselects support views, reads a query pose, or reads target
labels.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
    _load_frozen_layout,
)
from feature_extract.tools.vfm.build_global_context_support8_candidate_probe_features import (
    _fixed_support8_layout,
    _load_maplet_support_index,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    HIGHRES_LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_GRID_SIZE,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_WINDOW_SIZES,
    LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES,
    LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
    LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    LANDMARK_REGION_PROTOTYPE_REGION_GRID_SIZE,
    LANDMARK_REGION_PROTOTYPE_WINDOW_SIZE,
)
from feature_extract.vfm.localization.radio_final_context import (
    RADIO_FINAL_CONTEXT_PCA_FORMAT,
    load_radio_final_context_pca_cache,
)


ARTIFACT_FORMAT = LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT
MAPLET_FORMAT = "local_maplet_support_index_npz"
DEFAULT_PROFILE_NAME = "grid8_window7"


@dataclass(frozen=True)
class _RegionPrototypeProfile:
    name: str
    artifact_format: str
    grid_size: int
    window_sizes: tuple[int, ...]
    feature_names: tuple[str, ...]
    context_feature_names: tuple[str, ...]


_REGION_PROTOTYPE_PROFILES: Mapping[str, _RegionPrototypeProfile] = {
    DEFAULT_PROFILE_NAME: _RegionPrototypeProfile(
        name=DEFAULT_PROFILE_NAME,
        artifact_format=LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
        grid_size=8,
        window_sizes=(LANDMARK_REGION_PROTOTYPE_WINDOW_SIZE,),
        feature_names=LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        context_feature_names=LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
    ),
    "grid16_multiscale7_11": _RegionPrototypeProfile(
        name="grid16_multiscale7_11",
        artifact_format=HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
        grid_size=HIGHRES_LANDMARK_REGION_PROTOTYPE_GRID_SIZE,
        window_sizes=HIGHRES_LANDMARK_REGION_PROTOTYPE_WINDOW_SIZES,
        feature_names=HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
        context_feature_names=HIGHRES_LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
    ),
}


def _profile(name: str) -> _RegionPrototypeProfile:
    output = _REGION_PROTOTYPE_PROFILES.get(str(name))
    if output is None:
        raise ValueError(f"unsupported landmark-region prototype profile {name!r}")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(_REGION_PROTOTYPE_PROFILES),
        default=DEFAULT_PROFILE_NAME,
        help="predeclared spatial-grid and landmark-centred crop configuration",
    )
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="comma-separated devices; each gets a disjoint fixed candidate-view batch",
    )
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must contain unique non-empty device names")
    for name in devices:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device is unavailable: {name}")
    return devices


def _candidate_radio_final_anchor(layout: Mapping[str, np.ndarray]) -> np.ndarray:
    """Recover the candidate-only RADIO-final anchor from the frozen layout."""

    names = tuple(str(value) for value in np.asarray(layout["feature_names"]).tolist())
    if LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES[0] not in names:
        raise ValueError("frozen layout lacks the RADIO-final candidate anchor")
    values = np.asarray(layout["candidate_features"], dtype=np.float32)[
        ..., names.index(LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES[0])
    ]
    views = np.asarray(layout["candidate_view_valid"], dtype=bool)
    tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    if values.shape != views.shape or tracks.shape != values.shape[:2]:
        raise ValueError("frozen candidate anchor arrays are incompatible")
    has_view = np.any(views, axis=2)
    if np.any((tracks >= 0) & ~has_view):
        raise ValueError("a frozen candidate lacks a source support view")
    first_view = np.argmax(views, axis=2)
    rows = np.arange(values.shape[0])[:, None]
    columns = np.arange(values.shape[1])[None, :]
    output = values[rows, columns, first_view]
    valid = tracks >= 0
    if np.any(~np.isfinite(output[valid])):
        raise ValueError("frozen RADIO-final candidate anchor is non-finite")
    repeated = np.broadcast_to(output[..., None], values.shape)
    disagreement = views & (np.abs(values - repeated) > 5e-4)
    if np.any(disagreement):
        raise ValueError("frozen RADIO-final candidate anchor unexpectedly depends on support view")
    output = output.astype(np.float32, copy=False)
    output[~valid] = np.nan
    return output


def _cache_image_indices(
    *,
    cache_image_ids: np.ndarray,
    image_ids: np.ndarray,
    context: str,
) -> np.ndarray:
    lookup = {str(image_id): index for index, image_id in enumerate(cache_image_ids.tolist())}
    output = np.empty((len(image_ids),), dtype=np.int64)
    for row, image_id in enumerate(np.asarray(image_ids).astype(str).tolist()):
        position = lookup.get(str(image_id))
        if position is None:
            raise ValueError(f"RADIO-final spatial cache misses {context} image {image_id!r}")
        output[row] = int(position)
    return output


def _fixed_candidate_view_edges(
    *,
    layout: Mapping[str, np.ndarray],
    support_ids: np.ndarray,
    view_valid: np.ndarray,
    geometry: object,
    radio_image_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Resolve every fixed support slot to its real SfM observation xy."""

    tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    query_ids = np.asarray(layout["query_ids"]).astype(str)
    support = np.asarray(support_ids).astype(str)
    valid = np.asarray(view_valid, dtype=bool)
    if support.shape != valid.shape or tracks.shape != valid.shape[:2]:
        raise ValueError("fixed support8 candidate-view arrays are incompatible")
    edge_rows, edge_candidates, edge_views = np.nonzero(valid)
    if len(edge_rows) == 0:
        raise ValueError("fixed support8 layout has no valid candidate views")
    edge_support_ids = support[edge_rows, edge_candidates, edge_views]
    if np.any(edge_support_ids == ""):
        raise ValueError("a valid fixed support view lacks an image id")
    if set(query_ids.tolist()) & set(np.unique(edge_support_ids).tolist()):
        raise ValueError("query images leaked into fixed support-view region prototypes")
    edge_support_cache_rows = _cache_image_indices(
        cache_image_ids=radio_image_ids,
        image_ids=edge_support_ids,
        context="fixed support",
    )
    edge_support_xy = np.empty((len(edge_rows), 2), dtype=np.float32)
    order = np.argsort(edge_support_cache_rows, kind="stable")
    ordered_cache_rows = edge_support_cache_rows[order]
    starts = np.r_[0, np.flatnonzero(ordered_cache_rows[1:] != ordered_cache_rows[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for begin, end in zip(starts.tolist(), ends.tolist()):
        positions = order[int(begin) : int(end)]
        image_id = str(edge_support_ids[positions[0]])
        geometry_rows = geometry.geometry_rows_for_tracks(
            image_id,
            tracks[edge_rows[positions], edge_candidates[positions]],
        )
        if np.any(geometry_rows < 0):
            raise ValueError(
                "fixed maplet support view does not observe its candidate track: "
                f"{image_id!r}"
            )
        edge_support_xy[positions] = np.asarray(geometry.xy[geometry_rows], dtype=np.float32)
    flat_positions = np.ravel_multi_index(
        (edge_rows, edge_candidates, edge_views), dims=valid.shape
    )
    return {
        "row": edge_rows.astype(np.int64, copy=False),
        "candidate": edge_candidates.astype(np.int64, copy=False),
        "view": edge_views.astype(np.int64, copy=False),
        "flat_position": flat_positions.astype(np.int64, copy=False),
        "support_cache_row": edge_support_cache_rows.astype(np.int64, copy=False),
        "support_xy": edge_support_xy,
    }


def _crop_region_prototypes_torch(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    grid_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pooled and 3x3 crop prototypes plus their explicit masks."""

    if (
        image_grids.ndim != 4
        or image_grids.shape[1] != image_grids.shape[2]
        or image_sizes.shape != (image_grids.shape[0], 2)
        or image_indices.ndim != 1
        or xy.shape != (len(image_indices), 2)
    ):
        raise ValueError("landmark-region crop tensors are incompatible")
    actual_grid_size = int(image_grids.shape[1])
    grid_size = int(grid_size)
    window_size = int(window_size)
    if (
        actual_grid_size != grid_size
        or grid_size <= 0
        or window_size <= 0
        or window_size % 2 != 1
    ):
        raise ValueError("landmark-region prototype grid/window configuration is invalid")
    selected_sizes = image_sizes.index_select(0, image_indices).to(dtype=torch.float32)
    columns = torch.floor(
        xy[:, 0] / selected_sizes[:, 0].clamp_min(1.0) * float(grid_size)
    ).to(dtype=torch.long).clamp(0, grid_size - 1)
    rows = torch.floor(
        xy[:, 1] / selected_sizes[:, 1].clamp_min(1.0) * float(grid_size)
    ).to(dtype=torch.long).clamp(0, grid_size - 1)
    offsets = torch.arange(
        -(window_size // 2), window_size // 2 + 1, device=image_grids.device
    )
    raw_rows = rows[:, None] + offsets[None, :]
    raw_columns = columns[:, None] + offsets[None, :]
    cell_valid = (
        (raw_rows[:, :, None] >= 0)
        & (raw_rows[:, :, None] < grid_size)
        & (raw_columns[:, None, :] >= 0)
        & (raw_columns[:, None, :] < grid_size)
    )
    safe_rows = raw_rows.clamp(0, grid_size - 1)
    safe_columns = raw_columns.clamp(0, grid_size - 1)
    selected_grids = image_grids.index_select(0, image_indices)
    batch = torch.arange(len(image_indices), device=image_grids.device)[:, None, None]
    crop = selected_grids[
        batch,
        safe_rows[:, :, None].expand(-1, window_size, window_size),
        safe_columns[:, None, :].expand(-1, window_size, window_size),
    ]

    def pooled(rows_for_region: torch.Tensor, columns_for_region: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = cell_valid[:, rows_for_region[:, None], columns_for_region[None, :]]
        values = crop[:, rows_for_region[:, None], columns_for_region[None, :], :]
        count = mask.sum(dim=(1, 2))
        mean = (values * mask[..., None].to(dtype=values.dtype)).sum(dim=(1, 2))
        mean = mean / count.clamp_min(1).to(dtype=mean.dtype)[:, None]
        return F.normalize(mean, p=2, dim=1, eps=1e-8), count > 0

    all_cells = torch.arange(window_size, device=image_grids.device)
    vectors: list[torch.Tensor] = []
    vector_valid: list[torch.Tensor] = []
    value, present = pooled(all_cells, all_cells)
    vectors.append(value)
    vector_valid.append(present)
    splits = torch.tensor_split(
        all_cells, int(LANDMARK_REGION_PROTOTYPE_REGION_GRID_SIZE)
    )
    for row_cells in splits:
        for column_cells in splits:
            value, present = pooled(row_cells, column_cells)
            vectors.append(value)
            vector_valid.append(present)
    return (
        torch.stack(vectors, dim=1),
        torch.stack(vector_valid, dim=1),
        cell_valid,
    )


def _query_region_prototypes_torch(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    query_cache_rows: np.ndarray,
    query_xy: np.ndarray,
    batch_size: int,
    grid_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prototypes: list[torch.Tensor] = []
    present: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for begin in range(0, len(query_cache_rows), int(batch_size)):
        end = min(begin + int(batch_size), len(query_cache_rows))
        values, valid, cell_valid = _crop_region_prototypes_torch(
            image_grids=image_grids,
            image_sizes=image_sizes,
            image_indices=torch.from_numpy(query_cache_rows[begin:end]).to(image_grids.device),
            xy=torch.from_numpy(query_xy[begin:end]).to(
                image_grids.device, dtype=torch.float32
            ),
            grid_size=int(grid_size),
            window_size=int(window_size),
        )
        prototypes.append(values)
        present.append(valid)
        masks.append(cell_valid)
    return torch.cat(prototypes, dim=0), torch.cat(present, dim=0), torch.cat(masks, dim=0)


def _compute_region_features_partition(
    *,
    device_name: str,
    image_grids_np: np.ndarray,
    image_sizes_np: np.ndarray,
    query_cache_rows: np.ndarray,
    query_xy: np.ndarray,
    edge_rows: np.ndarray,
    edge_support_cache_rows: np.ndarray,
    edge_support_xy: np.ndarray,
    begin: int,
    end: int,
    batch_size: int,
    profile: _RegionPrototypeProfile,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute one contiguous fixed edge range on one device."""

    started = time.monotonic()
    device = torch.device(device_name)
    with torch.no_grad():
        image_grids = torch.from_numpy(image_grids_np).to(
            device=device, dtype=torch.float32
        )
        image_sizes = torch.from_numpy(image_sizes_np).to(device=device, dtype=torch.float32)
        query_context = {
            int(window_size): _query_region_prototypes_torch(
                image_grids=image_grids,
                image_sizes=image_sizes,
                query_cache_rows=query_cache_rows,
                query_xy=query_xy,
                batch_size=int(batch_size),
                grid_size=int(profile.grid_size),
                window_size=int(window_size),
            )
            for window_size in profile.window_sizes
        }
        context = np.empty(
            (int(end) - int(begin), len(profile.context_feature_names)),
            dtype=np.float16,
        )
        for offset in range(int(begin), int(end), int(batch_size)):
            stop = min(offset + int(batch_size), int(end))
            query_rows = torch.from_numpy(edge_rows[offset:stop]).to(device=device)
            support_cache_rows = torch.from_numpy(edge_support_cache_rows[offset:stop]).to(
                device=device
            )
            support_xy = torch.from_numpy(edge_support_xy[offset:stop]).to(
                device=device, dtype=torch.float32
            )
            values: list[torch.Tensor] = []
            for window_size in profile.window_sizes:
                query_prototypes, query_present, query_cells = query_context[int(window_size)]
                support_prototypes, support_present, support_cells = (
                    _crop_region_prototypes_torch(
                    image_grids=image_grids,
                    image_sizes=image_sizes,
                    image_indices=support_cache_rows,
                    xy=support_xy,
                    grid_size=int(profile.grid_size),
                    window_size=int(window_size),
                    )
                )
                query_values = query_prototypes.index_select(0, query_rows)
                query_valid = query_present.index_select(0, query_rows)
                query_cell_valid = query_cells.index_select(0, query_rows)
                scores = torch.sum(query_values * support_prototypes, dim=2)
                scores = torch.where(
                    query_valid & support_present,
                    scores,
                    torch.full_like(scores, float("nan")),
                )
                common_fraction = (query_cell_valid & support_cells).to(torch.float32).mean(
                    dim=(1, 2)
                )
                values.append(torch.cat([scores, common_fraction[:, None]], dim=1))
            value = torch.cat(values, dim=1)
            context[offset - int(begin) : stop - int(begin)] = (
                value.cpu().numpy().astype(np.float16, copy=False)
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return context, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def build_landmark_region_prototype_candidate_probe_features(
    *,
    frozen_layout_features: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    output: Path,
    summary_json: Path,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("landmark-region prototype batch_size must be positive")
    selected_devices = tuple(str(value) for value in devices)
    if not selected_devices:
        raise ValueError("landmark-region prototype needs at least one device")
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite landmark-region prototype outputs")
    profile = _profile(profile_name)
    started = time.monotonic()
    layout, layout_metadata = _load_frozen_layout(Path(frozen_layout_features))
    maplet, maplet_metadata = _load_maplet_support_index(Path(maplet_support_index))
    if maplet_metadata.get("format") != MAPLET_FORMAT:
        raise ValueError("unsupported maplet support index format")
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("support geometry must use real SfM observation coordinates")
    radio = load_radio_final_context_pca_cache(Path(radio_final_context_cache))
    if radio.metadata.get("format") != RADIO_FINAL_CONTEXT_PCA_FORMAT:
        raise ValueError("unsupported RADIO-final region cache")
    if radio.metadata.get("pca_fit_scope") != "mapping_train_images_only":
        raise ValueError("RADIO-final region PCA was not fit on mapping-train images only")
    if bool(radio.metadata.get("image_retrieval_or_submap_used", True)) or bool(
        radio.metadata.get("pose_or_ground_truth_used", True)
    ):
        raise ValueError("RADIO-final region cache violates the target-free protocol")
    expected_radio_checkpoint = str(layout_metadata.get("radio_checkpoint_sha256", ""))
    actual_radio_checkpoint = str(radio.metadata.get("radio_checkpoint_sha256", ""))
    if expected_radio_checkpoint and expected_radio_checkpoint != actual_radio_checkpoint:
        raise ValueError(
            "frozen candidate layout and RADIO-final region cache use different "
            "RADIO checkpoints"
        )
    image_grid_descriptors = getattr(radio, f"grid{int(profile.grid_size)}_descriptors")
    if image_grid_descriptors is None or int(profile.grid_size) not in set(
        radio.metadata.get("spatial_grid_sizes", ())
    ):
        raise ValueError(
            "landmark-region prototype requires the RADIO-final "
            f"grid{int(profile.grid_size)} cache"
        )
    proposals_sha256 = str(layout_metadata.get("proposals_sha256", ""))
    if not proposals_sha256:
        raise ValueError("frozen layout lacks proposals_sha256 lineage")
    support_ids, view_valid, coverage = _fixed_support8_layout(layout=layout, maplet=maplet)
    edge = _fixed_candidate_view_edges(
        layout=layout,
        support_ids=support_ids,
        view_valid=view_valid,
        geometry=geometry,
        radio_image_ids=radio.image_ids,
    )
    query_cache_rows = _cache_image_indices(
        cache_image_ids=radio.image_ids,
        image_ids=np.asarray(layout["query_ids"]).astype(str),
        context="query",
    )
    candidate_anchor = _candidate_radio_final_anchor(layout)
    image_grids = np.asarray(image_grid_descriptors, dtype=np.float32).reshape(
        len(radio.image_ids),
        int(profile.grid_size),
        int(profile.grid_size),
        radio.descriptor_dim,
    )
    image_sizes = np.asarray(radio.image_sizes, dtype=np.float32)
    edge_count = len(edge["row"])
    partitions = [
        (
            edge_count * index // len(selected_devices),
            edge_count * (index + 1) // len(selected_devices),
        )
        for index in range(len(selected_devices))
    ]
    if any(end <= begin for begin, end in partitions):
        raise ValueError("more devices than fixed candidate-view edges")
    with ThreadPoolExecutor(max_workers=len(selected_devices)) as executor:
        futures = [
            executor.submit(
                _compute_region_features_partition,
                device_name=device_name,
                image_grids_np=image_grids,
                image_sizes_np=image_sizes,
                query_cache_rows=query_cache_rows,
                query_xy=np.asarray(layout["xy"], dtype=np.float32),
                edge_rows=edge["row"],
                edge_support_cache_rows=edge["support_cache_row"],
                edge_support_xy=edge["support_xy"],
                begin=begin,
                end=end,
                batch_size=int(batch_size),
                profile=profile,
            )
            for device_name, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    context = np.concatenate([value for value, _worker in computed], axis=0)
    workers = [worker for _value, worker in computed]
    if context.shape != (edge_count, len(profile.context_feature_names)):
        raise RuntimeError("landmark-region prototype context feature shape drifted")
    common_fraction_columns = np.asarray(
        [
            index
            for index, name in enumerate(profile.context_feature_names)
            if name.endswith("_common_cell_fraction")
        ],
        dtype=np.int64,
    )
    if (
        np.any(np.isinf(context))
        or np.any(~np.isfinite(context[:, 0]))
        or not len(common_fraction_columns)
        or np.any(~np.isfinite(context[:, common_fraction_columns]))
    ):
        raise RuntimeError("landmark-region prototype emitted invalid pooled or coverage features")
    features = np.full(
        (*view_valid.shape, len(profile.feature_names)),
        np.nan,
        dtype=np.float16,
    )
    flat_features = features.reshape(-1, features.shape[-1])
    flat_features[edge["flat_position"], 0] = candidate_anchor[
        edge["row"], edge["candidate"]
    ].astype(np.float16, copy=False)
    flat_features[edge["flat_position"], 1:] = context
    valid_features = features[view_valid]
    if np.any(np.isinf(valid_features)) or np.any(~np.isfinite(valid_features[:, 0])):
        raise RuntimeError("landmark-region prototype valid feature rows are invalid")
    metadata: dict[str, Any] = {
        "format": profile.artifact_format,
        "feature_definition": (
            "fixed_maplet_support8_per_view_radio_final_"
            f"grid{int(profile.grid_size)}_landmark_centered_region_prototypes_v1"
        ),
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "candidate_anchor_conditioned_spatial_grid_only": True,
        "render": False,
        "candidate_set": "frozen_global_top20_tracks",
        "support_view_selection": "fixed_maplet_coverage_rank_top8_v1",
        "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
        "region_prototype": {
            "profile": profile.name,
            "source": (
                "real_radio_final_"
                f"grid{int(profile.grid_size)}_landmark_centered_observation_crop"
            ),
            "grid_size": int(profile.grid_size),
            "window_sizes": [int(value) for value in profile.window_sizes],
            "region_grid_size": int(LANDMARK_REGION_PROTOTYPE_REGION_GRID_SIZE),
            "comparison": "pooled_region_descriptor_cosine_with_explicit_common_cell_fraction_v1",
            "view_aggregation": "none_before_learned_per_view_mixture",
            "support_coordinate_source": "sfm_observation_xy",
        },
        "descriptor_feature_names": list(profile.feature_names),
        "feature_dtype": "float16_with_nan_only_for_boundary-missing_spatial_bins",
        "source_frozen_layout": str(frozen_layout_features),
        "source_frozen_layout_sha256": file_sha256_short(frozen_layout_features),
        "frozen_layout_features_sha256": _array_sha256_short(layout["candidate_features"]),
        "full_frozen_source_rows_sha256": _array_sha256_short(layout["source_row_indices"]),
        "frozen_candidate_tracks_sha256": _array_sha256_short(layout["candidate_track_ids"]),
        "proposals_sha256": proposals_sha256,
        "maplet_support_index": str(maplet_support_index),
        "maplet_support_index_sha256": file_sha256_short(maplet_support_index),
        "maplet_support_index_format": maplet_metadata.get("format"),
        "support_geometry_index": str(support_geometry_index),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_index),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "radio_final_context_cache": str(radio_final_context_cache),
        "radio_final_context_cache_sha256": file_sha256_short(radio_final_context_cache),
        "radio_checkpoint_sha256": radio.metadata.get("radio_checkpoint_sha256"),
        "frozen_layout_radio_checkpoint_sha256": (
            expected_radio_checkpoint or None
        ),
        "radio_final_pca_fit_scope": radio.metadata.get("pca_fit_scope"),
        "radio_final_source_image_manifest_sha256": radio.metadata.get(
            "source_image_manifest_sha256"
        ),
        "candidate_top_k": int(np.asarray(layout["candidate_track_ids"]).shape[1]),
        "support_view_count": int(view_valid.shape[2]),
        "query_row_count": int(len(layout["source_row_indices"])),
        "query_image_count": int(len(set(np.asarray(layout["query_ids"]).astype(str).tolist()))),
        "support_image_count": int(len(set(support_ids[view_valid].tolist()))),
        "valid_candidate_view_count": int(np.sum(view_valid)),
        "is_complete_frozen_layout": True,
        "diagnostic_max_queries": 0,
        "diagnostic_max_rows": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    with temporary_output.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_row_indices=np.asarray(layout["source_row_indices"], dtype=np.int64),
            query_ids=np.asarray(layout["query_ids"]).astype(str),
            split_names=np.asarray(layout["split_names"]).astype(str),
            xy=np.asarray(layout["xy"], dtype=np.float32),
            candidate_track_ids=np.asarray(layout["candidate_track_ids"], dtype=np.int64),
            candidate_canonical_rows=np.asarray(layout["candidate_canonical_rows"], dtype=np.int64),
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_ids,
            candidate_support_coverage_counts=coverage,
            feature_names=np.asarray(profile.feature_names, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary_output.replace(output)
    summary = {
        "stage": "build_fixed_maplet_support8_landmark_region_prototype_features",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": profile.artifact_format,
        "row_count": int(len(layout["source_row_indices"])),
        "feature_count": int(len(profile.feature_names)),
        "support_view_count": int(view_valid.shape[2]),
        "valid_candidate_view_count": int(np.sum(view_valid)),
        "devices": list(selected_devices),
        "workers": workers,
        "runtime_seconds": float(time.monotonic() - started),
        "protocol": {
            "image_retrieval_or_submap_used": False,
            "hard_image_retrieval_or_candidate_reselection": False,
            "pose_or_ground_truth_used": False,
            "view_features_averaged_before_inference": False,
            "render": False,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_landmark_region_prototype_candidate_probe_features(
        frozen_layout_features=Path(args.frozen_layout_features),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        devices=_parse_devices(str(args.devices)),
        batch_size=int(args.batch_size),
        force=bool(args.force),
        profile_name=str(args.profile),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
