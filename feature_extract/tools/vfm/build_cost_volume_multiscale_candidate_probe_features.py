"""Export frozen per-view candidate cost-volume features for the S1c probe.

This exporter preserves the exact held-out query rows, global top-L tracks,
and real SfM support views from the frozen S1 layout.  It adds no pose target,
residual, render, image retrieval, or whole-image descriptor.  The only new
signal is candidate-anchor-referenced 2-D query/support appearance layout.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.build_multiscale_candidate_probe_features import (
    _ImageDescriptorIndex,
    _split_lookup,
    _validate_inputs,
)
from feature_extract.tools.vfm.build_structured_multiscale_candidate_probe_features import (
    _StructuredSupportAppearance,
    _StructuredSupportAppearanceCache,
    _array_sha256_short,
    _contiguous_layout_shard_positions,
    _load_frozen_layout,
    _query_context_summaries,
    _query_image_index,
    _validate_layout_lineage,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    COST_VOLUME_CONTEXT_GRID_SIZE,
    COST_VOLUME_CONTEXT_SCALE_NAMES,
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    WIDE_FULL_CORRELATION_CONTEXT_GRID_SIZE,
    WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    batched_wide_context_full_correlation_features,
    batched_context_cost_volume_hough_features,
    cosine_similarity,
    cost_volume_scale_feature_slices,
    crop_spatial_grid_context,
    resample_context_descriptor_grid,
    wide_full_correlation_scale_feature_slices,
)


ARTIFACT_FORMAT = "cost_volume_multiscale_candidate_probe_features_v1"
WIDE_FULL_CORRELATION_ARTIFACT_FORMAT = (
    "wide_full_correlation_multiscale_candidate_probe_features_v1"
)


@dataclass(frozen=True)
class _CostVolumeProfile:
    name: str
    artifact_format: str
    feature_definition: str
    feature_names: tuple[str, ...]
    scale_names: tuple[str, ...]
    scale_sources: tuple[str, ...]
    context_grid_size: int
    scale_feature_slices: Mapping[str, slice]
    hough_grid_size: int | None


def _cost_volume_profile(value: str) -> _CostVolumeProfile:
    """Resolve an immutable S1 feature schema without changing S1c fields."""

    name = str(value).strip()
    if name == "hough3":
        return _CostVolumeProfile(
            name=name,
            artifact_format=ARTIFACT_FORMAT,
            feature_definition=(
                "per_view_candidate_specific_real_image_anchor_referenced_hough_cost_volume_v1"
            ),
            feature_names=COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
            scale_names=COST_VOLUME_CONTEXT_SCALE_NAMES,
            scale_sources=("final_window5", "final_window7", "intermediate11", "alike11"),
            context_grid_size=int(COST_VOLUME_CONTEXT_GRID_SIZE),
            scale_feature_slices=cost_volume_scale_feature_slices(),
            hough_grid_size=int(2 * COST_VOLUME_CONTEXT_GRID_SIZE - 1),
        )
    if name == "wide_fullcorr5":
        return _CostVolumeProfile(
            name=name,
            artifact_format=WIDE_FULL_CORRELATION_ARTIFACT_FORMAT,
            feature_definition=(
                "per_view_candidate_specific_real_image_anchor_referenced_"
                "wide_full_correlation_v1"
            ),
            feature_names=WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
            scale_names=WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES,
            scale_sources=("final_window7", "intermediate11", "alike11"),
            context_grid_size=int(WIDE_FULL_CORRELATION_CONTEXT_GRID_SIZE),
            scale_feature_slices=wide_full_correlation_scale_feature_slices(),
            hough_grid_size=None,
        )
    raise ValueError(f"unsupported cost-volume profile {value!r}")


def _resolve_materialized_scales(
    value: str, *, scale_names: Sequence[str] = COST_VOLUME_CONTEXT_SCALE_NAMES
) -> tuple[str, ...]:
    if str(value).strip() == "all":
        return tuple(str(name) for name in scale_names)
    requested = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("materialized cost-volume scales must be unique and non-empty")
    unknown = set(requested) - set(str(name) for name in scale_names)
    if unknown:
        raise ValueError(f"unsupported materialized cost-volume scales: {sorted(unknown)}")
    return requested


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--query_context_cache", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--radio_intermediate_cache", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--radius7_px", type=float, default=144.0)
    parser.add_argument("--radius11_px", type=float, default=288.0)
    parser.add_argument("--max_context_nodes", type=int, default=128)
    parser.add_argument("--duplicate_radius_px", type=float, default=2.0)
    parser.add_argument("--support_context_cache_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--cost_volume_temperature", type=float, default=0.07)
    parser.add_argument(
        "--profile",
        choices=("hough3", "wide_fullcorr5"),
        default="hough3",
        help=(
            "frozen evidence schema; wide_fullcorr5 retains every 5x5 query/support "
            "cell correlation and never changes the hough3 layout"
        ),
    )
    parser.add_argument(
        "--materialized_scales",
        default="all",
        help=(
            "all or a comma-separated subset of declared cost-volume scales; "
            "fit rejects any family that references an unmaterialized scale"
        ),
    )
    parser.add_argument(
        "--layout_shard_count",
        type=int,
        default=1,
        help="contiguous frozen-layout shard count; use the strict merger before fitting",
    )
    parser.add_argument("--layout_shard_index", type=int, default=0)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="diagnostic-only prefix limit; production artifacts use zero",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _support_cost_grids(
    appearance: _StructuredSupportAppearance,
    *,
    cache: OrderedDict[tuple[int, str], tuple[tuple[np.ndarray, np.ndarray], ...]],
    cache_key: tuple[int, str],
    max_entries: int,
    profile: _CostVolumeProfile,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    cached = cache.get(cache_key)
    if cached is not None:
        cache.move_to_end(cache_key)
        return cached
    value = tuple(
        resample_context_descriptor_grid(
            getattr(appearance, source), grid_size=int(profile.context_grid_size)
        )
        for source in profile.scale_sources
    )
    cache[cache_key] = value
    if len(cache) > int(max_entries):
        cache.popitem(last=False)
    return value


def _final_cost_grids(
    *,
    final_context: object,
    image_id: str,
    anchor_xy: np.ndarray,
    profile: _CostVolumeProfile,
    materialized_scale_names: set[str],
) -> tuple[tuple[np.ndarray, np.ndarray] | None, ...]:
    """Build only RADIO-final scale contexts when local grids are unnecessary."""

    final_grid, image_size = final_context.image_grid_descriptors(image_id, grid_size=8)
    output: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(profile.scale_names)
    for index, (scale, source) in enumerate(
        zip(profile.scale_names, profile.scale_sources)
    ):
        if scale not in materialized_scale_names:
            continue
        if source == "final_window5":
            window_size = 5
        elif source == "final_window7":
            window_size = 7
        else:
            continue
        output[index] = resample_context_descriptor_grid(
            crop_spatial_grid_context(
                final_grid,
                image_size=image_size,
                xy=np.asarray(anchor_xy, dtype=np.float32),
                window_size=window_size,
            ),
            grid_size=int(profile.context_grid_size),
        )
    return tuple(output)


def _flush_batch(
    *,
    features: np.ndarray,
    positions: list[tuple[int, int, int]],
    anchors: list[np.ndarray],
    query_grids: list[list[np.ndarray]],
    query_valid: list[list[np.ndarray]],
    support_grids: list[list[np.ndarray]],
    support_valid: list[list[np.ndarray]],
    materialized_scale_indices: tuple[int, ...],
    device: torch.device,
    temperature: float,
    profile: _CostVolumeProfile,
) -> None:
    if not positions:
        return
    with torch.no_grad():
        values = np.full(
            (
                len(positions),
                len(profile.feature_names),
            ),
            np.nan,
            dtype=np.float32,
        )
        values[:, :3] = np.stack(anchors, axis=0).astype(np.float32, copy=False)
        scale_slices = profile.scale_feature_slices
        for scale_index in materialized_scale_indices:
            scale_name = profile.scale_names[int(scale_index)]
            query = torch.as_tensor(
                np.stack(query_grids[scale_index], axis=0),
                dtype=torch.float32,
                device=device,
            )
            query_mask = torch.as_tensor(
                np.stack(query_valid[scale_index], axis=0), dtype=torch.bool, device=device
            )
            support = torch.as_tensor(
                np.stack(support_grids[scale_index], axis=0),
                dtype=torch.float32,
                device=device,
            )
            support_mask = torch.as_tensor(
                np.stack(support_valid[scale_index], axis=0), dtype=torch.bool, device=device
            )
            if profile.name == "hough3":
                encoded = batched_context_cost_volume_hough_features(
                    query, query_mask, support, support_mask, temperature=float(temperature)
                )
            elif profile.name == "wide_fullcorr5":
                encoded = batched_wide_context_full_correlation_features(
                    query, query_mask, support, support_mask
                )
            else:
                raise RuntimeError(f"unsupported cost-volume profile {profile.name!r}")
            values[:, scale_slices[scale_name]] = (
                encoded.cpu().numpy().astype(np.float32, copy=False)
            )
    expected_shape = (
        len(positions),
        len(profile.feature_names),
    )
    materialized_columns = np.concatenate(
        [
            np.arange(3, dtype=np.int64),
            *(
                np.arange(
                    profile.scale_feature_slices[profile.scale_names[int(index)]].start,
                    profile.scale_feature_slices[profile.scale_names[int(index)]].stop,
                    dtype=np.int64,
                )
                for index in materialized_scale_indices
            ),
        ]
    )
    if values.shape != expected_shape or not np.isfinite(values[:, materialized_columns]).all():
        raise RuntimeError("batched cost-volume output is invalid")
    for value, (row, candidate, view) in zip(values, positions):
        features[int(row), int(candidate), int(view)] = value.astype(np.float16)
    positions.clear()
    anchors.clear()
    for collection in (query_grids, query_valid, support_grids, support_valid):
        for values_for_scale in collection:
            values_for_scale.clear()


def build_cost_volume_multiscale_candidate_probe_features(
    *,
    frozen_layout_path: Path,
    proposals_path: Path,
    detector_path: Path,
    candidate_path: Path,
    query_context_path: Path,
    support_feature_path: Path,
    support_geometry_path: Path,
    bank_path: Path,
    maplet_path: Path,
    radio_path: Path,
    final_context_path: Path,
    split_json_path: Path,
    output_path: Path,
    summary_path: Path,
    radius7_px: float,
    radius11_px: float,
    max_context_nodes: int,
    duplicate_radius_px: float,
    support_context_cache_size: int,
    device: str,
    batch_size: int,
    cost_volume_temperature: float,
    materialized_scales: tuple[str, ...],
    layout_shard_count: int = 1,
    layout_shard_index: int = 0,
    max_rows: int = 0,
    force: bool = False,
    profile: str = "hough3",
) -> dict[str, object]:
    """Build GPU-batched, target-free cost-volume evidence on a frozen layout."""

    schema = _cost_volume_profile(profile)
    if (
        float(radius7_px) <= 0.0
        or float(radius11_px) < float(radius7_px)
        or int(max_context_nodes) <= 0
        or float(duplicate_radius_px) < 0.0
        or int(support_context_cache_size) <= 0
        or int(batch_size) <= 0
        or float(cost_volume_temperature) <= 0.0
        or int(max_rows) < 0
        or int(layout_shard_count) <= 0
        or not 0 <= int(layout_shard_index) < int(layout_shard_count)
    ):
        raise ValueError("cost-volume export parameters are invalid")
    if (
        not materialized_scales
        or len(set(materialized_scales)) != len(materialized_scales)
        or set(materialized_scales) - set(schema.scale_names)
    ):
        raise ValueError("cost-volume materialized scales are invalid")
    materialized_scale_indices = tuple(
        schema.scale_names.index(name) for name in materialized_scales
    )
    materialized_scale_set = set(materialized_scales)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("a CUDA cost-volume export was requested but CUDA is unavailable")
    if output_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if summary_path.exists() and not bool(force):
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    started = time.time()
    layout, layout_metadata = _load_frozen_layout(frozen_layout_path)
    (
        proposals,
        detector,
        _fit_rows,
        query_context,
        support_alike,
        support_scores,
        geometry,
        radio,
        final_context,
        landmark_bank,
        _maplet,
        input_metadata,
    ) = _validate_inputs(
        proposal_path=proposals_path,
        detector_path=detector_path,
        candidate_path=candidate_path,
        query_context_path=query_context_path,
        support_feature_path=support_feature_path,
        support_geometry_path=support_geometry_path,
        bank_path=bank_path,
        maplet_path=maplet_path,
        radio_path=radio_path,
        final_context_path=final_context_path,
    )
    split_by_query = _split_lookup(json.loads(Path(split_json_path).read_text()))
    _validate_layout_lineage(
        layout=layout,
        layout_metadata=layout_metadata,
        layout_path=frozen_layout_path,
        proposals=proposals,
        detector=detector,
        landmark_bank=landmark_bank,
        split_by_query=split_by_query,
        expected_hashes={
            "proposals_sha256": file_sha256_short(proposals_path),
            "detector_query_cache_sha256": file_sha256_short(detector_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "query_context_cache_sha256": file_sha256_short(query_context_path),
            "support_feature_cache_sha256": file_sha256_short(support_feature_path),
            "support_geometry_index_sha256": file_sha256_short(support_geometry_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "maplet_support_index_sha256": file_sha256_short(maplet_path),
            "radio_intermediate_cache_sha256": file_sha256_short(radio_path),
        },
    )
    if final_context.grid8_descriptors is None:
        raise ValueError("cost-volume S1c requires RADIO-final grid8 descriptors")
    if 8 not in set(final_context.metadata.get("spatial_grid_sizes", ())):
        raise ValueError("RADIO-final cache manifest does not declare grid8")
    final_image_ids = set(final_context.image_ids.tolist())
    if not set(np.asarray(layout["query_ids"]).astype(str)).issubset(final_image_ids):
        raise ValueError("RADIO-final grid8 cache misses a frozen layout query image")
    valid_support_ids = np.asarray(layout["candidate_support_image_ids"]).astype(str)[
        np.asarray(layout["candidate_view_valid"], dtype=bool)
    ]
    if not set(valid_support_ids.tolist()).issubset(final_image_ids):
        raise ValueError("RADIO-final grid8 cache misses a frozen layout support image")

    full_layout_row_count = int(len(layout["source_row_indices"]))
    full_source_rows_sha256 = _array_sha256_short(layout["source_row_indices"])
    full_candidate_tracks_sha256 = _array_sha256_short(layout["candidate_track_ids"])
    full_support_view_ids_sha256 = _array_sha256_short(
        layout["candidate_support_image_ids"]
    )
    layout_positions = _contiguous_layout_shard_positions(
        full_layout_row_count,
        shard_count=int(layout_shard_count),
        shard_index=int(layout_shard_index),
    )
    layout = {key: np.asarray(value)[layout_positions] for key, value in layout.items()}
    if int(max_rows) > 0:
        limit = min(int(max_rows), len(layout_positions))
        layout_positions = layout_positions[:limit]
        layout = {key: np.asarray(value)[:limit] for key, value in layout.items()}

    rows = np.asarray(layout["source_row_indices"], dtype=np.int64)
    tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    view_valid = np.asarray(layout["candidate_view_valid"], dtype=bool)
    support_ids = np.asarray(layout["candidate_support_image_ids"]).astype(str)
    row_count, candidate_count = tracks.shape
    view_count = int(view_valid.shape[2])
    features = np.full(
        (
            row_count,
            candidate_count,
            view_count,
            len(schema.feature_names),
        ),
        np.nan,
        dtype=np.float16,
    )
    support_cache = _StructuredSupportAppearanceCache(
        geometry=geometry,
        support_alike=support_alike,
        support_scores=support_scores,
        radio=radio,
        final_context=final_context,
        radius7_px=float(radius7_px),
        radius11_px=float(radius11_px),
        max_context_nodes=int(max_context_nodes),
        duplicate_radius_px=float(duplicate_radius_px),
        max_entries=int(support_context_cache_size),
    )
    support_grid_cache: OrderedDict[
        tuple[int, str], tuple[tuple[np.ndarray, np.ndarray], ...]
    ] = OrderedDict()
    query_indices: dict[str, _ImageDescriptorIndex] = {}
    needs_local_context = any(
        scale in materialized_scale_set and source in {"intermediate11", "alike11"}
        for scale, source in zip(schema.scale_names, schema.scale_sources)
    )
    positions: list[tuple[int, int, int]] = []
    anchors: list[np.ndarray] = []
    query_grids: list[list[np.ndarray]] = [[] for _ in schema.scale_names]
    query_valid: list[list[np.ndarray]] = [[] for _ in schema.scale_names]
    support_grids: list[list[np.ndarray]] = [[] for _ in schema.scale_names]
    support_valid: list[list[np.ndarray]] = [[] for _ in schema.scale_names]
    processed = 0
    for output_row, source_row in enumerate(rows.tolist()):
        query_id = str(layout["query_ids"][output_row])
        query_anchor_xy = np.asarray(detector["xy"], dtype=np.float32)[int(source_row)]
        query_alike_anchor = np.asarray(
            detector["local_descriptors"], dtype=np.float32
        )[int(source_row)]
        query_intermediate_anchor = np.asarray(
            radio.query_anchor_descriptors[int(source_row)], dtype=np.float32
        )
        query_cost: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(
            schema.scale_names
        )
        if needs_local_context:
            query_nodes = query_indices.get(query_id)
            if query_nodes is None:
                query_nodes = _query_image_index(
                    query_id=query_id,
                    detector=detector,
                    query_context=query_context,
                    radio=radio,
                )
                query_indices[query_id] = query_nodes
            (
                _query_anchor_xy,
                _query_alike_anchor,
                _query_intermediate_anchor,
                _query_intermediate7,
                query_intermediate11,
                _query_alike7,
                query_alike11,
                _query_final_window3,
                query_final_window5,
                query_final_window7,
            ) = _query_context_summaries(
                nodes=query_nodes,
                detector=detector,
                radio=radio,
                final_context=final_context,
                query_id=query_id,
                source_row=int(source_row),
                radius7_px=float(radius7_px),
                radius11_px=float(radius11_px),
                max_context_nodes=int(max_context_nodes),
                duplicate_radius_px=float(duplicate_radius_px),
            )
            summaries = {
                "final_window5": query_final_window5,
                "final_window7": query_final_window7,
                "intermediate11": query_intermediate11,
                "alike11": query_alike11,
            }
            for index, (scale, source) in enumerate(
                zip(schema.scale_names, schema.scale_sources)
            ):
                if scale in materialized_scale_set:
                    query_cost[index] = resample_context_descriptor_grid(
                        summaries[source], grid_size=int(schema.context_grid_size)
                    )
        else:
            query_cost = list(
                _final_cost_grids(
                    final_context=final_context,
                    image_id=query_id,
                    anchor_xy=query_anchor_xy,
                    profile=schema,
                    materialized_scale_names=materialized_scale_set,
                )
            )
        for candidate_column, track_id in enumerate(tracks[output_row].tolist()):
            if int(track_id) < 0:
                continue
            canonical_row = int(layout["candidate_canonical_rows"][output_row, candidate_column])
            final_anchor = cosine_similarity(
                np.asarray(detector["global_descriptors"], dtype=np.float32)[int(source_row)],
                landmark_bank.features[canonical_row],
            )
            if not np.isfinite(final_anchor):
                raise ValueError("cost-volume RADIO-final anchor similarity is invalid")
            for view_column in range(view_count):
                if not bool(view_valid[output_row, candidate_column, view_column]):
                    continue
                image_id = str(support_ids[output_row, candidate_column, view_column])
                cache_key = (int(track_id), image_id)
                if needs_local_context:
                    appearance = support_cache.get(
                        track_id=int(track_id), image_id=image_id
                    )
                    support_cost = _support_cost_grids(
                        appearance,
                        cache=support_grid_cache,
                        cache_key=cache_key,
                        max_entries=int(support_context_cache_size),
                        profile=schema,
                    )
                    support_intermediate_anchor = appearance.intermediate_anchor
                    support_alike_anchor = appearance.alike_anchor
                else:
                    support_anchor_xy, support_intermediate_anchor, support_alike_anchor = (
                        support_cache.anchor_observation(
                            track_id=int(track_id), image_id=image_id
                        )
                    )
                    support_cost = support_grid_cache.get(cache_key)
                    if support_cost is None:
                        support_cost = _final_cost_grids(
                            final_context=final_context,
                            image_id=image_id,
                            anchor_xy=support_anchor_xy,
                            profile=schema,
                            materialized_scale_names=materialized_scale_set,
                        )
                        support_grid_cache[cache_key] = support_cost
                        if len(support_grid_cache) > int(support_context_cache_size):
                            support_grid_cache.popitem(last=False)
                anchor = np.asarray(
                    [
                        float(final_anchor),
                        cosine_similarity(
                            query_intermediate_anchor, support_intermediate_anchor
                        ),
                        cosine_similarity(query_alike_anchor, support_alike_anchor),
                    ],
                    dtype=np.float32,
                )
                if not np.isfinite(anchor).all():
                    raise RuntimeError("cost-volume anchor vector is invalid")
                positions.append((output_row, candidate_column, view_column))
                anchors.append(anchor)
                for scale_index in materialized_scale_indices:
                    query_grid, query_mask = query_cost[scale_index]
                    support_grid, support_mask = support_cost[scale_index]
                    query_grids[scale_index].append(query_grid)
                    query_valid[scale_index].append(query_mask)
                    support_grids[scale_index].append(support_grid)
                    support_valid[scale_index].append(support_mask)
                if len(positions) >= int(batch_size):
                    _flush_batch(
                        features=features,
                        positions=positions,
                        anchors=anchors,
                        query_grids=query_grids,
                        query_valid=query_valid,
                        support_grids=support_grids,
                        support_valid=support_valid,
                        materialized_scale_indices=materialized_scale_indices,
                        device=target,
                        temperature=float(cost_volume_temperature),
                        profile=schema,
                    )
        processed += 1
        if processed % 256 == 0 or processed == row_count:
            elapsed = max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "progress_rows": processed,
                        "row_count": row_count,
                        "elapsed_seconds": round(elapsed, 1),
                        "rows_per_second": round(processed / elapsed, 2),
                        "valid_candidate_views": int(np.sum(view_valid[:processed])),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _flush_batch(
        features=features,
        positions=positions,
        anchors=anchors,
        query_grids=query_grids,
        query_valid=query_valid,
        support_grids=support_grids,
        support_valid=support_valid,
        materialized_scale_indices=materialized_scale_indices,
        device=target,
        temperature=float(cost_volume_temperature),
        profile=schema,
    )
    valid_feature_rows = features[view_valid]
    materialized_columns = np.concatenate(
        [
            np.arange(3, dtype=np.int64),
            *(
                np.arange(
                    schema.scale_feature_slices[scale].start,
                    schema.scale_feature_slices[scale].stop,
                    dtype=np.int64,
                )
                for scale in materialized_scales
            ),
        ]
    )
    if np.any(~np.isfinite(valid_feature_rows[:, materialized_columns])):
        raise RuntimeError("cost-volume valid feature tensor is non-finite")
    metadata = {
        "format": schema.artifact_format,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "candidate_anchor_conditioned_spatial_grid_only": True,
        "feature_definition": schema.feature_definition,
        "descriptor_feature_names": list(schema.feature_names),
        "feature_dtype": "float16_finite_for_materialized_candidate_support_view_fields",
        "source_row_selection": layout_metadata["source_row_selection"],
        "verification_point_count": int(layout_metadata["verification_point_count"]),
        "support_view_selection": layout_metadata["support_view_selection"],
        "candidate_fit_rows_sha256": layout_metadata.get("candidate_fit_rows_sha256"),
        "detector_log_merit_weight": layout_metadata.get("detector_log_merit_weight"),
        "frozen_layout_features": str(frozen_layout_path),
        "frozen_layout_features_sha256": file_sha256_short(frozen_layout_path),
        "frozen_source_rows_sha256": _array_sha256_short(rows),
        "frozen_candidate_tracks_sha256": _array_sha256_short(tracks),
        "frozen_support_view_ids_sha256": _array_sha256_short(support_ids),
        "full_frozen_layout_row_count": int(full_layout_row_count),
        "full_frozen_source_rows_sha256": full_source_rows_sha256,
        "full_frozen_candidate_tracks_sha256": full_candidate_tracks_sha256,
        "full_frozen_support_view_ids_sha256": full_support_view_ids_sha256,
        "layout_shard_count": int(layout_shard_count),
        "layout_shard_index": int(layout_shard_index),
        "layout_position_count": int(len(layout_positions)),
        "is_complete_frozen_layout": bool(int(layout_shard_count) == 1 and int(max_rows) == 0),
        "proposals_sha256": file_sha256_short(proposals_path),
        "detector_query_cache_sha256": file_sha256_short(detector_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "query_context_cache_sha256": file_sha256_short(query_context_path),
        "support_feature_cache_sha256": file_sha256_short(support_feature_path),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "maplet_support_index_sha256": file_sha256_short(maplet_path),
        "radio_intermediate_cache_sha256": file_sha256_short(radio_path),
        "radio_final_context_cache_sha256": file_sha256_short(final_context_path),
        "split_json_sha256": file_sha256_short(split_json_path),
        "descriptor_space_id": input_metadata["bank_metadata"].get("descriptor_space_id"),
        "radio_checkpoint_sha256": radio.metadata.get("radio_checkpoint_sha256"),
        "alike_checkpoint_sha256": input_metadata["detector_metadata"].get("alike_checkpoint_sha256"),
        "candidate_top_k": int(candidate_count),
        "support_view_count": int(view_count),
        "query_row_count": int(row_count),
        "diagnostic_max_rows": int(max_rows),
        "query_image_count": int(len(set(np.asarray(layout["query_ids"]).astype(str).tolist()))),
        "split_row_counts": {
            split: int(np.sum(np.asarray(layout["split_names"]).astype(str) == split))
            for split in ("train", "validation", "test")
        },
        "cost_volume": {
            "profile": schema.name,
            "context_grid_size": int(schema.context_grid_size),
            "hough_grid_size": schema.hough_grid_size,
            "temperature": (
                float(cost_volume_temperature) if schema.name == "hough3" else None
            ),
            "scale_names": list(schema.scale_names),
            "materialized_scale_names": list(materialized_scales),
            "representation": (
                "hough_translation_summary_v1"
                if schema.name == "hough3"
                else "unpooled_query_cell_by_support_cell_cosine_with_explicit_masks_v1"
            ),
            "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
        },
        "input_protocol": {
            "frozen_candidate_layout": "heldout_query_rows_fixed_global_topl_fixed_support_views_v1",
            "final_pca_fit_scope": final_context.metadata.get("pca_fit_scope"),
            "intermediate_projection": radio.metadata.get("projection"),
            "intermediate_support_descriptor_source": radio.metadata.get("support_descriptor_source"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_row_indices=rows,
            layout_positions=layout_positions,
            query_ids=np.asarray(layout["query_ids"]).astype(str),
            split_names=np.asarray(layout["split_names"]).astype(str),
            xy=np.asarray(layout["xy"], dtype=np.float32),
            candidate_track_ids=tracks,
            candidate_canonical_rows=np.asarray(layout["candidate_canonical_rows"], dtype=np.int64),
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_ids,
            candidate_support_coverage_counts=np.asarray(
                layout["candidate_support_coverage_counts"], dtype=np.int32
            ),
            feature_names=np.asarray(
                schema.feature_names, dtype=np.str_
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    temporary_path.replace(output_path)
    summary = {
        "stage": "frozen_cost_volume_multiscale_candidate_specific_appearance_features",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "metadata": metadata,
        "export_audit": {
            "valid_candidate_view_count": int(np.sum(view_valid)),
            "support_cache": dict(support_cache.stats),
            "support_cost_grid_cache_entries": int(len(support_grid_cache)),
            "runtime_seconds": float(time.time() - started),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    schema = _cost_volume_profile(str(args.profile))
    summary = build_cost_volume_multiscale_candidate_probe_features(
        frozen_layout_path=Path(args.frozen_layout_features),
        proposals_path=Path(args.proposals),
        detector_path=Path(args.detector_query_cache),
        candidate_path=Path(args.candidate_artifact),
        query_context_path=Path(args.query_context_cache),
        support_feature_path=Path(args.support_feature_cache),
        support_geometry_path=Path(args.support_geometry_index),
        bank_path=Path(args.projected_landmark_bank),
        maplet_path=Path(args.maplet_support_index),
        radio_path=Path(args.radio_intermediate_cache),
        final_context_path=Path(args.radio_final_context_cache),
        split_json_path=Path(args.split_json),
        output_path=Path(args.output),
        summary_path=Path(args.summary_json),
        radius7_px=float(args.radius7_px),
        radius11_px=float(args.radius11_px),
        max_context_nodes=int(args.max_context_nodes),
        duplicate_radius_px=float(args.duplicate_radius_px),
        support_context_cache_size=int(args.support_context_cache_size),
        device=str(args.device),
        batch_size=int(args.batch_size),
        cost_volume_temperature=float(args.cost_volume_temperature),
        materialized_scales=_resolve_materialized_scales(
            str(args.materialized_scales), scale_names=schema.scale_names
        ),
        layout_shard_count=int(args.layout_shard_count),
        layout_shard_index=int(args.layout_shard_index),
        max_rows=int(args.max_rows),
        force=bool(args.force),
        profile=str(args.profile),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
