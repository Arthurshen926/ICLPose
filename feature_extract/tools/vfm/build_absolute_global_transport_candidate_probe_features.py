"""Export frozen full-image transport evidence for candidate-specific probes.

The exporter follows a fixed query/candidate/support-view layout.  It uses no
pose hypothesis, target label, image retrieval, or candidate reselection.  A
central 3x3 descriptor neighbourhood around both frozen anchors is masked
before descriptor comparison, so the visual family can only use context from
the remaining full-image 16x16 lattice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    CONTEXT_ATTENTION_CENTER_MASK_RADIUS,
    ContextAttentionSource,
    anchor_grid_descriptors,
    build_fixed_candidate_context_runtime,
    image_xy_to_grid_indices,
    load_absolute_grid_context_sources,
    load_context_attention_frozen_layout,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES,
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT,
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES,
    ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES,
    ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE,
    ABSOLUTE_GLOBAL_TRANSPORT_SCALE_NAMES,
    batched_absolute_global_transport_features,
    batched_absolute_global_transport_position_features,
)


ARTIFACT_FORMAT = ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT
_GRID_SIZE = 16
_SOURCE_NAME_BY_SCALE = {
    "radio_final_grid16": "radio_final",
    "radio_intermediate_grid16": "radio_intermediate",
    "alike_grid16": "alike",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch_size", type=int, default=256)
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


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _full_grid_context_mask(
    *,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Keep all full-image cells except the fixed central anchor neighbourhood."""

    rows, columns = image_xy_to_grid_indices(
        image_sizes=image_sizes,
        image_indices=image_indices,
        xy=xy,
        grid_size=int(grid_size),
    )
    coordinates = torch.arange(int(grid_size), device=image_sizes.device)
    grid_rows, grid_columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius = int(CONTEXT_ATTENTION_CENTER_MASK_RADIUS)
    excluded = (
        (grid_rows[None] - rows[:, None, None]).abs() <= radius
    ) & ((grid_columns[None] - columns[:, None, None]).abs() <= radius)
    return ~excluded


def _source_by_name(sources: Sequence[ContextAttentionSource]) -> dict[str, ContextAttentionSource]:
    native = {str(source.name): source for source in sources}
    if set(native) != set(_SOURCE_NAME_BY_SCALE.values()):
        raise ValueError("absolute global-transport sources do not match the declared scales")
    output = {
        scale_name: native[source_name]
        for scale_name, source_name in _SOURCE_NAME_BY_SCALE.items()
    }
    if any(source.spatial_grid_size != _GRID_SIZE for source in output.values()):
        raise ValueError("absolute global-transport source grid sizes differ from grid16")
    return output


def _compute_partition(
    *,
    device_name: str,
    sources: Sequence[ContextAttentionSource],
    query_image_indices: np.ndarray,
    query_xy: np.ndarray,
    edge_rows: np.ndarray,
    edge_support_image_indices: np.ndarray,
    edge_support_xy: np.ndarray,
    begin: int,
    end: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Compute a disjoint fixed edge range on one device."""

    started = time.monotonic()
    device = torch.device(device_name)
    source_map = _source_by_name(sources)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad():
        grids = {
            name: torch.from_numpy(np.asarray(source.grid)).to(device=device, dtype=torch.float16)
            for name, source in source_map.items()
        }
        sizes = torch.from_numpy(np.asarray(sources[0].image_sizes, dtype=np.int64)).to(
            device=device, dtype=torch.float32
        )
        edge_count = int(end) - int(begin)
        context_output = np.empty(
            (edge_count, len(ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES)),
            dtype=np.float16,
        )
        position_output = np.empty(
            (edge_count, len(ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES)),
            dtype=np.float16,
        )
        anchor_output = np.empty((edge_count,), dtype=np.float16)
        progress_interval = max(int(batch_size) * 64, int(batch_size))
        for offset in range(int(begin), int(end), int(batch_size)):
            stop = min(offset + int(batch_size), int(end))
            rows = edge_rows[offset:stop]
            query_indices = torch.from_numpy(query_image_indices[rows]).to(device=device)
            support_indices = torch.from_numpy(edge_support_image_indices[offset:stop]).to(
                device=device
            )
            query_coordinates = torch.from_numpy(query_xy[rows]).to(
                device=device, dtype=torch.float32
            )
            support_coordinates = torch.from_numpy(edge_support_xy[offset:stop]).to(
                device=device, dtype=torch.float32
            )
            query_mask = _full_grid_context_mask(
                image_sizes=sizes,
                image_indices=query_indices,
                xy=query_coordinates,
                grid_size=_GRID_SIZE,
            )
            support_mask = _full_grid_context_mask(
                image_sizes=sizes,
                image_indices=support_indices,
                xy=support_coordinates,
                grid_size=_GRID_SIZE,
            )
            per_scale = []
            for scale_name in ABSOLUTE_GLOBAL_TRANSPORT_SCALE_NAMES:
                grid = grids[scale_name]
                per_scale.append(
                    batched_absolute_global_transport_features(
                        grid.index_select(0, query_indices),
                        query_mask,
                        grid.index_select(0, support_indices),
                        support_mask,
                        region_grid_size=ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE,
                    )
                )
            context = torch.cat(per_scale, dim=1)
            position = batched_absolute_global_transport_position_features(
                query_mask,
                support_mask,
                region_grid_size=ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE,
            )
            final_grid = grids["radio_final_grid16"]
            query_anchor = anchor_grid_descriptors(
                image_grids=final_grid,
                image_sizes=sizes,
                image_indices=query_indices,
                xy=query_coordinates,
            )
            support_anchor = anchor_grid_descriptors(
                image_grids=final_grid,
                image_sizes=sizes,
                image_indices=support_indices,
                xy=support_coordinates,
            )
            anchor = F.cosine_similarity(
                query_anchor.to(dtype=torch.float32),
                support_anchor.to(dtype=torch.float32),
                dim=1,
            )
            local_begin = offset - int(begin)
            local_end = stop - int(begin)
            context_output[local_begin:local_end] = context.cpu().numpy().astype(
                np.float16, copy=False
            )
            position_output[local_begin:local_end] = position.cpu().numpy().astype(
                np.float16, copy=False
            )
            anchor_output[local_begin:local_end] = anchor.cpu().numpy().astype(
                np.float16, copy=False
            )
            completed = int(stop) - int(begin)
            if completed % progress_interval == 0 or int(stop) == int(end):
                print(
                    json.dumps(
                        {
                            "stage": "absolute_global_transport",
                            "device": str(device),
                            "completed": completed,
                            "assigned": edge_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if (
        np.any(~np.isfinite(context_output))
        or np.any(~np.isfinite(position_output))
        or np.any(~np.isfinite(anchor_output))
    ):
        raise RuntimeError("absolute global-transport export produced non-finite values")
    return context_output, position_output, anchor_output, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def build_absolute_global_transport_candidate_probe_features(
    *,
    frozen_layout_features: Path,
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
    """Build a complete no-retrieval feature artifact from frozen inputs."""

    if int(batch_size) <= 0:
        raise ValueError("absolute global-transport batch_size must be positive")
    selected_devices = tuple(str(device) for device in devices)
    if not selected_devices:
        raise ValueError("absolute global-transport needs at least one device")
    if (Path(output).exists() or Path(summary_json).exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite absolute global-transport outputs")
    started = time.monotonic()
    layout, layout_metadata = load_context_attention_frozen_layout(Path(frozen_layout_features))
    sources = load_absolute_grid_context_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_context_cache=Path(radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
        expected_radio_checkpoint=str(layout_metadata.get("radio_checkpoint_sha256", "")),
    )
    source_map = _source_by_name(sources)
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("support geometry must use real SfM observation coordinates")
    runtime = build_fixed_candidate_context_runtime(
        query_ids=layout["query_ids"],
        query_xy=layout["xy"],
        candidate_track_ids=layout["candidate_track_ids"],
        candidate_support_image_ids=layout["candidate_support_image_ids"],
        candidate_view_valid=layout["candidate_view_valid"],
        cache_image_ids=sources[0].image_ids,
        support_geometry=geometry,
    )
    view_valid = np.asarray(runtime.view_valid, dtype=bool)
    edge_rows, _edge_candidates, edge_views = np.nonzero(view_valid)
    if len(edge_rows) == 0:
        raise ValueError("frozen layout has no candidate support-view edges")
    edge_support_indices = runtime.support_image_indices[edge_rows, _edge_candidates, edge_views]
    edge_support_xy = runtime.support_xy[edge_rows, _edge_candidates, edge_views]
    flat_positions = np.ravel_multi_index(
        (edge_rows, _edge_candidates, edge_views), dims=view_valid.shape
    )
    edge_count = len(edge_rows)
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
                _compute_partition,
                device_name=device,
                sources=sources,
                query_image_indices=runtime.query_image_indices,
                query_xy=np.asarray(layout["xy"], dtype=np.float32),
                edge_rows=edge_rows,
                edge_support_image_indices=edge_support_indices,
                edge_support_xy=edge_support_xy,
                begin=begin,
                end=end,
                batch_size=int(batch_size),
            )
            for device, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    context = np.concatenate([value[0] for value in computed], axis=0)
    position = np.concatenate([value[1] for value in computed], axis=0)
    anchor = np.concatenate([value[2] for value in computed], axis=0)
    if context.shape != (edge_count, len(ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES)):
        raise RuntimeError("absolute global-transport visual feature shape drifted")
    if position.shape != (edge_count, len(ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES)):
        raise RuntimeError("absolute global-transport position feature shape drifted")
    features = np.full(
        (*view_valid.shape, len(ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES)),
        np.nan,
        dtype=np.float16,
    )
    flat = features.reshape(-1, features.shape[-1])
    flat[flat_positions, 0] = anchor
    context_end = 1 + len(ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES)
    flat[flat_positions, 1:context_end] = context
    flat[flat_positions, context_end:] = position
    if np.any(~np.isfinite(features[view_valid])):
        raise RuntimeError("absolute global-transport valid feature rows are invalid")
    support_ids = np.asarray(layout["candidate_support_image_ids"]).astype(str)
    proposals_sha256 = str(layout_metadata.get("proposals_sha256", ""))
    if not proposals_sha256:
        raise ValueError("frozen layout lacks proposal lineage")
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "feature_definition": "masked_full_image_candidate_specific_absolute_transport_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "full_image_spatial_layout_used": True,
        "render": False,
        "candidate_set": "frozen_global_top20_tracks",
        "support_view_selection": layout_metadata.get("support_view_selection"),
        "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
        "absolute_global_transport": {
            "descriptor_sources": [
                "real_rgb_radio_final_grid16_pca64_v1",
                "real_rgb_radio_intermediate_grid16_pca64_v1",
                "real_rgb_alike_grid16_pca64_v1",
            ],
            "grid_size": _GRID_SIZE,
            "region_grid_size": ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE,
            "center_mask": {
                "shape": [3, 3],
                "radius": CONTEXT_ATTENTION_CENTER_MASK_RADIUS,
                "applied_before_visual_correlation": True,
            },
            "query_support_alignment": "frozen_query_xy_to_fixed_sfm_support_observation_xy_v1",
            "visual_transport": "full_grid_query_region_to_support_region_attention_v1",
            "position_control": "constant_descriptor_matched_mask_transport_v1",
        },
        "appearance_evidence_manifest": {
            name: {
                "path": str(source.path),
                "sha256": file_sha256_short(source.path),
                "descriptor_dim": source.descriptor_dim,
                "source_image_manifest_sha256": source.metadata.get(
                    "source_image_manifest_sha256"
                ),
                "radio_checkpoint_sha256": source.metadata.get("radio_checkpoint_sha256"),
                "alike_checkpoint_sha256": source.metadata.get("alike_checkpoint_sha256"),
            }
            for name, source in source_map.items()
        },
        "descriptor_feature_names": list(ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES),
        "source_frozen_layout": str(frozen_layout_features),
        "source_frozen_layout_sha256": file_sha256_short(frozen_layout_features),
        "full_frozen_source_rows_sha256": _array_sha256_short(layout["source_row_indices"]),
        "frozen_candidate_tracks_sha256": _array_sha256_short(layout["candidate_track_ids"]),
        "proposals_sha256": proposals_sha256,
        "support_geometry_index": str(support_geometry_index),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_index),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "radio_checkpoint_sha256": source_map["radio_final_grid16"].metadata.get(
            "radio_checkpoint_sha256"
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
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
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
            candidate_support_coverage_counts=np.asarray(
                layout["candidate_support_coverage_counts"], dtype=np.int32
            ),
            feature_names=np.asarray(ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    summary = {
        "stage": "build_absolute_global_transport_features",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": ARTIFACT_FORMAT,
        "row_count": int(len(layout["source_row_indices"])),
        "feature_count": int(len(ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES)),
        "visual_feature_count": int(len(ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES)),
        "position_control_feature_count": int(len(ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES)),
        "support_view_count": int(view_valid.shape[2]),
        "valid_candidate_view_count": int(np.sum(view_valid)),
        "devices": list(selected_devices),
        "workers": [item[3] for item in computed],
        "runtime_seconds": float(time.monotonic() - started),
        "protocol": {
            "image_retrieval_or_submap_used": False,
            "hard_image_retrieval_or_candidate_reselection": False,
            "pose_or_ground_truth_used": False,
            "view_features_averaged_before_inference": False,
            "render": False,
        },
    }
    summary_json = Path(summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_absolute_global_transport_candidate_probe_features(
        frozen_layout_features=Path(args.frozen_layout_features),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        devices=_parse_devices(args.devices),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
