"""Export fixed-view, anchor-aligned RADIO-final grid16 layout evidence.

The exporter consumes only a frozen top-L layout, real-image RADIO feature
maps, and real SfM observation coordinates for the already selected support
views.  It never retrieves images, chooses candidates, evaluates a pose, or
loads target errors.
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

from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
    _load_frozen_layout,
)
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    _cache_image_indices,
    _candidate_radio_final_anchor,
    _fixed_candidate_view_edges,
    _parse_devices,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMAT,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_SCALES,
    batched_masked_translation_correlation_features,
)
from feature_extract.vfm.localization.radio_final_context import (
    RADIO_FINAL_CONTEXT_PCA_FORMAT,
    load_radio_final_context_pca_cache,
)


ARTIFACT_FORMAT = ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMAT
_GRID_SIZE = 16


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_radio_final_cache(
    path: Path, *, expected_radio_checkpoint: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    cache = load_radio_final_context_pca_cache(Path(path))
    metadata = cache.metadata
    if metadata.get("format") != RADIO_FINAL_CONTEXT_PCA_FORMAT:
        raise ValueError("unsupported RADIO-final spatial cache")
    if metadata.get("pca_fit_scope") != "mapping_train_images_only":
        raise ValueError("RADIO-final PCA must be fit on mapping train images only")
    if bool(metadata.get("pose_or_ground_truth_used", True)) or bool(
        metadata.get("image_retrieval_or_submap_used", True)
    ) or bool(metadata.get("render", False)):
        raise ValueError("RADIO-final spatial cache violates the target-free protocol")
    if expected_radio_checkpoint and str(metadata.get("radio_checkpoint_sha256", "")) != str(
        expected_radio_checkpoint
    ):
        raise ValueError("frozen layout and RADIO-final cache checkpoints differ")
    if cache.grid16_descriptors is None:
        raise ValueError("RADIO-final spatial cache lacks grid16 descriptors")
    grids = np.asarray(cache.grid16_descriptors, dtype=np.float32)
    if grids.shape[1] != _GRID_SIZE * _GRID_SIZE:
        raise ValueError("RADIO-final grid16 descriptors have an invalid shape")
    manifest = str(metadata.get("source_image_manifest_sha256", ""))
    if not manifest:
        raise ValueError("RADIO-final spatial cache lacks its source-image manifest")
    return (
        grids.reshape(len(cache.image_ids), _GRID_SIZE, _GRID_SIZE, cache.descriptor_dim),
        np.asarray(cache.image_ids).astype(str),
        np.asarray(cache.image_sizes, dtype=np.int64),
        metadata,
    )


def _crop_grid_torch(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a masked grid around a frozen image-coordinate anchor."""

    if (
        image_grids.ndim != 4
        or image_grids.shape[1] != image_grids.shape[2]
        or image_sizes.shape != (image_grids.shape[0], 2)
        or image_indices.ndim != 1
        or xy.shape != (len(image_indices), 2)
        or int(window_size) <= 0
        or int(window_size) % 2 != 1
    ):
        raise ValueError("anchor-aligned grid crop inputs are incompatible")
    grid_size = int(image_grids.shape[1])
    selected_sizes = image_sizes.index_select(0, image_indices).to(dtype=torch.float32)
    columns = torch.floor(
        xy[:, 0] / selected_sizes[:, 0].clamp_min(1.0) * float(grid_size)
    ).to(dtype=torch.long).clamp(0, grid_size - 1)
    rows = torch.floor(
        xy[:, 1] / selected_sizes[:, 1].clamp_min(1.0) * float(grid_size)
    ).to(dtype=torch.long).clamp(0, grid_size - 1)
    offsets = torch.arange(
        -(int(window_size) // 2), int(window_size) // 2 + 1, device=image_grids.device
    )
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
    return crop, valid


def _compute_partition(
    *,
    device_name: str,
    grids: np.ndarray,
    image_sizes: np.ndarray,
    query_cache_rows: np.ndarray,
    query_xy: np.ndarray,
    edge_rows: np.ndarray,
    edge_support_cache_rows: np.ndarray,
    edge_support_xy: np.ndarray,
    begin: int,
    end: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.monotonic()
    device = torch.device(device_name)
    with torch.no_grad():
        image_grids = torch.from_numpy(grids).to(device=device, dtype=torch.float32)
        sizes = torch.from_numpy(image_sizes).to(device=device, dtype=torch.float32)
        output = np.empty(
            (int(end) - int(begin), len(ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES)),
            dtype=np.float16,
        )
        progress_interval = max(int(batch_size) * 64, int(batch_size))
        for offset in range(int(begin), int(end), int(batch_size)):
            stop = min(offset + int(batch_size), int(end))
            rows = edge_rows[offset:stop]
            query_indices = torch.from_numpy(query_cache_rows[rows]).to(device=device)
            support_indices = torch.from_numpy(edge_support_cache_rows[offset:stop]).to(
                device=device
            )
            query_coordinates = torch.from_numpy(query_xy[rows]).to(
                device=device, dtype=torch.float32
            )
            support_coordinates = torch.from_numpy(edge_support_xy[offset:stop]).to(
                device=device, dtype=torch.float32
            )
            values: list[torch.Tensor] = []
            for _name, window_size, maximum_shift in ANCHOR_ALIGNED_GLOBAL_LAYOUT_SCALES:
                query_crop, query_valid = _crop_grid_torch(
                    image_grids=image_grids,
                    image_sizes=sizes,
                    image_indices=query_indices,
                    xy=query_coordinates,
                    window_size=int(window_size),
                )
                support_crop, support_valid = _crop_grid_torch(
                    image_grids=image_grids,
                    image_sizes=sizes,
                    image_indices=support_indices,
                    xy=support_coordinates,
                    window_size=int(window_size),
                )
                values.append(
                    batched_masked_translation_correlation_features(
                        query_crop,
                        query_valid,
                        support_crop,
                        support_valid,
                        maximum_shift=int(maximum_shift),
                    )
                )
            value = torch.cat(values, dim=1)
            output[offset - int(begin) : stop - int(begin)] = (
                value.cpu().numpy().astype(np.float16, copy=False)
            )
            completed = int(stop) - int(begin)
            if completed % progress_interval == 0 or int(stop) == int(end):
                print(
                    json.dumps(
                        {
                            "stage": "anchor_aligned_global_layout",
                            "device": str(device),
                            "completed": completed,
                            "assigned": int(end) - int(begin),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if np.any(~np.isfinite(output)):
        raise RuntimeError("anchor-aligned global-layout export produced non-finite values")
    return output, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def build_anchor_aligned_global_layout_candidate_probe_features(
    *,
    frozen_layout_features: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    output: Path,
    summary_json: Path,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("anchor-aligned global-layout batch_size must be positive")
    selected_devices = tuple(str(device) for device in devices)
    if not selected_devices:
        raise ValueError("anchor-aligned global-layout needs at least one device")
    if (Path(output).exists() or Path(summary_json).exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite anchor-aligned global-layout outputs")
    started = time.monotonic()
    layout, layout_metadata = _load_frozen_layout(Path(frozen_layout_features))
    expected_checkpoint = str(layout_metadata.get("radio_checkpoint_sha256", ""))
    grids, image_ids, image_sizes, final_metadata = _validate_radio_final_cache(
        Path(radio_final_context_cache), expected_radio_checkpoint=expected_checkpoint
    )
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("support geometry must use real SfM observation coordinates")
    proposals_sha256 = str(layout_metadata.get("proposals_sha256", ""))
    if not proposals_sha256:
        raise ValueError("frozen layout lacks proposal lineage")
    support_ids = np.asarray(layout["candidate_support_image_ids"]).astype(str)
    view_valid = np.asarray(layout["candidate_view_valid"], dtype=bool)
    coverage = np.asarray(layout["candidate_support_coverage_counts"], dtype=np.int32)
    if support_ids.shape != view_valid.shape or coverage.shape != view_valid.shape:
        raise ValueError("frozen support-view arrays are incompatible")
    query_ids = np.asarray(layout["query_ids"]).astype(str)
    if set(query_ids.tolist()) & set(support_ids[view_valid].tolist()):
        raise ValueError("query images leaked into fixed support-view layout evidence")
    edge = _fixed_candidate_view_edges(
        layout=layout,
        support_ids=support_ids,
        view_valid=view_valid,
        geometry=geometry,
        radio_image_ids=image_ids,
    )
    query_cache_rows = _cache_image_indices(
        cache_image_ids=image_ids, image_ids=query_ids, context="query"
    )
    anchor = _candidate_radio_final_anchor(layout)
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
                _compute_partition,
                device_name=device,
                grids=grids,
                image_sizes=image_sizes,
                query_cache_rows=query_cache_rows,
                query_xy=np.asarray(layout["xy"], dtype=np.float32),
                edge_rows=edge["row"],
                edge_support_cache_rows=edge["support_cache_row"],
                edge_support_xy=edge["support_xy"],
                begin=begin,
                end=end,
                batch_size=int(batch_size),
            )
            for device, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    context = np.concatenate([values for values, _worker in computed], axis=0)
    if context.shape != (edge_count, len(ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES)):
        raise RuntimeError("anchor-aligned global-layout feature shape drifted")
    features = np.full(
        (*view_valid.shape, len(ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES)),
        np.nan,
        dtype=np.float16,
    )
    flat = features.reshape(-1, features.shape[-1])
    flat[edge["flat_position"], 0] = anchor[edge["row"], edge["candidate"]].astype(
        np.float16, copy=False
    )
    flat[edge["flat_position"], 1:] = context
    if np.any(~np.isfinite(features[view_valid])):
        raise RuntimeError("anchor-aligned global-layout valid feature rows are invalid")
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "feature_definition": (
            "per_view_candidate_specific_radio_final_grid16_"
            "anchor_aligned_translation_correlation_v1"
        ),
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
        "anchor_aligned_global_layout": {
            "descriptor_source": "real_rgb_radio_final_grid16_pca64_v1",
            "grid_size": _GRID_SIZE,
            "scales": [
                {
                    "name": name,
                    "window_size": int(window_size),
                    "maximum_shift": int(maximum_shift),
                }
                for name, window_size, maximum_shift in ANCHOR_ALIGNED_GLOBAL_LAYOUT_SCALES
            ],
            "correlation": "masked_corresponding_cells_per_bounded_translation_v1",
            "query_support_alignment": "frozen_query_xy_to_fixed_sfm_support_observation_xy_v1",
            "full_query_support_pair_matrix_exported": False,
        },
        "appearance_evidence_manifest": {
            "radio_final_context_cache": str(radio_final_context_cache),
            "radio_final_context_cache_sha256": file_sha256_short(radio_final_context_cache),
            "radio_checkpoint_sha256": final_metadata.get("radio_checkpoint_sha256"),
            "radio_final_pca_fit_scope": final_metadata.get("pca_fit_scope"),
            "source_image_manifest_sha256": final_metadata.get("source_image_manifest_sha256"),
        },
        "descriptor_feature_names": list(ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES),
        "source_frozen_layout": str(frozen_layout_features),
        "source_frozen_layout_sha256": file_sha256_short(frozen_layout_features),
        "frozen_layout_features_sha256": _array_sha256_short(layout["candidate_features"]),
        "full_frozen_source_rows_sha256": _array_sha256_short(layout["source_row_indices"]),
        "frozen_candidate_tracks_sha256": _array_sha256_short(layout["candidate_track_ids"]),
        "proposals_sha256": proposals_sha256,
        "support_geometry_index": str(support_geometry_index),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_index),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "radio_checkpoint_sha256": final_metadata.get("radio_checkpoint_sha256"),
        "candidate_top_k": int(np.asarray(layout["candidate_track_ids"]).shape[1]),
        "support_view_count": int(view_valid.shape[2]),
        "query_row_count": int(len(layout["source_row_indices"])),
        "query_image_count": int(len(set(query_ids.tolist()))),
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
            query_ids=query_ids,
            split_names=np.asarray(layout["split_names"]).astype(str),
            xy=np.asarray(layout["xy"], dtype=np.float32),
            candidate_track_ids=np.asarray(layout["candidate_track_ids"], dtype=np.int64),
            candidate_canonical_rows=np.asarray(
                layout["candidate_canonical_rows"], dtype=np.int64
            ),
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_ids,
            candidate_support_coverage_counts=coverage,
            feature_names=np.asarray(ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    summary = {
        "stage": "build_anchor_aligned_radio_final_grid16_layout_features",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": ARTIFACT_FORMAT,
        "row_count": int(len(layout["source_row_indices"])),
        "feature_count": int(len(ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES)),
        "support_view_count": int(view_valid.shape[2]),
        "valid_candidate_view_count": int(np.sum(view_valid)),
        "devices": list(selected_devices),
        "workers": [worker for _values, worker in computed],
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
    summary = build_anchor_aligned_global_layout_candidate_probe_features(
        frozen_layout_features=Path(args.frozen_layout_features),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        devices=_parse_devices(str(args.devices)),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
