"""Export dense ALIKE local translation-mode evidence on a frozen layout.

Every candidate track, support view, and query token comes from the supplied
frozen layout.  The exporter only compares local dense ALIKE cells around the
query token and the real SfM observation of that fixed candidate/view.
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

from feature_extract.tools.vfm.build_alike_image_spatial_context_cache import (
    ARTIFACT_FORMAT as ALIKE_SPATIAL_CONTEXT_FORMAT,
)
from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
    _load_frozen_layout,
)
from feature_extract.tools.vfm.build_global_context_support8_candidate_probe_features import (
    _fixed_support8_layout,
    _load_maplet_support_index,
)
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    MAPLET_FORMAT,
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
    DENSE_ALIKE_LOCAL_MODE_CONTEXT_FEATURE_NAMES,
    DENSE_ALIKE_LOCAL_MODE_FEATURE_ARTIFACT_FORMAT,
    DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES,
    batched_dense_local_translation_mode_features,
)
from feature_extract.vfm.localization.radio_final_context import (
    RADIO_FINAL_CONTEXT_PCA_FORMAT,
    load_radio_final_context_pca_cache,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    load_spatial_image_context_cache,
)


ARTIFACT_FORMAT = DENSE_ALIKE_LOCAL_MODE_FEATURE_ARTIFACT_FORMAT
_GRID_SIZE = 64
_MODE_SCALES = (
    ("alike_dense_grid64_window15_radius3", 15, 3),
    ("alike_dense_grid64_window31_radius4", 31, 4),
)
_TEMPERATURE = 0.07


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_source_caches(
    *,
    radio_final_context_cache: Path,
    alike_spatial_context_cache: Path,
    expected_radio_checkpoint: str,
) -> tuple[SpatialImageContextCache, Mapping[str, object], np.ndarray, np.ndarray]:
    final = load_radio_final_context_pca_cache(Path(radio_final_context_cache))
    if final.metadata.get("format") != RADIO_FINAL_CONTEXT_PCA_FORMAT:
        raise ValueError("unsupported RADIO-final anchor cache")
    if final.metadata.get("pca_fit_scope") != "mapping_train_images_only":
        raise ValueError("RADIO-final anchor cache was not fit on mapping train images")
    if bool(final.metadata.get("pose_or_ground_truth_used", True)) or bool(
        final.metadata.get("image_retrieval_or_submap_used", True)
    ) or bool(final.metadata.get("render", False)):
        raise ValueError("RADIO-final anchor cache violates the target-free protocol")
    if expected_radio_checkpoint and str(final.metadata.get("radio_checkpoint_sha256", "")) != str(
        expected_radio_checkpoint
    ):
        raise ValueError("frozen layout and RADIO-final anchor cache checkpoints differ")
    alike = load_spatial_image_context_cache(
        Path(alike_spatial_context_cache), expected_format=ALIKE_SPATIAL_CONTEXT_FORMAT
    )
    if bool(alike.metadata.get("pose_or_ground_truth_used", True)) or bool(
        alike.metadata.get("image_retrieval_or_submap_used", True)
    ) or bool(alike.metadata.get("render", False)):
        raise ValueError("ALIKE dense cache violates the target-free protocol")
    if not str(alike.metadata.get("alike_checkpoint_sha256", "")):
        raise ValueError("ALIKE dense cache lacks checkpoint lineage")
    if not np.array_equal(alike.image_ids.astype(str), final.image_ids.astype(str)) or not np.array_equal(
        alike.image_sizes, final.image_sizes
    ):
        raise ValueError("ALIKE dense cache images do not align with RADIO-final anchor cache")
    final_manifest = str(final.metadata.get("source_image_manifest_sha256", ""))
    if not final_manifest or str(alike.metadata.get("source_image_manifest_sha256", "")) != final_manifest:
        raise ValueError("ALIKE dense cache source-image manifest differs from RADIO-final")
    grid = alike.grid_descriptors(_GRID_SIZE)
    if grid.shape[1:] != (_GRID_SIZE * _GRID_SIZE, alike.descriptor_dim):
        raise ValueError("ALIKE dense cache grid64 has an invalid shape")
    return alike, final.metadata, final.image_ids.astype(str), final.image_sizes


def _crop_dense_grid_torch(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop one masked square dense grid around each image-coordinate anchor."""

    if (
        image_grids.ndim != 4
        or image_grids.shape[1] != image_grids.shape[2]
        or image_sizes.shape != (image_grids.shape[0], 2)
        or image_indices.ndim != 1
        or xy.shape != (len(image_indices), 2)
        or int(window_size) <= 0
        or int(window_size) % 2 != 1
    ):
        raise ValueError("dense ALIKE crop inputs are incompatible")
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
    alike_grid: np.ndarray,
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
    width = len(DENSE_ALIKE_LOCAL_MODE_CONTEXT_FEATURE_NAMES)
    with torch.no_grad():
        grids = torch.from_numpy(alike_grid).to(device=device, dtype=torch.float32)
        sizes = torch.from_numpy(image_sizes).to(device=device, dtype=torch.float32)
        output = np.empty((int(end) - int(begin), width), dtype=np.float16)
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
            for _name, window_size, maximum_shift in _MODE_SCALES:
                query_crop, query_valid = _crop_dense_grid_torch(
                    image_grids=grids,
                    image_sizes=sizes,
                    image_indices=query_indices,
                    xy=query_coordinates,
                    window_size=int(window_size),
                )
                support_crop, support_valid = _crop_dense_grid_torch(
                    image_grids=grids,
                    image_sizes=sizes,
                    image_indices=support_indices,
                    xy=support_coordinates,
                    window_size=int(window_size),
                )
                values.append(
                    batched_dense_local_translation_mode_features(
                        query_crop,
                        query_valid,
                        support_crop,
                        support_valid,
                        maximum_shift=int(maximum_shift),
                        temperature=_TEMPERATURE,
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
                            "stage": "dense_alike_local_mode",
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
        raise RuntimeError("dense ALIKE local-mode export produced non-finite values")
    return output, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def build_dense_alike_local_mode_candidate_probe_features(
    *,
    frozen_layout_features: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("dense ALIKE local-mode batch_size must be positive")
    selected_devices = tuple(str(device) for device in devices)
    if not selected_devices:
        raise ValueError("dense ALIKE local-mode needs at least one device")
    if (Path(output).exists() or Path(summary_json).exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite dense ALIKE local-mode outputs")
    started = time.monotonic()
    layout, layout_metadata = _load_frozen_layout(Path(frozen_layout_features))
    expected_checkpoint = str(layout_metadata.get("radio_checkpoint_sha256", ""))
    alike, final_metadata, image_ids, image_sizes = _validate_source_caches(
        radio_final_context_cache=Path(radio_final_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
        expected_radio_checkpoint=expected_checkpoint,
    )
    maplet, maplet_metadata = _load_maplet_support_index(Path(maplet_support_index))
    if maplet_metadata.get("format") != MAPLET_FORMAT:
        raise ValueError("unsupported maplet support index format")
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("support geometry must use real SfM observation coordinates")
    proposals_sha256 = str(layout_metadata.get("proposals_sha256", ""))
    if not proposals_sha256:
        raise ValueError("frozen layout lacks proposal lineage")
    support_ids, view_valid, coverage = _fixed_support8_layout(layout=layout, maplet=maplet)
    edge = _fixed_candidate_view_edges(
        layout=layout,
        support_ids=support_ids,
        view_valid=view_valid,
        geometry=geometry,
        radio_image_ids=image_ids,
    )
    query_cache_rows = _cache_image_indices(
        cache_image_ids=image_ids,
        image_ids=np.asarray(layout["query_ids"]).astype(str),
        context="query",
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
    alike_grid = alike.grid_descriptors(_GRID_SIZE).reshape(
        len(image_ids), _GRID_SIZE, _GRID_SIZE, alike.descriptor_dim
    )
    with ThreadPoolExecutor(max_workers=len(selected_devices)) as executor:
        futures = [
            executor.submit(
                _compute_partition,
                device_name=device,
                alike_grid=alike_grid,
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
    if context.shape != (edge_count, len(DENSE_ALIKE_LOCAL_MODE_CONTEXT_FEATURE_NAMES)):
        raise RuntimeError("dense ALIKE local-mode feature shape drifted")
    features = np.full(
        (*view_valid.shape, len(DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES)),
        np.nan,
        dtype=np.float16,
    )
    flat = features.reshape(-1, features.shape[-1])
    flat[edge["flat_position"], 0] = anchor[edge["row"], edge["candidate"]].astype(
        np.float16, copy=False
    )
    flat[edge["flat_position"], 1:] = context
    valid_features = features[view_valid]
    if np.any(~np.isfinite(valid_features)):
        raise RuntimeError("dense ALIKE local-mode valid feature rows are invalid")
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "feature_definition": "fixed_maplet_support8_dense_alike_grid64_bounded_translation_mode_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "render": False,
        "candidate_set": "frozen_global_top20_tracks",
        "support_view_selection": "fixed_maplet_coverage_rank_top8_v1",
        "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
        "source_image_manifest_sha256": alike.metadata.get("source_image_manifest_sha256"),
        "dense_local_modes": {
            "descriptor_source": "real_rgb_alike_dense_fpn_grid64_v1",
            "grid_size": _GRID_SIZE,
            "scales": [
                {"name": name, "window_size": window, "maximum_shift": shift}
                for name, window, shift in _MODE_SCALES
            ],
            "temperature": _TEMPERATURE,
            "explicit_crop_coverage_feature_used": False,
            "full_query_support_pair_matrix_exported": False,
        },
        "appearance_evidence_manifest": {
            "alike_spatial_context_cache": str(alike_spatial_context_cache),
            "alike_spatial_context_cache_sha256": file_sha256_short(alike_spatial_context_cache),
            "alike_checkpoint_sha256": alike.metadata.get("alike_checkpoint_sha256"),
            "alike_coordinate_convention": alike.metadata.get("coordinate_convention"),
            "radio_final_context_cache": str(radio_final_context_cache),
            "radio_final_context_cache_sha256": file_sha256_short(radio_final_context_cache),
            "radio_checkpoint_sha256": final_metadata.get("radio_checkpoint_sha256"),
            "radio_final_pca_fit_scope": final_metadata.get("pca_fit_scope"),
        },
        "descriptor_feature_names": list(DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES),
        "source_frozen_layout": str(frozen_layout_features),
        "source_frozen_layout_sha256": file_sha256_short(frozen_layout_features),
        "frozen_layout_features_sha256": _array_sha256_short(layout["candidate_features"]),
        "full_frozen_source_rows_sha256": _array_sha256_short(layout["source_row_indices"]),
        "frozen_candidate_tracks_sha256": _array_sha256_short(layout["candidate_track_ids"]),
        "proposals_sha256": proposals_sha256,
        "maplet_support_index": str(maplet_support_index),
        "maplet_support_index_sha256": file_sha256_short(maplet_support_index),
        "support_geometry_index": str(support_geometry_index),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_index),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "radio_checkpoint_sha256": final_metadata.get("radio_checkpoint_sha256"),
        "alike_checkpoint_sha256": alike.metadata.get("alike_checkpoint_sha256"),
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
            candidate_canonical_rows=np.asarray(
                layout["candidate_canonical_rows"], dtype=np.int64
            ),
            candidate_features=features,
            candidate_view_valid=view_valid,
            candidate_support_image_ids=support_ids,
            candidate_support_coverage_counts=coverage,
            feature_names=np.asarray(DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    summary = {
        "stage": "build_fixed_maplet_support8_dense_alike_local_mode_features",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": ARTIFACT_FORMAT,
        "row_count": int(len(layout["source_row_indices"])),
        "feature_count": int(len(DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES)),
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
    summary = build_dense_alike_local_mode_candidate_probe_features(
        frozen_layout_features=Path(args.frozen_layout_features),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
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
