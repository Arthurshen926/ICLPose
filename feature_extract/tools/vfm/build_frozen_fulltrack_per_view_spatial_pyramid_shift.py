"""Export frozen per-view spatial-pyramid shift evidence on the current CSR.

For every immutable ``(query point, global-top20 candidate track, real SfM
support observation)`` edge, retain descriptor correlations for each local
query-region and bounded relative shift.  The exporter does not read targets,
project a candidate with a pose, choose support views, use image retrieval, or
render.  Only an invalid centre anchor is omitted as unknown; border samples
use reflection from the real feature map, while original-crop overlap is
exported only in a separate control artifact.
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

from feature_extract.tools.vfm.build_frozen_fulltrack_global_context import (
    _array_sha256_short,
    _geometry_image_indices,
    _load_current_raw_per_view_source,
)
from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_multiscale_translation_mode import (
    _TranslationSource,
    _crop_grid_torch,
    _edge_offsets,
    _load_translation_sources,
    _source_by_name,
    _source_metadata,
)
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    _cache_image_indices,
    _parse_devices,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_spatial_pyramid_shift import (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS,
    SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE,
    SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE,
    SPATIAL_PYRAMID_SHIFT_PROFILES,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    batched_spatial_pyramid_shift_correlation_features,
    batched_spatial_pyramid_shift_overlap_features,
)


ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_spatial_pyramid_shift_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-per-view-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-pca256-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--mask-control-output", required=True)
    parser.add_argument("--mask-control-summary-json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _profile_feature_slices(*, mask_control: bool = False) -> dict[str, slice]:
    offset = 0
    output: dict[str, slice] = {}
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES:
        names = (
            SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE[profile.name]
            if bool(mask_control)
            else SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE[profile.name]
        )
        width = len(names)
        output[profile.name] = slice(offset, offset + width)
        offset += width
    expected_width = (
        len(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES)
        if bool(mask_control)
        else len(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES)
    )
    if offset != expected_width:
        raise RuntimeError("spatial-pyramid profile feature offsets drifted")
    return output


def _source_by_profile(
    sources: Sequence[_TranslationSource],
) -> dict[str, _TranslationSource]:
    by_name = _source_by_name(sources)
    expected = {profile.source_name for profile in SPATIAL_PYRAMID_SHIFT_PROFILES}
    if not expected.issubset(by_name):
        raise RuntimeError("spatial-pyramid sources are incomplete")
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES:
        if int(by_name[profile.source_name].grid_size) != int(profile.grid_size):
            raise ValueError(f"{profile.name}: feature grid size differs from profile")
    return by_name


def _full_crop_valid(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError("spatial-pyramid crop mask is invalid")
    return torch.all(mask.reshape(len(mask), -1), dim=1)


def _center_crop_valid(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3 or mask.shape[1] != mask.shape[2]:
        raise ValueError("spatial-pyramid crop mask is invalid")
    center = int(mask.shape[1]) // 2
    return mask[:, center, center]


def _compute_partition(
    *,
    device_name: str,
    sources: tuple[_TranslationSource, ...],
    query_cache_rows: np.ndarray,
    query_xy: np.ndarray,
    edge_rows: np.ndarray,
    edge_support_cache_rows: np.ndarray,
    edge_support_xy: np.ndarray,
    begin: int,
    end: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Materialize a disjoint immutable edge range on one device."""

    started = time.monotonic()
    device = torch.device(device_name)
    source_by_name = _source_by_profile(sources)
    feature_count = len(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES)
    profile_offsets = _profile_feature_slices()
    control_feature_count = len(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES
    )
    control_profile_offsets = _profile_feature_slices(mask_control=True)
    with torch.inference_mode():
        tensors = {
            name: (
                torch.from_numpy(source.grids).to(device=device, dtype=torch.float32),
                torch.from_numpy(source.image_sizes).to(
                    device=device, dtype=torch.float32
                ),
            )
            for name, source in source_by_name.items()
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
            for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
        }
        scores = np.full(
            (int(end) - int(begin), feature_count), np.nan, dtype=np.float16
        )
        valid = np.zeros(scores.shape, dtype=bool)
        control_scores = np.full(
            (int(end) - int(begin), control_feature_count), np.nan, dtype=np.float16
        )
        control_valid = np.zeros(control_scores.shape, dtype=bool)
        full_crop_coverage: dict[str, list[float]] = {
            profile.name: [] for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
        }
        center_coverage: dict[str, list[float]] = {
            profile.name: [] for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
        }
        for offset in range(int(begin), int(end), int(batch_size)):
            stop = min(offset + int(batch_size), int(end))
            point_rows = torch.from_numpy(edge_rows[offset:stop]).to(device=device)
            support_rows = torch.from_numpy(edge_support_cache_rows[offset:stop]).to(
                device=device
            )
            support_xy = torch.from_numpy(edge_support_xy[offset:stop]).to(
                device=device, dtype=torch.float32
            )
            local_scores = scores[offset - int(begin) : stop - int(begin)]
            local_valid = valid[offset - int(begin) : stop - int(begin)]
            local_control_scores = control_scores[offset - int(begin) : stop - int(begin)]
            local_control_valid = control_valid[offset - int(begin) : stop - int(begin)]
            for profile in SPATIAL_PYRAMID_SHIFT_PROFILES:
                source_tensors = tensors[profile.source_name]
                support_crop, support_mask = _crop_grid_torch(
                    image_grids=source_tensors[0],
                    image_sizes=source_tensors[1],
                    image_indices=support_rows,
                    xy=support_xy,
                    window_size=int(profile.window_size),
                    padding_mode="reflection",
                )
                query_crop, query_mask = query_context[profile.name]
                query_crop = query_crop.index_select(0, point_rows)
                query_mask = query_mask.index_select(0, point_rows)
                usable = _center_crop_valid(query_mask) & _center_crop_valid(support_mask)
                usable_np = usable.cpu().numpy().astype(bool, copy=False)
                full_crop_coverage[profile.name].append(
                    float(
                        (
                            _full_crop_valid(query_mask)
                            & _full_crop_valid(support_mask)
                        )
                        .to(dtype=torch.float32)
                        .mean()
                        .item()
                    )
                )
                center_coverage[profile.name].append(
                    float(usable.to(dtype=torch.float32).mean().item())
                )
                if not np.any(usable_np):
                    continue
                visual_valid = torch.ones_like(query_mask[usable])
                values = batched_spatial_pyramid_shift_correlation_features(
                    query_grid=query_crop[usable],
                    query_valid=visual_valid,
                    support_grid=support_crop[usable],
                    support_valid=torch.ones_like(support_mask[usable]),
                    maximum_shift=int(profile.maximum_shift),
                    spatial_bin_count=int(profile.spatial_bin_count),
                )
                controls = batched_spatial_pyramid_shift_overlap_features(
                    query_valid=query_mask[usable],
                    support_valid=support_mask[usable],
                    maximum_shift=int(profile.maximum_shift),
                    spatial_bin_count=int(profile.spatial_bin_count),
                )
                profile_slice = profile_offsets[profile.name]
                control_slice = control_profile_offsets[profile.name]
                values_np = values.cpu().numpy().astype(np.float16, copy=False)
                controls_np = controls.cpu().numpy().astype(np.float16, copy=False)
                local_scores[usable_np, profile_slice] = values_np
                local_valid[usable_np, profile_slice] = True
                local_control_scores[usable_np, control_slice] = controls_np
                local_control_valid[usable_np, control_slice] = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    if (
        np.any(np.isinf(scores))
        or np.any(~np.isfinite(scores[valid]))
        or np.any(np.isfinite(scores[~valid]))
        or np.any(np.isinf(control_scores))
        or np.any(~np.isfinite(control_scores[control_valid]))
        or np.any(np.isfinite(control_scores[~control_valid]))
    ):
        raise RuntimeError("spatial-pyramid exporter emitted invalid missingness")
    return scores, valid, control_scores, control_valid, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
        "profile_full_crop_coverage": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in full_crop_coverage.items()
        },
        "profile_center_coverage": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in center_coverage.items()
        },
    }


def _context_source_contract(sources: Sequence[_TranslationSource]) -> dict[str, Any]:
    by_name = _source_by_profile(sources)
    return {
        "mode": "candidate_specific_multiscale_spatial_pyramid_shift_correlation_v1",
        "support_coordinate_source": "sfm_observation_xy",
        "view_aggregation": "none_before_learned_logsumexp_mixture_v1",
        "visual_border_padding": "reflection_from_real_feature_map_v1",
        "missing_evidence": "invalid_center_anchor_edge_omitted_neutral_v1",
        "original_crop_mask_control_exported_separately": True,
        "explicit_availability_or_neighbor_count_feature": False,
        "full_crop_required": False,
        "center_anchor_required": True,
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "grid_size": int(profile.grid_size),
                "window_size": int(profile.window_size),
                "maximum_shift": int(profile.maximum_shift),
                "spatial_bin_count": int(profile.spatial_bin_count),
                "feature_names": list(
                    SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE[profile.name]
                ),
            }
            for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
        ],
        "mask_control_profiles": [
            {
                "name": profile.name,
                "feature_names": list(
                    SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE[
                        profile.name
                    ]
                ),
            }
            for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
        ],
        "sources": [_source_metadata(by_name[name]) for name in sorted(by_name)],
    }


def build_frozen_fulltrack_per_view_spatial_pyramid_shift(
    *,
    source_per_view_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_pca256_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    mask_control_output: Path,
    mask_control_summary_json: Path,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    """Copy one raw CSR shard and attach target-free spatial-layout evidence."""

    if int(batch_size) <= 0:
        raise ValueError("spatial-pyramid batch size must be positive")
    selected_devices = tuple(str(value) for value in devices)
    if not selected_devices or len(set(selected_devices)) != len(selected_devices):
        raise ValueError("spatial-pyramid devices must be unique and non-empty")
    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    final_path = Path(radio_final_context_cache)
    intermediate_path = Path(radio_intermediate_pca256_context_cache)
    alike_path = Path(alike_spatial_context_cache)
    output_path = Path(output)
    summary_path = Path(summary_json)
    control_output_path = Path(mask_control_output)
    control_summary_path = Path(mask_control_summary_json)
    if (
        output_path == control_output_path
        or summary_path == control_summary_path
        or len(
            {
                output_path,
                summary_path,
                control_output_path,
                control_summary_path,
            }
        )
        != 4
    ):
        raise ValueError("spatial-pyramid visual/control outputs must be distinct")
    if (
        output_path.exists()
        or summary_path.exists()
        or control_output_path.exists()
        or control_summary_path.exists()
    ) and not bool(force):
        raise FileExistsError("refusing to overwrite spatial-pyramid outputs")

    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("spatial-pyramid evidence requires real SfM observation xy")
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
        raise ValueError("raw per-view source and RADIO-final context differ")
    sources = _load_translation_sources(
        radio_final_context_cache=final_path,
        radio_intermediate_pca256_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
    )
    source_by_name = _source_by_profile(sources)
    cache_ids = source_by_name["radio_final"].image_ids
    query_cache_rows = _cache_image_indices(
        cache_image_ids=cache_ids,
        image_ids=np.asarray(source["verification_query_ids"]).astype(str),
        context="frozen query",
    )
    image_rows = _geometry_image_indices(geometry)
    geometry_rows = np.asarray(edges.geometry_rows, dtype=np.int64)
    support_ids = np.asarray(geometry.image_ids).astype(str)[image_rows[geometry_rows]]
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
        raise ValueError("spatial-pyramid artifact has no immutable CSR edges")
    partitions = [
        (
            edge_count * index // len(selected_devices),
            edge_count * (index + 1) // len(selected_devices),
        )
        for index in range(len(selected_devices))
    ]
    if any(end <= begin for begin, end in partitions):
        raise ValueError("more devices than immutable CSR edges")
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
                edge_support_xy=np.asarray(geometry.xy, dtype=np.float32)[geometry_rows],
                begin=begin,
                end=end,
                batch_size=int(batch_size),
            )
            for device, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    scores = np.concatenate([value[0] for value in computed], axis=0)
    valid = np.concatenate([value[1] for value in computed], axis=0)
    control_scores = np.concatenate([value[2] for value in computed], axis=0)
    control_valid = np.concatenate([value[3] for value in computed], axis=0)
    expected_width = len(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES)
    expected_control_width = len(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES
    )
    if (
        scores.shape != (edge_count, expected_width)
        or valid.shape != scores.shape
        or control_scores.shape != (edge_count, expected_control_width)
        or control_valid.shape != control_scores.shape
    ):
        raise RuntimeError("spatial-pyramid output shape drifted")
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
        raise RuntimeError("spatial-pyramid exporter changed CSR edge order")
    candidate = np.asarray(source["candidate_probabilities"], dtype=np.float32)
    if np.any((candidate > 0.0) & (edges.candidate_observation_counts <= 0)):
        raise RuntimeError("positive frozen candidate lost all support observations")

    query_id = str(source["verification_query_ids"][0])
    source_contract = _context_source_contract(sources)
    metadata: dict[str, Any] = {
        "format": FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT,
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
        "source_edge_feature_semantics": source_lineage["source_edge_feature_semantics"],
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
            "radio_intermediate_pca256": file_sha256_short(intermediate_path),
            "alike": file_sha256_short(alike_path),
        },
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_preserve_source_csr_order_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": (
            FULLTRACK_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
        ),
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": source_contract["profiles"],
        "spatial_pyramid_shift_contract": source_contract,
        "artifact_role": "visual_descriptor_correlation",
        "paired_mask_control_artifact": str(control_output_path),
        "appearance_config": {
            "candidate_specific": True,
            "per_view": True,
            "feature_definition": (
                "multiscale_candidate_relative_spatial_pyramid_shift_correlation_v1"
            ),
            "support_view_marginalization": "not_aggregated_export_per_view_v1",
            "whole_image_retrieval_or_candidate_reselection": False,
            "original_crop_mask_values_included": False,
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
            "explicit_availability_or_neighbor_count_is_not_a_learned_feature": True,
            "incomplete_center_anchor_is_unknown_not_visual_value": True,
            "visual_descriptor_values_included": True,
            "original_crop_mask_values_included": False,
            "paired_mask_control_artifact_required": True,
        },
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "spatial_pyramid_core_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm/localization/frozen_fulltrack_spatial_pyramid_shift.py"
            ),
            "spatial_pyramid_correlation_core_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm/localization/multiscale_candidate_probe.py"
            ),
            "crop_sampler_sha256": file_sha256_short(
                Path(__file__).with_name(
                    "build_frozen_fulltrack_per_view_multiscale_translation_mode.py"
                )
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    control_output_path.parent.mkdir(parents=True, exist_ok=True)
    control_summary_path.parent.mkdir(parents=True, exist_ok=True)
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
                FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES,
                dtype=np.str_,
            ),
            edge_candidate_offsets=offsets,
            edge_geometry_rows=geometry_rows,
            edge_profile_scores=scores.astype(np.float16, copy=False),
            edge_profile_valid=valid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output_path)
    control_metadata = {
        **metadata,
        "format": FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT,
        "version": (
            "frozen_fulltrack_candidate_per_view_spatial_pyramid_shift_mask_control_v1"
        ),
        "per_view_edge_feature_semantics": (
            FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
        "profiles": source_contract["mask_control_profiles"],
        "artifact_role": "original_crop_mask_overlap_control",
        "paired_visual_artifact": str(output_path),
        "appearance_config": {
            **metadata["appearance_config"],
            "feature_definition": (
                "original_crop_mask_overlap_control_matching_spatial_pyramid_layout_v1"
            ),
            "original_crop_mask_values_included": True,
            "visual_descriptor_values_included": False,
            "control_only": True,
        },
        "strict_fulltrack_appearance_contract": {
            **metadata["strict_fulltrack_appearance_contract"],
            "visual_descriptor_values_included": False,
            "original_crop_mask_values_included": True,
            "control_only": True,
        },
    }
    control_temporary = control_output_path.with_name(control_output_path.name + ".tmp")
    with control_temporary.open("wb") as handle:
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
                FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES,
                dtype=np.str_,
            ),
            edge_candidate_offsets=offsets,
            edge_geometry_rows=geometry_rows,
            edge_profile_scores=control_scores.astype(np.float16, copy=False),
            edge_profile_valid=control_valid,
            metadata_json=np.asarray(json.dumps(control_metadata, sort_keys=True)),
        )
    control_temporary.replace(control_output_path)
    profile_slices = _profile_feature_slices()
    control_profile_slices = _profile_feature_slices(mask_control=True)
    summary = {
        "stage": "build_frozen_fulltrack_per_view_spatial_pyramid_shift",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "mask_control_output": str(control_output_path),
        "mask_control_output_sha256": file_sha256_short(control_output_path),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "edge_count": edge_count,
        "feature_count": int(scores.shape[1]),
        "mask_control_feature_count": int(control_scores.shape[1]),
        "profile_full_crop_edge_coverage": {
            profile.name: float(
                sum(
                    worker["profile_full_crop_coverage"].get(profile.name, 0.0)
                    * int(worker["edge_count"])
                    for *_values, worker in computed
                )
                / edge_count
            )
            for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
        },
        "profile_center_anchor_edge_coverage": {
            profile.name: float(
                np.mean(np.all(valid[:, profile_slices[profile.name]], axis=1))
            )
            for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
        },
        "mask_control_profile_center_anchor_edge_coverage": {
            profile.name: float(
                np.mean(
                    np.all(control_valid[:, control_profile_slices[profile.name]], axis=1)
                )
            )
            for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
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
            "hard_image_retrieval_or_submap": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "diagnostic_only": True,
            "incomplete_center_anchor_is_unknown": True,
            "visual_border_padding": "reflection_from_real_feature_map_v1",
            "original_crop_mask_control_exported_separately": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    control_summary = {
        **summary,
        "stage": "build_frozen_fulltrack_per_view_spatial_pyramid_shift_mask_control",
        "output": str(control_output_path),
        "output_sha256": file_sha256_short(control_output_path),
        "paired_visual_output": str(output_path),
        "paired_visual_output_sha256": file_sha256_short(output_path),
        "feature_count": int(control_scores.shape[1]),
        "artifact_role": "original_crop_mask_overlap_control",
    }
    control_summary_path.write_text(
        json.dumps(control_summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frozen_fulltrack_per_view_spatial_pyramid_shift(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            args.radio_intermediate_pca256_context_cache
        ),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        mask_control_output=Path(args.mask_control_output),
        mask_control_summary_json=Path(args.mask_control_summary_json),
        devices=_parse_devices(str(args.devices)),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
