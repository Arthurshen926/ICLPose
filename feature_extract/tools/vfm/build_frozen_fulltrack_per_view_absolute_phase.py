"""Export full-CSR, candidate-specific absolute image-phase evidence.

Every output row starts from an immutable raw full-track CSR artifact.  For
each fixed ``(query point, candidate track, real SfM support observation)``
edge, the builder compares a small lattice of *absolute image regions* across
the query and support images.  It does not retrieve an image, select a view,
read a target, use a pose, or average views before the downstream diagnostic
mixture.

The local anchor core is removed from both images.  Visual columns contain
only descriptor transport; mask coverage is emitted in separately named
position-control columns so it cannot be accidentally used as visual evidence.
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
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    _cache_image_indices,
    _parse_devices,
)
from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_multiscale_translation_mode import (
    _TranslationSource,
    _load_translation_sources,
    _source_metadata,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_absolute_phase import (
    ABSOLUTE_PHASE_CENTER_MASK_RADIUS,
    ABSOLUTE_PHASE_REGION_GRID_SIZE,
    ABSOLUTE_PHASE_TEMPERATURE,
    ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE,
    ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE,
    ABSOLUTE_PHASE_VISUAL_PROFILES,
    FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
    FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES,
    absolute_phase_anchor_mask,
    batched_absolute_phase_position_control_features,
    batched_absolute_phase_region_transport_features,
    profile_feature_slices,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)


ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_absolute_phase_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-per-view-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-pca256-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _edge_offsets(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("absolute-phase candidate support observation counts are invalid")
    return np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(values.reshape(-1), dtype=np.int64))
    )


def _source_by_name(sources: Sequence[_TranslationSource]) -> dict[str, _TranslationSource]:
    output = {source.name: source for source in sources}
    expected = {profile.source_name for profile in ABSOLUTE_PHASE_VISUAL_PROFILES}
    if set(output) != expected:
        raise ValueError("absolute-phase sources are incomplete or duplicated")
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES:
        source = output[profile.source_name]
        if int(source.grid_size) != int(profile.grid_size):
            raise ValueError(f"{profile.name}: native source grid size drifted")
    return output


def _edge_context_inputs(
    *,
    geometry: SupportObservationGeometryIndex,
    candidate_shape: tuple[int, int],
    geometry_rows: np.ndarray,
    edge_candidate_indices: np.ndarray,
    cache_image_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(geometry_rows, dtype=np.int64).reshape(-1)
    candidates = np.asarray(edge_candidate_indices, dtype=np.int64).reshape(-1)
    if (
        len(candidate_shape) != 2
        or candidate_shape[0] <= 0
        or candidate_shape[1] <= 0
        or rows.shape != candidates.shape
        or np.any(rows < 0)
        or np.any(rows >= len(geometry.track_ids))
        or np.any(candidates < 0)
        or np.any(candidates >= int(candidate_shape[0]) * int(candidate_shape[1]))
    ):
        raise ValueError("absolute-phase immutable CSR edges are invalid")
    image_rows = _geometry_image_indices(geometry)
    support_ids = np.asarray(geometry.image_ids).astype(str)[image_rows[rows]]
    support_cache_rows = _cache_image_indices(
        cache_image_ids=np.asarray(cache_image_ids).astype(str),
        image_ids=support_ids,
        context="absolute-phase real full-track support observation",
    )
    return {
        "edge_rows": candidates // int(candidate_shape[1]),
        "support_cache_rows": support_cache_rows.astype(np.int64, copy=False),
        "support_xy": np.asarray(geometry.xy, dtype=np.float32)[rows],
    }


def _absolute_phase_contract(sources: Sequence[_TranslationSource]) -> dict[str, Any]:
    source_metadata = {_source_metadata(source)["name"]: _source_metadata(source) for source in sources}
    profiles: list[dict[str, Any]] = []
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES:
        profiles.append(
            {
                "name": profile.name,
                "source": profile.source_name,
                "grid_size": int(profile.grid_size),
                "feature_kind": "appearance_only_absolute_region_transport_v1",
                "feature_names": list(
                    ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE[profile.name]
                ),
            }
        )
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES:
        profiles.append(
            {
                "name": f"{profile.name}_position_control",
                "source": profile.source_name,
                "grid_size": int(profile.grid_size),
                "feature_kind": "anchor_mask_geometry_control_only_v1",
                "feature_names": list(
                    ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE[profile.name]
                ),
            }
        )
    return {
        "mode": "candidate_specific_absolute_image_region_transport_v1",
        "support_coordinate_source": "sfm_observation_xy",
        "view_aggregation": "none_before_learned_logsumexp_mixture_v1",
        "region_grid_size": int(ABSOLUTE_PHASE_REGION_GRID_SIZE),
        "center_mask": {
            "shape": [
                2 * int(ABSOLUTE_PHASE_CENTER_MASK_RADIUS) + 1,
                2 * int(ABSOLUTE_PHASE_CENTER_MASK_RADIUS) + 1,
            ],
            "anchor_content_excluded_from_visual_transport": True,
        },
        "temperature": float(ABSOLUTE_PHASE_TEMPERATURE),
        "visual_features_exclude_coverage_count_or_availability": True,
        "position_control_exported_separately": True,
        "profiles": profiles,
        "sources": [source_metadata[profile.source_name] for profile in ABSOLUTE_PHASE_VISUAL_PROFILES],
    }


def _compute_partition(
    *,
    device_name: str,
    sources: Sequence[_TranslationSource],
    query_cache_rows: np.ndarray,
    query_xy: np.ndarray,
    edge_rows: np.ndarray,
    edge_support_cache_rows: np.ndarray,
    edge_support_xy: np.ndarray,
    begin: int,
    end: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Materialize a disjoint immutable edge range on one GPU."""

    if int(end) <= int(begin) or int(batch_size) <= 0:
        raise ValueError("absolute-phase partition is invalid")
    started = time.monotonic()
    device = torch.device(device_name)
    source_by_name = _source_by_name(sources)
    profile_slices = profile_feature_slices()
    feature_count = len(FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES)
    if (
        feature_count != len(FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES)
        or max(item.stop for item in profile_slices.values()) != feature_count
    ):
        raise RuntimeError("absolute-phase feature layout drifted")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad():
        tensors = {
            name: (
                torch.from_numpy(np.asarray(source.grids)).to(device=device, dtype=torch.float16),
                torch.from_numpy(np.asarray(source.image_sizes, dtype=np.int64)).to(
                    device=device, dtype=torch.float32
                ),
            )
            for name, source in source_by_name.items()
        }
        query_rows = torch.as_tensor(query_cache_rows, device=device, dtype=torch.long)
        query_coordinates = torch.as_tensor(query_xy, device=device, dtype=torch.float32)
        query_context: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for profile in ABSOLUTE_PHASE_VISUAL_PROFILES:
            grids, sizes = tensors[profile.source_name]
            selected = grids.index_select(0, query_rows)
            mask = absolute_phase_anchor_mask(
                image_sizes=sizes,
                image_indices=query_rows,
                xy=query_coordinates,
                grid_size=int(profile.grid_size),
            )
            query_context[profile.name] = (selected, mask)
        score_output = np.empty((int(end) - int(begin), feature_count), dtype=np.float16)
        valid_output = np.ones(score_output.shape, dtype=bool)
        progress_interval = max(int(batch_size) * 128, int(batch_size))
        for offset in range(int(begin), int(end), int(batch_size)):
            stop = min(offset + int(batch_size), int(end))
            local_rows = torch.from_numpy(edge_rows[offset:stop]).to(device=device)
            support_rows = torch.from_numpy(edge_support_cache_rows[offset:stop]).to(
                device=device
            )
            support_coordinates = torch.from_numpy(edge_support_xy[offset:stop]).to(
                device=device, dtype=torch.float32
            )
            local_begin = offset - int(begin)
            local_end = stop - int(begin)
            for profile in ABSOLUTE_PHASE_VISUAL_PROFILES:
                grids, sizes = tensors[profile.source_name]
                query_grid, query_mask = query_context[profile.name]
                query_grid = query_grid.index_select(0, local_rows)
                query_mask = query_mask.index_select(0, local_rows)
                support_grid = grids.index_select(0, support_rows)
                support_mask = absolute_phase_anchor_mask(
                    image_sizes=sizes,
                    image_indices=support_rows,
                    xy=support_coordinates,
                    grid_size=int(profile.grid_size),
                )
                visual = batched_absolute_phase_region_transport_features(
                    query_grid,
                    query_mask,
                    support_grid,
                    support_mask,
                    region_grid_size=ABSOLUTE_PHASE_REGION_GRID_SIZE,
                    temperature=ABSOLUTE_PHASE_TEMPERATURE,
                )
                position = batched_absolute_phase_position_control_features(
                    query_mask,
                    support_mask,
                    region_grid_size=ABSOLUTE_PHASE_REGION_GRID_SIZE,
                )
                visual_slice = profile_slices[f"{profile.name}:visual"]
                position_slice = profile_slices[f"{profile.name}:position"]
                score_output[local_begin:local_end, visual_slice] = (
                    visual.cpu().numpy().astype(np.float16, copy=False)
                )
                score_output[local_begin:local_end, position_slice] = (
                    position.cpu().numpy().astype(np.float16, copy=False)
                )
            if (int(stop) - int(begin)) % progress_interval == 0 or int(stop) == int(end):
                print(
                    json.dumps(
                        {
                            "stage": "fulltrack_absolute_phase",
                            "device": str(device),
                            "completed": int(stop) - int(begin),
                            "assigned": int(end) - int(begin),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if np.any(~np.isfinite(score_output)):
        raise RuntimeError("absolute-phase exporter emitted non-finite features")
    return score_output, valid_output, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def build_frozen_fulltrack_per_view_absolute_phase(
    *,
    source_per_view_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_pca256_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    """Attach target-free full-image phase transport to one raw CSR shard."""

    if int(batch_size) <= 0:
        raise ValueError("absolute-phase batch size must be positive")
    selected_devices = tuple(str(value) for value in devices)
    if not selected_devices or len(set(selected_devices)) != len(selected_devices):
        raise ValueError("absolute-phase devices must be unique and non-empty")
    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    final_path = Path(radio_final_context_cache)
    intermediate_path = Path(radio_intermediate_pca256_context_cache)
    alike_path = Path(alike_spatial_context_cache)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite absolute-phase outputs")
    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("absolute-phase evidence requires real SfM observation xy")
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
    source_by_name = _source_by_name(sources)
    cache_ids = source_by_name["radio_final"].image_ids
    query_cache_rows = _cache_image_indices(
        cache_image_ids=cache_ids,
        image_ids=np.asarray(source["verification_query_ids"]).astype(str),
        context="absolute-phase frozen query",
    )
    edge_input = _edge_context_inputs(
        geometry=geometry,
        candidate_shape=tuple(edges.candidate_shape),
        geometry_rows=np.asarray(edges.geometry_rows, dtype=np.int64),
        edge_candidate_indices=np.asarray(edges.edge_candidate_indices, dtype=np.int64),
        cache_image_ids=cache_ids,
    )
    edge_count = int(edges.edge_count)
    if edge_count <= 0 or edge_count != len(edge_input["edge_rows"]):
        raise ValueError("absolute-phase artifact has no immutable CSR edges")
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
                edge_rows=edge_input["edge_rows"],
                edge_support_cache_rows=edge_input["support_cache_rows"],
                edge_support_xy=edge_input["support_xy"],
                begin=begin,
                end=end,
                batch_size=int(batch_size),
            )
            for device, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    scores = np.concatenate([item[0] for item in computed], axis=0)
    valid = np.concatenate([item[1] for item in computed], axis=0)
    expected_width = len(FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES)
    if scores.shape != (edge_count, expected_width) or valid.shape != scores.shape:
        raise RuntimeError("absolute-phase output shape drifted")
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
        raise RuntimeError("absolute-phase exporter changed CSR edge order")
    candidate = np.asarray(source["candidate_probabilities"], dtype=np.float32)
    if np.any((candidate > 0.0) & (edges.candidate_observation_counts <= 0)):
        raise RuntimeError("positive frozen candidate lost all support observations")
    query_id = str(source["verification_query_ids"][0])
    phase_contract = _absolute_phase_contract(sources)
    metadata: dict[str, Any] = {
        "format": FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
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
            "radio_final": file_sha256_short(final_path),
            "radio_intermediate_pca256": file_sha256_short(intermediate_path),
            "alike": file_sha256_short(alike_path),
        },
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_preserve_source_csr_order_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": phase_contract["profiles"],
        "absolute_phase_contract": phase_contract,
        "appearance_config": {
            "candidate_specific": True,
            "per_view": True,
            "feature_definition": "full_image_absolute_region_transport_v1",
            "support_view_marginalization": "not_aggregated_export_per_view_v1",
            "whole_image_retrieval_or_candidate_reselection": False,
            "visual_and_position_control_disjoint": True,
        },
        "strict_fulltrack_appearance_contract": {
            "candidate_identity_fixed": True,
            "candidate_posterior_preserved": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "support_view_count_cap": None,
            "support_view_features_averaged_before_inference": False,
            "candidate_3d_projection_or_pose_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "heldout_s0_verification_rows": True,
            "raw_summary_not_calibrated_likelihood": True,
            "source_fulltrack_csr_edges_preserved": True,
            "visual_feature_has_no_coverage_count_or_availability_field": True,
            "position_control_is_separate": True,
        },
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "absolute_phase_module_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm/localization/frozen_fulltrack_absolute_phase.py"
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
            profile_names=np.asarray(FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES, dtype=np.str_),
            edge_candidate_offsets=offsets,
            edge_geometry_rows=np.asarray(edges.geometry_rows, dtype=np.int64),
            edge_profile_scores=scores,
            edge_profile_valid=valid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output_path)
    summary = {
        "stage": "build_frozen_fulltrack_per_view_absolute_phase",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "edge_count": edge_count,
        "feature_count": expected_width,
        "visual_feature_count": int(
            sum(len(ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE[p.name]) for p in ABSOLUTE_PHASE_VISUAL_PROFILES)
        ),
        "position_control_feature_count": int(
            sum(len(ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE[p.name]) for p in ABSOLUTE_PHASE_VISUAL_PROFILES)
        ),
        "workers": [item[2] for item in computed],
        "runtime_seconds": float(time.monotonic() - started),
        "protocol": {
            "target_free": True,
            "identity_or_pose_targets_loaded": False,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "support_view_features_averaged_before_inference": False,
            "visual_feature_has_no_coverage_count_or_availability_field": True,
            "position_control_is_separate": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_frozen_fulltrack_per_view_absolute_phase(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            args.radio_intermediate_pca256_context_cache
        ),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        devices=_parse_devices(args.devices),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
