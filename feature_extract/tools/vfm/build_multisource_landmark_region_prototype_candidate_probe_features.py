"""Export fixed per-view multi-source landmark-centred region evidence.

The candidate tracks, top-L order, maplet support views and support
observations are all frozen inputs.  The only new values are local descriptor
comparisons between a query anchor and the corresponding real observation of
each candidate track in three separately manifested image spaces.
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
    _crop_region_prototypes_torch,
    _fixed_candidate_view_edges,
    _parse_devices,
    _query_region_prototypes_torch,
)
from feature_extract.tools.vfm.build_radio_intermediate_image_context_pca_cache import (
    PCA_FORMAT as RADIO_INTERMEDIATE_PCA_FORMAT,
)
from feature_extract.tools.vfm.build_alike_image_spatial_context_cache import (
    ARTIFACT_FORMAT as ALIKE_SPATIAL_CONTEXT_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_FINAL_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
)
from feature_extract.vfm.localization.radio_final_context import (
    RADIO_FINAL_CONTEXT_PCA_FORMAT,
    load_radio_final_context_pca_cache,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    load_spatial_image_context_cache,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    image_source_contract_signature,
    require_compatible_image_source_contracts,
)


ARTIFACT_FORMAT = MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT
_SAFE_PCA_FIT_SCOPES = frozenset(
    {
        "mapping_train_images_only",
        "mapping_support_images_excluding_all_query_splits_v1",
    }
)


@dataclass(frozen=True)
class _SourceProfile:
    name: str
    grid_size: int
    window_sizes: tuple[int, ...]
    context_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class _SourceArrays:
    profile: _SourceProfile
    image_ids: np.ndarray
    image_sizes: np.ndarray
    grids: np.ndarray
    metadata: Mapping[str, object]
    cache_path: Path


_SOURCE_PROFILES = (
    _SourceProfile(
        name="radio_final",
        grid_size=16,
        window_sizes=(7, 11),
        context_feature_names=MULTISOURCE_LANDMARK_REGION_FINAL_CONTEXT_FEATURE_NAMES,
    ),
    _SourceProfile(
        name="radio_intermediate",
        grid_size=16,
        window_sizes=(7, 11),
        context_feature_names=MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES,
    ),
    _SourceProfile(
        name="alike",
        grid_size=32,
        window_sizes=(7, 15),
        context_feature_names=MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES,
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _source_metadata(
    source: _SourceArrays,
) -> dict[str, object]:
    metadata = source.metadata
    return {
        "cache": str(source.cache_path),
        "cache_sha256": file_sha256_short(source.cache_path),
        "format": metadata.get("format"),
        "grid_size": int(source.profile.grid_size),
        "window_sizes": [int(value) for value in source.profile.window_sizes],
        "descriptor_dim": int(source.grids.shape[-1]),
        "source_image_manifest_sha256": metadata.get("source_image_manifest_sha256"),
        "radio_checkpoint_sha256": metadata.get("radio_checkpoint_sha256"),
        "alike_checkpoint_sha256": metadata.get("alike_checkpoint_sha256"),
        "pca_fit_scope": metadata.get("pca_fit_scope"),
        "pca_training_manifest_sha256": metadata.get("pca_training_manifest_sha256"),
        "intermediate_index": metadata.get("intermediate_index"),
    }


def _as_source(
    *,
    profile: _SourceProfile,
    cache_path: Path,
    image_ids: np.ndarray,
    image_sizes: np.ndarray,
    descriptors: np.ndarray,
    metadata: Mapping[str, object],
) -> _SourceArrays:
    values = np.asarray(descriptors, dtype=np.float32)
    expected = (
        len(image_ids),
        int(profile.grid_size) ** 2,
        int(values.shape[-1]),
    )
    if values.ndim != 3 or values.shape != expected:
        raise ValueError(f"{profile.name} spatial cache grid shape is incompatible")
    return _SourceArrays(
        profile=profile,
        image_ids=np.asarray(image_ids).astype(str),
        image_sizes=np.asarray(image_sizes, dtype=np.int64),
        grids=values.reshape(
            len(image_ids),
            int(profile.grid_size),
            int(profile.grid_size),
            values.shape[-1],
        ),
        metadata=dict(metadata),
        cache_path=Path(cache_path),
    )


def _load_sources(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
) -> tuple[_SourceArrays, ...]:
    final = load_radio_final_context_pca_cache(Path(radio_final_context_cache))
    if final.metadata.get("format") != RADIO_FINAL_CONTEXT_PCA_FORMAT:
        raise ValueError("unsupported RADIO-final spatial cache")
    if str(final.metadata.get("pca_fit_scope", "")) not in _SAFE_PCA_FIT_SCOPES:
        raise ValueError("RADIO-final spatial PCA has an unsupported training scope")
    if bool(final.metadata.get("image_retrieval_or_submap_used", True)) or bool(
        final.metadata.get("pose_or_ground_truth_used", True)
    ):
        raise ValueError("RADIO-final spatial cache violates the target-free protocol")
    if final.grid16_descriptors is None:
        raise ValueError("RADIO-final spatial cache lacks grid16")
    final_image_contract = dict(final.metadata.get("image_source_contract", {}))
    final_image_signature = image_source_contract_signature(final_image_contract)
    if str(final.metadata.get("source_image_manifest_sha256", "")) != str(
        final_image_signature["sampled_content_manifest_sha256"]
    ):
        raise ValueError("RADIO-final spatial cache has a stale image source manifest")
    intermediate = load_spatial_image_context_cache(
        Path(radio_intermediate_context_cache),
        expected_format=RADIO_INTERMEDIATE_PCA_FORMAT,
    )
    if str(intermediate.metadata.get("pca_fit_scope", "")) not in _SAFE_PCA_FIT_SCOPES:
        raise ValueError("RADIO-intermediate spatial PCA has an unsupported training scope")
    if int(intermediate.metadata.get("intermediate_index", 0)) != -6:
        raise ValueError("RADIO-intermediate spatial cache uses an unexpected layer")
    if str(intermediate.metadata.get("radio_checkpoint_sha256", "")) != str(
        final.metadata.get("radio_checkpoint_sha256", "")
    ):
        raise ValueError("RADIO final/intermediate spatial cache checkpoints differ")
    alike = load_spatial_image_context_cache(
        Path(alike_spatial_context_cache), expected_format=ALIKE_SPATIAL_CONTEXT_FORMAT
    )
    base_ids = np.asarray(final.image_ids).astype(str)
    base_sizes = np.asarray(final.image_sizes, dtype=np.int64)
    for name, cache in (("intermediate", intermediate), ("alike", alike)):
        if not np.array_equal(cache.image_ids.astype(str), base_ids) or not np.array_equal(
            cache.image_sizes, base_sizes
        ):
            raise ValueError(f"{name} spatial cache images do not align with RADIO-final")
        require_compatible_image_source_contracts(
            final_image_contract,
            dict(cache.metadata.get("image_source_contract", {})),
            context=f"RADIO-final and {name} spatial caches",
        )
        if str(cache.metadata.get("source_image_manifest_sha256", "")) != str(
            final_image_signature["sampled_content_manifest_sha256"]
        ):
            raise ValueError(f"{name} spatial cache has a stale image source manifest")
        if bool(cache.metadata.get("image_retrieval_or_submap_used", True)) or bool(
            cache.metadata.get("pose_or_ground_truth_used", True)
        ):
            raise ValueError(f"{name} spatial cache violates the target-free protocol")
    sources = (
        _as_source(
            profile=_SOURCE_PROFILES[0],
            cache_path=Path(radio_final_context_cache),
            image_ids=base_ids,
            image_sizes=base_sizes,
            descriptors=final.grid16_descriptors,
            metadata=final.metadata,
        ),
        _as_source(
            profile=_SOURCE_PROFILES[1],
            cache_path=Path(radio_intermediate_context_cache),
            image_ids=intermediate.image_ids,
            image_sizes=intermediate.image_sizes,
            descriptors=intermediate.grid_descriptors(16),
            metadata=intermediate.metadata,
        ),
        _as_source(
            profile=_SOURCE_PROFILES[2],
            cache_path=Path(alike_spatial_context_cache),
            image_ids=alike.image_ids,
            image_sizes=alike.image_sizes,
            descriptors=alike.grid_descriptors(32),
            metadata=alike.metadata,
        ),
    )
    return sources


def _compute_partition(
    *,
    device_name: str,
    sources: tuple[_SourceArrays, ...],
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
    context_width = len(MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES)
    with torch.no_grad():
        source_tensors = {
            source.profile.name: (
                torch.from_numpy(source.grids).to(device=device, dtype=torch.float32),
                torch.from_numpy(source.image_sizes).to(device=device, dtype=torch.float32),
            )
            for source in sources
        }
        query_context = {
            (source.profile.name, int(window_size)): _query_region_prototypes_torch(
                image_grids=source_tensors[source.profile.name][0],
                image_sizes=source_tensors[source.profile.name][1],
                query_cache_rows=query_cache_rows,
                query_xy=query_xy,
                batch_size=int(batch_size),
                grid_size=int(source.profile.grid_size),
                window_size=int(window_size),
            )
            for source in sources
            for window_size in source.profile.window_sizes
        }
        context = np.empty((int(end) - int(begin), context_width), dtype=np.float16)
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
            for source in sources:
                image_grids, image_sizes = source_tensors[source.profile.name]
                for window_size in source.profile.window_sizes:
                    query_values, query_present, query_cells = query_context[
                        (source.profile.name, int(window_size))
                    ]
                    support_values, support_present, support_cells = _crop_region_prototypes_torch(
                        image_grids=image_grids,
                        image_sizes=image_sizes,
                        image_indices=support_cache_rows,
                        xy=support_xy,
                        grid_size=int(source.profile.grid_size),
                        window_size=int(window_size),
                    )
                    query_values = query_values.index_select(0, query_rows)
                    query_present = query_present.index_select(0, query_rows)
                    query_cells = query_cells.index_select(0, query_rows)
                    scores = torch.sum(query_values * support_values, dim=2)
                    scores = torch.where(
                        query_present & support_present,
                        scores,
                        torch.full_like(scores, float("nan")),
                    )
                    common_fraction = (query_cells & support_cells).to(torch.float32).mean(
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


def build_multisource_landmark_region_prototype_candidate_probe_features(
    *,
    frozen_layout_features: Path,
    maplet_support_index: Path,
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
    if int(batch_size) <= 0:
        raise ValueError("multi-source landmark-region batch_size must be positive")
    selected_devices = tuple(str(device) for device in devices)
    if not selected_devices:
        raise ValueError("multi-source landmark-region needs at least one device")
    if (Path(output).exists() or Path(summary_json).exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite multi-source landmark-region outputs")
    started = time.monotonic()
    layout, layout_metadata = _load_frozen_layout(
        Path(frozen_layout_features),
        required_feature_names=("radio_final_anchor_cosine",),
    )
    expected_mapper_checkpoint = str(layout_metadata.get("mapper_checkpoint_sha256", ""))
    sources = _load_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_context_cache=Path(radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
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
        radio_image_ids=sources[0].image_ids,
    )
    query_cache_rows = _cache_image_indices(
        cache_image_ids=sources[0].image_ids,
        image_ids=np.asarray(layout["query_ids"]).astype(str),
        context="query",
    )
    candidate_anchor = _candidate_radio_final_anchor(layout)
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
                sources=sources,
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
    if context.shape != (edge_count, len(MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES)):
        raise RuntimeError("multi-source landmark-region context feature shape drifted")
    common_columns = np.asarray(
        [
            index
            for index, name in enumerate(MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES)
            if name.endswith("_common_cell_fraction")
        ],
        dtype=np.int64,
    )
    if (
        not len(common_columns)
        or np.any(np.isinf(context))
        or np.any(~np.isfinite(context[:, common_columns]))
        or np.any(~np.isfinite(context[:, [0, 11, 22, 33, 44, 55]]))
    ):
        raise RuntimeError("multi-source landmark-region emitted invalid pooled or coverage fields")
    features = np.full(
        (*view_valid.shape, len(MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES)),
        np.nan,
        dtype=np.float16,
    )
    flat = features.reshape(-1, features.shape[-1])
    flat[edge["flat_position"], 0] = candidate_anchor[
        edge["row"], edge["candidate"]
    ].astype(np.float16, copy=False)
    flat[edge["flat_position"], 1:] = context
    valid_features = features[view_valid]
    if np.any(np.isinf(valid_features)) or np.any(~np.isfinite(valid_features[:, 0])):
        raise RuntimeError("multi-source landmark-region valid rows are invalid")
    source_manifest = str(sources[0].metadata.get("source_image_manifest_sha256", ""))
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "feature_definition": "fixed_maplet_support8_per_view_multisource_landmark_centered_region_prototypes_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "render": False,
        "candidate_set": "frozen_global_top20_tracks",
        "support_view_selection": "fixed_maplet_coverage_rank_top8_v1",
        "support_view_marginalization": "per_view_features_preserved_for_later_log_mixture_v1",
        "source_image_manifest_sha256": source_manifest,
        "appearance_evidence_manifest": {
            "mode": "candidate_specific_landmark_centered_per_view_region_prototypes_v1",
            "support_coordinate_source": "sfm_observation_xy",
            "region_grid_size": 3,
            "sources": [_source_metadata(source) for source in sources],
            "feature_names": list(MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES),
        },
        "descriptor_feature_names": list(MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES),
        "feature_dtype": "float16_with_nan_only_for_boundary_missing_spatial_bins",
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
        "mapper_checkpoint_sha256": expected_mapper_checkpoint or None,
        "radio_backbone_checkpoint_sha256": sources[0].metadata.get(
            "radio_checkpoint_sha256"
        ),
        "alike_checkpoint_sha256": sources[2].metadata.get("alike_checkpoint_sha256"),
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
            feature_names=np.asarray(
                MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES, dtype=np.str_
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    summary = {
        "stage": "build_fixed_maplet_support8_multisource_landmark_region_features",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": ARTIFACT_FORMAT,
        "row_count": int(len(layout["source_row_indices"])),
        "feature_count": int(len(MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES)),
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
    summary = build_multisource_landmark_region_prototype_candidate_probe_features(
        frozen_layout_features=Path(args.frozen_layout_features),
        maplet_support_index=Path(args.maplet_support_index),
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
