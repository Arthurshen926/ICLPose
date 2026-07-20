"""Attach dense multiscale translation-mode evidence to immutable CSR edges.

This S1 exporter uses only real query/support image feature maps.  For each
already frozen ``(query point, candidate track, real support observation)``
edge, it retains a bounded relative-mode summary from RADIO-final, RADIO
intermediate PCA256, and dense ALIKE/FPN features.  It never projects a
candidate with a pose, rebuilds candidates, selects support views, reads
targets, or averages support evidence before the downstream per-view mixture.
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
    _load_sources,
)
from feature_extract.tools.vfm.build_alike_image_spatial_context_cache import (
    ARTIFACT_FORMAT as ALIKE_SPATIAL_CONTEXT_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_multiscale_translation_mode import (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE,
    MULTISCALE_TRANSLATION_MODE_PROFILES,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    batched_dense_local_translation_mode_features,
)
from feature_extract.vfm.localization.spatial_image_context import (
    load_spatial_image_context_cache,
)


ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_multiscale_translation_mode_v1"
_TEMPERATURE = 0.07


@dataclass(frozen=True)
class _TranslationSource:
    name: str
    grid_size: int
    image_ids: np.ndarray
    image_sizes: np.ndarray
    grids: np.ndarray
    metadata: Mapping[str, object]
    cache_path: Path

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.image_ids).astype(str).reshape(-1)
        image_sizes = np.asarray(self.image_sizes, dtype=np.int64)
        grids = np.asarray(self.grids, dtype=np.float32)
        if (
            not str(self.name)
            or int(self.grid_size) <= 0
            or image_sizes.shape != (len(image_ids), 2)
            or grids.ndim != 4
            or grids.shape[:3]
            != (len(image_ids), int(self.grid_size), int(self.grid_size))
            or not len(image_ids)
            or len(set(image_ids.tolist())) != len(image_ids)
            or np.any(image_sizes <= 1)
            or np.any(~np.isfinite(grids))
        ):
            raise ValueError("multiscale translation-mode source is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "grid_size", int(self.grid_size))
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_sizes", image_sizes)
        object.__setattr__(self, "grids", grids)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "cache_path", Path(self.cache_path))


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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _edge_offsets(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("candidate support observation counts are invalid")
    return np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(values.reshape(-1), dtype=np.int64))
    )


def _source_metadata(source: _TranslationSource) -> dict[str, object]:
    metadata = dict(source.metadata)
    return {
        "name": source.name,
        "cache": str(source.cache_path),
        "cache_sha256": file_sha256_short(source.cache_path),
        "format": metadata.get("format"),
        "grid_size": int(source.grid_size),
        "descriptor_dim": int(source.grids.shape[-1]),
        "source_image_manifest_sha256": metadata.get("source_image_manifest_sha256"),
        "radio_checkpoint_sha256": metadata.get("radio_checkpoint_sha256"),
        "alike_checkpoint_sha256": metadata.get("alike_checkpoint_sha256"),
        "pca_fit_scope": metadata.get("pca_fit_scope"),
        "pca_training_manifest_sha256": metadata.get("pca_training_manifest_sha256"),
        "intermediate_index": metadata.get("intermediate_index"),
    }


def _load_translation_sources(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_pca256_context_cache: Path,
    alike_spatial_context_cache: Path,
) -> tuple[_TranslationSource, ...]:
    # The shared loader validates checkpoint, image-manifest, PCA-fit, and
    # source-image contracts for final/intermediate/ALIKE before this builder
    # asks ALIKE for its additional dense grid32 representation.
    base = _load_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_context_cache=Path(radio_intermediate_pca256_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
    )
    final, intermediate, alike32 = base
    alike = load_spatial_image_context_cache(
        Path(alike_spatial_context_cache), expected_format=ALIKE_SPATIAL_CONTEXT_FORMAT
    )
    if (
        not np.array_equal(alike.image_ids.astype(str), final.image_ids.astype(str))
        or not np.array_equal(alike.image_sizes, final.image_sizes)
    ):
        raise ValueError("ALIKE dense source does not align with RADIO-final")
    sources = (
        _TranslationSource(
            name="radio_final",
            grid_size=int(final.profile.grid_size),
            image_ids=final.image_ids,
            image_sizes=final.image_sizes,
            grids=final.grids,
            metadata=final.metadata,
            cache_path=final.cache_path,
        ),
        _TranslationSource(
            name="radio_intermediate_pca256",
            grid_size=int(intermediate.profile.grid_size),
            image_ids=intermediate.image_ids,
            image_sizes=intermediate.image_sizes,
            grids=intermediate.grids,
            metadata=intermediate.metadata,
            cache_path=intermediate.cache_path,
        ),
        _TranslationSource(
            name="alike_fpn",
            grid_size=32,
            image_ids=alike.image_ids,
            image_sizes=alike.image_sizes,
            grids=alike.grid_descriptors(32).reshape(
                len(alike.image_ids), 32, 32, alike.descriptor_dim
            ),
            metadata=alike.metadata,
            cache_path=Path(alike_spatial_context_cache),
        ),
    )
    if alike32.profile.name != "alike":  # pragma: no cover - loader invariant
        raise RuntimeError("shared ALIKE source name drifted")
    expected = {profile.source_name for profile in MULTISCALE_TRANSLATION_MODE_PROFILES}
    by_name = {source.name: source for source in sources}
    if set(by_name) != expected:
        raise RuntimeError("translation-mode source names drifted")
    for profile in MULTISCALE_TRANSLATION_MODE_PROFILES:
        source = by_name[profile.source_name]
        if int(source.grid_size) != int(profile.grid_size):
            raise ValueError(f"{profile.name}: feature grid size differs from profile")
    return sources


def _source_by_name(sources: Sequence[_TranslationSource]) -> dict[str, _TranslationSource]:
    output = {source.name: source for source in sources}
    expected = {profile.source_name for profile in MULTISCALE_TRANSLATION_MODE_PROFILES}
    if set(output) != expected:
        raise ValueError("translation-mode sources are incomplete or duplicated")
    return output


def _crop_grid_torch(
    *,
    image_grids: torch.Tensor,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    window_size: int,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample one dense crop and retain its original in-image mask."""

    grids = torch.as_tensor(image_grids)
    sizes = torch.as_tensor(image_sizes, dtype=torch.float32, device=grids.device)
    indices = torch.as_tensor(image_indices, dtype=torch.long, device=grids.device)
    coordinates = torch.as_tensor(xy, dtype=torch.float32, device=grids.device)
    window = int(window_size)
    if (
        grids.ndim != 4
        or grids.shape[1] != grids.shape[2]
        or sizes.shape != (grids.shape[0], 2)
        or indices.ndim != 1
        or coordinates.shape != (len(indices), 2)
        or window <= 0
        or window % 2 != 1
        or window > int(grids.shape[1])
        or str(padding_mode) not in {"zeros", "border", "reflection"}
    ):
        raise ValueError("translation-mode dense crop inputs are invalid")
    grid_size = int(grids.shape[1])
    selected_sizes = sizes.index_select(0, indices)
    center_x = coordinates[:, 0] * float(grid_size - 1) / selected_sizes[:, 0].sub(1.0)
    center_y = coordinates[:, 1] * float(grid_size - 1) / selected_sizes[:, 1].sub(1.0)
    radius = window // 2
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
    crop = F.grid_sample(
        selected,
        normalized,
        mode="bilinear",
        padding_mode=str(padding_mode),
        align_corners=True,
    ).permute(0, 2, 3, 1)
    crop = F.normalize(crop, p=2, dim=3, eps=1e-8)
    return crop, valid


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
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Materialize a disjoint immutable edge range on one device."""

    started = time.monotonic()
    device = torch.device(device_name)
    source_by_name = _source_by_name(sources)
    feature_count = len(FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES)
    profile_offsets = _profile_feature_slices()
    if max(item.stop for item in profile_offsets.values()) != feature_count:
        raise RuntimeError("translation-mode profile feature offsets drifted")
    with torch.no_grad():
        tensors = {
            name: (
                torch.from_numpy(source.grids).to(device=device, dtype=torch.float32),
                torch.from_numpy(source.image_sizes).to(device=device, dtype=torch.float32),
            )
            for name, source in source_by_name.items()
        }
        query_context = {
            profile.name: _crop_grid_torch(
                image_grids=tensors[profile.source_name][0],
                image_sizes=tensors[profile.source_name][1],
                image_indices=torch.as_tensor(query_cache_rows, device=device),
                xy=torch.as_tensor(query_xy, device=device),
                window_size=int(profile.window_size),
            )
            for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
        }
        scores = np.full((int(end) - int(begin), feature_count), np.nan, dtype=np.float16)
        valid = np.zeros(scores.shape, dtype=bool)
        profile_coverage: dict[str, list[float]] = {
            profile.name: [] for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
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
            for profile in MULTISCALE_TRANSLATION_MODE_PROFILES:
                source_tensors = tensors[profile.source_name]
                support_crop, support_mask = _crop_grid_torch(
                    image_grids=source_tensors[0],
                    image_sizes=source_tensors[1],
                    image_indices=support_rows,
                    xy=support_xy,
                    window_size=int(profile.window_size),
                )
                query_crop, query_mask = query_context[profile.name]
                query_crop = query_crop.index_select(0, point_rows)
                query_mask = query_mask.index_select(0, point_rows)
                center = int(profile.window_size) // 2
                usable = query_mask[:, center, center] & support_mask[:, center, center]
                usable_np = usable.cpu().numpy().astype(bool, copy=False)
                profile_coverage[profile.name].append(float(np.mean(usable_np)))
                profile_slice = profile_offsets[profile.name]
                if np.any(usable_np):
                    values = batched_dense_local_translation_mode_features(
                        query_grid=query_crop[usable],
                        query_valid=query_mask[usable],
                        support_grid=support_crop[usable],
                        support_valid=support_mask[usable],
                        maximum_shift=int(profile.maximum_shift),
                        temperature=_TEMPERATURE,
                    )
                    values_np = values.cpu().numpy().astype(np.float16, copy=False)
                    local_scores[usable_np, profile_slice] = values_np
                    local_valid[usable_np, profile_slice] = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if np.any(np.isinf(scores)) or np.any(~np.isfinite(scores[valid])) or np.any(
        np.isfinite(scores[~valid])
    ):
        raise RuntimeError("translation-mode exporter emitted invalid missingness")
    return scores, valid, {
        "device": str(device),
        "edge_count": int(end) - int(begin),
        "elapsed_seconds": float(time.monotonic() - started),
        "profile_center_coverage": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in profile_coverage.items()
        },
    }


def _profile_feature_slices() -> dict[str, slice]:
    offset = 0
    output: dict[str, slice] = {}
    for profile in MULTISCALE_TRANSLATION_MODE_PROFILES:
        width = len(MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE[profile.name])
        output[profile.name] = slice(offset, offset + width)
        offset += width
    return output


def _context_source_contract(sources: Sequence[_TranslationSource]) -> dict[str, Any]:
    return {
        "mode": "candidate_specific_multiscale_dense_translation_mode_v1",
        "support_coordinate_source": "sfm_observation_xy",
        "view_aggregation": "none_before_learned_logsumexp_mixture_v1",
        "explicit_availability_or_neighbor_count_feature": False,
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "grid_size": int(profile.grid_size),
                "window_size": int(profile.window_size),
                "maximum_shift": int(profile.maximum_shift),
                "feature_names": list(
                    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE[profile.name]
                ),
            }
            for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
        ],
        "sources": [_source_metadata(source) for source in sources],
    }


def build_frozen_fulltrack_per_view_multiscale_translation_mode(
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
    """Copy one raw CSR shard and attach dense target-free visual modes."""

    if int(batch_size) <= 0:
        raise ValueError("translation-mode batch size must be positive")
    selected_devices = tuple(str(value) for value in devices)
    if not selected_devices or len(set(selected_devices)) != len(selected_devices):
        raise ValueError("translation-mode devices must be unique and non-empty")
    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    final_path = Path(radio_final_context_cache)
    intermediate_path = Path(radio_intermediate_pca256_context_cache)
    alike_path = Path(alike_spatial_context_cache)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite multiscale translation-mode outputs")
    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("translation-mode evidence requires real SfM observation xy")
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
        raise ValueError("translation-mode artifact has no immutable CSR edges")
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
    expected_width = len(FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES)
    if scores.shape != (edge_count, expected_width) or valid.shape != scores.shape:
        raise RuntimeError("translation-mode output shape drifted")
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
        raise RuntimeError("translation-mode exporter changed CSR edge order")
    candidate = np.asarray(source["candidate_probabilities"], dtype=np.float32)
    if np.any((candidate > 0.0) & (edges.candidate_observation_counts <= 0)):
        raise RuntimeError("positive frozen candidate lost all support observations")
    query_id = str(source["verification_query_ids"][0])
    source_contract = _context_source_contract(sources)
    metadata: dict[str, Any] = {
        "format": FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
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
        "per_view_edge_feature_semantics": (
            FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
        ),
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": source_contract["profiles"],
        "translation_mode_contract": {
            **source_contract,
            "temperature": _TEMPERATURE,
            "all_profiles_require_only_center_sample_validity": True,
        },
        "appearance_config": {
            "candidate_specific": True,
            "per_view": True,
            "feature_definition": "multiscale_dense_candidate_relative_translation_modes_v1",
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
            "explicit_availability_or_neighbor_count_is_not_a_learned_feature": True,
        },
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "translation_mode_core_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm/localization/frozen_fulltrack_multiscale_translation_mode.py"
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
                FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES,
                dtype=np.str_,
            ),
            edge_candidate_offsets=offsets,
            edge_geometry_rows=geometry_rows,
            edge_profile_scores=scores.astype(np.float16, copy=False),
            edge_profile_valid=valid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output_path)
    summary = {
        "stage": "build_frozen_fulltrack_per_view_multiscale_translation_mode",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "edge_count": edge_count,
        "feature_count": int(scores.shape[1]),
        "profile_edge_coverage": {
            profile.name: float(
                np.mean(np.all(valid[:, _profile_feature_slices()[profile.name]], axis=1))
            )
            for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
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
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frozen_fulltrack_per_view_multiscale_translation_mode(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            args.radio_intermediate_pca256_context_cache
        ),
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
