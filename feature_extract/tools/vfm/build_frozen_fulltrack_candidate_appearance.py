"""Export target-free appearance summaries over every SfM support observation.

This is a diagnostic S1 probe, not a production scorer.  It starts from an
already frozen S0 per-query appearance artifact, preserves its query anchors,
top-20 tracks, and top-L-plus-null posterior byte-for-byte, then replaces only
the maplet's bounded support-view list with the complete set of real mapping
observations for each candidate track.  It never searches images, selects a
support subset, reads a pose/target, or emits a pose score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.build_frozen_multiscale_candidate_appearance import (
    AppearanceProfile,
    _as_image_grid_sources,
    parse_profiles,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance import (
    aligned_patch_ncc,
)
from feature_extract.vfm.localization.full_track_support_view_probe import (
    FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT,
    FULL_TRACK_VIEW_STATISTIC_NAMES,
    FullTrackCandidateEdges,
    aggregate_full_track_view_scores,
    build_full_track_candidate_edges,
    build_track_observation_lookup,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ImageGridFeatureSource,
)


ARTIFACT_FORMAT = FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT
ARTIFACT_VERSION = "frozen_fulltrack_candidate_appearance_summary_v1"
PER_VIEW_ARTIFACT_FORMAT = "frozen_fulltrack_candidate_per_view_appearance_v1"
PER_VIEW_ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_appearance_v1"
_SOURCE_FORMAT = "frozen_multiscale_candidate_absolute_appearance_v1"
_RADIO_INTERMEDIATE_PCA_FORMAT = "radio_intermediate_image_context_pca_v1"
_INTERMEDIATE_PCA_SHARED_METADATA_FIELDS = (
    "format",
    "source_context_sha256",
    "source_image_manifest_sha256",
    "pca_training_manifest_sha256",
    "pca_training_image_list_sha256",
    "pca_training_image_count",
    "pca_fit_scope",
    "radio_checkpoint_sha256",
    "radio_version",
    "intermediate_index",
    "normalization",
    "spatial_grid_sizes",
    "pose_or_ground_truth_used",
    "image_retrieval_or_submap_used",
    "render",
)
_DEFAULT_PROFILES = (
    "radio_final_center:radio_final:1;"
    "radio_final_context3:radio_final:3;"
    "radio_final_context5:radio_final:5;"
    "radio_intermediate_center:radio_intermediate:1;"
    "radio_intermediate_context5:radio_intermediate:5;"
    "radio_intermediate_context9:radio_intermediate:9;"
    "radio_intermediate_context13:radio_intermediate:13;"
    "alike_center:alike:1;"
    "alike_context3:alike:3;"
    "alike_context5:alike:5;"
    "alike_context9:alike:9"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-appearance-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--profiles", default=_DEFAULT_PROFILES)
    parser.add_argument("--template-batch-size", type=int, default=8192)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--retain-per-view-edges",
        action="store_true",
        help=(
            "diagnostic-only: retain every real SfM support-observation NCC edge "
            "instead of only its candidate-level summary"
        ),
    )
    parser.add_argument(
        "--allow-radio-intermediate-projection-override",
        action="store_true",
        help=(
            "diagnostic-only: permit a higher-dimensional RADIO-intermediate PCA "
            "cache only when its source cache and PCA training lineage exactly match "
            "the frozen source artifact"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _source_cache_lineage(
    metadata: Mapping[str, Any], *, cache_name: str, cache_path: Path
) -> None:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("source appearance artifact lacks input lineage")
    source = inputs.get(cache_name)
    if not isinstance(source, Mapping):
        raise ValueError(f"source appearance artifact lacks {cache_name} lineage")
    if str(source.get("sha256", "")) != file_sha256_short(cache_path):
        raise ValueError(f"source appearance artifact and {cache_name} differ")


def _radio_intermediate_pca_contract(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Load only immutable lineage fields needed for a PCA-dimension sweep."""

    with np.load(Path(path), allow_pickle=False) as data:
        required = {"image_ids", "image_sizes", "metadata_json"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"RADIO-intermediate PCA cache lacks {sorted(missing)}")
        metadata = _metadata(data, context="RADIO-intermediate PCA cache")
        image_ids = np.asarray(data["image_ids"]).astype(str).copy()
        image_sizes = np.asarray(data["image_sizes"], dtype=np.int64).copy()
    if metadata.get("format") != _RADIO_INTERMEDIATE_PCA_FORMAT:
        raise ValueError("unsupported RADIO-intermediate PCA cache format")
    return metadata, image_ids, image_sizes


def _validate_radio_intermediate_projection_override(
    source_metadata: Mapping[str, Any], *, cache_path: Path
) -> dict[str, Any]:
    """Allow only a strictly equivalent higher-dimensional PCA representation."""

    inputs = source_metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("source appearance artifact lacks input lineage")
    source = inputs.get("radio_intermediate_context_cache")
    if not isinstance(source, Mapping):
        raise ValueError("source appearance artifact lacks intermediate PCA lineage")
    source_path_value = source.get("path")
    if not isinstance(source_path_value, str) or not source_path_value:
        raise ValueError("source appearance artifact lacks intermediate PCA cache path")
    source_path = Path(source_path_value)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if not source_path.is_file():
        raise FileNotFoundError(
            "source intermediate PCA cache is unavailable for a projection override"
        )
    if str(source.get("sha256", "")) != file_sha256_short(source_path):
        raise ValueError("source intermediate PCA cache is stale")
    source_metadata_pca, source_ids, source_sizes = _radio_intermediate_pca_contract(
        source_path
    )
    override_metadata_pca, override_ids, override_sizes = _radio_intermediate_pca_contract(
        Path(cache_path)
    )
    mismatches = {
        field: {
            "source": source_metadata_pca.get(field),
            "override": override_metadata_pca.get(field),
        }
        for field in _INTERMEDIATE_PCA_SHARED_METADATA_FIELDS
        if source_metadata_pca.get(field) != override_metadata_pca.get(field)
    }
    if mismatches or not np.array_equal(source_ids, override_ids) or not np.array_equal(
        source_sizes, override_sizes
    ):
        raise ValueError(
            "RADIO-intermediate PCA override changes source/image/PCA-fit lineage: "
            + json.dumps(mismatches, sort_keys=True)
        )
    source_dim = int(source_metadata_pca.get("projection_dim", 0))
    override_dim = int(override_metadata_pca.get("projection_dim", 0))
    if source_dim <= 0 or override_dim <= source_dim:
        raise ValueError(
            "RADIO-intermediate PCA override must increase projection dimension"
        )
    return {
        "enabled": True,
        "source_cache": str(source_path),
        "source_cache_sha256": file_sha256_short(source_path),
        "source_projection_dim": source_dim,
        "override_cache": str(Path(cache_path)),
        "override_cache_sha256": file_sha256_short(Path(cache_path)),
        "override_projection_dim": override_dim,
        "same_source_context_sha256": source_metadata_pca["source_context_sha256"],
        "same_pca_training_manifest_sha256": source_metadata_pca[
            "pca_training_manifest_sha256"
        ],
    }


def load_frozen_source_appearance(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only the frozen, target-free handoff fields required by this probe."""

    required = {
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_view_weights",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"source appearance artifact lacks {sorted(missing)}")
        arrays = {
            name: np.asarray(data[name]).copy()
            for name in required
            if name != "metadata_json"
        }
        metadata = _metadata(data, context="source appearance artifact")
    strict = metadata.get("strict_frozen_appearance_contract")
    if (
        metadata.get("format") != _SOURCE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or not isinstance(strict, Mapping)
        or strict.get("candidate_identity_fixed") is not True
        or strict.get("candidate_reselection") is not False
        or strict.get("support_reselection") is not False
        or strict.get("candidate_3d_projection_or_pose_used") is not False
        or strict.get("image_retrieval_or_submap_used") is not False
        or strict.get("render") is not False
        or int(strict.get("fixed_candidate_top_k", -1)) != 20
        or strict.get("heldout_s0_verification_rows") is not True
    ):
        raise ValueError("source appearance artifact is not a strict target-free S0 handoff")
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    source_rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
    xy = np.asarray(arrays["verification_xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    maplet_weights = np.asarray(arrays["candidate_view_weights"], dtype=np.float32)
    count = len(query_ids)
    if (
        count != 192
        or len(set(query_ids.tolist())) != 1
        or len(set(split_names.tolist())) != 1
        or split_names[0] not in {"train", "validation", "test"}
        or len(np.unique(source_rows)) != count
        or xy.shape != (count, 2)
        or tracks.shape != (count, 20)
        or probabilities.shape != tracks.shape
        or null.shape != (count,)
        or maplet_weights.ndim != 3
        or maplet_weights.shape[:2] != tracks.shape
        or np.any(~np.isfinite(xy))
        or np.any(~np.isfinite(probabilities))
        or np.any(~np.isfinite(null))
        or np.any(~np.isfinite(maplet_weights))
        or np.any(probabilities < 0.0)
        or np.any(null < 0.0)
        or np.any(maplet_weights < 0.0)
        or np.max(np.abs(probabilities.sum(axis=1) + null - 1.0)) > 2e-5
        or np.any((probabilities > 0.0) & (tracks < 0))
    ):
        raise ValueError("source appearance artifact frozen rows are invalid")
    return {
        "verification_query_ids": query_ids,
        "split_names": split_names,
        "verification_source_row_indices": source_rows,
        "verification_xy": xy,
        "candidate_track_ids": tracks,
        "candidate_probabilities": probabilities,
        "null_probabilities": null,
        "candidate_view_weights": maplet_weights,
    }, metadata


def _geometry_image_indices(geometry: SupportObservationGeometryIndex) -> np.ndarray:
    counts = np.diff(np.asarray(geometry.image_offsets, dtype=np.int64))
    image_indices = np.repeat(np.arange(len(geometry.image_ids), dtype=np.int64), counts)
    if image_indices.shape != geometry.track_ids.shape:
        raise RuntimeError("support geometry image offsets are inconsistent")
    return image_indices


def score_full_track_profile_edges(
    *,
    profile: AppearanceProfile,
    source: ImageGridFeatureSource,
    query_id: str,
    query_xy: np.ndarray,
    geometry: SupportObservationGeometryIndex,
    geometry_image_indices: np.ndarray,
    edges: FullTrackCandidateEdges,
    template_batch_size: int,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
    device: torch.device,
    source_grid_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure one raw patch profile at all fixed candidate observations."""

    if int(template_batch_size) <= 0:
        raise ValueError("template batch size must be positive")
    if not (0.0 < float(minimum_support_fraction) <= 1.0) or not (
        0.0 < float(minimum_overlap_fraction) <= 1.0
    ):
        raise ValueError("full-track appearance coverage thresholds are invalid")
    points = np.asarray(query_xy, dtype=np.float32)
    if points.shape != (edges.candidate_shape[0], 2):
        raise ValueError("query points differ from full-track edge layout")
    geometry_rows = np.asarray(edges.geometry_rows, dtype=np.int64)
    if np.any(geometry_rows >= len(geometry.track_ids)):
        raise ValueError("full-track edge references a missing support observation")
    support_ids = np.asarray(geometry.image_ids, dtype=np.str_)[
        np.asarray(geometry_image_indices, dtype=np.int64)[geometry_rows]
    ]
    support_xy = np.asarray(geometry.xy, dtype=np.float32)[geometry_rows]
    point_indices = edges.point_indices
    scores = np.full((edges.edge_count,), np.nan, dtype=np.float32)
    usable = np.zeros((edges.edge_count,), dtype=bool)
    grid_cache = {} if source_grid_cache is None else source_grid_cache
    with torch.inference_mode():
        query_patches, query_valid = source.context_patches_torch(
            np.full((len(points),), str(query_id)),
            points,
            window_size=profile.window_size,
            device=device,
            source_grid_cache=grid_cache,
        )
        for begin in range(0, edges.edge_count, int(template_batch_size)):
            end = min(begin + int(template_batch_size), edges.edge_count)
            selected_points = torch.as_tensor(
                point_indices[begin:end], dtype=torch.long, device=device
            )
            support_patches, support_valid = source.context_patches_torch(
                support_ids[begin:end],
                support_xy[begin:end],
                window_size=profile.window_size,
                device=device,
                source_grid_cache=grid_cache,
            )
            evidence = aligned_patch_ncc(
                query_patches=query_patches.index_select(0, selected_points),
                support_patches=support_patches,
                query_valid=query_valid.index_select(0, selected_points),
                support_valid=support_valid,
                minimum_support_fraction=float(minimum_support_fraction),
                minimum_overlap_fraction=float(minimum_overlap_fraction),
            )
            scores[begin:end] = evidence.score.detach().cpu().numpy()
            usable[begin:end] = evidence.usable.detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if np.any(usable & ~np.isfinite(scores)):
        raise RuntimeError("a usable all-observation support view lacks a score")
    return scores, usable


def _feature_arrays(
    *,
    summary: object,
    profiles: Sequence[AppearanceProfile],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep this conversion beside the exporter so the persisted schema cannot
    # accidentally make coverage a trainable appearance score.
    statistics = getattr(summary, "statistics")
    usable_counts = np.asarray(getattr(summary, "usable_counts"), dtype=np.int64)
    usable_fractions = np.asarray(getattr(summary, "usable_fractions"), dtype=np.float32)
    profile_count = len(profiles)
    if usable_counts.shape[2] != profile_count:
        raise ValueError("full-track summary profile count differs from requested profiles")
    values: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    names: list[str] = []
    for profile_index, profile in enumerate(profiles):
        support = usable_counts[..., profile_index] > 0
        for statistic in FULL_TRACK_VIEW_STATISTIC_NAMES:
            field = np.asarray(statistics[statistic], dtype=np.float32)[..., profile_index]
            values.append(field)
            valid.append(support)
            names.append(f"{profile.name}__{statistic}")
    feature_values = np.stack(values, axis=2).astype(np.float32, copy=False)
    feature_valid = np.stack(valid, axis=2).astype(bool, copy=False)
    if np.any(~np.isfinite(feature_values[feature_valid])) or np.any(
        np.isfinite(feature_values[~feature_valid])
    ):
        raise RuntimeError("full-track feature conversion lost explicit missingness")
    return (
        feature_values,
        feature_valid,
        np.asarray(names, dtype=np.str_),
        usable_counts,
        usable_fractions,
    )


def build_frozen_fulltrack_candidate_appearance(
    *,
    source_appearance_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    profiles: str,
    template_batch_size: int,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
    device: str,
    output: Path,
    summary_json: Path,
    force: bool,
    retain_per_view_edges: bool = False,
    allow_radio_intermediate_projection_override: bool = False,
) -> dict[str, Any]:
    """Build one complete, target-free full-track support-view probe artifact."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite full-track appearance outputs")
    if int(template_batch_size) <= 0:
        raise ValueError("template batch size must be positive")
    device_value = torch.device(str(device))
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA full-track appearance probe requested without CUDA")
    started = time.monotonic()
    source, source_metadata = load_frozen_source_appearance(Path(source_appearance_artifact))
    _source_cache_lineage(
        source_metadata,
        cache_name="radio_final_context_cache",
        cache_path=Path(radio_final_context_cache),
    )
    intermediate_override: dict[str, Any] | None = None
    if bool(allow_radio_intermediate_projection_override):
        intermediate_override = _validate_radio_intermediate_projection_override(
            source_metadata,
            cache_path=Path(radio_intermediate_context_cache),
        )
    else:
        _source_cache_lineage(
            source_metadata,
            cache_name="radio_intermediate_context_cache",
            cache_path=Path(radio_intermediate_context_cache),
        )
    _source_cache_lineage(
        source_metadata,
        cache_name="alike_spatial_context_cache",
        cache_path=Path(alike_spatial_context_cache),
    )
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("full-track support geometry must use real SfM observation xy")
    sources = _as_image_grid_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_context_cache=Path(radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
        allow_mismatched_descriptor_dimensions=bool(intermediate_override),
    )
    resolved_profiles = parse_profiles(
        profiles,
        source_grid_sizes={name: source.grid_size for name, source in sources.items()},
    )
    lookup = build_track_observation_lookup(geometry)
    edges = build_full_track_candidate_edges(
        candidate_track_ids=source["candidate_track_ids"],
        candidate_probabilities=source["candidate_probabilities"],
        lookup=lookup,
    )
    geometry_image_indices = _geometry_image_indices(geometry)
    score_columns: list[np.ndarray] = []
    usable_columns: list[np.ndarray] = []
    query_id = str(source["verification_query_ids"][0])
    source_grid_caches: dict[str, dict[str, torch.Tensor]] = {
        name: {} for name in sources
    }
    for profile in resolved_profiles:
        scores, usable = score_full_track_profile_edges(
            profile=profile,
            source=sources[profile.source_name],
            query_id=query_id,
            query_xy=source["verification_xy"],
            geometry=geometry,
            geometry_image_indices=geometry_image_indices,
            edges=edges,
            template_batch_size=int(template_batch_size),
            minimum_support_fraction=float(minimum_support_fraction),
            minimum_overlap_fraction=float(minimum_overlap_fraction),
            device=device_value,
            source_grid_cache=source_grid_caches[profile.source_name],
        )
        score_columns.append(scores)
        usable_columns.append(usable)
    per_edge_scores = np.stack(score_columns, axis=1).astype(np.float32, copy=False)
    per_edge_usable = np.stack(usable_columns, axis=1).astype(bool, copy=False)
    edge_offsets = np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(
                edges.candidate_observation_counts.reshape(-1), dtype=np.int64
            ),
        )
    )
    if (
        int(edge_offsets[-1]) != int(edges.edge_count)
        or not np.array_equal(
            np.repeat(
                np.arange(
                    len(edges.candidate_observation_counts.reshape(-1)),
                    dtype=np.int64,
                ),
                np.diff(edge_offsets),
            ),
            edges.edge_candidate_indices,
        )
    ):
        raise RuntimeError("full-track edge offsets are not candidate-major")
    all_view_summary = aggregate_full_track_view_scores(
        edges=edges,
        scores=per_edge_scores,
        usable=per_edge_usable,
    )
    (
        feature_values,
        feature_valid,
        feature_names,
        usable_counts,
        usable_fractions,
    ) = _feature_arrays(summary=all_view_summary, profiles=resolved_profiles)
    source_maplet_counts = np.sum(
        np.asarray(source["candidate_view_weights"], dtype=np.float32) > 0.0,
        axis=2,
        dtype=np.int64,
    )
    if np.any(
        (source["candidate_probabilities"] > 0.0)
        & (edges.candidate_observation_counts <= 0)
    ):
        raise RuntimeError("positive frozen candidate lost all support observations")
    metadata: dict[str, Any] = {
        "format": (
            PER_VIEW_ARTIFACT_FORMAT
            if bool(retain_per_view_edges)
            else ARTIFACT_FORMAT
        ),
        "version": (
            PER_VIEW_ARTIFACT_VERSION
            if bool(retain_per_view_edges)
            else ARTIFACT_VERSION
        ),
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
        "source_frozen_appearance_artifact": str(Path(source_appearance_artifact)),
        "source_frozen_appearance_artifact_sha256": file_sha256_short(
            Path(source_appearance_artifact)
        ),
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
        "source_strict_frozen_appearance_contract": source_metadata.get(
            "strict_frozen_appearance_contract"
        ),
        "support_geometry_index": str(Path(support_geometry_index)),
        "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_enumerate_all_track_observations_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "support_view_marginalization": "uncalibrated_uniform_all_observation_summary_v1",
        "per_view_edges_retained": bool(retain_per_view_edges),
        "per_view_edge_feature_semantics": (
            "raw_aligned_ncc_per_real_sfm_observation_v1"
            if bool(retain_per_view_edges)
            else "not_materialized"
        ),
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": [
            {
                "name": profile.name,
                "source": profile.source_name,
                "window_size": int(profile.window_size),
            }
            for profile in resolved_profiles
        ],
        "summary_statistics": list(FULL_TRACK_VIEW_STATISTIC_NAMES),
        "appearance_config": {
            "descriptor_sampling": "bilinear_align_corners_true_real_image_grid_v1",
            "minimum_support_fraction": float(minimum_support_fraction),
            "minimum_overlap_fraction": float(minimum_overlap_fraction),
            "template_batch_size": int(template_batch_size),
            "soft_best_temperature": 0.05,
            "top_view_count": 4,
        },
        "context_cache_sha256": {
            "radio_final": file_sha256_short(Path(radio_final_context_cache)),
            "radio_intermediate": file_sha256_short(Path(radio_intermediate_context_cache)),
            "alike": file_sha256_short(Path(alike_spatial_context_cache)),
        },
        "radio_intermediate_projection_override": intermediate_override
        if intermediate_override is not None
        else {"enabled": False},
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
            "raw_summary_not_calibrated_likelihood": True,
            "per_view_edges_retained": bool(retain_per_view_edges),
        },
        "implementation": {
            "script_sha256": file_sha256_short(Path(__file__)),
            "view_probe_module_sha256": file_sha256_short(
                Path("feature_extract/vfm/localization/full_track_support_view_probe.py")
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "verification_query_ids": source["verification_query_ids"],
        "split_names": source["split_names"],
        "verification_source_row_indices": source["verification_source_row_indices"],
        "verification_xy": source["verification_xy"],
        "candidate_track_ids": source["candidate_track_ids"],
        "candidate_probabilities": source["candidate_probabilities"],
        "null_probabilities": source["null_probabilities"],
        "candidate_support_observation_counts": edges.candidate_observation_counts,
        "source_maplet_support_view_counts": source_maplet_counts,
        "candidate_summary_features": feature_values,
        "candidate_summary_feature_valid": feature_valid,
        "feature_names": feature_names,
        "profile_names": np.asarray(
            [profile.name for profile in resolved_profiles], dtype=np.str_
        ),
        "candidate_profile_usable_counts": usable_counts,
        "candidate_profile_usable_fractions": usable_fractions,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if bool(retain_per_view_edges):
        payload.update(
            {
                "edge_candidate_offsets": edge_offsets,
                "edge_geometry_rows": np.asarray(
                    edges.geometry_rows, dtype=np.int64
                ),
                "edge_profile_scores": per_edge_scores.astype(
                    np.float16, copy=False
                ),
                "edge_profile_valid": per_edge_usable,
            }
        )
    with output.with_name(output.name + ".tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            **payload,
        )
    output.with_name(output.name + ".tmp").replace(output)
    summary = {
        "stage": "build_frozen_fulltrack_candidate_appearance",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": metadata["format"],
        "row_count": int(len(source["verification_query_ids"])),
        "candidate_edge_count": int(edges.edge_count),
        "source_maplet_support_edge_count": int(np.sum(source_maplet_counts)),
        "candidate_entries_expanded_beyond_maplet": int(
            np.sum(edges.candidate_observation_counts > source_maplet_counts)
        ),
        "candidate_observation_count_quantiles": {
            "p50": float(np.quantile(edges.candidate_observation_counts, 0.5)),
            "p90": float(np.quantile(edges.candidate_observation_counts, 0.9)),
            "p99": float(np.quantile(edges.candidate_observation_counts, 0.99)),
            "max": int(np.max(edges.candidate_observation_counts)),
        },
        "feature_count": int(feature_values.shape[2]),
        "per_view_edge_count": int(edges.edge_count) if bool(retain_per_view_edges) else 0,
        "per_view_edges_retained": bool(retain_per_view_edges),
        "runtime_seconds": float(time.monotonic() - started),
        "device": str(device_value),
        "protocol": {
            "image_retrieval_or_submap_used": False,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "pose_or_ground_truth_used": False,
            "render": False,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_frozen_fulltrack_candidate_appearance(
        source_appearance_artifact=Path(args.source_appearance_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        profiles=str(args.profiles),
        template_batch_size=int(args.template_batch_size),
        minimum_support_fraction=float(args.minimum_support_fraction),
        minimum_overlap_fraction=float(args.minimum_overlap_fraction),
        device=str(args.device),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
        retain_per_view_edges=bool(args.retain_per_view_edges),
        allow_radio_intermediate_projection_override=bool(
            args.allow_radio_intermediate_projection_override
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
