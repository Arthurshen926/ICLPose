"""Export compact pose-free spatial-layout evidence on frozen full-track edges.

This S1 exporter intentionally consumes an existing raw-NCC full-track
artifact rather than rebuilding candidates or support observations.  It keeps
the exact frozen global top-20 rows and every real SfM observation edge, then
replaces each scalar NCC value with a low-frequency DCT of same-relative-cell
query/support agreement.  The representation is candidate-specific but uses
neither candidate coordinates, pose hypotheses, targets, image retrieval, nor
rendered imagery.
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
    _as_image_grid_sources,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    _radio_intermediate_projection_contract,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS,
    _load_artifact,
    load_frozen_fulltrack_per_view_appearance_features,
)
from feature_extract.vfm.localization.fulltrack_aligned_layout_probe import (
    ALIGNED_LAYOUT_FEATURE_NAMES,
    ALIGNED_LAYOUT_PROFILES,
    FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_FORMAT,
    FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_VERSION,
    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    aligned_layout_profile_feature_names,
    aligned_spatial_dct_features,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ImageGridFeatureSource,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-per-view-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--template-batch-size", type=int, default=2048)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _geometry_image_indices(geometry: SupportObservationGeometryIndex) -> np.ndarray:
    counts = np.diff(np.asarray(geometry.image_offsets, dtype=np.int64))
    indices = np.repeat(np.arange(len(geometry.image_ids), dtype=np.int64), counts)
    if indices.shape != np.asarray(geometry.track_ids).shape:
        raise RuntimeError("support geometry image offsets are inconsistent")
    return indices


def _source_per_view_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the immutable raw-edge lineage inherited by the layout export."""

    return {
        "format": metadata.get("format"),
        "version": metadata.get("version"),
        "support_geometry_index_sha256": metadata.get("support_geometry_index_sha256"),
        "context_cache_sha256": metadata.get("context_cache_sha256"),
        "per_view_edge_feature_semantics": metadata.get(
            "per_view_edge_feature_semantics"
        ),
        "radio_intermediate_projection_override": metadata.get(
            "radio_intermediate_projection_override"
        ),
        "strict_fulltrack_appearance_contract": metadata.get(
            "strict_fulltrack_appearance_contract"
        ),
    }


def _validate_source_lineage(
    *,
    metadata: Mapping[str, Any],
    source_path: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
) -> dict[str, Any]:
    """Reject stale caches and a raw/PCA layout semantic mix before export."""

    if (
        metadata.get("per_view_edge_feature_semantics")
        != FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS
    ):
        raise ValueError("aligned-layout export requires a raw-NCC full-track source")
    expected_geometry_hash = str(metadata.get("support_geometry_index_sha256", ""))
    if expected_geometry_hash != file_sha256_short(support_geometry_index):
        raise ValueError("source per-view artifact and support geometry index differ")
    source_hashes = metadata.get("context_cache_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("source per-view artifact lacks context-cache lineage")
    actual_hashes = {
        "radio_final": file_sha256_short(radio_final_context_cache),
        "radio_intermediate": file_sha256_short(radio_intermediate_context_cache),
        "alike": file_sha256_short(alike_spatial_context_cache),
    }
    if {str(key): str(value) for key, value in source_hashes.items()} != actual_hashes:
        raise ValueError("source per-view artifact and feature context caches differ")
    projection = _radio_intermediate_projection_contract(
        metadata, path=Path(source_path)
    )
    if projection["enabled"]:
        if projection["override_cache_sha256"] != actual_hashes["radio_intermediate"]:
            raise ValueError("intermediate PCA override cache differs from source contract")
    return projection


def _validate_source_edges(
    *,
    arrays: Mapping[str, np.ndarray],
    geometry: SupportObservationGeometryIndex,
) -> np.ndarray:
    """Recover immutable query-row indices and validate candidate-major CSR."""

    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    counts = np.asarray(
        arrays["candidate_support_observation_counts"], dtype=np.int64
    )
    offsets = np.asarray(arrays["edge_candidate_offsets"], dtype=np.int64).reshape(-1)
    geometry_rows = np.asarray(arrays["edge_geometry_rows"], dtype=np.int64).reshape(-1)
    if (
        tracks.shape != probabilities.shape
        or counts.shape != tracks.shape
        or offsets.shape != (tracks.size + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(geometry_rows)
        or not np.array_equal(np.diff(offsets), counts.reshape(-1))
        or np.any(geometry_rows < 0)
        or np.any(geometry_rows >= len(geometry.track_ids))
    ):
        raise ValueError("source per-view CSR edges are invalid")
    candidate_flat = np.repeat(
        np.arange(tracks.size, dtype=np.int64), np.diff(offsets)
    )
    if candidate_flat.shape != geometry_rows.shape:
        raise RuntimeError("source per-view CSR edge expansion is inconsistent")
    expected_tracks = tracks.reshape(-1)[candidate_flat]
    if not np.array_equal(
        expected_tracks, np.asarray(geometry.track_ids, dtype=np.int64)[geometry_rows]
    ):
        raise ValueError("source per-view edges no longer match their SfM track ids")
    if np.any((probabilities > 0.0) & (counts <= 0)):
        raise ValueError("a positive frozen candidate has no retained support edge")
    return candidate_flat // int(tracks.shape[1])


def score_aligned_layout_profile_edges(
    *,
    profile_name: str,
    source: ImageGridFeatureSource,
    query_id: str,
    query_xy: np.ndarray,
    support_ids: np.ndarray,
    support_xy: np.ndarray,
    point_indices: np.ndarray,
    window_size: int,
    dct_size: int,
    template_batch_size: int,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure one fixed DCT layout profile on every immutable support edge."""

    points = np.asarray(query_xy, dtype=np.float32)
    ids = np.asarray(support_ids).astype(str).reshape(-1)
    support = np.asarray(support_xy, dtype=np.float32)
    edge_points = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or support.shape != (len(ids), 2)
        or edge_points.shape != (len(ids),)
        or np.any((edge_points < 0) | (edge_points >= len(points)))
        or int(template_batch_size) <= 0
    ):
        raise ValueError(f"{profile_name}: aligned-layout profile inputs are invalid")
    feature_count = int(dct_size) ** 2
    scores = np.full((len(ids), feature_count), np.nan, dtype=np.float32)
    usable = np.zeros((len(ids),), dtype=bool)
    source_grid_cache: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        query_patches, query_valid = source.context_patches_torch(
            np.full((len(points),), str(query_id)),
            points,
            window_size=int(window_size),
            device=device,
            source_grid_cache=source_grid_cache,
        )
        for begin in range(0, len(ids), int(template_batch_size)):
            end = min(begin + int(template_batch_size), len(ids))
            selected = torch.as_tensor(
                edge_points[begin:end], dtype=torch.long, device=device
            )
            support_patches, support_valid = source.context_patches_torch(
                ids[begin:end],
                support[begin:end],
                window_size=int(window_size),
                device=device,
                source_grid_cache=source_grid_cache,
            )
            values, valid = aligned_spatial_dct_features(
                query_patches=query_patches.index_select(0, selected),
                support_patches=support_patches,
                query_valid=query_valid.index_select(0, selected),
                support_valid=support_valid,
                dct_size=int(dct_size),
                minimum_support_fraction=float(minimum_support_fraction),
                minimum_overlap_fraction=float(minimum_overlap_fraction),
            )
            values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
            valid_np = valid.detach().cpu().numpy().astype(bool, copy=False)
            batch_scores = scores[begin:end]
            batch_scores[valid_np] = values_np[valid_np]
            usable[begin:end] = valid_np
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    if np.any(~np.isfinite(scores[usable])) or np.any(np.isfinite(scores[~usable])):
        raise RuntimeError(f"{profile_name}: aligned-layout missingness semantics failed")
    return scores, usable


def build_frozen_fulltrack_aligned_layout_appearance(
    *,
    source_per_view_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    template_batch_size: int,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
    device: str,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    """Build one frozen aligned-layout artifact from one raw full-track shard."""

    output_path = Path(output)
    summary_path = Path(summary_json)
    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    final_cache = Path(radio_final_context_cache)
    intermediate_cache = Path(radio_intermediate_context_cache)
    alike_cache = Path(alike_spatial_context_cache)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite aligned-layout outputs")
    if (
        int(template_batch_size) <= 0
        or not 0.0 < float(minimum_support_fraction) <= 1.0
        or not 0.0 < float(minimum_overlap_fraction) <= 1.0
    ):
        raise ValueError("aligned-layout exporter arguments are invalid")
    device_value = torch.device(str(device))
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA aligned-layout export requested without CUDA")
    started = time.monotonic()

    raw_arrays, raw_metadata = _load_artifact(source_path)
    # Constructing this target-free object revalidates posterior mass, active
    # candidate support, and NaN/valid semantics rather than trusting the NPZ
    # merely because its header passed the source loader.
    source_features = load_frozen_fulltrack_per_view_appearance_features((source_path,))
    projection_contract = _validate_source_lineage(
        metadata=raw_metadata,
        source_path=source_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=final_cache,
        radio_intermediate_context_cache=intermediate_cache,
        alike_spatial_context_cache=alike_cache,
    )
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("aligned-layout export requires real SfM observation xy")
    point_indices = _validate_source_edges(arrays=raw_arrays, geometry=geometry)
    geometry_rows = np.asarray(raw_arrays["edge_geometry_rows"], dtype=np.int64)
    geometry_image_indices = _geometry_image_indices(geometry)
    support_ids = np.asarray(geometry.image_ids).astype(str)[
        geometry_image_indices[geometry_rows]
    ]
    support_xy = np.asarray(geometry.xy, dtype=np.float32)[geometry_rows]
    query_ids = np.asarray(raw_arrays["verification_query_ids"]).astype(str).reshape(-1)
    if len(set(query_ids.tolist())) != 1:
        raise ValueError("aligned-layout source must contain exactly one query image")

    sources = _as_image_grid_sources(
        radio_final_context_cache=final_cache,
        radio_intermediate_context_cache=intermediate_cache,
        alike_spatial_context_cache=alike_cache,
        allow_mismatched_descriptor_dimensions=bool(projection_contract["enabled"]),
    )
    score_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    profile_contracts: list[dict[str, Any]] = []
    for profile in ALIGNED_LAYOUT_PROFILES:
        source = sources.get(profile.source_name)
        if source is None or int(profile.window_size) > int(source.grid_size):
            raise ValueError(f"aligned-layout profile source is unavailable: {profile.name}")
        values, valid = score_aligned_layout_profile_edges(
            profile_name=profile.name,
            source=source,
            query_id=str(query_ids[0]),
            query_xy=np.asarray(raw_arrays["verification_xy"], dtype=np.float32),
            support_ids=support_ids,
            support_xy=support_xy,
            point_indices=point_indices,
            window_size=int(profile.window_size),
            dct_size=int(profile.dct_size),
            template_batch_size=int(template_batch_size),
            minimum_support_fraction=float(minimum_support_fraction),
            minimum_overlap_fraction=float(minimum_overlap_fraction),
            device=device_value,
        )
        if values.shape != (len(geometry_rows), int(profile.dct_size) ** 2):
            raise RuntimeError("aligned-layout profile output has the wrong width")
        score_blocks.append(values)
        valid_blocks.append(np.broadcast_to(valid[:, None], values.shape).copy())
        profile_contracts.append(
            {
                "name": profile.name,
                "source": profile.source_name,
                "window_size": int(profile.window_size),
                "dct_size": int(profile.dct_size),
                "feature_names": list(aligned_layout_profile_feature_names(profile)),
            }
        )
    edge_scores = np.concatenate(score_blocks, axis=1).astype(np.float32, copy=False)
    edge_valid = np.concatenate(valid_blocks, axis=1).astype(bool, copy=False)
    if (
        edge_scores.shape != (len(geometry_rows), len(ALIGNED_LAYOUT_FEATURE_NAMES))
        or edge_valid.shape != edge_scores.shape
        or np.any(~np.isfinite(edge_scores[edge_valid]))
        or np.any(np.isfinite(edge_scores[~edge_valid]))
    ):
        raise RuntimeError("aligned-layout export lost feature or missingness semantics")
    source_contract = _source_per_view_contract(raw_metadata)
    inherited_keys = (
        "format",
        "version",
        "support_geometry_index_sha256",
        "context_cache_sha256",
        "per_view_edge_feature_semantics",
    )
    if any(
        source_features.compatibility.get(key) != source_contract.get(key)
        for key in inherited_keys
    ):
        # The source loader has its own normalized compatibility object.  This
        # check catches an accidental raw/layout format handoff before output.
        raise RuntimeError("source full-track compatibility is internally inconsistent")
    metadata: dict[str, Any] = {
        "format": FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_FORMAT,
        "version": FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_VERSION,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "query_id": str(query_ids[0]),
        "split_name": str(np.asarray(raw_arrays["split_names"]).astype(str)[0]),
        "row_count": int(len(query_ids)),
        "query_count": 1,
        "source_frozen_fulltrack_per_view_artifact": str(source_path),
        "source_frozen_fulltrack_per_view_artifact_sha256": file_sha256_short(
            source_path
        ),
        "source_candidate_tracks_sha256": _array_sha256_short(
            raw_arrays["candidate_track_ids"]
        ),
        "source_candidate_probabilities_sha256": _array_sha256_short(
            raw_arrays["candidate_probabilities"]
        ),
        "source_null_probabilities_sha256": _array_sha256_short(
            raw_arrays["null_probabilities"]
        ),
        "source_verification_rows_sha256": _array_sha256_short(
            raw_arrays["verification_source_row_indices"]
        ),
        "source_edge_offsets_sha256": _array_sha256_short(
            raw_arrays["edge_candidate_offsets"]
        ),
        "source_edge_geometry_rows_sha256": _array_sha256_short(
            raw_arrays["edge_geometry_rows"]
        ),
        "source_csr_array_hash_scheme": "dtype_shape_bytes_sha256_v1",
        "source_per_view_contract": source_contract,
        "support_geometry_index": str(geometry_path),
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_enumerate_all_track_observations_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "support_view_marginalization": "not_applied_raw_per_observation_edges_v1",
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": profile_contracts,
        "appearance_config": {
            "descriptor_sampling": "bilinear_align_corners_true_real_image_grid_v1",
            "minimum_support_fraction": float(minimum_support_fraction),
            "minimum_overlap_fraction": float(minimum_overlap_fraction),
            "template_batch_size": int(template_batch_size),
            "layout_representation": "same_relative_cell_cosine_centered_low_frequency_dct_v1",
        },
        "context_cache_sha256": {
            "radio_final": file_sha256_short(final_cache),
            "radio_intermediate": file_sha256_short(intermediate_cache),
            "alike": file_sha256_short(alike_cache),
        },
        "radio_intermediate_projection_override": projection_contract,
        "aligned_layout_contract": {
            "representation": "same_relative_cell_cosine_centered_low_frequency_dct_v1",
            "missing_evidence": "invalid_edge_omitted_neutral_no_mask_features_v1",
            "per_view_order": "all_real_sfm_observations_candidate_major_v1",
            "candidate_coordinates_or_pose_used": False,
            "profiles": profile_contracts,
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
            "per_view_edges_retained": True,
        },
        "implementation": {
            "script_sha256": file_sha256_short(Path(__file__)),
            "layout_probe_module_sha256": file_sha256_short(
                Path("feature_extract/vfm/localization/fulltrack_aligned_layout_probe.py")
            ),
        },
    }
    payload = {
        "verification_query_ids": np.asarray(raw_arrays["verification_query_ids"]),
        "split_names": np.asarray(raw_arrays["split_names"]),
        "verification_source_row_indices": np.asarray(
            raw_arrays["verification_source_row_indices"], dtype=np.int64
        ),
        "verification_xy": np.asarray(raw_arrays["verification_xy"], dtype=np.float32),
        "candidate_track_ids": np.asarray(raw_arrays["candidate_track_ids"], dtype=np.int64),
        "candidate_probabilities": np.asarray(
            raw_arrays["candidate_probabilities"], dtype=np.float32
        ),
        "null_probabilities": np.asarray(raw_arrays["null_probabilities"], dtype=np.float32),
        "candidate_support_observation_counts": np.asarray(
            raw_arrays["candidate_support_observation_counts"], dtype=np.int64
        ),
        "profile_names": np.asarray(ALIGNED_LAYOUT_FEATURE_NAMES, dtype=np.str_),
        "edge_candidate_offsets": np.asarray(
            raw_arrays["edge_candidate_offsets"], dtype=np.int64
        ),
        "edge_geometry_rows": geometry_rows,
        "edge_profile_scores": edge_scores.astype(np.float16, copy=False),
        "edge_profile_valid": edge_valid,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(output_path)
    summary = {
        "stage": "build_frozen_fulltrack_aligned_layout_appearance",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "format": metadata["format"],
        "row_count": int(len(query_ids)),
        "candidate_edge_count": int(len(geometry_rows)),
        "feature_count": int(len(ALIGNED_LAYOUT_FEATURE_NAMES)),
        "profile_count": int(len(ALIGNED_LAYOUT_PROFILES)),
        "profile_edge_usable_fraction": {
            profile.name: float(np.mean(valid_blocks[index][:, 0]))
            for index, profile in enumerate(ALIGNED_LAYOUT_PROFILES)
        },
        "runtime_seconds": float(time.monotonic() - started),
        "device": str(device_value),
        "protocol": {
            "source_edges_preserved": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_frozen_fulltrack_aligned_layout_appearance(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        template_batch_size=int(args.template_batch_size),
        minimum_support_fraction=float(args.minimum_support_fraction),
        minimum_overlap_fraction=float(args.minimum_overlap_fraction),
        device=str(args.device),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
