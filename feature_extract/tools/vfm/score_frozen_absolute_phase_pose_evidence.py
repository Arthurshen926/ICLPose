"""Run the target-free P1 multiscale absolute-phase pose probe.

The scorer fixes the global top-20 landmark candidates, explicit null mass,
two mapping-side support observations per candidate, held-out 64/64/64 query
points, and every pre-generated pose hypothesis.  A pose only changes the
candidate-projected query crop.  Evidence comes from a feature-only bounded
two-dimensional translation cost volume; neither a learned V5 head nor any
pose-error/ground-truth array is loaded here.

This is a diagnostic producer, not a pose selector.  Its target-free output
must be joined to pose targets only by the companion audit script.  Raw scores
must not be routed into PnP or mixed with an identity posterior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    FixedCandidateViews,
    _canonical_hash,
    _fixed_candidate_views,
    _input_manifest,
    _load_bank_xyz,
    _load_exact_hypotheses,
    _load_maplet_support_fields,
    _load_npz_allowlist,
    _resolve_rows,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
)
from feature_extract.tools.vfm.score_v5_dynamic_absolute_context_pose_evidence import (
    _array_digest,
    _first_fixed_support_views,
    _formal_p1_mixed_evidence_for_query,
    _load_contract,
    _validate_sources_against_contract,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
    load_context_attention_sources,
)
from feature_extract.vfm.localization.frozen_absolute_phase_probe import (
    FROZEN_ABSOLUTE_CONTEXT_CORRELATION_PROBE_VERSION,
    FROZEN_ABSOLUTE_CONTEXT_CORRELATION_PROFILES,
    FROZEN_ABSOLUTE_PHASE_PROBE_VERSION,
    FROZEN_ABSOLUTE_PHASE_PROFILES,
    PHASE_PROBE_SOURCE_NAMES,
    AbsolutePhaseProfile,
    FrozenCandidateMultiscaleCropRuntime,
    deterministic_support_channel_permutation,
    evaluate_absolute_phase_profile,
    profiles_by_source,
)
from feature_extract.vfm.localization.radio_final_context import (
    load_radio_final_context_pca_cache,
)
from feature_extract.vfm.localization.frozen_multiscale_pose_evidence import (
    fixed_candidate_point_log_ratios,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    project_simple_radial_torch,
)


SCORE_FORMAT = "frozen_absolute_phase_pose_scores_v1"
SCORE_VERSION = "p1_feature_only_multiscale_absolute_phase_v1"
CONTEXT_CORRELATION_SCORE_FORMAT = "frozen_absolute_context_correlation_pose_scores_v2"
CONTEXT_CORRELATION_SCORE_VERSION = "p1_feature_only_multiscale_absolute_context_correlation_v2"
POINT_SIDECAR_FORMAT = "frozen_absolute_phase_pose_point_sidecar_v1"
FIXED_CANDIDATE_TOP_K = 20
FIXED_SUPPORT_VIEW_COUNT = 2
FORMAL_POINT_COUNT = 192
EVIDENCE_VARIANTS = ("visual", "support_channel_permutation_control")
POINT_SOURCE_TO_FEATURE_SOURCES = {
    POINT_SOURCE_ALIKE: ("alike",),
    POINT_SOURCE_RADIO_INTERMEDIATE: ("radio_intermediate",),
    POINT_SOURCE_RADIO_FINAL: ("radio_final_grid4", "radio_final_grid8", "radio_final"),
}

# Combination rows remain feature-only.  The two RADIO phase profiles occupy
# disjoint held-out point sets, so their robust aggregation cannot double-count
# one query crop.  ALIKE remains an explicit diagnostic local-detail branch.
FAMILY_COMPONENTS = (
    ("radio_final_center", ("radio_final_center",)),
    ("radio_final_phase3", ("radio_final_phase3",)),
    ("radio_final_phase5", ("radio_final_phase5",)),
    ("radio_intermediate_center", ("radio_intermediate_center",)),
    ("radio_intermediate_phase5", ("radio_intermediate_phase5",)),
    ("radio_intermediate_phase9", ("radio_intermediate_phase9",)),
    ("alike_center_diagnostic", ("alike_center",)),
    (
        "radio_final_phase5_plus_radio_intermediate_phase9",
        ("radio_final_phase5", "radio_intermediate_phase9"),
    ),
    (
        "radio_final_phase5_plus_radio_intermediate_phase9_plus_alike_center_diagnostic",
        ("radio_final_phase5", "radio_intermediate_phase9", "alike_center"),
    ),
)


# Grid4/grid8 are separate large-context RADIO-final views of the same
# mapping-side image cache.  They are reported independently.  Only the
# cross-source combinations below sum profiles because their held-out query
# point sets are disjoint; raw P1 never silently double-counts a final token.
CONTEXT_CORRELATION_FAMILY_COMPONENTS = (
    ("radio_final_grid4_local_bipartite3", ("radio_final_grid4_local_bipartite3",)),
    ("radio_final_grid8_local_bipartite5", ("radio_final_grid8_local_bipartite5",)),
    ("radio_final_grid16_local_bipartite5", ("radio_final_grid16_local_bipartite5",)),
    ("radio_intermediate_local_bipartite7", ("radio_intermediate_local_bipartite7",)),
    ("radio_intermediate_local_bipartite9", ("radio_intermediate_local_bipartite9",)),
    ("alike_local_bipartite5_diagnostic", ("alike_local_bipartite5_diagnostic",)),
    (
        "radio_final_grid8_plus_radio_intermediate_local_bipartite9",
        (
            "radio_final_grid8_local_bipartite5",
            "radio_intermediate_local_bipartite9",
        ),
    ),
    (
        "radio_final_grid8_plus_radio_intermediate9_plus_alike_diagnostic",
        (
            "radio_final_grid8_local_bipartite5",
            "radio_intermediate_local_bipartite9",
            "alike_local_bipartite5_diagnostic",
        ),
    ),
)

PROFILE_SET_PHASE_V1 = "phase_v1"
PROFILE_SET_CONTEXT_CORRELATION_V2 = "context_correlation_v2"
PROFILE_SET_CONFIGS: dict[str, dict[str, object]] = {
    PROFILE_SET_PHASE_V1: {
        "profiles": FROZEN_ABSOLUTE_PHASE_PROFILES,
        "family_components": FAMILY_COMPONENTS,
        "score_format": SCORE_FORMAT,
        "score_version": SCORE_VERSION,
        "probe_version": FROZEN_ABSOLUTE_PHASE_PROBE_VERSION,
        "requires_multiresolution_radio_final": False,
        "raw_score_semantics": (
            "feature_only_bounded_2d_translation_phase_log_ratio_against_uniform_valid_shift_lattice;"
            "center_profiles_are_descriptor_similarity_controls;"
            "mixed_only_under_fixed_candidate_null_denominator"
        ),
    },
    PROFILE_SET_CONTEXT_CORRELATION_V2: {
        "profiles": FROZEN_ABSOLUTE_CONTEXT_CORRELATION_PROFILES,
        "family_components": CONTEXT_CORRELATION_FAMILY_COMPONENTS,
        "score_format": CONTEXT_CORRELATION_SCORE_FORMAT,
        "score_version": CONTEXT_CORRELATION_SCORE_VERSION,
        "probe_version": FROZEN_ABSOLUTE_CONTEXT_CORRELATION_PROBE_VERSION,
        "requires_multiresolution_radio_final": True,
        "raw_score_semantics": (
            "feature_only_bidirectional_per_token_bounded_2d_correlation_log_ratio_against_"
            "uniform_valid_local_displacement_lattice;"
            "radio_final_grid4_grid8_are_large_absolute_context_views;"
            "mixed_only_under_fixed_candidate_null_denominator"
        ),
    },
}


def _profile_set_config(name: str) -> dict[str, object]:
    config = PROFILE_SET_CONFIGS.get(str(name))
    if config is None:
        raise ValueError("absolute-context profile set is unknown")
    return dict(config)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifact", required=True)
    parser.add_argument("--baseline_score_artifact", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--fixed_candidate_prior_overlay", required=True)
    parser.add_argument("--mixed_verification_points_artifact", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument("--context_contract", required=True)
    parser.add_argument(
        "--profile_set",
        choices=tuple(PROFILE_SET_CONFIGS),
        default=PROFILE_SET_PHASE_V1,
        help=(
            "immutable feature-only P1 profile set; profile sets never share output artifacts"
        ),
    )
    parser.add_argument("--fixed_candidate_top_k", type=int, default=FIXED_CANDIDATE_TOP_K)
    parser.add_argument("--verification_point_count", type=int, default=FORMAL_POINT_COUNT)
    parser.add_argument("--fixed_support_view_count", type=int, default=FIXED_SUPPORT_VIEW_COUNT)
    parser.add_argument("--hypothesis_batch_size", type=int, default=64)
    parser.add_argument("--lookup_point_batch_size", type=int, default=4)
    parser.add_argument("--lookup_anchor_batch_size", type=int, default=4)
    parser.add_argument(
        "--evidence_variant",
        choices=EVIDENCE_VARIANTS,
        default="visual",
        help="visual descriptors or the paired fixed support-channel permutation control",
    )
    parser.add_argument(
        "--hypothesis_limit",
        type=int,
        default=0,
        help="development-only prefix; formal audit requires zero and scores every frozen row",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--point_sidecar_output",
        default=None,
        help="target-free per-point evidence sidecar; default derives from --output",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _grid_anchor_coordinates(
    *, image_width: int, image_height: int, grid_size: int, device: torch.device
) -> torch.Tensor:
    if int(image_width) <= 1 or int(image_height) <= 1 or int(grid_size) <= 0:
        raise ValueError("absolute-phase feature-grid geometry is invalid")
    rows = torch.arange(int(grid_size), dtype=torch.float32, device=device)
    columns = torch.arange(int(grid_size), dtype=torch.float32, device=device)
    grid_rows, grid_columns = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack(
        [
            (grid_columns.reshape(-1) + 0.5) / float(grid_size) * float(image_width),
            (grid_rows.reshape(-1) + 0.5) / float(grid_size) * float(image_height),
        ],
        dim=1,
    )


def _profile_payload(profile: AbsolutePhaseProfile) -> dict[str, object]:
    return {
        "name": str(profile.name),
        "source_name": str(profile.source_name),
        "window_size": int(profile.window_size),
        "score_kind": str(profile.score_kind),
        "maximum_translation_cells": int(profile.maximum_translation_cells),
        "temperature": float(profile.temperature),
        "alignment_sigma_cells": float(profile.alignment_sigma_cells),
        "minimum_zero_shift_coverage": float(profile.minimum_zero_shift_coverage),
        "log_ratio_clip": float(profile.log_ratio_clip),
    }


def _source_point_rows(
    point_sources: np.ndarray, *, profile_sources: Sequence[str]
) -> dict[str, np.ndarray]:
    values = np.asarray(point_sources).astype(str).reshape(-1)
    output: dict[str, np.ndarray] = {}
    observed_rows: list[np.ndarray] = []
    requested = {str(source) for source in profile_sources}
    for point_source, feature_sources in POINT_SOURCE_TO_FEATURE_SOURCES.items():
        rows = np.flatnonzero(values == point_source).astype(np.int64)
        if len(rows) != 64:
            raise ValueError(
                f"formal absolute-phase P1 needs 64 {point_source} rows, found {len(rows)}"
            )
        observed_rows.append(rows)
        for feature_source in feature_sources:
            if str(feature_source) in requested:
                output[str(feature_source)] = rows
    if len(np.concatenate(observed_rows)) != len(values) or set(output) != requested:
        raise ValueError("formal P1 contains an unknown point-source family")
    return output


def _profile_source_grids(
    *,
    sources: Sequence[object],
    radio_final_context_cache: Path,
    required_sources: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Mapping[str, object]]]:
    """Expose only the frozen descriptor grids requested by one profile set.

    The context-attention loader supplies the normal grid16 RADIO/ALIKE
    sources.  The correlation-v2 probe additionally opens the exact same
    RADIO-final cache at grid4 and grid8.  These pooled grids are not a new
    retrieval source: they have the identical image manifest, checkpoint and
    mapping-side PCA lineage as grid16, which is checked here before use.
    """

    standard = {str(source.name): source for source in sources}
    required = {str(name) for name in required_sources}
    if not required or not required.issubset(set(PHASE_PROBE_SOURCE_NAMES)):
        raise ValueError("absolute-context profile requests an unknown source")
    reference = standard.get("radio_final")
    if reference is None:
        raise ValueError("absolute-context source loader lacks RADIO-final grid16")
    grids: dict[str, np.ndarray] = {}
    lineage: dict[str, Mapping[str, object]] = {}
    for name in sorted(required.intersection(standard)):
        source = standard[name]
        grids[name] = np.asarray(source.grid)
        lineage[name] = {
            "source_image_manifest_sha256": source.metadata.get("source_image_manifest_sha256"),
            "pca_fit_scope": source.metadata.get("pca_fit_scope"),
            "radio_checkpoint_sha256": source.metadata.get("radio_checkpoint_sha256"),
            "alike_checkpoint_sha256": source.metadata.get("alike_checkpoint_sha256"),
            "grid_size": int(np.asarray(source.grid).shape[1]),
        }
    multiresolution = required.intersection({"radio_final_grid4", "radio_final_grid8"})
    if multiresolution:
        final = load_radio_final_context_pca_cache(Path(radio_final_context_cache))
        if (
            not np.array_equal(np.asarray(final.image_ids).astype(str), np.asarray(reference.image_ids).astype(str))
            or not np.array_equal(np.asarray(final.image_sizes), np.asarray(reference.image_sizes))
            or str(final.metadata.get("source_image_manifest_sha256", ""))
            != str(reference.metadata.get("source_image_manifest_sha256", ""))
            or str(final.metadata.get("radio_checkpoint_sha256", ""))
            != str(reference.metadata.get("radio_checkpoint_sha256", ""))
        ):
            raise ValueError("RADIO-final multiresolution cache lineage differs from grid16")
        cache_by_name = {
            "radio_final_grid4": (4, final.grid4_descriptors),
            "radio_final_grid8": (8, final.grid8_descriptors),
        }
        for name in sorted(multiresolution):
            grid_size, values = cache_by_name[name]
            if values is None:
                raise ValueError(f"RADIO-final cache lacks {name} descriptors")
            array = np.asarray(values)
            if array.ndim != 3 or array.shape[0] != len(final.image_ids) or array.shape[1] != grid_size * grid_size:
                raise ValueError(f"RADIO-final {name} descriptors have invalid geometry")
            grids[name] = array.reshape(len(final.image_ids), grid_size, grid_size, array.shape[2])
            lineage[name] = {
                "source_image_manifest_sha256": final.metadata.get("source_image_manifest_sha256"),
                "pca_fit_scope": final.metadata.get("pca_fit_scope"),
                "radio_checkpoint_sha256": final.metadata.get("radio_checkpoint_sha256"),
                "alike_checkpoint_sha256": None,
                "grid_size": int(grid_size),
            }
    if set(grids) != required or set(lineage) != required:
        raise ValueError("absolute-context profile source grids are incomplete")
    return grids, lineage


def _precompute_source_profile_lookups(
    *,
    runtime: FrozenCandidateMultiscaleCropRuntime,
    source_name: str,
    profiles: Sequence[AbsolutePhaseProfile],
    source_rows: np.ndarray,
    image_width: int,
    image_height: int,
    point_batch_size: int,
    anchor_batch_size: int,
    support_channel_permutation_control: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    """Materialize source-specific raw evidence at every discrete query anchor."""

    profile_values = tuple(profiles)
    if (
        not profile_values
        or any(str(profile.source_name) != str(source_name) for profile in profile_values)
        or int(point_batch_size) <= 0
        or int(anchor_batch_size) <= 0
    ):
        raise ValueError("absolute-phase source lookup configuration is invalid")
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("absolute-phase source point rows are invalid")
    grid_size = runtime.grid_size(str(source_name))
    anchors = _grid_anchor_coordinates(
        image_width=int(image_width),
        image_height=int(image_height),
        grid_size=grid_size,
        device=next(runtime.buffers()).device,
    )
    candidate_count = runtime.candidate_count
    view_count = runtime.view_count
    outputs: dict[str, dict[str, torch.Tensor]] = {
        profile.name: {
            "raw": torch.empty(
                (len(rows), len(anchors), candidate_count, view_count),
                dtype=torch.float32,
                device=anchors.device,
            ),
            "available": torch.empty(
                (len(rows), len(anchors), candidate_count, view_count),
                dtype=torch.bool,
                device=anchors.device,
            ),
            "coverage": torch.empty(
                (len(rows), len(anchors), candidate_count, view_count),
                dtype=torch.float32,
                device=anchors.device,
            ),
        }
        for profile in profile_values
    }
    maximum_window = max(int(profile.window_size) for profile in profile_values)
    descriptor_dim = int(getattr(runtime, f"_grid_{source_name}").shape[-1])
    permutation = (
        deterministic_support_channel_permutation(
            source_name=str(source_name), descriptor_dim=descriptor_dim, device=anchors.device
        )
        if bool(support_channel_permutation_control)
        else None
    )
    with torch.inference_mode():
        for point_begin in range(0, len(rows), int(point_batch_size)):
            point_end = min(point_begin + int(point_batch_size), len(rows))
            source_batch_rows = torch.as_tensor(
                rows[point_begin:point_end], dtype=torch.long, device=anchors.device
            )
            for anchor_begin in range(0, len(anchors), int(anchor_batch_size)):
                anchor_end = min(anchor_begin + int(anchor_batch_size), len(anchors))
                local_anchors = anchors[anchor_begin:anchor_end]
                row_indices = source_batch_rows[:, None].expand(-1, len(local_anchors)).reshape(-1)
                coordinates = (
                    local_anchors[None]
                    .expand(len(source_batch_rows), -1, -1)
                    .reshape(-1, 2)[:, None]
                    .expand(-1, candidate_count, -1)
                )
                crops, _view_valid, _in_image = runtime.pairwise_crops_at_query_xy(
                    row_indices,
                    coordinates,
                    source_windows={str(source_name): maximum_window},
                )
                source_crops = crops[str(source_name)]
                for profile in profile_values:
                    evidence = evaluate_absolute_phase_profile(
                        source_crops,
                        profile=profile,
                        support_channel_permutation=permutation,
                    )
                    shape = (
                        len(source_batch_rows),
                        len(local_anchors),
                        candidate_count,
                        view_count,
                    )
                    outputs[profile.name]["raw"][point_begin:point_end, anchor_begin:anchor_end] = (
                        evidence.log_ratios.reshape(shape).float()
                    )
                    outputs[profile.name]["available"][point_begin:point_end, anchor_begin:anchor_end] = (
                        evidence.available.reshape(shape)
                    )
                    outputs[profile.name]["coverage"][point_begin:point_end, anchor_begin:anchor_end] = (
                        evidence.zero_shift_coverage.reshape(shape).float()
                    )
    return outputs


def _lookup_source_tensor(
    *,
    lookup: torch.Tensor,
    projected_xy: torch.Tensor,
    image_width: int,
    image_height: int,
    grid_size: int,
) -> torch.Tensor:
    """Apply the cropper's floor-cell rule to a source-specific lookup table."""

    coordinates = torch.as_tensor(projected_xy, dtype=torch.float32, device=lookup.device)
    if (
        coordinates.ndim != 4
        or coordinates.shape[1] != lookup.shape[0]
        or coordinates.shape[2] != lookup.shape[2]
        or coordinates.shape[3] != 2
    ):
        raise ValueError("absolute-phase lookup projections are incompatible")
    hypothesis_count, point_count, candidate_count, _ = coordinates.shape
    safe = torch.nan_to_num(coordinates, nan=0.0, posinf=0.0, neginf=0.0)
    columns = torch.floor(safe[..., 0] / float(image_width) * float(grid_size)).to(torch.long)
    rows = torch.floor(safe[..., 1] / float(image_height) * float(grid_size)).to(torch.long)
    anchor_indices = rows.clamp(0, grid_size - 1) * grid_size + columns.clamp(0, grid_size - 1)
    point_indices = torch.arange(point_count, device=lookup.device)[None, :, None].expand(
        hypothesis_count, -1, candidate_count
    )
    candidate_indices = torch.arange(candidate_count, device=lookup.device)[None, None, :].expand(
        hypothesis_count, point_count, -1
    )
    return lookup[point_indices, anchor_indices, candidate_indices]


def _masked_spatial_2x2(
    *, values: torch.Tensor, active: torch.Tensor, query_xy: np.ndarray, image_width: int, image_height: int
) -> torch.Tensor:
    """Median of available fixed 2x2 block means, with no zero-filled blocks."""

    if values.ndim != 2 or active.shape != values.shape:
        raise ValueError("absolute-phase spatial statistics are incompatible")
    xy = np.asarray(query_xy, dtype=np.float32)
    if xy.shape != (values.shape[1], 2):
        raise ValueError("absolute-phase query coordinates are incompatible")
    block = (
        (xy[:, 0] >= float(image_width) * 0.5).astype(np.int64)
        + 2 * (xy[:, 1] >= float(image_height) * 0.5).astype(np.int64)
    )
    block_values: list[torch.Tensor] = []
    block_available: list[torch.Tensor] = []
    for index in range(4):
        rows = torch.as_tensor(block == index, dtype=torch.bool, device=values.device)
        selected = active & rows[None]
        count = selected.sum(dim=1)
        mean = (values * selected.to(dtype=values.dtype)).sum(dim=1) / count.clamp_min(1).to(
            dtype=values.dtype
        )
        block_values.append(mean)
        block_available.append(count > 0)
    stacked = torch.stack(block_values, dim=1)
    available = torch.stack(block_available, dim=1)
    ordered = torch.sort(
        torch.where(available, stacked, torch.full_like(stacked, torch.inf)), dim=1
    ).values
    count = available.sum(dim=1)
    indices = ((count.clamp_min(1) - 1) // 2)[:, None]
    result = torch.gather(ordered, 1, indices).squeeze(1)
    return torch.where(count > 0, result, torch.zeros_like(result))


def _masked_statistics(
    *,
    values: torch.Tensor,
    active: torch.Tensor,
    point_view_masses: torch.Tensor,
    query_xy: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, torch.Tensor]:
    """Robust summaries over only points with real feature evidence."""

    if (
        values.ndim != 2
        or active.shape != values.shape
        or point_view_masses.shape != values.shape
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(point_view_masses).all())
        or torch.any(point_view_masses < 0.0)
    ):
        raise ValueError("absolute-phase point statistics are invalid")
    count = active.sum(dim=1)
    active_float = active.to(dtype=values.dtype)
    means = (values * active_float).sum(dim=1) / count.clamp_min(1).to(dtype=values.dtype)
    ordered = torch.sort(
        torch.where(active, values, torch.full_like(values, torch.inf)), dim=1
    ).values
    median_indices = ((count.clamp_min(1) - 1) // 2)[:, None]
    medians = torch.gather(ordered, 1, median_indices).squeeze(1)
    worst_count = (count + 3) // 4
    ranks = torch.arange(values.shape[1], device=values.device)[None]
    worst_mask = ranks < worst_count[:, None]
    worst_values = torch.where(worst_mask, ordered, torch.zeros_like(ordered))
    worst_means = worst_values.sum(dim=1) / worst_count.clamp_min(1).to(dtype=ordered.dtype)
    empty = count == 0
    zeros = torch.zeros_like(means)
    return {
        "means": torch.where(empty, zeros, means),
        "medians": torch.where(empty, zeros, medians),
        "worst_quartile_means": torch.where(empty, zeros, worst_means),
        "spatial_median_of_means_2x2": _masked_spatial_2x2(
            values=values,
            active=active,
            query_xy=query_xy,
            image_width=int(image_width),
            image_height=int(image_height),
        ),
        "effective_point_counts": count,
        "effective_view_masses": (point_view_masses * active_float).sum(dim=1),
    }


def _score_hypotheses(
    *,
    profile_lookups: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    source_rows: Mapping[str, np.ndarray],
    profiles: Sequence[AbsolutePhaseProfile] = FROZEN_ABSOLUTE_PHASE_PROFILES,
    family_components: Sequence[tuple[str, Sequence[str]]] = FAMILY_COMPONENTS,
    poses_w2c: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    candidate_view_weights: np.ndarray,
    query_xy: np.ndarray,
    camera: object,
    image_width: int,
    image_height: int,
    hypothesis_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Score all fixed hypotheses from precomputed candidate-projected crops."""

    if int(getattr(camera, "model_id")) != 2:
        raise ValueError("absolute-phase scorer requires COLMAP SIMPLE_RADIAL cameras")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4 or int(hypothesis_batch_size) <= 0:
        raise ValueError("absolute-phase camera/batch configuration is invalid")
    xyz = np.asarray(candidate_xyz, dtype=np.float32)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    weights = np.asarray(candidate_view_weights, dtype=np.float32)
    point_count, candidate_count, view_count = weights.shape
    if (
        xyz.shape != (point_count, candidate_count, 3)
        or probabilities.shape != (point_count, candidate_count)
        or null.shape != (point_count,)
        or np.asarray(query_xy).shape != (point_count, 2)
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("absolute-phase candidate geometry is incompatible")
    profile_specs = tuple(profiles)
    profile_by_name = {profile.name: profile for profile in profile_specs}
    if len(profile_by_name) != len(profile_specs) or set(profile_lookups) != set(profile_by_name):
        raise ValueError("absolute-phase lookup profile set is incomplete")
    family_components = tuple(
        (str(name), tuple(str(component) for component in components))
        for name, components in family_components
    )
    if (
        not family_components
        or len({name for name, _components in family_components}) != len(family_components)
        or any(
            not components or any(component not in profile_by_name for component in components)
            for _name, components in family_components
        )
    ):
        raise ValueError("absolute-phase family components are invalid")
    family_names = tuple(name for name, _components in family_components)
    outputs: dict[str, list[np.ndarray]] = {
        "means": [],
        "medians": [],
        "worst_quartile_means": [],
        "spatial_median_of_means_2x2": [],
        "effective_point_counts": [],
        "effective_view_masses": [],
    }
    sidecar: dict[str, list[np.ndarray]] = {
        "point_log_ratios": [],
        "point_active": [],
        "point_effective_view_masses": [],
        "point_zero_shift_coverages": [],
    }
    xyz_tensor = torch.as_tensor(xyz.reshape(-1, 3), dtype=torch.float32, device=device)
    probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    null_tensor = torch.as_tensor(null, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for begin in range(0, len(poses_w2c), int(hypothesis_batch_size)):
            end = min(begin + int(hypothesis_batch_size), len(poses_w2c))
            pose_tensor = torch.as_tensor(poses_w2c[begin:end], dtype=torch.float32, device=device)
            projected, projection_valid = project_simple_radial_torch(
                xyz_tensor,
                pose_tensor,
                focal_length=params[0],
                principal_x=params[1],
                principal_y=params[2],
                radial_k=params[3],
                image_width=int(image_width),
                image_height=int(image_height),
            )
            projected = projected.reshape(end - begin, point_count, candidate_count, 2)
            projection_valid = projection_valid.reshape(end - begin, point_count, candidate_count)
            profile_tensor_values = torch.zeros(
                (end - begin, point_count, len(profile_specs)), dtype=torch.float32, device=device
            )
            profile_active = torch.zeros_like(profile_tensor_values, dtype=torch.bool)
            profile_masses = torch.zeros_like(profile_tensor_values)
            profile_coverages = torch.zeros_like(profile_tensor_values)
            for profile_index, profile in enumerate(profile_specs):
                point_rows_np = np.asarray(source_rows[profile.source_name], dtype=np.int64)
                point_rows = torch.as_tensor(point_rows_np, dtype=torch.long, device=device)
                lookup = profile_lookups[profile.name]
                source_grid_size = int(lookup["raw"].shape[1] ** 0.5)
                local_projected = projected.index_select(1, point_rows)
                local_visible = projection_valid.index_select(1, point_rows)
                raw = _lookup_source_tensor(
                    lookup=lookup["raw"],
                    projected_xy=local_projected,
                    image_width=int(image_width),
                    image_height=int(image_height),
                    grid_size=source_grid_size,
                )
                available = _lookup_source_tensor(
                    lookup=lookup["available"],
                    projected_xy=local_projected,
                    image_width=int(image_width),
                    image_height=int(image_height),
                    grid_size=source_grid_size,
                ).to(dtype=torch.bool)
                coverage = _lookup_source_tensor(
                    lookup=lookup["coverage"],
                    projected_xy=local_projected,
                    image_width=int(image_width),
                    image_height=int(image_height),
                    grid_size=source_grid_size,
                )
                logs, _candidate_ratios, contributed = fixed_candidate_point_log_ratios(
                    view_log_ratios=raw,
                    view_available=available,
                    view_geometric_in_window=local_visible[..., None].expand_as(available),
                    candidate_view_weights=weight_tensor.index_select(0, point_rows),
                    candidate_probabilities=probability_tensor.index_select(0, point_rows),
                    null_probabilities=null_tensor.index_select(0, point_rows),
                )
                effective_mass = contributed.sum(dim=2)
                active = effective_mass > 0.0
                view_mass = (
                    available
                    & local_visible[..., None].expand_as(available)
                ).to(dtype=coverage.dtype) * weight_tensor.index_select(0, point_rows)[None]
                weighted_coverage = (coverage * view_mass).sum(dim=(2, 3)) / effective_mass.clamp_min(
                    torch.finfo(coverage.dtype).tiny
                )
                weighted_coverage = torch.where(active, weighted_coverage, torch.zeros_like(weighted_coverage))
                profile_tensor_values[:, point_rows, profile_index] = logs
                profile_active[:, point_rows, profile_index] = active
                profile_masses[:, point_rows, profile_index] = effective_mass
                profile_coverages[:, point_rows, profile_index] = weighted_coverage
            family_values: list[torch.Tensor] = []
            family_active: list[torch.Tensor] = []
            family_masses: list[torch.Tensor] = []
            family_coverages: list[torch.Tensor] = []
            profile_positions = {profile.name: index for index, profile in enumerate(profile_specs)}
            for _family_name, component_names in family_components:
                component_indices = [profile_positions[name] for name in component_names]
                values = profile_tensor_values[:, :, component_indices].sum(dim=2)
                active = profile_active[:, :, component_indices].any(dim=2)
                masses = profile_masses[:, :, component_indices].sum(dim=2)
                coverage_weight = profile_coverages[:, :, component_indices] * profile_masses[
                    :, :, component_indices
                ]
                coverages = coverage_weight.sum(dim=2) / masses.clamp_min(
                    torch.finfo(profile_tensor_values.dtype).tiny
                )
                coverages = torch.where(active, coverages, torch.zeros_like(coverages))
                family_values.append(values)
                family_active.append(active)
                family_masses.append(masses)
                family_coverages.append(coverages)
            values = torch.stack(family_values, dim=2)
            active = torch.stack(family_active, dim=2)
            masses = torch.stack(family_masses, dim=2)
            coverages = torch.stack(family_coverages, dim=2)
            statistics = [
                _masked_statistics(
                    values=values[:, :, family_index],
                    active=active[:, :, family_index],
                    point_view_masses=masses[:, :, family_index],
                    query_xy=np.asarray(query_xy, dtype=np.float32),
                    image_width=int(image_width),
                    image_height=int(image_height),
                )
                for family_index in range(len(family_names))
            ]
            for key in outputs:
                outputs[key].append(
                    torch.stack([item[key] for item in statistics], dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            sidecar["point_log_ratios"].append(values.detach().cpu().numpy())
            sidecar["point_active"].append(active.detach().cpu().numpy())
            sidecar["point_effective_view_masses"].append(masses.detach().cpu().numpy())
            sidecar["point_zero_shift_coverages"].append(coverages.detach().cpu().numpy())
    result = {
        key: np.concatenate(parts, axis=0).astype(
            np.int64 if key == "effective_point_counts" else np.float64, copy=False
        )
        for key, parts in outputs.items()
    }
    point_sidecar = {
        key: np.concatenate(parts, axis=0).astype(
            bool if key == "point_active" else np.float32, copy=False
        )
        for key, parts in sidecar.items()
    }
    return result, point_sidecar


def _default_sidecar_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_point_sidecar{output.suffix}")


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True), dtype=np.str_),
        )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile_config = _profile_set_config(str(args.profile_set))
    profiles = tuple(profile_config["profiles"])
    family_components = tuple(profile_config["family_components"])
    if (
        not profiles
        or any(not isinstance(profile, AbsolutePhaseProfile) for profile in profiles)
        or not family_components
    ):
        raise ValueError("absolute-context profile configuration is invalid")
    start = time.time()
    output_path = Path(args.output)
    sidecar_path = (
        _default_sidecar_path(output_path)
        if args.point_sidecar_output is None
        else Path(args.point_sidecar_output)
    )
    if output_path == sidecar_path:
        raise ValueError("absolute-phase score and point sidecar paths must differ")
    if (output_path.exists() or sidecar_path.exists()) and not bool(args.force):
        raise FileExistsError("absolute-phase output or point sidecar already exists")
    if (
        int(args.fixed_candidate_top_k) != FIXED_CANDIDATE_TOP_K
        or int(args.verification_point_count) != FORMAL_POINT_COUNT
        or int(args.fixed_support_view_count) != FIXED_SUPPORT_VIEW_COUNT
        or int(args.hypothesis_batch_size) <= 0
        or int(args.lookup_point_batch_size) <= 0
        or int(args.lookup_anchor_batch_size) <= 0
        or int(args.hypothesis_limit) < 0
    ):
        raise ValueError("absolute-phase P1 config differs from its frozen protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    paths = {
        "hypothesis_artifact": Path(args.hypothesis_artifact),
        "baseline_score_artifact": Path(args.baseline_score_artifact),
        "detector_query_cache": Path(args.detector_query_cache),
        "proposals": Path(args.proposals),
        "candidate_artifact": Path(args.candidate_artifact),
        "fixed_candidate_prior_overlay": Path(args.fixed_candidate_prior_overlay),
        "mixed_verification_points_artifact": Path(args.mixed_verification_points_artifact),
        "maplet_support_index": Path(args.maplet_support_index),
        "support_geometry_index": Path(args.support_geometry_index),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "radio_final_context_cache": Path(args.radio_final_context_cache),
        "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        "context_contract": Path(args.context_contract),
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin_camera_ownership_only": Path(args.colmap_model_dir) / "images.bin",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    contract = _load_contract(paths["context_contract"])
    exact, hypothesis_metadata, baseline_metadata = _load_exact_hypotheses(
        hypothesis_path=paths["hypothesis_artifact"],
        baseline_path=paths["baseline_score_artifact"],
        detector_path=paths["detector_query_cache"],
        proposals_path=paths["proposals"],
        candidate_path=paths["candidate_artifact"],
        prior_path=paths["fixed_candidate_prior_overlay"],
        fixed_candidate_top_k=FIXED_CANDIDATE_TOP_K,
    )
    if int(args.hypothesis_limit) > 0:
        count = min(int(args.hypothesis_limit), len(exact["query_ids"]))
        exact = {key: np.asarray(value)[:count] for key, value in exact.items()}
    query_id = str(np.asarray(exact["query_ids"]).astype(str)[0])
    query_split = str(np.asarray(exact["split_names"]).astype(str)[0])
    if query_split not in {"validation", "test"}:
        raise ValueError("absolute-phase P1 accepts only held-out validation/test query shards")
    detector, detector_metadata, _detector_names = _load_npz_allowlist(
        paths["detector_query_cache"], ("image_ids", "offsets", "xy", "detector_scores")
    )
    proposals, proposal_metadata, _proposal_names = _load_npz_allowlist(
        paths["proposals"], ("query_ids", "candidate_track_ids", "coarse_scores"), metadata_required=False
    )
    candidate_artifact, candidate_metadata, _candidate_names = _load_npz_allowlist(
        paths["candidate_artifact"], ("selected_rows",)
    )
    if (
        detector_metadata.get("format") != "alike_detector_mapped_radio_query_cache_v1"
        or candidate_metadata.get("contains_ground_truth") is not False
        or candidate_metadata.get("contains_pose_derived_selection") is not False
        or proposal_metadata.get("format") not in {None, "detector_support_reranked_proposals_v1"}
    ):
        raise ValueError("absolute-phase P1 inputs violate the target-free frozen-row contract")
    _prior_overlay, prior_metadata = _load_candidate_prior_overlay(
        paths["fixed_candidate_prior_overlay"],
        proposals_path=paths["proposals"],
        proposals=proposals,
    )
    mixed, selection_audit, mixed_metadata = _formal_p1_mixed_evidence_for_query(
        points_path=paths["mixed_verification_points_artifact"],
        query_id=query_id,
        query_split=query_split,
        detector_path=paths["detector_query_cache"],
        candidate_path=paths["candidate_artifact"],
        landmark_bank_path=paths["projected_landmark_bank"],
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray(candidate_artifact["selected_rows"], dtype=np.int64),
        point_count=FORMAL_POINT_COUNT,
    )
    candidate_tracks = np.asarray(mixed["candidate_track_ids"], dtype=np.int64)
    candidate_probabilities = np.asarray(mixed["candidate_probabilities"], dtype=np.float32)
    null_probabilities = np.asarray(mixed["null_probabilities"], dtype=np.float32)
    query_xy = np.asarray(mixed["xy"], dtype=np.float32)
    candidate_bank_rows = np.asarray(mixed["candidate_bank_rows"], dtype=np.int64)
    verification_source_row_indices = np.asarray(mixed["source_point_ids"], dtype=np.int64)
    verification_point_sources = np.asarray(mixed["point_sources"], dtype=np.str_)
    verification_source_detector_rows = np.asarray(mixed["source_detector_rows"], dtype=np.int64)
    if candidate_tracks.shape != candidate_probabilities.shape or candidate_tracks.shape != (
        FORMAL_POINT_COUNT,
        FIXED_CANDIDATE_TOP_K,
    ):
        raise ValueError("absolute-phase formal P1 candidate layout is invalid")
    source_rows = _source_point_rows(
        verification_point_sources,
        profile_sources=tuple(profile.source_name for profile in profiles),
    )

    bank_tracks, bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    maplet_tracks, maplet_image_ids, maplet_image_indices, maplet_coverage, maplet_metadata = (
        _load_maplet_support_fields(paths["maplet_support_index"])
    )
    if str(maplet_metadata.get("source_landmark_index_sha256", "")) != file_sha256_short(
        paths["projected_landmark_bank"]
    ):
        raise ValueError("maplet support index was built from another landmark bank")
    if (
        candidate_bank_rows.shape != candidate_tracks.shape
        or np.any(candidate_bank_rows[candidate_tracks >= 0] < 0)
        or np.any(candidate_bank_rows[candidate_tracks >= 0] >= len(bank_tracks))
        or not np.array_equal(
            bank_tracks[candidate_bank_rows[candidate_tracks >= 0]],
            candidate_tracks[candidate_tracks >= 0],
        )
    ):
        raise ValueError("absolute-phase candidate rows do not identify their physical tracks")
    candidate_xyz = np.zeros((*candidate_tracks.shape, 3), dtype=np.float32)
    valid_tracks = candidate_bank_rows >= 0
    candidate_xyz[valid_tracks] = bank_xyz[candidate_bank_rows[valid_tracks]]
    if np.any((candidate_probabilities > 0.0) & ~valid_tracks):
        raise ValueError("a positive-mass absolute-phase candidate lacks landmark geometry")
    all_views: FixedCandidateViews = _fixed_candidate_views(
        candidate_track_ids=candidate_tracks,
        candidate_probabilities=candidate_probabilities,
        maplet_track_ids=maplet_tracks,
        support_image_ids=maplet_image_ids,
        support_image_indices=maplet_image_indices,
        support_coverage_counts=maplet_coverage,
    )
    views = _first_fixed_support_views(all_views, count=FIXED_SUPPORT_VIEW_COUNT)
    support_geometry, support_geometry_metadata = load_support_observation_geometry_index_npz(
        paths["support_geometry_index"]
    )
    sources = load_context_attention_sources(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
        expected_radio_checkpoint=str(contract.get("radio_checkpoint_sha256", "")),
        require_equal_descriptor_dimensions=False,
    )
    _validate_sources_against_contract(contract=contract, sources=sources)
    reference_source = sources[0]
    source_grids, source_lineage = _profile_source_grids(
        sources=sources,
        radio_final_context_cache=paths["radio_final_context_cache"],
        required_sources=tuple(profile.source_name for profile in profiles),
    )
    runtime_arrays = build_fixed_candidate_context_runtime(
        query_ids=np.full((FORMAL_POINT_COUNT,), query_id),
        query_xy=query_xy,
        candidate_track_ids=candidate_tracks,
        candidate_support_image_ids=views.support_image_ids,
        candidate_view_valid=views.valid,
        cache_image_ids=reference_source.image_ids,
        support_geometry=support_geometry,
    )
    phase_runtime = FrozenCandidateMultiscaleCropRuntime(
        sources={name: torch.from_numpy(grid) for name, grid in source_grids.items()},
        image_sizes=torch.from_numpy(np.asarray(reference_source.image_sizes, dtype=np.float32)),
        runtime=runtime_arrays,
    ).to(device).eval()
    query_positions = np.flatnonzero(np.asarray(reference_source.image_ids).astype(str) == query_id)
    if len(query_positions) != 1:
        raise ValueError("query image is absent or duplicated in the absolute-phase context cache")
    image_width, image_height = (
        int(value) for value in np.asarray(reference_source.image_sizes)[int(query_positions[0])]
    )
    if not np.all((query_xy[:, 0] >= 0.0) & (query_xy[:, 0] <= image_width - 1.0)) or not np.all(
        (query_xy[:, 1] >= 0.0) & (query_xy[:, 1] <= image_height - 1.0)
    ):
        raise ValueError("absolute-phase held-out verification coordinates exceed the query image")
    cameras = read_colmap_cameras_binary(paths["colmap_cameras_bin"])
    camera_ids = read_colmap_image_camera_ids_binary(paths["colmap_images_bin_camera_ownership_only"])
    camera_id = camera_ids.get(query_id)
    if camera_id is None or int(camera_id) not in cameras:
        raise ValueError("absolute-phase query image has no declared COLMAP camera")
    camera = cameras[int(camera_id)]
    if (int(camera.width), int(camera.height)) != (image_width, image_height):
        raise ValueError("absolute-phase query camera geometry differs from the context cache")
    grouped_profiles = profiles_by_source(profiles)
    profile_lookups: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for source_name, profile_values in grouped_profiles.items():
        per_source = _precompute_source_profile_lookups(
            runtime=phase_runtime,
            source_name=source_name,
            profiles=profile_values,
            source_rows=source_rows[source_name],
            image_width=int(image_width),
            image_height=int(image_height),
            point_batch_size=int(args.lookup_point_batch_size),
            anchor_batch_size=int(args.lookup_anchor_batch_size),
            support_channel_permutation_control=(
                str(args.evidence_variant) == "support_channel_permutation_control"
            ),
        )
        profile_lookups.update(per_source)
    statistics, point_sidecar = _score_hypotheses(
        profile_lookups=profile_lookups,
        source_rows=source_rows,
        profiles=profiles,
        family_components=family_components,
        poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64),
        candidate_xyz=candidate_xyz,
        candidate_probabilities=candidate_probabilities,
        null_probabilities=null_probabilities,
        candidate_view_weights=views.weights,
        query_xy=query_xy,
        camera=camera,
        image_width=int(image_width),
        image_height=int(image_height),
        hypothesis_batch_size=int(args.hypothesis_batch_size),
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    family_names = np.asarray([name for name, _components in family_components], dtype=np.str_)
    frozen_evidence = {
        "verification_source_row_indices": verification_source_row_indices,
        "verification_point_sources": verification_point_sources,
        "verification_source_detector_rows": verification_source_detector_rows,
        "verification_xy": query_xy,
        "candidate_track_ids": candidate_tracks,
        "candidate_probabilities": candidate_probabilities,
        "null_probabilities": null_probabilities,
        "candidate_view_weights": views.weights,
        "candidate_support_image_ids": views.support_image_ids,
        "profile_set": np.asarray(str(args.profile_set)),
        "profiles": np.asarray(
            json.dumps([_profile_payload(profile) for profile in profiles], sort_keys=True)
        ),
        "family_components": np.asarray(
            json.dumps({name: list(components) for name, components in family_components}, sort_keys=True)
        ),
    }
    frozen_evidence_sha = _array_digest(frozen_evidence)
    arrays: dict[str, np.ndarray] = {
        "query_ids": np.asarray(exact["query_ids"]).astype(str),
        "split_names": np.asarray(exact["split_names"]).astype(str),
        "evaluation_labels": np.asarray(exact["evaluation_labels"]).astype(str),
        "hypothesis_indices": np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray(
            exact["source_chosen_for_optional_pose"], dtype=bool
        ),
        "baseline_score_top1": np.asarray(exact["independent_score_top1"], dtype=bool),
        "baseline_selection_scores": np.asarray(
            exact["independent_selection_scores"], dtype=np.float64
        ),
        "family_names": family_names,
        "family_log_likelihood_means": statistics["means"],
        "family_log_likelihood_medians": statistics["medians"],
        "family_log_likelihood_worst_quartile_means": statistics["worst_quartile_means"],
        "family_spatial_median_of_means_2x2": statistics["spatial_median_of_means_2x2"],
        "family_effective_point_counts": statistics["effective_point_counts"],
        "family_effective_view_masses": statistics["effective_view_masses"],
        "verification_query_ids": np.full((FORMAL_POINT_COUNT,), query_id),
        "verification_source_row_indices": verification_source_row_indices,
        "verification_point_sources": verification_point_sources,
        "verification_source_detector_rows": verification_source_detector_rows,
        "verification_xy": query_xy,
    }
    strict_contract = {
        "heldout_query_rows": True,
        "verification_point_selector_fixed_across_hypotheses": True,
        "formal_p1_mixed_multiscale_points": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": FIXED_CANDIDATE_TOP_K,
        "explicit_null_mass": True,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "support_view_descriptor_averaging": False,
        "support_views": "fixed_maplet_rank_prefix_two_v1",
        "candidate_specific_query_projection": True,
        "query_center_gaussian_fallback": False,
        "source_specific_heldout_feature_roles": True,
        "learned_context_head_used": False,
        "learned_identity_head_used": False,
        "feature_only_bounded_2d_translation_cost_volume": True,
        "feature_only_bidirectional_per_token_2d_correlation": (
            str(args.profile_set) == PROFILE_SET_CONTEXT_CORRELATION_V2
        ),
        "radio_final_multiresolution_absolute_context": bool(
            profile_config["requires_multiresolution_radio_final"]
        ),
        "support_channel_permutation_control_only": (
            str(args.evidence_variant) == "support_channel_permutation_control"
        ),
        "image_retrieval_or_submap_used": False,
        "render": False,
        "raw_scores_calibrated_as_independent_pose_likelihood": False,
    }
    metadata: dict[str, Any] = {
        "format": str(profile_config["score_format"]),
        "version": str(profile_config["score_version"]),
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "row_count": int(len(arrays["query_ids"])),
        "query_count": 1,
        "query_id": query_id,
        "split_name": query_split,
        "evidence_variant": str(args.evidence_variant),
        "profile_set": str(args.profile_set),
        "raw_score_semantics": str(profile_config["raw_score_semantics"]),
        "raw_score_is_calibrated_independent_pose_likelihood": False,
        "profiles": [_profile_payload(profile) for profile in profiles],
        "family_components": {name: list(components) for name, components in family_components},
        "frozen_query_evidence_sha256": frozen_evidence_sha,
        "verification_point_selection": selection_audit,
        "hypothesis_scope": {
            "all_frozen_hypotheses": int(args.hypothesis_limit) == 0,
            "development_prefix_limit": int(args.hypothesis_limit),
            "scored_hypothesis_count": int(len(np.asarray(exact["query_ids"]))),
        },
        "strict_frozen_evidence_contract": strict_contract,
        "camera": {
            "camera_id": int(camera.camera_id),
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
            "ownership_parser": "image_name_to_camera_id_pose_discarded_v1",
        },
        "source_cache_lineage": source_lineage,
        "source_metadata_hashes": {
            "contract": _canonical_hash(contract),
            "hypothesis": _canonical_hash(hypothesis_metadata),
            "baseline_s0": _canonical_hash(baseline_metadata),
            "detector": _canonical_hash(detector_metadata),
            "proposals": _canonical_hash(proposal_metadata),
            "candidate": _canonical_hash(candidate_metadata),
            "candidate_prior": _canonical_hash(prior_metadata),
            "maplet": _canonical_hash(maplet_metadata),
            "support_geometry": _canonical_hash(support_geometry_metadata),
            "landmark_bank": _canonical_hash(bank_metadata),
            "mixed_verification_points": _canonical_hash(mixed_metadata),
        },
        "inputs": _input_manifest(paths),
        "point_sidecar": str(sidecar_path),
        "implementation": {
            "script_sha256": file_sha256_short(Path(__file__)),
            "phase_probe_module_sha256": file_sha256_short(
                Path("feature_extract/vfm/localization/frozen_absolute_phase_probe.py")
            ),
            "phase_probe_version": str(profile_config["probe_version"]),
        },
        "elapsed_seconds": float(time.time() - start),
    }
    sidecar_arrays = {
        "query_ids": arrays["query_ids"],
        "split_names": arrays["split_names"],
        "evaluation_labels": arrays["evaluation_labels"],
        "hypothesis_indices": arrays["hypothesis_indices"],
        "family_names": family_names,
        "point_log_ratios": point_sidecar["point_log_ratios"],
        "point_active": point_sidecar["point_active"],
        "point_effective_view_masses": point_sidecar["point_effective_view_masses"],
        "point_zero_shift_coverages": point_sidecar["point_zero_shift_coverages"],
        "verification_xy": query_xy,
        "verification_point_sources": verification_point_sources,
    }
    sidecar_metadata = {
        "format": POINT_SIDECAR_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "frozen_query_evidence_sha256": frozen_evidence_sha,
        "score_artifact_path": str(output_path),
        "score_format": str(profile_config["score_format"]),
        "evidence_variant": str(args.evidence_variant),
        "profile_set": str(args.profile_set),
        "family_names": family_names.tolist(),
        "strict_frozen_evidence_contract": strict_contract,
        "row_count": int(len(arrays["query_ids"])),
    }
    _atomic_savez(output_path, arrays, metadata)
    _atomic_savez(sidecar_path, sidecar_arrays, sidecar_metadata)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "stage": "frozen_absolute_phase_pose_evidence",
                "output": str(output_path),
                "point_sidecar": str(sidecar_path),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output_path), "point_sidecar": str(sidecar_path), "metadata": metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
