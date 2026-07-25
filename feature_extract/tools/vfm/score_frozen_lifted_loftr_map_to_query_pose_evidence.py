"""Score fixed pose hypotheses with independent lifted LoFTR map-to-query evidence.

LoFTR has already been run between the query and every mapping image.  This
target-free P1 producer never fits a pose from those matches.  Instead it:

1. freezes the top-20 candidate maplet union used by the existing PnP fit;
2. removes LoFTR query matches near all 128 PnP fit detector points;
3. lifts support-anchor matches into multi-view-consistent query-to-3D modes;
4. evaluates every already-generated hypothesis with a fixed out-of-image
   penalty and robust spatial aggregation.

The target-side companion audit is the only consumer permitted to read pose
errors.  These raw scores are diagnostic and must never be fed to PnP.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.build_frozen_loftr_candidate_anchor_evidence import (
    _validate_pair_cache_lineage,
)
from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _input_manifest,
    _load_bank_xyz,
    _load_exact_hypotheses,
    _load_maplet_support_fields,
    _load_npz_allowlist,
    _resolve_rows,
)
from feature_extract.tools.vfm.score_v5_dynamic_absolute_context_pose_evidence import (
    _array_digest,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.frozen_lifted_loftr_map_to_query import (
    FrozenCandidateGroupLayout,
    FrozenCandidateSupportViewGroupLayout,
    FrozenLiftedTrackModeBank,
    FrozenLiftedSupportViewModeBank,
    build_frozen_candidate_support_view_group_layout,
    build_frozen_candidate_group_layout,
    build_frozen_lifted_track_mode_bank,
    build_frozen_lifted_support_view_mode_bank,
    deterministic_track_xyz_permutation,
    candidate_support_view_log_mixture_terms_from_support_view_log_ratios,
    fixed_prior_group_log_ratios_from_track_log_ratios,
    group_log_ratios_from_track_log_ratios,
    robust_track_log_ratio_statistics,
    select_loftr_modes_near_support_anchors,
    support_view_log_ratios_from_projected,
    track_mode_log_ratios_from_projected,
)
from feature_extract.vfm.localization.frozen_loftr_coordinate_contract import (
    build_frozen_loftr_colmap_coordinate_contract,
)
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    load_frozen_loftr_pair_cache,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    project_simple_radial_torch,
)


SCORE_FORMAT = "frozen_lifted_loftr_map_to_query_pose_scores_v1"
SCORE_VERSION = "p1_candidate_specific_support_view_lifted_loftr_map_to_query_v6"
FIXED_CANDIDATE_TOP_K = 20
DEFAULT_FIXED_SUPPORT_VIEW_COUNT = 4
DEFAULT_FIT_EXCLUSION_RADIUS_PX = 8.0
DEFAULT_SUPPORT_SNAP_RADIUS_PX = 6.0
DEFAULT_QUERY_MODE_NMS_RADIUS_PX = 4.0
DEFAULT_MIN_SUPPORT_VIEWS = 2
DEFAULT_CONSENSUS_RADIUS_PX = 12.0
DEFAULT_MAX_MODES_PER_ANCHOR = 2
DEFAULT_MAX_MODES_PER_TRACK = 3
# These fixed scales span LoFTR fine-localization uncertainty after mapping
# original 1920x1080 cache coordinates onto the 1024x576 COLMAP grid.  They
# are all reported as raw P1 profiles; no validation target selects one here.
DEFAULT_SIGMA_PROFILES = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
DEFAULT_GROUP_NULL_MASSES = (0.25, 0.5, 0.75)
EVIDENCE_VARIANTS = ("visual", "xyz_permutation_control")
EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION = "global_track_union_crossview_v1"
EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW = "candidate_specific_support_view_v1"
_SUPPORT_VIEW_OVERLAY_FORMAT = "candidate_maplet_support_view_overlay_v1"
_SUPPORT_VIEW_OVERLAY_SEMANTICS = (
    "identity_conditioned_support_view_probability_normalized_per_candidate_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis-artifact", required=True)
    parser.add_argument("--baseline-score-artifact", required=True)
    parser.add_argument("--detector-query-cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--fixed-candidate-prior-overlay", required=True)
    parser.add_argument(
        "--fixed-candidate-support-view-overlay",
        default="",
        help=(
            "proposal-aligned target-free p(view | query,candidate) overlay; "
            "required by --evidence-layout candidate_specific_support_view"
        ),
    )
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--loftr-pair-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--loftr-checkpoint", required=True)
    parser.add_argument("--fixed-candidate-top-k", type=int, default=FIXED_CANDIDATE_TOP_K)
    parser.add_argument(
        "--evidence-layout",
        choices=(
            EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION,
            EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW,
        ),
        default=EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION,
    )
    parser.add_argument("--fixed-support-view-count", type=int, default=DEFAULT_FIXED_SUPPORT_VIEW_COUNT)
    parser.add_argument("--fit-exclusion-radius-px", type=float, default=DEFAULT_FIT_EXCLUSION_RADIUS_PX)
    parser.add_argument("--support-snap-radius-px", type=float, default=DEFAULT_SUPPORT_SNAP_RADIUS_PX)
    parser.add_argument("--query-mode-nms-radius-px", type=float, default=DEFAULT_QUERY_MODE_NMS_RADIUS_PX)
    parser.add_argument("--min-support-views", type=int, default=DEFAULT_MIN_SUPPORT_VIEWS)
    parser.add_argument("--consensus-radius-px", type=float, default=DEFAULT_CONSENSUS_RADIUS_PX)
    parser.add_argument("--max-modes-per-anchor", type=int, default=DEFAULT_MAX_MODES_PER_ANCHOR)
    parser.add_argument("--max-modes-per-track", type=int, default=DEFAULT_MAX_MODES_PER_TRACK)
    parser.add_argument(
        "--base-support-reliability",
        type=float,
        default=0.75,
        help=(
            "fixed LoFTR endpoint reliability before candidate/view marginalization"
        ),
    )
    parser.add_argument(
        "--sigma-px-profiles",
        default=",".join(str(value) for value in DEFAULT_SIGMA_PROFILES),
        help="predeclared raw P1 projection sigmas, comma-separated",
    )
    parser.add_argument(
        "--group-null-masses",
        default=",".join(str(value) for value in DEFAULT_GROUP_NULL_MASSES),
        help="predeclared fixed null masses for candidate-group identity mixtures",
    )
    parser.add_argument("--out-of-image-ratio", type=float, default=0.01)
    parser.add_argument("--max-log-ratio", type=float, default=12.0)
    parser.add_argument("--hypothesis-batch-size", type=int, default=256)
    parser.add_argument("--hypothesis-limit", type=int, default=0)
    parser.add_argument(
        "--diagnostic-hypothesis-indices",
        default="",
        help=(
            "comma-separated immutable hypothesis indices for an explicitly "
            "target-only tail attribution rerun; cannot be combined with "
            "--hypothesis-limit or formal P1 audit"
        ),
    )
    parser.add_argument(
        "--diagnostic-dump-group-terms",
        action="store_true",
        help=(
            "write per-support-view, candidate, and group terms for an "
            "explicit diagnostic hypothesis subset only"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evidence-variant", choices=EVIDENCE_VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _parse_sigmas(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("LoFTR sigma profiles are malformed") from error
    if (
        not values
        or len(set(values)) != len(values)
        or any(not np.isfinite(item) or item <= 0.0 for item in values)
    ):
        raise ValueError("LoFTR sigma profiles are invalid")
    return values


def _parse_group_null_masses(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("candidate-group null masses are malformed") from error
    if (
        not values
        or len(set(values)) != len(values)
        or any(not np.isfinite(item) or not 0.0 < item < 1.0 for item in values)
    ):
        raise ValueError("candidate-group null masses are invalid")
    return values


def _parse_diagnostic_hypothesis_indices(value: str) -> tuple[int, ...]:
    """Parse an explicit, immutable diagnostic hypothesis subset."""

    raw = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not raw:
        return ()
    try:
        values = tuple(int(item) for item in raw)
    except ValueError as error:
        raise ValueError("diagnostic hypothesis indices are malformed") from error
    if any(item < 0 for item in values) or len(set(values)) != len(values):
        raise ValueError("diagnostic hypothesis indices must be unique non-negative integers")
    return values


def _select_explicit_hypothesis_indices(
    exact: Mapping[str, np.ndarray], requested_indices: Sequence[int]
) -> dict[str, np.ndarray]:
    """Select exact frozen rows by immutable hypothesis index, preserving order."""

    requested = tuple(int(value) for value in requested_indices)
    if not requested:
        raise ValueError("explicit diagnostic hypothesis selection is empty")
    if "hypothesis_indices" not in exact:
        raise ValueError("frozen hypotheses lack immutable hypothesis indices")
    indices = np.asarray(exact["hypothesis_indices"], dtype=np.int64).reshape(-1)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("frozen hypothesis indices are malformed")
    lookup = {int(value): row for row, value in enumerate(indices.tolist())}
    missing = [int(value) for value in requested if int(value) not in lookup]
    if missing:
        raise ValueError(
            "requested diagnostic hypothesis indices are absent from frozen rows: "
            + ", ".join(str(value) for value in missing)
        )
    rows = np.asarray([lookup[int(value)] for value in requested], dtype=np.int64)
    output = {key: np.asarray(value)[rows] for key, value in exact.items()}
    if not np.array_equal(
        np.asarray(output["hypothesis_indices"], dtype=np.int64),
        np.asarray(requested, dtype=np.int64),
    ):
        raise ValueError("diagnostic frozen hypothesis selection lost immutable order")
    return output


def _load_fit_rows_and_points(
    *,
    candidate_artifact: Path,
    detector_query_cache: Path,
    proposals: Path,
    query_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    candidate, candidate_metadata, _ = _load_npz_allowlist(
        Path(candidate_artifact), ("selected_rows",)
    )
    detector, _detector_metadata, _ = _load_npz_allowlist(
        Path(detector_query_cache), ("image_ids", "offsets", "xy")
    )
    # This legacy proposal artifact also stores target-only diagnostics, but
    # has no metadata envelope.  Read a strict inference-only allowlist rather
    # than handing its whole payload to a generic loader.
    with np.load(Path(proposals), allow_pickle=False) as payload:
        proposal_fields = ("query_ids", "candidate_track_ids")
        missing = sorted(set(proposal_fields).difference(payload.files))
        if missing:
            raise ValueError(f"{proposals}: proposal artifact lacks {missing}")
        proposal = {field: np.asarray(payload[field]).copy() for field in proposal_fields}
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64).reshape(-1)
    proposal_ids = np.asarray(proposal["query_ids"]).astype(str).reshape(-1)
    candidate_tracks = np.asarray(proposal["candidate_track_ids"], dtype=np.int64)
    detector_ids = np.asarray(detector["image_ids"]).astype(str).reshape(-1)
    detector_offsets = np.asarray(detector["offsets"], dtype=np.int64).reshape(-1)
    detector_xy = np.asarray(detector["xy"], dtype=np.float32)
    if (
        candidate_metadata.get("format") != "detector_maplet_geometry_features_v1"
        or candidate_metadata.get("contains_ground_truth") is not False
        or candidate_metadata.get("contains_pose_derived_selection") is not False
        or int(candidate_metadata.get("candidate_top_k", -1)) != FIXED_CANDIDATE_TOP_K
        or int(candidate_metadata.get("query_points_per_image", -1)) != 128
        or str(candidate_metadata.get("detector_query_cache_sha256", ""))
        != file_sha256_short(Path(detector_query_cache))
        or str(candidate_metadata.get("proposals_sha256", ""))
        != file_sha256_short(Path(proposals))
        or detector_offsets.shape != (len(detector_ids) + 1,)
        or detector_offsets[0] != 0
        or detector_offsets[-1] != len(detector_xy)
        or candidate_tracks.shape[0] != len(proposal_ids)
        or proposal_ids.shape != (len(detector_xy),)
        or np.any(selected_rows < 0)
        or np.any(selected_rows >= len(proposal_ids))
    ):
        raise ValueError("target-free PnP fit-row source contract is invalid")
    rows = selected_rows[proposal_ids[selected_rows] == str(query_id)]
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("candidate artifact has no unique PnP fit rows for query")
    if len(rows) != 128:
        raise ValueError("lifted LoFTR P1 requires the complete fixed 128-point PnP fit set")
    detector_position = np.flatnonzero(detector_ids == str(query_id))
    if len(detector_position) != 1:
        raise ValueError("detector cache does not uniquely own the query")
    start, stop = (
        int(detector_offsets[int(detector_position[0])]),
        int(detector_offsets[int(detector_position[0]) + 1]),
    )
    if np.any((rows < start) | (rows >= stop)):
        raise ValueError("candidate fit rows do not belong to query detector image")
    tracks = candidate_tracks[rows]
    if tracks.ndim != 2 or tracks.shape[1] != FIXED_CANDIDATE_TOP_K:
        raise ValueError("PnP fit candidate source does not retain fixed top-20 tracks")
    return (
        rows.astype(np.int64, copy=False),
        detector_xy[rows].astype(np.float32, copy=False),
        tracks.astype(np.int64, copy=False),
        dict(candidate_metadata),
    )


def _load_candidate_support_view_overlay(
    *,
    path: Path,
    proposals_path: Path,
    proposals: Mapping[str, np.ndarray],
    candidate_prior_path: Path,
    maplet_support_index: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load only a strictly co-lineaged target-free support-view overlay."""

    fields = (
        "candidate_track_ids",
        "support_view_probabilities",
        "candidate_support_image_ids",
        "candidate_support_view_valid",
        "metadata_json",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(fields).difference(payload.files))
        if missing:
            raise ValueError(f"support-view overlay lacks fields: {missing}")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields[:-1]}
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("support-view overlay metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("support-view overlay metadata must be an object")
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["support_view_probabilities"], dtype=np.float32)
    support_ids = np.asarray(arrays["candidate_support_image_ids"]).astype(str)
    valid = np.asarray(arrays["candidate_support_view_valid"], dtype=bool)
    proposal_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    if (
        metadata.get("format") != _SUPPORT_VIEW_OVERLAY_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
        or metadata.get("candidate_reselection") is not False
        or metadata.get("support_reselection") is not False
        or metadata.get("probability_semantics") != _SUPPORT_VIEW_OVERLAY_SEMANTICS
        or str(metadata.get("proposals_sha256", ""))
        != file_sha256_short(Path(proposals_path))
        or str(metadata.get("candidate_prior_overlay_sha256", ""))
        != file_sha256_short(Path(candidate_prior_path))
        or str(metadata.get("maplet_support_index_sha256", ""))
        != file_sha256_short(Path(maplet_support_index))
        or tracks.shape != proposal_tracks.shape
        or not np.array_equal(tracks, proposal_tracks)
        or probabilities.ndim != 3
        or probabilities.shape[:2] != tracks.shape
        or probabilities.shape[2] != int(metadata.get("support_view_count", -1))
        or support_ids.shape != probabilities.shape
        or valid.shape != probabilities.shape
        or not np.all(valid)
        or np.any(support_ids == "")
        or np.any(~np.isfinite(probabilities))
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or np.any(
            np.abs(probabilities.sum(axis=2, dtype=np.float64) - 1.0) > 1e-4
        )
    ):
        raise ValueError("support-view overlay violates the frozen candidate contract")
    return {
        "candidate_track_ids": tracks,
        "support_view_probabilities": probabilities,
        "candidate_support_image_ids": support_ids,
        "candidate_support_view_valid": valid,
    }, dict(metadata)


def _fixed_candidate_maplet_anchor_layout(
    *,
    candidate_tracks: np.ndarray,
    fixed_support_view_count: int,
    maplet_track_ids: np.ndarray,
    maplet_image_ids: Sequence[str],
    maplet_image_indices: np.ndarray,
    maplet_coverage: np.ndarray,
    bank_tracks: np.ndarray,
    bank_xyz: np.ndarray,
    geometry: SupportObservationGeometryIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Use a fixed maplet prefix, never an image search or pose-specific view."""

    candidates = np.asarray(candidate_tracks, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[1] != FIXED_CANDIDATE_TOP_K:
        raise ValueError("candidate maplet union needs fixed top-20 candidate groups")
    if not 0 < int(fixed_support_view_count) <= maplet_image_indices.shape[1]:
        raise ValueError("fixed maplet support-view prefix is invalid")
    tracks = np.unique(candidates[candidates >= 0])
    if len(tracks) == 0:
        raise ValueError("candidate maplet union is empty")
    maplet_rows = _resolve_rows(tracks, canonical_track_ids=maplet_track_ids)
    bank_rows = _resolve_rows(tracks, canonical_track_ids=bank_tracks)
    support_indices = maplet_image_indices[maplet_rows, : int(fixed_support_view_count)]
    coverage = maplet_coverage[maplet_rows, : int(fixed_support_view_count)]
    source_ids = np.asarray(tuple(str(value) for value in maplet_image_ids), dtype=np.str_)
    pairs: list[tuple[str, int, np.ndarray]] = []
    for track_index, track_id in enumerate(tracks.tolist()):
        for view_index in range(int(fixed_support_view_count)):
            image_index = int(support_indices[track_index, view_index])
            if image_index < 0 or int(coverage[track_index, view_index]) <= 0:
                continue
            pairs.append(
                (
                    str(source_ids[image_index]),
                    int(track_id),
                    np.asarray(bank_xyz[bank_rows[track_index]], dtype=np.float32),
                )
            )
    if not pairs:
        raise ValueError("fixed candidate maplet union has no usable support anchors")
    pairs.sort(key=lambda item: (item[0], item[1]))
    if len({(image_id, track_id) for image_id, track_id, _xyz in pairs}) != len(pairs):
        raise ValueError("fixed maplet union repeats a track/support observation")
    output_tracks: list[int] = []
    output_xyz: list[np.ndarray] = []
    output_ids: list[str] = []
    output_xy: list[np.ndarray] = []
    for image_id in sorted({item[0] for item in pairs}):
        image_pairs = [item for item in pairs if item[0] == image_id]
        image_tracks = np.asarray([item[1] for item in image_pairs], dtype=np.int64)
        rows = geometry.geometry_rows_for_tracks(image_id, image_tracks)
        if np.any(rows < 0):
            raise ValueError("maplet support anchor is absent from frozen support geometry")
        for item, row in zip(image_pairs, rows.tolist()):
            output_ids.append(image_id)
            output_tracks.append(int(item[1]))
            output_xyz.append(np.asarray(item[2], dtype=np.float32))
            output_xy.append(np.asarray(geometry.xy[int(row)], dtype=np.float32))
    return (
        np.asarray(output_tracks, dtype=np.int64),
        np.stack(output_xyz, axis=0).astype(np.float32),
        np.asarray(output_ids, dtype=np.str_),
        np.stack(output_xy, axis=0).astype(np.float32),
        {
            "candidate_track_count": int(len(tracks)),
            "maplet_support_anchor_count": int(len(output_tracks)),
            "maplet_support_image_count": int(len(set(output_ids))),
        },
    )


def _fixed_candidate_support_view_anchor_layout(
    *,
    candidate_tracks: np.ndarray,
    candidate_support_image_ids: np.ndarray,
    bank_tracks: np.ndarray,
    bank_xyz: np.ndarray,
    geometry: SupportObservationGeometryIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Freeze exactly the candidate/view support observations from the overlay."""

    candidates = np.asarray(candidate_tracks, dtype=np.int64)
    support_ids = np.asarray(candidate_support_image_ids).astype(str)
    canonical_tracks = np.asarray(bank_tracks, dtype=np.int64).reshape(-1)
    canonical_xyz = np.asarray(bank_xyz, dtype=np.float32).reshape(-1, 3)
    if (
        candidates.ndim != 2
        or candidates.shape != (128, FIXED_CANDIDATE_TOP_K)
        or np.any(candidates < 0)
        or support_ids.ndim != 3
        or support_ids.shape[:2] != candidates.shape
        or support_ids.shape[2] <= 0
        or np.any(support_ids == "")
        or canonical_xyz.shape != (len(canonical_tracks), 3)
        or len(np.unique(canonical_tracks)) != len(canonical_tracks)
    ):
        raise ValueError("candidate-specific support-view anchor layout is invalid")
    xyz_lookup = {
        int(track_id): canonical_xyz[row]
        for row, track_id in enumerate(canonical_tracks.tolist())
    }
    pairs = sorted(
        {
            (str(image_id), int(track_id))
            for track_row, support_row in zip(candidates.tolist(), support_ids.tolist())
            for track_id, views in zip(track_row, support_row)
            for image_id in views
        },
        key=lambda item: (item[0], item[1]),
    )
    if not pairs or any(track_id not in xyz_lookup for _image_id, track_id in pairs):
        raise ValueError("candidate-specific support-view anchor has an unknown track")
    output_tracks: list[int] = []
    output_xyz: list[np.ndarray] = []
    output_ids: list[str] = []
    output_xy: list[np.ndarray] = []
    for image_id in sorted({image_id for image_id, _track_id in pairs}):
        image_tracks = np.asarray(
            [track_id for pair_image, track_id in pairs if pair_image == image_id],
            dtype=np.int64,
        )
        geometry_rows = geometry.geometry_rows_for_tracks(str(image_id), image_tracks)
        if np.any(geometry_rows < 0):
            raise ValueError(
                "candidate-specific support view is absent from frozen support geometry"
            )
        for track_id, geometry_row in zip(image_tracks.tolist(), geometry_rows.tolist()):
            output_tracks.append(int(track_id))
            output_xyz.append(np.asarray(xyz_lookup[int(track_id)], dtype=np.float32))
            output_ids.append(str(image_id))
            output_xy.append(np.asarray(geometry.xy[int(geometry_row)], dtype=np.float32))
    return (
        np.asarray(output_tracks, dtype=np.int64),
        np.stack(output_xyz, axis=0).astype(np.float32),
        np.asarray(output_ids, dtype=np.str_),
        np.stack(output_xy, axis=0).astype(np.float32),
        {
            "candidate_track_count": int(len(np.unique(candidates))),
            "maplet_support_anchor_count": int(len(output_tracks)),
            "maplet_support_image_count": int(len(set(output_ids))),
            "candidate_support_view_count": int(support_ids.shape[2]),
        },
    )


def _lift_fixed_maplet_union_modes(
    *,
    cache: object,
    coordinate_contract: object,
    query_id: str,
    fit_query_xy: np.ndarray,
    fit_exclusion_radius_px: float,
    anchor_track_ids: np.ndarray,
    anchor_xyz: np.ndarray,
    anchor_support_image_ids: np.ndarray,
    anchor_support_xy: np.ndarray,
    support_snap_radius_px: float,
    query_mode_nms_radius_px: float,
    max_modes_per_anchor: int,
    min_support_views: int,
    consensus_radius_px: float,
    max_modes_per_track: int,
) -> tuple[FrozenLiftedTrackModeBank, dict[str, int]]:
    """Lift cache endpoints at frozen maplet anchors, excluding PnP fit content."""

    fit_xy = np.asarray(fit_query_xy, dtype=np.float32).reshape(-1, 2)
    anchor_ids = np.asarray(anchor_support_image_ids).astype(str).reshape(-1)
    tracks = np.asarray(anchor_track_ids, dtype=np.int64).reshape(-1)
    xyz = np.asarray(anchor_xyz, dtype=np.float32).reshape(-1, 3)
    support_xy = np.asarray(anchor_support_xy, dtype=np.float32).reshape(-1, 2)
    if (
        len(fit_xy) == 0
        or len(tracks) == 0
        or anchor_ids.shape != tracks.shape
        or xyz.shape != (len(tracks), 3)
        or support_xy.shape != (len(tracks), 2)
        or np.any(anchor_ids == "")
        or np.any(tracks < 0)
        or not np.isfinite(float(fit_exclusion_radius_px))
        or float(fit_exclusion_radius_px) <= 0.0
    ):
        raise ValueError("fixed maplet LoFTR lifting inputs are invalid")
    fit_tree = cKDTree(fit_xy)
    raw_tracks: list[np.ndarray] = []
    raw_xyz: list[np.ndarray] = []
    raw_support_ids: list[np.ndarray] = []
    raw_query_xy: list[np.ndarray] = []
    raw_confidence: list[np.ndarray] = []
    raw_distances: list[np.ndarray] = []
    support_images_with_matches = 0
    support_images_with_lifted_modes = 0
    for image_id in np.unique(anchor_ids).tolist():
        selected = np.flatnonzero(anchor_ids == str(image_id))
        pair_index = cache.image_index(str(image_id))
        query_source_xy, support_source_xy, confidence = cache.matches_for_index(pair_index)
        if len(confidence) == 0:
            continue
        support_images_with_matches += 1
        query_model_xy = coordinate_contract.source_to_model(
            query_source_xy,
            image_ids=np.repeat(np.asarray([query_id]), len(query_source_xy)),
        )
        support_model_xy = coordinate_contract.source_to_model(
            support_source_xy,
            image_ids=np.repeat(np.asarray([str(image_id)]), len(support_source_xy)),
        )
        heldout = fit_tree.query(query_model_xy, k=1)[0] > float(fit_exclusion_radius_px)
        if not np.any(heldout):
            continue
        modes = select_loftr_modes_near_support_anchors(
            support_anchor_xy=support_xy[selected],
            matched_query_xy=query_model_xy[heldout],
            matched_support_xy=support_model_xy[heldout],
            match_confidence=np.asarray(confidence, dtype=np.float32)[heldout],
            support_snap_radius_px=float(support_snap_radius_px),
            max_modes_per_anchor=int(max_modes_per_anchor),
            query_mode_nms_radius_px=float(query_mode_nms_radius_px),
        )
        if len(modes.anchor_indices) == 0:
            continue
        support_images_with_lifted_modes += 1
        source_anchor_rows = selected[modes.anchor_indices]
        raw_tracks.append(tracks[source_anchor_rows])
        raw_xyz.append(xyz[source_anchor_rows])
        raw_support_ids.append(
            np.repeat(np.asarray([str(image_id)]), len(source_anchor_rows))
        )
        raw_query_xy.append(modes.query_xy)
        raw_confidence.append(modes.confidence)
        raw_distances.append(modes.support_distance_px)
    if not raw_tracks:
        raise ValueError("LoFTR maplet union has no held-out support-anchor modes")
    bank = build_frozen_lifted_track_mode_bank(
        track_ids=np.concatenate(raw_tracks, axis=0),
        xyz=np.concatenate(raw_xyz, axis=0),
        support_image_ids=np.concatenate(raw_support_ids, axis=0),
        query_xy=np.concatenate(raw_query_xy, axis=0),
        confidence=np.concatenate(raw_confidence, axis=0),
        support_distance_px=np.concatenate(raw_distances, axis=0),
        min_support_views=int(min_support_views),
        consensus_radius_px=float(consensus_radius_px),
        max_modes_per_track=int(max_modes_per_track),
    )
    return bank, {
        "support_images_with_cached_matches": int(support_images_with_matches),
        "support_images_with_lifted_anchor_modes": int(support_images_with_lifted_modes),
        "raw_anchor_mode_count": int(sum(len(values) for values in raw_tracks)),
        "cross_view_track_count": int(len(bank.track_ids)),
        "cross_view_mode_count": int(len(bank.mode_query_xy)),
    }


def _lift_fixed_candidate_support_view_modes(
    *,
    cache: object,
    coordinate_contract: object,
    query_id: str,
    fit_query_xy: np.ndarray,
    fit_exclusion_radius_px: float,
    anchor_track_ids: np.ndarray,
    anchor_xyz: np.ndarray,
    anchor_support_image_ids: np.ndarray,
    anchor_support_xy: np.ndarray,
    support_snap_radius_px: float,
    query_mode_nms_radius_px: float,
    max_modes_per_anchor: int,
    base_support_reliability: float,
) -> tuple[FrozenLiftedSupportViewModeBank, dict[str, int]]:
    """Lift held-out endpoints while preserving support-view identities."""

    fit_xy = np.asarray(fit_query_xy, dtype=np.float32).reshape(-1, 2)
    anchor_ids = np.asarray(anchor_support_image_ids).astype(str).reshape(-1)
    tracks = np.asarray(anchor_track_ids, dtype=np.int64).reshape(-1)
    xyz = np.asarray(anchor_xyz, dtype=np.float32).reshape(-1, 3)
    support_xy = np.asarray(anchor_support_xy, dtype=np.float32).reshape(-1, 2)
    if (
        len(fit_xy) == 0
        or len(tracks) == 0
        or anchor_ids.shape != tracks.shape
        or xyz.shape != (len(tracks), 3)
        or support_xy.shape != (len(tracks), 2)
        or np.any(anchor_ids == "")
        or np.any(tracks < 0)
        or not np.isfinite(float(fit_exclusion_radius_px))
        or float(fit_exclusion_radius_px) <= 0.0
    ):
        raise ValueError("candidate support-view LoFTR lifting inputs are invalid")
    fit_tree = cKDTree(fit_xy)
    raw_tracks: list[np.ndarray] = []
    raw_xyz: list[np.ndarray] = []
    raw_support_ids: list[np.ndarray] = []
    raw_query_xy: list[np.ndarray] = []
    raw_confidence: list[np.ndarray] = []
    raw_distances: list[np.ndarray] = []
    support_images_with_matches = 0
    support_images_with_lifted_modes = 0
    for image_id in np.unique(anchor_ids).tolist():
        selected = np.flatnonzero(anchor_ids == str(image_id))
        pair_index = cache.image_index(str(image_id))
        query_source_xy, support_source_xy, confidence = cache.matches_for_index(pair_index)
        if len(confidence) == 0:
            continue
        support_images_with_matches += 1
        query_model_xy = coordinate_contract.source_to_model(
            query_source_xy,
            image_ids=np.repeat(np.asarray([query_id]), len(query_source_xy)),
        )
        support_model_xy = coordinate_contract.source_to_model(
            support_source_xy,
            image_ids=np.repeat(np.asarray([str(image_id)]), len(support_source_xy)),
        )
        heldout = fit_tree.query(query_model_xy, k=1)[0] > float(fit_exclusion_radius_px)
        if not np.any(heldout):
            continue
        modes = select_loftr_modes_near_support_anchors(
            support_anchor_xy=support_xy[selected],
            matched_query_xy=query_model_xy[heldout],
            matched_support_xy=support_model_xy[heldout],
            match_confidence=np.asarray(confidence, dtype=np.float32)[heldout],
            support_snap_radius_px=float(support_snap_radius_px),
            max_modes_per_anchor=int(max_modes_per_anchor),
            query_mode_nms_radius_px=float(query_mode_nms_radius_px),
        )
        if len(modes.anchor_indices) == 0:
            continue
        support_images_with_lifted_modes += 1
        source_anchor_rows = selected[modes.anchor_indices]
        raw_tracks.append(tracks[source_anchor_rows])
        raw_xyz.append(xyz[source_anchor_rows])
        raw_support_ids.append(
            np.repeat(np.asarray([str(image_id)]), len(source_anchor_rows))
        )
        raw_query_xy.append(modes.query_xy)
        raw_confidence.append(modes.confidence)
        raw_distances.append(modes.support_distance_px)
    if not raw_tracks:
        raise ValueError("candidate support-view union has no held-out LoFTR modes")
    bank = build_frozen_lifted_support_view_mode_bank(
        track_ids=np.concatenate(raw_tracks, axis=0),
        xyz=np.concatenate(raw_xyz, axis=0),
        support_image_ids=np.concatenate(raw_support_ids, axis=0),
        query_xy=np.concatenate(raw_query_xy, axis=0),
        confidence=np.concatenate(raw_confidence, axis=0),
        support_distance_px=np.concatenate(raw_distances, axis=0),
        max_modes_per_support_view=int(max_modes_per_anchor),
        base_support_reliability=float(base_support_reliability),
    )
    return bank, {
        "support_images_with_cached_matches": int(support_images_with_matches),
        "support_images_with_lifted_anchor_modes": int(support_images_with_lifted_modes),
        "raw_anchor_mode_count": int(sum(len(values) for values in raw_tracks)),
        "support_view_slot_count": int(len(bank.support_slot_track_indices)),
        "support_view_mode_count": int(len(bank.mode_query_xy)),
    }


def _active_bank_for_variant(
    bank: FrozenLiftedTrackModeBank, *, query_id: str, evidence_variant: str
) -> tuple[FrozenLiftedTrackModeBank, np.ndarray | None]:
    if str(evidence_variant) == "visual":
        return bank, None
    if str(evidence_variant) != "xyz_permutation_control":
        raise ValueError("lifted LoFTR evidence variant is invalid")
    permutation = deterministic_track_xyz_permutation(bank.track_ids, query_id=str(query_id))
    return (
        FrozenLiftedTrackModeBank(
            track_ids=bank.track_ids,
            xyz=bank.xyz[permutation],
            mode_offsets=bank.mode_offsets,
            mode_query_xy=bank.mode_query_xy,
            mode_weights=bank.mode_weights,
            mode_support_image_ids=bank.mode_support_image_ids,
            mode_support_view_counts=bank.mode_support_view_counts,
            mode_confidence_sums=bank.mode_confidence_sums,
            track_reliabilities=bank.track_reliabilities,
            track_reference_xy=bank.track_reference_xy,
        ),
        permutation,
    )


def _active_support_view_bank_for_variant(
    bank: FrozenLiftedSupportViewModeBank,
    *,
    query_id: str,
    evidence_variant: str,
) -> tuple[FrozenLiftedSupportViewModeBank, np.ndarray | None]:
    if str(evidence_variant) == "visual":
        return bank, None
    if str(evidence_variant) != "xyz_permutation_control":
        raise ValueError("candidate support-view LoFTR evidence variant is invalid")
    permutation = deterministic_track_xyz_permutation(bank.track_ids, query_id=str(query_id))
    return (
        FrozenLiftedSupportViewModeBank(
            track_ids=bank.track_ids,
            xyz=bank.xyz[permutation],
            support_slot_track_indices=bank.support_slot_track_indices,
            support_slot_image_ids=bank.support_slot_image_ids,
            mode_offsets=bank.mode_offsets,
            mode_query_xy=bank.mode_query_xy,
            mode_weights=bank.mode_weights,
            mode_confidence_sums=bank.mode_confidence_sums,
            support_reliabilities=bank.support_reliabilities,
            support_reference_xy=bank.support_reference_xy,
        ),
        permutation,
    )


def _score_fixed_hypotheses(
    *,
    poses_w2c: np.ndarray,
    bank: FrozenLiftedTrackModeBank,
    candidate_groups: FrozenCandidateGroupLayout,
    camera: object,
    sigma_profiles: Sequence[float],
    group_null_masses: Sequence[float],
    out_of_image_ratio: float,
    max_log_ratio: float,
    hypothesis_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, object]]]:
    if (
        int(getattr(camera, "model_id")) != 2
        or int(hypothesis_batch_size) <= 0
        or not 0.0 < float(out_of_image_ratio) < 1.0
        or float(max_log_ratio) <= 0.0
    ):
        raise ValueError("lifted LoFTR hypothesis scorer configuration is invalid")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4:
        raise ValueError("lifted LoFTR map-to-query scorer requires SIMPLE_RADIAL camera")
    poses = np.asarray(poses_w2c, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or np.any(~np.isfinite(poses)):
        raise ValueError("frozen hypotheses are invalid")
    profile_specs: list[dict[str, object]] = []
    for sigma in sigma_profiles:
        profile_specs.append(
            {
                "name": f"track_sigma{float(sigma):g}",
                "aggregation": "global_track_union_diagnostic",
                "sigma_px": float(sigma),
                "null_mass": None,
            }
        )
        for null_mass in group_null_masses:
            profile_specs.append(
                {
                    "name": f"group_uniform_null{float(null_mass):g}_sigma{float(sigma):g}",
                    "aggregation": "fixed_candidate_group_latent_identity_mixture",
                    "sigma_px": float(sigma),
                    "null_mass": float(null_mass),
                }
            )
        profile_specs.append(
            {
                "name": f"group_fixedprior_sigma{float(sigma):g}",
                "aggregation": "fixed_candidate_group_latent_identity_prior_mixture",
                "sigma_px": float(sigma),
                "null_mass": "per_group_fixed_candidate_prior",
            }
        )
    profile_names = [str(item["name"]) for item in profile_specs]
    outputs: dict[str, list[np.ndarray]] = {
        f"{field}": []
        for field in (
            "log_likelihood_means",
            "log_likelihood_medians",
            "log_likelihood_worst_quartile_means",
            "spatial_median_of_means_2x2",
        )
    }
    outputs["projection_visible_fractions"] = []
    xyz = torch.as_tensor(bank.xyz, dtype=torch.float32, device=device)
    width, height = (int(getattr(camera, "width")), int(getattr(camera, "height")))
    for start in range(0, len(poses), int(hypothesis_batch_size)):
        stop = min(start + int(hypothesis_batch_size), len(poses))
        pose_tensor = torch.as_tensor(poses[start:stop], dtype=torch.float32, device=device)
        projected, projection_valid = project_simple_radial_torch(
            xyz,
            pose_tensor,
            focal_length=params[0],
            principal_x=params[1],
            principal_y=params[2],
            radial_k=params[3],
            image_width=width,
            image_height=height,
        )
        outputs["projection_visible_fractions"].append(
            projection_valid.float().mean(dim=1).detach().cpu().numpy()
        )
        per_profile: dict[str, dict[str, torch.Tensor]] = {}
        for sigma in sigma_profiles:
            track_profile_name = f"track_sigma{float(sigma):g}"
            ratios = track_mode_log_ratios_from_projected(
                projected_xy=projected,
                projection_valid=projection_valid,
                bank=bank,
                image_width=width,
                image_height=height,
                sigma_px=float(sigma),
                out_of_image_ratio=float(out_of_image_ratio),
                max_log_ratio=float(max_log_ratio),
            )
            per_profile[track_profile_name] = robust_track_log_ratio_statistics(
                track_log_ratios=ratios,
                track_reference_xy=bank.track_reference_xy,
                image_width=width,
                image_height=height,
            )
            for null_mass in group_null_masses:
                group_profile_name = (
                    f"group_uniform_null{float(null_mass):g}_sigma{float(sigma):g}"
                )
                group_ratios = group_log_ratios_from_track_log_ratios(
                    track_log_ratios=ratios,
                    layout=candidate_groups,
                    null_mass=float(null_mass),
                    max_log_ratio=float(max_log_ratio),
                )
                per_profile[group_profile_name] = robust_track_log_ratio_statistics(
                    track_log_ratios=group_ratios[:, candidate_groups.active_group_mask],
                    track_reference_xy=candidate_groups.group_reference_xy[
                        candidate_groups.active_group_mask
                    ],
                    image_width=width,
                    image_height=height,
                )
            fixed_prior_name = f"group_fixedprior_sigma{float(sigma):g}"
            fixed_prior_ratios = fixed_prior_group_log_ratios_from_track_log_ratios(
                track_log_ratios=ratios,
                layout=candidate_groups,
                max_log_ratio=float(max_log_ratio),
            )
            per_profile[fixed_prior_name] = robust_track_log_ratio_statistics(
                track_log_ratios=fixed_prior_ratios[:, candidate_groups.active_group_mask],
                track_reference_xy=candidate_groups.group_reference_xy[
                    candidate_groups.active_group_mask
                ],
                image_width=width,
                image_height=height,
            )
        for field, key in (
            ("log_likelihood_means", "means"),
            ("log_likelihood_medians", "medians"),
            ("log_likelihood_worst_quartile_means", "worst_quartile_means"),
            ("spatial_median_of_means_2x2", "spatial_median_of_means_2x2"),
        ):
            outputs[field].append(
                np.column_stack(
                    [per_profile[name][key].detach().cpu().numpy() for name in profile_names]
                )
            )
    return (
        {
            key: np.concatenate(values, axis=0).astype(np.float64, copy=False)
            for key, values in outputs.items()
        },
        np.asarray(profile_names, dtype=np.str_),
        profile_specs,
    )


def _score_candidate_support_view_hypotheses(
    *,
    poses_w2c: np.ndarray,
    bank: FrozenLiftedSupportViewModeBank,
    candidate_groups: FrozenCandidateSupportViewGroupLayout,
    camera: object,
    sigma_profiles: Sequence[float],
    out_of_image_ratio: float,
    max_log_ratio: float,
    hypothesis_batch_size: int,
    device: torch.device,
    diagnostic_dump_group_terms: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    list[dict[str, object]],
    dict[str, np.ndarray] | None,
]:
    """Score fixed poses with explicit candidate and support-view latents."""

    if (
        int(getattr(camera, "model_id")) != 2
        or int(hypothesis_batch_size) <= 0
        or not 0.0 < float(out_of_image_ratio) < 1.0
        or float(max_log_ratio) <= 0.0
    ):
        raise ValueError("candidate support-view LoFTR scorer configuration is invalid")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4:
        raise ValueError("candidate support-view LoFTR scorer requires SIMPLE_RADIAL camera")
    poses = np.asarray(poses_w2c, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or np.any(~np.isfinite(poses)):
        raise ValueError("candidate support-view frozen hypotheses are invalid")
    profile_specs = [
        {
            "name": f"candidate_view_fixedprior_sigma{float(sigma):g}",
            "aggregation": (
                "candidate_specific_fixed_identity_and_support_view_prior_mixture"
            ),
            "sigma_px": float(sigma),
            "null_mass": "per_group_fixed_candidate_prior",
        }
        for sigma in sigma_profiles
    ]
    profile_names = [str(item["name"]) for item in profile_specs]
    outputs: dict[str, list[np.ndarray]] = {
        field: []
        for field in (
            "log_likelihood_means",
            "log_likelihood_medians",
            "log_likelihood_worst_quartile_means",
            "spatial_median_of_means_2x2",
        )
    }
    outputs["projection_visible_fractions"] = []
    diagnostic_outputs: dict[str, list[np.ndarray]] | None = None
    if bool(diagnostic_dump_group_terms):
        diagnostic_outputs = {
            "support_view_slot_log_ratios": [],
            "candidate_log_ratios": [],
            "group_log_ratios": [],
            "projected_xy": [],
            "projection_valid": [],
        }
    xyz = torch.as_tensor(bank.xyz, dtype=torch.float32, device=device)
    width, height = (int(getattr(camera, "width")), int(getattr(camera, "height")))
    for start in range(0, len(poses), int(hypothesis_batch_size)):
        stop = min(start + int(hypothesis_batch_size), len(poses))
        pose_tensor = torch.as_tensor(poses[start:stop], dtype=torch.float32, device=device)
        projected, projection_valid = project_simple_radial_torch(
            xyz,
            pose_tensor,
            focal_length=params[0],
            principal_x=params[1],
            principal_y=params[2],
            radial_k=params[3],
            image_width=width,
            image_height=height,
        )
        outputs["projection_visible_fractions"].append(
            projection_valid.float().mean(dim=1).detach().cpu().numpy()
        )
        per_profile_support_view: list[np.ndarray] = []
        per_profile_candidate: list[np.ndarray] = []
        per_profile_group: list[np.ndarray] = []
        per_profile: dict[str, dict[str, torch.Tensor]] = {}
        for sigma in sigma_profiles:
            support_ratios = support_view_log_ratios_from_projected(
                projected_xy=projected,
                projection_valid=projection_valid,
                bank=bank,
                image_width=width,
                image_height=height,
                sigma_px=float(sigma),
                out_of_image_ratio=float(out_of_image_ratio),
                max_log_ratio=float(max_log_ratio),
            )
            candidate_ratios, group_ratios = (
                candidate_support_view_log_mixture_terms_from_support_view_log_ratios(
                    support_view_log_ratios=support_ratios,
                    layout=candidate_groups,
                    max_log_ratio=float(max_log_ratio),
                )
            )
            profile_name = f"candidate_view_fixedprior_sigma{float(sigma):g}"
            per_profile[profile_name] = robust_track_log_ratio_statistics(
                track_log_ratios=group_ratios[:, candidate_groups.active_group_mask],
                track_reference_xy=candidate_groups.group_reference_xy[
                    candidate_groups.active_group_mask
                ],
                image_width=width,
                image_height=height,
            )
            if diagnostic_outputs is not None:
                per_profile_support_view.append(
                    support_ratios.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                per_profile_candidate.append(
                    candidate_ratios.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                per_profile_group.append(
                    group_ratios.detach().cpu().numpy().astype(np.float32, copy=False)
                )
        for field, key in (
            ("log_likelihood_means", "means"),
            ("log_likelihood_medians", "medians"),
            ("log_likelihood_worst_quartile_means", "worst_quartile_means"),
            ("spatial_median_of_means_2x2", "spatial_median_of_means_2x2"),
        ):
            outputs[field].append(
                np.column_stack(
                    [per_profile[name][key].detach().cpu().numpy() for name in profile_names]
                )
            )
        if diagnostic_outputs is not None:
            diagnostic_outputs["support_view_slot_log_ratios"].append(
                np.stack(per_profile_support_view, axis=1)
            )
            diagnostic_outputs["candidate_log_ratios"].append(
                np.stack(per_profile_candidate, axis=1)
            )
            diagnostic_outputs["group_log_ratios"].append(
                np.stack(per_profile_group, axis=1)
            )
            diagnostic_outputs["projected_xy"].append(
                projected.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            diagnostic_outputs["projection_valid"].append(
                projection_valid.detach().cpu().numpy().astype(bool, copy=False)
            )
    diagnostics: dict[str, np.ndarray] | None = None
    if diagnostic_outputs is not None:
        diagnostics = {
            key: np.concatenate(values, axis=0)
            for key, values in diagnostic_outputs.items()
        }
    return (
        {
            key: np.concatenate(values, axis=0).astype(np.float64, copy=False)
            for key, values in outputs.items()
        },
        np.asarray(profile_names, dtype=np.str_),
        profile_specs,
        diagnostics,
    )


def _bank_layout_digest(bank: FrozenLiftedTrackModeBank) -> str:
    return _array_digest(
        {
            "track_ids": bank.track_ids,
            "mode_offsets": bank.mode_offsets,
            "mode_query_xy": bank.mode_query_xy,
            "mode_weights": bank.mode_weights,
            "mode_support_image_ids": bank.mode_support_image_ids,
            "mode_support_view_counts": bank.mode_support_view_counts,
            "mode_confidence_sums": bank.mode_confidence_sums,
            "track_reliabilities": bank.track_reliabilities,
            "track_reference_xy": bank.track_reference_xy,
        }
    )


def _support_view_bank_layout_digest(bank: FrozenLiftedSupportViewModeBank) -> str:
    return _array_digest(
        {
            "track_ids": bank.track_ids,
            "support_slot_track_indices": bank.support_slot_track_indices,
            "support_slot_image_ids": bank.support_slot_image_ids,
            "mode_offsets": bank.mode_offsets,
            "mode_query_xy": bank.mode_query_xy,
            "mode_weights": bank.mode_weights,
            "mode_confidence_sums": bank.mode_confidence_sums,
            "support_reliabilities": bank.support_reliabilities,
            "support_reference_xy": bank.support_reference_xy,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite lifted LoFTR score artifact: {output_path}")
    sigma_profiles = _parse_sigmas(args.sigma_px_profiles)
    group_null_masses = _parse_group_null_masses(args.group_null_masses)
    diagnostic_hypothesis_indices = _parse_diagnostic_hypothesis_indices(
        args.diagnostic_hypothesis_indices
    )
    diagnostic_dump_group_terms = bool(args.diagnostic_dump_group_terms)
    evidence_layout = str(args.evidence_layout)
    support_view_overlay_requested = bool(
        str(args.fixed_candidate_support_view_overlay).strip()
    )
    if (
        int(args.fixed_candidate_top_k) != FIXED_CANDIDATE_TOP_K
        or int(args.fixed_support_view_count) <= 0
        or int(args.max_modes_per_anchor) <= 0
        or int(args.hypothesis_batch_size) <= 0
        or int(args.hypothesis_limit) < 0
        or (
            bool(diagnostic_hypothesis_indices)
            and int(args.hypothesis_limit) != 0
        )
        or (bool(diagnostic_hypothesis_indices) != diagnostic_dump_group_terms)
        or (
            diagnostic_dump_group_terms
            and evidence_layout != EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
        )
        or not np.isfinite(
            [
                args.fit_exclusion_radius_px,
                args.support_snap_radius_px,
                args.query_mode_nms_radius_px,
                args.consensus_radius_px,
                args.out_of_image_ratio,
                args.max_log_ratio,
                args.base_support_reliability,
            ]
        ).all()
        or args.fit_exclusion_radius_px <= 0.0
        or args.support_snap_radius_px <= 0.0
        or args.query_mode_nms_radius_px < 0.0
        or args.consensus_radius_px <= 0.0
        or not 0.0 < args.out_of_image_ratio < 1.0
        or args.max_log_ratio <= 0.0
        or not 0.0 < args.base_support_reliability <= 1.0
        or (
            evidence_layout == EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION
            and (int(args.max_modes_per_track) <= 0 or int(args.min_support_views) < 2)
        )
        or (
            evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
            and not support_view_overlay_requested
        )
        or (
            evidence_layout == EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION
            and support_view_overlay_requested
        )
    ):
        raise ValueError("lifted LoFTR P1 configuration is invalid")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("lifted LoFTR P1 requested unavailable CUDA device")
    paths = {
        "hypothesis_artifact": Path(args.hypothesis_artifact),
        "baseline_score_artifact": Path(args.baseline_score_artifact),
        "detector_query_cache": Path(args.detector_query_cache),
        "proposals": Path(args.proposals),
        "candidate_artifact": Path(args.candidate_artifact),
        "fixed_candidate_prior_overlay": Path(args.fixed_candidate_prior_overlay),
        "maplet_support_index": Path(args.maplet_support_index),
        "support_geometry_index": Path(args.support_geometry_index),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "loftr_pair_cache": Path(args.loftr_pair_cache),
        "loftr_checkpoint": Path(args.loftr_checkpoint),
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin_camera_ownership_only": Path(args.colmap_model_dir) / "images.bin",
    }
    if support_view_overlay_requested:
        paths["fixed_candidate_support_view_overlay"] = Path(
            args.fixed_candidate_support_view_overlay
        )
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    exact, hypothesis_metadata, baseline_metadata = _load_exact_hypotheses(
        hypothesis_path=paths["hypothesis_artifact"],
        baseline_path=paths["baseline_score_artifact"],
        detector_path=paths["detector_query_cache"],
        proposals_path=paths["proposals"],
        candidate_path=paths["candidate_artifact"],
        prior_path=paths["fixed_candidate_prior_overlay"],
        fixed_candidate_top_k=FIXED_CANDIDATE_TOP_K,
    )
    if diagnostic_hypothesis_indices:
        # The scorer remains target-free.  A separate target-only P0 audit may
        # name this immutable subset, but this artifact is intentionally
        # ineligible for any formal P1 audit or production use.
        exact = _select_explicit_hypothesis_indices(
            exact, diagnostic_hypothesis_indices
        )
    query_ids = np.unique(np.asarray(exact["query_ids"]).astype(str))
    if len(query_ids) != 1:
        raise ValueError("one lifted LoFTR scoring artifact must contain exactly one query")
    query_id = str(query_ids[0])
    cache = load_frozen_loftr_pair_cache(paths["loftr_pair_cache"])
    _validate_pair_cache_lineage(
        cache=cache,
        query_id=query_id,
        maplet_support_index=paths["maplet_support_index"],
        image_root=Path(args.image_root),
        loftr_checkpoint=paths["loftr_checkpoint"],
    )
    coordinate_contract = build_frozen_loftr_colmap_coordinate_contract(
        cache_metadata=cache.metadata,
        query_id=query_id,
        support_image_ids=cache.support_image_ids.tolist(),
        colmap_model_dir=Path(args.colmap_model_dir),
    )
    fit_rows, fit_xy, fit_candidate_tracks, candidate_metadata = _load_fit_rows_and_points(
        candidate_artifact=paths["candidate_artifact"],
        detector_query_cache=paths["detector_query_cache"],
        proposals=paths["proposals"],
        query_id=query_id,
    )
    with np.load(paths["proposals"], allow_pickle=False) as proposal_payload:
        proposal_fields = ("query_ids", "candidate_track_ids")
        missing = sorted(set(proposal_fields).difference(proposal_payload.files))
        if missing:
            raise ValueError(
                "fixed candidate prior source lacks proposal fields: " + ", ".join(missing)
            )
        proposal_for_prior = {
            field: np.asarray(proposal_payload[field]).copy()
            for field in proposal_fields
        }
    prior_overlay, prior_metadata = _load_candidate_prior_overlay(
        paths["fixed_candidate_prior_overlay"],
        proposals_path=paths["proposals"],
        proposals=proposal_for_prior,
    )
    fit_candidate_probabilities = np.asarray(
        prior_overlay["candidate_probabilities"], dtype=np.float32
    )[fit_rows]
    fit_null_probabilities = np.asarray(
        prior_overlay["null_probabilities"], dtype=np.float32
    )[fit_rows]
    if not np.array_equal(
        fit_candidate_tracks,
        np.asarray(prior_overlay["candidate_track_ids"], dtype=np.int64)[fit_rows],
    ):
        raise ValueError("fixed candidate identity priors do not match PnP top-L tracks")
    support_view_overlay: dict[str, np.ndarray] | None = None
    support_view_metadata: dict[str, object] | None = None
    fit_support_view_probabilities: np.ndarray | None = None
    fit_candidate_support_image_ids: np.ndarray | None = None
    if evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW:
        support_view_overlay, support_view_metadata = _load_candidate_support_view_overlay(
            path=paths["fixed_candidate_support_view_overlay"],
            proposals_path=paths["proposals"],
            proposals=proposal_for_prior,
            candidate_prior_path=paths["fixed_candidate_prior_overlay"],
            maplet_support_index=paths["maplet_support_index"],
        )
        if not np.array_equal(
            fit_candidate_tracks,
            np.asarray(support_view_overlay["candidate_track_ids"], dtype=np.int64)[fit_rows],
        ):
            raise ValueError("support-view overlay does not match fixed PnP top-L tracks")
        fit_support_view_probabilities = np.asarray(
            support_view_overlay["support_view_probabilities"], dtype=np.float32
        )[fit_rows]
        fit_candidate_support_image_ids = np.asarray(
            support_view_overlay["candidate_support_image_ids"]
        ).astype(str)[fit_rows]
        if (
            fit_support_view_probabilities.shape[:2] != fit_candidate_tracks.shape
            or fit_candidate_support_image_ids.shape != fit_support_view_probabilities.shape
            or int(args.fixed_support_view_count)
            != int(fit_support_view_probabilities.shape[2])
        ):
            raise ValueError(
                "candidate-specific LoFTR support-view count differs from its frozen overlay"
            )
    bank_tracks, bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    maplet_tracks, maplet_image_ids, maplet_image_indices, maplet_coverage, maplet_metadata = (
        _load_maplet_support_fields(paths["maplet_support_index"])
    )
    if str(maplet_metadata.get("source_landmark_index_sha256", "")) != file_sha256_short(
        paths["projected_landmark_bank"]
    ):
        raise ValueError("maplet support index was built from a different landmark bank")
    for metadata_key, path_key in (
        ("maplet_support_index_sha256", "maplet_support_index"),
        ("support_geometry_index_sha256", "support_geometry_index"),
        ("projected_landmark_bank_sha256", "projected_landmark_bank"),
    ):
        if str(candidate_metadata.get(metadata_key, "")) != file_sha256_short(paths[path_key]):
            raise ValueError("candidate maplet union lineage is stale or mismatched")
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        paths["support_geometry_index"]
    )
    if evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW:
        assert fit_support_view_probabilities is not None
        assert fit_candidate_support_image_ids is not None
        anchor_tracks, anchor_xyz, anchor_ids, anchor_xy, union_stats = (
            _fixed_candidate_support_view_anchor_layout(
                candidate_tracks=fit_candidate_tracks,
                candidate_support_image_ids=fit_candidate_support_image_ids,
                bank_tracks=bank_tracks,
                bank_xyz=bank_xyz,
                geometry=geometry,
            )
        )
    else:
        anchor_tracks, anchor_xyz, anchor_ids, anchor_xy, union_stats = (
            _fixed_candidate_maplet_anchor_layout(
                candidate_tracks=fit_candidate_tracks,
                fixed_support_view_count=int(args.fixed_support_view_count),
                maplet_track_ids=maplet_tracks,
                maplet_image_ids=maplet_image_ids,
                maplet_image_indices=maplet_image_indices,
                maplet_coverage=maplet_coverage,
                bank_tracks=bank_tracks,
                bank_xyz=bank_xyz,
                geometry=geometry,
            )
        )
    if set(anchor_ids.tolist()).difference(set(cache.support_image_ids.tolist())):
        raise ValueError("fixed maplet union references a support image absent from global LoFTR cache")
    if evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW:
        canonical_bank: FrozenLiftedTrackModeBank | FrozenLiftedSupportViewModeBank
        canonical_bank, lifting_stats = _lift_fixed_candidate_support_view_modes(
            cache=cache,
            coordinate_contract=coordinate_contract,
            query_id=query_id,
            fit_query_xy=fit_xy,
            fit_exclusion_radius_px=float(args.fit_exclusion_radius_px),
            anchor_track_ids=anchor_tracks,
            anchor_xyz=anchor_xyz,
            anchor_support_image_ids=anchor_ids,
            anchor_support_xy=anchor_xy,
            support_snap_radius_px=float(args.support_snap_radius_px),
            query_mode_nms_radius_px=float(args.query_mode_nms_radius_px),
            max_modes_per_anchor=int(args.max_modes_per_anchor),
            base_support_reliability=float(args.base_support_reliability),
        )
        assert isinstance(canonical_bank, FrozenLiftedSupportViewModeBank)
        active_bank, xyz_permutation = _active_support_view_bank_for_variant(
            canonical_bank,
            query_id=query_id,
            evidence_variant=str(args.evidence_variant),
        )
        assert fit_support_view_probabilities is not None
        assert fit_candidate_support_image_ids is not None
        candidate_groups: FrozenCandidateGroupLayout | FrozenCandidateSupportViewGroupLayout
        candidate_groups = build_frozen_candidate_support_view_group_layout(
            candidate_track_ids=fit_candidate_tracks,
            candidate_identity_probabilities=fit_candidate_probabilities,
            null_probabilities=fit_null_probabilities,
            candidate_support_image_ids=fit_candidate_support_image_ids,
            support_view_probabilities=fit_support_view_probabilities,
            bank=canonical_bank,
        )
    else:
        canonical_bank, lifting_stats = _lift_fixed_maplet_union_modes(
            cache=cache,
            coordinate_contract=coordinate_contract,
            query_id=query_id,
            fit_query_xy=fit_xy,
            fit_exclusion_radius_px=float(args.fit_exclusion_radius_px),
            anchor_track_ids=anchor_tracks,
            anchor_xyz=anchor_xyz,
            anchor_support_image_ids=anchor_ids,
            anchor_support_xy=anchor_xy,
            support_snap_radius_px=float(args.support_snap_radius_px),
            query_mode_nms_radius_px=float(args.query_mode_nms_radius_px),
            max_modes_per_anchor=int(args.max_modes_per_anchor),
            min_support_views=int(args.min_support_views),
            consensus_radius_px=float(args.consensus_radius_px),
            max_modes_per_track=int(args.max_modes_per_track),
        )
        active_bank, xyz_permutation = _active_bank_for_variant(
            canonical_bank, query_id=query_id, evidence_variant=str(args.evidence_variant)
        )
        candidate_groups = build_frozen_candidate_group_layout(
            candidate_track_ids=fit_candidate_tracks,
            bank=canonical_bank,
            candidate_identity_probabilities=fit_candidate_probabilities,
            null_probabilities=fit_null_probabilities,
        )
    cameras = read_colmap_cameras_binary(paths["colmap_cameras_bin"])
    image_camera_ids = read_colmap_image_camera_ids_binary(
        paths["colmap_images_bin_camera_ownership_only"]
    )
    camera_id = image_camera_ids.get(query_id)
    if camera_id is None or int(camera_id) not in cameras:
        raise ValueError("query has no declared COLMAP camera ownership")
    camera = cameras[int(camera_id)]
    if (
        (int(camera.width), int(camera.height)) != coordinate_contract.query_model_size
        or int(camera.model_id) != 2
    ):
        raise ValueError("LoFTR coordinate contract and query camera differ")
    poses = np.asarray(exact["poses_w2c"], dtype=np.float64)
    if int(args.hypothesis_limit) > 0:
        poses = poses[: int(args.hypothesis_limit)]
        exact = {key: np.asarray(value)[: len(poses)] for key, value in exact.items()}
    if evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW:
        assert isinstance(canonical_bank, FrozenLiftedSupportViewModeBank)
        assert isinstance(active_bank, FrozenLiftedSupportViewModeBank)
        assert isinstance(candidate_groups, FrozenCandidateSupportViewGroupLayout)
        scored, profile_names, profile_specs, diagnostic_terms = (
            _score_candidate_support_view_hypotheses(
                poses_w2c=poses,
                bank=active_bank,
                candidate_groups=candidate_groups,
                camera=camera,
                sigma_profiles=sigma_profiles,
                out_of_image_ratio=float(args.out_of_image_ratio),
                max_log_ratio=float(args.max_log_ratio),
                hypothesis_batch_size=int(args.hypothesis_batch_size),
                device=device,
                diagnostic_dump_group_terms=diagnostic_dump_group_terms,
            )
        )
        layout_digest = _support_view_bank_layout_digest(canonical_bank)
    else:
        assert isinstance(canonical_bank, FrozenLiftedTrackModeBank)
        assert isinstance(active_bank, FrozenLiftedTrackModeBank)
        assert isinstance(candidate_groups, FrozenCandidateGroupLayout)
        scored, profile_names, profile_specs = _score_fixed_hypotheses(
            poses_w2c=poses,
            bank=active_bank,
            candidate_groups=candidate_groups,
            camera=camera,
            sigma_profiles=sigma_profiles,
            group_null_masses=group_null_masses,
            out_of_image_ratio=float(args.out_of_image_ratio),
            max_log_ratio=float(args.max_log_ratio),
            hypothesis_batch_size=int(args.hypothesis_batch_size),
            device=device,
        )
        diagnostic_terms = None
        layout_digest = _bank_layout_digest(canonical_bank)
    candidate_union_digest = _array_digest(
        {
            "fit_rows": fit_rows,
            "fit_candidate_tracks": fit_candidate_tracks,
            "anchor_tracks": anchor_tracks,
            "anchor_support_image_ids": anchor_ids,
            "anchor_support_xy": anchor_xy,
        }
    )
    candidate_group_digest_payload: dict[str, np.ndarray] = {
        "candidate_track_indices": candidate_groups.candidate_track_indices,
        "group_reference_xy": candidate_groups.group_reference_xy,
        "active_group_mask": candidate_groups.active_group_mask,
        "candidate_identity_probabilities": candidate_groups.candidate_identity_probabilities,
        "null_probabilities": candidate_groups.null_probabilities,
    }
    if isinstance(candidate_groups, FrozenCandidateSupportViewGroupLayout):
        candidate_group_digest_payload.update(
            {
                "candidate_support_slot_indices": (
                    candidate_groups.candidate_support_slot_indices
                ),
                "candidate_support_image_ids": candidate_groups.candidate_support_image_ids,
                "support_view_probabilities": candidate_groups.support_view_probabilities,
            }
        )
    candidate_group_digest = _array_digest(candidate_group_digest_payload)
    strict_contract = {
        "heldout_query_image_content_excludes_pnp_fit_neighborhoods": True,
        "fixed_pnp_fit_topl_maplet_union": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_group_latent_identity_marginalized": True,
        "candidate_group_topl_denominator_fixed": True,
        "candidate_group_explicit_null": True,
        "candidate_group_identity_prior_fixed_before_pose_scoring": True,
        "candidate_group_identity_prior_target_free": True,
        "candidate_groups_without_lifted_evidence_fixed_null_only": True,
        "candidate_group_active_mask_fixed_across_hypotheses": True,
        "pnp_query_center_used_for_group_scoring": False,
        "support_maplet_prefix_fixed": True,
        "support_view_descriptor_averaging": False,
        "support_view_endpoints_averaged_before_likelihood": False,
        "support_view_endpoint_mixture_explicit": True,
        "cross_view_modes_marginalized_not_argmaxed": True,
        "candidate_specific_support_view_posterior": (
            evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
        ),
        "candidate_support_view_posterior_fixed_before_pose_scoring": (
            evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
        ),
        "candidate_support_view_posterior_target_free": (
            evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
        ),
        "missing_support_view_evidence_is_neutral_ratio": (
            evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
        ),
        "query_center_used_for_loftr_mode_selection": False,
        "pose_dependent_correspondence_selection": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "full_mapping_image_pair_cache": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "out_of_image_projection_is_negative_likelihood": True,
        "raw_scores_calibrated_or_promoted": False,
        "raw_scores_must_not_feed_pnp": True,
        "xyz_permutation_control": str(args.evidence_variant) == "xyz_permutation_control",
    }
    metadata: dict[str, Any] = {
        "format": SCORE_FORMAT,
        "version": 1,
        "score_version": SCORE_VERSION,
        "query_id": query_id,
        "split_name": str(np.asarray(exact["split_names"]).astype(str)[0]),
        "row_count": int(len(poses)),
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "evidence_variant": str(args.evidence_variant),
        "evidence_layout": evidence_layout,
        "strict_frozen_lifted_map_to_query_contract": strict_contract,
        "candidate_maplet_union": {
            **union_stats,
            "fixed_candidate_top_k": FIXED_CANDIDATE_TOP_K,
            "fixed_support_view_count": int(args.fixed_support_view_count),
            "candidate_union_digest": candidate_union_digest,
            "candidate_group_count": int(candidate_groups.candidate_track_indices.shape[0]),
            "candidate_group_active_count": int(candidate_groups.active_group_mask.sum()),
            "candidate_group_top_k": int(candidate_groups.candidate_track_indices.shape[1]),
            "candidate_group_digest": candidate_group_digest,
            "candidate_group_prior_source": "fixed_candidate_maplet_probability_overlay",
            "candidate_support_view_posterior_source": (
                None
                if support_view_metadata is None
                else "fixed_candidate_maplet_support_view_overlay"
            ),
        },
        "lifting_config": {
            "fit_exclusion_radius_px": float(args.fit_exclusion_radius_px),
            "support_snap_radius_px": float(args.support_snap_radius_px),
            "query_mode_nms_radius_px": float(args.query_mode_nms_radius_px),
            "min_support_views": int(args.min_support_views),
            "consensus_radius_px": float(args.consensus_radius_px),
            "max_modes_per_anchor": int(args.max_modes_per_anchor),
            "max_modes_per_track": int(args.max_modes_per_track),
            "base_support_reliability": float(args.base_support_reliability),
            "support_anchor_mode_selection": "support_only_no_query_center_v1",
        },
        "score_profiles": [
            {
                **item,
                "out_of_image_ratio": float(args.out_of_image_ratio),
                "max_log_ratio": float(args.max_log_ratio),
            }
            for item in profile_specs
        ],
        "canonical_mode_layout_sha256": layout_digest,
        "canonical_xyz_sha256": _array_digest({"xyz": canonical_bank.xyz}),
        "active_xyz_sha256": _array_digest({"xyz": active_bank.xyz}),
        "xyz_permutation": None
        if xyz_permutation is None
        else xyz_permutation.astype(np.int64).tolist(),
        "coordinate_contract": coordinate_contract.metadata(),
        "pair_cache_contract": cache.metadata["strict_global_pair_contract"],
        "input_metadata_hashes": {
            "hypothesis": _array_digest({"metadata": np.asarray(json.dumps(hypothesis_metadata, sort_keys=True))}),
            "baseline": _array_digest({"metadata": np.asarray(json.dumps(baseline_metadata, sort_keys=True))}),
            "candidate": _array_digest({"metadata": np.asarray(json.dumps(candidate_metadata, sort_keys=True))}),
            "candidate_prior": _array_digest({"metadata": np.asarray(json.dumps(prior_metadata, sort_keys=True))}),
            "maplet": _array_digest({"metadata": np.asarray(json.dumps(maplet_metadata, sort_keys=True))}),
            "geometry": _array_digest({"metadata": np.asarray(json.dumps(geometry_metadata, sort_keys=True))}),
            "landmark_bank": _array_digest({"metadata": np.asarray(json.dumps(bank_metadata, sort_keys=True))}),
            "candidate_support_view": (
                None
                if support_view_metadata is None
                else _array_digest(
                    {"metadata": np.asarray(json.dumps(support_view_metadata, sort_keys=True))}
                )
            ),
        },
        "inputs": _input_manifest(paths),
        "runtime": {
            "device": str(device),
            "hypothesis_batch_size": int(args.hypothesis_batch_size),
            "hypothesis_limit": int(args.hypothesis_limit),
            "all_frozen_hypotheses": bool(
                int(args.hypothesis_limit) == 0 and not diagnostic_hypothesis_indices
            ),
            "diagnostic_hypothesis_selection": bool(diagnostic_hypothesis_indices),
            "diagnostic_hypothesis_indices": [
                int(value) for value in diagnostic_hypothesis_indices
            ],
            "diagnostic_group_terms_dumped": diagnostic_dump_group_terms,
            "elapsed_seconds": float(time.time() - started),
        },
        "implementation": {
            "script_path": str(Path(__file__)),
            "script_sha256": file_sha256_short(Path(__file__)),
            "lifted_mode_module_sha256": file_sha256_short(
                Path("feature_extract/vfm/localization/frozen_lifted_loftr_map_to_query.py")
            ),
        },
    }
    payload: dict[str, np.ndarray] = {
        "query_ids": np.asarray(exact["query_ids"]).astype(np.str_),
        "split_names": np.asarray(exact["split_names"]).astype(np.str_),
        "evaluation_labels": np.asarray(exact["evaluation_labels"]).astype(np.str_),
        "hypothesis_indices": np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray(
            exact["source_chosen_for_optional_pose"], dtype=bool
        ),
        "baseline_score_top1": np.asarray(exact["independent_score_top1"], dtype=bool),
        "baseline_selection_scores": np.asarray(
            exact["independent_selection_scores"], dtype=np.float64
        ),
        "profile_names": profile_names,
        "profile_log_likelihood_means": scored["log_likelihood_means"],
        "profile_log_likelihood_medians": scored["log_likelihood_medians"],
        "profile_log_likelihood_worst_quartile_means": scored[
            "log_likelihood_worst_quartile_means"
        ],
        "profile_spatial_median_of_means_2x2": scored[
            "spatial_median_of_means_2x2"
        ],
        "profile_projection_visible_fractions": np.asarray(
            scored["projection_visible_fractions"], dtype=np.float64
        ),
        "fit_query_xy": fit_xy,
        "candidate_group_track_indices": candidate_groups.candidate_track_indices,
        "candidate_group_reference_xy": candidate_groups.group_reference_xy,
        "candidate_group_active_mask": candidate_groups.active_group_mask,
        "candidate_group_identity_probabilities": np.asarray(
            candidate_groups.candidate_identity_probabilities, dtype=np.float32
        ),
        "candidate_group_null_probabilities": np.asarray(
            candidate_groups.null_probabilities, dtype=np.float32
        ),
        "canonical_track_ids": canonical_bank.track_ids,
        "canonical_xyz": canonical_bank.xyz,
        "active_xyz": active_bank.xyz,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if isinstance(canonical_bank, FrozenLiftedSupportViewModeBank):
        assert isinstance(candidate_groups, FrozenCandidateSupportViewGroupLayout)
        payload.update(
            {
                "candidate_group_support_slot_indices": (
                    candidate_groups.candidate_support_slot_indices
                ),
                "candidate_group_support_image_ids": (
                    candidate_groups.candidate_support_image_ids
                ),
                "candidate_group_support_view_probabilities": (
                    candidate_groups.support_view_probabilities
                ),
                "support_slot_track_indices": canonical_bank.support_slot_track_indices,
                "support_slot_image_ids": canonical_bank.support_slot_image_ids,
                "mode_offsets": canonical_bank.mode_offsets,
                "mode_query_xy": canonical_bank.mode_query_xy,
                "mode_weights": canonical_bank.mode_weights,
                "mode_confidence_sums": canonical_bank.mode_confidence_sums,
                "support_reliabilities": canonical_bank.support_reliabilities,
                "support_reference_xy": canonical_bank.support_reference_xy,
            }
        )
    else:
        assert isinstance(canonical_bank, FrozenLiftedTrackModeBank)
        payload.update(
            {
                "mode_offsets": canonical_bank.mode_offsets,
                "mode_query_xy": canonical_bank.mode_query_xy,
                "mode_weights": canonical_bank.mode_weights,
                "mode_support_image_ids": canonical_bank.mode_support_image_ids,
                "mode_support_view_counts": canonical_bank.mode_support_view_counts,
                "mode_confidence_sums": canonical_bank.mode_confidence_sums,
                "track_reliabilities": canonical_bank.track_reliabilities,
                "track_reference_xy": canonical_bank.track_reference_xy,
            }
        )
    if diagnostic_terms is not None:
        if evidence_layout != EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW:
            raise AssertionError("only candidate support-view layout can dump group terms")
        expected_rows = len(poses)
        expected_profiles = len(profile_names)
        if (
            diagnostic_terms["support_view_slot_log_ratios"].shape
            != (expected_rows, expected_profiles, len(canonical_bank.support_slot_track_indices))
            or diagnostic_terms["candidate_log_ratios"].shape
            != (expected_rows, expected_profiles, 128, FIXED_CANDIDATE_TOP_K)
            or diagnostic_terms["group_log_ratios"].shape
            != (expected_rows, expected_profiles, 128)
            or diagnostic_terms["projected_xy"].shape
            != (expected_rows, len(canonical_bank.track_ids), 2)
            or diagnostic_terms["projection_valid"].shape
            != (expected_rows, len(canonical_bank.track_ids))
        ):
            raise ValueError("candidate support-view diagnostic term shapes are invalid")
        payload.update(
            {
                "diagnostic_support_view_slot_log_ratios": diagnostic_terms[
                    "support_view_slot_log_ratios"
                ],
                "diagnostic_candidate_log_ratios": diagnostic_terms[
                    "candidate_log_ratios"
                ],
                "diagnostic_group_log_ratios": diagnostic_terms["group_log_ratios"],
                "diagnostic_projected_xy": diagnostic_terms["projected_xy"],
                "diagnostic_projection_valid": diagnostic_terms["projection_valid"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    summary = {
        "stage": "score_frozen_lifted_loftr_map_to_query_pose_evidence",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": metadata["split_name"],
        "row_count": int(len(poses)),
        "profile_names": profile_names.tolist(),
        "mode_bank": {
            **lifting_stats,
            "canonical_mode_layout_sha256": layout_digest,
            "mean_reliability": float(
                np.mean(
                    canonical_bank.support_reliabilities
                    if isinstance(canonical_bank, FrozenLiftedSupportViewModeBank)
                    else canonical_bank.track_reliabilities
                )
            ),
        },
        "protocol": {
            "target_free": True,
            "fixed_global_top_l": True,
            "fixed_maplet_union": True,
            "candidate_specific_support_view": (
                evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
            ),
            "pnp_fit_neighborhood_excluded_from_query_visual_evidence": True,
            "candidate_or_support_reselection_per_pose": False,
            "query_center_shortcut": False,
            "full_mapping_pair_cache": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "raw_score_promotion_allowed": False,
            "target_only_tail_attribution_subset": bool(diagnostic_hypothesis_indices),
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
