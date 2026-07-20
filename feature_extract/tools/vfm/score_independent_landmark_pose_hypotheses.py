"""Score frozen pose hypotheses with strictly held-out global map evidence.

The scorer is inference-only.  It reads image-to-camera ownership but never
loads or retains COLMAP qvec/tvec.  Pose targets are joined later by a separate
evaluation command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.local_maplet_matching import (
    build_disjoint_maplet_cluster_ids,
    load_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.independent_landmark_pose_likelihood import (
    INDEPENDENT_LANDMARK_POSE_LIKELIHOOD_VERSION,
    IndependentLandmarkPoseLikelihoodConfig,
    IndependentLandmarkPoseVerifier,
    IndependentVerificationPoints,
    LandmarkObservationViewIndex,
    load_landmark_prototype_view_index_npz,
)
from feature_extract.vfm.localization import (
    independent_landmark_pose_likelihood as likelihood_module,
)
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
    MixedVerificationPoints,
    load_mixed_verification_points,
)


SCORE_ARTIFACT_FORMAT = "independent_landmark_hypothesis_scores_v1"
VERIFICATION_POINT_SELECTOR_FORMAT = "identity_posterior_verification_selector_v1"
MIXED_VERIFICATION_POINT_SOURCE = "frozen_mixed_multiscale_full_global_topl_v1"
SELECTION_STATISTICS = (
    "mean",
    "median",
    "trimmed_mean_10",
    "worst_quartile_mean",
    "lcb95",
    "spatial_median_of_means_2x2",
    "unique_track_assignment_log_joint",
)
STATISTIC_SCORE_FIELDS = {
    "mean": "independent_log_likelihood_means",
    "median": "independent_log_likelihood_medians",
    "trimmed_mean_10": "independent_log_likelihood_trimmed_means_10",
    "worst_quartile_mean": "independent_log_likelihood_worst_quartile_means",
    "lcb95": "independent_log_likelihood_lcb95s",
    "spatial_median_of_means_2x2": (
        "independent_spatial_median_of_means_2x2"
    ),
    "unique_track_assignment_log_joint": (
        "unique_track_assignment_log_joints"
    ),
}
CANDIDATE_PRIOR_OVERLAY_FORMATS = frozenset(
    {
        "candidate_maplet_prior_overlay_v1",
        "independent_rgb_candidate_prior_overlay_v1",
        "candidate_image_context_prior_overlay_v1",
        "multiscale_candidate_probe_prior_overlay_v1",
        "multiscale_candidate_probe_prior_overlay_v2",
        "current_v3_multisource_candidate_probe_prior_overlay_v1",
        "frozen_fulltrack_per_view_candidate_probe_prior_overlay_v1",
        "frozen_fulltrack_summary_top4_candidate_probe_prior_overlay_v1",
    }
)
CANDIDATE_PRIOR_PROBABILITY_SEMANTICS = frozenset(
    {
        "candidate_identity_probability_plus_explicit_null_equals_one",
        "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one",
        "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one",
    }
)


# The grouped-hypothesis artifact also contains diagnostic relation tensors.
# Their edge axis is intentionally shard-local and can vary with the number of
# hypotheses generated for a query.  Pose scoring needs only this fixed,
# row-aligned subset, so loading the full tensor schema makes the documented
# comma-separated shard interface both memory-heavy and invalid to merge.
_HYPOTHESIS_SCORING_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "shortlisted_for_verification",
    "chosen_for_optional_pose",
    "preliminary_log_likelihood_means",
    "poses_w2c",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis_artifacts",
        required=True,
        help="comma-separated grouped inference-only hypothesis NPZ shards",
    )
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    candidate_source = parser.add_mutually_exclusive_group(required=True)
    candidate_source.add_argument(
        "--fixed_candidate_prior_overlay",
        help=(
            "Target-free, proposal-aligned candidate probabilities plus explicit "
            "null mass. Strict absolute scoring refuses an implicit denominator."
        ),
    )
    candidate_source.add_argument(
        "--mixed_verification_points_artifact",
        help=(
            "Frozen target-free mixed verifier points with per-point full-bank "
            "top-L candidates and explicit null mass. This diagnostic mode is "
            "incompatible with selector, RGB-spatial, and top-K overlay options."
        ),
    )
    parser.add_argument(
        "--fixed_candidate_top_k",
        type=int,
        default=None,
        help=(
            "optional target-free ablation: retain the top-K frozen overlay "
            "probabilities per query point and transfer every removed candidate "
            "mass to the explicit null state without re-normalization"
        ),
    )
    parser.add_argument(
        "--fixed_candidate_prior_source",
        choices=(
            "learned_probability",
            "prototype_similarity_with_learned_null",
        ),
        default="learned_probability",
        help=(
            "Pose-independent identity prior frozen before any hypothesis is "
            "scored. Both supported modes preserve the overlay null mass."
        ),
    )
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument(
        "--independent_verification_landmark_bank",
        default=None,
        help=(
            "optional projection-compatible alternate representation used only "
            "for held-out scoring"
        ),
    )
    parser.add_argument("--support_geometry_index", default=None)
    parser.add_argument("--prototype_view_geometry", default=None)
    parser.add_argument(
        "--candidate_spatial_likelihood",
        default=None,
        help=(
            "optional comma-separated target-free candidate_spatial_likelihood_v7 "
            "artifacts aligned by source query row and physical track"
        ),
    )
    parser.add_argument(
        "--score_splits",
        default=None,
        help=(
            "optional comma-separated query splits to score; use this when a "
            "spatial artifact was materialized only for a subset of splits"
        ),
    )
    parser.add_argument(
        "--allow_unmaterialized_spatial_queries",
        action="store_true",
        help=(
            "diagnostic-only escape hatch: permit a spatial scorer to emit "
            "neutral unknown scores for a query with no RGB modes. Production "
            "spatial scoring fails closed instead."
        ),
    )
    parser.add_argument(
        "--emit_unique_track_assignment_diagnostic",
        action="store_true",
        help=(
            "write an exact null-or-unique-physical-track RGB assignment "
            "diagnostic for every frozen hypothesis; disabled by default"
        ),
    )
    parser.add_argument(
        "--unique_track_assignment_null_probability_floor",
        type=float,
        default=1e-12,
        help=(
            "numerical null floor used only by the optional unique-track "
            "assignment diagnostic"
        ),
    )
    parser.add_argument(
        "--unique_track_assignment_minimum_active_visual_mass",
        type=float,
        default=1e-6,
        help=(
            "minimum non-dustbin support-view mass for an optional assignment edge"
        ),
    )
    parser.add_argument("--maplet_support_index", default=None)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verification_point_count", type=int, default=192)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    parser.add_argument(
        "--verification_point_selection_artifact",
        default=None,
        help=(
            "optional target-free fixed verifier-row selector. It must be "
            "disjoint from the frozen hypothesis-fit rows and is fixed before "
            "any pose hypothesis is scored."
        ),
    )
    parser.add_argument("--nearest_landmarks", type=int, default=4)
    parser.add_argument("--maximum_reprojection_distance_px", type=float, default=8.0)
    parser.add_argument("--spatial_sigma_px", type=float, default=3.0)
    parser.add_argument(
        "--candidate_spatial_null_density",
        type=float,
        default=None,
        help=(
            "optional fixed RGB null density; the default is the uniform density "
            "of the frozen offset-grid support"
        ),
    )
    parser.add_argument(
        "--candidate_spatial_ineligible_likelihood_ratio",
        type=float,
        default=0.0,
        help=(
            "likelihood ratio for a candidate that fails positive-depth/image/"
            "view geometry; zero preserves the existing hard geometry gate"
        ),
    )
    parser.add_argument("--descriptor_temperature", type=float, default=0.04)
    parser.add_argument("--outlier_likelihood", type=float, default=0.01)
    parser.add_argument(
        "--selection_statistic",
        choices=SELECTION_STATISTICS,
        default="mean",
        help=(
            "target-free fixed-point aggregation used to rank frozen pose "
            "hypotheses; all statistics are written for later paired audits"
        ),
    )
    parser.add_argument("--minimum_observation_count", type=int, default=2)
    parser.add_argument("--maximum_view_angle_deg", type=float, default=90.0)
    parser.add_argument("--disable_view_gate", action="store_true")
    parser.add_argument("--purge_maplet_clusters", action="store_true")
    parser.add_argument("--kdtree_workers", type=int, default=1)
    parser.add_argument("--query_shard_count", type=int, default=1)
    parser.add_argument("--query_shard_index", type=int, default=0)
    parser.add_argument(
        "--hypothesis_scope",
        choices=("shortlisted", "preliminary_topk", "all"),
        default="shortlisted",
    )
    parser.add_argument(
        "--hypothesis_limit",
        type=int,
        default=128,
        help="per-query limit for preliminary_topk; zero means unlimited",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _strict_absolute_likelihood_config(
    args: argparse.Namespace,
) -> IndependentLandmarkPoseLikelihoodConfig:
    """Build the non-self-selecting production absolute-evidence protocol."""

    return IndependentLandmarkPoseLikelihoodConfig(
        nearest_landmarks=int(args.nearest_landmarks),
        maximum_reprojection_distance_px=float(
            args.maximum_reprojection_distance_px
        ),
        spatial_sigma_px=float(args.spatial_sigma_px),
        descriptor_temperature=float(args.descriptor_temperature),
        outlier_likelihood=float(args.outlier_likelihood),
        minimum_observation_count=int(args.minimum_observation_count),
        maximum_view_angle_deg=(
            None
            if bool(args.disable_view_gate)
            else float(args.maximum_view_angle_deg)
        ),
        kdtree_workers=int(args.kdtree_workers),
        candidate_mode="fixed_global_topl",
        fixed_candidate_prior_source=str(args.fixed_candidate_prior_source),
        candidate_spatial_null_density=args.candidate_spatial_null_density,
        candidate_spatial_ineligible_likelihood_ratio=float(
            args.candidate_spatial_ineligible_likelihood_ratio
        ),
    )


def _canonical_hash(payload: object) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()[:16]


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = (
            {}
            if "metadata_json" not in payload.files
            else json.loads(str(payload["metadata_json"].item()))
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must decode to an object")
    return arrays, metadata


def _load_npz_fields(
    path: Path,
    fields: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load an explicit inference-only field allowlist from an NPZ artifact."""

    requested = tuple(str(field) for field in fields)
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(requested).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: missing required fields: {missing}")
        arrays = {field: np.asarray(payload[field]).copy() for field in requested}
        metadata = (
            {}
            if "metadata_json" not in payload.files
            else json.loads(str(payload["metadata_json"].item()))
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must decode to an object")
    return arrays, metadata


def _landmark_bank_metadata(path: Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: landmark bank has no metadata_json")
        metadata = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: landmark metadata must decode to an object")
    return metadata


def _load_candidate_prior_overlay(
    path: Path,
    *,
    proposals_path: Path,
    proposals: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, metadata = _load_npz(path)
    required = {
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
    }
    if set(arrays) != required:
        raise ValueError("candidate prior overlay fields differ from contract")
    if metadata.get("format") not in CANDIDATE_PRIOR_OVERLAY_FORMATS:
        raise ValueError("unsupported candidate prior overlay format")
    if metadata.get("contains_ground_truth") is not False or metadata.get(
        "contains_target_errors"
    ) is not False:
        raise ValueError("candidate prior overlay is not target-free")
    if str(metadata.get("probability_semantics")) not in (
        CANDIDATE_PRIOR_PROBABILITY_SEMANTICS
    ):
        raise ValueError("candidate prior overlay probability semantics differ")
    if str(metadata.get("proposals_sha256")) != str(
        file_sha256_short(proposals_path)
    ):
        raise ValueError("candidate prior overlay references different proposals")
    proposal_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    overlay_tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    if not np.array_equal(overlay_tracks, proposal_tracks):
        raise ValueError("candidate prior overlay track identities differ")
    if probabilities.shape != proposal_tracks.shape or null.shape != (
        proposal_tracks.shape[0],
    ):
        raise ValueError("candidate prior overlay arrays have incompatible shapes")
    valid = proposal_tracks >= 0
    if np.any(~np.isfinite(probabilities)) or np.any(~np.isfinite(null)):
        raise ValueError("candidate prior overlay contains non-finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)) or np.any(
        (null < 0.0) | (null > 1.0)
    ):
        raise ValueError("candidate prior overlay values must be probabilities")
    if np.any(np.abs(probabilities[~valid]) > 1e-6):
        raise ValueError("invalid candidate columns carry posterior mass")
    mass = np.sum(np.where(valid, probabilities, 0.0), axis=1) + null
    if np.any(np.abs(mass - 1.0) > 1e-4):
        raise ValueError("candidate prior overlay probability mass differs from one")
    return {
        "candidate_track_ids": overlay_tracks,
        "candidate_probabilities": probabilities,
        "null_probabilities": null,
    }, metadata


def _mask_fixed_candidate_posterior_topk(
    *,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask a frozen candidate posterior without reallocating removed mass.

    This is deliberately an inference-time ablation, not a new candidate
    selector.  Ranking is fixed by the full posterior before a pose is scored;
    removed candidates have exactly zero mass and their original mass moves to
    the explicit null state.  That makes top-K comparisons meaningful even
    when a candidate spatial artifact was materialized for the full top-20.
    """

    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    if (
        tracks.ndim != 2
        or probabilities.shape != tracks.shape
        or null.shape != (tracks.shape[0],)
    ):
        raise ValueError("fixed candidate posterior arrays are not aligned")
    valid = tracks >= 0
    if np.any(~np.isfinite(probabilities)) or np.any(~np.isfinite(null)):
        raise ValueError("fixed candidate posterior must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)) or np.any(
        (null < 0.0) | (null > 1.0)
    ):
        raise ValueError("fixed candidate posterior must be in [0, 1]")
    if np.any(np.abs(probabilities[~valid]) > 1e-6):
        raise ValueError("invalid fixed candidate columns carry posterior mass")
    original_total = np.sum(np.where(valid, probabilities, 0.0), axis=1) + null
    if np.any(np.abs(original_total - 1.0) > 1e-4):
        raise ValueError("fixed candidate posterior mass is not conserved")
    if top_k is None:
        return probabilities.copy(), null.copy(), valid.copy()
    count = int(top_k)
    if count <= 0 or count > tracks.shape[1]:
        raise ValueError(
            "fixed_candidate_top_k must be in [1, candidate_column_count]"
        )

    # Stable ordering makes exact ties deterministic by original candidate
    # column, which is part of the frozen proposal layout.
    rank_scores = np.where(valid, probabilities, -np.inf)
    order = np.argsort(-rank_scores, axis=1, kind="mergesort")
    retained = np.zeros_like(valid)
    retained[np.arange(len(tracks))[:, None], order[:, :count]] = True
    retained &= valid
    effective_probabilities = np.where(retained, probabilities, 0.0).astype(
        np.float32, copy=False
    )
    removed_mass = np.sum(
        np.where(valid & ~retained, probabilities, 0.0), axis=1, dtype=np.float64
    )
    effective_null = (null.astype(np.float64) + removed_mass).astype(
        np.float32, copy=False
    )
    effective_total = (
        np.sum(np.where(valid, effective_probabilities, 0.0), axis=1)
        + effective_null
    )
    if np.any(np.abs(effective_total - 1.0) > 2e-5):
        raise RuntimeError("top-K candidate mask did not conserve posterior mass")
    return effective_probabilities, effective_null, retained


def _load_candidate_spatial_mode_index(
    paths: Sequence[Path],
) -> tuple[
    dict[tuple[int, int], list[tuple[int, float, float, float, float, float, int, int]]],
    np.ndarray,
    list[np.ndarray],
    list[dict[str, object]],
]:
    index: dict[
        tuple[int, int],
        list[tuple[int, float, float, float, float, float, int, int]],
    ] = {}
    offsets: np.ndarray | None = None
    log_probability_arrays: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    required = {
        "source_query_rows",
        "candidate_track_ids",
        "support_view_ranks",
        "support_view_probabilities",
        "candidate_prior_probabilities",
        "center_xy",
        "offsets_xy",
        "local_log_probabilities",
        "dustbin_probabilities",
        "metadata_json",
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"{path}: candidate spatial artifact lacks {missing}")
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata.get("format") != "candidate_spatial_likelihood_v7":
                raise ValueError(f"{path}: unsupported candidate spatial format")
            if metadata.get("ground_truth_loaded_by_inference_process") is not False:
                raise ValueError(f"{path}: candidate spatial inference loaded targets")
            if metadata.get("pose_or_ground_truth_used_for_inference") is not False:
                raise ValueError(f"{path}: candidate spatial modes are not target-free")
            local_offsets = np.asarray(payload["offsets_xy"], dtype=np.float32)
            if offsets is None:
                offsets = local_offsets.copy()
            elif not np.array_equal(offsets, local_offsets):
                raise ValueError("candidate spatial shards use different offset grids")
            rows = np.asarray(payload["source_query_rows"], dtype=np.int64)
            tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
            ranks = np.asarray(payload["support_view_ranks"], dtype=np.int64)
            view_probability = np.asarray(
                payload["support_view_probabilities"], dtype=np.float32
            )
            candidate_prior = np.asarray(
                payload["candidate_prior_probabilities"], dtype=np.float32
            )
            centers = np.asarray(payload["center_xy"], dtype=np.float32)
            local_log = np.asarray(payload["local_log_probabilities"], dtype=np.float16)
            dustbin = np.asarray(payload["dustbin_probabilities"], dtype=np.float32)
            count = len(rows)
            if not (
                tracks.shape == ranks.shape == view_probability.shape
                == candidate_prior.shape == dustbin.shape == (count,)
                and centers.shape == (count, 2)
                and local_log.shape == (count, len(local_offsets))
            ):
                raise ValueError(f"{path}: candidate spatial rows are not aligned")
            artifact_index = len(log_probability_arrays)
            log_probability_arrays.append(local_log)
            for row_index in range(count):
                key = (int(rows[row_index]), int(tracks[row_index]))
                index.setdefault(key, []).append(
                    (
                        int(ranks[row_index]),
                        float(view_probability[row_index]),
                        float(candidate_prior[row_index]),
                        float(centers[row_index, 0]),
                        float(centers[row_index, 1]),
                        float(dustbin[row_index]),
                        artifact_index,
                        row_index,
                    )
                )
            metadata_rows.append(metadata)
    if offsets is None:
        raise ValueError("at least one candidate spatial artifact is required")
    for key, views in index.items():
        ranks = [int(view[0]) for view in views]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"candidate spatial views repeat for {key}")
        views.sort(key=lambda view: int(view[0]))
    return index, offsets, log_probability_arrays, metadata_rows


def _candidate_spatial_modes_for_points(
    *,
    source_rows: np.ndarray,
    xy: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    mode_index: Mapping[
        tuple[int, int],
        Sequence[tuple[int, float, float, float, float, float, int, int]],
    ],
    offsets_xy: np.ndarray,
    log_probability_arrays: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    priors = np.asarray(candidate_probabilities, dtype=np.float32)
    centers = np.asarray(xy, dtype=np.float32)
    maximum_views = max(
        (int(view[0]) + 1 for views in mode_index.values() for view in views),
        default=0,
    )
    if maximum_views <= 0:
        raise ValueError("candidate spatial artifact contains no support views")
    shape = (*tracks.shape, maximum_views)
    local_log = np.zeros((*shape, len(offsets_xy)), dtype=np.float16)
    dustbin = np.ones(shape, dtype=np.float32)
    view_probability = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    for point_index, source_row in enumerate(rows.tolist()):
        for candidate_column, track_id in enumerate(tracks[point_index].tolist()):
            if int(track_id) < 0:
                continue
            views = mode_index.get((int(source_row), int(track_id)), ())
            for view in views:
                rank, view_prob, candidate_prior, center_x, center_y, dust, artifact, row = view
                if not np.allclose(
                    np.asarray([center_x, center_y], dtype=np.float32),
                    centers[point_index],
                    rtol=0.0,
                    atol=1e-4,
                ):
                    raise ValueError("candidate spatial center differs from query point")
                if not np.isclose(
                    float(candidate_prior),
                    float(priors[point_index, candidate_column]),
                    rtol=0.0,
                    atol=2e-5,
                ):
                    raise ValueError(
                        "candidate spatial identity prior differs from fixed overlay"
                    )
                local_log[point_index, candidate_column, rank] = np.asarray(
                    log_probability_arrays[int(artifact)][int(row)], dtype=np.float16
                )
                dustbin[point_index, candidate_column, rank] = float(dust)
                view_probability[point_index, candidate_column, rank] = float(view_prob)
                valid[point_index, candidate_column, rank] = True
    return {
        "candidate_spatial_offsets_xy": np.asarray(offsets_xy, dtype=np.float32),
        "candidate_spatial_log_probabilities": local_log,
        "candidate_spatial_dustbin_probabilities": dustbin,
        "candidate_support_view_probabilities": view_probability,
        "candidate_spatial_valid_mask": valid,
    }


def _validate_alternate_verification_bank(
    source_metadata: Mapping[str, object],
    verification_metadata: Mapping[str, object],
) -> None:
    source_manifest = source_metadata.get("descriptor_space_manifest")
    verification_manifest = verification_metadata.get("descriptor_space_manifest")
    if not isinstance(source_manifest, Mapping) or not isinstance(
        verification_manifest, Mapping
    ):
        raise ValueError("both landmark banks require descriptor-space manifests")
    required_equal = (
        "projection_space_id",
        "checkpoint_sha256",
        "feature_key",
        "projection_source",
        "output_branch",
        "image_manifest_hash",
        "sfm_track_hash",
        "descriptor_dimension",
        "observation_selection",
    )
    mismatches = {
        key: {
            "source": source_manifest.get(key),
            "verification": verification_manifest.get(key),
        }
        for key in required_equal
        if source_manifest.get(key) != verification_manifest.get(key)
    }
    if mismatches:
        raise ValueError(
            "alternate verification bank is not projection/map compatible: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _merge_hypothesis_artifacts(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], dict[str, object], str]:
    # `load_inference_artifact_fields` still validates the complete no-target
    # artifact contract, while avoiding variable-width diagnostic arrays that
    # are irrelevant to the frozen pose scorer.
    loaded = [
        (
            {
                key: arrays[key]
                for key in _HYPOTHESIS_SCORING_FIELDS
            },
            metadata,
        )
        for arrays, metadata in (
            load_inference_artifact_fields(path, _HYPOTHESIS_SCORING_FIELDS)
            for path in paths
        )
    ]
    compatibility = [
        {
            "inputs": metadata.get("inputs"),
            "grouped_config": metadata.get("grouped_config"),
        }
        for _arrays, metadata in loaded
    ]
    fingerprints = [_canonical_hash(value) for value in compatibility]
    if len(set(fingerprints)) != 1:
        raise ValueError(f"hypothesis shards are incompatible: {fingerprints}")
    keys = set(_HYPOTHESIS_SCORING_FIELDS)
    if any(set(arrays) != keys for arrays, _ in loaded):
        raise ValueError("hypothesis shards have different schemas")
    merged = {
        key: np.concatenate([arrays[key] for arrays, _ in loaded], axis=0)
        for key in sorted(keys)
    }
    query_ids = merged["query_ids"].astype(str)
    labels = merged["evaluation_labels"].astype(str)
    indices = merged["hypothesis_indices"].astype(np.int64)
    row_keys = list(zip(query_ids.tolist(), labels.tolist(), indices.tolist()))
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("merged hypothesis artifacts contain duplicate rows")
    return merged, loaded[0][1], fingerprints[0]


def _validate_declared_input(
    metadata: Mapping[str, object],
    *,
    key: str,
    path: Path,
) -> None:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("hypothesis metadata has no input manifest")
    expected_hash = inputs.get(f"{key}_sha256")
    if expected_hash is None:
        raise ValueError(f"hypothesis metadata does not declare {key}_sha256")
    actual_hash = file_sha256_short(path)
    if str(expected_hash) != str(actual_hash):
        raise ValueError(
            f"{key} differs from hypothesis source: expected {expected_hash}, "
            f"found {actual_hash}"
        )


def _maplet_purged_tracks(
    excluded_tracks: np.ndarray,
    *,
    maplet_track_ids: np.ndarray,
    maplet_cluster_ids: np.ndarray,
) -> np.ndarray:
    tracks = np.asarray(maplet_track_ids, dtype=np.int64).reshape(-1)
    clusters = np.asarray(maplet_cluster_ids, dtype=np.int64).reshape(-1)
    if tracks.shape != clusters.shape:
        raise ValueError("maplet tracks and clusters are not aligned")
    order = np.argsort(tracks, kind="mergesort")
    sorted_tracks = tracks[order]
    positions = np.searchsorted(sorted_tracks, excluded_tracks)
    found = positions < len(sorted_tracks)
    found_indices = np.flatnonzero(found)
    if found_indices.size:
        found[found_indices] &= (
            sorted_tracks[positions[found_indices]]
            == excluded_tracks[found_indices]
        )
    excluded_clusters = np.unique(clusters[order[positions[found]]])
    if excluded_clusters.size == 0:
        return np.unique(excluded_tracks)
    return np.unique(
        np.concatenate(
            [excluded_tracks, tracks[np.isin(clusters, excluded_clusters)]]
        )
    )


def _query_detector_rows(
    query_id: str,
    *,
    detector: Mapping[str, np.ndarray],
) -> np.ndarray:
    image_ids = detector["image_ids"].astype(str)
    matches = np.flatnonzero(image_ids == str(query_id))
    if len(matches) != 1:
        raise ValueError(f"detector cache does not uniquely contain {query_id}")
    image_row = int(matches[0])
    offsets = detector["offsets"].astype(np.int64)
    return np.arange(int(offsets[image_row]), int(offsets[image_row + 1]))


def _verification_points_for_query(
    query_id: str,
    *,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
    point_count: int,
    detector_log_merit_weight: float,
    descriptor_key: str = "global_descriptors",
    candidate_prior_overlay: Mapping[str, np.ndarray] | None = None,
    candidate_spatial_mode_index: Mapping[
        tuple[int, int],
        Sequence[tuple[int, float, float, float, float, float, int, int]],
    ] | None = None,
    candidate_spatial_offsets_xy: np.ndarray | None = None,
    candidate_spatial_log_probability_arrays: Sequence[np.ndarray] | None = None,
    candidate_spatial_source_probabilities: np.ndarray | None = None,
    preselected_verification_rows: np.ndarray | None = None,
) -> tuple[IndependentVerificationPoints, np.ndarray, dict[str, int]]:
    all_rows = _query_detector_rows(query_id, detector=detector)
    proposal_query_ids = proposals["query_ids"].astype(str)
    if proposal_query_ids.shape[0] != detector["xy"].shape[0]:
        raise ValueError("proposal and detector row counts differ")
    if not np.all(proposal_query_ids[all_rows] == str(query_id)):
        raise ValueError("proposal query row ownership differs from detector cache")
    fit_rows = selected_rows[proposal_query_ids[selected_rows] == str(query_id)]
    if len(np.unique(fit_rows)) != len(fit_rows):
        raise ValueError(f"{query_id}: candidate artifact repeats selected rows")
    unused_rows = np.setdiff1d(all_rows, fit_rows, assume_unique=True)
    if np.intersect1d(unused_rows, fit_rows).size:
        raise RuntimeError("fit and verification query rows overlap")
    if preselected_verification_rows is None:
        coarse_reference = np.max(
            np.asarray(proposals["coarse_scores"][unused_rows], dtype=np.float64),
            axis=1,
        )
        detector_scores = np.asarray(
            detector["detector_scores"][unused_rows], dtype=np.float64
        )
        merit = coarse_reference + float(detector_log_merit_weight) * np.log(
            np.maximum(detector_scores, 1e-12)
        )
        keep_count = min(int(point_count), int(len(unused_rows)))
        kept = unused_rows[
            np.argsort(-merit, kind="mergesort")[:keep_count]
        ]
    else:
        kept = np.asarray(preselected_verification_rows, dtype=np.int64).reshape(-1)
        if (
            kept.size == 0
            or len(np.unique(kept)) != len(kept)
            or np.any(~np.isin(kept, unused_rows))
        ):
            raise ValueError(
                f"{query_id}: fixed verification selector is not a unique held-out row set"
            )
        if kept.size > int(point_count):
            raise ValueError(
                f"{query_id}: fixed verification selector exceeds point budget"
            )
        coarse_reference = np.max(
            np.asarray(proposals["coarse_scores"][unused_rows], dtype=np.float64),
            axis=1,
        )
    reference_by_row = {
        int(row): float(score) for row, score in zip(unused_rows, coarse_reference)
    }
    if str(descriptor_key) not in detector:
        raise ValueError(f"detector cache has no descriptor field {descriptor_key}")
    if candidate_prior_overlay is None:
        candidate_scores = proposals["coarse_scores"][kept]
        candidate_null = None
    else:
        overlay_tracks = np.asarray(
            candidate_prior_overlay["candidate_track_ids"], dtype=np.int64
        )
        if not np.array_equal(
            overlay_tracks[kept], proposals["candidate_track_ids"][kept]
        ):
            raise ValueError("candidate prior overlay rows differ from proposals")
        candidate_scores = candidate_prior_overlay["candidate_probabilities"][kept]
        candidate_null = candidate_prior_overlay["null_probabilities"][kept]
    spatial_kwargs: dict[str, np.ndarray] = {}
    if candidate_spatial_mode_index is not None:
        if (
            candidate_prior_overlay is None
            or candidate_spatial_offsets_xy is None
            or candidate_spatial_log_probability_arrays is None
        ):
            raise ValueError("candidate spatial modes require the fixed prior overlay")
        source_probabilities = (
            candidate_scores
            if candidate_spatial_source_probabilities is None
            else np.asarray(candidate_spatial_source_probabilities, dtype=np.float32)[
                kept
            ]
        )
        if source_probabilities.shape != candidate_scores.shape:
            raise ValueError(
                "candidate spatial source probabilities are not point-aligned"
            )
        spatial_kwargs = _candidate_spatial_modes_for_points(
            source_rows=kept,
            xy=detector["xy"][kept],
            candidate_track_ids=proposals["candidate_track_ids"][kept],
            candidate_probabilities=source_probabilities,
            mode_index=candidate_spatial_mode_index,
            offsets_xy=candidate_spatial_offsets_xy,
            log_probability_arrays=candidate_spatial_log_probability_arrays,
        )
    points = IndependentVerificationPoints(
        xy=detector["xy"][kept],
        descriptors=detector[str(descriptor_key)][kept],
        descriptor_reference_scores=np.asarray(
            [reference_by_row[int(row)] for row in kept], dtype=np.float64
        ),
        source_row_indices=kept,
        candidate_track_ids=proposals["candidate_track_ids"][kept],
        candidate_descriptor_scores=candidate_scores,
        candidate_null_probabilities=candidate_null,
        **spatial_kwargs,
    )
    candidate_tracks = np.asarray(
        proposals["candidate_track_ids"][fit_rows], dtype=np.int64
    ).reshape(-1)
    candidate_tracks = np.unique(candidate_tracks[candidate_tracks >= 0])
    return points, candidate_tracks, {
        "fit_query_point_count": int(len(fit_rows)),
        "available_unused_query_point_count": int(len(unused_rows)),
        "selected_verification_point_count": int(len(kept)),
    }


def _load_mixed_verification_points_for_scoring(
    path: Path,
    *,
    detector_path: Path,
    candidate_path: Path,
    source_bank_path: Path,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
    landmark_track_ids: np.ndarray,
    bank_metadata: Mapping[str, object],
) -> MixedVerificationPoints:
    """Load a frozen full-bank verifier and prove it matches this pose run.

    Mixed verifier points deliberately have their own immutable global top-L
    candidates.  They must therefore not silently inherit a detector-row
    overlay or a candidate-specific RGB likelihood generated for a different
    proposal universe.  The original fit rows remain useful only for the
    cross-fit track exclusion below.
    """

    artifact_path = Path(path)
    points = load_mixed_verification_points(artifact_path)
    metadata = dict(points.metadata)
    if metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT:
        raise ValueError("unsupported mixed verification artifact format")
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
    ):
        raise ValueError("mixed verification artifact violates the target-free contract")
    candidate_fit_hash = metadata.get("candidate_fit_artifact_sha256")
    if candidate_fit_hash is None:
        # Older point artifacts predate the explicit fit-artifact field and
        # can only be used when the candidate-evidence artifact itself was
        # the hypothesis fit artifact.
        candidate_fit_hash = metadata.get("candidate_evidence_sha256")
    expected_hashes = {
        "candidate_fit_artifact_sha256": file_sha256_short(candidate_path),
        "detector_query_cache_sha256": file_sha256_short(detector_path),
        "projected_landmark_bank_sha256": file_sha256_short(source_bank_path),
    }
    observed_hashes = {
        "candidate_fit_artifact_sha256": candidate_fit_hash,
        "detector_query_cache_sha256": metadata.get("detector_query_cache_sha256"),
        "projected_landmark_bank_sha256": metadata.get(
            "projected_landmark_bank_sha256"
        ),
    }
    for key, expected in expected_hashes.items():
        if str(observed_hashes.get(key, "")) != str(expected):
            raise ValueError(f"mixed verification artifact lineage mismatch for {key}")
    if (
        metadata.get("candidate_reselection") is not False
        or metadata.get("candidate_set")
        != "fixed_full_global_faiss_top_l_unique_tracks"
        or metadata.get("global_landmark_ann_scope")
        != "full_projected_landmark_bank_only"
    ):
        raise ValueError("mixed verification candidates are not fixed full-bank top-L")
    if int(metadata.get("candidate_top_k", -1)) != int(points.candidate_track_ids.shape[1]):
        raise ValueError("mixed verification candidate count provenance is inconsistent")
    if str(metadata.get("descriptor_space_id", "")) != str(
        bank_metadata.get("descriptor_space_id", "")
    ):
        raise ValueError("mixed verification and landmark descriptor spaces differ")

    detector_ids = np.asarray(detector["image_ids"]).astype(str).reshape(-1)
    detector_offsets = np.asarray(detector["offsets"], dtype=np.int64).reshape(-1)
    proposal_queries = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    if (
        detector_offsets.shape != (len(detector_ids) + 1,)
        or detector_offsets[0] != 0
        or detector_offsets[-1] != len(proposal_queries)
        or np.any(detector_offsets[1:] < detector_offsets[:-1])
        or set(points.query_ids.tolist()) - set(detector_ids.tolist())
    ):
        raise ValueError("mixed verification query ownership differs from detector cache")
    detector_rows = np.asarray(points.source_detector_rows, dtype=np.int64)
    alike = np.asarray(points.point_sources).astype(str) == POINT_SOURCE_ALIKE
    if np.any((detector_rows < -1) | (detector_rows >= len(proposal_queries))):
        raise ValueError("mixed verification source detector rows are invalid")
    if np.any((detector_rows >= 0) != alike):
        raise ValueError("only ALIKE mixed verifier points may reference detector rows")
    if np.any(detector_rows[alike] < 0):
        raise ValueError("ALIKE mixed verifier points must retain detector-row lineage")
    if np.any(detector_rows[~alike] != -1):
        raise ValueError("dense mixed verifier points must not claim detector-row lineage")
    referenced = detector_rows[alike]
    if (
        np.any(proposal_queries[referenced] != points.query_ids[alike])
        or np.any(np.isin(referenced, np.asarray(selected_rows, dtype=np.int64)))
    ):
        raise ValueError("mixed verifier reuses candidate-fit detector rows")

    bank_tracks = np.asarray(landmark_track_ids, dtype=np.int64).reshape(-1)
    bank_rows = np.asarray(points.candidate_bank_rows, dtype=np.int64)
    candidate_tracks = np.asarray(points.candidate_track_ids, dtype=np.int64)
    valid = candidate_tracks >= 0
    if (
        np.any(bank_rows[valid] < 0)
        or np.any(bank_rows[valid] >= len(bank_tracks))
        or not np.array_equal(bank_tracks[bank_rows[valid]], candidate_tracks[valid])
    ):
        raise ValueError("mixed verifier candidate rows differ from landmark bank tracks")
    return points


def _mixed_verification_points_for_query(
    query_id: str,
    *,
    mixed_points: MixedVerificationPoints,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
) -> tuple[IndependentVerificationPoints, np.ndarray, dict[str, int]]:
    """Materialize one query's immutable multi-source verifier points."""

    point_rows = mixed_points.rows_for_query(str(query_id))
    if point_rows.size == 0:
        raise ValueError(f"{query_id}: mixed verification artifact has no points")
    source_ids = np.asarray(mixed_points.source_point_ids, dtype=np.int64)[point_rows]
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError(f"{query_id}: mixed verifier source IDs are not unique")
    candidate_scores = np.asarray(
        mixed_points.candidate_coarse_similarities, dtype=np.float32
    )[point_rows]
    candidate_tracks = np.asarray(mixed_points.candidate_track_ids, dtype=np.int64)[
        point_rows
    ]
    valid = candidate_tracks >= 0
    reference = np.max(
        np.where(valid, candidate_scores, -np.inf), axis=1
    ).astype(np.float64, copy=False)
    if np.any(~np.isfinite(reference)):
        raise ValueError(f"{query_id}: mixed verifier has a point without a candidate")
    points = IndependentVerificationPoints(
        xy=np.asarray(mixed_points.xy, dtype=np.float32)[point_rows],
        descriptors=np.asarray(mixed_points.descriptors, dtype=np.float32)[point_rows],
        descriptor_reference_scores=reference,
        source_row_indices=source_ids,
        candidate_track_ids=candidate_tracks,
        candidate_descriptor_scores=np.asarray(
            mixed_points.candidate_prior_probabilities, dtype=np.float32
        )[point_rows],
        candidate_null_probabilities=np.asarray(
            mixed_points.null_probabilities, dtype=np.float32
        )[point_rows],
    )
    fit_rows = np.asarray(selected_rows, dtype=np.int64)
    proposal_query_ids = np.asarray(proposals["query_ids"]).astype(str)
    fit_rows = fit_rows[proposal_query_ids[fit_rows] == str(query_id)]
    fit_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)[fit_rows]
    excluded_tracks = np.unique(fit_tracks[fit_tracks >= 0])
    all_detector_rows = _query_detector_rows(query_id, detector=detector)
    sources = np.asarray(mixed_points.point_sources).astype(str)[point_rows]
    return points, excluded_tracks, {
        "fit_query_point_count": int(len(fit_rows)),
        "available_unused_query_point_count": int(
            len(np.setdiff1d(all_detector_rows, fit_rows, assume_unique=True))
        ),
        "selected_verification_point_count": int(len(points)),
        "mixed_alike_verification_point_count": int(
            np.count_nonzero(sources == POINT_SOURCE_ALIKE)
        ),
        "mixed_radio_intermediate_verification_point_count": int(
            np.count_nonzero(sources == POINT_SOURCE_RADIO_INTERMEDIATE)
        ),
        "mixed_radio_final_verification_point_count": int(
            np.count_nonzero(sources == POINT_SOURCE_RADIO_FINAL)
        ),
    }


def _spatial_materialization_audit(
    points: IndependentVerificationPoints,
) -> dict[str, int]:
    """Count frozen RGB support before ranking any pose hypothesis.

    A score with no materialized RGB modes is a neutral unknown likelihood.
    It cannot rank poses safely on its own, even though it is useful for an
    internal likelihood-invariance check.  Keep this audit at query scope so
    callers can fail closed instead of accidentally promoting a tie-broken
    unknown score.
    """

    valid = points.candidate_spatial_valid_mask
    if valid is None:
        return {
            "materialized_verification_point_count": 0,
            "materialized_candidate_view_count": 0,
        }
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 3 or mask.shape[0] != len(points):
        raise ValueError("candidate spatial valid mask is not point-aligned")
    return {
        "materialized_verification_point_count": int(
            np.count_nonzero(np.any(mask, axis=(1, 2)))
        ),
        "materialized_candidate_view_count": int(np.count_nonzero(mask)),
    }


def _load_verification_point_selection_artifact(
    path: Path,
    *,
    detector: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    selected_rows: np.ndarray,
    detector_path: Path,
    proposals_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load a fixed target-free verifier selector with strict row lineage."""

    arrays, metadata = _load_npz_fields(
        path,
        ("query_ids", "offsets", "source_row_indices", "identity_confidence"),
    )
    if metadata.get("format") != VERIFICATION_POINT_SELECTOR_FORMAT:
        raise ValueError("unsupported verification point selector format")
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
    ):
        raise ValueError("verification point selector violates the target-free contract")
    expected_hashes = {
        "detector_query_cache_sha256": file_sha256_short(detector_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
    }
    for key, expected in expected_hashes.items():
        if str(metadata.get(key, "")) != str(expected):
            raise ValueError(f"verification point selector lineage mismatch for {key}")
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64).reshape(-1)
    rows = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    confidence = np.asarray(arrays["identity_confidence"], dtype=np.float32).reshape(-1)
    detector_ids = np.asarray(detector["image_ids"]).astype(str).reshape(-1)
    detector_offsets = np.asarray(detector["offsets"], dtype=np.int64).reshape(-1)
    proposal_queries = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    if (
        query_ids.shape != detector_ids.shape
        or not np.array_equal(query_ids, detector_ids)
        or offsets.shape != (len(query_ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(rows)
        or np.any(offsets[1:] < offsets[:-1])
        or confidence.shape != rows.shape
        or np.any(~np.isfinite(confidence))
        or np.any(confidence < 0.0)
        or len(np.unique(rows)) != len(rows)
        or np.any((rows < 0) | (rows >= len(proposal_queries)))
    ):
        raise ValueError("verification point selector arrays are invalid")
    selection_metadata = metadata.get("selection")
    if not isinstance(selection_metadata, Mapping):
        raise ValueError("verification point selector lacks selection metadata")
    point_count = int(selection_metadata.get("point_count", -1))
    if point_count <= 0 or np.any(np.diff(offsets) != point_count):
        raise ValueError("verification point selector has an inconsistent point budget")
    fit_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if len(np.unique(fit_rows)) != len(fit_rows):
        raise ValueError("candidate artifact repeats fit rows")
    output: dict[str, np.ndarray] = {}
    for image_index, query_id in enumerate(query_ids.tolist()):
        begin, end = int(offsets[image_index]), int(offsets[image_index + 1])
        query_rows = rows[begin:end]
        detector_begin = int(detector_offsets[image_index])
        detector_end = int(detector_offsets[image_index + 1])
        if (
            np.any(query_rows < detector_begin)
            or np.any(query_rows >= detector_end)
            or not np.all(proposal_queries[query_rows] == str(query_id))
            or np.any(np.isin(query_rows, fit_rows))
        ):
            raise ValueError(
                f"{query_id}: selector rows are not detector-owned held-out rows"
            )
        output[str(query_id)] = query_rows
    return output, metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mixed_verification_path = (
        None
        if args.mixed_verification_points_artifact is None
        else Path(args.mixed_verification_points_artifact)
    )
    mixed_verification_mode = mixed_verification_path is not None
    implementation_manifest = {
        "scorer_source_sha256_at_process_start": file_sha256_short(Path(__file__)),
        "likelihood_source_sha256_at_process_start": file_sha256_short(
            Path(likelihood_module.__file__)
        ),
    }
    if int(args.verification_point_count) <= 0:
        raise ValueError("verification_point_count must be positive")
    if int(args.query_shard_count) <= 0 or not (
        0 <= int(args.query_shard_index) < int(args.query_shard_count)
    ):
        raise ValueError("invalid query shard")
    if int(args.hypothesis_limit) < 0:
        raise ValueError("hypothesis_limit must be non-negative")
    if mixed_verification_mode:
        incompatible = {
            "candidate_spatial_likelihood": args.candidate_spatial_likelihood,
            "verification_point_selection_artifact": (
                args.verification_point_selection_artifact
            ),
            "fixed_candidate_top_k": args.fixed_candidate_top_k,
            "allow_unmaterialized_spatial_queries": (
                bool(args.allow_unmaterialized_spatial_queries)
            ),
            "emit_unique_track_assignment_diagnostic": (
                bool(args.emit_unique_track_assignment_diagnostic)
            ),
        }
        supplied = [name for name, value in incompatible.items() if value not in (None, False)]
        if supplied:
            raise ValueError(
                "mixed verification mode is incompatible with " + ", ".join(supplied)
            )
        if str(args.fixed_candidate_prior_source) != "learned_probability":
            raise ValueError(
                "mixed verification uses its frozen per-point probability mass; "
                "fixed_candidate_prior_source must be learned_probability"
            )
    assignment_diagnostic_enabled = bool(
        args.emit_unique_track_assignment_diagnostic
    )
    if (
        str(args.selection_statistic) == "unique_track_assignment_log_joint"
        and not assignment_diagnostic_enabled
    ):
        raise ValueError(
            "unique-track assignment selection requires "
            "--emit_unique_track_assignment_diagnostic"
        )
    null_floor = float(args.unique_track_assignment_null_probability_floor)
    active_mass_floor = float(
        args.unique_track_assignment_minimum_active_visual_mass
    )
    if not np.isfinite(null_floor) or not 0.0 < null_floor <= 1.0:
        raise ValueError(
            "unique_track_assignment_null_probability_floor must be in (0, 1]"
        )
    if not np.isfinite(active_mass_floor) or active_mass_floor < 0.0:
        raise ValueError(
            "unique_track_assignment_minimum_active_visual_mass must be non-negative"
        )
    requested_splits = None
    if args.score_splits is not None:
        requested_splits = {
            value.strip()
            for value in str(args.score_splits).split(",")
            if value.strip()
        }
        if not requested_splits:
            raise ValueError("score_splits must contain at least one split")
    if bool(args.support_geometry_index) == bool(args.prototype_view_geometry):
        raise ValueError(
            "provide exactly one of --support_geometry_index and "
            "--prototype_view_geometry"
        )
    paths = tuple(
        Path(value.strip())
        for value in str(args.hypothesis_artifacts).split(",")
        if value.strip()
    )
    if not paths:
        raise ValueError("at least one hypothesis artifact is required")
    output_dir = Path(args.output_dir)
    output_path = output_dir / "independent_landmark_hypothesis_scores_v1.npz"
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_path}")

    merged, hypothesis_metadata, compatibility_hash = (
        _merge_hypothesis_artifacts(paths)
    )
    proposal_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    source_bank_path = Path(args.projected_landmark_bank)
    bank_path = (
        source_bank_path
        if args.independent_verification_landmark_bank is None
        else Path(args.independent_verification_landmark_bank)
    )
    if mixed_verification_mode and bank_path != source_bank_path:
        raise ValueError(
            "mixed verification points are tied to their source landmark bank; "
            "alternate verification banks are unsupported"
        )
    _validate_declared_input(
        hypothesis_metadata, key="proposals", path=proposal_path
    )
    _validate_declared_input(
        hypothesis_metadata, key="candidate_artifact", path=candidate_path
    )
    _validate_declared_input(
        hypothesis_metadata,
        key="projected_landmark_bank",
        path=source_bank_path,
    )

    detector, detector_metadata = _load_npz_fields(
        Path(args.detector_query_cache),
        (
            "image_ids",
            "offsets",
            "xy",
            "global_descriptors",
            "detector_scores",
        ),
    )
    proposals, proposal_metadata = _load_npz_fields(
        proposal_path,
        ("query_ids", "coarse_scores", "candidate_track_ids"),
    )
    candidate, candidate_metadata = _load_npz_fields(
        candidate_path, ("selected_rows",)
    )
    prior_overlay_path: Path | None = None
    prior_overlay: dict[str, np.ndarray] | None = None
    prior_overlay_metadata: dict[str, object] = {}
    spatial_source_probabilities: np.ndarray | None = None
    fixed_candidate_topk_retained: np.ndarray | None = None
    fixed_candidate_topk_removed_mass: np.ndarray | None = None
    if not mixed_verification_mode:
        prior_overlay_path = Path(args.fixed_candidate_prior_overlay)
        prior_overlay, prior_overlay_metadata = _load_candidate_prior_overlay(
            prior_overlay_path,
            proposals_path=proposal_path,
            proposals=proposals,
        )
        spatial_source_probabilities = np.asarray(
            prior_overlay["candidate_probabilities"], dtype=np.float32
        ).copy()
        (
            effective_candidate_probabilities,
            effective_null_probabilities,
            fixed_candidate_topk_retained,
        ) = _mask_fixed_candidate_posterior_topk(
            candidate_track_ids=np.asarray(
                prior_overlay["candidate_track_ids"], dtype=np.int64
            ),
            candidate_probabilities=spatial_source_probabilities,
            null_probabilities=np.asarray(
                prior_overlay["null_probabilities"], dtype=np.float32
            ),
            top_k=args.fixed_candidate_top_k,
        )
        prior_overlay = {
            "candidate_track_ids": np.asarray(
                prior_overlay["candidate_track_ids"], dtype=np.int64
            ),
            "candidate_probabilities": effective_candidate_probabilities,
            "null_probabilities": effective_null_probabilities,
        }
        fixed_candidate_topk_removed_mass = np.sum(
            spatial_source_probabilities - effective_candidate_probabilities,
            axis=1,
            dtype=np.float64,
        )
    spatial_mode_index = None
    spatial_offsets_xy = None
    spatial_log_probability_arrays: list[np.ndarray] = []
    spatial_mode_metadata: list[dict[str, object]] = []
    spatial_mode_paths: tuple[Path, ...] = tuple()
    if args.candidate_spatial_likelihood is not None:
        spatial_mode_paths = tuple(
            Path(value.strip())
            for value in str(args.candidate_spatial_likelihood).split(",")
            if value.strip()
        )
        if not spatial_mode_paths:
            raise ValueError("candidate spatial likelihood path list is empty")
        (
            spatial_mode_index,
            spatial_offsets_xy,
            spatial_log_probability_arrays,
            spatial_mode_metadata,
        ) = _load_candidate_spatial_mode_index(spatial_mode_paths)
    if assignment_diagnostic_enabled and spatial_mode_index is None:
        raise ValueError(
            "unique-track assignment diagnostic requires --candidate_spatial_likelihood"
        )
    support_geometry = None
    support_geometry_metadata: dict[str, object] = {}
    if args.support_geometry_index is not None:
        support_geometry, support_geometry_metadata = _load_npz(
            Path(args.support_geometry_index)
        )
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    if np.any((selected_rows < 0) | (selected_rows >= len(proposals["query_ids"]))):
        raise ValueError("candidate artifact contains invalid selected rows")
    verification_selector_path = None
    verification_selector_rows: dict[str, np.ndarray] | None = None
    verification_selector_metadata: dict[str, object] = {}
    if args.verification_point_selection_artifact is not None:
        verification_selector_path = Path(args.verification_point_selection_artifact)
        (
            verification_selector_rows,
            verification_selector_metadata,
        ) = _load_verification_point_selection_artifact(
            verification_selector_path,
            detector=detector,
            proposals=proposals,
            selected_rows=selected_rows,
            detector_path=Path(args.detector_query_cache),
            proposals_path=proposal_path,
            candidate_path=candidate_path,
        )
        selector_config = verification_selector_metadata.get("selection")
        if not isinstance(selector_config, Mapping) or int(
            selector_config.get("point_count", -1)
        ) != int(args.verification_point_count):
            raise ValueError(
                "verification selector point budget must equal --verification_point_count"
            )

    source_bank_metadata = _landmark_bank_metadata(source_bank_path)
    landmark_index, bank_metadata = load_landmark_index_npz(bank_path)
    if bank_path != source_bank_path:
        _validate_alternate_verification_bank(
            source_bank_metadata, bank_metadata
        )
    bank_descriptor_space_id = str(bank_metadata.get("descriptor_space_id", ""))
    detector_descriptor_space_id = str(
        detector_metadata.get("descriptor_space_id", "")
    )
    bank_manifest = bank_metadata.get("descriptor_space_manifest")
    if not isinstance(bank_manifest, Mapping):
        raise ValueError("landmark bank has no descriptor-space manifest")
    projection_space_id = str(bank_manifest.get("projection_space_id", ""))
    detector_projection_space_id = str(
        detector_metadata.get("projection_space_id", "")
    )
    if detector_projection_space_id:
        if detector_projection_space_id != projection_space_id:
            raise ValueError("detector and landmark projection spaces differ")
        projection_compatibility = "explicit_projection_space_id"
    elif detector_descriptor_space_id == bank_descriptor_space_id:
        projection_compatibility = "legacy_exact_descriptor_space_id"
    else:
        legacy_fields_match = (
            str(detector_metadata.get("matcha_joint_checkpoint_sha256", ""))
            == str(bank_manifest.get("checkpoint_sha256", ""))
            and str(detector_metadata.get("feature_key", ""))
            == str(bank_manifest.get("feature_key", ""))
            and int(detector_metadata.get("global_descriptor_dimension", -1))
            == int(bank_manifest.get("descriptor_dimension", -2))
        )
        if not legacy_fields_match:
            raise ValueError(
                "legacy detector cache cannot be proven projection-compatible "
                "with the landmark bank"
            )
        projection_compatibility = (
            "legacy_checkpoint_feature_dimension_projection_equivalence_v1"
        )
    mixed_points: MixedVerificationPoints | None = None
    if mixed_verification_path is not None:
        mixed_points = _load_mixed_verification_points_for_scoring(
            mixed_verification_path,
            detector_path=Path(args.detector_query_cache),
            candidate_path=candidate_path,
            source_bank_path=source_bank_path,
            detector=detector,
            proposals=proposals,
            selected_rows=selected_rows,
            landmark_track_ids=landmark_index.track_ids,
            bank_metadata=bank_metadata,
        )
        mixed_counts = {
            str(query_id): int(len(mixed_points.rows_for_query(str(query_id))))
            for query_id in np.unique(mixed_points.query_ids).tolist()
        }
        if not mixed_counts or any(
            count != int(args.verification_point_count)
            for count in mixed_counts.values()
        ):
            raise ValueError(
                "mixed verification point count must equal --verification_point_count "
                "for every artifact query"
            )
    if args.prototype_view_geometry is None:
        view_index = LandmarkObservationViewIndex.from_track_observations(
            landmark_index.track_ids,
            support_geometry["track_ids"],
            support_geometry["viewing_rays"],
        )
        view_geometry_mode = "all_observation_rays_with_track_aggregated_descriptor"
    else:
        view_index, support_geometry_metadata = (
            load_landmark_prototype_view_index_npz(
                Path(args.prototype_view_geometry),
                landmark_index,
                expected_descriptor_space_id=bank_descriptor_space_id,
            )
        )
        view_geometry_mode = "descriptor_prototype_aligned_view_distribution"
    config = _strict_absolute_likelihood_config(args)
    verifier = IndependentLandmarkPoseVerifier(
        landmark_index, view_index, config
    )

    maplet_track_ids = None
    maplet_cluster_ids = None
    maplet_metadata: dict[str, object] | None = None
    if bool(args.purge_maplet_clusters):
        if args.maplet_support_index is None:
            raise ValueError("maplet purge requires --maplet_support_index")
        maplet_index, maplet_metadata = load_local_maplet_support_index_npz(
            Path(args.maplet_support_index)
        )
        maplet_track_ids = maplet_index.anchor_track_ids
        maplet_cluster_ids = build_disjoint_maplet_cluster_ids(maplet_index)

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    image_camera_ids = read_colmap_image_camera_ids_binary(
        model_dir / "images.bin"
    )
    query_ids = merged["query_ids"].astype(str)
    split_names = merged["split_names"].astype(str)
    labels = merged["evaluation_labels"].astype(str)
    hypothesis_indices = merged["hypothesis_indices"].astype(np.int64)
    shortlisted = merged["shortlisted_for_verification"].astype(bool)
    chosen = merged["chosen_for_optional_pose"].astype(bool)
    preliminary_scores = np.asarray(
        merged["preliminary_log_likelihood_means"], dtype=np.float64
    )
    poses = np.asarray(merged["poses_w2c"], dtype=np.float64)
    all_split_names = set(split_names.tolist())
    if requested_splits is not None:
        unknown_splits = sorted(requested_splits.difference(all_split_names))
        if unknown_splits:
            raise ValueError(f"score_splits are absent from hypotheses: {unknown_splits}")
    group_keys = sorted(
        key
        for key in set(zip(split_names.tolist(), labels.tolist(), query_ids.tolist()))
        if requested_splits is None or key[0] in requested_splits
    )
    group_keys = [
        key
        for index, key in enumerate(group_keys)
        if index % int(args.query_shard_count) == int(args.query_shard_index)
    ]
    if mixed_points is not None:
        missing = [
            f"{split_name}:{query_id}"
            for split_name, _label, query_id in group_keys
            if not np.any(
                (mixed_points.query_ids == str(query_id))
                & (mixed_points.split_names == str(split_name))
            )
        ]
        if missing:
            raise ValueError(
                "mixed verification artifact lacks scored query/split points: "
                + ", ".join(missing[:8])
            )

    output_rows: list[dict[str, object]] = []
    start = time.time()
    for group_index, (split_name, label, query_id) in enumerate(group_keys):
        if query_id not in image_camera_ids:
            raise ValueError(f"query absent from COLMAP image ownership: {query_id}")
        camera_id = int(image_camera_ids[query_id])
        if camera_id not in cameras:
            raise ValueError(f"camera {camera_id} for {query_id} is missing")
        all_group = np.flatnonzero(
            (query_ids == query_id)
            & (split_names == split_name)
            & (labels == label)
        )
        if str(args.hypothesis_scope) == "shortlisted":
            group = all_group[shortlisted[all_group]]
        elif str(args.hypothesis_scope) == "all":
            group = all_group
        else:
            finite = all_group[np.isfinite(preliminary_scores[all_group])]
            order = np.argsort(
                -preliminary_scores[finite], kind="mergesort"
            )
            limit = int(args.hypothesis_limit)
            group = finite[order if limit == 0 else order[:limit]]
            chosen_rows = all_group[chosen[all_group]]
            if chosen_rows.size:
                group = np.unique(np.concatenate([group, chosen_rows]))
        if group.size == 0:
            raise ValueError(f"{query_id}: hypothesis scope selected no rows")
        if mixed_points is not None:
            points, excluded_tracks, point_audit = (
                _mixed_verification_points_for_query(
                    query_id,
                    mixed_points=mixed_points,
                    detector=detector,
                    proposals=proposals,
                    selected_rows=selected_rows,
                )
            )
        else:
            preselected_rows = None
            if verification_selector_rows is not None:
                preselected_rows = verification_selector_rows.get(str(query_id))
                if preselected_rows is None:
                    raise ValueError(
                        f"{query_id}: verification point selector has no fixed rows"
                    )
            if prior_overlay is None:
                raise RuntimeError("detector verifier is missing its fixed prior overlay")
            points, excluded_tracks, point_audit = _verification_points_for_query(
                query_id,
                detector=detector,
                proposals=proposals,
                selected_rows=selected_rows,
                point_count=int(args.verification_point_count),
                detector_log_merit_weight=float(args.detector_log_merit_weight),
                candidate_prior_overlay=prior_overlay,
                candidate_spatial_mode_index=spatial_mode_index,
                candidate_spatial_offsets_xy=spatial_offsets_xy,
                candidate_spatial_log_probability_arrays=spatial_log_probability_arrays,
                candidate_spatial_source_probabilities=spatial_source_probabilities,
                preselected_verification_rows=preselected_rows,
            )
        spatial_audit = _spatial_materialization_audit(points)
        if (
            spatial_mode_index is not None
            and not bool(args.allow_unmaterialized_spatial_queries)
            and int(spatial_audit["materialized_verification_point_count"]) == 0
        ):
            raise ValueError(
                f"{query_id}: no candidate RGB spatial modes are materialized "
                "for this held-out verification set; restrict --score_splits "
                "or generate an OOF spatial artifact before scoring"
            )
        direct_excluded_count = int(len(excluded_tracks))
        if bool(args.purge_maplet_clusters):
            excluded_tracks = _maplet_purged_tracks(
                excluded_tracks,
                maplet_track_ids=np.asarray(maplet_track_ids),
                maplet_cluster_ids=np.asarray(maplet_cluster_ids),
            )
        eligible = verifier.eligible_mask_excluding_tracks(excluded_tracks)
        group_output_start = len(output_rows)
        for source_row in group.tolist():
            score = verifier.score_pose(
                poses[source_row],
                cameras[camera_id],
                points,
                eligible_landmark_mask=eligible,
                emit_unique_track_assignment_diagnostic=(
                    assignment_diagnostic_enabled
                ),
                unique_track_assignment_null_probability_floor=null_floor,
                unique_track_assignment_minimum_active_visual_mass=(
                    active_mass_floor
                ),
            )
            assignment_row: dict[str, object] = {}
            if assignment_diagnostic_enabled:
                assignment = score.unique_track_assignment
                if assignment is None:
                    raise RuntimeError(
                        "unique-track assignment diagnostic was requested but missing"
                    )
                assignment_row = {
                    "unique_track_assignment_log_joint": float(
                        assignment.log_joint
                    ),
                    "unique_track_assignment_log_gain_over_null": float(
                        assignment.log_gain_over_null
                    ),
                    "unique_track_assignment_independent_log_joint": float(
                        assignment.independent_log_joint
                    ),
                    "unique_track_assignment_independent_log_gain_over_null": float(
                        assignment.independent_log_gain_over_null
                    ),
                    "unique_track_assignment_collision_penalty": float(
                        assignment.collision_penalty
                    ),
                    "unique_track_assignment_active_point_count": int(
                        assignment.active_point_count
                    ),
                    "unique_track_assignment_eligible_edge_count": int(
                        assignment.eligible_edge_count
                    ),
                    "unique_track_assignment_selected_candidate_count": int(
                        assignment.selected_candidate_count
                    ),
                    "unique_track_assignment_selected_unique_track_count": int(
                        assignment.selected_unique_track_count
                    ),
                }
            if str(args.selection_statistic) == "unique_track_assignment_log_joint":
                if not assignment_diagnostic_enabled:
                    raise RuntimeError(
                        "unique-track assignment selection was not materialized"
                    )
                selection_score = float(
                    assignment_row["unique_track_assignment_log_joint"]
                )
            else:
                selection_score = float(score.statistic(str(args.selection_statistic)))
            output_rows.append(
                {
                    "query_id": query_id,
                    "split_name": split_name,
                    "evaluation_label": label,
                    "hypothesis_index": int(hypothesis_indices[source_row]),
                    "source_chosen_for_optional_pose": bool(chosen[source_row]),
                    "independent_log_likelihood_sum": float(
                        score.log_likelihood_sum
                    ),
                    "independent_log_likelihood_mean": float(
                        score.log_likelihood_mean
                    ),
                    "independent_log_likelihood_median": float(
                        score.log_likelihood_median
                    ),
                    "independent_log_likelihood_trimmed_mean_10": float(
                        score.log_likelihood_trimmed_mean_10
                    ),
                    "independent_log_likelihood_worst_quartile_mean": float(
                        score.log_likelihood_worst_quartile_mean
                    ),
                    "independent_log_likelihood_lcb95": float(
                        score.log_likelihood_lcb95
                    ),
                    "independent_spatial_median_of_means_2x2": float(
                        score.spatial_median_of_means_2x2
                    ),
                    "independent_selection_score": selection_score,
                    "independent_effective_point_count": int(
                        score.effective_point_count
                    ),
                    "independent_evidence_coverage": float(
                        score.evidence_coverage
                    ),
                    "projected_landmark_count": int(
                        score.projected_landmark_count
                    ),
                    "verification_point_count": int(
                        score.verification_point_count
                    ),
                    "direct_excluded_track_count": direct_excluded_count,
                    "total_excluded_track_count": int(len(excluded_tracks)),
                    **point_audit,
                    **spatial_audit,
                    **assignment_row,
                }
            )
        group_rows = output_rows[group_output_start:]
        best_local = int(
            np.argmax(
                [
                    float(row["independent_selection_score"])
                    for row in group_rows
                ]
            )
        )
        for local_index, row in enumerate(group_rows):
            row["independent_score_top1"] = bool(local_index == best_local)
        print(
            json.dumps(
                {
                    "stage": "independent_landmark_hypothesis_scoring",
                    "completed_queries": int(group_index + 1),
                    "total_queries": int(len(group_keys)),
                    "query_id": query_id,
                    "hypothesis_count": int(len(group)),
                    "elapsed_seconds": float(time.time() - start),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    field_specs = {
        "query_ids": ("query_id", str),
        "split_names": ("split_name", str),
        "evaluation_labels": ("evaluation_label", str),
        "hypothesis_indices": ("hypothesis_index", np.int64),
        "source_chosen_for_optional_pose": (
            "source_chosen_for_optional_pose",
            bool,
        ),
        "independent_score_top1": ("independent_score_top1", bool),
        "independent_log_likelihood_sums": (
            "independent_log_likelihood_sum",
            np.float64,
        ),
        "independent_log_likelihood_means": (
            "independent_log_likelihood_mean",
            np.float64,
        ),
        "independent_log_likelihood_medians": (
            "independent_log_likelihood_median",
            np.float64,
        ),
        "independent_log_likelihood_trimmed_means_10": (
            "independent_log_likelihood_trimmed_mean_10",
            np.float64,
        ),
        "independent_log_likelihood_worst_quartile_means": (
            "independent_log_likelihood_worst_quartile_mean",
            np.float64,
        ),
        "independent_log_likelihood_lcb95s": (
            "independent_log_likelihood_lcb95",
            np.float64,
        ),
        "independent_spatial_median_of_means_2x2": (
            "independent_spatial_median_of_means_2x2",
            np.float64,
        ),
        "independent_selection_scores": (
            "independent_selection_score",
            np.float64,
        ),
        "independent_effective_point_counts": (
            "independent_effective_point_count",
            np.int64,
        ),
        "independent_evidence_coverages": (
            "independent_evidence_coverage",
            np.float64,
        ),
        "projected_landmark_counts": ("projected_landmark_count", np.int64),
        "verification_point_counts": ("verification_point_count", np.int64),
        "direct_excluded_track_counts": (
            "direct_excluded_track_count",
            np.int64,
        ),
        "total_excluded_track_counts": (
            "total_excluded_track_count",
            np.int64,
        ),
        "fit_query_point_counts": ("fit_query_point_count", np.int64),
        "available_unused_query_point_counts": (
            "available_unused_query_point_count",
            np.int64,
        ),
        "selected_verification_point_counts": (
            "selected_verification_point_count",
            np.int64,
        ),
        "candidate_spatial_materialized_verification_point_counts": (
            "materialized_verification_point_count",
            np.int64,
        ),
        "candidate_spatial_materialized_candidate_view_counts": (
            "materialized_candidate_view_count",
            np.int64,
        ),
    }
    if assignment_diagnostic_enabled:
        field_specs.update(
            {
                "unique_track_assignment_log_joints": (
                    "unique_track_assignment_log_joint",
                    np.float64,
                ),
                "unique_track_assignment_log_gains_over_null": (
                    "unique_track_assignment_log_gain_over_null",
                    np.float64,
                ),
                "unique_track_assignment_independent_log_joints": (
                    "unique_track_assignment_independent_log_joint",
                    np.float64,
                ),
                "unique_track_assignment_independent_log_gains_over_null": (
                    "unique_track_assignment_independent_log_gain_over_null",
                    np.float64,
                ),
                "unique_track_assignment_collision_penalties": (
                    "unique_track_assignment_collision_penalty",
                    np.float64,
                ),
                "unique_track_assignment_active_point_counts": (
                    "unique_track_assignment_active_point_count",
                    np.int64,
                ),
                "unique_track_assignment_eligible_edge_counts": (
                    "unique_track_assignment_eligible_edge_count",
                    np.int64,
                ),
                "unique_track_assignment_selected_candidate_counts": (
                    "unique_track_assignment_selected_candidate_count",
                    np.int64,
                ),
                "unique_track_assignment_selected_unique_track_counts": (
                    "unique_track_assignment_selected_unique_track_count",
                    np.int64,
                ),
            }
        )
    arrays = {
        output_name: np.asarray(
            [row[row_name] for row in output_rows], dtype=dtype
        )
        for output_name, (row_name, dtype) in field_specs.items()
    }
    if mixed_points is None:
        if (
            prior_overlay is None
            or prior_overlay_path is None
            or fixed_candidate_topk_retained is None
            or fixed_candidate_topk_removed_mass is None
        ):
            raise RuntimeError("detector verifier metadata lacks fixed overlay state")
        query_point_selection = {
            "source": (
                "detector_cache_points_unused_by_candidate_artifact"
                if verification_selector_path is None
                else "target_free_identity_posterior_spatial_quota_selector"
            ),
            "verification_point_count": int(args.verification_point_count),
            "merit": (
                "global_coarse_top1_plus_weighted_log_detector_score"
                if verification_selector_path is None
                else verification_selector_metadata.get("selection_strategy")
            ),
            "detector_log_merit_weight": (
                float(args.detector_log_merit_weight)
                if verification_selector_path is None
                else None
            ),
            "verification_point_selection_artifact": (
                None
                if verification_selector_path is None
                else str(verification_selector_path)
            ),
            "verification_point_selection_artifact_sha256": (
                None
                if verification_selector_path is None
                else file_sha256_short(verification_selector_path)
            ),
            "verification_point_selection_metadata_sha256": (
                None
                if verification_selector_path is None
                else _canonical_hash(verification_selector_metadata)
            ),
            "mixed_verification_points_artifact": None,
            "mixed_verification_points_artifact_sha256": None,
            "score_splits": (
                None if requested_splits is None else sorted(requested_splits)
            ),
        }
        fixed_candidate_topk_metadata = {
            "applied": args.fixed_candidate_top_k is not None,
            "top_k": args.fixed_candidate_top_k,
            "candidate_column_count": int(
                np.asarray(prior_overlay["candidate_track_ids"]).shape[1]
            ),
            "ranking": (
                "frozen_overlay_probability_descending_stable_candidate_column_v1"
            ),
            "removed_candidate_mass_transferred_to_null": True,
            "removed_candidate_mass": {
                "mean": float(np.mean(fixed_candidate_topk_removed_mass)),
                "maximum": float(np.max(fixed_candidate_topk_removed_mass)),
            },
            "retained_candidate_slot_count": int(
                np.count_nonzero(fixed_candidate_topk_retained)
            ),
        }
        fixed_prior_input = {
            "fixed_candidate_prior_overlay": str(prior_overlay_path),
            "fixed_candidate_prior_overlay_sha256": file_sha256_short(
                prior_overlay_path
            ),
            "fixed_candidate_prior_overlay_metadata_sha256": _canonical_hash(
                prior_overlay_metadata
            ),
            "mixed_verification_points_artifact": None,
            "mixed_verification_points_artifact_sha256": None,
            "mixed_verification_points_metadata_sha256": None,
        }
    else:
        if mixed_verification_path is None:
            raise RuntimeError("mixed verifier source path is missing")
        query_point_selection = {
            "source": MIXED_VERIFICATION_POINT_SOURCE,
            "verification_point_count": int(args.verification_point_count),
            "merit": (
                "frozen_per_point_full_global_topl_coarse_probability_plus_"
                "explicit_null_v1"
            ),
            "detector_log_merit_weight": None,
            "verification_point_selection_artifact": None,
            "verification_point_selection_artifact_sha256": None,
            "verification_point_selection_metadata_sha256": None,
            "mixed_verification_points_artifact": str(mixed_verification_path),
            "mixed_verification_points_artifact_sha256": file_sha256_short(
                mixed_verification_path
            ),
            "mixed_verification_points_metadata_sha256": _canonical_hash(
                mixed_points.metadata
            ),
            "mixed_point_sources": sorted(
                set(np.asarray(mixed_points.point_sources).astype(str).tolist())
            ),
            "score_splits": (
                None if requested_splits is None else sorted(requested_splits)
            ),
        }
        fixed_candidate_topk_metadata = {
            "applied": False,
            "top_k": None,
            "candidate_column_count": int(
                np.asarray(mixed_points.candidate_track_ids).shape[1]
            ),
            "ranking": "not_applicable_mixed_verifier_has_immutable_topl_v1",
            "removed_candidate_mass_transferred_to_null": True,
            "removed_candidate_mass": {"mean": 0.0, "maximum": 0.0},
            "retained_candidate_slot_count": int(
                np.count_nonzero(mixed_points.candidate_track_ids >= 0)
            ),
        }
        fixed_prior_input = {
            "fixed_candidate_prior_overlay": None,
            "fixed_candidate_prior_overlay_sha256": None,
            "fixed_candidate_prior_overlay_metadata_sha256": None,
            "mixed_verification_points_artifact": str(mixed_verification_path),
            "mixed_verification_points_artifact_sha256": file_sha256_short(
                mixed_verification_path
            ),
            "mixed_verification_points_metadata_sha256": _canonical_hash(
                mixed_points.metadata
            ),
        }
    metadata = {
        "format": SCORE_ARTIFACT_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "strict_absolute_evidence_contract": {
            "heldout_query_rows": True,
            "fixed_global_topl": True,
            "explicit_null_mass": True,
            "identity_prior_fixed_across_hypotheses": True,
            "verification_point_selector_fixed_across_hypotheses": True,
            "support_appearance_posterior_pose_independent": True,
            "pose_local_candidate_reselection": False,
            "pose_conditioned_refinement": False,
            "pose_conditioned_unique_track_assignment": (
                "fixed_topl_null_or_unique_physical_track_map_diagnostic_v1"
                if assignment_diagnostic_enabled
                else "not_used"
            ),
            "unique_track_assignment_dustbin_missing_and_crossfit_neutral": bool(
                assignment_diagnostic_enabled
            ),
            "unique_track_assignment_changes_default_mixture_score": False,
            "candidate_specific_rgb_spatial_modes": bool(
                spatial_mode_index is not None
            ),
            "candidate_spatial_semantics": (
                "per_view_normalized_continuous_gaussian_mixture_relative_to_"
                "grid_uniform_null_v1"
                if spatial_mode_index is not None
                else "not_used"
            ),
            "candidate_spatial_dustbin_and_missing_pose_independent": bool(
                spatial_mode_index is not None
            ),
            "candidate_spatial_omitted_topk_mass_is_null": bool(
                spatial_mode_index is not None
            ),
            "fixed_candidate_topk_ablation": fixed_candidate_topk_metadata,
            "candidate_spatial_query_materialization": (
                "required_at_least_one_heldout_verification_point"
                if spatial_mode_index is not None
                and not bool(args.allow_unmaterialized_spatial_queries)
                else (
                    "diagnostic_unknown_queries_allowed"
                    if spatial_mode_index is not None
                    else "not_used"
                )
            ),
            "pose_effects": (
                "positive_depth_image_bounds_and_optional_loose_view_gate_only"
            ),
        },
        "camera_ownership_parser": "image_name_to_camera_id_pose_discarded_v1",
        "version": INDEPENDENT_LANDMARK_POSE_LIKELIHOOD_VERSION,
        "implementation": implementation_manifest,
        "row_count": int(len(output_rows)),
        "query_count": int(len(group_keys)),
        "query_shard_count": int(args.query_shard_count),
        "query_shard_index": int(args.query_shard_index),
        "hypothesis_compatibility_sha256": compatibility_hash,
        "config": config.to_dict(),
        "unique_track_assignment_diagnostic": {
            "enabled": assignment_diagnostic_enabled,
            "solver": (
                "exact_sparse_topl_bipartite_assignment_with_per_point_null_v1"
                if assignment_diagnostic_enabled
                else "not_used"
            ),
            "edge_source": (
                "frozen_candidate_prior_times_candidate_specific_rgb_spatial_"
                "likelihood_ratio"
                if assignment_diagnostic_enabled
                else "not_used"
            ),
            "edge_requirements": (
                "in_bank_pose_visible_materialized_non_dustbin_rgb_mass"
                if assignment_diagnostic_enabled
                else "not_used"
            ),
            "null_probability_floor": (
                null_floor if assignment_diagnostic_enabled else None
            ),
            "minimum_active_visual_mass": (
                active_mass_floor if assignment_diagnostic_enabled else None
            ),
            "selection_is_opt_in": bool(
                str(args.selection_statistic)
                == "unique_track_assignment_log_joint"
            ),
        },
        "selection": {
            "statistic": str(args.selection_statistic),
            "score_field": "independent_selection_scores",
            "available_statistics": list(SELECTION_STATISTICS),
            "statistic_score_fields": dict(STATISTIC_SCORE_FIELDS),
            "tie_break": "first_frozen_hypothesis_row_v1",
        },
        "query_point_selection": query_point_selection,
        "fixed_candidate_topk_ablation": fixed_candidate_topk_metadata,
        "hypothesis_scope": {
            "mode": str(args.hypothesis_scope),
            "limit": int(args.hypothesis_limit),
            "source_score": (
                None
                if str(args.hypothesis_scope) != "preliminary_topk"
                else "preliminary_log_likelihood_means"
            ),
            "source_chosen_pose_forced_into_scope": bool(
                str(args.hypothesis_scope) == "preliminary_topk"
            ),
        },
        "crossfit": {
            "query_token_disjoint": True,
            "physical_track_disjoint": True,
            "maplet_cluster_disjoint": bool(args.purge_maplet_clusters),
            "fit_track_definition": "all_top20_tracks_for_each_fit_query_point",
            "denominator_fixed_across_hypotheses": True,
        },
        "view_geometry_mode": view_geometry_mode,
        "inputs": {
            "hypothesis_artifacts": [str(path) for path in paths],
            "hypothesis_artifact_sha256": [
                file_sha256_short(path) for path in paths
            ],
            "detector_query_cache": str(args.detector_query_cache),
            "detector_query_cache_sha256": file_sha256_short(
                Path(args.detector_query_cache)
            ),
            "proposals": str(proposal_path),
            "proposals_sha256": file_sha256_short(proposal_path),
            "candidate_artifact": str(candidate_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "verification_point_selection_artifact": (
                None
                if verification_selector_path is None
                else str(verification_selector_path)
            ),
            "verification_point_selection_artifact_sha256": (
                None
                if verification_selector_path is None
                else file_sha256_short(verification_selector_path)
            ),
            "verification_point_selection_metadata_sha256": (
                None
                if verification_selector_path is None
                else _canonical_hash(verification_selector_metadata)
            ),
            **fixed_prior_input,
            "candidate_spatial_likelihood": [
                str(path) for path in spatial_mode_paths
            ],
            "candidate_spatial_likelihood_sha256": [
                file_sha256_short(path) for path in spatial_mode_paths
            ],
            "candidate_spatial_likelihood_metadata_sha256": [
                _canonical_hash(row) for row in spatial_mode_metadata
            ],
            "projected_landmark_bank": str(source_bank_path),
            "projected_landmark_bank_sha256": file_sha256_short(
                source_bank_path
            ),
            "independent_verification_landmark_bank": str(bank_path),
            "independent_verification_landmark_bank_sha256": file_sha256_short(
                bank_path
            ),
            "support_geometry_index": args.support_geometry_index,
            "support_geometry_index_sha256": file_sha256_short(
                Path(args.support_geometry_index)
            ) if args.support_geometry_index is not None else None,
            "prototype_view_geometry": args.prototype_view_geometry,
            "prototype_view_geometry_sha256": (
                None
                if args.prototype_view_geometry is None
                else file_sha256_short(Path(args.prototype_view_geometry))
            ),
            "maplet_support_index": args.maplet_support_index,
            "maplet_support_index_sha256": (
                None
                if args.maplet_support_index is None
                else file_sha256_short(Path(args.maplet_support_index))
            ),
            "colmap_cameras_bin": str(model_dir / "cameras.bin"),
            "colmap_cameras_bin_sha256": file_sha256_short(
                model_dir / "cameras.bin"
            ),
            "colmap_images_bin_camera_ownership_only": str(
                model_dir / "images.bin"
            ),
            "colmap_images_bin_sha256": file_sha256_short(
                model_dir / "images.bin"
            ),
            "detector_descriptor_space_id": detector_descriptor_space_id,
            "landmark_descriptor_space_id": bank_descriptor_space_id,
            "projection_space_id": projection_space_id,
            "projection_compatibility": projection_compatibility,
            "candidate_metadata_sha256": _canonical_hash(candidate_metadata),
            "support_geometry_metadata_sha256": _canonical_hash(
                support_geometry_metadata
            ),
            "maplet_metadata_sha256": (
                None if maplet_metadata is None else _canonical_hash(maplet_metadata)
            ),
        },
        "elapsed_seconds": float(time.time() - start),
    }
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "independent_landmark_hypothesis_scoring",
        "output": str(output_path),
        "metadata": metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
