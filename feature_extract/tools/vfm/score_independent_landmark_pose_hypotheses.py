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
    load_inference_artifact,
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
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
)


SCORE_ARTIFACT_FORMAT = "independent_landmark_hypothesis_scores_v1"
CANDIDATE_PRIOR_OVERLAY_FORMATS = frozenset(
    {
        "candidate_maplet_prior_overlay_v1",
        "candidate_image_context_prior_overlay_v1",
    }
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
    parser.add_argument(
        "--fixed_candidate_prior_overlay",
        required=True,
        help=(
            "Target-free, proposal-aligned candidate probabilities plus explicit "
            "null mass. Strict absolute scoring refuses an implicit denominator."
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
    parser.add_argument("--maplet_support_index", default=None)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verification_point_count", type=int, default=192)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    parser.add_argument("--nearest_landmarks", type=int, default=4)
    parser.add_argument("--maximum_reprojection_distance_px", type=float, default=8.0)
    parser.add_argument("--spatial_sigma_px", type=float, default=3.0)
    parser.add_argument("--descriptor_temperature", type=float, default=0.04)
    parser.add_argument("--outlier_likelihood", type=float, default=0.01)
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
    if str(metadata.get("probability_semantics")) != (
        "candidate_identity_probability_plus_explicit_null_equals_one"
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
    loaded = [load_inference_artifact(path) for path in paths]
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
    keys = set(loaded[0][0]) - {"metadata_json"}
    if any((set(arrays) - {"metadata_json"}) != keys for arrays, _ in loaded):
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.verification_point_count) <= 0:
        raise ValueError("verification_point_count must be positive")
    if int(args.query_shard_count) <= 0 or not (
        0 <= int(args.query_shard_index) < int(args.query_shard_count)
    ):
        raise ValueError("invalid query shard")
    if int(args.hypothesis_limit) < 0:
        raise ValueError("hypothesis_limit must be non-negative")
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
    prior_overlay_path = Path(args.fixed_candidate_prior_overlay)
    prior_overlay, prior_overlay_metadata = _load_candidate_prior_overlay(
        prior_overlay_path,
        proposals_path=proposal_path,
        proposals=proposals,
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
    group_keys = sorted(set(zip(split_names.tolist(), labels.tolist(), query_ids.tolist())))
    group_keys = [
        key
        for index, key in enumerate(group_keys)
        if index % int(args.query_shard_count) == int(args.query_shard_index)
    ]

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
        points, excluded_tracks, point_audit = _verification_points_for_query(
            query_id,
            detector=detector,
            proposals=proposals,
            selected_rows=selected_rows,
            point_count=int(args.verification_point_count),
            detector_log_merit_weight=float(args.detector_log_merit_weight),
            candidate_prior_overlay=prior_overlay,
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
            )
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
                }
            )
        group_rows = output_rows[group_output_start:]
        best_local = int(
            np.argmax(
                [
                    float(row["independent_log_likelihood_mean"])
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
    }
    arrays = {
        output_name: np.asarray(
            [row[row_name] for row in output_rows], dtype=dtype
        )
        for output_name, (row_name, dtype) in field_specs.items()
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
            "support_appearance_posterior_pose_independent": True,
            "pose_local_candidate_reselection": False,
            "pose_conditioned_refinement": False,
            "pose_effects": (
                "positive_depth_image_bounds_and_optional_loose_view_gate_only"
            ),
        },
        "camera_ownership_parser": "image_name_to_camera_id_pose_discarded_v1",
        "version": INDEPENDENT_LANDMARK_POSE_LIKELIHOOD_VERSION,
        "row_count": int(len(output_rows)),
        "query_count": int(len(group_keys)),
        "query_shard_count": int(args.query_shard_count),
        "query_shard_index": int(args.query_shard_index),
        "hypothesis_compatibility_sha256": compatibility_hash,
        "config": config.to_dict(),
        "query_point_selection": {
            "source": "detector_cache_points_unused_by_candidate_artifact",
            "verification_point_count": int(args.verification_point_count),
            "merit": "global_coarse_top1_plus_weighted_log_detector_score",
            "detector_log_merit_weight": float(args.detector_log_merit_weight),
        },
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
            "fixed_candidate_prior_overlay": str(prior_overlay_path),
            "fixed_candidate_prior_overlay_sha256": file_sha256_short(
                prior_overlay_path
            ),
            "fixed_candidate_prior_overlay_metadata_sha256": _canonical_hash(
                prior_overlay_metadata
            ),
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
