"""Refine and select frozen pose hypotheses with strict three-way map evidence.

This command is target-free. Query points and physical landmark/maplet identities
are independently partitioned into refine, rank, and final-audit roles. The audit
fold can only promote one rank-frozen optional pose over the immutable source pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _canonical_hash,
    _landmark_bank_metadata,
    _load_candidate_prior_overlay,
    _load_candidate_spatial_mode_index,
    _load_npz,
    _maplet_purged_tracks,
    _merge_hypothesis_artifacts,
    _spatial_materialization_audit,
    _validate_alternate_verification_bank,
    _validate_declared_input,
    _verification_points_for_query,
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
    IndependentPoseCorrespondences,
    IndependentPoseRefinementConfig,
    IndependentPoseRefinementResult,
    LandmarkObservationViewIndex,
    deterministic_identity_folds,
    load_landmark_prototype_view_index_npz,
    spatially_balanced_point_folds,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    pose_information_diagnostics,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


ARTIFACT_FORMAT = "independent_crossfit_pose_alignment_v1"
SELECTED_POSE_ARTIFACT_FORMAT = "selected_pose_inference_only_v1"
ROLE_NAMES = ("refine", "rank", "audit")
SCORE_STATISTICS = (
    "mean",
    "median",
    "trimmed_mean_10",
    "worst_quartile_mean",
    "lcb95",
    "spatial_median_of_means_2x2",
)
SPLIT_FILTERS = ("all", "train", "validation", "test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument(
        "--fixed_candidate_prior_overlay",
        default="",
        help=(
            "Target-free learned candidate/null posterior aligned to every "
            "proposal row; required by learned_probability prior mode."
        ),
    )
    parser.add_argument(
        "--candidate_spatial_likelihood",
        default="",
        help=(
            "Optional comma-separated target-free candidate_spatial_likelihood_v7 "
            "artifacts. When supplied, both rank and audit folds must contain "
            "materialized RGB spatial evidence."
        ),
    )
    parser.add_argument(
        "--allow_unmaterialized_spatial_roles",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch. Production cross-fit refuses a rank "
            "or audit fold without materialized candidate RGB spatial modes."
        ),
    )
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--independent_verification_landmark_bank", required=True)
    parser.add_argument("--prototype_view_geometry", default="")
    parser.add_argument("--support_geometry_index", default="")
    parser.add_argument(
        "--verification_descriptor_source",
        choices=("global", "local"),
        default="global",
    )
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--immutable_source_pose_artifact",
        default="",
        help=(
            "Frozen selected-pose artifact used as the bit-exact fallback. By "
            "default this is resolved from the grouped-hypothesis input manifest."
        ),
    )
    parser.add_argument(
        "--immutable_source_pose_evaluation_label",
        default="",
        help=(
            "Policy label inside the immutable source artifact. By default this "
            "is resolved from the grouped-hypothesis input manifest."
        ),
    )
    parser.add_argument(
        "--allow_immutable_source_override",
        action="store_true",
        help=(
            "Allow an explicitly supplied external source pose artifact in place "
            "of the hypothesis-declared one. The artifact must retain the same "
            "candidate/map/scene lineage and the override is recorded in outputs."
        ),
    )
    parser.add_argument(
        "--source_pose_policy",
        choices=("immutable_artifact", "grouped_artifact_chosen"),
        default="immutable_artifact",
        help=(
            "Use the historical external immutable pose or the already-frozen "
            "chosen grouped pose as the source baseline. The latter is useful "
            "only when every calibration and held-out split uses it."
        ),
    )
    parser.add_argument(
        "--allow_missing_immutable_source_for_train_calibration",
        action="store_true",
        help=(
            "Train-only calibration escape hatch. When the declared immutable "
            "source artifact omits a train query, use the grouped artifact's "
            "already-frozen chosen pose as the baseline. Validation/test and "
            "production-style runs always reject missing immutable sources."
        ),
    )
    parser.add_argument("--verification_point_count", type=int, default=288)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    parser.add_argument("--point_fold_seed", type=int, default=173)
    parser.add_argument("--landmark_fold_seed", type=int, default=271)
    parser.add_argument(
        "--crossfit_role_count",
        type=int,
        choices=(2, 3),
        default=3,
        help=(
            "Use two active rank/audit roles when refinement and staged "
            "shortlisting are disabled, otherwise use refine/rank/audit."
        ),
    )
    parser.add_argument(
        "--rank_uses_complement_of_audit_fold",
        action="store_true",
        help=(
            "With three spatial/map folds and no refinement, rank hypotheses "
            "on two folds while reserving the remaining fold for the independent "
            "audit. This keeps rank/audit disjoint but gives rank two-thirds of "
            "the frozen RGB evidence."
        ),
    )
    parser.add_argument("--nearest_landmarks", type=int, default=4)
    parser.add_argument("--maximum_reprojection_distance_px", type=float, default=8.0)
    parser.add_argument("--spatial_sigma_px", type=float, default=3.0)
    parser.add_argument("--descriptor_temperature", type=float, default=0.04)
    parser.add_argument("--outlier_likelihood", type=float, default=0.01)
    parser.add_argument("--minimum_observation_count", type=int, default=2)
    parser.add_argument("--maximum_view_angle_deg", type=float, default=15.0)
    parser.add_argument("--kdtree_workers", type=int, default=1)
    parser.add_argument(
        "--score_candidate_mode",
        choices=("fixed_global_topl", "pose_local_knn"),
        default="fixed_global_topl",
    )
    parser.add_argument(
        "--fixed_candidate_prior_source",
        choices=(
            "prototype_similarity",
            "prototype_similarity_with_learned_null",
            "coarse_score",
            "learned_probability",
        ),
        default="prototype_similarity",
    )
    parser.add_argument(
        "--rank_score_statistic", choices=SCORE_STATISTICS, default="mean"
    )
    parser.add_argument(
        "--audit_score_statistic", choices=SCORE_STATISTICS, default="mean"
    )
    parser.add_argument(
        "--crossfit_rank_shortlist_size",
        type=int,
        default=0,
        help=(
            "When positive, fold 0 shortlists hypotheses, fold 1 selects one, "
            "and fold 2 only audits it against the immutable source."
        ),
    )
    parser.add_argument("--refine_nearest_landmarks", type=int, default=8)
    parser.add_argument("--refine_radius_px", type=float, default=12.0)
    parser.add_argument("--refine_spatial_sigma_px", type=float, default=4.0)
    parser.add_argument("--refine_minimum_match_evidence", type=float, default=0.05)
    parser.add_argument("--refine_minimum_correspondences", type=int, default=8)
    parser.add_argument("--refine_iterations", type=int, default=2)
    parser.add_argument("--refine_max_translation_step_m", type=float, default=0.25)
    parser.add_argument("--refine_max_rotation_step_deg", type=float, default=3.0)
    parser.add_argument("--minimum_rank_refine_gain", type=float, default=0.0)
    parser.add_argument("--minimum_audit_gain", type=float, default=0.0)
    parser.add_argument("--minimum_rank_effective_points", type=int, default=8)
    parser.add_argument("--minimum_audit_effective_points", type=int, default=8)
    parser.add_argument("--minimum_information_matches", type=int, default=8)
    parser.add_argument("--minimum_bearing_span_deg", type=float, default=3.0)
    parser.add_argument("--minimum_depth_span_ratio", type=float, default=0.0)
    parser.add_argument("--minimum_xyz_second_ratio", type=float, default=1e-3)
    parser.add_argument("--minimum_translation_information_eigenvalue", type=float, default=0.0)
    parser.add_argument("--maximum_translation_information_condition", type=float, default=0.0)
    parser.add_argument("--maximum_joint_information_condition", type=float, default=0.0)
    parser.add_argument(
        "--hypothesis_scope",
        choices=("shortlisted", "preliminary_topk", "all"),
        default="shortlisted",
    )
    parser.add_argument("--hypothesis_limit", type=int, default=128)
    parser.add_argument(
        "--split_filter",
        choices=SPLIT_FILTERS,
        default="all",
        help="Filter query splits before sharding; use this to keep tuning off test.",
    )
    parser.add_argument("--query_shard_count", type=int, default=1)
    parser.add_argument("--query_shard_index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_immutable_source_pose_artifact(
    path: Path,
    *,
    expected_colmap_cameras_sha256: str,
    expected_colmap_images_sha256: str,
    evaluation_label: str,
) -> dict[str, object]:
    """Load a target-free pose artifact and reject stale or cross-scene input."""

    arrays, metadata = _load_npz(path)
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "success",
        "poses_w2c",
        "match_counts",
        "inlier_counts",
    }
    if set(arrays) != required:
        raise ValueError(
            "immutable source pose artifact fields differ: "
            f"missing={sorted(required - set(arrays))}, "
            f"extra={sorted(set(arrays) - required)}"
        )
    if metadata.get("format") != SELECTED_POSE_ARTIFACT_FORMAT:
        raise ValueError("unsupported immutable source pose artifact format")
    if metadata.get("contains_ground_truth") is not False or metadata.get(
        "contains_target_errors"
    ) is not False:
        raise ValueError("immutable source pose artifact is not target-free")
    source_manifest = metadata.get("source_manifest")
    if not isinstance(source_manifest, Mapping) or metadata.get(
        "source_manifest_sha256"
    ) != _canonical_hash(source_manifest):
        raise ValueError("immutable source pose manifest is stale")
    source_inputs = source_manifest.get("inputs")
    if not isinstance(source_inputs, Mapping):
        raise ValueError("immutable source pose manifest has no input section")
    scene_mismatches = {
        key: {"expected": expected, "actual": source_inputs.get(key)}
        for key, expected in (
            ("colmap_cameras_bin_sha256", expected_colmap_cameras_sha256),
            ("colmap_images_bin_sha256", expected_colmap_images_sha256),
        )
        if source_inputs.get(key) != expected
    }
    if scene_mismatches:
        raise ValueError(
            "immutable source pose belongs to another COLMAP scene: "
            f"{json.dumps(scene_mismatches, sort_keys=True)}"
        )

    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    labels = np.asarray(arrays["evaluation_labels"]).astype(str).reshape(-1)
    success = np.asarray(arrays["success"], dtype=bool).reshape(-1)
    poses = np.asarray(arrays["poses_w2c"], dtype=np.float64)
    match_counts = np.asarray(arrays["match_counts"], dtype=np.int64).reshape(-1)
    inlier_counts = np.asarray(arrays["inlier_counts"], dtype=np.int64).reshape(-1)
    row_count = len(query_ids)
    if (
        any(
            len(value) != row_count
            for value in (split_names, labels, success, match_counts, inlier_counts)
        )
        or poses.shape != (row_count, 4, 4)
        or int(metadata.get("row_count", -1)) != row_count
    ):
        raise ValueError("immutable source pose artifact dimensions differ")
    available_labels = tuple(sorted(set(labels.tolist())))
    selected_label = str(evaluation_label)
    if selected_label:
        if selected_label not in available_labels:
            raise ValueError(
                f"immutable source pose label is missing: {selected_label}"
            )
    elif len(available_labels) == 1:
        selected_label = available_labels[0]
    else:
        raise ValueError(
            "immutable source pose artifact has multiple policies; select a label"
        )

    records: dict[tuple[str, str], dict[str, object]] = {}
    for index in np.flatnonzero(labels == selected_label).tolist():
        identity = (str(split_names[index]), str(query_ids[index]))
        if identity in records:
            raise ValueError(f"duplicate immutable source pose: {identity}")
        match_count = int(match_counts[index])
        inlier_count = int(inlier_counts[index])
        if match_count < 0 or not 0 <= inlier_count <= match_count:
            raise ValueError("immutable source pose match/inlier counts are invalid")
        row_success = bool(success[index])
        pose = np.asarray(poses[index], dtype=np.float64).reshape(4, 4)
        if row_success and not np.all(np.isfinite(pose)):
            raise ValueError("successful immutable source pose is non-finite")
        if not row_success and np.any(np.isfinite(pose)):
            raise ValueError("failed immutable source pose carries finite values")
        records[identity] = {
            "success": row_success,
            "pose_w2c": pose.copy() if row_success else None,
            "match_count": match_count,
            "inlier_count": inlier_count,
        }
    if not records:
        raise ValueError("immutable source pose policy has no rows")
    return {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "evaluation_label": selected_label,
        "metadata": metadata,
        "records": records,
    }


def _resolve_immutable_source_pose_artifact(
    hypothesis_metadata: Mapping[str, object],
    *,
    requested_path: str,
    requested_evaluation_label: str,
    expected_colmap_cameras_sha256: str,
    expected_colmap_images_sha256: str,
    allow_override: bool = False,
) -> dict[str, object]:
    """Resolve the source from the hypothesis manifest and enforce its hash."""

    inputs = hypothesis_metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("grouped hypothesis metadata has no input manifest")
    declared_path = str(inputs.get("immutable_baseline_pose_artifact", ""))
    declared_sha256 = str(
        inputs.get("immutable_baseline_pose_artifact_sha256", "")
    )
    declared_label = str(
        inputs.get("immutable_baseline_pose_evaluation_label", "")
    )
    if not declared_path or not declared_sha256:
        raise ValueError(
            "grouped hypothesis artifact does not declare an immutable pose source"
        )
    requested = str(requested_path)
    if bool(allow_override) and not requested:
        raise ValueError(
            "immutable source override requires --immutable_source_pose_artifact"
        )
    path = Path(requested or declared_path)
    actual_sha256 = file_sha256_short(path)
    overridden = bool(allow_override) and path != Path(declared_path)
    if actual_sha256 != declared_sha256 and not overridden:
        raise ValueError(
            "immutable source pose hash differs from grouped hypothesis manifest: "
            f"expected={declared_sha256}, actual={actual_sha256}"
        )
    selected_label = str(requested_evaluation_label) or declared_label
    source = _load_immutable_source_pose_artifact(
        path,
        expected_colmap_cameras_sha256=expected_colmap_cameras_sha256,
        expected_colmap_images_sha256=expected_colmap_images_sha256,
        evaluation_label=selected_label,
    )
    if overridden:
        source_manifest = source["metadata"].get("source_manifest")
        source_inputs = (
            None
            if not isinstance(source_manifest, Mapping)
            else source_manifest.get("inputs")
        )
        hypothesis_inputs = inputs
        if not isinstance(source_inputs, Mapping):
            raise ValueError("immutable source override has no lineage inputs")
        lineage_keys = (
            "proposals_sha256",
            "candidate_artifact_sha256",
            "score_artifact_sha256",
            "candidate_evidence_sha256",
            "projected_landmark_bank_sha256",
            "colmap_cameras_bin_sha256",
            "colmap_images_bin_sha256",
        )
        mismatches = {
            key: {
                "hypothesis": hypothesis_inputs.get(key),
                "source": source_inputs.get(key),
            }
            for key in lineage_keys
            if hypothesis_inputs.get(key) is not None
            and source_inputs.get(key) != hypothesis_inputs.get(key)
        }
        if mismatches:
            raise ValueError(
                "immutable source override lineage differs from grouped hypotheses: "
                + json.dumps(mismatches, sort_keys=True)
            )
    source["declared_path"] = declared_path
    source["declared_sha256"] = declared_sha256
    source["declared_evaluation_label"] = declared_label
    source["override"] = overridden
    return source


def _maplet_partition_identities(
    bank_track_ids: np.ndarray,
    maplet_track_ids: np.ndarray,
    maplet_cluster_ids: np.ndarray,
) -> np.ndarray:
    bank_tracks = np.asarray(bank_track_ids, dtype=np.int64).reshape(-1)
    maplet_tracks = np.asarray(maplet_track_ids, dtype=np.int64).reshape(-1)
    clusters = np.asarray(maplet_cluster_ids, dtype=np.int64).reshape(-1)
    if maplet_tracks.shape != clusters.shape:
        raise ValueError("maplet track IDs and clusters are not aligned")
    if len(np.unique(maplet_tracks)) != len(maplet_tracks):
        raise ValueError("maplet support index repeats anchor track IDs")
    order = np.argsort(maplet_tracks, kind="mergesort")
    sorted_tracks = maplet_tracks[order]
    positions = np.searchsorted(sorted_tracks, bank_tracks)
    clipped = np.minimum(positions, max(len(sorted_tracks) - 1, 0))
    found = (
        (positions < len(sorted_tracks))
        & (sorted_tracks[clipped] == bank_tracks)
    ) if len(sorted_tracks) else np.zeros(bank_tracks.shape, dtype=bool)
    # Negative IDs namespace unclustered physical tracks; positive IDs namespace
    # connected repeated-structure maplets. Duplicate descriptor prototypes remain
    # tied because they share the same physical track ID.
    identities = -(bank_tracks + 1)
    cluster_offset = max(int(np.max(bank_tracks, initial=0)) + 2, 2)
    identities[found] = clusters[order[positions[found]]] + cluster_offset
    return identities


def _correspondence_information(
    pose_w2c: np.ndarray,
    correspondences: IndependentPoseCorrespondences,
    camera,
) -> dict[str, float | int | None]:
    matches = [
        QueryTo3DMatch(
            token_index=int(point_index),
            xy=np.asarray(xy, dtype=np.float64),
            track_id=int(track_id),
            xyz=np.asarray(xyz, dtype=np.float64),
            similarity=float(similarity),
            ratio=1.0,
            landmark_variance=0.0,
            source="independent_rank_crossfit",
        )
        for point_index, xy, track_id, xyz, similarity in zip(
            correspondences.point_indices,
            correspondences.xy,
            correspondences.track_ids,
            correspondences.xyz,
            correspondences.descriptor_similarities,
        )
    ]
    return pose_information_diagnostics(pose_w2c, matches, camera)


def _observability_failures(
    diagnostics: Mapping[str, object], args: argparse.Namespace
) -> tuple[str, ...]:
    minimums = {
        "information_match_count": float(args.minimum_information_matches),
        "bearing_max_angle_deg": float(args.minimum_bearing_span_deg),
        "camera_depth_span_ratio": float(args.minimum_depth_span_ratio),
        "xyz_second_singular_ratio": float(args.minimum_xyz_second_ratio),
        "translation_information_min_eigenvalue": float(
            args.minimum_translation_information_eigenvalue
        ),
    }
    maximums = {
        "translation_information_condition": float(
            args.maximum_translation_information_condition
        ),
        "joint_information_condition": float(args.maximum_joint_information_condition),
    }
    failures: list[str] = []
    for key, threshold in minimums.items():
        if threshold <= 0.0:
            continue
        value = diagnostics.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) < threshold:
            failures.append(key)
    for key, threshold in maximums.items():
        if threshold <= 0.0:
            continue
        value = diagnostics.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) > threshold:
            failures.append(key)
    return tuple(failures)


def _validate_local_descriptor_verification_bank(
    source_bank_path: Path,
    verification_bank,
    verification_metadata: Mapping[str, object],
    detector_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Prove ALIKE descriptor and physical-map alignment without loading RADIO."""

    manifest = verification_metadata.get("descriptor_space_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("local verification bank has no descriptor-space manifest")
    expected_checkpoint = str(detector_metadata.get("alike_checkpoint_sha256", ""))
    expected_dimension = int(detector_metadata.get("local_descriptor_dimension", -1))
    if not expected_checkpoint or expected_dimension <= 0:
        raise ValueError("detector cache does not declare its ALIKE descriptor space")
    contract = {
        "alike_checkpoint_sha256": expected_checkpoint,
        "descriptor_dimension": expected_dimension,
        "projection_source": "real_image_alike_observation_full_map",
        "coordinate_source": "sfm_observation_xy",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in contract.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "local verification descriptor contract differs: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if int(verification_bank.features.shape[1]) != expected_dimension:
        raise ValueError("local verification bank feature dimension differs")
    with np.load(source_bank_path, allow_pickle=False) as source:
        required = {"track_ids", "xyz"}
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"source landmark bank is missing geometry: {missing}")
        source_tracks = np.asarray(source["track_ids"], dtype=np.int64)
        source_xyz = np.asarray(source["xyz"], dtype=np.float64)
    if not np.array_equal(source_tracks, verification_bank.track_ids):
        raise ValueError("local verification bank physical track rows differ")
    if not np.allclose(
        source_xyz,
        np.asarray(verification_bank.xyz, dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("local verification bank XYZ differs from production map")
    return {
        "mode": "alike_local_descriptor_independent_evidence",
        "alike_checkpoint_sha256": expected_checkpoint,
        "descriptor_dimension": expected_dimension,
        "physical_track_geometry_exact": True,
    }


def _validate_support_geometry_index(
    metadata: Mapping[str, object],
    detector_metadata: Mapping[str, object],
    *,
    local_descriptor_mode: bool,
) -> None:
    if str(metadata.get("format", "")) != "support_observation_geometry_index_npz":
        raise ValueError("unsupported observation-view geometry index")
    if local_descriptor_mode and str(metadata.get("alike_checkpoint_sha256", "")) != str(
        detector_metadata.get("alike_checkpoint_sha256", "")
    ):
        raise ValueError("support geometry ALIKE checkpoint differs from query cache")


def _hypothesis_group(
    all_group: np.ndarray,
    *,
    shortlisted: np.ndarray,
    artifact_chosen: np.ndarray,
    preliminary_scores: np.ndarray,
    scope: str,
    limit: int,
) -> np.ndarray:
    if scope == "shortlisted":
        group = all_group[shortlisted[all_group]]
    elif scope == "all":
        group = all_group
    else:
        finite = all_group[np.isfinite(preliminary_scores[all_group])]
        order = np.argsort(-preliminary_scores[finite], kind="mergesort")
        group = finite[order if int(limit) == 0 else order[: int(limit)]]
    artifact_source = all_group[artifact_chosen[all_group]]
    if len(artifact_source) != 1:
        raise ValueError("each query requires exactly one artifact-chosen pose")
    return np.unique(np.concatenate([group, artifact_source]))


def _filtered_sharded_group_keys(
    group_keys: Sequence[tuple[str, str, str]],
    *,
    split_filter: str,
    shard_count: int,
    shard_index: int,
) -> list[tuple[str, str, str]]:
    """Filter before sharding so validation and test runs have stable coverage."""

    if split_filter not in SPLIT_FILTERS:
        raise ValueError(f"unsupported split filter: {split_filter}")
    filtered = [
        key
        for key in group_keys
        if split_filter == "all" or str(key[0]) == split_filter
    ]
    return [
        key
        for index, key in enumerate(filtered)
        if index % int(shard_count) == int(shard_index)
    ]


def _missing_immutable_source_is_allowed(
    split_name: str,
    *,
    train_calibration_escape_hatch: bool,
) -> bool:
    """Keep the artifact-chosen fallback strictly out of held-out inference."""

    return bool(train_calibration_escape_hatch) and str(split_name) == "train"


def _exactly_one_artifact_chosen_entry(
    entries: Sequence[Mapping[str, object]], *, query_id: str
) -> int:
    """Return the frozen grouped source entry or reject an ambiguous baseline."""

    selected = [
        index
        for index, entry in enumerate(entries)
        if bool(entry["artifact_chosen"])
    ]
    if len(selected) != 1:
        raise RuntimeError(f"{query_id}: artifact-chosen source pose is not unique")
    return int(selected[0])


def _crossfit_rank_shortlist(
    scores: np.ndarray,
    effective_points: np.ndarray,
    hypothesis_indices: np.ndarray,
    *,
    size: int,
    forced_indices: Sequence[int],
) -> np.ndarray:
    """Select a deterministic top-K while retaining required fallback poses."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    effective = np.asarray(effective_points, dtype=np.int64).reshape(-1)
    hypotheses = np.asarray(hypothesis_indices, dtype=np.int64).reshape(-1)
    if values.shape != effective.shape or values.shape != hypotheses.shape:
        raise ValueError("cross-fit shortlist arrays must be aligned")
    if int(size) <= 0:
        raise ValueError("cross-fit shortlist size must be positive")
    forced = np.asarray(tuple(forced_indices), dtype=np.int64).reshape(-1)
    if np.any((forced < 0) | (forced >= len(values))):
        raise ValueError("forced shortlist index is outside the candidate group")
    forced = np.unique(forced)
    if len(forced) > int(size):
        raise ValueError("cross-fit shortlist cannot contain all forced poses")
    if np.any(~np.isfinite(values)):
        raise ValueError("cross-fit exploration scores must be finite")
    order = sorted(
        range(len(values)),
        key=lambda index: (
            -float(values[index]),
            -int(effective[index]),
            int(hypotheses[index]),
        ),
    )
    selected = forced.tolist()
    selected_set = set(selected)
    target_size = min(int(size), len(values))
    if len(selected) >= target_size:
        return np.asarray(sorted(selected), dtype=np.int64)
    for index in order:
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
        if len(selected) >= target_size:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _crossfit_role_partitions(
    points,
    point_folds: np.ndarray,
    base_eligible: np.ndarray,
    landmark_folds: np.ndarray,
    *,
    role_count: int,
    rank_uses_complement_of_audit_fold: bool = False,
):
    """Return legacy three slots while keeping only active roles independent."""

    if int(role_count) == 2:
        rank_points = points.subset(point_folds == 0)
        audit_points = points.subset(point_folds == 1)
        rank_landmarks = base_eligible & (landmark_folds == 0)
        audit_landmarks = base_eligible & (landmark_folds == 1)
        return (
            [rank_points, rank_points, audit_points],
            [rank_landmarks, rank_landmarks, audit_landmarks],
            (1, 2),
        )
    if int(role_count) == 3:
        if bool(rank_uses_complement_of_audit_fold):
            # Refinement is disabled for this mode.  Reusing the rank partition
            # in the legacy refine slot keeps the surrounding artifact schema
            # stable while rank and audit remain strictly disjoint.
            rank_points = points.subset(point_folds != 2)
            audit_points = points.subset(point_folds == 2)
            rank_landmarks = base_eligible & (landmark_folds != 2)
            audit_landmarks = base_eligible & (landmark_folds == 2)
            return (
                [rank_points, rank_points, audit_points],
                [rank_landmarks, rank_landmarks, audit_landmarks],
                (1, 2),
            )
        return (
            [points.subset(point_folds == fold) for fold in range(3)],
            [base_eligible & (landmark_folds == fold) for fold in range(3)],
            (0, 1, 2),
        )
    raise ValueError("cross-fit role count must be two or three")


def _optional_pose_differs_from_source(
    optional_pose: np.ndarray,
    source_pose: np.ndarray,
) -> bool:
    """Prevent a no-op source selection from being counted as a promotion."""

    return not np.array_equal(
        np.asarray(optional_pose, dtype=np.float64),
        np.asarray(source_pose, dtype=np.float64),
    )


def _disabled_refinement_result(initial_pose: np.ndarray) -> IndependentPoseRefinementResult:
    """Avoid scoring a refinement fold when refinement is explicitly disabled."""

    pose = np.asarray(initial_pose, dtype=np.float64).reshape(4, 4).copy()
    return IndependentPoseRefinementResult(
        success=False,
        pose_w2c=pose,
        accepted_iterations=0,
        final_correspondence_count=0,
        fit_log_likelihood_before=float("nan"),
        fit_log_likelihood_after=float("nan"),
        translation_step_m=0.0,
        rotation_step_deg=0.0,
        used_track_ids=np.zeros((0,), dtype=np.int64),
        failure_reason="disabled_by_config",
    )


def _require_spatial_materialization_for_active_roles(
    role_points: Sequence[object],
    active_role_indices: Sequence[int],
    *,
    allow_unmaterialized: bool,
) -> tuple[dict[str, int], ...]:
    """Audit RGB-mode coverage separately for rank/audit point partitions.

    A full-query materialization check is insufficient here: a spatially
    balanced fold may otherwise contain no RGB mode at all, which turns its
    likelihood into a neutral unknown and lets it pass an audit by tie break.
    This check happens before any hypothesis score is evaluated.
    """

    audits = tuple(
        _spatial_materialization_audit(role_points[index])
        for index in active_role_indices
    )
    if not bool(allow_unmaterialized):
        missing = [
            str(index)
            for index, audit in zip(active_role_indices, audits)
            if int(audit["materialized_verification_point_count"]) <= 0
        ]
        if missing:
            raise ValueError(
                "candidate RGB spatial modes are unmaterialized for active "
                f"cross-fit roles: {', '.join(missing)}"
            )
    return audits


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.verification_point_count) < int(args.crossfit_role_count):
        raise ValueError("cross-fit requires at least one point per active role")
    if int(args.hypothesis_limit) < 0:
        raise ValueError("hypothesis_limit must be non-negative")
    if int(args.crossfit_rank_shortlist_size) < 0:
        raise ValueError("crossfit rank shortlist size must be non-negative")
    if int(args.crossfit_rank_shortlist_size) > 0 and int(args.refine_iterations) != 0:
        raise ValueError(
            "staged cross-fit ranking currently requires --refine_iterations 0"
        )
    if int(args.crossfit_role_count) == 2 and (
        int(args.refine_iterations) != 0
        or int(args.crossfit_rank_shortlist_size) != 0
    ):
        raise ValueError(
            "two-role cross-fit requires disabled refinement and staged shortlist"
        )
    if bool(args.rank_uses_complement_of_audit_fold) and (
        int(args.crossfit_role_count) != 3
        or int(args.refine_iterations) != 0
        or int(args.crossfit_rank_shortlist_size) != 0
    ):
        raise ValueError(
            "rank_uses_complement_of_audit_fold requires three folds with "
            "disabled refinement and staged shortlist"
        )
    if bool(args.allow_missing_immutable_source_for_train_calibration) and (
        str(args.split_filter) != "train"
    ):
        raise ValueError(
            "allow_missing_immutable_source_for_train_calibration requires "
            "--split_filter train"
        )
    if (
        str(args.source_pose_policy) != "immutable_artifact"
        and bool(args.allow_missing_immutable_source_for_train_calibration)
    ):
        raise ValueError(
            "allow_missing_immutable_source_for_train_calibration only applies "
            "to --source_pose_policy immutable_artifact"
        )
    if bool(args.allow_immutable_source_override) and (
        str(args.source_pose_policy) != "immutable_artifact"
    ):
        raise ValueError(
            "allow_immutable_source_override requires --source_pose_policy "
            "immutable_artifact"
        )
    if int(args.query_shard_count) <= 0 or not (
        0 <= int(args.query_shard_index) < int(args.query_shard_count)
    ):
        raise ValueError("invalid query shard")
    if bool(str(args.prototype_view_geometry)) == bool(
        str(args.support_geometry_index)
    ):
        raise ValueError(
            "provide exactly one of --prototype_view_geometry and "
            "--support_geometry_index"
        )
    if (
        str(args.verification_descriptor_source) == "local"
        and str(args.score_candidate_mode) != "fixed_global_topl"
    ):
        raise ValueError("local descriptor verification requires fixed top-L candidates")
    hypothesis_paths = tuple(
        Path(value.strip())
        for value in str(args.hypothesis_artifacts).split(",")
        if value.strip()
    )
    merged, hypothesis_metadata, compatibility_hash = _merge_hypothesis_artifacts(
        hypothesis_paths
    )
    output_dir = Path(args.output_dir)
    output_path = output_dir / "independent_crossfit_pose_alignment_v1.npz"
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_path}")

    proposal_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    source_bank_path = Path(args.projected_landmark_bank)
    verification_bank_path = Path(args.independent_verification_landmark_bank)
    for key, path in (
        ("proposals", proposal_path),
        ("candidate_artifact", candidate_path),
        ("projected_landmark_bank", source_bank_path),
    ):
        _validate_declared_input(hypothesis_metadata, key=key, path=path)
    detector, detector_metadata = _load_npz(Path(args.detector_query_cache))
    proposals, _proposal_metadata = _load_npz(proposal_path)
    candidate, candidate_metadata = _load_npz(candidate_path)
    prior_overlay_path = (
        None
        if not str(args.fixed_candidate_prior_overlay)
        else Path(args.fixed_candidate_prior_overlay)
    )
    learned_prior_sources = {
        "learned_probability",
        "prototype_similarity_with_learned_null",
    }
    if str(args.fixed_candidate_prior_source) in learned_prior_sources:
        if prior_overlay_path is None:
            raise ValueError(
                "learned candidate availability requires "
                "--fixed_candidate_prior_overlay"
            )
        prior_overlay, prior_overlay_metadata = _load_candidate_prior_overlay(
            prior_overlay_path,
            proposals_path=proposal_path,
            proposals=proposals,
        )
    else:
        if prior_overlay_path is not None:
            raise ValueError(
                "fixed candidate prior overlay requires a learned availability mode"
            )
        prior_overlay = None
        prior_overlay_metadata = {}
    spatial_mode_paths = tuple(
        Path(value.strip())
        for value in str(args.candidate_spatial_likelihood).split(",")
        if value.strip()
    )
    spatial_mode_index = None
    spatial_mode_offsets = None
    spatial_mode_log_probabilities = None
    spatial_mode_metadata: list[dict[str, object]] = []
    if spatial_mode_paths:
        if str(args.score_candidate_mode) != "fixed_global_topl":
            raise ValueError(
                "candidate RGB spatial modes require --score_candidate_mode "
                "fixed_global_topl"
            )
        if prior_overlay is None:
            raise ValueError(
                "candidate RGB spatial modes require a fixed learned candidate "
                "prior overlay"
            )
        (
            spatial_mode_index,
            spatial_mode_offsets,
            spatial_mode_log_probabilities,
            spatial_mode_metadata,
        ) = _load_candidate_spatial_mode_index(spatial_mode_paths)
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    if np.any((selected_rows < 0) | (selected_rows >= len(proposals["query_ids"]))):
        raise ValueError("candidate artifact contains invalid selected rows")

    source_bank_metadata = _landmark_bank_metadata(source_bank_path)
    landmark_index, bank_metadata = load_landmark_index_npz(verification_bank_path)
    bank_manifest = bank_metadata.get("descriptor_space_manifest")
    if not isinstance(bank_manifest, Mapping):
        raise ValueError("verification bank has no descriptor-space manifest")
    descriptor_source = str(args.verification_descriptor_source)
    bank_projection = str(bank_manifest.get("projection_space_id", ""))
    if descriptor_source == "global":
        _validate_alternate_verification_bank(source_bank_metadata, bank_metadata)
        detector_projection = str(detector_metadata.get("projection_space_id", ""))
        if detector_projection and detector_projection != bank_projection:
            raise ValueError(
                "detector and verification landmark projection spaces differ"
            )
        if not detector_projection:
            legacy_match = (
                str(detector_metadata.get("matcha_joint_checkpoint_sha256", ""))
                == str(bank_manifest.get("checkpoint_sha256", ""))
                and str(detector_metadata.get("feature_key", ""))
                == str(bank_manifest.get("feature_key", ""))
                and int(detector_metadata.get("global_descriptor_dimension", -1))
                == int(bank_manifest.get("descriptor_dimension", -2))
            )
            if not legacy_match:
                raise ValueError(
                    "detector cache is not projection-compatible with the bank"
                )
        descriptor_compatibility = {
            "mode": "joint_global_descriptor_projection_compatible"
        }
    else:
        descriptor_compatibility = _validate_local_descriptor_verification_bank(
            source_bank_path,
            landmark_index,
            bank_metadata,
            detector_metadata,
        )

    if str(args.prototype_view_geometry):
        view_index, view_metadata = load_landmark_prototype_view_index_npz(
            Path(args.prototype_view_geometry),
            landmark_index,
            expected_descriptor_space_id=str(
                bank_metadata.get("descriptor_space_id", "")
            ),
        )
        view_geometry_path = Path(args.prototype_view_geometry)
        view_geometry_mode = "descriptor_prototype_aligned_view_distribution"
    else:
        view_geometry_path = Path(args.support_geometry_index)
        support_geometry, view_metadata = _load_npz(view_geometry_path)
        _validate_support_geometry_index(
            view_metadata,
            detector_metadata,
            local_descriptor_mode=descriptor_source == "local",
        )
        view_index = LandmarkObservationViewIndex.from_track_observations(
            landmark_index.track_ids,
            support_geometry["track_ids"],
            support_geometry["viewing_rays"],
        )
        view_geometry_mode = "all_observation_rays_with_track_aggregated_descriptor"
    score_config = IndependentLandmarkPoseLikelihoodConfig(
        nearest_landmarks=int(args.nearest_landmarks),
        maximum_reprojection_distance_px=float(args.maximum_reprojection_distance_px),
        spatial_sigma_px=float(args.spatial_sigma_px),
        descriptor_temperature=float(args.descriptor_temperature),
        outlier_likelihood=float(args.outlier_likelihood),
        minimum_observation_count=int(args.minimum_observation_count),
        maximum_view_angle_deg=float(args.maximum_view_angle_deg),
        kdtree_workers=int(args.kdtree_workers),
        candidate_mode=str(args.score_candidate_mode),
        fixed_candidate_prior_source=str(args.fixed_candidate_prior_source),
    )
    refine_config = IndependentPoseRefinementConfig(
        nearest_landmarks=int(args.refine_nearest_landmarks),
        maximum_reprojection_distance_px=float(args.refine_radius_px),
        spatial_sigma_px=float(args.refine_spatial_sigma_px),
        descriptor_temperature=float(args.descriptor_temperature),
        minimum_match_evidence=float(args.refine_minimum_match_evidence),
        minimum_correspondences=int(args.refine_minimum_correspondences),
        iterations=int(args.refine_iterations),
        maximum_translation_step_m=float(args.refine_max_translation_step_m),
        maximum_rotation_step_deg=float(args.refine_max_rotation_step_deg),
    )
    verifier = IndependentLandmarkPoseVerifier(landmark_index, view_index, score_config)

    maplet_path = Path(args.maplet_support_index)
    maplet_index, maplet_metadata = load_local_maplet_support_index_npz(maplet_path)
    maplet_track_ids = np.asarray(maplet_index.anchor_track_ids, dtype=np.int64)
    maplet_cluster_ids = build_disjoint_maplet_cluster_ids(maplet_index)
    partition_identities = _maplet_partition_identities(
        landmark_index.track_ids, maplet_track_ids, maplet_cluster_ids
    )
    role_count = int(args.crossfit_role_count)
    landmark_folds = deterministic_identity_folds(
        partition_identities,
        fold_count=role_count,
        seed=int(args.landmark_fold_seed),
    )
    if any(
        np.count_nonzero(landmark_folds == fold) == 0
        for fold in range(role_count)
    ):
        raise RuntimeError("landmark cross-fit produced an empty role")

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    image_camera_ids = read_colmap_image_camera_ids_binary(model_dir / "images.bin")
    query_ids = merged["query_ids"].astype(str)
    split_names = merged["split_names"].astype(str)
    labels = merged["evaluation_labels"].astype(str)
    hypothesis_indices = merged["hypothesis_indices"].astype(np.int64)
    shortlisted = merged["shortlisted_for_verification"].astype(bool)
    artifact_chosen = merged["chosen_for_optional_pose"].astype(bool)
    preliminary = np.asarray(merged["preliminary_log_likelihood_means"], dtype=np.float64)
    poses = np.asarray(merged["poses_w2c"], dtype=np.float64)
    group_keys = _filtered_sharded_group_keys(
        sorted(set(zip(split_names.tolist(), labels.tolist(), query_ids.tolist()))),
        split_filter=str(args.split_filter),
        shard_count=int(args.query_shard_count),
        shard_index=int(args.query_shard_index),
    )

    immutable_source: Mapping[str, object] | None = None
    immutable_records: Mapping[tuple[str, str], Mapping[str, object]] = {}
    if str(args.source_pose_policy) == "immutable_artifact":
        immutable_source = _resolve_immutable_source_pose_artifact(
            hypothesis_metadata,
            requested_path=str(args.immutable_source_pose_artifact),
            requested_evaluation_label=str(
                args.immutable_source_pose_evaluation_label
            ),
            expected_colmap_cameras_sha256=file_sha256_short(
                model_dir / "cameras.bin"
            ),
            expected_colmap_images_sha256=file_sha256_short(
                model_dir / "images.bin"
            ),
            allow_override=bool(args.allow_immutable_source_override),
        )
        immutable_records = immutable_source["records"]
        missing_source_queries = [
            (split_name, query_id)
            for split_name, _label, query_id in group_keys
            if (split_name, query_id) not in immutable_records
        ]
        if missing_source_queries and not all(
            _missing_immutable_source_is_allowed(
                split_name,
                train_calibration_escape_hatch=bool(
                    args.allow_missing_immutable_source_for_train_calibration
                ),
            )
            for split_name, _query_id in missing_source_queries
        ):
            raise ValueError(
                "immutable source pose artifact does not cover this execution: "
                f"{missing_source_queries[:5]}"
            )

    rows: list[dict[str, object]] = []
    immutable_source_success_count = 0
    immutable_source_failure_fallback_count = 0
    missing_immutable_source_train_calibration_count = 0
    grouped_artifact_source_count = 0
    start = time.time()
    diagnostic_keys = (
        "information_match_count",
        "translation_information_min_eigenvalue",
        "translation_information_condition",
        "rotation_information_min_eigenvalue",
        "rotation_information_condition",
        "joint_information_min_eigenvalue",
        "joint_information_condition",
        "bearing_max_angle_deg",
        "camera_depth_span_m",
        "camera_depth_span_ratio",
        "xyz_second_singular_ratio",
        "xyz_third_singular_ratio",
    )
    for query_number, (split_name, label, query_id) in enumerate(group_keys):
        camera_id = image_camera_ids.get(query_id)
        if camera_id is None or int(camera_id) not in cameras:
            raise ValueError(f"query camera is unavailable: {query_id}")
        camera = cameras[int(camera_id)]
        all_group = np.flatnonzero(
            (query_ids == query_id) & (split_names == split_name) & (labels == label)
        )
        group = _hypothesis_group(
            all_group,
            shortlisted=shortlisted,
            artifact_chosen=artifact_chosen,
            preliminary_scores=preliminary,
            scope=str(args.hypothesis_scope),
            limit=int(args.hypothesis_limit),
        )
        points, excluded_tracks, point_audit = _verification_points_for_query(
            query_id,
            detector=detector,
            proposals=proposals,
            selected_rows=selected_rows,
            point_count=int(args.verification_point_count),
            detector_log_merit_weight=float(args.detector_log_merit_weight),
            descriptor_key=(
                "local_descriptors"
                if descriptor_source == "local"
                else "global_descriptors"
            ),
            candidate_prior_overlay=prior_overlay,
            candidate_spatial_mode_index=spatial_mode_index,
            candidate_spatial_offsets_xy=spatial_mode_offsets,
            candidate_spatial_log_probability_arrays=spatial_mode_log_probabilities,
        )
        excluded_tracks = _maplet_purged_tracks(
            excluded_tracks,
            maplet_track_ids=maplet_track_ids,
            maplet_cluster_ids=maplet_cluster_ids,
        )
        base_eligible = verifier.eligible_mask_excluding_tracks(excluded_tracks)
        point_folds = spatially_balanced_point_folds(
            points,
            image_width=int(camera.width),
            image_height=int(camera.height),
            fold_count=role_count,
            seed=int(args.point_fold_seed),
        )
        role_points, role_landmarks, active_role_indices = _crossfit_role_partitions(
            points,
            point_folds,
            base_eligible,
            landmark_folds,
            role_count=role_count,
            rank_uses_complement_of_audit_fold=bool(
                args.rank_uses_complement_of_audit_fold
            ),
        )
        role_spatial_audits = tuple(
            _spatial_materialization_audit(role_points[index])
            for index in range(len(role_points))
        )
        if spatial_mode_index is not None:
            _require_spatial_materialization_for_active_roles(
                role_points,
                active_role_indices,
                allow_unmaterialized=bool(args.allow_unmaterialized_spatial_roles),
            )
        if any(
            len(role_points[index]) < int(args.refine_minimum_correspondences)
            for index in active_role_indices
        ):
            raise ValueError(f"{query_id}: a query cross-fit role is too small")

        source_record = immutable_records.get((split_name, query_id))
        immutable_source_missing = source_record is None
        immutable_success = bool(source_record["success"]) if source_record else False
        candidate_entries = [
            {
                "source_row": int(source_row),
                "hypothesis_index": int(hypothesis_indices[source_row]),
                "initial_pose": np.asarray(poses[source_row], dtype=np.float64),
                "artifact_chosen": bool(artifact_chosen[source_row]),
                "source_chosen": False,
                "hypothesis_origin": "grouped_hypothesis",
            }
            for source_row in group.tolist()
        ]
        if str(args.source_pose_policy) == "grouped_artifact_chosen":
            source_entry = _exactly_one_artifact_chosen_entry(
                candidate_entries, query_id=query_id
            )
            candidate_entries[source_entry]["source_chosen"] = True
            candidate_entries[source_entry]["hypothesis_origin"] = (
                "grouped_artifact_chosen_source"
            )
            grouped_artifact_source_count += 1
            source_origin = "grouped_artifact_chosen"
        elif immutable_source_missing:
            if not _missing_immutable_source_is_allowed(
                split_name,
                train_calibration_escape_hatch=bool(
                    args.allow_missing_immutable_source_for_train_calibration
                ),
            ):
                raise RuntimeError(
                    f"{query_id}: missing immutable source escaped validation"
                )
            missing_immutable_source_train_calibration_count += 1
            source_entry = _exactly_one_artifact_chosen_entry(
                candidate_entries, query_id=query_id
            )
            candidate_entries[source_entry]["source_chosen"] = True
            source_origin = "artifact_chosen_missing_immutable_train_calibration"
        elif immutable_success:
            immutable_source_success_count += 1
            immutable_pose = np.asarray(
                source_record["pose_w2c"], dtype=np.float64
            ).reshape(4, 4)
            exact_matches = [
                index
                for index, entry in enumerate(candidate_entries)
                if np.array_equal(entry["initial_pose"], immutable_pose)
            ]
            if exact_matches:
                source_entry = exact_matches[0]
                candidate_entries[source_entry]["source_chosen"] = True
                candidate_entries[source_entry]["hypothesis_origin"] = (
                    "grouped_hypothesis_bit_exact_immutable"
                )
            else:
                synthetic_index = int(
                    np.min(hypothesis_indices[all_group], initial=0)
                ) - 1
                candidate_entries.append(
                    {
                        "source_row": -1,
                        "hypothesis_index": synthetic_index,
                        "initial_pose": immutable_pose,
                        "artifact_chosen": False,
                        "source_chosen": True,
                        "hypothesis_origin": "immutable_pose_artifact_synthetic",
                    }
                )
            source_origin = "immutable_pose_artifact"
        else:
            immutable_source_failure_fallback_count += 1
            source_entry = _exactly_one_artifact_chosen_entry(
                candidate_entries, query_id=query_id
            )
            candidate_entries[source_entry]["source_chosen"] = True
            source_origin = "artifact_chosen_after_immutable_failure"

        staged_rank_size = int(args.crossfit_rank_shortlist_size)
        group_rows: list[dict[str, object]] = []
        for entry in candidate_entries:
            initial_pose = np.asarray(entry["initial_pose"], dtype=np.float64)
            refinement = (
                _disabled_refinement_result(initial_pose)
                if int(args.refine_iterations) == 0
                else verifier.refine_pose(
                    initial_pose,
                    camera,
                    role_points[0],
                    refine_config,
                    eligible_landmark_mask=role_landmarks[0],
                )
            )
            if staged_rank_size > 0:
                exploration = verifier.score_pose(
                    initial_pose,
                    camera,
                    role_points[0],
                    eligible_landmark_mask=role_landmarks[0],
                )
                exploration_value = exploration.statistic(
                    str(args.rank_score_statistic)
                )
                use_refined = False
                candidate_pose = initial_pose
                initial_rank_value = None
                refined_rank_value = None
                candidate_rank_value = None
                initial_rank_mean = None
                refined_rank_mean = None
                candidate_rank_mean = None
                candidate_rank_effective_points = 0
                candidate_rank_coverage = 0.0
            else:
                exploration = None
                exploration_value = None
                initial_rank = verifier.score_pose(
                    initial_pose,
                    camera,
                    role_points[1],
                    eligible_landmark_mask=role_landmarks[1],
                )
                refined_rank = verifier.score_pose(
                    refinement.pose_w2c,
                    camera,
                    role_points[1],
                    eligible_landmark_mask=role_landmarks[1],
                )
                initial_rank_value = initial_rank.statistic(
                    str(args.rank_score_statistic)
                )
                refined_rank_value = refined_rank.statistic(
                    str(args.rank_score_statistic)
                )
                use_refined = bool(
                    refinement.success
                    and refined_rank_value
                    >= initial_rank_value + float(args.minimum_rank_refine_gain)
                )
                candidate_pose = refinement.pose_w2c if use_refined else initial_pose
                candidate_rank = refined_rank if use_refined else initial_rank
                candidate_rank_value = (
                    refined_rank_value if use_refined else initial_rank_value
                )
                initial_rank_mean = float(initial_rank.log_likelihood_mean)
                refined_rank_mean = float(refined_rank.log_likelihood_mean)
                candidate_rank_mean = float(candidate_rank.log_likelihood_mean)
                candidate_rank_effective_points = int(
                    candidate_rank.effective_point_count
                )
                candidate_rank_coverage = float(candidate_rank.evidence_coverage)
            group_rows.append(
                {
                    "source_row": int(entry["source_row"]),
                    "hypothesis_index": int(entry["hypothesis_index"]),
                    "artifact_chosen": bool(entry["artifact_chosen"]),
                    "source_chosen": bool(entry["source_chosen"]),
                    "hypothesis_origin": str(entry["hypothesis_origin"]),
                    "source_pose": initial_pose,
                    "refined_pose": refinement.pose_w2c,
                    "candidate_pose": candidate_pose,
                    "refinement_success": bool(refinement.success),
                    "refinement_used": use_refined,
                    "refinement_iterations": int(refinement.accepted_iterations),
                    "refinement_correspondence_count": int(
                        refinement.final_correspondence_count
                    ),
                    "refinement_translation_step_m": float(
                        refinement.translation_step_m
                    ),
                    "refinement_rotation_step_deg": float(
                        refinement.rotation_step_deg
                    ),
                    "refinement_failure_reason": refinement.failure_reason or "",
                    "exploration_score": (
                        None if exploration_value is None else float(exploration_value)
                    ),
                    "exploration_score_mean": (
                        None
                        if exploration is None
                        else float(exploration.log_likelihood_mean)
                    ),
                    "exploration_effective_points": (
                        0
                        if exploration is None
                        else int(exploration.effective_point_count)
                    ),
                    "exploration_coverage": (
                        0.0
                        if exploration is None
                        else float(exploration.evidence_coverage)
                    ),
                    "rank_shortlisted": bool(staged_rank_size == 0),
                    "initial_rank_score": initial_rank_value,
                    "refined_rank_score": refined_rank_value,
                    "candidate_rank_score": candidate_rank_value,
                    "initial_rank_score_mean": initial_rank_mean,
                    "refined_rank_score_mean": refined_rank_mean,
                    "candidate_rank_score_mean": candidate_rank_mean,
                    "candidate_rank_effective_points": candidate_rank_effective_points,
                    "candidate_rank_coverage": candidate_rank_coverage,
                }
            )
        source_locals = [
            index for index, value in enumerate(group_rows) if bool(value["source_chosen"])
        ]
        if len(source_locals) != 1:
            raise RuntimeError(f"{query_id}: immutable source pose is not unique")
        source_local = source_locals[0]
        if staged_rank_size > 0:
            shortlist = _crossfit_rank_shortlist(
                np.asarray(
                    [value["exploration_score"] for value in group_rows],
                    dtype=np.float64,
                ),
                np.asarray(
                    [value["exploration_effective_points"] for value in group_rows],
                    dtype=np.int64,
                ),
                np.asarray(
                    [value["hypothesis_index"] for value in group_rows],
                    dtype=np.int64,
                ),
                size=staged_rank_size,
                forced_indices=[source_local],
            )
            for local_index in shortlist.tolist():
                value = group_rows[local_index]
                rank_score = verifier.score_pose(
                    np.asarray(value["candidate_pose"], dtype=np.float64),
                    camera,
                    role_points[1],
                    eligible_landmark_mask=role_landmarks[1],
                )
                rank_value = rank_score.statistic(str(args.rank_score_statistic))
                value.update(
                    {
                        "rank_shortlisted": True,
                        "initial_rank_score": float(rank_value),
                        "refined_rank_score": float(rank_value),
                        "candidate_rank_score": float(rank_value),
                        "initial_rank_score_mean": float(
                            rank_score.log_likelihood_mean
                        ),
                        "refined_rank_score_mean": float(
                            rank_score.log_likelihood_mean
                        ),
                        "candidate_rank_score_mean": float(
                            rank_score.log_likelihood_mean
                        ),
                        "candidate_rank_effective_points": int(
                            rank_score.effective_point_count
                        ),
                        "candidate_rank_coverage": float(
                            rank_score.evidence_coverage
                        ),
                    }
                )
            rank_candidates = shortlist.tolist()
        else:
            rank_candidates = list(range(len(group_rows)))
        optional_local = max(
            rank_candidates,
            key=lambda index: (
                float(group_rows[index]["candidate_rank_score"]),
                int(group_rows[index]["candidate_rank_effective_points"]),
                -int(group_rows[index]["hypothesis_index"]),
            ),
        )
        source_pose = np.asarray(group_rows[source_local]["source_pose"], dtype=np.float64)
        optional_pose = np.asarray(
            group_rows[optional_local]["candidate_pose"], dtype=np.float64
        )
        source_audit = verifier.score_pose(
            source_pose,
            camera,
            role_points[2],
            eligible_landmark_mask=role_landmarks[2],
        )
        optional_audit = verifier.score_pose(
            optional_pose,
            camera,
            role_points[2],
            eligible_landmark_mask=role_landmarks[2],
        )
        optional_correspondences = verifier.pose_conditioned_correspondences(
            optional_pose,
            camera,
            role_points[1],
            refine_config,
            eligible_landmark_mask=role_landmarks[1],
        )
        information = _correspondence_information(
            optional_pose, optional_correspondences, camera
        )
        observability_failures = _observability_failures(information, args)
        source_audit_value = source_audit.statistic(
            str(args.audit_score_statistic)
        )
        optional_audit_value = optional_audit.statistic(
            str(args.audit_score_statistic)
        )
        audit_delta = float(optional_audit_value - source_audit_value)
        promotion_failures: list[str] = list(observability_failures)
        optional_differs_from_source = _optional_pose_differs_from_source(
            optional_pose, source_pose
        )
        if not optional_differs_from_source:
            promotion_failures.append("optional_matches_immutable_source")
        if int(group_rows[optional_local]["candidate_rank_effective_points"]) < int(
            args.minimum_rank_effective_points
        ):
            promotion_failures.append("rank_effective_points")
        if min(
            int(source_audit.effective_point_count),
            int(optional_audit.effective_point_count),
        ) < int(args.minimum_audit_effective_points):
            promotion_failures.append("audit_effective_points")
        if audit_delta < float(args.minimum_audit_gain):
            promotion_failures.append("audit_likelihood_gain")
        promoted = not promotion_failures
        returned_pose = optional_pose if promoted else source_pose
        for local_index, value in enumerate(group_rows):
            rows.append(
                {
                    "query_id": query_id,
                    "split_name": split_name,
                    "evaluation_label": label,
                    "immutable_source_origin": source_origin,
                    "immutable_source_artifact_success": immutable_success,
                    **value,
                    "optional_rank_top1": bool(local_index == optional_local),
                    "optional_differs_from_source": optional_differs_from_source,
                    "promoted": bool(promoted),
                    "promotion_failures": ",".join(promotion_failures),
                    "optional_pose": optional_pose,
                    "returned_pose": returned_pose,
                    "source_audit_score": float(source_audit_value),
                    "optional_audit_score": float(optional_audit_value),
                    "source_audit_score_mean": float(
                        source_audit.log_likelihood_mean
                    ),
                    "optional_audit_score_mean": float(
                        optional_audit.log_likelihood_mean
                    ),
                    "audit_score_delta": audit_delta,
                    "source_audit_effective_points": int(source_audit.effective_point_count),
                    "optional_audit_effective_points": int(
                        optional_audit.effective_point_count
                    ),
                    "refine_point_count": (
                        int(len(role_points[0]))
                        if role_count == 3
                        and not bool(args.rank_uses_complement_of_audit_fold)
                        else 0
                    ),
                    "rank_point_count": int(len(role_points[1])),
                    "audit_point_count": int(len(role_points[2])),
                    "refine_landmark_count": (
                        int(np.count_nonzero(role_landmarks[0]))
                        if role_count == 3
                        and not bool(args.rank_uses_complement_of_audit_fold)
                        else 0
                    ),
                    "rank_landmark_count": int(np.count_nonzero(role_landmarks[1])),
                    "audit_landmark_count": int(np.count_nonzero(role_landmarks[2])),
                    "excluded_track_count": int(len(excluded_tracks)),
                    "candidate_spatial_refine_materialized_point_count": int(
                        role_spatial_audits[0]["materialized_verification_point_count"]
                        if role_count == 3
                        and not bool(args.rank_uses_complement_of_audit_fold)
                        else 0
                    ),
                    "candidate_spatial_refine_materialized_view_count": int(
                        role_spatial_audits[0]["materialized_candidate_view_count"]
                        if role_count == 3
                        and not bool(args.rank_uses_complement_of_audit_fold)
                        else 0
                    ),
                    "candidate_spatial_rank_materialized_point_count": int(
                        role_spatial_audits[1]["materialized_verification_point_count"]
                    ),
                    "candidate_spatial_rank_materialized_view_count": int(
                        role_spatial_audits[1]["materialized_candidate_view_count"]
                    ),
                    "candidate_spatial_audit_materialized_point_count": int(
                        role_spatial_audits[2]["materialized_verification_point_count"]
                    ),
                    "candidate_spatial_audit_materialized_view_count": int(
                        role_spatial_audits[2]["materialized_candidate_view_count"]
                    ),
                    **{
                        key: information.get(key)
                        for key in diagnostic_keys
                    },
                    **point_audit,
                }
            )
        print(
            json.dumps(
                {
                    "stage": "independent_crossfit_pose_alignment",
                    "completed_queries": int(query_number + 1),
                    "total_queries": int(len(group_keys)),
                    "query_id": query_id,
                    "hypothesis_count": int(len(group_rows)),
                    "refined_count": int(
                        sum(bool(value["refinement_used"]) for value in group_rows)
                    ),
                    "promoted": bool(promoted),
                    "audit_delta": audit_delta,
                    "promotion_failures": promotion_failures,
                    "elapsed_seconds": float(time.time() - start),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    string_fields = {
        "query_ids": "query_id",
        "split_names": "split_name",
        "evaluation_labels": "evaluation_label",
        "hypothesis_origins": "hypothesis_origin",
        "immutable_source_origins": "immutable_source_origin",
        "refinement_failure_reasons": "refinement_failure_reason",
        "promotion_failures": "promotion_failures",
    }
    int_fields = {
        "hypothesis_indices": "hypothesis_index",
        "refinement_iterations": "refinement_iterations",
        "refinement_correspondence_counts": "refinement_correspondence_count",
        "candidate_rank_effective_points": "candidate_rank_effective_points",
        "exploration_effective_points": "exploration_effective_points",
        "source_audit_effective_points": "source_audit_effective_points",
        "optional_audit_effective_points": "optional_audit_effective_points",
        "refine_point_counts": "refine_point_count",
        "rank_point_counts": "rank_point_count",
        "audit_point_counts": "audit_point_count",
        "refine_landmark_counts": "refine_landmark_count",
        "rank_landmark_counts": "rank_landmark_count",
        "audit_landmark_counts": "audit_landmark_count",
        "excluded_track_counts": "excluded_track_count",
        "information_match_counts": "information_match_count",
        "fit_query_point_counts": "fit_query_point_count",
        "available_unused_query_point_counts": "available_unused_query_point_count",
        "selected_verification_point_counts": "selected_verification_point_count",
        "candidate_spatial_refine_materialized_point_counts": (
            "candidate_spatial_refine_materialized_point_count"
        ),
        "candidate_spatial_refine_materialized_view_counts": (
            "candidate_spatial_refine_materialized_view_count"
        ),
        "candidate_spatial_rank_materialized_point_counts": (
            "candidate_spatial_rank_materialized_point_count"
        ),
        "candidate_spatial_rank_materialized_view_counts": (
            "candidate_spatial_rank_materialized_view_count"
        ),
        "candidate_spatial_audit_materialized_point_counts": (
            "candidate_spatial_audit_materialized_point_count"
        ),
        "candidate_spatial_audit_materialized_view_counts": (
            "candidate_spatial_audit_materialized_view_count"
        ),
    }
    bool_fields = {
        "artifact_chosen_for_optional_pose": "artifact_chosen",
        "source_chosen": "source_chosen",
        "immutable_source_artifact_success": "immutable_source_artifact_success",
        "refinement_success": "refinement_success",
        "refinement_used": "refinement_used",
        "rank_shortlisted": "rank_shortlisted",
        "optional_rank_top1": "optional_rank_top1",
        "optional_differs_from_source": "optional_differs_from_source",
        "promoted": "promoted",
    }
    float_fields = {
        "refinement_translation_steps_m": "refinement_translation_step_m",
        "refinement_rotation_steps_deg": "refinement_rotation_step_deg",
        "exploration_scores": "exploration_score",
        "exploration_score_means": "exploration_score_mean",
        "exploration_coverages": "exploration_coverage",
        "initial_rank_scores": "initial_rank_score",
        "refined_rank_scores": "refined_rank_score",
        "candidate_rank_scores": "candidate_rank_score",
        "initial_rank_score_means": "initial_rank_score_mean",
        "refined_rank_score_means": "refined_rank_score_mean",
        "candidate_rank_score_means": "candidate_rank_score_mean",
        "candidate_rank_coverages": "candidate_rank_coverage",
        "source_audit_scores": "source_audit_score",
        "optional_audit_scores": "optional_audit_score",
        "source_audit_score_means": "source_audit_score_mean",
        "optional_audit_score_means": "optional_audit_score_mean",
        "audit_score_deltas": "audit_score_delta",
        **{f"observability_{key}": key for key in diagnostic_keys if key != "information_match_count"},
    }
    arrays: dict[str, np.ndarray] = {
        name: np.asarray([row[field] for row in rows], dtype=str)
        for name, field in string_fields.items()
    }
    arrays.update(
        {
            name: np.asarray([row[field] for row in rows], dtype=np.int64)
            for name, field in int_fields.items()
        }
    )
    arrays.update(
        {
            name: np.asarray([row[field] for row in rows], dtype=bool)
            for name, field in bool_fields.items()
        }
    )
    arrays.update(
        {
            name: np.asarray(
                [np.nan if row[field] is None else row[field] for row in rows],
                dtype=np.float64,
            )
            for name, field in float_fields.items()
        }
    )
    for output_name, field in (
        ("source_poses_w2c", "source_pose"),
        ("refined_poses_w2c", "refined_pose"),
        ("candidate_poses_w2c", "candidate_pose"),
        ("optional_poses_w2c", "optional_pose"),
        ("returned_poses_w2c", "returned_pose"),
    ):
        arrays[output_name] = np.stack(
            [np.asarray(row[field], dtype=np.float64) for row in rows], axis=0
        )
    score_thresholds = {
        "rank_score_statistic": str(args.rank_score_statistic),
        "audit_score_statistic": str(args.audit_score_statistic),
        "minimum_rank_refine_gain": float(args.minimum_rank_refine_gain),
        "minimum_audit_gain": float(args.minimum_audit_gain),
        "minimum_rank_effective_points": int(args.minimum_rank_effective_points),
        "minimum_audit_effective_points": int(args.minimum_audit_effective_points),
        "crossfit_rank_shortlist_size": int(args.crossfit_rank_shortlist_size),
    }
    observability_thresholds = {
        "minimum_information_matches": int(args.minimum_information_matches),
        "minimum_bearing_span_deg": float(args.minimum_bearing_span_deg),
        "minimum_depth_span_ratio": float(args.minimum_depth_span_ratio),
        "minimum_xyz_second_ratio": float(args.minimum_xyz_second_ratio),
        "minimum_translation_information_eigenvalue": float(
            args.minimum_translation_information_eigenvalue
        ),
        "maximum_translation_information_condition": float(
            args.maximum_translation_information_condition
        ),
        "maximum_joint_information_condition": float(
            args.maximum_joint_information_condition
        ),
    }
    metadata = {
        "format": ARTIFACT_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "camera_ownership_parser": "image_name_to_camera_id_pose_discarded_v1",
        "version": INDEPENDENT_LANDMARK_POSE_LIKELIHOOD_VERSION,
        "row_count": int(len(rows)),
        "query_count": int(len(group_keys)),
        "split_filter": str(args.split_filter),
        "query_shard_count": int(args.query_shard_count),
        "query_shard_index": int(args.query_shard_index),
        "hypothesis_compatibility_sha256": compatibility_hash,
        "score_config": score_config.to_dict(),
        "descriptor_evidence": {
            "source": descriptor_source,
            "compatibility": descriptor_compatibility,
            "view_geometry_mode": view_geometry_mode,
            "candidate_prior_source": str(args.fixed_candidate_prior_source),
            "explicit_candidate_null_probability": bool(
                prior_overlay is not None
            ),
            "candidate_specific_rgb_spatial_modes": bool(spatial_mode_paths),
            "candidate_spatial_semantics": (
                "normalized_gaussian_mixture_relative_to_uniform_grid_null_v1"
                if spatial_mode_paths
                else None
            ),
        },
        "refinement_config": refine_config.to_dict(),
        "score_thresholds": score_thresholds,
        "observability_thresholds": observability_thresholds,
        "hypothesis_scope": {
            "mode": str(args.hypothesis_scope),
            "limit": int(args.hypothesis_limit),
            "immutable_source_forced_into_scope": bool(
                str(args.source_pose_policy) == "immutable_artifact"
            ),
            "artifact_chosen_pose_forced_into_scope": True,
        },
        "crossfit": {
            "roles": (
                ROLE_NAMES
                if role_count == 3
                and not bool(args.rank_uses_complement_of_audit_fold)
                else ("rank", "audit")
            ),
            "active_role_count": (
                2 if role_count == 2 or bool(args.rank_uses_complement_of_audit_fold)
                else 3
            ),
            "refinement_role_inactive": bool(
                role_count == 2 or bool(args.rank_uses_complement_of_audit_fold)
            ),
            "query_token_disjoint": True,
            "physical_track_disjoint": True,
            "maplet_cluster_disjoint": True,
            "point_fold_seed": int(args.point_fold_seed),
            "landmark_fold_seed": int(args.landmark_fold_seed),
            "rank_selects_hypothesis_only": True,
            "explore_shortlists_hypotheses_only": bool(
                int(args.crossfit_rank_shortlist_size) > 0
            ),
            "rank_selects_from_explore_shortlist_only": bool(
                int(args.crossfit_rank_shortlist_size) > 0
            ),
            "rank_uses_complement_of_audit_fold": bool(
                args.rank_uses_complement_of_audit_fold
            ),
            "role_semantics": (
                ["explore", "rank", "audit"]
                if int(args.crossfit_rank_shortlist_size) > 0
                else (
                    ["rank_union_two_spatial_track_folds", "audit_remaining_fold"]
                    if bool(args.rank_uses_complement_of_audit_fold)
                    else (
                        ["refine", "rank", "audit"]
                        if role_count == 3
                        else ["rank", "audit"]
                    )
                )
            ),
            "audit_compares_frozen_optional_to_immutable_source_only": True,
            "denominator_fixed_across_hypotheses_within_each_role": True,
            "candidate_spatial_role_materialization_required": bool(
                spatial_mode_paths
                and not bool(args.allow_unmaterialized_spatial_roles)
            ),
            "candidate_spatial_missing_is_neutral_unknown": bool(
                spatial_mode_paths
            ),
            "fallback_pose_bit_exact": True,
            "promotion_requires_distinct_optional_pose": True,
            "source_pose_policy": str(args.source_pose_policy),
            "immutable_source_override": bool(
                immutable_source is not None and immutable_source.get("override")
            ),
            "audit_compares_frozen_optional_to_fixed_source_only": True,
            "audit_compares_frozen_optional_to_immutable_source_only": bool(
                str(args.source_pose_policy) == "immutable_artifact"
            ),
            "immutable_source_failure_policy": (
                "artifact_chosen_pose_only_when_immutable_source_failed"
                if str(args.source_pose_policy) == "immutable_artifact"
                else "not_applicable_grouped_artifact_chosen_source"
            ),
            "missing_immutable_source_train_calibration_only": bool(
                args.allow_missing_immutable_source_for_train_calibration
            ),
        },
        "immutable_source_coverage": {
            "success_query_count": int(immutable_source_success_count),
            "failure_fallback_query_count": int(
                immutable_source_failure_fallback_count
            ),
            "missing_train_calibration_fallback_query_count": int(
                missing_immutable_source_train_calibration_count
            ),
            "grouped_artifact_source_query_count": int(
                grouped_artifact_source_count
            ),
        },
        "inputs": {
            "hypothesis_artifacts": [str(path) for path in hypothesis_paths],
            "hypothesis_artifact_sha256": [
                file_sha256_short(path) for path in hypothesis_paths
            ],
            "source_pose_policy": str(args.source_pose_policy),
            "immutable_source_pose_artifact": (
                None if immutable_source is None else str(immutable_source["path"])
            ),
            "immutable_source_pose_artifact_sha256": (
                None if immutable_source is None else str(immutable_source["sha256"])
            ),
            "immutable_source_pose_evaluation_label": (
                None
                if immutable_source is None
                else str(immutable_source["evaluation_label"])
            ),
            "immutable_source_declared_path": (
                None
                if immutable_source is None
                else str(immutable_source["declared_path"])
            ),
            "immutable_source_declared_sha256": (
                None
                if immutable_source is None
                else str(immutable_source["declared_sha256"])
            ),
            "immutable_source_override": (
                None
                if immutable_source is None
                else bool(immutable_source.get("override"))
            ),
            "detector_query_cache": str(args.detector_query_cache),
            "detector_query_cache_sha256": file_sha256_short(
                Path(args.detector_query_cache)
            ),
            "proposals": str(proposal_path),
            "proposals_sha256": file_sha256_short(proposal_path),
            "candidate_artifact": str(candidate_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "candidate_metadata_sha256": _canonical_hash(candidate_metadata),
            "fixed_candidate_prior_overlay": (
                None if prior_overlay_path is None else str(prior_overlay_path)
            ),
            "fixed_candidate_prior_overlay_sha256": (
                None
                if prior_overlay_path is None
                else file_sha256_short(prior_overlay_path)
            ),
            "fixed_candidate_prior_overlay_metadata_sha256": (
                None
                if prior_overlay_path is None
                else _canonical_hash(prior_overlay_metadata)
            ),
            "candidate_spatial_likelihood": [
                str(path) for path in spatial_mode_paths
            ],
            "candidate_spatial_likelihood_sha256": [
                file_sha256_short(path) for path in spatial_mode_paths
            ],
            "candidate_spatial_likelihood_metadata_sha256": [
                _canonical_hash(value) for value in spatial_mode_metadata
            ],
            "projected_landmark_bank": str(source_bank_path),
            "projected_landmark_bank_sha256": file_sha256_short(source_bank_path),
            "independent_verification_landmark_bank": str(verification_bank_path),
            "independent_verification_landmark_bank_sha256": file_sha256_short(
                verification_bank_path
            ),
            "prototype_view_geometry": str(args.prototype_view_geometry),
            "prototype_view_geometry_sha256": (
                None
                if not str(args.prototype_view_geometry)
                else file_sha256_short(Path(args.prototype_view_geometry))
            ),
            "support_geometry_index": str(args.support_geometry_index),
            "support_geometry_index_sha256": (
                None
                if not str(args.support_geometry_index)
                else file_sha256_short(Path(args.support_geometry_index))
            ),
            "view_geometry": str(view_geometry_path),
            "view_geometry_sha256": file_sha256_short(view_geometry_path),
            "view_geometry_metadata_sha256": _canonical_hash(view_metadata),
            "maplet_support_index": str(maplet_path),
            "maplet_support_index_sha256": file_sha256_short(maplet_path),
            "maplet_metadata_sha256": _canonical_hash(maplet_metadata),
            "colmap_cameras_bin": str(model_dir / "cameras.bin"),
            "colmap_cameras_bin_sha256": file_sha256_short(
                model_dir / "cameras.bin"
            ),
            "colmap_images_bin_camera_ownership_only": str(model_dir / "images.bin"),
            "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
            "projection_space_id": bank_projection,
            "verification_descriptor_space_id": str(
                bank_metadata.get("descriptor_space_id", "")
            ),
            "verification_descriptor_source": descriptor_source,
        },
        "elapsed_seconds": float(time.time() - start),
    }
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "independent_crossfit_pose_alignment",
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
