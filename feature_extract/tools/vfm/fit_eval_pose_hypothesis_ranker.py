"""Fit a train-only PnP hypothesis ranker and evaluate a frozen pose policy.

The ranker sees only inference-time fit, held-out verification, and
cross-hypothesis consensus features. Ground-truth pose is used only to create
pairwise labels on the train query split and to report held-out metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)
from feature_extract.vfm.localization.measurement_pose_evidence import (
    FrozenMeasurementPoseEvidence,
    FrozenMeasurementUpdateEvidence,
    load_frozen_measurement_pose_evidence,
    load_frozen_measurement_update_evidence,
)
from feature_extract.vfm.localization.pose_hypothesis_ranking import (
    HypothesisRankingGroup,
    POSE_HYPOTHESIS_FEATURE_NAMES,
    PoseHypothesisRanker,
    eligible_hypothesis_indices,
    evaluate_legacy_hypothesis_ranking,
    evaluate_pose_hypothesis_ranker,
    fit_pose_hypothesis_ranker,
    pose_hypothesis_features,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    PoseVerificationCandidatePool,
    VerifiedPnPConfig,
    VerifiedPnPResult,
    estimate_pose_with_heldout_verification,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch, pnp_pose_error


def _positive_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or min(output) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return output


def _positive_float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or min(output) <= 0.0 or not np.all(np.isfinite(output)):
        raise argparse.ArgumentTypeError("expected positive comma-separated floats")
    return output


def _integer_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected_policy_artifact", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--proposal_score_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--measurement_fit_summary", default="")
    parser.add_argument("--measurement_late_apply_summary", default="")
    parser.add_argument(
        "--measurement_feature_set", default="measurement_plus_support"
    )
    parser.add_argument("--measurement_update_fit_summary", default="")
    parser.add_argument("--measurement_update_late_apply_summary", default="")
    parser.add_argument(
        "--coordinate_geometry_probability_threshold", type=float, default=0.85
    )
    parser.add_argument("--score_key", default=None)
    parser.add_argument(
        "--fit_match_counts", type=_positive_int_list, default=(32, 64)
    )
    parser.add_argument(
        "--hypothesis_selection_modes",
        default="score_topk,spatial_round_robin,geometry_diverse",
    )
    parser.add_argument(
        "--ransac_thresholds_px", type=_positive_float_list, default=(2.0, 4.0, 8.0)
    )
    parser.add_argument("--rng_seed_offsets", type=_integer_list, default=(0,))
    parser.add_argument("--ransac_iterations", type=int, default=2000)
    parser.add_argument("--holdout_folds", type=int, default=4)
    parser.add_argument("--holdout_fold", type=int, default=0)
    parser.add_argument("--final_audit_fold", type=int, default=1)
    parser.add_argument(
        "--disable_final_audit",
        action="store_true",
        help="use a two-way fit/rank split; valid only when final refinement is disabled",
    )
    parser.add_argument("--verification_strict_px", type=float, default=2.0)
    parser.add_argument("--verification_loose_px", type=float, default=5.0)
    parser.add_argument("--final_consensus_px", type=float, default=4.0)
    parser.add_argument("--final_refine_f_scale_px", type=float, default=2.0)
    parser.add_argument("--min_final_inliers", type=int, default=6)
    parser.add_argument("--enable_final_refine", action="store_true")
    parser.add_argument("--candidate_pool_residual_sigma_px", type=float, default=2.0)
    parser.add_argument("--candidate_pool_hard_threshold_px", type=float, default=8.0)
    parser.add_argument(
        "--candidate_pool_descriptor_rank_weight", type=float, default=0.02
    )
    parser.add_argument("--candidate_pool_refine_iterations", type=int, default=2)
    parser.add_argument("--measurement_verified_min_matches", type=int, default=8)
    parser.add_argument("--measurement_verified_min_grid_cells", type=int, default=6)
    parser.add_argument(
        "--ranker_c_values", type=_positive_float_list, default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    )
    parser.add_argument("--ranker_cross_validation_folds", type=int, default=5)
    parser.add_argument("--rotation_equivalent_m_per_deg", type=float, default=0.02)
    parser.add_argument("--minimum_pair_gap_m", type=float, default=0.005)
    parser.add_argument(
        "--skip_late_on_validation_pass",
        action="store_true",
        help="do not replay the frozen ranker on the reused late-development block",
    )
    parser.add_argument(
        "--force_late_development_replay",
        action="store_true",
        help="diagnostic only: replay late even when validation promotion fails",
    )
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _query_seed(query_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(str(query_id).encode("utf8")).digest()[:4], "little"
    )


@dataclass(frozen=True)
class FrozenPoseInputs:
    query_ids: np.ndarray
    token_indices: np.ndarray
    query_xy: np.ndarray
    selected_track_ids: np.ndarray
    selected_prototype_ids: np.ndarray
    selected_canonical_rows: np.ndarray
    selected_scores: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_prototype_ids: np.ndarray
    candidate_canonical_rows: np.ndarray
    candidate_scores: np.ndarray
    selected_policy_metadata: dict[str, object]
    selected_policy_summary: dict[str, object]
    landmark_index: object
    landmark_metadata: dict[str, object]
    measurement_evidence: FrozenMeasurementPoseEvidence | None
    measurement_update_evidence: FrozenMeasurementUpdateEvidence | None
    coordinate_geometry_probability_threshold: float


def _require_hash(metadata: dict[str, object], key: str, path: Path) -> None:
    actual = file_sha256_short(path)
    expected = str(metadata.get(key, ""))
    if not expected or actual != expected:
        raise ValueError(
            f"frozen artifact mismatch for {key}: expected={expected!r}, actual={actual!r}"
        )


def _load_frozen_inputs(args: argparse.Namespace) -> FrozenPoseInputs:
    policy_path = Path(args.selected_policy_artifact)
    proposals_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    score_path = Path(args.proposal_score_artifact)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    policy = _load_npz(policy_path)
    metadata = json.loads(str(policy["metadata_json"].item()))
    if str(metadata.get("format")) != "pose_safe_selected_policy_v1":
        raise ValueError("unsupported selected pose policy artifact")
    for key, path in (
        ("proposals_sha256", proposals_path),
        ("candidate_artifact_sha256", candidate_path),
        ("score_artifact_sha256", score_path),
        ("projected_landmark_bank_sha256", bank_path),
        ("split_json_sha256", split_path),
    ):
        _require_hash(metadata, key, path)

    policy_summary_path = policy_path.parent / "summary.json"
    if not policy_summary_path.exists():
        raise ValueError("selected policy requires a sibling summary.json")
    policy_summary = json.loads(policy_summary_path.read_text())
    recorded_policy_hash = str(
        policy_summary.get("outputs", {}).get("selected_policy_artifact_sha256", "")
    )
    if recorded_policy_hash != file_sha256_short(policy_path):
        raise ValueError("selected policy hash differs from its sibling summary")

    score_summary_path = score_path.parent / "summary.json"
    if not score_summary_path.exists():
        raise ValueError("proposal score artifact requires a sibling summary.json")
    score_summary = json.loads(score_summary_path.read_text())
    recorded_score_hash = str(
        score_summary.get("outputs", {}).get("scores_sha256", "")
    )
    if recorded_score_hash != file_sha256_short(score_path):
        raise ValueError("proposal score artifact hash differs from its summary")

    proposals = _load_npz(proposals_path)
    candidate = _load_npz(candidate_path)
    scores = _load_npz(score_path)
    selected_rows = np.asarray(policy["selected_rows"], dtype=np.int64)
    candidate_columns = np.asarray(policy["candidate_pool_columns"], dtype=np.int64)
    if not np.array_equal(selected_rows, np.asarray(candidate["selected_rows"], dtype=np.int64)):
        raise ValueError("selected policy rows differ from the candidate artifact")
    if not np.array_equal(
        candidate_columns, np.asarray(candidate["selected_columns"], dtype=np.int64)
    ):
        raise ValueError("selected policy candidate columns differ from the candidate artifact")

    source_query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    source_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    query_ids = np.asarray(policy["query_ids"]).astype(str)
    query_xy = np.asarray(policy["query_xy"], dtype=np.float32)
    if not np.array_equal(query_ids, source_query_ids) or not np.allclose(
        query_xy, source_xy, atol=1e-5
    ):
        raise ValueError("selected policy query rows are stale or misaligned")

    score_key = (
        str(metadata.get("chosen_score_key"))
        if args.score_key is None
        else str(args.score_key)
    )
    if score_key != str(metadata.get("chosen_score_key")):
        raise ValueError("score_key must replay the frozen selected policy score")
    if score_key not in scores:
        raise ValueError(f"proposal score array is missing: {score_key}")
    candidate_scores = np.asarray(scores[score_key], dtype=np.float32)
    if candidate_scores.shape != candidate_columns.shape:
        raise ValueError("frozen proposal score array has an incompatible shape")

    source_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    source_prototypes = np.asarray(
        proposals["candidate_prototype_ids"], dtype=np.int64
    )
    candidate_track_ids = np.take_along_axis(
        source_tracks[selected_rows], candidate_columns, axis=1
    )
    candidate_prototype_ids = np.take_along_axis(
        source_prototypes[selected_rows], candidate_columns, axis=1
    )
    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    if str(metadata.get("descriptor_space_id")) != str(
        landmark_metadata.get("descriptor_space_id")
    ):
        raise ValueError("selected policy and landmark bank descriptor spaces differ")
    candidate_canonical_rows = canonical_rows_for_track_candidates(
        candidate_track_ids, landmark_index.track_ids
    )
    selected_track_ids = np.asarray(policy["selected_track_ids"], dtype=np.int64)
    selected_canonical_rows = canonical_rows_for_track_candidates(
        selected_track_ids[:, None], landmark_index.track_ids
    )[:, 0]
    if not np.array_equal(
        selected_canonical_rows,
        np.asarray(policy["selected_canonical_rows"], dtype=np.int64),
    ):
        raise ValueError("selected policy canonical landmark rows are stale")
    if np.any(selected_canonical_rows < 0):
        raise ValueError("selected policy contains tracks absent from the landmark bank")
    measurement_evidence = (
        None
        if not str(args.measurement_fit_summary)
        else load_frozen_measurement_pose_evidence(
            fit_summary_path=Path(args.measurement_fit_summary),
            late_apply_summary_path=(
                None
                if not str(args.measurement_late_apply_summary)
                else Path(args.measurement_late_apply_summary)
            ),
            feature_set=str(args.measurement_feature_set),
        )
    )
    measurement_update_evidence = (
        None
        if not str(args.measurement_update_fit_summary)
        else load_frozen_measurement_update_evidence(
            fit_summary_path=Path(args.measurement_update_fit_summary),
            late_apply_summary_path=(
                None
                if not str(args.measurement_update_late_apply_summary)
                else Path(args.measurement_update_late_apply_summary)
            ),
        )
    )
    coordinate_threshold = float(args.coordinate_geometry_probability_threshold)
    if not 0.0 <= coordinate_threshold <= 1.0:
        raise ValueError("coordinate geometry probability threshold must be in [0, 1]")
    if measurement_update_evidence is not None and measurement_evidence is None:
        raise ValueError("coordinate updates require frozen geometry evidence")
    return FrozenPoseInputs(
        query_ids=query_ids,
        token_indices=selected_rows,
        query_xy=query_xy,
        selected_track_ids=selected_track_ids,
        selected_prototype_ids=np.asarray(
            policy["selected_prototype_ids"], dtype=np.int64
        ),
        selected_canonical_rows=selected_canonical_rows,
        selected_scores=np.asarray(
            policy["selected_pose_selection_scores"], dtype=np.float32
        ),
        candidate_track_ids=candidate_track_ids,
        candidate_prototype_ids=candidate_prototype_ids,
        candidate_canonical_rows=candidate_canonical_rows,
        candidate_scores=candidate_scores,
        selected_policy_metadata=metadata,
        selected_policy_summary=policy_summary,
        landmark_index=landmark_index,
        landmark_metadata=landmark_metadata,
        measurement_evidence=measurement_evidence,
        measurement_update_evidence=measurement_update_evidence,
        coordinate_geometry_probability_threshold=coordinate_threshold,
    )


def _query_matches_and_pool(
    frozen: FrozenPoseInputs, query_id: str
) -> tuple[list[QueryTo3DMatch], PoseVerificationCandidatePool]:
    rows = np.flatnonzero(frozen.query_ids == str(query_id))
    index = frozen.landmark_index
    measurement_probabilities = (
        None
        if frozen.measurement_evidence is None
        else frozen.measurement_evidence.candidate_probability_matrix(
            query_id=str(query_id),
            token_indices=frozen.token_indices[rows],
            measured_track_ids=frozen.selected_track_ids[rows],
            candidate_track_ids=frozen.candidate_track_ids[rows],
        )
    )
    selected_measurement_probabilities: list[float | None] = []
    for local_row, selected_track in enumerate(frozen.selected_track_ids[rows]):
        if measurement_probabilities is None:
            selected_measurement_probabilities.append(None)
            continue
        columns = np.flatnonzero(
            frozen.candidate_track_ids[rows][local_row] == int(selected_track)
        )
        values = measurement_probabilities[local_row, columns]
        finite = values[np.isfinite(values)]
        selected_measurement_probabilities.append(
            None if len(finite) == 0 else float(finite[0])
        )
    if frozen.measurement_update_evidence is None:
        update_probabilities = np.full((len(rows),), np.nan, dtype=np.float64)
        updated_xy = np.full((len(rows), 2), np.nan, dtype=np.float64)
    else:
        update_probabilities, updated_xy = (
            frozen.measurement_update_evidence.assignment_updates(
                query_id=str(query_id),
                token_indices=frozen.token_indices[rows],
                measured_track_ids=frozen.selected_track_ids[rows],
            )
        )
    approved_refined_xy: list[np.ndarray | None] = []
    for geometry_probability, update_probability, refined_xy in zip(
        selected_measurement_probabilities, update_probabilities, updated_xy
    ):
        approved = bool(
            geometry_probability is not None
            and float(geometry_probability)
            >= float(frozen.coordinate_geometry_probability_threshold)
            and np.isfinite(float(update_probability))
            and frozen.measurement_update_evidence is not None
            and float(update_probability)
            >= float(frozen.measurement_update_evidence.update_threshold)
            and np.all(np.isfinite(refined_xy))
        )
        approved_refined_xy.append(
            np.asarray(refined_xy, dtype=np.float64) if approved else None
        )
    matches = [
        QueryTo3DMatch(
            token_index=int(frozen.token_indices[row]),
            xy=np.asarray(frozen.query_xy[row], dtype=np.float64),
            track_id=int(frozen.selected_track_ids[row]),
            xyz=np.asarray(
                index.xyz[int(frozen.selected_canonical_rows[row])], dtype=np.float64
            ),
            similarity=float(frozen.selected_scores[row]),
            ratio=0.0,
            landmark_variance=float(
                index.mean_variances[int(frozen.selected_canonical_rows[row])]
            ),
            source="frozen_l97_selected_policy",
            prototype_id=int(frozen.selected_prototype_ids[row]),
            geometry_probability=selected_measurement_probabilities[local_row],
            measurement_refined_xy=approved_refined_xy[local_row],
        )
        for local_row, row in enumerate(rows.tolist())
    ]
    canonical = frozen.candidate_canonical_rows[rows]
    valid = (canonical >= 0) & np.isfinite(frozen.candidate_scores[rows])
    xyz = np.zeros((*canonical.shape, 3), dtype=np.float64)
    xyz[valid] = index.xyz[canonical[valid]]
    pool = PoseVerificationCandidatePool(
        token_indices=frozen.token_indices[rows],
        xy=frozen.query_xy[rows],
        track_ids=frozen.candidate_track_ids[rows],
        prototype_ids=frozen.candidate_prototype_ids[rows],
        xyz=xyz,
        descriptor_scores=frozen.candidate_scores[rows],
        valid_mask=valid,
        measurement_geometry_probabilities=(
            measurement_probabilities
        ),
        measurement_verification_threshold=(
            0.5
            if frozen.measurement_evidence is None
            else float(frozen.measurement_evidence.verification_threshold)
        ),
    )
    return matches, pool


def _gt_pose(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return pose


def _ranking_group(
    query_id: str, result: VerifiedPnPResult, gt_pose: np.ndarray
) -> HypothesisRankingGroup:
    translation: list[float] = []
    rotation: list[float] = []
    for pose in result.hypothesis_poses_w2c:
        error = pnp_pose_error(pose, gt_pose)
        translation.append(float(error.translation_m))
        rotation.append(float(error.rotation_deg))
    return HypothesisRankingGroup(
        query_id=str(query_id),
        records=tuple(result.hypotheses),
        poses_w2c=tuple(result.hypothesis_poses_w2c),
        translation_errors_m=tuple(translation),
        rotation_errors_deg=tuple(rotation),
    )


def _pose_row(
    query_id: str,
    result: VerifiedPnPResult,
    group: HypothesisRankingGroup,
    gt_pose: np.ndarray,
) -> dict[str, object]:
    error = pnp_pose_error(result.pose_w2c, gt_pose)
    pre_error = pnp_pose_error(result.pre_refine_pose_w2c, gt_pose)
    finite_translation = np.asarray(group.translation_errors_m, dtype=np.float64)
    finite_rotation = np.asarray(group.rotation_errors_deg, dtype=np.float64)
    finite = np.isfinite(finite_translation) & np.isfinite(finite_rotation)
    if np.any(finite):
        finite_indices = np.flatnonzero(finite)
        oracle_index = int(
            min(
                finite_indices.tolist(),
                key=lambda index: (
                    float(finite_translation[index]), float(finite_rotation[index])
                ),
            )
        )
        oracle_translation = float(finite_translation[oracle_index])
        oracle_rotation = float(finite_rotation[oracle_index])
    else:
        oracle_index = None
        oracle_translation = float("inf")
        oracle_rotation = float("inf")
    chosen_index = result.chosen_hypothesis_index
    chosen_rank = (
        None
        if chosen_index is None or not np.isfinite(finite_translation[chosen_index])
        else 1 + int(np.sum(finite_translation[finite] < finite_translation[chosen_index]))
    )
    return {
        "query_id": str(query_id),
        "success": bool(
            result.success
            and np.isfinite(error.translation_m)
            and np.isfinite(error.rotation_deg)
        ),
        "match_count": int(result.match_count),
        "inlier_count": int(result.inlier_count),
        "translation_m": (
            None if not np.isfinite(error.translation_m) else float(error.translation_m)
        ),
        "rotation_deg": (
            None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg)
        ),
        "pre_refine_translation_m": (
            None
            if not np.isfinite(pre_error.translation_m)
            else float(pre_error.translation_m)
        ),
        "pre_refine_rotation_deg": (
            None if not np.isfinite(pre_error.rotation_deg) else float(pre_error.rotation_deg)
        ),
        "chosen_hypothesis_index": chosen_index,
        "chosen_hypothesis_translation_rank": chosen_rank,
        "oracle_hypothesis_index_TARGET_ONLY": oracle_index,
        "oracle_translation_m_TARGET_ONLY": (
            None if not np.isfinite(oracle_translation) else oracle_translation
        ),
        "oracle_rotation_deg_TARGET_ONLY": (
            None if not np.isfinite(oracle_rotation) else oracle_rotation
        ),
        "inference_verification": result.summary(),
    }


def _pose_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if bool(row.get("success"))]
    translation = np.asarray(
        [float(row["translation_m"]) for row in valid], dtype=np.float64
    )
    rotation = np.asarray(
        [float(row["rotation_deg"]) for row in valid], dtype=np.float64
    )
    if translation.size == 0:
        return {"query_count": len(rows), "success_count": 0, "success_rate": 0.0}
    output = {
        "query_count": int(len(rows)),
        "success_count": int(len(valid)),
        "success_rate": float(len(valid) / max(len(rows), 1)),
        "median_translation_m": float(np.median(translation)),
        "p90_translation_m": float(np.quantile(translation, 0.9)),
        "median_rotation_deg": float(np.median(rotation)),
        "recall_25cm_2deg": float(
            np.mean(
                [
                    bool(row.get("success"))
                    and float(row["translation_m"]) <= 0.25
                    and float(row["rotation_deg"]) <= 2.0
                    for row in rows
                ]
            )
        ),
        "recall_10cm_5deg": float(
            np.mean(
                [
                    bool(row.get("success"))
                    and float(row["translation_m"]) <= 0.10
                    and float(row["rotation_deg"]) <= 5.0
                    for row in rows
                ]
            )
        ),
        "recall_5cm_5deg": float(
            np.mean(
                [
                    bool(row.get("success"))
                    and float(row["translation_m"]) <= 0.05
                    and float(row["rotation_deg"]) <= 5.0
                    for row in rows
                ]
            )
        ),
        "median_inlier_count": float(
            np.median([int(row["inlier_count"]) for row in valid])
        ),
    }
    ranks = [
        int(row["chosen_hypothesis_translation_rank"])
        for row in rows
        if row.get("chosen_hypothesis_translation_rank") is not None
    ]
    oracle = [
        row for row in rows if row.get("oracle_translation_m_TARGET_ONLY") is not None
    ]
    if ranks:
        output["median_chosen_hypothesis_translation_rank"] = float(np.median(ranks))
    if oracle:
        output["oracle_median_translation_m_TARGET_ONLY"] = float(
            np.median([float(row["oracle_translation_m_TARGET_ONLY"]) for row in oracle])
        )
        output["oracle_p90_translation_m_TARGET_ONLY"] = float(
            np.quantile(
                [float(row["oracle_translation_m_TARGET_ONLY"]) for row in oracle], 0.9
            )
        )
    return output


def _run_split(
    *,
    split_name: str,
    query_ids: Sequence[str],
    frozen: FrozenPoseInputs,
    cameras: dict[int, object],
    images_by_name: dict[str, object],
    config: VerifiedPnPConfig,
    ranker: PoseHypothesisRanker | None,
) -> tuple[
    list[HypothesisRankingGroup],
    list[dict[str, object]],
    list[VerifiedPnPResult],
]:
    groups: list[HypothesisRankingGroup] = []
    rows: list[dict[str, object]] = []
    results: list[VerifiedPnPResult] = []
    for position, query_id in enumerate(query_ids, start=1):
        print(
            f"[{split_name}] {position}/{len(query_ids)} {query_id}", flush=True
        )
        image = images_by_name.get(str(query_id))
        if image is None:
            raise ValueError(f"query image is missing from COLMAP: {query_id}")
        matches, pool = _query_matches_and_pool(frozen, str(query_id))
        result = estimate_pose_with_heldout_verification(
            matches,
            cameras[int(image.camera_id)],
            config=config,
            query_seed=_query_seed(str(query_id)),
            candidate_pool=pool,
            hypothesis_selector=None if ranker is None else ranker.select,
        )
        group = _ranking_group(str(query_id), result, _gt_pose(image))
        if len(eligible_hypothesis_indices(group.records, group.poses_w2c)) < 2:
            raise RuntimeError(f"query has fewer than two eligible hypotheses: {query_id}")
        groups.append(group)
        rows.append(_pose_row(str(query_id), result, group, _gt_pose(image)))
        results.append(result)
    return groups, rows, results


def _hypothesis_audit_rows(
    groups: Sequence[HypothesisRankingGroup], ranker: PoseHypothesisRanker
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for group in groups:
        indices, features = pose_hypothesis_features(group.records, group.poses_w2c)
        scored_indices, scores = ranker.scores(group.records, group.poses_w2c, indices)
        if not np.array_equal(indices, scored_indices):
            raise RuntimeError("ranker changed hypothesis row identity")
        for index, feature, score in zip(indices.tolist(), features, scores):
            record = asdict(group.records[int(index)])
            output.append(
                {
                    "query_id": str(group.query_id),
                    "hypothesis_index": int(index),
                    "features": {
                        name: float(value)
                        for name, value in zip(POSE_HYPOTHESIS_FEATURE_NAMES, feature)
                    },
                    "ranker_score": float(score),
                    "translation_m_TARGET_ONLY": float(
                        group.translation_errors_m[int(index)]
                    ),
                    "rotation_deg_TARGET_ONLY": float(
                        group.rotation_errors_deg[int(index)]
                    ),
                    "record": record,
                }
            )
    return output


def _mainline_pose(summary: dict[str, object], split_name: str) -> dict[str, object]:
    if split_name == "validation":
        source = summary["validation"]["chosen"]["pose"]
    elif split_name == "test":
        source = summary["late_development_replay"]["pose"]
    else:
        raise ValueError("mainline pose is available only for validation/test")
    return {
        "query_count": int(source["query_count"]),
        "success_count": int(source["success_count"]),
        "success_rate": float(source["success_rate"]),
        "median_translation_m": float(source["median_translation_m_success"]),
        "p90_translation_m": float(source["p90_translation_m_success"]),
        "median_rotation_deg": float(source["median_rotation_deg_success"]),
        "recall_25cm_2deg": float(source["recall_25cm_2deg"]),
        "recall_10cm_5deg": float(source["recall_10cm_5deg"]),
        "recall_5cm_5deg": float(source["recall_5cm_5deg"]),
    }


def _strict_pose_gate(candidate: dict[str, object], baseline: dict[str, object]) -> bool:
    required = (
        "success_rate",
        "median_translation_m",
        "p90_translation_m",
        "median_rotation_deg",
    )
    return bool(
        all(key in candidate and key in baseline for key in required)
        and float(candidate["success_rate"]) >= float(baseline["success_rate"])
        and float(candidate["median_translation_m"])
        < float(baseline["median_translation_m"])
        and float(candidate["p90_translation_m"])
        < float(baseline["p90_translation_m"])
        and float(candidate["median_rotation_deg"])
        <= float(baseline["median_rotation_deg"])
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.disable_final_audit) and bool(args.enable_final_refine):
        raise ValueError(
            "final audit may be disabled only when final refinement is disabled"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen_inputs(args)
    split_path = Path(args.split_json)
    split = json.loads(split_path.read_text())
    for name in ("train", "validation", "test"):
        if name not in split or not isinstance(split[name], list) or not split[name]:
            raise ValueError("split JSON requires non-empty train/validation/test lists")
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("train and validation query splits overlap")
    if set(split["train"]) & set(split["test"]):
        raise ValueError("train and late query splits overlap")
    if set(split["validation"]) & set(split["test"]):
        raise ValueError("validation and late query splits overlap")
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    modes = tuple(
        item.strip()
        for item in str(args.hypothesis_selection_modes).split(",")
        if item.strip()
    )
    config = VerifiedPnPConfig(
        fit_match_counts=tuple(args.fit_match_counts),
        selection_modes=modes,
        ransac_thresholds_px=tuple(args.ransac_thresholds_px),
        rng_seed_offsets=tuple(args.rng_seed_offsets),
        ransac_iterations=int(args.ransac_iterations),
        holdout_folds=int(args.holdout_folds),
        holdout_fold=int(args.holdout_fold),
        final_audit_fold=(
            None if bool(args.disable_final_audit) else int(args.final_audit_fold)
        ),
        verification_strict_px=float(args.verification_strict_px),
        verification_loose_px=float(args.verification_loose_px),
        final_consensus_px=float(args.final_consensus_px),
        final_refine_f_scale_px=float(args.final_refine_f_scale_px),
        min_final_inliers=int(args.min_final_inliers),
        enable_final_refine=bool(args.enable_final_refine),
        candidate_pool_residual_sigma_px=float(
            args.candidate_pool_residual_sigma_px
        ),
        candidate_pool_hard_threshold_px=float(
            args.candidate_pool_hard_threshold_px
        ),
        candidate_pool_descriptor_rank_weight=float(
            args.candidate_pool_descriptor_rank_weight
        ),
        candidate_pool_refine_iterations=int(args.candidate_pool_refine_iterations),
        measurement_verified_threshold=(
            0.64
            if frozen.measurement_evidence is None
            else float(frozen.measurement_evidence.verification_threshold)
        ),
        measurement_verified_min_matches=int(args.measurement_verified_min_matches),
        measurement_verified_min_grid_cells=int(
            args.measurement_verified_min_grid_cells
        ),
    )

    train_groups, train_legacy_rows, _train_legacy_results = _run_split(
        split_name="train_hypothesis_generation",
        query_ids=[str(value) for value in split["train"]],
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        config=config,
        ranker=None,
    )
    ranker, fit_summary = fit_pose_hypothesis_ranker(
        train_groups,
        c_values=tuple(args.ranker_c_values),
        cross_validation_folds=int(args.ranker_cross_validation_folds),
        rotation_equivalent_m_per_deg=float(args.rotation_equivalent_m_per_deg),
        minimum_pair_gap_m=float(args.minimum_pair_gap_m),
    )
    validation_groups, validation_legacy_rows, _validation_legacy_results = _run_split(
        split_name="validation_legacy_selector",
        query_ids=[str(value) for value in split["validation"]],
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        config=config,
        ranker=None,
    )
    validation_legacy_pre = evaluate_legacy_hypothesis_ranking(validation_groups)
    validation_learned_pre = evaluate_pose_hypothesis_ranker(
        validation_groups, ranker
    )
    validation_rank_gate = bool(
        float(validation_learned_pre["median_translation_m"])
        < float(validation_legacy_pre["median_translation_m"])
        and float(validation_learned_pre["p90_translation_m"])
        < float(validation_legacy_pre["p90_translation_m"])
        and float(validation_learned_pre["median_translation_rank"])
        < float(validation_legacy_pre["median_translation_rank"])
    )
    (
        _validation_learned_groups,
        validation_learned_rows,
        _validation_learned_results,
    ) = _run_split(
        split_name="validation_frozen_learned_selector",
        query_ids=[str(value) for value in split["validation"]],
        frozen=frozen,
        cameras=cameras,
        images_by_name=images_by_name,
        config=config,
        ranker=ranker,
    )
    validation_legacy_pose = _pose_metrics(validation_legacy_rows)
    validation_learned_pose = _pose_metrics(validation_learned_rows)
    validation_mainline_pose = _mainline_pose(
        frozen.selected_policy_summary, "validation"
    )
    validation_pose_gate = _strict_pose_gate(
        validation_learned_pose, validation_mainline_pose
    )
    validation_promotion_passes = bool(validation_rank_gate and validation_pose_gate)

    late_groups: list[HypothesisRankingGroup] = []
    late_rows: list[dict[str, object]] = []
    if (
        validation_promotion_passes or bool(args.force_late_development_replay)
    ) and not bool(args.skip_late_on_validation_pass):
        late_groups, late_rows, _late_results = _run_split(
            split_name="late_development_frozen_learned_selector",
            query_ids=[str(value) for value in split["test"]],
            frozen=frozen,
            cameras=cameras,
            images_by_name=images_by_name,
            config=config,
            ranker=ranker,
        )

    late_candidate_pose = None if not late_rows else _pose_metrics(late_rows)
    late_mainline_pose = (
        None
        if not late_rows
        else _mainline_pose(frozen.selected_policy_summary, "test")
    )
    late_pose_gate = bool(
        late_candidate_pose is not None
        and late_mainline_pose is not None
        and _strict_pose_gate(late_candidate_pose, late_mainline_pose)
    )
    production_promotion_passes = bool(
        validation_promotion_passes and late_pose_gate
    )

    model_path = output_dir / "pose_hypothesis_ranker.json"
    model_payload = {
        "format": "pose_hypothesis_ranker_v1",
        "model": ranker.to_dict(),
        "inputs": {
            "selected_policy_artifact_sha256": file_sha256_short(
                Path(args.selected_policy_artifact)
            ),
            "proposals_sha256": file_sha256_short(Path(args.proposals)),
            "candidate_artifact_sha256": file_sha256_short(
                Path(args.candidate_artifact)
            ),
            "proposal_score_artifact_sha256": file_sha256_short(
                Path(args.proposal_score_artifact)
            ),
            "projected_landmark_bank_sha256": file_sha256_short(
                Path(args.projected_landmark_bank)
            ),
            "split_json_sha256": file_sha256_short(split_path),
            "measurement_evidence": (
                None
                if frozen.measurement_evidence is None
                else dict(frozen.measurement_evidence.manifest)
            ),
            "measurement_update_evidence": (
                None
                if frozen.measurement_update_evidence is None
                else dict(frozen.measurement_update_evidence.manifest)
            ),
            "coordinate_geometry_probability_threshold": float(
                frozen.coordinate_geometry_probability_threshold
            ),
        },
        "verified_pnp_config": {
            **config.__dict__,
            "fit_match_counts": list(config.fit_match_counts),
            "selection_modes": list(config.selection_modes),
            "ransac_thresholds_px": list(config.ransac_thresholds_px),
            "rng_seed_offsets": list(config.rng_seed_offsets),
        },
        "fit_summary": fit_summary,
    }
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_payload = {
        "train_legacy_generation": train_legacy_rows,
        "validation_legacy_selector": validation_legacy_rows,
        "validation_learned_selector": validation_learned_rows,
        "late_development_learned_selector": late_rows,
    }
    pose_rows_path.write_text(
        json.dumps(pose_rows_payload, indent=2, sort_keys=True) + "\n"
    )
    train_audit_path = output_dir / "train_hypothesis_audit.json"
    train_audit_path.write_text(
        json.dumps(_hypothesis_audit_rows(train_groups, ranker), indent=2, sort_keys=True)
        + "\n"
    )
    validation_audit_path = output_dir / "validation_hypothesis_audit.json"
    validation_audit_path.write_text(
        json.dumps(
            _hypothesis_audit_rows(validation_groups, ranker),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    late_audit_path = output_dir / "late_hypothesis_audit.json"
    if late_groups:
        late_audit_path.write_text(
            json.dumps(
                _hypothesis_audit_rows(late_groups, ranker),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    summary = {
        "stage": "train_only_learned_pose_hypothesis_ranking",
        "protocol": {
            "fit_split": "train_queries_only",
            "ranker_hyperparameter_selection": "train_grouped_cross_validation_only",
            "validation_used_for_model_fit": False,
            "late_used_for_model_or_policy_selection": False,
            "late_replay_forced_for_diagnostic": bool(
                args.force_late_development_replay
            ),
            "ground_truth_available_to_inference_ranker": False,
            "ranker_feature_source": "fit_plus_heldout_verification_plus_cross_hypothesis_consensus",
            "candidate_hypotheses_are_mutually_exclusive_per_query_token": True,
            "frozen_upstream_policy": str(args.selected_policy_artifact),
            "proposal_scope": "full_bank_global_faiss_top20",
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": frozen.measurement_evidence is not None,
            "measurement_role": (
                "disabled"
                if frozen.measurement_evidence is None
                else "pose_free_heldout_hypothesis_verification_only"
            ),
            "measurement_coordinate_update": (
                frozen.measurement_update_evidence is not None
            ),
            "measurement_coordinate_update_role": (
                "disabled"
                if frozen.measurement_update_evidence is None
                else "measurement_verified_refined_fit_hypotheses_only"
            ),
            "late_block_is_untouched_test": False,
            "production_promoted": production_promotion_passes,
        },
        "inputs": model_payload["inputs"],
        "descriptor_space_id": frozen.landmark_metadata.get("descriptor_space_id"),
        "score_key": frozen.selected_policy_metadata.get("chosen_score_key"),
        "ranker_fit": fit_summary,
        "train": {
            "query_count": len(train_groups),
            "legacy_final_pose": _pose_metrics(train_legacy_rows),
        },
        "validation": {
            "legacy_pre_refine_ranking": validation_legacy_pre,
            "learned_pre_refine_ranking": validation_learned_pre,
            "legacy_verifier_final_pose": validation_legacy_pose,
            "learned_verifier_final_pose": validation_learned_pose,
            "frozen_l97_mainline_pose": validation_mainline_pose,
            "passes_rank_regret_gate": validation_rank_gate,
            "passes_strict_l97_pose_gate": validation_pose_gate,
            "promotion_passes": validation_promotion_passes,
        },
        "late_development_replay": {
            "executed": bool(late_rows),
            "query_count": len(late_groups),
            "learned_verifier_final_pose": late_candidate_pose,
            "frozen_l97_mainline_pose": late_mainline_pose,
            "passes_strict_l97_pose_gate": late_pose_gate,
            "used_for_selection": False,
        },
        "production_promotion_passes": production_promotion_passes,
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "pose_rows": str(pose_rows_path),
            "pose_rows_sha256": file_sha256_short(pose_rows_path),
            "train_hypothesis_audit": str(train_audit_path),
            "train_hypothesis_audit_sha256": file_sha256_short(train_audit_path),
            "validation_hypothesis_audit": str(validation_audit_path),
            "validation_hypothesis_audit_sha256": file_sha256_short(
                validation_audit_path
            ),
            "late_hypothesis_audit": (
                None if not late_groups else str(late_audit_path)
            ),
            "late_hypothesis_audit_sha256": (
                None if not late_groups else file_sha256_short(late_audit_path)
            ),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
