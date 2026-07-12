"""Evaluate held-out multi-hypothesis PnP on frozen global top-L proposals.

All pose hypotheses are selected without ground-truth pose access. Ground
truth is used only after a final pose has been returned, to report validation
metrics. A reused late block is evaluated only under the explicit development
cross-block flag and can never produce a production claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_global_partial_assignment import (
    _load_frozen_baseline_policy,
    _validate_frozen_baseline_pose,
)
from feature_extract.tools.vfm.probe_detector_maplet_geometry import _pose_gate
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
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    PoseVerificationCandidatePool,
    VerifiedPnPConfig,
    estimate_pose_with_heldout_verification,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)


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
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--score_keys",
        default="ensemble__geometry_p05px",
        help="comma-separated compact matcher score arrays",
    )
    parser.add_argument("--baseline_score_key", default="baseline_scores")
    parser.add_argument("--frozen_baseline_summary", default=None)
    parser.add_argument(
        "--frozen_baseline_source_score_key",
        default="strategy__alike_support_top2_mean",
        help="baseline score identity recorded by the frozen global sweep",
    )
    parser.add_argument(
        "--assignment_modes", default="row_argmax,global_bipartite"
    )
    parser.add_argument(
        "--fit_match_counts", type=_positive_int_list, default=(24, 32, 48, 64)
    )
    parser.add_argument(
        "--hypothesis_selection_modes",
        default="score_topk,spatial_round_robin,geometry_diverse",
    )
    parser.add_argument(
        "--ransac_thresholds_px", type=_positive_float_list, default=(2.0, 4.0, 8.0)
    )
    parser.add_argument("--rng_seed_offsets", type=_integer_list, default=(0, 1))
    parser.add_argument("--ransac_iterations", type=int, default=3000)
    parser.add_argument("--holdout_folds", type=int, default=4)
    parser.add_argument("--holdout_fold", type=int, default=0)
    parser.add_argument("--verification_strict_px", type=float, default=2.0)
    parser.add_argument("--verification_loose_px", type=float, default=5.0)
    parser.add_argument("--final_consensus_px", type=float, default=4.0)
    parser.add_argument("--final_refine_f_scale_px", type=float, default=2.0)
    parser.add_argument("--min_final_inliers", type=int, default=6)
    parser.add_argument("--enable_final_refine", action="store_true")
    parser.add_argument(
        "--disable_topl_candidate_pool_verification", action="store_true"
    )
    parser.add_argument("--candidate_pool_residual_sigma_px", type=float, default=2.0)
    parser.add_argument("--candidate_pool_hard_threshold_px", type=float, default=8.0)
    parser.add_argument(
        "--candidate_pool_descriptor_rank_weight", type=float, default=0.02
    )
    parser.add_argument("--candidate_pool_refine_iterations", type=int, default=2)
    parser.add_argument("--single_ransac_match_count", type=int, default=32)
    parser.add_argument("--single_ransac_selection_mode", default="score_topk")
    parser.add_argument("--single_ransac_threshold_px", type=float, default=8.0)
    parser.add_argument("--single_ransac_iterations", type=int, default=5000)
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
    )
    parser.add_argument(
        "--development_cross_block_audit",
        action="store_true",
        help="replay validation-frozen policies on the reused late block",
    )
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _compact(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    valid = columns >= 0
    safe_columns = np.maximum(columns, 0)
    output = np.take_along_axis(np.asarray(values)[rows], safe_columns, axis=1).copy()
    if np.issubdtype(output.dtype, np.floating):
        output[~valid] = -np.inf
    else:
        output[~valid] = -1
    return output


def _selected_columns(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid & np.isfinite(scores), scores, -np.inf)
    selected = np.argmax(safe, axis=1).astype(np.int64)
    selected[~np.any(np.isfinite(safe), axis=1)] = -1
    return selected


def _score_array(
    payload: dict[str, np.ndarray],
    proposals: dict[str, np.ndarray],
    key: str,
    *,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
) -> np.ndarray:
    source = payload.get(str(key), proposals.get(str(key)))
    if source is None:
        raise ValueError(f"score array is missing: {key}")
    values = np.asarray(source)
    if values.shape == selected_columns.shape:
        return values.astype(np.float32)
    proposal_shape = np.asarray(proposals["candidate_track_ids"]).shape
    if values.shape == proposal_shape:
        return _compact(values, selected_rows, selected_columns).astype(np.float32)
    raise ValueError(
        f"score array {key} has shape {values.shape}; expected {selected_columns.shape} "
        f"or {proposal_shape}"
    )


def _query_seed(query_id: str) -> int:
    return int.from_bytes(hashlib.sha256(str(query_id).encode("utf8")).digest()[:4], "little")


def _set_cv2_seed(seed: int) -> None:
    try:
        import cv2

        cv2.setRNGSeed(int(int(seed) % (2**31 - 1)))
    except ImportError:  # pragma: no cover
        return


def _pose_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    success = [row for row in rows if bool(row.get("success"))]
    translations = np.asarray(
        [float(row["translation_m"]) for row in success], dtype=np.float64
    )
    rotations = np.asarray(
        [float(row["rotation_deg"]) for row in success], dtype=np.float64
    )
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "success_count": int(len(success)),
        "success_rate": 0.0 if not rows else float(len(success) / len(rows)),
        "median_translation_m_success": (
            None if translations.size == 0 else float(np.median(translations))
        ),
        "p90_translation_m_success": (
            None if translations.size == 0 else float(np.percentile(translations, 90))
        ),
        "median_rotation_deg_success": (
            None if rotations.size == 0 else float(np.median(rotations))
        ),
        "median_matches": (
            None
            if not rows
            else float(np.median([int(row.get("match_count", 0)) for row in rows]))
        ),
        "median_inliers_success": (
            None
            if not success
            else float(np.median([int(row.get("inlier_count", 0)) for row in success]))
        ),
    }
    for distance, angle, name in (
        (0.25, 2.0, "25cm_2deg"),
        (0.10, 5.0, "10cm_5deg"),
        (0.05, 5.0, "5cm_5deg"),
    ):
        output[f"recall_{name}"] = (
            0.0
            if not rows
            else float(
                np.mean(
                    [
                        bool(row.get("success"))
                        and float(row["translation_m"]) <= distance
                        and float(row["rotation_deg"]) <= angle
                        for row in rows
                    ]
                )
            )
        )
    pre_refine = [
        row
        for row in rows
        if row.get("pre_refine_translation_m") is not None
        and row.get("pre_refine_rotation_deg") is not None
    ]
    oracle = [
        row
        for row in rows
        if row.get("hypothesis_oracle_translation_m") is not None
        and row.get("hypothesis_oracle_rotation_deg") is not None
    ]
    if pre_refine:
        output["pre_refine_median_translation_m"] = float(
            np.median([float(row["pre_refine_translation_m"]) for row in pre_refine])
        )
        output["pre_refine_p90_translation_m"] = float(
            np.percentile(
                [float(row["pre_refine_translation_m"]) for row in pre_refine], 90
            )
        )
        output["pre_refine_median_rotation_deg"] = float(
            np.median([float(row["pre_refine_rotation_deg"]) for row in pre_refine])
        )
        output["median_final_minus_pre_refine_translation_m"] = float(
            np.median(
                [
                    float(row["translation_m"])
                    - float(row["pre_refine_translation_m"])
                    for row in pre_refine
                    if row.get("translation_m") is not None
                ]
            )
        )
    if oracle:
        output["hypothesis_oracle_median_translation_m"] = float(
            np.median(
                [float(row["hypothesis_oracle_translation_m"]) for row in oracle]
            )
        )
        output["hypothesis_oracle_p90_translation_m"] = float(
            np.percentile(
                [float(row["hypothesis_oracle_translation_m"]) for row in oracle], 90
            )
        )
        output["hypothesis_oracle_median_rotation_deg"] = float(
            np.median([float(row["hypothesis_oracle_rotation_deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_10cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_10cm_5deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_5cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_5cm_5deg"]) for row in oracle])
        )
        output["median_chosen_hypothesis_translation_rank"] = float(
            np.median(
                [float(row["chosen_hypothesis_translation_rank"]) for row in oracle]
            )
        )
    return output


def _policy_key(trial: dict[str, object]) -> tuple[float, ...]:
    pose = dict(trial["verified_pose"])
    values = np.asarray(
        [
            float(pose["median_translation_m_success"]),
            float(pose["p90_translation_m_success"]),
            float(pose["median_rotation_deg_success"]),
        ],
        dtype=np.float64,
    )
    geomean = float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))
    return (
        float(pose["success_rate"]),
        -geomean,
        -float(pose["p90_translation_m_success"]),
        -float(pose["median_translation_m_success"]),
        -float(pose["median_rotation_deg_success"]),
        float(pose["recall_10cm_5deg"]),
        float(pose["recall_5cm_5deg"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.development_cross_block_audit) and str(args.evaluation_role) != "development":
        raise ValueError("cross-block replay is development-only")
    if int(args.single_ransac_match_count) < 4:
        raise ValueError("single_ransac_match_count must be at least four")
    if str(args.single_ransac_selection_mode) not in {
        "score_topk",
        "spatial_round_robin",
    }:
        raise ValueError("unsupported single-RANSAC selection mode")
    assignment_modes = tuple(
        item.strip() for item in str(args.assignment_modes).split(",") if item.strip()
    )
    if not assignment_modes or set(assignment_modes) - {
        "row_argmax",
        "global_bipartite",
    }:
        raise ValueError("unsupported assignment mode")
    hypothesis_modes = tuple(
        item.strip()
        for item in str(args.hypothesis_selection_modes).split(",")
        if item.strip()
    )
    score_keys = tuple(
        item.strip() for item in str(args.score_keys).split(",") if item.strip()
    )
    if not score_keys:
        raise ValueError("score_keys cannot be empty")

    proposals_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    score_path = Path(args.score_artifact)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals = _load_npz(proposals_path)
    candidate = _load_npz(candidate_path)
    score_payload = _load_npz(score_path)
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(candidate["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(candidate["valid_edges"], dtype=bool)
    if selected_columns.ndim != 2 or valid_edges.shape != selected_columns.shape:
        raise ValueError("candidate artifact arrays have incompatible shapes")
    candidate_metadata = json.loads(str(candidate["metadata_json"].item()))
    expected_candidate = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    candidate_mismatches = {
        key: {"expected": value, "actual": candidate_metadata.get(key)}
        for key, value in expected_candidate.items()
        if candidate_metadata.get(key) != value
    }
    if candidate_mismatches:
        raise ValueError(
            "stale candidate artifact: "
            f"{json.dumps(candidate_mismatches, sort_keys=True)}"
        )
    score_summary_path = score_path.parent / "summary.json"
    if not score_summary_path.exists():
        raise ValueError("score artifact requires sibling summary.json")
    score_summary = json.loads(score_summary_path.read_text())
    score_manifest = score_summary.get("data_manifest")
    if not isinstance(score_manifest, dict):
        raise ValueError("score summary is missing data_manifest")
    score_outputs = score_summary.get("outputs")
    recorded_score_hash = (
        None
        if not isinstance(score_outputs, dict)
        else score_outputs.get("scores_sha256")
    )
    actual_score_hash = file_sha256_short(score_path)
    if not recorded_score_hash or str(recorded_score_hash) != actual_score_hash:
        raise ValueError("score artifact hash differs from its sibling summary")
    expected_score = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "feature_artifact_sha256": file_sha256_short(candidate_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    score_mismatches = {
        key: {"expected": value, "actual": score_manifest.get(key)}
        for key, value in expected_score.items()
        if score_manifest.get(key) != value
    }
    if score_mismatches:
        raise ValueError(
            "stale or misaligned score artifact: "
            f"{json.dumps(score_mismatches, sort_keys=True)}"
        )
    frozen_baseline_source = (
        None
        if args.frozen_baseline_summary is None
        else _load_frozen_baseline_policy(
            Path(args.frozen_baseline_summary),
            proposals_path=proposals_path,
            candidate_path=candidate_path,
            bank_path=bank_path,
            split_path=split_path,
            baseline_score_key=str(args.frozen_baseline_source_score_key),
        )
    )
    if frozen_baseline_source is not None:
        score_protocol = score_summary.get("protocol")
        score_baseline = (
            None
            if not isinstance(score_protocol, dict)
            else score_protocol.get("baseline_strategy")
        )
        normalized_score_baseline = str(score_baseline)
        if not normalized_score_baseline.startswith("strategy__"):
            normalized_score_baseline = f"strategy__{normalized_score_baseline}"
        if normalized_score_baseline != str(args.frozen_baseline_source_score_key):
            raise ValueError(
                "score artifact baseline identity differs from frozen baseline"
            )
        if (
            int(args.single_ransac_match_count)
            != int(frozen_baseline_source["max_matches"])
            or str(args.single_ransac_selection_mode)
            != str(frozen_baseline_source["selection_mode"])
        ):
            raise ValueError(
                "single-RANSAC policy must replay the frozen baseline K/mode"
            )

    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    compact_tracks = _compact(
        proposals["candidate_track_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    compact_prototypes = _compact(
        proposals["candidate_prototype_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    canonical_rows = canonical_rows_for_track_candidates(
        compact_tracks, landmark_index.track_ids
    )
    valid_edges &= canonical_rows >= 0
    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    split = json.loads(split_path.read_text())
    for name in ("train", "validation", "test"):
        if name not in split or not isinstance(split[name], list) or not split[name]:
            raise ValueError("split JSON requires non-empty train/validation/test lists")
    split_masks = {
        name: np.isin(query_ids, np.asarray(split[name], dtype=np.str_))
        for name in ("validation", "test")
    }
    if np.any(split_masks["validation"] & split_masks["test"]):
        raise ValueError("validation and late split overlap")
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    config = VerifiedPnPConfig(
        fit_match_counts=tuple(args.fit_match_counts),
        selection_modes=hypothesis_modes,
        ransac_thresholds_px=tuple(args.ransac_thresholds_px),
        rng_seed_offsets=tuple(args.rng_seed_offsets),
        ransac_iterations=int(args.ransac_iterations),
        holdout_folds=int(args.holdout_folds),
        holdout_fold=int(args.holdout_fold),
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
    )
    raw_scores = {
        key: _score_array(
            score_payload,
            proposals,
            key,
            selected_rows=selected_rows,
            selected_columns=selected_columns,
        )
        for key in (*score_keys, str(args.baseline_score_key))
    }
    for values in raw_scores.values():
        values[~valid_edges] = -np.inf

    policy_scores: dict[str, np.ndarray] = {}
    policy_metadata: dict[str, dict[str, object]] = {}
    for score_key, values in raw_scores.items():
        if "row_argmax" in assignment_modes:
            key = f"row_argmax__{score_key}"
            policy_scores[key] = values
            policy_metadata[key] = {
                "assignment_mode": "row_argmax_then_conflict_resolution",
                "score_key": score_key,
            }
        if "global_bipartite" in assignment_modes:
            key = f"global_bipartite__{score_key}"
            resolved, _selected = global_assignment_score_matrix(
                compact_tracks,
                values,
                query_ids,
                valid_mask=valid_edges,
                dustbin_score=None,
            )
            policy_scores[key] = resolved
            policy_metadata[key] = {
                "assignment_mode": "whole_image_sparse_bipartite_per_query_dustbin",
                "score_key": score_key,
            }

    def matches_by_query(scores: np.ndarray, split_name: str) -> dict[str, list[QueryTo3DMatch]]:
        selected = _selected_columns(scores, valid_edges)
        output: dict[str, list[QueryTo3DMatch]] = {}
        for row in np.flatnonzero(split_masks[split_name]).tolist():
            column = int(selected[row])
            if column < 0:
                continue
            query_id = str(query_ids[row])
            output.setdefault(query_id, []).append(
                QueryTo3DMatch(
                    token_index=int(selected_rows[row]),
                    xy=np.asarray(query_xy[row], dtype=np.float64),
                    track_id=int(compact_tracks[row, column]),
                    xyz=np.asarray(
                        landmark_index.xyz[int(canonical_rows[row, column])],
                        dtype=np.float64,
                    ),
                    similarity=float(scores[row, column]),
                    ratio=0.0,
                    landmark_variance=float(
                        landmark_index.mean_variances[
                            int(canonical_rows[row, column])
                        ]
                    ),
                    source="heldout_pose_hypothesis_eval",
                    prototype_id=int(compact_prototypes[row, column]),
                )
            )
        return output

    def candidate_pools_by_query(
        scores: np.ndarray, split_name: str
    ) -> dict[str, PoseVerificationCandidatePool]:
        output: dict[str, PoseVerificationCandidatePool] = {}
        for query_id in split[split_name]:
            rows = np.flatnonzero(
                split_masks[split_name] & (query_ids == str(query_id))
            )
            xyz = np.zeros((*canonical_rows[rows].shape, 3), dtype=np.float64)
            local_valid = valid_edges[rows]
            xyz[local_valid] = landmark_index.xyz[
                canonical_rows[rows][local_valid]
            ]
            output[str(query_id)] = PoseVerificationCandidatePool(
                token_indices=selected_rows[rows],
                xy=query_xy[rows],
                track_ids=compact_tracks[rows],
                prototype_ids=compact_prototypes[rows],
                xyz=xyz,
                descriptor_scores=scores[rows],
                valid_mask=local_valid,
            )
        return output

    def evaluate(
        scores: np.ndarray,
        split_name: str,
        backend: str,
        *,
        candidate_pool_scores: np.ndarray | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        grouped = matches_by_query(scores, split_name)
        grouped_candidate_pools = (
            {}
            if candidate_pool_scores is None
            else candidate_pools_by_query(candidate_pool_scores, split_name)
        )
        expected_ids = [str(value) for value in split[split_name]]
        rows: list[dict[str, object]] = []
        for query_id in expected_ids:
            image = images_by_name.get(query_id)
            if image is None:
                rows.append(
                    {
                        "query_id": query_id,
                        "success": False,
                        "failure_reason": "missing_colmap_image",
                        "match_count": 0,
                        "inlier_count": 0,
                    }
                )
                continue
            camera = cameras[int(image.camera_id)]
            matches = grouped.get(query_id, [])
            verified_result = None
            if backend == "verified":
                verified_result = estimate_pose_with_heldout_verification(
                    matches,
                    camera,
                    config=config,
                    query_seed=_query_seed(query_id),
                    candidate_pool=grouped_candidate_pools.get(query_id),
                )
                pose = verified_result.pose_w2c
                solver_success = bool(verified_result.success)
                match_count = int(verified_result.match_count)
                inlier_count = int(verified_result.inlier_count)
                backend_summary = verified_result.summary()
            elif backend == "single_ransac":
                selected_matches = select_pose_safe_matches(
                    matches,
                    max_matches=int(args.single_ransac_match_count),
                    image_width=int(camera.width),
                    image_height=int(camera.height),
                    mode=str(args.single_ransac_selection_mode),
                )
                selected_matches = stable_uniform_ransac_order(selected_matches)
                _set_cv2_seed(_query_seed(query_id))
                result = estimate_pose_pnp_ransac(
                    selected_matches,
                    camera,
                    reprojection_error_px=float(args.single_ransac_threshold_px),
                    iterations=int(args.single_ransac_iterations),
                    refine_method="LM",
                )
                pose = result.pose_w2c
                solver_success = bool(result.success)
                match_count = int(result.match_count)
                inlier_count = int(result.inlier_count)
                backend_summary = None
            else:
                raise ValueError(f"unsupported backend: {backend}")
            gt_pose = np.eye(4, dtype=np.float64)
            gt_pose[:3, :3] = qvec_to_rotmat(image.qvec)
            gt_pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
            error = pnp_pose_error(pose, gt_pose)
            pre_refine_error = None
            oracle_error = None
            chosen_translation_rank = None
            if verified_result is not None:
                pre_refine_error = pnp_pose_error(
                    verified_result.pre_refine_pose_w2c, gt_pose
                )
                hypothesis_errors = [
                    pnp_pose_error(hypothesis_pose, gt_pose)
                    for hypothesis_pose in verified_result.hypothesis_poses_w2c
                    if hypothesis_pose is not None
                ]
                finite_hypothesis_errors = [
                    value
                    for value in hypothesis_errors
                    if np.isfinite(value.translation_m)
                    and np.isfinite(value.rotation_deg)
                ]
                if finite_hypothesis_errors:
                    oracle_error = min(
                        finite_hypothesis_errors,
                        key=lambda value: (
                            float(value.translation_m), float(value.rotation_deg)
                        ),
                    )
                    chosen_translation_rank = 1 + int(
                        np.sum(
                            [
                                float(value.translation_m)
                                < float(pre_refine_error.translation_m)
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
            success = bool(
                solver_success
                and np.isfinite(error.translation_m)
                and np.isfinite(error.rotation_deg)
            )
            rows.append(
                {
                    "query_id": query_id,
                    "success": success,
                    "solver_success": solver_success,
                    "failure_reason": None if success else "pnp_solver_failure",
                    "match_count": match_count,
                    "inlier_count": inlier_count,
                    "translation_m": (
                        None if not np.isfinite(error.translation_m) else float(error.translation_m)
                    ),
                    "rotation_deg": (
                        None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg)
                    ),
                    "pre_refine_translation_m": (
                        None
                        if pre_refine_error is None
                        or not np.isfinite(pre_refine_error.translation_m)
                        else float(pre_refine_error.translation_m)
                    ),
                    "pre_refine_rotation_deg": (
                        None
                        if pre_refine_error is None
                        or not np.isfinite(pre_refine_error.rotation_deg)
                        else float(pre_refine_error.rotation_deg)
                    ),
                    "hypothesis_oracle_translation_m": (
                        None if oracle_error is None else float(oracle_error.translation_m)
                    ),
                    "hypothesis_oracle_rotation_deg": (
                        None if oracle_error is None else float(oracle_error.rotation_deg)
                    ),
                    "hypothesis_oracle_10cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.10
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "hypothesis_oracle_5cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.05
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "chosen_hypothesis_translation_rank": chosen_translation_rank,
                    "inference_verification": backend_summary,
                }
            )
        return _pose_summary(rows), rows

    validation_trials: list[dict[str, object]] = []
    validation_rows: dict[str, dict[str, object]] = {}
    for policy_key, scores in policy_scores.items():
        source_scores = raw_scores[str(policy_metadata[policy_key]["score_key"])]
        candidate_pool_scores = (
            None
            if bool(args.disable_topl_candidate_pool_verification)
            else source_scores
        )
        verified_pose, verified_rows = evaluate(
            scores,
            "validation",
            "verified",
            candidate_pool_scores=candidate_pool_scores,
        )
        single_pose, single_rows = evaluate(scores, "validation", "single_ransac")
        trial = {
            "trial_id": int(len(validation_trials)),
            "policy_key": policy_key,
            **policy_metadata[policy_key],
            "verified_pose": verified_pose,
            "matched_single_ransac_pose": single_pose,
            "passes_matched_pose_gate": _pose_gate(verified_pose, single_pose),
            "passes_frozen_baseline_pose_gate": (
                None
                if frozen_baseline_source is None
                else _pose_gate(verified_pose, frozen_baseline_source["pose"])
            ),
        }
        validation_trials.append(trial)
        validation_rows[policy_key] = {
            "verified": verified_rows,
            "matched_single_ransac": single_rows,
        }
    if frozen_baseline_source is not None:
        baseline_policy_key = f"global_bipartite__{str(args.baseline_score_key)}"
        baseline_trials = [
            trial
            for trial in validation_trials
            if str(trial["policy_key"]) == baseline_policy_key
        ]
        if len(baseline_trials) != 1:
            raise ValueError(
                "frozen baseline replay requires global_bipartite baseline scores"
            )
        _validate_frozen_baseline_pose(
            baseline_trials[0]["matched_single_ransac_pose"],
            frozen_baseline_source,
        )
    finite_trials = [
        trial
        for trial in validation_trials
        if trial["verified_pose"]["median_translation_m_success"] is not None
        and trial["verified_pose"]["p90_translation_m_success"] is not None
    ]
    if not finite_trials:
        raise RuntimeError("no verification policy produced a finite validation pose")
    gate_key = (
        "passes_matched_pose_gate"
        if frozen_baseline_source is None
        else "passes_frozen_baseline_pose_gate"
    )
    gated = [trial for trial in finite_trials if bool(trial[gate_key])]
    chosen = max(gated or finite_trials, key=_policy_key)
    chosen["selection_fallback_without_strict_pose_gate"] = not bool(gated)

    late_trials: list[dict[str, object]] = []
    late_rows: dict[str, dict[str, object]] = {}
    if bool(args.development_cross_block_audit):
        for trial in validation_trials:
            policy_key = str(trial["policy_key"])
            source_scores = raw_scores[str(policy_metadata[policy_key]["score_key"])]
            candidate_pool_scores = (
                None
                if bool(args.disable_topl_candidate_pool_verification)
                else source_scores
            )
            verified_pose, verified_rows = evaluate(
                policy_scores[policy_key],
                "test",
                "verified",
                candidate_pool_scores=candidate_pool_scores,
            )
            single_pose, single_rows = evaluate(
                policy_scores[policy_key], "test", "single_ransac"
            )
            late_trials.append(
                {
                    "trial_id": int(trial["trial_id"]),
                    "policy_key": policy_key,
                    "verified_pose": verified_pose,
                    "matched_single_ransac_pose": single_pose,
                    "passes_matched_pose_gate": _pose_gate(verified_pose, single_pose),
                    "passes_frozen_baseline_pose_gate": (
                        None
                        if frozen_baseline_source is None
                        or not isinstance(
                            frozen_baseline_source.get("late_development_pose"),
                            dict,
                        )
                        else _pose_gate(
                            verified_pose,
                            frozen_baseline_source["late_development_pose"],
                        )
                    ),
                    "selected_by_late_metrics": False,
                }
            )
            late_rows[policy_key] = {
                "verified": verified_rows,
                "matched_single_ransac": single_rows,
            }
        if frozen_baseline_source is not None:
            expected_late_pose = frozen_baseline_source.get("late_development_pose")
            if not isinstance(expected_late_pose, dict):
                raise ValueError(
                    "frozen baseline summary has no late development replay"
                )
            baseline_policy_key = f"global_bipartite__{str(args.baseline_score_key)}"
            baseline_late_trials = [
                trial
                for trial in late_trials
                if str(trial["policy_key"]) == baseline_policy_key
            ]
            if len(baseline_late_trials) != 1:
                raise ValueError("late replay is missing the frozen baseline policy")
            _validate_frozen_baseline_pose(
                baseline_late_trials[0]["matched_single_ransac_pose"],
                {"pose": expected_late_pose},
            )

    rows_path = output_dir / "pose_rows.json"
    rows_path.write_text(
        json.dumps(
            {"validation": validation_rows, "late_development": late_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = {
        "proposals": str(proposals_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "score_artifact": str(score_path),
        "score_artifact_sha256": file_sha256_short(score_path),
        "projected_landmark_bank": str(bank_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json": str(split_path),
        "split_json_sha256": file_sha256_short(split_path),
        "frozen_baseline_summary": (
            None
            if frozen_baseline_source is None
            else str(frozen_baseline_source["source_path"])
        ),
        "frozen_baseline_summary_sha256": (
            None
            if frozen_baseline_source is None
            else str(frozen_baseline_source["source_sha256"])
        ),
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
    }
    summary = {
        "stage": "heldout_multi_hypothesis_pose_verification",
        "inputs": manifest,
        "protocol": {
            "proposal_scope": "full_bank_global_faiss_top_l",
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "ground_truth_available_to_pose_selector": False,
            "pre_pnp_depth_proxy": False,
            "pre_pnp_geometry": "image_bearing_coverage_plus_world_xyz_covariance",
            "post_pose_geometry": (
                "heldout_topl_pose_guided_bipartite_assignment_plus_cheirality_plus_camera_depth"
                if not bool(args.disable_topl_candidate_pool_verification)
                else "heldout_hard_assignment_reprojection_plus_cheirality_plus_camera_depth"
            ),
            "topl_candidate_pool_verification": not bool(
                args.disable_topl_candidate_pool_verification
            ),
            "policy_selected_on_validation_only": True,
            "validation_gate_reference": (
                "matched_single_ransac"
                if frozen_baseline_source is None
                else "externally_frozen_global_baseline_exact_replay"
            ),
            "late_block_is_untouched_test": False,
            "production_promoted": False,
        },
        "config": {
            "verified_pnp": {
                **config.__dict__,
                "fit_match_counts": list(config.fit_match_counts),
                "selection_modes": list(config.selection_modes),
                "ransac_thresholds_px": list(config.ransac_thresholds_px),
                "rng_seed_offsets": list(config.rng_seed_offsets),
            },
            "single_ransac": {
                "match_count": int(args.single_ransac_match_count),
                "selection_mode": str(args.single_ransac_selection_mode),
                "threshold_px": float(args.single_ransac_threshold_px),
                "iterations": int(args.single_ransac_iterations),
            },
        },
        "validation": {
            "trial_count": int(len(validation_trials)),
            "chosen": chosen,
            "trials": validation_trials,
        },
        "late_development_replay": {
            "enabled": bool(args.development_cross_block_audit),
            "trials": late_trials,
            "development_only": True,
            "used_for_policy_selection": False,
        },
        "outputs": {
            "pose_rows": str(rows_path),
            "pose_rows_sha256": file_sha256_short(rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
