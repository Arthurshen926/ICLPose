"""Evaluate one validation-frozen candidate-maplet policy on a query set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_detector_maplet_geometry import (
    _identity_metrics,
    _pose_gate,
)
from feature_extract.tools.vfm.probe_local_assignment_support_views import (
    _evaluate_pose_strategy,
)
from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _assignment_identity_gate,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
)
from feature_extract.vfm.localization.local_assignment_linear import (
    selective_baseline_gain_switch_scores,
    selective_switch_scores,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument(
        "--ground_truth_proposals",
        default=None,
        help=(
            "optional supervised copy with identical query/candidate rows; used "
            "only after frozen score inference for metrics and pose GT"
        ),
    )
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument(
        "--score_artifact",
        default=None,
        help="required for learned policies; baseline-only replay reads proposal scores",
    )
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--frozen_policy_summary")
    policy_group.add_argument(
        "--frozen_global_policy_summary",
        help=(
            "validation-only whole-image assignment policy produced by "
            "eval_global_partial_assignment.py"
        ),
    )
    parser.add_argument(
        "--allow_source_baseline_fallback",
        action="store_true",
        help=(
            "allow a validation-frozen baseline-only global policy that failed "
            "the improvement gate; learned fallback policies remain forbidden"
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_score_key", default="baseline_scores")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument(
        "--exclude_query_ids_from_split_json",
        default=None,
        help=(
            "optional development split JSON; every query id in all list-valued "
            "fields is excluded from this frozen evaluation"
        ),
    )
    parser.add_argument(
        "--evaluation_role",
        choices=("development_reused", "untouched_test"),
        default="development_reused",
    )
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _split_query_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("excluded query split must be a JSON object")
    values = {
        str(query_id)
        for rows in payload.values()
        if isinstance(rows, list)
        for query_id in rows
    }
    if not values:
        raise ValueError("excluded query split contains no query ids")
    return values


def _compact(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return np.take_along_axis(np.asarray(values)[rows], columns, axis=1)


def _compact_candidate_scores(
    values: np.ndarray,
    *,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
    full_candidate_shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape == selected_columns.shape:
        return array.copy()
    if array.shape == full_candidate_shape:
        return _compact(array, selected_rows, selected_columns).astype(np.float32)
    raise ValueError(
        "score arrays do not align with compact or full proposal candidates"
    )


def _validate_ground_truth_join(
    inference: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray]
) -> None:
    shared_keys = (
        "query_ids",
        "xy",
        "candidate_track_ids",
        "candidate_prototype_ids",
        "coarse_scores",
        "pose_keep_mask",
    )
    missing = [
        key
        for key in shared_keys
        if key not in inference or key not in ground_truth
    ]
    if missing:
        raise ValueError(f"proposal GT join is missing shared arrays: {missing}")
    mismatches = []
    for key in shared_keys:
        left = np.asarray(inference[key])
        right = np.asarray(ground_truth[key])
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if np.issubdtype(left.dtype, np.inexact)
            else np.array_equal(left, right)
        )
        if not equal:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            f"proposal GT join changes inference identities or coordinates: {mismatches}"
        )
    required_gt = (
        "nearest_visible_track_ids",
        "nearest_visible_residuals_px",
        "candidate_gt_residuals_px",
    )
    missing_gt = [key for key in required_gt if key not in ground_truth]
    if missing_gt:
        raise ValueError(f"ground-truth proposal artifact lacks: {missing_gt}")


def _frozen_score_key(summary: dict[str, object]) -> tuple[str, dict[str, object]]:
    validation = summary.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("frozen policy summary is missing validation results")
    chosen = validation.get("chosen")
    if not isinstance(chosen, dict):
        raise ValueError("frozen policy summary is missing its chosen policy")
    if not bool(validation.get("passes_stage_gate")):
        raise ValueError("source validation policy did not pass its stage gate")
    mode = str(chosen.get("mode", ""))
    if mode not in {"unconditional", "selective"}:
        raise ValueError(f"unsupported frozen policy mode: {mode}")
    margin_threshold = chosen.get("margin_threshold")
    action_margin_threshold = chosen.get("action_margin_threshold")
    if mode == "unconditional":
        if margin_threshold is not None or action_margin_threshold is not None:
            raise ValueError(
                "frozen unconditional policy unexpectedly carries a threshold"
            )
    else:
        if (
            margin_threshold is None
            or not np.isfinite(float(margin_threshold))
            or float(margin_threshold) < 0.0
            or action_margin_threshold is not None
        ):
            raise ValueError("frozen selective policy has an invalid margin threshold")
    strategy = str(chosen.get("strategy", ""))
    if not strategy or strategy == "baseline":
        raise ValueError("frozen policy has no learned score strategy")
    return f"ensemble__{strategy}", dict(chosen)


def _frozen_global_policy(
    summary: dict[str, object], *, allow_baseline_fallback: bool = False
) -> dict[str, object]:
    validation = summary.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("frozen global policy summary is missing validation results")
    chosen = validation.get("chosen")
    if not isinstance(chosen, dict):
        raise ValueError("frozen global policy summary is missing its chosen policy")
    baseline_block = summary.get("baseline")
    frozen_baseline = (
        None
        if not isinstance(baseline_block, dict)
        else baseline_block.get("frozen_validation_policy")
    )
    if not isinstance(frozen_baseline, dict):
        raise ValueError("frozen global policy summary has no best baseline policy")
    baseline_max_matches = int(frozen_baseline.get("max_matches", 0))
    baseline_selection_mode = str(frozen_baseline.get("selection_mode", ""))
    if baseline_max_matches <= 0 or baseline_selection_mode not in {
        "score_topk",
        "spatial_round_robin",
    }:
        raise ValueError("frozen global baseline policy is invalid")
    strict_gate_passed = bool(chosen.get("passes_pose_gate")) and not bool(
        chosen.get("selection_fallback_without_strict_pose_gate")
    )
    baseline_only = False
    if not strict_gate_passed:
        if not bool(allow_baseline_fallback):
            raise ValueError("source global policy did not pass its strict pose gate")
        inputs = summary.get("inputs")
        source_baseline_key = (
            "" if not isinstance(inputs, dict) else str(inputs.get("baseline_score_key", ""))
        )
        baseline_only = bool(
            source_baseline_key
            and str(chosen.get("score_key", "")) == source_baseline_key
            and str(chosen.get("pre_global_assignment_policy", "")) == "direct_score"
            and str(chosen.get("assignment_mode", ""))
            == "whole_image_sparse_bipartite_with_per_query_dustbin"
            and chosen.get("dustbin_score") is None
            and bool(chosen.get("selection_fallback_without_strict_pose_gate"))
        )
        if not baseline_only:
            raise ValueError(
                "source fallback is learned or differs from the declared baseline"
            )

    assignment_mode = str(chosen.get("assignment_mode", ""))
    if assignment_mode not in {
        "row_argmax_then_greedy_track_conflict",
        "whole_image_sparse_bipartite_with_per_query_dustbin",
    }:
        raise ValueError(f"unsupported frozen global assignment mode: {assignment_mode}")
    selection_mode = str(chosen.get("selection_mode", ""))
    if selection_mode not in {"score_topk", "spatial_round_robin"}:
        raise ValueError(f"unsupported frozen pose selection mode: {selection_mode}")
    max_matches = int(chosen.get("max_matches", 0))
    if max_matches <= 0:
        raise ValueError("frozen global policy has an invalid match budget")

    dustbin = chosen.get("dustbin_score")
    if dustbin is not None and not np.isfinite(float(dustbin)):
        raise ValueError("frozen global policy has a non-finite dustbin score")
    if assignment_mode == "row_argmax_then_greedy_track_conflict" and dustbin is not None:
        raise ValueError("row-argmax frozen policy cannot carry a dustbin score")

    pre_policy = str(chosen.get("pre_global_assignment_policy", ""))
    if pre_policy == "direct_score":
        source_score_key = str(chosen.get("score_key", ""))
    elif pre_policy == (
        "keep_baseline_unless_candidate_probability_gain_and_baseline_invalid"
    ):
        source_score_key = str(chosen.get("source_score_key", ""))
        minimum_gain = chosen.get("minimum_gain")
        maximum_validity = chosen.get("maximum_baseline_validity")
        if (
            minimum_gain is None
            or maximum_validity is None
            or not np.isfinite(float(minimum_gain))
            or float(minimum_gain) < 0.0
            or not np.isfinite(float(maximum_validity))
            or not 0.0 <= float(maximum_validity) <= 1.0
        ):
            raise ValueError("frozen baseline-gain policy has invalid thresholds")
    else:
        raise ValueError(f"unsupported frozen pre-assignment policy: {pre_policy}")
    if not source_score_key:
        raise ValueError("frozen global policy has no source score key")

    protocol = summary.get("protocol")
    if not isinstance(protocol, dict) or not bool(
        protocol.get("policy_selected_on_validation_only")
    ):
        raise ValueError("global policy was not selected exclusively on validation")
    output = dict(chosen)
    output["source_score_key"] = source_score_key
    output["baseline_only"] = bool(baseline_only)
    output["frozen_baseline_max_matches"] = baseline_max_matches
    output["frozen_baseline_selection_mode"] = baseline_selection_mode
    return output


def _resolve_frozen_global_scores(
    *,
    policy: dict[str, object],
    source_scores: np.ndarray,
    baseline_scores: np.ndarray,
    valid_edges: np.ndarray,
    track_ids: np.ndarray,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    source = np.where(valid_edges, np.asarray(source_scores), -np.inf).astype(
        np.float32
    )
    baseline = np.where(valid_edges, np.asarray(baseline_scores), -np.inf).astype(
        np.float32
    )
    if source.shape != valid_edges.shape or baseline.shape != valid_edges.shape:
        raise ValueError("global policy scores do not align with candidate edges")

    switched = None
    pre_policy = str(policy["pre_global_assignment_policy"])
    if pre_policy == (
        "keep_baseline_unless_candidate_probability_gain_and_baseline_invalid"
    ):
        _selected, source, switched, _gains = selective_baseline_gain_switch_scores(
            source,
            baseline,
            min_gain=float(policy["minimum_gain"]),
            valid_mask=valid_edges,
            baseline_validity_scores=source,
            max_baseline_validity=float(policy["maximum_baseline_validity"]),
            preserve_baseline_row_confidence=True,
        )

    if str(policy["assignment_mode"]) == (
        "whole_image_sparse_bipartite_with_per_query_dustbin"
    ):
        source, _selected = global_assignment_score_matrix(
            track_ids,
            source,
            query_ids,
            valid_mask=valid_edges,
            dustbin_score=(
                None
                if policy.get("dustbin_score") is None
                else float(policy["dustbin_score"])
            ),
        )
    baseline, _baseline_selected = global_assignment_score_matrix(
        track_ids,
        baseline,
        query_ids,
        valid_mask=valid_edges,
        dustbin_score=None,
    )
    return source.astype(np.float32), baseline.astype(np.float32), switched


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.allow_source_baseline_fallback) and args.frozen_global_policy_summary is None:
        raise ValueError("baseline fallback is only valid for a frozen global policy")
    proposals_path = Path(args.proposals)
    ground_truth_path = (
        proposals_path
        if args.ground_truth_proposals is None
        else Path(args.ground_truth_proposals)
    )
    candidate_path = Path(args.candidate_artifact)
    score_path = None if args.score_artifact is None else Path(args.score_artifact)
    bank_path = Path(args.projected_landmark_bank)
    policy_path = Path(
        args.frozen_policy_summary
        if args.frozen_policy_summary is not None
        else args.frozen_global_policy_summary
    )
    global_policy_mode = args.frozen_global_policy_summary is not None
    excluded_split_path = (
        None
        if args.exclude_query_ids_from_split_json is None
        else Path(args.exclude_query_ids_from_split_json)
    )
    excluded_query_ids = (
        set() if excluded_split_path is None else _split_query_ids(excluded_split_path)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals = _load_npz(proposals_path)
    candidate = _load_npz(candidate_path)
    policy_summary = json.loads(policy_path.read_text())
    if global_policy_mode:
        frozen_policy = _frozen_global_policy(
            policy_summary,
            allow_baseline_fallback=bool(args.allow_source_baseline_fallback),
        )
        baseline_only = bool(frozen_policy["baseline_only"])
        score_key = str(frozen_policy["source_score_key"])
    else:
        baseline_only = False
        score_key, frozen_policy = _frozen_score_key(policy_summary)
    if score_path is None and not baseline_only:
        raise ValueError("learned frozen policy evaluation requires a score artifact")
    if baseline_only and score_path is not None:
        raise ValueError("baseline-only replay must read its score from proposals")
    scores_payload = proposals if baseline_only else _load_npz(score_path)
    effective_baseline_score_key = (
        score_key if baseline_only else str(args.baseline_score_key)
    )
    if score_key not in scores_payload or effective_baseline_score_key not in scores_payload:
        raise ValueError("score artifact lacks frozen policy or baseline scores")

    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(candidate["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(candidate["valid_edges"], dtype=bool)
    if (
        selected_columns.ndim != 2
        or valid_edges.shape != selected_columns.shape
        or np.any(selected_columns < 0)
    ):
        raise ValueError("candidate artifact must contain a fixed valid top-L pool")
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
            f"stale candidate artifact: {json.dumps(candidate_mismatches, sort_keys=True)}"
        )
    score_summary = None
    if score_path is not None:
        score_summary_path = score_path.parent / "summary.json"
        if not score_summary_path.exists():
            raise ValueError("score artifact requires sibling summary.json")
        score_summary = json.loads(score_summary_path.read_text())
        score_manifest = score_summary.get("data_manifest")
        if not isinstance(score_manifest, dict):
            raise ValueError("score summary is missing data_manifest")
        expected_scores = {
            "proposals_sha256": file_sha256_short(proposals_path),
            "feature_artifact_sha256": file_sha256_short(candidate_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        }
        score_mismatches = {
            key: {"expected": value, "actual": score_manifest.get(key)}
            for key, value in expected_scores.items()
            if score_manifest.get(key) != value
        }
        if score_mismatches:
            raise ValueError(
                f"stale score artifact: {json.dumps(score_mismatches, sort_keys=True)}"
            )
    if global_policy_mode:
        source_inputs = policy_summary.get("inputs")
        if not isinstance(source_inputs, dict):
            raise ValueError("global policy summary lacks input metadata")
        source_baseline_key = str(source_inputs.get("baseline_score_key", ""))
        if baseline_only:
            if source_baseline_key != score_key:
                raise ValueError("frozen baseline policy changed its score semantics")
        else:
            score_protocol = score_summary.get("protocol")
            if not isinstance(score_protocol, dict):
                raise ValueError("inference score summary lacks protocol metadata")
            inference_baseline_strategy = str(
                score_protocol.get("baseline_strategy", "")
            )
            if (
                not source_baseline_key
                or not inference_baseline_strategy
                or source_baseline_key != f"strategy__{inference_baseline_strategy}"
            ):
                raise ValueError(
                    "frozen global policy and inference scores use different baseline semantics"
                )

    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    tracks = _compact(
        proposals["candidate_track_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    prototypes = _compact(
        proposals["candidate_prototype_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    canonical_rows = canonical_rows_for_track_candidates(
        tracks, landmark_index.track_ids
    )
    valid_edges &= canonical_rows >= 0
    full_candidate_shape = np.asarray(proposals["candidate_track_ids"]).shape
    learned_scores = _compact_candidate_scores(
        scores_payload[score_key],
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        full_candidate_shape=full_candidate_shape,
    )
    baseline_scores = _compact_candidate_scores(
        scores_payload[effective_baseline_score_key],
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        full_candidate_shape=full_candidate_shape,
    )
    all_query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    all_query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    all_query_count = int(len(np.unique(all_query_ids)))
    all_point_count = int(len(selected_rows))
    if excluded_query_ids:
        keep = ~np.isin(all_query_ids, np.asarray(sorted(excluded_query_ids)))
        if not np.any(keep):
            raise ValueError("query exclusion removed the entire evaluation set")
        selected_rows = selected_rows[keep]
        selected_columns = selected_columns[keep]
        valid_edges = valid_edges[keep]
        tracks = tracks[keep]
        prototypes = prototypes[keep]
        canonical_rows = canonical_rows[keep]
        learned_scores = learned_scores[keep]
        baseline_scores = baseline_scores[keep]
        query_ids = all_query_ids[keep]
        query_xy = all_query_xy[keep]
    else:
        query_ids = all_query_ids
        query_xy = all_query_xy
    learned_scores = np.where(valid_edges, learned_scores, -np.inf)
    baseline_scores = np.where(valid_edges, baseline_scores, -np.inf)
    switched = None
    margins = None
    if global_policy_mode:
        learned_scores, baseline_scores, switched = _resolve_frozen_global_scores(
            policy=frozen_policy,
            source_scores=learned_scores,
            baseline_scores=baseline_scores,
            valid_edges=valid_edges,
            track_ids=tracks,
            query_ids=query_ids,
        )
    elif str(frozen_policy["mode"]) == "selective":
        _selected, learned_scores, switched, margins = selective_switch_scores(
            learned_scores,
            baseline_scores,
            margin_threshold=float(frozen_policy["margin_threshold"]),
            valid_mask=valid_edges,
            preserve_baseline_row_confidence=str(
                frozen_policy["strategy"]
            ).endswith("_prior_row_confidence"),
        )
    candidates = UniqueTrackCandidateSet(
        canonical_rows,
        tracks,
        prototypes,
        baseline_scores,
    )

    # Scores and whole-image assignment are complete before the supervised
    # proposal copy is loaded. GT can only affect metrics and pose error here.
    ground_truth = (
        proposals
        if ground_truth_path == proposals_path
        else _load_npz(ground_truth_path)
    )
    _validate_ground_truth_join(proposals, ground_truth)
    nearest_tracks = np.asarray(
        ground_truth["nearest_visible_track_ids"], dtype=np.int64
    )[selected_rows]
    observations = [
        ColmapTrackObservation(
            track_id=int(nearest_tracks[row]),
            image_id=str(query_ids[row]),
            point2d_idx=int(selected_rows[row]),
            xy=(float(query_xy[row, 0]), float(query_xy[row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for row in range(len(selected_rows))
    ]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    def pose(
        scores: np.ndarray, strategy: str, *, use_frozen_baseline_policy: bool = False
    ):
        policy_max_matches = None
        policy_selection_mode = "score_topk"
        if global_policy_mode:
            policy_max_matches = (
                int(frozen_policy["frozen_baseline_max_matches"])
                if use_frozen_baseline_policy
                else int(frozen_policy["max_matches"])
            )
            policy_selection_mode = (
                str(frozen_policy["frozen_baseline_selection_mode"])
                if use_frozen_baseline_policy
                else str(frozen_policy["selection_mode"])
            )
        return _evaluate_pose_strategy(
            strategy=strategy,
            scores=scores,
            candidates=candidates,
            query_observations=observations,
            query_ids=query_ids.tolist(),
            landmark_index=landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
            max_matches=(
                policy_max_matches
                if global_policy_mode
                else None
            ),
            pose_selection_mode=(
                policy_selection_mode
                if global_policy_mode
                else "score_topk"
            ),
        )

    residuals = _compact(
        ground_truth["candidate_gt_residuals_px"], selected_rows, selected_columns
    ).astype(np.float32)
    nearest_residuals = np.asarray(
        ground_truth["nearest_visible_residuals_px"], dtype=np.float32
    )[selected_rows]

    def identity(scores: np.ndarray) -> dict[str, object]:
        return _identity_metrics(
            nearest_residuals=nearest_residuals,
            candidate_residuals=residuals,
            scores=scores,
            query_ids=query_ids,
            labels=residuals <= 2.0,
            valid_edges=valid_edges,
        )

    baseline_pose, baseline_rows = pose(
        baseline_scores,
        "frozen_policy_baseline",
        use_frozen_baseline_policy=global_policy_mode,
    )
    learned_pose, learned_rows = pose(learned_scores, "frozen_validation_policy")
    baseline_identity = identity(baseline_scores)
    learned_identity = identity(learned_scores)
    pose_gate = _pose_gate(learned_pose, baseline_pose)
    identity_gate = _assignment_identity_gate(learned_identity, baseline_identity)
    rows_path = output_dir / "pose_rows.json"
    rows_path.write_text(
        json.dumps(
            {"baseline": baseline_rows, "frozen_policy": learned_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "stage": (
            (
                "frozen_global_baseline_pose_evaluation"
                if bool(frozen_policy.get("baseline_only"))
                else "frozen_global_candidate_maplet_pose_policy_evaluation"
            )
            if global_policy_mode
            else "frozen_candidate_maplet_pose_policy_evaluation"
        ),
        "protocol": {
            "policy_source": str(policy_path),
            "policy_source_sha256": file_sha256_short(policy_path),
            "policy_selected_on_current_query_set": False,
            "ground_truth_join_after_inference": ground_truth_path != proposals_path,
            "query_exclusion_split_json": (
                None if excluded_split_path is None else str(excluded_split_path)
            ),
            "query_exclusion_split_json_sha256": (
                None
                if excluded_split_path is None
                else file_sha256_short(excluded_split_path)
            ),
            "excluded_development_query_count_present": int(
                all_query_count - len(np.unique(query_ids))
            ),
            "inference_proposals_sha256": file_sha256_short(proposals_path),
            "ground_truth_proposals_sha256": file_sha256_short(ground_truth_path),
            "score_source": (
                "inference_proposal_baseline"
                if baseline_only
                else "external_no_gt_matcher_score_artifact"
            ),
            "score_artifact": None if score_path is None else str(score_path),
            "score_artifact_sha256": (
                None if score_path is None else file_sha256_short(score_path)
            ),
            "frozen_score_key": score_key,
            "source_baseline_fallback": bool(
                global_policy_mode and frozen_policy.get("baseline_only")
            ),
            "frozen_policy_mode": (
                "whole_image_global_assignment"
                if global_policy_mode
                else str(frozen_policy["mode"])
            ),
            "selective_switch_count": (
                None if switched is None else int(np.sum(switched))
            ),
            "selective_margin_median": (
                None
                if margins is None or not np.any(np.isfinite(margins))
                else float(np.median(margins[np.isfinite(margins)]))
            ),
            "assignment": (
                str(frozen_policy["assignment_mode"])
                if global_policy_mode
                else "row_argmax_then_unique_query_track_conflict_resolution"
            ),
            "match_budget": (
                int(frozen_policy["max_matches"])
                if global_policy_mode
                else None
            ),
            "pose_selection_mode": (
                str(frozen_policy["selection_mode"])
                if global_policy_mode
                else None
            ),
            "frozen_baseline_match_budget": (
                int(frozen_policy["frozen_baseline_max_matches"])
                if global_policy_mode
                else None
            ),
            "frozen_baseline_pose_selection_mode": (
                str(frozen_policy["frozen_baseline_selection_mode"])
                if global_policy_mode
                else None
            ),
            "pnp": "uniform_ransac_epnp_then_lm",
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "evaluation_role": str(args.evaluation_role),
            "production_promoted": False,
        },
        "frozen_policy": frozen_policy,
        "query_count": int(len(np.unique(query_ids))),
        "candidate_shape": {
            "query_point_count": int(len(selected_rows)),
            "candidate_top_l": int(selected_columns.shape[1]),
            "source_query_count_before_exclusion": all_query_count,
            "source_query_point_count_before_exclusion": all_point_count,
        },
        "baseline": {"identity": baseline_identity, "pose": baseline_pose},
        "selected": {
            "identity": learned_identity,
            "pose": learned_pose,
            "passes_pose_gate": bool(pose_gate),
            "passes_identity_gate": bool(identity_gate),
            "passes_stage_gate": bool(pose_gate and identity_gate),
        },
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
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
