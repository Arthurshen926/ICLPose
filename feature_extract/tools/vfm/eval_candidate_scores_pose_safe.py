"""Select a pose-safe match budget on validation and replay it once on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_detector_maplet_geometry import _identity_metrics, _pose_gate
from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
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
from feature_extract.vfm.localization.local_assignment_linear import (
    selective_baseline_gain_switch_scores,
    selective_switch_scores,
)
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)


def _positive_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed or min(parsed) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def _finite_float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if any(not np.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("expected finite comma-separated floats")
    return parsed


def _score_key_map(value: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in (part.strip() for part in str(value).split(",")):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError("score maps must use assignment_key=selection_key")
        assignment_key, selection_key = (part.strip() for part in item.split("=", 1))
        if not assignment_key or not selection_key or assignment_key in output:
            raise argparse.ArgumentTypeError("score maps require unique non-empty keys")
        output[assignment_key] = selection_key
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument(
        "--candidate_artifact",
        default=None,
        help="optional artifact providing selected_rows/selected_columns",
    )
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_keys", required=True)
    parser.add_argument("--baseline_score_key", default="baseline_scores")
    parser.add_argument(
        "--selection_score_map",
        type=_score_key_map,
        default={},
        help=(
            "comma-separated assignment_key=selection_key mappings; candidate identity is "
            "resolved with the first score while conflict resolution and pose filtering use "
            "the second"
        ),
    )
    parser.add_argument(
        "--selective_switch_score_keys",
        default="",
        help=(
            "optional raw reranker arrays used only through conservative baseline-to-reranker "
            "switch policies"
        ),
    )
    parser.add_argument(
        "--switch_margin_thresholds",
        type=_finite_float_list,
        default=(0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7),
    )
    parser.add_argument(
        "--baseline_gain_switch_score_keys",
        default="",
        help="optional reranker arrays used with best-minus-baseline candidate gain",
    )
    parser.add_argument(
        "--baseline_gain_thresholds",
        type=_finite_float_list,
        default=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3),
    )
    parser.add_argument(
        "--baseline_validity_score_map",
        type=_score_key_map,
        default={},
        help="candidate_score_key=baseline_validity_probability_key mappings",
    )
    parser.add_argument(
        "--max_baseline_validity_thresholds",
        type=_finite_float_list,
        default=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    )
    parser.add_argument(
        "--max_validation_switch_worsen_rate",
        type=float,
        default=1.0,
        help="maximum validation residual-worsening rate for an assignment policy",
    )
    parser.add_argument(
        "--min_validation_switch_improve_rate",
        type=float,
        default=0.0,
        help="minimum validation residual-improvement rate for an assignment policy",
    )
    parser.add_argument("--match_counts", type=_positive_int_list, default=(24, 32, 48, 64, 96, 128))
    parser.add_argument("--selection_modes", default="score_topk,spatial_round_robin")
    parser.add_argument(
        "--adaptive_confidence_thresholds",
        type=_finite_float_list,
        default=(),
        help="optional calibrated confidence thresholds for adaptive match counts",
    )
    parser.add_argument(
        "--adaptive_min_match_counts",
        type=_positive_int_list,
        default=(12, 16, 20),
    )
    parser.add_argument("--adaptive_max_matches", type=int, default=32)
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
        help="reused development data can never produce a production promotion",
    )
    parser.add_argument(
        "--development_cross_block_audit",
        action="store_true",
        help=(
            "diagnostically replay every policy on both reused development blocks; this "
            "invalidates the late block as a held-out test and cannot promote production"
        ),
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _compact(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return np.take_along_axis(np.asarray(values)[rows], columns, axis=1)


def _switch_residual_audit(
    residuals_px: np.ndarray,
    baseline_scores: np.ndarray,
    selected_scores: np.ndarray,
    *,
    row_mask: np.ndarray,
    correctness_threshold_px: float = 5.0,
) -> dict[str, object]:
    """Audit UPDATE risk from GT residuals without feeding GT into inference."""

    residuals = np.asarray(residuals_px, dtype=np.float32)
    baseline = np.asarray(baseline_scores, dtype=np.float32)
    selected = np.asarray(selected_scores, dtype=np.float32)
    rows = np.asarray(row_mask, dtype=bool).reshape(-1)
    if (
        residuals.ndim != 2
        or baseline.shape != residuals.shape
        or selected.shape != residuals.shape
        or rows.shape != (residuals.shape[0],)
    ):
        raise ValueError("switch residual audit inputs have incompatible shapes")
    threshold = float(correctness_threshold_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("correctness threshold must be finite and positive")
    baseline_choice = np.argmax(
        np.where(np.isfinite(baseline), baseline, -np.inf), axis=1
    )
    selected_choice = np.argmax(
        np.where(np.isfinite(selected), selected, -np.inf), axis=1
    )
    switched = rows & (selected_choice != baseline_choice)
    indices = np.arange(len(residuals), dtype=np.int64)
    baseline_residual = residuals[indices, baseline_choice]
    selected_residual = residuals[indices, selected_choice]
    improved = selected_residual < baseline_residual
    worsened = selected_residual > baseline_residual
    true_rescue = (baseline_residual > threshold) & (
        selected_residual <= threshold
    )
    finite_delta = switched & np.isfinite(baseline_residual) & np.isfinite(
        selected_residual
    )
    deltas = selected_residual[finite_delta] - baseline_residual[finite_delta]
    switch_count = int(np.sum(switched))
    return {
        "switch_count": switch_count,
        "true_rescue_count": int(np.sum(switched & true_rescue)),
        "true_rescue_precision": float(
            np.sum(switched & true_rescue) / max(switch_count, 1)
        ),
        "improved_count": int(np.sum(switched & improved)),
        "improved_rate": float(np.sum(switched & improved) / max(switch_count, 1)),
        "worsened_count": int(np.sum(switched & worsened)),
        "worsened_rate": float(np.sum(switched & worsened) / max(switch_count, 1)),
        "baseline_valid_false_switch_count": int(
            np.sum(switched & (baseline_residual <= threshold))
        ),
        "finite_residual_delta_count": int(len(deltas)),
        "median_selected_minus_baseline_residual_px": (
            None if len(deltas) == 0 else float(np.median(deltas))
        ),
        "mean_selected_minus_baseline_residual_px": (
            None if len(deltas) == 0 else float(np.mean(deltas))
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = Path(args.proposals)
    scores_path = Path(args.score_artifact)
    candidate_path = (
        scores_path if args.candidate_artifact is None else Path(args.candidate_artifact)
    )
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    proposals = _load_npz(proposals_path)
    score_payload = _load_npz(scores_path)
    candidate_payload = (
        score_payload if candidate_path == scores_path else _load_npz(candidate_path)
    )
    split = json.loads(split_path.read_text())
    score_keys = tuple(value.strip() for value in str(args.score_keys).split(",") if value.strip())
    selective_switch_score_keys = tuple(
        value.strip()
        for value in str(args.selective_switch_score_keys).split(",")
        if value.strip()
    )
    baseline_gain_switch_score_keys = tuple(
        value.strip()
        for value in str(args.baseline_gain_switch_score_keys).split(",")
        if value.strip()
    )
    selection_modes = tuple(
        value.strip() for value in str(args.selection_modes).split(",") if value.strip()
    )
    if not score_keys or any(key not in score_payload for key in score_keys):
        raise ValueError("score_keys must name arrays in score_artifact")
    missing_switch_keys = set(selective_switch_score_keys) - set(score_payload)
    if missing_switch_keys:
        raise ValueError(
            f"selective switch score arrays are missing: {sorted(missing_switch_keys)}"
        )
    missing_gain_keys = set(baseline_gain_switch_score_keys) - set(score_payload)
    if missing_gain_keys:
        raise ValueError(
            f"baseline-gain switch score arrays are missing: {sorted(missing_gain_keys)}"
        )
    baseline_validity_score_map = dict(args.baseline_validity_score_map)
    unknown_validity_sources = set(baseline_validity_score_map) - set(
        baseline_gain_switch_score_keys
    )
    if unknown_validity_sources:
        raise ValueError(
            "baseline_validity_score_map contains unused sources: "
            f"{sorted(unknown_validity_sources)}"
        )
    missing_validity_keys = set(baseline_validity_score_map.values()) - set(score_payload)
    if missing_validity_keys:
        raise ValueError(
            f"baseline validity score arrays are missing: {sorted(missing_validity_keys)}"
        )
    if any(float(value) < 0.0 for value in tuple(args.baseline_gain_thresholds)):
        raise ValueError("baseline gain thresholds must be non-negative")
    if any(
        not 0.0 <= float(value) <= 1.0
        for value in tuple(args.max_baseline_validity_thresholds)
    ):
        raise ValueError("maximum baseline validity thresholds must be in [0, 1]")
    if not 0.0 <= float(args.max_validation_switch_worsen_rate) <= 1.0:
        raise ValueError("maximum validation switch worsen rate must be in [0, 1]")
    if not 0.0 <= float(args.min_validation_switch_improve_rate) <= 1.0:
        raise ValueError("minimum validation switch improve rate must be in [0, 1]")
    if args.baseline_score_key not in score_payload:
        raise ValueError("baseline score array is missing")
    selection_score_map = dict(args.selection_score_map)
    unknown_assignment_keys = set(selection_score_map) - set(score_keys)
    if unknown_assignment_keys:
        raise ValueError(
            f"selection_score_map contains unrequested assignment keys: {sorted(unknown_assignment_keys)}"
        )
    missing_selection_keys = set(selection_score_map.values()) - set(score_payload)
    if missing_selection_keys:
        raise ValueError(
            f"selection_score_map arrays are missing: {sorted(missing_selection_keys)}"
        )
    if not selection_modes or set(selection_modes) - {"score_topk", "spatial_round_robin"}:
        raise ValueError("unsupported pose-safe selection mode")
    if int(args.adaptive_max_matches) <= 0:
        raise ValueError("adaptive_max_matches must be positive")
    if bool(args.development_cross_block_audit) and args.evaluation_role != "development":
        raise ValueError("development_cross_block_audit is forbidden for untouched_test evaluation")
    if any(
        int(count) > int(args.adaptive_max_matches)
        for count in tuple(args.adaptive_min_match_counts)
    ):
        raise ValueError("adaptive minimum match counts cannot exceed adaptive_max_matches")

    selected_rows = np.asarray(candidate_payload["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(candidate_payload["selected_columns"], dtype=np.int64)
    if selected_columns.ndim != 2 or np.any(selected_columns < 0):
        raise ValueError("score artifact requires a fixed valid candidate pool")
    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    query_ids_all = np.asarray(proposals["query_ids"]).astype(str)
    query_xy_all = np.asarray(proposals["xy"], dtype=np.float32)
    query_ids = query_ids_all[selected_rows]
    query_xy = query_xy_all[selected_rows]
    candidate_tracks = _compact(
        proposals["candidate_track_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    candidate_prototypes = _compact(
        proposals["candidate_prototype_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    canonical_rows = canonical_rows_for_track_candidates(
        candidate_tracks, landmark_index.track_ids
    )
    baseline_scores = np.asarray(score_payload[args.baseline_score_key], dtype=np.float32)
    if baseline_scores.shape != selected_columns.shape:
        raise ValueError("baseline scores do not align with selected candidates")
    candidates = UniqueTrackCandidateSet(
        canonical_rows,
        candidate_tracks,
        candidate_prototypes,
        baseline_scores,
    )
    residuals = _compact(
        proposals["candidate_gt_residuals_px"], selected_rows, selected_columns
    ).astype(np.float32)
    nearest_residuals = np.asarray(
        proposals["nearest_visible_residuals_px"], dtype=np.float32
    )[selected_rows]
    labels = residuals <= 2.0
    valid_edges = canonical_rows >= 0
    nearest_tracks = np.asarray(proposals["nearest_visible_track_ids"], dtype=np.int64)[
        selected_rows
    ]
    observations = [
        ColmapTrackObservation(
            track_id=int(nearest_tracks[index]),
            image_id=str(query_ids[index]),
            point2d_idx=int(selected_rows[index]),
            xy=(float(query_xy[index, 0]), float(query_xy[index, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for index in range(len(selected_rows))
    ]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    split_masks = {
        name: np.isin(query_ids, np.asarray(split[name], dtype=np.str_))
        for name in ("validation", "test")
    }

    def identity(scores: np.ndarray, split_name: str) -> dict[str, object]:
        mask = split_masks[split_name]
        return _identity_metrics(
            nearest_residuals=nearest_residuals[mask],
            candidate_residuals=residuals[mask],
            scores=scores[mask],
            query_ids=query_ids[mask],
            labels=labels[mask],
            valid_edges=valid_edges[mask],
        )

    def pose(
        scores: np.ndarray,
        split_name: str,
        strategy: str,
        *,
        selection_scores: np.ndarray | None = None,
        max_matches: int | None = None,
        selection_mode: str = "score_topk",
        min_matches: int | None = None,
        min_selection_score: float | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        rows = np.flatnonzero(split_masks[split_name])
        subset = UniqueTrackCandidateSet(
            candidates.bank_row_indices[rows],
            candidates.track_ids[rows],
            candidates.prototype_ids[rows],
            candidates.coarse_scores[rows],
        )
        return _evaluate_pose_strategy(
            strategy=str(strategy),
            scores=scores[rows],
            selection_scores=(
                None if selection_scores is None else selection_scores[rows]
            ),
            candidates=subset,
            query_observations=[observations[int(row)] for row in rows],
            query_ids=query_ids[rows].tolist(),
            landmark_index=landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
            max_matches=max_matches,
            pose_selection_mode=str(selection_mode),
            min_matches=min_matches,
            min_selection_score=min_selection_score,
        )

    baseline_validation_identity = identity(baseline_scores, "validation")
    baseline_validation_pose, baseline_validation_rows = pose(
        baseline_scores, "validation", "baseline_all_validation"
    )
    validation_trials: list[dict[str, object]] = []
    validation_trial_rows: list[dict[str, object]] = []
    all_scores = {key: np.asarray(score_payload[key], dtype=np.float32) for key in score_keys}
    all_selection_scores = {
        key: np.asarray(score_payload[selection_score_map.get(key, key)], dtype=np.float32)
        for key in score_keys
    }
    strategy_metadata: dict[str, dict[str, object]] = {
        key: {
            "type": "direct_score_array",
            "source_score_key": str(key),
            "selection_score_key": str(selection_score_map.get(key, key)),
        }
        for key in score_keys
    }
    for raw_key in selective_switch_score_keys:
        raw_scores = np.asarray(score_payload[raw_key], dtype=np.float32)
        if raw_scores.shape != selected_columns.shape:
            raise ValueError(f"selective switch score array has incompatible shape: {raw_key}")
        for margin in tuple(args.switch_margin_thresholds):
            _selected, resolved, switched, _margins = selective_switch_scores(
                raw_scores,
                baseline_scores,
                margin_threshold=float(margin),
                valid_mask=valid_edges,
                preserve_baseline_row_confidence=True,
            )
            margin_name = f"{float(margin):g}".replace("-", "m").replace(".", "p")
            derived_key = f"selective__{raw_key}__margin_{margin_name}"
            all_scores[derived_key] = resolved
            all_selection_scores[derived_key] = raw_scores
            strategy_metadata[derived_key] = {
                "type": "keep_baseline_unless_reranker_margin",
                "source_score_key": str(raw_key),
                "selection_score_key": str(raw_key),
                "margin_threshold": float(margin),
                "switch_count_all": int(np.sum(switched)),
                "switch_count_validation": int(np.sum(switched & split_masks["validation"])),
                "switch_count_test": int(np.sum(switched & split_masks["test"])),
            }
    for raw_key in baseline_gain_switch_score_keys:
        raw_scores = np.asarray(score_payload[raw_key], dtype=np.float32)
        if raw_scores.shape != selected_columns.shape:
            raise ValueError(f"baseline-gain score array has incompatible shape: {raw_key}")
        validity_key = baseline_validity_score_map.get(raw_key, raw_key)
        validity_scores = np.asarray(score_payload[validity_key], dtype=np.float32)
        if validity_scores.shape != selected_columns.shape:
            raise ValueError(
                f"baseline validity score array has incompatible shape: {validity_key}"
            )
        for gain_threshold in tuple(args.baseline_gain_thresholds):
            for validity_threshold in tuple(args.max_baseline_validity_thresholds):
                _selected, resolved, switched, _gains = (
                    selective_baseline_gain_switch_scores(
                        raw_scores,
                        baseline_scores,
                        min_gain=float(gain_threshold),
                        valid_mask=valid_edges,
                        baseline_validity_scores=validity_scores,
                        max_baseline_validity=float(validity_threshold),
                        preserve_baseline_row_confidence=True,
                    )
                )
                gain_name = f"{float(gain_threshold):g}".replace("-", "m").replace(".", "p")
                validity_name = f"{float(validity_threshold):g}".replace("-", "m").replace(".", "p")
                derived_key = (
                    f"baseline_gain__{raw_key}__gain_{gain_name}"
                    f"__validity_{validity_key}__max_{validity_name}"
                )
                all_scores[derived_key] = resolved
                all_selection_scores[derived_key] = raw_scores
                strategy_metadata[derived_key] = {
                    "type": "keep_baseline_unless_reranker_beats_baseline",
                    "source_score_key": str(raw_key),
                    "selection_score_key": str(raw_key),
                    "baseline_validity_score_key": str(validity_key),
                    "minimum_gain": float(gain_threshold),
                    "maximum_baseline_validity": float(validity_threshold),
                    "switch_count_all": int(np.sum(switched)),
                    "switch_count_validation": int(
                        np.sum(switched & split_masks["validation"])
                    ),
                    "switch_count_test": int(np.sum(switched & split_masks["test"])),
                }

    def add_validation_trial(
        *,
        score_key: str,
        scores: np.ndarray,
        confidence_key: str,
        confidences: np.ndarray,
        score_identity: dict[str, object],
        max_matches: int,
        selection_mode: str,
        min_matches: int | None = None,
        min_selection_score: float | None = None,
    ) -> None:
        trial_id = len(validation_trials)
        policy = (
            f"{score_key}_{selection_mode}_max{int(max_matches)}"
            f"_min{'none' if min_matches is None else int(min_matches)}"
            f"_threshold{'none' if min_selection_score is None else f'{float(min_selection_score):g}'}"
        )
        pose_summary, pose_rows = pose(
            scores,
            "validation",
            f"{policy}_validation",
            selection_scores=confidences,
            max_matches=int(max_matches),
            selection_mode=str(selection_mode),
            min_matches=min_matches,
            min_selection_score=min_selection_score,
        )
        switch_residual_audit = _switch_residual_audit(
            residuals,
            baseline_scores,
            scores,
            row_mask=split_masks["validation"],
        )
        switch_count = int(switch_residual_audit["switch_count"])
        passes_switch_residual_gate = bool(
            switch_count == 0
            or (
                float(switch_residual_audit["worsened_rate"])
                <= float(args.max_validation_switch_worsen_rate)
                and float(switch_residual_audit["improved_rate"])
                >= float(args.min_validation_switch_improve_rate)
            )
        )
        validation_trials.append(
            {
                "trial_id": int(trial_id),
                "score_key": str(score_key),
                "selection_score_key": str(confidence_key),
                "assignment_policy": strategy_metadata[str(score_key)],
                "max_matches": int(max_matches),
                "min_matches": None if min_matches is None else int(min_matches),
                "min_selection_score": (
                    None
                    if min_selection_score is None
                    else float(min_selection_score)
                ),
                "selection_mode": str(selection_mode),
                "identity": score_identity,
                "pose": pose_summary,
                "switch_residual_audit": switch_residual_audit,
                "passes_switch_residual_gate": passes_switch_residual_gate,
                "passes_pose_gate": _pose_gate(
                    pose_summary, baseline_validation_pose
                ),
                "passes_identity_gate": _assignment_identity_gate(
                    score_identity, baseline_validation_identity
                ),
            }
        )
        validation_trial_rows.append(
            {
                "trial_id": int(trial_id),
                "policy": str(policy),
                "rows": pose_rows,
            }
        )

    for score_key, scores in all_scores.items():
        if scores.shape != selected_columns.shape:
            raise ValueError(f"score array has incompatible shape: {score_key}")
        selection_scores = all_selection_scores[score_key]
        if selection_scores.shape != selected_columns.shape:
            raise ValueError(f"selection score array has incompatible shape: {score_key}")
        selection_score_key = str(strategy_metadata[score_key]["selection_score_key"])
        score_identity = identity(scores, "validation")
        for count in tuple(args.match_counts):
            for mode in selection_modes:
                add_validation_trial(
                    score_key=str(score_key),
                    scores=scores,
                    confidence_key=str(selection_score_key),
                    confidences=selection_scores,
                    max_matches=int(count),
                    selection_mode=str(mode),
                    score_identity=score_identity,
                )
        for threshold in tuple(args.adaptive_confidence_thresholds):
            for minimum in tuple(args.adaptive_min_match_counts):
                for mode in selection_modes:
                    add_validation_trial(
                        score_key=str(score_key),
                        scores=scores,
                        confidence_key=str(selection_score_key),
                        confidences=selection_scores,
                        max_matches=int(args.adaptive_max_matches),
                        selection_mode=str(mode),
                        min_matches=int(minimum),
                        min_selection_score=float(threshold),
                        score_identity=score_identity,
                    )
    for trial in validation_trials:
        trial["passes_stage_gate"] = bool(
            trial["passes_pose_gate"]
            and trial["passes_identity_gate"]
            and trial["passes_switch_residual_gate"]
        )
    eligible = [trial for trial in validation_trials if bool(trial["passes_stage_gate"])]
    if eligible:
        chosen = max(
            eligible,
            key=lambda trial: (
                float(trial["pose"]["recall_10cm_5deg"]),
                float(trial["pose"]["recall_5cm_5deg"]),
                -float(trial["pose"]["median_translation_m_success"]),
                -float(trial["pose"]["p90_translation_m_success"]),
                -float(trial["max_matches"]),
            ),
        )
        chosen_scores = all_scores[str(chosen["score_key"])]
        chosen_selection_scores = all_selection_scores[str(chosen["score_key"])]
        chosen_validation_rows = validation_trial_rows[int(chosen["trial_id"])]["rows"]
        test_pose, test_rows = pose(
            chosen_scores,
            "test",
            "validation_selected_pose_safe_test",
            selection_scores=chosen_selection_scores,
            max_matches=int(chosen["max_matches"]),
            selection_mode=str(chosen["selection_mode"]),
            min_matches=chosen["min_matches"],
            min_selection_score=chosen["min_selection_score"],
        )
        test_identity = identity(chosen_scores, "test")
    else:
        chosen_scores = baseline_scores
        chosen_selection_scores = baseline_scores
        chosen = {
            "score_key": str(args.baseline_score_key),
            "selection_score_key": str(args.baseline_score_key),
            "max_matches": None,
            "min_matches": None,
            "min_selection_score": None,
            "selection_mode": "all",
            "identity": baseline_validation_identity,
            "pose": baseline_validation_pose,
            "switch_residual_audit": _switch_residual_audit(
                residuals,
                baseline_scores,
                baseline_scores,
                row_mask=split_masks["validation"],
            ),
            "passes_switch_residual_gate": True,
            "passes_pose_gate": False,
            "passes_identity_gate": True,
            "passes_stage_gate": False,
        }
        chosen_validation_rows = baseline_validation_rows
        test_pose, test_rows = pose(baseline_scores, "test", "baseline_all_test")
        test_identity = identity(baseline_scores, "test")
    baseline_test_pose, baseline_test_rows = pose(
        baseline_scores, "test", "baseline_all_test_reference"
    )
    baseline_test_identity = identity(baseline_scores, "test")
    test_pose_gate = _pose_gate(test_pose, baseline_test_pose)
    test_identity_gate = _assignment_identity_gate(
        test_identity, baseline_test_identity
    )
    test_switch_residual_audit = _switch_residual_audit(
        residuals,
        baseline_scores,
        chosen_scores,
        row_mask=split_masks["test"],
    )
    test_gate = bool(test_pose_gate and test_identity_gate)
    cross_block_trials: list[dict[str, object]] = []
    cross_block_pose_rows: list[dict[str, object]] = []
    cross_block_chosen: dict[str, object] | None = None
    if bool(args.development_cross_block_audit):
        for validation_trial in validation_trials:
            score_key = str(validation_trial["score_key"])
            late_pose, late_rows = pose(
                all_scores[score_key],
                "test",
                f"cross_block_{int(validation_trial['trial_id'])}_late_development",
                selection_scores=all_selection_scores[score_key],
                max_matches=int(validation_trial["max_matches"]),
                selection_mode=str(validation_trial["selection_mode"]),
                min_matches=validation_trial["min_matches"],
                min_selection_score=validation_trial["min_selection_score"],
            )
            late_identity = identity(all_scores[score_key], "test")
            late_pose_passed = _pose_gate(late_pose, baseline_test_pose)
            late_identity_passed = _assignment_identity_gate(
                late_identity, baseline_test_identity
            )
            late_switch_residual_audit = _switch_residual_audit(
                residuals,
                baseline_scores,
                all_scores[score_key],
                row_mask=split_masks["test"],
            )
            both_passed = bool(
                validation_trial["passes_stage_gate"]
                and late_pose_passed
                and late_identity_passed
            )
            cross_block_trials.append(
                {
                    "trial_id": int(validation_trial["trial_id"]),
                    "score_key": score_key,
                    "selection_score_key": str(validation_trial["selection_score_key"]),
                    "assignment_policy": validation_trial["assignment_policy"],
                    "max_matches": int(validation_trial["max_matches"]),
                    "min_matches": validation_trial["min_matches"],
                    "min_selection_score": validation_trial["min_selection_score"],
                    "selection_mode": str(validation_trial["selection_mode"]),
                    "validation_pose": validation_trial["pose"],
                    "validation_switch_residual_audit": validation_trial[
                        "switch_residual_audit"
                    ],
                    "late_development_identity": late_identity,
                    "late_development_pose": late_pose,
                    "late_development_switch_residual_audit": (
                        late_switch_residual_audit
                    ),
                    "late_development_pose_passed": bool(late_pose_passed),
                    "late_development_identity_passed": bool(late_identity_passed),
                    "both_development_blocks_passed": both_passed,
                }
            )
            cross_block_pose_rows.append(
                {
                    "trial_id": int(validation_trial["trial_id"]),
                    "rows": late_rows,
                }
            )
        cross_block_eligible = [
            trial
            for trial in cross_block_trials
            if bool(trial["both_development_blocks_passed"])
        ]
        if cross_block_eligible:
            cross_block_chosen = max(
                cross_block_eligible,
                key=lambda trial: (
                    min(
                        float(trial["validation_pose"]["recall_10cm_5deg"]),
                        float(trial["late_development_pose"]["recall_10cm_5deg"]),
                    ),
                    min(
                        float(trial["validation_pose"]["recall_25cm_2deg"]),
                        float(trial["late_development_pose"]["recall_25cm_2deg"]),
                    ),
                    -max(
                        float(trial["validation_pose"]["median_translation_m_success"])
                        / float(baseline_validation_pose["median_translation_m_success"]),
                        float(trial["late_development_pose"]["median_translation_m_success"])
                        / float(baseline_test_pose["median_translation_m_success"]),
                    ),
                    -max(
                        float(trial["validation_pose"]["p90_translation_m_success"])
                        / float(baseline_validation_pose["p90_translation_m_success"]),
                        float(trial["late_development_pose"]["p90_translation_m_success"])
                        / float(baseline_test_pose["p90_translation_m_success"]),
                    ),
                ),
            )
    candidate_valid = canonical_rows >= 0
    safe_chosen_scores = np.where(
        candidate_valid & np.isfinite(chosen_scores), chosen_scores, -np.inf
    )
    if np.any(np.sum(np.isfinite(safe_chosen_scores), axis=1) <= 0):
        raise ValueError("chosen policy leaves at least one query point without an identity")
    chosen_columns = np.argmax(safe_chosen_scores, axis=1).astype(np.int64)
    baseline_columns = np.argmax(
        np.where(candidate_valid & np.isfinite(baseline_scores), baseline_scores, -np.inf),
        axis=1,
    ).astype(np.int64)
    row_indices = np.arange(len(chosen_columns), dtype=np.int64)
    support_view_keys = tuple(
        sorted(
            key
            for key in score_payload
            if key.startswith("ensemble__support_view_probability_")
        )
    )
    selected_support_view_probabilities = np.zeros(
        (len(chosen_columns), len(support_view_keys)), dtype=np.float32
    )
    for view_index, key in enumerate(support_view_keys):
        values = np.asarray(score_payload[key], dtype=np.float32)
        if values.shape != selected_columns.shape:
            raise ValueError(f"support-view probability has incompatible shape: {key}")
        selected_support_view_probabilities[:, view_index] = values[
            row_indices, chosen_columns
        ]
    geometry_probability_keys = tuple(
        key
        for key in (
            "ensemble__geometry_p01px",
            "ensemble__geometry_p02px",
            "ensemble__geometry_p05px",
        )
        if key in score_payload
    )
    selected_geometry_probabilities = np.zeros(
        (len(chosen_columns), len(geometry_probability_keys)), dtype=np.float32
    )
    for probability_index, key in enumerate(geometry_probability_keys):
        values = np.asarray(score_payload[key], dtype=np.float32)
        if values.shape != selected_columns.shape:
            raise ValueError(f"geometry probability has incompatible shape: {key}")
        selected_geometry_probabilities[:, probability_index] = values[
            row_indices, chosen_columns
        ]
    policy_artifact_path = output_dir / "selected_policy_artifact.npz"
    candidate_metadata = (
        json.loads(str(candidate_payload["metadata_json"].item()))
        if "metadata_json" in candidate_payload
        else {}
    )
    policy_metadata = {
        "format": "pose_safe_selected_policy_v1",
        "evaluation_role": str(args.evaluation_role),
        "chosen_score_key": str(chosen["score_key"]),
        "chosen_selection_score_key": str(chosen["selection_score_key"]),
        "assignment_policy": chosen.get("assignment_policy"),
        "max_matches": chosen["max_matches"],
        "selection_mode": chosen["selection_mode"],
        "proposals_sha256": file_sha256_short(proposals_path),
        "score_artifact_sha256": file_sha256_short(scores_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
        "support_geometry_index_sha256": candidate_metadata.get(
            "support_geometry_index_sha256"
        ),
        "maplet_support_index_sha256": candidate_metadata.get(
            "maplet_support_index_sha256"
        ),
        "support_view_probability_keys": list(support_view_keys),
        "geometry_probability_keys": list(geometry_probability_keys),
        "gt_residuals_are_target_only": True,
    }
    np.savez(
        policy_artifact_path,
        selected_rows=selected_rows,
        candidate_pool_columns=selected_columns,
        selected_compact_columns=chosen_columns,
        selected_source_columns=selected_columns[row_indices, chosen_columns],
        baseline_compact_columns=baseline_columns,
        switched_from_baseline=chosen_columns != baseline_columns,
        query_ids=query_ids.astype(np.str_),
        query_xy=query_xy.astype(np.float32),
        selected_track_ids=candidate_tracks[row_indices, chosen_columns],
        selected_prototype_ids=candidate_prototypes[row_indices, chosen_columns],
        selected_canonical_rows=canonical_rows[row_indices, chosen_columns],
        selected_assignment_scores=chosen_scores[row_indices, chosen_columns],
        selected_pose_selection_scores=chosen_selection_scores[
            row_indices, chosen_columns
        ],
        selected_gt_residuals_px=residuals[row_indices, chosen_columns],
        selected_geometry_probabilities=selected_geometry_probabilities,
        selected_support_view_probabilities=selected_support_view_probabilities,
        metadata_json=np.asarray(json.dumps(policy_metadata, sort_keys=True), dtype=np.str_),
    )
    summary = {
        "stage": "candidate_score_pose_safe_validation_selection",
        "protocol": {
            "evaluation_role": str(args.evaluation_role),
            "development_data_reused": bool(args.evaluation_role == "development"),
            "development_cross_block_audit": bool(args.development_cross_block_audit),
            "test_used_for_selection": False,
            "query_conflicts_resolved": True,
            "track_conflicts_resolved": True,
            "assignment_and_pose_selection_scores_decoupled": True,
            "selective_switch": {
                "default_action": "keep_baseline",
                "score_keys": [str(key) for key in selective_switch_score_keys],
                "margin_thresholds": [
                    float(value) for value in tuple(args.switch_margin_thresholds)
                ],
            },
            "baseline_gain_switch": {
                "default_action": "keep_baseline",
                "score_keys": [str(key) for key in baseline_gain_switch_score_keys],
                "minimum_gain_thresholds": [
                    float(value) for value in tuple(args.baseline_gain_thresholds)
                ],
                "baseline_validity_score_map": baseline_validity_score_map,
                "maximum_baseline_validity_thresholds": [
                    float(value)
                    for value in tuple(args.max_baseline_validity_thresholds)
                ],
            },
            "validation_switch_residual_gate": {
                "target_only_not_inference_input": True,
                "correctness_threshold_px": 5.0,
                "maximum_worsen_rate": float(
                    args.max_validation_switch_worsen_rate
                ),
                "minimum_improve_rate": float(
                    args.min_validation_switch_improve_rate
                ),
            },
            "uniform_ransac_canonical_input_order": True,
            "adaptive_selection": {
                "confidence_thresholds": [
                    float(value) for value in tuple(args.adaptive_confidence_thresholds)
                ],
                "minimum_match_counts": [
                    int(value) for value in tuple(args.adaptive_min_match_counts)
                ],
                "maximum_match_count": int(args.adaptive_max_matches),
            },
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "pnp_reprojection_error_px": float(args.pnp_reprojection_error_px),
            "pnp_iterations": int(args.pnp_iterations),
        },
        "inputs": {
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "score_artifact": str(scores_path),
            "score_artifact_sha256": file_sha256_short(scores_path),
            "candidate_artifact": str(candidate_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
            "split_json": str(split_path),
            "split_json_sha256": file_sha256_short(split_path),
        },
        "validation": {
            "baseline": {
                "identity": baseline_validation_identity,
                "pose": baseline_validation_pose,
            },
            "trials": validation_trials,
            "chosen": chosen,
            "passes_pose_gate": bool(
                any(bool(trial["passes_pose_gate"]) for trial in validation_trials)
            ),
            "passes_stage_gate": bool(eligible),
        },
        "test": {
            "baseline": {
                "identity": baseline_test_identity,
                "pose": baseline_test_pose,
            },
            "selected": {
                "identity": test_identity,
                "pose": test_pose,
                "switch_residual_audit": test_switch_residual_audit,
                "passes_pose_gate": bool(test_pose_gate),
                "passes_identity_gate": bool(test_identity_gate),
                "passes_stage_gate": bool(test_gate),
            },
        },
        "development_cross_block": {
            "enabled": bool(args.development_cross_block_audit),
            "late_block_is_held_out_test": False
            if bool(args.development_cross_block_audit)
            else bool(args.evaluation_role == "untouched_test"),
            "trial_count": int(len(cross_block_trials)),
            "trials": cross_block_trials,
            "chosen": cross_block_chosen,
            "learned_policy_passed_both_blocks": bool(cross_block_chosen is not None),
        },
        "gate": {
            "validation_passed": bool(eligible),
            "test_passed": bool(test_gate),
            "test_pose_passed": bool(test_pose_gate),
            "test_identity_passed": bool(test_identity_gate),
            "evaluation_role_allows_production_promotion": bool(
                args.evaluation_role == "untouched_test"
            ),
            "development_cross_block_passed": bool(cross_block_chosen is not None),
            "production_promoted": bool(
                args.evaluation_role == "untouched_test" and eligible and test_gate
            ),
        },
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "pose_rows": str(output_dir / "pose_rows.json"),
            "validation_trial_pose_rows": str(
                output_dir / "validation_trial_pose_rows.json"
            ),
            "selected_policy_artifact": str(policy_artifact_path),
            "selected_policy_artifact_sha256": file_sha256_short(
                policy_artifact_path
            ),
        },
    }
    (output_dir / "pose_rows.json").write_text(
        json.dumps(
            {
                "validation_baseline": baseline_validation_rows,
                "validation_chosen": chosen_validation_rows,
                "test_baseline": baseline_test_rows,
                "test_selected": test_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "validation_trial_pose_rows.json").write_text(
        json.dumps(validation_trial_rows, indent=2, sort_keys=True) + "\n"
    )
    if bool(args.development_cross_block_audit):
        (output_dir / "development_cross_block_pose_rows.json").write_text(
            json.dumps(cross_block_pose_rows, indent=2, sort_keys=True) + "\n"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "gate": summary["gate"],
                "validation_chosen": chosen,
                "development_cross_block_chosen": cross_block_chosen,
                "test": summary["test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
