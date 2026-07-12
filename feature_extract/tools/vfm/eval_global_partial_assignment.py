"""Audit whole-image query-track partial assignment before PnP.

This tool operates on a frozen no-retrieval, full-bank top-L proposal artifact.
Policy selection uses the validation split only. A reused late block can be
replayed for development diagnosis, but can never produce a production claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_local_assignment_support_views import (
    _evaluate_pose_strategy,
)
from feature_extract.tools.vfm.probe_detector_maplet_geometry import (
    _identity_metrics,
    _pose_gate,
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
    resolve_rescue_policy_scores,
    selective_baseline_gain_switch_scores,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
    select_pose_safe_matches,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


def _positive_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or min(output) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return output


def _finite_float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if any(not np.isfinite(item) for item in output):
        raise argparse.ArgumentTypeError("expected finite comma-separated floats")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument(
        "--score_artifact",
        default=None,
        help=(
            "optional compact matcher score NPZ; its sibling summary.json must carry "
            "the exact candidate/proposal/bank data manifest"
        ),
    )
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--score_keys",
        default="strategy__alike_support_top2_mean",
        help="comma-separated proposal arrays used as assignment scores",
    )
    parser.add_argument(
        "--baseline_score_key", default="strategy__alike_support_top2_mean"
    )
    parser.add_argument(
        "--frozen_baseline_summary",
        default=None,
        help=(
            "optional prior validation-only global baseline summary; when set, "
            "its exact policy and pose are replayed instead of reselecting a "
            "baseline from this invocation's match-count sweep"
        ),
    )
    parser.add_argument(
        "--dustbin_scores",
        type=_finite_float_list,
        default=(0.50, 0.60, 0.65, 0.70, 0.75, 0.80),
        help="explicit global-assignment no-match scores; no-threshold is always included",
    )
    parser.add_argument(
        "--baseline_gain_score_keys",
        default="",
        help=(
            "optional comma-separated probability scores used to build conservative "
            "KEEP-baseline policies before whole-image assignment"
        ),
    )
    parser.add_argument(
        "--baseline_gain_thresholds",
        type=_finite_float_list,
        default=(0.05, 0.10, 0.15, 0.20),
    )
    parser.add_argument(
        "--max_baseline_validity_thresholds",
        type=_finite_float_list,
        default=(0.20, 0.30, 0.40, 0.50),
    )
    parser.add_argument(
        "--rescue_candidate_score_keys",
        default="",
        help=(
            "optional comma-separated raw rescue-candidate probability arrays; "
            "the matching keep-probability array is resolved automatically"
        ),
    )
    parser.add_argument(
        "--rescue_action_margin_thresholds",
        type=_finite_float_list,
        default=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
    )
    parser.add_argument(
        "--match_counts", type=_positive_int_list, default=(32, 48, 64, 96, 128)
    )
    parser.add_argument(
        "--selection_modes", default="score_topk,spatial_round_robin"
    )
    parser.add_argument(
        "--assignment_modes",
        default="row_argmax,global_bipartite",
        help=(
            "comma-separated pre-PnP conflict resolvers; use global_bipartite "
            "alone for the formal S4 partial-assignment gate"
        ),
    )
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
    )
    parser.add_argument(
        "--development_cross_block_audit",
        action="store_true",
        help="replay the validation-frozen policy once on the reused late block",
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _compact(
    values: np.ndarray, selected_rows: np.ndarray, selected_columns: np.ndarray
) -> np.ndarray:
    valid = selected_columns >= 0
    safe_columns = np.maximum(selected_columns, 0)
    output = np.take_along_axis(
        np.asarray(values)[selected_rows], safe_columns, axis=1
    ).copy()
    if np.issubdtype(output.dtype, np.floating):
        output[~valid] = -np.inf
    else:
        output[~valid] = -1
    return output


def _selected_columns(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid & np.isfinite(scores), scores, -np.inf)
    output = np.argmax(safe, axis=1).astype(np.int64)
    output[~np.any(np.isfinite(safe), axis=1)] = -1
    return output


def _assignment_quality(
    *,
    scores: np.ndarray,
    valid_edges: np.ndarray,
    track_ids: np.ndarray,
    residuals_px: np.ndarray,
    query_ids: np.ndarray,
    baseline_columns: np.ndarray,
) -> dict[str, object]:
    selected = _selected_columns(scores, valid_edges)
    accepted = selected >= 0
    rows = np.arange(len(selected), dtype=np.int64)
    selected_residuals = np.full((len(selected),), np.inf, dtype=np.float32)
    selected_tracks = np.full((len(selected),), -1, dtype=np.int64)
    selected_residuals[accepted] = residuals_px[
        rows[accepted], selected[accepted]
    ]
    selected_tracks[accepted] = track_ids[rows[accepted], selected[accepted]]
    duplicate_counts: list[int] = []
    accepted_counts: list[int] = []
    for query_id in dict.fromkeys(query_ids.tolist()):
        image_rows = np.flatnonzero(query_ids == str(query_id))
        image_tracks = selected_tracks[image_rows]
        image_tracks = image_tracks[image_tracks >= 0]
        accepted_counts.append(int(len(image_tracks)))
        duplicate_counts.append(int(len(image_tracks) - len(np.unique(image_tracks))))
    finite_residuals = selected_residuals[np.isfinite(selected_residuals)]
    thresholds: dict[str, object] = {}
    for threshold in (1.0, 2.0, 5.0, 8.0):
        correct = accepted & (selected_residuals <= threshold)
        mappable = np.any(
            valid_edges & np.isfinite(residuals_px) & (residuals_px <= threshold),
            axis=1,
        )
        thresholds[f"{threshold:g}"] = {
            "correct_count": int(np.sum(correct)),
            "precision_among_accepted": float(np.sum(correct) / max(np.sum(accepted), 1)),
            "recall_given_mappable": float(np.sum(correct) / max(np.sum(mappable), 1)),
            "mappable_count": int(np.sum(mappable)),
        }
    comparable = accepted & (baseline_columns >= 0)
    return {
        "query_point_count": int(len(selected)),
        "accepted_count": int(np.sum(accepted)),
        "accepted_rate": float(np.mean(accepted)),
        "median_accepted_per_image": float(np.median(accepted_counts)),
        "minimum_accepted_per_image": int(min(accepted_counts)),
        "duplicate_track_count": int(sum(duplicate_counts)),
        "median_duplicate_tracks_per_image": float(np.median(duplicate_counts)),
        "reassigned_from_row_argmax_count": int(
            np.sum(comparable & (selected != baseline_columns))
        ),
        "median_selected_residual_px": (
            None if finite_residuals.size == 0 else float(np.median(finite_residuals))
        ),
        "thresholds_px": thresholds,
    }


def _pose_input_quality(
    *,
    scores: np.ndarray,
    valid_edges: np.ndarray,
    track_ids: np.ndarray,
    residuals_px: np.ndarray,
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    image_sizes: dict[str, tuple[int, int]],
    max_matches: int,
    selection_mode: str,
) -> dict[str, object]:
    """Measure the exact correspondence subset passed to uniform PnP."""

    selected_columns = _selected_columns(scores, valid_edges)
    selected_residuals: list[float] = []
    selected_counts: list[int] = []
    for query_id in dict.fromkeys(query_ids.tolist()):
        rows = np.flatnonzero(query_ids == str(query_id))
        matches: list[QueryTo3DMatch] = []
        for row in rows.tolist():
            column = int(selected_columns[row])
            if column < 0:
                continue
            matches.append(
                QueryTo3DMatch(
                    token_index=int(row),
                    xy=np.asarray(query_xy[row], dtype=np.float64),
                    track_id=int(track_ids[row, column]),
                    xyz=np.zeros((3,), dtype=np.float64),
                    similarity=float(scores[row, column]),
                    ratio=0.0,
                    landmark_variance=0.0,
                    source="global_partial_assignment_quality",
                )
            )
        width, height = image_sizes[str(query_id)]
        selected = select_pose_safe_matches(
            matches,
            max_matches=int(max_matches),
            image_width=int(width),
            image_height=int(height),
            mode=str(selection_mode),
        )
        selected_counts.append(len(selected))
        for match in selected:
            row = int(match.token_index)
            column = int(selected_columns[row])
            selected_residuals.append(float(residuals_px[row, column]))
    residual_array = np.asarray(selected_residuals, dtype=np.float32)
    finite = residual_array[np.isfinite(residual_array)]
    thresholds = {
        f"{threshold:g}": {
            "correct_count": int(np.sum(residual_array <= threshold)),
            "precision": float(
                np.sum(residual_array <= threshold) / max(len(residual_array), 1)
            ),
        }
        for threshold in (1.0, 2.0, 5.0, 8.0)
    }
    return {
        "selected_match_count": int(len(residual_array)),
        "median_matches_per_image": float(np.median(selected_counts)),
        "minimum_matches_per_image": int(min(selected_counts)),
        "finite_residual_count": int(len(finite)),
        "median_residual_px": None if len(finite) == 0 else float(np.median(finite)),
        "p90_residual_px": (
            None if len(finite) == 0 else float(np.percentile(finite, 90))
        ),
        "thresholds_px": thresholds,
    }


def _relative_pose_risk(
    pose: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    keys = (
        "median_translation_m_success",
        "p90_translation_m_success",
        "median_rotation_deg_success",
    )
    values: dict[str, tuple[float, float]] = {}
    for key in keys:
        candidate_value = pose.get(key)
        baseline_value = baseline.get(key)
        if candidate_value is None or baseline_value is None:
            return {
                **{name: None for name in keys},
                "worst_error_ratio": None,
                "mean_log_error_ratio": None,
                "valid": False,
                "failure_reason": "missing_pose_error_metric",
            }
        candidate_float = float(candidate_value)
        baseline_float = float(baseline_value)
        if not np.isfinite(candidate_float) or not np.isfinite(baseline_float):
            return {
                **{name: None for name in keys},
                "worst_error_ratio": None,
                "mean_log_error_ratio": None,
                "valid": False,
                "failure_reason": "non_finite_pose_error_metric",
            }
        values[key] = (candidate_float, baseline_float)
    ratios = {
        key: candidate / max(baseline_value, 1e-12)
        for key, (candidate, baseline_value) in values.items()
    }
    return {
        **ratios,
        "worst_error_ratio": float(max(ratios.values())),
        "mean_log_error_ratio": float(
            np.mean(np.log(np.maximum(tuple(ratios.values()), 1e-12)))
        ),
        "valid": True,
        "failure_reason": None,
    }


def _has_finite_pose_errors(pose: dict[str, object]) -> bool:
    values = (
        pose.get("median_translation_m_success"),
        pose.get("p90_translation_m_success"),
        pose.get("median_rotation_deg_success"),
    )
    return all(value is not None and np.isfinite(float(value)) for value in values)


def _policy_key(trial: dict[str, object]) -> tuple[float, ...]:
    pose = trial["pose"]
    absolute_error_geomean = float(
        np.exp(
            np.mean(
                np.log(
                    np.maximum(
                        (
                            float(pose["median_translation_m_success"]),
                            float(pose["p90_translation_m_success"]),
                            float(pose["median_rotation_deg_success"]),
                        ),
                        1e-12,
                    )
                )
            )
        )
    )
    return (
        float(pose["success_rate"]),
        -absolute_error_geomean,
        -float(pose["p90_translation_m_success"]),
        -float(pose["median_translation_m_success"]),
        -float(pose["median_rotation_deg_success"]),
        float(pose["recall_10cm_5deg"]),
        float(pose["recall_25cm_2deg"]),
        float(pose["recall_5cm_5deg"]),
        -float(trial["max_matches"]),
    )


def _load_frozen_baseline_policy(
    path: Path,
    *,
    proposals_path: Path,
    candidate_path: Path,
    bank_path: Path,
    split_path: Path,
    baseline_score_key: str,
) -> dict[str, object]:
    summary = json.loads(Path(path).read_text())
    if str(summary.get("stage", "")) != "whole_image_global_partial_assignment_audit":
        raise ValueError("frozen baseline summary has an unsupported stage")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict) or not bool(
        protocol.get("policy_selected_on_validation_only")
    ):
        raise ValueError("frozen baseline policy was not selected on validation only")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("frozen baseline summary is missing its input manifest")
    expected_inputs = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
        "baseline_score_key": str(baseline_score_key),
    }
    mismatches = {
        key: {"expected": expected, "actual": inputs.get(key)}
        for key, expected in expected_inputs.items()
        if inputs.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "frozen baseline summary uses different evaluation inputs: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    baseline = summary.get("baseline")
    policy = baseline.get("frozen_validation_policy") if isinstance(baseline, dict) else None
    if not isinstance(policy, dict):
        raise ValueError("frozen baseline summary has no validation policy")
    max_matches = int(policy.get("max_matches", 0))
    selection_mode = str(policy.get("selection_mode", ""))
    pose = policy.get("pose")
    if (
        max_matches <= 0
        or selection_mode not in {"score_topk", "spatial_round_robin"}
        or not isinstance(pose, dict)
    ):
        raise ValueError("frozen baseline summary contains an invalid policy")
    return {
        "source_path": str(path),
        "source_sha256": file_sha256_short(path),
        "policy_key": str(policy.get("policy_key", "")),
        "max_matches": max_matches,
        "selection_mode": selection_mode,
        "pose": pose,
        "late_development_pose": (
            baseline.get("late_development_pose")
            if isinstance(baseline, dict)
            else None
        ),
    }


def _validate_frozen_baseline_pose(
    actual: dict[str, object], source: dict[str, object]
) -> None:
    expected = source.get("pose")
    if not isinstance(expected, dict):
        raise ValueError("frozen baseline source has no validation pose")
    keys = (
        "query_count",
        "success_count",
        "success_rate",
        "median_translation_m_success",
        "p90_translation_m_success",
        "median_rotation_deg_success",
        "recall_25cm_2deg",
        "recall_10cm_5deg",
        "recall_5cm_5deg",
    )
    mismatches: dict[str, dict[str, float]] = {}
    for key in keys:
        try:
            expected_value = float(expected[key])
            actual_value = float(actual[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"frozen baseline pose is missing {key}") from error
        if not np.isclose(actual_value, expected_value, rtol=0.0, atol=1e-12):
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    if mismatches:
        raise ValueError(
            "baseline replay differs from the frozen validation sweep: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.development_cross_block_audit) and str(args.evaluation_role) != "development":
        raise ValueError("cross-block audit is only valid for development evaluation")
    proposals_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    score_path = None if args.score_artifact is None else Path(args.score_artifact)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    frozen_baseline_summary_path = (
        None
        if args.frozen_baseline_summary is None
        else Path(args.frozen_baseline_summary)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals = _load_npz(proposals_path)
    candidate = _load_npz(candidate_path)
    score_payload = proposals if score_path is None else _load_npz(score_path)
    split = json.loads(split_path.read_text())
    for name in ("train", "validation", "test"):
        if name not in split or not isinstance(split[name], list) or not split[name]:
            raise ValueError("split JSON requires non-empty train/validation/test lists")
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(candidate["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(candidate["valid_edges"], dtype=bool)
    if selected_columns.shape != valid_edges.shape or len(selected_rows) != len(selected_columns):
        raise ValueError("candidate artifact arrays have incompatible shapes")
    metadata = (
        json.loads(str(candidate["metadata_json"].item()))
        if "metadata_json" in candidate
        else {}
    )
    expected_hashes = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_hashes.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"stale candidate artifact: {json.dumps(mismatches, sort_keys=True)}")
    if score_path is not None:
        score_summary_path = score_path.parent / "summary.json"
        if not score_summary_path.exists():
            raise ValueError("external score artifact requires a sibling summary.json")
        score_summary = json.loads(score_summary_path.read_text())
        score_manifest = score_summary.get("data_manifest")
        if not isinstance(score_manifest, dict):
            raise ValueError("external score summary is missing data_manifest")
        score_outputs = score_summary.get("outputs")
        recorded_score_hash = (
            None
            if not isinstance(score_outputs, dict)
            else score_outputs.get("scores_sha256")
        )
        actual_score_hash = file_sha256_short(score_path)
        if not recorded_score_hash or str(recorded_score_hash) != actual_score_hash:
            raise ValueError(
                "external score artifact hash differs from its sibling summary"
            )
        expected_score_manifest = {
            "proposals_sha256": file_sha256_short(proposals_path),
            "feature_artifact_sha256": file_sha256_short(candidate_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        }
        score_mismatches = {
            key: {"expected": value, "actual": score_manifest.get(key)}
            for key, value in expected_score_manifest.items()
            if score_manifest.get(key) != value
        }
        if score_mismatches:
            raise ValueError(
                "stale or semantically misaligned score artifact: "
                f"{json.dumps(score_mismatches, sort_keys=True)}"
            )

    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    all_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    all_canonical_rows = canonical_rows_for_track_candidates(
        all_tracks, landmark_index.track_ids
    )
    compact_tracks = _compact(all_tracks, selected_rows, selected_columns).astype(np.int64)
    compact_rows = _compact(
        all_canonical_rows, selected_rows, selected_columns
    ).astype(np.int64)
    compact_prototypes = _compact(
        proposals["candidate_prototype_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    compact_coarse = _compact(
        proposals["coarse_scores"], selected_rows, selected_columns
    ).astype(np.float32)
    candidates = UniqueTrackCandidateSet(
        compact_rows, compact_tracks, compact_prototypes, compact_coarse
    )
    valid_edges &= candidates.valid_mask
    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    residuals = _compact(
        proposals["candidate_gt_residuals_px"], selected_rows, selected_columns
    ).astype(np.float32)
    nearest_residuals = np.asarray(
        proposals["nearest_visible_residuals_px"], dtype=np.float32
    )[selected_rows]
    labels = np.asarray(candidate["labels"], dtype=bool)
    if labels.shape != valid_edges.shape:
        raise ValueError("candidate identity labels have an incompatible shape")
    nearest_tracks = np.asarray(
        proposals["nearest_visible_track_ids"], dtype=np.int64
    )
    observations = [
        ColmapTrackObservation(
            track_id=int(nearest_tracks[global_row]),
            image_id=str(proposals["query_ids"][global_row]),
            point2d_idx=int(global_row),
            xy=(float(proposals["xy"][global_row, 0]), float(proposals["xy"][global_row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for global_row in selected_rows.tolist()
    ]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    image_sizes = {
        image_id: (
            int(cameras[int(image.camera_id)].width),
            int(cameras[int(image.camera_id)].height),
        )
        for image_id, image in images_by_name.items()
    }
    masks = {
        name: np.isin(query_ids, np.asarray(split[name], dtype=np.str_))
        for name in ("train", "validation", "test")
    }
    if np.any(np.sum(np.stack(list(masks.values()), axis=0), axis=0) != 1):
        raise ValueError("candidate query rows must belong to exactly one split")
    score_keys = tuple(
        item.strip() for item in str(args.score_keys).split(",") if item.strip()
    )
    if not score_keys or args.baseline_score_key not in proposals:
        raise ValueError("score keys are empty or the baseline score is missing")
    baseline_gain_score_keys = tuple(
        item.strip()
        for item in str(args.baseline_gain_score_keys).split(",")
        if item.strip()
    )
    rescue_candidate_score_keys = tuple(
        item.strip()
        for item in str(args.rescue_candidate_score_keys).split(",")
        if item.strip()
    )
    if set(baseline_gain_score_keys) - set(score_keys):
        raise ValueError("baseline gain score keys must also appear in score_keys")
    if any(float(value) < 0.0 for value in tuple(args.baseline_gain_thresholds)):
        raise ValueError("baseline gain thresholds must be non-negative")
    if any(
        not 0.0 <= float(value) <= 1.0
        for value in tuple(args.max_baseline_validity_thresholds)
    ):
        raise ValueError("baseline validity thresholds must be in [0, 1]")
    if any(
        float(value) < 0.0
        for value in tuple(args.rescue_action_margin_thresholds)
    ):
        raise ValueError("rescue action-margin thresholds must be non-negative")
    rescue_keep_score_keys: dict[str, str] = {}
    rescue_suffix = "rescue_candidate_probability"
    for key in rescue_candidate_score_keys:
        if not key.endswith(rescue_suffix):
            raise ValueError(
                "rescue candidate score keys must end with rescue_candidate_probability"
            )
        rescue_keep_score_keys[key] = (
            key[: -len(rescue_suffix)]
            + "rescue_keep_probability_DIAGNOSTIC_ONLY"
        )
    required_score_keys = {
        *score_keys,
        *rescue_candidate_score_keys,
        *rescue_keep_score_keys.values(),
    }
    missing_scores = required_score_keys - set(score_payload)
    if missing_scores:
        raise ValueError(f"proposal score arrays are missing: {sorted(missing_scores)}")
    selection_modes = tuple(
        item.strip()
        for item in str(args.selection_modes).split(",")
        if item.strip()
    )
    if not selection_modes or set(selection_modes) - {"score_topk", "spatial_round_robin"}:
        raise ValueError("unsupported pose selection mode")
    assignment_modes = tuple(
        item.strip()
        for item in str(args.assignment_modes).split(",")
        if item.strip()
    )
    if not assignment_modes or set(assignment_modes) - {
        "row_argmax",
        "global_bipartite",
    }:
        raise ValueError("unsupported assignment mode")
    def compact_score(key: str) -> np.ndarray:
        values = np.asarray(score_payload[key])
        if values.shape == selected_columns.shape:
            compact_values = values.astype(np.float32)
        elif values.shape == np.asarray(proposals["candidate_track_ids"]).shape:
            compact_values = _compact(
                values, selected_rows, selected_columns
            ).astype(np.float32)
        else:
            raise ValueError(
                f"score array {key} has shape {values.shape}, expected compact "
                f"{selected_columns.shape} or full proposal shape"
            )
        compact_values[~valid_edges] = -np.inf
        return compact_values

    compact_scores = {key: compact_score(key) for key in score_keys}
    compact_rescue_candidates = {
        key: compact_score(key) for key in rescue_candidate_score_keys
    }
    compact_rescue_keep = {
        key: compact_score(keep_key)
        for key, keep_key in rescue_keep_score_keys.items()
    }
    baseline_scores = _compact(
        proposals[args.baseline_score_key], selected_rows, selected_columns
    ).astype(np.float32)
    baseline_columns = _selected_columns(baseline_scores, valid_edges)

    def identity(scores: np.ndarray, split_name: str) -> dict[str, object]:
        mask = masks[split_name]
        return _identity_metrics(
            nearest_residuals=nearest_residuals[mask],
            candidate_residuals=residuals[mask],
            scores=np.asarray(scores)[mask],
            query_ids=query_ids[mask],
            labels=labels[mask],
            valid_edges=valid_edges[mask],
        )

    baseline_validation_identity = identity(baseline_scores, "validation")
    baseline_late_identity = None

    def evaluate_pose(
        scores: np.ndarray,
        split_name: str,
        strategy: str,
        *,
        max_matches: int | None,
        selection_mode: str = "score_topk",
    ):
        rows = np.flatnonzero(masks[split_name])
        subset = UniqueTrackCandidateSet(
            candidates.bank_row_indices[rows],
            candidates.track_ids[rows],
            candidates.prototype_ids[rows],
            candidates.coarse_scores[rows],
        )
        return _evaluate_pose_strategy(
            strategy=strategy,
            scores=scores[rows],
            candidates=subset,
            query_observations=[observations[int(row)] for row in rows.tolist()],
            query_ids=query_ids[rows].tolist(),
            landmark_index=landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
            max_matches=max_matches,
            pose_selection_mode=selection_mode,
        )

    row_argmax_all_validation_pose, _row_argmax_all_validation_rows = evaluate_pose(
        baseline_scores,
        "validation",
        "row_argmax_all_validation",
        max_matches=None,
    )
    baseline_global_scores, _baseline_global_columns = global_assignment_score_matrix(
        compact_tracks,
        baseline_scores,
        query_ids,
        valid_mask=valid_edges,
        dustbin_score=None,
    )
    baseline_by_budget: dict[tuple[int, str], dict[str, object]] = {}
    baseline_rows_by_budget: dict[tuple[int, str], list[dict[str, object]]] = {}
    baseline_policy_trials: list[dict[str, object]] = []
    frozen_baseline_source = (
        None
        if frozen_baseline_summary_path is None
        else _load_frozen_baseline_policy(
            frozen_baseline_summary_path,
            proposals_path=proposals_path,
            candidate_path=candidate_path,
            bank_path=bank_path,
            split_path=split_path,
            baseline_score_key=str(args.baseline_score_key),
        )
    )
    baseline_policy_pairs = {
        (int(count), str(mode))
        for count in tuple(args.match_counts)
        for mode in selection_modes
    }
    if frozen_baseline_source is not None:
        baseline_policy_pairs.add(
            (
                int(frozen_baseline_source["max_matches"]),
                str(frozen_baseline_source["selection_mode"]),
            )
        )
    for count, mode in sorted(baseline_policy_pairs):
        baseline_pose, baseline_rows = evaluate_pose(
            baseline_global_scores,
            "validation",
            f"baseline_global_max{int(count)}_{mode}_validation",
            max_matches=int(count),
            selection_mode=mode,
        )
        baseline_by_budget[(int(count), str(mode))] = baseline_pose
        baseline_rows_by_budget[(int(count), str(mode))] = baseline_rows
        baseline_policy_trials.append(
            {
                "policy_key": f"baseline_global_max{int(count)}_{str(mode)}",
                "max_matches": int(count),
                "selection_mode": str(mode),
                "pose": baseline_pose,
            }
        )
    finite_baseline_trials = [
        trial
        for trial in baseline_policy_trials
        if _has_finite_pose_errors(trial["pose"])
    ]
    if not finite_baseline_trials:
        raise RuntimeError("no frozen baseline policy produced a finite pose")
    if frozen_baseline_source is None:
        frozen_baseline_policy = max(finite_baseline_trials, key=_policy_key)
    else:
        frozen_key = (
            int(frozen_baseline_source["max_matches"]),
            str(frozen_baseline_source["selection_mode"]),
        )
        frozen_baseline_policy = next(
            trial
            for trial in finite_baseline_trials
            if (int(trial["max_matches"]), str(trial["selection_mode"])) == frozen_key
        )
        _validate_frozen_baseline_pose(
            frozen_baseline_policy["pose"], frozen_baseline_source
        )
    frozen_baseline_pose = frozen_baseline_policy["pose"]
    assignment_scores_by_policy = dict(compact_scores)
    assignment_policy_metadata: dict[str, dict[str, object]] = {
        key: {"pre_global_assignment_policy": "direct_score"}
        for key in compact_scores
    }
    for key in baseline_gain_score_keys:
        raw_scores = compact_scores[key]
        for gain_threshold in tuple(args.baseline_gain_thresholds):
            for validity_threshold in tuple(args.max_baseline_validity_thresholds):
                _selected, resolved, switched, _gains = (
                    selective_baseline_gain_switch_scores(
                        raw_scores,
                        baseline_scores,
                        min_gain=float(gain_threshold),
                        valid_mask=valid_edges,
                        baseline_validity_scores=raw_scores,
                        max_baseline_validity=float(validity_threshold),
                        preserve_baseline_row_confidence=True,
                    )
                )
                gain_name = f"{float(gain_threshold):g}".replace(".", "p")
                validity_name = f"{float(validity_threshold):g}".replace(".", "p")
                derived_key = (
                    f"baseline_gain__{key}__gain_{gain_name}"
                    f"__validity_max_{validity_name}"
                )
                assignment_scores_by_policy[derived_key] = resolved
                assignment_policy_metadata[derived_key] = {
                    "pre_global_assignment_policy": (
                        "keep_baseline_unless_candidate_probability_gain_and_baseline_invalid"
                    ),
                    "source_score_key": key,
                    "minimum_gain": float(gain_threshold),
                    "maximum_baseline_validity": float(validity_threshold),
                    "switch_count_all": int(np.sum(switched)),
                    "switch_count_validation": int(
                        np.sum(switched & masks["validation"])
                    ),
                    "switch_count_late_development": int(
                        np.sum(switched & masks["test"])
                    ),
                }
    for key in rescue_candidate_score_keys:
        candidate_probabilities = compact_rescue_candidates[key]
        keep_matrix = compact_rescue_keep[key]
        finite_keep = valid_edges & np.isfinite(keep_matrix)
        keep_probabilities = np.max(
            np.where(finite_keep, keep_matrix, -np.inf), axis=1
        )
        if not np.all(np.isfinite(keep_probabilities)):
            raise ValueError("every rescue row requires a finite keep probability")
        keep_deviation = np.where(
            finite_keep,
            np.abs(keep_matrix - keep_probabilities[:, None]),
            0.0,
        )
        if np.any(keep_deviation > 1e-6):
            raise ValueError("rescue keep probability differs within a candidate row")
        for margin_threshold in tuple(args.rescue_action_margin_thresholds):
            _selected, resolved, switched, action_margins, _action_scores = (
                resolve_rescue_policy_scores(
                    candidate_probabilities,
                    keep_probabilities,
                    baseline_scores,
                    action_margin_threshold=float(margin_threshold),
                    valid_mask=valid_edges,
                    preserve_baseline_alternatives=True,
                    lock_rescue_updates=True,
                )
            )
            margin_name = f"{float(margin_threshold):g}".replace(".", "p")
            derived_key = f"rescue_margin__{key}__margin_{margin_name}"
            assignment_scores_by_policy[derived_key] = resolved
            finite_switched_margins = action_margins[
                switched & np.isfinite(action_margins)
            ]
            assignment_policy_metadata[derived_key] = {
                "pre_global_assignment_policy": (
                    "keep_baseline_unless_rescue_action_margin_exceeds_threshold"
                ),
                "source_score_key": key,
                "keep_score_key": rescue_keep_score_keys[key],
                "action_margin_threshold": float(margin_threshold),
                "baseline_alternative_policy": (
                    "keep_full_row_and_lock_rescue_candidate_on_update"
                ),
                "switch_count_all": int(np.sum(switched)),
                "switch_count_validation": int(
                    np.sum(switched & masks["validation"])
                ),
                "switch_count_late_development": int(
                    np.sum(switched & masks["test"])
                ),
                "median_action_margin_switched": (
                    None
                    if len(finite_switched_margins) == 0
                    else float(np.median(finite_switched_margins))
                ),
            }

    resolved_by_policy: dict[str, np.ndarray] = {}
    policy_metadata: dict[str, dict[str, object]] = {}
    for key, values in assignment_scores_by_policy.items():
        if "row_argmax" in assignment_modes:
            row_key = f"row_argmax__{key}"
            resolved_by_policy[row_key] = values
            policy_metadata[row_key] = {
                "assignment_mode": "row_argmax_then_greedy_track_conflict",
                "score_key": key,
                "dustbin_score": None,
                **assignment_policy_metadata[key],
            }
        if "global_bipartite" in assignment_modes:
            for dustbin in (None, *tuple(args.dustbin_scores)):
                suffix = (
                    "none"
                    if dustbin is None
                    else f"{float(dustbin):g}".replace(".", "p")
                )
                policy_key = f"global_bipartite__{key}__dustbin_{suffix}"
                resolved, _selected = global_assignment_score_matrix(
                    compact_tracks,
                    values,
                    query_ids,
                    valid_mask=valid_edges,
                    dustbin_score=dustbin,
                )
                resolved_by_policy[policy_key] = resolved
                policy_metadata[policy_key] = {
                    "assignment_mode": (
                        "whole_image_sparse_bipartite_with_per_query_dustbin"
                    ),
                    "score_key": key,
                    "dustbin_score": None if dustbin is None else float(dustbin),
                    **assignment_policy_metadata[key],
                }

    validation_identity_by_score = {
        key: identity(values, "validation")
        for key, values in assignment_scores_by_policy.items()
    }
    trials: list[dict[str, object]] = []
    trial_pose_rows: list[dict[str, object]] = []
    for policy_key, scores in resolved_by_policy.items():
        validation_mask = masks["validation"]
        score_key = str(policy_metadata[policy_key]["score_key"])
        validation_identity = validation_identity_by_score[score_key]
        identity_passed = _assignment_identity_gate(
            validation_identity, baseline_validation_identity
        )
        quality = _assignment_quality(
            scores=scores[validation_mask],
            valid_edges=valid_edges[validation_mask],
            track_ids=compact_tracks[validation_mask],
            residuals_px=residuals[validation_mask],
            query_ids=query_ids[validation_mask],
            baseline_columns=baseline_columns[validation_mask],
        )
        for count in tuple(args.match_counts):
            for mode in selection_modes:
                pose, rows = evaluate_pose(
                    scores,
                    "validation",
                    f"{policy_key}_max{int(count)}_{mode}_validation",
                    max_matches=int(count),
                    selection_mode=mode,
                )
                trial_id = len(trials)
                same_budget_baseline = baseline_by_budget[(int(count), str(mode))]
                pose_input_quality = _pose_input_quality(
                    scores=scores[validation_mask],
                    valid_edges=valid_edges[validation_mask],
                    track_ids=compact_tracks[validation_mask],
                    residuals_px=residuals[validation_mask],
                    query_ids=query_ids[validation_mask],
                    query_xy=query_xy[validation_mask],
                    image_sizes=image_sizes,
                    max_matches=int(count),
                    selection_mode=mode,
                )
                pose_passed = _pose_gate(pose, frozen_baseline_pose)
                trials.append(
                    {
                        "trial_id": trial_id,
                        **policy_metadata[policy_key],
                        "policy_key": policy_key,
                        "max_matches": int(count),
                        "selection_mode": mode,
                        "assignment_quality": quality,
                        "pose_input_quality": pose_input_quality,
                        "identity": validation_identity,
                        "pose": pose,
                        "relative_pose_risk": _relative_pose_risk(
                            pose, frozen_baseline_pose
                        ),
                        "passes_pose_gate": bool(pose_passed),
                        "passes_identity_gate": bool(identity_passed),
                        "passes_stage_gate": bool(
                            pose_passed and identity_passed
                        ),
                        "baseline_reference_pose": frozen_baseline_pose,
                        "same_budget_baseline_reference_pose_DIAGNOSTIC_ONLY": (
                            same_budget_baseline
                        ),
                    }
                )
                trial_pose_rows.append({"trial_id": trial_id, "rows": rows})
    finite_trials = [
        trial
        for trial in trials
        if _has_finite_pose_errors(trial["pose"])
    ]
    if not finite_trials:
        raise RuntimeError("no validation assignment policy produced a finite pose")
    gated_trials = [trial for trial in finite_trials if trial["passes_stage_gate"]]
    chosen = max(gated_trials or finite_trials, key=_policy_key)
    chosen["selection_fallback_without_strict_stage_gate"] = not bool(gated_trials)
    chosen["selection_fallback_without_strict_pose_gate"] = not any(
        bool(trial["passes_pose_gate"]) for trial in finite_trials
    )
    chosen_scores = resolved_by_policy[str(chosen["policy_key"])]
    chosen_validation_rows = trial_pose_rows[int(chosen["trial_id"])]
    chosen_baseline_validation_rows = baseline_rows_by_budget[
        (
            int(frozen_baseline_policy["max_matches"]),
            str(frozen_baseline_policy["selection_mode"]),
        )
    ]
    late_pose = None
    late_rows: list[dict[str, object]] = []
    baseline_late_pose = None
    baseline_late_rows: list[dict[str, object]] = []
    late_quality = None
    late_pose_input_quality = None
    baseline_late_pose_input_quality = None
    late_identity = None
    late_identity_gate_passed = None
    late_pose_gate_passed = None
    late_stage_gate_passed = None
    if bool(args.development_cross_block_audit):
        baseline_late_identity = identity(baseline_scores, "test")
        late_pose, late_rows = evaluate_pose(
            chosen_scores,
            "test",
            "validation_frozen_policy_late_development",
            max_matches=int(chosen["max_matches"]),
            selection_mode=str(chosen["selection_mode"]),
        )
        baseline_late_pose, baseline_late_rows = evaluate_pose(
            baseline_global_scores,
            "test",
            "validation_frozen_global_baseline_late_development",
            max_matches=int(frozen_baseline_policy["max_matches"]),
            selection_mode=str(frozen_baseline_policy["selection_mode"]),
        )
        chosen_pre_assignment_scores = assignment_scores_by_policy[
            str(chosen["score_key"])
        ]
        late_identity = identity(chosen_pre_assignment_scores, "test")
        late_identity_gate_passed = _assignment_identity_gate(
            late_identity, baseline_late_identity
        )
        late_pose_gate_passed = _pose_gate(late_pose, baseline_late_pose)
        late_stage_gate_passed = bool(
            late_identity_gate_passed and late_pose_gate_passed
        )
        late_mask = masks["test"]
        late_quality = _assignment_quality(
            scores=chosen_scores[late_mask],
            valid_edges=valid_edges[late_mask],
            track_ids=compact_tracks[late_mask],
            residuals_px=residuals[late_mask],
            query_ids=query_ids[late_mask],
            baseline_columns=baseline_columns[late_mask],
        )
        late_pose_input_quality = _pose_input_quality(
            scores=chosen_scores[late_mask],
            valid_edges=valid_edges[late_mask],
            track_ids=compact_tracks[late_mask],
            residuals_px=residuals[late_mask],
            query_ids=query_ids[late_mask],
            query_xy=query_xy[late_mask],
            image_sizes=image_sizes,
            max_matches=int(chosen["max_matches"]),
            selection_mode=str(chosen["selection_mode"]),
        )
        baseline_late_pose_input_quality = _pose_input_quality(
            scores=baseline_global_scores[late_mask],
            valid_edges=valid_edges[late_mask],
            track_ids=compact_tracks[late_mask],
            residuals_px=residuals[late_mask],
            query_ids=query_ids[late_mask],
            query_xy=query_xy[late_mask],
            image_sizes=image_sizes,
            max_matches=int(frozen_baseline_policy["max_matches"]),
            selection_mode=str(frozen_baseline_policy["selection_mode"]),
        )
    artifact_metadata = {
        "format": "global_query_track_partial_assignment_v1",
        "evaluation_role": str(args.evaluation_role),
        "policy_selected_on": "validation",
        "baseline_score_key": str(args.baseline_score_key),
        "chosen_policy": {
            key: chosen[key]
            for key in (
                "policy_key",
                "assignment_mode",
                "score_key",
                "dustbin_score",
                "max_matches",
                "selection_mode",
            )
        },
        "frozen_baseline_policy": {
            "policy_key": str(frozen_baseline_policy["policy_key"]),
            "max_matches": int(frozen_baseline_policy["max_matches"]),
            "selection_mode": str(frozen_baseline_policy["selection_mode"]),
        },
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
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "score_artifact": None if score_path is None else str(score_path),
        "score_artifact_sha256": (
            None if score_path is None else file_sha256_short(score_path)
        ),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
    }
    artifact_path = output_dir / "resolved_assignment_scores.npz"
    chosen_compact_columns = _selected_columns(chosen_scores, valid_edges)
    np.savez(
        artifact_path,
        selected_rows=selected_rows,
        candidate_pool_columns=selected_columns,
        resolved_scores=chosen_scores.astype(np.float32),
        selected_compact_columns=chosen_compact_columns,
        query_ids=query_ids.astype(np.str_),
        query_xy=query_xy.astype(np.float32),
        metadata_json=np.asarray(json.dumps(artifact_metadata, sort_keys=True), dtype=np.str_),
    )
    selected_policy_artifact_path: Path | None = None
    chosen_score_key = str(chosen["score_key"])
    chosen_prefix = (
        chosen_score_key.rsplit("__", 1)[0]
        if "__" in chosen_score_key
        else ""
    )
    geometry_probability_keys = tuple(
        f"{chosen_prefix}__geometry_p{threshold}px"
        for threshold in ("01", "02", "05")
    )
    support_view_probability_keys = tuple(
        sorted(
            key
            for key in score_payload
            if chosen_prefix
            and key.startswith(f"{chosen_prefix}__support_view_probability_")
        )
    )
    if (
        chosen_prefix
        and set(geometry_probability_keys).issubset(score_payload)
        and support_view_probability_keys
    ):
        if np.any(chosen_compact_columns < 0):
            raise ValueError(
                "selected measurement policy contains an unmatched query row"
            )
        selected_index = np.arange(len(chosen_compact_columns), dtype=np.int64)
        chosen_pre_assignment_scores = assignment_scores_by_policy[chosen_score_key]
        geometry_probabilities = np.stack(
            [compact_score(key) for key in geometry_probability_keys], axis=2
        )
        support_view_probabilities = np.stack(
            [compact_score(key) for key in support_view_probability_keys], axis=2
        )
        selected_policy_metadata = {
            "format": "pose_safe_selected_policy_v1",
            "evaluation_role": str(args.evaluation_role),
            "chosen_score_key": chosen_score_key,
            "chosen_selection_score_key": chosen_score_key,
            "assignment_policy": str(chosen["assignment_mode"]),
            "max_matches": int(chosen["max_matches"]),
            "selection_mode": str(chosen["selection_mode"]),
            "proposals_sha256": file_sha256_short(proposals_path),
            "score_artifact_sha256": (
                None if score_path is None else file_sha256_short(score_path)
            ),
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json_sha256": file_sha256_short(split_path),
            "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
            "support_geometry_index_sha256": metadata.get(
                "support_geometry_index_sha256"
            ),
            "maplet_support_index_sha256": metadata.get(
                "maplet_support_index_sha256"
            ),
            "support_view_probability_keys": list(support_view_probability_keys),
            "geometry_probability_keys": list(geometry_probability_keys),
            "gt_residuals_are_target_only": True,
            "source_global_assignment_summary": str(output_dir / "summary.json"),
        }
        selected_policy_artifact_path = output_dir / "selected_policy_artifact.npz"
        np.savez(
            selected_policy_artifact_path,
            selected_rows=selected_rows,
            candidate_pool_columns=selected_columns,
            selected_compact_columns=chosen_compact_columns,
            selected_source_columns=selected_columns[
                selected_index, chosen_compact_columns
            ],
            baseline_compact_columns=baseline_columns,
            switched_from_baseline=chosen_compact_columns != baseline_columns,
            query_ids=query_ids.astype(np.str_),
            query_xy=query_xy.astype(np.float32),
            selected_track_ids=compact_tracks[
                selected_index, chosen_compact_columns
            ],
            selected_prototype_ids=compact_prototypes[
                selected_index, chosen_compact_columns
            ],
            selected_canonical_rows=compact_rows[
                selected_index, chosen_compact_columns
            ],
            selected_assignment_scores=chosen_pre_assignment_scores[
                selected_index, chosen_compact_columns
            ],
            selected_pose_selection_scores=chosen_scores[
                selected_index, chosen_compact_columns
            ],
            selected_gt_residuals_px=residuals[
                selected_index, chosen_compact_columns
            ],
            selected_geometry_probabilities=geometry_probabilities[
                selected_index, chosen_compact_columns
            ],
            selected_support_view_probabilities=support_view_probabilities[
                selected_index, chosen_compact_columns
            ],
            metadata_json=np.asarray(
                json.dumps(selected_policy_metadata, sort_keys=True),
                dtype=np.str_,
            ),
        )
    (output_dir / "validation_trial_pose_rows.json").write_text(
        json.dumps(trial_pose_rows, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "chosen_pose_rows.json").write_text(
        json.dumps(
            {
                "validation": chosen_validation_rows["rows"],
                "late_development": late_rows,
                "baseline_validation": chosen_baseline_validation_rows,
                "baseline_late_development": baseline_late_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "stage": "whole_image_global_partial_assignment_audit",
        "inputs": artifact_metadata,
        "protocol": {
            "proposal_scope": "full_bank_global_faiss_top_l",
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": False,
            "query_track_partial_assignment": True,
            "allowed_assignment_modes": list(assignment_modes),
            "rescue_action_margin_sweep": {
                "candidate_score_keys": list(rescue_candidate_score_keys),
                "thresholds": [
                    float(value)
                    for value in tuple(args.rescue_action_margin_thresholds)
                ],
                "selected_on": "validation_only",
            },
            "per_query_dustbin": True,
            "uniform_ransac_canonical_input_order": True,
            "policy_selected_on_validation_only": True,
            "stage_gate": "pose_and_pre_global_assignment_identity",
            "baseline_policy_source": (
                "current_validation_budget_sweep"
                if frozen_baseline_source is None
                else "externally_frozen_validation_summary_exact_replay"
            ),
            "baseline_score_key": str(args.baseline_score_key),
            "late_block_is_untouched_test": False,
            "production_promoted": False,
        },
        "candidate_shape": {
            "query_point_count": int(len(selected_rows)),
            "query_points_per_image": float(len(selected_rows) / len(np.unique(query_ids))),
            "proposal_top_l": int(selected_columns.shape[1]),
        },
        "baseline": {
            "row_argmax_all_validation_pose_DIAGNOSTIC_ONLY": (
                row_argmax_all_validation_pose
            ),
            "global_partial_assignment_validation_pose_by_budget": {
                f"max{count}_{mode}": pose
                for (count, mode), pose in baseline_by_budget.items()
            },
            "frozen_validation_policy": frozen_baseline_policy,
            "chosen_policy_validation_reference_pose": frozen_baseline_pose,
            "validation_identity": baseline_validation_identity,
            "late_development_pose": baseline_late_pose,
            "late_development_identity": baseline_late_identity,
            "late_development_pose_input_quality": (
                baseline_late_pose_input_quality
            ),
        },
        "validation": {
            "trial_count": int(len(trials)),
            "chosen": chosen,
            "trials": trials,
        },
        "late_development_replay": {
            "enabled": bool(args.development_cross_block_audit),
            "pose": late_pose,
            "identity": late_identity,
            "assignment_quality": late_quality,
            "pose_input_quality": late_pose_input_quality,
            "passes_pose_gate": late_pose_gate_passed,
            "passes_identity_gate": late_identity_gate_passed,
            "passes_stage_gate": late_stage_gate_passed,
            "development_only": True,
        },
        "outputs": {
            "resolved_assignment_scores": str(artifact_path),
            "resolved_assignment_scores_sha256": file_sha256_short(artifact_path),
            "selected_policy_artifact": (
                None
                if selected_policy_artifact_path is None
                else str(selected_policy_artifact_path)
            ),
            "selected_policy_artifact_sha256": (
                None
                if selected_policy_artifact_path is None
                else file_sha256_short(selected_policy_artifact_path)
            ),
            "validation_trial_pose_rows": str(output_dir / "validation_trial_pose_rows.json"),
            "chosen_pose_rows": str(output_dir / "chosen_pose_rows.json"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
