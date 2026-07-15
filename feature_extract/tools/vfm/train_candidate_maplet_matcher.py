"""Train assignment-only candidate-maplet matching on real hard proposals."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

from feature_extract.tools.vfm.probe_detector_maplet_geometry import _identity_metrics, _pose_gate
from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.localization.candidate_maplet_data import CandidateMapletEpisodeStore
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    CandidateMapletBatch,
    CandidateMapletMatcher,
    CandidateMapletMatcherConfig,
    candidate_maplet_group_loss,
    factorized_candidate_posterior,
)
from feature_extract.vfm.localization.local_assignment_linear import (
    resolve_rescue_policy_scores,
    selective_switch_scores,
)
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.pose_safe_selection import (
    candidate_admission_utility_scores,
    global_assignment_score_matrix,
    paired_pose_safety_report,
)


_T = TypeVar("_T")
_R = TypeVar("_R")


def _ordered_prefetch(
    items: Iterable[_T],
    build: Callable[[_T], _R],
    *,
    enabled: bool,
) -> Iterator[_R]:
    """Build at most one future item while the caller consumes the current one."""

    iterator = iter(items)
    try:
        first = next(iterator)
    except StopIteration:
        return
    if not bool(enabled):
        yield build(first)
        for item in iterator:
            yield build(item)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="candidate-batch") as pool:
        future = pool.submit(build, first)
        for item in iterator:
            value = future.result()
            future = pool.submit(build, item)
            yield value
        yield future.result()


def _float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return parsed


def _validation_strategy_names(
    value: str, *, rescue_policy_enabled: bool
) -> tuple[str, ...]:
    """Return a complete validation contract before training starts."""

    names = tuple(
        dict.fromkeys(item.strip() for item in str(value).split(",") if item.strip())
    )
    if not names:
        raise ValueError("validation_strategies must not be empty")
    if bool(rescue_policy_enabled) and "rescue_policy_resolved" not in names:
        names = (*names, "rescue_policy_resolved")
    return names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--query_context_detector_cache", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--feature_artifact", required=True)
    parser.add_argument("--radio_intermediate_cache", default=None)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
    )
    parser.add_argument(
        "--refit_selection_checkpoint",
        default=None,
        help=(
            "validation-selected checkpoint whose fixed epoch and inference policy are reused "
            "while refitting on train+validation"
        ),
    )
    parser.add_argument("--baseline_strategy", default="alike_support_top2_mean")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--model_dim", type=int, default=96)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iterations", type=int, default=10)
    parser.add_argument("--assignment_loss_weight", type=float, default=1.0)
    parser.add_argument("--pair_loss_weight", type=float, default=0.25)
    parser.add_argument("--candidate_aux_loss_weight", type=float, default=0.25)
    parser.add_argument("--candidate_set_loss_weight", type=float, default=1.0)
    parser.add_argument("--matched_query_weight", type=float, default=3.0)
    parser.add_argument("--candidate_pos_weight", type=float, default=3.0)
    parser.add_argument("--geometry_validity_loss_weight", type=float, default=0.0)
    parser.add_argument("--candidate_visibility_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--decoupled_candidate_heads",
        action="store_true",
        help="use independent candidate-set branches for identity, geometry, and visibility",
    )
    parser.add_argument(
        "--candidate_view_marginalization",
        action="store_true",
        help="retain per-support-view identity residuals until candidate-logit marginalization",
    )
    parser.add_argument(
        "--full_candidate_view_mixture",
        action="store_true",
        help="run candidate-set reasoning over all candidate/view latent nodes",
    )
    parser.add_argument(
        "--identity_conditioned_view_posterior",
        action="store_true",
        help="export the view posterior after applying per-view identity evidence",
    )
    parser.add_argument(
        "--explicit_anchor_role_embedding",
        action="store_true",
        help="mark the semantic query/support anchor with learned role embeddings",
    )
    parser.add_argument(
        "--prior_free_set_identity",
        action="store_true",
        help=(
            "derive top-L identity/view evidence only from query/support appearance; "
            "apply the coarse retrieval score solely as the explicit final prior"
        ),
    )
    parser.add_argument(
        "--deployable_identity_context",
        action="store_true",
        help=(
            "feed target-free detector, track, maplet, support-view, and anchor "
            "assignment evidence to the prior-free identity branch while "
            "excluding coarse/baseline scores and ranks"
        ),
    )
    parser.add_argument(
        "--prior_free_identity_loss_weight",
        type=float,
        default=0.0,
        help="auxiliary joint candidate/null loss before adding the coarse prior",
    )
    parser.add_argument(
        "--prior_free_conditional_identity_loss_weight",
        type=float,
        default=0.0,
        help=(
            "set-valued candidate ranking loss conditioned on group non-null, "
            "before adding the coarse prior"
        ),
    )
    parser.add_argument(
        "--factorized_set_posterior",
        action="store_true",
        help=(
            "model P(group mappable) separately from P(track | group mappable) "
            "instead of placing one dustbin in an L+1 softmax"
        ),
    )
    parser.add_argument(
        "--factorized_top_l_availability_loss_weight",
        type=float,
        default=0.0,
        help="binary top-L proposal-availability loss for the factorized posterior",
    )
    parser.add_argument("--support_view_dropout", type=float, default=0.0)
    parser.add_argument("--rescue_policy_loss_weight", type=float, default=0.0)
    parser.add_argument("--rescue_candidate_threshold_px", type=float, default=5.0)
    parser.add_argument(
        "--rescue_baseline_invalid_threshold_px", type=float, default=5.0
    )
    parser.add_argument(
        "--geometry_validity_thresholds_px",
        type=_float_list,
        default=(1.0, 2.0, 5.0),
    )
    parser.add_argument("--candidate_set_layers", type=int, default=1)
    parser.add_argument("--candidate_prior_feature", default="baseline_score")
    parser.add_argument("--candidate_prior_scale", type=float, default=20.0)
    parser.add_argument("--no_match_group_ratio", type=float, default=1.0)
    parser.add_argument("--max_train_groups", type=int, default=0)
    parser.add_argument(
        "--system_hard_group_oversample_factor",
        type=float,
        default=1.0,
        help=(
            "Identity-only curriculum multiplier for complete top-L groups with "
            "a rank-2..L positive, a low-margin wrong competitor, or a high-score "
            "no-match proposal. Values above one require geometry/rescue losses off."
        ),
    )
    parser.add_argument(
        "--system_hard_conditional_identity_weight",
        type=float,
        default=1.0,
        help=(
            "Per-group weight applied only to the prior-free conditional identity "
            "loss for mined hard groups. Unlike oversampling, this preserves the "
            "natural distribution used by top-L availability and null losses."
        ),
    )
    parser.add_argument(
        "--system_hard_ambiguous_margin", type=float, default=0.05
    )
    parser.add_argument(
        "--system_hard_no_match_quantile", type=float, default=0.75
    )
    parser.add_argument(
        "--system_hard_score_artifact",
        default="",
        help=(
            "Optional frozen same-query score artifact used to define system hard "
            "groups. Its data manifest must match the training candidate store."
        ),
    )
    parser.add_argument(
        "--system_hard_score_key",
        default="ensemble__set_candidate_probability",
        help="[G,L] score key read from --system_hard_score_artifact",
    )
    parser.add_argument(
        "--pose_conditioned_hard_negative_artifact",
        default="",
        help=(
            "Training-only artifact of concrete wrong candidate identities that "
            "coherently support target-free bad pose hypotheses"
        ),
    )
    parser.add_argument(
        "--pose_conditioned_hard_negative_margin_weight",
        type=float,
        default=0.0,
        help="weight of the prior-free system-error candidate margin loss",
    )
    parser.add_argument(
        "--pose_conditioned_hard_negative_margin",
        type=float,
        default=0.2,
        help="required prior-free logit gap from each system-error candidate",
    )
    parser.add_argument(
        "--pose_conditioned_hard_mode_artifact",
        default="",
        help=(
            "Training-only v2 artifact preserving coherent bad-pose mode "
            "membership across query groups"
        ),
    )
    parser.add_argument(
        "--pose_conditioned_hard_mode_margin_weight",
        type=float,
        default=0.0,
        help="weight of the structured bad-pose mode margin loss",
    )
    parser.add_argument(
        "--pose_conditioned_hard_mode_margin",
        type=float,
        default=0.2,
        help="required aggregate positive-vs-bad-mode logit gap",
    )
    parser.add_argument(
        "--pose_conditioned_hard_mode_top_group_fraction",
        type=float,
        default=0.5,
        help="fraction of strongest bad-mode support groups used by the margin",
    )
    parser.add_argument(
        "--query_grouped_training_batches",
        action="store_true",
        help=(
            "keep all candidate groups from each training query in one batch; "
            "required by structured pose-mode losses"
        ),
    )
    parser.add_argument("--train_query_count", type=int, default=60)
    parser.add_argument("--validation_query_count", type=int, default=15)
    parser.add_argument(
        "--split_strategy",
        default="contiguous_temporal_blocks_v1",
        choices=("contiguous_temporal_blocks_v1", "interleaved_development_v1"),
    )
    parser.add_argument(
        "--query_split_manifest",
        default="",
        help="explicit leakage-audited train/validation/test query split; overrides count-based splitting",
    )
    parser.add_argument("--smoke_validation_query_count", type=int, default=0)
    parser.add_argument("--support_view_count", type=int, default=2)
    parser.add_argument(
        "--static_feature_count",
        type=int,
        default=17,
        help="17 uses priors only; 74 also exposes inference-safe maplet probe summaries",
    )
    parser.add_argument(
        "--standardize_static_features",
        action="store_true",
        help="fit per-field mean/std on training query groups and store them in model config",
    )
    parser.add_argument("--query_radius_px", type=float, default=96.0)
    parser.add_argument("--max_query_nodes", type=int, default=48)
    parser.add_argument("--max_support_tracks", type=int, default=33)
    parser.add_argument("--query_cache_size", type=int, default=16384)
    parser.add_argument("--support_cache_size", type=int, default=131072)
    parser.add_argument("--episode_cache_size", type=int, default=262144)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--assignment_threshold_px", type=float, default=5.0)
    parser.add_argument(
        "--switch_margin_thresholds",
        type=_float_list,
        default=(0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7),
    )
    parser.add_argument(
        "--validation_strategies",
        default=(
            "set_candidate_probability,"
            "set_assignment_geomean,"
            "set_candidate_probability_dustbin_p50,"
            "set_candidate_probability_dustbin_p70,"
            "set_candidate_probability_dustbin_p90,"
            "set_assignment_geomean_dustbin_p50,"
            "set_assignment_geomean_dustbin_p70,"
            "set_assignment_geomean_dustbin_p90"
        ),
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument("--paired_catastrophic_translation_m", type=float, default=1.0)
    parser.add_argument("--paired_max_translation_regression_m", type=float, default=0.25)
    parser.add_argument(
        "--global_assignment_validation",
        action="store_true",
        help=(
            "also select best_global.pt with whole-image one-to-one assignment and a "
            "fixed pose budget; legacy best.pt selection remains unchanged"
        ),
    )
    parser.add_argument(
        "--global_assignment_joint_posterior",
        action="store_true",
        help=(
            "for set-candidate probabilities, solve whole-image assignment in "
            "candidate-vs-learned-null log-odds space"
        ),
    )
    parser.add_argument(
        "--global_assignment_candidate_admission_log_bonuses",
        type=_float_list,
        default=(0.0,),
        help=(
            "validation-only task utility sweep for admitting uncertain candidate "
            "groups without modifying their identity/null posterior"
        ),
    )
    parser.add_argument(
        "--global_assignment_baseline_summary",
        default="",
        help=(
            "validation-only baseline sweep summary that freezes the assignment "
            "match budget and selection mode"
        ),
    )
    parser.add_argument("--global_assignment_match_count", type=int, default=48)
    parser.add_argument(
        "--global_assignment_selection_mode",
        choices=("score_topk", "spatial_round_robin"),
        default="score_topk",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--profile_training_timing", action="store_true")
    parser.add_argument(
        "--training_batch_prefetch",
        action="store_true",
        help="construct the next lazy real-image episode batch during GPU compute",
    )
    return parser.parse_args(argv)


def _compact_values(values: np.ndarray, store: CandidateMapletEpisodeStore) -> np.ndarray:
    return np.take_along_axis(
        np.asarray(values)[store.selected_rows], store.selected_columns, axis=1
    ).astype(np.float32)


def _preserve_prior_row_confidence(
    candidate_scores: np.ndarray,
    prior_scores: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Use learned within-row ranking while retaining prior no-match confidence."""

    learned = np.asarray(candidate_scores, dtype=np.float32)
    prior = np.asarray(prior_scores, dtype=np.float32)
    if learned.ndim != 2 or prior.shape != learned.shape:
        raise ValueError("learned and prior candidate scores must have equal 2D shape")
    valid = np.isfinite(prior) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != learned.shape:
        raise ValueError("row confidence valid mask has an incompatible shape")
    output = np.full(learned.shape, -np.inf, dtype=np.float32)
    for row in range(len(learned)):
        candidates = valid[row] & np.isfinite(learned[row])
        prior_candidates = valid[row] & np.isfinite(prior[row])
        if not np.any(candidates) or not np.any(prior_candidates):
            continue
        learned_max = float(np.max(learned[row, candidates]))
        prior_max = float(np.max(prior[row, prior_candidates]))
        output[row, candidates] = (
            learned[row, candidates] - learned_max + prior_max
        ).astype(np.float32)
    return output


def _fit_static_feature_normalization(
    features: np.ndarray,
    *,
    row_mask: np.ndarray,
    valid_edges: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = np.asarray(features, dtype=np.float32)
    selected_rows = np.asarray(row_mask, dtype=bool)
    valid = np.asarray(valid_edges, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2] or selected_rows.shape != values.shape[:1]:
        raise ValueError("static normalization arrays have incompatible shapes")
    train_values = values[selected_rows][valid[selected_rows]].astype(np.float64)
    if train_values.shape[0] < 2 or not np.all(np.isfinite(train_values)):
        raise ValueError("static normalization requires finite training edges")
    mean = np.mean(train_values, axis=0)
    scale = np.std(train_values, axis=0)
    scale = np.where(scale >= 1e-6, scale, 1.0)
    return tuple(mean.astype(float).tolist()), tuple(scale.astype(float).tolist())


def _assignment_identity_gate(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """S4 identity gate: local assignment may not trade identity for pose luck."""

    candidate_thresholds = candidate["geometry"]["thresholds_px"]
    baseline_thresholds = baseline["geometry"]["thresholds_px"]
    comparisons = (
        (
            candidate_thresholds["1"]["recall_at_1_given_mappable"],
            baseline_thresholds["1"]["recall_at_1_given_mappable"],
        ),
        (
            candidate_thresholds["2"]["recall_at_1_given_mappable"],
            baseline_thresholds["2"]["recall_at_1_given_mappable"],
        ),
        (
            candidate["pair_positive_average_precision"],
            baseline["pair_positive_average_precision"],
        ),
        (
            candidate["wrong_pool_rejection_average_precision"],
            baseline["wrong_pool_rejection_average_precision"],
        ),
    )
    return all(
        float(candidate_value) + float(tolerance) >= float(baseline_value)
        for candidate_value, baseline_value in comparisons
    )


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


def _relative_pose_risk_rank_key(
    risk: dict[str, object],
) -> tuple[float, float, float]:
    if not bool(risk.get("valid", False)):
        return (0.0, -float("inf"), -float("inf"))
    return (
        1.0,
        -float(risk["worst_error_ratio"]),
        -float(risk["mean_log_error_ratio"]),
    )


def _candidate_data_manifest(
    args: argparse.Namespace, store: CandidateMapletEpisodeStore
) -> dict[str, object]:
    paths = {
        "proposals_sha256": Path(args.proposals),
        "detector_query_cache_sha256": Path(args.detector_query_cache),
        "query_context_detector_cache_sha256": Path(args.query_context_detector_cache),
        "support_feature_cache_sha256": Path(args.support_feature_cache),
        "support_geometry_index_sha256": Path(args.support_geometry_index),
        "projected_landmark_bank_sha256": Path(args.projected_landmark_bank),
        "maplet_support_index_sha256": Path(args.maplet_support_index),
        "feature_artifact_sha256": Path(args.feature_artifact),
        "colmap_cameras_sha256": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_sha256": Path(args.colmap_model_dir) / "images.bin",
        "colmap_points3d_sha256": Path(args.colmap_model_dir) / "points3D.bin",
    }
    manifest = {key: file_sha256_short(path) for key, path in paths.items()}
    manifest.update(
        {
            "query_split_manifest_sha256": (
                None
                if not str(getattr(args, "query_split_manifest", ""))
                else file_sha256_short(Path(args.query_split_manifest))
            ),
            "global_assignment_baseline_summary_sha256": (
                None
                if not str(getattr(args, "global_assignment_baseline_summary", ""))
                else file_sha256_short(Path(args.global_assignment_baseline_summary))
            ),
            "radio_intermediate_cache_sha256": (
                None
                if args.radio_intermediate_cache is None
                else file_sha256_short(Path(args.radio_intermediate_cache))
            ),
            "query_input_dim": int(store.query_input_dim),
            "support_input_dim": int(store.support_input_dim),
            "static_input_dim": int(store.static_input_dim),
            "candidate_top_k": int(store.candidate_top_k),
            "static_feature_names": list(store.static_feature_names),
            "positive_threshold_px": float(args.positive_threshold_px),
            "assignment_threshold_px": float(args.assignment_threshold_px),
            "support_view_count": int(args.support_view_count),
            "query_radius_px": float(args.query_radius_px),
            "max_query_nodes": int(args.max_query_nodes),
            "max_support_tracks": int(args.max_support_tracks),
        }
    )
    return manifest


_SYSTEM_HARD_SCORE_MANIFEST_KEYS = (
    "proposals_sha256",
    "detector_query_cache_sha256",
    "query_context_detector_cache_sha256",
    "support_feature_cache_sha256",
    "support_geometry_index_sha256",
    "projected_landmark_bank_sha256",
    "maplet_support_index_sha256",
    "feature_artifact_sha256",
    "query_split_manifest_sha256",
    "global_assignment_baseline_summary_sha256",
    "radio_intermediate_cache_sha256",
    "query_input_dim",
    "support_input_dim",
    "static_input_dim",
    "candidate_top_k",
    "static_feature_names",
    "positive_threshold_px",
    "assignment_threshold_px",
    "support_view_count",
    "query_radius_px",
    "max_query_nodes",
    "max_support_tracks",
)


def _load_system_hard_score_artifact(
    path: Path,
    *,
    score_key: str,
    expected_manifest: dict[str, object],
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a frozen same-query system score with strict lineage checks."""

    artifact_path = Path(path)
    key = str(score_key)
    if not key:
        raise ValueError("system hard score key must not be empty")
    with np.load(artifact_path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("system hard score artifact has no metadata_json")
        metadata = json.loads(str(data["metadata_json"].item()))
        source_manifest = dict(metadata.get("data_manifest") or {})
        if key not in data.files:
            raise ValueError(f"system hard score artifact has no key {key!r}")
        scores = np.asarray(data[key], dtype=np.float32)
    if scores.shape != tuple(int(value) for value in expected_shape):
        raise ValueError(
            "system hard scores have shape "
            f"{scores.shape}, expected {tuple(expected_shape)}"
        )
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("system hard scores contain NaN or positive infinity")
    mismatches = {
        field: {
            "score_artifact": source_manifest.get(field),
            "training": expected_manifest.get(field),
        }
        for field in _SYSTEM_HARD_SCORE_MANIFEST_KEYS
        if source_manifest.get(field) != expected_manifest.get(field)
    }
    if mismatches:
        raise ValueError(
            "system hard score artifact is stale or from a different candidate store: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    audit = {
        "mode": "frozen_same_query_score_artifact",
        "path": str(artifact_path),
        "sha256": file_sha256_short(artifact_path),
        "score_key": key,
        "format": metadata.get("format"),
        "finite_score_fraction": float(np.mean(np.isfinite(scores))),
    }
    return scores, audit


_POSE_CONDITIONED_HARD_NEGATIVE_FIELDS = frozenset(
    {
        "selected_rows",
        "selected_columns",
        "query_ids",
        "valid_edges",
        "positive_mask_TARGET_ONLY",
        "hard_negative_mask_TARGET_ONLY",
        "candidate_bad_mode_counts_TARGET_ONLY",
        "group_hard_mask_TARGET_ONLY",
        "group_bad_mode_counts_TARGET_ONLY",
        "metadata_json",
    }
)


def _load_pose_conditioned_hard_negative_artifact(
    path: Path,
    *,
    expected_manifest: dict[str, object],
    expected_selected_rows: np.ndarray,
    expected_selected_columns: np.ndarray,
    expected_query_ids: np.ndarray,
    expected_valid_edges: np.ndarray,
    expected_positive_mask: np.ndarray,
    split: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load training-only system errors and reject stale or leaky artifacts."""

    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as data:
        fields = set(data.files)
        if fields != _POSE_CONDITIONED_HARD_NEGATIVE_FIELDS:
            raise ValueError(
                "pose-conditioned hard-negative artifact fields differ: "
                f"missing={sorted(_POSE_CONDITIONED_HARD_NEGATIVE_FIELDS - fields)}, "
                f"extra={sorted(fields - _POSE_CONDITIONED_HARD_NEGATIVE_FIELDS)}"
            )
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != "pose_conditioned_system_hard_negatives_v1":
        raise ValueError("unsupported pose-conditioned hard-negative artifact format")
    if (
        metadata.get("training_only_target_artifact") is not True
        or metadata.get("pose_or_ground_truth_used_for_hypothesis_generation")
        is not False
        or metadata.get("ground_truth_joined_after_generation") is not True
        or metadata.get("split_names") != ["train"]
    ):
        raise ValueError(
            "pose-conditioned hard negatives must be target-free-generated, "
            "target-joined training-only rows"
        )
    inputs = metadata.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("pose-conditioned hard-negative input manifest is missing")
    expected_sources = {
        "candidate_artifact_sha256": expected_manifest.get(
            "feature_artifact_sha256"
        ),
        "proposals_sha256": expected_manifest.get("proposals_sha256"),
        "projected_landmark_bank_sha256": expected_manifest.get(
            "projected_landmark_bank_sha256"
        ),
        "split_json_sha256": expected_manifest.get(
            "query_split_manifest_sha256"
        ),
        "colmap_cameras_sha256": expected_manifest.get(
            "colmap_cameras_sha256"
        ),
        "colmap_images_sha256": expected_manifest.get("colmap_images_sha256"),
    }
    mismatches = {
        key: {"artifact": inputs.get(key), "training": value}
        for key, value in expected_sources.items()
        if value is None or inputs.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "pose-conditioned hard-negative artifact is stale or misaligned: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    selected_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(payload["selected_columns"], dtype=np.int64)
    query_ids = np.asarray(payload["query_ids"]).astype(str)
    valid = np.asarray(payload["valid_edges"], dtype=bool)
    positives = np.asarray(payload["positive_mask_TARGET_ONLY"], dtype=bool)
    hard = np.asarray(payload["hard_negative_mask_TARGET_ONLY"], dtype=bool)
    candidate_mode_counts = np.asarray(
        payload["candidate_bad_mode_counts_TARGET_ONLY"], dtype=np.int64
    )
    group_hard = np.asarray(payload["group_hard_mask_TARGET_ONLY"], dtype=bool)
    group_mode_counts = np.asarray(
        payload["group_bad_mode_counts_TARGET_ONLY"], dtype=np.int64
    )
    expected_shape = np.asarray(expected_valid_edges).shape
    if (
        selected_rows.shape != np.asarray(expected_selected_rows).shape
        or selected_columns.shape != np.asarray(expected_selected_columns).shape
        or query_ids.shape != np.asarray(expected_query_ids).shape
        or valid.shape != expected_shape
        or positives.shape != expected_shape
        or hard.shape != expected_shape
        or candidate_mode_counts.shape != expected_shape
        or group_hard.shape != (expected_shape[0],)
        or group_mode_counts.shape != (expected_shape[0],)
    ):
        raise ValueError("pose-conditioned hard-negative artifact dimensions differ")
    exact_arrays = (
        ("selected rows", selected_rows, np.asarray(expected_selected_rows)),
        ("selected columns", selected_columns, np.asarray(expected_selected_columns)),
        ("query ids", query_ids, np.asarray(expected_query_ids).astype(str)),
        ("valid edges", valid, np.asarray(expected_valid_edges, dtype=bool)),
        ("positive targets", positives, np.asarray(expected_positive_mask, dtype=bool)),
    )
    for label, actual, expected in exact_arrays:
        if not np.array_equal(actual, expected):
            raise ValueError(f"pose-conditioned hard-negative {label} differ")
    if np.any(candidate_mode_counts < 0) or np.any(group_mode_counts < 0):
        raise ValueError("pose-conditioned bad-mode counts must be non-negative")
    if np.any(hard & ~valid) or np.any(hard & positives):
        raise ValueError("pose-conditioned hard-negative candidate targets are invalid")
    if np.any(hard & (candidate_mode_counts <= 0)):
        raise ValueError("pose-conditioned hard candidates require a bad-mode count")
    if not np.array_equal(group_hard, np.any(hard, axis=1)):
        raise ValueError("pose-conditioned hard group mask differs from candidate masks")
    if np.any(group_hard & ~np.any(positives, axis=1)):
        raise ValueError("pose-conditioned hard groups require a positive candidate")
    if np.any(group_hard & (group_mode_counts <= 0)):
        raise ValueError("pose-conditioned hard groups require a bad-mode count")
    train_ids = np.asarray(split.get("train", ())).astype(str)
    known_ids = np.asarray(
        [
            str(query_id)
            for name in ("train", "validation", "test")
            for query_id in split.get(name, ())
        ]
    )
    if len(train_ids) == 0 or np.any(~np.isin(query_ids, known_ids)):
        raise ValueError("pose-conditioned artifact query ids differ from the split")
    train_rows = np.isin(query_ids, train_ids)
    if (
        np.any(hard[~train_rows])
        or np.any(candidate_mode_counts[~train_rows])
        or np.any(group_hard[~train_rows])
        or np.any(group_mode_counts[~train_rows])
    ):
        raise ValueError(
            "pose-conditioned hard-negative artifact leaks validation/test targets"
        )
    audit = {
        "mode": "target_free_pose_hypothesis_then_train_only_target_join",
        "path": str(artifact_path),
        "sha256": file_sha256_short(artifact_path),
        "hard_group_count": int(np.sum(group_hard)),
        "hard_candidate_count": int(np.sum(hard)),
        "multi_mode_hard_group_count": int(np.sum(group_mode_counts >= 2)),
        "training_query_count": int(len(set(query_ids[group_hard].tolist()))),
        "source_score_artifact_sha256": inputs.get("score_artifact_sha256"),
        "source_score_key": inputs.get("score_key"),
        "artifact_config": metadata.get("config"),
        "validation_or_test_training_leakage": False,
    }
    return hard, candidate_mode_counts, audit


_POSE_CONDITIONED_HARD_MODE_FIELDS = _POSE_CONDITIONED_HARD_NEGATIVE_FIELDS | {
    "hard_mode_ids_TARGET_ONLY",
    "hard_mode_candidate_mask_TARGET_ONLY",
    "hard_mode_query_ids_TARGET_ONLY",
    "hard_mode_support_group_counts_TARGET_ONLY",
}


def _load_pose_conditioned_hard_mode_artifact(
    path: Path,
    *,
    expected_manifest: dict[str, object],
    expected_selected_rows: np.ndarray,
    expected_selected_columns: np.ndarray,
    expected_query_ids: np.ndarray,
    expected_valid_edges: np.ndarray,
    expected_positive_mask: np.ndarray,
    split: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load coherent train-only pose modes with exact group membership."""

    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as data:
        fields = set(data.files)
        if fields != _POSE_CONDITIONED_HARD_MODE_FIELDS:
            raise ValueError(
                "pose-conditioned hard-mode artifact fields differ: "
                f"missing={sorted(_POSE_CONDITIONED_HARD_MODE_FIELDS - fields)}, "
                f"extra={sorted(fields - _POSE_CONDITIONED_HARD_MODE_FIELDS)}"
            )
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != "pose_conditioned_system_hard_modes_v2":
        raise ValueError("unsupported pose-conditioned hard-mode artifact format")
    if (
        metadata.get("training_only_target_artifact") is not True
        or metadata.get("pose_or_ground_truth_used_for_hypothesis_generation")
        is not False
        or metadata.get("ground_truth_joined_after_generation") is not True
        or metadata.get("split_names") != ["train"]
    ):
        raise ValueError(
            "pose-conditioned hard modes must be target-free-generated, "
            "target-joined training-only rows"
        )
    inputs = metadata.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("pose-conditioned hard-mode input manifest is missing")
    expected_sources = {
        "candidate_artifact_sha256": expected_manifest.get(
            "feature_artifact_sha256"
        ),
        "proposals_sha256": expected_manifest.get("proposals_sha256"),
        "projected_landmark_bank_sha256": expected_manifest.get(
            "projected_landmark_bank_sha256"
        ),
        "split_json_sha256": expected_manifest.get(
            "query_split_manifest_sha256"
        ),
        "colmap_cameras_sha256": expected_manifest.get(
            "colmap_cameras_sha256"
        ),
        "colmap_images_sha256": expected_manifest.get("colmap_images_sha256"),
    }
    mismatches = {
        key: {"artifact": inputs.get(key), "training": value}
        for key, value in expected_sources.items()
        if value is None or inputs.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "pose-conditioned hard-mode artifact is stale or misaligned: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    selected_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(payload["selected_columns"], dtype=np.int64)
    query_ids = np.asarray(payload["query_ids"]).astype(str)
    valid = np.asarray(payload["valid_edges"], dtype=bool)
    positives = np.asarray(payload["positive_mask_TARGET_ONLY"], dtype=bool)
    hard = np.asarray(payload["hard_negative_mask_TARGET_ONLY"], dtype=bool)
    candidate_mode_counts = np.asarray(
        payload["candidate_bad_mode_counts_TARGET_ONLY"], dtype=np.int64
    )
    group_hard = np.asarray(payload["group_hard_mask_TARGET_ONLY"], dtype=bool)
    group_mode_counts = np.asarray(
        payload["group_bad_mode_counts_TARGET_ONLY"], dtype=np.int64
    )
    mode_ids = np.asarray(payload["hard_mode_ids_TARGET_ONLY"], dtype=np.int64)
    mode_candidates = np.asarray(
        payload["hard_mode_candidate_mask_TARGET_ONLY"], dtype=bool
    )
    mode_query_ids = np.asarray(
        payload["hard_mode_query_ids_TARGET_ONLY"]
    ).astype(str)
    mode_support_counts = np.asarray(
        payload["hard_mode_support_group_counts_TARGET_ONLY"], dtype=np.int64
    )
    expected_shape = np.asarray(expected_valid_edges).shape
    if (
        selected_rows.shape != np.asarray(expected_selected_rows).shape
        or selected_columns.shape != np.asarray(expected_selected_columns).shape
        or query_ids.shape != np.asarray(expected_query_ids).shape
        or valid.shape != expected_shape
        or positives.shape != expected_shape
        or hard.shape != expected_shape
        or candidate_mode_counts.shape != expected_shape
        or group_hard.shape != (expected_shape[0],)
        or group_mode_counts.shape != (expected_shape[0],)
        or mode_ids.ndim != 2
        or mode_ids.shape[0] != expected_shape[0]
        or mode_candidates.shape
        != (expected_shape[0], mode_ids.shape[1], expected_shape[1])
        or mode_query_ids.shape != mode_support_counts.shape
    ):
        raise ValueError("pose-conditioned hard-mode artifact dimensions differ")
    exact_arrays = (
        ("selected rows", selected_rows, np.asarray(expected_selected_rows)),
        (
            "selected columns",
            selected_columns,
            np.asarray(expected_selected_columns),
        ),
        ("query ids", query_ids, np.asarray(expected_query_ids).astype(str)),
        ("valid edges", valid, np.asarray(expected_valid_edges, dtype=bool)),
        (
            "positive targets",
            positives,
            np.asarray(expected_positive_mask, dtype=bool),
        ),
    )
    for label, actual, expected in exact_arrays:
        if not np.array_equal(actual, expected):
            raise ValueError(f"pose-conditioned hard-mode {label} differ")
    if np.any(mode_ids < -1):
        raise ValueError("pose-conditioned hard-mode IDs must be -1 or non-negative")
    mode_present = mode_ids >= 0
    if not np.array_equal(mode_present, np.any(mode_candidates, axis=2)):
        raise ValueError("pose-conditioned hard-mode IDs and membership differ")
    if np.any(mode_candidates & ~valid[:, None, :]) or np.any(
        mode_candidates & positives[:, None, :]
    ):
        raise ValueError("pose-conditioned hard-mode candidate targets are invalid")
    if np.any(candidate_mode_counts < 0) or np.any(group_mode_counts < 0):
        raise ValueError("pose-conditioned hard-mode counts must be non-negative")
    if not np.array_equal(
        candidate_mode_counts, np.sum(mode_candidates, axis=1, dtype=np.int64)
    ):
        raise ValueError("pose-conditioned candidate mode counts differ from membership")
    if not np.array_equal(
        group_mode_counts, np.sum(mode_present, axis=1, dtype=np.int64)
    ):
        raise ValueError("pose-conditioned group mode counts differ from membership")
    mode_union = np.any(mode_candidates, axis=1)
    if np.any(hard & ~mode_union) or not np.array_equal(
        group_hard, np.any(mode_present, axis=1)
    ):
        raise ValueError("pose-conditioned hard-mode union differs from v1 targets")
    if np.any(group_hard & ~np.any(positives, axis=1)):
        raise ValueError("pose-conditioned hard-mode groups require a positive")

    mode_count = int(len(mode_query_ids))
    if mode_count == 0:
        raise ValueError("pose-conditioned hard-mode artifact contains no modes")
    present_ids = mode_ids[mode_present]
    if np.any(present_ids >= mode_count) or not np.array_equal(
        np.unique(present_ids), np.arange(mode_count, dtype=np.int64)
    ):
        raise ValueError("pose-conditioned hard-mode IDs are not contiguous")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("pose-conditioned hard-mode config is missing")
    minimum_mode_groups = int(config.get("min_consistent_groups", 0))
    if minimum_mode_groups <= 0:
        raise ValueError("pose-conditioned hard-mode minimum group count is invalid")
    for mode_id in range(mode_count):
        locations = np.argwhere(mode_ids == mode_id)
        rows = locations[:, 0]
        if len(np.unique(rows)) != len(rows):
            raise ValueError("pose-conditioned hard mode repeats a query group")
        if len(rows) != int(mode_support_counts[mode_id]):
            raise ValueError("pose-conditioned hard-mode support count differs")
        if len(rows) < minimum_mode_groups:
            raise ValueError("pose-conditioned hard mode has too few groups")
        if not np.all(query_ids[rows] == mode_query_ids[mode_id]):
            raise ValueError("pose-conditioned hard mode crosses query images")

    train_ids = np.asarray(split.get("train", ())).astype(str)
    known_ids = np.asarray(
        [
            str(query_id)
            for name in ("train", "validation", "test")
            for query_id in split.get(name, ())
        ]
    )
    if len(train_ids) == 0 or np.any(~np.isin(query_ids, known_ids)):
        raise ValueError("pose-conditioned hard-mode query ids differ from the split")
    train_rows = np.isin(query_ids, train_ids)
    if (
        np.any(mode_present[~train_rows])
        or np.any(mode_candidates[~train_rows])
        or np.any(~np.isin(mode_query_ids, train_ids))
    ):
        raise ValueError("pose-conditioned hard-mode artifact leaks validation/test targets")
    audit = {
        "mode": "structured_target_free_bad_pose_modes_with_train_only_target_join",
        "path": str(artifact_path),
        "sha256": file_sha256_short(artifact_path),
        "hard_mode_count": mode_count,
        "hard_mode_group_incidence_count": int(np.sum(mode_present)),
        "hard_group_count": int(np.sum(group_hard)),
        "hard_candidate_membership_count": int(np.sum(mode_candidates)),
        "training_query_count": int(len(set(mode_query_ids.tolist()))),
        "minimum_mode_groups": minimum_mode_groups,
        "source_score_artifact_sha256": inputs.get("score_artifact_sha256"),
        "source_score_key": inputs.get("score_key"),
        "artifact_config": config,
        "validation_or_test_training_leakage": False,
    }
    return mode_ids, mode_candidates, audit


_COLMAP_PROVENANCE_MANIFEST_KEYS = frozenset(
    {
        "colmap_cameras_sha256",
        "colmap_images_sha256",
        "colmap_points3d_sha256",
    }
)

_LEGACY_TRAINING_ONLY_DATA_MANIFEST_KEYS = frozenset(
    {
        "system_hard_score_artifact_sha256",
        "system_hard_score_key",
    }
)


def _candidate_data_manifest_mismatches(
    checkpoint_manifest: dict[str, object],
    current_manifest: dict[str, object],
    *,
    allow_legacy_missing_colmap_provenance: bool,
) -> dict[str, dict[str, object]]:
    """Compare runtime data contracts while tolerating known legacy metadata."""

    mismatches: dict[str, dict[str, object]] = {}
    for key in sorted(set(checkpoint_manifest) | set(current_manifest)):
        # Early hard-negative checkpoints incorrectly stored training-curriculum
        # provenance in the runtime data contract. It is not an inference input.
        if key in _LEGACY_TRAINING_ONLY_DATA_MANIFEST_KEYS:
            continue
        if (
            allow_legacy_missing_colmap_provenance
            and key in _COLMAP_PROVENANCE_MANIFEST_KEYS
            and key not in checkpoint_manifest
        ):
            continue
        if checkpoint_manifest.get(key) != current_manifest.get(key):
            mismatches[key] = {
                "checkpoint": checkpoint_manifest.get(key),
                "current": current_manifest.get(key),
            }
    return mismatches


def _load_refit_selection(
    path: Path,
    *,
    data_manifest: dict[str, object],
    split: dict[str, object],
    model_config: dict[str, object],
    epochs: int,
) -> tuple[dict[str, object], dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    checkpoint_format = str(checkpoint.get("format", ""))
    if checkpoint_format not in {
        "candidate_maplet_matcher_checkpoint_v5",
        "candidate_maplet_matcher_checkpoint_v6",
        "candidate_maplet_matcher_checkpoint_v7",
        "candidate_maplet_matcher_checkpoint_v8",
        "candidate_maplet_matcher_checkpoint_v9",
        "candidate_maplet_matcher_checkpoint_v10",
        "candidate_maplet_matcher_checkpoint_v11",
    }:
        raise ValueError("refit selection checkpoint has an unsupported format")
    manifest_mismatches = _candidate_data_manifest_mismatches(
        dict(checkpoint.get("data_manifest") or {}),
        data_manifest,
        allow_legacy_missing_colmap_provenance=checkpoint_format
        not in {
            "candidate_maplet_matcher_checkpoint_v8",
            "candidate_maplet_matcher_checkpoint_v9",
            "candidate_maplet_matcher_checkpoint_v10",
            "candidate_maplet_matcher_checkpoint_v11",
        },
    )
    if manifest_mismatches:
        raise ValueError(
            "refit selection checkpoint uses different data inputs: "
            f"{json.dumps(manifest_mismatches, sort_keys=True)}"
        )
    if checkpoint.get("split") != split:
        raise ValueError("refit selection checkpoint uses a different development split")
    if checkpoint.get("model_config") != model_config:
        raise ValueError("refit selection checkpoint uses a different model configuration")
    source_epoch = int(checkpoint.get("epoch", -1))
    if int(epochs) != source_epoch + 1:
        raise ValueError(
            f"refit epochs must equal validation-selected epoch + 1 ({source_epoch + 1})"
        )
    selection = checkpoint.get("selection")
    if not isinstance(selection, dict) or not bool(selection.get("validation_gate_passed")):
        raise ValueError("refit requires a validation-passed fixed inference selection")
    if str(selection.get("mode", "")) not in {"unconditional", "selective"}:
        raise ValueError("refit source selection has no deployable inference mode")
    return dict(selection), {
        "path": str(path),
        "sha256": file_sha256_short(Path(path)),
        "source_epoch": int(source_epoch),
        "source_seed": int(checkpoint.get("seed", -1)),
    }


def _build_query_split(
    query_ids: Sequence[str],
    *,
    strategy: str,
    train_count: int,
    validation_count: int,
    smoke_validation_count: int = 0,
) -> dict[str, object]:
    unique_ids = tuple(dict.fromkeys(str(value) for value in query_ids))
    development_count = int(train_count) + int(validation_count)
    if int(train_count) <= 0 or int(validation_count) <= 0 or development_count >= len(unique_ids):
        raise ValueError("split counts must leave non-empty train, validation, and test blocks")
    if str(strategy) == "contiguous_temporal_blocks_v1":
        train = list(unique_ids[: int(train_count)])
        validation = list(unique_ids[int(train_count) : development_count])
    elif str(strategy) == "interleaved_development_v1":
        validation_positions = (
            np.arange(int(validation_count), dtype=np.int64) * development_count
        ) // int(validation_count)
        validation_position_set = set(validation_positions.tolist())
        development = unique_ids[:development_count]
        validation = [
            image_id
            for position, image_id in enumerate(development)
            if position in validation_position_set
        ]
        train = [
            image_id
            for position, image_id in enumerate(development)
            if position not in validation_position_set
        ]
        if len(train) != int(train_count) or len(validation) != int(validation_count):
            raise RuntimeError("interleaved split produced the wrong block sizes")
    else:
        raise ValueError(f"unsupported split strategy: {strategy}")
    if int(smoke_validation_count) > 0:
        validation = validation[: int(smoke_validation_count)]
    return {
        "strategy": str(strategy),
        "train": train,
        "validation": validation,
        "test": list(unique_ids[development_count:]),
    }


def _load_query_split_manifest(
    path: Path,
    *,
    query_ids: Sequence[str],
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if str(payload.get("format", "")) != "stratified_landmark_query_split_v1":
        raise ValueError("unsupported explicit query split manifest format")
    split: dict[str, list[str]] = {}
    flattened: list[str] = []
    for name in ("train", "validation", "test"):
        values = [str(value) for value in payload.get(name, [])]
        if not values:
            raise ValueError(f"explicit query split has an empty {name} block")
        if len(values) != len(set(values)):
            raise ValueError(f"explicit query split has duplicate ids in {name}")
        split[name] = values
        flattened.extend(values)
    if len(flattened) != len(set(flattened)):
        raise ValueError("explicit query split blocks overlap")
    available = tuple(dict.fromkeys(str(value) for value in query_ids))
    missing = sorted(set(available) - set(flattened))
    unexpected = sorted(set(flattened) - set(available))
    if missing or unexpected:
        raise ValueError(
            "explicit query split does not equal the proposal query set: "
            f"missing={missing[:10]!r}, unexpected={unexpected[:10]!r}"
        )
    return {
        "strategy": "explicit_query_split_manifest_v1",
        "source_format": str(payload["format"]),
        "source_strategy": str(payload.get("strategy", "")),
        "source_sha256": file_sha256_short(Path(path)),
        **split,
    }


def _split_query_ids(store: CandidateMapletEpisodeStore, args: argparse.Namespace):
    query_ids = tuple(dict.fromkeys(store.query_ids[store.selected_rows].tolist()))
    if str(getattr(args, "query_split_manifest", "")):
        return _load_query_split_manifest(
            Path(args.query_split_manifest),
            query_ids=query_ids,
        )
    return _build_query_split(
        query_ids,
        strategy=str(args.split_strategy),
        train_count=int(args.train_query_count),
        validation_count=int(args.validation_query_count),
        smoke_validation_count=int(args.smoke_validation_query_count),
    )


def _load_frozen_global_baseline_policy(
    path: Path,
    *,
    args: argparse.Namespace,
    data_manifest: dict[str, object],
) -> dict[str, object]:
    summary = json.loads(Path(path).read_text())
    if str(summary.get("stage", "")) != "whole_image_global_partial_assignment_audit":
        raise ValueError("global assignment baseline summary has an unsupported stage")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict) or not bool(protocol.get("policy_selected_on_validation_only")):
        raise ValueError("global assignment baseline policy was not selected on validation only")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("global assignment baseline summary is missing its input manifest")
    baseline_score_key = str(args.baseline_strategy)
    if not baseline_score_key.startswith("strategy__"):
        baseline_score_key = f"strategy__{baseline_score_key}"
    expected = {
        "proposals_sha256": data_manifest.get("proposals_sha256"),
        "candidate_artifact_sha256": data_manifest.get("feature_artifact_sha256"),
        "projected_landmark_bank_sha256": data_manifest.get("projected_landmark_bank_sha256"),
        "split_json_sha256": data_manifest.get("query_split_manifest_sha256"),
        "baseline_score_key": baseline_score_key,
    }
    mismatches = {
        key: {"expected": value, "actual": inputs.get(key)}
        for key, value in expected.items()
        if value is None or inputs.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "global assignment baseline summary uses different training inputs: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    baseline = summary.get("baseline")
    policy = baseline.get("frozen_validation_policy") if isinstance(baseline, dict) else None
    if not isinstance(policy, dict):
        raise ValueError("global assignment baseline summary has no frozen validation policy")
    max_matches = int(policy.get("max_matches", 0))
    selection_mode = str(policy.get("selection_mode", ""))
    if max_matches <= 0 or selection_mode not in {"score_topk", "spatial_round_robin"}:
        raise ValueError("global assignment baseline policy is invalid")
    args.global_assignment_match_count = max_matches
    args.global_assignment_selection_mode = selection_mode
    return {
        "path": str(path),
        "sha256": file_sha256_short(Path(path)),
        "policy_key": str(policy.get("policy_key", "")),
        "max_matches": max_matches,
        "selection_mode": selection_mode,
        "validation_pose": policy.get("pose"),
    }


def _validate_frozen_baseline_pose(
    actual: dict[str, object],
    source: dict[str, object],
) -> None:
    expected = source.get("validation_pose")
    if not isinstance(expected, dict):
        raise ValueError("frozen global baseline source has no validation pose")
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
    mismatches = {}
    for key in keys:
        try:
            expected_value = float(expected[key])
            actual_value = float(actual[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"frozen global baseline pose is missing {key}") from error
        if not np.isclose(actual_value, expected_value, rtol=0.0, atol=1e-12):
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    if mismatches:
        raise ValueError(
            "training baseline replay differs from the frozen validation sweep: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def _balanced_train_groups(
    store: CandidateMapletEpisodeStore,
    train_edges: np.ndarray,
    *,
    no_match_ratio: float,
    max_groups: int,
    rng: np.random.Generator,
    system_hard_group_mask: np.ndarray | None = None,
    system_hard_oversample_factor: float = 1.0,
) -> np.ndarray:
    """Sample complete top-L rows so ranking and no-match targets stay well defined."""

    if float(no_match_ratio) < 0.0:
        raise ValueError("no_match_ratio must be non-negative")
    hard_factor = float(system_hard_oversample_factor)
    if not np.isfinite(hard_factor) or hard_factor < 1.0:
        raise ValueError("system hard-group oversample factor must be at least one")
    top_l = store.candidate_top_k
    edges = np.asarray(train_edges, dtype=np.int64).reshape(-1)
    groups = np.unique(edges // top_l)
    expected = (groups[:, None] * top_l + np.arange(top_l)[None]).reshape(-1)
    if not np.array_equal(np.sort(edges), expected):
        raise ValueError("training split does not contain complete candidate groups")
    has_positive = np.any(store.labels[groups], axis=1)
    positive_groups = groups[has_positive]
    no_match_groups = groups[~has_positive]
    if positive_groups.size == 0 or no_match_groups.size == 0:
        raise ValueError("grouped training requires both mappable and no-match rows")
    no_match_count = min(
        len(no_match_groups), int(round(len(positive_groups) * float(no_match_ratio)))
    )
    selected_no_match = rng.choice(
        no_match_groups, size=no_match_count, replace=False
    ).astype(np.int64)
    output = np.concatenate([positive_groups, selected_no_match]).astype(np.int64)
    if int(max_groups) > 0 and len(output) > int(max_groups):
        target_positive = min(len(positive_groups), max(1, int(max_groups) // 2))
        target_no_match = min(len(selected_no_match), int(max_groups) - target_positive)
        if target_positive + target_no_match < int(max_groups):
            target_positive = min(
                len(positive_groups), int(max_groups) - target_no_match
            )
        output = np.concatenate(
            [
                rng.choice(positive_groups, size=target_positive, replace=False),
                rng.choice(selected_no_match, size=target_no_match, replace=False),
            ]
        ).astype(np.int64)
    if system_hard_group_mask is not None and hard_factor > 1.0:
        hard_mask = np.asarray(system_hard_group_mask, dtype=bool).reshape(-1)
        if hard_mask.shape[0] != store.selected_rows.shape[0]:
            raise ValueError("system hard-group mask is not aligned with candidate groups")
        selected_hard = output[hard_mask[output]]
        if len(selected_hard) > 0:
            additional_factor = hard_factor - 1.0
            whole_repeats = int(np.floor(additional_factor))
            extras = [selected_hard.copy() for _ in range(whole_repeats)]
            fractional = additional_factor - float(whole_repeats)
            fractional_count = int(round(fractional * len(selected_hard)))
            if fractional_count > 0:
                extras.append(
                    rng.choice(
                        selected_hard, size=fractional_count, replace=False
                    ).astype(np.int64)
                )
            if extras:
                output = np.concatenate([output, *extras]).astype(np.int64)
    rng.shuffle(output)
    return output


def _query_grouped_training_batches(
    train_groups: np.ndarray,
    query_ids: np.ndarray,
    *,
    groups_per_batch: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Pack complete per-query group blocks without splitting an image."""

    groups = np.asarray(train_groups, dtype=np.int64).reshape(-1)
    owners = np.asarray(query_ids).astype(str).reshape(-1)
    capacity = int(groups_per_batch)
    if capacity <= 0:
        raise ValueError("groups_per_batch must be positive")
    if np.any((groups < 0) | (groups >= len(owners))):
        raise ValueError("training groups are outside the query owner array")
    if len(np.unique(groups)) != len(groups):
        raise ValueError("query-grouped batches do not support repeated groups")
    query_order = np.unique(owners[groups])
    query_order = query_order[rng.permutation(len(query_order))]
    blocks = [
        np.sort(groups[owners[groups] == query_id]) for query_id in query_order
    ]
    if any(len(block) > capacity for block in blocks):
        raise ValueError("one query has more groups than the batch capacity")
    batches: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    pending_count = 0
    for block in blocks:
        if pending and pending_count + len(block) > capacity:
            batches.append(np.concatenate(pending).astype(np.int64))
            pending = []
            pending_count = 0
        pending.append(block)
        pending_count += len(block)
    if pending:
        batches.append(np.concatenate(pending).astype(np.int64))
    if not np.array_equal(
        np.sort(np.concatenate(batches)), np.sort(groups)
    ):
        raise RuntimeError("query-grouped batching changed the training group set")
    return batches


def _validate_complete_hard_modes_in_batches(
    group_batches: Sequence[np.ndarray],
    hard_mode_ids: np.ndarray,
) -> dict[str, int]:
    """Require every structured pose mode to remain complete in one batch."""

    mode_ids = np.asarray(hard_mode_ids, dtype=np.int64)
    if mode_ids.ndim != 2:
        raise ValueError("hard_mode_ids must have shape (G, M)")
    group_batch_ids = np.full((mode_ids.shape[0],), -1, dtype=np.int64)
    for batch_index, raw_groups in enumerate(group_batches):
        groups = np.asarray(raw_groups, dtype=np.int64).reshape(-1)
        if np.any((groups < 0) | (groups >= mode_ids.shape[0])):
            raise ValueError("hard-mode training batch contains an invalid group")
        if np.any(group_batch_ids[groups] >= 0):
            raise ValueError("hard-mode training batches repeat a group")
        group_batch_ids[groups] = int(batch_index)

    present = mode_ids >= 0
    unique_modes = np.unique(mode_ids[present])
    for mode_id in unique_modes.tolist():
        mode_rows = np.flatnonzero(np.any(mode_ids == int(mode_id), axis=1))
        assigned = group_batch_ids[mode_rows]
        if np.any(assigned < 0):
            raise ValueError(
                "structured hard-mode training selection omitted support groups"
            )
        if len(np.unique(assigned)) != 1:
            raise ValueError(
                "structured hard-mode support groups cross training batches"
            )
    return {
        "hard_mode_count": int(len(unique_modes)),
        "hard_mode_group_incidence_count": int(np.sum(present)),
    }


def _system_hard_candidate_group_mask(
    labels: np.ndarray,
    prior_scores: np.ndarray,
    valid_mask: np.ndarray,
    eligible_groups: np.ndarray,
    *,
    ambiguous_margin: float,
    no_match_quantile: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mine complete top-L groups that reproduce inference-time identity errors."""

    positive = np.asarray(labels, dtype=bool)
    scores = np.asarray(prior_scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if positive.shape != scores.shape or valid.shape != scores.shape:
        raise ValueError("system hard-group arrays must have equal [G, L] shape")
    margin = float(ambiguous_margin)
    quantile = float(no_match_quantile)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("system hard-group ambiguous margin must be non-negative")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("system hard-group no-match quantile must be in [0, 1]")
    groups = np.asarray(eligible_groups, dtype=np.int64).reshape(-1)
    if np.any(groups < 0) or np.any(groups >= len(scores)):
        raise ValueError("eligible hard-group index is outside the candidate store")
    if np.any(np.sum(valid[groups], axis=1) <= 0):
        raise ValueError("every eligible hard group requires a valid candidate")

    masked_scores = np.where(valid, scores, -np.inf)
    best_columns = np.argmax(masked_scores, axis=1)
    rows = np.arange(len(scores), dtype=np.int64)
    has_positive = np.any(positive & valid, axis=1)
    rank1_positive = positive[rows, best_columns] & valid[rows, best_columns]
    rank2_to_l_positive = has_positive & ~rank1_positive

    positive_scores = np.max(
        np.where(positive & valid, scores, -np.inf), axis=1
    )
    wrong_scores = np.max(
        np.where(~positive & valid, scores, -np.inf), axis=1
    )
    ambiguous_wrong = (
        has_positive
        & np.isfinite(wrong_scores)
        & ((positive_scores - wrong_scores) <= margin)
    )

    no_match = ~has_positive
    eligible_no_match = groups[no_match[groups]]
    no_match_threshold = np.inf
    high_score_no_match = np.zeros((len(scores),), dtype=bool)
    if len(eligible_no_match) > 0:
        maxima = masked_scores[eligible_no_match, best_columns[eligible_no_match]]
        finite = maxima[np.isfinite(maxima)]
        if len(finite) > 0:
            no_match_threshold = float(np.quantile(finite, quantile))
            high_score_no_match = no_match & (
                masked_scores[rows, best_columns] >= no_match_threshold
            )

    eligible_mask = np.zeros((len(scores),), dtype=bool)
    eligible_mask[groups] = True
    hard = eligible_mask & (
        rank2_to_l_positive | ambiguous_wrong | high_score_no_match
    )
    audit = {
        "eligible_group_count": int(len(groups)),
        "hard_group_count": int(np.sum(hard)),
        "hard_group_fraction": float(np.sum(hard) / max(len(groups), 1)),
        "rank2_to_l_positive_group_count": int(
            np.sum(eligible_mask & rank2_to_l_positive)
        ),
        "ambiguous_wrong_competitor_group_count": int(
            np.sum(eligible_mask & ambiguous_wrong)
        ),
        "high_score_no_match_group_count": int(
            np.sum(eligible_mask & high_score_no_match)
        ),
        "ambiguous_margin": margin,
        "no_match_quantile": quantile,
        "no_match_score_threshold": (
            None if not np.isfinite(no_match_threshold) else no_match_threshold
        ),
    }
    return hard, audit


def _identity_transition_report(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
) -> dict[str, object]:
    """Report inference-relevant top-L identity changes against a fixed baseline."""

    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    if not (
        positive.shape == valid.shape == baseline.shape == candidate.shape
    ):
        raise ValueError("identity transition arrays must have equal [G, L] shape")
    if positive.ndim != 2 or np.any(np.sum(valid, axis=1) <= 0):
        raise ValueError("identity transition report requires non-empty top-L groups")

    baseline_columns = np.argmax(np.where(valid, baseline, -np.inf), axis=1)
    candidate_columns = np.argmax(np.where(valid, candidate, -np.inf), axis=1)
    rows = np.arange(len(positive), dtype=np.int64)
    has_positive = np.any(positive & valid, axis=1)
    baseline_correct = positive[rows, baseline_columns] & valid[rows, baseline_columns]
    candidate_correct = positive[rows, candidate_columns] & valid[rows, candidate_columns]
    switched = candidate_columns != baseline_columns

    # These are the exact rows where a correct top-L identity exists but the
    # frozen system prior hard-selects the wrong candidate.
    rescue_eligible = has_positive & ~baseline_correct
    rescued = rescue_eligible & candidate_correct
    corruption_eligible = baseline_correct
    corrupted = corruption_eligible & ~candidate_correct
    beneficial_switch = switched & ~baseline_correct & candidate_correct
    harmful_switch = switched & baseline_correct & ~candidate_correct
    no_match_switch = switched & ~has_positive

    def rate(count: int, denominator: int) -> float:
        return float(count / max(denominator, 1))

    rescue_count = int(np.sum(rescued))
    rescue_eligible_count = int(np.sum(rescue_eligible))
    corrupted_count = int(np.sum(corrupted))
    corruption_eligible_count = int(np.sum(corruption_eligible))
    switch_count = int(np.sum(switched))
    beneficial_count = int(np.sum(beneficial_switch))
    harmful_count = int(np.sum(harmful_switch))
    return {
        "group_count": int(len(positive)),
        "mappable_group_count": int(np.sum(has_positive)),
        "baseline_correct_count": int(np.sum(baseline_correct)),
        "candidate_correct_count": int(np.sum(candidate_correct)),
        "rank2_to_l_rescue_eligible_count": rescue_eligible_count,
        "rank2_to_l_rescue_count": rescue_count,
        "rank2_to_l_rescue_rate": rate(rescue_count, rescue_eligible_count),
        "baseline_correct_corruption_eligible_count": corruption_eligible_count,
        "baseline_correct_corruption_count": corrupted_count,
        "wrong_switch_rate": rate(corrupted_count, corruption_eligible_count),
        "switch_count": switch_count,
        "beneficial_switch_count": beneficial_count,
        "harmful_switch_count": harmful_count,
        "beneficial_switch_precision": rate(beneficial_count, switch_count),
        "no_match_group_switch_count": int(np.sum(no_match_switch)),
        "unchanged_wrong_mappable_count": int(
            np.sum(has_positive & ~baseline_correct & ~candidate_correct & ~switched)
        ),
    }


@torch.no_grad()
def _predict_edges(
    model: CandidateMapletMatcher,
    store: CandidateMapletEpisodeStore,
    edge_indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    dustbin_thresholds: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
    geometry_min_probability_thresholds: Sequence[float] = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
    ),
    export_candidate_view_embeddings: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    thresholds = tuple(float(value) for value in dustbin_thresholds)
    if not thresholds or any(value <= 0.0 or value >= 1.0 for value in thresholds):
        raise ValueError("dustbin thresholds must be within (0, 1)")
    geometry_min_thresholds = tuple(
        float(value) for value in geometry_min_probability_thresholds
    )
    if not geometry_min_thresholds or any(
        value <= 0.0 or value >= 1.0 for value in geometry_min_thresholds
    ):
        raise ValueError("geometry probability thresholds must be within (0, 1)")
    outputs = {
        "candidate_probability": np.full((store.edge_count,), -np.inf, dtype=np.float32),
        "anchor_assignment_probability": np.full((store.edge_count,), -np.inf, dtype=np.float32),
        "candidate_assignment_geomean": np.full((store.edge_count,), -np.inf, dtype=np.float32),
        "set_candidate_probability": np.full((store.edge_count,), -np.inf, dtype=np.float32),
        "set_candidate_conditional_probability_DIAGNOSTIC_ONLY": np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        ),
        "set_identity_evidence_probability_DIAGNOSTIC_ONLY": np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        ),
        "set_identity_evidence_conditional_probability_DIAGNOSTIC_ONLY": np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        ),
        "set_identity_evidence_dustbin_probability_DIAGNOSTIC_ONLY": np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        ),
        "set_candidate_probability_dustbin_argmax_DIAGNOSTIC_ONLY": np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        ),
        "set_assignment_geomean": np.full((store.edge_count,), -np.inf, dtype=np.float32),
        "set_assignment_geomean_dustbin_argmax_DIAGNOSTIC_ONLY": np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        ),
        "set_dustbin_probability_DIAGNOSTIC_ONLY": np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        ),
    }
    if bool(model.config.factorized_set_posterior_enabled):
        outputs.update(
            {
                "factorized_set_candidate_probability": np.full(
                    (store.edge_count,), -np.inf, dtype=np.float32
                ),
                "factorized_set_dustbin_probability_DIAGNOSTIC_ONLY": np.full(
                    (store.edge_count,), -np.inf, dtype=np.float32
                ),
                "factorized_identity_evidence_probability_DIAGNOSTIC_ONLY": np.full(
                    (store.edge_count,), -np.inf, dtype=np.float32
                ),
                "factorized_top_l_availability_probability_DIAGNOSTIC_ONLY": np.full(
                    (store.edge_count,), -np.inf, dtype=np.float32
                ),
            }
        )
    geometry_tags = tuple(
        f"p{int(round(float(value))):02d}px"
        for value in model.config.geometry_validity_thresholds_px
    )
    if bool(model.config.geometry_validity_enabled):
        for tag in geometry_tags:
            outputs[f"geometry_{tag}"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
            outputs[f"geometry_{tag}_prior_row_confidence"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
        outputs["geometry_ordinal_geomean"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs["geometry_ordinal_visibility_geomean"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs["geometry_center_precision_score"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs["geometry_center_precision_prior_row_confidence"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs["candidate_visibility_probability"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        for threshold in geometry_min_thresholds:
            suffix = f"p{int(round(threshold * 100)):02d}"
            outputs[f"geometry_p02px_min_{suffix}"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
            outputs[f"geometry_p02px_min_{suffix}_prior_row_confidence"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
            outputs[f"geometry_center_precision_min_{suffix}"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
    if bool(model.config.rescue_policy_enabled):
        outputs["rescue_candidate_probability"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs["rescue_keep_probability_DIAGNOSTIC_ONLY"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs["rescue_policy_resolved"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs["rescue_action_probability"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
    for view_rank in range(store.support_view_count):
        outputs[f"support_view_probability_{view_rank}"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs[f"support_view_prior_probability_{view_rank}"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
    for threshold in thresholds:
        suffix = f"p{int(round(threshold * 100)):02d}"
        outputs[f"set_candidate_probability_dustbin_{suffix}"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        outputs[f"set_assignment_geomean_dustbin_{suffix}"] = np.full(
            (store.edge_count,), -np.inf, dtype=np.float32
        )
        if bool(model.config.geometry_validity_enabled):
            outputs[f"geometry_ordinal_geomean_dustbin_{suffix}"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
            outputs[
                f"geometry_ordinal_visibility_geomean_dustbin_{suffix}"
            ] = np.full((store.edge_count,), -np.inf, dtype=np.float32)
            outputs[f"geometry_p02px_dustbin_{suffix}"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
            outputs[f"geometry_center_precision_dustbin_{suffix}"] = np.full(
                (store.edge_count,), -np.inf, dtype=np.float32
            )
    view_embeddings = []
    for view_rank in range(store.support_view_count):
        embeddings = np.full(
            (store.edge_count, int(model.config.model_dim)), np.nan, dtype=np.float32
        )
        for start in range(0, len(edge_indices), int(batch_size)):
            edges = edge_indices[start : start + int(batch_size)]
            ranks = np.full(edges.shape, view_rank, dtype=np.int64)
            batch = store.batch(edges, view_ranks=ranks).to(device)
            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                result = model(batch, return_ragged_query_probabilities=False)
            candidate = torch.sigmoid(result["candidate_logits"].float()).cpu().numpy()
            candidate_embeddings = result.get("set_identity_candidate_embeddings")
            if not isinstance(candidate_embeddings, torch.Tensor):
                raise TypeError(
                    "matcher output is missing set_identity_candidate_embeddings"
                )
            embeddings[edges] = candidate_embeddings.float().cpu().numpy()
            batched_assignment = result.get("query_log_probabilities_batched")
            if not isinstance(batched_assignment, torch.Tensor):
                raise TypeError("matcher output is missing batched query probabilities")
            assignment = (
                torch.exp(batched_assignment[:, 0, 0])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            combined = np.sqrt(np.clip(candidate, 0.0, 1.0) * np.clip(assignment, 0.0, 1.0))
            for name, values in (
                ("candidate_probability", candidate),
                ("anchor_assignment_probability", assignment),
                ("candidate_assignment_geomean", combined),
            ):
                outputs[name][edges] = np.maximum(outputs[name][edges], values)
        view_embeddings.append(embeddings)

    if bool(export_candidate_view_embeddings):
        for view_rank, embeddings in enumerate(view_embeddings):
            outputs[f"candidate_view_embedding_{view_rank}"] = embeddings

    top_l = store.candidate_top_k
    requested = np.zeros((store.edge_count,), dtype=bool)
    requested[np.asarray(edge_indices, dtype=np.int64)] = True
    group_ids = np.unique(np.asarray(edge_indices, dtype=np.int64) // top_l)
    complete_edges = group_ids[:, None] * top_l + np.arange(top_l)[None]
    if not np.all(requested[complete_edges]):
        raise ValueError("candidate-set prediction requires complete top-L groups")
    group_batch_size = max(1, int(batch_size) // top_l)
    for start in range(0, len(group_ids), group_batch_size):
        groups = group_ids[start : start + group_batch_size]
        edge_matrix = groups[:, None] * top_l + np.arange(top_l)[None]
        flat_edges = edge_matrix.reshape(-1)
        values = np.stack(
            [embeddings[flat_edges] for embeddings in view_embeddings], axis=1
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("candidate-set prediction encountered missing view embeddings")
        candidate_views = torch.from_numpy(values).to(device)
        prior_scores = torch.from_numpy(
            np.asarray(
                store.static_features[
                    groups, :, int(model.config.candidate_prior_index)
                ],
                dtype=np.float32,
            )
        ).to(device)
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            aggregated, view_weights_t = model.aggregate_candidate_views(candidate_views)
            if bool(model.config.candidate_view_marginalization_enabled):
                resolved = model.resolve_candidate_view_sets(
                    candidate_views.reshape(
                        len(groups),
                        top_l,
                        store.support_view_count,
                        int(candidate_views.shape[2]),
                    ),
                    view_weights_t.reshape(
                        len(groups), top_l, store.support_view_count
                    ),
                    candidate_prior_scores=prior_scores,
                )
            else:
                resolved = model.resolve_candidate_sets(
                    aggregated.reshape(len(groups), top_l, int(aggregated.shape[1])),
                    candidate_prior_scores=prior_scores,
                )
            joint_logits = torch.cat(
                [resolved["candidate_logits"], resolved["dustbin_logits"][:, None]], dim=1
            )
            joint_probabilities = torch.softmax(joint_logits.float(), dim=1)
            evidence_logits = resolved.get("candidate_evidence_logits")
            if not isinstance(evidence_logits, torch.Tensor):
                raise TypeError("candidate-set output is missing identity evidence logits")
            evidence_joint_probabilities = torch.softmax(
                torch.cat(
                    [evidence_logits, resolved["dustbin_logits"][:, None]], dim=1
                ).float(),
                dim=1,
            )
            candidate_conditional_probabilities_t = torch.softmax(
                resolved["candidate_logits"].float(), dim=1
            )
            evidence_conditional_probabilities_t = torch.softmax(
                evidence_logits.float(), dim=1
            )
            factorized_candidate_probabilities_t = None
            factorized_evidence_probabilities_t = None
            factorized_dustbin_probabilities_t = None
            factorized_top_l_availability_probabilities_t = None
            if bool(model.config.factorized_set_posterior_enabled):
                availability_logits = resolved.get("top_l_availability_logits")
                if not isinstance(availability_logits, torch.Tensor):
                    raise TypeError(
                        "factorized candidate-set output is missing top-L availability logits"
                    )
                (
                    factorized_candidate_probabilities_t,
                    factorized_dustbin_probabilities_t,
                    _factorized_conditional,
                ) = factorized_candidate_posterior(
                    resolved["candidate_logits"], availability_logits
                )
                factorized_top_l_availability_probabilities_t = (
                    1.0 - factorized_dustbin_probabilities_t
                )
                factorized_evidence_probabilities_t = (
                    factorized_top_l_availability_probabilities_t[:, None]
                    * evidence_conditional_probabilities_t
                )
            geometry_logits = resolved.get("geometry_validity_logits")
            visibility_logits = resolved.get("candidate_visibility_logits")
            if bool(model.config.geometry_validity_enabled):
                if not isinstance(geometry_logits, torch.Tensor) or not isinstance(
                    visibility_logits, torch.Tensor
                ):
                    raise TypeError("enabled geometry validity heads produced no logits")
                geometry_probabilities_t = torch.sigmoid(geometry_logits.float())
                visibility_probabilities_t = torch.sigmoid(visibility_logits.float())
            rescue_candidate_logits = resolved.get("rescue_candidate_logits")
            rescue_keep_logits = resolved.get("rescue_keep_logits")
            baseline_candidate_indices = resolved.get("baseline_candidate_indices")
            if bool(model.config.rescue_policy_enabled):
                if (
                    not isinstance(rescue_candidate_logits, torch.Tensor)
                    or not isinstance(rescue_keep_logits, torch.Tensor)
                    or not isinstance(baseline_candidate_indices, torch.Tensor)
                ):
                    raise TypeError("enabled rescue policy heads produced no logits")
                rescue_joint_probabilities_t = torch.softmax(
                    torch.cat(
                        [rescue_candidate_logits, rescue_keep_logits[:, None]], dim=1
                    ).float(),
                    dim=1,
                )
        resolved_view_weights_t = resolved.get("support_view_probabilities")
        resolved_view_prior_t = resolved.get("support_view_prior_probabilities")
        if not bool(model.config.candidate_view_marginalization_enabled):
            resolved_view_weights_t = view_weights_t.reshape(
                len(groups), top_l, store.support_view_count
            )
            resolved_view_prior_t = resolved_view_weights_t
        if not isinstance(resolved_view_weights_t, torch.Tensor):
            raise TypeError("candidate-set output is missing support-view probabilities")
        if not isinstance(resolved_view_prior_t, torch.Tensor):
            raise TypeError("candidate-set output is missing support-view priors")
        view_weights = (
            resolved_view_weights_t.reshape(-1, store.support_view_count)
            .float()
            .cpu()
            .numpy()
        )
        view_priors = (
            resolved_view_prior_t.reshape(-1, store.support_view_count)
            .float()
            .cpu()
            .numpy()
        )
        for view_rank in range(store.support_view_count):
            outputs[f"support_view_probability_{view_rank}"][flat_edges] = (
                view_weights[:, view_rank].astype(np.float32)
            )
            outputs[f"support_view_prior_probability_{view_rank}"][flat_edges] = (
                view_priors[:, view_rank].astype(np.float32)
            )
        candidate_probabilities = joint_probabilities[:, :top_l].cpu().numpy()
        dustbin_probabilities = joint_probabilities[:, top_l].cpu().numpy()
        evidence_candidate_probabilities = (
            evidence_joint_probabilities[:, :top_l].cpu().numpy()
        )
        evidence_dustbin_probabilities = (
            evidence_joint_probabilities[:, top_l].cpu().numpy()
        )
        candidate_conditional_probabilities = (
            candidate_conditional_probabilities_t.cpu().numpy()
        )
        evidence_conditional_probabilities = (
            evidence_conditional_probabilities_t.cpu().numpy()
        )
        prior_probabilities = prior_scores.float().cpu().numpy()
        if bool(model.config.rescue_policy_enabled):
            rescue_joint_probabilities = rescue_joint_probabilities_t.cpu().numpy()
            rescue_candidate_probabilities = rescue_joint_probabilities[:, :top_l]
            rescue_keep_probabilities = rescue_joint_probabilities[:, top_l]
            (
                selected_indices,
                resolved_policy,
                switch,
                _action_margin,
                action_probability,
            ) = resolve_rescue_policy_scores(
                rescue_candidate_probabilities,
                rescue_keep_probabilities,
                prior_probabilities,
                valid_mask=np.isfinite(prior_probabilities),
            )
            expected_baseline_indices = baseline_candidate_indices.long().cpu().numpy()
            baseline_indices = np.argmax(prior_probabilities, axis=1)
            if not np.array_equal(baseline_indices, expected_baseline_indices):
                raise RuntimeError("NumPy and Torch baseline candidate resolution disagree")
            outputs["rescue_candidate_probability"][flat_edges] = (
                rescue_candidate_probabilities.reshape(-1).astype(np.float32)
            )
            outputs["rescue_keep_probability_DIAGNOSTIC_ONLY"][flat_edges] = np.repeat(
                rescue_keep_probabilities[:, None], top_l, axis=1
            ).reshape(-1).astype(np.float32)
            outputs["rescue_policy_resolved"][flat_edges] = resolved_policy.reshape(
                -1
            )
            outputs["rescue_action_probability"][flat_edges] = action_probability.reshape(
                -1
            )
        outputs["set_dustbin_probability_DIAGNOSTIC_ONLY"][flat_edges] = np.repeat(
            dustbin_probabilities[:, None], top_l, axis=1
        ).reshape(-1).astype(np.float32)
        outputs["set_identity_evidence_probability_DIAGNOSTIC_ONLY"][flat_edges] = (
            evidence_candidate_probabilities.reshape(-1).astype(np.float32)
        )
        outputs[
            "set_candidate_conditional_probability_DIAGNOSTIC_ONLY"
        ][flat_edges] = candidate_conditional_probabilities.reshape(-1).astype(
            np.float32
        )
        outputs[
            "set_identity_evidence_conditional_probability_DIAGNOSTIC_ONLY"
        ][flat_edges] = evidence_conditional_probabilities.reshape(-1).astype(
            np.float32
        )
        outputs[
            "set_identity_evidence_dustbin_probability_DIAGNOSTIC_ONLY"
        ][flat_edges] = np.repeat(
            evidence_dustbin_probabilities[:, None], top_l, axis=1
        ).reshape(-1).astype(np.float32)
        if bool(model.config.factorized_set_posterior_enabled):
            if (
                factorized_candidate_probabilities_t is None
                or factorized_evidence_probabilities_t is None
                or factorized_dustbin_probabilities_t is None
                or factorized_top_l_availability_probabilities_t is None
            ):
                raise RuntimeError("factorized posterior tensors were not initialized")
            factorized_candidate_probabilities = (
                factorized_candidate_probabilities_t.cpu().numpy()
            )
            factorized_evidence_probabilities = (
                factorized_evidence_probabilities_t.cpu().numpy()
            )
            factorized_dustbin_probabilities = (
                factorized_dustbin_probabilities_t.cpu().numpy()
            )
            factorized_top_l_availability_probabilities = (
                factorized_top_l_availability_probabilities_t.cpu().numpy()
            )
            outputs["factorized_set_candidate_probability"][flat_edges] = (
                factorized_candidate_probabilities.reshape(-1).astype(np.float32)
            )
            outputs[
                "factorized_identity_evidence_probability_DIAGNOSTIC_ONLY"
            ][flat_edges] = factorized_evidence_probabilities.reshape(-1).astype(
                np.float32
            )
            outputs[
                "factorized_set_dustbin_probability_DIAGNOSTIC_ONLY"
            ][flat_edges] = np.repeat(
                factorized_dustbin_probabilities[:, None], top_l, axis=1
            ).reshape(-1).astype(np.float32)
            outputs[
                "factorized_top_l_availability_probability_DIAGNOSTIC_ONLY"
            ][flat_edges] = np.repeat(
                factorized_top_l_availability_probabilities[:, None], top_l, axis=1
            ).reshape(-1).astype(np.float32)
        assignment_probabilities = outputs["anchor_assignment_probability"][flat_edges].reshape(
            len(groups), top_l
        )
        combined = np.sqrt(
            np.clip(candidate_probabilities, 0.0, 1.0)
            * np.clip(assignment_probabilities, 0.0, 1.0)
        ).astype(np.float32)
        argmax_rejected = dustbin_probabilities >= np.max(candidate_probabilities, axis=1)
        candidate_argmax_rejected = candidate_probabilities.copy()
        combined_argmax_rejected = combined.copy()
        candidate_argmax_rejected[argmax_rejected] = -np.inf
        combined_argmax_rejected[argmax_rejected] = -np.inf
        for name, values in (
            ("set_candidate_probability", candidate_probabilities),
            (
                "set_candidate_probability_dustbin_argmax_DIAGNOSTIC_ONLY",
                candidate_argmax_rejected,
            ),
            ("set_assignment_geomean", combined),
            (
                "set_assignment_geomean_dustbin_argmax_DIAGNOSTIC_ONLY",
                combined_argmax_rejected,
            ),
        ):
            outputs[name][flat_edges] = values.reshape(-1).astype(np.float32)
        geometry_ordinal = None
        geometry_ordinal_visibility = None
        geometry_p2 = None
        geometry_center_precision = None
        if bool(model.config.geometry_validity_enabled):
            geometry_probabilities = geometry_probabilities_t.cpu().numpy()
            visibility_probabilities = visibility_probabilities_t.cpu().numpy()
            for threshold_index, tag in enumerate(geometry_tags):
                threshold_probabilities = geometry_probabilities[:, :, threshold_index]
                outputs[f"geometry_{tag}"][flat_edges] = threshold_probabilities.reshape(
                    -1
                ).astype(np.float32)
                outputs[f"geometry_{tag}_prior_row_confidence"][flat_edges] = (
                    _preserve_prior_row_confidence(
                        threshold_probabilities, prior_probabilities
                    ).reshape(-1)
                )
            geometry_ordinal = np.prod(
                np.clip(geometry_probabilities, 1e-8, 1.0), axis=2
            ) ** (1.0 / float(len(geometry_tags)))
            geometry_ordinal_visibility = (
                np.prod(np.clip(geometry_probabilities, 1e-8, 1.0), axis=2)
                * np.clip(visibility_probabilities, 1e-8, 1.0)
            ) ** (1.0 / float(len(geometry_tags) + 1))
            geometry_p2 = geometry_probabilities[:, :, 1]
            geometry_center_precision = (
                np.clip(geometry_probabilities[:, :, 0], 1e-8, 1.0)
                * np.clip(geometry_p2, 1e-8, 1.0) ** 2
                * np.clip(geometry_probabilities[:, :, 2], 1e-8, 1.0)
            ) ** 0.25
            outputs["geometry_ordinal_geomean"][flat_edges] = geometry_ordinal.reshape(
                -1
            ).astype(np.float32)
            outputs["geometry_ordinal_visibility_geomean"][flat_edges] = (
                geometry_ordinal_visibility.reshape(-1).astype(np.float32)
            )
            outputs["geometry_center_precision_score"][flat_edges] = (
                geometry_center_precision.reshape(-1).astype(np.float32)
            )
            outputs["geometry_center_precision_prior_row_confidence"][flat_edges] = (
                _preserve_prior_row_confidence(
                    geometry_center_precision, prior_probabilities
                ).reshape(-1)
            )
            outputs["candidate_visibility_probability"][flat_edges] = (
                visibility_probabilities.reshape(-1).astype(np.float32)
            )
            for threshold in geometry_min_thresholds:
                suffix = f"p{int(round(threshold * 100)):02d}"
                accepted = geometry_p2 >= float(threshold)
                filtered_p2 = geometry_p2.copy()
                filtered_center = geometry_center_precision.copy()
                filtered_p2[~accepted] = -np.inf
                filtered_center[~accepted] = -np.inf
                outputs[f"geometry_p02px_min_{suffix}"][flat_edges] = (
                    filtered_p2.reshape(-1).astype(np.float32)
                )
                outputs[
                    f"geometry_p02px_min_{suffix}_prior_row_confidence"
                ][flat_edges] = _preserve_prior_row_confidence(
                    filtered_p2, prior_probabilities
                ).reshape(-1)
                outputs[f"geometry_center_precision_min_{suffix}"][flat_edges] = (
                    filtered_center.reshape(-1).astype(np.float32)
                )
        for threshold in thresholds:
            suffix = f"p{int(round(threshold * 100)):02d}"
            rejected = dustbin_probabilities >= float(threshold)
            candidate_rejected = candidate_probabilities.copy()
            combined_rejected = combined.copy()
            candidate_rejected[rejected] = -np.inf
            combined_rejected[rejected] = -np.inf
            outputs[f"set_candidate_probability_dustbin_{suffix}"][flat_edges] = (
                candidate_rejected.reshape(-1).astype(np.float32)
            )
            outputs[f"set_assignment_geomean_dustbin_{suffix}"][flat_edges] = (
                combined_rejected.reshape(-1).astype(np.float32)
            )
            if geometry_ordinal is not None and geometry_ordinal_visibility is not None:
                ordinal_rejected = geometry_ordinal.copy()
                ordinal_visibility_rejected = geometry_ordinal_visibility.copy()
                ordinal_rejected[rejected] = -np.inf
                ordinal_visibility_rejected[rejected] = -np.inf
                outputs[f"geometry_ordinal_geomean_dustbin_{suffix}"][flat_edges] = (
                    ordinal_rejected.reshape(-1).astype(np.float32)
                )
                outputs[
                    f"geometry_ordinal_visibility_geomean_dustbin_{suffix}"
                ][flat_edges] = ordinal_visibility_rejected.reshape(-1).astype(np.float32)
            if geometry_p2 is not None and geometry_center_precision is not None:
                p2_rejected = geometry_p2.copy()
                center_rejected = geometry_center_precision.copy()
                p2_rejected[rejected] = -np.inf
                center_rejected[rejected] = -np.inf
                outputs[f"geometry_p02px_dustbin_{suffix}"][flat_edges] = (
                    p2_rejected.reshape(-1).astype(np.float32)
                )
                outputs[f"geometry_center_precision_dustbin_{suffix}"][flat_edges] = (
                    center_rejected.reshape(-1).astype(np.float32)
                )
    return outputs


def _global_assignment_strategy_scores(
    predictions: dict[str, np.ndarray],
    strategy_name: str,
    *,
    group_count: int,
    candidate_top_k: int,
) -> tuple[np.ndarray, float | np.ndarray | None, str]:
    """Put set probabilities and their learned null in a joint MAP score space."""

    strategy = str(strategy_name)
    scores = np.asarray(predictions[strategy], dtype=np.float32).reshape(
        int(group_count), int(candidate_top_k)
    )
    factorized_strategy = strategy == "factorized_set_candidate_probability"
    if not strategy.startswith("set_candidate_probability") and not factorized_strategy:
        return scores, None, "raw_strategy_score_without_learned_null"

    calibrated_prefix = "set_candidate_probability_cal_"
    if strategy.startswith(calibrated_prefix):
        dustbin_key = (
            "set_dustbin_probability_cal_"
            f"{strategy[len(calibrated_prefix):]}_DIAGNOSTIC_ONLY"
        )
    elif factorized_strategy:
        dustbin_key = "factorized_set_dustbin_probability_DIAGNOSTIC_ONLY"
    else:
        dustbin_key = "set_dustbin_probability_DIAGNOSTIC_ONLY"
    if dustbin_key not in predictions:
        raise ValueError(
            f"{strategy} requires the jointly normalized set dustbin probability"
        )
    dustbin_matrix = np.asarray(predictions[dustbin_key], dtype=np.float32).reshape(
        int(group_count), int(candidate_top_k)
    )
    requested = np.any(np.isfinite(scores), axis=1)
    if np.any(requested):
        requested_dustbins = dustbin_matrix[requested]
        if not np.all(np.isfinite(requested_dustbins)):
            raise ValueError("requested set predictions have missing dustbin probability")
        if not np.allclose(
            requested_dustbins,
            requested_dustbins[:, :1],
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError("set dustbin probability differs within a candidate group")
    dustbins = dustbin_matrix[:, 0]
    finite = np.isfinite(scores)
    if np.any((scores[finite] < 0.0) | (scores[finite] > 1.0)):
        raise ValueError("set candidate probabilities must be within [0, 1]")
    if np.any(requested & ((dustbins < 0.0) | (dustbins > 1.0))):
        raise ValueError("set dustbin probabilities must be within [0, 1]")
    if (
        strategy in {
            "set_candidate_probability",
            "factorized_set_candidate_probability",
        }
        or strategy.startswith(calibrated_prefix)
    ) and np.any(requested):
        probability_mass = np.sum(scores[requested], axis=1) + dustbins[requested]
        if not np.allclose(probability_mass, 1.0, atol=2e-5, rtol=2e-5):
            raise ValueError("candidate and dustbin probabilities do not conserve mass")

    # Maximizing the product of independent group posteriors under track
    # capacity is equivalent to maximizing summed candidate-vs-null log odds.
    # Rows not requested in this split remain all -inf and deterministically
    # choose their zero-valued private dustbin.
    log_odds = np.full_like(scores, -np.inf, dtype=np.float32)
    safe_dustbins = np.clip(dustbins, 1e-8, 1.0)
    log_odds[finite] = (
        np.log(np.clip(scores[finite], 1e-8, 1.0))
        - np.repeat(np.log(safe_dustbins)[:, None], int(candidate_top_k), axis=1)[
            finite
        ]
    ).astype(np.float32)
    return (
        log_odds,
        np.zeros((int(group_count),), dtype=np.float32),
        "joint_set_posterior_candidate_vs_null_log_odds",
    )


def _recalibrate_set_posterior(
    candidate_probabilities: np.ndarray,
    dustbin_probabilities: np.ndarray,
    candidate_prior_scores: np.ndarray,
    *,
    source_prior_scale: float,
    target_prior_scale: float,
    dustbin_logit_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly rescale the coarse prior and null logit of a joint set posterior."""

    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    dustbin = np.asarray(dustbin_probabilities, dtype=np.float64)
    prior = np.asarray(candidate_prior_scores, dtype=np.float64)
    if candidate.ndim != 2 or prior.shape != candidate.shape:
        raise ValueError("candidate probabilities and prior scores must be aligned matrices")
    if dustbin.shape != (candidate.shape[0],):
        raise ValueError("dustbin probabilities must provide one value per candidate group")
    for name, value in (
        ("source_prior_scale", source_prior_scale),
        ("target_prior_scale", target_prior_scale),
        ("dustbin_logit_bias", dustbin_logit_bias),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(source_prior_scale) < 0.0 or float(target_prior_scale) < 0.0:
        raise ValueError("candidate prior scales must be non-negative")

    output_candidate = np.full(candidate.shape, -np.inf, dtype=np.float32)
    output_dustbin = np.full(dustbin.shape, -np.inf, dtype=np.float32)
    requested = np.isfinite(dustbin) & np.any(np.isfinite(candidate), axis=1)
    if not np.any(requested):
        return output_candidate, output_dustbin
    valid = np.isfinite(candidate[requested]) & np.isfinite(prior[requested])
    if np.any(np.sum(valid, axis=1) <= 0):
        raise ValueError("every requested group requires a finite candidate and prior")
    requested_candidate = candidate[requested]
    requested_dustbin = dustbin[requested]
    if np.any(requested_candidate[valid] < 0.0) or np.any(
        requested_candidate[valid] > 1.0
    ):
        raise ValueError("candidate probabilities must be within [0, 1]")
    if np.any(requested_dustbin < 0.0) or np.any(requested_dustbin > 1.0):
        raise ValueError("dustbin probabilities must be within [0, 1]")
    probability_mass = np.sum(
        np.where(valid, requested_candidate, 0.0), axis=1
    ) + requested_dustbin
    if not np.allclose(probability_mass, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("candidate and dustbin probabilities do not conserve mass")

    requested_prior = prior[requested]
    prior_count = np.sum(valid, axis=1, keepdims=True)
    prior_center = np.sum(
        np.where(valid, requested_prior, 0.0), axis=1, keepdims=True
    ) / prior_count
    candidate_logits = np.full(requested_candidate.shape, -np.inf, dtype=np.float64)
    candidate_logits[valid] = np.log(
        np.clip(requested_candidate[valid], 1e-12, 1.0)
    )
    candidate_logits[valid] += (
        float(target_prior_scale) - float(source_prior_scale)
    ) * np.broadcast_to(requested_prior - prior_center, valid.shape)[valid]
    dustbin_logits = np.log(np.clip(requested_dustbin, 1e-12, 1.0)) + float(
        dustbin_logit_bias
    )
    joint_logits = np.concatenate([candidate_logits, dustbin_logits[:, None]], axis=1)
    maximum = np.max(joint_logits, axis=1, keepdims=True)
    joint_exp = np.exp(joint_logits - maximum)
    joint_probability = joint_exp / np.sum(joint_exp, axis=1, keepdims=True)
    output_candidate[requested] = joint_probability[:, :-1].astype(np.float32)
    output_dustbin[requested] = joint_probability[:, -1].astype(np.float32)
    return output_candidate, output_dustbin


def _factorized_top_l_availability_report(
    predictions: dict[str, np.ndarray],
    store: CandidateMapletEpisodeStore,
    *,
    row_mask: np.ndarray,
) -> dict[str, object] | None:
    key = "factorized_top_l_availability_probability_DIAGNOSTIC_ONLY"
    if key not in predictions:
        return None
    probabilities = np.asarray(predictions[key], dtype=np.float32).reshape(
        len(store.selected_rows), store.candidate_top_k
    )
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    if mask.shape != (len(store.selected_rows),):
        raise ValueError("top-L availability row mask is not aligned")
    selected = probabilities[mask]
    if len(selected) == 0:
        return None
    if not np.all(np.isfinite(selected)):
        raise ValueError("top-L availability predictions are incomplete")
    if not np.allclose(selected, selected[:, :1], atol=1e-7, rtol=1e-7):
        raise ValueError("top-L availability probability differs within a group")
    target = np.any(store.labels[mask] & store.valid_edges[mask], axis=1)
    probability = selected[:, 0]
    report = confidence_metrics(target, probability)
    prediction = probability >= 0.5
    report.update(
        {
            "target_semantics": "fixed_top_l_contains_track_with_residual_le_positive_threshold",
            "group_count": int(len(target)),
            "positive_group_count": int(np.sum(target)),
            "accuracy_at_0p5": float(np.mean(prediction == target)),
            "predicted_positive_rate_at_0p5": float(np.mean(prediction)),
        }
    )
    return report


def _geometry_probability_report(
    predictions: dict[str, np.ndarray],
    store: CandidateMapletEpisodeStore,
    *,
    row_mask: np.ndarray,
    thresholds_px: Sequence[float],
    visibility_supervised: bool = True,
) -> dict[str, object] | None:
    tags = tuple(f"p{int(round(float(value))):02d}px" for value in thresholds_px)
    required = [
        *(f"geometry_{tag}" for tag in tags),
        "candidate_visibility_probability",
        "set_dustbin_probability_DIAGNOSTIC_ONLY",
    ]
    if not all(name in predictions for name in required):
        return None
    rows = np.asarray(row_mask, dtype=bool)
    if rows.shape != (len(store.selected_rows),):
        raise ValueError("geometry calibration row mask has an incompatible shape")
    valid = np.asarray(store.valid_edges, dtype=bool) & rows[:, None]
    residuals = np.asarray(store.candidate_residuals_px, dtype=np.float32)
    report: dict[str, object] = {"thresholds_px": {}}
    probabilities_by_threshold = []
    for threshold, tag in zip(thresholds_px, tags):
        probabilities = np.asarray(predictions[f"geometry_{tag}"], dtype=np.float32).reshape(
            residuals.shape
        )
        metric_mask = valid & np.isfinite(probabilities)
        labels = residuals <= float(threshold)
        report["thresholds_px"][str(float(threshold))] = confidence_metrics(
            labels[metric_mask], probabilities[metric_mask]
        )
        probabilities_by_threshold.append(probabilities)
    stacked = np.stack(probabilities_by_threshold, axis=2)
    monotonic_valid = valid & np.all(np.isfinite(stacked), axis=2)
    violations = (stacked[:, :, 0] > stacked[:, :, 1]) | (
        stacked[:, :, 1] > stacked[:, :, 2]
    )
    report["monotonic_violation_rate"] = float(
        np.mean(violations[monotonic_valid])
    )
    visibility_probabilities = np.asarray(
        predictions["candidate_visibility_probability"], dtype=np.float32
    ).reshape(residuals.shape)
    visibility_mask = valid & np.isfinite(visibility_probabilities)
    report["visibility"] = (
        confidence_metrics(
            np.isfinite(residuals)[visibility_mask],
            visibility_probabilities[visibility_mask],
        )
        if bool(visibility_supervised)
        else {"supervision_enabled": False}
    )
    dustbin_probabilities = np.asarray(
        predictions["set_dustbin_probability_DIAGNOSTIC_ONLY"], dtype=np.float32
    ).reshape(residuals.shape)[:, 0]
    valid_rows = rows & np.isfinite(dustbin_probabilities)
    positive_threshold = float(thresholds_px[1])
    no_match_labels = ~np.any(
        (residuals <= positive_threshold) & np.asarray(store.valid_edges, dtype=bool),
        axis=1,
    )
    report["no_match"] = confidence_metrics(
        no_match_labels[valid_rows], dustbin_probabilities[valid_rows]
    )
    return report


def _rescue_policy_report(
    predictions: dict[str, np.ndarray],
    store: CandidateMapletEpisodeStore,
    baseline_scores: np.ndarray,
    *,
    row_mask: np.ndarray,
    candidate_threshold_px: float,
    baseline_invalid_threshold_px: float,
    action_margin_threshold: float = 0.0,
) -> dict[str, object] | None:
    required = {
        "rescue_candidate_probability",
        "rescue_keep_probability_DIAGNOSTIC_ONLY",
    }
    if not required.issubset(predictions):
        return None
    top_l = int(store.candidate_top_k)
    candidate_probabilities = np.asarray(
        predictions["rescue_candidate_probability"], dtype=np.float32
    ).reshape(-1, top_l)
    keep_probabilities = np.asarray(
        predictions["rescue_keep_probability_DIAGNOSTIC_ONLY"], dtype=np.float32
    ).reshape(-1, top_l)[:, 0]
    baseline = np.asarray(baseline_scores, dtype=np.float32)
    valid = np.asarray(store.valid_edges, dtype=bool)
    baseline_columns = np.argmax(np.where(valid, baseline, -np.inf), axis=1)
    residuals = np.asarray(store.candidate_residuals_px, dtype=np.float32)
    baseline_residuals = residuals[np.arange(len(residuals)), baseline_columns]
    targets = (
        valid
        & (baseline_residuals[:, None] > float(baseline_invalid_threshold_px))
        & (residuals <= float(candidate_threshold_px))
    )
    targets[np.arange(len(targets)), baseline_columns] = False
    rows = np.asarray(row_mask, dtype=bool) & np.isfinite(keep_probabilities)
    candidate_samples = rows[:, None] & valid & np.isfinite(candidate_probabilities)
    candidate_metrics = confidence_metrics(
        targets[candidate_samples], candidate_probabilities[candidate_samples]
    )
    has_rescue = np.any(targets, axis=1)
    keep_metrics = confidence_metrics(~has_rescue[rows], keep_probabilities[rows])
    row_candidate_probabilities = candidate_probabilities[rows]
    row_keep_probabilities = keep_probabilities[rows]
    row_baseline = baseline[rows]
    row_valid = valid[rows]
    row_targets = targets[rows]
    row_has_rescue = has_rescue[rows]
    row_residuals = residuals[rows]
    row_baseline_columns = baseline_columns[rows]
    (
        selected_candidates,
        _resolved,
        switched,
        action_margins,
        _action_scores,
    ) = resolve_rescue_policy_scores(
        row_candidate_probabilities,
        row_keep_probabilities,
        row_baseline,
        action_margin_threshold=float(action_margin_threshold),
        valid_mask=row_valid,
    )
    action_correct = np.where(
        switched,
        row_targets[np.arange(len(row_targets)), selected_candidates],
        ~row_has_rescue,
    )
    baseline_residuals = row_residuals[
        np.arange(len(row_residuals)), row_baseline_columns
    ]
    selected_residuals = row_residuals[
        np.arange(len(row_residuals)), selected_candidates
    ]
    true_rescue = (
        (baseline_residuals > float(baseline_invalid_threshold_px))
        & (selected_residuals <= float(candidate_threshold_px))
    )
    improved = selected_residuals < baseline_residuals
    worsened = selected_residuals > baseline_residuals
    finite_delta = switched & np.isfinite(baseline_residuals) & np.isfinite(
        selected_residuals
    )
    residual_delta = selected_residuals[finite_delta] - baseline_residuals[finite_delta]
    switch_count = int(np.sum(switched))
    return {
        "candidate": candidate_metrics,
        "keep": keep_metrics,
        "action_accuracy": float(np.mean(action_correct)),
        "action_margin_threshold": float(action_margin_threshold),
        "predicted_switch_rate": float(np.mean(switched)),
        "target_rescue_group_rate": float(np.mean(row_has_rescue)),
        "target_rescue_candidate_rate": float(np.mean(targets[candidate_samples])),
        "row_count": int(np.sum(rows)),
        "switch_residual_audit": {
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
                np.sum(
                    switched
                    & (baseline_residuals <= float(baseline_invalid_threshold_px))
                )
            ),
            "finite_residual_delta_count": int(len(residual_delta)),
            "median_selected_minus_baseline_residual_px": (
                None if len(residual_delta) == 0 else float(np.median(residual_delta))
            ),
            "mean_selected_minus_baseline_residual_px": (
                None if len(residual_delta) == 0 else float(np.mean(residual_delta))
            ),
            "median_action_margin_switched": (
                None
                if switch_count == 0
                else float(np.median(action_margins[switched]))
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    admission_log_bonuses = tuple(
        float(value)
        for value in args.global_assignment_candidate_admission_log_bonuses
    )
    if str(args.global_assignment_baseline_summary) and not bool(
        args.global_assignment_validation
    ):
        raise ValueError(
            "--global_assignment_baseline_summary requires --global_assignment_validation"
        )
    if bool(args.global_assignment_joint_posterior) and not bool(
        args.global_assignment_validation
    ):
        raise ValueError(
            "--global_assignment_joint_posterior requires --global_assignment_validation"
        )
    if any(not np.isfinite(value) for value in admission_log_bonuses):
        raise ValueError("candidate admission log bonuses must be finite")
    if len(set(admission_log_bonuses)) != len(admission_log_bonuses):
        raise ValueError("candidate admission log bonuses must be unique")
    if admission_log_bonuses != (0.0,) and not bool(
        args.global_assignment_joint_posterior
    ):
        raise ValueError(
            "candidate admission log bonuses require joint-posterior global assignment"
        )
    if (
        not np.isfinite(float(args.paired_catastrophic_translation_m))
        or float(args.paired_catastrophic_translation_m) <= 0.0
        or not np.isfinite(float(args.paired_max_translation_regression_m))
        or float(args.paired_max_translation_regression_m) < 0.0
    ):
        raise ValueError("paired pose safety thresholds are invalid")
    if min(
        int(args.epochs),
        int(args.batch_size),
        int(args.eval_batch_size),
        int(args.global_assignment_match_count),
    ) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if (
        float(args.geometry_validity_loss_weight) < 0.0
        or float(args.candidate_visibility_loss_weight) < 0.0
        or float(args.rescue_policy_loss_weight) < 0.0
    ):
        raise ValueError("geometry-aware loss weights must be non-negative")
    pose_hard_margin_weight = float(
        args.pose_conditioned_hard_negative_margin_weight
    )
    pose_hard_margin = float(args.pose_conditioned_hard_negative_margin)
    if (
        not np.isfinite(pose_hard_margin_weight)
        or pose_hard_margin_weight < 0.0
        or not np.isfinite(pose_hard_margin)
        or pose_hard_margin < 0.0
    ):
        raise ValueError(
            "pose-conditioned hard-negative weight and margin must be finite "
            "and non-negative"
        )
    pose_hard_artifact_enabled = bool(
        str(args.pose_conditioned_hard_negative_artifact)
    )
    if pose_hard_artifact_enabled != bool(pose_hard_margin_weight > 0.0):
        raise ValueError(
            "--pose_conditioned_hard_negative_artifact and a positive margin "
            "weight must be enabled together"
        )
    pose_hard_mode_weight = float(
        args.pose_conditioned_hard_mode_margin_weight
    )
    pose_hard_mode_margin = float(args.pose_conditioned_hard_mode_margin)
    pose_hard_mode_fraction = float(
        args.pose_conditioned_hard_mode_top_group_fraction
    )
    if (
        not np.isfinite(pose_hard_mode_weight)
        or pose_hard_mode_weight < 0.0
        or not np.isfinite(pose_hard_mode_margin)
        or pose_hard_mode_margin < 0.0
        or not np.isfinite(pose_hard_mode_fraction)
        or not 0.0 < pose_hard_mode_fraction <= 1.0
    ):
        raise ValueError(
            "pose-conditioned hard-mode weight/margin/fraction are invalid"
        )
    pose_hard_mode_artifact_enabled = bool(
        str(args.pose_conditioned_hard_mode_artifact)
    )
    if pose_hard_mode_artifact_enabled != bool(pose_hard_mode_weight > 0.0):
        raise ValueError(
            "--pose_conditioned_hard_mode_artifact and a positive mode margin "
            "weight must be enabled together"
        )
    if pose_hard_margin_weight > 0.0 and pose_hard_mode_weight > 0.0:
        raise ValueError(
            "candidate-level and structured pose-mode margins are separate ablations"
        )
    if pose_hard_mode_weight > 0.0 and not bool(
        args.query_grouped_training_batches
    ):
        raise ValueError(
            "structured pose-mode margins require --query_grouped_training_batches"
        )
    if not 0.0 <= float(args.support_view_dropout) < 1.0:
        raise ValueError("support_view_dropout must be in [0, 1)")
    geometry_thresholds = tuple(
        float(value) for value in args.geometry_validity_thresholds_px
    )
    if len(geometry_thresholds) != 3 or not np.isclose(
        geometry_thresholds[1], float(args.positive_threshold_px)
    ):
        raise ValueError(
            "geometry validity needs three thresholds whose middle value equals positive_threshold_px"
        )
    geometry_validity_enabled = bool(
        float(args.geometry_validity_loss_weight) > 0.0
        or float(args.candidate_visibility_loss_weight) > 0.0
    )
    if bool(args.candidate_view_marginalization) and not bool(
        args.decoupled_candidate_heads
    ):
        raise ValueError(
            "--candidate_view_marginalization requires --decoupled_candidate_heads"
        )
    if bool(args.full_candidate_view_mixture) and not bool(
        args.candidate_view_marginalization
    ):
        raise ValueError(
            "--full_candidate_view_mixture requires --candidate_view_marginalization"
        )
    if bool(args.identity_conditioned_view_posterior) and not bool(
        args.candidate_view_marginalization
    ):
        raise ValueError(
            "--identity_conditioned_view_posterior requires --candidate_view_marginalization"
        )
    if bool(args.full_candidate_view_mixture) and not bool(
        args.identity_conditioned_view_posterior
    ):
        raise ValueError(
            "--full_candidate_view_mixture requires --identity_conditioned_view_posterior"
        )
    prior_free_loss_weights = (
        float(args.prior_free_identity_loss_weight),
        float(args.prior_free_conditional_identity_loss_weight),
        pose_hard_margin_weight,
        pose_hard_mode_weight,
    )
    if min(prior_free_loss_weights) < 0.0:
        raise ValueError("prior-free identity loss weights must be non-negative")
    if bool(args.prior_free_set_identity) != bool(
        max(prior_free_loss_weights) > 0.0
    ):
        raise ValueError(
            "--prior_free_set_identity and at least one positive prior-free "
            "identity loss weight must be enabled together"
        )
    if bool(args.deployable_identity_context) and not bool(
        args.prior_free_set_identity
    ):
        raise ValueError(
            "--deployable_identity_context requires --prior_free_set_identity"
        )
    if float(args.factorized_top_l_availability_loss_weight) < 0.0:
        raise ValueError("factorized top-L availability loss weight must be non-negative")
    if bool(args.factorized_set_posterior) != bool(
        float(args.factorized_top_l_availability_loss_weight) > 0.0
    ):
        raise ValueError(
            "--factorized_set_posterior and a positive factorized top-L availability "
            "loss weight must be enabled together"
        )
    if bool(args.factorized_set_posterior) and (
        not bool(args.prior_free_set_identity)
        or float(args.prior_free_conditional_identity_loss_weight) <= 0.0
    ):
        raise ValueError(
            "factorized set posterior requires prior-free conditional identity "
            "training for P(track | non-null)"
        )
    rescue_policy_enabled = bool(float(args.rescue_policy_loss_weight) > 0.0)
    hard_oversample_factor = float(args.system_hard_group_oversample_factor)
    if not np.isfinite(hard_oversample_factor) or hard_oversample_factor < 1.0:
        raise ValueError("system hard-group oversample factor must be at least one")
    hard_conditional_weight = float(
        args.system_hard_conditional_identity_weight
    )
    if not np.isfinite(hard_conditional_weight) or hard_conditional_weight < 1.0:
        raise ValueError(
            "system hard conditional-identity weight must be at least one"
        )
    if hard_oversample_factor > 1.0 and hard_conditional_weight > 1.0:
        raise ValueError(
            "hard-group oversampling and conditional identity weighting cannot "
            "be enabled together"
        )
    if max(pose_hard_margin_weight, pose_hard_mode_weight) > 0.0 and (
        hard_oversample_factor > 1.0 or hard_conditional_weight > 1.0
    ):
        raise ValueError(
            "pose-conditioned margins cannot be combined with legacy "
            "hard-group oversampling or whole-group identity upweighting"
        )
    if hard_conditional_weight > 1.0 and (
        not bool(args.prior_free_set_identity)
        or float(args.prior_free_conditional_identity_loss_weight) <= 0.0
    ):
        raise ValueError(
            "hard conditional identity weighting requires the prior-free "
            "conditional identity loss"
        )
    if hard_oversample_factor > 1.0 and (
        geometry_validity_enabled or rescue_policy_enabled
    ):
        raise ValueError(
            "system hard-group oversampling is identity-only; geometry, visibility, "
            "and rescue losses must be disabled to preserve their calibration"
        )
    if bool(args.factorized_set_posterior) and hard_oversample_factor > 1.0:
        raise ValueError(
            "factorized top-L availability calibration requires the natural group "
            "distribution; hard-group oversampling must be disabled"
        )
    if (
        not np.isfinite(float(args.rescue_candidate_threshold_px))
        or not np.isfinite(float(args.rescue_baseline_invalid_threshold_px))
        or float(args.rescue_candidate_threshold_px) <= 0.0
        or float(args.rescue_baseline_invalid_threshold_px) <= 0.0
        or float(args.rescue_candidate_threshold_px)
        > float(args.rescue_baseline_invalid_threshold_px)
    ):
        raise ValueError("invalid rescue policy residual thresholds")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = CandidateMapletEpisodeStore(
        proposals=Path(args.proposals),
        detector_query_cache=Path(args.detector_query_cache),
        query_context_detector_cache=Path(args.query_context_detector_cache),
        support_feature_cache=Path(args.support_feature_cache),
        support_geometry_index=Path(args.support_geometry_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        maplet_support_index=Path(args.maplet_support_index),
        feature_artifact=Path(args.feature_artifact),
        colmap_model_dir=Path(args.colmap_model_dir),
        radio_intermediate_cache=(
            None
            if args.radio_intermediate_cache is None
            else Path(args.radio_intermediate_cache)
        ),
        query_radius_px=float(args.query_radius_px),
        max_query_nodes=int(args.max_query_nodes),
        max_support_tracks=int(args.max_support_tracks),
        positive_threshold_px=float(args.positive_threshold_px),
        assignment_threshold_px=float(args.assignment_threshold_px),
        support_view_count=int(args.support_view_count),
        static_feature_count=int(args.static_feature_count),
        query_cache_size=int(args.query_cache_size),
        support_cache_size=int(args.support_cache_size),
        episode_cache_size=int(args.episode_cache_size),
    )
    split = _split_query_ids(store, args)
    data_manifest = _candidate_data_manifest(args, store)
    frozen_global_baseline_source = (
        None
        if not str(args.global_assignment_baseline_summary)
        else _load_frozen_global_baseline_policy(
            Path(args.global_assignment_baseline_summary),
            args=args,
            data_manifest=data_manifest,
        )
    )
    (output_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    split_edges = {
        name: store.split_edge_indices(values)
        for name, values in (("train", split["train"]), ("validation", split["validation"]), ("test", split["test"]))
    }
    selected_query_ids = store.query_ids[store.selected_rows]
    split_row_masks = {
        name: np.isin(selected_query_ids, np.asarray(split[name], dtype=np.str_))
        for name in ("train", "validation", "test")
    }
    pose_conditioned_hard_negative_mask = np.zeros_like(
        store.valid_edges, dtype=bool
    )
    pose_conditioned_hard_negative_mode_counts = np.zeros_like(
        store.valid_edges, dtype=np.int64
    )
    pose_conditioned_hard_negative_audit: dict[str, object] = {
        "mode": "disabled",
        "hard_group_count": 0,
        "hard_candidate_count": 0,
        "validation_or_test_training_leakage": False,
    }
    if pose_hard_artifact_enabled:
        (
            pose_conditioned_hard_negative_mask,
            pose_conditioned_hard_negative_mode_counts,
            pose_conditioned_hard_negative_audit,
        ) = _load_pose_conditioned_hard_negative_artifact(
            Path(args.pose_conditioned_hard_negative_artifact),
            expected_manifest=data_manifest,
            expected_selected_rows=store.selected_rows,
            expected_selected_columns=store.selected_columns,
            expected_query_ids=selected_query_ids,
            expected_valid_edges=store.valid_edges,
            expected_positive_mask=store.labels,
            split=split,
        )
    pose_conditioned_hard_mode_ids = np.full(
        (len(store.selected_rows), 0), -1, dtype=np.int64
    )
    pose_conditioned_hard_mode_candidate_mask = np.zeros(
        (len(store.selected_rows), 0, store.candidate_top_k), dtype=bool
    )
    pose_conditioned_hard_mode_audit: dict[str, object] = {
        "mode": "disabled",
        "hard_mode_count": 0,
        "hard_mode_group_incidence_count": 0,
        "validation_or_test_training_leakage": False,
    }
    pose_hard_mode_minimum_groups = 1
    if pose_hard_mode_artifact_enabled:
        (
            pose_conditioned_hard_mode_ids,
            pose_conditioned_hard_mode_candidate_mask,
            pose_conditioned_hard_mode_audit,
        ) = _load_pose_conditioned_hard_mode_artifact(
            Path(args.pose_conditioned_hard_mode_artifact),
            expected_manifest=data_manifest,
            expected_selected_rows=store.selected_rows,
            expected_selected_columns=store.selected_columns,
            expected_query_ids=selected_query_ids,
            expected_valid_edges=store.valid_edges,
            expected_positive_mask=store.labels,
            split=split,
        )
        pose_hard_mode_minimum_groups = int(
            pose_conditioned_hard_mode_audit["minimum_mode_groups"]
        )
    prototype_ids = _compact_values(store.proposals["candidate_prototype_ids"], store).astype(np.int64)
    coarse_scores = _compact_values(store.proposals["coarse_scores"], store)
    candidates = UniqueTrackCandidateSet(
        store.compact_canonical_rows,
        store.compact_candidate_tracks,
        prototype_ids,
        coarse_scores,
    )
    baseline_scores = _compact_values(
        store.proposals[f"strategy__{args.baseline_strategy}"], store
    )
    system_hard_scores = baseline_scores
    system_hard_score_source: dict[str, object] = {
        "mode": "proposal_baseline_strategy",
        "baseline_strategy": str(args.baseline_strategy),
        "score_key": f"strategy__{args.baseline_strategy}",
    }
    if str(args.system_hard_score_artifact):
        system_hard_scores, system_hard_score_source = (
            _load_system_hard_score_artifact(
                Path(args.system_hard_score_artifact),
                score_key=str(args.system_hard_score_key),
                expected_manifest=data_manifest,
                expected_shape=(
                    len(store.selected_rows),
                    int(store.candidate_top_k),
                ),
            )
        )
    candidate_residuals = _compact_values(store.proposals["candidate_gt_residuals_px"], store)
    nearest_residuals = np.asarray(store.proposals["nearest_visible_residuals_px"], dtype=np.float32)[
        store.selected_rows
    ]
    query_observations = [
        ColmapTrackObservation(
            track_id=int(store.proposals["nearest_visible_track_ids"][global_row]),
            image_id=str(store.query_ids[global_row]),
            point2d_idx=int(global_row),
            xy=(float(store.query_xy[global_row, 0]), float(store.query_xy[global_row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for global_row in store.selected_rows.tolist()
    ]
    cameras = read_colmap_cameras_binary(Path(args.colmap_model_dir) / "cameras.bin")
    images = read_colmap_images_binary(Path(args.colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    def identity(scores: np.ndarray, split_name: str) -> dict[str, object]:
        mask = split_row_masks[split_name]
        report = _identity_metrics(
            nearest_residuals=nearest_residuals[mask],
            candidate_residuals=candidate_residuals[mask],
            scores=scores[mask],
            query_ids=selected_query_ids[mask],
            labels=store.labels[mask],
            valid_edges=store.valid_edges[mask],
        )
        report["transition_vs_frozen_baseline"] = _identity_transition_report(
            store.labels[mask],
            store.valid_edges[mask],
            baseline_scores[mask],
            scores[mask],
        )
        if str(args.system_hard_score_artifact):
            report["transition_vs_system_hard_reference"] = (
                _identity_transition_report(
                    store.labels[mask],
                    store.valid_edges[mask],
                    system_hard_scores[mask],
                    scores[mask],
                )
            )
        return report

    def pose(
        scores: np.ndarray,
        split_name: str,
        strategy: str,
        *,
        max_matches: int | None = None,
        selection_mode: str = "score_topk",
    ):
        rows = np.flatnonzero(split_row_masks[split_name])
        subset = UniqueTrackCandidateSet(
            candidates.bank_row_indices[rows],
            candidates.track_ids[rows],
            candidates.prototype_ids[rows],
            candidates.coarse_scores[rows],
        )
        summary, pose_rows = _evaluate_pose_strategy(
            strategy=str(strategy),
            scores=scores[rows],
            candidates=subset,
            query_observations=[query_observations[int(row)] for row in rows.tolist()],
            query_ids=selected_query_ids[rows].tolist(),
            landmark_index=store.landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
            max_matches=max_matches,
            pose_selection_mode=str(selection_mode),
        )
        return summary, pose_rows

    baseline_validation_pose, _rows = pose(baseline_scores, "validation", "baseline_validation")
    baseline_validation_identity = identity(baseline_scores, "validation")
    baseline_global_scores = None
    baseline_global_validation_pose = None
    baseline_global_validation_rows = None
    if bool(args.global_assignment_validation):
        baseline_global_scores, _baseline_global_columns = global_assignment_score_matrix(
            candidates.track_ids,
            baseline_scores,
            selected_query_ids,
            valid_mask=store.valid_edges,
            dustbin_score=None,
        )
        baseline_global_validation_pose, baseline_global_validation_rows = pose(
            baseline_global_scores,
            "validation",
            "baseline_global_partial_assignment_validation",
            max_matches=int(args.global_assignment_match_count),
            selection_mode=str(args.global_assignment_selection_mode),
        )
        if frozen_global_baseline_source is not None:
            _validate_frozen_baseline_pose(
                baseline_global_validation_pose,
                frozen_global_baseline_source,
            )
    try:
        candidate_prior_index = store.static_feature_names.index(
            str(args.candidate_prior_feature)
        )
    except ValueError as error:
        raise ValueError(
            f"candidate prior feature is unavailable: {args.candidate_prior_feature!r}"
        ) from error
    deployable_identity_static_start_index = 0
    if bool(args.deployable_identity_context):
        try:
            deployable_identity_static_start_index = (
                store.static_feature_names.index("query_detector_score")
            )
        except ValueError as error:
            raise ValueError(
                "deployable identity context requires query_detector_score"
            ) from error
        excluded = tuple(
            store.static_feature_names[:deployable_identity_static_start_index]
        )
        expected_excluded = (
            "coarse_score",
            "baseline_score",
            "coarse_rank_fraction",
            "baseline_rank_fraction",
            "coarse_gap_to_row_best",
            "baseline_gap_to_row_best",
        )
        if excluded != expected_excluded:
            raise ValueError(
                "deployable identity context prior/rank exclusion schema differs"
            )
    if bool(args.standardize_static_features):
        static_feature_mean, static_feature_scale = _fit_static_feature_normalization(
            store.static_features,
            row_mask=split_row_masks["train"],
            valid_edges=store.valid_edges,
        )
    else:
        static_feature_mean = None
        static_feature_scale = None
    config = CandidateMapletMatcherConfig(
        query_input_dim=store.query_input_dim,
        support_input_dim=store.support_input_dim,
        static_input_dim=store.static_input_dim,
        descriptor_dim=64,
        model_dim=int(args.model_dim),
        num_heads=int(args.num_heads),
        layers=int(args.layers),
        dropout=float(args.dropout),
        sinkhorn_iterations=int(args.sinkhorn_iterations),
        candidate_set_layers=int(args.candidate_set_layers),
        candidate_prior_index=int(candidate_prior_index),
        candidate_prior_scale=float(args.candidate_prior_scale),
        static_feature_mean=static_feature_mean,
        static_feature_scale=static_feature_scale,
        geometry_validity_enabled=geometry_validity_enabled,
        decoupled_candidate_heads=bool(args.decoupled_candidate_heads),
        candidate_view_marginalization_enabled=bool(
            args.candidate_view_marginalization
        ),
        identity_conditioned_view_posterior_enabled=bool(
            args.identity_conditioned_view_posterior
        ),
        full_candidate_view_mixture_enabled=bool(
            args.full_candidate_view_mixture
        ),
        explicit_anchor_role_embedding=bool(
            args.explicit_anchor_role_embedding
        ),
        prior_free_set_identity_enabled=bool(args.prior_free_set_identity),
        deployable_identity_context_enabled=bool(
            args.deployable_identity_context
        ),
        deployable_identity_static_start_index=int(
            deployable_identity_static_start_index
        ),
        factorized_set_posterior_enabled=bool(args.factorized_set_posterior),
        geometry_validity_thresholds_px=geometry_thresholds,
        rescue_policy_enabled=rescue_policy_enabled,
        rescue_candidate_threshold_px=float(args.rescue_candidate_threshold_px),
        rescue_baseline_invalid_threshold_px=float(
            args.rescue_baseline_invalid_threshold_px
        ),
    )
    device = torch.device(str(args.device))
    model = CandidateMapletMatcher(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(int(args.seed))
    history = []
    best_key = None
    best_epoch = -1
    best_selection = None
    checkpoint_path = output_dir / "best.pt"
    best_rescue_key = None
    best_rescue_epoch = -1
    best_rescue_selection = None
    best_rescue_checkpoint_path = output_dir / "best_rescue.pt"
    best_global_key = None
    best_global_epoch = -1
    best_global_selection = None
    best_global_checkpoint_path = output_dir / "best_global.pt"
    last_checkpoint_path = output_dir / "last.pt"
    checkpoint_format = (
        "candidate_maplet_matcher_checkpoint_v11"
        if bool(args.deployable_identity_context)
        else "candidate_maplet_matcher_checkpoint_v10"
        if bool(args.factorized_set_posterior)
        else "candidate_maplet_matcher_checkpoint_v9"
        if bool(args.prior_free_set_identity)
        else "candidate_maplet_matcher_checkpoint_v8"
        if bool(args.full_candidate_view_mixture)
        or bool(args.explicit_anchor_role_embedding)
        else "candidate_maplet_matcher_checkpoint_v7"
        if rescue_policy_enabled
        else (
            "candidate_maplet_matcher_checkpoint_v6"
            if geometry_validity_enabled
            else "candidate_maplet_matcher_checkpoint_v5"
        )
    )
    validation_strategy_names = _validation_strategy_names(
        str(args.validation_strategies),
        rescue_policy_enabled=rescue_policy_enabled,
    )
    refit_metadata = None
    if args.refit_selection_checkpoint is not None:
        if bool(args.global_assignment_validation):
            raise ValueError(
                "global assignment checkpoint selection is unavailable during fixed refit"
            )
        best_selection, refit_metadata = _load_refit_selection(
            Path(args.refit_selection_checkpoint),
            data_manifest=data_manifest,
            split=split,
            model_config=config.to_dict(),
            epochs=int(args.epochs),
        )
        training_edges = np.concatenate(
            [split_edges["train"], split_edges["validation"]]
        ).astype(np.int64)
    else:
        training_edges = split_edges["train"]
    available_training_groups = np.unique(training_edges // store.candidate_top_k)
    available_group_has_positive = np.any(store.labels[available_training_groups], axis=1)
    available_positive_group_count = int(np.sum(available_group_has_positive))
    available_no_match_group_count = int(np.sum(~available_group_has_positive))
    system_hard_group_mask, system_hard_group_audit = (
        _system_hard_candidate_group_mask(
            store.labels,
            system_hard_scores,
            store.valid_edges,
            available_training_groups,
            ambiguous_margin=float(args.system_hard_ambiguous_margin),
            no_match_quantile=float(args.system_hard_no_match_quantile),
        )
    )
    required_no_match_ratio = float(
        available_no_match_group_count / max(available_positive_group_count, 1)
    )
    full_group_distribution_training = bool(
        int(args.max_train_groups) <= 0
        and float(args.no_match_group_ratio) + 1e-12 >= required_no_match_ratio
    )
    if (
        bool(args.factorized_set_posterior)
        and not full_group_distribution_training
        and int(args.smoke_validation_query_count) <= 0
    ):
        raise ValueError(
            "factorized top-L availability training requires every natural train group; "
            f"set no_match_group_ratio >= {required_no_match_ratio:.6g} and "
            "max_train_groups=0"
        )
    if (
        (geometry_validity_enabled or rescue_policy_enabled)
        and not full_group_distribution_training
        and int(args.smoke_validation_query_count) <= 0
    ):
        raise ValueError(
            "calibrated geometry validity training requires the full train-group distribution; "
            f"set no_match_group_ratio >= {required_no_match_ratio:.6g} and max_train_groups=0"
        )

    def save_training_checkpoint(
        path: Path,
        *,
        epoch: int,
        selection: dict[str, object] | None,
        checkpoint_role: str,
    ) -> None:
        payload: dict[str, object] = {
            "format": checkpoint_format,
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "data_manifest": data_manifest,
            "split": split,
            "epoch": int(epoch),
            "selection": selection,
            "seed": int(args.seed),
            "checkpoint_role": str(checkpoint_role),
            "training_curriculum": {
                "system_hard_score_source": system_hard_score_source,
                "system_hard_oversample_factor": float(hard_oversample_factor),
                "system_hard_conditional_identity_weight": float(
                    hard_conditional_weight
                ),
                "pose_conditioned_hard_negative_source": (
                    pose_conditioned_hard_negative_audit
                ),
                "pose_conditioned_hard_negative_margin_weight": float(
                    pose_hard_margin_weight
                ),
                "pose_conditioned_hard_negative_margin": float(
                    pose_hard_margin
                ),
                "pose_conditioned_hard_mode_source": (
                    pose_conditioned_hard_mode_audit
                ),
                "pose_conditioned_hard_mode_margin_weight": float(
                    pose_hard_mode_weight
                ),
                "pose_conditioned_hard_mode_margin": float(
                    pose_hard_mode_margin
                ),
                "pose_conditioned_hard_mode_top_group_fraction": float(
                    pose_hard_mode_fraction
                ),
                "pose_conditioned_hard_mode_minimum_groups": int(
                    pose_hard_mode_minimum_groups
                ),
                "query_grouped_training_batches": bool(
                    args.query_grouped_training_batches
                ),
            },
        }
        if refit_metadata is not None:
            payload["training_query_ids"] = [
                *list(split["train"]),
                *list(split["validation"]),
            ]
            payload["refit_selection_source"] = refit_metadata
        torch.save(payload, path)

    for epoch in range(int(args.epochs)):
        epoch_start = time.time()
        model.train()
        train_groups = _balanced_train_groups(
            store,
            training_edges,
            no_match_ratio=float(args.no_match_group_ratio),
            max_groups=int(args.max_train_groups),
            rng=rng,
            system_hard_group_mask=system_hard_group_mask,
            system_hard_oversample_factor=hard_oversample_factor,
        )
        metric_sums: dict[str, float] = {}
        batch_count = 0
        data_build_seconds = 0.0
        gpu_compute_seconds = 0.0
        groups_per_batch = max(1, int(args.batch_size) // store.candidate_top_k)
        if bool(args.query_grouped_training_batches):
            training_group_batches = _query_grouped_training_batches(
                train_groups,
                selected_query_ids,
                groups_per_batch=groups_per_batch,
                rng=rng,
            )
        else:
            training_group_batches = [
                train_groups[batch_start : batch_start + groups_per_batch]
                for batch_start in range(0, len(train_groups), groups_per_batch)
            ]
        hard_mode_batch_audit = {
            "hard_mode_count": 0,
            "hard_mode_group_incidence_count": 0,
        }
        if pose_hard_mode_weight > 0.0:
            hard_mode_batch_audit = _validate_complete_hard_modes_in_batches(
                training_group_batches,
                pose_conditioned_hard_mode_ids,
            )
        training_edge_batches = [
            (
                group_batch[:, None]
                * store.candidate_top_k
                + np.arange(store.candidate_top_k, dtype=np.int64)[None]
            ).reshape(-1)
            for group_batch in training_group_batches
        ]

        def build_training_batch(
            edges: np.ndarray,
        ) -> tuple[
            list[CandidateMapletBatch],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
        ]:
            data_start = time.perf_counter()
            cpu_batches = []
            for view_rank in range(store.support_view_count):
                ranks = np.full(edges.shape, view_rank, dtype=np.int64)
                cpu_batches.append(store.batch(edges, view_ranks=ranks))
            batch_groups = edges.reshape(-1, store.candidate_top_k)[:, 0]
            batch_groups = batch_groups // store.candidate_top_k
            conditional_weights = np.where(
                system_hard_group_mask[batch_groups],
                hard_conditional_weight,
                1.0,
            ).astype(np.float32)
            return (
                cpu_batches,
                conditional_weights,
                pose_conditioned_hard_negative_mask[batch_groups],
                pose_conditioned_hard_negative_mode_counts[batch_groups],
                pose_conditioned_hard_mode_ids[batch_groups],
                pose_conditioned_hard_mode_candidate_mask[batch_groups],
                store.valid_edges[batch_groups],
                float(time.perf_counter() - data_start),
            )

        for (
            cpu_batches,
            conditional_weights,
            pose_hard_mask,
            pose_hard_mode_counts,
            pose_hard_structured_ids,
            pose_hard_structured_candidate_mask,
            batch_candidate_mask,
            build_seconds,
        ) in _ordered_prefetch(
            training_edge_batches,
            build_training_batch,
            enabled=bool(args.training_batch_prefetch),
        ):
            optimizer.zero_grad(set_to_none=True)
            transfer_start = time.perf_counter()
            batches = [batch.to(device) for batch in cpu_batches]
            data_build_seconds += float(build_seconds) + (
                time.perf_counter() - transfer_start
            )
            if bool(args.profile_training_timing) and device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = [
                    model(batch, return_ragged_query_probabilities=False)
                    for batch in batches
                ]
                loss, metrics = candidate_maplet_group_loss(
                    model,
                    outputs,
                    batches,
                    candidate_group_size=store.candidate_top_k,
                    assignment_weight=float(args.assignment_loss_weight),
                    pair_weight=float(args.pair_loss_weight),
                    candidate_aux_weight=float(args.candidate_aux_loss_weight),
                    candidate_set_weight=float(args.candidate_set_loss_weight),
                    matched_query_weight=float(args.matched_query_weight),
                    candidate_pos_weight=float(args.candidate_pos_weight),
                    geometry_validity_weight=float(args.geometry_validity_loss_weight),
                    candidate_visibility_weight=float(
                        args.candidate_visibility_loss_weight
                    ),
                    rescue_policy_weight=float(args.rescue_policy_loss_weight),
                    prior_free_identity_weight=float(
                        args.prior_free_identity_loss_weight
                    ),
                    prior_free_conditional_identity_weight=float(
                        args.prior_free_conditional_identity_loss_weight
                    ),
                    factorized_top_l_availability_weight=float(
                        args.factorized_top_l_availability_loss_weight
                    ),
                    pose_conditioned_hard_negative_weight=float(
                        pose_hard_margin_weight
                    ),
                    pose_conditioned_hard_negative_margin=float(
                        pose_hard_margin
                    ),
                    pose_conditioned_hard_negative_mask=torch.as_tensor(
                        pose_hard_mask,
                        dtype=torch.bool,
                        device=device,
                    ),
                    pose_conditioned_hard_negative_mode_counts=torch.as_tensor(
                        pose_hard_mode_counts,
                        dtype=torch.float32,
                        device=device,
                    ),
                    pose_conditioned_hard_mode_weight=float(
                        pose_hard_mode_weight
                    ),
                    pose_conditioned_hard_mode_margin=float(
                        pose_hard_mode_margin
                    ),
                    pose_conditioned_hard_mode_top_group_fraction=float(
                        pose_hard_mode_fraction
                    ),
                    pose_conditioned_hard_mode_minimum_groups=int(
                        pose_hard_mode_minimum_groups
                    ),
                    pose_conditioned_hard_mode_ids=torch.as_tensor(
                        pose_hard_structured_ids,
                        dtype=torch.long,
                        device=device,
                    ),
                    pose_conditioned_hard_mode_candidate_mask=torch.as_tensor(
                        pose_hard_structured_candidate_mask,
                        dtype=torch.bool,
                        device=device,
                    ),
                    pose_conditioned_candidate_mask=torch.as_tensor(
                        batch_candidate_mask,
                        dtype=torch.bool,
                        device=device,
                    ),
                    support_view_dropout=float(args.support_view_dropout),
                    conditional_identity_group_weights=torch.as_tensor(
                        conditional_weights,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            if bool(args.profile_training_timing) and device.type == "cuda":
                torch.cuda.synchronize(device)
            gpu_compute_seconds += time.perf_counter() - compute_start
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
            batch_count += 1
        train_metrics = {
            name: value / max(batch_count, 1) for name, value in metric_sums.items()
        }
        train_metrics["data_build_seconds"] = float(data_build_seconds)
        train_metrics["gpu_compute_seconds"] = float(gpu_compute_seconds)
        train_metrics["system_hard_group_draw_count"] = int(
            np.sum(system_hard_group_mask[train_groups])
        )
        train_metrics["system_hard_unique_group_count"] = int(
            len(np.unique(train_groups[system_hard_group_mask[train_groups]]))
        )
        train_metrics["system_hard_conditional_identity_weight"] = float(
            hard_conditional_weight
        )
        pose_hard_draw_mask = pose_conditioned_hard_negative_mask[train_groups]
        train_metrics["pose_conditioned_hard_group_draw_count"] = int(
            np.sum(np.any(pose_hard_draw_mask, axis=1))
        )
        train_metrics["pose_conditioned_hard_candidate_draw_count"] = int(
            np.sum(pose_hard_draw_mask)
        )
        train_metrics["pose_conditioned_hard_unique_group_count"] = int(
            len(
                np.unique(
                    train_groups[np.any(pose_hard_draw_mask, axis=1)]
                )
            )
        )
        pose_hard_mode_draw_ids = pose_conditioned_hard_mode_ids[train_groups]
        pose_hard_mode_draw_present = pose_hard_mode_draw_ids >= 0
        pose_hard_mode_draw_candidates = (
            pose_conditioned_hard_mode_candidate_mask[train_groups]
        )
        train_metrics["pose_conditioned_hard_mode_draw_count"] = int(
            len(np.unique(pose_hard_mode_draw_ids[pose_hard_mode_draw_present]))
        )
        train_metrics[
            "pose_conditioned_hard_mode_group_incidence_draw_count"
        ] = int(np.sum(pose_hard_mode_draw_present))
        train_metrics[
            "pose_conditioned_hard_mode_candidate_membership_draw_count"
        ] = int(np.sum(pose_hard_mode_draw_candidates))
        train_metrics["query_grouped_training_batch_count"] = int(
            len(training_group_batches)
        )
        train_metrics["maximum_training_groups_per_batch"] = int(
            max((len(batch) for batch in training_group_batches), default=0)
        )
        train_metrics.update(
            {
                f"validated_batch_{key}": int(value)
                for key, value in hard_mode_batch_audit.items()
            }
        )

        if refit_metadata is not None:
            epoch_summary = {
                "epoch": int(epoch),
                "train_group_count": int(len(train_groups)),
                "train_edge_count": int(len(train_groups) * store.candidate_top_k),
                "train_metrics": train_metrics,
                "selection_source": "fixed_validation_checkpoint",
                "epoch_seconds": float(time.time() - epoch_start),
            }
            history.append(epoch_summary)
            print(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "train": train_metrics,
                        "refit": True,
                        "epoch_seconds": epoch_summary["epoch_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            save_training_checkpoint(
                last_checkpoint_path,
                epoch=int(epoch),
                selection=best_selection,
                checkpoint_role="last",
            )
            if epoch == int(args.epochs) - 1:
                best_epoch = int(epoch)
                save_training_checkpoint(
                    checkpoint_path,
                    epoch=int(epoch),
                    selection=best_selection,
                    checkpoint_role="development_refit_final",
                )
            continue

        validation_predictions = _predict_edges(
            model,
            store,
            split_edges["validation"],
            device=device,
            batch_size=int(args.eval_batch_size),
            use_amp=use_amp,
        )
        validation_top_l_availability = _factorized_top_l_availability_report(
            validation_predictions,
            store,
            row_mask=split_row_masks["validation"],
        )
        strategy_trials = []
        validation_geometry_probability = _geometry_probability_report(
            validation_predictions,
            store,
            row_mask=split_row_masks["validation"],
            thresholds_px=geometry_thresholds,
            visibility_supervised=bool(
                float(args.candidate_visibility_loss_weight) > 0.0
            ),
        )
        validation_rescue_policy = _rescue_policy_report(
            validation_predictions,
            store,
            baseline_scores,
            row_mask=split_row_masks["validation"],
            candidate_threshold_px=float(args.rescue_candidate_threshold_px),
            baseline_invalid_threshold_px=float(
                args.rescue_baseline_invalid_threshold_px
            ),
        )
        missing_strategies = set(validation_strategy_names) - set(validation_predictions)
        if missing_strategies:
            raise ValueError(
                f"unknown validation strategies: {sorted(missing_strategies)}"
            )
        for strategy_name in validation_strategy_names:
            flat_scores = validation_predictions[strategy_name]
            scores = flat_scores.reshape(len(store.selected_rows), store.candidate_top_k)
            unconditional_identity = identity(scores, "validation")
            unconditional_pose, _pose_rows = pose(
                scores, "validation", f"epoch{epoch}_{strategy_name}_unconditional"
            )
            unconditional_pose_passed = _pose_gate(
                unconditional_pose, baseline_validation_pose
            )
            unconditional_identity_passed = _assignment_identity_gate(
                unconditional_identity, baseline_validation_identity
            )
            unconditional_choices = np.argmax(scores, axis=1)
            baseline_choices = np.argmax(baseline_scores, axis=1)
            unconditional_switch_count = int(
                np.sum(
                    (unconditional_choices != baseline_choices)
                    & split_row_masks["validation"]
                )
            )
            unconditional_trial = {
                "mode": "unconditional",
                "margin_threshold": None,
                "switch_count": unconditional_switch_count,
                "identity": unconditional_identity,
                "pose": unconditional_pose,
                "passes_pose_gate": bool(unconditional_pose_passed),
                "passes_identity_gate": bool(unconditional_identity_passed),
                "passes_stage_gate": bool(
                    unconditional_pose_passed and unconditional_identity_passed
                ),
            }
            threshold_trials = []
            strategy_switch_thresholds = (
                ()
                if str(strategy_name) == "rescue_policy_resolved"
                else tuple(args.switch_margin_thresholds)
            )
            for threshold in strategy_switch_thresholds:
                _selected, resolved, switched, _margins = selective_switch_scores(
                    scores,
                    baseline_scores,
                    margin_threshold=float(threshold),
                    valid_mask=store.valid_edges,
                    preserve_baseline_row_confidence=str(strategy_name).endswith(
                        "_prior_row_confidence"
                    ),
                )
                selected_pose, _selected_rows = pose(
                    resolved,
                    "validation",
                    f"epoch{epoch}_{strategy_name}_margin{float(threshold):g}",
                )
                selected_identity = identity(resolved, "validation")
                selected_pose_passed = _pose_gate(
                    selected_pose, baseline_validation_pose
                )
                selected_identity_passed = _assignment_identity_gate(
                    selected_identity, baseline_validation_identity
                )
                threshold_trials.append(
                    {
                        "mode": "selective",
                        "margin_threshold": float(threshold),
                        "switch_count": int(np.sum(switched[split_row_masks["validation"]])),
                        "identity": selected_identity,
                        "pose": selected_pose,
                        "passes_pose_gate": bool(selected_pose_passed),
                        "passes_identity_gate": bool(selected_identity_passed),
                        "passes_stage_gate": bool(
                            selected_pose_passed and selected_identity_passed
                        ),
                    }
                )
            selection_trials = [unconditional_trial, *threshold_trials]
            eligible = [trial for trial in selection_trials if bool(trial["passes_stage_gate"])]
            if eligible:
                selected_trial = max(
                    eligible,
                    key=lambda trial: (
                        float(trial["pose"]["recall_10cm_5deg"]),
                        float(trial["pose"]["recall_5cm_5deg"]),
                        -float(trial["pose"]["median_translation_m_success"]),
                        -float(trial["pose"]["p90_translation_m_success"]),
                        -float(trial["pose"]["median_rotation_deg_success"]),
                        -float(
                            len(store.selected_rows)
                            if trial["switch_count"] is None
                            else trial["switch_count"]
                        ),
                    ),
                )
                gate_passed = True
            else:
                selected_trial = {
                    "mode": "fallback",
                    "margin_threshold": None,
                    "switch_count": 0,
                    "identity": baseline_validation_identity,
                    "pose": baseline_validation_pose,
                    "passes_pose_gate": False,
                    "passes_identity_gate": False,
                    "passes_stage_gate": False,
                }
                gate_passed = False
            r1 = float(
                unconditional_identity["geometry"]["thresholds_px"]["2"]["recall_at_1_given_mappable"]
            )
            strategy_trials.append(
                {
                    "strategy": strategy_name,
                    "unconditional": unconditional_trial,
                    "unconditional_identity": unconditional_identity,
                    "unconditional_pose": unconditional_pose,
                    "threshold_trials": threshold_trials,
                    "selected": selected_trial,
                    "selection_key": [
                        float(gate_passed),
                        float(selected_trial["pose"]["recall_10cm_5deg"]),
                        float(selected_trial["pose"]["recall_5cm_5deg"]),
                        -float(selected_trial["pose"]["median_translation_m_success"]),
                        -float(selected_trial["pose"]["p90_translation_m_success"]),
                        -float(selected_trial["pose"]["median_rotation_deg_success"]),
                        r1,
                        float(unconditional_identity["pair_positive_average_precision"]),
                    ],
                }
            )
        global_strategy_trials: list[dict[str, object]] = []
        if bool(args.global_assignment_validation):
            if baseline_global_validation_pose is None:
                raise RuntimeError("global assignment baseline was not initialized")
            for strategy_name in validation_strategy_names:
                raw_scores = validation_predictions[strategy_name].reshape(
                    len(store.selected_rows), store.candidate_top_k
                )
                if bool(args.global_assignment_joint_posterior):
                    scores, assignment_dustbins, assignment_score_space = (
                        _global_assignment_strategy_scores(
                            validation_predictions,
                            str(strategy_name),
                            group_count=len(store.selected_rows),
                            candidate_top_k=int(store.candidate_top_k),
                        )
                    )
                else:
                    scores = raw_scores
                    assignment_dustbins = None
                    assignment_score_space = (
                        "legacy_raw_strategy_score_without_learned_null"
                    )
                raw_identity = identity(raw_scores, "validation")
                identity_passed = _assignment_identity_gate(
                    raw_identity, baseline_validation_identity
                )
                strategy_admission_bonuses = (
                    admission_log_bonuses
                    if bool(args.global_assignment_joint_posterior)
                    and (
                        str(strategy_name).startswith("set_candidate_probability")
                        or str(strategy_name)
                        == "factorized_set_candidate_probability"
                    )
                    else (0.0,)
                )
                for admission_log_bonus in strategy_admission_bonuses:
                    utility_scores = candidate_admission_utility_scores(
                        scores,
                        admission_log_bonus=float(admission_log_bonus),
                    )
                    resolved_scores, selected_columns = global_assignment_score_matrix(
                        candidates.track_ids,
                        utility_scores,
                        selected_query_ids,
                        valid_mask=store.valid_edges & np.isfinite(utility_scores),
                        dustbin_score=assignment_dustbins,
                    )
                    global_pose, global_rows = pose(
                        resolved_scores,
                        "validation",
                        (
                            f"epoch{epoch}_{strategy_name}_global_partial_assignment_"
                            f"admission{float(admission_log_bonus):g}"
                        ),
                        max_matches=int(args.global_assignment_match_count),
                        selection_mode=str(args.global_assignment_selection_mode),
                    )
                    if baseline_global_validation_rows is None:
                        raise RuntimeError("global baseline pose rows were not initialized")
                    paired_safety = paired_pose_safety_report(
                        global_rows,
                        baseline_global_validation_rows,
                        catastrophic_translation_m=float(
                            args.paired_catastrophic_translation_m
                        ),
                        max_translation_regression_m=float(
                            args.paired_max_translation_regression_m
                        ),
                    )
                    aggregate_pose_passed = _pose_gate(
                        global_pose, baseline_global_validation_pose
                    )
                    pose_passed = bool(
                        aggregate_pose_passed and paired_safety["passes"]
                    )
                    validation_selected = selected_columns[
                        split_row_masks["validation"]
                    ]
                    risk = _relative_pose_risk(
                        global_pose, baseline_global_validation_pose
                    )
                    global_strategy_trials.append(
                        {
                            "strategy": str(strategy_name),
                            "assignment_mode": "whole_image_sparse_bipartite_with_per_query_dustbin",
                            "assignment_score_space": assignment_score_space,
                            "candidate_admission_log_bonus": float(
                                admission_log_bonus
                            ),
                            "posterior_modified_by_admission_bonus": False,
                            "dustbin_score": (
                                None
                                if assignment_dustbins is None
                                else "per_query_zero_in_candidate_vs_null_log_odds_space"
                            ),
                            "max_matches": int(args.global_assignment_match_count),
                            "selection_mode": str(
                                args.global_assignment_selection_mode
                            ),
                            "accepted_query_count": int(
                                np.sum(validation_selected >= 0)
                            ),
                            "validation_identity_before_global_conflict_resolution": raw_identity,
                            "validation_pose": global_pose,
                            "paired_pose_safety": paired_safety,
                            "relative_pose_risk": risk,
                            "passes_aggregate_pose_gate": bool(
                                aggregate_pose_passed
                            ),
                            "passes_pose_gate": bool(pose_passed),
                            "passes_identity_gate": bool(identity_passed),
                            "passes_stage_gate": bool(
                                pose_passed and identity_passed
                            ),
                        }
                    )
            global_epoch_choice = max(
                global_strategy_trials,
                key=lambda trial: (
                    float(trial["passes_stage_gate"]),
                    *_relative_pose_risk_rank_key(
                        trial["relative_pose_risk"]
                    ),
                    float(trial["validation_pose"]["recall_10cm_5deg"]),
                    float(trial["validation_pose"]["recall_25cm_2deg"]),
                    float(trial["validation_pose"]["recall_5cm_5deg"]),
                ),
            )
            global_epoch_key = (
                float(global_epoch_choice["passes_stage_gate"]),
                *_relative_pose_risk_rank_key(
                    global_epoch_choice["relative_pose_risk"]
                ),
                float(
                    global_epoch_choice["validation_pose"]["recall_10cm_5deg"]
                ),
                float(
                    global_epoch_choice["validation_pose"]["recall_25cm_2deg"]
                ),
            )
            if best_global_key is None or global_epoch_key > best_global_key:
                best_global_key = global_epoch_key
                best_global_epoch = int(epoch)
                best_global_selection = dict(global_epoch_choice)
                best_global_selection["validation_gate_passed"] = bool(
                    global_epoch_choice["passes_stage_gate"]
                )
                save_training_checkpoint(
                    best_global_checkpoint_path,
                    epoch=int(epoch),
                    selection=best_global_selection,
                    checkpoint_role="best_global_partial_assignment_validation",
                )
        epoch_choice = max(strategy_trials, key=lambda trial: tuple(trial["selection_key"]))
        epoch_key = tuple(epoch_choice["selection_key"])
        epoch_summary = {
            "epoch": int(epoch),
            "train_group_count": int(len(train_groups)),
            "train_edge_count": int(len(train_groups) * store.candidate_top_k),
            "train_metrics": train_metrics,
            "validation_strategies": strategy_trials,
            "chosen_validation_strategy": str(epoch_choice["strategy"]),
            "chosen_validation_mode": str(epoch_choice["selected"]["mode"]),
            "chosen_validation_margin": epoch_choice["selected"]["margin_threshold"],
            "validation_gate_passed": bool(epoch_choice["selected"]["passes_stage_gate"]),
            "validation_geometry_probability": validation_geometry_probability,
            "validation_top_l_availability": validation_top_l_availability,
            "validation_rescue_policy": validation_rescue_policy,
            "global_assignment_validation": global_strategy_trials,
            "epoch_seconds": float(time.time() - epoch_start),
        }
        history.append(epoch_summary)
        selected_pose = epoch_choice["selected"]["pose"]
        print(
            json.dumps(
                {
                    "event": "candidate_maplet_epoch_complete",
                    "epoch": int(epoch),
                    "epoch_seconds": float(epoch_summary["epoch_seconds"]),
                    "train_group_count": int(len(train_groups)),
                    "train_data_build_seconds": float(
                        train_metrics["data_build_seconds"]
                    ),
                    "train_gpu_compute_seconds": float(
                        train_metrics["gpu_compute_seconds"]
                    ),
                    "train_loss": float(train_metrics["loss"]),
                    "train_assignment_loss": float(train_metrics["assignment_loss"]),
                    "train_candidate_set_loss": float(train_metrics["candidate_set_loss"]),
                    "train_geometry_validity_loss": (
                        None
                        if "geometry_validity_loss" not in train_metrics
                        else float(train_metrics["geometry_validity_loss"])
                    ),
                    "train_rescue_policy_loss": (
                        None
                        if "rescue_policy_loss" not in train_metrics
                        else float(train_metrics["rescue_policy_loss"])
                    ),
                    "validation_strategy": str(epoch_choice["strategy"]),
                    "validation_mode": str(epoch_choice["selected"]["mode"]),
                    "validation_gate_passed": bool(
                        epoch_choice["selected"]["passes_stage_gate"]
                    ),
                    "validation_median_translation_m": float(
                        selected_pose["median_translation_m_success"]
                    ),
                    "validation_p90_translation_m": float(
                        selected_pose["p90_translation_m_success"]
                    ),
                    "validation_median_rotation_deg": float(
                        selected_pose["median_rotation_deg_success"]
                    ),
                    "global_validation_strategy": (
                        None
                        if not global_strategy_trials
                        else str(global_epoch_choice["strategy"])
                    ),
                    "global_validation_worst_error_ratio": (
                        None
                        if not global_strategy_trials
                        else global_epoch_choice["relative_pose_risk"].get(
                            "worst_error_ratio"
                        )
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if rescue_policy_enabled and validation_rescue_policy is not None:
            rescue_strategy = next(
                (
                    trial
                    for trial in strategy_trials
                    if str(trial["strategy"]) == "rescue_policy_resolved"
                ),
                None,
            )
            if rescue_strategy is None:
                raise ValueError(
                    "rescue policy training requires rescue_policy_resolved validation"
                )
            rescue_trial = rescue_strategy["unconditional"]
            rescue_pose = rescue_trial["pose"]
            rescue_worst_error_ratio = max(
                float(rescue_pose["median_translation_m_success"])
                / float(baseline_validation_pose["median_translation_m_success"]),
                float(rescue_pose["p90_translation_m_success"])
                / float(baseline_validation_pose["p90_translation_m_success"]),
                float(rescue_pose["median_rotation_deg_success"])
                / float(baseline_validation_pose["median_rotation_deg_success"]),
            )
            rescue_mean_error_ratio = float(
                np.mean(
                    [
                        float(rescue_pose["median_translation_m_success"])
                        / float(baseline_validation_pose["median_translation_m_success"]),
                        float(rescue_pose["p90_translation_m_success"])
                        / float(baseline_validation_pose["p90_translation_m_success"]),
                        float(rescue_pose["median_rotation_deg_success"])
                        / float(baseline_validation_pose["median_rotation_deg_success"]),
                    ]
                )
            )
            rescue_key = (
                float(rescue_trial["passes_stage_gate"]),
                float(rescue_pose["recall_10cm_5deg"]),
                float(rescue_pose["recall_5cm_5deg"]),
                float(rescue_pose["recall_25cm_2deg"]),
                -float(rescue_worst_error_ratio),
                -float(rescue_mean_error_ratio),
                float(validation_rescue_policy["candidate"]["auprc"]),
                float(validation_rescue_policy["keep"]["auprc"]),
            )
            if best_rescue_key is None or rescue_key > best_rescue_key:
                best_rescue_key = rescue_key
                best_rescue_epoch = int(epoch)
                best_rescue_selection = {
                    "strategy": "rescue_policy_resolved",
                    "mode": "unconditional",
                    "validation_gate_passed": bool(
                        rescue_trial["passes_stage_gate"]
                    ),
                    "validation_identity": rescue_trial["identity"],
                    "validation_pose": rescue_pose,
                    "validation_rescue_policy": validation_rescue_policy,
                    "worst_error_ratio_vs_baseline": float(
                        rescue_worst_error_ratio
                    ),
                    "mean_error_ratio_vs_baseline": float(
                        rescue_mean_error_ratio
                    ),
                }
                save_training_checkpoint(
                    best_rescue_checkpoint_path,
                    epoch=int(epoch),
                    selection=best_rescue_selection,
                    checkpoint_role="best_rescue_policy",
                )
        print(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "train": train_metrics,
                    "chosen_strategy": epoch_choice["strategy"],
                    "chosen_mode": epoch_choice["selected"]["mode"],
                    "chosen_margin": epoch_choice["selected"]["margin_threshold"],
                    "validation_gate_passed": epoch_choice["selected"]["passes_stage_gate"],
                    "validation_pose": epoch_choice["selected"]["pose"],
                    "epoch_seconds": epoch_summary["epoch_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if best_key is None or epoch_key > best_key:
            best_key = epoch_key
            best_epoch = int(epoch)
            best_selection = {
                "strategy": str(epoch_choice["strategy"]),
                "mode": str(epoch_choice["selected"]["mode"]),
                "margin_threshold": epoch_choice["selected"]["margin_threshold"],
                "validation_gate_passed": bool(epoch_choice["selected"]["passes_stage_gate"]),
                "validation_identity": epoch_choice["selected"]["identity"],
                "validation_pose": epoch_choice["selected"]["pose"],
                "validation_geometry_probability": validation_geometry_probability,
                "validation_top_l_availability": validation_top_l_availability,
            }
            save_training_checkpoint(
                checkpoint_path,
                epoch=int(epoch),
                selection=best_selection,
                checkpoint_role="best_overall_validation_policy",
            )
        save_training_checkpoint(
            last_checkpoint_path,
            epoch=int(epoch),
            selection=best_selection,
            checkpoint_role="last",
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_strategy = str(best_selection["strategy"])
    test_predictions = _predict_edges(
        model,
        store,
        split_edges["test"],
        device=device,
        batch_size=int(args.eval_batch_size),
        use_amp=use_amp,
    )
    test_top_l_availability = _factorized_top_l_availability_report(
        test_predictions,
        store,
        row_mask=split_row_masks["test"],
    )
    test_geometry_probability = _geometry_probability_report(
        test_predictions,
        store,
        row_mask=split_row_masks["test"],
        thresholds_px=geometry_thresholds,
        visibility_supervised=bool(
            float(args.candidate_visibility_loss_weight) > 0.0
        ),
    )
    test_rescue_policy = _rescue_policy_report(
        test_predictions,
        store,
        baseline_scores,
        row_mask=split_row_masks["test"],
        candidate_threshold_px=float(args.rescue_candidate_threshold_px),
        baseline_invalid_threshold_px=float(
            args.rescue_baseline_invalid_threshold_px
        ),
    )
    test_model_scores = test_predictions[best_strategy].reshape(
        len(store.selected_rows), store.candidate_top_k
    )
    unconditional_test_identity = identity(test_model_scores, "test")
    unconditional_test_pose, unconditional_pose_rows = pose(
        test_model_scores, "test", "candidate_maplet_unconditional_test"
    )
    baseline_test_identity = identity(baseline_scores, "test")
    baseline_test_pose, baseline_pose_rows = pose(baseline_scores, "test", "baseline_test")
    chosen_mode = str(best_selection["mode"])
    chosen_margin = best_selection["margin_threshold"]
    if bool(best_selection["validation_gate_passed"]) and chosen_mode == "selective":
        _selected, selected_scores, switched, margins = selective_switch_scores(
            test_model_scores,
            baseline_scores,
            margin_threshold=float(chosen_margin),
            valid_mask=store.valid_edges,
            preserve_baseline_row_confidence=best_strategy.endswith(
                "_prior_row_confidence"
            ),
        )
        selected_test_identity = identity(selected_scores, "test")
        selected_test_pose, selected_pose_rows = pose(
            selected_scores, "test", "candidate_maplet_selective_test"
        )
        selected_test_switch_count = int(np.sum(switched[split_row_masks["test"]]))
        selected_test_margin_median = float(
            np.median(margins[split_row_masks["test"]][np.isfinite(margins[split_row_masks["test"]])])
        )
    elif bool(best_selection["validation_gate_passed"]) and chosen_mode == "unconditional":
        selected_scores = test_model_scores
        selected_test_identity = unconditional_test_identity
        selected_test_pose = unconditional_test_pose
        selected_pose_rows = unconditional_pose_rows
        selected_test_switch_count = None
        selected_test_margin_median = None
    else:
        selected_scores = baseline_scores
        selected_test_identity = baseline_test_identity
        selected_test_pose = baseline_test_pose
        selected_pose_rows = baseline_pose_rows
        selected_test_switch_count = 0
        selected_test_margin_median = None
    test_pose_gate_passed = _pose_gate(selected_test_pose, baseline_test_pose)
    test_identity_gate_passed = _assignment_identity_gate(
        selected_test_identity, baseline_test_identity
    )
    test_gate_passed = bool(test_pose_gate_passed and test_identity_gate_passed)
    best_rescue_test = None
    best_rescue_pose_rows: list[dict[str, object]] = []
    if best_rescue_checkpoint_path.exists():
        rescue_checkpoint = torch.load(best_rescue_checkpoint_path, map_location=device)
        model.load_state_dict(rescue_checkpoint["model_state_dict"])
        rescue_predictions = _predict_edges(
            model,
            store,
            split_edges["test"],
            device=device,
            batch_size=int(args.eval_batch_size),
            use_amp=use_amp,
        )
        if "rescue_policy_resolved" not in rescue_predictions:
            raise ValueError("best rescue checkpoint produced no resolved policy")
        rescue_scores = rescue_predictions["rescue_policy_resolved"].reshape(
            len(store.selected_rows), store.candidate_top_k
        )
        rescue_test_identity = identity(rescue_scores, "test")
        rescue_test_pose, best_rescue_pose_rows = pose(
            rescue_scores, "test", "best_rescue_policy_test"
        )
        rescue_pose_passed = _pose_gate(rescue_test_pose, baseline_test_pose)
        rescue_identity_passed = _assignment_identity_gate(
            rescue_test_identity, baseline_test_identity
        )
        rescue_choices = np.argmax(rescue_scores, axis=1)
        baseline_choices = np.argmax(baseline_scores, axis=1)
        best_rescue_test = {
            "checkpoint_epoch": int(rescue_checkpoint["epoch"]),
            "checkpoint_selection": rescue_checkpoint.get("selection"),
            "switch_count": int(
                np.sum(
                    (rescue_choices != baseline_choices)
                    & split_row_masks["test"]
                )
            ),
            "identity": rescue_test_identity,
            "pose": rescue_test_pose,
            "probability": _rescue_policy_report(
                rescue_predictions,
                store,
                baseline_scores,
                row_mask=split_row_masks["test"],
                candidate_threshold_px=float(args.rescue_candidate_threshold_px),
                baseline_invalid_threshold_px=float(
                    args.rescue_baseline_invalid_threshold_px
                ),
            ),
            "passes_pose_gate": bool(rescue_pose_passed),
            "passes_identity_gate": bool(rescue_identity_passed),
            "passes_stage_gate": bool(
                rescue_pose_passed and rescue_identity_passed
            ),
        }
    smoke_protocol = bool(int(args.smoke_validation_query_count) > 0)
    production_promoted = bool(
        best_selection["validation_gate_passed"]
        and test_gate_passed
        and not smoke_protocol
        and args.evaluation_role == "untouched_test"
    )
    baseline_columns = np.argmax(
        np.where(store.valid_edges, baseline_scores, -np.inf), axis=1
    )
    baseline_residuals = candidate_residuals[
        np.arange(len(candidate_residuals)), baseline_columns
    ]
    rescue_targets = (
        store.valid_edges
        & (baseline_residuals[:, None] > float(args.rescue_baseline_invalid_threshold_px))
        & (candidate_residuals <= float(args.rescue_candidate_threshold_px))
    )
    rescue_targets[np.arange(len(rescue_targets)), baseline_columns] = False
    (output_dir / "pose_rows_test.json").write_text(
        json.dumps(
            {
                "baseline": baseline_pose_rows,
                "unconditional": unconditional_pose_rows,
                "selected": selected_pose_rows,
                "best_rescue": best_rescue_pose_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "stage": (
            "candidate_maplet_assignment_only_development_refit"
            if refit_metadata is not None
            else "candidate_maplet_assignment_only_training"
        ),
        "protocol": {
            "evaluation_role": str(args.evaluation_role),
            "development_data_reused": bool(args.evaluation_role == "development"),
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "global_mapper_frozen": True,
            "pixel_measurement_enabled": False,
            "query_pose_is_input": False,
            "query_pose_target_only": True,
            "anchor_positive_threshold_px": float(args.positive_threshold_px),
            "context_assignment_threshold_px": float(args.assignment_threshold_px),
            "support_view_count": int(args.support_view_count),
            "support_view_aggregation": (
                "identity_conditioned_posterior_logsumexp_full_view_mixture"
                if bool(args.identity_conditioned_view_posterior)
                and bool(args.full_candidate_view_mixture)
                else "identity_conditioned_posterior_logsumexp_identity_marginal"
                if bool(args.identity_conditioned_view_posterior)
                else "learned_prior_logsumexp_identity_marginal"
                if bool(args.candidate_view_marginalization)
                else "learned_posterior_embedding_mean"
            ),
            "support_view_dropout": float(args.support_view_dropout),
            "system_hard_identity_curriculum": {
                "enabled": bool(
                    hard_oversample_factor > 1.0
                    or hard_conditional_weight > 1.0
                ),
                "complete_top_l_groups_preserved": True,
                "geometry_visibility_rescue_losses_reweighted": False,
                "oversample_factor": hard_oversample_factor,
                "conditional_identity_weight": hard_conditional_weight,
                "availability_and_null_distribution": (
                    "natural_unweighted_groups"
                    if hard_oversample_factor == 1.0
                    else "oversampled_DIAGNOSTIC_ONLY"
                ),
                "score_source": system_hard_score_source,
                "training_groups_only": True,
                **system_hard_group_audit,
            },
            "pose_conditioned_system_hard_negative_training": {
                "enabled": bool(pose_hard_margin_weight > 0.0),
                "source": pose_conditioned_hard_negative_audit,
                "target_join_role": "training_only_after_target_free_generation",
                "identity_logit_space": "prior_free_candidate_evidence",
                "loss": "best_positive_vs_concrete_bad_candidate_margin",
                "within_group_weighting": "sqrt_bad_pose_mode_count",
                "across_group_weighting": "equal",
                "margin": float(pose_hard_margin),
                "weight": float(pose_hard_margin_weight),
                "availability_null_geometry_targets_reweighted": False,
                "validation_or_test_targets_consumed": False,
            },
            "pose_conditioned_system_hard_mode_training": {
                "enabled": bool(pose_hard_mode_weight > 0.0),
                "source": pose_conditioned_hard_mode_audit,
                "target_join_role": "training_only_after_target_free_generation",
                "identity_logit_space": "prior_free_candidate_evidence",
                "loss": "strongest_support_group_mean_pose_mode_margin",
                "structured_unit": "coherent_bad_pose_mode_across_query_groups",
                "top_group_fraction": float(pose_hard_mode_fraction),
                "minimum_mode_groups": int(pose_hard_mode_minimum_groups),
                "margin": float(pose_hard_mode_margin),
                "weight": float(pose_hard_mode_weight),
                "query_grouped_batches": bool(
                    args.query_grouped_training_batches
                ),
                "complete_mode_membership_required_per_batch": True,
                "availability_null_geometry_targets_reweighted": False,
                "validation_or_test_targets_consumed": False,
            },
            "candidate_resolution": "top_l_set_attention_multi_positive_with_dustbin",
            "candidate_view_reasoning": (
                "all_candidate_view_nodes_before_identity_marginalization"
                if bool(args.full_candidate_view_mixture)
                else "candidate_embedding_before_identity_marginalization"
            ),
            "anchor_role_encoding": (
                "learned_explicit_query_support_role_embedding"
                if bool(args.explicit_anchor_role_embedding)
                else "implicit_node_zero_DIAGNOSTIC_ONLY"
            ),
            "set_identity_evidence": (
                "deployable_context_without_coarse_or_baseline_prior_fields"
                if bool(args.deployable_identity_context)
                else "query_support_appearance_only_with_explicit_final_coarse_prior"
                if bool(args.prior_free_set_identity)
                else "legacy_static_fused_candidate_embedding"
            ),
            "deployable_identity_context": (
                None
                if not bool(args.deployable_identity_context)
                else {
                    "static_start_index": int(
                        deployable_identity_static_start_index
                    ),
                    "static_feature_names": list(
                        store.static_feature_names[
                            deployable_identity_static_start_index:
                        ]
                    ),
                    "anchor_evidence": [
                        "descriptor_cosine",
                        "contextual_cosine",
                        "optimal_transport_anchor_probability",
                        "optimal_transport_dustbin_probability",
                    ],
                    "excluded_prior_or_rank_features": list(
                        store.static_feature_names[
                            :deployable_identity_static_start_index
                        ]
                    ),
                    "query_pose_or_ground_truth_used": False,
                }
            ),
            "prior_free_identity_loss_weight": float(
                args.prior_free_identity_loss_weight
            ),
            "prior_free_conditional_identity_loss_weight": float(
                args.prior_free_conditional_identity_loss_weight
            ),
            "factorized_set_posterior": bool(args.factorized_set_posterior),
            "factorized_top_l_availability_loss_weight": float(
                args.factorized_top_l_availability_loss_weight
            ),
            "global_assignment_validation": (
                None
                if not bool(args.global_assignment_validation)
                else {
                    "checkpoint_role": "best_global_partial_assignment_validation",
                    "assignment": "whole_image_sparse_bipartite_with_per_query_dustbin",
                    "set_probability_score_space": (
                        "candidate_vs_learned_null_log_odds_for_joint_MAP"
                        if bool(args.global_assignment_joint_posterior)
                        else "legacy_raw_probability_without_learned_null"
                    ),
                    "non_set_strategy_dustbin_score": None,
                    "max_matches": int(args.global_assignment_match_count),
                    "selection_mode": str(
                        args.global_assignment_selection_mode
                    ),
                    "candidate_admission_log_bonuses": list(
                        admission_log_bonuses
                    ),
                    "candidate_admission_bonus_semantics": (
                        "task_utility_only_posterior_unchanged"
                    ),
                    "paired_pose_safety": {
                        "catastrophic_translation_threshold_m": float(
                            args.paired_catastrophic_translation_m
                        ),
                        "allowed_max_translation_regression_m": float(
                            args.paired_max_translation_regression_m
                        ),
                    },
                    "frozen_baseline_source": frozen_global_baseline_source,
                    "test_used_for_selection": False,
                }
            ),
            "real_no_match_rows": True,
            "geometry_validity_supervision": (
                None
                if not geometry_validity_enabled
                else {
                    "source": "real_query_pose_anchor_reprojection_residual_target_only",
                    "thresholds_px": list(geometry_thresholds),
                    "loss": "unweighted_binary_cross_entropy_proper_scoring_rule",
                    "training_group_distribution": (
                        "full"
                        if full_group_distribution_training
                        else "smoke_subsample_not_calibration_valid"
                    ),
                    "monotonic_parameterization": True,
                    "weight": float(args.geometry_validity_loss_weight),
                    "visibility_weight": float(args.candidate_visibility_loss_weight),
                }
            ),
            "rescue_policy_supervision": (
                None
                if not rescue_policy_enabled
                else {
                    "source": "real_query_pose_anchor_reprojection_residual_target_only",
                    "action_space": "keep_baseline_or_switch_to_set_valued_rescue_candidate",
                    "candidate_threshold_px": float(args.rescue_candidate_threshold_px),
                    "baseline_invalid_threshold_px": float(
                        args.rescue_baseline_invalid_threshold_px
                    ),
                    "loss": "set_valued_cross_entropy_with_keep_dustbin",
                    "weight": float(args.rescue_policy_loss_weight),
                }
            ),
            "static_features_standardized_on_train_split": bool(
                args.standardize_static_features
            ),
            "development_refit": bool(refit_metadata is not None),
            "test_used_for_epoch_or_policy_selection": False,
            "smoke_protocol": smoke_protocol,
            "local_features": (
                "alike_64_plus_radio_intermediate_pca64"
                if args.radio_intermediate_cache is not None
                else "alike_64_only"
            ),
        },
        "split": split,
        "data": {
            "manifest": data_manifest,
            "edge_count": int(store.edge_count),
            "candidate_positive_rate": float(np.mean(store.edge_labels)),
            "candidate_visible_rate": float(np.mean(store.candidate_visible)),
            "candidate_geometry_positive_rates": {
                str(float(threshold)): float(
                    np.mean(store.candidate_residuals_px <= float(threshold))
                )
                for threshold in geometry_thresholds
            },
            "rescue_candidate_positive_rate": float(np.mean(rescue_targets)),
            "rescue_group_positive_rate": float(
                np.mean(np.any(rescue_targets, axis=1))
            ),
            "available_training_group_count": int(len(available_training_groups)),
            "available_positive_group_count": available_positive_group_count,
            "available_no_match_group_count": available_no_match_group_count,
            "required_no_match_group_ratio_for_full_distribution": required_no_match_ratio,
            "full_group_distribution_training": full_group_distribution_training,
            "system_hard_group_audit": system_hard_group_audit,
            "pose_conditioned_hard_negative_audit": (
                pose_conditioned_hard_negative_audit
            ),
            "pose_conditioned_hard_mode_audit": (
                pose_conditioned_hard_mode_audit
            ),
            "train_edge_count": int(len(training_edges)),
            "validation_edge_count": int(len(split_edges["validation"])),
            "test_edge_count": int(len(split_edges["test"])),
        },
        "model_config": config.to_dict(),
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "training_batch_prefetch": bool(args.training_batch_prefetch),
            "pose_conditioned_hard_negative_margin_weight": float(
                pose_hard_margin_weight
            ),
            "pose_conditioned_hard_negative_margin": float(pose_hard_margin),
            "pose_conditioned_hard_mode_margin_weight": float(
                pose_hard_mode_weight
            ),
            "pose_conditioned_hard_mode_margin": float(
                pose_hard_mode_margin
            ),
            "pose_conditioned_hard_mode_top_group_fraction": float(
                pose_hard_mode_fraction
            ),
            "pose_conditioned_hard_mode_minimum_groups": int(
                pose_hard_mode_minimum_groups
            ),
            "query_grouped_training_batches": bool(
                args.query_grouped_training_batches
            ),
            "best_epoch": int(best_epoch),
            "best_selection": best_selection,
            "best_rescue_epoch": int(best_rescue_epoch),
            "best_rescue_selection": best_rescue_selection,
            "best_global_epoch": int(best_global_epoch),
            "best_global_selection": best_global_selection,
            "baseline_global_validation_pose": baseline_global_validation_pose,
            "refit_selection_source": refit_metadata,
            "history": history,
        },
        "test": {
            "baseline": {"identity": baseline_test_identity, "pose": baseline_test_pose},
            "top_l_availability": test_top_l_availability,
            "geometry_probability": test_geometry_probability,
            "rescue_policy": test_rescue_policy,
            "best_rescue": best_rescue_test,
            "unconditional": {
                "strategy": best_strategy,
                "identity": unconditional_test_identity,
                "pose": unconditional_test_pose,
            },
            "selected": {
                "mode": chosen_mode,
                "margin_threshold": chosen_margin,
                "switch_count": selected_test_switch_count,
                "margin_median": selected_test_margin_median,
                "identity": selected_test_identity,
                "pose": selected_test_pose,
                "passes_pose_gate": bool(test_pose_gate_passed),
                "passes_identity_gate": bool(test_identity_gate_passed),
                "passes_stage_gate": bool(test_gate_passed),
            },
        },
        "gate": {
            "validation_passed": bool(best_selection["validation_gate_passed"]),
            "test_passed": bool(test_gate_passed),
            "test_pose_passed": bool(test_pose_gate_passed),
            "test_identity_passed": bool(test_identity_gate_passed),
            "protocol_eligible_for_production": bool(
                not smoke_protocol and args.evaluation_role == "untouched_test"
            ),
            "evaluation_role_allows_production_promotion": bool(
                args.evaluation_role == "untouched_test"
            ),
            "production_promoted": bool(production_promoted),
        },
        "runtime_seconds": float(time.time() - start_time),
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256_short(checkpoint_path),
            "best_rescue_checkpoint": (
                None
                if not best_rescue_checkpoint_path.exists()
                else str(best_rescue_checkpoint_path)
            ),
            "best_rescue_checkpoint_sha256": (
                None
                if not best_rescue_checkpoint_path.exists()
                else file_sha256_short(best_rescue_checkpoint_path)
            ),
            "best_global_checkpoint": (
                None
                if not best_global_checkpoint_path.exists()
                else str(best_global_checkpoint_path)
            ),
            "best_global_checkpoint_sha256": (
                None
                if not best_global_checkpoint_path.exists()
                else file_sha256_short(best_global_checkpoint_path)
            ),
            "last_checkpoint": (
                None if not last_checkpoint_path.exists() else str(last_checkpoint_path)
            ),
            "last_checkpoint_sha256": (
                None
                if not last_checkpoint_path.exists()
                else file_sha256_short(last_checkpoint_path)
            ),
            "summary": str(output_dir / "summary.json"),
            "split": str(output_dir / "split.json"),
            "pose_rows_test": str(output_dir / "pose_rows_test.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": summary["gate"], "test": summary["test"], "runtime_seconds": summary["runtime_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
