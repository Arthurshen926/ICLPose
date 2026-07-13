"""Train assignment-only candidate-maplet matching on real hard proposals."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

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
    CandidateMapletMatcher,
    CandidateMapletMatcherConfig,
    candidate_maplet_group_loss,
)
from feature_extract.vfm.localization.local_assignment_linear import (
    resolve_rescue_policy_scores,
    selective_switch_scores,
)
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
)


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
    parser.add_argument(
        "--global_assignment_validation",
        action="store_true",
        help=(
            "also select best_global.pt with whole-image one-to-one assignment and a "
            "fixed pose budget; legacy best.pt selection remains unchanged"
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
) -> dict[str, float]:
    ratios = {
        key: float(pose[key]) / max(float(baseline[key]), 1e-12)
        for key in (
            "median_translation_m_success",
            "p90_translation_m_success",
            "median_rotation_deg_success",
        )
    }
    return {
        **ratios,
        "worst_error_ratio": float(max(ratios.values())),
        "mean_log_error_ratio": float(
            np.mean(np.log(np.maximum(tuple(ratios.values()), 1e-12)))
        ),
    }


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


def _load_refit_selection(
    path: Path,
    *,
    data_manifest: dict[str, object],
    split: dict[str, object],
    model_config: dict[str, object],
    epochs: int,
) -> tuple[dict[str, object], dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(checkpoint.get("format", "")) not in {
        "candidate_maplet_matcher_checkpoint_v5",
        "candidate_maplet_matcher_checkpoint_v6",
        "candidate_maplet_matcher_checkpoint_v7",
    }:
        raise ValueError("refit selection checkpoint has an unsupported format")
    if checkpoint.get("data_manifest") != data_manifest:
        raise ValueError("refit selection checkpoint uses different data inputs")
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
) -> np.ndarray:
    """Sample complete top-L rows so ranking and no-match targets stay well defined."""

    if float(no_match_ratio) < 0.0:
        raise ValueError("no_match_ratio must be non-negative")
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
    rng.shuffle(output)
    return output


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
                result = model(batch)
            candidate = torch.sigmoid(result["candidate_logits"].float()).cpu().numpy()
            candidate_embeddings = result.get("candidate_embeddings")
            if not isinstance(candidate_embeddings, torch.Tensor):
                raise TypeError("matcher output is missing candidate_embeddings")
            embeddings[edges] = candidate_embeddings.float().cpu().numpy()
            assignment = np.asarray(
                [
                    float(torch.exp(values[0, 0]).detach().cpu().item())
                    for values in result["query_log_probabilities"]
                ],
                dtype=np.float32,
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
        view_weights = view_weights_t.float().cpu().numpy()
        for view_rank in range(store.support_view_count):
            outputs[f"support_view_probability_{view_rank}"][flat_edges] = (
                view_weights[:, view_rank].astype(np.float32)
            )
        candidate_probabilities = joint_probabilities[:, :top_l].cpu().numpy()
        dustbin_probabilities = joint_probabilities[:, top_l].cpu().numpy()
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
    if str(args.global_assignment_baseline_summary) and not bool(
        args.global_assignment_validation
    ):
        raise ValueError(
            "--global_assignment_baseline_summary requires --global_assignment_validation"
        )
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
    if bool(args.decoupled_candidate_heads) and not geometry_validity_enabled:
        raise ValueError(
            "--decoupled_candidate_heads requires geometry or visibility supervision"
        )
    if bool(args.candidate_view_marginalization) and not bool(
        args.decoupled_candidate_heads
    ):
        raise ValueError(
            "--candidate_view_marginalization requires --decoupled_candidate_heads"
        )
    rescue_policy_enabled = bool(float(args.rescue_policy_loss_weight) > 0.0)
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
        return _identity_metrics(
            nearest_residuals=nearest_residuals[mask],
            candidate_residuals=candidate_residuals[mask],
            scores=scores[mask],
            query_ids=selected_query_ids[mask],
            labels=store.labels[mask],
            valid_edges=store.valid_edges[mask],
        )

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
    if bool(args.global_assignment_validation):
        baseline_global_scores, _baseline_global_columns = global_assignment_score_matrix(
            candidates.track_ids,
            baseline_scores,
            selected_query_ids,
            valid_mask=store.valid_edges,
            dustbin_score=None,
        )
        baseline_global_validation_pose, _baseline_global_rows = pose(
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
        "candidate_maplet_matcher_checkpoint_v7"
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
    required_no_match_ratio = float(
        available_no_match_group_count / max(available_positive_group_count, 1)
    )
    full_group_distribution_training = bool(
        int(args.max_train_groups) <= 0
        and float(args.no_match_group_ratio) + 1e-12 >= required_no_match_ratio
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
        )
        metric_sums: dict[str, float] = {}
        batch_count = 0
        data_build_seconds = 0.0
        gpu_compute_seconds = 0.0
        groups_per_batch = max(1, int(args.batch_size) // store.candidate_top_k)
        for batch_start in range(0, len(train_groups), groups_per_batch):
            groups = train_groups[batch_start : batch_start + groups_per_batch]
            edges = (
                groups[:, None] * store.candidate_top_k
                + np.arange(store.candidate_top_k, dtype=np.int64)[None]
            ).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            data_start = time.perf_counter()
            batches = []
            for view_rank in range(store.support_view_count):
                ranks = np.full(edges.shape, view_rank, dtype=np.int64)
                batches.append(store.batch(edges, view_ranks=ranks).to(device))
            data_build_seconds += time.perf_counter() - data_start
            if bool(args.profile_training_timing) and device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = [model(batch) for batch in batches]
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
                    support_view_dropout=float(args.support_view_dropout),
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
                scores = validation_predictions[strategy_name].reshape(
                    len(store.selected_rows), store.candidate_top_k
                )
                resolved_scores, selected_columns = global_assignment_score_matrix(
                    candidates.track_ids,
                    scores,
                    selected_query_ids,
                    valid_mask=store.valid_edges & np.isfinite(scores),
                    dustbin_score=None,
                )
                global_pose, _global_rows = pose(
                    resolved_scores,
                    "validation",
                    f"epoch{epoch}_{strategy_name}_global_partial_assignment",
                    max_matches=int(args.global_assignment_match_count),
                    selection_mode=str(args.global_assignment_selection_mode),
                )
                raw_identity = identity(scores, "validation")
                pose_passed = _pose_gate(
                    global_pose, baseline_global_validation_pose
                )
                identity_passed = _assignment_identity_gate(
                    raw_identity, baseline_validation_identity
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
                        "dustbin_score": None,
                        "max_matches": int(args.global_assignment_match_count),
                        "selection_mode": str(
                            args.global_assignment_selection_mode
                        ),
                        "accepted_query_count": int(
                            np.sum(validation_selected >= 0)
                        ),
                        "validation_identity_before_global_conflict_resolution": raw_identity,
                        "validation_pose": global_pose,
                        "relative_pose_risk": risk,
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
                    -float(trial["relative_pose_risk"]["worst_error_ratio"]),
                    -float(trial["relative_pose_risk"]["mean_log_error_ratio"]),
                    float(trial["validation_pose"]["recall_10cm_5deg"]),
                    float(trial["validation_pose"]["recall_25cm_2deg"]),
                    float(trial["validation_pose"]["recall_5cm_5deg"]),
                ),
            )
            global_epoch_key = (
                float(global_epoch_choice["passes_stage_gate"]),
                -float(
                    global_epoch_choice["relative_pose_risk"][
                        "worst_error_ratio"
                    ]
                ),
                -float(
                    global_epoch_choice["relative_pose_risk"][
                        "mean_log_error_ratio"
                    ]
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
                    "train_geometry_validity_loss": float(
                        train_metrics["geometry_validity_loss"]
                    ),
                    "train_rescue_policy_loss": float(
                        train_metrics["rescue_policy_loss"]
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
                        else float(
                            global_epoch_choice["relative_pose_risk"][
                                "worst_error_ratio"
                            ]
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
                "learned_posterior_logsumexp_identity_marginal"
                if bool(args.candidate_view_marginalization)
                else "learned_posterior_embedding_mean"
            ),
            "support_view_dropout": float(args.support_view_dropout),
            "candidate_resolution": "top_l_set_attention_multi_positive_with_dustbin",
            "global_assignment_validation": (
                None
                if not bool(args.global_assignment_validation)
                else {
                    "checkpoint_role": "best_global_partial_assignment_validation",
                    "assignment": "whole_image_sparse_bipartite_with_per_query_dustbin",
                    "dustbin_score": None,
                    "max_matches": int(args.global_assignment_match_count),
                    "selection_mode": str(
                        args.global_assignment_selection_mode
                    ),
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
            "train_edge_count": int(len(training_edges)),
            "validation_edge_count": int(len(split_edges["validation"])),
            "test_edge_count": int(len(split_edges["test"])),
        },
        "model_config": config.to_dict(),
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
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
