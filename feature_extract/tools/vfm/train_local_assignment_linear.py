"""Fit and gate low-capacity local assignment and no-match models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.tools.vfm.train_local_assignment_matcher import _pose_inputs, _query_rows
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.local_assignment_data import LocalAssignmentFeatureStore
from feature_extract.vfm.localization.local_assignment_linear import (
    LinearLogitModel,
    build_identity_candidate_features,
    build_no_match_features,
    selective_switch_scores,
)
from feature_extract.vfm.localization.local_assignment_probe import (
    binary_average_precision,
    summarize_assignment_strategy,
)


def _float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("expected at least one floating-point value")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_arrays", required=True)
    parser.add_argument("--real_feature_cache", required=True)
    parser.add_argument("--query_global_cache", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_query_count", type=int, default=60)
    parser.add_argument("--validation_query_count", type=int, default=15)
    parser.add_argument("--identity_c_values", type=_float_list, default=(0.001, 0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--no_match_c_values", type=_float_list, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument(
        "--switch_margin_thresholds",
        type=_float_list,
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    )
    parser.add_argument("--baseline_strategy", default="alike_support_top4_mean")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _split_query_ids(
    query_ids: Sequence[str],
    *,
    train_count: int,
    validation_count: int,
) -> dict[str, list[str]]:
    train_end = int(train_count)
    validation_end = train_end + int(validation_count)
    if train_end <= 0 or validation_end >= len(query_ids):
        raise ValueError("train/validation counts must leave a non-empty contiguous test block")
    return {
        "strategy": "contiguous_temporal_blocks_v1",
        "train": list(query_ids[:train_end]),
        "validation": list(query_ids[train_end:validation_end]),
        "test": list(query_ids[validation_end:]),
    }


def _row_mask(query_ids: np.ndarray, selected: Sequence[str]) -> np.ndarray:
    return np.isin(query_ids.astype(str), np.asarray(selected, dtype=np.str_))


def _identity_summary(
    store: LocalAssignmentFeatureStore,
    rows: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    return summarize_assignment_strategy(
        candidates=store.subset_candidates(rows),
        correct_track_ids=store.correct_track_ids[rows],
        query_ids=[store.query_ids[int(row)] for row in rows],
        scores=np.asarray(scores, dtype=np.float32)[rows],
    )


def _fit_logistic(features: np.ndarray, labels: np.ndarray, *, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=0,
    )
    model.fit(np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=bool))
    return model


def _safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.asarray(labels, dtype=bool)
    if np.unique(target).size < 2:
        return 0.0
    return float(roc_auc_score(target, probabilities))


def _no_match_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    target = np.asarray(labels, dtype=bool)
    values = np.asarray(probabilities, dtype=np.float32)
    return {
        "positive_rate": float(np.mean(target)),
        "average_precision": float(average_precision_score(target, values)),
        "roc_auc": _safe_roc_auc(target, values),
    }


def _evaluate_pose(
    *,
    name: str,
    store: LocalAssignmentFeatureStore,
    rows: np.ndarray,
    scores: np.ndarray,
    cameras,
    images_by_name,
    reprojection_error_px: float,
    iterations: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = store.subset_candidates(rows)
    observations, query_ids = _pose_inputs(store, rows)
    return _evaluate_pose_strategy(
        strategy=str(name),
        scores=np.asarray(scores, dtype=np.float32)[rows],
        candidates=candidates,
        query_observations=observations,
        query_ids=query_ids,
        landmark_index=store.landmark_index,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(reprojection_error_px),
        iterations=int(iterations),
    )


def _pose_gate(candidate: dict[str, object], baseline: dict[str, object]) -> bool:
    lower_is_better = (
        "median_translation_m_success",
        "p90_translation_m_success",
        "median_rotation_deg_success",
    )
    higher_is_better = (
        "success_rate",
        "recall_25cm_2deg",
        "recall_10cm_5deg",
        "recall_5cm_5deg",
    )
    tolerance = 1e-12
    no_regression = all(
        float(candidate[name]) <= float(baseline[name]) + tolerance for name in lower_is_better
    ) and all(
        float(candidate[name]) + tolerance >= float(baseline[name]) for name in higher_is_better
    )
    strict_improvement = any(
        float(candidate[name]) < float(baseline[name]) - tolerance for name in lower_is_better
    ) or any(
        float(candidate[name]) > float(baseline[name]) + tolerance for name in higher_is_better
    )
    return bool(no_regression and strict_improvement)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = Path(args.probe_arrays)
    with np.load(probe_path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    store = LocalAssignmentFeatureStore(
        probe_arrays=probe_path,
        real_feature_cache=Path(args.real_feature_cache),
        query_global_cache=Path(args.query_global_cache),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        maplet_support_index=Path(args.maplet_support_index),
    )
    query_ids = np.asarray(store.query_ids, dtype=np.str_)
    split = _split_query_ids(
        store.unique_query_ids,
        train_count=int(args.train_query_count),
        validation_count=int(args.validation_query_count),
    )
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    split_rows = {
        name: _query_rows(store, split[name])
        for name in ("train", "validation", "test")
    }
    row_masks = {
        name: _row_mask(query_ids, split[name])
        for name in ("train", "validation", "test")
    }

    identity_features, identity_feature_names = build_identity_candidate_features(payload)
    candidate_tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
    candidate_valid = candidate_tracks >= 0
    identity_labels = candidate_tracks == store.correct_track_ids[:, None]
    identity_trials: list[dict[str, object]] = []
    identity_models: list[LogisticRegression] = []
    identity_scores_by_c: list[np.ndarray] = []
    for c_value in args.identity_c_values:
        train_pairs = row_masks["train"][:, None] & candidate_valid
        model = _fit_logistic(
            identity_features[train_pairs],
            identity_labels[train_pairs],
            c_value=float(c_value),
        )
        scores = model.decision_function(identity_features.reshape(-1, identity_features.shape[2])).reshape(
            candidate_tracks.shape
        ).astype(np.float32)
        scores[~candidate_valid] = -np.inf
        validation = _identity_summary(store, split_rows["validation"], scores)
        validation_pair_mask = row_masks["validation"][:, None] & candidate_valid
        pair_ap = binary_average_precision(
            identity_labels[validation_pair_mask],
            scores[validation_pair_mask],
        )
        identity_trials.append(
            {
                "c": float(c_value),
                "validation": validation,
                "validation_pair_average_precision": float(pair_ap),
            }
        )
        identity_models.append(model)
        identity_scores_by_c.append(scores)
    best_identity_index = max(
        range(len(identity_trials)),
        key=lambda index: (
            float(identity_trials[index]["validation"]["recall_at_1"]),
            float(identity_trials[index]["validation_pair_average_precision"]),
            -abs(np.log10(float(identity_trials[index]["c"]))),
        ),
    )
    identity_model_sklearn = identity_models[best_identity_index]
    identity_scores = identity_scores_by_c[best_identity_index]
    chosen_identity_c = float(identity_trials[best_identity_index]["c"])

    identity_metrics: dict[str, object] = {}
    for name in ("train", "validation", "test"):
        pair_mask = row_masks[name][:, None] & candidate_valid
        identity_metrics[name] = {
            **_identity_summary(store, split_rows[name], identity_scores),
            "pair_average_precision": binary_average_precision(
                identity_labels[pair_mask],
                identity_scores[pair_mask],
            ),
        }
    baseline_scores = np.asarray(payload[f"strategy__{args.baseline_strategy}"], dtype=np.float32)
    baseline_identity = {
        name: _identity_summary(store, split_rows[name], baseline_scores)
        for name in ("validation", "test")
    }

    with np.load(Path(args.real_feature_cache), allow_pickle=False) as data:
        query_detector_scores = np.asarray(data["query_detector_scores"], dtype=np.float32)
    no_match_features, no_match_feature_names = build_no_match_features(payload, query_detector_scores)
    no_match_labels = ~np.any(identity_labels & candidate_valid, axis=1)
    no_match_trials: list[dict[str, object]] = []
    no_match_models: list[LogisticRegression] = []
    no_match_probabilities_by_c: list[np.ndarray] = []
    for c_value in args.no_match_c_values:
        model = _fit_logistic(
            no_match_features[row_masks["train"]],
            no_match_labels[row_masks["train"]],
            c_value=float(c_value),
        )
        probabilities = model.predict_proba(no_match_features)[:, 1].astype(np.float32)
        metrics = _no_match_metrics(
            no_match_labels[row_masks["validation"]],
            probabilities[row_masks["validation"]],
        )
        no_match_trials.append({"c": float(c_value), "validation": metrics})
        no_match_models.append(model)
        no_match_probabilities_by_c.append(probabilities)
    best_no_match_index = max(
        range(len(no_match_trials)),
        key=lambda index: (
            float(no_match_trials[index]["validation"]["average_precision"]),
            float(no_match_trials[index]["validation"]["roc_auc"]),
            -abs(np.log10(float(no_match_trials[index]["c"]))),
        ),
    )
    no_match_model_sklearn = no_match_models[best_no_match_index]
    no_match_probabilities = no_match_probabilities_by_c[best_no_match_index]
    chosen_no_match_c = float(no_match_trials[best_no_match_index]["c"])
    no_match_metrics = {
        name: _no_match_metrics(
            no_match_labels[row_masks[name]],
            no_match_probabilities[row_masks[name]],
        )
        for name in ("train", "validation", "test")
    }

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    validation_rows = split_rows["validation"]
    baseline_validation_pose, baseline_validation_pose_rows = _evaluate_pose(
        name=str(args.baseline_strategy),
        store=store,
        rows=validation_rows,
        scores=baseline_scores,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(args.pnp_reprojection_error_px),
        iterations=int(args.pnp_iterations),
    )
    threshold_trials: list[dict[str, object]] = []
    resolved_by_threshold: list[np.ndarray] = []
    for threshold in args.switch_margin_thresholds:
        _selected, resolved, switched, _margins = selective_switch_scores(
            identity_scores,
            baseline_scores,
            margin_threshold=float(threshold),
            valid_mask=candidate_valid,
        )
        pose, _rows = _evaluate_pose(
            name=f"linear_selective_switch_margin_{float(threshold):g}",
            store=store,
            rows=validation_rows,
            scores=resolved,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        identity = _identity_summary(store, validation_rows, resolved)
        threshold_trials.append(
            {
                "margin_threshold": float(threshold),
                "validation_pose": pose,
                "validation_identity": identity,
                "validation_switch_count": int(np.sum(switched[validation_rows])),
                "passes_pose_gate": _pose_gate(pose, baseline_validation_pose),
            }
        )
        resolved_by_threshold.append(resolved)
    eligible = [index for index, trial in enumerate(threshold_trials) if bool(trial["passes_pose_gate"])]
    if eligible:
        chosen_threshold_index = max(
            eligible,
            key=lambda index: (
                float(threshold_trials[index]["validation_pose"]["recall_10cm_5deg"]),
                float(threshold_trials[index]["validation_identity"]["recall_at_1"]),
                -float(threshold_trials[index]["validation_pose"]["median_translation_m_success"]),
                -float(threshold_trials[index]["margin_threshold"]),
            ),
        )
        pose_gate_passed = True
    else:
        chosen_threshold_index = -1
        pose_gate_passed = False

    if chosen_threshold_index >= 0:
        chosen_threshold = float(threshold_trials[chosen_threshold_index]["margin_threshold"])
        selected_columns, resolved_scores, switched, reranker_margins = selective_switch_scores(
            identity_scores,
            baseline_scores,
            margin_threshold=chosen_threshold,
            valid_mask=candidate_valid,
        )
    else:
        chosen_threshold = None
        selected_columns = np.argmax(np.where(candidate_valid, baseline_scores, -np.inf), axis=1)
        resolved_scores = np.full_like(baseline_scores, -np.inf, dtype=np.float32)
        resolved_scores[np.arange(len(selected_columns)), selected_columns] = baseline_scores[
            np.arange(len(selected_columns)), selected_columns
        ]
        switched = np.zeros((len(selected_columns),), dtype=bool)
        reranker_margins = np.zeros((len(selected_columns),), dtype=np.float32)

    pose: dict[str, object] = {
        "validation": {
            "baseline": baseline_validation_pose,
            "selected": (
                baseline_validation_pose
                if chosen_threshold_index < 0
                else threshold_trials[chosen_threshold_index]["validation_pose"]
            ),
        }
    }
    pose_rows: dict[str, object] = {
        "validation": {"baseline": baseline_validation_pose_rows}
    }
    test_rows = split_rows["test"]
    test_baseline_pose, test_baseline_pose_rows = _evaluate_pose(
        name=str(args.baseline_strategy),
        store=store,
        rows=test_rows,
        scores=baseline_scores,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(args.pnp_reprojection_error_px),
        iterations=int(args.pnp_iterations),
    )
    test_selected_pose, test_selected_pose_rows = _evaluate_pose(
        name="linear_selective_switch" if pose_gate_passed else str(args.baseline_strategy),
        store=store,
        rows=test_rows,
        scores=resolved_scores,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(args.pnp_reprojection_error_px),
        iterations=int(args.pnp_iterations),
    )
    pose["test"] = {"baseline": test_baseline_pose, "selected": test_selected_pose}
    pose_rows["test"] = {"baseline": test_baseline_pose_rows, "selected": test_selected_pose_rows}

    artifact_hashes = {
        "probe_arrays": file_sha256_short(probe_path),
        "real_feature_cache": file_sha256_short(Path(args.real_feature_cache)),
        "query_global_cache": file_sha256_short(Path(args.query_global_cache)),
        "projected_landmark_bank": file_sha256_short(Path(args.projected_landmark_bank)),
        "maplet_support_index": file_sha256_short(Path(args.maplet_support_index)),
        "split": file_sha256_short(split_path),
    }
    common_metadata = {
        "stage": "s4_l1_linear_assignment_v1",
        "probe_arrays": str(probe_path),
        "artifact_hashes": artifact_hashes,
        "split": split,
        "training_scope": "train_block_only",
    }
    identity_model = LinearLogitModel(
        coefficients=identity_model_sklearn.coef_[0],
        intercept=float(identity_model_sklearn.intercept_[0]),
        feature_names=identity_feature_names,
        metadata={**common_metadata, "task": "candidate_same_track", "logistic_c": chosen_identity_c},
    )
    no_match_model = LinearLogitModel(
        coefficients=no_match_model_sklearn.coef_[0],
        intercept=float(no_match_model_sklearn.intercept_[0]),
        feature_names=no_match_feature_names,
        metadata={**common_metadata, "task": "correct_track_absent", "logistic_c": chosen_no_match_c},
    )
    identity_model_path = output_dir / "identity_reranker.npz"
    no_match_model_path = output_dir / "no_match_calibrator.npz"
    identity_model.save(identity_model_path)
    no_match_model.save(no_match_model_path)
    predictions_path = output_dir / "predictions.npz"
    np.savez(
        predictions_path,
        identity_scores=identity_scores.astype(np.float32),
        baseline_scores=baseline_scores.astype(np.float32),
        selected_columns=selected_columns.astype(np.int64),
        resolved_scores=resolved_scores.astype(np.float32),
        switched=switched.astype(bool),
        reranker_margins=reranker_margins.astype(np.float32),
        no_match_probabilities=no_match_probabilities.astype(np.float32),
        no_match_labels=no_match_labels.astype(bool),
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps(pose_rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "s4_l1_linear_assignment_and_no_match_calibration",
        "protocol": {
            "split": split,
            "test_used_for_selection": False,
            "identity_confidence_used_as_pnp_weight": False,
            "no_match_hard_filter_enabled": False,
            "pnp_confidence_source": str(args.baseline_strategy),
        },
        "artifact_hashes": artifact_hashes,
        "identity": {
            "chosen_c": chosen_identity_c,
            "trials": identity_trials,
            "metrics": identity_metrics,
            "baseline": baseline_identity,
        },
        "no_match": {
            "chosen_c": chosen_no_match_c,
            "trials": no_match_trials,
            "metrics": no_match_metrics,
            "deployment": "diagnostic_probability_only_until_pose-safe_filtering_is_validated",
        },
        "selective_switch": {
            "baseline_strategy": str(args.baseline_strategy),
            "trials": threshold_trials,
            "pose_gate_passed": bool(pose_gate_passed),
            "chosen_margin_threshold": chosen_threshold,
            "validation_switch_count": int(np.sum(switched[validation_rows])),
            "test_switch_count": int(np.sum(switched[test_rows])),
        },
        "pose": pose,
        "runtime_seconds": float(time.time() - start),
        "limitations": [
            "query nodes are held-out GT SfM observations rather than detector-generated points",
            "the 15-query validation and test cohorts are small and must also be compared on the fixed full90/full182 protocol",
            "no-match probability is not a calibrated pose covariance and is not used for hard filtering",
            "pixel coordinates remain the input observation centers; local measurement is not active",
        ],
        "outputs": {
            "identity_model": str(identity_model_path),
            "no_match_model": str(no_match_model_path),
            "predictions": str(predictions_path),
            "pose_rows": str(pose_rows_path),
            "split": str(split_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
