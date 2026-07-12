"""Pose-free calibration of RGB measurement evidence for 2D-3D assignments."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.action_calibration import (
    BinaryLinearProbabilityModel,
    _fit_binary_model,
    _load_diagnostic_manifest,
    _measurement_support_selector_rows,
    build_action_examples,
)


MEASUREMENT_CORE_FEATURE_NAMES: tuple[str, ...] = (
    "measurement_accept_probability",
    "likelihood_confidence",
    "likelihood_peak_probability",
    "likelihood_peak_margin",
    "likelihood_covariance_quality",
    "log_fused_offset_norm",
    "view_offset_agreement_quality",
    "mean_mode_offset_agreement_quality",
    "support_view_top_probability",
    "support_view_concentration",
    "log_support_view_count",
)

SUPPORT_MAP_CONTROL_FEATURE_NAMES: tuple[str, ...] = (
    "log_track_length",
    "support_reprojection_quality",
    "support_view_angle_quality",
    "support_view_top_probability",
    "support_view_concentration",
    "log_support_view_count",
)

MEASUREMENT_PLUS_SUPPORT_FEATURE_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(MEASUREMENT_CORE_FEATURE_NAMES + SUPPORT_MAP_CONTROL_FEATURE_NAMES)
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "measurement_core": MEASUREMENT_CORE_FEATURE_NAMES,
    "support_map_control": SUPPORT_MAP_CONTROL_FEATURE_NAMES,
    "measurement_plus_support": MEASUREMENT_PLUS_SUPPORT_FEATURE_NAMES,
}

_FORBIDDEN_FEATURE_FRAGMENTS: tuple[str, ...] = (
    "target",
    "ground_truth",
    "gt_",
    "pose_",
    "assignment_score",
    "geometry_p",
    "query_reprojection",
    "actual_query_observation",
    "switched_from_baseline",
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _validate_feature_schema(feature_names: Sequence[str]) -> None:
    names = tuple(str(name) for name in feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("pose-free geometry feature names must be non-empty and unique")
    for name in names:
        lowered = name.lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_FEATURE_FRAGMENTS):
            raise ValueError(f"pose-free geometry feature leaks unavailable context: {name}")


for _feature_names in FEATURE_SETS.values():
    _validate_feature_schema(_feature_names)


@dataclass(frozen=True)
class PoseFreeGeometryVerifier:
    feature_set: str
    feature_names: tuple[str, ...]
    base_model: BinaryLinearProbabilityModel
    calibration_slope: float
    calibration_intercept: float
    verification_threshold: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        base_probability = self.base_model.predict(features)
        calibrated_logit = (
            float(self.calibration_slope) * _logit(base_probability)
            + float(self.calibration_intercept)
        )
        return _sigmoid(calibrated_logit)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "pose_free_measurement_geometry_verifier_v1",
            "feature_set": self.feature_set,
            "feature_names": list(self.feature_names),
            "base_model": self.base_model.to_dict(),
            "calibration_slope": float(self.calibration_slope),
            "calibration_intercept": float(self.calibration_intercept),
            "verification_threshold": float(self.verification_threshold),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PoseFreeGeometryVerifier":
        if payload.get("format") != "pose_free_measurement_geometry_verifier_v1":
            raise ValueError("unsupported pose-free geometry verifier format")
        feature_set = str(payload.get("feature_set", ""))
        feature_names = tuple(str(value) for value in payload.get("feature_names", ()))
        if FEATURE_SETS.get(feature_set) != feature_names:
            raise ValueError("pose-free geometry verifier feature schema mismatch")
        _validate_feature_schema(feature_names)
        return cls(
            feature_set=feature_set,
            feature_names=feature_names,
            base_model=BinaryLinearProbabilityModel.from_dict(payload["base_model"]),
            calibration_slope=float(payload["calibration_slope"]),
            calibration_intercept=float(payload["calibration_intercept"]),
            verification_threshold=float(payload["verification_threshold"]),
        )


def build_pose_free_geometry_examples(
    *,
    source_rows_csv: Path,
    diagnostic_rows_csv: Path,
    measurement_support_selector_rows_csv: Path,
) -> list[dict[str, object]]:
    action_examples = build_action_examples(
        source_rows_csv=Path(source_rows_csv),
        diagnostic_rows_csv=Path(diagnostic_rows_csv),
        coarse_pose_context_csv=None,
        measurement_support_selector_rows_csv=Path(
            measurement_support_selector_rows_csv
        ),
    )
    output: list[dict[str, object]] = []
    for example in action_examples:
        source_features = example["features"]
        view_count = max(int(example["support_view_count"]), 1)
        transformed = {
            "measurement_accept_probability": float(
                source_features["measurement_accept_probability"]
            ),
            "likelihood_confidence": float(source_features["likelihood_confidence"]),
            "likelihood_peak_probability": float(
                source_features["likelihood_peak_probability"]
            ),
            "likelihood_peak_margin": float(
                source_features["likelihood_peak_margin"]
            ),
            "likelihood_covariance_quality": float(
                source_features["likelihood_covariance_quality"]
            ),
            "log_fused_offset_norm": math.log1p(
                max(float(source_features["fused_offset_norm"]), 0.0)
            ),
            "view_offset_agreement_quality": 1.0
            / (1.0 + max(float(source_features["view_offset_disagreement"]), 0.0)),
            "mean_mode_offset_agreement_quality": 1.0
            / (
                1.0
                + max(
                    float(source_features["mean_mode_offset_disagreement"]), 0.0
                )
            ),
            "support_view_top_probability": float(
                source_features["support_view_top_probability"]
            ),
            "support_view_concentration": 1.0
            - float(source_features["support_view_normalized_entropy"]),
            "log_support_view_count": math.log1p(float(view_count)),
            "log_track_length": float(source_features["log_track_length"]),
            "support_reprojection_quality": float(
                source_features["support_reprojection_quality"]
            ),
            "support_view_angle_quality": float(
                source_features["support_view_angle_quality"]
            ),
        }
        output.append(
            {
                "policy_row_index": int(example["policy_row_index"]),
                "query_id": str(example["query_id"]),
                "track_id": int(example["track_id"]),
                "center_x": float(example["center_x"]),
                "center_y": float(example["center_y"]),
                "updated_x": float(example["updated_x"]),
                "updated_y": float(example["updated_y"]),
                "mode_updated_x": float(example["mode_updated_x"]),
                "mode_updated_y": float(example["mode_updated_y"]),
                "has_valid_geometry_target": bool(
                    example["has_valid_geometry_target"]
                ),
                "target_geometry_correct_5px": bool(
                    example["target_geometry_correct_5px"]
                ),
                "baseline_residual_px_TARGET_ONLY": float(
                    example["baseline_residual_px"]
                ),
                "updated_residual_px_TARGET_ONLY": float(
                    example["updated_residual_px"]
                ),
                "target_update_beneficial": bool(
                    example["target_update_beneficial"]
                ),
                "target_update_safe": bool(example["target_update_safe"]),
                "features": transformed,
            }
        )
    return output


def _feature_matrix(
    examples: Sequence[Mapping[str, object]], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [float(example["features"][name]) for name in feature_names]
            for example in examples
        ],
        dtype=np.float64,
    )


def _valid_target_arrays(
    examples: Sequence[Mapping[str, object]], feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = np.asarray(
        [
            index
            for index, example in enumerate(examples)
            if bool(example["has_valid_geometry_target"])
        ],
        dtype=np.int64,
    )
    if not len(valid_indices):
        raise ValueError("pose-free geometry calibration has no valid GT projections")
    features = _feature_matrix(examples, feature_names)[valid_indices]
    labels = np.asarray(
        [bool(examples[index]["target_geometry_correct_5px"]) for index in valid_indices],
        dtype=np.int64,
    )
    groups = np.asarray(
        [str(examples[index]["query_id"]) for index in valid_indices], dtype=object
    )
    if len(np.unique(labels)) < 2:
        raise ValueError("pose-free geometry calibration requires both label classes")
    return valid_indices, features, labels, groups


def _fit_platt(base_probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    logits = _logit(base_probabilities).reshape(-1, 1)
    model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=2000, random_state=0)
    model.fit(logits, np.asarray(labels, dtype=np.int64))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _group_metrics(
    examples: Sequence[Mapping[str, object]],
    valid_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    per_query: list[dict[str, object]] = []
    query_ids = np.asarray(
        [str(examples[index]["query_id"]) for index in valid_indices], dtype=object
    )
    for query_id in sorted(set(query_ids.tolist())):
        mask = query_ids == query_id
        metrics = confidence_metrics(labels[mask], probabilities[mask])
        per_query.append({"query_id": query_id, **metrics})
    finite_auroc = np.asarray(
        [float(row["auroc"]) for row in per_query if row["auroc"] is not None],
        dtype=np.float64,
    )
    return {
        "query_count": len(per_query),
        "query_count_with_both_classes": int(len(finite_auroc)),
        "mean_query_auroc": (
            None if not len(finite_auroc) else float(np.mean(finite_auroc))
        ),
        "median_query_auroc": (
            None if not len(finite_auroc) else float(np.median(finite_auroc))
        ),
    }


def _metrics(
    examples: Sequence[Mapping[str, object]],
    valid_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    return {
        **confidence_metrics(labels, probabilities),
        "grouped": _group_metrics(
            examples, valid_indices, labels, probabilities
        ),
    }


def _oof_base_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    c_value: float,
    fold_count: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    splits = min(int(fold_count), int(len(unique_groups)))
    if splits < 2:
        raise ValueError("grouped calibration requires at least two query groups")
    output = np.full((len(labels),), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=splits)
    for fit_indices, holdout_indices in splitter.split(features, labels, groups):
        model = _fit_binary_model(
            features[fit_indices], labels[fit_indices], c_value=float(c_value)
        )
        output[holdout_indices] = model.predict(features[holdout_indices])
    if not np.all(np.isfinite(output)):
        raise RuntimeError("grouped calibration did not produce every OOF probability")
    return output


def _oof_all_example_probabilities(
    examples: Sequence[Mapping[str, object]],
    feature_names: Sequence[str],
    valid_indices: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    fold_count: int,
) -> np.ndarray:
    all_features = _feature_matrix(examples, feature_names)
    all_groups = np.asarray(
        [str(example["query_id"]) for example in examples], dtype=object
    )
    valid_groups = all_groups[valid_indices]
    unique_groups = np.unique(valid_groups)
    splits = min(int(fold_count), int(len(unique_groups)))
    if splits < 2:
        raise ValueError("grouped calibration requires at least two query groups")
    output = np.full((len(examples),), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=splits)
    valid_features = all_features[valid_indices]
    for fit_indices, holdout_indices in splitter.split(
        valid_features, labels, valid_groups
    ):
        model = _fit_binary_model(
            valid_features[fit_indices],
            labels[fit_indices],
            c_value=float(c_value),
        )
        holdout_queries = set(valid_groups[holdout_indices].tolist())
        all_holdout = np.asarray(
            [group in holdout_queries for group in all_groups], dtype=bool
        )
        output[all_holdout] = model.predict(all_features[all_holdout])
    if not np.all(np.isfinite(output)):
        missing_queries = sorted(set(all_groups[~np.isfinite(output)].tolist()))
        raise RuntimeError(
            f"grouped calibration did not cover all example queries: {missing_queries[:5]}"
        )
    return output


def _choose_c_value(
    examples: Sequence[Mapping[str, object]],
    valid_indices: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    c_values: Sequence[float],
    fold_count: int,
) -> tuple[float, np.ndarray, list[dict[str, object]]]:
    candidates: list[tuple[tuple[float, ...], float, np.ndarray, dict[str, object]]] = []
    for c_value in c_values:
        if float(c_value) <= 0.0:
            raise ValueError("all C values must be positive")
        probabilities = _oof_base_probabilities(
            features,
            labels,
            groups,
            c_value=float(c_value),
            fold_count=int(fold_count),
        )
        metrics = _metrics(examples, valid_indices, labels, probabilities)
        auprc = float(metrics["auprc"] or 0.0)
        auroc = float(metrics["auroc"] or 0.0)
        query_auroc = float(metrics["grouped"]["mean_query_auroc"] or 0.0)
        key = (-query_auroc, -auprc, -auroc, float(metrics["brier"]), float(c_value))
        candidates.append((key, float(c_value), probabilities, metrics))
    _key, selected_c, probabilities, _metrics_value = min(
        candidates, key=lambda item: item[0]
    )
    audit = [
        {"c_value": c_value, "metrics": metrics}
        for _candidate_key, c_value, _probabilities, metrics in candidates
    ]
    return selected_c, probabilities, audit


def _threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    selected = np.asarray(probabilities, dtype=np.float64) >= float(threshold)
    selected_count = int(np.sum(selected))
    positive_count = int(np.sum(labels == 1))
    true_positive = int(np.sum(selected & (labels == 1)))
    return {
        "threshold": float(threshold),
        "selected_count": selected_count,
        "selected_fraction": float(np.mean(selected)),
        "precision": (
            None if selected_count == 0 else float(true_positive / selected_count)
        ),
        "recall": (
            None if positive_count == 0 else float(true_positive / positive_count)
        ),
    }


def _choose_verification_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_precision: float,
    minimum_selected_fraction: float,
) -> tuple[float, list[dict[str, object]]]:
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = [
        _threshold_metrics(labels, probabilities, float(threshold))
        for threshold in thresholds
    ]
    eligible = [
        row
        for row in rows
        if row["precision"] is not None
        and float(row["precision"]) >= float(minimum_precision)
        and float(row["selected_fraction"]) >= float(minimum_selected_fraction)
    ]
    if not eligible:
        return 1.0, rows
    selected = max(
        eligible,
        key=lambda row: (
            float(row["recall"] or 0.0),
            float(row["precision"] or 0.0),
            -float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), rows


def _fit_feature_set(
    train_examples: Sequence[Mapping[str, object]],
    validation_examples: Sequence[Mapping[str, object]],
    *,
    feature_set: str,
    c_values: Sequence[float],
    fold_count: int,
    minimum_verification_precision: float,
    minimum_selected_fraction: float,
) -> tuple[PoseFreeGeometryVerifier, dict[str, object], np.ndarray, np.ndarray]:
    feature_names = FEATURE_SETS[feature_set]
    train_valid, train_features, train_labels, train_groups = _valid_target_arrays(
        train_examples, feature_names
    )
    validation_valid, validation_features, validation_labels, _validation_groups = (
        _valid_target_arrays(validation_examples, feature_names)
    )
    selected_c, _selected_valid_oof_base, c_audit = _choose_c_value(
        train_examples,
        train_valid,
        train_features,
        train_labels,
        train_groups,
        c_values=c_values,
        fold_count=int(fold_count),
    )
    train_oof_base = _oof_all_example_probabilities(
        train_examples,
        feature_names,
        train_valid,
        train_labels,
        c_value=float(selected_c),
        fold_count=int(fold_count),
    )
    slope, intercept = _fit_platt(train_oof_base[train_valid], train_labels)
    train_oof_probability = _sigmoid(slope * _logit(train_oof_base) + intercept)
    base_model = _fit_binary_model(
        train_features, train_labels, c_value=float(selected_c)
    )
    provisional = PoseFreeGeometryVerifier(
        feature_set=feature_set,
        feature_names=feature_names,
        base_model=base_model,
        calibration_slope=slope,
        calibration_intercept=intercept,
        verification_threshold=1.0,
    )
    validation_all_features = _feature_matrix(validation_examples, feature_names)
    validation_probability = provisional.predict(validation_all_features)
    threshold, threshold_audit = _choose_verification_threshold(
        validation_labels,
        validation_probability[validation_valid],
        minimum_precision=float(minimum_verification_precision),
        minimum_selected_fraction=float(minimum_selected_fraction),
    )
    verifier = PoseFreeGeometryVerifier(
        feature_set=feature_set,
        feature_names=feature_names,
        base_model=base_model,
        calibration_slope=slope,
        calibration_intercept=intercept,
        verification_threshold=threshold,
    )
    summary = {
        "feature_set": feature_set,
        "feature_names": list(feature_names),
        "selected_c_value": float(selected_c),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "train_oof_metrics": _metrics(
            train_examples,
            train_valid,
            train_labels,
            train_oof_probability[train_valid],
        ),
        "validation_metrics": _metrics(
            validation_examples,
            validation_valid,
            validation_labels,
            validation_probability[validation_valid],
        ),
        "verification_threshold": float(threshold),
        "validation_threshold_metrics": _threshold_metrics(
            validation_labels, validation_probability[validation_valid], threshold
        ),
        "c_value_audit": c_audit,
        "threshold_audit": threshold_audit,
    }
    return verifier, summary, train_oof_probability, validation_probability


def _write_prediction_rows(
    path: Path,
    examples: Sequence[Mapping[str, object]],
    probabilities_by_feature_set: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> None:
    fields = [
        "policy_row_index",
        "query_id",
        "track_id",
        *[f"{name}_geometry_probability" for name in probabilities_by_feature_set],
        *[f"{name}_verified" for name in probabilities_by_feature_set],
        "has_valid_geometry_target_TARGET_ONLY",
        "target_geometry_correct_5px_TARGET_ONLY",
        "baseline_residual_px_TARGET_ONLY",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, example in enumerate(examples):
            row: dict[str, object] = {
                "policy_row_index": int(example["policy_row_index"]),
                "query_id": str(example["query_id"]),
                "track_id": int(example["track_id"]),
                "has_valid_geometry_target_TARGET_ONLY": bool(
                    example["has_valid_geometry_target"]
                ),
                "target_geometry_correct_5px_TARGET_ONLY": bool(
                    example["target_geometry_correct_5px"]
                ),
                "baseline_residual_px_TARGET_ONLY": float(
                    example["baseline_residual_px_TARGET_ONLY"]
                ),
            }
            for name, probabilities in probabilities_by_feature_set.items():
                probability = float(probabilities[index])
                row[f"{name}_geometry_probability"] = (
                    "" if not math.isfinite(probability) else probability
                )
                row[f"{name}_verified"] = bool(
                    math.isfinite(probability)
                    and probability >= float(thresholds[name])
                )
            writer.writerow(row)


def _input_manifest(
    *,
    source_rows_csv: Path,
    diagnostic_rows_csv: Path,
    support_selector_rows_csv: Path,
) -> dict[str, object]:
    diagnostic = _load_diagnostic_manifest(
        Path(diagnostic_rows_csv),
        expected_rows_csv=Path(source_rows_csv),
        required=True,
    )
    _selector_rows, selector = _measurement_support_selector_rows(
        Path(support_selector_rows_csv)
    )
    if selector is None:
        raise ValueError("pose-free geometry verifier requires support selector rows")
    return {
        "source_rows_csv": str(source_rows_csv),
        "source_rows_sha256": file_sha256_short(Path(source_rows_csv)),
        "diagnostics": diagnostic,
        "support_selector": selector,
    }


def fit_pose_free_geometry_verifier(
    *,
    train_rows_csv: Path,
    train_diagnostics_csv: Path,
    train_support_selector_rows_csv: Path,
    validation_rows_csv: Path,
    validation_diagnostics_csv: Path,
    validation_support_selector_rows_csv: Path,
    output_dir: Path,
    c_values: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0),
    fold_count: int = 5,
    minimum_verification_precision: float = 0.8,
    minimum_selected_fraction: float = 0.05,
) -> dict[str, object]:
    train_manifest = _input_manifest(
        source_rows_csv=Path(train_rows_csv),
        diagnostic_rows_csv=Path(train_diagnostics_csv),
        support_selector_rows_csv=Path(train_support_selector_rows_csv),
    )
    validation_manifest = _input_manifest(
        source_rows_csv=Path(validation_rows_csv),
        diagnostic_rows_csv=Path(validation_diagnostics_csv),
        support_selector_rows_csv=Path(validation_support_selector_rows_csv),
    )
    train_checkpoint = train_manifest["diagnostics"]["checkpoint_sha256"]
    validation_checkpoint = validation_manifest["diagnostics"]["checkpoint_sha256"]
    if train_checkpoint != validation_checkpoint:
        raise ValueError("train and validation measurement checkpoint mismatch")
    train_selector = train_manifest["support_selector"]["selector_sha256"]
    validation_selector = validation_manifest["support_selector"]["selector_sha256"]
    if train_selector != validation_selector:
        raise ValueError("train and validation support selector mismatch")

    train_examples = build_pose_free_geometry_examples(
        source_rows_csv=Path(train_rows_csv),
        diagnostic_rows_csv=Path(train_diagnostics_csv),
        measurement_support_selector_rows_csv=Path(train_support_selector_rows_csv),
    )
    validation_examples = build_pose_free_geometry_examples(
        source_rows_csv=Path(validation_rows_csv),
        diagnostic_rows_csv=Path(validation_diagnostics_csv),
        measurement_support_selector_rows_csv=Path(
            validation_support_selector_rows_csv
        ),
    )
    train_query_ids = {str(example["query_id"]) for example in train_examples}
    validation_query_ids = {
        str(example["query_id"]) for example in validation_examples
    }
    overlap = sorted(train_query_ids & validation_query_ids)
    if overlap:
        raise ValueError(f"train/validation query leakage: {overlap[:5]}")

    verifiers: dict[str, PoseFreeGeometryVerifier] = {}
    feature_summaries: dict[str, dict[str, object]] = {}
    train_probabilities: dict[str, np.ndarray] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for feature_set in FEATURE_SETS:
        verifier, feature_summary, train_probability, validation_probability = (
            _fit_feature_set(
                train_examples,
                validation_examples,
                feature_set=feature_set,
                c_values=c_values,
                fold_count=int(fold_count),
                minimum_verification_precision=float(
                    minimum_verification_precision
                ),
                minimum_selected_fraction=float(minimum_selected_fraction),
            )
        )
        verifiers[feature_set] = verifier
        feature_summaries[feature_set] = feature_summary
        train_probabilities[feature_set] = train_probability
        validation_probabilities[feature_set] = validation_probability

    core_metrics = feature_summaries["measurement_core"]["validation_metrics"]
    core_prior = float(core_metrics["positive_prior"])
    core_auprc = float(core_metrics["auprc"] or 0.0)
    core_train_auroc = float(
        feature_summaries["measurement_core"]["train_oof_metrics"]["auroc"]
        or 0.0
    )
    core_validation_auroc = float(core_metrics["auroc"] or 0.0)
    combined_metrics = feature_summaries["measurement_plus_support"][
        "validation_metrics"
    ]
    promotion_passes = bool(
        core_validation_auroc >= 0.65
        and core_auprc >= core_prior + 0.05
        and abs(core_train_auroc - core_validation_auroc) <= 0.10
        and float(combined_metrics["auroc"] or 0.0) >= core_validation_auroc - 0.02
        and verifiers["measurement_plus_support"].verification_threshold < 1.0
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_paths: dict[str, Path] = {}
    for feature_set, verifier in verifiers.items():
        path = output / f"{feature_set}_verifier.json"
        payload = {
            **verifier.to_dict(),
            "measurement_checkpoint_sha256": train_checkpoint,
            "support_selector_sha256": train_selector,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        model_paths[feature_set] = path
    train_predictions = output / "train_oof_predictions.csv"
    validation_predictions = output / "validation_predictions.csv"
    thresholds = {
        name: verifier.verification_threshold for name, verifier in verifiers.items()
    }
    _write_prediction_rows(
        train_predictions, train_examples, train_probabilities, thresholds
    )
    _write_prediction_rows(
        validation_predictions,
        validation_examples,
        validation_probabilities,
        thresholds,
    )
    summary = {
        "stage": "pose_free_measurement_geometry_verifier_fit",
        "protocol": {
            "fit_split": "train",
            "regularization_selection": "grouped_train_OOF",
            "probability_calibration": "Platt_scaling_on_grouped_train_OOF",
            "verification_threshold_selection": "validation",
            "test_used": False,
            "query_pose_used_as_feature": False,
            "coarse_pose_used_as_feature": False,
            "assignment_score_used_as_feature": False,
            "GT_pose_residual_used_as_target_only": True,
            "coordinate_update_enabled": False,
            "render": False,
        },
        "feature_sets": feature_summaries,
        "promotion_gate": {
            "passes": promotion_passes,
            "minimum_measurement_core_validation_auroc": 0.65,
            "minimum_measurement_core_auprc_gain_over_prior": 0.05,
            "maximum_train_validation_auroc_gap": 0.10,
            "maximum_combined_auroc_regression": 0.02,
            "requires_nonempty_combined_verification_set": True,
        },
        "inputs": {"train": train_manifest, "validation": validation_manifest},
        "outputs": {
            "models": {
                name: {"path": str(path), "sha256": file_sha256_short(path)}
                for name, path in model_paths.items()
            },
            "train_oof_predictions": str(train_predictions),
            "train_oof_predictions_sha256": file_sha256_short(train_predictions),
            "validation_predictions": str(validation_predictions),
            "validation_predictions_sha256": file_sha256_short(
                validation_predictions
            ),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def apply_pose_free_geometry_verifier(
    *,
    rows_csv: Path,
    diagnostics_csv: Path,
    support_selector_rows_csv: Path,
    verifier_json: Path,
    output_dir: Path,
) -> dict[str, object]:
    manifest = _input_manifest(
        source_rows_csv=Path(rows_csv),
        diagnostic_rows_csv=Path(diagnostics_csv),
        support_selector_rows_csv=Path(support_selector_rows_csv),
    )
    payload = json.loads(Path(verifier_json).read_text())
    verifier = PoseFreeGeometryVerifier.from_dict(payload)
    checkpoint = manifest["diagnostics"]["checkpoint_sha256"]
    selector = manifest["support_selector"]["selector_sha256"]
    if str(payload.get("measurement_checkpoint_sha256", "")) != str(checkpoint):
        raise ValueError("pose-free geometry verifier measurement checkpoint mismatch")
    if str(payload.get("support_selector_sha256", "")) != str(selector):
        raise ValueError("pose-free geometry verifier support selector mismatch")
    examples = build_pose_free_geometry_examples(
        source_rows_csv=Path(rows_csv),
        diagnostic_rows_csv=Path(diagnostics_csv),
        measurement_support_selector_rows_csv=Path(support_selector_rows_csv),
    )
    features = _feature_matrix(examples, verifier.feature_names)
    probabilities = verifier.predict(features)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "geometry_predictions.csv"
    _write_prediction_rows(
        predictions,
        examples,
        {verifier.feature_set: probabilities},
        {verifier.feature_set: verifier.verification_threshold},
    )
    valid_indices, _valid_features, labels, _groups = _valid_target_arrays(
        examples, verifier.feature_names
    )
    summary = {
        "stage": "pose_free_measurement_geometry_verifier_apply",
        "protocol": {
            "threshold_search": False,
            "target_fields_used_as_features": False,
            "query_pose_used_as_feature": False,
            "coarse_pose_used_as_feature": False,
            "coordinate_update_enabled": False,
        },
        "feature_set": verifier.feature_set,
        "feature_names": list(verifier.feature_names),
        "verification_threshold": float(verifier.verification_threshold),
        "metrics_TARGET_ONLY": _metrics(
            examples, valid_indices, labels, probabilities[valid_indices]
        ),
        "verification_metrics_TARGET_ONLY": _threshold_metrics(
            labels,
            probabilities[valid_indices],
            verifier.verification_threshold,
        ),
        "inputs": {
            "data": manifest,
            "verifier": str(verifier_json),
            "verifier_sha256": file_sha256_short(Path(verifier_json)),
        },
        "outputs": {
            "predictions": str(predictions),
            "predictions_sha256": file_sha256_short(predictions),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
