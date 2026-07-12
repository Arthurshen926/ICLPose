"""Pose-free calibration of whether an RGB measurement offset is beneficial."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.action_calibration import (
    BinaryLinearProbabilityModel,
    _fit_binary_model,
)
from feature_extract.vfm.measurement_v1.pose_free_geometry_verifier import (
    FEATURE_SETS,
    _choose_c_value,
    _choose_verification_threshold,
    _feature_matrix,
    _fit_platt,
    _input_manifest,
    _logit,
    _metrics,
    _oof_all_example_probabilities,
    _sigmoid,
    _threshold_metrics,
    build_pose_free_geometry_examples,
)


UPDATE_FEATURE_SET = "measurement_plus_support"
UPDATE_FEATURE_NAMES = FEATURE_SETS[UPDATE_FEATURE_SET]


@dataclass(frozen=True)
class PoseFreeUpdateVerifier:
    base_model: BinaryLinearProbabilityModel
    calibration_slope: float
    calibration_intercept: float
    update_threshold: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        probability = self.base_model.predict(features)
        return _sigmoid(
            float(self.calibration_slope) * _logit(probability)
            + float(self.calibration_intercept)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "pose_free_measurement_update_verifier_v1",
            "feature_set": UPDATE_FEATURE_SET,
            "feature_names": list(UPDATE_FEATURE_NAMES),
            "base_model": self.base_model.to_dict(),
            "calibration_slope": float(self.calibration_slope),
            "calibration_intercept": float(self.calibration_intercept),
            "update_threshold": float(self.update_threshold),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PoseFreeUpdateVerifier":
        if payload.get("format") != "pose_free_measurement_update_verifier_v1":
            raise ValueError("unsupported pose-free update verifier format")
        if str(payload.get("feature_set", "")) != UPDATE_FEATURE_SET or tuple(
            str(value) for value in payload.get("feature_names", ())
        ) != UPDATE_FEATURE_NAMES:
            raise ValueError("pose-free update verifier feature schema mismatch")
        return cls(
            base_model=BinaryLinearProbabilityModel.from_dict(payload["base_model"]),
            calibration_slope=float(payload["calibration_slope"]),
            calibration_intercept=float(payload["calibration_intercept"]),
            update_threshold=float(payload["update_threshold"]),
        )


def _target_arrays(
    examples: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(
        [
            index
            for index, example in enumerate(examples)
            if bool(example["has_valid_geometry_target"])
        ],
        dtype=np.int64,
    )
    if not len(indices):
        raise ValueError("pose-free update calibration has no valid GT projections")
    features = _feature_matrix(examples, UPDATE_FEATURE_NAMES)[indices]
    labels = np.asarray(
        [bool(examples[index]["target_update_beneficial"]) for index in indices],
        dtype=np.int64,
    )
    groups = np.asarray(
        [str(examples[index]["query_id"]) for index in indices], dtype=object
    )
    if len(np.unique(labels)) < 2:
        raise ValueError("pose-free update calibration requires both target classes")
    return indices, features, labels, groups


def _write_predictions(
    path: Path,
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    fields = (
        "policy_row_index",
        "query_id",
        "track_id",
        "update_beneficial_probability",
        "update_approved",
        "center_x",
        "center_y",
        "updated_x",
        "updated_y",
        "mode_updated_x",
        "mode_updated_y",
        "has_valid_geometry_target_TARGET_ONLY",
        "target_update_beneficial_TARGET_ONLY",
        "target_update_safe_TARGET_ONLY",
        "baseline_residual_px_TARGET_ONLY",
        "updated_residual_px_TARGET_ONLY",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example, probability in zip(examples, probabilities):
            writer.writerow(
                {
                    "policy_row_index": int(example["policy_row_index"]),
                    "query_id": str(example["query_id"]),
                    "track_id": int(example["track_id"]),
                    "update_beneficial_probability": float(probability),
                    "update_approved": bool(float(probability) >= float(threshold)),
                    "center_x": float(example["center_x"]),
                    "center_y": float(example["center_y"]),
                    "updated_x": float(example["updated_x"]),
                    "updated_y": float(example["updated_y"]),
                    "mode_updated_x": float(example["mode_updated_x"]),
                    "mode_updated_y": float(example["mode_updated_y"]),
                    "has_valid_geometry_target_TARGET_ONLY": bool(
                        example["has_valid_geometry_target"]
                    ),
                    "target_update_beneficial_TARGET_ONLY": bool(
                        example["target_update_beneficial"]
                    ),
                    "target_update_safe_TARGET_ONLY": bool(
                        example["target_update_safe"]
                    ),
                    "baseline_residual_px_TARGET_ONLY": float(
                        example["baseline_residual_px_TARGET_ONLY"]
                    ),
                    "updated_residual_px_TARGET_ONLY": float(
                        example["updated_residual_px_TARGET_ONLY"]
                    ),
                }
            )


def _update_metrics(
    examples: Sequence[Mapping[str, object]],
    valid_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    selected = probabilities >= float(threshold)
    before = np.asarray(
        [float(examples[index]["baseline_residual_px_TARGET_ONLY"]) for index in valid_indices]
    )
    after = np.asarray(
        [float(examples[index]["updated_residual_px_TARGET_ONLY"]) for index in valid_indices]
    )
    return {
        "confidence": _metrics(
            examples, valid_indices, labels, probabilities
        ),
        "threshold": _threshold_metrics(labels, probabilities, threshold),
        "selected_residual": {
            "count": int(np.sum(selected)),
            "baseline_median_px": (
                None if not np.any(selected) else float(np.median(before[selected]))
            ),
            "updated_median_px": (
                None if not np.any(selected) else float(np.median(after[selected]))
            ),
            "improve_fraction": (
                None
                if not np.any(selected)
                else float(np.mean(after[selected] + 0.1 < before[selected]))
            ),
            "worsen_fraction": (
                None
                if not np.any(selected)
                else float(np.mean(after[selected] > before[selected] + 0.1))
            ),
        },
    }


def fit_pose_free_update_verifier(
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
    minimum_update_precision: float = 0.8,
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
    if train_manifest["diagnostics"]["checkpoint_sha256"] != validation_manifest[
        "diagnostics"
    ]["checkpoint_sha256"]:
        raise ValueError("train/validation measurement checkpoint mismatch")
    if train_manifest["support_selector"]["selector_sha256"] != validation_manifest[
        "support_selector"
    ]["selector_sha256"]:
        raise ValueError("train/validation support selector mismatch")
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
    train_queries = {str(example["query_id"]) for example in train_examples}
    validation_queries = {
        str(example["query_id"]) for example in validation_examples
    }
    if train_queries & validation_queries:
        raise ValueError("train/validation query leakage in update calibration")
    train_valid, train_features, train_labels, train_groups = _target_arrays(
        train_examples
    )
    validation_valid, _validation_features, validation_labels, _groups = (
        _target_arrays(validation_examples)
    )
    selected_c, _valid_oof, c_audit = _choose_c_value(
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
        UPDATE_FEATURE_NAMES,
        train_valid,
        train_labels,
        c_value=float(selected_c),
        fold_count=int(fold_count),
    )
    slope, intercept = _fit_platt(train_oof_base[train_valid], train_labels)
    train_probability = _sigmoid(slope * _logit(train_oof_base) + intercept)
    base_model = _fit_binary_model(
        train_features, train_labels, c_value=float(selected_c)
    )
    provisional = PoseFreeUpdateVerifier(base_model, slope, intercept, 1.0)
    validation_probability = provisional.predict(
        _feature_matrix(validation_examples, UPDATE_FEATURE_NAMES)
    )
    threshold, threshold_audit = _choose_verification_threshold(
        validation_labels,
        validation_probability[validation_valid],
        minimum_precision=float(minimum_update_precision),
        minimum_selected_fraction=float(minimum_selected_fraction),
    )
    verifier = PoseFreeUpdateVerifier(base_model, slope, intercept, threshold)
    train_metrics = _update_metrics(
        train_examples,
        train_valid,
        train_labels,
        train_probability[train_valid],
        threshold,
    )
    validation_metrics = _update_metrics(
        validation_examples,
        validation_valid,
        validation_labels,
        validation_probability[validation_valid],
        threshold,
    )
    train_auroc = float(train_metrics["confidence"]["auroc"] or 0.0)
    validation_auroc = float(validation_metrics["confidence"]["auroc"] or 0.0)
    selected_residual = validation_metrics["selected_residual"]
    promotion_passes = bool(
        validation_auroc >= 0.60
        and abs(train_auroc - validation_auroc) <= 0.10
        and threshold < 1.0
        and float(selected_residual["improve_fraction"] or 0.0) >= 0.75
        and float(selected_residual["worsen_fraction"] or 1.0) <= 0.20
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "pose_free_update_verifier.json"
    model_payload = {
        **verifier.to_dict(),
        "measurement_checkpoint_sha256": train_manifest["diagnostics"][
            "checkpoint_sha256"
        ],
        "support_selector_sha256": train_manifest["support_selector"][
            "selector_sha256"
        ],
    }
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    train_predictions = output / "train_oof_update_predictions.csv"
    validation_predictions = output / "validation_update_predictions.csv"
    _write_predictions(train_predictions, train_examples, train_probability, threshold)
    _write_predictions(
        validation_predictions,
        validation_examples,
        validation_probability,
        threshold,
    )
    summary = {
        "stage": "pose_free_measurement_update_verifier_fit",
        "protocol": {
            "fit_split": "train",
            "regularization_selection": "grouped_train_OOF",
            "probability_calibration": "Platt_scaling_on_grouped_train_OOF",
            "threshold_selection": "validation",
            "GT_pose_residual_improvement_used_as_target_only": True,
            "query_pose_used_as_feature": False,
            "coarse_pose_used_as_feature": False,
            "assignment_score_used_as_feature": False,
            "test_used": False,
            "coordinate_update_applied": False,
        },
        "feature_names": list(UPDATE_FEATURE_NAMES),
        "selected_c_value": float(selected_c),
        "update_threshold": float(threshold),
        "train_oof_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "c_value_audit": c_audit,
        "threshold_audit": threshold_audit,
        "promotion_passes": promotion_passes,
        "inputs": {"train": train_manifest, "validation": validation_manifest},
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "train_predictions": str(train_predictions),
            "train_predictions_sha256": file_sha256_short(train_predictions),
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


def apply_pose_free_update_verifier(
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
    verifier = PoseFreeUpdateVerifier.from_dict(payload)
    if str(payload.get("measurement_checkpoint_sha256", "")) != str(
        manifest["diagnostics"]["checkpoint_sha256"]
    ):
        raise ValueError("pose-free update verifier checkpoint mismatch")
    if str(payload.get("support_selector_sha256", "")) != str(
        manifest["support_selector"]["selector_sha256"]
    ):
        raise ValueError("pose-free update verifier support selector mismatch")
    examples = build_pose_free_geometry_examples(
        source_rows_csv=Path(rows_csv),
        diagnostic_rows_csv=Path(diagnostics_csv),
        measurement_support_selector_rows_csv=Path(support_selector_rows_csv),
    )
    probability = verifier.predict(_feature_matrix(examples, UPDATE_FEATURE_NAMES))
    valid, _features, labels, _groups = _target_arrays(examples)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "update_predictions.csv"
    _write_predictions(predictions, examples, probability, verifier.update_threshold)
    summary = {
        "stage": "pose_free_measurement_update_verifier_apply",
        "protocol": {
            "threshold_search": False,
            "target_fields_used_as_features": False,
            "query_pose_used_as_feature": False,
            "coordinate_update_applied": False,
        },
        "update_threshold": float(verifier.update_threshold),
        "metrics_TARGET_ONLY": _update_metrics(
            examples,
            valid,
            labels,
            probability[valid],
            verifier.update_threshold,
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
