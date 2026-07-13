"""Calibrate candidate-specific RGB coordinate updates against true geometry."""

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
from feature_extract.vfm.measurement_v1.candidate_geometry_verifier import (
    CANDIDATE_GEOMETRY_FEATURE_NAMES,
    CandidateGeometryVerifier,
    build_candidate_geometry_examples,
)


CANDIDATE_UPDATE_FEATURE_NAMES = (
    *CANDIDATE_GEOMETRY_FEATURE_NAMES,
    "calibrated_geometry_probability",
    "log_refined_offset_norm",
    "weighted_dustbin_probability",
    "weighted_likelihood_entropy",
    "weighted_measurement_gate_probability",
    "weighted_predicted_improvement",
    "direct_offset_agreement_quality",
)


@dataclass(frozen=True)
class CandidateUpdateVerifier:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    update_threshold: float
    minimum_baseline_residual_px: float
    maximum_baseline_residual_px: float
    minimum_improvement_px: float
    measurement_checkpoint_sha256: str
    candidate_geometry_verifier_sha256: str

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        if self.feature_names != CANDIDATE_UPDATE_FEATURE_NAMES:
            raise ValueError("candidate update feature schema mismatch")
        if any(
            len(values) != count
            for values in (self.feature_mean, self.feature_scale, self.coefficients)
        ):
            raise ValueError("candidate update model vector lengths differ")
        if any(float(value) <= 0.0 for value in self.feature_scale):
            raise ValueError("candidate update feature scales must be positive")
        if not 0.0 <= float(self.update_threshold) <= 1.0:
            raise ValueError("candidate update threshold must be in [0, 1]")
        if not 0.0 <= float(self.minimum_baseline_residual_px) < float(
            self.maximum_baseline_residual_px
        ):
            raise ValueError("candidate update target residual interval is invalid")
        if float(self.minimum_improvement_px) < 0.0:
            raise ValueError("minimum candidate update improvement must be non-negative")
        if len(self.measurement_checkpoint_sha256) < 8:
            raise ValueError("candidate update verifier lacks measurement checkpoint identity")
        if len(self.candidate_geometry_verifier_sha256) < 8:
            raise ValueError("candidate update verifier lacks geometry verifier identity")

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(
            -1, len(self.feature_names)
        )
        standardized = (
            values - np.asarray(self.feature_mean, dtype=np.float64)[None]
        ) / np.asarray(self.feature_scale, dtype=np.float64)[None]
        base_logit = (
            standardized @ np.asarray(self.coefficients, dtype=np.float64)
            + float(self.intercept)
        )
        calibrated = (
            float(self.calibration_slope) * base_logit
            + float(self.calibration_intercept)
        )
        output = np.empty_like(calibrated)
        positive = calibrated >= 0.0
        output[positive] = 1.0 / (1.0 + np.exp(-calibrated[positive]))
        exp_values = np.exp(calibrated[~positive])
        output[~positive] = exp_values / (1.0 + exp_values)
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "candidate_coordinate_update_verifier_v1",
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "calibration_slope": float(self.calibration_slope),
            "calibration_intercept": float(self.calibration_intercept),
            "update_threshold": float(self.update_threshold),
            "minimum_baseline_residual_px": float(
                self.minimum_baseline_residual_px
            ),
            "maximum_baseline_residual_px": float(
                self.maximum_baseline_residual_px
            ),
            "minimum_improvement_px": float(self.minimum_improvement_px),
            "measurement_checkpoint_sha256": self.measurement_checkpoint_sha256,
            "candidate_geometry_verifier_sha256": (
                self.candidate_geometry_verifier_sha256
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CandidateUpdateVerifier":
        if payload.get("format") != "candidate_coordinate_update_verifier_v1":
            raise ValueError("unsupported candidate coordinate update verifier format")
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            calibration_slope=float(payload["calibration_slope"]),
            calibration_intercept=float(payload["calibration_intercept"]),
            update_threshold=float(payload["update_threshold"]),
            minimum_baseline_residual_px=float(
                payload["minimum_baseline_residual_px"]
            ),
            maximum_baseline_residual_px=float(
                payload["maximum_baseline_residual_px"]
            ),
            minimum_improvement_px=float(payload["minimum_improvement_px"]),
            measurement_checkpoint_sha256=str(
                payload["measurement_checkpoint_sha256"]
            ),
            candidate_geometry_verifier_sha256=str(
                payload["candidate_geometry_verifier_sha256"]
            ),
        )


def _float(row: Mapping[str, object], key: str, *, default: float | None = None) -> float:
    text = str(row.get(key, "")).strip()
    if text:
        return float(text)
    if default is None:
        raise ValueError(f"candidate update diagnostic row lacks {key}")
    return float(default)


def _consistent_float(
    rows: Sequence[Mapping[str, object]], key: str, *, atol: float = 1e-5
) -> float:
    values = np.asarray([_float(row, key) for row in rows], dtype=np.float64)
    if not np.allclose(values, values[:1], rtol=0.0, atol=float(atol)):
        raise ValueError(f"candidate support views disagree on {key}")
    return float(values[0])


def _geometry_feature_matrix(
    examples: Sequence[Mapping[str, object]],
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(example["features"][name])
                for name in CANDIDATE_GEOMETRY_FEATURE_NAMES
            ]
            for example in examples
        ],
        dtype=np.float64,
    )


def _load_train_oof_geometry_probabilities(
    *,
    path: Path,
    geometry_summary: Mapping[str, object],
    diagnostic_rows_csv: Path,
) -> dict[str, float]:
    outputs = geometry_summary.get("outputs", {})
    inputs = geometry_summary.get("inputs", {})
    if str(geometry_summary.get("stage", "")) != "candidate_geometry_verifier_fit":
        raise ValueError("candidate geometry summary is not a fit artifact")
    if str(outputs.get("train_oof_probabilities_sha256", "")) != file_sha256_short(
        Path(path)
    ):
        raise ValueError("candidate geometry train OOF probability hash mismatch")
    if str(inputs.get("train_diagnostic_rows_sha256", "")) != file_sha256_short(
        Path(diagnostic_rows_csv)
    ):
        raise ValueError("candidate geometry OOF diagnostics differ from update fit")
    with Path(path).open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    probabilities: dict[str, float] = {}
    for row in rows:
        identity = str(row.get("candidate_identity_key", "")).strip()
        if not identity or identity in probabilities:
            raise ValueError("candidate geometry OOF identities must be unique")
        probability = _float(row, "geometry_probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("candidate geometry OOF probability is outside [0, 1]")
        probabilities[identity] = probability
    return probabilities


def build_candidate_update_examples(
    diagnostic_rows_csv: Path,
    *,
    calibrated_geometry_probabilities: Mapping[str, float],
    minimum_baseline_residual_px: float = 1.0,
    maximum_baseline_residual_px: float = 5.0,
    minimum_improvement_px: float = 0.1,
) -> list[dict[str, object]]:
    """Aggregate support views without using pose or target values as features."""

    geometry_examples = build_candidate_geometry_examples(Path(diagnostic_rows_csv))
    geometry_by_identity = {
        str(example["candidate_identity_key"]): example
        for example in geometry_examples
    }
    with Path(diagnostic_rows_csv).open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        identity = str(row.get("candidate_identity_key", "")).strip()
        if not identity:
            raise ValueError("candidate update diagnostic row lacks identity")
        grouped.setdefault(identity, []).append(row)
    if set(grouped) != set(geometry_by_identity) or set(grouped) != set(
        calibrated_geometry_probabilities
    ):
        raise ValueError("candidate update and geometry identities differ")

    output: list[dict[str, object]] = []
    for identity, views in grouped.items():
        first = views[0]
        geometry = geometry_by_identity[identity]
        support = np.asarray(
            [_float(row, "support_view_probability", default=1.0) for row in views],
            dtype=np.float64,
        )
        dustbin = np.asarray(
            [_float(row, "dustbin_probability") for row in views],
            dtype=np.float64,
        )
        if np.any(support < 0.0) or np.any((dustbin < 0.0) | (dustbin > 1.0)):
            raise ValueError("candidate update support probability is invalid")
        weights = support * (1.0 - dustbin)
        if float(np.sum(weights)) <= 1e-12:
            weights = support.copy()
        if float(np.sum(weights)) <= 1e-12:
            weights = np.ones((len(views),), dtype=np.float64)
        weights /= float(np.sum(weights))

        def weighted(key: str, *, default: float | None = None) -> float:
            values = np.asarray(
                [_float(row, key, default=default) for row in views],
                dtype=np.float64,
            )
            return float(np.sum(weights * values))

        center_xy = np.asarray(
            [
                _consistent_float(views, "center_x"),
                _consistent_float(views, "center_y"),
            ],
            dtype=np.float64,
        )
        refined_xy = np.asarray(
            [weighted("query_pred_x"), weighted("query_pred_y")], dtype=np.float64
        )
        direct_xy = np.asarray(
            [weighted("query_direct_x"), weighted("query_direct_y")],
            dtype=np.float64,
        )
        geometry_probability = float(calibrated_geometry_probabilities[identity])
        if not 0.0 <= geometry_probability <= 1.0:
            raise ValueError("calibrated candidate geometry probability is invalid")
        features = {
            **{
                name: float(geometry["features"][name])
                for name in CANDIDATE_GEOMETRY_FEATURE_NAMES
            },
            "calibrated_geometry_probability": geometry_probability,
            "log_refined_offset_norm": math.log1p(
                float(np.linalg.norm(refined_xy - center_xy))
            ),
            "weighted_dustbin_probability": weighted("dustbin_probability"),
            "weighted_likelihood_entropy": weighted(
                "likelihood_normalized_entropy"
            ),
            "weighted_measurement_gate_probability": weighted(
                "measurement_gate_probability"
            ),
            "weighted_predicted_improvement": weighted(
                "predicted_improvement_px"
            ),
            "direct_offset_agreement_quality": 1.0
            / (1.0 + float(np.linalg.norm(refined_xy - direct_xy))),
        }
        values = np.asarray(
            [features[name] for name in CANDIDATE_UPDATE_FEATURE_NAMES],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(refined_xy)):
            raise ValueError("candidate update features and coordinates must be finite")

        target_text = str(first.get("target_gt_projected_x", "")).strip()
        target_y_text = str(first.get("target_gt_projected_y", "")).strip()
        has_target = bool(target_text and target_y_text)
        baseline_residual = updated_residual = float("nan")
        safe_beneficial = False
        if has_target:
            target_xy = np.asarray(
                [
                    _consistent_float(views, "target_gt_projected_x"),
                    _consistent_float(views, "target_gt_projected_y"),
                ],
                dtype=np.float64,
            )
            baseline_residual = float(np.linalg.norm(center_xy - target_xy))
            updated_residual = float(np.linalg.norm(refined_xy - target_xy))
            safe_beneficial = bool(
                baseline_residual > float(minimum_baseline_residual_px)
                and baseline_residual <= float(maximum_baseline_residual_px)
                and updated_residual + float(minimum_improvement_px)
                < baseline_residual
            )
        output.append(
            {
                "candidate_identity_key": identity,
                "query_id": str(geometry["query_id"]),
                "source_query_row": int(geometry["source_query_row"]),
                "candidate_measurement_rank": int(
                    geometry["candidate_measurement_rank"]
                ),
                "track_id": int(geometry["track_id"]),
                "prototype_id": int(geometry["prototype_id"]),
                "center_xy": center_xy,
                "refined_xy": refined_xy,
                "measurement_checkpoint_sha256": str(
                    geometry["measurement_checkpoint_sha256"]
                ),
                "has_target": has_target,
                "target_safe_update_beneficial": safe_beneficial,
                "baseline_residual_px_TARGET_ONLY": baseline_residual,
                "updated_residual_px_TARGET_ONLY": updated_residual,
                "features": features,
            }
        )
    return output


def _matrix(examples: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [
            [float(example["features"][name]) for name in CANDIDATE_UPDATE_FEATURE_NAMES]
            for example in examples
        ],
        dtype=np.float64,
    )


def _fit_base(
    features: np.ndarray, labels: np.ndarray, *, c_value: float
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = LogisticRegression(
        C=float(c_value), solver="lbfgs", max_iter=2000, random_state=0
    )
    model.fit((features - mean[None]) / scale[None], labels)
    return mean, scale, model


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _fit_calibration(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    model = LogisticRegression(
        C=1000.0, solver="lbfgs", max_iter=2000, random_state=0
    )
    model.fit(_logit(probabilities).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _threshold_at_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_precision: float,
    minimum_selected_fraction: float,
) -> tuple[float, dict[str, object]]:
    order = np.argsort(-probabilities, kind="stable")
    precision = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    minimum_count = max(1, int(math.ceil(len(labels) * minimum_selected_fraction)))
    valid = np.flatnonzero(
        (precision >= float(target_precision))
        & (np.arange(1, len(labels) + 1) >= minimum_count)
    )
    if not len(valid):
        return 1.0, {
            "selected_count": 0,
            "selected_fraction": 0.0,
            "precision": None,
            "target_precision": float(target_precision),
            "minimum_selected_fraction": float(minimum_selected_fraction),
        }
    end = int(valid[-1])
    return float(probabilities[order[end]]), {
        "selected_count": int(end + 1),
        "selected_fraction": float((end + 1) / len(labels)),
        "precision": float(precision[end]),
        "target_precision": float(target_precision),
        "minimum_selected_fraction": float(minimum_selected_fraction),
    }


def _evaluate(
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    valid = np.asarray([bool(example["has_target"]) for example in examples])
    if not np.any(valid):
        return {"target_count": 0}
    labels = np.asarray(
        [bool(example["target_safe_update_beneficial"]) for example in examples],
        dtype=bool,
    )[valid]
    values = np.asarray(probabilities, dtype=np.float64)[valid]
    selected = values >= float(threshold)
    baseline = np.asarray(
        [float(example["baseline_residual_px_TARGET_ONLY"]) for example in examples]
    )[valid]
    updated = np.asarray(
        [float(example["updated_residual_px_TARGET_ONLY"]) for example in examples]
    )[valid]
    report = confidence_metrics(labels, values)
    return {
        **report,
        "target_count": int(np.sum(valid)),
        "positive_count": int(np.sum(labels)),
        "positive_prior": float(np.mean(labels)),
        "selected_count": int(np.sum(selected)),
        "selected_fraction": float(np.mean(selected)),
        "selected_safe_update_precision": (
            None if not np.any(selected) else float(np.mean(labels[selected]))
        ),
        "selected_actual_improve_fraction": (
            None
            if not np.any(selected)
            else float(np.mean(updated[selected] + 0.1 < baseline[selected]))
        ),
        "selected_actual_worsen_fraction": (
            None
            if not np.any(selected)
            else float(np.mean(updated[selected] > baseline[selected] + 0.1))
        ),
        "selected_baseline_median_px": (
            None if not np.any(selected) else float(np.median(baseline[selected]))
        ),
        "selected_updated_median_px": (
            None if not np.any(selected) else float(np.median(updated[selected]))
        ),
    }


def _write_predictions(
    path: Path,
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> None:
    fieldnames = [
        "candidate_identity_key",
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "update_beneficial_probability",
        "update_approved",
        "center_x",
        "center_y",
        "refined_x",
        "refined_y",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example, probability in zip(examples, probabilities.tolist()):
            center = np.asarray(example["center_xy"], dtype=np.float64)
            refined = np.asarray(example["refined_xy"], dtype=np.float64)
            writer.writerow(
                {
                    **{name: example[name] for name in fieldnames[:6]},
                    "update_beneficial_probability": float(probability),
                    "update_approved": bool(float(probability) >= float(threshold)),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "refined_x": float(refined[0]),
                    "refined_y": float(refined[1]),
                }
            )


def fit_candidate_update_verifier(
    *,
    train_diagnostic_rows_csv: Path,
    validation_diagnostic_rows_csv: Path,
    train_oof_geometry_probabilities_csv: Path,
    candidate_geometry_verifier_json: Path,
    measurement_checkpoint: Path,
    output_dir: Path,
    c_value: float = 0.1,
    fold_count: int = 5,
    target_precision: float = 0.8,
    minimum_selected_fraction: float = 0.005,
    minimum_baseline_residual_px: float = 1.0,
    maximum_baseline_residual_px: float = 5.0,
    minimum_improvement_px: float = 0.1,
) -> dict[str, object]:
    geometry_model_path = Path(candidate_geometry_verifier_json)
    geometry_model_sha = file_sha256_short(geometry_model_path)
    geometry_model = CandidateGeometryVerifier.from_dict(
        json.loads(geometry_model_path.read_text())
    )
    checkpoint_sha = file_sha256_short(Path(measurement_checkpoint))
    if geometry_model.measurement_checkpoint_sha256 != checkpoint_sha:
        raise ValueError("geometry verifier and measurement checkpoint differ")
    geometry_summary_path = geometry_model_path.parent / "summary.json"
    geometry_summary = json.loads(geometry_summary_path.read_text())
    if str(geometry_summary.get("outputs", {}).get("model_sha256", "")) != geometry_model_sha:
        raise ValueError("candidate geometry model hash differs from fit summary")
    train_geometry = _load_train_oof_geometry_probabilities(
        path=Path(train_oof_geometry_probabilities_csv),
        geometry_summary=geometry_summary,
        diagnostic_rows_csv=Path(train_diagnostic_rows_csv),
    )
    validation_geometry_examples = build_candidate_geometry_examples(
        Path(validation_diagnostic_rows_csv)
    )
    validation_geometry_values = geometry_model.predict(
        _geometry_feature_matrix(validation_geometry_examples)
    )
    validation_geometry = {
        str(example["candidate_identity_key"]): float(probability)
        for example, probability in zip(
            validation_geometry_examples, validation_geometry_values.tolist()
        )
    }
    build_kwargs = {
        "minimum_baseline_residual_px": float(minimum_baseline_residual_px),
        "maximum_baseline_residual_px": float(maximum_baseline_residual_px),
        "minimum_improvement_px": float(minimum_improvement_px),
    }
    train = build_candidate_update_examples(
        Path(train_diagnostic_rows_csv),
        calibrated_geometry_probabilities=train_geometry,
        **build_kwargs,
    )
    validation = build_candidate_update_examples(
        Path(validation_diagnostic_rows_csv),
        calibrated_geometry_probabilities=validation_geometry,
        **build_kwargs,
    )
    observed_checkpoint_hashes = {
        str(example["measurement_checkpoint_sha256"])
        for example in (*train, *validation)
    }
    if observed_checkpoint_hashes != {checkpoint_sha}:
        raise ValueError("candidate update diagnostics and checkpoint differ")
    train_queries = {str(example["query_id"]) for example in train}
    validation_queries = {str(example["query_id"]) for example in validation}
    if train_queries & validation_queries:
        raise ValueError("train/validation query leakage in candidate update verifier")
    features = _matrix(train)
    labels = np.asarray(
        [bool(example["target_safe_update_beneficial"]) for example in train],
        dtype=np.int64,
    )
    groups = np.asarray([str(example["query_id"]) for example in train], dtype=object)
    if len(np.unique(labels)) != 2:
        raise ValueError("candidate update verifier requires both target classes")
    split_count = min(int(fold_count), len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("candidate update verifier requires at least two query groups")
    oof = np.zeros((len(train),), dtype=np.float64)
    splitter = GroupKFold(n_splits=split_count)
    for fit_indices, heldout_indices in splitter.split(features, labels, groups):
        mean, scale, model = _fit_base(
            features[fit_indices], labels[fit_indices], c_value=float(c_value)
        )
        oof[heldout_indices] = model.predict_proba(
            (features[heldout_indices] - mean[None]) / scale[None]
        )[:, 1]
    calibration_slope, calibration_intercept = _fit_calibration(oof, labels)
    calibrated_oof = 1.0 / (
        1.0
        + np.exp(
            -(
                calibration_slope * _logit(oof)
                + calibration_intercept
            )
        )
    )
    mean, scale, model = _fit_base(features, labels, c_value=float(c_value))
    provisional = CandidateUpdateVerifier(
        feature_names=CANDIDATE_UPDATE_FEATURE_NAMES,
        feature_mean=tuple(mean.tolist()),
        feature_scale=tuple(scale.tolist()),
        coefficients=tuple(model.coef_[0].tolist()),
        intercept=float(model.intercept_[0]),
        calibration_slope=calibration_slope,
        calibration_intercept=calibration_intercept,
        update_threshold=1.0,
        minimum_baseline_residual_px=float(minimum_baseline_residual_px),
        maximum_baseline_residual_px=float(maximum_baseline_residual_px),
        minimum_improvement_px=float(minimum_improvement_px),
        measurement_checkpoint_sha256=checkpoint_sha,
        candidate_geometry_verifier_sha256=geometry_model_sha,
    )
    validation_probability = provisional.predict(_matrix(validation))
    validation_labels = np.asarray(
        [bool(example["target_safe_update_beneficial"]) for example in validation],
        dtype=np.int64,
    )
    threshold, threshold_audit = _threshold_at_precision(
        validation_labels,
        validation_probability,
        target_precision=float(target_precision),
        minimum_selected_fraction=float(minimum_selected_fraction),
    )
    verifier = CandidateUpdateVerifier(
        **{
            **provisional.__dict__,
            "update_threshold": float(threshold),
        }
    )
    train_metrics = _evaluate(train, calibrated_oof, threshold=threshold)
    validation_metrics = _evaluate(
        validation, validation_probability, threshold=threshold
    )
    promotion_passes = bool(
        float(validation_metrics.get("auroc") or 0.0) >= 0.75
        and float(validation_metrics.get("selected_safe_update_precision") or 0.0)
        >= float(target_precision)
        and float(validation_metrics.get("selected_actual_improve_fraction") or 0.0)
        >= 0.85
        and float(validation_metrics.get("selected_actual_worsen_fraction") or 1.0)
        <= 0.10
        and threshold < 1.0
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "candidate_update_verifier.json"
    model_path.write_text(json.dumps(verifier.to_dict(), indent=2, sort_keys=True) + "\n")
    train_predictions = output / "train_oof_update_predictions.csv"
    validation_predictions = output / "validation_update_predictions.csv"
    _write_predictions(
        train_predictions, train, calibrated_oof, threshold=float(threshold)
    )
    _write_predictions(
        validation_predictions,
        validation,
        validation_probability,
        threshold=float(threshold),
    )
    summary = {
        "stage": "candidate_coordinate_update_verifier_fit",
        "protocol": {
            "target": (
                "GT_pose_projection_residual_improves_within_local_1_to_5px_"
                "window_TARGET_ONLY"
            ),
            "features_use_pose_or_ground_truth": False,
            "grouped_oof_by_query": True,
            "probability_calibration": "Platt_scaling_on_grouped_train_OOF",
            "threshold_selection": "validation_precision_gate",
            "test_used": False,
            "coordinate_update_applied": False,
        },
        "feature_names": list(CANDIDATE_UPDATE_FEATURE_NAMES),
        "c_value": float(c_value),
        "update_threshold": float(threshold),
        "target_definition": {
            "minimum_baseline_residual_px_exclusive": float(
                minimum_baseline_residual_px
            ),
            "maximum_baseline_residual_px_inclusive": float(
                maximum_baseline_residual_px
            ),
            "minimum_improvement_px": float(minimum_improvement_px),
        },
        "train_oof": train_metrics,
        "validation": validation_metrics,
        "threshold_audit": threshold_audit,
        "promotion_passes": promotion_passes,
        "inputs": {
            "train_diagnostic_rows_csv": str(train_diagnostic_rows_csv),
            "train_diagnostic_rows_sha256": file_sha256_short(
                Path(train_diagnostic_rows_csv)
            ),
            "validation_diagnostic_rows_csv": str(validation_diagnostic_rows_csv),
            "validation_diagnostic_rows_sha256": file_sha256_short(
                Path(validation_diagnostic_rows_csv)
            ),
            "train_oof_geometry_probabilities_csv": str(
                train_oof_geometry_probabilities_csv
            ),
            "train_oof_geometry_probabilities_sha256": file_sha256_short(
                Path(train_oof_geometry_probabilities_csv)
            ),
            "candidate_geometry_verifier": str(geometry_model_path),
            "candidate_geometry_verifier_sha256": geometry_model_sha,
            "measurement_checkpoint": str(measurement_checkpoint),
            "measurement_checkpoint_sha256": checkpoint_sha,
        },
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "train_oof_predictions": str(train_predictions),
            "train_oof_predictions_sha256": file_sha256_short(train_predictions),
            "validation_predictions": str(validation_predictions),
            "validation_predictions_sha256": file_sha256_short(
                validation_predictions
            ),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def apply_candidate_update_verifier(
    *,
    model_path: Path,
    candidate_geometry_verifier_json: Path,
    diagnostic_rows_csv: Path,
    output_path: Path,
) -> dict[str, object]:
    payload = json.loads(Path(model_path).read_text())
    verifier = CandidateUpdateVerifier.from_dict(payload)
    geometry_path = Path(candidate_geometry_verifier_json)
    geometry_sha = file_sha256_short(geometry_path)
    if geometry_sha != verifier.candidate_geometry_verifier_sha256:
        raise ValueError("candidate update and geometry verifier identities differ")
    geometry_model = CandidateGeometryVerifier.from_dict(
        json.loads(geometry_path.read_text())
    )
    if (
        geometry_model.measurement_checkpoint_sha256
        != verifier.measurement_checkpoint_sha256
    ):
        raise ValueError("candidate update and geometry verifier checkpoints differ")
    geometry_examples = build_candidate_geometry_examples(Path(diagnostic_rows_csv))
    observed_checkpoint_hashes = {
        str(example["measurement_checkpoint_sha256"])
        for example in geometry_examples
    }
    if observed_checkpoint_hashes != {verifier.measurement_checkpoint_sha256}:
        raise ValueError("candidate update diagnostics use a different checkpoint")
    geometry_values = geometry_model.predict(_geometry_feature_matrix(geometry_examples))
    geometry_probabilities = {
        str(example["candidate_identity_key"]): float(probability)
        for example, probability in zip(geometry_examples, geometry_values.tolist())
    }
    examples = build_candidate_update_examples(
        Path(diagnostic_rows_csv),
        calibrated_geometry_probabilities=geometry_probabilities,
        minimum_baseline_residual_px=verifier.minimum_baseline_residual_px,
        maximum_baseline_residual_px=verifier.maximum_baseline_residual_px,
        minimum_improvement_px=verifier.minimum_improvement_px,
    )
    probabilities = verifier.predict(_matrix(examples))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(
        output,
        examples,
        probabilities,
        threshold=float(verifier.update_threshold),
    )
    summary = {
        "stage": "candidate_coordinate_update_verifier_apply",
        "protocol": {
            "threshold_search": False,
            "features_use_pose_or_ground_truth": False,
            "coordinate_update_applied": False,
        },
        "update_threshold": float(verifier.update_threshold),
        "metrics_TARGET_ONLY": _evaluate(
            examples, probabilities, threshold=float(verifier.update_threshold)
        ),
        "inputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(Path(model_path)),
            "candidate_geometry_verifier": str(geometry_path),
            "candidate_geometry_verifier_sha256": geometry_sha,
            "diagnostic_rows_csv": str(diagnostic_rows_csv),
            "diagnostic_rows_sha256": file_sha256_short(Path(diagnostic_rows_csv)),
        },
        "outputs": {
            "predictions": str(output),
            "predictions_sha256": file_sha256_short(output),
            "summary": str(output.with_suffix(".summary.json")),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
