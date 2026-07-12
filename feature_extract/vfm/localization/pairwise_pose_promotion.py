"""Abstaining pairwise promotion from an immutable baseline pose."""

from __future__ import annotations

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


OPTIONAL_STRATEGIES = (
    "candidate_pgeometry_argmax_DIAGNOSTIC_ONLY",
    "candidate_prior_assignment",
)

ABSOLUTE_EVIDENCE_NAMES = (
    "success",
    "log_match_count",
    "log_inlier_count",
    "inlier_ratio",
)

PAIRWISE_FEATURE_NAMES = tuple(
    [f"baseline_{name}" for name in ABSOLUTE_EVIDENCE_NAMES]
    + [f"optional_{name}" for name in ABSOLUTE_EVIDENCE_NAMES]
    + [f"delta_{name}" for name in ABSOLUTE_EVIDENCE_NAMES]
    + [f"strategy_{name}" for name in OPTIONAL_STRATEGIES]
)


@dataclass(frozen=True)
class PairwisePosePromotionGate:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    promotion_threshold: float
    target_precision: float

    def __post_init__(self) -> None:
        if self.feature_names != PAIRWISE_FEATURE_NAMES:
            raise ValueError("pairwise promotion feature schema mismatch")
        count = len(self.feature_names)
        if any(
            len(values) != count
            for values in (self.feature_mean, self.feature_scale, self.coefficients)
        ):
            raise ValueError("pairwise promotion vector lengths differ")

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(-1, len(self.feature_names))
        standardized = (
            values - np.asarray(self.feature_mean)[None]
        ) / np.asarray(self.feature_scale)[None]
        base_logit = standardized @ np.asarray(self.coefficients) + float(self.intercept)
        calibrated = (
            float(self.calibration_slope) * base_logit
            + float(self.calibration_intercept)
        )
        output = np.empty_like(calibrated)
        positive = calibrated >= 0.0
        output[positive] = 1.0 / (1.0 + np.exp(-calibrated[positive]))
        exponential = np.exp(calibrated[~positive])
        output[~positive] = exponential / (1.0 + exponential)
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "pairwise_pose_promotion_gate_v1",
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "calibration_slope": float(self.calibration_slope),
            "calibration_intercept": float(self.calibration_intercept),
            "promotion_threshold": float(self.promotion_threshold),
            "target_precision": float(self.target_precision),
        }


def _load_rows(directory: Path, strategy: str) -> list[dict[str, object]]:
    path = Path(directory) / f"{strategy}.json"
    return [dict(row) for row in json.loads(path.read_text())]


def _absolute_evidence(row: Mapping[str, object]) -> np.ndarray:
    match_count = max(float(row.get("match_count", 0.0)), 0.0)
    inlier_count = max(float(row.get("inlier_count", 0.0)), 0.0)
    return np.asarray(
        [
            float(bool(row.get("success"))),
            math.log1p(match_count),
            math.log1p(inlier_count),
            inlier_count / max(match_count, 1.0),
        ],
        dtype=np.float64,
    )


def _is_beneficial(
    baseline: Mapping[str, object], optional: Mapping[str, object]
) -> bool:
    if not bool(optional.get("success")):
        return False
    if not bool(baseline.get("success")):
        return True
    baseline_translation = float(baseline["translation_m"])
    optional_translation = float(optional["translation_m"])
    baseline_rotation = float(baseline["rotation_deg"])
    optional_rotation = float(optional["rotation_deg"])
    translation_gain = baseline_translation - optional_translation
    rotation_gain = baseline_rotation - optional_rotation
    return bool(
        (translation_gain >= 0.02 and optional_rotation <= baseline_rotation + 0.05)
        or (rotation_gain >= 0.10 and optional_translation <= baseline_translation + 0.01)
    )


def build_pairwise_examples(directory: Path) -> list[dict[str, object]]:
    baseline_rows = _load_rows(Path(directory), "frozen_L97_replay")
    baseline_by_query = {str(row["query_id"]): row for row in baseline_rows}
    examples: list[dict[str, object]] = []
    for strategy_index, strategy in enumerate(OPTIONAL_STRATEGIES):
        for optional in _load_rows(Path(directory), strategy):
            query_id = str(optional["query_id"])
            baseline = baseline_by_query.get(query_id)
            if baseline is None:
                raise ValueError("optional pose lacks its frozen baseline")
            baseline_evidence = _absolute_evidence(baseline)
            optional_evidence = _absolute_evidence(optional)
            strategy_one_hot = np.zeros((len(OPTIONAL_STRATEGIES),), dtype=np.float64)
            strategy_one_hot[strategy_index] = 1.0
            features = np.concatenate(
                [
                    baseline_evidence,
                    optional_evidence,
                    optional_evidence - baseline_evidence,
                    strategy_one_hot,
                ]
            )
            if not np.all(np.isfinite(features)):
                raise ValueError("pairwise promotion features must be finite")
            examples.append(
                {
                    "query_id": query_id,
                    "strategy": strategy,
                    "baseline": baseline,
                    "optional": optional,
                    "features": features,
                    "target_beneficial": _is_beneficial(baseline, optional),
                }
            )
    return examples


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
    values = np.clip(np.asarray(probabilities), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _calibrate(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=2000, random_state=0)
    model.fit(_logit(probabilities).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _threshold_at_precision(
    labels: np.ndarray, probabilities: np.ndarray, target_precision: float
) -> float:
    order = np.argsort(-probabilities, kind="stable")
    cumulative_precision = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    valid = np.flatnonzero(cumulative_precision >= float(target_precision))
    return 1.0 if not len(valid) else float(probabilities[order[int(valid[-1])]])


def _pose_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successes = [row for row in rows if bool(row.get("success"))]
    translations = np.asarray([float(row["translation_m"]) for row in successes])
    rotations = np.asarray([float(row["rotation_deg"]) for row in successes])
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "success_rate": 0.0 if not rows else float(len(successes) / len(rows)),
        "median_translation_m_success": None if not len(translations) else float(np.median(translations)),
        "p90_translation_m_success": None if not len(translations) else float(np.quantile(translations, 0.9)),
        "median_rotation_deg_success": None if not len(rotations) else float(np.median(rotations)),
    }
    for distance, angle, name in (
        (0.25, 2.0, "25cm_2deg"),
        (0.10, 5.0, "10cm_5deg"),
        (0.05, 5.0, "5cm_5deg"),
    ):
        output[f"recall_{name}"] = float(
            np.mean(
                [
                    bool(row.get("success"))
                    and float(row.get("translation_m") or np.inf) <= distance
                    and float(row.get("rotation_deg") or np.inf) <= angle
                    for row in rows
                ]
            )
        )
    return output


def _select_poses(
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(str(example["query_id"]), []).append(index)
    selected: list[dict[str, object]] = []
    promotion_count = 0
    beneficial_count = 0
    for query_id in sorted(grouped):
        indices = grouped[query_id]
        best = max(indices, key=lambda index: (float(probabilities[index]), str(examples[index]["strategy"])))
        baseline = dict(examples[best]["baseline"])
        if float(probabilities[best]) < float(threshold):
            selected.append(baseline)
            continue
        optional = dict(examples[best]["optional"])
        optional["promoted_strategy"] = str(examples[best]["strategy"])
        optional["promotion_probability"] = float(probabilities[best])
        selected.append(optional)
        promotion_count += 1
        beneficial_count += int(bool(examples[best]["target_beneficial"]))
    return selected, {
        "promotion_count": int(promotion_count),
        "beneficial_promotion_count_TARGET_ONLY": int(beneficial_count),
        "promotion_precision_TARGET_ONLY": (
            None if promotion_count == 0 else float(beneficial_count / promotion_count)
        ),
    }


def fit_pairwise_pose_promotion_gate(
    *,
    train_pose_dir: Path,
    validation_pose_dir: Path,
    output_dir: Path,
    target_precision: float = 0.8,
    c_value: float = 1.0,
    fold_count: int = 5,
) -> dict[str, object]:
    train = build_pairwise_examples(Path(train_pose_dir))
    validation = build_pairwise_examples(Path(validation_pose_dir))
    features = np.asarray([example["features"] for example in train], dtype=np.float64)
    labels = np.asarray([bool(example["target_beneficial"]) for example in train], dtype=np.int64)
    groups = np.asarray([str(example["query_id"]) for example in train], dtype=object)
    if len(np.unique(labels)) != 2:
        raise ValueError("pairwise promotion requires both target classes")
    splitter = GroupKFold(n_splits=min(int(fold_count), len(np.unique(groups))))
    oof = np.zeros((len(train),), dtype=np.float64)
    for train_indices, heldout_indices in splitter.split(features, labels, groups):
        mean, scale, model = _fit_base(features[train_indices], labels[train_indices], c_value=float(c_value))
        oof[heldout_indices] = model.predict_proba(
            (features[heldout_indices] - mean[None]) / scale[None]
        )[:, 1]
    calibration_slope, calibration_intercept = _calibrate(oof, labels)
    calibrated_oof = 1.0 / (
        1.0 + np.exp(-(calibration_slope * _logit(oof) + calibration_intercept))
    )
    threshold = _threshold_at_precision(labels, calibrated_oof, float(target_precision))
    mean, scale, model = _fit_base(features, labels, c_value=float(c_value))
    gate = PairwisePosePromotionGate(
        feature_names=PAIRWISE_FEATURE_NAMES,
        feature_mean=tuple(mean.tolist()),
        feature_scale=tuple(scale.tolist()),
        coefficients=tuple(model.coef_[0].tolist()),
        intercept=float(model.intercept_[0]),
        calibration_slope=calibration_slope,
        calibration_intercept=calibration_intercept,
        promotion_threshold=threshold,
        target_precision=float(target_precision),
    )
    validation_features = np.asarray(
        [example["features"] for example in validation], dtype=np.float64
    )
    validation_labels = np.asarray(
        [bool(example["target_beneficial"]) for example in validation], dtype=np.int64
    )
    validation_probabilities = gate.predict(validation_features)
    train_selected, train_promotions = _select_poses(
        train, calibrated_oof, threshold=float(threshold)
    )
    validation_selected, validation_promotions = _select_poses(
        validation, validation_probabilities, threshold=float(threshold)
    )
    validation_baseline = [
        dict(example["baseline"])
        for example in validation[:: len(OPTIONAL_STRATEGIES)]
    ]
    # Build by identity rather than relying on example order.
    validation_baseline = list(
        {
            str(example["query_id"]): dict(example["baseline"])
            for example in validation
        }.values()
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "pairwise_pose_promotion_gate.json"
    model_path.write_text(json.dumps(gate.to_dict(), indent=2, sort_keys=True) + "\n")
    (output / "validation_selected_rows.json").write_text(
        json.dumps(validation_selected, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "stage": "pairwise_pose_promotion_gate_fit",
        "protocol": {
            "baseline_immutable": True,
            "features": "baseline_absolute_optional_absolute_fixed_difference_only",
            "pool_relative_features": False,
            "pose_error_features": False,
            "GT_pose_errors_target_only": True,
            "grouped_oof_by_query": True,
        },
        "promotion_threshold": float(threshold),
        "train_oof_classifier": confidence_metrics(labels, calibrated_oof),
        "validation_classifier": confidence_metrics(validation_labels, validation_probabilities),
        "train_oof_promotions": train_promotions,
        "validation_promotions": validation_promotions,
        "validation_baseline_pose": _pose_summary(validation_baseline),
        "validation_selected_pose": _pose_summary(validation_selected),
        "inputs": {
            "train_pose_summary_sha256": file_sha256_short(Path(train_pose_dir) / "summary.json"),
            "validation_pose_summary_sha256": file_sha256_short(Path(validation_pose_dir) / "summary.json"),
        },
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "validation_selected_rows": str(output / "validation_selected_rows.json"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
