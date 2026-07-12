"""Tail-safe pairwise promotion from an immutable baseline pose."""

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


ABSOLUTE_EVIDENCE_NAMES = (
    "success",
    "log_match_count",
    "log_inlier_count",
    "inlier_ratio",
    "all_grid_coverage",
    "inlier_grid_coverage",
    "log_inlier_median_residual",
    "log_inlier_p90_residual",
    "selection_confidence_min",
    "selection_confidence_median",
)
PAIRWISE_FEATURE_NAMES = tuple(
    [f"baseline_{name}" for name in ABSOLUTE_EVIDENCE_NAMES]
    + [f"optional_{name}" for name in ABSOLUTE_EVIDENCE_NAMES]
    + [f"delta_{name}" for name in ABSOLUTE_EVIDENCE_NAMES]
)
NO_PROMOTION_THRESHOLD = 1.000001


def _finite(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def absolute_pose_evidence(row: Mapping[str, object]) -> np.ndarray:
    """Build inference-safe evidence without reading GT pose-error fields."""

    match_count = max(_finite(row.get("match_count"), 0.0), 0.0)
    inlier_count = max(_finite(row.get("inlier_count"), 0.0), 0.0)
    inlier_ratio = _finite(
        row.get("inlier_ratio"), inlier_count / max(match_count, 1.0)
    )
    values = np.asarray(
        [
            float(bool(row.get("success"))),
            math.log1p(match_count),
            math.log1p(inlier_count),
            inlier_ratio,
            _finite(row.get("all_grid_4x4_occupancy_frac"), 0.0),
            _finite(row.get("inlier_grid_4x4_occupancy_frac"), 0.0),
            math.log1p(
                max(_finite(row.get("pnp_reproj_inlier_median_px"), 1e3), 0.0)
            ),
            math.log1p(
                max(_finite(row.get("pnp_reproj_inlier_p90_px"), 1e3), 0.0)
            ),
            _finite(row.get("selection_confidence_min"), -1.0),
            _finite(row.get("selection_confidence_median"), -1.0),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("absolute pose evidence must be finite")
    return values


def is_beneficial(
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
    return bool(
        (
            baseline_translation - optional_translation >= 0.02
            and optional_rotation <= baseline_rotation + 0.05
        )
        or (
            baseline_rotation - optional_rotation >= 0.10
            and optional_translation <= baseline_translation + 0.01
        )
    )


def build_pairwise_examples(
    baseline_rows: Sequence[Mapping[str, object]],
    optional_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = {str(row["query_id"]): dict(row) for row in baseline_rows}
    optional = {str(row["query_id"]): dict(row) for row in optional_rows}
    if len(baseline) != len(baseline_rows) or len(optional) != len(optional_rows):
        raise ValueError("pose rows contain duplicate query ids")
    if set(baseline) != set(optional):
        raise ValueError("baseline and optional query identities differ")
    examples = []
    for query_id in sorted(baseline):
        baseline_evidence = absolute_pose_evidence(baseline[query_id])
        optional_evidence = absolute_pose_evidence(optional[query_id])
        examples.append(
            {
                "query_id": query_id,
                "baseline": baseline[query_id],
                "optional": optional[query_id],
                "features": np.concatenate(
                    [
                        baseline_evidence,
                        optional_evidence,
                        optional_evidence - baseline_evidence,
                    ]
                ),
                "target_beneficial": is_beneficial(
                    baseline[query_id], optional[query_id]
                ),
            }
        )
    return examples


@dataclass(frozen=True)
class PairwisePosePromotionGateV2:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    promotion_threshold: float

    def __post_init__(self) -> None:
        if self.feature_names != PAIRWISE_FEATURE_NAMES:
            raise ValueError("pairwise promotion v2 feature schema mismatch")
        size = len(self.feature_names)
        if any(
            len(values) != size
            for values in (self.feature_mean, self.feature_scale, self.coefficients)
        ):
            raise ValueError("pairwise promotion v2 vector lengths differ")

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(
            -1, len(self.feature_names)
        )
        standardized = (
            values - np.asarray(self.feature_mean)[None]
        ) / np.asarray(self.feature_scale)[None]
        logits = standardized @ np.asarray(self.coefficients) + float(self.intercept)
        logits = (
            float(self.calibration_slope) * logits
            + float(self.calibration_intercept)
        )
        output = np.empty_like(logits)
        positive = logits >= 0.0
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        output[~positive] = exp_logits / (1.0 + exp_logits)
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "pairwise_pose_promotion_gate_v2",
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "calibration_slope": float(self.calibration_slope),
            "calibration_intercept": float(self.calibration_intercept),
            "promotion_threshold": float(self.promotion_threshold),
        }


def _fit_base(
    features: np.ndarray, labels: np.ndarray, *, c_value: float
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = LogisticRegression(
        C=float(c_value), solver="lbfgs", max_iter=3000, random_state=0
    )
    model.fit((features - mean[None]) / scale[None], labels)
    return mean, scale, model


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _calibrate(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(
        C=1000.0, solver="lbfgs", max_iter=3000, random_state=0
    )
    model.fit(_logit(probabilities).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def pose_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successes = [row for row in rows if bool(row.get("success"))]
    translations = np.asarray(
        [float(row["translation_m"]) for row in successes], dtype=np.float64
    )
    rotations = np.asarray(
        [float(row["rotation_deg"]) for row in successes], dtype=np.float64
    )
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "success_count": int(len(successes)),
        "success_rate": 0.0 if not rows else float(len(successes) / len(rows)),
        "median_translation_m_success": (
            None if not len(translations) else float(np.median(translations))
        ),
        "p90_translation_m_success": (
            None if not len(translations) else float(np.quantile(translations, 0.9))
        ),
        "median_rotation_deg_success": (
            None if not len(rotations) else float(np.median(rotations))
        ),
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
                    and row.get("translation_m") is not None
                    and row.get("rotation_deg") is not None
                    and float(row["translation_m"]) <= distance
                    and float(row["rotation_deg"]) <= angle
                    for row in rows
                ]
            )
        )
    return output


def _pose_gate(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    checks = {
        "success_rate": float(candidate["success_rate"])
        >= float(baseline["success_rate"]),
        "median_translation": float(candidate["median_translation_m_success"])
        <= float(baseline["median_translation_m_success"]),
        "p90_translation": float(candidate["p90_translation_m_success"])
        <= float(baseline["p90_translation_m_success"]),
        "median_rotation": float(candidate["median_rotation_deg_success"])
        <= float(baseline["median_rotation_deg_success"]),
        "recall_25cm_2deg": float(candidate["recall_25cm_2deg"])
        >= float(baseline["recall_25cm_2deg"]),
        "recall_10cm_5deg": float(candidate["recall_10cm_5deg"])
        >= float(baseline["recall_10cm_5deg"]),
        "recall_5cm_5deg": float(candidate["recall_5cm_5deg"])
        >= float(baseline["recall_5cm_5deg"]),
    }
    return {"passes": bool(all(checks.values())), "checks": checks}


def select_rows(
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = []
    promoted = 0
    beneficial = 0
    for example, probability in zip(examples, probabilities):
        baseline = dict(example["baseline"])
        if float(probability) < float(threshold):
            selected.append(baseline)
            continue
        optional = dict(example["optional"])
        optional["promotion_probability"] = float(probability)
        optional["promoted_from_immutable_baseline"] = True
        selected.append(optional)
        promoted += 1
        beneficial += int(bool(example["target_beneficial"]))
    return selected, {
        "promotion_count": int(promoted),
        "fallback_count": int(len(examples) - promoted),
        "beneficial_promotion_count_TARGET_ONLY": int(beneficial),
        "promotion_precision_TARGET_ONLY": (
            None if promoted == 0 else float(beneficial / promoted)
        ),
    }


def _validation_threshold(
    examples: Sequence[Mapping[str, object]], probabilities: np.ndarray
) -> tuple[float, list[dict[str, object]], dict[str, object]]:
    baseline_rows = [dict(example["baseline"]) for example in examples]
    baseline_summary = pose_summary(baseline_rows)
    thresholds = [NO_PROMOTION_THRESHOLD]
    thresholds.extend(sorted(set(float(value) for value in probabilities), reverse=True))
    eligible = []
    for threshold in thresholds:
        rows, promotion = select_rows(examples, probabilities, threshold=threshold)
        summary = pose_summary(rows)
        gate = _pose_gate(summary, baseline_summary)
        trial = {
            "threshold": float(threshold),
            "pose": summary,
            "pose_gate": gate,
            "promotion": promotion,
            "rows": rows,
        }
        if bool(gate["passes"]):
            eligible.append(trial)
    chosen = max(
        eligible,
        key=lambda trial: (
            float(trial["pose"]["recall_10cm_5deg"]),
            float(trial["pose"]["recall_25cm_2deg"]),
            float(trial["pose"]["recall_5cm_5deg"]),
            -float(trial["pose"]["median_translation_m_success"]),
            -float(trial["pose"]["p90_translation_m_success"]),
            -float(trial["pose"]["median_rotation_deg_success"]),
            -int(trial["promotion"]["promotion_count"]),
        ),
    )
    effective_promotion = int(chosen["promotion"]["promotion_count"]) > 0
    if not effective_promotion:
        fallback_rows, fallback_promotion = select_rows(
            examples, probabilities, threshold=NO_PROMOTION_THRESHOLD
        )
        chosen = {
            "threshold": NO_PROMOTION_THRESHOLD,
            "pose": pose_summary(fallback_rows),
            "pose_gate": _pose_gate(pose_summary(fallback_rows), baseline_summary),
            "promotion": fallback_promotion,
            "rows": fallback_rows,
        }
    return float(chosen["threshold"]), list(chosen["rows"]), {
        "pose": chosen["pose"],
        "pose_gate": chosen["pose_gate"],
        "promotion": chosen["promotion"],
        "eligible_threshold_count": int(len(eligible)),
        "effective_promotion": bool(effective_promotion),
    }


def _paired_report(
    baseline_rows: Sequence[Mapping[str, object]],
    selected_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline = {str(row["query_id"]): row for row in baseline_rows}
    selected = {str(row["query_id"]): row for row in selected_rows}
    ids = sorted(baseline)
    translation_delta = np.asarray(
        [
            float(selected[q]["translation_m"])
            - float(baseline[q]["translation_m"])
            for q in ids
        ],
        dtype=np.float64,
    )
    rotation_delta = np.asarray(
        [
            float(selected[q]["rotation_deg"])
            - float(baseline[q]["rotation_deg"])
            for q in ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(0)
    bootstrap = []
    for _ in range(2000):
        indices = rng.integers(0, len(ids), size=len(ids))
        bootstrap.append(
            (
                float(np.median(translation_delta[indices])),
                float(np.median(rotation_delta[indices])),
            )
        )
    boot = np.asarray(bootstrap, dtype=np.float64)
    return {
        "translation_wins": int(np.sum(translation_delta < 0.0)),
        "translation_losses": int(np.sum(translation_delta > 0.0)),
        "translation_ties": int(np.sum(translation_delta == 0.0)),
        "median_translation_delta_m": float(np.median(translation_delta)),
        "median_rotation_delta_deg": float(np.median(rotation_delta)),
        "catastrophic_regression_count": int(
            np.sum((translation_delta > 0.25) | (rotation_delta > 0.5))
        ),
        "bootstrap_95ci": {
            "median_translation_delta_m": [
                float(np.quantile(boot[:, 0], 0.025)),
                float(np.quantile(boot[:, 0], 0.975)),
            ],
            "median_rotation_delta_deg": [
                float(np.quantile(boot[:, 1], 0.025)),
                float(np.quantile(boot[:, 1], 0.975)),
            ],
        },
    }


def fit_pairwise_pose_promotion_gate_v2(
    *,
    train_baseline_rows: Sequence[Mapping[str, object]],
    train_optional_rows: Sequence[Mapping[str, object]],
    validation_baseline_rows: Sequence[Mapping[str, object]],
    validation_optional_rows: Sequence[Mapping[str, object]],
    test_baseline_rows: Sequence[Mapping[str, object]],
    test_optional_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    input_paths: Sequence[Path],
    c_value: float = 0.1,
    fold_count: int = 5,
) -> dict[str, object]:
    train = build_pairwise_examples(train_baseline_rows, train_optional_rows)
    validation = build_pairwise_examples(
        validation_baseline_rows, validation_optional_rows
    )
    test = build_pairwise_examples(test_baseline_rows, test_optional_rows)
    features = np.asarray([row["features"] for row in train], dtype=np.float64)
    labels = np.asarray(
        [bool(row["target_beneficial"]) for row in train], dtype=np.int64
    )
    groups = np.asarray([str(row["query_id"]) for row in train], dtype=object)
    if len(np.unique(labels)) != 2:
        raise ValueError("pairwise promotion v2 requires both target classes")
    splitter = GroupKFold(n_splits=min(int(fold_count), len(np.unique(groups))))
    oof = np.zeros((len(train),), dtype=np.float64)
    for fit_indices, heldout_indices in splitter.split(features, labels, groups):
        mean, scale, model = _fit_base(
            features[fit_indices], labels[fit_indices], c_value=float(c_value)
        )
        oof[heldout_indices] = model.predict_proba(
            (features[heldout_indices] - mean[None]) / scale[None]
        )[:, 1]
    calibration_slope, calibration_intercept = _calibrate(oof, labels)
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
    validation_features = np.asarray(
        [row["features"] for row in validation], dtype=np.float64
    )
    uncalibrated_validation = model.predict_proba(
        (validation_features - mean[None]) / scale[None]
    )[:, 1]
    validation_probabilities = 1.0 / (
        1.0
        + np.exp(
            -(
                calibration_slope * _logit(uncalibrated_validation)
                + calibration_intercept
            )
        )
    )
    threshold, validation_selected, validation_selection = _validation_threshold(
        validation, validation_probabilities
    )
    gate = PairwisePosePromotionGateV2(
        feature_names=PAIRWISE_FEATURE_NAMES,
        feature_mean=tuple(mean.tolist()),
        feature_scale=tuple(scale.tolist()),
        coefficients=tuple(model.coef_[0].tolist()),
        intercept=float(model.intercept_[0]),
        calibration_slope=float(calibration_slope),
        calibration_intercept=float(calibration_intercept),
        promotion_threshold=float(threshold),
    )
    test_features = np.asarray([row["features"] for row in test], dtype=np.float64)
    test_probabilities = gate.predict(test_features)
    test_selected, test_promotion = select_rows(
        test, test_probabilities, threshold=float(threshold)
    )
    validation_labels = np.asarray(
        [bool(row["target_beneficial"]) for row in validation], dtype=np.int64
    )
    test_labels = np.asarray(
        [bool(row["target_beneficial"]) for row in test], dtype=np.int64
    )
    validation_baseline_summary = pose_summary(validation_baseline_rows)
    test_baseline_summary = pose_summary(test_baseline_rows)
    test_selected_summary = pose_summary(test_selected)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "pairwise_pose_promotion_gate_v2.json"
    model_path.write_text(json.dumps(gate.to_dict(), indent=2, sort_keys=True) + "\n")
    (output / "validation_selected_rows.json").write_text(
        json.dumps(validation_selected, indent=2, sort_keys=True) + "\n"
    )
    (output / "test_selected_rows.json").write_text(
        json.dumps(test_selected, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "stage": "pairwise_pose_promotion_gate_v2",
        "protocol": {
            "baseline_immutable": True,
            "pool_relative_features": False,
            "pose_error_features": False,
            "GT_pose_errors_target_only": True,
            "train_grouped_oof": True,
            "validation_policy_selection_only": True,
            "test_used_for_selection": False,
            "feature_names": list(PAIRWISE_FEATURE_NAMES),
        },
        "promotion_threshold": float(threshold),
        "train_oof_classifier": confidence_metrics(labels, calibrated_oof),
        "validation_classifier": confidence_metrics(
            validation_labels, validation_probabilities
        ),
        "test_classifier": confidence_metrics(test_labels, test_probabilities),
        "validation": {
            "baseline_pose": validation_baseline_summary,
            "optional_pose": pose_summary(validation_optional_rows),
            "selected_pose": validation_selection["pose"],
            "selection": validation_selection,
            "paired": _paired_report(
                validation_baseline_rows, validation_selected
            ),
        },
        "test": {
            "baseline_pose": test_baseline_summary,
            "optional_pose": pose_summary(test_optional_rows),
            "selected_pose": test_selected_summary,
            "pose_gate": _pose_gate(test_selected_summary, test_baseline_summary),
            "promotion": test_promotion,
            "paired": _paired_report(test_baseline_rows, test_selected),
        },
        "gate": {
            "validation_passed": bool(
                validation_selection["pose_gate"]["passes"]
                and validation_selection["effective_promotion"]
            ),
            "test_passed": bool(
                validation_selection["effective_promotion"]
                and _pose_gate(test_selected_summary, test_baseline_summary)["passes"]
            ),
            "production_promoted": False,
        },
        "inputs": [
            {"path": str(path), "sha256": file_sha256_short(path)}
            for path in dict.fromkeys(Path(path) for path in input_paths)
        ],
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "validation_selected_rows": str(output / "validation_selected_rows.json"),
            "test_selected_rows": str(output / "test_selected_rows.json"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
