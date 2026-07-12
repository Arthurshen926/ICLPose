"""Conservative train-only selection between two frozen pose backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


POSE_BACKEND_GATE_FEATURE_NAMES: tuple[str, ...] = (
    "ranker_top_score",
    "ranker_top_margin",
    "ranker_probability_entropy",
    "chosen_pose_cluster_50cm_5deg_fraction",
    "chosen_verification_strict_inlier_ratio",
    "chosen_verification_loose_inlier_ratio",
    "chosen_verification_descriptor_score_mean",
    "chosen_verification_descriptor_rank_mean",
    "learned_final_audit_strict_inlier_ratio",
    "learned_final_audit_loose_inlier_ratio",
    "learned_final_audit_soft_consensus_ratio",
    "learned_final_audit_median_residual_log1p",
    "learned_full_inlier_ratio",
    "baseline_ransac_inlier_ratio",
    "pose_disagreement_translation_log1p",
    "pose_disagreement_rotation_log1p",
)


@dataclass(frozen=True)
class PoseBackendSelectionExample:
    query_id: str
    features: tuple[float, ...]
    learned_translation_m: float
    learned_rotation_deg: float
    baseline_translation_m: float
    baseline_rotation_deg: float

    def __post_init__(self) -> None:
        if len(self.features) != len(POSE_BACKEND_GATE_FEATURE_NAMES):
            raise ValueError("pose backend gate feature dimension is incompatible")
        values = np.asarray(
            [
                *self.features,
                self.learned_translation_m,
                self.learned_rotation_deg,
                self.baseline_translation_m,
                self.baseline_rotation_deg,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("pose backend selection example contains non-finite values")


@dataclass(frozen=True)
class PoseBackendGate:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    threshold: float
    c_value: float
    rotation_equivalent_m_per_deg: float

    def __post_init__(self) -> None:
        count = len(POSE_BACKEND_GATE_FEATURE_NAMES)
        if not (
            len(self.mean) == count
            and len(self.scale) == count
            and len(self.weights) == count
        ):
            raise ValueError("pose backend gate model dimension is incompatible")
        if min(self.scale) <= 0.0 or not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("pose backend gate scale or threshold is invalid")

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != len(POSE_BACKEND_GATE_FEATURE_NAMES):
            raise ValueError("pose backend gate input dimension is incompatible")
        standardized = (
            values - np.asarray(self.mean, dtype=np.float64)[None, :]
        ) / np.asarray(self.scale, dtype=np.float64)[None, :]
        logits = standardized @ np.asarray(self.weights, dtype=np.float64) + float(
            self.bias
        )
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))

    def choose_learned(self, features: np.ndarray) -> np.ndarray:
        return self.probabilities(features) >= float(self.threshold)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(POSE_BACKEND_GATE_FEATURE_NAMES),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "threshold": float(self.threshold),
            "c_value": float(self.c_value),
            "rotation_equivalent_m_per_deg": float(
                self.rotation_equivalent_m_per_deg
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PoseBackendGate":
        names = tuple(str(value) for value in payload["feature_names"])
        if names != POSE_BACKEND_GATE_FEATURE_NAMES:
            raise ValueError("pose backend gate feature schema is incompatible")
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            weights=tuple(float(value) for value in payload["weights"]),
            bias=float(payload["bias"]),
            threshold=float(payload["threshold"]),
            c_value=float(payload["c_value"]),
            rotation_equivalent_m_per_deg=float(
                payload["rotation_equivalent_m_per_deg"]
            ),
        )


@dataclass(frozen=True)
class MonotonicPoseBackendPolicy:
    descriptor_score_threshold: float
    baseline_inlier_ratio_threshold: float
    final_audit_loose_ratio_threshold: float
    maximum_pose_disagreement_log1p: float

    def choose_learned(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != len(POSE_BACKEND_GATE_FEATURE_NAMES):
            raise ValueError("monotonic backend policy input dimension is incompatible")
        columns = {
            name: index for index, name in enumerate(POSE_BACKEND_GATE_FEATURE_NAMES)
        }
        primary = (
            values[:, columns["chosen_verification_descriptor_score_mean"]]
            >= float(self.descriptor_score_threshold)
        )
        rescue = (
            (
                values[:, columns["baseline_ransac_inlier_ratio"]]
                <= float(self.baseline_inlier_ratio_threshold)
            )
            & (
                values[:, columns["learned_final_audit_loose_inlier_ratio"]]
                >= float(self.final_audit_loose_ratio_threshold)
            )
            & (
                values[:, columns["pose_disagreement_translation_log1p"]]
                <= float(self.maximum_pose_disagreement_log1p)
            )
        )
        return primary | rescue

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(POSE_BACKEND_GATE_FEATURE_NAMES),
            "descriptor_score_threshold": float(self.descriptor_score_threshold),
            "baseline_inlier_ratio_threshold": float(
                self.baseline_inlier_ratio_threshold
            ),
            "final_audit_loose_ratio_threshold": float(
                self.final_audit_loose_ratio_threshold
            ),
            "maximum_pose_disagreement_log1p": float(
                self.maximum_pose_disagreement_log1p
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MonotonicPoseBackendPolicy":
        names = tuple(str(value) for value in payload["feature_names"])
        if names != POSE_BACKEND_GATE_FEATURE_NAMES:
            raise ValueError("monotonic backend policy feature schema is incompatible")
        return cls(
            descriptor_score_threshold=float(payload["descriptor_score_threshold"]),
            baseline_inlier_ratio_threshold=float(
                payload["baseline_inlier_ratio_threshold"]
            ),
            final_audit_loose_ratio_threshold=float(
                payload["final_audit_loose_ratio_threshold"]
            ),
            maximum_pose_disagreement_log1p=float(
                payload["maximum_pose_disagreement_log1p"]
            ),
        )


def _arrays(
    examples: Sequence[PoseBackendSelectionExample],
    *,
    rotation_equivalent_m_per_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([example.features for example in examples], dtype=np.float64)
    learned_quality = np.asarray(
        [
            example.learned_translation_m
            + float(rotation_equivalent_m_per_deg) * example.learned_rotation_deg
            for example in examples
        ],
        dtype=np.float64,
    )
    baseline_quality = np.asarray(
        [
            example.baseline_translation_m
            + float(rotation_equivalent_m_per_deg) * example.baseline_rotation_deg
            for example in examples
        ],
        dtype=np.float64,
    )
    return features, (learned_quality < baseline_quality).astype(np.int64)


def _fit_model(
    examples: Sequence[PoseBackendSelectionExample],
    *,
    c_value: float,
    rotation_equivalent_m_per_deg: float,
    threshold: float,
) -> PoseBackendGate:
    features, labels = _arrays(
        examples,
        rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
    )
    if len(set(labels.tolist())) < 2:
        raise ValueError("pose backend gate training needs both backend labels")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    classifier = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=0,
    )
    classifier.fit((features - mean[None, :]) / scale[None, :], labels)
    return PoseBackendGate(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in classifier.coef_[0]),
        bias=float(classifier.intercept_[0]),
        threshold=float(threshold),
        c_value=float(c_value),
        rotation_equivalent_m_per_deg=float(rotation_equivalent_m_per_deg),
    )


def pose_backend_selection_metrics(
    examples: Sequence[PoseBackendSelectionExample],
    choose_learned: Sequence[bool],
) -> dict[str, object]:
    selected = np.asarray(choose_learned, dtype=bool)
    if len(selected) != len(examples) or not examples:
        raise ValueError("pose backend decisions do not align with examples")
    learned_t = np.asarray(
        [example.learned_translation_m for example in examples], dtype=np.float64
    )
    learned_r = np.asarray(
        [example.learned_rotation_deg for example in examples], dtype=np.float64
    )
    baseline_t = np.asarray(
        [example.baseline_translation_m for example in examples], dtype=np.float64
    )
    baseline_r = np.asarray(
        [example.baseline_rotation_deg for example in examples], dtype=np.float64
    )
    translation = np.where(selected, learned_t, baseline_t)
    rotation = np.where(selected, learned_r, baseline_r)
    oracle = (
        learned_t + 0.02 * learned_r < baseline_t + 0.02 * baseline_r
    )
    return {
        "query_count": int(len(examples)),
        "success_count": int(len(examples)),
        "success_rate": 1.0,
        "learned_selected_count": int(np.sum(selected)),
        "learned_selected_rate": float(np.mean(selected)),
        "decision_accuracy_TARGET_ONLY": float(np.mean(selected == oracle)),
        "median_translation_m": float(np.median(translation)),
        "p90_translation_m": float(np.quantile(translation, 0.9)),
        "median_rotation_deg": float(np.median(rotation)),
        "recall_25cm_2deg": float(
            np.mean((translation <= 0.25) & (rotation <= 2.0))
        ),
        "recall_10cm_5deg": float(
            np.mean((translation <= 0.10) & (rotation <= 5.0))
        ),
        "recall_5cm_5deg": float(
            np.mean((translation <= 0.05) & (rotation <= 5.0))
        ),
        "rows": [
            {
                "query_id": str(example.query_id),
                "selected_backend": "learned_multi_hypothesis" if choice else "l97_all_match",
                "translation_m_TARGET_ONLY": float(value_t),
                "rotation_deg_TARGET_ONLY": float(value_r),
            }
            for example, choice, value_t, value_r in zip(
                examples, selected.tolist(), translation.tolist(), rotation.tolist()
            )
        ],
    }


def _risk(
    metrics: Mapping[str, object], *, rotation_equivalent_m_per_deg: float
) -> float:
    return float(
        float(metrics["median_translation_m"])
        + 0.5 * float(metrics["p90_translation_m"])
        + float(rotation_equivalent_m_per_deg)
        * float(metrics["median_rotation_deg"])
    )


def fit_pose_backend_gate(
    examples: Sequence[PoseBackendSelectionExample],
    *,
    fold_assignments: Mapping[str, int],
    c_values: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0),
    thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    rotation_equivalent_m_per_deg: float = 0.02,
) -> tuple[PoseBackendGate, dict[str, object]]:
    if not examples:
        raise ValueError("pose backend gate requires train examples")
    fold_values = sorted(
        set(int(fold_assignments[str(example.query_id)]) for example in examples)
    )
    baseline_metrics = pose_backend_selection_metrics(
        examples, np.zeros((len(examples),), dtype=bool)
    )
    candidates: list[dict[str, object]] = []
    features = np.asarray([example.features for example in examples], dtype=np.float64)
    for c_value in c_values:
        oof_probabilities = np.zeros((len(examples),), dtype=np.float64)
        for fold in fold_values:
            train_indices = [
                index
                for index, example in enumerate(examples)
                if int(fold_assignments[str(example.query_id)]) != int(fold)
            ]
            heldout_indices = [
                index
                for index, example in enumerate(examples)
                if int(fold_assignments[str(example.query_id)]) == int(fold)
            ]
            model = _fit_model(
                [examples[index] for index in train_indices],
                c_value=float(c_value),
                rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
                threshold=0.5,
            )
            oof_probabilities[heldout_indices] = model.probabilities(
                features[heldout_indices]
            )
        for threshold in thresholds:
            metrics = pose_backend_selection_metrics(
                examples, oof_probabilities >= float(threshold)
            )
            candidates.append(
                {
                    "c_value": float(c_value),
                    "threshold": float(threshold),
                    "risk": _risk(
                        metrics,
                        rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
                    ),
                    "metrics": {key: value for key, value in metrics.items() if key != "rows"},
                }
            )
    passing = [
        candidate
        for candidate in candidates
        if float(candidate["metrics"]["median_translation_m"])
        < float(baseline_metrics["median_translation_m"])
        and float(candidate["metrics"]["p90_translation_m"])
        < float(baseline_metrics["p90_translation_m"])
        and float(candidate["metrics"]["median_rotation_deg"])
        <= float(baseline_metrics["median_rotation_deg"])
    ]
    if passing:
        chosen = min(
            passing,
            key=lambda value: (
                float(value["risk"]),
                -float(value["threshold"]),
                float(value["c_value"]),
            ),
        )
        fallback_only = False
    else:
        chosen = {
            "c_value": float(min(c_values)),
            "threshold": 1.0,
            "risk": _risk(
                baseline_metrics,
                rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
            ),
            "metrics": {key: value for key, value in baseline_metrics.items() if key != "rows"},
        }
        fallback_only = True
    model = _fit_model(
        examples,
        c_value=float(chosen["c_value"]),
        rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
        threshold=float(chosen["threshold"]),
    )
    probabilities = model.probabilities(features)
    train_metrics = pose_backend_selection_metrics(
        examples, model.choose_learned(features)
    )
    importance = sorted(
        (
            {"feature": name, "weight": float(weight)}
            for name, weight in zip(POSE_BACKEND_GATE_FEATURE_NAMES, model.weights)
        ),
        key=lambda value: -abs(float(value["weight"])),
    )
    return model, {
        "fold_count": int(len(fold_values)),
        "positive_count_TARGET_ONLY": int(
            np.sum(
                _arrays(
                    examples,
                    rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
                )[1]
            )
        ),
        "baseline_metrics": {key: value for key, value in baseline_metrics.items() if key != "rows"},
        "candidates": candidates,
        "chosen": chosen,
        "fallback_only_due_to_failed_oof_gate": bool(fallback_only),
        "train_in_sample_metrics": {
            key: value for key, value in train_metrics.items() if key != "rows"
        },
        "train_probabilities": [float(value) for value in probabilities],
        "feature_importance": importance,
    }


def _strict_improvement(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> bool:
    return bool(
        float(candidate["median_translation_m"])
        < float(baseline["median_translation_m"])
        and float(candidate["p90_translation_m"])
        < float(baseline["p90_translation_m"])
        and float(candidate["median_rotation_deg"])
        <= float(baseline["median_rotation_deg"])
    )


def select_monotonic_pose_backend_policy(
    train_oof_examples: Sequence[PoseBackendSelectionExample],
    validation_examples: Sequence[PoseBackendSelectionExample],
    *,
    descriptor_score_thresholds: Sequence[float] = (0.07, 0.08, 0.09),
    baseline_inlier_ratio_thresholds: Sequence[float] = (0.45, 0.50),
    final_audit_loose_ratio_thresholds: Sequence[float] = (0.60, 0.70),
    maximum_pose_disagreement_log1p_values: Sequence[float] = (0.30, 0.50),
    rotation_equivalent_m_per_deg: float = 0.02,
) -> tuple[MonotonicPoseBackendPolicy, dict[str, object]]:
    if not train_oof_examples or not validation_examples:
        raise ValueError("monotonic policy selection requires train OOF and validation")
    train_features = np.asarray(
        [example.features for example in train_oof_examples], dtype=np.float64
    )
    validation_features = np.asarray(
        [example.features for example in validation_examples], dtype=np.float64
    )
    train_baseline = pose_backend_selection_metrics(
        train_oof_examples, np.zeros((len(train_oof_examples),), dtype=bool)
    )
    validation_baseline = pose_backend_selection_metrics(
        validation_examples, np.zeros((len(validation_examples),), dtype=bool)
    )
    candidates: list[dict[str, object]] = []
    for descriptor_threshold in descriptor_score_thresholds:
        for baseline_threshold in baseline_inlier_ratio_thresholds:
            for audit_threshold in final_audit_loose_ratio_thresholds:
                for disagreement_threshold in maximum_pose_disagreement_log1p_values:
                    policy = MonotonicPoseBackendPolicy(
                        descriptor_score_threshold=float(descriptor_threshold),
                        baseline_inlier_ratio_threshold=float(baseline_threshold),
                        final_audit_loose_ratio_threshold=float(audit_threshold),
                        maximum_pose_disagreement_log1p=float(disagreement_threshold),
                    )
                    train_metrics = pose_backend_selection_metrics(
                        train_oof_examples, policy.choose_learned(train_features)
                    )
                    validation_metrics = pose_backend_selection_metrics(
                        validation_examples,
                        policy.choose_learned(validation_features),
                    )
                    passes_train = _strict_improvement(train_metrics, train_baseline)
                    passes_validation = _strict_improvement(
                        validation_metrics, validation_baseline
                    )
                    candidates.append(
                        {
                            "policy": policy.to_dict(),
                            "passes_train_oof_gate": bool(passes_train),
                            "passes_validation_gate": bool(passes_validation),
                            "train_oof_metrics": {
                                key: value
                                for key, value in train_metrics.items()
                                if key != "rows"
                            },
                            "validation_metrics": {
                                key: value
                                for key, value in validation_metrics.items()
                                if key != "rows"
                            },
                            "combined_risk": float(
                                _risk(
                                    train_metrics,
                                    rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
                                )
                                + _risk(
                                    validation_metrics,
                                    rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
                                )
                            ),
                        }
                    )
    eligible = [
        candidate
        for candidate in candidates
        if bool(candidate["passes_train_oof_gate"])
        and bool(candidate["passes_validation_gate"])
    ]
    if not eligible:
        raise ValueError("no monotonic pose backend policy passes both development gates")
    chosen = min(
        eligible,
        key=lambda candidate: (
            float(candidate["combined_risk"]),
            -float(candidate["policy"]["descriptor_score_threshold"]),
            float(candidate["policy"]["baseline_inlier_ratio_threshold"]),
            -float(candidate["policy"]["final_audit_loose_ratio_threshold"]),
            float(candidate["policy"]["maximum_pose_disagreement_log1p"]),
        ),
    )
    policy = MonotonicPoseBackendPolicy.from_dict(chosen["policy"])
    return policy, {
        "candidate_count": int(len(candidates)),
        "eligible_count": int(len(eligible)),
        "selection_splits": ["train_oof", "validation"],
        "train_oof_baseline": {
            key: value for key, value in train_baseline.items() if key != "rows"
        },
        "validation_baseline": {
            key: value for key, value in validation_baseline.items() if key != "rows"
        },
        "chosen": chosen,
        "candidates": candidates,
    }
