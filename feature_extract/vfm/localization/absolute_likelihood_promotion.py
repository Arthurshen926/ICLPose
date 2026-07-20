"""Train-only abstention policy for absolute RGB likelihood pose proposals.

The policy deliberately has a small, target-free feature surface.  It chooses
between an immutable no-RGB baseline hypothesis and a hypothesis selected by a
fixed, candidate-specific RGB likelihood.  Pose targets are used only after
the frozen scores have been produced, to fit and audit the abstention rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


PROMOTION_FEATURE_SCHEMA = "rgb_spatial_mode_consensus_v2"

# The first eight terms describe the selected RGB-mode hypothesis itself.  The
# remaining terms deliberately compare its rank across precomputed robust
# aggregations.  They are target-free: all of them are available before any
# pose target is joined and prevent a single mean aggregation from being the
# only reason to replace the immutable baseline.
FEATURE_NAMES = (
    "optional_gain_over_baseline_mean",
    "optional_mean_top1_top2_margin",
    "optional_spatial_median_gap_at_top1",
    "optional_spatial_median_rank_fraction",
    "optional_evidence_coverage",
    "optional_effective_point_fraction",
    "optional_materialized_point_fraction",
    "optional_log_materialized_views_per_point",
    "baseline_rank_under_optional_mean",
    "optional_max_profile_rank_fraction",
    "optional_mean_profile_rank_fraction",
    "optional_profile_top1_fraction",
    "optional_beats_baseline_profile_fraction",
    "optional_mean_profile_margin",
    "optional_min_profile_margin",
)
NO_PROMOTION_THRESHOLD = 1.000001


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    output[~positive] = exp_logits / (1.0 + exp_logits)
    return output


@dataclass(frozen=True)
class AbsoluteLikelihoodPromotionExample:
    """One immutable-baseline versus RGB-optional choice for a query group."""

    query_id: str
    split_name: str
    evaluation_label: str
    baseline_hypothesis_index: int
    optional_hypothesis_index: int
    features: tuple[float, ...]
    baseline_translation_m: float
    baseline_rotation_deg: float
    optional_translation_m: float
    optional_rotation_deg: float

    def __post_init__(self) -> None:
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("absolute-likelihood promotion feature schema mismatch")
        values = np.asarray(self.features, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("absolute-likelihood promotion features must be finite")
        errors = np.asarray(
            (
                self.baseline_translation_m,
                self.baseline_rotation_deg,
                self.optional_translation_m,
                self.optional_rotation_deg,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
            raise ValueError("pose-target audit errors must be finite and non-negative")

    @property
    def changed_hypothesis(self) -> bool:
        return int(self.baseline_hypothesis_index) != int(self.optional_hypothesis_index)

    @property
    def target_beneficial(self) -> bool:
        """Conservative post-hoc label; never available to inference.

        Small translation wins count only when rotation does not regress.  A
        catastrophic baseline rescue is also useful, provided it does not
        introduce a large rotation regression.
        """

        if not self.changed_hypothesis:
            return False
        if (
            float(self.baseline_translation_m) > 1.0
            and float(self.optional_translation_m) <= 1.0
            and float(self.optional_rotation_deg)
            <= float(self.baseline_rotation_deg) + 0.5
        ):
            return True
        return bool(
            (
                float(self.baseline_translation_m)
                - float(self.optional_translation_m)
                >= 0.02
                and float(self.optional_rotation_deg)
                <= float(self.baseline_rotation_deg) + 0.05
            )
            or (
                float(self.baseline_rotation_deg)
                - float(self.optional_rotation_deg)
                >= 0.10
                and float(self.optional_translation_m)
                <= float(self.baseline_translation_m) + 0.01
            )
        )


@dataclass(frozen=True)
class AbsoluteLikelihoodPromotionModel:
    """A serializable target-free probability model."""

    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("absolute-likelihood promotion model feature mismatch")
        expected = len(FEATURE_NAMES)
        if any(
            len(values) != expected
            for values in (self.feature_mean, self.feature_scale, self.coefficients)
        ):
            raise ValueError("absolute-likelihood promotion model vector mismatch")
        if np.any(np.asarray(self.feature_scale, dtype=np.float64) <= 0.0):
            raise ValueError("absolute-likelihood promotion scales must be positive")

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
        standardized = (
            values - np.asarray(self.feature_mean, dtype=np.float64)[None]
        ) / np.asarray(self.feature_scale, dtype=np.float64)[None]
        return _sigmoid(
            standardized @ np.asarray(self.coefficients, dtype=np.float64)
            + float(self.intercept)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "absolute_likelihood_promotion_model_v2",
            "feature_schema": PROMOTION_FEATURE_SCHEMA,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
        }


def feature_matrix(examples: Sequence[AbsoluteLikelihoodPromotionExample]) -> np.ndarray:
    if not examples:
        raise ValueError("at least one promotion example is required")
    return np.asarray([example.features for example in examples], dtype=np.float64)


def target_labels(examples: Sequence[AbsoluteLikelihoodPromotionExample]) -> np.ndarray:
    return np.asarray(
        [bool(example.target_beneficial) for example in examples], dtype=np.int64
    )


def _fit_model(
    features: np.ndarray, labels: np.ndarray, *, c_value: float
) -> AbsoluteLikelihoodPromotionModel | None:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("promotion model feature matrix has the wrong shape")
    if len(values) != len(targets):
        raise ValueError("promotion feature/label rows differ")
    if len(np.unique(targets)) < 2:
        return None
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=0,
    )
    model.fit((values - mean[None]) / scale[None], targets)
    return AbsoluteLikelihoodPromotionModel(
        feature_names=FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in mean.tolist()),
        feature_scale=tuple(float(value) for value in scale.tolist()),
        coefficients=tuple(float(value) for value in model.coef_[0].tolist()),
        intercept=float(model.intercept_[0]),
    )


def crossfit_probabilities(
    examples: Sequence[AbsoluteLikelihoodPromotionExample],
    *,
    fold_count: int,
    c_value: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return query-grouped OOF probabilities without reading held-out labels."""

    if int(fold_count) < 2:
        raise ValueError("fold_count must be at least two")
    features = feature_matrix(examples)
    labels = target_labels(examples)
    groups = np.asarray(
        [f"{example.query_id}\x1f{example.evaluation_label}" for example in examples],
        dtype=object,
    )
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("query-grouped crossfit needs at least two query groups")
    n_splits = min(int(fold_count), int(len(unique_groups)))
    probabilities = np.zeros((len(examples),), dtype=np.float64)
    fold_assignments = np.full((len(examples),), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (fit_rows, heldout_rows) in enumerate(
        splitter.split(features, labels, groups)
    ):
        model = _fit_model(features[fit_rows], labels[fit_rows], c_value=float(c_value))
        if model is not None:
            probabilities[heldout_rows] = model.probabilities(features[heldout_rows])
        fold_assignments[heldout_rows] = int(fold)
    if np.any(fold_assignments < 0):
        raise RuntimeError("crossfit did not assign every promotion example")
    return probabilities, tuple(int(value) for value in fold_assignments.tolist())


def fit_final_model(
    examples: Sequence[AbsoluteLikelihoodPromotionExample], *, c_value: float
) -> AbsoluteLikelihoodPromotionModel | None:
    return _fit_model(feature_matrix(examples), target_labels(examples), c_value=c_value)


def promotion_decisions(
    examples: Sequence[AbsoluteLikelihoodPromotionExample],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) != len(examples):
        raise ValueError("promotion probability count differs from examples")
    if not np.all(np.isfinite(values)):
        raise ValueError("promotion probabilities must be finite")
    return np.asarray(
        [
            bool(example.changed_hypothesis and probability >= float(threshold))
            for example, probability in zip(examples, values.tolist())
        ],
        dtype=bool,
    )


def _pose_summary(
    translations: np.ndarray, rotations: np.ndarray
) -> dict[str, object]:
    if translations.ndim != 1 or rotations.shape != translations.shape:
        raise ValueError("pose summary arrays are not aligned")
    return {
        "query_count": int(len(translations)),
        "median_translation_m": float(np.median(translations)),
        "p90_translation_m": float(np.quantile(translations, 0.9)),
        "median_rotation_deg": float(np.median(rotations)),
        "p90_rotation_deg": float(np.quantile(rotations, 0.9)),
        "catastrophic_1m_count": int(np.count_nonzero(translations > 1.0)),
        "recall_5cm_5deg": float(
            np.mean((translations <= 0.05) & (rotations <= 5.0))
        ),
        "recall_10cm_5deg": float(
            np.mean((translations <= 0.10) & (rotations <= 5.0))
        ),
        "recall_25cm_2deg": float(
            np.mean((translations <= 0.25) & (rotations <= 2.0))
        ),
    }


def policy_metrics(
    examples: Sequence[AbsoluteLikelihoodPromotionExample], decisions: Sequence[bool]
) -> dict[str, object]:
    if not examples:
        raise ValueError("cannot audit an empty promotion set")
    selected = np.asarray(decisions, dtype=bool).reshape(-1)
    if len(selected) != len(examples):
        raise ValueError("promotion decision count differs from examples")
    baseline_translation = np.asarray(
        [example.baseline_translation_m for example in examples], dtype=np.float64
    )
    baseline_rotation = np.asarray(
        [example.baseline_rotation_deg for example in examples], dtype=np.float64
    )
    optional_translation = np.asarray(
        [example.optional_translation_m for example in examples], dtype=np.float64
    )
    optional_rotation = np.asarray(
        [example.optional_rotation_deg for example in examples], dtype=np.float64
    )
    selected_translation = np.where(selected, optional_translation, baseline_translation)
    selected_rotation = np.where(selected, optional_rotation, baseline_rotation)
    delta = selected_translation - baseline_translation
    tolerance = 1e-12
    promoted_targets = np.asarray(
        [bool(example.target_beneficial) for example in examples], dtype=bool
    )
    return {
        "baseline": _pose_summary(baseline_translation, baseline_rotation),
        "selected": _pose_summary(selected_translation, selected_rotation),
        "promotion": {
            "count": int(np.count_nonzero(selected)),
            "fallback_count": int(len(selected) - np.count_nonzero(selected)),
            "beneficial_count_TARGET_ONLY": int(
                np.count_nonzero(selected & promoted_targets)
            ),
            "precision_TARGET_ONLY": (
                None
                if not np.count_nonzero(selected)
                else float(
                    np.count_nonzero(selected & promoted_targets)
                    / np.count_nonzero(selected)
                )
            ),
            "new_catastrophic_count": int(
                np.count_nonzero(
                    selected
                    & (baseline_translation <= 1.0)
                    & (optional_translation > 1.0)
                )
            ),
        },
        "paired": {
            "translation_wins": int(np.count_nonzero(delta < -tolerance)),
            "translation_losses": int(np.count_nonzero(delta > tolerance)),
            "translation_ties": int(np.count_nonzero(np.abs(delta) <= tolerance)),
            "median_translation_delta_m": float(np.median(delta)),
            "median_rotation_delta_deg": float(
                np.median(selected_rotation - baseline_rotation)
            ),
        },
    }


def tail_safe_gate(metrics: Mapping[str, object]) -> dict[str, object]:
    baseline = metrics.get("baseline")
    selected = metrics.get("selected")
    promotion = metrics.get("promotion")
    paired = metrics.get("paired")
    if not all(
        isinstance(value, Mapping) for value in (baseline, selected, promotion, paired)
    ):
        raise ValueError("promotion metrics are incomplete")
    checks = {
        "effective_promotion": int(promotion["count"]) > 0,
        "median_translation_strictly_improved": float(
            selected["median_translation_m"]
        )
        < float(baseline["median_translation_m"]),
        "p90_translation_not_worse": float(selected["p90_translation_m"])
        <= float(baseline["p90_translation_m"]),
        "median_rotation_not_worse": float(selected["median_rotation_deg"])
        <= float(baseline["median_rotation_deg"]),
        "p90_rotation_not_worse": float(selected["p90_rotation_deg"])
        <= float(baseline["p90_rotation_deg"]),
        "catastrophic_1m_not_worse": int(selected["catastrophic_1m_count"])
        <= int(baseline["catastrophic_1m_count"]),
        "no_new_catastrophic_promotion": int(promotion["new_catastrophic_count"]) == 0,
        "paired_wins_greater_than_losses": int(paired["translation_wins"])
        > int(paired["translation_losses"]),
    }
    return {"checks": checks, "passes": bool(all(checks.values()))}


def select_oof_tail_safe_threshold(
    examples: Sequence[AbsoluteLikelihoodPromotionExample],
    probabilities: Sequence[float],
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Select a frozen threshold solely from grouped OOF train predictions."""

    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) != len(examples):
        raise ValueError("OOF probability count differs from examples")
    if not np.all(np.isfinite(values)):
        raise ValueError("OOF probabilities must be finite")
    thresholds = [NO_PROMOTION_THRESHOLD]
    thresholds.extend(sorted(set(float(value) for value in values), reverse=True))
    candidates: list[tuple[float, np.ndarray, dict[str, object], dict[str, object]]] = []
    for threshold in thresholds:
        decisions = promotion_decisions(examples, values, threshold=float(threshold))
        metrics = policy_metrics(examples, decisions)
        gate = tail_safe_gate(metrics)
        if bool(gate["passes"]):
            candidates.append((float(threshold), decisions, metrics, gate))
    if not candidates:
        decisions = promotion_decisions(
            examples, values, threshold=NO_PROMOTION_THRESHOLD
        )
        metrics = policy_metrics(examples, decisions)
        return NO_PROMOTION_THRESHOLD, decisions, {
            "metrics": metrics,
            "gate": tail_safe_gate(metrics),
            "eligible_threshold_count": 0,
            "effective_promotion": False,
        }
    threshold, decisions, metrics, gate = max(
        candidates,
        key=lambda value: (
            float(value[2]["selected"]["recall_10cm_5deg"]),
            float(value[2]["selected"]["recall_25cm_2deg"]),
            -float(value[2]["selected"]["median_translation_m"]),
            -float(value[2]["selected"]["p90_translation_m"]),
            int(value[2]["paired"]["translation_wins"])
            - int(value[2]["paired"]["translation_losses"]),
            int(value[2]["promotion"]["count"]),
        ),
    )
    return threshold, decisions, {
        "metrics": metrics,
        "gate": gate,
        "eligible_threshold_count": int(len(candidates)),
        "effective_promotion": True,
    }
