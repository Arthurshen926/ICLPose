"""Train-only calibration helpers for independent cross-fit audit gains.

The pose scorer emits a target-free audit likelihood delta between a frozen
optional hypothesis and the immutable source pose.  This module turns that
single scalar into a conservative abstention policy only after the scores have
been frozen and pose targets have been joined for train-only calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


POLICY_FORMAT = "crossfit_audit_gain_promotion_policy_v1"


@dataclass(frozen=True)
class CrossfitAuditPromotionExample:
    """One immutable source versus rank-frozen optional pose decision."""

    query_id: str
    split_name: str
    evaluation_label: str
    source_hypothesis_index: int
    optional_hypothesis_index: int
    audit_score_delta: float
    eligible_without_audit_gain: bool
    source_translation_m: float
    source_rotation_deg: float
    optional_translation_m: float
    optional_rotation_deg: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.audit_score_delta,
                self.source_translation_m,
                self.source_rotation_deg,
                self.optional_translation_m,
                self.optional_rotation_deg,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("audit-promotion values must be finite")
        if np.any(values[1:] < 0.0):
            raise ValueError("pose-target errors must be non-negative")

    @property
    def changed_hypothesis(self) -> bool:
        return int(self.source_hypothesis_index) != int(self.optional_hypothesis_index)

    @property
    def sequence_group(self) -> str:
        """Keep adjacent frames from one capture sequence in one OOF fold."""

        parent = PurePosixPath(str(self.query_id)).parent.as_posix()
        return (
            f"{self.evaluation_label}\x1f{parent}"
            if parent not in ("", ".")
            else f"{self.evaluation_label}\x1f{self.query_id}"
        )


def promotion_decisions(
    examples: Sequence[CrossfitAuditPromotionExample],
    *,
    minimum_audit_gain: float | None,
) -> np.ndarray:
    """Apply a target-free scalar threshold without changing identities."""

    if minimum_audit_gain is None:
        return np.zeros((len(examples),), dtype=bool)
    threshold = float(minimum_audit_gain)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum audit gain must be finite and non-negative")
    return np.asarray(
        [
            bool(
                example.changed_hypothesis
                and example.eligible_without_audit_gain
                and float(example.audit_score_delta) >= threshold
            )
            for example in examples
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
    examples: Sequence[CrossfitAuditPromotionExample], decisions: Sequence[bool]
) -> dict[str, object]:
    """Post-hoc target audit for a frozen promote/fallback decision vector."""

    if not examples:
        raise ValueError("cannot audit an empty promotion set")
    selected = np.asarray(decisions, dtype=bool).reshape(-1)
    if len(selected) != len(examples):
        raise ValueError("promotion decision count differs from examples")
    source_translation = np.asarray(
        [example.source_translation_m for example in examples], dtype=np.float64
    )
    source_rotation = np.asarray(
        [example.source_rotation_deg for example in examples], dtype=np.float64
    )
    optional_translation = np.asarray(
        [example.optional_translation_m for example in examples], dtype=np.float64
    )
    optional_rotation = np.asarray(
        [example.optional_rotation_deg for example in examples], dtype=np.float64
    )
    selected_translation = np.where(
        selected, optional_translation, source_translation
    )
    selected_rotation = np.where(selected, optional_rotation, source_rotation)
    translation_delta = selected_translation - source_translation
    tolerance = 1e-12
    return {
        "baseline": _pose_summary(source_translation, source_rotation),
        "selected": _pose_summary(selected_translation, selected_rotation),
        "promotion": {
            "count": int(np.count_nonzero(selected)),
            "fallback_count": int(len(selected) - np.count_nonzero(selected)),
            "new_catastrophic_count": int(
                np.count_nonzero(
                    selected
                    & (source_translation <= 1.0)
                    & (optional_translation > 1.0)
                )
            ),
        },
        "paired": {
            "translation_wins": int(
                np.count_nonzero(translation_delta < -tolerance)
            ),
            "translation_losses": int(
                np.count_nonzero(translation_delta > tolerance)
            ),
            "translation_ties": int(
                np.count_nonzero(np.abs(translation_delta) <= tolerance)
            ),
            "median_translation_delta_m": float(np.median(translation_delta)),
            "median_rotation_delta_deg": float(
                np.median(selected_rotation - source_rotation)
            ),
        },
    }


def tail_safe_gate(
    metrics: Mapping[str, object], *, minimum_promotion_count: int
) -> dict[str, object]:
    """Require a genuine gain without permitting a new catastrophic tail."""

    if int(minimum_promotion_count) <= 0:
        raise ValueError("minimum promotion count must be positive")
    baseline = metrics.get("baseline")
    selected = metrics.get("selected")
    promotion = metrics.get("promotion")
    paired = metrics.get("paired")
    if not all(
        isinstance(value, Mapping) for value in (baseline, selected, promotion, paired)
    ):
        raise ValueError("promotion metrics are incomplete")
    checks = {
        "minimum_effective_promotions": int(promotion["count"])
        >= int(minimum_promotion_count),
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
        "no_new_catastrophic_promotion": int(promotion["new_catastrophic_count"])
        == 0,
        "paired_wins_greater_than_losses": int(paired["translation_wins"])
        > int(paired["translation_losses"]),
    }
    return {"checks": checks, "passes": bool(all(checks.values()))}


def select_tail_safe_threshold(
    examples: Sequence[CrossfitAuditPromotionExample],
    *,
    minimum_promotion_count: int,
) -> tuple[float | None, np.ndarray, dict[str, object]]:
    """Select a scalar threshold using only the supplied calibration examples."""

    if not examples:
        raise ValueError("at least one audit-promotion example is required")
    thresholds = {0.0}
    thresholds.update(
        max(0.0, float(example.audit_score_delta))
        for example in examples
        if example.changed_hypothesis and example.eligible_without_audit_gain
    )
    candidates: list[
        tuple[float, np.ndarray, dict[str, object], dict[str, object]]
    ] = []
    for threshold in sorted(thresholds, reverse=True):
        decisions = promotion_decisions(examples, minimum_audit_gain=float(threshold))
        metrics = policy_metrics(examples, decisions)
        gate = tail_safe_gate(
            metrics, minimum_promotion_count=int(minimum_promotion_count)
        )
        if bool(gate["passes"]):
            candidates.append((float(threshold), decisions, metrics, gate))
    if not candidates:
        decisions = promotion_decisions(examples, minimum_audit_gain=None)
        metrics = policy_metrics(examples, decisions)
        return None, decisions, {
            "metrics": metrics,
            "gate": tail_safe_gate(
                metrics, minimum_promotion_count=int(minimum_promotion_count)
            ),
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


def sequence_grouped_oof_decisions(
    examples: Sequence[CrossfitAuditPromotionExample],
    *,
    fold_count: int,
    minimum_promotion_count: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[float | None, ...], dict[str, object]]:
    """Evaluate threshold fitting on mutually exclusive capture sequences."""

    if int(fold_count) < 2:
        raise ValueError("fold count must be at least two")
    if int(minimum_promotion_count) <= 0:
        raise ValueError("minimum promotion count must be positive")
    if not examples:
        raise ValueError("at least one audit-promotion example is required")
    groups = np.asarray([example.sequence_group for example in examples], dtype=object)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("sequence-grouped OOF needs at least two capture groups")
    splitter = GroupKFold(n_splits=min(int(fold_count), int(len(unique_groups))))
    indices = np.arange(len(examples), dtype=np.int64)
    decisions = np.zeros((len(examples),), dtype=bool)
    assignments = np.full((len(examples),), -1, dtype=np.int64)
    thresholds: list[float | None] = []
    folds: list[dict[str, object]] = []
    for fold, (fit_rows, heldout_rows) in enumerate(splitter.split(indices, groups=groups)):
        fit = [examples[int(row)] for row in fit_rows.tolist()]
        heldout = [examples[int(row)] for row in heldout_rows.tolist()]
        scaled_minimum = max(
            1,
            int(
                np.ceil(
                    float(minimum_promotion_count) * len(fit) / len(examples)
                )
            ),
        )
        threshold, _fit_decisions, fit_audit = select_tail_safe_threshold(
            fit, minimum_promotion_count=scaled_minimum
        )
        heldout_decisions = promotion_decisions(
            heldout, minimum_audit_gain=threshold
        )
        decisions[heldout_rows] = heldout_decisions
        assignments[heldout_rows] = int(fold)
        thresholds.append(threshold)
        folds.append(
            {
                "fold": int(fold),
                "fit_sequence_groups": sorted(
                    set(groups[fit_rows].astype(str).tolist())
                ),
                "heldout_sequence_groups": sorted(
                    set(groups[heldout_rows].astype(str).tolist())
                ),
                "fit_minimum_promotion_count": int(scaled_minimum),
                "selected_minimum_audit_gain": threshold,
                "fit": fit_audit,
                "heldout_metrics_TARGET_ONLY": policy_metrics(
                    heldout, heldout_decisions
                ),
            }
        )
    if np.any(assignments < 0):
        raise RuntimeError("sequence-grouped OOF did not assign every example")
    metrics = policy_metrics(examples, decisions)
    return decisions, tuple(int(value) for value in assignments.tolist()), tuple(
        thresholds
    ), {
        "fold_count": int(len(thresholds)),
        "minimum_promotion_count": int(minimum_promotion_count),
        "folds": folds,
        "metrics": metrics,
        "gate": tail_safe_gate(
            metrics, minimum_promotion_count=int(minimum_promotion_count)
        ),
    }
