"""Deterministic linear baselines for local landmark assignment evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


IDENTITY_STRATEGIES = (
    "coarse_prototype",
    "all_support_best",
    "all_support_mean",
    "all_support_top2_mean",
    "all_support_top4_mean",
    "alike_support_best",
    "alike_support_mean",
    "alike_support_top2_mean",
    "alike_support_top4_mean",
    "alike_support_logmeanexp_tau0p05",
)

NO_MATCH_STRATEGIES = (
    "coarse_prototype",
    "all_support_top2_mean",
    "alike_support_best",
    "alike_support_mean",
    "alike_support_top2_mean",
    "alike_support_top4_mean",
    "alike_support_logmeanexp_tau0p05",
)


def _candidate_shape(payload: Mapping[str, np.ndarray]) -> tuple[int, int]:
    tracks = np.asarray(payload["candidate_track_ids"])
    if tracks.ndim != 2 or tracks.shape[1] < 2:
        raise ValueError("candidate_track_ids must have shape (N, L) with L >= 2")
    return int(tracks.shape[0]), int(tracks.shape[1])


def _strategy(payload: Mapping[str, np.ndarray], name: str, shape: tuple[int, int]) -> np.ndarray:
    key = f"strategy__{name}"
    if key not in payload:
        raise ValueError(f"assignment probe is missing strategy evidence: {name}")
    values = np.asarray(payload[key], dtype=np.float32)
    if values.shape != shape:
        raise ValueError(f"strategy evidence has wrong shape: {name}")
    return values


def build_identity_candidate_features(
    payload: Mapping[str, np.ndarray],
    *,
    strategy_names: Sequence[str] = IDENTITY_STRATEGIES,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build deployable per-candidate evidence without any GT-derived feature."""

    shape = _candidate_shape(payload)
    tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
    valid = tracks >= 0
    feature_values: list[np.ndarray] = []
    feature_names: list[str] = []
    for name in strategy_names:
        values = _strategy(payload, str(name), shape)
        finite_valid = valid & np.isfinite(values)
        safe = np.where(finite_valid, values, -np.inf)
        row_max = np.max(safe, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = 0.0
        raw = np.where(finite_valid, values, 0.0).astype(np.float32, copy=False)
        gap = np.where(finite_valid, row_max - values, 0.0).astype(np.float32, copy=False)
        feature_values.extend((raw, gap))
        feature_names.extend((str(name), f"{name}_gap_to_row_max"))

    top_l = int(shape[1])
    rank = np.broadcast_to(
        np.arange(top_l, dtype=np.float32)[None, :] / max(float(top_l - 1), 1.0),
        shape,
    )
    maplet_counts = np.asarray(payload["maplet_support_counts"], dtype=np.float32)
    support_counts = np.asarray(payload["all_support_counts"], dtype=np.float32)
    prototype_ids = np.asarray(payload["candidate_prototype_ids"], dtype=np.float32)
    for name, values in (
        ("proposal_rank_normalized", rank),
        ("log1p_maplet_support_count", np.log1p(np.maximum(maplet_counts, 0.0))),
        ("log1p_all_support_count", np.log1p(np.maximum(support_counts, 0.0))),
        ("prototype_id", prototype_ids),
    ):
        if values.shape != shape:
            raise ValueError(f"candidate evidence has wrong shape: {name}")
        feature_values.append(np.where(valid, values, 0.0).astype(np.float32, copy=False))
        feature_names.append(name)
    features = np.stack(feature_values, axis=2).astype(np.float32, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError("identity candidate features contain non-finite values")
    return features, tuple(feature_names)


def build_no_match_features(
    payload: Mapping[str, np.ndarray],
    query_detector_scores: np.ndarray,
    *,
    strategy_names: Sequence[str] = NO_MATCH_STRATEGIES,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize each proposal row for learned correct-track-absent probability."""

    shape = _candidate_shape(payload)
    tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
    valid = tracks >= 0
    feature_values: list[np.ndarray] = []
    feature_names: list[str] = []
    for name in strategy_names:
        values = _strategy(payload, str(name), shape)
        safe = np.where(valid & np.isfinite(values), values, -np.inf)
        sorted_values = np.sort(safe, axis=1)[:, ::-1]
        top1 = sorted_values[:, 0]
        top2 = sorted_values[:, 1]
        finite_values = np.where(valid & np.isfinite(values), values, np.nan)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(finite_values, axis=1)
            std = np.nanstd(finite_values, axis=1)
        rows = (
            np.nan_to_num(top1, nan=0.0, neginf=0.0, posinf=0.0),
            np.nan_to_num(top1 - top2, nan=0.0, neginf=0.0, posinf=0.0),
            np.nan_to_num(mean, nan=0.0),
            np.nan_to_num(std, nan=0.0),
        )
        feature_values.extend(rows)
        feature_names.extend(
            (f"{name}_top1", f"{name}_top1_gap", f"{name}_mean", f"{name}_std")
        )
    detector = np.asarray(query_detector_scores, dtype=np.float32).reshape(-1)
    if detector.shape[0] != shape[0]:
        raise ValueError("query detector scores do not match candidate rows")
    detector = np.nan_to_num(detector, nan=0.0, neginf=0.0, posinf=1.0)
    detector = np.clip(detector, 0.0, 1.0)
    feature_values.extend((detector, np.log10(np.maximum(detector, 1e-8))))
    feature_names.extend(("query_detector_score", "query_detector_log10_score"))
    features = np.stack(feature_values, axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError("no-match features contain non-finite values")
    return features, tuple(feature_names)


@dataclass(frozen=True)
class LinearLogitModel:
    """Portable NumPy inference form of a binary logistic regression model."""

    coefficients: np.ndarray
    intercept: float
    feature_names: tuple[str, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64).reshape(-1)
        names = tuple(str(name) for name in self.feature_names)
        if coefficients.shape[0] != len(names):
            raise ValueError("coefficient and feature-name counts differ")
        if not np.all(np.isfinite(coefficients)) or not np.isfinite(float(self.intercept)):
            raise ValueError("linear model parameters must be finite")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != self.coefficients.shape[0]:
            raise ValueError("feature dimension does not match linear model")
        return (values @ self.coefficients + float(self.intercept)).astype(np.float32)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        logits = self.decision_function(features).astype(np.float64)
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_values = np.exp(logits[~positive])
        probabilities[~positive] = exp_values / (1.0 + exp_values)
        return probabilities.astype(np.float32)

    def save(self, path: Path) -> None:
        payload = {
            "format_version": 1,
            "feature_names": list(self.feature_names),
            **dict(self.metadata),
        }
        np.savez(
            Path(path),
            coefficients=self.coefficients.astype(np.float64),
            intercept=np.asarray(float(self.intercept), dtype=np.float64),
            metadata_json=np.asarray(json.dumps(payload, sort_keys=True), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: Path) -> "LinearLogitModel":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            coefficients = np.asarray(data["coefficients"], dtype=np.float64)
            intercept = float(data["intercept"].item())
        names = tuple(str(name) for name in metadata.pop("feature_names"))
        if int(metadata.pop("format_version", -1)) != 1:
            raise ValueError("unsupported linear assignment model format")
        return cls(coefficients, intercept, names, metadata)


def selective_switch_scores(
    reranker_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    margin_threshold: float,
    valid_mask: np.ndarray | None = None,
    preserve_baseline_row_confidence: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the baseline identity unless a disagreeing reranker is decisive."""

    reranker = np.asarray(reranker_scores, dtype=np.float32)
    baseline = np.asarray(baseline_scores, dtype=np.float32)
    if reranker.shape != baseline.shape or reranker.ndim != 2 or reranker.shape[1] < 2:
        raise ValueError("reranker and baseline scores must have matching shape (N, L>=2)")
    valid = np.ones_like(reranker, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != reranker.shape:
        raise ValueError("valid mask shape does not match scores")
    reranker_valid = np.where(valid & np.isfinite(reranker), reranker, -np.inf)
    baseline_valid = np.where(valid & np.isfinite(baseline), baseline, -np.inf)
    order = np.argsort(-reranker_valid, axis=1, kind="stable")
    reranker_choice = order[:, 0]
    first = reranker_valid[np.arange(len(order)), order[:, 0]]
    second = reranker_valid[np.arange(len(order)), order[:, 1]]
    reranker_margin = np.full((len(order),), -np.inf, dtype=np.float32)
    both_finite = np.isfinite(first) & np.isfinite(second)
    reranker_margin[both_finite] = first[both_finite] - second[both_finite]
    reranker_margin[np.isfinite(first) & ~np.isfinite(second)] = np.inf
    baseline_choice = np.argmax(baseline_valid, axis=1)
    switch = (reranker_choice != baseline_choice) & (reranker_margin >= float(margin_threshold))
    selected = baseline_choice.copy()
    selected[switch] = reranker_choice[switch]
    resolved = np.full_like(baseline_valid, -np.inf, dtype=np.float32)
    if bool(preserve_baseline_row_confidence):
        selected_scores = np.max(baseline_valid, axis=1)
    else:
        selected_scores = baseline_valid[np.arange(len(selected)), selected]
    resolved[np.arange(len(selected)), selected] = selected_scores
    return selected.astype(np.int64), resolved, switch, reranker_margin.astype(np.float32)


def selective_baseline_gain_switch_scores(
    reranker_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    min_gain: float,
    valid_mask: np.ndarray | None = None,
    baseline_validity_scores: np.ndarray | None = None,
    max_baseline_validity: float | None = None,
    preserve_baseline_row_confidence: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Switch only when the reranker explicitly beats the baseline candidate."""

    reranker = np.asarray(reranker_scores, dtype=np.float32)
    baseline = np.asarray(baseline_scores, dtype=np.float32)
    if reranker.shape != baseline.shape or reranker.ndim != 2 or reranker.shape[1] < 2:
        raise ValueError("reranker and baseline scores must have matching shape (N, L>=2)")
    if not np.isfinite(float(min_gain)) or float(min_gain) < 0.0:
        raise ValueError("min_gain must be finite and non-negative")
    valid = (
        np.ones_like(reranker, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != reranker.shape:
        raise ValueError("valid mask shape does not match scores")
    if (baseline_validity_scores is None) != (max_baseline_validity is None):
        raise ValueError(
            "baseline validity scores and maximum threshold must be provided together"
        )
    validity = None
    if baseline_validity_scores is not None:
        validity = np.asarray(baseline_validity_scores, dtype=np.float32)
        if validity.shape != reranker.shape:
            raise ValueError("baseline validity scores do not match reranker scores")
        if (
            not np.isfinite(float(max_baseline_validity))
            or not 0.0 <= float(max_baseline_validity) <= 1.0
        ):
            raise ValueError("max_baseline_validity must be in [0, 1]")

    reranker_valid = np.where(valid & np.isfinite(reranker), reranker, -np.inf)
    baseline_valid = np.where(valid & np.isfinite(baseline), baseline, -np.inf)
    reranker_choice = np.argmax(reranker_valid, axis=1)
    baseline_choice = np.argmax(baseline_valid, axis=1)
    row_indices = np.arange(len(reranker), dtype=np.int64)
    best_score = reranker_valid[row_indices, reranker_choice]
    baseline_candidate_score = reranker_valid[row_indices, baseline_choice]
    gain = np.full((len(reranker),), -np.inf, dtype=np.float32)
    finite_gain = np.isfinite(best_score) & np.isfinite(baseline_candidate_score)
    gain[finite_gain] = best_score[finite_gain] - baseline_candidate_score[finite_gain]
    switch = (
        (reranker_choice != baseline_choice)
        & finite_gain
        & (gain >= float(min_gain))
    )
    if validity is not None:
        baseline_validity = validity[row_indices, baseline_choice]
        switch &= np.isfinite(baseline_validity) & (
            baseline_validity <= float(max_baseline_validity)
        )

    selected = baseline_choice.copy()
    selected[switch] = reranker_choice[switch]
    resolved = np.full_like(baseline_valid, -np.inf, dtype=np.float32)
    if bool(preserve_baseline_row_confidence):
        selected_scores = np.max(baseline_valid, axis=1)
    else:
        selected_scores = baseline_valid[row_indices, selected]
    resolved[row_indices, selected] = selected_scores
    return selected.astype(np.int64), resolved, switch, gain.astype(np.float32)


def resolve_rescue_policy_scores(
    candidate_probabilities: np.ndarray,
    keep_probabilities: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    action_margin_threshold: float = 0.0,
    valid_mask: np.ndarray | None = None,
    preserve_baseline_alternatives: bool = False,
    lock_rescue_updates: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve KEEP versus a non-baseline rescue candidate after probability fusion.

    ``preserve_baseline_alternatives`` keeps the complete baseline score row for
    downstream whole-image assignment. A rescue action swaps the baseline and
    rescue ranks by default. With ``lock_rescue_updates``, only switched rows
    become single-candidate rows; KEEP rows remain exactly equal to baseline.
    """

    candidate = np.asarray(candidate_probabilities, dtype=np.float32)
    keep = np.asarray(keep_probabilities, dtype=np.float32).reshape(-1)
    baseline = np.asarray(baseline_scores, dtype=np.float32)
    if candidate.shape != baseline.shape or candidate.ndim != 2 or candidate.shape[1] < 2:
        raise ValueError(
            "candidate probabilities and baseline scores must have matching shape (N, L>=2)"
        )
    if keep.shape != (candidate.shape[0],):
        raise ValueError("keep probabilities must contain one value per candidate row")
    if bool(lock_rescue_updates) and not bool(preserve_baseline_alternatives):
        raise ValueError(
            "lock_rescue_updates requires preserve_baseline_alternatives"
        )
    threshold = float(action_margin_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("action margin threshold must be finite and non-negative")
    valid = (
        np.ones_like(candidate, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != candidate.shape:
        raise ValueError("valid mask shape does not match rescue probabilities")
    candidate_valid = valid & np.isfinite(candidate)
    baseline_valid = valid & np.isfinite(baseline)
    if np.any(np.sum(baseline_valid, axis=1) <= 0):
        raise ValueError("every rescue row requires a finite baseline candidate")
    if not np.all(np.isfinite(keep)):
        raise ValueError("keep probabilities must be finite")

    row_indices = np.arange(len(candidate), dtype=np.int64)
    safe_baseline = np.where(baseline_valid, baseline, -np.inf)
    baseline_choice = np.argmax(safe_baseline, axis=1)
    rescue_valid = candidate_valid.copy()
    rescue_valid[row_indices, baseline_choice] = False
    safe_candidate = np.where(rescue_valid, candidate, -np.inf)
    rescue_choice = np.argmax(safe_candidate, axis=1)
    rescue_probability = safe_candidate[row_indices, rescue_choice]
    action_margin = np.full((len(candidate),), -np.inf, dtype=np.float32)
    finite_rescue = np.isfinite(rescue_probability)
    action_margin[finite_rescue] = (
        rescue_probability[finite_rescue] - keep[finite_rescue]
    ).astype(np.float32)
    switch = finite_rescue & (action_margin > threshold)
    selected = baseline_choice.copy()
    selected[switch] = rescue_choice[switch]

    baseline_row_confidence = np.max(safe_baseline, axis=1)
    if bool(preserve_baseline_alternatives):
        resolved = safe_baseline.copy()
        switched_rows = row_indices[switch]
        if len(switched_rows):
            switched_baseline = baseline_choice[switch]
            switched_rescue = rescue_choice[switch]
            if bool(lock_rescue_updates):
                resolved[switched_rows] = -np.inf
                resolved[switched_rows, switched_rescue] = baseline_row_confidence[
                    switch
                ]
            else:
                baseline_values = resolved[switched_rows, switched_baseline].copy()
                rescue_values = resolved[switched_rows, switched_rescue].copy()
                resolved[switched_rows, switched_rescue] = baseline_values
                resolved[switched_rows, switched_baseline] = rescue_values
    else:
        resolved = np.full_like(candidate, -np.inf, dtype=np.float32)
        resolved[row_indices, selected] = baseline_row_confidence
    action_scores = np.full_like(candidate, -np.inf, dtype=np.float32)
    selected_action_probability = keep.copy()
    selected_action_probability[switch] = rescue_probability[switch]
    action_scores[row_indices, selected] = selected_action_probability
    return (
        selected.astype(np.int64),
        resolved,
        switch,
        action_margin,
        action_scores,
    )
