"""Learned pose-candidate scoring for render-query evaluation."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence
from feature_extract.vfm.rendered_pose_scoring import PoseHypothesisScore


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    return np.where(logits >= 0.0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))


def _float_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return float(default)
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if math.isfinite(output) else float(default)


def _bool_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _match_validity_probability(match: Any) -> float:
    score = getattr(match, "pnp_soft_score", None)
    if score is not None:
        try:
            value = float(score)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            return float(np.clip(value, 0.0, 1.0))
    logit = getattr(match, "pairwise_inlier_logit", None)
    if logit is not None:
        try:
            value = float(logit)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            return float(_sigmoid(np.asarray([value], dtype=np.float64))[0])
    logprob = getattr(match, "pairwise_inlier_logprob", None)
    if logprob is not None:
        try:
            value = float(logprob)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            return float(np.clip(math.exp(value), 0.0, 1.0))
    similarity = getattr(match, "similarity", 0.0)
    try:
        sim = float(similarity)
    except (TypeError, ValueError):
        sim = 0.0
    return float(np.clip((sim + 1.0) * 0.5, 0.0, 1.0))


def match_validity_probability_stats(
    matches: Sequence[Any],
    *,
    inlier_mask: Sequence[bool] | np.ndarray | None = None,
) -> dict[str, object]:
    """Aggregate observable match-validity probabilities for pose candidate scoring."""

    values = list(matches)
    if not values:
        return {
            "match_validity_probability_mean": None,
            "match_validity_probability_median": None,
            "match_validity_probability_min": None,
            "match_validity_probability_top20_mean": None,
            "match_validity_expected_good_count": 0.0,
            "match_validity_inlier_probability_mean": None,
            "match_validity_outlier_probability_mean": None,
            "match_validity_inlier_outlier_gap": None,
        }
    probabilities = np.asarray([_match_validity_probability(match) for match in values], dtype=np.float64)
    top_count = max(1, int(math.ceil(0.20 * float(probabilities.shape[0]))))
    sorted_probabilities = np.sort(probabilities)[::-1]
    stats: dict[str, object] = {
        "match_validity_probability_mean": float(np.mean(probabilities)),
        "match_validity_probability_median": float(np.median(probabilities)),
        "match_validity_probability_min": float(np.min(probabilities)),
        "match_validity_probability_top20_mean": float(np.mean(sorted_probabilities[:top_count])),
        "match_validity_expected_good_count": float(np.sum(probabilities)),
        "match_validity_inlier_probability_mean": None,
        "match_validity_outlier_probability_mean": None,
        "match_validity_inlier_outlier_gap": None,
    }
    if inlier_mask is None:
        return stats
    inliers = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if inliers.shape[0] != probabilities.shape[0]:
        return stats
    if np.any(inliers):
        stats["match_validity_inlier_probability_mean"] = float(np.mean(probabilities[inliers]))
    if np.any(~inliers):
        stats["match_validity_outlier_probability_mean"] = float(np.mean(probabilities[~inliers]))
    if stats["match_validity_inlier_probability_mean"] is not None and stats["match_validity_outlier_probability_mean"] is not None:
        stats["match_validity_inlier_outlier_gap"] = float(
            float(stats["match_validity_inlier_probability_mean"])
            - float(stats["match_validity_outlier_probability_mean"])
        )
    return stats


def label_pose_row(
    row: Mapping[str, Any],
    *,
    translation_threshold_m: float = 0.10,
    rotation_threshold_deg: float = 5.0,
) -> int:
    translation = _float_value(row, "translation_error_m", float("inf"))
    rotation = _float_value(row, "rotation_error_deg", float("inf"))
    return int(translation <= float(translation_threshold_m) and rotation <= float(rotation_threshold_deg))


def _pose_feature_specs() -> list[tuple[str, Any]]:
    return [
        ("pnp_success", lambda row: 1.0 if _bool_value(row, "pnp_success") else 0.0),
        ("pnp_inlier_count", lambda row: math.log1p(max(_float_value(row, "pnp_inlier_count"), 0.0))),
        ("pnp_inlier_ratio", lambda row: _float_value(row, "pnp_inlier_ratio")),
        ("pose_candidate_match_count", lambda row: math.log1p(max(_float_value(row, "pose_candidate_match_count"), 0.0))),
        ("unfiltered_depth_valid_match_count", lambda row: math.log1p(max(_float_value(row, "unfiltered_depth_valid_match_count"), 0.0))),
        ("pose_score", lambda row: _float_value(row, "pose_update_selected_score", _float_value(row, "pose_score"))),
        ("alignment_score", lambda row: _float_value(row, "pose_update_selected_alignment_score", _float_value(row, "alignment_score"))),
        ("pose_score_weighted_residual", lambda row: _float_value(row, "pose_score_weighted_residual", 16.0)),
        ("pose_score_confidence_mean", lambda row: _float_value(row, "pose_score_confidence_mean", _float_value(row, "pnp_match_confidence_mean"))),
        ("pose_score_coverage", lambda row: _float_value(row, "pose_score_coverage")),
        ("pose_score_degeneracy_penalty", lambda row: _float_value(row, "pose_score_degeneracy_penalty")),
        ("pnp_match_confidence_mean", lambda row: _float_value(row, "pnp_match_confidence_mean")),
        ("pnp_inlier_confidence_mean", lambda row: _float_value(row, "pnp_inlier_confidence_mean")),
        ("pnp_outlier_confidence_mean", lambda row: _float_value(row, "pnp_outlier_confidence_mean")),
        ("pnp_confidence_gap", lambda row: _float_value(row, "pnp_inlier_confidence_mean") - _float_value(row, "pnp_outlier_confidence_mean")),
        ("match_validity_probability_mean", lambda row: _float_value(row, "match_validity_probability_mean")),
        ("match_validity_probability_median", lambda row: _float_value(row, "match_validity_probability_median")),
        ("match_validity_probability_top20_mean", lambda row: _float_value(row, "match_validity_probability_top20_mean")),
        ("match_validity_expected_good_count", lambda row: math.log1p(max(_float_value(row, "match_validity_expected_good_count"), 0.0))),
        ("match_validity_inlier_outlier_gap", lambda row: _float_value(row, "match_validity_inlier_outlier_gap")),
        ("candidate_rank_score", lambda row: 1.0 / (1.0 + max(_float_value(row, "initial_render_index"), 0.0))),
    ]


def vectorize_pose_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    specs = _pose_feature_specs()
    matrix = np.zeros((len(rows), len(specs)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, (_name, getter) in enumerate(specs):
            matrix[row_idx, col_idx] = float(getter(row))
    return matrix, [name for name, _getter in specs]


def vectorize_pose_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    translation_threshold_m: float = 0.10,
    rotation_threshold_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features, names = vectorize_pose_candidate_rows(rows)
    labels = np.asarray(
        [
            label_pose_row(
                row,
                translation_threshold_m=float(translation_threshold_m),
                rotation_threshold_deg=float(rotation_threshold_deg),
            )
            for row in rows
        ],
        dtype=np.int64,
    )
    return features, labels, names


class PairwisePoseRanker:
    """Linear query-level pose candidate ranker trained with pairwise logistic loss."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 800,
        l2: float = 1e-3,
        mean_: np.ndarray | None = None,
        scale_: np.ndarray | None = None,
        weights_: np.ndarray | None = None,
        bias_: float = 0.0,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.mean_ = None if mean_ is None else np.asarray(mean_, dtype=np.float64)
        self.scale_ = None if scale_ is None else np.asarray(scale_, dtype=np.float64)
        self.weights_ = None if weights_ is None else np.asarray(weights_, dtype=np.float64)
        self.bias_ = float(bias_)
        self.feature_names = list(feature_names or [])

    def fit(self, features: np.ndarray, preference_pairs: Sequence[tuple[int, int]]) -> "PairwisePoseRanker":
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must have shape (N, C)")
        pairs = [(int(pos), int(neg)) for pos, neg in preference_pairs]
        if not pairs:
            raise ValueError("at least one pairwise preference is required")
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        z = (x - self.mean_) / self.scale_
        weights = np.zeros((z.shape[1],), dtype=np.float64)
        bias = 0.0
        pair_array = np.asarray(pairs, dtype=np.int64)
        pos = pair_array[:, 0]
        neg = pair_array[:, 1]
        diff = z[pos] - z[neg]
        denominator = max(float(diff.shape[0]), 1.0)
        for _ in range(int(self.max_iter)):
            margin = diff @ weights + bias
            residual = _sigmoid(margin) - 1.0
            grad_w = (diff.T @ residual) / denominator + float(self.l2) * weights
            grad_b = float(np.sum(residual) / denominator)
            weights -= float(self.learning_rate) * grad_w
            bias -= float(self.learning_rate) * grad_b
        self.weights_ = weights.astype(np.float64, copy=False)
        self.bias_ = float(bias)
        return self

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise ValueError("model is not fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must have shape (N, C)")
        z = (x - self.mean_) / self.scale_
        return (z @ self.weights_ + float(self.bias_)).astype(np.float64, copy=False)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_scores(features))

    def predict_scores_from_rows(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        features, names = vectorize_pose_candidate_rows(rows)
        if self.feature_names and list(names) != list(self.feature_names):
            raise ValueError("pose ranker feature names do not match current feature extractor")
        return self.predict_scores(features)

    def to_json_dict(self, feature_names: Sequence[str] | None = None) -> dict[str, Any]:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise ValueError("model is not fitted")
        names = list(feature_names if feature_names is not None else self.feature_names)
        return {
            "model_type": "pairwise_pose_ranker",
            "feature_names": names,
            "mean": [float(value) for value in self.mean_],
            "scale": [float(value) for value in self.scale_],
            "weights": [float(value) for value in self.weights_],
            "bias": float(self.bias_),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "PairwisePoseRanker":
        return cls(
            mean_=np.asarray(data["mean"], dtype=np.float64),
            scale_=np.asarray(data["scale"], dtype=np.float64),
            weights_=np.asarray(data["weights"], dtype=np.float64),
            bias_=float(data.get("bias", 0.0)),
            feature_names=list(data.get("feature_names", [])),
        )

    def save_json(self, path: str | Path, feature_names: Sequence[str] | None = None) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(feature_names), indent=2, sort_keys=True) + "\n")


def _pose_row_quality(
    row: Mapping[str, Any],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> float:
    translation = _float_value(row, "translation_error_m", float("inf"))
    rotation = _float_value(row, "rotation_error_deg", float("inf"))
    if not math.isfinite(translation) or not math.isfinite(rotation):
        return float("-inf")
    return -(
        float(translation) / max(float(translation_threshold_m), 1e-6)
        + float(rotation) / max(float(rotation_threshold_deg), 1e-6)
    )


def build_pairwise_pose_preferences(
    rows: Sequence[Mapping[str, Any]],
    *,
    translation_threshold_m: float = 0.10,
    rotation_threshold_deg: float = 5.0,
    min_quality_gap: float = 0.25,
) -> list[tuple[int, int]]:
    groups: dict[str, list[int]] = {}
    values = list(rows)
    for idx, row in enumerate(values):
        groups.setdefault(str(row.get("query_id", "")), []).append(idx)
    labels = [
        label_pose_row(
            row,
            translation_threshold_m=float(translation_threshold_m),
            rotation_threshold_deg=float(rotation_threshold_deg),
        )
        for row in values
    ]
    qualities = [
        _pose_row_quality(
            row,
            translation_threshold_m=float(translation_threshold_m),
            rotation_threshold_deg=float(rotation_threshold_deg),
        )
        for row in values
    ]
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        for left in indices:
            for right in indices:
                if left == right:
                    continue
                if labels[left] > labels[right]:
                    pairs.append((left, right))
                elif labels[left] == labels[right] and qualities[left] > qualities[right] + float(min_quality_gap):
                    pairs.append((left, right))
    return pairs


def fit_pairwise_pose_ranker(
    rows: Sequence[Mapping[str, Any]],
    *,
    translation_threshold_m: float = 0.10,
    rotation_threshold_deg: float = 5.0,
    learning_rate: float = 0.05,
    max_iter: int = 800,
    l2: float = 1e-3,
) -> tuple[PairwisePoseRanker, dict[str, object]]:
    values = list(rows)
    features, names = vectorize_pose_candidate_rows(values)
    pairs = build_pairwise_pose_preferences(
        values,
        translation_threshold_m=float(translation_threshold_m),
        rotation_threshold_deg=float(rotation_threshold_deg),
    )
    model = PairwisePoseRanker(
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        l2=float(l2),
        feature_names=names,
    ).fit(features, pairs)
    scores = model.predict_scores(features)
    return model, {
        "objective": "pairwise_rank",
        "row_count": int(len(values)),
        "pair_count": int(len(pairs)),
        "feature_names": names,
        "selection": pose_candidate_selection_report(
            values,
            learned_scores=scores,
            translation_threshold_m=float(translation_threshold_m),
            rotation_threshold_deg=float(rotation_threshold_deg),
        ),
    }


def pose_candidate_selection_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    learned_scores: Sequence[float] | None = None,
    translation_threshold_m: float = 0.10,
    rotation_threshold_deg: float = 5.0,
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    values = list(rows)
    for row_idx, row in enumerate(values):
        query_id = str(row.get("query_id", ""))
        groups.setdefault(query_id, []).append(row_idx)
    labels = [
        label_pose_row(
            row,
            translation_threshold_m=float(translation_threshold_m),
            rotation_threshold_deg=float(rotation_threshold_deg),
        )
        for row in values
    ]
    learned = None if learned_scores is None else [float(score) for score in learned_scores]

    def finite_values(indices: Sequence[int], key: str) -> list[float]:
        out = []
        for idx in indices:
            value = _float_value(values[idx], key, float("nan"))
            if math.isfinite(value):
                out.append(float(value))
        return out

    def select_by_score(indices: Sequence[int], scores: Sequence[float]) -> int | None:
        if not indices:
            return None
        return max(
            indices,
            key=lambda idx: (
                float(scores[idx]),
                -_float_value(values[idx], "candidate_rank", _float_value(values[idx], "initial_render_index", 0.0)),
            ),
        )

    def strategy_report(name: str, selected_indices: Sequence[int | None]) -> dict[str, object]:
        selected = [int(idx) for idx in selected_indices if idx is not None]
        selected_labels = [labels[idx] for idx in selected]
        translations = finite_values(selected, "translation_error_m")
        rotations = finite_values(selected, "rotation_error_deg")
        return {
            f"{name}_query_count": int(len(selected)),
            f"{name}_success_rate": None if not selected_labels else float(np.mean(selected_labels)),
            f"{name}_median_translation_error_m": None if not translations else float(np.median(translations)),
            f"{name}_median_rotation_error_deg": None if not rotations else float(np.median(rotations)),
        }

    oracle_labels = [1 if any(labels[idx] > 0 for idx in indices) else 0 for indices in groups.values()]
    report: dict[str, object] = {
        "query_count": int(len(groups)),
        "row_count": int(len(values)),
        "oracle_success_rate": None if not oracle_labels else float(np.mean(oracle_labels)),
    }
    if not values:
        return report
    rank0_selected = [
        min(
            indices,
            key=lambda idx: _float_value(values[idx], "candidate_rank", _float_value(values[idx], "initial_render_index", 0.0)),
        )
        for indices in groups.values()
    ]
    report.update(strategy_report("rank0", rank0_selected))
    pose_scores = [
        _float_value(row, "pose_update_selected_score", _float_value(row, "pose_score", float("-inf")))
        for row in values
    ]
    report.update(
        strategy_report(
            "pose_score",
            [select_by_score(indices, pose_scores) for indices in groups.values()],
        )
    )
    alignment_scores = [
        _float_value(row, "pose_update_selected_alignment_score", _float_value(row, "alignment_score", float("-inf")))
        for row in values
    ]
    report.update(
        strategy_report(
            "alignment_score",
            [select_by_score(indices, alignment_scores) for indices in groups.values()],
        )
    )
    if learned is not None:
        report.update(
            strategy_report(
                "learned",
                [select_by_score(indices, learned) for indices in groups.values()],
            )
        )
    return report


def pose_candidate_row_from_eval_candidate(candidate: Mapping[str, Any]) -> dict[str, object]:
    pnp = candidate.get("pnp")
    score = candidate.get("iteration_pose_score")
    pose_matches = list(candidate.get("pose_pnp_matches", []) or [])
    inlier_mask = getattr(pnp, "inlier_mask", None)
    row: dict[str, object] = {
        "pnp_success": bool(getattr(pnp, "success", False)),
        "pnp_inlier_count": int(getattr(pnp, "inlier_count", 0)),
        "pnp_inlier_ratio": float(getattr(pnp, "inlier_ratio", 0.0)),
        "pose_candidate_match_count": int(len(pose_matches)),
        "unfiltered_depth_valid_match_count": int(candidate.get("unfiltered_pnp_match_count", 0) or 0),
        "alignment_score": float(candidate.get("iteration_alignment_score", 0.0) or 0.0),
        "initial_render_index": int(candidate.get("initial_render_index", 0) or 0),
    }
    row.update(match_validity_probability_stats(pose_matches, inlier_mask=inlier_mask))
    if isinstance(score, PoseHypothesisScore):
        row.update(
            {
                "pose_score": float(score.score),
                "pose_score_weighted_residual": float(score.weighted_residual),
                "pose_score_confidence_mean": float(score.confidence_mean),
                "pose_score_coverage": float(score.coverage),
                "pose_score_degeneracy_penalty": float(score.degeneracy_penalty),
            }
        )
    return row


def load_pose_scorer_model(path: str | Path) -> CalibratedLogisticConfidence | PairwisePoseRanker:
    data = json.loads(Path(path).read_text())
    if str(data.get("model_type", "calibrated_logistic_confidence")) == "pairwise_pose_ranker":
        return PairwisePoseRanker.from_json_dict(data)
    return CalibratedLogisticConfidence.from_json_dict(data)


def score_pose_candidates_with_model(
    candidates: Sequence[dict[str, Any]],
    model: CalibratedLogisticConfidence | PairwisePoseRanker,
    *,
    feature_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    values = list(candidates)
    if not values:
        return []
    rows = [pose_candidate_row_from_eval_candidate(candidate) for candidate in values]
    features, names = vectorize_pose_candidate_rows(rows)
    if feature_names is not None and list(feature_names) and list(feature_names) != names:
        raise ValueError("pose scorer feature names do not match current feature extractor")
    if isinstance(model, PairwisePoseRanker):
        scores = model.predict_scores(features)
        probabilities = _sigmoid(scores)
    else:
        scores = model.predict_proba(features)
        probabilities = scores
    scored: list[dict[str, Any]] = []
    for candidate, score, probability in zip(values, scores, probabilities):
        value = float(np.clip(float(probability), 0.0, 1.0))
        candidate["learned_pose_score"] = float(score)
        candidate["learned_pose_probability"] = value
        score = candidate.get("iteration_pose_score")
        if isinstance(score, PoseHypothesisScore):
            candidate["iteration_pose_score"] = replace(score, score=float(candidate["learned_pose_score"]), learned_probability=value)
        scored.append(candidate)
    scored.sort(
        key=lambda item: (
            float(item.get("learned_pose_score", item.get("learned_pose_probability", 0.0)) or 0.0),
            float(getattr(item.get("iteration_pose_score"), "inlier_count", 0)),
        ),
        reverse=True,
    )
    return scored
