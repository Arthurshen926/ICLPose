"""Held-out linear calibration for rendered-map evidence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np

from feature_extract.vfm.score_table import ScoreRow, ScoreTableReport, evaluate_score_table


_RANK_SUFFIX_RE = re.compile(r":(\d+)$")


@dataclass(frozen=True)
class LinearEvidenceModel:
    feature_names: tuple[str, ...]
    intercept: float
    weights: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    l2: float
    target: str = "negative_cost"

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "intercept": float(self.intercept),
            "weights": [float(value) for value in self.weights],
            "feature_means": [float(value) for value in self.feature_means],
            "feature_scales": [float(value) for value in self.feature_scales],
            "l2": float(self.l2),
            "target": self.target,
        }


@dataclass(frozen=True)
class HeldoutEvidenceScorerResult:
    model: LinearEvidenceModel
    all_rows: tuple[ScoreRow, ...]
    calibration_rows: tuple[ScoreRow, ...]
    evaluation_rows: tuple[ScoreRow, ...]
    calibration_report: ScoreTableReport
    evaluation_report: ScoreTableReport


def _candidate_key(row: ScoreRow, alignment: str) -> tuple[str, str]:
    if alignment == "candidate_id":
        return (str(row.query_id), str(row.candidate_id))
    if alignment == "query_rank":
        match = _RANK_SUFFIX_RE.search(str(row.candidate_id))
        if match is None:
            raise ValueError(f"candidate_id has no numeric rank suffix: {row.candidate_id}")
        return (str(row.query_id), match.group(1))
    if alignment == "init_lattice_id":
        candidate_id = str(row.candidate_id)
        if ":init_lattice:" not in candidate_id:
            raise ValueError(f"candidate_id is not an init-lattice id: {row.candidate_id}")
        if "__" in candidate_id:
            candidate_id = candidate_id.split("__", 1)[1]
        return (str(row.query_id), candidate_id)
    raise ValueError("alignment must be one of: candidate_id, query_rank, init_lattice_id")


def _index_rows(rows: Sequence[ScoreRow], alignment: str, table_name: str) -> dict[tuple[str, str], ScoreRow]:
    index: dict[tuple[str, str], ScoreRow] = {}
    for row in rows:
        key = _candidate_key(row, alignment)
        if key in index:
            raise ValueError(f"{table_name} rows contain duplicate candidate key {key}")
        index[key] = row
    if not index:
        raise ValueError(f"{table_name} rows must be non-empty")
    return index


def _validate_aligned_pair(key: tuple[str, str], baseline: ScoreRow, evidence: ScoreRow) -> None:
    if baseline.protocol_kind != evidence.protocol_kind:
        raise ValueError(f"aligned rows for {key} mix protocol kinds")
    if abs(float(baseline.cost_m) - float(evidence.cost_m)) > 1e-6:
        raise ValueError(f"aligned rows for {key} have inconsistent costs")
    if bool(baseline.basin_label) != bool(evidence.basin_label):
        raise ValueError(f"aligned rows for {key} have inconsistent basin labels")


def _aligned_rows(
    baseline_rows: Sequence[ScoreRow],
    evidence_rows: Sequence[ScoreRow],
    alignment: str,
) -> list[tuple[tuple[str, str], ScoreRow, ScoreRow]]:
    baseline_index = _index_rows(baseline_rows, alignment, "baseline")
    evidence_index = _index_rows(evidence_rows, alignment, "evidence")
    if set(baseline_index) != set(evidence_index):
        missing = sorted(set(baseline_index) - set(evidence_index))
        extra = sorted(set(evidence_index) - set(baseline_index))
        detail = missing[0] if missing else extra[0]
        raise ValueError(f"evidence rows have mismatched candidate key {detail}")
    aligned = []
    for key in sorted(baseline_index):
        baseline = baseline_index[key]
        evidence = evidence_index[key]
        _validate_aligned_pair(key, baseline, evidence)
        aligned.append((key, baseline, evidence))
    return aligned


def _optional_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _feature_value(name: str, baseline: ScoreRow, evidence: ScoreRow) -> float:
    if name == "baseline_score":
        return float(baseline.score)
    if name == "evidence_score":
        return float(evidence.score)
    if name == "baseline_risk":
        return float(baseline.risk)
    if name in {"risk", "evidence_risk"}:
        return float(evidence.risk)
    if name == "mean_similarity":
        return _optional_float(evidence.mean_similarity)
    if name == "inlier_fraction":
        return _optional_float(evidence.inlier_fraction)
    if name == "visibility_fraction":
        return _optional_float(evidence.visibility_fraction)
    if name == "match_count":
        return _optional_float(evidence.match_count)
    if name == "match_count_log1p":
        return math.log1p(max(0.0, _optional_float(evidence.match_count)))
    if name == "empty_evidence":
        match_count = _optional_float(evidence.match_count)
        visibility = evidence.visibility_fraction
        if match_count <= 0.0:
            return 1.0
        if visibility is not None and float(visibility) <= 0.0:
            return 1.0
        return 0.0
    raise ValueError(f"unsupported evidence feature: {name}")


def _feature_matrix(
    pairs: Sequence[tuple[tuple[str, str], ScoreRow, ScoreRow]],
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [[_feature_value(name, baseline, evidence) for name in feature_names] for _, baseline, evidence in pairs],
        dtype=np.float64,
    )


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / float(values.size - 1)


def _target_values(
    pairs: Sequence[tuple[tuple[str, str], ScoreRow, ScoreRow]],
    target: str,
) -> np.ndarray:
    costs = np.asarray([baseline.cost_m for _, baseline, _ in pairs], dtype=np.float64)
    if target == "negative_cost":
        return -costs

    values = np.zeros_like(costs, dtype=np.float64)
    query_to_indices: dict[str, list[int]] = {}
    for idx, (_, baseline, _) in enumerate(pairs):
        query_to_indices.setdefault(str(baseline.query_id), []).append(idx)
    for indices in query_to_indices.values():
        query_costs = costs[indices]
        if target == "negative_query_centered_cost":
            values[indices] = -(query_costs - float(np.mean(query_costs)))
        elif target == "negative_query_zscore_cost":
            std = float(np.std(query_costs))
            if std < 1e-12:
                values[indices] = 0.0
            else:
                values[indices] = -(query_costs - float(np.mean(query_costs))) / std
        elif target == "negative_query_rank_cost":
            values[indices] = 1.0 - _rank_percentile(query_costs)
        else:
            raise ValueError(
                "target must be one of: negative_cost, negative_query_centered_cost, "
                "negative_query_zscore_cost, negative_query_rank_cost"
            )
    return values


def _fit_ridge_model(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    l2: float,
    target: str,
) -> LinearEvidenceModel:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid feature matrix or target shape")
    if x.shape[0] == 0:
        raise ValueError("cannot fit evidence scorer with zero calibration rows")
    if float(l2) < 0.0:
        raise ValueError("l2 must be non-negative")
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    z = (x - means) / scales
    design = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        params = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return LinearEvidenceModel(
        feature_names=tuple(str(name) for name in feature_names),
        intercept=float(params[0]),
        weights=tuple(float(value) for value in params[1:]),
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        l2=float(l2),
        target=target,
    )


def _score_matrix(model: LinearEvidenceModel, x: np.ndarray) -> np.ndarray:
    means = np.asarray(model.feature_means, dtype=np.float64)
    scales = np.asarray(model.feature_scales, dtype=np.float64)
    weights = np.asarray(model.weights, dtype=np.float64)
    z = (x - means) / scales
    return float(model.intercept) + z @ weights


def _row_with_score(baseline: ScoreRow, evidence: ScoreRow, score: float, method: str) -> ScoreRow:
    return ScoreRow(
        query_id=baseline.query_id,
        candidate_id=baseline.candidate_id,
        score=float(score),
        cost_m=baseline.cost_m,
        basin_label=baseline.basin_label,
        protocol_kind=baseline.protocol_kind,
        method=method,
        risk=evidence.risk,
        mean_similarity=evidence.mean_similarity,
        inlier_fraction=evidence.inlier_fraction,
        match_count=evidence.match_count,
        visibility_fraction=evidence.visibility_fraction,
    )


def fit_heldout_evidence_linear_scorer(
    baseline_rows: Sequence[ScoreRow],
    evidence_rows: Sequence[ScoreRow],
    calibration_query_ids: Sequence[str],
    evaluation_query_ids: Sequence[str],
    feature_names: Sequence[str],
    method: str,
    l2: float = 1e-3,
    alignment: str = "candidate_id",
    target: str = "negative_cost",
) -> HeldoutEvidenceScorerResult:
    """Fit a linear scorer on calibration queries and evaluate held-out queries."""

    if not method:
        raise ValueError("method must be non-empty")
    if not feature_names:
        raise ValueError("feature_names must be non-empty")
    calibration_queries = {str(query_id) for query_id in calibration_query_ids}
    evaluation_queries = {str(query_id) for query_id in evaluation_query_ids}
    if not calibration_queries or not evaluation_queries:
        raise ValueError("calibration and evaluation query ids must be non-empty")
    overlap = calibration_queries.intersection(evaluation_queries)
    if overlap:
        raise ValueError(f"calibration/evaluation query split overlaps at {sorted(overlap)[0]!r}")

    aligned = _aligned_rows(baseline_rows, evidence_rows, alignment)
    calibration_pairs = [item for item in aligned if str(item[1].query_id) in calibration_queries]
    evaluation_pairs = [item for item in aligned if str(item[1].query_id) in evaluation_queries]
    if not calibration_pairs:
        raise ValueError("calibration query split produced no aligned rows")
    if not evaluation_pairs:
        raise ValueError("evaluation query split produced no aligned rows")

    x_calibration = _feature_matrix(calibration_pairs, feature_names)
    y_calibration = _target_values(calibration_pairs, target=target)
    model = _fit_ridge_model(x_calibration, y_calibration, feature_names, l2=float(l2), target=target)

    x_all = _feature_matrix(aligned, feature_names)
    scores = _score_matrix(model, x_all)
    all_rows = [
        _row_with_score(baseline, evidence, score, method)
        for (_, baseline, evidence), score in zip(aligned, scores)
    ]
    calibration_rows = [row for row in all_rows if str(row.query_id) in calibration_queries]
    evaluation_rows = [row for row in all_rows if str(row.query_id) in evaluation_queries]
    return HeldoutEvidenceScorerResult(
        model=model,
        all_rows=tuple(all_rows),
        calibration_rows=tuple(calibration_rows),
        evaluation_rows=tuple(evaluation_rows),
        calibration_report=evaluate_score_table(calibration_rows),
        evaluation_report=evaluate_score_table(evaluation_rows),
    )
