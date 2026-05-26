"""Fuse aligned hypothesis score-row tables."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable, Sequence

import numpy as np

from feature_extract.vfm.score_table import ScoreRow, ScoreTableReport, evaluate_score_table, group_rows_by_query


_RANK_SUFFIX_RE = re.compile(r":(\d+)$")


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


def _rank_percentile(scores: np.ndarray) -> np.ndarray:
    if scores.size == 1:
        return np.zeros_like(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(scores.size, dtype=np.float64)
    return ranks / float(scores.size - 1)


def _normalize_query_scores(scores: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "none":
        return scores.astype(np.float64, copy=False)
    if normalization == "zscore":
        std = float(np.std(scores))
        if std < 1e-12:
            return np.zeros_like(scores, dtype=np.float64)
        return (scores - float(np.mean(scores))) / std
    if normalization == "minmax":
        low = float(np.min(scores))
        high = float(np.max(scores))
        if high - low < 1e-12:
            return np.zeros_like(scores, dtype=np.float64)
        return (scores - low) / (high - low)
    if normalization == "rank_percentile":
        return _rank_percentile(scores)
    raise ValueError("normalization must be one of: none, zscore, minmax, rank_percentile")


def normalize_scores_by_query(rows: Sequence[ScoreRow], normalization: str = "zscore") -> list[ScoreRow]:
    """Normalize scores within each query group while preserving row metadata."""

    normalized: list[ScoreRow] = []
    grouped = group_rows_by_query(rows)
    for query_id in sorted(grouped):
        group = grouped[query_id]
        values = np.asarray([row.score for row in group], dtype=np.float64)
        scores = _normalize_query_scores(values, normalization)
        normalized.extend(replace(row, score=float(score)) for row, score in zip(group, scores))
    return normalized


def _validate_aligned_tables(
    tables: Sequence[Sequence[ScoreRow]],
    alignment: str,
) -> list[tuple[str, str]]:
    if len(tables) < 2:
        raise ValueError("at least two score-row tables are required")
    key_sets = [{_candidate_key(row, alignment) for row in table} for table in tables]
    for idx, table in enumerate(tables):
        if len(key_sets[idx]) != len(table):
            raise ValueError(f"score table {idx} has duplicate candidate key under {alignment} alignment")
    expected = key_sets[0]
    for idx, keys in enumerate(key_sets[1:], start=1):
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            detail = missing[0] if missing else extra[0]
            raise ValueError(f"score table {idx} has mismatched candidate key {detail}")
    return sorted(expected)


def _validate_row_metadata(rows: Sequence[ScoreRow], key: tuple[str, str]) -> None:
    base = rows[0]
    for row in rows[1:]:
        if row.protocol_kind != base.protocol_kind:
            raise ValueError(f"aligned rows for {key} mix protocol kinds")
        if abs(float(row.cost_m) - float(base.cost_m)) > 1e-6:
            raise ValueError(f"aligned rows for {key} have inconsistent costs")
        if bool(row.basin_label) != bool(base.basin_label):
            raise ValueError(f"aligned rows for {key} have inconsistent basin labels")


def fuse_score_rows(
    tables: Sequence[Sequence[ScoreRow]],
    weights: Sequence[float],
    method: str,
    normalization: str = "zscore",
    alignment: str = "candidate_id",
) -> list[ScoreRow]:
    """Linearly fuse aligned score-row tables after per-query normalization."""

    if len(tables) != len(weights):
        raise ValueError("weights length must match score-row table count")
    if not method:
        raise ValueError("method must be non-empty")
    keys = _validate_aligned_tables(tables, alignment=alignment)
    normalized_tables = [normalize_scores_by_query(table, normalization=normalization) for table in tables]
    normalized_indices = [{_candidate_key(row, alignment): row for row in table} for table in normalized_tables]
    original_indices = [{_candidate_key(row, alignment): row for row in table} for table in tables]
    base_index = original_indices[0]
    fused: list[ScoreRow] = []
    for key in keys:
        base = base_index[key]
        _validate_row_metadata([index[key] for index in original_indices], key)
        score = 0.0
        for weight, index in zip(weights, normalized_indices):
            score += float(weight) * float(index[key].score)
        fused.append(
            ScoreRow(
                query_id=base.query_id,
                candidate_id=base.candidate_id,
                score=float(score),
                cost_m=base.cost_m,
                basin_label=base.basin_label,
                protocol_kind=base.protocol_kind,
                method=method,
                risk=base.risk,
            )
        )
    return fused


def fuse_two_score_rows(
    primary_rows: Sequence[ScoreRow],
    secondary_rows: Sequence[ScoreRow],
    alpha: float,
    method: str,
    normalization: str = "zscore",
    alignment: str = "candidate_id",
) -> list[ScoreRow]:
    """Fuse primary and secondary rows with score = (1-alpha) primary + alpha secondary."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return fuse_score_rows(
        [primary_rows, secondary_rows],
        weights=[1.0 - float(alpha), float(alpha)],
        method=method,
        normalization=normalization,
        alignment=alignment,
    )


def slice_score_rows_by_query_ids(rows: Sequence[ScoreRow], query_ids: Sequence[str]) -> list[ScoreRow]:
    """Return rows whose query_id is in query_ids."""

    allowed = {str(query_id) for query_id in query_ids}
    filtered = [row for row in rows if str(row.query_id) in allowed]
    if not filtered:
        raise ValueError("query slice produced no score rows")
    return filtered


@dataclass(frozen=True)
class FusionAlphaReport:
    calibration: ScoreTableReport
    evaluation: ScoreTableReport


@dataclass(frozen=True)
class FusionCalibrationResult:
    selected_alpha: float
    selected_rows: tuple[ScoreRow, ...]
    calibration_report: ScoreTableReport
    evaluation_report: ScoreTableReport
    alpha_reports: dict[float, FusionAlphaReport]


def _filter_rows_by_query_ids(rows: Sequence[ScoreRow], query_ids: Sequence[str]) -> list[ScoreRow]:
    return slice_score_rows_by_query_ids(rows, query_ids)


def _metric_direction(metric: str) -> int:
    if metric in {"pred_cost_m", "oracle_cost_m", "oracle_gap_m"}:
        return -1
    if metric in {
        "top1_acc",
        "spearman",
        "ndcg_at_10",
        "basin_recall_at_1",
        "basin_recall_at_2",
        "basin_recall_at_5",
        "basin_recall_at_10",
    }:
        return 1
    raise ValueError(f"unsupported calibration metric: {metric}")


def _report_metric(report: ScoreTableReport, metric: str) -> float:
    try:
        return float(getattr(report, f"mean_{metric}"))
    except AttributeError as exc:
        raise ValueError(f"unsupported calibration metric: {metric}") from exc


def _alpha_method(method_prefix: str, alpha: float) -> str:
    return f"{method_prefix}_a{int(round(float(alpha) * 1000)):03d}"


def calibrate_two_score_fusion(
    primary_rows: Sequence[ScoreRow],
    secondary_rows: Sequence[ScoreRow],
    alphas: Sequence[float],
    metric: str,
    calibration_query_ids: Sequence[str],
    evaluation_query_ids: Sequence[str],
    method_prefix: str,
    normalization: str = "zscore",
    alignment: str = "candidate_id",
) -> FusionCalibrationResult:
    """Choose a two-score fusion alpha on calibration queries and evaluate held-out queries."""

    if not alphas:
        raise ValueError("at least one alpha is required")
    if not calibration_query_ids or not evaluation_query_ids:
        raise ValueError("calibration and evaluation query ids must be non-empty")
    overlap = set(str(query_id) for query_id in calibration_query_ids).intersection(
        str(query_id) for query_id in evaluation_query_ids
    )
    if overlap:
        raise ValueError(f"calibration/evaluation query split overlaps at {sorted(overlap)[0]!r}")
    direction = _metric_direction(metric)

    alpha_reports: dict[float, FusionAlphaReport] = {}
    selected_alpha: float | None = None
    selected_score: float | None = None
    selected_rows: list[ScoreRow] | None = None
    selected_calibration_report: ScoreTableReport | None = None
    selected_evaluation_report: ScoreTableReport | None = None

    for alpha in sorted(float(value) for value in alphas):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha values must be in [0, 1]")
        fused = fuse_score_rows(
            [primary_rows, secondary_rows],
            weights=[1.0 - alpha, alpha],
            method=_alpha_method(method_prefix, alpha),
            normalization=normalization,
            alignment=alignment,
        )
        calibration_report = evaluate_score_table(_filter_rows_by_query_ids(fused, calibration_query_ids))
        evaluation_report = evaluate_score_table(_filter_rows_by_query_ids(fused, evaluation_query_ids))
        alpha_reports[alpha] = FusionAlphaReport(
            calibration=calibration_report,
            evaluation=evaluation_report,
        )
        calibration_score = _report_metric(calibration_report, metric)
        if selected_score is None or direction * calibration_score > direction * selected_score:
            selected_alpha = alpha
            selected_score = calibration_score
            selected_rows = fused
            selected_calibration_report = calibration_report
            selected_evaluation_report = evaluation_report

    assert selected_alpha is not None
    assert selected_rows is not None
    assert selected_calibration_report is not None
    assert selected_evaluation_report is not None
    return FusionCalibrationResult(
        selected_alpha=selected_alpha,
        selected_rows=tuple(selected_rows),
        calibration_report=selected_calibration_report,
        evaluation_report=selected_evaluation_report,
        alpha_reports=alpha_reports,
    )


def rows_from_json_payload(payload: Iterable[dict]) -> list[ScoreRow]:
    return [ScoreRow(**dict(item)) for item in payload]


def rows_to_json_payload(rows: Sequence[ScoreRow]) -> list[dict[str, object]]:
    payload = []
    for row in rows:
        item = dict(row.__dict__)
        item["protocol_kind"] = row.protocol_kind.value
        payload.append(item)
    return payload
