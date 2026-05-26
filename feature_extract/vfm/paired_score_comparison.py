"""Paired query-level comparisons for VFM score rows."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.metrics import RankingSummary, ranking_summary
from feature_extract.vfm.score_table import ScoreRow, group_rows_by_query
from feature_extract.vfm.statistics import (
    mcnemar_exact_pvalue,
    paired_bootstrap_delta_ci,
    wilcoxon_signed_rank,
)


DEFAULT_PAIRED_METRICS = (
    "pred_cost_m",
    "oracle_gap_m",
    "top1_acc",
    "spearman",
    "ndcg_at_10",
    "basin_recall_at_1",
    "basin_recall_at_5",
    "basin_recall_at_10",
)

_BINARY_METRICS = {
    "top1_acc",
    "basin_recall_at_1",
    "basin_recall_at_2",
    "basin_recall_at_5",
    "basin_recall_at_10",
}


def load_score_rows_json(path: Path | str) -> list[ScoreRow]:
    items = json.loads(Path(path).read_text())
    rows = []
    for item in items:
        rows.append(
            ScoreRow(
                query_id=str(item["query_id"]),
                candidate_id=str(item["candidate_id"]),
                score=float(item["score"]),
                cost_m=float(item["cost_m"]),
                basin_label=bool(item["basin_label"]),
                protocol_kind=item["protocol_kind"],
                method=str(item["method"]),
                risk=float(item.get("risk", 0.0)),
            )
        )
    return rows


def _query_summaries(rows: Iterable[ScoreRow]) -> tuple[dict[str, RankingSummary], str, str]:
    row_list = list(rows)
    if not row_list:
        raise ValueError("score rows must be non-empty")
    protocols = {row.protocol_kind.value for row in row_list}
    methods = {row.method for row in row_list}
    if len(protocols) != 1:
        raise ValueError("score rows must have one protocol_kind")
    if len(methods) != 1:
        raise ValueError("score rows must have one method")
    grouped = group_rows_by_query(row_list)
    summaries = {}
    for query_id, group in grouped.items():
        summaries[query_id] = ranking_summary(
            scores=[row.score for row in group],
            costs_m=[row.cost_m for row in group],
            basin_labels=[row.basin_label for row in group],
        )
    return summaries, next(iter(protocols)), next(iter(methods))


def _metric_values(summaries: Mapping[str, RankingSummary], query_ids: Sequence[str], metric: str) -> np.ndarray:
    try:
        return np.asarray([float(getattr(summaries[query_id], metric)) for query_id in query_ids], dtype=np.float64)
    except AttributeError as exc:
        raise ValueError(f"unknown ranking metric: {metric}") from exc


def compare_score_rows(
    method_rows: Iterable[ScoreRow],
    baseline_rows: Iterable[ScoreRow],
    label: str,
    metrics: Sequence[str] = DEFAULT_PAIRED_METRICS,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, object]:
    method_summaries, method_protocol, method_name = _query_summaries(method_rows)
    baseline_summaries, baseline_protocol, baseline_name = _query_summaries(baseline_rows)
    if method_protocol != baseline_protocol:
        raise ValueError("method and baseline protocol_kind must match")
    method_queries = set(method_summaries)
    baseline_queries = set(baseline_summaries)
    if method_queries != baseline_queries:
        missing_method = sorted(baseline_queries - method_queries)
        missing_baseline = sorted(method_queries - baseline_queries)
        raise ValueError(
            "method and baseline query sets must match"
            f"; missing_method={missing_method[:3]} missing_baseline={missing_baseline[:3]}"
        )
    query_ids = sorted(method_queries)
    metric_payload = {}
    for metric in metrics:
        method_values = _metric_values(method_summaries, query_ids, metric)
        baseline_values = _metric_values(baseline_summaries, query_ids, metric)
        delta_mean, ci_low, ci_high = paired_bootstrap_delta_ci(
            method_values,
            baseline_values,
            resamples=resamples,
            seed=seed,
        )
        payload: dict[str, object] = {
            "method_mean": float(np.mean(method_values)),
            "baseline_mean": float(np.mean(baseline_values)),
            "delta_mean": float(delta_mean),
            "bootstrap_ci_low": float(ci_low),
            "bootstrap_ci_high": float(ci_high),
        }
        if metric in _BINARY_METRICS:
            payload["mcnemar_pvalue"] = mcnemar_exact_pvalue(
                baseline_success=baseline_values > 0.5,
                method_success=method_values > 0.5,
            )
        else:
            payload["wilcoxon"] = asdict(wilcoxon_signed_rank(method_values, baseline_values))
        metric_payload[metric] = payload

    return {
        "label": str(label),
        "method": method_name,
        "baseline": baseline_name,
        "protocol_kind": method_protocol,
        "query_count": len(query_ids),
        "resamples": int(resamples),
        "seed": int(seed),
        "metrics": metric_payload,
    }


def attach_comparison_inputs(
    summary: Mapping[str, object],
    method_rows_path: Path,
    baseline_rows_path: Path,
) -> dict[str, object]:
    payload = dict(summary)
    payload["inputs"] = {
        "method_rows": {
            "path": str(method_rows_path),
            "sha256": file_sha256_short(method_rows_path),
        },
        "baseline_rows": {
            "path": str(baseline_rows_path),
            "sha256": file_sha256_short(baseline_rows_path),
        },
    }
    return payload


def paired_score_comparison_markdown(summary: Mapping[str, object]) -> str:
    metrics = dict(summary["metrics"])  # type: ignore[index]
    lines = [
        f"# {summary['label']}",
        "",
        f"- method: {summary['method']}",
        f"- baseline: {summary['baseline']}",
        f"- protocol_kind: {summary['protocol_kind']}",
        f"- query_count: {summary['query_count']}",
        "",
        "| metric | method | baseline | delta | 95% CI | test |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric, value in metrics.items():
        item = dict(value)
        test = ""
        if "mcnemar_pvalue" in item:
            test = f"McNemar p={item['mcnemar_pvalue']:.6f}"
        elif "wilcoxon" in item:
            wilcoxon = dict(item["wilcoxon"])
            test = f"Wilcoxon z={wilcoxon['normal_approx_z']:.6f}"
        lines.append(
            f"| {metric} | {item['method_mean']:.6f} | {item['baseline_mean']:.6f} | "
            f"{item['delta_mean']:.6f} | [{item['bootstrap_ci_low']:.6f}, "
            f"{item['bootstrap_ci_high']:.6f}] | {test} |"
        )
    return "\n".join(lines) + "\n"
