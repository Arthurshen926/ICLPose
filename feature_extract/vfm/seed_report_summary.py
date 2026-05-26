"""Summaries for repeated-seed VFM score reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


DEFAULT_SEED_REPORT_METRICS = (
    "mean_pred_cost_m",
    "mean_top1_acc",
    "mean_spearman",
    "mean_basin_recall_at_5",
    "mean_oracle_gap_m",
)


def _metric_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("metric values must be a non-empty 1D sequence")
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _load_report(path: Path) -> Mapping[str, object]:
    return json.loads(Path(path).read_text())


def summarize_seed_reports(
    report_paths: Sequence[Path | str],
    label: str,
    metrics: Sequence[str] = DEFAULT_SEED_REPORT_METRICS,
) -> dict[str, object]:
    paths = [Path(path) for path in report_paths]
    if not paths:
        raise ValueError("at least one report path is required")
    reports = [_load_report(path) for path in paths]

    protocol_kinds = {str(report.get("protocol_kind", "")) for report in reports}
    if len(protocol_kinds) != 1:
        raise ValueError("seed reports must share one protocol_kind")
    query_counts = {int(report.get("query_count", -1)) for report in reports}
    if len(query_counts) != 1:
        raise ValueError("seed reports must share one query_count")

    metric_payload: dict[str, dict[str, float]] = {}
    for metric in metrics:
        missing = [str(path) for path, report in zip(paths, reports) if metric not in report]
        if missing:
            raise ValueError(f"metric {metric!r} is missing from {missing[0]}")
        metric_payload[metric] = _metric_stats(float(report[metric]) for report in reports)

    return {
        "label": str(label),
        "seed_count": len(paths),
        "protocol_kind": next(iter(protocol_kinds)),
        "query_count": next(iter(query_counts)),
        "metrics": metric_payload,
        "inputs": [
            {
                "path": str(path),
                "sha256": file_sha256_short(path),
            }
            for path in paths
        ],
    }


def seed_report_summary_markdown(summary: Mapping[str, object]) -> str:
    metrics = dict(summary["metrics"])  # type: ignore[index]
    lines = [
        f"# {summary['label']}",
        "",
        f"- seeds: {summary['seed_count']}",
        f"- protocol_kind: {summary['protocol_kind']}",
        f"- query_count: {summary['query_count']}",
        "",
        "| metric | mean | std | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, stats in metrics.items():
        value = dict(stats)
        lines.append(
            f"| {metric} | {value['mean']:.6f} | {value['std']:.6f} | "
            f"{value['min']:.6f} | {value['max']:.6f} |"
        )
    return "\n".join(lines) + "\n"
