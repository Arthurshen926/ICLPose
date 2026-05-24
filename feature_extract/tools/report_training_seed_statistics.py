#!/usr/bin/env python3
"""Summarize multi-seed training logs with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.statistics import paired_bootstrap_mean_ci  # noqa: E402


def _parse_label_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    return label, path


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_best_eval_row(
    path: str | Path,
    *,
    selection_metric: str = "pred_cost_m",
    selection_mode: str = "min",
) -> dict:
    """Return the best eval row from a train_log JSONL."""

    rows = [row for row in _load_jsonl(path) if str(row.get("split", "")) == "eval" and selection_metric in row]
    if not rows:
        raise ValueError(f"No eval rows with metric {selection_metric} in {path}")
    reverse = str(selection_mode) == "max"
    if str(selection_mode) not in {"min", "max"}:
        raise ValueError("selection_mode must be 'min' or 'max'")
    return sorted(rows, key=lambda row: float(row[selection_metric]), reverse=reverse)[0]


def _metric_summary(values: Sequence[float], *, num_bootstrap: int, seed: int) -> dict:
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    ci = paired_bootstrap_mean_ci(tensor, num_bootstrap=int(num_bootstrap), seed=int(seed))
    return {
        "mean": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() else 0.0,
        "values": [float(value) for value in values],
    }


def build_training_seed_report(
    *,
    runs: Sequence[tuple[str, str | Path]],
    expected_seeds: int = 5,
    selection_metric: str = "pred_cost_m",
    selection_mode: str = "min",
    metrics: Sequence[str] = ("pred_cost_m", "top1_acc", "spearman", "basin_recall@5"),
    num_bootstrap: int = 10000,
    seed: int = 20260524,
) -> dict:
    """Group repeated train logs and report seed-level mean/CI."""

    grouped: dict[str, list[dict]] = {}
    for label, path in runs:
        row = load_best_eval_row(path, selection_metric=selection_metric, selection_mode=selection_mode)
        row = dict(row)
        row["train_log"] = str(path)
        grouped.setdefault(str(label), []).append(row)
    groups = []
    for label in sorted(grouped):
        rows = grouped[label]
        metric_summary = {}
        for metric in metrics:
            values = [float(row[metric]) for row in rows if metric in row]
            if values:
                metric_summary[str(metric)] = _metric_summary(values, num_bootstrap=num_bootstrap, seed=seed)
        groups.append(
            {
                "label": label,
                "num_seeds": int(len(rows)),
                "expected_seeds": int(expected_seeds),
                "complete": bool(len(rows) >= int(expected_seeds)),
                "selection_metric": str(selection_metric),
                "selection_mode": str(selection_mode),
                "rows": rows,
                "metrics": metric_summary,
            }
        )
    return {
        "protocol": "training_seed_statistics",
        "expected_seeds": int(expected_seeds),
        "num_groups": int(len(groups)),
        "groups": groups,
    }


def format_training_seed_report_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Training Seed Statistics",
        "",
        "| group | seeds | complete | pred mean | pred 95% CI | top1 mean | Spearman mean | basin@5 mean |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for group in report.get("groups", []):  # type: ignore[union-attr]
        group_map = dict(group)
        metrics = dict(group_map.get("metrics", {}))
        pred = dict(metrics.get("pred_cost_m", {}))
        top1 = dict(metrics.get("top1_acc", {}))
        spearman = dict(metrics.get("spearman", {}))
        basin5 = dict(metrics.get("basin_recall@5", {}))
        seeds = f"{int(group_map.get('num_seeds', 0))}/{int(group_map.get('expected_seeds', 0))}"
        ci = "" if not pred else f"[{float(pred['ci_low']):.4g}, {float(pred['ci_high']):.4g}]"
        lines.append(
            "| {label} | {seeds} | {complete} | {pred:.4g} | {ci} | {top1:.4g} | {spearman:.4g} | {basin5:.4g} |".format(
                label=str(group_map.get("label", "")),
                seeds=seeds,
                complete="yes" if group_map.get("complete") else "no",
                pred=float(pred.get("mean", float("nan"))),
                ci=ci,
                top1=float(top1.get("mean", float("nan"))),
                spearman=float(spearman.get("mean", float("nan"))),
                basin5=float(basin5.get("mean", float("nan"))),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Run spec GROUP=TRAIN_LOG_JSONL. Repeatable.")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--selection-metric", default="pred_cost_m")
    parser.add_argument("--selection-mode", choices=("min", "max"), default="min")
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_training_seed_report(
        runs=[_parse_label_path(value) for value in args.run],
        expected_seeds=int(args.expected_seeds),
        selection_metric=str(args.selection_metric),
        selection_mode=str(args.selection_mode),
        metrics=tuple(args.metric) if args.metric else ("pred_cost_m", "top1_acc", "spearman", "basin_recall@5"),
        num_bootstrap=int(args.num_bootstrap),
        seed=int(args.seed),
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(format_training_seed_report_markdown(report), encoding="utf-8")
    print(json.dumps({"num_groups": report["num_groups"]}, indent=2))


if __name__ == "__main__":
    main()
