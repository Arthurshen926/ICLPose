#!/usr/bin/env python3
"""Summarize POFD-FS selector seed train logs against promotion gates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_GATE = {
    "pred_cost_m_max": 0.21,
    "top1_acc_min": 0.75,
    "oracle_gap_m_max": 0.09,
    "spearman_min": 0.55,
}
METRIC_KEYS = ("pred_cost_m", "oracle_gap_m", "top1_acc", "spearman", "basin_recall@1", "basin_recall@5")


def _resolve_train_log(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "train_log.jsonl"
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"train log is empty: {path}")
    return rows


def _best_row(rows: Sequence[Mapping[str, object]], *, metric: str = "pred_cost_m") -> Mapping[str, object]:
    if not all(metric in row for row in rows):
        raise KeyError(f"best metric {metric!r} is missing from at least one train-log row")
    return min(rows, key=lambda row: float(row[metric]))


def _passes_gate(row: Mapping[str, object], gate: Mapping[str, float]) -> bool:
    return (
        float(row.get("pred_cost_m", float("inf"))) <= float(gate["pred_cost_m_max"])
        and float(row.get("top1_acc", float("-inf"))) >= float(gate["top1_acc_min"])
        and float(row.get("oracle_gap_m", float("inf"))) <= float(gate["oracle_gap_m_max"])
        and float(row.get("spearman", float("-inf"))) >= float(gate["spearman_min"])
    )


def _mean_std(rows: Sequence[Mapping[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        mean[key] = float(statistics.fmean(values))
        std[key] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean, std


def summarize_selector_seed_runs(
    train_logs: Sequence[str | Path],
    *,
    gate: Mapping[str, float] | None = None,
    best_metric: str = "pred_cost_m",
) -> dict:
    """Summarize best rows from multiple selector seed train logs."""

    gate = dict(DEFAULT_GATE if gate is None else gate)
    runs = []
    for path in train_logs:
        log_path = _resolve_train_log(path)
        best = dict(_best_row(_load_jsonl(log_path), metric=str(best_metric)))
        run = {
            "path": str(log_path),
            "best_step": int(best.get("step", -1)),
            "promotion_pass": _passes_gate(best, gate),
        }
        for key in METRIC_KEYS:
            if key in best:
                run[key] = float(best[key])
        runs.append(run)
    mean, std = _mean_std(runs)
    return {
        "num_runs": int(len(runs)),
        "num_promotion_pass": int(sum(1 for run in runs if run["promotion_pass"])),
        "promotion_pass_fraction": float(sum(1 for run in runs if run["promotion_pass"]) / max(len(runs), 1)),
        "gate": gate,
        "best_metric": str(best_metric),
        "mean": mean,
        "std": std,
        "runs": runs,
    }


def format_summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Selector Seed Summary",
        "",
        f"- runs: `{summary.get('num_runs')}`",
        f"- promotion pass: `{summary.get('num_promotion_pass')}`",
        "",
        "| run | step | pass | pred | top1 | gap | Spearman | basin@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary.get("runs", []):  # type: ignore[assignment]
        row = dict(run)
        lines.append(
            "| {name} | {step} | {passed} | {pred:.4g} | {top1:.4g} | {gap:.4g} | {spear:.4g} | {basin5:.4g} |".format(
                name=Path(str(row.get("path", ""))).parent.name,
                step=int(row.get("best_step", -1)),
                passed="yes" if bool(row.get("promotion_pass")) else "no",
                pred=float(row.get("pred_cost_m", 0.0)),
                top1=float(row.get("top1_acc", 0.0)),
                gap=float(row.get("oracle_gap_m", 0.0)),
                spear=float(row.get("spearman", 0.0)),
                basin5=float(row.get("basin_recall@5", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", action="append", required=True, help="Path to train_log.jsonl or its run directory.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--best-metric", default="pred_cost_m")
    parser.add_argument("--pred-cost-max", type=float, default=DEFAULT_GATE["pred_cost_m_max"])
    parser.add_argument("--top1-min", type=float, default=DEFAULT_GATE["top1_acc_min"])
    parser.add_argument("--oracle-gap-max", type=float, default=DEFAULT_GATE["oracle_gap_m_max"])
    parser.add_argument("--spearman-min", type=float, default=DEFAULT_GATE["spearman_min"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = {
        "pred_cost_m_max": float(args.pred_cost_max),
        "top1_acc_min": float(args.top1_min),
        "oracle_gap_m_max": float(args.oracle_gap_max),
        "spearman_min": float(args.spearman_min),
    }
    summary = summarize_selector_seed_runs(args.train_log, gate=gate, best_metric=str(args.best_metric))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(format_summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
