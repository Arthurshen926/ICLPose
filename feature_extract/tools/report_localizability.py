#!/usr/bin/env python3
"""Summarize POFD-FS localizability audit/training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_jsonl(path: str | Path, *, best_metric: str = "pred_cost_m", mode: str = "min") -> dict:
    path = Path(path)
    rows = _read_jsonl(path)
    eval_rows = [row for row in rows if row.get("split", "eval") == "eval" and best_metric in row]
    if not eval_rows:
        eval_rows = [row for row in rows if best_metric in row]
    if not eval_rows:
        raise ValueError(f"No rows with metric {best_metric!r} in {path}")
    reverse = str(mode).lower() == "max"
    best = sorted(eval_rows, key=lambda row: float(row[best_metric]), reverse=reverse)[0]
    row = dict(best)
    row.setdefault("run", path.parent.name if path.name == "train_log.jsonl" else path.stem)
    if "oracle_gap_m" not in row and "pred_cost_m" in row and "oracle_cost_m" in row:
        row["oracle_gap_m"] = round(float(row["pred_cost_m"]) - float(row["oracle_cost_m"]), 10)
    return row


def _fmt(value, precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def format_markdown_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    headers = ["run", "step", "pred", "oracle", "gap", "top1", "spearman", "basin@5"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run", "")),
                    _fmt(row.get("step"), 0),
                    _fmt(row.get("pred_cost_m")),
                    _fmt(row.get("oracle_cost_m")),
                    _fmt(row.get("oracle_gap_m")),
                    _fmt(row.get("top1_acc")),
                    _fmt(row.get("spearman")),
                    _fmt(row.get("basin_recall@5", row.get("basin_recall@4"))),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="JSONL logs to summarize")
    parser.add_argument("--best-metric", default="pred_cost_m")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize_jsonl(path, best_metric=args.best_metric, mode=args.mode) for path in args.logs]
    markdown = format_markdown_table(rows)
    if args.out:
        Path(args.out).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
