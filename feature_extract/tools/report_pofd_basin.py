#!/usr/bin/env python3
"""Summarize POFD/NVS pose-basin experiments from JSONL training logs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Iterable


METRIC_COLUMNS = [
    "run",
    "step",
    "pred_cost_m",
    "oracle_cost_m",
    "oracle_gap_m",
    "pred_trans_m",
    "pred_rot_deg",
    "top1_acc",
    "spearman",
    "align_cos",
    "pred_success_5cm_2deg",
    "pred_success_10cm_5deg",
    "pred_success_25cm_10deg",
    "pred_success_50cm_10deg",
    "init_success_25cm_10deg",
    "oracle_success_25cm_10deg",
    "candidate_observability_gap",
    "observability_gap",
    "nvs_pair_match_gap",
    "candidate_teacher_quality_top1_acc",
]


DISPLAY_COLUMNS = [
    ("run", "run"),
    ("step", "step"),
    ("pred_cost_m", "pred_m"),
    ("oracle_cost_m", "oracle_m"),
    ("oracle_gap_m", "gap_m"),
    ("pred_trans_m", "trans_m"),
    ("pred_rot_deg", "rot_deg"),
    ("top1_acc", "top1"),
    ("spearman", "spearman"),
    ("align_cos", "align"),
    ("pred_success_5cm_2deg", "succ@5cm/2deg"),
    ("pred_success_10cm_5deg", "succ@10cm/5deg"),
    ("pred_success_25cm_10deg", "succ@25cm/10deg"),
    ("pred_success_50cm_10deg", "succ@50cm/10deg"),
    ("candidate_observability_gap", "cand_obs_gap"),
    ("observability_gap", "gt_obs_gap"),
    ("nvs_pair_match_gap", "pair_gap"),
]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _metric_is_better(candidate: float | None, current: float | None, mode: str) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    if mode == "max":
        return candidate > current
    return candidate < current


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _candidate_eval_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_rows = [row for row in rows if str(row.get("split", "")).lower() == "eval"]
    return eval_rows if eval_rows else list(rows)


def summarize_log(
    path: str | Path,
    *,
    best_metric: str = "pred_cost_m",
    best_metric_mode: str = "min",
) -> dict[str, Any]:
    """Return one best-eval summary row for a training log."""
    path = Path(path)
    rows = _candidate_eval_rows(load_jsonl(path))
    if not rows:
        raise ValueError(f"{path} does not contain any JSON rows")

    best_row: dict[str, Any] | None = None
    best_value: float | None = None
    for row in rows:
        value = _as_float(row.get(best_metric))
        if _metric_is_better(value, best_value, best_metric_mode):
            best_row = row
            best_value = value
    if best_row is None:
        best_row = rows[-1]

    summary: dict[str, Any] = {
        "run": path.parent.name,
        "log_path": str(path),
        "best_metric": best_metric,
        "best_metric_mode": best_metric_mode,
    }
    for key in METRIC_COLUMNS:
        if key == "run":
            continue
        value = best_row.get(key)
        if value is None and key == "oracle_gap_m":
            pred = _as_float(best_row.get("pred_cost_m"))
            oracle = _as_float(best_row.get("oracle_cost_m"))
            if pred is not None and oracle is not None:
                value = pred - oracle
            else:
                value = best_row.get("rank_oracle_gap_m")
        summary[key] = _as_float(value)
    if summary.get("step") is not None:
        summary["step"] = int(summary["step"])
    return summary


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [header for _, header in DISPLAY_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(key)) for key, _ in DISPLAY_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def resolve_log_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = [Path(p) for p in glob.glob(pattern)]
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(Path(pattern))
    unique = sorted({path.resolve() for path in paths if path.exists()})
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="JSONL log paths or shell-style glob patterns")
    parser.add_argument("--glob", dest="glob_patterns", action="append", default=[])
    parser.add_argument("--best-metric", default="pred_cost_m")
    parser.add_argument("--best-metric-mode", choices=("min", "max"), default="min")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = list(args.logs) + list(args.glob_patterns)
    if not patterns:
        raise SystemExit("provide at least one log path or --glob pattern")
    paths = resolve_log_paths(patterns)
    if not paths:
        raise SystemExit("no log files matched")

    rows = [
        summarize_log(path, best_metric=args.best_metric, best_metric_mode=args.best_metric_mode)
        for path in paths
    ]
    markdown = format_markdown_table(rows)

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
