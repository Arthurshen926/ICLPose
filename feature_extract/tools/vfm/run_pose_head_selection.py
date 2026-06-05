"""Run offline calibrated pose-head selection over fixed row JSONLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.pose_head_selection import (
    common_query_ids,
    evaluate_pose_head_selection,
    load_jsonl_rows,
    query_split,
    selector_training_diagnostics,
    train_pose_head_selector,
    write_json,
    write_jsonl_rows,
)


def parse_head_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--head must be formatted as name=/path/to/rows.jsonl")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("head name must be non-empty")
    return name, Path(path)


def _load_rows(head_specs: Sequence[tuple[str, Path]]) -> dict[str, list[dict[str, object]]]:
    rows_by_head = {}
    for name, path in head_specs:
        if name in rows_by_head:
            raise ValueError(f"duplicate head name: {name}")
        rows_by_head[name] = load_jsonl_rows(path, head_name=name)
    return rows_by_head


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _method_table(per_head: dict[str, object]) -> str:
    lines = [
        "| head | S@10 | S@25 | S@50 | median t | median r | inlier P@1 | inliers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for head_name, metrics in sorted(per_head.items()):
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(head_name),
                    _fmt(metrics.get("success_10cm_5deg")),
                    _fmt(metrics.get("success_25cm_10deg")),
                    _fmt(metrics.get("success_50cm_10deg")),
                    _fmt(metrics.get("median_translation_error_m")),
                    _fmt(metrics.get("median_rotation_error_deg")),
                    _fmt(metrics.get("mean_pnp_inlier_patch_at_1")),
                    _fmt(metrics.get("mean_pnp_inlier_count")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    train = summary["train"]
    eval_summary = summary["eval"]
    lines = [
        "# Stage C2.12 Pose Head Selection",
        "",
        f"Scene: `{summary.get('scene', '')}`",
        "",
        "## Split",
        "",
        f"- train queries: {len(summary['train_query_ids'])}",
        f"- eval queries: {len(summary['eval_query_ids'])}",
        f"- heads: {', '.join(summary['heads'])}",
        "",
        "## Training Diagnostics",
        "",
        "```json",
        json.dumps(summary["training_diagnostics"], indent=2, sort_keys=True),
        "```",
        "",
        "## Eval Per-Head Metrics",
        "",
        _method_table(eval_summary["per_head"]),
        "",
        "## Selected Head",
        "",
        "| split | S@10 | S@25 | S@50 | median t | median r | selected heads |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split_name, split_summary in (("train", train), ("eval", eval_summary)):
        lines.append(
            "| "
            + " | ".join(
                [
                    split_name,
                    _fmt(split_summary.get("success_10cm_5deg")),
                    _fmt(split_summary.get("success_25cm_10deg")),
                    _fmt(split_summary.get("success_50cm_10deg")),
                    _fmt(split_summary.get("median_translation_error_m")),
                    _fmt(split_summary.get("median_rotation_error_deg")),
                    json.dumps(split_summary.get("selected_head_counts", {}), sort_keys=True),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Eval Oracle Upper Bound",
            "",
            "```json",
            json.dumps(eval_summary.get("oracle", {}), indent=2, sort_keys=True),
            "```",
        ]
    )
    if "rescue_break_vs_baseline" in eval_summary:
        lines.extend(
            [
                "",
                "## Rescue / Break",
                "",
                "```json",
                json.dumps(eval_summary["rescue_break_vs_baseline"], indent=2, sort_keys=True),
                "```",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", action="append", required=True, type=parse_head_arg)
    parser.add_argument("--scene", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_head", default=None)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_query", action="append", default=[])
    parser.add_argument("--eval_query", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args(argv)

    rows_by_head = _load_rows(args.head)
    query_ids = common_query_ids(rows_by_head)
    if args.train_query or args.eval_query:
        train_ids = {str(query_id) for query_id in args.train_query}
        eval_ids = {str(query_id) for query_id in args.eval_query}
        if not train_ids:
            train_ids = set(query_ids - eval_ids)
        if not eval_ids:
            eval_ids = set(query_ids - train_ids)
    else:
        train_ids, eval_ids = query_split(sorted(query_ids), train_fraction=args.train_fraction, seed=args.seed)
    train_ids &= query_ids
    eval_ids &= query_ids
    if not train_ids or not eval_ids:
        raise ValueError("pose-head selection requires non-empty train and eval query sets")

    model = train_pose_head_selector(
        rows_by_head,
        train_query_ids=train_ids,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    train_summary = evaluate_pose_head_selection(
        rows_by_head,
        query_ids=train_ids,
        model=model,
        baseline_head=args.baseline_head,
    )
    eval_summary = evaluate_pose_head_selection(
        rows_by_head,
        query_ids=eval_ids,
        model=model,
        baseline_head=args.baseline_head,
    )
    summary = {
        "scene": args.scene,
        "heads": sorted(rows_by_head),
        "train_query_ids": sorted(train_ids),
        "eval_query_ids": sorted(eval_ids),
        "training_diagnostics": selector_training_diagnostics(rows_by_head, query_ids=train_ids, model=model),
        "train": train_summary,
        "eval": eval_summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pose_head_selection_summary.json", summary)
    write_jsonl_rows(output_dir / "selected_train_rows.jsonl", train_summary["rows"])
    write_jsonl_rows(output_dir / "selected_eval_rows.jsonl", eval_summary["rows"])
    write_markdown(output_dir / "pose_head_selection_summary.md", summary)


if __name__ == "__main__":
    main()
