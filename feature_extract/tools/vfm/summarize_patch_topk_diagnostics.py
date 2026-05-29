"""Summarize soft-mutual patch topK diagnostic evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


FIELDS = (
    "scene",
    "method",
    "top_k",
    "query_count",
    "mean_match_count",
    "mean_patch_at_5",
    "mean_gt_precision_2stride",
    "mean_pnp_inlier_patch_at_5",
    "success_25cm_10deg",
    "success_50cm_10deg",
    "median_translation_error_m",
    "median_rotation_error_deg",
    "summary_path",
)


def _parse_run(text: str) -> tuple[str, str, int, Path]:
    parts = [part.strip() for part in text.split(",", 3)]
    if len(parts) != 4 or not all(parts):
        raise ValueError("--run must be formatted as scene,method,top_k,evaluation_summary")
    return parts[0], parts[1], int(parts[2]), Path(parts[3])


def _row(scene: str, method: str, top_k: int, summary_path: Path) -> dict[str, object]:
    summary = json.loads(Path(summary_path).read_text())
    return {
        "scene": scene,
        "method": method,
        "top_k": int(top_k),
        "query_count": summary.get("query_count"),
        "mean_match_count": summary.get("mean_match_count"),
        "mean_patch_at_5": summary.get("mean_patch_at_5"),
        "mean_gt_precision_2stride": summary.get("mean_gt_precision_2stride"),
        "mean_pnp_inlier_patch_at_5": summary.get("mean_pnp_inlier_patch_at_5"),
        "success_25cm_10deg": summary.get("success_25cm_10deg"),
        "success_50cm_10deg": summary.get("success_50cm_10deg"),
        "median_translation_error_m": summary.get("median_translation_error_m"),
        "median_rotation_error_deg": summary.get("median_rotation_error_deg"),
        "summary_path": str(summary_path),
    }


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scene",
        "method",
        "top_k",
        "mean_patch_at_5",
        "mean_gt_precision_2stride",
        "mean_pnp_inlier_patch_at_5",
        "success_25cm_10deg",
        "success_50cm_10deg",
        "median_translation_error_m",
        "mean_match_count",
    )
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize soft-mutual patch topK diagnostic runs")
    parser.add_argument("--run", action="append", required=True, help="scene,method,top_k,evaluation_summary")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    rows = [_row(*_parse_run(item)) for item in args.run]
    rows.sort(key=lambda row: (str(row["scene"]), int(row["top_k"]), str(row["method"])))
    report = {"stage": "patch_topk_diagnostics_summary", "rows": rows}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        _write_markdown(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
