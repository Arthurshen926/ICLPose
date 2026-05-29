"""Summarize feature-only, stats-only, and feature+stats patch matching modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


FIELDS = (
    "scene",
    "method",
    "query_count",
    "success_25cm_10deg",
    "success_50cm_10deg",
    "median_translation_error_m",
    "median_rotation_error_deg",
    "mean_pnp_inlier_patch_at_1",
    "mean_pnp_inlier_count",
    "mean_match_count",
    "summary_path",
)


def _parse_run(text: str) -> tuple[str, str, Path]:
    parts = [part.strip() for part in text.split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--run must be formatted as scene,method,evaluation_summary")
    return parts[0], parts[1], Path(parts[2])


def _row(scene: str, method: str, summary_path: Path) -> dict[str, object]:
    summary = json.loads(Path(summary_path).read_text())
    return {
        "scene": scene,
        "method": method,
        "query_count": summary.get("query_count"),
        "success_25cm_10deg": summary.get("success_25cm_10deg"),
        "success_50cm_10deg": summary.get("success_50cm_10deg"),
        "median_translation_error_m": summary.get("median_translation_error_m"),
        "median_rotation_error_deg": summary.get("median_rotation_error_deg"),
        "mean_pnp_inlier_patch_at_1": summary.get("mean_pnp_inlier_patch_at_1"),
        "mean_pnp_inlier_count": summary.get("mean_pnp_inlier_count"),
        "mean_match_count": summary.get("mean_match_count"),
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
        "success_25cm_10deg",
        "success_50cm_10deg",
        "median_translation_error_m",
        "mean_pnp_inlier_patch_at_1",
        "mean_pnp_inlier_count",
        "mean_match_count",
    )
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize patch map-quality scoring mode runs")
    parser.add_argument("--run", action="append", required=True, help="scene,method,evaluation_summary")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    rows = [_row(*_parse_run(item)) for item in args.run]
    rows.sort(key=lambda row: (str(row["scene"]), str(row["method"])))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"stage": "patch_map_quality_modes_summary", "rows": rows}, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        _write_markdown(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
