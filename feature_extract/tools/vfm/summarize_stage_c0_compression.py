"""Summarize Stage C0 compression and patch-to-3D evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence


FIELDS = (
    "label",
    "method",
    "output_dim",
    "query_count",
    "mean_match_count",
    "mean_pnp_inlier_count",
    "mean_pnp_inlier_patch_at_1",
    "success_25cm_10deg",
    "success_50cm_10deg",
    "success_1m_10deg",
    "median_translation_error_m",
    "median_rotation_error_deg",
    "query_storage_bytes",
    "landmark_storage_bytes",
    "transform_storage_bytes",
    "total_storage_bytes",
    "compression_elapsed_sec",
    "evaluation_elapsed_sec",
    "total_elapsed_sec",
    "compression_summary_path",
    "evaluation_summary_path",
)


def _parse_run(text: str) -> tuple[str, Path, Path]:
    parts = [part.strip() for part in text.split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--run must be formatted as label,compression_summary,evaluation_summary")
    return parts[0], Path(parts[1]), Path(parts[2])


def _row(label: str, compression_path: Path, evaluation_path: Path) -> dict[str, object]:
    compression = json.loads(Path(compression_path).read_text())
    evaluation = json.loads(Path(evaluation_path).read_text())
    storage = dict(compression.get("storage_bytes") or {})
    query_storage = int(storage.get("query_tokens", 0) or 0)
    landmark_storage = int(storage.get("landmark_bank", 0) or 0)
    transform_storage = int(storage.get("transform", 0) or 0)
    compression_elapsed = float(compression.get("elapsed_sec", 0.0) or 0.0)
    evaluation_elapsed = float(evaluation.get("elapsed_sec", 0.0) or 0.0)
    return {
        "label": label,
        "method": compression.get("method"),
        "output_dim": compression.get("output_dim"),
        "query_count": evaluation.get("query_count"),
        "mean_match_count": evaluation.get("mean_match_count"),
        "mean_pnp_inlier_count": evaluation.get("mean_pnp_inlier_count"),
        "mean_pnp_inlier_patch_at_1": evaluation.get("mean_pnp_inlier_patch_at_1"),
        "success_25cm_10deg": evaluation.get("success_25cm_10deg"),
        "success_50cm_10deg": evaluation.get("success_50cm_10deg"),
        "success_1m_10deg": evaluation.get("success_1m_10deg"),
        "median_translation_error_m": evaluation.get("median_translation_error_m"),
        "median_rotation_error_deg": evaluation.get("median_rotation_error_deg"),
        "query_storage_bytes": query_storage,
        "landmark_storage_bytes": landmark_storage,
        "transform_storage_bytes": transform_storage,
        "total_storage_bytes": query_storage + landmark_storage + transform_storage,
        "compression_elapsed_sec": compression_elapsed,
        "evaluation_elapsed_sec": evaluation_elapsed,
        "total_elapsed_sec": compression_elapsed + evaluation_elapsed,
        "compression_summary_path": str(compression_path),
        "evaluation_summary_path": str(evaluation_path),
    }


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def _write_markdown(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "label",
        "method",
        "output_dim",
        "success_25cm_10deg",
        "success_50cm_10deg",
        "median_translation_error_m",
        "median_rotation_error_deg",
        "mean_pnp_inlier_patch_at_1",
        "total_storage_bytes",
        "total_elapsed_sec",
    )
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage C0 compression runs")
    parser.add_argument("--run", action="append", required=True, help="label,compression_summary,evaluation_summary")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    rows = [_row(label, compression, evaluation) for label, compression, evaluation in map(_parse_run, args.run)]
    report = {"stage": "stage_c0_compression_summary", "rows": rows}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_csv:
        _write_csv(rows, Path(args.output_csv))
    if args.output_md:
        _write_markdown(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
