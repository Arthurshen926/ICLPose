"""Build fine-stage RGB patch rows from coarse measurement outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FIELDNAMES = [
    "query_id",
    "match_index",
    "center_x",
    "center_y",
    "query_gt_x",
    "query_gt_y",
    "render_x",
    "render_y",
    "render_depth",
    "world_x",
    "world_y",
    "world_z",
    "center_residual_px",
    "center_residual_dx",
    "center_residual_dy",
    "target_is_dustbin",
    "requested_residual_px",
    "measurement_policy",
    "coarse_valid_prob",
]


def _read_csv(path: Path, *, max_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows: list[dict[str, str]] = []
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
        return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FIELDNAMES)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _optional_float(row: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            number = float(text)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return float(number)
    return None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def build_rgb_patch_fine_rows_from_coarse_measurements(
    *,
    coarse_rows_csv: Path,
    output_dir: Path,
    dustbin_residual_px: float,
    max_rows: int | None = None,
    requested_residual_bin_px: float = 0.5,
) -> dict[str, Any]:
    input_rows = _read_csv(Path(coarse_rows_csv), max_rows=max_rows)
    output_rows: list[dict[str, Any]] = []
    skipped = 0
    dustbin_count = 0
    residuals: list[float] = []
    for row in input_rows:
        center_x = _optional_float(row, "query_refined_x", "query_pred_x")
        center_y = _optional_float(row, "query_refined_y", "query_pred_y")
        gt_x = _optional_float(row, "query_gt_x", "gt_query_x")
        gt_y = _optional_float(row, "query_gt_y", "gt_query_y")
        render_x = _optional_float(row, "render_x")
        render_y = _optional_float(row, "render_y")
        if center_x is None or center_y is None or gt_x is None or gt_y is None or render_x is None or render_y is None:
            skipped += 1
            continue
        dx = float(gt_x - center_x)
        dy = float(gt_y - center_y)
        residual = float(math.hypot(dx, dy))
        target_is_dustbin = residual > float(dustbin_residual_px)
        if target_is_dustbin:
            dustbin_count += 1
        bin_px = float(requested_residual_bin_px)
        requested_residual = float(round(residual / bin_px) * bin_px) if bin_px > 0.0 else residual
        residuals.append(residual)
        output_rows.append(
            {
                "query_id": row.get("query_id", ""),
                "match_index": row.get("match_index", ""),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "query_gt_x": float(gt_x),
                "query_gt_y": float(gt_y),
                "render_x": float(render_x),
                "render_y": float(render_y),
                "render_depth": row.get("render_depth", ""),
                "world_x": row.get("world_x", ""),
                "world_y": row.get("world_y", ""),
                "world_z": row.get("world_z", ""),
                "center_residual_px": residual,
                "center_residual_dx": dx,
                "center_residual_dy": dy,
                "target_is_dustbin": target_is_dustbin,
                "requested_residual_px": requested_residual,
                "measurement_policy": "fine_after_coarse",
                "coarse_valid_prob": row.get("measurement_valid_prob", ""),
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_csv = output / "fine_rows.csv"
    _write_csv(rows_csv, output_rows)
    summary = {
        "stage": "rgb_patch_fine_rows_from_coarse_measurements",
        "coarse_rows_csv": str(coarse_rows_csv),
        "input_count": int(len(input_rows)),
        "row_count": int(len(output_rows)),
        "skipped_count": int(skipped),
        "dustbin_count": int(dustbin_count),
        "dustbin_rate": float(dustbin_count / len(output_rows)) if output_rows else 0.0,
        "center_residual_median_px": _percentile(residuals, 50.0),
        "center_residual_p90_px": _percentile(residuals, 90.0),
        "dustbin_residual_px": float(dustbin_residual_px),
        "requested_residual_bin_px": float(requested_residual_bin_px),
        "outputs": {"rows_csv": str(rows_csv), "summary": str(output / "summary.json")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dustbin_residual_px", type=float, default=4.0)
    parser.add_argument("--requested_residual_bin_px", type=float, default=0.5)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_rgb_patch_fine_rows_from_coarse_measurements(
        coarse_rows_csv=Path(args.coarse_rows_csv),
        output_dir=Path(args.output_dir),
        dustbin_residual_px=float(args.dustbin_residual_px),
        requested_residual_bin_px=float(args.requested_residual_bin_px),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
