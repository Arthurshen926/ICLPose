"""Build RGB patch measurement training rows from a dense-depth match table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


OUTPUT_FIELDNAMES = [
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
    "radio_match_score",
    "source_match_table",
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
    seen = set(OUTPUT_FIELDNAMES)
    fieldnames = list(OUTPUT_FIELDNAMES)
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


def _load_camera_from_model_dir(path: Path) -> ColmapCamera:
    root = Path(path)
    cameras_path = root if root.name == "cameras.bin" else root / "cameras.bin"
    cameras = read_colmap_cameras_binary(cameras_path)
    if not cameras:
        raise ValueError(f"no COLMAP cameras found in {cameras_path}")
    ordered = sorted(cameras.values(), key=lambda item: item.camera_id)
    return ordered[len(ordered) // 2]


def _pose_lookup(path: Path) -> dict[str, np.ndarray]:
    return {str(record.image_id): np.asarray(record.pose_w2c, dtype=np.float64).reshape(4, 4) for record in parse_cambridge_pose_file(Path(path))}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def build_rgb_patch_measurement_rows_from_match_table(
    *,
    match_table_csv: Path,
    query_pose_file: Path,
    camera_model_dir: Path,
    output_dir: Path,
    max_rows: int | None = None,
    max_center_residual_px: float | None = None,
    min_center_residual_px: float | None = None,
    dustbin_residual_px: float | None = None,
    requested_residual_bin_px: float = 1.0,
) -> dict[str, Any]:
    rows = _read_csv(Path(match_table_csv), max_rows=max_rows)
    poses = _pose_lookup(Path(query_pose_file))
    camera = _load_camera_from_model_dir(Path(camera_model_dir))
    output_rows: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_filter = 0
    residuals: list[float] = []
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        pose = poses.get(query_id)
        center_x = _optional_float(row, "query_center_x", "center_x", "query_x")
        center_y = _optional_float(row, "query_center_y", "center_y", "query_y")
        render_x = _optional_float(row, "render_x")
        render_y = _optional_float(row, "render_y")
        world_x = _optional_float(row, "world_x", "x")
        world_y = _optional_float(row, "world_y", "y")
        world_z = _optional_float(row, "world_z", "z")
        if (
            not query_id
            or pose is None
            or center_x is None
            or center_y is None
            or render_x is None
            or render_y is None
            or world_x is None
            or world_y is None
            or world_z is None
        ):
            skipped_missing += 1
            continue
        projected = project_world_to_image(
            np.asarray([[world_x, world_y, world_z]], dtype=np.float64),
            pose,
            camera,
        )[0]
        query_gt_x = float(projected[0])
        query_gt_y = float(projected[1])
        dx = query_gt_x - float(center_x)
        dy = query_gt_y - float(center_y)
        residual = float(math.hypot(dx, dy))
        if max_center_residual_px is not None and residual > float(max_center_residual_px):
            skipped_filter += 1
            continue
        if min_center_residual_px is not None and residual < float(min_center_residual_px):
            skipped_filter += 1
            continue
        target_is_dustbin = bool(dustbin_residual_px is not None and residual > float(dustbin_residual_px))
        bin_px = float(requested_residual_bin_px)
        requested_residual = float(round(residual / bin_px) * bin_px) if bin_px > 0.0 else residual
        residuals.append(residual)
        output_rows.append(
            {
                "query_id": query_id,
                "match_index": row.get("match_index", ""),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "query_gt_x": query_gt_x,
                "query_gt_y": query_gt_y,
                "render_x": float(render_x),
                "render_y": float(render_y),
                "render_depth": row.get("render_depth", ""),
                "world_x": float(world_x),
                "world_y": float(world_y),
                "world_z": float(world_z),
                "center_residual_px": residual,
                "center_residual_dx": dx,
                "center_residual_dy": dy,
                "target_is_dustbin": target_is_dustbin,
                "requested_residual_px": requested_residual,
                "measurement_policy": "match_table_projected_query_gt",
                "radio_match_score": row.get("radio_match_score", row.get("confidence", row.get("similarity", ""))),
                "source_match_table": str(match_table_csv),
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_csv = output / "measurement_rows.csv"
    _write_csv(rows_csv, output_rows)
    summary = {
        "stage": "rgb_patch_measurement_rows_from_match_table",
        "match_table_csv": str(match_table_csv),
        "query_pose_file": str(query_pose_file),
        "camera_model_dir": str(camera_model_dir),
        "input_count": int(len(rows)),
        "row_count": int(len(output_rows)),
        "skipped_missing_count": int(skipped_missing),
        "skipped_filter_count": int(skipped_filter),
        "center_residual_median_px": _percentile(residuals, 50.0),
        "center_residual_p90_px": _percentile(residuals, 90.0),
        "center_residual_max_px": max(residuals) if residuals else None,
        "max_center_residual_px": None if max_center_residual_px is None else float(max_center_residual_px),
        "min_center_residual_px": None if min_center_residual_px is None else float(min_center_residual_px),
        "dustbin_residual_px": None if dustbin_residual_px is None else float(dustbin_residual_px),
        "requested_residual_bin_px": float(requested_residual_bin_px),
        "outputs": {
            "rows_csv": str(rows_csv),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table_csv", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_center_residual_px", type=float, default=-1.0)
    parser.add_argument("--min_center_residual_px", type=float, default=-1.0)
    parser.add_argument("--dustbin_residual_px", type=float, default=-1.0)
    parser.add_argument("--requested_residual_bin_px", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_rgb_patch_measurement_rows_from_match_table(
        match_table_csv=Path(args.match_table_csv),
        query_pose_file=Path(args.query_pose_file),
        camera_model_dir=Path(args.camera_model_dir),
        output_dir=Path(args.output_dir),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        max_center_residual_px=float(args.max_center_residual_px) if float(args.max_center_residual_px) >= 0.0 else None,
        min_center_residual_px=float(args.min_center_residual_px) if float(args.min_center_residual_px) >= 0.0 else None,
        dustbin_residual_px=float(args.dustbin_residual_px) if float(args.dustbin_residual_px) >= 0.0 else None,
        requested_residual_bin_px=float(args.requested_residual_bin_px),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
