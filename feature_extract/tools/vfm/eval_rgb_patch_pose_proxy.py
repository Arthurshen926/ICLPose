"""Evaluate RGB-patch measurement predictions through a COLMAP PnP pose proxy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
    read_colmap_points3d_binary,
)
from feature_extract.vfm.measurement_v1.rgb_patch_pose_proxy import evaluate_pose_proxy_from_prediction_rows


POSE_ROW_FIELDNAMES = [
    "query_id",
    "match_count",
    "success",
    "inlier_count",
    "translation_error_m",
    "rotation_error_deg",
    "failure_reason",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _prediction_rows_with_selected_columns(
    rows: Sequence[Mapping[str, object]],
    *,
    prediction_x_key: str,
    prediction_y_key: str,
) -> list[dict[str, object]]:
    x_key = str(prediction_x_key).strip()
    y_key = str(prediction_y_key).strip()
    if not x_key or not y_key:
        raise ValueError("prediction_x_key and prediction_y_key must be non-empty")
    out: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if str(row.get(x_key, "")).strip() == "":
            raise ValueError(f"row {index} missing prediction_x_key={x_key!r}")
        if str(row.get(y_key, "")).strip() == "":
            raise ValueError(f"row {index} missing prediction_y_key={y_key!r}")
        out.append({**row, "query_pred_x": row.get(x_key), "query_pred_y": row.get(y_key)})
    return out


def _filter_rows_by_baseline_epe(
    rows: Sequence[Mapping[str, object]],
    *,
    min_baseline_epe_px: float | None,
    max_baseline_epe_px: float | None,
) -> list[Mapping[str, object]]:
    if min_baseline_epe_px is None and max_baseline_epe_px is None:
        return list(rows)
    out: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        text = str(row.get("baseline_epe_px", "")).strip()
        if not text:
            raise ValueError(f"row {index} missing baseline_epe_px required for baseline filtering")
        value = float(text)
        if min_baseline_epe_px is not None and value < float(min_baseline_epe_px):
            continue
        if max_baseline_epe_px is not None and value > float(max_baseline_epe_px):
            continue
        out.append(row)
    return out


def evaluate_rgb_patch_pose_proxy(
    *,
    prediction_rows_csv: Path,
    model_dir: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    prediction_x_key: str = "query_pred_x",
    prediction_y_key: str = "query_pred_y",
    min_baseline_epe_px: float | None = None,
    max_baseline_epe_px: float | None = None,
    dustbin_threshold: float = 0.5,
    reprojection_error_px: float = 8.0,
    min_inliers: int = 4,
) -> dict[str, Any]:
    input_rows = _read_csv(Path(prediction_rows_csv))
    filtered_rows = _filter_rows_by_baseline_epe(
        input_rows,
        min_baseline_epe_px=min_baseline_epe_px,
        max_baseline_epe_px=max_baseline_epe_px,
    )
    rows = _prediction_rows_with_selected_columns(
        filtered_rows,
        prediction_x_key=str(prediction_x_key),
        prediction_y_key=str(prediction_y_key),
    )
    model = Path(model_dir)
    cameras = read_colmap_cameras_binary(model / "cameras.bin")
    images = read_colmap_images_binary(model / "images.bin")
    points = read_colmap_points3d_binary(model / "points3D.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    xyz_by_track = {int(point_id): np.asarray(point.xyz, dtype=np.float64).reshape(3) for point_id, point in points.items()}
    summary = evaluate_pose_proxy_from_prediction_rows(
        rows,
        cameras=cameras,
        images_by_name=images_by_name,
        xyz_by_track=xyz_by_track,
        image_width=int(image_width),
        image_height=int(image_height),
        dustbin_threshold=float(dustbin_threshold),
        reprojection_error_px=float(reprojection_error_px),
        min_inliers=int(min_inliers),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pose_rows = list(summary.pop("pose_rows"))
    _write_csv(output / "pose_rows.csv", pose_rows, POSE_ROW_FIELDNAMES)
    summary = {
        **summary,
        "stage": "measurement_v1_rgb_patch_pose_proxy",
        "prediction_rows_csv": str(prediction_rows_csv),
        "model_dir": str(model_dir),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "prediction_x_key": str(prediction_x_key),
        "prediction_y_key": str(prediction_y_key),
        "input_row_count": int(len(input_rows)),
        "filtered_row_count": int(len(filtered_rows)),
        "min_baseline_epe_px": None if min_baseline_epe_px is None else float(min_baseline_epe_px),
        "max_baseline_epe_px": None if max_baseline_epe_px is None else float(max_baseline_epe_px),
        "outputs": {
            "pose_rows": str(output / "pose_rows.csv"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_rows_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--prediction_x_key", default="query_pred_x")
    parser.add_argument("--prediction_y_key", default="query_pred_y")
    parser.add_argument("--min_baseline_epe_px", type=float, default=None)
    parser.add_argument("--max_baseline_epe_px", type=float, default=None)
    parser.add_argument("--dustbin_threshold", type=float, default=0.5)
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--min_inliers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_rgb_patch_pose_proxy(
        prediction_rows_csv=Path(args.prediction_rows_csv),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        prediction_x_key=str(args.prediction_x_key),
        prediction_y_key=str(args.prediction_y_key),
        min_baseline_epe_px=args.min_baseline_epe_px,
        max_baseline_epe_px=args.max_baseline_epe_px,
        dustbin_threshold=float(args.dustbin_threshold),
        reprojection_error_px=float(args.reprojection_error_px),
        min_inliers=int(args.min_inliers),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
