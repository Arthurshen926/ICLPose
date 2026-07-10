from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.landmark_hybrid import cameras_by_image_name
from feature_extract.vfm.measurement_v1.d0_row_builder import project_world_points_to_image


@dataclass(frozen=True)
class ResidualAuditConfig:
    thresholds_px: tuple[float, ...] = (2.0, 5.0, 8.0)
    group_key: str = "query_id"


def _parse_xyz(value: str) -> np.ndarray:
    parsed = ast.literal_eval(str(value))
    arr = np.asarray(parsed, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"xyz must contain three values, got {value!r}")
    return arr


def _read_match_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def summarize_match_reprojection_residuals(
    rows: Sequence[Mapping[str, Any]],
    *,
    pose_w2c_by_query: Mapping[str, np.ndarray],
    camera_by_query: Mapping[str, ColmapCamera],
    config: ResidualAuditConfig = ResidualAuditConfig(),
) -> dict[str, Any]:
    """Summarize GT-pose reprojection residuals for match CSV rows."""

    thresholds = tuple(sorted(float(value) for value in config.thresholds_px))
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get(config.group_key, "")).strip()
        if query_id:
            groups[query_id].append(row)

    per_query: list[dict[str, Any]] = []
    residuals_all: list[float] = []
    visible_all = 0
    missing_pose = 0
    missing_camera = 0
    processed_match_count = 0

    for query_id in sorted(groups):
        query_rows = groups[query_id]
        pose = pose_w2c_by_query.get(query_id)
        camera = camera_by_query.get(query_id)
        if pose is None:
            missing_pose += len(query_rows)
            continue
        if camera is None:
            missing_camera += len(query_rows)
            continue
        points: list[np.ndarray] = []
        xy: list[tuple[float, float]] = []
        token_indices: list[int] = []
        for row in query_rows:
            try:
                point = _parse_xyz(str(row.get("xyz", "")))
                x = _finite_float(row.get("x"))
                y = _finite_float(row.get("y"))
            except (SyntaxError, ValueError):
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            points.append(point)
            xy.append((float(x), float(y)))
            try:
                token_indices.append(int(row.get("token_index", -1)))
            except (TypeError, ValueError):
                token_indices.append(-1)
        if not points:
            continue
        projected, visible = project_world_points_to_image(
            np.stack(points, axis=0),
            np.asarray(pose, dtype=np.float64).reshape(4, 4),
            camera,
        )
        query_xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        residuals = np.linalg.norm(projected - query_xy, axis=1).astype(np.float64)
        finite = np.isfinite(residuals)
        processed_match_count += int(finite.sum())
        visible_count = int(np.asarray(visible, dtype=bool).reshape(-1)[finite].sum())
        visible_all += visible_count
        residual_list = [float(value) for value in residuals[finite]]
        residuals_all.extend(residual_list)
        unique_token_best: dict[int, float] = {}
        for token_index, residual in zip(token_indices, residuals):
            if not np.isfinite(residual):
                continue
            previous = unique_token_best.get(int(token_index))
            if previous is None or float(residual) < previous:
                unique_token_best[int(token_index)] = float(residual)
        row_out: dict[str, Any] = {
            "query_id": query_id,
            "match_count": int(len(residual_list)),
            "gt_visible_count": visible_count,
            "median_residual_px": None if not residual_list else float(np.median(residual_list)),
            "p90_residual_px": None if not residual_list else float(np.percentile(residual_list, 90)),
            "oracle_unique_token_count": int(len(unique_token_best)),
            "oracle_unique_token_median_residual_px": None
            if not unique_token_best
            else float(np.median(list(unique_token_best.values()))),
        }
        for threshold in thresholds:
            key = _threshold_key(threshold)
            count = int(np.sum(np.asarray(residual_list, dtype=np.float64) <= threshold)) if residual_list else 0
            row_out[f"valid_{key}px_count"] = count
            row_out[f"valid_{key}px_rate"] = 0.0 if not residual_list else float(count / len(residual_list))
            oracle_count = int(
                np.sum(np.asarray(list(unique_token_best.values()), dtype=np.float64) <= threshold)
            ) if unique_token_best else 0
            row_out[f"oracle_unique_valid_{key}px_count"] = oracle_count
            row_out[f"oracle_unique_valid_{key}px_rate"] = (
                0.0 if not unique_token_best else float(oracle_count / len(unique_token_best))
            )
        per_query.append(row_out)

    residual_array = np.asarray(residuals_all, dtype=np.float64)
    summary: dict[str, Any] = {
        "query_count": int(len(per_query)),
        "input_row_count": int(len(rows)),
        "processed_match_count": int(processed_match_count),
        "missing_pose_match_count": int(missing_pose),
        "missing_camera_match_count": int(missing_camera),
        "gt_visible_count": int(visible_all),
        "median_residual_px": None if residual_array.size == 0 else float(np.median(residual_array)),
        "p90_residual_px": None if residual_array.size == 0 else float(np.percentile(residual_array, 90)),
        "mean_match_count": 0.0 if not per_query else float(np.mean([row["match_count"] for row in per_query])),
        "median_match_count": 0.0 if not per_query else float(np.median([row["match_count"] for row in per_query])),
        "mean_oracle_unique_token_count": 0.0
        if not per_query
        else float(np.mean([row["oracle_unique_token_count"] for row in per_query])),
        "median_oracle_unique_token_count": 0.0
        if not per_query
        else float(np.median([row["oracle_unique_token_count"] for row in per_query])),
        "per_query": per_query,
    }
    for threshold in thresholds:
        key = _threshold_key(threshold)
        count = int(np.sum(residual_array <= threshold)) if residual_array.size else 0
        summary[f"valid_{key}px_count"] = count
        summary[f"valid_{key}px_rate"] = 0.0 if residual_array.size == 0 else float(count / residual_array.size)
        per_counts = [row[f"valid_{key}px_count"] for row in per_query]
        per_rates = [row[f"valid_{key}px_rate"] for row in per_query]
        summary[f"mean_valid_{key}px_count_per_query"] = 0.0 if not per_counts else float(np.mean(per_counts))
        summary[f"median_valid_{key}px_count_per_query"] = 0.0 if not per_counts else float(np.median(per_counts))
        summary[f"mean_valid_{key}px_rate_per_query"] = 0.0 if not per_rates else float(np.mean(per_rates))
        oracle_counts = [row[f"oracle_unique_valid_{key}px_count"] for row in per_query]
        oracle_rates = [row[f"oracle_unique_valid_{key}px_rate"] for row in per_query]
        summary[f"mean_oracle_unique_valid_{key}px_count_per_query"] = (
            0.0 if not oracle_counts else float(np.mean(oracle_counts))
        )
        summary[f"median_oracle_unique_valid_{key}px_count_per_query"] = (
            0.0 if not oracle_counts else float(np.median(oracle_counts))
        )
        summary[f"mean_oracle_unique_valid_{key}px_rate_per_query"] = (
            0.0 if not oracle_rates else float(np.mean(oracle_rates))
        )
    return summary


def _threshold_key(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    return str(float(value)).replace(".", "p")


def _pose_lookup(path: Path) -> dict[str, np.ndarray]:
    return {str(record.image_id): np.asarray(record.pose_w2c, dtype=np.float64).reshape(4, 4) for record in parse_cambridge_pose_file(Path(path))}


def _load_cameras(model_dir: Path, image_root: Path) -> dict[str, ColmapCamera]:
    cameras = read_colmap_cameras_binary(Path(model_dir) / "cameras.bin")
    images = read_colmap_images_binary(Path(model_dir) / "images.bin")
    return cameras_by_image_name(cameras=cameras, colmap_images=images, image_root=Path(image_root))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches_csv", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--thresholds_px", nargs="+", type=float, default=[2.0, 5.0, 8.0])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    rows = _read_match_rows(Path(args.matches_csv))
    summary = summarize_match_reprojection_residuals(
        rows,
        pose_w2c_by_query=_pose_lookup(Path(args.query_pose_file)),
        camera_by_query=_load_cameras(Path(args.colmap_model_dir), Path(args.image_root)),
        config=ResidualAuditConfig(thresholds_px=tuple(float(value) for value in args.thresholds_px)),
    )
    per_query = list(summary.pop("per_query"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "per_query_csv": str(output_dir / "per_query.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf8")
    if per_query:
        fieldnames = list(per_query[0].keys())
        _write_csv(output_dir / "per_query.csv", per_query, fieldnames)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
