from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.dense_depth_measurement_fusion import (
    DENSE_DEPTH_FUSION_FIELDNAMES,
    GEOMETRY_SOURCE_CHOICES,
    augment_rows_with_query_gt_projection,
    dense_depth_measurement_summary,
    dense_depth_pose_ablation_from_rows,
    dense_depth_rows_from_rows,
)


ABLATION_FIELDNAMES = [
    "group_id",
    "query_id",
    "candidate_id",
    "variant",
    "solver",
    "success",
    "match_count",
    "inlier_count",
    "inlier_ratio",
    "translation_error_m",
    "rotation_error_deg",
    "residual_median_px",
    "residual_p90_px",
]

MATRIX_FIELDNAMES = [
    "variant",
    "solver",
    "query_count",
    "pnp_success_rate",
    "median_translation_error_m",
    "p90_translation_error_m",
    "median_rotation_error_deg",
    "p90_rotation_error_deg",
    "success_3cm_1deg",
    "success_5cm_2deg",
    "success_10cm_5deg",
    "median_residual_median_px",
    "median_inlier_count",
    "depth_valid_rate",
    "measurement_epe_median_px",
    "measurement_epe_p90_px",
    "measurement_improve_ratio",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _pose_from_json(text: str) -> np.ndarray:
    pose = np.asarray(json.loads(str(text)), dtype=np.float64)
    return pose.reshape(4, 4)


def _load_camera_from_model_dir(path: Path) -> ColmapCamera:
    root = Path(path)
    cameras_path = root if root.name == "cameras.bin" else root / "cameras.bin"
    cameras = read_colmap_cameras_binary(cameras_path)
    if not cameras:
        raise ValueError(f"no COLMAP cameras found in {cameras_path}")
    ordered = sorted(cameras.values(), key=lambda item: item.camera_id)
    return ordered[len(ordered) // 2]


def _build_camera(args: argparse.Namespace) -> ColmapCamera:
    if str(args.camera_model_dir).strip():
        return _load_camera_from_model_dir(Path(args.camera_model_dir))
    if args.camera_width is None or args.camera_height is None or not args.camera_params:
        raise ValueError("--camera_model_dir or all of --camera_width/--camera_height/--camera_params is required")
    return ColmapCamera(
        camera_id=1,
        model_id=int(args.camera_model_id),
        width=int(args.camera_width),
        height=int(args.camera_height),
        params=tuple(float(value) for value in args.camera_params),
    )


def _query_pose_lookup(path: Path) -> dict[str, np.ndarray]:
    return {str(record.image_id): np.asarray(record.pose_w2c, dtype=np.float64).reshape(4, 4) for record in parse_cambridge_pose_file(Path(path))}


def _ablation_rows(report: Mapping[str, Any], *, query_id: str = "", candidate_id: str = "", group_id: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, solvers in dict(report.get("variants", {})).items():
        for solver, values in dict(solvers).items():
            rows.append(
                {
                    "group_id": str(group_id),
                    "query_id": str(query_id),
                    "candidate_id": str(candidate_id),
                    "variant": str(variant),
                    "solver": str(solver),
                    **dict(values),
                }
            )
    return rows


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def _aggregate_pose_rows(rows: Sequence[Mapping[str, Any]], measurement_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("variant", "")), str(row.get("solver", ""))), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (variant, solver), values in sorted(by_key.items()):
        query_count = int(len(values))
        successes = [_bool_value(row.get("success")) for row in values]
        translations = [_finite_float(row.get("translation_error_m")) for row in values]
        rotations = [_finite_float(row.get("rotation_error_deg")) for row in values]
        residuals = [_finite_float(row.get("residual_median_px")) for row in values]
        inliers = [_finite_float(row.get("inlier_count")) for row in values]

        def success_rate(t_threshold: float, r_threshold: float) -> float:
            count = 0
            for ok, t_err, r_err in zip(successes, translations, rotations):
                if ok and t_err is not None and r_err is not None and t_err <= t_threshold and r_err <= r_threshold:
                    count += 1
            return float(count / query_count) if query_count else 0.0

        summary_rows.append(
            {
                "variant": variant,
                "solver": solver,
                "query_count": query_count,
                "pnp_success_rate": float(sum(1 for item in successes if item) / query_count) if query_count else 0.0,
                "median_translation_error_m": _percentile([item for item in translations if item is not None], 50.0),
                "p90_translation_error_m": _percentile([item for item in translations if item is not None], 90.0),
                "median_rotation_error_deg": _percentile([item for item in rotations if item is not None], 50.0),
                "p90_rotation_error_deg": _percentile([item for item in rotations if item is not None], 90.0),
                "success_3cm_1deg": success_rate(0.03, 1.0),
                "success_5cm_2deg": success_rate(0.05, 2.0),
                "success_10cm_5deg": success_rate(0.10, 5.0),
                "median_residual_median_px": _percentile([item for item in residuals if item is not None], 50.0),
                "median_inlier_count": _percentile([item for item in inliers if item is not None], 50.0),
                "depth_valid_rate": measurement_summary.get("depth_valid_rate", ""),
                "measurement_epe_median_px": measurement_summary.get("measurement_epe_median_px", ""),
                "measurement_epe_p90_px": measurement_summary.get("measurement_epe_p90_px", ""),
                "measurement_improve_ratio": measurement_summary.get("measurement_improve_ratio", ""),
            }
        )
    return summary_rows


def _candidate_id(row: Mapping[str, Any]) -> str:
    # Many match tables use candidate_id for match-level/top-L cell candidates,
    # not for pose hypotheses. Treat only explicit pose-level fields as a pose
    # group key; otherwise the correct grouping is query-level.
    for name in ("render_pose_id", "pose_candidate_label", "initial_render_pose_label", "render_pose_label"):
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _group_id(query_id: str, candidate_id: str) -> str:
    return str(query_id) if not str(candidate_id).strip() else f"{query_id}::{candidate_id}"


def _rows_by_pose_group(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        candidate_id = _candidate_id(row)
        grouped.setdefault((query_id, candidate_id), []).append(row)
    return grouped


def _candidate_group_summary(groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    per_query: dict[str, set[str]] = {}
    for query_id, candidate_id in groups.keys():
        per_query.setdefault(str(query_id), set()).add(str(candidate_id))
    counts = [len(values) for values in per_query.values()]
    return {
        "query_count": int(len(per_query)),
        "pose_group_count": int(len(groups)),
        "max_candidates_per_query": int(max(counts)) if counts else 0,
        "min_candidates_per_query": int(min(counts)) if counts else 0,
        "candidate_count_by_query": {query_id: int(len(values)) for query_id, values in sorted(per_query.items())},
    }


def evaluate_dense_depth_measurement_fusion(
    *,
    match_table_csv: Path,
    output_dir: Path,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray | None = None,
    gt_pose_w2c: np.ndarray | None = None,
    query_pose_w2c_by_id: Mapping[str, np.ndarray] | None = None,
    variants: Sequence[str] = ("center", "measurement", "oracle"),
    solvers: Sequence[str] = ("ransac", "weighted", "covariance", "oracle_uncertainty"),
    reprojection_error_px: float = 8.0,
    geometry_source: str = "force_backproject",
    world_xyz_consistency_threshold_m: float = 1e-4,
    strict_measurement_schema: bool = True,
) -> dict[str, Any]:
    rows = _read_csv(Path(match_table_csv))
    query_gt_projection_summary: dict[str, Any] | None = None
    if query_pose_w2c_by_id:
        rows, query_gt_projection_summary = augment_rows_with_query_gt_projection(
            rows,
            query_pose_w2c_by_id=query_pose_w2c_by_id,
            camera=camera,
        )
    output = Path(output_dir)
    groups = _rows_by_pose_group(rows)
    if render_pose_w2c is not None and len(groups) != 1:
        raise ValueError("--render_pose_w2c_json is only valid for a single query/candidate group")
    dense_rows = dense_depth_rows_from_rows(
        rows,
        camera=camera,
        render_pose_w2c=render_pose_w2c,
        geometry_source=str(geometry_source),
        world_xyz_consistency_threshold_m=float(world_xyz_consistency_threshold_m),
    )
    measurement_summary = dense_depth_measurement_summary(dense_rows)
    query_reports: dict[str, Any] = {}
    ablation_rows: list[dict[str, Any]] = []
    for (query_id, candidate_id), query_rows in sorted(groups.items()):
        query_gt_pose = (
            None
            if query_pose_w2c_by_id is None
            else query_pose_w2c_by_id.get(str(query_id))
        )
        if query_gt_pose is None:
            query_gt_pose = gt_pose_w2c
        report = dense_depth_pose_ablation_from_rows(
            query_rows,
            camera=camera,
            render_pose_w2c=render_pose_w2c,
            gt_pose_w2c=query_gt_pose,
            variants=tuple(str(value) for value in variants),
            solvers=tuple(str(value) for value in solvers),
            reprojection_error_px=float(reprojection_error_px),
            geometry_source=str(geometry_source),
            world_xyz_consistency_threshold_m=float(world_xyz_consistency_threshold_m),
            strict_measurement_schema=bool(strict_measurement_schema),
        )
        gid = _group_id(str(query_id), str(candidate_id))
        query_reports[gid] = report
        ablation_rows.extend(_ablation_rows(report, query_id=str(query_id), candidate_id=str(candidate_id), group_id=gid))
    aggregate_rows = _aggregate_pose_rows(ablation_rows, measurement_summary)
    _write_csv(output / "match_table.csv", dense_rows, DENSE_DEPTH_FUSION_FIELDNAMES)
    _write_jsonl(output / "match_table.jsonl", dense_rows)
    _write_csv(output / "pose_rows.csv", ablation_rows, ABLATION_FIELDNAMES)
    _write_csv(output / "ablation_summary.tsv", ablation_rows, ABLATION_FIELDNAMES, delimiter="\t")
    summary = {
        "stage": "radio_matcha_dense_depth_measurement_fusion_cli",
        "input_count": int(len(rows)),
        "query_count": int(len({str(row.get("query_id", "")) for row in rows})),
        "pose_group_count": int(len(groups)),
        "candidate_group_summary": _candidate_group_summary(groups),
        "camera": {
            "camera_id": int(camera.camera_id),
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        },
        "query_gt_projection_summary": query_gt_projection_summary,
        "geometry_source": str(geometry_source),
        "world_xyz_consistency_threshold_m": float(world_xyz_consistency_threshold_m),
        "strict_measurement_schema": bool(strict_measurement_schema),
        "dense_depth_summary": measurement_summary,
        "pose_summary": aggregate_rows,
        "pose_ablation_by_query": query_reports,
        "outputs": {
            "match_table_csv": str(output / "match_table.csv"),
            "match_table_jsonl": str(output / "match_table.jsonl"),
            "pose_rows_csv": str(output / "pose_rows.csv"),
            "ablation_summary_tsv": str(output / "ablation_summary.tsv"),
            "matrix_summary_tsv": str(output / "matrix_summary.tsv"),
            "summary": str(output / "summary.json"),
        },
    }
    _write_csv(output / "matrix_summary.tsv", aggregate_rows, MATRIX_FIELDNAMES, delimiter="\t")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--camera_width", type=int)
    parser.add_argument("--camera_height", type=int)
    parser.add_argument("--camera_model_id", type=int, default=1)
    parser.add_argument("--camera_params", nargs="+", type=float, default=[])
    parser.add_argument("--query_pose_file", default="")
    parser.add_argument(
        "--render_pose_w2c_json",
        default="",
        help="Optional single render pose for backprojecting render_x/render_y/render_depth. "
        "Omit when match_table already contains world_x/world_y/world_z.",
    )
    parser.add_argument("--gt_pose_w2c_json", default="")
    parser.add_argument("--variants", nargs="+", default=["center", "measurement", "oracle"])
    parser.add_argument("--solvers", nargs="+", default=["ransac", "weighted", "covariance", "oracle_uncertainty"])
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--geometry_source", default="force_backproject", choices=list(GEOMETRY_SOURCE_CHOICES))
    parser.add_argument("--world_xyz_consistency_threshold_m", type=float, default=1e-4)
    parser.add_argument("--no_strict_measurement_schema", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_dense_depth_measurement_fusion(
        match_table_csv=Path(args.match_table_csv),
        output_dir=Path(args.output_dir),
        camera=_build_camera(args),
        render_pose_w2c=None if not str(args.render_pose_w2c_json).strip() else _pose_from_json(args.render_pose_w2c_json),
        gt_pose_w2c=None if not str(args.gt_pose_w2c_json).strip() else _pose_from_json(args.gt_pose_w2c_json),
        query_pose_w2c_by_id=(
            None if not str(args.query_pose_file).strip() else _query_pose_lookup(Path(args.query_pose_file))
        ),
        variants=[str(value) for value in args.variants],
        solvers=[str(value) for value in args.solvers],
        reprojection_error_px=float(args.reprojection_error_px),
        geometry_source=str(args.geometry_source),
        world_xyz_consistency_threshold_m=float(args.world_xyz_consistency_threshold_m),
        strict_measurement_schema=not bool(args.no_strict_measurement_schema),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
