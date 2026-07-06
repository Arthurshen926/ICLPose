"""Run D-1 measurement error budget diagnostics on fixed-anchor rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.measurement_v1.error_budget import ErrorBudgetConfig, run_error_budget_for_rows


def _read_csv(path: Path, max_rows: int = 0) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
        return rows


def _csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in str(text).split(",") if item.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True, help="match_table.csv or measurement rows with query/world coordinates")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--noise_sigmas_px", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--quantization_strides_px", default="0,2,4,8,16")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=2000)
    parser.add_argument("--pnp_min_inliers", type=int, default=6)
    parser.add_argument("--rng_seed", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_model_dir = _infer_camera_model_dir(str(args.query_pose_file), str(args.camera_model_dir))
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(str(args.default_camera)))
    pose_records = parse_cambridge_pose_file(Path(args.query_pose_file))
    gt_pose_by_query = {record.image_id: record.pose_w2c for record in pose_records}
    config = ErrorBudgetConfig(
        noise_sigmas_px=_csv_floats(str(args.noise_sigmas_px)),
        quantization_strides_px=_csv_floats(str(args.quantization_strides_px)),
        trials=int(args.trials),
        pnp_reprojection_error_px=float(args.pnp_reprojection_error_px),
        pnp_iterations=int(args.pnp_iterations),
        pnp_min_inliers=int(args.pnp_min_inliers),
        rng_seed=int(args.rng_seed),
    )
    rows = _read_csv(Path(args.rows_csv), max_rows=int(args.max_rows))
    report = run_error_budget_for_rows(rows=rows, gt_pose_by_query=gt_pose_by_query, camera=camera, config=config)
    summary = {
        "stage": "measurement_v1_error_budget_cli",
        "config": {
            "rows_csv": str(args.rows_csv),
            "query_pose_file": str(args.query_pose_file),
            "camera_source": str(camera_source),
            "noise_sigmas_px": list(config.noise_sigmas_px),
            "quantization_strides_px": list(config.quantization_strides_px),
            "trials": int(config.trials),
        },
        "metrics": {key: value for key, value in report.items() if key != "variant_rows"},
        "outputs": {"summary": str(output_dir / "summary.json")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
