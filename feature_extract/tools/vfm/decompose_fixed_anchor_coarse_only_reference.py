"""Decompose fixed-anchor coarse-only reference topK results into oracle stages."""

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
from feature_extract.vfm.measurement_v1.reference_oracle_decomposition import (
    OracleDecompositionConfig,
    decompose_match_table,
    summarize_decomposition_rows,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=2000)
    parser.add_argument("--pnp_min_inliers", type=int, default=6)
    parser.add_argument("--validity_thresholds_px", default="2,5,10,16")
    parser.add_argument("--topk_values", default="1,3,5,10")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_model_dir = _infer_camera_model_dir(str(args.query_pose_file), str(args.camera_model_dir))
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(str(args.default_camera)))
    pose_records = parse_cambridge_pose_file(Path(args.query_pose_file))
    gt_pose_by_query = {record.image_id: record.pose_w2c for record in pose_records}
    config = OracleDecompositionConfig(
        pnp_reprojection_error_px=float(args.pnp_reprojection_error_px),
        pnp_iterations=int(args.pnp_iterations),
        pnp_min_inliers=int(args.pnp_min_inliers),
        validity_thresholds_px=tuple(
            float(item) for item in str(args.validity_thresholds_px).split(",") if item.strip()
        ),
        topk_values=tuple(int(float(item)) for item in str(args.topk_values).split(",") if item.strip()),
    )
    match_rows = _read_csv(Path(args.match_table))
    candidate_rows = decompose_match_table(
        match_rows,
        gt_pose_by_query=gt_pose_by_query,
        camera=camera,
        config=config,
    )
    summary = {
        "stage": "fixed_anchor_coarse_only_reference_oracle_decomposition",
        "config": {
            "match_table": str(args.match_table),
            "query_pose_file": str(args.query_pose_file),
            "camera_source": str(camera_source),
            "pnp_reprojection_error_px": float(config.pnp_reprojection_error_px),
            "pnp_iterations": int(config.pnp_iterations),
            "pnp_min_inliers": int(config.pnp_min_inliers),
            "validity_thresholds_px": list(config.validity_thresholds_px),
            "topk_values": list(config.topk_values),
        },
        "metrics": summarize_decomposition_rows(candidate_rows, topk_values=config.topk_values),
        "outputs": {
            "candidate_oracle_rows": str(output_dir / "candidate_oracle_rows.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    _write_csv(output_dir / "candidate_oracle_rows.csv", candidate_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
