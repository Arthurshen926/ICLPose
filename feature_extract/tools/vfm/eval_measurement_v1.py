from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.d0_probe import parse_feature_specs, run_d0_probe
from feature_extract.vfm.measurement_v1.gt_measurement_export import export_gt_measurement_from_legacy_eval
from feature_extract.vfm.measurement_v1.local_likelihood import compute_local_likelihood
from feature_extract.vfm.measurement_v1.legacy_report import report_to_markdown, summarize_legacy_measurement_eval
from feature_extract.vfm.measurement_v1.probabilistic_pnp import estimate_pose_from_measurements
from feature_extract.vfm.measurement_v1.protocol_lock import build_protocol_lock_report
from feature_extract.vfm.measurement_v1.types import QueryMeasurement, SurfaceAnchor
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _synthetic_camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _synthetic_anchor(anchor_id: int, xy: tuple[float, float], depth: float, camera: ColmapCamera) -> SurfaceAnchor:
    x = (float(xy[0]) - float(camera.params[2])) / float(camera.params[0]) * float(depth)
    y = (float(xy[1]) - float(camera.params[3])) / float(camera.params[1]) * float(depth)
    return SurfaceAnchor(
        anchor_id=anchor_id,
        token_index=anchor_id,
        subanchor_index=0,
        render_xy_px=np.asarray(xy, dtype=np.float64),
        world_xyz=np.asarray([x, y, depth], dtype=np.float64),
        cov_world_3x3=np.eye(3, dtype=np.float64) * 1e-6,
        depth_m=float(depth),
        alpha=1.0,
        quality=1.0,
        normal_world=None,
        surface_id=None,
    )


def _build_synthetic_feature_map(anchors: Sequence[SurfaceAnchor]) -> np.ndarray:
    channels = len(anchors)
    fmap = np.zeros((channels, 100, 100), dtype=np.float32)
    for idx, anchor in enumerate(anchors):
        x = int(np.clip(round(float(anchor.render_xy_px[0])), 0, 99))
        y = int(np.clip(round(float(anchor.render_xy_px[1])), 0, 99))
        fmap[idx, y, x] = 10.0
    return fmap


def _run_synthetic_smoke(output_dir: Path) -> None:
    camera = _synthetic_camera()
    xy_values = [
        (25.0, 25.0),
        (75.0, 25.0),
        (25.0, 75.0),
        (75.0, 75.0),
        (50.0, 30.0),
        (60.0, 70.0),
    ]
    anchors = [_synthetic_anchor(idx, xy, 4.0 + 0.25 * idx, camera) for idx, xy in enumerate(xy_values)]
    anchor_rows = [
        {
            "anchor_id": int(anchor.anchor_id),
            "token_index": int(anchor.token_index),
            "subanchor_index": int(anchor.subanchor_index),
            "render_x": float(anchor.render_xy_px[0]),
            "render_y": float(anchor.render_xy_px[1]),
            "X": float(anchor.world_xyz[0]),
            "Y": float(anchor.world_xyz[1]),
            "Z": float(anchor.world_xyz[2]),
            "depth": float(anchor.depth_m),
            "alpha": float(anchor.alpha),
            "depth_gradient": 0.0,
            "depth_variance": 0.0,
            "quality": float(anchor.quality),
            "rejection_reason": "",
        }
        for anchor in anchors
    ]
    feature_map = _build_synthetic_feature_map(anchors)
    measurements: list[QueryMeasurement] = []
    measurement_rows: list[dict[str, Any]] = []
    for idx, anchor in enumerate(anchors):
        descriptor = np.zeros((len(anchors),), dtype=np.float32)
        descriptor[idx] = 1.0
        likelihood = compute_local_likelihood(
            anchor_descriptor=descriptor,
            query_feature_map=feature_map,
            center_xy_px=np.asarray(anchor.render_xy_px, dtype=np.float64),
            image_width=camera.width,
            image_height=camera.height,
            search_radius_px=2.0,
            step_px=1.0,
            temperature=0.25,
            gt_xy_px=np.asarray(anchor.render_xy_px, dtype=np.float64),
        )
        measurement = QueryMeasurement(
            anchor_id=int(anchor.anchor_id),
            query_xy_mean_px=likelihood.mean_xy_px,
            cov_query_2x2=likelihood.cov_query_2x2,
            p_visible=1.0,
            p_assignment=1.0,
            local_log_likelihood=likelihood.local_log_probs,
            mode_probability=float(likelihood.mode_probability),
            diagnostics={"dustbin_probability": float(likelihood.dustbin_probability)},
        )
        measurements.append(measurement)
        residual_before = 0.0
        residual_after = float(np.linalg.norm(np.asarray(measurement.query_xy_mean_px) - np.asarray(anchor.render_xy_px)))
        measurement_rows.append(
            {
                "query_id": "synthetic_smoke",
                "candidate_id": "identity",
                "anchor_id": int(anchor.anchor_id),
                "fit_or_verify": "fit",
                "query_gt_x": float(anchor.render_xy_px[0]),
                "query_gt_y": float(anchor.render_xy_px[1]),
                "query_pred_x": float(measurement.query_xy_mean_px[0]),
                "query_pred_y": float(measurement.query_xy_mean_px[1]),
                "residual_before_px": residual_before,
                "residual_after_px": residual_after,
                "p_visible": float(measurement.p_visible),
                "p_assignment": float(measurement.p_assignment),
                "p_valid": float(measurement.p_valid),
                "cov_xx": float(measurement.cov_query_2x2[0, 0]),
                "cov_xy": float(measurement.cov_query_2x2[0, 1]),
                "cov_yy": float(measurement.cov_query_2x2[1, 1]),
                "mahalanobis2": 0.0,
                "anchor_quality": float(anchor.quality),
                "depth_gradient": None,
                "surface_type": "synthetic",
            }
        )
    pose = estimate_pose_from_measurements(anchors, measurements, camera, use_covariance=True)
    error = pnp_pose_error(pose.pose_w2c, np.eye(4, dtype=np.float64))
    pose_rows = [
        {
            "query_id": "synthetic_smoke",
            "candidate_id": "identity",
            "solver": "covariance_pnp",
            "success": bool(pose.success),
            "translation_error_m": float(error.translation_m),
            "rotation_error_deg": float(error.rotation_deg),
            "objective": float(pose.objective),
            "hessian_condition": float(pose.hessian_condition),
        }
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "anchor_rows.csv",
        [row for row in anchor_rows if row.get("anchor_id") is not None],
        [
            "anchor_id",
            "token_index",
            "subanchor_index",
            "render_x",
            "render_y",
            "X",
            "Y",
            "Z",
            "depth",
            "alpha",
            "depth_gradient",
            "depth_variance",
            "quality",
            "rejection_reason",
        ],
    )
    _write_csv(
        output_dir / "measurement_rows.csv",
        measurement_rows,
        [
            "query_id",
            "candidate_id",
            "anchor_id",
            "fit_or_verify",
            "query_gt_x",
            "query_gt_y",
            "query_pred_x",
            "query_pred_y",
            "residual_before_px",
            "residual_after_px",
            "p_visible",
            "p_assignment",
            "p_valid",
            "cov_xx",
            "cov_xy",
            "cov_yy",
            "mahalanobis2",
            "anchor_quality",
            "depth_gradient",
            "surface_type",
        ],
    )
    _write_csv(
        output_dir / "pose_rows.csv",
        pose_rows,
        [
            "query_id",
            "candidate_id",
            "solver",
            "success",
            "translation_error_m",
            "rotation_error_deg",
            "objective",
            "hessian_condition",
        ],
    )
    summary = {
        "stage": "measurement_v1_synthetic_smoke",
        "anchor_count": int(len(anchors)),
        "measurement_count": int(len(measurements)),
        "pose": pose_rows[0],
        "outputs": {
            "anchor_rows": str(output_dir / "anchor_rows.csv"),
            "measurement_rows": str(output_dir / "measurement_rows.csv"),
            "pose_rows": str(output_dir / "pose_rows.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate measurement_v1 minimal closed-loop diagnostics.")
    parser.add_argument("--synthetic_smoke", action="store_true", help="Run a deterministic synthetic smoke diagnostic.")
    parser.add_argument("--legacy_eval_dir", default="", help="Summarize an existing legacy eval directory as a measurement diagnostic.")
    parser.add_argument("--protocol_lock_eval_dir", default="", help="Build a deterministic protocol lock report from an existing eval directory.")
    parser.add_argument(
        "--gt_measurement_eval_dir",
        default="",
        help="Export GT-render measurement_v1 audit tables from an existing eval directory containing match_table.csv.",
    )
    parser.add_argument("--d0_probe_rows", default="", help="CSV rows for cache-backed D0 token decodability probe.")
    parser.add_argument("--d0_feature_specs", default="radio_dual:radio_dual")
    parser.add_argument("--d0_query_feature_cache_dir", default="")
    parser.add_argument("--d0_render_feature_cache_dir", default="")
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--search_radius_px", type=float, default=8.0)
    parser.add_argument("--step_px", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    mode_count = sum(
        [
            bool(args.synthetic_smoke),
            bool(str(args.legacy_eval_dir)),
            bool(str(args.protocol_lock_eval_dir)),
            bool(str(args.gt_measurement_eval_dir)),
            bool(str(args.d0_probe_rows)),
        ]
    )
    if mode_count != 1:
        raise ValueError(
            "choose exactly one mode: --synthetic_smoke, --legacy_eval_dir, --protocol_lock_eval_dir, --gt_measurement_eval_dir, or --d0_probe_rows"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if bool(args.synthetic_smoke):
        _run_synthetic_smoke(output_dir)
        return
    if bool(str(args.protocol_lock_eval_dir)):
        build_protocol_lock_report(eval_dir=Path(args.protocol_lock_eval_dir), output_dir=output_dir)
        return
    if bool(str(args.gt_measurement_eval_dir)):
        export_gt_measurement_from_legacy_eval(
            eval_dir=Path(args.gt_measurement_eval_dir),
            output_dir=output_dir,
            max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        )
        return
    if bool(str(args.d0_probe_rows)):
        run_d0_probe(
            rows_csv=Path(args.d0_probe_rows),
            output_dir=output_dir,
            feature_specs=parse_feature_specs(str(args.d0_feature_specs)),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            search_radius_px=float(args.search_radius_px),
            step_px=float(args.step_px),
            temperature=float(args.temperature),
            query_feature_cache_dir=Path(args.d0_query_feature_cache_dir) if str(args.d0_query_feature_cache_dir) else None,
            render_feature_cache_dir=Path(args.d0_render_feature_cache_dir) if str(args.d0_render_feature_cache_dir) else None,
            max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        )
        return
    report = summarize_legacy_measurement_eval(Path(args.legacy_eval_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "measurement_v1_legacy_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "measurement_v1_legacy_report.md").write_text(report_to_markdown(report) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
