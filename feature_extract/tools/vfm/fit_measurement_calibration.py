"""Fit RGB measurement confidence and uncertainty calibration from eval matches."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.landmark_hybrid import cameras_by_image_name
from feature_extract.vfm.localization.measurement_calibration import (
    calibration_rows_from_matches,
    fit_confidence_temperature_bias,
    fit_geometry_probability_model,
    fit_uncertainty_scale_floor,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches_csv", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_samples_csv", default="")
    parser.add_argument("--inlier_threshold_px", type=float, default=2.0)
    parser.add_argument("--uncertainty_max_residual_px", type=float, default=None)
    parser.add_argument("--geometry_inlier_threshold_px", type=float, default=None)
    parser.add_argument("--source_measurement_confidence_temperature", type=float, default=None)
    parser.add_argument("--source_measurement_confidence_bias", type=float, default=None)
    parser.add_argument("--source_measurement_uncertainty_scale", type=float, default=None)
    parser.add_argument("--source_measurement_uncertainty_floor_px", type=float, default=None)
    return parser.parse_args(argv)


def _write_samples_csv(path: Path, rows) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [asdict(row) for row in rows]
    fieldnames = list(values[0]) if values else [
        "query_id",
        "measured_x",
        "measured_y",
        "gt_x",
        "gt_y",
        "confidence",
        "uncertainty_px",
        "residual_px",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    cameras_by_query = cameras_by_image_name(cameras=cameras, colmap_images=images, image_root=Path(args.image_root))
    pose_w2c_by_query = {
        record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.query_pose_file))
    }
    with Path(args.matches_csv).open() as handle:
        match_rows = list(csv.DictReader(handle))
    samples = calibration_rows_from_matches(
        match_rows,
        cameras_by_query=cameras_by_query,
        pose_w2c_by_query=pose_w2c_by_query,
    )
    confidence = fit_confidence_temperature_bias(samples, inlier_threshold_px=float(args.inlier_threshold_px))
    uncertainty = fit_uncertainty_scale_floor(samples, max_residual_px=args.uncertainty_max_residual_px)
    geometry_threshold = (
        float(args.geometry_inlier_threshold_px)
        if args.geometry_inlier_threshold_px is not None
        else float(args.inlier_threshold_px)
    )
    geometry = fit_geometry_probability_model(samples, inlier_threshold_px=geometry_threshold)
    geometry_output = dict(geometry)
    geometry_output["model"] = geometry["model"].to_dict()
    source_measurement_config = {
        "measurement_confidence_temperature": args.source_measurement_confidence_temperature,
        "measurement_confidence_bias": args.source_measurement_confidence_bias,
        "measurement_uncertainty_scale": args.source_measurement_uncertainty_scale,
        "measurement_uncertainty_floor_px": args.source_measurement_uncertainty_floor_px,
    }
    geometry_eval_args = {
        "measurement_geometry_probability_model": str(args.output_json),
        "min_measurement_geometry_probability": geometry["best_f1_threshold"],
    }
    for key, value in source_measurement_config.items():
        if value is not None:
            geometry_eval_args[key] = float(value)
    confidence_uncertainty_eval_args = {
        "measurement_confidence_temperature": confidence["confidence_temperature"],
        "measurement_confidence_bias": confidence["confidence_bias"],
        "measurement_uncertainty_scale": uncertainty["uncertainty_scale"],
        "measurement_uncertainty_floor_px": uncertainty["uncertainty_floor_px"],
    }
    output = {
        "stage": "measurement_confidence_uncertainty_calibration",
        "matches_csv": str(args.matches_csv),
        "query_pose_file": str(args.query_pose_file),
        "colmap_model_dir": str(args.colmap_model_dir),
        "image_root": str(args.image_root),
        "sample_count": int(len(samples)),
        "confidence": confidence,
        "uncertainty": uncertainty,
        "geometry_probability": geometry_output,
        "source_measurement_adapter_config": source_measurement_config,
        "recommended_eval_args": geometry_eval_args,
        "recommended_geometry_probability_eval_args": geometry_eval_args,
        "recommended_confidence_uncertainty_eval_args": confidence_uncertainty_eval_args,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if str(args.output_samples_csv):
        _write_samples_csv(Path(args.output_samples_csv), samples)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
