"""Build measurement-v1 real-real SfM-track local alignment rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.real_real_tracks import (
    build_real_real_measurement_rows_from_colmap_model,
    build_real_real_measurement_rows_from_jsonl,
    build_real_real_query_measurement_rows_from_colmap_model,
    build_real_real_query_measurement_rows_from_jsonl,
    build_real_same_image_measurement_rows_from_jsonl,
)


def _load_image_id_allowlist(path: Path) -> set[str]:
    values: set[str] = set()
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        first = text.split()[0].rstrip(",")
        if first in {"Visual", "ImageFile"}:
            continue
        values.add(first)
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--track_observations_jsonl", default="")
    parser.add_argument("--image_id_file", default="")
    parser.add_argument("--same_image", action="store_true")
    parser.add_argument("--query_measurement_rows", action="store_true")
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--image_width", type=int, default=0)
    parser.add_argument("--image_height", type=int, default=0)
    parser.add_argument("--search_radius_px", type=float, default=4.0)
    parser.add_argument("--context_radius_px", type=float, default=8.0)
    parser.add_argument("--residual_bins_px", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--dustbin_residual_bins_px", nargs="*", type=float, default=[])
    parser.add_argument("--wrong_support_rows_per_positive", type=int, default=0)
    parser.add_argument("--min_track_length", type=int, default=2)
    parser.add_argument("--max_reprojection_error", type=float, default=0.0)
    parser.add_argument("--max_view_angle_deg", type=float, default=0.0)
    parser.add_argument("--same_sequence_only", action="store_true")
    parser.add_argument("--max_frame_gap", type=int, default=0)
    parser.add_argument("--max_tracks_per_query", type=int, default=256)
    parser.add_argument("--emit_all_residual_bins", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    image_id_allowlist = _load_image_id_allowlist(Path(args.image_id_file)) if str(args.image_id_file) else None
    common = {
        "output_rows_csv": Path(args.output_rows_csv),
        "image_width": int(args.image_width) if int(args.image_width) > 0 else None,
        "image_height": int(args.image_height) if int(args.image_height) > 0 else None,
        "search_radius_px": float(args.search_radius_px),
        "context_radius_px": float(args.context_radius_px),
        "residual_bins_px": [float(item) for item in args.residual_bins_px],
        "dustbin_residual_bins_px": [float(item) for item in args.dustbin_residual_bins_px],
        "min_track_length": int(args.min_track_length),
        "max_reprojection_error": float(args.max_reprojection_error) if float(args.max_reprojection_error) > 0.0 else None,
        "max_view_angle_deg": float(args.max_view_angle_deg) if float(args.max_view_angle_deg) > 0.0 else None,
        "max_rows": int(args.max_rows) if int(args.max_rows) > 0 else None,
        "seed": int(args.seed),
        "image_id_allowlist": image_id_allowlist,
    }
    if bool(args.same_image) and bool(args.query_measurement_rows):
        raise SystemExit("--query_measurement_rows is only valid for real-real rows")
    if bool(args.same_image):
        if not str(args.track_observations_jsonl):
            raise SystemExit("--same_image requires --track_observations_jsonl")
        summary = build_real_same_image_measurement_rows_from_jsonl(
            track_observations_jsonl=Path(args.track_observations_jsonl),
            wrong_support_rows_per_positive=max(0, int(args.wrong_support_rows_per_positive)),
            **common,
        )
    elif bool(args.query_measurement_rows) and str(args.track_observations_jsonl):
        summary = build_real_real_query_measurement_rows_from_jsonl(
            track_observations_jsonl=Path(args.track_observations_jsonl),
            output_rows_csv=Path(args.output_rows_csv),
            image_width=int(args.image_width) if int(args.image_width) > 0 else None,
            image_height=int(args.image_height) if int(args.image_height) > 0 else None,
            search_radius_px=float(args.search_radius_px),
            context_radius_px=float(args.context_radius_px),
            residual_bins_px=[float(item) for item in args.residual_bins_px],
            min_track_length=int(args.min_track_length),
            max_reprojection_error=float(args.max_reprojection_error) if float(args.max_reprojection_error) > 0.0 else None,
            max_view_angle_deg=float(args.max_view_angle_deg) if float(args.max_view_angle_deg) > 0.0 else None,
            same_sequence_only=bool(args.same_sequence_only),
            max_frame_gap=int(args.max_frame_gap) if int(args.max_frame_gap) > 0 else None,
            max_tracks_per_query=int(args.max_tracks_per_query),
            emit_all_residual_bins=bool(args.emit_all_residual_bins),
            seed=int(args.seed),
            image_id_allowlist=image_id_allowlist,
        )
    elif str(args.track_observations_jsonl):
        summary = build_real_real_measurement_rows_from_jsonl(
            track_observations_jsonl=Path(args.track_observations_jsonl),
            same_sequence_only=bool(args.same_sequence_only),
            max_frame_gap=int(args.max_frame_gap) if int(args.max_frame_gap) > 0 else None,
            wrong_support_rows_per_positive=max(0, int(args.wrong_support_rows_per_positive)),
            **common,
        )
    elif bool(args.query_measurement_rows) and str(args.model_dir):
        summary = build_real_real_query_measurement_rows_from_colmap_model(
            model_dir=Path(args.model_dir),
            output_rows_csv=Path(args.output_rows_csv),
            image_width=int(args.image_width) if int(args.image_width) > 0 else None,
            image_height=int(args.image_height) if int(args.image_height) > 0 else None,
            search_radius_px=float(args.search_radius_px),
            context_radius_px=float(args.context_radius_px),
            residual_bins_px=[float(item) for item in args.residual_bins_px],
            min_track_length=int(args.min_track_length),
            max_reprojection_error=float(args.max_reprojection_error) if float(args.max_reprojection_error) > 0.0 else None,
            max_view_angle_deg=float(args.max_view_angle_deg) if float(args.max_view_angle_deg) > 0.0 else None,
            same_sequence_only=bool(args.same_sequence_only),
            max_frame_gap=int(args.max_frame_gap) if int(args.max_frame_gap) > 0 else None,
            max_tracks_per_query=int(args.max_tracks_per_query),
            emit_all_residual_bins=bool(args.emit_all_residual_bins),
            seed=int(args.seed),
            image_id_allowlist=image_id_allowlist,
        )
    elif str(args.model_dir):
        summary = build_real_real_measurement_rows_from_colmap_model(
            model_dir=Path(args.model_dir),
            same_sequence_only=bool(args.same_sequence_only),
            max_frame_gap=int(args.max_frame_gap) if int(args.max_frame_gap) > 0 else None,
            wrong_support_rows_per_positive=max(0, int(args.wrong_support_rows_per_positive)),
            **common,
        )
    else:
        raise SystemExit("one of --track_observations_jsonl or --model_dir is required")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
