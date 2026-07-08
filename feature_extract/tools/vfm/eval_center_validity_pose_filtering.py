from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.center_validity_model import evaluate_center_validity_pose_filtering
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table_csv", required=True)
    parser.add_argument("--score_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_key", required=True)
    parser.add_argument("--secondary_score_key", default="")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--camera_width", type=int)
    parser.add_argument("--camera_height", type=int)
    parser.add_argument("--camera_model_id", type=int, default=1)
    parser.add_argument("--camera_params", nargs="+", type=float, default=[])
    parser.add_argument("--query_pose_file", default="")
    parser.add_argument("--budgets", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--score_thresholds", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    parser.add_argument("--solvers", nargs="+", default=["ransac"])
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--geometry_source", default="prefer_world_xyz")
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_center_validity_pose_filtering(
        match_table_csv=Path(args.match_table_csv),
        score_rows_csv=Path(args.score_rows_csv),
        output_dir=Path(args.output_dir),
        camera=_build_camera(args),
        query_pose_w2c_by_id=None if not str(args.query_pose_file).strip() else _query_pose_lookup(Path(args.query_pose_file)),
        score_key=str(args.score_key),
        secondary_score_key=None if not str(args.secondary_score_key).strip() else str(args.secondary_score_key),
        budgets=[int(value) for value in args.budgets],
        score_thresholds=[float(value) for value in args.score_thresholds],
        solvers=[str(value) for value in args.solvers],
        reprojection_error_px=float(args.reprojection_error_px),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        geometry_source=str(args.geometry_source),
        seed=int(args.seed),
    )
    printable = dict(summary)
    printable.pop("pose_rows", None)
    printable.pop("group_coverage_rows", None)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
