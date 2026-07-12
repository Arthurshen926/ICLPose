"""Audit correct-track coverage and oracle PnP for top-L landmark proposals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.landmark_topl_oracle import (
    TopLOracleObservation,
    summarize_topl_oracle,
)


def _optional_int(value: object) -> int | None:
    text = str(value).strip()
    return None if not text else int(float(text))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall_rows_csv", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--top_ls", default="1,5,10,20")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=4.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    top_ls = tuple(int(value) for value in str(args.top_ls).split(",") if value.strip())
    with Path(args.recall_rows_csv).open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"query_x", "query_y", "landmark_x", "landmark_y", "landmark_z"}
    missing = sorted(required - set(rows[0] if rows else {}))
    if missing:
        raise ValueError(f"recall rows are missing geometry columns: {missing!r}; rerun recall oracle")
    observations = [
        TopLOracleObservation(
            query_id=str(row["query_id"]),
            track_id=int(row["correct_track_id"]),
            correct_rank=_optional_int(row.get("correct_rank")),
            xy=np.asarray([float(row["query_x"]), float(row["query_y"])], dtype=np.float64),
            xyz=np.asarray(
                [float(row["landmark_x"]), float(row["landmark_y"]), float(row["landmark_z"])],
                dtype=np.float64,
            ),
        )
        for row in rows
    ]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_query = {str(image.image_name): image for image in images.values()}
    cameras_by_query = {
        str(image.image_name): cameras[int(image.camera_id)]
        for image in images.values()
        if int(image.camera_id) in cameras
    }
    summary = summarize_topl_oracle(
        observations,
        top_ls=top_ls,
        cameras_by_query=cameras_by_query,
        images_by_query=images_by_query,
        reprojection_error_px=float(args.pnp_reprojection_error_px),
        iterations=int(args.pnp_iterations),
    )
    summary.update(
        {
            "stage": "landmark_topl_correct_track_pose_oracle",
            "recall_rows_csv": str(args.recall_rows_csv),
            "colmap_model_dir": str(model_dir),
            "top_ls": list(top_ls),
            "pnp_reprojection_error_px": float(args.pnp_reprojection_error_px),
            "pnp_iterations": int(args.pnp_iterations),
        }
    )
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
