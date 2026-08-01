"""Measure affine-versus-projective chart-frame error from geometry only.

This diagnostic reads the fixed 2DGS atlas and contributor buffers.  It does
not load mapping or query RGB, descriptors, or pose estimates.  Ground truth
is used only to compare two observation models on the same visible charts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _select_oracle_charts,
    _visible_charts,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _visibility
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    frame_control_error_px,
    ground_truth_chart_frame,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--charts_per_query", type=int, default=4)
    parser.add_argument("--feature_stride", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("output exists; pass --force to replace it")
    atlas_path = Path(args.atlas)
    atlas = MapletFeatureAtlasBank.load_npz(atlas_path)
    rows = []
    query_ids = []
    for path in sorted(Path(args.query_contributor_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            image_id = str(metadata["image_id"])
            pose = np.asarray(data["pose_w2c"], dtype=np.float64)
            camera = ColmapCamera(
                camera_id=0,
                model_id=int(data["camera_model_id"]),
                width=int(data["camera_width"]),
                height=int(data["camera_height"]),
                params=tuple(
                    np.asarray(data["camera_params"], dtype=np.float64)
                ),
            )
            visible_rows, _xy = _visibility(
                atlas,
                pose,
                camera,
                np.asarray(data["topk_ids"], dtype=np.int64),
                np.asarray(data["topk_weights"], dtype=np.float32),
            )
        visible_ids, visible_counts = _visible_charts(
            SimpleNamespace(visible_rows=visible_rows), atlas
        )
        selected = _select_oracle_charts(
            visible_ids,
            visible_counts,
            atlas,
            int(args.charts_per_query),
        )
        query_ids.append(image_id)
        for chart_id in selected.tolist():
            homography = ground_truth_chart_frame(
                atlas,
                int(chart_id),
                pose,
                camera,
                feature_stride=int(args.feature_stride),
                feature_level="diagnostic_gt_homography",
                model="homography",
            )
            affine = ground_truth_chart_frame(
                atlas,
                int(chart_id),
                pose,
                camera,
                feature_stride=int(args.feature_stride),
                feature_level="diagnostic_gt_affine",
                model="affine",
            )
            if homography is None or affine is None:
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "chart_id": int(chart_id),
                    "affine_to_homography_control_error_px": (
                        frame_control_error_px(affine, homography)
                    ),
                }
            )
    values = np.asarray(
        [
            float(row["affine_to_homography_control_error_px"])
            for row in rows
        ],
        dtype=np.float64,
    )
    summary = {
        "chart_count": int(values.size),
        "median_control_error_px": (
            float(np.median(values)) if values.size else None
        ),
        "p90_control_error_px": (
            float(np.quantile(values, 0.90)) if values.size else None
        ),
        "fraction_above_8px": (
            float(np.mean(values > 8.0)) if values.size else None
        ),
    }
    report = {
        "stage": "v6_projection_model_gap_geometry_only",
        "atlas_sha256": _sha256(atlas_path),
        "query_count": len(query_ids),
        "query_ids": query_ids,
        "configuration": {
            "charts_per_query": int(args.charts_per_query),
            "feature_stride": int(args.feature_stride),
        },
        "map_contract": {
            "loads_rgb": False,
            "loads_descriptors": False,
            "uses_point_correspondences": False,
            "uses_pnp": False,
        },
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
