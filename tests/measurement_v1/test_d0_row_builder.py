from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.d0_row_builder import (
    build_d0_rows_from_match_table,
    project_world_points_to_image,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_project_world_points_to_image_uses_camera_pose_and_distortion() -> None:
    camera = ColmapCamera(camera_id=1, model_id=2, width=16, height=16, params=(8.0, 8.0, 8.0, 0.0))
    projected, valid = project_world_points_to_image(
        np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        np.eye(4, dtype=np.float64),
        camera,
    )

    assert bool(valid[0]) is True
    assert np.allclose(projected[0], [8.0, 8.0], atol=1e-6)
    assert projected[1, 0] > projected[0, 0]


def test_build_d0_rows_from_match_table_writes_gt_targets_and_stride4_rgb_cache(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    query_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    query_rgb[8, 8] = [255, 0, 0]
    Image.fromarray(query_rgb).save(image_root / "seq0" / "frame00000.png")
    render_cache = tmp_path / "render_rgb_depth_cache" / "render.npz"
    render_cache.parent.mkdir(parents=True)
    np.savez_compressed(render_cache, rgb=query_rgb, depth=np.ones((16, 16), dtype=np.float32), alpha=np.ones((16, 16), dtype=np.float32))
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(
        manifest,
        [{"query_id": "seq0/frame00000.png", "rgb_depth_cache_path": str(render_cache)}],
    )
    match_table = tmp_path / "match_table.csv"
    _write_csv(
        match_table,
        [
            {
                "query_id": "seq0/frame00000.png",
                "render_x": 8.0,
                "render_y": 8.0,
                "world_x": 0.0,
                "world_y": 0.0,
                "world_z": 4.0,
            }
        ],
    )
    output_rows = tmp_path / "d0_rows.csv"
    stride4_cache_dir = tmp_path / "stride4_cache"

    build_d0_rows_from_match_table(
        match_table_csv=match_table,
        output_rows_csv=output_rows,
        image_root=image_root,
        render_cache_manifest_csv=manifest,
        pose_by_query={"seq0/frame00000.png": np.eye(4, dtype=np.float64)},
        camera_by_query={"seq0/frame00000.png": ColmapCamera(camera_id=1, model_id=2, width=16, height=16, params=(8.0, 8.0, 8.0, 0.0))},
        stride4_rgb_cache_dir=stride4_cache_dir,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert len(rows) == 1
    assert rows[0]["query_gt_x"] == "8.0"
    assert rows[0]["query_gt_y"] == "8.0"
    assert Path(rows[0]["query_stride4_rgb_feature_cache_path"]).exists()
    assert Path(rows[0]["render_stride4_rgb_feature_cache_path"]).exists()
    with np.load(rows[0]["query_stride4_rgb_feature_cache_path"]) as data:
        assert data["stride4_rgb"].shape == (3, 4, 4)
