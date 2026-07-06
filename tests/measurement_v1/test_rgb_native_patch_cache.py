from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_native_patch_cache import materialize_rgb_native_patch_cache_rows
from feature_extract.vfm.measurement_v1.measurement_training import train_cached_measurement_branch
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import crop_rgb_window


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def test_materialize_rgb_native_patch_cache_matches_online_real_render_crop(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    query_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    query_rgb[..., 0] = np.arange(8, dtype=np.uint8).reshape(1, 8) * 10
    query_rgb[..., 1] = np.arange(8, dtype=np.uint8).reshape(8, 1) * 10
    _save_rgb(image_root / "seq0" / "frame000.png", query_rgb)
    render_rgb = np.zeros((8, 8, 3), dtype=np.float32)
    render_rgb[..., 0] = np.arange(8, dtype=np.float32).reshape(1, 8) / 8.0
    render_rgb[..., 1] = np.arange(8, dtype=np.float32).reshape(8, 1) / 8.0
    render_cache = tmp_path / "render_q0.npz"
    np.savez_compressed(render_cache, rgb=render_rgb, depth=np.ones((8, 8), dtype=np.float32))
    manifest = tmp_path / "render_manifest.csv"
    _write_csv(manifest, [{"query_id": "seq0/frame000.png", "rgb_depth_cache_path": str(render_cache)}])
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/frame000.png",
                "center_x": 4.0,
                "center_y": 3.0,
                "render_x": 5.0,
                "render_y": 4.0,
                "query_gt_x": 4.5,
                "query_gt_y": 3.0,
            }
        ],
    )

    summary = materialize_rgb_native_patch_cache_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "rgb_patch_rows.csv",
        output_patch_dir=tmp_path / "patches",
        image_root=image_root,
        render_cache_manifest_csv=manifest,
        image_width=8,
        image_height=8,
        crop_radius_px=1.0,
        step_px=1.0,
        query_source="real",
    )

    rows = _read_csv(tmp_path / "rgb_patch_rows.csv")
    with np.load(rows[0]["query_rgb_native_patch_cache_path"]) as data:
        query_patch = torch.from_numpy(np.asarray(data["rgb_native"]))
    with np.load(rows[0]["render_rgb_native_patch_cache_path"]) as data:
        render_patch = torch.from_numpy(np.asarray(data["rgb_native"]))
    query_tensor = torch.from_numpy(np.moveaxis(query_rgb.astype(np.float32) / 255.0, -1, 0)).unsqueeze(0)
    render_tensor = torch.from_numpy(np.moveaxis(render_rgb, -1, 0)).unsqueeze(0)
    expected_query, _ = crop_rgb_window(
        query_tensor,
        torch.tensor([[4.0, 3.0]], dtype=torch.float32),
        radius_px=1.0,
        step_px=1.0,
        image_width=8,
        image_height=8,
    )
    expected_render, _ = crop_rgb_window(
        render_tensor,
        torch.tensor([[5.0, 4.0]], dtype=torch.float32),
        radius_px=1.0,
        step_px=1.0,
        image_width=8,
        image_height=8,
    )

    assert summary["patch_count"] == 2
    assert rows[0]["patch_cache_source_feature_height"] == "8"
    assert rows[0]["patch_cache_source_feature_width"] == "8"
    assert float(rows[0]["patch_cache_source_feature_pixel_pitch_x"]) == 1.0
    assert torch.allclose(query_patch, expected_query[0])
    assert torch.allclose(render_patch, expected_render[0])


def test_rgb_native_patch_cache_can_feed_cached_measurement_training(tmp_path: Path) -> None:
    query_patch = np.zeros((3, 3, 3), dtype=np.float32)
    render_patch = np.zeros((3, 3, 3), dtype=np.float32)
    query_patch[2, 1, 2] = 1.0
    render_patch[2, 1, 1] = 1.0
    query_cache = tmp_path / "query_rgb_patch.npz"
    render_cache = tmp_path / "render_rgb_patch.npz"
    np.savez_compressed(query_cache, rgb_native=query_patch)
    np.savez_compressed(render_cache, rgb_native=render_patch)
    rows_csv = tmp_path / "rows_train.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_rgb_native_patch_cache_path": str(query_cache),
                "render_rgb_native_patch_cache_path": str(render_cache),
                "patch_cache_source_feature_height": 8,
                "patch_cache_source_feature_width": 8,
                "patch_cache_source_feature_pixel_pitch_x": 1.0,
                "patch_cache_source_feature_pixel_pitch_y": 1.0,
                "center_x": 4.0,
                "center_y": 4.0,
                "render_x": 4.0,
                "render_y": 4.0,
                "query_gt_x": 5.0,
                "query_gt_y": 4.0,
            }
        ],
    )

    summary = train_cached_measurement_branch(
        rows_csv=rows_csv,
        output_dir=tmp_path / "train",
        feature_name="rgb_native",
        feature_key="rgb_native",
        image_width=8,
        image_height=8,
        search_radius_px=1.0,
        context_radius_px=0.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        hidden_dim=4,
        output_dim=4,
        device="cpu",
    )

    assert summary["patch_cache_mode"] == "precomputed"
    assert summary["feature_geometry"]["source_feature_pixel_pitch_x"] == 1.0
    assert summary["feature_geometry"]["requested_step_sub_source_pitch"] is False
