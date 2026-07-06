from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_extract.vfm.measurement_v1.feature_patch_cache import materialize_feature_patch_cache_rows
from feature_extract.vfm.measurement_v1.measurement_training import train_cached_measurement_branch
from feature_extract.vfm.measurement_v1.stride4_fine_feature import crop_feature_window


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_materialize_feature_patch_cache_matches_online_crop(tmp_path: Path) -> None:
    query = np.zeros((3, 8, 8), dtype=np.float32)
    render = np.zeros((3, 8, 8), dtype=np.float32)
    query[0] = np.arange(8, dtype=np.float32).reshape(1, 8)
    query[1] = np.arange(8, dtype=np.float32).reshape(8, 1)
    render[0] = query[0] + 10.0
    render[1] = query[1] + 20.0
    query_cache = tmp_path / "query.npz"
    render_cache = tmp_path / "render.npz"
    np.savez_compressed(query_cache, stride4_rgb=query)
    np.savez_compressed(render_cache, stride4_rgb=render)
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_stride4_rgb_feature_cache_path": str(query_cache),
                "render_stride4_rgb_feature_cache_path": str(render_cache),
                "center_x": 4.0,
                "center_y": 3.0,
                "render_x": 5.0,
                "render_y": 4.0,
                "query_gt_x": 4.5,
                "query_gt_y": 3.0,
            }
        ],
    )

    summary = materialize_feature_patch_cache_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "patch_rows.csv",
        output_patch_dir=tmp_path / "patches",
        feature_name="stride4_rgb",
        feature_key="stride4_rgb",
        image_width=8,
        image_height=8,
        crop_radius_px=1.0,
        step_px=1.0,
    )

    rows = _read_csv(tmp_path / "patch_rows.csv")
    assert rows[0]["patch_cache_source_feature_height"] == "8"
    assert rows[0]["patch_cache_source_feature_width"] == "8"
    assert float(rows[0]["patch_cache_source_feature_pixel_pitch_x"]) == 1.0
    query_patch_path = Path(rows[0]["query_stride4_rgb_patch_cache_path"])
    render_patch_path = Path(rows[0]["render_stride4_rgb_patch_cache_path"])
    with np.load(query_patch_path) as data:
        query_patch = torch.from_numpy(np.asarray(data["stride4_rgb"]))
    with np.load(render_patch_path) as data:
        render_patch = torch.from_numpy(np.asarray(data["stride4_rgb"]))
    expected_query, _ = crop_feature_window(
        torch.from_numpy(query).unsqueeze(0),
        torch.tensor([[4.0, 3.0]], dtype=torch.float32),
        radius_px=1.0,
        step_px=1.0,
        image_width=8,
        image_height=8,
    )
    expected_render, _ = crop_feature_window(
        torch.from_numpy(render).unsqueeze(0),
        torch.tensor([[5.0, 4.0]], dtype=torch.float32),
        radius_px=1.0,
        step_px=1.0,
        image_width=8,
        image_height=8,
    )

    assert summary["patch_count"] == 2
    assert summary["source_feature_spatial_shape"] == [8, 8]
    assert summary["source_feature_pixel_pitch_x"] == 1.0
    assert query_patch.shape == (3, 3, 3)
    assert torch.allclose(query_patch, expected_query[0])
    assert torch.allclose(render_patch, expected_render[0])

    reused_summary = materialize_feature_patch_cache_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "patch_rows_reused.csv",
        output_patch_dir=tmp_path / "patches",
        feature_name="stride4_rgb",
        feature_key="stride4_rgb",
        image_width=8,
        image_height=8,
        crop_radius_px=1.0,
        step_px=1.0,
    )
    assert reused_summary["reused_patch_count"] == 2
    assert reused_summary["source_feature_spatial_shape"] == [8, 8]

    training_summary = train_cached_measurement_branch(
        rows_csv=tmp_path / "patch_rows.csv",
        output_dir=tmp_path / "train_from_materialized_patch",
        feature_name="stride4_rgb",
        feature_key="stride4_rgb",
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
    assert training_summary["feature_geometry"]["source_feature_spatial_shape"] == [8, 8]
    assert training_summary["feature_geometry"]["source_feature_pixel_pitch_x"] == 1.0


def test_materialize_feature_patch_cache_preserves_chw_small_channel_large_spatial_layout(tmp_path: Path) -> None:
    query = np.zeros((3, 20, 30), dtype=np.float32)
    render = np.zeros((3, 20, 30), dtype=np.float32)
    query_cache = tmp_path / "query_chw_large.npz"
    render_cache = tmp_path / "render_chw_large.npz"
    np.savez_compressed(query_cache, stride4_rgb=query)
    np.savez_compressed(render_cache, stride4_rgb=render)
    rows_csv = tmp_path / "rows_chw_large.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_stride4_rgb_feature_cache_path": str(query_cache),
                "render_stride4_rgb_feature_cache_path": str(render_cache),
                "center_x": 15.0,
                "center_y": 10.0,
                "render_x": 15.0,
                "render_y": 10.0,
                "query_gt_x": 15.0,
                "query_gt_y": 10.0,
            }
        ],
    )

    materialize_feature_patch_cache_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "patch_rows_chw_large.csv",
        output_patch_dir=tmp_path / "patches_chw_large",
        feature_name="stride4_rgb",
        feature_key="stride4_rgb",
        image_width=30,
        image_height=20,
        crop_radius_px=1.0,
        step_px=1.0,
    )
    rows = _read_csv(tmp_path / "patch_rows_chw_large.csv")
    with np.load(rows[0]["query_stride4_rgb_patch_cache_path"]) as data:
        patch = np.asarray(data["stride4_rgb"])

    assert patch.shape == (3, 3, 3)


def test_materialize_feature_patch_cache_reuses_source_feature_maps(tmp_path: Path) -> None:
    query = np.zeros((3, 8, 8), dtype=np.float32)
    render = np.zeros((3, 8, 8), dtype=np.float32)
    query_cache = tmp_path / "query.npz"
    render_cache = tmp_path / "render.npz"
    np.savez_compressed(query_cache, stride4_rgb=query)
    np.savez_compressed(render_cache, stride4_rgb=render)
    rows_csv = tmp_path / "rows_reuse.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_stride4_rgb_feature_cache_path": str(query_cache),
                "render_stride4_rgb_feature_cache_path": str(render_cache),
                "center_x": 4.0,
                "center_y": 3.0,
                "render_x": 5.0,
                "render_y": 4.0,
                "query_gt_x": 4.5,
                "query_gt_y": 3.0,
            },
            {
                "query_id": "q0.png",
                "query_stride4_rgb_feature_cache_path": str(query_cache),
                "render_stride4_rgb_feature_cache_path": str(render_cache),
                "center_x": 4.5,
                "center_y": 3.5,
                "render_x": 5.5,
                "render_y": 4.5,
                "query_gt_x": 4.5,
                "query_gt_y": 3.0,
            },
        ],
    )

    summary = materialize_feature_patch_cache_rows(
        rows_csv=rows_csv,
        output_rows_csv=tmp_path / "patch_rows_reuse.csv",
        output_patch_dir=tmp_path / "patches_reuse",
        feature_name="stride4_rgb",
        feature_key="stride4_rgb",
        image_width=8,
        image_height=8,
        crop_radius_px=1.0,
        step_px=1.0,
    )

    assert summary["source_feature_cache_miss_count"] == 2
    assert summary["source_feature_cache_hit_count"] == 2


def test_materialize_feature_patch_cache_rejects_excessive_patch_size(tmp_path: Path) -> None:
    query = np.zeros((8, 8, 8), dtype=np.float32)
    render = np.zeros((8, 8, 8), dtype=np.float32)
    query_cache = tmp_path / "query_big.npz"
    render_cache = tmp_path / "render_big.npz"
    np.savez_compressed(query_cache, feat=query)
    np.savez_compressed(render_cache, feat=render)
    rows_csv = tmp_path / "rows_big.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_feat_feature_cache_path": str(query_cache),
                "render_feat_feature_cache_path": str(render_cache),
                "center_x": 4.0,
                "center_y": 4.0,
                "render_x": 4.0,
                "render_y": 4.0,
                "query_gt_x": 4.0,
                "query_gt_y": 4.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="estimated feature patch cache item is too large"):
        materialize_feature_patch_cache_rows(
            rows_csv=rows_csv,
            output_rows_csv=tmp_path / "patch_rows_big.csv",
            output_patch_dir=tmp_path / "patches_big",
            feature_name="feat",
            feature_key="feat",
            image_width=8,
            image_height=8,
            crop_radius_px=4.0,
            step_px=0.5,
            max_patch_bytes=100,
        )


def test_cached_measurement_training_uses_precomputed_patch_cache(tmp_path: Path) -> None:
    query_patch = np.zeros((3, 3, 3), dtype=np.float32)
    render_patch = np.zeros((3, 3, 3), dtype=np.float32)
    query_patch[2, 1, 2] = 1.0
    render_patch[2, 1, 1] = 1.0
    query_patch_cache = tmp_path / "query_patch.npz"
    render_patch_cache = tmp_path / "render_patch.npz"
    np.savez_compressed(query_patch_cache, stride4_rgb=query_patch)
    np.savez_compressed(render_patch_cache, stride4_rgb=render_patch)
    rows_csv = tmp_path / "patch_rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_stride4_rgb_patch_cache_path": str(query_patch_cache),
                "render_stride4_rgb_patch_cache_path": str(render_patch_cache),
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
        feature_name="stride4_rgb",
        feature_key="stride4_rgb",
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
    assert summary["feature_geometry"]["source"] == "precomputed_patch"
    assert summary["crop_before_projection"] is True
    assert np.isfinite(summary["val_metrics"]["epe_px"])


def test_cached_measurement_training_flags_sub_source_pitch_measurements(tmp_path: Path) -> None:
    query_patch = np.zeros((3, 5, 5), dtype=np.float32)
    render_patch = np.zeros((3, 5, 5), dtype=np.float32)
    query_patch_cache = tmp_path / "query_patch_pitch.npz"
    render_patch_cache = tmp_path / "render_patch_pitch.npz"
    np.savez_compressed(query_patch_cache, stride4_rgb=query_patch)
    np.savez_compressed(render_patch_cache, stride4_rgb=render_patch)
    rows_csv = tmp_path / "patch_rows_pitch.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_stride4_rgb_patch_cache_path": str(query_patch_cache),
                "render_stride4_rgb_patch_cache_path": str(render_patch_cache),
                "patch_cache_source_feature_height": 270,
                "patch_cache_source_feature_width": 480,
                "patch_cache_source_feature_pixel_pitch_x": 4.0,
                "patch_cache_source_feature_pixel_pitch_y": 4.0,
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
        output_dir=tmp_path / "train_pitch",
        feature_name="stride4_rgb",
        feature_key="stride4_rgb",
        image_width=1920,
        image_height=1080,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        hidden_dim=4,
        output_dim=4,
        device="cpu",
    )

    assert summary["feature_geometry"]["step_to_source_pitch_ratio_x"] == 0.25
    assert summary["feature_geometry"]["requested_step_sub_source_pitch"] is True
    assert summary["feature_geometry"]["search_radius_below_one_source_cell"] is True
