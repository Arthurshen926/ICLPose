from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.measurement_v1.measurement_training import _split_train_val, train_cached_measurement_branch
from feature_extract.vfm.measurement_v1.stride4_fine_feature import crop_feature_window, local_template_correlation_logits


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_split_train_val_can_keep_requested_groups_disjoint() -> None:
    rows = [
        {"query_id": "q0", "center_x": "0"},
        {"query_id": "q0", "center_x": "1"},
        {"query_id": "q1", "center_x": "2"},
        {"query_id": "q1", "center_x": "3"},
        {"query_id": "q2", "center_x": "4"},
        {"query_id": "q2", "center_x": "5"},
    ]

    train_rows, val_rows = _split_train_val(rows, val_fraction=0.34, seed=3, group_key="query_id")

    train_groups = {row["query_id"] for row in train_rows}
    val_groups = {row["query_id"] for row in val_rows}
    assert train_rows
    assert val_rows
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == {"q0", "q1", "q2"}


def test_train_cached_measurement_branch_writes_checkpoint_and_continuous_metrics(tmp_path: Path) -> None:
    query = np.zeros((3, 8, 8), dtype=np.float32)
    render = np.zeros((3, 8, 8), dtype=np.float32)
    query[2, 4, 5] = 1.0
    render[2, 4, 4] = 1.0
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
        context_radius_px=1.0,
        step_px=1.0,
        steps=2,
        batch_size=1,
        hidden_dim=4,
        output_dim=4,
        device="cpu",
        feature_cache_capacity=1,
    )

    assert summary["steps"] == 2
    assert summary["feature_cache_capacity"] == 1
    assert summary["feature_geometry"]["input_spatial_shape"] == [8, 8]
    assert summary["feature_geometry"]["feature_pixel_pitch_x"] == 1.0
    assert summary["feature_geometry"]["step_to_pitch_ratio_x"] == 1.0
    assert np.isfinite(summary["final_metrics"]["loss"])
    assert np.isfinite(summary["final_metrics"]["epe_px"])
    assert np.isfinite(summary["val_metrics"]["epe_px"])
    assert np.isfinite(summary["val_center_baseline"]["epe_px"])
    assert "recall_0p5px" in summary["final_metrics"]
    assert "recall_0p5px" in summary["val_metrics"]
    assert "improve_ratio" in summary["val_metrics"]
    assert "recall_0p5px" in summary["val_center_baseline"]
    assert (tmp_path / "train" / "measurement_branch.pt").exists()
    assert (tmp_path / "train" / "summary.json").exists()


def test_train_cached_measurement_branch_can_use_conv3_projection(tmp_path: Path) -> None:
    query = np.zeros((3, 5, 5), dtype=np.float32)
    render = np.zeros((3, 5, 5), dtype=np.float32)
    query[2, 2, 3] = 1.0
    render[2, 2, 2] = 1.0
    query_cache = tmp_path / "query_patch.npz"
    render_cache = tmp_path / "render_patch.npz"
    np.savez_compressed(query_cache, rgb_native=query)
    np.savez_compressed(render_cache, rgb_native=render)
    rows_csv = tmp_path / "rows_conv3.csv"
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
                "center_x": 2.0,
                "center_y": 2.0,
                "render_x": 2.0,
                "render_y": 2.0,
                "query_gt_x": 3.0,
                "query_gt_y": 2.0,
            }
        ],
    )

    summary = train_cached_measurement_branch(
        rows_csv=rows_csv,
        output_dir=tmp_path / "train_conv3",
        feature_name="rgb_native",
        feature_key="rgb_native",
        image_width=8,
        image_height=8,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        hidden_dim=4,
        output_dim=4,
        projection_type="conv3",
        device="cpu",
    )

    assert summary["projection_type"] == "conv3"
    assert summary["feature_geometry"]["source"] == "precomputed_patch"


def test_train_cached_measurement_branch_can_use_texture_norm_graygrad_projection(tmp_path: Path) -> None:
    query = np.zeros((3, 5, 5), dtype=np.float32)
    render = np.zeros((3, 5, 5), dtype=np.float32)
    query[:, 2, 3] = 0.9
    render[:, 2, 2] = 0.9
    query_cache = tmp_path / "query_texture_patch.npz"
    render_cache = tmp_path / "render_texture_patch.npz"
    np.savez_compressed(query_cache, rgb_native=query)
    np.savez_compressed(render_cache, rgb_native=render)
    rows_csv = tmp_path / "rows_texture.csv"
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
                "center_x": 2.0,
                "center_y": 2.0,
                "render_x": 2.0,
                "render_y": 2.0,
                "query_gt_x": 3.0,
                "query_gt_y": 2.0,
            }
        ],
    )

    summary = train_cached_measurement_branch(
        rows_csv=rows_csv,
        output_dir=tmp_path / "train_texture_norm_graygrad",
        feature_name="rgb_native",
        feature_key="rgb_native",
        image_width=8,
        image_height=8,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        hidden_dim=4,
        output_dim=4,
        projection_type="texture_norm_graygrad",
        device="cpu",
    )

    assert summary["projection_type"] == "texture_norm_graygrad"
    assert summary["feature_geometry"]["input_channel_count"] == 3


def test_train_cached_measurement_branch_can_use_policy_target_and_weight(tmp_path: Path) -> None:
    query = np.zeros((3, 5, 5), dtype=np.float32)
    render = np.zeros((3, 5, 5), dtype=np.float32)
    query[:, 2, 2] = 1.0
    render[:, 2, 2] = 1.0
    query_cache = tmp_path / "query_policy_patch.npz"
    render_cache = tmp_path / "render_policy_patch.npz"
    np.savez_compressed(query_cache, rgb_native=query)
    np.savez_compressed(render_cache, rgb_native=render)
    rows_csv = tmp_path / "rows_policy.csv"
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
                "center_x": 2.0,
                "center_y": 2.0,
                "render_x": 2.0,
                "render_y": 2.0,
                "query_gt_x": 4.0,
                "query_gt_y": 2.0,
                "policy_target_x": 2.0,
                "policy_target_y": 2.0,
                "policy_loss_weight": 0.25,
            }
        ],
    )

    summary = train_cached_measurement_branch(
        rows_csv=rows_csv,
        output_dir=tmp_path / "train_policy_target",
        feature_name="rgb_native",
        feature_key="rgb_native",
        image_width=8,
        image_height=8,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        hidden_dim=4,
        output_dim=4,
        projection_type="texture_rgb_graygrad",
        target_x_key="policy_target_x",
        target_y_key="policy_target_y",
        sample_weight_key="policy_loss_weight",
        device="cpu",
    )

    assert summary["target_x_key"] == "policy_target_x"
    assert summary["target_y_key"] == "policy_target_y"
    assert summary["sample_weight_key"] == "policy_loss_weight"
    assert summary["val_center_baseline"]["epe_median_px"] == 0.0
    saved = torch.load(tmp_path / "train_policy_target" / "measurement_branch.pt", map_location="cpu")
    assert saved["config"]["target_x_key"] == "policy_target_x"
    assert saved["config"]["sample_weight_key"] == "policy_loss_weight"


def test_local_template_correlation_logits_recovers_shifted_feature_template() -> None:
    render = np.zeros((4, 13, 13), dtype=np.float32)
    query = np.zeros((4, 13, 13), dtype=np.float32)
    rng = np.random.default_rng(7)
    template = rng.normal(size=(4, 3, 3)).astype(np.float32)
    render[:, 5:8, 5:8] = template
    dx, dy = 2.0, -1.0
    query[:, 4:7, 7:10] = template

    logits, sample_xy = local_template_correlation_logits(
        torch.from_numpy(query).unsqueeze(0),
        torch.from_numpy(render).unsqueeze(0),
        query_centers_xy=torch.tensor([[6.0, 6.0]], dtype=torch.float32),
        render_anchor_xy=torch.tensor([[6.0, 6.0]], dtype=torch.float32),
        image_width=13,
        image_height=13,
        search_radius_px=3.0,
        context_radius_px=1.0,
        step_px=1.0,
    )

    mode_idx = torch.argmax(logits, dim=1)[0]
    mode = sample_xy[0, mode_idx] if sample_xy.ndim == 3 else sample_xy[mode_idx]
    assert torch.allclose(mode, torch.tensor([6.0 + dx, 6.0 + dy]), atol=1e-5)


def test_crop_feature_window_uses_original_pixel_coordinates_for_arbitrary_channels() -> None:
    feature = torch.zeros((1, 5, 4, 8), dtype=torch.float32)
    feature[:, 0] = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    feature[:, 1] = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    patch, offsets = crop_feature_window(
        feature,
        torch.tensor([[960.0, 540.0]], dtype=torch.float32),
        radius_px=4.0,
        step_px=4.0,
        image_width=1920,
        image_height=1080,
    )

    assert patch.shape == (1, 5, 3, 3)
    assert offsets.shape == (9, 2)
    assert torch.allclose(patch[0, :2, 1, 1], torch.tensor([3.501824, 1.501390]), atol=1e-5)


def test_cached_measurement_training_accepts_hwc_feature_caches(tmp_path: Path) -> None:
    query = np.zeros((8, 8, 3), dtype=np.float32)
    render = np.zeros((8, 8, 3), dtype=np.float32)
    query[4, 5, 2] = 1.0
    render[4, 4, 2] = 1.0
    query_cache = tmp_path / "query_hwc.npz"
    render_cache = tmp_path / "render_hwc.npz"
    np.savez_compressed(query_cache, stride4_rgb=query)
    np.savez_compressed(render_cache, stride4_rgb=render)
    rows_csv = tmp_path / "rows_hwc.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_stride4_rgb_feature_cache_path": str(query_cache),
                "render_stride4_rgb_feature_cache_path": str(render_cache),
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
        output_dir=tmp_path / "train_hwc",
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

    assert summary["input_dim"] == 3
    assert summary["crop_before_projection"] is True
    assert np.isfinite(summary["val_metrics"]["epe_px"])


def test_cached_measurement_training_accepts_hwc_large_channel_feature_caches(tmp_path: Path) -> None:
    query = np.zeros((8, 8, 16), dtype=np.float32)
    render = np.zeros((8, 8, 16), dtype=np.float32)
    query[4, 5, 7] = 1.0
    render[4, 4, 7] = 1.0
    query_cache = tmp_path / "query_hwc_large.npz"
    render_cache = tmp_path / "render_hwc_large.npz"
    np.savez_compressed(query_cache, radio_dual=query)
    np.savez_compressed(render_cache, radio_dual=render)
    rows_csv = tmp_path / "rows_hwc_large.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_radio_dual_feature_cache_path": str(query_cache),
                "render_radio_dual_feature_cache_path": str(render_cache),
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
        output_dir=tmp_path / "train_hwc_large",
        feature_name="radio_dual",
        feature_key="radio_dual",
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

    assert summary["input_dim"] == 16
    assert np.isfinite(summary["val_metrics"]["epe_px"])
