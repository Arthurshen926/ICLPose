from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from feature_extract.vfm.measurement_v1 import rgb_patch_training
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    _configure_head_only_training,
    _residual_bin_metrics,
    _ResidualBalancedBatchSampler,
    train_rgb_patch_measurement_branch,
)


def test_measurement_gate_head_only_training_freezes_every_other_parameter() -> None:
    model = rgb_patch_training.RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
    )

    scope = _configure_head_only_training(
        model,
        train_dustbin_head_only=False,
        train_measurement_gate_head_only=True,
    )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert scope == "measurement_gate_head"
    assert trainable
    assert all(name.startswith("measurement_gate_head.") for name in trainable)


def test_append_roll_hard_negatives_marks_rolled_support_patches_as_dustbin() -> None:
    query_patch = torch.arange(2 * 3 * 5 * 5, dtype=torch.float32).reshape(2, 3, 5, 5)
    render_patch = query_patch + 1000.0
    target = torch.tensor([[0.25, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target_is_dustbin = torch.tensor([False, False])

    out_query, out_render, out_target, out_dustbin = rgb_patch_training._append_roll_hard_negatives(
        query_patch,
        render_patch,
        target,
        target_is_dustbin,
        fraction=1.0,
    )

    assert out_query.shape[0] == 4
    assert out_render.shape[0] == 4
    assert out_target.shape == (4, 2)
    assert out_dustbin.tolist() == [False, False, True, True]
    assert torch.equal(out_query[2:], query_patch)
    assert torch.equal(out_render[2], render_patch[1])
    assert torch.equal(out_render[3], render_patch[0])


def _write_query_image(path: Path, *, size: tuple[int, int]) -> None:
    width, height = size
    xx = np.linspace(0, 255, width, dtype=np.uint8).reshape(1, width)
    yy = np.linspace(0, 255, height, dtype=np.uint8).reshape(height, 1)
    image = np.stack(
        [
            np.broadcast_to(xx, (height, width)),
            np.broadcast_to(yy, (height, width)),
            np.full((height, width), 128, dtype=np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(image, mode="RGB").save(path)


def _write_render_cache(path: Path, *, size: tuple[int, int]) -> None:
    width, height = size
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, width)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    rgb = np.stack(
        [
            np.broadcast_to(xx, (height, width)),
            np.broadcast_to(yy, (height, width)),
            0.5 * np.ones((height, width), dtype=np.float32),
        ],
        axis=-1,
    )
    np.savez(path, rgb=rgb.astype(np.float32), depth=np.ones((height, width), dtype=np.float32))


def test_rgb_patch_training_supports_different_query_and_render_sizes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png", size=(16, 16))
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache, size=(8, 8))
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": "q0.png", "rgb_depth_cache_path": str(render_cache)})
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "center_x",
                "center_y",
                "query_gt_x",
                "query_gt_y",
                "render_x",
                "render_y",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "center_x": "8.0",
                "center_y": "8.0",
                "query_gt_x": "8.5",
                "query_gt_y": "8.0",
                "render_x": "4.0",
                "render_y": "4.0",
            }
        )

    summary = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out",
        image_width=16,
        image_height=16,
        query_image_width=16,
        query_image_height=16,
        render_image_width=8,
        render_image_height=8,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
        gate_target_mode="utility",
        gate_utility_temperature_px=0.5,
        val_fraction=0.0,
        max_eval_rows=1,
        device="cpu",
    )

    assert summary["query_image_width"] == 16
    assert summary["query_image_height"] == 16
    assert summary["render_image_width"] == 8
    assert summary["gate_target_mode"] == "utility"
    assert summary["gate_utility_temperature_px"] == 0.5
    assert summary["render_image_height"] == 8
    assert Path(summary["outputs"]["checkpoint"]).exists()
    assert json.loads(Path(summary["outputs"]["summary"]).read_text())["render_image_width"] == 8


def test_rgb_patch_training_accepts_match_table_query_center_columns(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png", size=(16, 16))
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache, size=(8, 8))
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": "q0.png", "rgb_depth_cache_path": str(render_cache)})
    rows_csv = tmp_path / "match_table_rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query_center_x",
                "query_center_y",
                "query_gt_x",
                "query_gt_y",
                "render_x",
                "render_y",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "query_gt_x": "8.5",
                "query_gt_y": "8.0",
                "render_x": "4.0",
                "render_y": "4.0",
            }
        )

    summary = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out_match_table",
        image_width=16,
        image_height=16,
        query_image_width=16,
        query_image_height=16,
        render_image_width=8,
        render_image_height=8,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
        val_fraction=0.0,
        max_eval_rows=1,
        device="cpu",
    )

    assert summary["val_center_baseline"]["epe_median_px"] == 0.5


def test_rgb_patch_training_can_filter_valid_rows_by_dustbin_and_residual(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png", size=(16, 16))
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache, size=(8, 8))
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": "q0.png", "rgb_depth_cache_path": str(render_cache)})
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "center_x",
                "center_y",
                "query_gt_x",
                "query_gt_y",
                "render_x",
                "render_y",
                "target_is_dustbin",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "center_x": "8.0",
                "center_y": "8.0",
                "query_gt_x": "8.4",
                "query_gt_y": "8.0",
                "render_x": "4.0",
                "render_y": "4.0",
                "target_is_dustbin": "false",
            }
        )
        writer.writerow(
            {
                "query_id": "q0.png",
                "center_x": "8.0",
                "center_y": "8.0",
                "query_gt_x": "9.2",
                "query_gt_y": "8.0",
                "render_x": "4.0",
                "render_y": "4.0",
                "target_is_dustbin": "false",
            }
        )
        writer.writerow(
            {
                "query_id": "q0.png",
                "center_x": "8.0",
                "center_y": "8.0",
                "query_gt_x": "12.0",
                "query_gt_y": "8.0",
                "render_x": "4.0",
                "render_y": "4.0",
                "target_is_dustbin": "true",
            }
        )

    summary = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out_filter",
        image_width=16,
        image_height=16,
        search_radius_px=2.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
        val_fraction=0.0,
        max_eval_rows=2,
        device="cpu",
        target_dustbin_filter="valid",
        baseline_epe_min_px=0.5,
        baseline_epe_max_px=2.0,
    )

    assert summary["raw_row_count"] == 3
    assert summary["row_count"] == 1
    assert summary["row_filter"]["kept_count"] == 1
    assert summary["row_filter"]["dropped_count"] == 2
    assert summary["target_dustbin_filter"] == "valid"


def test_two_stage_rgb_patch_training_uses_full_coarse_fine_radius_for_valid_filter(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png", size=(16, 16))
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache, size=(16, 16))
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": "q0.png", "rgb_depth_cache_path": str(render_cache)})
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "center_x", "center_y", "query_gt_x", "query_gt_y", "render_x", "render_y"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "center_x": "8.0",
                "center_y": "8.0",
                "query_gt_x": "9.5",
                "query_gt_y": "8.0",
                "render_x": "8.0",
                "render_y": "8.0",
            }
        )

    summary = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out_two_stage_filter",
        image_width=16,
        image_height=16,
        coarse_search_radius_px=2.0,
        coarse_step_px=1.0,
        search_radius_px=0.5,
        context_radius_px=1.0,
        step_px=0.5,
        steps=1,
        batch_size=1,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
        val_fraction=0.0,
        max_eval_rows=1,
        device="cpu",
        target_dustbin_filter="valid",
    )

    assert summary["coarse_search_radius_px"] == 2.0
    assert summary["coarse_step_px"] == 1.0
    assert summary["measurement_search_radius_px"] == 2.5
    assert summary["row_count"] == 1
    assert summary["row_filter"]["kept_count"] == 1


def test_rgb_patch_training_filters_zero_weight_rows_before_sampling(tmp_path: Path) -> None:
    rows = [
        {
            "center_x": "8.0",
            "center_y": "8.0",
            "query_gt_x": "8.0",
            "query_gt_y": "8.0",
            "dustbin_supervision_weight": "0.0",
        },
        {
            "center_x": "8.0",
            "center_y": "8.0",
            "query_gt_x": "8.0",
            "query_gt_y": "8.0",
            "dustbin_supervision_weight": "1.0",
        },
    ]

    kept, summary = rgb_patch_training._filter_rows_for_training(
        rows,
        search_radius_px=2.0,
        target_x_key="query_gt_x",
        target_y_key="query_gt_y",
        loss_weight_key="dustbin_supervision_weight",
        min_loss_weight=0.5,
    )

    assert len(kept) == 1
    assert kept[0]["dustbin_supervision_weight"] == "1.0"
    assert summary["dropped_by_loss_weight_count"] == 1
    assert summary["min_loss_weight"] == 0.5


def test_rgb_patch_training_records_data_parallel_device_ids_on_cpu(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png", size=(16, 16))
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache, size=(8, 8))
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": "q0.png", "rgb_depth_cache_path": str(render_cache)})
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "center_x", "center_y", "query_gt_x", "query_gt_y", "render_x", "render_y"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "center_x": "8.0",
                "center_y": "8.0",
                "query_gt_x": "8.5",
                "query_gt_y": "8.0",
                "render_x": "4.0",
                "render_y": "4.0",
            }
        )

    summary = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out_parallel",
        image_width=16,
        image_height=16,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
        val_fraction=0.0,
        max_eval_rows=1,
        device="cpu",
        data_parallel_device_ids=[0, 1],
    )

    assert summary["requested_data_parallel_device_ids"] == [0, 1]
    assert summary["active_data_parallel_device_ids"] == []


def test_residual_bin_metrics_quantizes_continuous_residual_values() -> None:
    rows = [{"requested_residual_px": str(index * 0.013)} for index in range(100)]
    epe = torch.ones((100,), dtype=torch.float32)
    baseline = torch.ones((100,), dtype=torch.float32) * 2.0

    metrics = _residual_bin_metrics(rows, epe, baseline, prefix="likelihood")
    count_keys = [key for key in metrics if key.endswith("_count")]

    assert len(count_keys) < 100
    assert "likelihood_bin_overflow_grouped_count" in metrics


def test_residual_balanced_sampler_draws_from_each_non_empty_residual_bin() -> None:
    rows = []
    for residual in [0.25, 0.5, 2.5, 3.0, 9.0, 12.0, 24.0, 30.0]:
        rows.append(
            {
                "query_id": f"q_{residual}",
                "center_x": "10.0",
                "center_y": "10.0",
                "query_gt_x": str(10.0 + residual),
                "query_gt_y": "10.0",
            }
        )
    sampler = _ResidualBalancedBatchSampler(
        rows,
        residual_bin_edges_px=(0.0, 2.0, 5.0, 20.0, 28.0),
        target_x_key="query_gt_x",
        target_y_key="query_gt_y",
        search_radius_px=28.0,
    )

    batch = sampler.sample_batch(batch_size=10, rng=rgb_patch_training.random.Random(7))
    residuals = [round(float(row["query_gt_x"]) - float(row["center_x"]), 2) for row in batch]

    assert sampler.summary["enabled"] is True
    assert sampler.summary["non_empty_bin_count"] == 5
    assert any(value < 2.0 for value in residuals)
    assert any(2.0 <= value < 5.0 for value in residuals)
    assert any(5.0 <= value < 20.0 for value in residuals)
    assert any(20.0 <= value < 28.0 for value in residuals)
    assert any(value >= 28.0 for value in residuals)


def test_tensor_image_lru_cache_enforces_shared_byte_budget() -> None:
    cache = rgb_patch_training.TensorImageLRUCache(max_bytes=32)
    cache.record_miss()
    cache["first"] = torch.zeros((4,), dtype=torch.float32)
    cache.record_miss()
    cache["second"] = torch.ones((4,), dtype=torch.float32)
    assert list(cache) == ["first", "second"]

    assert torch.equal(cache["first"], torch.zeros((4,)))
    cache.record_miss()
    cache["third"] = torch.full((4,), 2.0)

    assert list(cache) == ["first", "third"]
    assert cache.summary() == {
        "entry_count": 2,
        "max_bytes": 32,
        "current_bytes": 32,
        "peak_bytes": 32,
        "hits": 1,
        "misses": 3,
        "evictions": 1,
    }
