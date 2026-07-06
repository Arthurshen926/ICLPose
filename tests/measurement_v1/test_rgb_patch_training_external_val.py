from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_training import train_rgb_patch_measurement_branch


def _write_query_image(path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = np.arange(16, dtype=np.uint8).reshape(1, 16) * 8
    image[..., 1] = np.arange(16, dtype=np.uint8).reshape(16, 1) * 8
    Image.fromarray(image, mode="RGB").save(path)


def _write_render_cache(path: Path) -> None:
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[..., 0] = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16)
    rgb[..., 1] = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(16, 1)
    np.savez(path, rgb=rgb, depth=np.ones((16, 16), dtype=np.float32))


def _write_rows(path: Path, query_id: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "center_x", "center_y", "query_gt_x", "query_gt_y", "render_x", "render_y"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": query_id,
                "center_x": "8.0",
                "center_y": "8.0",
                "query_gt_x": "8.5",
                "query_gt_y": "8.0",
                "render_x": "8.0",
                "render_y": "8.0",
            }
        )


def _write_two_rows(path: Path, query_id: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "center_x", "center_y", "query_gt_x", "query_gt_y", "render_x", "render_y"],
        )
        writer.writeheader()
        for offset in [0.5, 1.0]:
            writer.writerow(
                {
                    "query_id": query_id,
                    "center_x": "8.0",
                    "center_y": "8.0",
                    "query_gt_x": str(8.0 + offset),
                    "query_gt_y": "8.0",
                    "render_x": "8.0",
                    "render_y": "8.0",
                }
            )


def test_rgb_patch_training_uses_explicit_external_validation_rows(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for query_id in ["train.png", "val.png"]:
        _write_query_image(image_root / query_id)
        _write_render_cache(tmp_path / f"{query_id}.npz")
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        for query_id in ["train.png", "val.png"]:
            writer.writerow({"query_id": query_id, "rgb_depth_cache_path": str(tmp_path / f"{query_id}.npz")})
    train_rows = tmp_path / "train_rows.csv"
    val_rows = tmp_path / "val_rows.csv"
    _write_rows(train_rows, "train.png")
    _write_rows(val_rows, "val.png")

    summary = train_rgb_patch_measurement_branch(
        rows_csv=train_rows,
        val_rows_csv=val_rows,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out",
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
        val_fraction=0.5,
        max_eval_rows=1,
        device="cpu",
    )

    assert summary["row_count"] == 1
    assert summary["train_count"] == 1
    assert summary["val_count"] == 1
    assert summary["val_rows_csv"] == str(val_rows)
    assert summary["val_group_key"] == ""
    assert "mode_valid_epe_median_px" in summary["val_metrics"]
    assert "mode_valid_improve_ratio" in summary["val_metrics"]


def test_rgb_patch_training_marks_acceptance_gate_as_sampled_when_val_eval_is_truncated(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for query_id in ["train.png", "val.png"]:
        _write_query_image(image_root / query_id)
        _write_render_cache(tmp_path / f"{query_id}.npz")
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        for query_id in ["train.png", "val.png"]:
            writer.writerow({"query_id": query_id, "rgb_depth_cache_path": str(tmp_path / f"{query_id}.npz")})
    train_rows = tmp_path / "train_rows.csv"
    val_rows = tmp_path / "val_rows.csv"
    _write_rows(train_rows, "train.png")
    _write_two_rows(val_rows, "val.png")

    summary = train_rgb_patch_measurement_branch(
        rows_csv=train_rows,
        val_rows_csv=val_rows,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out",
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
        val_fraction=0.5,
        max_eval_rows=1,
        device="cpu",
    )

    assert summary["val_count"] == 2
    assert summary["val_eval_count"] == 1
    assert summary["acceptance_gate"]["is_sampled"] is True
    assert summary["acceptance_gate"]["evaluated_row_count"] == 1
    assert summary["acceptance_gate"]["total_row_count"] == 2


def test_rgb_patch_training_allows_eval_batch_larger_than_legacy_cap(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for query_id in ["train.png", "val.png"]:
        _write_query_image(image_root / query_id)
        _write_render_cache(tmp_path / f"{query_id}.npz")
    manifest = tmp_path / "render_cache_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        for query_id in ["train.png", "val.png"]:
            writer.writerow({"query_id": query_id, "rgb_depth_cache_path": str(tmp_path / f"{query_id}.npz")})
    train_rows = tmp_path / "train_rows.csv"
    val_rows = tmp_path / "val_rows.csv"
    _write_rows(train_rows, "train.png")
    _write_two_rows(val_rows, "val.png")

    summary = train_rgb_patch_measurement_branch(
        rows_csv=train_rows,
        val_rows_csv=val_rows,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "out",
        image_width=16,
        image_height=16,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        steps=1,
        batch_size=1,
        eval_batch_size=32,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
        val_fraction=0.5,
        max_eval_rows=0,
        device="cpu",
    )

    assert summary["eval_batch_size"] == 32
    assert summary["val_eval_count"] == 2
