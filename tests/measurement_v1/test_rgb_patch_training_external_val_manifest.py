from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_training import train_rgb_patch_measurement_branch


def _write_image(path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _write_cache(path: Path) -> None:
    np.savez(path, rgb=np.zeros((16, 16, 3), dtype=np.float32), depth=np.ones((16, 16), dtype=np.float32))


def _write_manifest(path: Path, query_id: str, cache_path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        writer.writerow({"query_id": query_id, "rgb_depth_cache_path": str(cache_path)})


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


def test_external_validation_can_use_separate_render_manifest(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for query_id in ["train.png", "val.png"]:
        _write_image(image_root / query_id)
        _write_cache(tmp_path / f"{query_id}.npz")
    train_manifest = tmp_path / "train_manifest.csv"
    val_manifest = tmp_path / "val_manifest.csv"
    _write_manifest(train_manifest, "train.png", tmp_path / "train.png.npz")
    _write_manifest(val_manifest, "val.png", tmp_path / "val.png.npz")
    train_rows = tmp_path / "train_rows.csv"
    val_rows = tmp_path / "val_rows.csv"
    _write_rows(train_rows, "train.png")
    _write_rows(val_rows, "val.png")

    summary = train_rgb_patch_measurement_branch(
        rows_csv=train_rows,
        val_rows_csv=val_rows,
        render_cache_manifest_csv=train_manifest,
        val_render_cache_manifest_csv=val_manifest,
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
        max_eval_rows=1,
        device="cpu",
    )

    assert summary["render_cache_manifest_csv"] == str(train_manifest)
    assert summary["val_render_cache_manifest_csv"] == str(val_manifest)
    assert summary["val_count"] == 1
