from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.measurement_v1.observability_audit import (
    classify_observability,
    export_measurement_observability_audit,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_classify_observability_separates_window_texture_ambiguity_and_mismatch() -> None:
    assert classify_observability(gt_in_window=False, render_texture=1.0, mode_epe_px=0.0, gt_rank=1, entropy_norm=0.1, peak_gap_z=4.0) == "window_out"
    assert classify_observability(gt_in_window=True, render_texture=0.01, mode_epe_px=0.0, gt_rank=1, entropy_norm=0.1, peak_gap_z=4.0) == "textureless"
    assert classify_observability(gt_in_window=True, render_texture=1.0, mode_epe_px=0.25, gt_rank=1, entropy_norm=0.1, peak_gap_z=4.0) == "observable_subpixel"
    assert classify_observability(gt_in_window=True, render_texture=1.0, mode_epe_px=3.0, gt_rank=4, entropy_norm=0.2, peak_gap_z=3.0) == "observable_coarse_only"
    assert classify_observability(gt_in_window=True, render_texture=1.0, mode_epe_px=12.0, gt_rank=30, entropy_norm=0.95, peak_gap_z=0.1) == "ambiguous"
    assert classify_observability(gt_in_window=True, render_texture=1.0, mode_epe_px=12.0, gt_rank=30, entropy_norm=0.2, peak_gap_z=3.0) == "domain_mismatch"


def test_export_measurement_observability_audit_writes_layer_rows_and_summary(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[16, 20] = [255, 255, 255]
    rgb[16, 21] = [128, 128, 128]
    rgb[17, 20] = [64, 64, 64]
    Image.fromarray(rgb).save(image_root / "seq0" / "frame00000.png")
    render_cache = tmp_path / "render.npz"
    np.savez_compressed(render_cache, rgb=rgb.astype(np.float32) / 255.0, depth=np.ones((32, 32), dtype=np.float32), alpha=np.ones((32, 32), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    _write_csv(manifest, [{"query_id": "seq0/frame00000.png", "rgb_depth_cache_path": str(render_cache)}])
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/frame00000.png",
                "render_x": 20.0,
                "render_y": 16.0,
                "center_x": 18.0,
                "center_y": 16.0,
                "query_gt_x": 20.0,
                "query_gt_y": 16.0,
                "target_is_dustbin": "False",
                "requested_residual_px": "2.0",
                "measurement_policy": "match_table_projected_query_gt",
            },
            {
                "query_id": "seq0/frame00000.png",
                "render_x": 4.0,
                "render_y": 4.0,
                "center_x": 4.0,
                "center_y": 4.0,
                "query_gt_x": 20.0,
                "query_gt_y": 16.0,
                "target_is_dustbin": "False",
                "requested_residual_px": "20.0",
                "measurement_policy": "match_table_projected_query_gt",
            },
        ],
    )

    summary = export_measurement_observability_audit(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "audit",
        image_width=32,
        image_height=32,
        search_radius_px=4.0,
        context_radius_px=2.0,
        step_px=1.0,
        max_rows=None,
        batch_size=2,
        device="cpu",
        cache_images_on_device=False,
    )

    assert summary["row_count"] == 2
    assert summary["class_counts"]["observable_subpixel"] >= 1
    assert summary["class_counts"]["window_out"] >= 1
    assert Path(summary["outputs"]["rows_csv"]).exists()
    assert Path(summary["outputs"]["weighted_rows_csv"]).exists()
    weighted_row = next(csv.DictReader(Path(summary["outputs"]["weighted_rows_csv"]).open()))
    assert weighted_row["measurement_policy"] == "match_table_projected_query_gt"
    assert "observability_class" in weighted_row
    assert "train_weight" in weighted_row
    row = next(csv.DictReader(Path(summary["outputs"]["rows_csv"]).open()))
    assert {
        "observability_class",
        "gt_rank",
        "mode_epe_px",
        "entropy_norm",
        "peak_gap_z",
        "render_texture",
        "train_weight",
    }.issubset(row.keys())
