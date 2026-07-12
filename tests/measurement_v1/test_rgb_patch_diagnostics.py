from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_diagnostics import (
    _group_measurement_metrics,
    export_rgb_patch_diagnostics,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import RGBPatchMeasurementBranch


def test_group_metrics_skip_rows_with_missing_target_baseline() -> None:
    metrics = _group_measurement_metrics(
        [
            {
                "row_index": 0,
                "target_gt_projected_x": "12.0",
                "target_gt_projected_y": "8.0",
                "target_gt_projected_residual_px": "",
            }
        ],
        target_x_key="target_gt_projected_x",
        target_y_key="target_gt_projected_y",
        baseline_key="target_gt_projected_residual_px",
        require_non_dustbin=False,
    )

    assert metrics["policy_group_count"] == 0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_export_rgb_patch_diagnostics_writes_rows_and_visualizations(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, 8] = [255, 255, 255]
    Image.fromarray(rgb).save(image_root / "seq0" / "frame00000.png")
    render_cache = tmp_path / "render_rgb_depth" / "r0.npz"
    render_cache.parent.mkdir(parents=True)
    np.savez_compressed(render_cache, rgb=rgb, depth=np.ones((16, 16), dtype=np.float32), alpha=np.ones((16, 16), dtype=np.float32))
    manifest = tmp_path / "render_manifest.csv"
    _write_csv(manifest, [{"query_id": "seq0/frame00000.png", "rgb_depth_cache_path": str(render_cache)}])
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/frame00000.png",
                "track_id": 7,
                "support_track_id": 7,
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 7.5,
                "center_y": 8.0,
                "query_gt_x": 8.0,
                "query_gt_y": 8.0,
                "target_is_dustbin": "False",
            }
        ],
    )
    model = RGBPatchMeasurementBranch(search_radius_px=1.0, context_radius_px=2.0, step_px=0.5, feature_dim=4)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "search_radius_px": 1.0,
                "context_radius_px": 2.0,
                "step_px": 0.5,
                "feature_dim": 4,
                "template_scale_factors": [1.0],
            },
        },
        checkpoint,
    )
    output_dir = tmp_path / "diagnostics"

    summary = export_rgb_patch_diagnostics(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        checkpoint=checkpoint,
        output_dir=output_dir,
        image_width=16,
        image_height=16,
        query_source="render",
        support_patch_warp="none",
        max_rows=1,
        visualize_limit=1,
        device="cpu",
    )

    assert summary["row_count"] == 1
    assert summary["support_patch_warp"] == "none"
    assert summary["metrics_scope"] == "valid_non_dustbin_rows_only"
    assert summary["metrics"]["count"] == 1
    assert summary["dustbin_metrics"]["dustbin_count"] == 0
    assert summary["support_patch_source_audit"]["support_patch_source"] == "render_cache_by_query"
    assert "mode_median_px" in summary["metrics"]
    assert (output_dir / "diagnostic_rows.csv").exists()
    assert list((output_dir / "visualizations").glob("*.png"))
    row = next(csv.DictReader((output_dir / "diagnostic_rows.csv").open()))
    assert {
        "baseline_epe_px",
        "likelihood_epe_px",
        "mode_epe_px",
        "direct_epe_px",
        "gated_epe_px",
        "pred_dx",
        "pred_dy",
        "peak_dx",
        "peak_dy",
        "query_mode_x",
        "query_mode_y",
        "query_pred_x",
        "query_pred_y",
        "query_gated_x",
        "query_gated_y",
        "measurement_gate_probability",
        "track_id",
        "support_track_id",
        "target_is_dustbin",
    }.issubset(row.keys())


def test_export_rgb_patch_diagnostics_batching_matches_single_row_mode(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[8, 8] = [255, 255, 255]
    rgb[12, 12] = [255, 255, 255]
    Image.fromarray(rgb).save(image_root / "seq0" / "frame00000.png")
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/frame00000.png",
                "support_image_id": "seq0/frame00000.png",
                "track_id": 7,
                "support_track_id": 7,
                "support_x": 8.0,
                "support_y": 8.0,
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 7.5,
                "center_y": 8.0,
                "query_gt_x": 8.0,
                "query_gt_y": 8.0,
                "requested_residual_px": 0.5,
                "target_is_dustbin": "False",
            },
            {
                "query_id": "seq0/frame00000.png",
                "support_image_id": "seq0/frame00000.png",
                "track_id": 8,
                "support_track_id": 8,
                "support_x": 12.0,
                "support_y": 12.0,
                "render_x": 12.0,
                "render_y": 12.0,
                "center_x": 12.0,
                "center_y": 11.5,
                "query_gt_x": 12.0,
                "query_gt_y": 12.0,
                "requested_residual_px": 0.5,
                "target_is_dustbin": "False",
            },
        ],
    )
    model = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=0.5,
        feature_dim=4,
        condition_on_prior_scale=True,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "search_radius_px": 1.0,
                "context_radius_px": 2.0,
                "step_px": 0.5,
                "feature_dim": 4,
                "template_scale_factors": [1.0],
                "condition_on_prior_scale": True,
            },
        },
        checkpoint,
    )

    export_rgb_patch_diagnostics(
        rows_csv=rows_csv,
        render_cache_manifest_csv=None,
        image_root=image_root,
        checkpoint=checkpoint,
        output_dir=tmp_path / "single",
        image_width=20,
        image_height=20,
        query_source="real_pair",
        max_rows=2,
        visualize_limit=0,
        batch_size=1,
        prior_scale_key="requested_residual_px",
        device="cpu",
    )
    export_rgb_patch_diagnostics(
        rows_csv=rows_csv,
        render_cache_manifest_csv=None,
        image_root=image_root,
        checkpoint=checkpoint,
        output_dir=tmp_path / "batched",
        image_width=20,
        image_height=20,
        query_source="real_pair",
        max_rows=2,
        visualize_limit=0,
        batch_size=2,
        prior_scale_key="requested_residual_px",
        device="cpu",
    )

    single_rows = list(csv.DictReader((tmp_path / "single" / "diagnostic_rows.csv").open()))
    batched_rows = list(csv.DictReader((tmp_path / "batched" / "diagnostic_rows.csv").open()))
    assert len(single_rows) == len(batched_rows) == 2
    for single, batched in zip(single_rows, batched_rows):
        assert single["track_id"] == batched["track_id"]
        for key in (
            "query_pred_x",
            "query_pred_y",
            "pred_dx",
            "pred_dy",
            "dustbin_probability",
            "measurement_gate_probability",
        ):
            # Grouped convolutions may accumulate channels in a different order
            # for batch one versus batch many.
            assert np.isclose(float(single[key]), float(batched[key]), atol=1e-5)

    filtered_source = list(csv.DictReader(rows_csv.open()))
    filtered_source[0]["target_is_dustbin"] = "True"
    filtered_rows_csv = tmp_path / "filtered_rows.csv"
    _write_csv(filtered_rows_csv, filtered_source)
    export_rgb_patch_diagnostics(
        rows_csv=filtered_rows_csv,
        render_cache_manifest_csv=None,
        image_root=image_root,
        checkpoint=checkpoint,
        output_dir=tmp_path / "filtered",
        image_width=20,
        image_height=20,
        query_source="real_pair",
        visualize_limit=0,
        batch_size=2,
        prior_scale_key="requested_residual_px",
        target_dustbin_filter="valid",
        device="cpu",
    )
    filtered = list(
        csv.DictReader(
            (tmp_path / "filtered" / "diagnostic_rows.csv").open()
        )
    )
    assert len(filtered) == 1
    assert filtered[0]["row_index"] == "1"


def test_export_rgb_patch_diagnostics_loads_legacy_checkpoint_without_empty_prior_buffer(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, 8] = [255, 255, 255]
    Image.fromarray(rgb).save(image_root / "seq0" / "frame00000.png")
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/frame00000.png",
                "support_image_id": "seq0/frame00000.png",
                "support_x": 8.0,
                "support_y": 8.0,
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 7.5,
                "center_y": 8.0,
                "query_gt_x": 8.0,
                "query_gt_y": 8.0,
                "target_is_dustbin": "False",
            }
        ],
    )
    model = RGBPatchMeasurementBranch(search_radius_px=1.0, context_radius_px=2.0, step_px=0.5, feature_dim=4)
    state = dict(model.state_dict())
    state.pop("prior_scale_expert_centers")
    checkpoint = tmp_path / "legacy_model.pt"
    torch.save(
        {
            "model": state,
            "config": {
                "search_radius_px": 1.0,
                "context_radius_px": 2.0,
                "step_px": 0.5,
                "feature_dim": 4,
                "template_scale_factors": [1.0],
            },
        },
        checkpoint,
    )

    summary = export_rgb_patch_diagnostics(
        rows_csv=rows_csv,
        render_cache_manifest_csv=None,
        image_root=image_root,
        checkpoint=checkpoint,
        output_dir=tmp_path / "diagnostics",
        image_width=16,
        image_height=16,
        query_source="real_pair",
        max_rows=1,
        visualize_limit=0,
        device="cpu",
    )

    assert summary["row_count"] == 1
