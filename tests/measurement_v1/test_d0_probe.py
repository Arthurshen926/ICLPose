from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.measurement_v1.d0_probe import FeatureSpec, run_d0_probe


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_d0_probe_reads_query_and_render_feature_caches_and_reports_metrics(tmp_path: Path) -> None:
    query_feature = np.zeros((2, 16, 16), dtype=np.float32)
    render_feature = np.zeros((2, 16, 16), dtype=np.float32)
    query_feature[1, 8, 10] = 8.0
    render_feature[1, 8, 8] = 8.0
    query_cache = tmp_path / "query.npz"
    render_cache = tmp_path / "render.npz"
    np.savez_compressed(query_cache, dual=query_feature)
    np.savez_compressed(render_cache, dual=render_feature)
    rows_csv = tmp_path / "probe_rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_feature_cache_path": str(query_cache),
                "render_feature_cache_path": str(render_cache),
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 8.0,
                "center_y": 8.0,
                "query_gt_x": 10.0,
                "query_gt_y": 8.0,
            }
        ],
    )
    output_dir = tmp_path / "d0"

    summary = run_d0_probe(
        rows_csv=rows_csv,
        output_dir=output_dir,
        feature_specs=[FeatureSpec(name="dual", query_key="dual", render_key="dual")],
        image_width=16,
        image_height=16,
        search_radius_px=3.0,
        step_px=1.0,
        temperature=0.25,
    )

    assert (output_dir / "d0_probe_rows.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert summary["features"]["dual"]["valid_count"] == 1
    assert summary["features"]["dual"]["recall_0p5px"] == 1.0
    assert summary["features"]["dual"]["epe_mean_px"] < 0.1
    assert summary["features"]["dual"]["nll_mean"] < 0.01
    assert summary["features"]["dual"]["by_residual_bin"]["2-4px"]["valid_count"] == 1
    assert summary["features"]["dual"]["top5_mode_recall_1px"] == 1.0
    assert json.loads((output_dir / "summary.json").read_text())["features"]["dual"]["entropy_mean"] >= 0.0


def test_d0_probe_accepts_hwc_large_channel_feature_caches(tmp_path: Path) -> None:
    query_feature = np.zeros((16, 16, 12), dtype=np.float32)
    render_feature = np.zeros((16, 16, 12), dtype=np.float32)
    query_feature[8, 10, 5] = 8.0
    render_feature[8, 8, 5] = 8.0
    query_cache = tmp_path / "query_hwc.npz"
    render_cache = tmp_path / "render_hwc.npz"
    np.savez_compressed(query_cache, dual=query_feature)
    np.savez_compressed(render_cache, dual=render_feature)
    rows_csv = tmp_path / "probe_rows_hwc.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_feature_cache_path": str(query_cache),
                "render_feature_cache_path": str(render_cache),
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 8.0,
                "center_y": 8.0,
                "query_gt_x": 10.0,
                "query_gt_y": 8.0,
            }
        ],
    )

    summary = run_d0_probe(
        rows_csv=rows_csv,
        output_dir=tmp_path / "d0_hwc",
        feature_specs=[FeatureSpec(name="dual", query_key="dual", render_key="dual")],
        image_width=16,
        image_height=16,
        search_radius_px=3.0,
        step_px=1.0,
        temperature=0.25,
    )

    assert summary["features"]["dual"]["valid_count"] == 1
    assert summary["features"]["dual"]["recall_0p5px"] == 1.0


def test_d0_probe_records_missing_feature_keys_without_faking_metrics(tmp_path: Path) -> None:
    query_cache = tmp_path / "query.npz"
    render_cache = tmp_path / "render.npz"
    np.savez_compressed(query_cache, present=np.zeros((1, 4, 4), dtype=np.float32))
    np.savez_compressed(render_cache, present=np.zeros((1, 4, 4), dtype=np.float32))
    rows_csv = tmp_path / "probe_rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "query_feature_cache_path": str(query_cache),
                "render_feature_cache_path": str(render_cache),
                "render_x": 1.0,
                "render_y": 1.0,
                "center_x": 1.0,
                "center_y": 1.0,
                "query_gt_x": 1.0,
                "query_gt_y": 1.0,
            }
        ],
    )

    summary = run_d0_probe(
        rows_csv=rows_csv,
        output_dir=tmp_path / "d0_missing",
        feature_specs=[FeatureSpec(name="radio_final", query_key="radio_final", render_key="radio_final")],
        image_width=4,
        image_height=4,
        search_radius_px=1.0,
        step_px=1.0,
    )

    feature_summary = summary["features"]["radio_final"]
    assert feature_summary["valid_count"] == 0
    assert feature_summary["missing_count"] == 1
    assert feature_summary["missing_reasons"]["missing_query_key:radio_final"] == 1


def test_d0_probe_xy_only_baseline_runs_without_feature_cache_and_records_residual_bins(tmp_path: Path) -> None:
    rows_csv = tmp_path / "probe_rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q0.png",
                "render_x": 4.0,
                "render_y": 4.0,
                "center_x": 4.0,
                "center_y": 4.0,
                "query_gt_x": 4.0,
                "query_gt_y": 4.0,
            },
            {
                "query_id": "q1.png",
                "render_x": 4.0,
                "render_y": 4.0,
                "center_x": 4.0,
                "center_y": 4.0,
                "query_gt_x": 7.0,
                "query_gt_y": 4.0,
            },
        ],
    )

    summary = run_d0_probe(
        rows_csv=rows_csv,
        output_dir=tmp_path / "d0_xy",
        feature_specs=[FeatureSpec(name="xy_only", query_key="__xy_only__", render_key="__xy_only__")],
        image_width=16,
        image_height=16,
        search_radius_px=4.0,
        step_px=1.0,
        temperature=0.25,
    )
    rows = list(csv.DictReader((tmp_path / "d0_xy" / "d0_probe_rows.csv").open()))

    assert summary["features"]["xy_only"]["valid_count"] == 2
    assert summary["features"]["xy_only"]["by_residual_bin"]["0-1px"]["valid_count"] == 1
    assert summary["features"]["xy_only"]["by_residual_bin"]["2-4px"]["valid_count"] == 1
    assert "peak_second_margin" in rows[0]
    assert "residual_bin" in rows[0]


def test_d0_probe_accepts_small_feature_cache_capacity_for_streaming_large_caches(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(3):
        query_feature = np.zeros((1, 8, 8), dtype=np.float32)
        render_feature = np.zeros((1, 8, 8), dtype=np.float32)
        query_feature[0, 4, 4 + idx] = 4.0
        render_feature[0, 4, 4] = 4.0
        query_cache = tmp_path / f"query_{idx}.npz"
        render_cache = tmp_path / f"render_{idx}.npz"
        np.savez_compressed(query_cache, dual=query_feature)
        np.savez_compressed(render_cache, dual=render_feature)
        rows.append(
            {
                "query_id": f"q{idx}.png",
                "query_feature_cache_path": str(query_cache),
                "render_feature_cache_path": str(render_cache),
                "render_x": 4.0,
                "render_y": 4.0,
                "center_x": 4.0,
                "center_y": 4.0,
                "query_gt_x": float(4 + idx),
                "query_gt_y": 4.0,
            }
        )
    rows_csv = tmp_path / "probe_rows.csv"
    _write_csv(rows_csv, rows)

    summary = run_d0_probe(
        rows_csv=rows_csv,
        output_dir=tmp_path / "d0_lru",
        feature_specs=[FeatureSpec(name="dual", query_key="dual", render_key="dual")],
        image_width=8,
        image_height=8,
        search_radius_px=3.0,
        step_px=1.0,
        temperature=0.25,
        feature_cache_capacity=1,
    )

    assert summary["feature_cache_capacity"] == 1
    assert summary["features"]["dual"]["valid_count"] == 3
