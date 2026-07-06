from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm import dense_depth_pose_error_budget as budget


def _rows() -> list[dict[str, object]]:
    return [
        {
            "query_id": "q0.png",
            "match_index": "0",
            "query_center_x": "10.0",
            "query_center_y": "10.0",
            "query_gt_x": "11.0",
            "query_gt_y": "10.0",
            "world_x": "0.0",
            "world_y": "0.0",
            "world_z": "4.0",
            "render_depth": "4.0",
            "radio_match_score": "0.9",
            "measurement_valid_prob": "0.8",
            "local_cost_entropy": "0.2",
        },
        {
            "query_id": "q0.png",
            "match_index": "1",
            "query_center_x": "20.0",
            "query_center_y": "20.0",
            "query_gt_x": "27.0",
            "query_gt_y": "20.0",
            "world_x": "1.0",
            "world_y": "0.0",
            "world_z": "4.0",
            "render_depth": "5.0",
            "radio_match_score": "0.1",
            "measurement_valid_prob": "0.2",
            "local_cost_entropy": "0.9",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_pose_budget_variant_rows_filter_and_set_pixels() -> None:
    center_rows, center_summary = budget._variant_rows(_rows(), variant="center_valid_2px", seed=0)
    assert center_summary["input_row_count"] == 2
    assert center_summary["kept_row_count"] == 1
    assert center_rows[0]["query_refined_x"] == 10.0
    assert center_rows[0]["query_refined_y"] == 10.0

    oracle_rows, oracle_summary = budget._variant_rows(_rows(), variant="oracle_noise_0px", seed=0)
    assert oracle_summary["kept_row_count"] == 2
    assert oracle_rows[0]["query_refined_x"] == 11.0
    assert oracle_rows[1]["query_refined_x"] == 27.0

    quantized_rows, _summary = budget._variant_rows(_rows(), variant="oracle_quantized_stride4px", seed=0)
    assert quantized_rows[0]["query_refined_x"] == 12.0
    assert quantized_rows[1]["query_refined_x"] == 28.0


def test_center_validity_score_summary_reports_discrimination() -> None:
    validity_rows = budget.build_center_validity_rows(_rows())
    summary = budget.center_validity_score_summary(validity_rows, score_columns=("measurement_valid_prob", "inverse_local_cost_entropy"))
    valid_2 = summary["thresholds"]["valid_2px"]
    assert valid_2["positive_count"] == 1
    assert valid_2["score_columns"]["measurement_valid_prob"]["auroc"] == 1.0
    assert valid_2["score_columns"]["inverse_local_cost_entropy"]["auroc"] == 1.0
    assert valid_2["score_columns"]["measurement_valid_prob"]["precision_at_coverage"][0]["precision"] == 1.0


def test_evaluate_pose_error_budget_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    rows_csv = tmp_path / "rows.csv"
    _write_csv(rows_csv, _rows())

    def fake_matches(rows, **_kwargs):
        return list(rows), {"valid_match_count": len(rows)}

    def fake_pnp(matches, _camera, **_kwargs):
        return {
            "plain": {
                "success": bool(len(matches) >= 1),
                "match_count": len(matches),
                "inlier_count": len(matches),
                "inlier_ratio": 1.0 if matches else 0.0,
                "translation_error_m": 0.01 * len(matches),
                "rotation_error_deg": 0.1,
                "residual_median_px": 1.0,
                "residual_p90_px": 2.0,
            }
        }

    monkeypatch.setattr(budget, "dense_depth_matches_from_rows", fake_matches)
    monkeypatch.setattr(budget, "run_pnp_solver_ablation", fake_pnp)

    summary = budget.evaluate_pose_error_budget(
        match_table_csv=rows_csv,
        output_dir=tmp_path / "out",
        camera=ColmapCamera(camera_id=1, model_id=1, width=40, height=40, params=(10.0, 10.0, 20.0, 20.0)),
        variants=("center", "center_valid_2px", "oracle"),
        solvers=("plain",),
        image_width=40,
        image_height=40,
    )

    assert summary["row_count"] == 2
    assert summary["variant_preparation"]["center_valid_2px"]["kept_row_count"] == 1
    assert (tmp_path / "out" / "pose_rows.csv").exists()
    assert (tmp_path / "out" / "center_validity_rows.csv").exists()
    assert (tmp_path / "out" / "coverage_rows.csv").exists()
    pose_rows = list(csv.DictReader((tmp_path / "out" / "pose_rows.csv").open()))
    assert {row["variant"] for row in pose_rows} == {"center", "center_valid_2px", "oracle"}

