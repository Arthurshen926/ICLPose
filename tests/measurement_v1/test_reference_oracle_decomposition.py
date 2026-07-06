from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.measurement_v1.reference_oracle_decomposition import (
    coarse_cell_centers_for_pixels,
    decompose_candidate_rows,
    summarize_decomposition_rows,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera


def test_coarse_cell_centers_for_pixels_uses_original_pixel_coordinates() -> None:
    centers, valid = coarse_cell_centers_for_pixels(
        np.asarray([[0.0, 0.0], [19.9, 9.9], [20.1, 0.0]], dtype=np.float64),
        image_width=20,
        image_height=10,
        grid_width=4,
        grid_height=2,
    )

    assert valid.tolist() == [True, True, False]
    assert np.allclose(centers[0], [2.5, 2.5])
    assert np.allclose(centers[1], [17.5, 7.5])


def test_summarize_decomposition_rows_reports_oracle_best_by_topk() -> None:
    rows = [
        {
            "query_id": "q0",
            "candidate_rank": 0,
            "variant": "predicted",
            "pnp_success": True,
            "translation_error_m": 0.30,
            "rotation_error_deg": 1.0,
        },
        {
            "query_id": "q0",
            "candidate_rank": 1,
            "variant": "predicted",
            "pnp_success": True,
            "translation_error_m": 0.05,
            "rotation_error_deg": 1.0,
        },
        {
            "query_id": "q1",
            "candidate_rank": 0,
            "variant": "predicted",
            "pnp_success": False,
            "translation_error_m": None,
            "rotation_error_deg": None,
        },
        {
            "query_id": "q1",
            "candidate_rank": 1,
            "variant": "predicted",
            "pnp_success": True,
            "translation_error_m": 0.08,
            "rotation_error_deg": 2.0,
        },
    ]

    summary = summarize_decomposition_rows(rows, topk_values=(1, 2))

    assert summary["query_count"] == 2
    assert summary["predicted_solve_rate"] == pytest.approx(0.75)
    assert summary["predicted_best_at_1_success_10cm_5deg"] == pytest.approx(0.0)
    assert summary["predicted_best_at_2_success_10cm_5deg"] == pytest.approx(1.0)
    assert summary["predicted_best_at_2_median_translation_error_m"] == pytest.approx(0.065)
    assert summary["predicted_rank0_selected_by_existing_rule_median_translation_error_m"] == pytest.approx(0.30)
    assert summary["predicted_per_candidate_median_translation_error_m"] == pytest.approx(0.08)
    assert summary["predicted_best_at_2_minus_rank0_median_translation_delta_m"] == pytest.approx(-0.235)


def test_decomposition_splits_visible_reprojection_oracle_from_raw_validity(monkeypatch) -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(50.0, 50.0, 50.0, 50.0))
    rows = [
        {
            "query_id": "q0",
            "candidate_rank": 0,
            "query_x": 50.0,
            "query_y": 50.0,
            "world_x": 0.0,
            "world_y": 0.0,
            "world_z": 5.0,
            "gt_reproj_error_px": 1.0,
        },
        {
            "query_id": "q0",
            "candidate_rank": 0,
            "query_x": 50.0,
            "query_y": 50.0,
            "world_x": 0.0,
            "world_y": 0.0,
            "world_z": -5.0,
            "gt_reproj_error_px": 1.0,
        },
    ]

    def fake_pose_row(**kwargs):
        return {
            "query_id": kwargs["query_id"],
            "candidate_rank": kwargs["candidate_rank"],
            "variant": kwargs["variant"],
            "match_count": len(kwargs["matches"]),
            "pnp_success": True,
            "translation_error_m": 0.0,
            "rotation_error_deg": 0.0,
            **dict(kwargs.get("extra") or {}),
        }

    monkeypatch.setattr(
        "feature_extract.vfm.measurement_v1.reference_oracle_decomposition._pose_row",
        fake_pose_row,
    )

    out = decompose_candidate_rows(
        query_id="q0",
        candidate_rank=0,
        rows=rows,
        camera=camera,
        gt_pose_w2c=np.eye(4),
    )
    by_variant = {row["variant"]: row for row in out}

    assert by_variant["oracle_validity_2px"]["match_count"] == 2
    assert by_variant["oracle_visible_reprojection_2px"]["match_count"] == 1
    assert by_variant["oracle_visible_reprojection_2px"]["visibility_model"] == "positive_depth_in_image"
    assert by_variant["oracle_visible_reprojection_2px"]["z_buffer_strict"] is False
