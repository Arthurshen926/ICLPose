from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feature_retrieval.tools.sweep_loftr_render_selectors import (
    parse_list_cell,
    sweep_selectors,
)


def test_parse_list_cell_accepts_json_python_lists_and_empty_values():
    assert parse_list_cell("[1, 2.5, null]") == [1.0, 2.5, float("-inf")]
    assert parse_list_cell("['3', 'nan', 4]") == [3.0, float("-inf"), 4.0]
    assert parse_list_cell("") == []
    assert parse_list_cell(None) == []


def test_sweep_selectors_scores_inliers_and_consistency(tmp_path):
    csv_path = tmp_path / "sample_metrics.csv"
    rows = [
        {
            "image_name": "a.png",
            "hyp_final_rot_err_deg": json.dumps([2.0, 0.2, 0.8]),
            "hyp_final_trans_err_mm": json.dumps([200.0, 80.0, 90.0]),
            "hyp_loftr_render_scores": json.dumps([0.1, 0.9, 0.3]),
            "hyp_loftr_render_inliers": json.dumps([5, 1, 10]),
            "hyp_loftr_render_matches": json.dumps([20, 8, 30]),
            "hyp_loftr_render_mean_conf": json.dumps([0.1, 0.8, 0.5]),
            "hyp_loftr_render_consistency_m": json.dumps([0.5, 0.4, 0.05]),
        },
        {
            "image_name": "b.png",
            "hyp_final_rot_err_deg": json.dumps([0.1, 1.5, 0.3]),
            "hyp_final_trans_err_mm": json.dumps([70.0, 160.0, 95.0]),
            "hyp_loftr_render_scores": json.dumps([0.7, 0.2, 0.6]),
            "hyp_loftr_render_inliers": json.dumps([2, 20, 5]),
            "hyp_loftr_render_matches": json.dumps([10, 35, 12]),
            "hyp_loftr_render_mean_conf": json.dumps([0.9, 0.4, 0.5]),
            "hyp_loftr_render_consistency_m": json.dumps([0.2, 0.1, 0.03]),
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_rule = {row.rule: row for row in sweep_selectors(csv_path)}

    assert by_rule["current_score_max"].selected_indices == [1, 0]
    assert by_rule["inliers_max"].selected_indices == [2, 1]
    assert by_rule["consistency_min"].selected_indices == [2, 2]
    assert np.isclose(by_rule["current_score_max"].median_trans_mm, 75.0)
    assert np.isclose(by_rule["inliers_max"].success_trans_pct, 50.0)


def test_quality_tiebreak_and_support_loftr_hybrid_use_optional_columns(tmp_path):
    csv_path = tmp_path / "sample_metrics.csv"
    row = {
        "image_name": "a.png",
        "hyp_final_rot_err_deg": json.dumps([1.0, 0.2, 0.5]),
        "hyp_final_trans_err_mm": json.dumps([150.0, 80.0, 90.0]),
        "hyp_loftr_render_scores": json.dumps([0.5, 0.5, 0.2]),
        "hyp_loftr_render_inliers": json.dumps([2, 3, 4]),
        "hyp_loftr_render_matches": json.dumps([5, 6, 7]),
        "hyp_loftr_render_mean_conf": json.dumps([0.1, 0.2, 0.3]),
        "hyp_loftr_render_consistency_m": json.dumps([0.2, 0.3, 0.4]),
        "hyp_candidate_quality_scores": json.dumps([0.1, 0.9, 0.3]),
        "hyp_consensus_support": json.dumps([1, 2, 10]),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    by_rule = {row.rule: row for row in sweep_selectors(csv_path)}

    assert by_rule["score_quality_tiebreak"].selected_indices == [1]
    assert by_rule["support_loftr_hybrid"].selected_indices == [2]
