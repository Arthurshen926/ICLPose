from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feature_retrieval.tools.train_candidate_selector import (
    DEFAULT_FEATURE_COLUMNS,
    build_candidate_dataset,
    evaluate_rule_selector,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_candidate_dataset_merges_feature_csv_without_error_leakage(tmp_path):
    label_csv = tmp_path / "label.csv"
    feature_csv = tmp_path / "feature.csv"
    _write_csv(
        label_csv,
        [
            {
                "image_name": "seq/frame1.png",
                "hyp_final_rot_err_deg": json.dumps([0.5, 0.1, 0.2]),
                "hyp_final_trans_err_mm": json.dumps([150.0, 80.0, 120.0]),
                "hyp_init_trans_err_mm": json.dumps([1.0, 2.0, 3.0]),
                "hyp_candidate_quality_scores": json.dumps([0.2, 0.8, 0.5]),
            },
            {
                "image_name": "seq/frame2.png",
                "hyp_final_rot_err_deg": json.dumps([0.4, 0.2, 0.3]),
                "hyp_final_trans_err_mm": json.dumps([90.0, 110.0, 70.0]),
                "hyp_init_trans_err_mm": json.dumps([4.0, 5.0, 6.0]),
                "hyp_candidate_quality_scores": json.dumps([0.7, 0.1, 0.9]),
            },
        ],
    )
    _write_csv(
        feature_csv,
        [
            {
                "image_name": "seq/frame1.png",
                "hyp_loftr_render_scores": json.dumps([10.0, 30.0, 20.0]),
                "hyp_loftr_render_inlier_ratio": json.dumps([0.2, 0.9, 0.4]),
            },
            {
                "image_name": "seq/frame2.png",
                "hyp_loftr_render_scores": json.dumps([40.0, 10.0, 50.0]),
                "hyp_loftr_render_inlier_ratio": json.dumps([0.8, 0.1, 0.95]),
            },
        ],
    )

    dataset = build_candidate_dataset([label_csv], [feature_csv])

    assert dataset.features.shape[0] == 6
    assert dataset.sample_ids.tolist() == [0, 0, 0, 1, 1, 1]
    assert dataset.candidate_indices.tolist() == [0, 1, 2, 0, 1, 2]
    assert dataset.best_indices == [1, 2]
    assert dataset.image_names == ["seq/frame1.png", "seq/frame2.png"]
    assert "rank_norm" in dataset.feature_names
    assert "hyp_loftr_render_scores" in dataset.feature_names
    assert "hyp_loftr_render_scores_row01" in dataset.feature_names
    assert "hyp_init_trans_err_mm" not in dataset.feature_names
    assert all("err" not in name for name in dataset.feature_names)


def test_score_max_rule_uses_merged_feature_columns(tmp_path):
    label_csv = tmp_path / "label.csv"
    feature_csv = tmp_path / "feature.csv"
    _write_csv(
        label_csv,
        [
            {
                "image_name": "a.png",
                "hyp_final_rot_err_deg": json.dumps([1.0, 0.2, 0.8]),
                "hyp_final_trans_err_mm": json.dumps([200.0, 70.0, 90.0]),
            },
            {
                "image_name": "b.png",
                "hyp_final_rot_err_deg": json.dumps([0.3, 0.9, 0.2]),
                "hyp_final_trans_err_mm": json.dumps([120.0, 150.0, 60.0]),
            },
        ],
    )
    _write_csv(
        feature_csv,
        [
            {"image_name": "a.png", "hyp_loftr_render_scores": json.dumps([0.1, 0.8, 0.2])},
            {"image_name": "b.png", "hyp_loftr_render_scores": json.dumps([0.3, 0.2, 0.7])},
        ],
    )

    dataset = build_candidate_dataset([label_csv], [feature_csv], feature_columns=DEFAULT_FEATURE_COLUMNS)
    summary = evaluate_rule_selector(dataset, "score_max", "hyp_loftr_render_scores")

    assert summary.selected_indices == [1, 2]
    assert np.isclose(summary.median_trans_mm, 65.0)
    assert np.isclose(summary.success_pose_pct, 100.0)
