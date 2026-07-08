from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from feature_extract.vfm import center_validity_model as cvm


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "query_id": "q0",
            "candidate_id": "c0",
            "match_index": "0",
            "valid_5px": "True",
            "valid_2px": "True",
            "center_error_px": "1.0",
            "radio_match_score": "0.9",
            "measurement_valid_prob": "0.8",
            "local_cost_entropy": "0.1",
            "local_cost_peak_prob": "0.7",
            "local_cost_top2_gap": "0.4",
            "render_depth": "4.0",
            "query_center_x": "10",
            "query_center_y": "10",
        },
        {
            "query_id": "q0",
            "candidate_id": "c0",
            "match_index": "1",
            "valid_5px": "False",
            "valid_2px": "False",
            "center_error_px": "8.0",
            "radio_match_score": "0.2",
            "measurement_valid_prob": "0.3",
            "local_cost_entropy": "0.9",
            "local_cost_peak_prob": "0.1",
            "local_cost_top2_gap": "0.05",
            "render_depth": "8.0",
            "query_center_x": "20",
            "query_center_y": "20",
        },
        {
            "query_id": "q1",
            "candidate_id": "c0",
            "match_index": "2",
            "valid_5px": "True",
            "valid_2px": "False",
            "center_error_px": "4.0",
            "radio_match_score": "0.5",
            "measurement_valid_prob": "0.6",
            "local_cost_entropy": "0.2",
            "local_cost_peak_prob": "0.5",
            "local_cost_top2_gap": "0.2",
            "render_depth": "5.0",
            "query_center_x": "100",
            "query_center_y": "100",
        },
    ]


def test_build_center_validity_feature_rows_adds_group_ranks_without_gt_leakage() -> None:
    feature_rows, feature_columns = cvm.build_feature_rows(_rows(), image_width=200, image_height=200)

    assert "center_error_px" not in feature_columns
    assert "valid_5px" not in feature_columns
    assert "radio_match_score_rank_pct" in feature_columns
    assert "inverse_local_cost_entropy_rank_pct" in feature_columns
    q0 = [row for row in feature_rows if row["query_id"] == "q0"]
    assert float(q0[0]["radio_match_score_rank_pct"]) == 1.0
    assert float(q0[1]["radio_match_score_rank_pct"]) == 0.5
    assert float(q0[0]["inverse_local_cost_entropy_rank_pct"]) == 1.0
    assert float(q0[1]["inverse_local_cost_entropy_rank_pct"]) == 0.5
    assert float(q0[0]["query_center_x_norm"]) == 0.05


def test_query_level_split_keeps_queries_disjoint() -> None:
    splits = cvm.split_rows_by_query(_rows(), train_fraction=0.34, calibration_fraction=0.33, seed=7)
    query_sets = {name: {row["query_id"] for row in values} for name, values in splits.items()}

    assert query_sets["train"].isdisjoint(query_sets["calibration"])
    assert query_sets["train"].isdisjoint(query_sets["validation"])
    assert query_sets["calibration"].isdisjoint(query_sets["validation"])
    assert query_sets["train"] | query_sets["calibration"] | query_sets["validation"] == {"q0", "q1"}


def test_precision_coverage_metrics_use_positive_ranking() -> None:
    rows = cvm.evaluate_scores(
        scores=np.asarray([0.9, 0.8, 0.1], dtype=np.float64),
        labels=np.asarray([1, 0, 1], dtype=np.float64),
        coverages=(1 / 3, 2 / 3, 1.0),
    )

    assert rows["positive_rate"] == 2 / 3
    assert rows["precision_at_coverage"][0]["precision"] == 1.0
    assert rows["precision_at_coverage"][1]["precision"] == 0.5


def test_grid_balanced_selection_improves_grid_coverage_over_top_score() -> None:
    rows = []
    for index, (x, y, score) in enumerate(
        [
            (10, 10, 0.99),
            (11, 10, 0.98),
            (12, 10, 0.97),
            (150, 10, 0.80),
            (10, 150, 0.79),
            (150, 150, 0.78),
        ]
    ):
        rows.append({"query_center_x": x, "query_center_y": y, "p_valid_5px": score, "match_index": index})

    top = cvm.select_rows_by_score(rows, score_key="p_valid_5px", budget=4)
    balanced = cvm.select_rows_grid_balanced(rows, score_key="p_valid_5px", budget=4, image_width=200, image_height=200)

    assert cvm.grid4_coverage(top, image_width=200, image_height=200) == 2
    assert cvm.grid4_coverage(balanced, image_width=200, image_height=200) == 4


def test_grid_depth_balanced_selection_keeps_spatial_and_depth_diversity() -> None:
    rows = []
    for index, (x, y, depth, score) in enumerate(
        [
            (10, 10, 4.0, 0.99),
            (11, 10, 4.1, 0.98),
            (12, 10, 12.0, 0.97),
            (150, 10, 6.0, 0.80),
            (10, 150, 8.0, 0.79),
            (150, 150, 10.0, 0.78),
        ]
    ):
        rows.append(
            {
                "query_center_x": x,
                "query_center_y": y,
                "render_depth": depth,
                "p_valid_5px": score,
                "match_index": index,
            }
        )

    selected = cvm.select_rows_grid_depth_balanced(
        rows,
        score_key="p_valid_5px",
        budget=5,
        image_width=200,
        image_height=200,
    )

    assert cvm.grid4_coverage(selected, image_width=200, image_height=200) == 4
    assert len({cvm.depth_bin(row, rows=rows, bin_count=4) for row in selected}) >= 3


def test_coverage_gate_falls_back_to_full_group_when_selection_is_unsafe() -> None:
    rows = [
        {"query_center_x": 10, "query_center_y": 10, "render_depth": 4.0, "p_valid_5px": 0.9, "match_index": "0"},
        {"query_center_x": 11, "query_center_y": 10, "render_depth": 4.1, "p_valid_5px": 0.8, "match_index": "1"},
        {"query_center_x": 150, "query_center_y": 150, "render_depth": 8.0, "p_valid_5px": 0.1, "match_index": "2"},
    ]

    gated = cvm.coverage_gate_selection(
        selected_rows=rows[:2],
        fallback_rows=rows,
        score_key="p_valid_5px",
        image_width=200,
        image_height=200,
        min_kept=2,
        min_grid4_coverage=2,
        min_depth_span_m=0.1,
        min_score_threshold=0.5,
        min_score_count=2,
    )

    assert len(gated) == len(rows)


def test_p5_then_secondary_grid_selection_uses_p5_pool_and_p2_ranking() -> None:
    rows = []
    for index, (x, y, p5, p2) in enumerate(
        [
            (10, 10, 0.99, 0.10),
            (20, 10, 0.98, 0.90),
            (150, 10, 0.97, 0.80),
            (10, 150, 0.20, 0.99),
            (150, 150, 0.96, 0.70),
        ]
    ):
        rows.append({"query_center_x": x, "query_center_y": y, "p5": p5, "p2": p2, "match_index": str(index)})

    selected = cvm.select_rows_p5_then_secondary_grid(
        rows,
        p5_score_key="p5",
        secondary_score_key="p2",
        budget=3,
        image_width=200,
        image_height=200,
        p5_pool_fraction=0.8,
    )

    assert {row["match_index"] for row in selected} == {"1", "2", "4"}


def test_train_validity_model_returns_calibrated_validation_scores() -> None:
    rows = []
    for query_idx in range(6):
        for match_idx in range(8):
            positive = match_idx < 2
            rows.append(
                {
                    "query_id": f"q{query_idx}",
                    "candidate_id": "c0",
                    "match_index": f"{query_idx}_{match_idx}",
                    "valid_5px": str(positive),
                    "valid_2px": str(match_idx == 0),
                    "center_error_px": "1.0" if positive else "9.0",
                    "radio_match_score": 1.0 - 0.1 * match_idx,
                    "measurement_valid_prob": 0.8 if positive else 0.2,
                    "local_cost_entropy": 0.1 if positive else 0.9,
                    "local_cost_peak_prob": 0.9 if positive else 0.1,
                    "local_cost_top2_gap": 0.5 if positive else 0.01,
                    "render_depth": 4.0 + match_idx,
                    "query_center_x": 10.0 * match_idx,
                    "query_center_y": 10.0 * query_idx,
                }
            )
    feature_rows, feature_columns = cvm.build_feature_rows(rows, image_width=100, image_height=100)

    result = cvm.train_validity_model(
        feature_rows,
        feature_columns,
        target="valid_5px",
        model_type="logistic",
        seed=3,
        steps=60,
        learning_rate=5e-2,
        train_fraction=0.5,
        calibration_fraction=0.25,
    )
    scores = cvm.predict_validity_scores(result, feature_rows)

    assert scores.shape == (len(feature_rows),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert result["metrics"]["validation"]["auprc"] is not None


def test_evaluate_center_validity_pose_filtering_outputs_selection_variants(tmp_path: Path, monkeypatch) -> None:
    match_rows = []
    score_rows = []
    for index, (x, y, score, valid) in enumerate(
        [
            (10, 10, 0.99, True),
            (11, 10, 0.98, True),
            (150, 10, 0.70, False),
            (10, 150, 0.69, False),
            (150, 150, 0.68, True),
            (80, 80, 0.10, False),
        ]
    ):
        row = {
            "query_id": "q0",
            "candidate_id": "",
            "match_index": str(index),
            "query_center_x": str(x),
            "query_center_y": str(y),
            "query_gt_x": str(x),
            "query_gt_y": str(y),
            "world_x": str(index),
            "world_y": "0",
            "world_z": "5",
            "render_depth": "5",
            "local_cost_entropy": str(1.0 - score),
        }
        match_rows.append(row)
        score_item = dict(row)
        score_item["p_valid_5px"] = str(score)
        score_item["p_valid_2px"] = str(1.0 - 0.1 * index)
        score_item["valid_5px"] = str(valid)
        score_item["valid_2px"] = str(valid and index == 0)
        score_rows.append(score_item)
    match_csv = tmp_path / "match.csv"
    score_csv = tmp_path / "scores.csv"
    _write_csv(match_csv, match_rows)
    _write_csv(score_csv, score_rows)

    def fake_matches(rows, **_kwargs):
        return list(rows), {"valid_match_count": len(rows)}

    def fake_pnp(matches, _camera, **_kwargs):
        return {
            "ransac": {
                "solver": "ransac",
                "success": bool(len(matches) >= 1),
                "match_count": len(matches),
                "inlier_count": len(matches),
                "inlier_ratio": 1.0 if matches else 0.0,
                "translation_error_m": 1.0 / max(len(matches), 1),
                "rotation_error_deg": 0.1,
                "residual_median_px": 1.0,
                "residual_p90_px": 2.0,
            }
        }

    monkeypatch.setattr(cvm, "dense_depth_matches_from_rows", fake_matches)
    monkeypatch.setattr(cvm, "_run_pnp_solver_ablation_safe", fake_pnp)

    summary = cvm.evaluate_center_validity_pose_filtering(
        match_table_csv=match_csv,
        score_rows_csv=score_csv,
        output_dir=tmp_path / "out",
        camera=None,
        score_key="p_valid_5px",
        secondary_score_key="p_valid_2px",
        budgets=(2, 4),
        solvers=("ransac",),
        image_width=200,
        image_height=200,
    )

    variants = {row["variant"] for row in summary["pose_rows"]}
    assert "center_all" in variants
    assert "learned_top2" in variants
    assert "learned_grid_top4" in variants
    assert "learned_grid_depth_top4" in variants
    assert "learned_top4_gated" in variants
    assert "p2_top4" in variants
    assert "p5_then_p2_grid_top4" in variants
    assert "oracle_grid_depth_top4" in variants
    assert "learned_threshold_0p500" in variants
    assert "random_top2" in variants
    assert "oracle_center_valid_5px" in variants
    grid_rows = [row for row in summary["pose_rows"] if row["variant"] == "learned_grid_top4"]
    assert grid_rows[0]["grid4_coverage"] == 4
    assert Path(summary["outputs"]["group_coverage_tsv"]).exists()


def test_evaluate_selection_oracle_update_gap_replays_same_selected_rows(tmp_path: Path, monkeypatch) -> None:
    selected_rows = []
    for index in range(4):
        selected_rows.append(
            {
                "query_id": "q0",
                "candidate_id": "",
                "match_index": str(index),
                "variant": "learned_top4",
                "query_center_x": str(10 + index),
                "query_center_y": "10",
                "query_gt_x": str(11 + index),
                "query_gt_y": "10",
                "world_x": str(index),
                "world_y": "0",
                "world_z": "5",
                "render_depth": "5",
            }
        )
    selected_csv = tmp_path / "selected.csv"
    _write_csv(selected_csv, selected_rows)

    def fake_matches(rows, **_kwargs):
        return list(rows), {"valid_match_count": len(rows)}

    def fake_pnp(matches, _camera, **_kwargs):
        first = matches[0]
        return {
            "ransac": {
                "solver": "ransac",
                "success": bool(matches),
                "match_count": len(matches),
                "inlier_count": len(matches),
                "inlier_ratio": 1.0,
                "translation_error_m": float(first["query_refined_x"]),
                "rotation_error_deg": 0.1,
                "residual_median_px": 1.0,
                "residual_p90_px": 2.0,
            }
        }

    monkeypatch.setattr(cvm, "dense_depth_matches_from_rows", fake_matches)
    monkeypatch.setattr(cvm, "_run_pnp_solver_ablation_safe", fake_pnp)

    summary = cvm.evaluate_selection_oracle_update_gap(
        selected_rows_csv=selected_csv,
        output_dir=tmp_path / "gap",
        camera=None,
        source_variants=("learned_top4",),
        update_variants=("center", "oracle_xy", "oracle_noise_1px"),
        solvers=("ransac",),
        seed=7,
    )

    variants = {row["variant"] for row in summary["pose_rows"]}
    assert "learned_top4_center" in variants
    assert "learned_top4_oracle_xy" in variants
    assert "learned_top4_oracle_noise_1px" in variants
    assert Path(summary["outputs"]["pose_summary_tsv"]).exists()
