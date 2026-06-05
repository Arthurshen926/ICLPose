from __future__ import annotations

from feature_extract.vfm.pose_head_selection import (
    FEATURE_COLUMNS,
    evaluate_pose_head_selection,
    guarded_multi_pose_head_selection,
    guarded_pairwise_pose_head_selection,
    train_pose_head_selector,
)


def _row(query_id: str, risk: float, residual: float, translation: float, head: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "head_name": head,
        "pose_risk": risk,
        "pnp_inlier_count": 100,
        "pnp_inlier_ratio": 1.0,
        "translation_error_m": translation,
        "rotation_error_deg": 1.0,
        "pnp_reprojection": {
            "pnp_reproj_inlier_median_px": residual,
            "pnp_reproj_inlier_p90_px": residual * 1.5,
        },
        "pnp_inlier_spatial": {
            "grid_4x4_occupancy_frac": 0.8,
            "convex_hull_area_frac": 0.6,
            "depth_range_m": 4.0,
            "xyz_linearity_ratio": 0.1,
            "xyz_planarity_ratio": 0.1,
        },
        "patch_geometry": {
            "pnp_inlier_patch_at_1": 0.7,
            "pnp_inlier_gt_precision_stride": 0.9,
        },
        "patch_offset_refinement": {
            "refined_count": 0,
            "fixed_same_inlier_count": 100,
        },
    }


def test_pose_head_selector_learns_to_choose_lower_risk_successful_head() -> None:
    rows_by_head = {
        "stable": [
            _row("train_a", 0.1, 2.0, 0.10, "stable"),
            _row("train_b", 0.2, 3.0, 0.12, "stable"),
            _row("eval_a", 0.1, 2.2, 0.11, "stable"),
        ],
        "risky": [
            _row("train_a", 0.8, 12.0, 0.80, "risky"),
            _row("train_b", 0.7, 10.0, 0.90, "risky"),
            _row("eval_a", 0.9, 11.0, 0.95, "risky"),
        ],
    }

    model = train_pose_head_selector(
        rows_by_head,
        train_query_ids={"train_a", "train_b"},
        iterations=250,
        learning_rate=0.2,
    )
    summary = evaluate_pose_head_selection(
        rows_by_head,
        query_ids={"eval_a"},
        model=model,
        baseline_head="risky",
    )

    assert summary["selected_head_counts"] == {"stable": 1}
    assert summary["success_25cm_10deg"] == 1.0
    assert summary["median_translation_error_m"] == 0.11
    assert summary["rescue_break_vs_baseline"]["rescued"] == 1
    assert summary["rescue_break_vs_baseline"]["broken"] == 0


def test_pose_head_selector_feature_columns_exclude_gt_only_diagnostics() -> None:
    blocked = [column for column in FEATURE_COLUMNS if "gt_" in column or "patch_at_" in column]

    assert blocked == []


def test_guarded_pose_head_selection_uses_residual_improvement_without_training() -> None:
    rows_by_head = {
        "baseline": [
            _row("q0", 0.2, 5.0, 0.30, "baseline"),
            _row("q1", 0.2, 5.0, 0.10, "baseline"),
        ],
        "confidence": [
            _row("q0", 0.2, 3.0, 0.12, "confidence"),
            _row("q1", 0.2, 6.0, 0.40, "confidence"),
        ],
    }

    summary = guarded_pairwise_pose_head_selection(
        rows_by_head,
        baseline_head="baseline",
        candidate_head="confidence",
        min_reproj_improvement_px=0.0,
        max_inlier_count_drop=0,
    )

    assert summary["selected_head_counts"] == {"baseline": 1, "confidence": 1}
    assert summary["success_25cm_10deg"] == 1.0
    assert summary["rescue_break_vs_baseline"]["rescued"] == 1
    assert summary["rescue_break_vs_baseline"]["broken"] == 0


def test_guarded_multi_pose_head_selection_picks_lowest_residual_safe_candidate() -> None:
    rows_by_head = {
        "baseline": [
            _row("q0", 0.2, 5.0, 0.30, "baseline"),
            _row("q1", 0.2, 5.0, 0.10, "baseline"),
            _row("q2", 0.2, 5.0, 0.12, "baseline"),
        ],
        "confidence": [
            _row("q0", 0.2, 4.0, 0.18, "confidence"),
            _row("q1", 0.2, 6.0, 0.40, "confidence"),
            dict(_row("q2", 0.2, 2.0, 0.60, "confidence"), pnp_inlier_count=80),
        ],
        "diagonal": [
            _row("q0", 0.2, 3.0, 0.11, "diagonal"),
            _row("q1", 0.2, 4.0, 0.20, "diagonal"),
            dict(_row("q2", 0.2, 1.0, 0.50, "diagonal"), pnp_inlier_count=80),
        ],
    }

    summary = guarded_multi_pose_head_selection(
        rows_by_head,
        baseline_head="baseline",
        candidate_heads=("confidence", "diagonal"),
        min_reproj_improvement_px=0.0,
        max_inlier_count_drop=0,
    )

    per_query = {row["query_id"]: row["selected_head"] for row in summary["rows"]}
    assert per_query == {"q0": "diagonal", "q1": "diagonal", "q2": "baseline"}
    assert summary["success_25cm_10deg"] == 1.0
    assert summary["rescue_break_vs_baseline"]["rescued"] == 1
    assert summary["rescue_break_vs_baseline"]["broken"] == 0


def test_pose_head_selector_reports_oracle_upper_bound() -> None:
    rows_by_head = {
        "a": [_row("q0", 0.5, 5.0, 0.40, "a"), _row("q1", 0.1, 2.0, 0.10, "a")],
        "b": [_row("q0", 0.2, 3.0, 0.12, "b"), _row("q1", 0.8, 9.0, 0.60, "b")],
    }
    model = train_pose_head_selector(rows_by_head, train_query_ids={"q0", "q1"}, iterations=20)
    summary = evaluate_pose_head_selection(rows_by_head, query_ids={"q0", "q1"}, model=model)

    assert summary["oracle"]["success_25cm_10deg"] == 1.0
    assert summary["oracle"]["median_translation_error_m"] == 0.11


def test_pose_head_selection_cli_writes_eval_summary(tmp_path) -> None:
    from feature_extract.tools.vfm.run_pose_head_selection import main

    stable_path = tmp_path / "stable.jsonl"
    risky_path = tmp_path / "risky.jsonl"
    stable_path.write_text(
        "\n".join(
            [
                __import__("json").dumps(_row("train_a", 0.1, 2.0, 0.10, "stable")),
                __import__("json").dumps(_row("train_b", 0.2, 3.0, 0.12, "stable")),
                __import__("json").dumps(_row("eval_a", 0.1, 2.2, 0.11, "stable")),
            ]
        )
        + "\n"
    )
    risky_path.write_text(
        "\n".join(
            [
                __import__("json").dumps(_row("train_a", 0.8, 12.0, 0.80, "risky")),
                __import__("json").dumps(_row("train_b", 0.7, 10.0, 0.90, "risky")),
                __import__("json").dumps(_row("eval_a", 0.9, 11.0, 0.95, "risky")),
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--head",
            f"stable={stable_path}",
            "--head",
            f"risky={risky_path}",
            "--train_query",
            "train_a",
            "--train_query",
            "train_b",
            "--eval_query",
            "eval_a",
            "--baseline_head",
            "risky",
            "--output_dir",
            str(output_dir),
        ]
    )

    summary = __import__("json").loads((output_dir / "pose_head_selection_summary.json").read_text())
    selected_rows = (output_dir / "selected_eval_rows.jsonl").read_text().splitlines()
    assert summary["eval"]["selected_head_counts"] == {"stable": 1}
    assert summary["eval"]["rescue_break_vs_baseline"]["rescued"] == 1
    assert len(selected_rows) == 1
