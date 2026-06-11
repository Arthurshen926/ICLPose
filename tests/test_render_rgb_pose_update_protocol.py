from __future__ import annotations

from types import SimpleNamespace

import pytest

from feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose import (
    _render_lock_diagnostics_from_rows,
    _select_pose_update_iteration,
    parse_args,
)
from feature_extract.tools.vfm.report_perturbation_adapter_eval import main as report_perturbation_adapter_eval


def _base_args() -> list[str]:
    return [
        "--query_manifest",
        "queries.json",
        "--query_pose_file",
        "poses.txt",
        "--image_root",
        "images",
        "--gaussian_rgb_ply",
        "scene.ply",
        "--output_dir",
        "out",
    ]


def test_confidence_coverage_filter_is_default_for_render_rgb_eval() -> None:
    args = parse_args(_base_args())

    assert args.coverage_filter_grid == 8
    assert args.coverage_filter_max_per_cell == 8
    assert args.coverage_filter_max_total == 512


def test_anti_lock_render_search_preset_enables_render_side_measurement_relocation() -> None:
    args = parse_args(_base_args() + ["--matcha_eval_preset", "anti_lock_render_search"])

    assert args.matcha_pair_fine_side == "query"
    assert args.fine_render_search_radius_px == 24.0
    assert args.fine_render_search_step_px == 2.0
    assert args.matcha_fine_mode == "fine_attention_argmax"
    assert args.pnp_soft_order_mode == "confidence"
    assert args.pnp_soft_order_top_n == 800
    assert args.measurement_sigma_px == 16.0
    assert args.coverage_filter_min_confidence == 0.05


def test_post_pair_render_refine_controls_are_exposed() -> None:
    args = parse_args(
        _base_args()
        + [
            "--post_pair_render_refine_radius_px",
            "20",
            "--post_pair_render_refine_step_px",
            "2",
            "--post_pair_render_refine_mode",
            "softargmax",
            "--post_pair_render_refine_query_sigma_px",
            "6",
        ]
    )

    assert args.post_pair_render_refine_radius_px == 20.0
    assert args.post_pair_render_refine_step_px == 2.0
    assert args.post_pair_render_refine_mode == "softargmax"
    assert args.post_pair_render_refine_query_sigma_px == 6.0


def test_anti_lock_post_pair_preset_refines_render_after_query_pair_fine() -> None:
    args = parse_args(_base_args() + ["--matcha_eval_preset", "anti_lock_post_pair_render_search"])

    assert args.matcha_pair_fine_side == "query"
    assert args.fine_render_search_radius_px == 0.0
    assert args.post_pair_render_refine_radius_px == 24.0
    assert args.post_pair_render_refine_step_px == 2.0
    assert args.post_pair_render_refine_mode == "argmax"
    assert args.pnp_soft_order_mode == "confidence"
    assert args.coverage_filter_min_confidence == 0.05


def test_pose_update_iterations_is_explicitly_configurable() -> None:
    default_args = parse_args(_base_args())
    updated_args = parse_args(
        _base_args() + ["--pose_update_iterations", "3", "--pose_update_selection", "best_alignment"]
    )

    assert default_args.pose_update_iterations == 1
    assert updated_args.pose_update_iterations == 3
    assert updated_args.pose_update_selection == "best_alignment"
    assert default_args.pose_update_selection == "guarded_score"
    assert default_args.pose_update_score_margin == 0.03
    assert default_args.pose_update_filter_schedule == "final_only"


def test_matcha_learned_heads_are_exposed_as_optional_eval_controls() -> None:
    args = parse_args(
        _base_args()
        + [
            "--matcha_confidence_mode",
            "blend",
            "--matcha_confidence_blend",
            "0.7",
            "--matcha_min_detector_confidence",
            "0.2",
            "--matcha_use_pair_fine_head",
        ]
    )

    assert args.matcha_confidence_mode == "blend"
    assert args.matcha_confidence_blend == 0.7
    assert args.matcha_min_detector_confidence == 0.2
    assert args.matcha_use_pair_fine_head is True


def test_matcha_pair_fine_side_defaults_to_query() -> None:
    args = parse_args(_base_args() + ["--matcha_use_pair_fine_head"])

    assert args.matcha_pair_fine_side == "query"


def test_render_lock_diagnostics_reports_render_and_pnp_delta() -> None:
    rows = [
        {
            "render_translation_error_m": 0.25,
            "render_rotation_error_deg": 0.0,
            "translation_error_m": 0.243,
            "rotation_error_deg": 0.05,
            "pnp_render_translation_delta_m": 0.011,
            "pnp_render_rotation_delta_deg": 0.02,
        },
        {
            "render_translation_error_m": 0.50,
            "render_rotation_error_deg": 0.0,
            "translation_error_m": 0.10,
            "rotation_error_deg": 0.04,
            "pnp_render_translation_delta_m": 0.39,
            "pnp_render_rotation_delta_deg": 0.10,
        },
    ]

    metrics = _render_lock_diagnostics_from_rows(rows)

    assert metrics["median_render_translation_error_m"] == pytest.approx(0.375)
    assert metrics["median_abs_pnp_minus_render_translation_error_m"] == pytest.approx(0.2035)
    assert metrics["locked_to_render_within_3cm_rate"] == pytest.approx(0.5)
    assert metrics["median_pnp_render_translation_delta_m"] == pytest.approx(0.2005)


def test_radio_dual_feature_mode_defaults_to_radio_dual_layer() -> None:
    args = parse_args(_base_args() + ["--feature_mode", "radio_dual"])

    assert args.layer_name == "radio_dual"


def test_reference_top5_render_pose_mode_is_exposed() -> None:
    args = parse_args(_base_args() + ["--render_pose_mode", "reference_top5", "--candidate_bank", "bank.jsonl"])

    assert args.render_pose_mode == "reference_top5"
    assert args.reference_top_k == 5


def test_report_perturbation_adapter_eval_writes_key_metrics(tmp_path) -> None:
    summary_path = tmp_path / "gt" / "summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text(
        """
{
  "stage": "render_rgb_feature_keypoint_pose",
  "inputs": {"matcha_joint_checkpoint": "model.pt"},
  "config": {"render_pose_mode": "gt", "pose_update_iterations": 1, "match_mode": "matcha_c2f"},
  "metrics": {"median_translation_error_m": 0.12, "success_25cm_10deg": 0.8, "match_gt_16px": 0.7}
}
""".strip()
        + "\n"
    )
    output_csv = tmp_path / "report.csv"
    output_json = tmp_path / "report.json"

    report_perturbation_adapter_eval(
        [
            "--summaries",
            str(summary_path),
            "--output_csv",
            str(output_csv),
            "--output_json",
            str(output_json),
        ]
    )

    csv_text = output_csv.read_text()
    json_text = output_json.read_text()
    assert "median_translation_error_m" in csv_text
    assert "success_25cm_10deg" in csv_text
    assert "model.pt" in csv_text
    assert "\"render_pose_mode\": \"gt\"" in json_text


def test_guarded_pose_update_keeps_initial_until_score_margin_is_met() -> None:
    initial = {"status": "ok", "iteration_pose_score": SimpleNamespace(score=1.0), "iteration_index": 0}
    weak_update = {"status": "ok", "iteration_pose_score": SimpleNamespace(score=1.02), "iteration_index": 1}
    strong_update = {"status": "ok", "iteration_pose_score": SimpleNamespace(score=1.04), "iteration_index": 1}

    selected, label = _select_pose_update_iteration(
        [initial, weak_update],
        mode="guarded_score",
        score_margin=0.03,
    )
    assert selected is initial
    assert label == "guarded_score:kept_initial"

    selected, label = _select_pose_update_iteration(
        [initial, strong_update],
        mode="guarded_score",
        score_margin=0.03,
    )
    assert selected is strong_update
    assert label == "guarded_score:accepted"
