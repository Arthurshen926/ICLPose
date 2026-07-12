from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from feature_extract.vfm.measurement_v1.action_calibration import (
    ACTION_FEATURE_NAMES,
    apply_measurement_action_calibrator,
    build_action_examples,
    fit_measurement_action_calibrator,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows(group_count: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for group in range(group_count):
        target_x = 1.0 if group % 2 == 0 else 8.0
        for view, probability in enumerate((0.75, 0.25)):
            row_index = len(source)
            source.append(
                {
                    "query_id": f"query_{group}",
                    "track_id": group + 10,
                    "support_track_id": group + 10,
                    "policy_row_index": group,
                    "support_view_probability": probability,
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "target_gt_projected_x": target_x,
                    "target_gt_projected_y": 0.0,
                    "target_gt_projected_residual_px": target_x,
                    "target_gt_projection_in_front": True,
                    "target_gt_projection_in_image": True,
                    "assignment_score": 0.8 - 0.03 * group,
                    "pose_selection_score": 0.7 - 0.02 * group,
                    "geometry_p01": 0.8 if target_x <= 1.0 else 0.1,
                    "geometry_p02": 0.9 if target_x <= 2.0 else 0.2,
                    "geometry_p05": 0.95 if target_x <= 5.0 else 0.3,
                    "track_length": 8,
                    "query_reprojection_error": 0.4,
                    "support_reprojection_error": 0.5,
                    "support_view_angle_deg": 15.0 + view,
                    "switched_from_baseline": False,
                }
            )
            diagnostics.append(
                {
                    "row_index": row_index,
                    "query_id": f"query_{group}",
                    "track_id": group + 10,
                    "support_track_id": group + 10,
                    "pred_dx": 0.8 if view == 0 else 0.0,
                    "pred_dy": 0.0,
                    "dustbin_probability": 0.1 + 0.05 * (group % 2),
                    "likelihood_normalized_entropy": 0.2 + 0.1 * view,
                    "likelihood_peak_probability": 0.7 - 0.1 * view,
                    "likelihood_peak_margin": 0.4 - 0.1 * view,
                    "likelihood_covariance_max_sigma_px": 0.5 + 0.1 * view,
                }
            )
    return source, diagnostics


def test_build_action_examples_fuses_support_views_without_target_features(
    tmp_path: Path,
) -> None:
    source, diagnostics = _rows(2)
    source_csv = tmp_path / "source.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    _write_csv(source_csv, source)
    _write_csv(diagnostics_csv, diagnostics)

    examples = build_action_examples(
        source_rows_csv=source_csv,
        diagnostic_rows_csv=diagnostics_csv,
    )

    assert len(examples) == 2
    assert np.isclose(examples[0]["updated_x"], 0.6)
    assert np.isclose(examples[0]["updated_residual_px"], 0.4)
    assert examples[0]["target_update_beneficial"] is True
    assert examples[1]["target_geometry_correct_5px"] is False
    assert set(examples[0]["features"]) == set(ACTION_FEATURE_NAMES)
    assert not any("target" in name for name in ACTION_FEATURE_NAMES)


def test_build_action_examples_accepts_source_indexed_diagnostic_subset(
    tmp_path: Path,
) -> None:
    source, diagnostics = _rows(3)
    source_csv = tmp_path / "source.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    _write_csv(source_csv, source)
    _write_csv(
        diagnostics_csv,
        [row for row in diagnostics if int(row["row_index"]) >= 2],
    )

    examples = build_action_examples(
        source_rows_csv=source_csv,
        diagnostic_rows_csv=diagnostics_csv,
    )

    assert [example["policy_row_index"] for example in examples] == [1, 2]


def test_build_action_examples_marks_missing_projection_as_unsafe(
    tmp_path: Path,
) -> None:
    source, diagnostics = _rows(1)
    for row in source:
        row["target_gt_projected_x"] = ""
        row["target_gt_projected_y"] = ""
        row["target_gt_projected_residual_px"] = ""
        row["target_gt_projection_in_front"] = False
        row["target_gt_projection_in_image"] = False
    source_csv = tmp_path / "source.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    _write_csv(source_csv, source)
    _write_csv(diagnostics_csv, diagnostics)

    example = build_action_examples(
        source_rows_csv=source_csv,
        diagnostic_rows_csv=diagnostics_csv,
    )[0]

    assert example["has_valid_geometry_target"] is False
    assert np.isinf(example["baseline_residual_px"])
    assert example["target_geometry_correct_5px"] is False
    assert example["target_update_beneficial"] is False
    assert example["target_update_safe"] is False


def test_build_action_examples_uses_no_gt_coarse_pose_context(
    tmp_path: Path,
) -> None:
    source, diagnostics = _rows(1)
    source_csv = tmp_path / "source.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    context_csv = tmp_path / "coarse_pose_context.csv"
    _write_csv(source_csv, source)
    _write_csv(diagnostics_csv, diagnostics)
    _write_csv(
        context_csv,
        [
            {
                "policy_row_index": 0,
                "query_id": "query_0",
                "track_id": 10,
                "coarse_pose_success": True,
                "coarse_pose_inlier": True,
                "coarse_pose_projection_in_front": True,
                "coarse_pose_query_inlier_ratio": 0.75,
                "coarse_pose_offset_dx": 0.5,
                "coarse_pose_offset_dy": 0.0,
                "coarse_pose_reprojection_residual_px": 0.5,
            }
        ],
    )

    example = build_action_examples(
        source_rows_csv=source_csv,
        diagnostic_rows_csv=diagnostics_csv,
        coarse_pose_context_csv=context_csv,
    )[0]

    assert example["coarse_pose_success"] is True
    assert example["coarse_pose_inlier"] is True
    assert example["features"]["coarse_pose_center_le_1px"] == 1.0
    assert example["features"]["coarse_pose_center_1_to_5px"] == 0.0
    assert np.isclose(
        example["features"]["measurement_pose_offset_cosine"], 1.0
    )


def test_fit_action_calibrator_writes_validation_only_policy(tmp_path: Path) -> None:
    train_source, train_diagnostics = _rows(8)
    validation_source, validation_diagnostics = _rows(6)
    paths = {
        "train_rows": tmp_path / "train_rows.csv",
        "train_diagnostics": tmp_path / "train_diagnostics.csv",
        "validation_rows": tmp_path / "validation_rows.csv",
        "validation_diagnostics": tmp_path / "validation_diagnostics.csv",
    }
    _write_csv(paths["train_rows"], train_source)
    _write_csv(paths["train_diagnostics"], train_diagnostics)
    _write_csv(paths["validation_rows"], validation_source)
    _write_csv(paths["validation_diagnostics"], validation_diagnostics)

    summary = fit_measurement_action_calibrator(
        train_rows_csv=paths["train_rows"],
        train_diagnostics_csv=paths["train_diagnostics"],
        validation_rows_csv=paths["validation_rows"],
        validation_diagnostics_csv=paths["validation_diagnostics"],
        output_dir=tmp_path / "output",
    )

    assert summary["protocol"]["fit_split"] == "train"
    assert summary["protocol"]["threshold_selection_split"] == "validation"
    assert summary["protocol"]["test_used"] is False
    assert summary["target_fields_excluded_from_features"] is True
    assert (tmp_path / "output" / "measurement_action_calibrator.json").exists()
    assert (tmp_path / "output" / "validation_action_predictions.csv").exists()

    apply_summary = apply_measurement_action_calibrator(
        rows_csv=paths["validation_rows"],
        diagnostics_csv=paths["validation_diagnostics"],
        calibrator_json=tmp_path / "output" / "measurement_action_calibrator.json",
        output_dir=tmp_path / "applied",
    )
    assert apply_summary["protocol"]["threshold_search"] is False
    assert apply_summary["protocol"]["target_fields_used_as_features"] is False
    assert sum(apply_summary["action_counts"].values()) == 6
