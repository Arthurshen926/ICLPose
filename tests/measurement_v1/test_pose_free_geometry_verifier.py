from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.pose_free_geometry_verifier import (
    FEATURE_SETS,
    apply_pose_free_geometry_verifier,
    build_pose_free_geometry_examples,
    fit_pose_free_geometry_verifier,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dataset(
    root: Path,
    *,
    prefix: str,
    query_count: int,
    selector_sha256: str = "selector-shared",
) -> tuple[Path, Path, Path]:
    source_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    policy_row = 0
    for query_index in range(query_count):
        query_id = f"{prefix}_query_{query_index:02d}"
        for assignment_index in range(4):
            correct = assignment_index % 2 == 0
            residual = 1.0 if correct else 9.0
            for view_index, selector_probability in enumerate((0.7, 0.3)):
                row_index = len(source_rows)
                source_rows.append(
                    {
                        "query_id": query_id,
                        "track_id": 1000 + policy_row,
                        "support_track_id": 1000 + policy_row,
                        "policy_row_index": policy_row,
                        "support_view_probability": 0.6 if view_index == 0 else 0.4,
                        "center_x": 10.0,
                        "center_y": 12.0,
                        "target_gt_projected_x": 10.0 + residual,
                        "target_gt_projected_y": 12.0,
                        "target_gt_projected_residual_px": residual,
                        "target_gt_projection_in_front": True,
                        "target_gt_projection_in_image": True,
                        "assignment_score": 0.99 if not correct else 0.01,
                        "pose_selection_score": 0.99 if not correct else 0.01,
                        "geometry_p01": 0.99 if not correct else 0.01,
                        "geometry_p02": 0.99 if not correct else 0.01,
                        "geometry_p05": 0.99 if not correct else 0.01,
                        "track_length": 8 if correct else 3,
                        "query_reprojection_error": 0.1 if not correct else 5.0,
                        "support_reprojection_error": 0.3 if correct else 2.0,
                        "support_view_angle_deg": 10.0 if correct else 70.0,
                        "switched_from_baseline": not correct,
                    }
                )
                diagnostic_rows.append(
                    {
                        "row_index": row_index,
                        "query_id": query_id,
                        "track_id": 1000 + policy_row,
                        "support_track_id": 1000 + policy_row,
                        "policy_row_index": policy_row,
                        "pred_dx": 0.2 if correct else 4.0,
                        "pred_dy": 0.1 if correct else -3.0,
                        "peak_dx": 0.25 if correct else -4.0,
                        "peak_dy": 0.1 if correct else 3.0,
                        "dustbin_probability": 0.1 if correct else 0.9,
                        "likelihood_normalized_entropy": 0.2 if correct else 0.9,
                        "likelihood_peak_probability": 0.7 if correct else 0.05,
                        "likelihood_peak_margin": 0.5 if correct else 0.01,
                        "likelihood_covariance_max_sigma_px": 0.4 if correct else 5.0,
                    }
                )
                selector_rows.append(
                    {
                        "diagnostic_row_index": row_index,
                        "policy_row_index": policy_row,
                        "query_id": query_id,
                        "track_id": 1000 + policy_row,
                        "support_view_rank": view_index + 1,
                        "selector_score": 1.0 - view_index,
                        "selector_probability": selector_probability,
                    }
                )
            policy_row += 1

    source_path = root / prefix / "rows.csv"
    diagnostics_path = root / prefix / "diagnostics" / "diagnostic_rows.csv"
    selector_path = root / prefix / "selector" / "selector_rows.csv"
    _write_csv(source_path, source_rows)
    _write_csv(diagnostics_path, diagnostic_rows)
    _write_csv(selector_path, selector_rows)
    diagnostic_summary = {
        "stage": "measurement_v1_rgb_patch_diagnostics",
        "checkpoint": "measurement.pt",
        "checkpoint_sha256": "measurement-shared",
        "rows_csv_sha256": file_sha256_short(source_path),
        "outputs": {
            "diagnostic_rows_sha256": file_sha256_short(diagnostics_path)
        },
    }
    (diagnostics_path.parent / "summary.json").write_text(
        json.dumps(diagnostic_summary)
    )
    selector_summary = {
        "stage": "measurement_support_selector_apply",
        "inputs": {"selector_sha256": selector_sha256},
        "outputs": {
            "predictions": str(selector_path),
            "predictions_sha256": file_sha256_short(selector_path),
        },
    }
    (selector_path.parent / "summary.json").write_text(json.dumps(selector_summary))
    return source_path, diagnostics_path, selector_path


def test_pose_free_feature_schemas_exclude_pose_and_target_context() -> None:
    forbidden = (
        "target",
        "gt_",
        "pose_",
        "assignment_score",
        "geometry_p",
        "query_reprojection",
    )
    for feature_names in FEATURE_SETS.values():
        assert not any(
            fragment in name for name in feature_names for fragment in forbidden
        )


def test_fit_and_apply_pose_free_geometry_verifier(tmp_path: Path) -> None:
    train = _dataset(tmp_path, prefix="train", query_count=8)
    validation = _dataset(tmp_path, prefix="validation", query_count=4)
    output = tmp_path / "fit"

    summary = fit_pose_free_geometry_verifier(
        train_rows_csv=train[0],
        train_diagnostics_csv=train[1],
        train_support_selector_rows_csv=train[2],
        validation_rows_csv=validation[0],
        validation_diagnostics_csv=validation[1],
        validation_support_selector_rows_csv=validation[2],
        output_dir=output,
        c_values=(0.1, 1.0),
        fold_count=4,
    )

    assert summary["protocol"]["query_pose_used_as_feature"] is False
    assert summary["protocol"]["GT_pose_residual_used_as_target_only"] is True
    assert summary["feature_sets"]["measurement_core"]["validation_metrics"][
        "auroc"
    ] == 1.0
    assert summary["promotion_gate"]["passes"] is True
    model = output / "measurement_plus_support_verifier.json"

    apply_summary = apply_pose_free_geometry_verifier(
        rows_csv=validation[0],
        diagnostics_csv=validation[1],
        support_selector_rows_csv=validation[2],
        verifier_json=model,
        output_dir=tmp_path / "apply",
    )
    assert apply_summary["protocol"]["threshold_search"] is False
    assert apply_summary["metrics_TARGET_ONLY"]["auroc"] == 1.0
    assert (tmp_path / "apply" / "geometry_predictions.csv").exists()


def test_pose_free_verifier_rejects_selector_identity_mismatch(
    tmp_path: Path,
) -> None:
    train = _dataset(tmp_path, prefix="train", query_count=6)
    validation = _dataset(
        tmp_path,
        prefix="validation",
        query_count=3,
        selector_sha256="different-selector",
    )

    with pytest.raises(ValueError, match="support selector mismatch"):
        fit_pose_free_geometry_verifier(
            train_rows_csv=train[0],
            train_diagnostics_csv=train[1],
            train_support_selector_rows_csv=train[2],
            validation_rows_csv=validation[0],
            validation_diagnostics_csv=validation[1],
            validation_support_selector_rows_csv=validation[2],
            output_dir=tmp_path / "fit",
            fold_count=3,
        )


def test_pose_free_examples_ignore_adversarial_pose_features(tmp_path: Path) -> None:
    rows, diagnostics, selector = _dataset(
        tmp_path, prefix="audit", query_count=2
    )
    examples = build_pose_free_geometry_examples(
        source_rows_csv=rows,
        diagnostic_rows_csv=diagnostics,
        measurement_support_selector_rows_csv=selector,
    )

    assert examples
    assert "assignment_score" not in examples[0]["features"]
    assert "geometry_p05" not in examples[0]["features"]
    assert "query_reprojection_quality" not in examples[0]["features"]


def test_fit_writes_oof_probabilities_for_rows_without_valid_gt_projection(
    tmp_path: Path,
) -> None:
    train = _dataset(tmp_path, prefix="train", query_count=6)
    validation = _dataset(tmp_path, prefix="validation", query_count=3)
    rows = list(csv.DictReader(train[0].open(newline="")))
    rows[0]["target_gt_projected_x"] = ""
    rows[0]["target_gt_projected_y"] = ""
    rows[0]["target_gt_projected_residual_px"] = ""
    rows[0]["target_gt_projection_in_front"] = False
    rows[0]["target_gt_projection_in_image"] = False
    _write_csv(train[0], rows)
    diagnostic_summary_path = train[1].parent / "summary.json"
    diagnostic_summary = json.loads(diagnostic_summary_path.read_text())
    diagnostic_summary["rows_csv_sha256"] = file_sha256_short(train[0])
    diagnostic_summary_path.write_text(json.dumps(diagnostic_summary))

    output = tmp_path / "fit"
    fit_pose_free_geometry_verifier(
        train_rows_csv=train[0],
        train_diagnostics_csv=train[1],
        train_support_selector_rows_csv=train[2],
        validation_rows_csv=validation[0],
        validation_diagnostics_csv=validation[1],
        validation_support_selector_rows_csv=validation[2],
        output_dir=output,
        c_values=(0.1,),
        fold_count=3,
    )
    prediction_rows = list(
        csv.DictReader((output / "train_oof_predictions.csv").open(newline=""))
    )
    assert prediction_rows[0]["measurement_core_geometry_probability"] != ""
