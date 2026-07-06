from __future__ import annotations

import csv
from pathlib import Path

from feature_extract.tools.vfm.build_measurement_policy_rows import parse_args
from feature_extract.vfm.measurement_v1.policy_rows import build_measurement_policy_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_measurement_policy_rows_preserves_micro_and_uses_valid_teacher(tmp_path: Path) -> None:
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q.png",
                "support_image_id": "s.png",
                "track_id": 1,
                "center_x": 10.0,
                "center_y": 12.0,
                "query_gt_x": 10.5,
                "query_gt_y": 12.0,
                "requested_residual_px": 0.5,
                "target_is_dustbin": "False",
            },
            {
                "query_id": "q.png",
                "support_image_id": "s.png",
                "track_id": 2,
                "center_x": 20.0,
                "center_y": 22.0,
                "query_gt_x": 22.0,
                "query_gt_y": 22.0,
                "requested_residual_px": 2.0,
                "target_is_dustbin": "False",
            },
            {
                "query_id": "q.png",
                "support_image_id": "s.png",
                "track_id": 3,
                "center_x": 30.0,
                "center_y": 32.0,
                "query_gt_x": 33.0,
                "query_gt_y": 32.0,
                "requested_residual_px": 3.0,
                "target_is_dustbin": "False",
            },
        ],
    )
    teacher_csv = tmp_path / "teacher.csv"
    _write_csv(
        teacher_csv,
        [
            {
                "row_index": 0,
                "lk_applied": "True",
                "lk_epe_px": 0.2,
                "lk_pred_x": 10.2,
                "lk_pred_y": 12.0,
                "lk_reason": "applied",
            },
            {
                "row_index": 1,
                "lk_applied": "True",
                "lk_epe_px": 0.4,
                "lk_pred_x": 21.7,
                "lk_pred_y": 22.1,
                "lk_reason": "applied",
            },
            {
                "row_index": 2,
                "lk_applied": "False",
                "lk_epe_px": "",
                "lk_pred_x": "",
                "lk_pred_y": "",
                "lk_reason": "high_fb_error",
            },
        ],
    )
    output_csv = tmp_path / "policy.csv"

    summary = build_measurement_policy_rows(
        rows_csv=rows_csv,
        dense_teacher_rows_csv=teacher_csv,
        output_rows_csv=output_csv,
        center_preserve_below_px=0.5,
        teacher_valid_max_epe_px=1.0,
        fallback_loss_weight=0.25,
    )

    rows = list(csv.DictReader(output_csv.open()))
    assert summary["policy_counts"] == {
        "center_preserve": 1,
        "gt_fallback": 1,
        "teacher_correction": 1,
    }
    assert rows[0]["measurement_policy"] == "center_preserve"
    assert float(rows[0]["policy_target_x"]) == 10.0
    assert float(rows[0]["policy_target_y"]) == 12.0
    assert rows[0]["policy_target_source"] == "center"
    assert float(rows[0]["policy_loss_weight"]) == 1.0

    assert rows[1]["measurement_policy"] == "teacher_correction"
    assert float(rows[1]["policy_target_x"]) == 21.7
    assert float(rows[1]["policy_target_y"]) == 22.1
    assert rows[1]["teacher_valid"] == "True"

    assert rows[2]["measurement_policy"] == "gt_fallback"
    assert float(rows[2]["policy_target_x"]) == 33.0
    assert float(rows[2]["policy_target_y"]) == 32.0
    assert rows[2]["teacher_valid"] == "False"
    assert rows[2]["teacher_reason"] == "high_fb_error"
    assert float(rows[2]["policy_loss_weight"]) == 0.25


def test_build_measurement_policy_rows_rejects_mismatched_teacher_indices(tmp_path: Path) -> None:
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "q.png",
                "center_x": 10.0,
                "center_y": 12.0,
                "query_gt_x": 11.0,
                "query_gt_y": 12.0,
                "requested_residual_px": 1.0,
            }
        ],
    )
    teacher_csv = tmp_path / "teacher.csv"
    _write_csv(
        teacher_csv,
        [{"row_index": 7, "lk_applied": "False", "lk_reason": "missing"}],
    )

    try:
        build_measurement_policy_rows(
            rows_csv=rows_csv,
            dense_teacher_rows_csv=teacher_csv,
            output_rows_csv=tmp_path / "out.csv",
        )
    except ValueError as exc:
        assert "missing dense teacher row" in str(exc)
    else:
        raise AssertionError("expected mismatched teacher rows to fail")


def test_build_measurement_policy_rows_cli_parses_thresholds() -> None:
    args = parse_args(
        [
            "--rows_csv",
            "rows.csv",
            "--dense_teacher_rows_csv",
            "teacher.csv",
            "--output_rows_csv",
            "policy.csv",
            "--center_preserve_below_px",
            "0.25",
            "--teacher_valid_max_epe_px",
            "0.75",
            "--fallback_loss_weight",
            "0.1",
        ]
    )

    assert args.rows_csv == "rows.csv"
    assert args.dense_teacher_rows_csv == "teacher.csv"
    assert args.output_rows_csv == "policy.csv"
    assert args.center_preserve_below_px == 0.25
    assert args.teacher_valid_max_epe_px == 0.75
    assert args.fallback_loss_weight == 0.1
