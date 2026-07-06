from __future__ import annotations

import csv
import json
from pathlib import Path

from feature_extract.tools.vfm.run_fixed_anchor_identity_shortcut_audit import (
    build_audit_jobs,
    parse_args,
    summarize,
)


def _args(tmp_path: Path):
    return parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--query_pose_file",
            "poses.txt",
            "--image_root",
            "images",
            "--gaussian_rgb_ply",
            "scene.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
        ]
    )


def test_identity_shortcut_audit_builds_full_control_labels(tmp_path: Path) -> None:
    jobs = build_audit_jobs(_args(tmp_path))
    labels = [label for label, _cmd in jobs]

    assert labels[:5] == ["learned_gt", "same_cell_gt", "constant_gt", "random_gt", "spatial_shuffle_gt"]
    assert "shift_xpos_1cell_gt" in labels
    assert "shift_xneg_4cell_gt" in labels
    assert "shift_ypos_2cell_gt" in labels
    assert "shift_yneg_4cell_gt" in labels
    assert len(labels) == 17
    joined = " ".join(dict(jobs)["shift_xneg_4cell_gt"])
    assert "--coarse_control_mode shift_query_cells" in joined
    assert "--coarse_control_shift_cells=-4,0" in joined


def test_identity_shortcut_audit_label_filter_is_applied(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.labels = "constant_gt,shift_ypos_1cell_gt"

    jobs = build_audit_jobs(args)

    assert [label for label, _cmd in jobs] == ["constant_gt", "shift_ypos_1cell_gt"]


def test_identity_shortcut_audit_summary_computes_shift_slope(tmp_path: Path) -> None:
    label_dir = tmp_path / "shift_xpos_2cell_gt"
    label_dir.mkdir(parents=True)
    (label_dir / "summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "query_count": 2,
                    "median_translation_error_m": 0.1,
                    "median_rotation_error_deg": 0.2,
                    "success_10cm_5deg": 0.5,
                    "pnp_solve_rate": 1.0,
                }
            }
        )
        + "\n"
    )
    with (label_dir / "rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "translation_error_m",
                "rotation_error_deg",
                "coarse_same_cell_fraction",
                "coarse_mean_cell_delta_x",
                "coarse_mean_cell_delta_y",
                "coarse_control_shift_dx_cells",
                "coarse_control_shift_dy_cells",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "translation_error_m": "0.02",
                "rotation_error_deg": "0.5",
                "coarse_same_cell_fraction": "0.1",
                "coarse_mean_cell_delta_x": "-1.8",
                "coarse_mean_cell_delta_y": "0.0",
                "coarse_control_shift_dx_cells": "2",
                "coarse_control_shift_dy_cells": "0",
            }
        )

    report = summarize(tmp_path, ["shift_xpos_2cell_gt"])

    row = report["rows"][0]
    assert row["success_3cm_1deg"] == 1.0
    assert row["shift_tracking_slope_x"] == 0.9
    assert row["shift_tracking_slope_y"] is None
