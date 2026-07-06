from __future__ import annotations

from pathlib import Path

from feature_extract.tools.vfm.run_fixed_anchor_coarse_only_basin_sweep import build_basin_jobs, parse_args, summarize


def _args():
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
            "out",
            "--matcha_joint_checkpoint",
            "model.pt",
            "--translation_magnitudes_m",
            "0,0.05",
            "--rotation_magnitudes_deg",
            "0,1",
            "--translation_directions",
            "xpos,zneg",
            "--rotation_axes",
            "y",
        ]
    )


def test_basin_sweep_builds_directional_translation_and_rotation_jobs() -> None:
    jobs = build_basin_jobs(_args())
    labels = [label for label, _cmd in jobs]

    assert labels == ["gt", "trans_xpos_0p050m", "trans_zneg_0p050m", "rot_ypos_1p000deg", "rot_yneg_1p000deg"]
    joined = " ".join(jobs[1][1])
    assert "--render_pose_mode gt_offset" in joined
    assert "--render_pose_world_offset=0.05,0,0" in joined
    joined_neg = " ".join(jobs[2][1])
    assert "--render_pose_world_offset=0,0,-0.05" in joined_neg
    joined_rot = " ".join(jobs[3][1])
    assert "--render_pose_mode gt_rotation_offset" in joined_rot
    assert "--render_pose_rotation_offset_deg=0,1,0" in joined_rot
    joined_rot_neg = " ".join(jobs[4][1])
    assert "--render_pose_rotation_offset_deg=0,-1,0" in joined_rot_neg


def test_basin_sweep_label_filter_is_applied() -> None:
    args = _args()
    args.labels = "trans_zneg_0p050m"

    jobs = build_basin_jobs(args)

    assert [label for label, _cmd in jobs] == ["trans_zneg_0p050m"]


def _write_eval_output(root: Path, label: str, rows: list[dict[str, object]]) -> None:
    import csv
    import json

    out = root / label
    out.mkdir(parents=True)
    with (out / "rows.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(
        json.dumps({"metrics": {"query_count": len(rows), "median_translation_error_m": 0.0}}) + "\n"
    )


def test_basin_sweep_summary_reports_requested_and_realized_protocol_gates(tmp_path: Path) -> None:
    _write_eval_output(
        tmp_path,
        "rot_ypos_1p000deg",
        [
            {
                "query_id": "q0",
                "realized_initial_translation_error_m": 0.0,
                "realized_initial_rotation_error_deg": 1.0,
                "requested_translation_m": 0.0,
                "requested_rotation_deg": 1.0,
            }
        ],
    )
    _write_eval_output(
        tmp_path,
        "trans_xpos_0p050m",
        [
            {
                "query_id": "q0",
                "realized_initial_translation_error_m": 0.05,
                "realized_initial_rotation_error_deg": 0.0,
                "requested_translation_m": 0.05,
                "requested_rotation_deg": 0.0,
            }
        ],
    )

    report = summarize(tmp_path, ["rot_ypos_1p000deg", "trans_xpos_0p050m"])
    by_label = {row["label"]: row for row in report["rows"]}

    assert by_label["rot_ypos_1p000deg"]["protocol_pure_rotation_center_pass"] is True
    assert by_label["rot_ypos_1p000deg"]["max_realized_initial_translation_error_m"] == 0.0
    assert by_label["trans_xpos_0p050m"]["protocol_pure_translation_rotation_pass"] is True
    assert by_label["trans_xpos_0p050m"]["max_realized_initial_rotation_error_deg"] == 0.0
