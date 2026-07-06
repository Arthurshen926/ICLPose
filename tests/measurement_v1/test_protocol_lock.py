from __future__ import annotations

import csv
import json
from pathlib import Path

from feature_extract.vfm.measurement_v1.protocol_lock import build_protocol_lock_report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_protocol_lock_replays_topk_from_existing_pose_candidate_table(tmp_path: Path) -> None:
    eval_dir = tmp_path / "reference_top10"
    eval_dir.mkdir()
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {"render_pose_mode": "reference_top10", "enable_pose_rescore": False},
                "inputs": {"matcha_joint_checkpoint": "ckpt.pt", "query_manifest": "test182.json"},
                "metrics": {"query_count": 2},
            }
        )
        + "\n"
    )
    _write_csv(eval_dir / "rows.csv", [{"query_id": "q0"}, {"query_id": "q1"}])
    _write_csv(
        eval_dir / "match_table.csv",
        [
            {
                "query_id": "q0",
                "match_index": 0,
                "render_index": 12,
                "query_x": 10.0,
                "query_y": 20.0,
                "render_x": 11.0,
                "render_y": 21.0,
                "world_x": 1.0,
                "world_y": 2.0,
                "world_z": 3.0,
                "confidence": 0.7,
                "render_alpha": 1.0,
                "gt_reproj_error_px": 2.5,
                "gt_correct_5px": True,
                "gt_correct_10px": True,
                "pnp_inlier": True,
            }
        ],
    )
    _write_csv(
        eval_dir / "pose_candidate_table.csv",
        [
            {
                "query_id": "q0",
                "candidate_rank": 0,
                "translation_error_m": 0.30,
                "rotation_error_deg": 1.0,
                "render_translation_error_m": 0.50,
                "render_rotation_error_deg": 3.0,
                "pose_score": 1.0,
            },
            {
                "query_id": "q0",
                "candidate_rank": 4,
                "translation_error_m": 0.05,
                "rotation_error_deg": 0.2,
                "render_translation_error_m": 0.20,
                "render_rotation_error_deg": 2.0,
                "pose_score": 0.1,
            },
            {
                "query_id": "q1",
                "candidate_rank": 0,
                "translation_error_m": 0.40,
                "rotation_error_deg": 2.0,
                "render_translation_error_m": 0.60,
                "render_rotation_error_deg": 3.0,
                "pose_score": 0.2,
            },
            {
                "query_id": "q1",
                "candidate_rank": 2,
                "translation_error_m": 0.10,
                "rotation_error_deg": 0.5,
                "render_translation_error_m": 0.15,
                "render_rotation_error_deg": 1.0,
                "pose_score": 2.0,
            },
        ],
    )

    report = build_protocol_lock_report(eval_dir=eval_dir, output_dir=tmp_path / "lock", topks=(1, 3, 5, 10))
    repeated = build_protocol_lock_report(eval_dir=eval_dir, output_dir=tmp_path / "lock2", topks=(1, 3, 5, 10))

    assert report["stage"] == "measurement_v1_protocol_lock"
    assert report["deterministic_replay"]["top5"]["oracle_solver_best"]["median_translation_error_m"] == 0.07500000000000001
    assert report["deterministic_replay"]["top3"]["score_selected"]["median_translation_error_m"] == 0.2
    assert report["deterministic_replay"]["top1"]["rank0"]["median_translation_error_m"] == 0.35
    assert report["protocol"]["scorer_enabled"] is False
    assert report["artifacts"]["pose_candidate_table"]["row_count"] == 4
    assert report["artifacts"]["query_summary"]["row_count"] == 2
    assert report["artifacts"]["measurement_table"]["row_count"] == 1
    assert (tmp_path / "lock" / "query_summary.csv").exists()
    assert (tmp_path / "lock" / "measurement_table.csv").exists()
    assert report["deterministic_replay"] == repeated["deterministic_replay"]
