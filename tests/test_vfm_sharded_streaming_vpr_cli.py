from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    CambridgePoseRecord,
    pose_w2c_from_center_rotation,
    write_cambridge_pose_file,
)


def _record(image_id: str, center: tuple[float, float, float]) -> CambridgePoseRecord:
    rotation = np.eye(3, dtype=np.float64)
    camera_center = np.asarray(center, dtype=np.float64)
    return CambridgePoseRecord(
        image_id=image_id,
        camera_center=camera_center,
        rotation_w2c=rotation,
        pose_w2c=pose_w2c_from_center_rotation(camera_center, rotation),
    )


def test_sharded_streaming_vpr_cli_can_dry_run_grid_count(tmp_path) -> None:
    reference_pose_file = tmp_path / "refs.txt"
    query_pose_file = tmp_path / "queries.txt"
    summary = tmp_path / "summary.json"
    write_cambridge_pose_file([_record("ref.png", (0.0, 2.0, 0.0))], reference_pose_file)
    write_cambridge_pose_file([_record("q.png", (0.0, 2.0, 0.0))], query_pose_file)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.run_sharded_2dgs_vpr_verifier",
            "--reference_pose_file",
            str(reference_pose_file),
            "--query_pose_file",
            str(query_pose_file),
            "--summary_json",
            str(summary),
            "--grid_step_m",
            "1.0",
            "--grid_margin_m",
            "0.0",
            "--grid_height_offsets_m",
            "0,0.5",
            "--grid_orientation_knn",
            "1",
            "--yaw_offsets_deg=-5,0",
            "--dry_run_count_only",
        ],
        check=True,
    )

    payload = json.loads(summary.read_text())
    assert payload["dry_run_count_only"] is True
    assert payload["estimated_pose_count"] == 4
