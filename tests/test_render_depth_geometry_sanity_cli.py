from __future__ import annotations

from pathlib import Path

import pytest

from feature_extract.tools.vfm.eval_render_depth_geometry_sanity import main


def test_render_depth_geometry_sanity_dry_run_lists_solver_matrix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            str(tmp_path),
            "--max_queries",
            "2",
            "--solvers",
            "plain,ransac,magsac,weighted,covariance,oracle_uncertainty",
            "--dry_run",
        ]
    )

    out = capsys.readouterr().out
    assert "eval_render_depth_geometry_sanity" in out
    assert "--render_pose_mode gt" in out
    assert "plain,ransac,magsac,weighted,covariance,oracle_uncertainty" in out
    assert "--sample_grid 32x18" in out
