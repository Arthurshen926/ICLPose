import json
import subprocess
import sys

import pytest

from feature_extract.vfm.virtual_pose_coverage import (
    summarize_pose_file_oracle_coverage,
    summarize_virtual_grid_oracle_coverage,
)


def _write_pose_file(path, lines):
    path.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                *lines,
            ]
        )
        + "\n"
    )


def test_pose_file_oracle_coverage_reports_candidate_existence(tmp_path):
    queries = tmp_path / "queries.txt"
    references = tmp_path / "references.txt"
    _write_pose_file(queries, ["q0.png 0.10 0.0 0.0 1.0 0.0 0.0 0.0"])
    _write_pose_file(
        references,
        [
            "r_bad.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
            "r_good.png 0.2 0.0 0.0 1.0 0.0 0.0 0.0",
        ],
    )

    summary = summarize_pose_file_oracle_coverage(
        query_pose_file=queries,
        reference_pose_file=references,
        thresholds=((0.25, 5.0),),
    )

    assert summary["query_count"] == 1
    assert summary["reference_count"] == 2
    assert summary["exist_recall_25cm_5deg"] == pytest.approx(1.0)
    assert summary["min_translation_m_median"] == pytest.approx(0.1)


def test_virtual_grid_oracle_coverage_checks_grid_without_query_gt_generation(tmp_path):
    references = tmp_path / "references.txt"
    queries = tmp_path / "queries.txt"
    _write_pose_file(
        references,
        [
            "r0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
            "r1.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
        ],
    )
    _write_pose_file(queries, ["q0.png 0.5 0.0 0.0 1.0 0.0 0.0 0.0"])

    summary = summarize_virtual_grid_oracle_coverage(
        reference_pose_file=references,
        query_pose_file=queries,
        grid_step_m=0.25,
        yaw_offsets_deg=(0.0,),
        height_offsets_m=(0.0,),
        orientation_knn=1,
        thresholds=((0.25, 5.0),),
    )

    assert summary["query_count"] == 1
    assert summary["grid_step_m"] == pytest.approx(0.25)
    assert summary["orientation_knn"] == 1
    assert summary["exist_recall_25cm_5deg"] == pytest.approx(1.0)
    assert summary["best_joint_translation_m_median"] == pytest.approx(0.0)


def test_virtual_grid_oracle_coverage_cli_writes_summary(tmp_path):
    references = tmp_path / "references.txt"
    queries = tmp_path / "queries.txt"
    output = tmp_path / "coverage.json"
    _write_pose_file(references, ["r0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0"])
    _write_pose_file(queries, ["q0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0"])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.benchmark_virtual_pose_coverage",
            "--query_pose_file",
            str(queries),
            "--reference_pose_file",
            str(references),
            "--mode",
            "grid",
            "--grid_step_m",
            "0.25",
            "--thresholds",
            "0.25,5;0.5,10",
            "--output",
            str(output),
        ],
        check=True,
    )

    summary = json.loads(output.read_text())
    assert summary["exist_recall_25cm_5deg"] == pytest.approx(1.0)
    assert summary["exist_recall_50cm_10deg"] == pytest.approx(1.0)
