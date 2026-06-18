from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_extract.tools.vfm import run_matcha_real_render_query_eval_matrix as matrix
from feature_extract.tools.vfm.run_matcha_real_render_query_eval_matrix import main


def test_eval_matrix_dry_run_contains_required_oldhospital_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
            "--candidate_bank",
            "candidates.jsonl",
            "--dry_run",
        ]
    )

    out = capsys.readouterr().out
    assert "--render_pose_mode gt" in out
    assert "--render_pose_mode gt_offset --render_pose_world_offset 0.250,0.000,0.000" in out
    assert "--render_pose_mode gt_rotation_offset --render_pose_rotation_offset_deg 0.000,6.000,0.000" in out
    assert "--render_pose_mode reference_top1" in out
    assert "--render_pose_mode reference_top5" in out
    assert "--render_pose_mode reference_top10" in out
    assert "--max_queries 0" in out
    assert "--feature_mode radio_dual" in out
    assert "--match_mode matcha_c2f" in out


def test_eval_matrix_accepts_cache_roots_and_extra_eval_args(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
            "--candidate_bank",
            "candidates.jsonl",
            "--query_token_cache_dir",
            "query_cache",
            "--render_token_cache_root",
            "render_tokens",
            "--render_rgb_depth_cache_root",
            "render_rgb_depth",
            "--extract_query_features_from_image",
            "--skip_existing_query_tokens",
            "--extra_eval_arg=--coverage_filter_min_confidence=0.05",
            "--dry_run",
        ]
    )

    out = capsys.readouterr().out
    assert "--query_token_cache_dir query_cache" in out
    assert "--render_token_cache_dir render_tokens/gt" in out
    assert "--render_rgb_depth_cache_dir render_rgb_depth/gt" in out
    assert "--extract_query_features_from_image" in out
    assert "--skip_existing_query_tokens" in out
    assert "--coverage_filter_min_confidence=0.05" in out


def test_eval_matrix_can_filter_labels_for_parallel_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
            "--candidate_bank",
            "candidates.jsonl",
            "--labels",
            "gt_offset_0p050m,reference_top10",
            "--dry_run",
        ]
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "gt_offset_0p050m" in lines[0]
    assert "reference_top10" in lines[1]
    assert "gt_rotation_y_3p000deg" not in "\n".join(lines)


def test_eval_matrix_forwards_query_shard_controls(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
            "--candidate_bank",
            "candidates.jsonl",
            "--labels",
            "gt",
            "--start_index",
            "64",
            "--max_queries",
            "32",
            "--view_selection",
            "uniform",
            "--dry_run",
        ]
    )

    out = capsys.readouterr().out
    assert "--start_index 64" in out
    assert "--max_queries 32" in out
    assert "--view_selection uniform" in out


def test_eval_matrix_applies_rotation_search_only_to_rotation_jobs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
            "--candidate_bank",
            "candidates.jsonl",
            "--rotation_search_offsets_deg",
            "-3,0,3",
            "--rotation_search_axis",
            "y",
            "--labels",
            "gt,gt_rotation_y_3p000deg",
            "--dry_run",
        ]
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "--render_pose_mode gt " in f"{lines[0]} "
    assert "--render_pose_rotation_search_offsets_deg" not in lines[0]
    assert "--render_pose_mode gt_rotation_offset" in lines[1]
    assert "--render_pose_rotation_search_offsets_deg=-3,0,3" in lines[1]
    assert "--render_pose_rotation_search_axis y" in lines[1]


def test_eval_matrix_can_forward_resume_and_diagnostic_table_controls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
            "--candidate_bank",
            "candidates.jsonl",
            "--labels",
            "gt,reference_top5",
            "--resume_eval_rows",
            "--diagnostic_labels",
            "gt,reference_top5",
            "--dry_run",
        ]
    )

    out = capsys.readouterr().out
    assert "--stream_rows" in out
    assert "--resume_existing_rows" in out
    assert "--save_match_table" in out
    assert "--match_table_stage candidate" in out
    assert "--save_coarse_oracle_table" in out
    assert "--save_coarse_oracle_candidate_table" in out


def test_eval_matrix_summary_writer_collects_existing_label_summaries(tmp_path: Path) -> None:
    gt_summary = tmp_path / "gt" / "summary.json"
    top5_summary = tmp_path / "reference_top5" / "summary.json"
    gt_summary.parent.mkdir(parents=True)
    top5_summary.parent.mkdir(parents=True)
    gt_summary.write_text(
        json.dumps(
            {
                "stage": "render_rgb_feature_keypoint_pose",
                "config": {"render_pose_mode": "gt"},
                "metrics": {"query_count": 2, "median_translation_error_m": 0.10, "success_10cm_5deg": 0.5},
            }
        )
        + "\n"
    )
    top5_summary.write_text(
        json.dumps(
            {
                "stage": "render_rgb_feature_keypoint_pose",
                "config": {"render_pose_mode": "reference_top5"},
                "metrics": {"query_count": 2, "median_translation_error_m": 0.27, "success_10cm_5deg": 0.1},
            }
        )
        + "\n"
    )

    payload = matrix.write_matrix_summary(tmp_path, labels=["gt", "reference_top5"])

    assert (tmp_path / "matrix_summary.json").exists()
    assert (tmp_path / "matrix_summary.tsv").exists()
    assert payload["rows"][0]["label"] == "gt"
    assert "median_translation_error_m" in (tmp_path / "matrix_summary.tsv").read_text()


def test_eval_matrix_execute_skips_completed_label_and_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = tmp_path / "gt" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "stage": "render_rgb_feature_keypoint_pose",
                "config": {"render_pose_mode": "gt"},
                "metrics": {"query_count": 1, "median_translation_error_m": 0.1},
            }
        )
        + "\n"
    )
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(list(command))

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    main(
        [
            "--query_manifest",
            "test_manifest.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_root",
            str(tmp_path),
            "--matcha_joint_checkpoint",
            "model.pt",
            "--candidate_bank",
            "candidates.jsonl",
            "--labels",
            "gt",
            "--skip_completed",
            "--execute",
        ]
    )

    assert calls == []
    assert (tmp_path / "matrix_summary.json").exists()


def test_eval_matrix_controlled_comparison_reports_metric_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "old"
    candidate = tmp_path / "new"
    for root, value in ((baseline, 0.20), (candidate, 0.12)):
        summary = root / "gt" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps(
                {
                    "stage": "render_rgb_feature_keypoint_pose",
                    "metrics": {"median_translation_error_m": value, "success_10cm_5deg": 0.5},
                }
            )
            + "\n"
        )

    payload = matrix.write_controlled_comparison(
        baseline,
        candidate,
        labels=["gt"],
        output_json=tmp_path / "old_vs_new_controlled_comparison.json",
        output_tsv=tmp_path / "old_vs_new_controlled_comparison.tsv",
    )

    assert payload["rows"][0]["median_translation_error_m_delta"] == pytest.approx(-0.08)
    assert (tmp_path / "old_vs_new_controlled_comparison.tsv").exists()
