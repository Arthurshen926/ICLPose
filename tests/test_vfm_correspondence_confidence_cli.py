import json
from pathlib import Path

from feature_extract.tools.vfm.train_stage_c28_correspondence_confidence import main


def test_train_stage_c28_correspondence_confidence_writes_model_and_metrics(tmp_path: Path) -> None:
    matches_path = tmp_path / "matches.jsonl"
    rows = [
        {"query_id": "q1", "similarity": 0.95, "similarity_margin": 0.30, "match_rank": 0, "landmark_variance": 0.01, "observation_count": 8, "gt_reproj_error_stride": 0.2},
        {"query_id": "q1", "similarity": 0.90, "similarity_margin": 0.25, "match_rank": 1, "landmark_variance": 0.02, "observation_count": 7, "gt_reproj_error_stride": 0.4},
        {"query_id": "q2", "similarity": 0.45, "similarity_margin": 0.02, "match_rank": 2, "landmark_variance": 0.90, "observation_count": 1, "gt_reproj_error_stride": 3.0},
        {"query_id": "q2", "similarity": 0.40, "similarity_margin": 0.01, "match_rank": 3, "landmark_variance": 0.80, "observation_count": 1, "gt_reproj_error_stride": 4.0},
    ]
    matches_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output_dir = tmp_path / "out"

    main(
        [
            "--match_jsonl",
            str(matches_path),
            "--output_dir",
            str(output_dir),
            "--eval_on_train",
        ]
    )

    summary = json.loads((output_dir / "confidence_summary.json").read_text())
    assert summary["stage"] == "stage_c28_correspondence_confidence"
    assert summary["models"]["descriptor_map"]["eval"]["auroc"] == 1.0
    assert summary["models"]["descriptor_map"]["eval"]["auprc"] == 1.0
    assert (output_dir / "descriptor_map_model.json").exists()


def test_train_stage_c28_correspondence_confidence_supports_pixel_validity_labels(tmp_path: Path) -> None:
    matches_path = tmp_path / "matches.jsonl"
    rows = [
        {"query_id": "q1", "similarity": 0.95, "similarity_margin": 0.30, "match_rank": 0, "gt_reproj_error_px": 2.0},
        {"query_id": "q1", "similarity": 0.90, "similarity_margin": 0.25, "match_rank": 1, "gt_reproj_error_px": 4.0},
        {"query_id": "q2", "similarity": 0.45, "similarity_margin": 0.02, "match_rank": 2, "gt_reproj_error_px": 12.0},
        {"query_id": "q2", "similarity": 0.40, "similarity_margin": 0.01, "match_rank": 3, "gt_reproj_error_px": 24.0},
    ]
    matches_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output_dir = tmp_path / "out_pixel"

    main(
        [
            "--match_jsonl",
            str(matches_path),
            "--output_dir",
            str(output_dir),
            "--feature_sets",
            "descriptor",
            "--label_mode",
            "pixel",
            "--positive_px",
            "5.0",
            "--eval_on_train",
        ]
    )

    summary = json.loads((output_dir / "confidence_summary.json").read_text())
    assert summary["label_policy"]["mode"] == "pixel"
    assert summary["label_policy"]["positive_px"] == 5.0
    assert summary["models"]["descriptor"]["eval"]["auroc"] == 1.0
    assert (output_dir / "descriptor_model.json").exists()
