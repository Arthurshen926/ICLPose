import csv
import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_stage_c0_compression import main


def test_summarize_stage_c0_compression_writes_accuracy_storage_runtime_table(tmp_path: Path) -> None:
    compression = tmp_path / "pca64_compression.json"
    evaluation = tmp_path / "pca64_eval.json"
    compression.write_text(
        json.dumps(
            {
                "method": "pca",
                "output_dim": 64,
                "storage_bytes": {"query_tokens": 1000, "landmark_bank": 250, "transform": 50},
                "elapsed_sec": 3.0,
            }
        )
        + "\n"
    )
    evaluation.write_text(
        json.dumps(
            {
                "query_count": 2,
                "elapsed_sec": 7.0,
                "mean_match_count": 12.0,
                "mean_pnp_inlier_count": 5.0,
                "mean_pnp_inlier_patch_at_1": 0.6,
                "success_25cm_10deg": 0.5,
                "success_50cm_10deg": 1.0,
                "median_translation_error_m": 0.2,
                "median_rotation_error_deg": 1.0,
            }
        )
        + "\n"
    )
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"

    main(
        [
            "--run",
            f"pca64,{compression},{evaluation}",
            "--output_json",
            str(output_json),
            "--output_csv",
            str(output_csv),
            "--output_md",
            str(output_md),
        ]
    )

    report = json.loads(output_json.read_text())
    assert report["rows"][0]["label"] == "pca64"
    assert report["rows"][0]["method"] == "pca"
    assert report["rows"][0]["output_dim"] == 64
    assert report["rows"][0]["total_storage_bytes"] == 1300
    assert report["rows"][0]["total_elapsed_sec"] == 10.0
    csv_rows = list(csv.DictReader(output_csv.open()))
    assert csv_rows[0]["label"] == "pca64"
    assert "pca64" in output_md.read_text()
