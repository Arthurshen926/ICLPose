import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_patch_map_quality_modes import main


def test_summarize_patch_map_quality_modes_writes_comparison_table(tmp_path: Path) -> None:
    eval_summary = tmp_path / "eval_summary.json"
    eval_summary.write_text(
        json.dumps(
            {
                "query_count": 3,
                "success_25cm_10deg": 0.5,
                "success_50cm_10deg": 0.75,
                "median_translation_error_m": 0.2,
                "median_rotation_error_deg": 1.0,
                "mean_pnp_inlier_patch_at_1": 0.4,
                "mean_pnp_inlier_count": 10.0,
                "mean_match_count": 20.0,
            }
        )
        + "\n"
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    main(
        [
            "--run",
            f"OldHospital,feature_only,{eval_summary}",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ]
    )

    report = json.loads(output_json.read_text())
    assert report["stage"] == "patch_map_quality_modes_summary"
    assert report["rows"][0]["method"] == "feature_only"
    assert "feature_only" in output_md.read_text()
