import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_patch_topk_diagnostics import main


def test_summarize_patch_topk_diagnostics_writes_method_k_table(tmp_path: Path) -> None:
    eval_path = tmp_path / "raw_soft3" / "eval_summary.json"
    eval_path.parent.mkdir()
    eval_path.write_text(
        json.dumps(
            {
                "query_count": 2,
                "mean_match_count": 20.0,
                "mean_patch_at_5": 0.7,
                "mean_gt_precision_2stride": 0.8,
                "mean_pnp_inlier_patch_at_5": 0.9,
                "success_25cm_10deg": 0.5,
                "success_50cm_10deg": 0.6,
                "median_translation_error_m": 0.2,
                "median_rotation_error_deg": 1.0,
            }
        )
        + "\n"
    )

    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    main(
        [
            "--run",
            f"ShopFacade,raw1280,3,{eval_path}",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ]
    )

    report = json.loads(output_json.read_text())
    assert report["stage"] == "patch_topk_diagnostics_summary"
    assert report["rows"][0]["scene"] == "ShopFacade"
    assert report["rows"][0]["top_k"] == 3
    assert report["rows"][0]["mean_patch_at_5"] == 0.7
    assert "raw1280" in output_md.read_text()
