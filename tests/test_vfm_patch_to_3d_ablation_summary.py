import csv
import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_patch_to_3d_ablation import main


def _summary(path: Path, label: str, median_t: float, success: float) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "patch_to_3d_vfm_matching_baseline",
                "query_count": 2,
                "inputs": {"landmark_bank": f"/banks/{label}.npz"},
                "submap": {"mode": "reference_visibility", "top_n": 5},
                "matching_config": {
                    "match_mode": "soft_mutual",
                    "top_k": 5,
                    "mutual_top_k": 5,
                    "ratio_threshold": None,
                    "min_similarity_margin": 0.02,
                    "landmark_quality": {"enabled": True},
                },
                "mean_match_count": 10.0,
                "mean_patch_at_1": 0.2,
                "mean_patch_at_5": 0.4,
                "mean_gt_precision_stride": 0.3,
                "mean_pnp_inlier_patch_at_1": 0.5,
                "mean_pnp_inlier_patch_at_5": 0.6,
                "mean_pnp_inlier_gt_precision_stride": 0.7,
                "mean_pnp_inlier_count": 8.0,
                "mean_pnp_inlier_convex_hull_area_frac": 0.25,
                "positive_set_summary": {
                    "mean_positives_per_token": 1.5,
                    "mean_zero_positive_token_ratio": 0.75,
                },
                "visible_landmark_recall": {
                    "mean": 0.8,
                    "median": 0.85,
                    "p25": 0.7,
                    "p75": 0.9,
                },
                "reference_prior": {
                    "top1": {
                        "median_translation_error_m": 1.1,
                        "median_rotation_error_deg": 2.2,
                        "success_25cm_10deg": 0.25,
                    },
                    "oracle": {
                        "median_translation_error_m": 0.4,
                        "median_rotation_error_deg": 1.0,
                        "success_25cm_10deg": 0.5,
                    },
                },
                "pnp_solve_rate": 1.0,
                "success_25cm_10deg": success,
                "success_50cm_10deg": 0.75,
                "success_1m_10deg": 0.9,
                "median_translation_error_m": median_t,
                "median_rotation_error_deg": 1.2,
            }
        )
        + "\n"
    )


def test_summarize_patch_to_3d_ablation_sorts_and_writes_tables(tmp_path: Path) -> None:
    worse = tmp_path / "worse_summary.json"
    better = tmp_path / "better_summary.json"
    _summary(worse, "mean", median_t=0.8, success=0.1)
    _summary(better, "medoid", median_t=0.3, success=0.4)
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"

    main(
        [
            "--summary_json",
            str(worse),
            str(better),
            "--sort_metric",
            "median_translation_error_m",
            "--output_json",
            str(output_json),
            "--output_csv",
            str(output_csv),
            "--output_md",
            str(output_md),
        ]
    )

    report = json.loads(output_json.read_text())
    assert report["best"]["label"] == "better"
    assert report["rows"][0]["label"] == "better"
    assert report["rows"][0]["landmark_bank_name"] == "medoid.npz"
    assert report["rows"][0]["visible_landmark_recall_median"] == 0.85
    assert report["rows"][0]["reference_top1_median_translation_error_m"] == 1.1
    assert report["rows"][0]["reference_oracle_median_translation_error_m"] == 0.4
    csv_rows = list(csv.DictReader(output_csv.open()))
    assert csv_rows[0]["label"] == "better"
    assert csv_rows[0]["visible_landmark_recall_median"] == "0.85"
    assert "better" in output_md.read_text()
