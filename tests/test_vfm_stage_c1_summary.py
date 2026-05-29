import csv
import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_stage_c1_patch_selector import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_summarize_stage_c1_patch_selector_groups_seed_statistics(tmp_path: Path) -> None:
    runs = []
    for seed, success in [(0, 0.4), (1, 0.6)]:
        train = tmp_path / f"learned64_seed{seed}" / "train.json"
        compression = tmp_path / f"learned64_seed{seed}" / "compression.json"
        eval_summary = tmp_path / f"learned64_seed{seed}" / "eval.json"
        _write_json(
            train,
            {
                "training": {
                    "output_dim": 64,
                    "eval_top1_acc": 0.5 + seed * 0.1,
                    "final_loss": 1.0 - seed * 0.1,
                },
                "sample_summary": {"sample_count": 100 + seed},
            },
        )
        _write_json(
            compression,
            {
                "method": "learned_linear_patch",
                "output_dim": 64,
                "elapsed_sec": 2.0 + seed,
                "storage_bytes": {"query_tokens": 10, "landmark_bank": 20, "transform": 1},
            },
        )
        _write_json(
            eval_summary,
            {
                "query_count": 5,
                "elapsed_sec": 3.0 + seed,
                "success_25cm_10deg": success,
                "success_50cm_10deg": 0.8,
                "median_translation_error_m": 0.3 - seed * 0.1,
                "median_rotation_error_deg": 1.0,
                "mean_pnp_inlier_patch_at_1": 0.7,
                "mean_pnp_inlier_patch_at_5": 0.9,
            },
        )
        runs.extend(["--run", f"OldHospital,learned,64,{seed},{train},{compression},{eval_summary}"])

    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"
    main(runs + ["--output_json", str(output_json), "--output_csv", str(output_csv), "--output_md", str(output_md)])

    report = json.loads(output_json.read_text())
    assert report["stage"] == "stage_c1_patch_selector_summary"
    row = report["groups"][0]
    assert row["scene"] == "OldHospital"
    assert row["method"] == "learned"
    assert row["output_dim"] == 64
    assert row["seed_count"] == 2
    assert row["success_25cm_10deg_mean"] == 0.5
    assert row["success_25cm_10deg_best"] == 0.6
    assert row["median_translation_error_m_best"] == 0.19999999999999998
    csv_rows = list(csv.DictReader(output_csv.open()))
    assert csv_rows[0]["method"] == "learned"
    assert "learned" in output_md.read_text()


def test_summarize_stage_c1_patch_selector_accepts_untrained_controls(tmp_path: Path) -> None:
    compression = tmp_path / "random64" / "compression.json"
    evaluation = tmp_path / "random64" / "eval.json"
    _write_json(
        compression,
        {
            "method": "random",
            "output_dim": 64,
            "elapsed_sec": 1.0,
            "storage_bytes": {"query_tokens": 1, "landmark_bank": 2, "transform": 3},
        },
    )
    _write_json(
        evaluation,
        {
            "query_count": 2,
            "elapsed_sec": 4.0,
            "success_25cm_10deg": 0.25,
            "success_50cm_10deg": 0.5,
            "median_translation_error_m": 0.7,
            "median_rotation_error_deg": 1.2,
            "mean_pnp_inlier_patch_at_1": 0.3,
            "mean_pnp_inlier_patch_at_5": 0.4,
        },
    )

    output_json = tmp_path / "summary.json"
    main(
        [
            "--run",
            f"ShopFacade,random,64,0,-,{compression},{evaluation}",
            "--output_json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert report["rows"][0]["train_summary_path"] == ""
    assert report["groups"][0]["train_eval_top1_acc_mean"] is None
