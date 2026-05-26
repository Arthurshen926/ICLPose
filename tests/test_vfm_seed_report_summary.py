import json

import pytest

from feature_extract.tools.vfm.summarize_seed_reports import main as summarize_seed_reports_cli_main
from feature_extract.vfm.seed_report_summary import summarize_seed_reports


def _write_report(path, seed, pred, top1, spearman):
    path.write_text(
        json.dumps(
            {
                "method": f"selected_seed{seed}",
                "protocol_kind": "reference_pose",
                "query_count": 10,
                "mean_pred_cost_m": pred,
                "mean_top1_acc": top1,
                "mean_spearman": spearman,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_seed_report_summary_computes_metric_statistics(tmp_path):
    reports = []
    for seed, pred in enumerate([3.0, 5.0, 7.0]):
        path = tmp_path / f"seed{seed}.json"
        _write_report(path, seed, pred=pred, top1=0.5 + 0.1 * seed, spearman=0.2 + 0.1 * seed)
        reports.append(path)

    summary = summarize_seed_reports(
        reports,
        label="selected64",
        metrics=("mean_pred_cost_m", "mean_top1_acc", "mean_spearman"),
    )

    assert summary["label"] == "selected64"
    assert summary["seed_count"] == 3
    assert summary["protocol_kind"] == "reference_pose"
    assert summary["query_count"] == 10
    assert summary["metrics"]["mean_pred_cost_m"]["mean"] == pytest.approx(5.0)
    assert summary["metrics"]["mean_pred_cost_m"]["std"] == pytest.approx(2.0)
    assert summary["metrics"]["mean_top1_acc"]["max"] == pytest.approx(0.7)
    assert summary["inputs"][0]["sha256"]


def test_seed_report_summary_cli_writes_json_and_markdown(tmp_path):
    report_paths = []
    for seed, pred in enumerate([1.0, 2.0]):
        path = tmp_path / f"report_seed{seed}.json"
        _write_report(path, seed, pred=pred, top1=0.4 + 0.1 * seed, spearman=0.1 + 0.1 * seed)
        report_paths.append(path)
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    summarize_seed_reports_cli_main(
        [
            "--label",
            "toy",
            "--reports",
            *(str(path) for path in report_paths),
            "--metrics",
            "mean_pred_cost_m",
            "mean_top1_acc",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert payload["metrics"]["mean_pred_cost_m"]["mean"] == pytest.approx(1.5)
    table = output_md.read_text()
    assert "| mean_pred_cost_m |" in table
    assert "toy" in table
