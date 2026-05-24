from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.report_training_seed_statistics import (  # noqa: E402
    build_training_seed_report,
    format_training_seed_report_markdown,
    load_best_eval_row,
)


def _write_log(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_training_seed_report_marks_incomplete_groups_and_reports_ci(tmp_path):
    log1 = tmp_path / "seed1.jsonl"
    log2 = tmp_path / "seed2.jsonl"
    _write_log(
        log1,
        [
            {"split": "train", "step": 1, "pred_cost_m": 0.5},
            {"split": "eval", "step": 1, "pred_cost_m": 0.3, "top1_acc": 0.5},
            {"split": "eval", "step": 2, "pred_cost_m": 0.2, "top1_acc": 0.7},
        ],
    )
    _write_log(log2, [{"split": "eval", "step": 1, "pred_cost_m": 0.4, "top1_acc": 0.2}])

    report = build_training_seed_report(
        runs=[("selector", log1), ("selector", log2)],
        expected_seeds=5,
        selection_metric="pred_cost_m",
        selection_mode="min",
        metrics=["pred_cost_m", "top1_acc"],
    )
    markdown = format_training_seed_report_markdown(report)

    assert load_best_eval_row(log1, selection_metric="pred_cost_m", selection_mode="min")["step"] == 2
    assert report["groups"][0]["num_seeds"] == 2
    assert report["groups"][0]["complete"] is False
    assert report["groups"][0]["metrics"]["pred_cost_m"]["mean"] == pytest.approx(0.3)
    assert "selector" in markdown
    assert "2/5" in markdown
