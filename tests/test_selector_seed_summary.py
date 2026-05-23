from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.summarize_selector_seed_runs import summarize_selector_seed_runs


def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_summarize_selector_seed_runs_uses_best_pred_row_and_gate(tmp_path):
    run_a = tmp_path / "seed_a" / "train_log.jsonl"
    run_b = tmp_path / "seed_b" / "train_log.jsonl"
    _write_log(
        run_a,
        [
            {"step": 10, "pred_cost_m": 0.30, "oracle_gap_m": 0.17, "top1_acc": 0.50, "spearman": 0.40},
            {"step": 20, "pred_cost_m": 0.19, "oracle_gap_m": 0.06, "top1_acc": 0.82, "spearman": 0.61},
        ],
    )
    _write_log(
        run_b,
        [
            {"step": 10, "pred_cost_m": 0.21, "oracle_gap_m": 0.08, "top1_acc": 0.75, "spearman": 0.55},
            {"step": 20, "pred_cost_m": 0.22, "oracle_gap_m": 0.09, "top1_acc": 0.74, "spearman": 0.54},
        ],
    )

    summary = summarize_selector_seed_runs([run_a, run_b])

    assert summary["num_runs"] == 2
    assert summary["num_promotion_pass"] == 2
    assert summary["runs"][0]["best_step"] == 20
    assert summary["runs"][1]["best_step"] == 10
    assert summary["mean"]["pred_cost_m"] == 0.20
    assert summary["mean"]["top1_acc"] == pytest.approx(0.785)
