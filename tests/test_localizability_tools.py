import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.report_localizability import format_markdown_table, summarize_jsonl


def test_report_localizability_summarizes_best_eval_row(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    rows = [
        {"split": "train", "step": 1, "pred_cost_m": 0.40},
        {
            "split": "eval",
            "step": 10,
            "pred_cost_m": 0.31,
            "oracle_cost_m": 0.12,
            "top1_acc": 0.60,
            "spearman": 0.20,
            "basin_recall@5": 0.75,
        },
        {
            "split": "eval",
            "step": 20,
            "pred_cost_m": 0.25,
            "oracle_cost_m": 0.11,
            "top1_acc": 0.70,
            "spearman": 0.40,
            "basin_recall@5": 0.80,
        },
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = summarize_jsonl(log_path)
    markdown = format_markdown_table([summary])

    assert summary["step"] == 20
    assert summary["oracle_gap_m"] == 0.14
    assert "| run | step | pred | oracle | gap | top1 | spearman | basin@5 |" in markdown
