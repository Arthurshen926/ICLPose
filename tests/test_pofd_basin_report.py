import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.report_pofd_basin import format_markdown_table, summarize_log


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_summarize_log_selects_best_eval_and_computes_oracle_gap(tmp_path):
    run_dir = tmp_path / "pofd_probe"
    run_dir.mkdir()
    log_path = run_dir / "train_log.jsonl"
    _write_jsonl(
        log_path,
        [
            {"step": 1, "split": "train", "pred_cost_m": 0.40},
            {
                "step": 20,
                "split": "eval",
                "pred_cost_m": 0.32,
                "oracle_cost_m": 0.12,
                "pred_trans_m": 0.31,
                "pred_rot_deg": 5.0,
                "top1_acc": 0.50,
                "align_cos": 0.80,
                "spearman": 0.10,
                "pred_success_25cm_10deg": 0.60,
                "oracle_success_25cm_10deg": 0.80,
                "init_success_25cm_10deg": 0.20,
            },
            {
                "step": 40,
                "split": "eval",
                "pred_cost_m": 0.28,
                "oracle_cost_m": 0.11,
                "pred_trans_m": 0.27,
                "pred_rot_deg": 4.0,
                "top1_acc": 0.70,
                "align_cos": 0.75,
                "spearman": 0.20,
                "pred_success_25cm_10deg": 0.70,
                "oracle_success_25cm_10deg": 0.90,
                "init_success_25cm_10deg": 0.30,
            },
        ],
    )

    row = summarize_log(log_path)

    assert row["run"] == "pofd_probe"
    assert row["step"] == 40
    assert row["pred_cost_m"] == 0.28
    assert abs(row["oracle_gap_m"] - 0.17) < 1e-8
    assert row["pred_success_25cm_10deg"] == 0.70


def test_format_markdown_table_emits_basin_columns(tmp_path):
    run_dir = tmp_path / "pofd_probe"
    run_dir.mkdir()
    log_path = run_dir / "train_log.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "step": 10,
                "split": "eval",
                "pred_cost_m": 0.10,
                "oracle_cost_m": 0.05,
                "pred_success_5cm_2deg": 0.25,
                "pred_success_10cm_5deg": 0.50,
                "pred_success_25cm_10deg": 1.0,
                "pred_success_50cm_10deg": 1.0,
            }
        ],
    )

    markdown = format_markdown_table([summarize_log(log_path)])

    assert "succ@5cm/2deg" in markdown
    assert "succ@50cm/10deg" in markdown
    assert "pofd_probe" in markdown
