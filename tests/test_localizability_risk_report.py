from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.risk_report import summarize_candidate_table_risk  # noqa: E402


def test_candidate_table_risk_uses_top1_margin_and_topk_basin_recall():
    rows = [
        {"sample_name": "q1", "candidate_idx": 0, "score": 0.90, "pose_cost_m": 0.10, "valid": True, "in_basin": True},
        {"sample_name": "q1", "candidate_idx": 1, "score": 0.20, "pose_cost_m": 0.40, "valid": True, "in_basin": False},
        {"sample_name": "q2", "candidate_idx": 0, "score": 0.80, "pose_cost_m": 0.50, "valid": True, "in_basin": False},
        {"sample_name": "q2", "candidate_idx": 1, "score": 0.70, "pose_cost_m": 0.12, "valid": True, "in_basin": True},
        {"sample_name": "q3", "candidate_idx": 0, "score": 0.30, "pose_cost_m": 0.30, "valid": True, "in_basin": False},
        {"sample_name": "q3", "candidate_idx": 1, "score": 0.10, "pose_cost_m": 0.20, "valid": True, "in_basin": True},
    ]

    summary = summarize_candidate_table_risk(rows, topk=(1, 2), confidence_mode="top1_margin")

    assert summary["num_samples"] == 3
    assert summary["selected_success_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["top1_acc"] == pytest.approx(1.0 / 3.0)
    assert summary["basin_recall@1"] == pytest.approx(1.0 / 3.0)
    assert summary["basin_recall@2"] == pytest.approx(1.0)
    assert summary["risk@50"] == pytest.approx(0.5)
    assert summary["high_conf_false_accept@50"] == pytest.approx(0.5)
