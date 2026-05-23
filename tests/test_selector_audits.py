from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.eval_localizability_selector_audits import channel_counterfactual_report, spatial_counterfactual_report


def test_spatial_counterfactual_report_detects_high_utility_damage():
    score_maps = torch.tensor([[[[2.0, 0.0]], [[0.0, 1.0]]]])
    utility = torch.tensor([[[[0.9, 0.1]]]])
    costs = torch.tensor([[0.1, 0.5]])
    valid = torch.tensor([[True, True]])
    basin = torch.tensor([[True, False]])

    report = spatial_counterfactual_report(
        score_maps,
        utility,
        costs,
        valid_mask=valid,
        basin_label=basin,
        drop_fraction=0.5,
    )

    assert report["base"]["pred_cost_m"] == pytest.approx(0.1)
    assert report["drop_high"]["pred_cost_m"] == pytest.approx(0.5)
    assert report["drop_low"]["pred_cost_m"] == pytest.approx(0.1)
    assert report["drop_high_cost_delta_m"] > report["drop_low_cost_delta_m"]


def test_spatial_counterfactual_report_resizes_utility_to_score_map_grid():
    score_maps = torch.tensor([[[[2.0, 0.0]], [[0.0, 1.0]]]])
    utility = torch.tensor([[[[0.9, 0.9, 0.1, 0.1]]]])
    costs = torch.tensor([[0.1, 0.5]])

    report = spatial_counterfactual_report(score_maps, utility, costs, drop_fraction=0.5)

    assert report["drop_high"]["pred_cost_m"] >= report["base"]["pred_cost_m"]


def test_channel_counterfactual_report_detects_high_gate_damage():
    base_scores = torch.tensor([[2.0, 0.0]])
    ablated = torch.tensor(
        [
            [[0.0, 1.0]],
            [[2.0, 0.0]],
        ]
    )
    costs = torch.tensor([[0.1, 0.5]])
    group_importance = torch.tensor([0.9, 0.1])

    report = channel_counterfactual_report(base_scores, ablated, costs, group_importance)

    assert report["drop_high_group_idx"] == 0
    assert report["drop_low_group_idx"] == 1
    assert report["base"]["pred_cost_m"] == pytest.approx(0.1)
    assert report["drop_high"]["pred_cost_m"] == pytest.approx(0.5)
    assert report["drop_low"]["pred_cost_m"] == pytest.approx(0.1)
    assert report["drop_high_cost_delta_m"] > report["drop_low_cost_delta_m"]
