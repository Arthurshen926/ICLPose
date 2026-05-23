from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.scorer import PoseHypothesisScorer  # noqa: E402
from feature_extract.localizability.selected_feature_map import (  # noqa: E402
    aggregate_selected_track_features,
    hard_hypothesis_selection_targets,
    load_selected_track_feature_bank,
    localization_feature_utility,
    save_selected_track_feature_bank,
    score_query_with_selected_map_features,
    weak_joint_selected_feature_loss,
)
from feature_extract.localizability.selector import LocalizationFeatureSelector  # noqa: E402


def test_localization_feature_utility_prefers_low_pose_cost_and_masks_invalid():
    costs = torch.tensor([[0.02, 0.30, 0.08]])
    valid = torch.tensor([[True, False, True]])

    utility = localization_feature_utility(costs, valid_mask=valid, temperature_m=0.05)

    assert utility.shape == costs.shape
    assert utility[0, 0] > utility[0, 2]
    assert utility[0, 1] == 0.0
    assert torch.allclose(utility.sum(dim=1), torch.ones(1))


def test_hard_hypothesis_targets_find_oracle_and_hard_false_positive():
    scores = torch.tensor([[0.1, 0.8, 0.2], [0.6, 0.1, 0.9]])
    costs = torch.tensor([[0.02, 0.35, 0.08], [0.30, 0.10, 0.12]])

    targets = hard_hypothesis_selection_targets(scores, costs, cost_gap_m=0.12)

    assert targets["positive_index"].tolist() == [0, 1]
    assert targets["hard_negative_index"].tolist() == [1, 0]
    assert targets["active_mask"].tolist() == [True, True]


def test_aggregate_selected_track_features_uses_visibility_and_geometry():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
            [9.0, 9.0],
        ]
    )
    track_ids = torch.tensor([10, 10, 20, 20, 30])
    utility = torch.tensor([1.0, 1.0, 1.0, 3.0, 1.0])
    geometry_valid = torch.tensor([True, True, True, True, False])

    bank = aggregate_selected_track_features(
        features,
        track_ids,
        utility=utility,
        geometry_valid=geometry_valid,
        min_observations=2,
        l2_normalize=False,
    )

    assert bank.track_ids.tolist() == [10, 20]
    assert bank.visibility_count.tolist() == [2, 2]
    assert torch.allclose(bank.features[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(bank.features[1], torch.tensor([0.0, 3.5]))


def test_selected_track_feature_bank_round_trips_npz(tmp_path):
    bank = aggregate_selected_track_features(
        torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]),
        torch.tensor([10, 10, 20, 20]),
        xyz=torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [1.0, 0.0, 2.0], [1.0, 0.2, 2.0]]),
        l2_normalize=False,
    )
    path = tmp_path / "selected_bank.npz"

    save_selected_track_feature_bank(bank, path, metadata={"scene": "unit"})
    loaded, metadata = load_selected_track_feature_bank(path)

    assert metadata["scene"] == "unit"
    assert loaded.track_ids.tolist() == [10, 20]
    assert torch.allclose(loaded.features, bank.features)
    assert loaded.xyz is not None
    assert torch.allclose(loaded.xyz, bank.xyz)


def test_same_selector_scores_query_against_rendered_selected_map_features():
    query = torch.zeros(1, 4, 5, 5)
    query[:, 0, 2, 2] = 1.0
    rendered = torch.zeros(1, 2, 4, 5, 5)
    rendered[:, 0, 0, 2, 2] = 1.0
    rendered[:, 1, 1, 2, 2] = 1.0

    selector = LocalizationFeatureSelector(
        in_channels=4,
        out_channels=4,
        group_size=2,
        spatial_utility=True,
        uncertainty=False,
        identity_init=True,
    )
    scorer = PoseHypothesisScorer(mode="same_pixel", temperature=0.1)

    scores, aux = score_query_with_selected_map_features(query, rendered, selector=selector, scorer=scorer)

    assert scores.shape == (1, 2)
    assert int(scores.argmax(dim=1)[0]) == 0
    assert aux["query_selected"]["z"].shape == query.shape
    assert aux["render_selected"]["z"].shape == rendered.shape


def test_weak_joint_path_backpropagates_through_selector_map_adapter_and_scorer():
    class TinyMapAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(4, 4, kernel_size=1)

        def forward(self, feature, domain="query", rgb=None):
            return self.proj(feature)

    class TinyScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, query, render, *, query_utility=None, render_valid_mask=None):
            query_n = torch.nn.functional.normalize(query.float(), dim=1, eps=1.0e-6)
            render_n = torch.nn.functional.normalize(render.float(), dim=2, eps=1.0e-6)
            score_maps = (query_n[:, None] * render_n).sum(dim=2)
            weight = query_utility[:, None, 0] if query_utility is not None else torch.ones_like(score_maps)
            scores = self.scale * (score_maps * weight).flatten(2).mean(dim=-1)
            return scores, {"score_maps": score_maps, "valid_mask": torch.ones_like(scores, dtype=torch.bool)}

    torch.manual_seed(3)
    query = torch.randn(1, 4, 5, 5)
    rendered = torch.randn(1, 3, 4, 5, 5)
    selector = LocalizationFeatureSelector(in_channels=4, out_channels=4, group_size=2)
    adapter = TinyMapAdapter()
    scorer = TinyScorer()

    scores, aux = score_query_with_selected_map_features(
        query,
        rendered,
        selector=selector,
        map_adapter=adapter,
        scorer=scorer,
    )
    loss, _ = weak_joint_selected_feature_loss(
        scores,
        torch.tensor([[0.02, 0.35, 0.08]]),
        selector_outputs=aux,
        sparsity_weight=0.01,
        utility_entropy_weight=0.01,
    )
    loss.backward()

    assert selector.proj.weight.grad is not None
    assert adapter.proj.weight.grad is not None
    assert scorer.scale.grad is not None


def test_weak_joint_selected_feature_loss_combines_rank_hard_and_regularizers():
    scores = torch.tensor([[0.1, 0.8, 0.2]], requires_grad=True)
    costs = torch.tensor([[0.02, 0.35, 0.08]])
    selector_outputs = {
        "channel_gate": torch.full((1, 2), 0.75),
        "utility": torch.full((1, 1, 3, 3), 0.8),
    }
    map_bank_aux = {"map_feature_variance": torch.tensor(0.05)}

    loss, metrics = weak_joint_selected_feature_loss(
        scores,
        costs,
        selector_outputs=selector_outputs,
        map_bank_aux=map_bank_aux,
        sparsity_weight=0.01,
        utility_entropy_weight=0.01,
        map_variance_weight=0.5,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert metrics["rank_loss"].item() > 0.0
    assert metrics["hard_loss"].item() > 0.0
    assert metrics["map_variance_loss"].item() == pytest.approx(0.05)
    assert scores.grad is not None
