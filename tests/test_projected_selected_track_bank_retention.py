from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.scorer import PoseHypothesisScorer
from feature_extract.localizability.rendered_map_scoring import densify_projected_feature_maps, render_selected_track_feature_maps
from feature_extract.localizability.selected_feature_map import SelectedTrackFeatureBank
from feature_extract.tools.eval_projected_selected_track_bank_retention import (
    intrinsics_dicts_to_scaled_k,
    projected_valid_coverage,
    score_selected_query_against_projected_bank,
)


class IdentitySelector(nn.Module):
    def forward(self, feature):
        return {
            "z": feature,
            "utility": torch.ones((feature.shape[0], 1, feature.shape[-2], feature.shape[-1]), device=feature.device),
            "channel_gate": torch.ones((feature.shape[0], 1), device=feature.device),
        }


def test_intrinsics_dicts_to_scaled_k_scales_from_source_to_target_hw():
    intrinsics = [{"fx": 100.0, "fy": 80.0, "cx": 50.0, "cy": 40.0}]

    k = intrinsics_dicts_to_scaled_k(intrinsics, source_hw=(100, 200), target_hw=(10, 20), device=torch.device("cpu"))

    assert k.shape == (1, 3, 3)
    assert k[0, 0, 0].item() == pytest.approx(10.0)
    assert k[0, 1, 1].item() == pytest.approx(8.0)
    assert k[0, 0, 2].item() == pytest.approx(5.0)
    assert k[0, 1, 2].item() == pytest.approx(4.0)


def test_score_selected_query_against_projected_bank_does_not_reselect_bank_features():
    query = torch.zeros(1, 2, 5, 5)
    query[:, 0, 2, 2] = 1.0
    bank = SelectedTrackFeatureBank(
        track_ids=torch.tensor([1]),
        features=torch.tensor([[1.0, 0.0]]),
        visibility_count=torch.tensor([2]),
        feature_variance=torch.tensor([0.0]),
        utility_mean=torch.tensor([1.0]),
        xyz=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    candidate = torch.eye(4).view(1, 1, 4, 4)
    intrinsics = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]])
    scorer = PoseHypothesisScorer(mode="same_pixel", temperature=0.1)

    scores, aux = score_selected_query_against_projected_bank(
        query,
        bank,
        candidate,
        intrinsics,
        image_hw=(5, 5),
        selector=IdentitySelector(),
        scorer=scorer,
    )

    assert scores.shape == (1, 1)
    assert scores[0, 0].item() > 0.0
    assert aux["rendered_selected_map_feature"].shape == (1, 1, 2, 5, 5)


def test_projected_valid_coverage_accepts_boolean_masks():
    mask = torch.tensor([[[[[True, False], [True, True]]]]])

    coverage = projected_valid_coverage(mask)

    assert coverage["projected_valid_pixel_frac"] == pytest.approx(0.75)
    assert coverage["projected_valid_pixels_per_candidate"] == pytest.approx(3.0)


def test_render_selected_track_feature_maps_splat_radius_expands_coverage():
    bank = SelectedTrackFeatureBank(
        track_ids=torch.tensor([1]),
        features=torch.tensor([[1.0, 0.0]]),
        visibility_count=torch.tensor([2]),
        feature_variance=torch.tensor([0.0]),
        utility_mean=torch.tensor([1.0]),
        xyz=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    candidate = torch.eye(4).view(1, 1, 4, 4)
    intrinsics = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]])

    _feature, valid = render_selected_track_feature_maps(
        bank,
        candidate,
        intrinsics,
        image_hw=(5, 5),
        splat_radius=1,
    )

    assert int(valid.sum().item()) == 9


def test_densify_projected_feature_maps_expands_sparse_projection_by_average_pool():
    rendered = torch.zeros(1, 1, 2, 5, 5)
    rendered[:, :, 0, 2, 2] = 2.0
    rendered[:, :, 1, 2, 2] = 4.0
    valid = torch.zeros(1, 1, 1, 5, 5, dtype=torch.bool)
    valid[:, :, :, 2, 2] = True

    dense, dense_valid = densify_projected_feature_maps(rendered, valid, radius=1)

    assert int(dense_valid.sum().item()) == 9
    assert dense[0, 0, 0, 1, 1].item() == pytest.approx(2.0)
    assert dense[0, 0, 1, 3, 3].item() == pytest.approx(4.0)
