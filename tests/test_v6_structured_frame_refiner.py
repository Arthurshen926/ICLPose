from pathlib import Path

import torch

from feature_extract.tools.vfm.train_v6_structured_frame_refiner import (
    _replay_correctability,
)
from feature_extract.vfm.localization_v6.structured_frame_refiner import (
    StructuredFrameRefiner,
    StructuredFrameRefinerConfig,
    load_structured_frame_refiner,
    save_structured_frame_refiner,
    structured_frame_correlation,
)


def test_structured_frame_refiner_shapes_and_identity_initialization():
    config = StructuredFrameRefinerConfig(
        feature_dim=8,
        correlation_radius=1,
        hidden_dim=16,
        transformer_layers=1,
        attention_heads=4,
    )
    model = StructuredFrameRefiner(config)
    correlation = torch.randn(2, 12, 18)
    canonical = torch.rand(2, 12, 2) * 2.0 - 1.0
    candidate = torch.rand(2, 12, 2)
    output = model(correlation, canonical, candidate)
    assert output["linear"].shape == (2, 2, 2)
    assert output["translation"].shape == (2, 2)
    assert output["usable_logit"].shape == (2,)
    assert torch.allclose(
        output["linear"], torch.eye(2)[None].expand(2, -1, -1)
    )
    assert torch.allclose(output["translation"], torch.zeros(2, 2))


def test_matrix_exponential_parameterization_starts_at_identity():
    model = StructuredFrameRefiner(
        StructuredFrameRefinerConfig(
            feature_dim=8,
            correlation_radius=1,
            hidden_dim=16,
            transformer_layers=1,
            attention_heads=4,
            maximum_linear_residual=3.2,
            linear_parameterization="matrix_exponential",
        )
    )
    output = model(
        torch.randn(2, 12, 18),
        torch.rand(2, 12, 2) * 2.0 - 1.0,
        torch.rand(2, 12, 2),
    )
    assert torch.allclose(
        output["linear"], torch.eye(2)[None].expand(2, -1, -1)
    )
    assert torch.all(
        torch.linalg.det(output["linear"]) > 0.0
    )


def test_structured_correlation_keeps_mean_and_appearance_mode_channels():
    query = torch.zeros(1, 4, 7, 7)
    query[:, 0] = 1.0
    mean = torch.zeros(2, 5, 4)
    mean[..., 0] = 1.0
    modes = mean[:, :, None].repeat(1, 1, 2, 1)
    weights = torch.full((2, 5, 2), 0.5)
    valid = torch.ones(2, 5, 2, dtype=torch.bool)
    xy = torch.full((2, 5, 2), 3.0)
    value = structured_frame_correlation(
        query, mean, modes, weights, valid, xy, radius=1
    )
    assert value.shape == (2, 5, 18)
    assert torch.all(value > 0.9)


def test_structured_update_is_applied_about_each_chart_center():
    candidate = torch.tensor(
        [[[9.0, 19.0], [11.0, 19.0], [11.0, 21.0], [9.0, 21.0]]]
    )
    quarter_turn = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])
    model = StructuredFrameRefiner()
    corrected = model.apply_update(
        candidate, quarter_turn, torch.tensor([[2.0, -3.0]])
    )
    expected = torch.tensor(
        [[[13.0, 16.0], [13.0, 18.0], [11.0, 18.0], [11.0, 16.0]]]
    )
    assert torch.allclose(corrected, expected)


def test_canonical_residual_update_is_well_conditioned_for_tiny_chart():
    model = StructuredFrameRefiner(
        StructuredFrameRefinerConfig(
            update_parameterization="canonical_residual"
        )
    )
    candidate = torch.tensor(
        [[[20.0, 15.0], [20.1, 15.0], [20.1, 15.1], [20.0, 15.1]]]
    )
    canonical = torch.tensor(
        [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]]
    )
    residual = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])
    corrected = model.apply_update(
        candidate,
        residual,
        torch.tensor([[0.5, -0.25]]),
        canonical,
    )
    expected = (
        candidate
        + torch.einsum("bnc,bdc->bnd", canonical, residual)
        + torch.tensor([[[0.5, -0.25]]])
    )
    assert torch.allclose(corrected, expected)


def test_deployment_replay_accepts_bounded_canonical_reflection_residual():
    canonical = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    ).numpy()
    candidate = 20.0 + 0.05 * canonical
    residual = canonical @ torch.tensor(
        [[-1.0, 0.2], [0.1, 0.8]]
    ).numpy().T
    target = candidate + residual + torch.tensor([0.4, -0.3]).numpy()
    usable, before, fitted = _replay_correctability(
        candidate,
        target,
        canonical,
        maximum_fitted_error_cells=0.01,
        maximum_linear_residual=3.2,
        maximum_translation_cells=3.5,
        linear_parameterization="additive",
        update_parameterization="canonical_residual",
    )
    assert usable
    assert before > 0.5
    assert fitted < 1e-5


def test_structured_frame_refiner_roundtrip_enforces_contract(
    tmp_path: Path,
):
    model = StructuredFrameRefiner(
        StructuredFrameRefinerConfig(
            feature_dim=8,
            correlation_radius=1,
            hidden_dim=16,
            transformer_layers=1,
            attention_heads=4,
        )
    )
    path = tmp_path / "refiner.pt"
    save_structured_frame_refiner(
        path,
        model,
        {
            "vfm_layer": "radio_final",
            "stores_mapping_rgb": False,
            "uses_alike_descriptors": False,
        },
    )
    restored, metadata = load_structured_frame_refiner(path)
    assert restored.config == model.config
    assert metadata["vfm_layer"] == "radio_final"
