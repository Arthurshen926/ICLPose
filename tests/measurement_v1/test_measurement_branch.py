from __future__ import annotations

import torch

from feature_extract.vfm.measurement_v1.measurement_branch import (
    CorrelationMeasurementBranch,
    SharedFeatureProjection,
    continuous_window_nll_and_moments,
)
from feature_extract.vfm.measurement_v1.stride4_fine_feature import local_correlation_logits


def test_continuous_window_nll_recovers_subpixel_mean_and_positive_covariance() -> None:
    xs, ys = torch.meshgrid(torch.arange(5, dtype=torch.float32), torch.arange(5, dtype=torch.float32), indexing="xy")
    xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1) + 10.0
    target = torch.tensor([[12.25, 11.75]], dtype=torch.float32)
    dist2 = torch.sum((xy[None] - target[:, None]) ** 2, dim=2)
    logits = -dist2 / 0.12

    loss, pred = continuous_window_nll_and_moments(logits, xy, target)

    assert float(loss.item()) < 0.8
    assert torch.linalg.norm(pred.mean_xy_px[0] - target[0]).item() < 0.5
    assert torch.linalg.eigvalsh(pred.cov_query_2x2[0]).min().item() > 0.0
    assert pred.epe_px.item() < 0.5


def test_correlation_measurement_branch_outputs_local_likelihood_from_fixed_render_anchor() -> None:
    query = torch.zeros(1, 4, 8, 8)
    render = torch.zeros(1, 4, 8, 8)
    query[0, 2, 4, 5] = 1.0
    render[0, 2, 4, 4] = 1.0
    branch = CorrelationMeasurementBranch(search_radius_px=1.0, step_px=1.0, temperature=0.1)

    pred = branch(
        query_features=query,
        render_features=render,
        query_centers_xy=torch.tensor([[4.0, 4.0]]),
        render_anchor_xy=torch.tensor([[4.0, 4.0]]),
        image_width=8,
        image_height=8,
    )

    assert pred.local_log_probs.shape == (1, 9)
    assert torch.linalg.norm(pred.mean_xy_px[0] - torch.tensor([5.0, 4.0])).item() < 0.15
    assert pred.mode_probability[0].item() > 0.99


def test_shared_feature_projection_preserves_map_shape_and_normalizes_channels() -> None:
    projection = SharedFeatureProjection(input_dim=3, hidden_dim=8, output_dim=5)
    features = torch.randn(2, 3, 6, 7)

    out = projection(features)

    assert out.shape == (2, 5, 6, 7)
    assert torch.allclose(torch.linalg.norm(out, dim=1).mean(), torch.tensor(1.0), atol=0.1)


def test_local_correlation_logits_returns_per_sample_window_coordinates() -> None:
    query = torch.zeros(2, 1, 8, 8)
    render = torch.zeros(2, 1, 8, 8)

    logits, xy = local_correlation_logits(
        query,
        render,
        query_centers_xy=torch.tensor([[4.0, 4.0], [6.0, 4.0]]),
        render_anchor_xy=torch.tensor([[4.0, 4.0], [4.0, 4.0]]),
        image_width=8,
        image_height=8,
        search_radius_px=1.0,
        step_px=1.0,
    )

    assert logits.shape == (2, 9)
    assert xy.shape == (2, 9, 2)
    assert torch.allclose(xy[0, 4], torch.tensor([4.0, 4.0]))
    assert torch.allclose(xy[1, 4], torch.tensor([6.0, 4.0]))
