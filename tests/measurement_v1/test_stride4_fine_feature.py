from __future__ import annotations

import torch

from feature_extract.vfm.measurement_v1.stride4_fine_feature import (
    Stride4FineFeatureExtractor,
    local_correlation_logits,
)


def test_stride4_fine_feature_extractor_encodes_whole_image_once() -> None:
    model = Stride4FineFeatureExtractor(output_dim=64, base_dim=16)
    images = torch.randn(2, 3, 64, 80)

    features = model(images)

    assert features.shape == (2, 64, 16, 20)
    norms = torch.linalg.norm(features, dim=1)
    assert torch.allclose(norms.mean(), torch.tensor(1.0), atol=0.1)


def test_local_correlation_logits_scores_query_window_from_fixed_render_descriptor() -> None:
    query = torch.zeros(1, 4, 8, 8)
    render = torch.zeros(1, 4, 8, 8)
    query[0, 2, 4, 5] = 1.0
    render[0, 2, 4, 4] = 1.0

    logits, xy = local_correlation_logits(
        query,
        render,
        query_centers_xy=torch.tensor([[4.0, 4.0]]),
        render_anchor_xy=torch.tensor([[4.0, 4.0]]),
        image_width=8,
        image_height=8,
        search_radius_px=1.0,
        step_px=1.0,
    )

    assert logits.shape == (1, 9)
    assert torch.argmax(logits, dim=1).item() == 5
    assert torch.allclose(xy[5], torch.tensor([5.0, 4.0]))
