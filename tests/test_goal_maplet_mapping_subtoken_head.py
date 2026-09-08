import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import (
    MappingSurfaceCoordinateHead,
    MappingSurfaceCoordinateContextHead,
    MappingSurfaceCoordinateDeepContextHead,
    MappingSurfaceCoordinateHomographyContextHead,
    MappingSurfaceCoordinateLocalCorrelationHead,
    MappingSurfaceCoordinateMixtureHead,
    MappingSubtokenHead,
    _directed_pairs,
    _metrics,
    _project_world_to_pixel,
    _token_centres,
)


def test_surface_coordinate_head_has_bounded_image_and_chart_outputs():
    model = MappingSurfaceCoordinateHead(4, hidden_dimension=8)
    query = torch.randn(3, 4)
    mapping = torch.randn(3, 4)
    token = torch.asarray([0, 17, 2303])
    image, image_variance, chart, chart_variance, match = model(query, mapping, token)
    assert image.shape == chart.shape == (3, 2)
    assert image_variance.shape == chart_variance.shape == match.shape == (3, 1)
    assert torch.all(torch.abs(image) <= 2.0)
    assert torch.all(torch.abs(chart) <= 0.5)
    assert torch.all(image_variance > 0.0)
    assert torch.all(chart_variance > 0.0)


def test_surface_coordinate_mixture_head_has_normalized_multimodal_contract():
    model = MappingSurfaceCoordinateMixtureHead(4, hidden_dimension=8, mixture_modes=3)
    query = torch.nn.functional.normalize(torch.randn(5, 4), dim=1)
    mapping = torch.nn.functional.normalize(torch.randn(5, 4), dim=1)
    image, image_variance, chart, chart_variance, logits, match = model(
        query, mapping, torch.arange(5),
    )
    assert image.shape == (5, 2) and image_variance.shape == (5, 1)
    assert chart.shape == (5, 3, 2) and chart_variance.shape == logits.shape == (5, 3)
    assert match.shape == (5, 1)
    assert torch.all(torch.abs(chart) <= 0.5) and torch.all(chart_variance > 0.0)
    torch.testing.assert_close(torch.softmax(logits, dim=1).sum(1), torch.ones(5))


def test_surface_coordinate_context_head_requires_five_finite_geometry_values():
    model = MappingSurfaceCoordinateContextHead(4, hidden_dimension=8)
    query = torch.randn(3, 4); mapping = torch.randn(3, 4); token = torch.arange(3)
    output = model(query, mapping, token, torch.zeros(3, 5))
    assert output[0].shape == output[2].shape == (3, 2)
    with pytest.raises(ValueError, match="geometric context"):
        model(query, mapping, token, torch.zeros(3, 4))


def test_deep_surface_coordinate_context_head_has_two_hidden_layers_and_same_contract():
    model = MappingSurfaceCoordinateDeepContextHead(4, hidden_dimension=12)
    query = torch.randn(3, 4); mapping = torch.randn(3, 4); token = torch.arange(3)
    output = model(query, mapping, token, torch.zeros(3, 5))
    assert output[0].shape == output[2].shape == (3, 2)
    assert output[1].shape == output[3].shape == output[4].shape == (3, 1)
    assert model.hidden_1.out_features == model.hidden_2.in_features == 12
    with pytest.raises(ValueError, match="geometric context"):
        model(query, mapping, token, torch.full((3, 5), float("nan")))


def test_homography_surface_coordinate_context_head_requires_eight_values():
    model = MappingSurfaceCoordinateHomographyContextHead(4, hidden_dimension=8)
    query = torch.randn(3, 4); mapping = torch.randn(3, 4); token = torch.arange(3)
    output = model(query, mapping, token, torch.zeros(3, 8))
    assert output[0].shape == output[2].shape == (3, 2)
    with pytest.raises(ValueError, match="homography context"):
        model(query, mapping, token, torch.zeros(3, 7))


def test_local_correlation_head_requires_full_candidate_conditioned_volume():
    model = MappingSurfaceCoordinateLocalCorrelationHead(4, hidden_dimension=8)
    query = torch.randn(3, 4); mapping = torch.randn(3, 4); token = torch.arange(3)
    output = model(query, mapping, token, torch.zeros(3, 107))
    assert output[0].shape == output[2].shape == (3, 2)
    with pytest.raises(ValueError, match="local-correlation context"):
        model(query, mapping, token, torch.zeros(3, 106))


def test_project_world_to_pixel_and_token_phase():
    pose = np.eye(4)[None]
    K = np.asarray([[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]])
    pixel, depth = _project_world_to_pixel(np.asarray([[0.0, 0.0, 2.0]]), pose, K, 0.0)
    np.testing.assert_allclose(pixel, [[127.5, 71.5]])
    np.testing.assert_allclose(depth, [2.0])
    np.testing.assert_allclose(_token_centres(np.asarray([0, 64, 2303])), [[1.5, 1.5], [1.5, 5.5], [253.5, 141.5]])


def test_directed_pairs_are_cross_observation_and_capped():
    rows = np.arange(12, dtype=np.int64)
    identities = np.asarray([0] * 10 + [1] * 2, np.int64)
    query, mapping = _directed_pairs(rows, identities, maximum_views=4)
    assert len(query) == 12
    assert np.all(query != mapping)
    assert set(query[-4:].tolist()) == {10, 11}


def test_metrics_requires_real_coordinate_improvement():
    target = np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32)
    exact = _metrics(target, target, np.ones((2, 1), np.float32))
    centre = _metrics(target, np.zeros_like(target), np.ones((2, 1), np.float32))
    assert exact["relative_median_improvement"] == pytest.approx(1.0)
    assert exact["predicted_better_fraction"] == 1.0
    assert centre["predicted_better_fraction"] == 0.0


def test_head_bounds_coordinates_and_positive_variance():
    head = MappingSubtokenHead(64, 16)
    q = torch.randn(5, 64); q = torch.nn.functional.normalize(q, dim=1)
    m = torch.randn(5, 64); m = torch.nn.functional.normalize(m, dim=1)
    mean, variance, logit = head(q, m, torch.arange(5))
    assert mean.shape == (5, 2) and variance.shape == (5, 1) and logit.shape == (5, 1)
    assert torch.max(torch.abs(mean)).item() < 2.0
    assert torch.min(variance).item() >= 0.01
