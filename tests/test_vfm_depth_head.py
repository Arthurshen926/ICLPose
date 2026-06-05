from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap
from feature_extract.vfm.vfm_depth_head import (
    RADIO_TOKEN_DEPTH_INVALID,
    RadioTokenDepthHead,
    TokenDepthRasterConfig,
    compute_depth_metrics,
    masked_log_depth_l1,
    render_surface_token_depth_label,
    rasterize_surface_token_depth,
    splat_surface_token_depth,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=8, height=8, params=(8.0, 8.0, 4.0, 4.0))


def _surface(points: np.ndarray) -> SurfaceElementMap:
    count = int(points.shape[0])
    return SurfaceElementMap(
        element_ids=np.arange(count, dtype=np.int64),
        parent_gaussian_indices=np.arange(count, dtype=np.int64),
        centers=points.astype(np.float64),
        tangent1=np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        tangent2=np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (count, 1)),
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        scale1=np.ones((count,), dtype=np.float32),
        scale2=np.ones((count,), dtype=np.float32),
        opacity=np.ones((count,), dtype=np.float32),
        area=np.ones((count,), dtype=np.float32),
        adjacency=tuple(np.zeros((0,), dtype=np.int64) for _ in range(count)),
    )


def test_rasterize_surface_token_depth_uses_front_surface_median() -> None:
    # Both points project into the same 2x2 token. The nearer point must win the
    # token depth rather than being averaged with the far layer.
    points = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 8.0],
            [1.5, 1.5, 4.0],
        ],
        dtype=np.float64,
    )
    depth, valid = rasterize_surface_token_depth(
        _surface(points),
        np.eye(4, dtype=np.float64),
        _camera(),
        token_height=2,
        token_width=2,
        config=TokenDepthRasterConfig(depth_epsilon=0.25, min_points_per_token=1),
    )

    assert depth.shape == (2, 2)
    assert valid.shape == (2, 2)
    assert valid.any()
    assert np.nanmin(np.where(valid, depth, np.nan)) == np.float32(4.0)
    assert np.all(depth[~valid] == RADIO_TOKEN_DEPTH_INVALID)


def test_splat_surface_token_depth_covers_projected_disk_footprint() -> None:
    points = np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64)
    surface = _surface(points)
    # A wide 2DGS disk should supervise several token cells, not only the cell
    # that contains its center projection.
    object.__setattr__(surface, "scale1", np.asarray([1.0], dtype=np.float32))
    object.__setattr__(surface, "scale2", np.asarray([1.0], dtype=np.float32))

    center_depth, center_valid = rasterize_surface_token_depth(
        surface,
        np.eye(4, dtype=np.float64),
        _camera(),
        token_height=8,
        token_width=8,
        config=TokenDepthRasterConfig(min_points_per_token=1),
    )
    splat_depth, splat_valid = splat_surface_token_depth(
        surface,
        np.eye(4, dtype=np.float64),
        _camera(),
        token_height=8,
        token_width=8,
        config=TokenDepthRasterConfig(min_points_per_token=1),
        sigma_scale=1.0,
        min_opacity=0.0,
    )

    assert int(center_valid.sum()) == 1
    assert int(splat_valid.sum()) > int(center_valid.sum())
    assert np.all(splat_depth[splat_valid] == np.float32(4.0))


def test_render_surface_token_depth_label_reports_confidence_and_coverage_reason() -> None:
    points = np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64)
    surface = _surface(points)
    object.__setattr__(surface, "scale1", np.asarray([1.0], dtype=np.float32))
    object.__setattr__(surface, "scale2", np.asarray([1.0], dtype=np.float32))

    label = render_surface_token_depth_label(
        surface,
        np.eye(4, dtype=np.float64),
        _camera(),
        token_height=8,
        token_width=8,
        config=TokenDepthRasterConfig(min_points_per_token=1),
        sigma_scale=1.0,
    )

    assert label.depth.shape == (8, 8)
    assert label.valid.shape == (8, 8)
    assert label.confidence.shape == (8, 8)
    assert label.support_count.shape == (8, 8)
    assert int(label.valid.sum()) > int(label.center_valid.sum())
    assert int(label.filled_by_splat.sum()) == int(label.valid.sum() - label.center_valid.sum())
    assert np.all(label.confidence[label.valid] > 0.0)


def test_render_surface_token_depth_label_vectorized_radius_cap_path() -> None:
    points = np.asarray([[0.0, 0.0, 4.0], [0.1, 0.1, 4.05]], dtype=np.float64)
    surface = _surface(points)
    object.__setattr__(surface, "scale1", np.asarray([1.0, 1.0], dtype=np.float32))
    object.__setattr__(surface, "scale2", np.asarray([1.0, 1.0], dtype=np.float32))

    label = render_surface_token_depth_label(
        surface,
        np.eye(4, dtype=np.float64),
        _camera(),
        token_height=8,
        token_width=8,
        config=TokenDepthRasterConfig(depth_epsilon=0.10, min_points_per_token=1),
        sigma_scale=1.0,
        max_radius_tokens=3.0,
    )

    assert label.depth.shape == (8, 8)
    assert int(label.valid.sum()) > 1
    assert int(label.support_count[label.valid].max()) >= 1
    assert np.all(label.confidence[label.valid] > 0.0)


def test_depth_metrics_ignore_invalid_tokens() -> None:
    pred = np.asarray([[2.0, 4.0], [8.0, 1.0]], dtype=np.float32)
    target = np.asarray([[2.0, 2.0], [RADIO_TOKEN_DEPTH_INVALID, 1.0]], dtype=np.float32)
    valid = target > 0.0

    metrics = compute_depth_metrics(pred, target, valid)

    assert metrics["valid_count"] == 3
    assert metrics["abs_rel"] > 0.0
    assert metrics["delta1"] < 1.0


def test_radio_token_depth_head_forward_and_masked_loss() -> None:
    model = RadioTokenDepthHead(in_channels=4, hidden_channels=8)
    tokens = torch.randn(2, 4, 3, 5)
    target = torch.exp(torch.randn(2, 3, 5))
    valid = torch.ones(2, 3, 5, dtype=torch.bool)
    valid[0, 0, 0] = False

    pred = model(tokens)
    loss = masked_log_depth_l1(pred, target, valid)

    assert pred.shape == (2, 3, 5)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
    assert float(loss.detach().cpu()) >= 0.0
