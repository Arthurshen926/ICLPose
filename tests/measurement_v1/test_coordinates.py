from __future__ import annotations

import numpy as np

from feature_extract.vfm.measurement_v1.coordinates import (
    ImageResizeTransform,
    PixelGridTransform,
)


def test_image_resize_transform_round_trips_pixels_and_covariance() -> None:
    transform = ImageResizeTransform(
        source_width=1920,
        source_height=1080,
        target_width=960,
        target_height=540,
    )
    xy = np.asarray([[100.0, 200.0], [1918.5, 1077.25]], dtype=np.float64)
    cov = np.asarray([[4.0, 1.0], [1.0, 9.0]], dtype=np.float64)

    target_xy = transform.to_target_xy(xy)
    round_trip = transform.to_source_xy(target_xy)
    target_cov = transform.covariance_to_target(cov)
    source_cov = transform.covariance_to_source(target_cov)

    assert np.allclose(round_trip, xy, atol=1e-6)
    assert np.allclose(target_cov, [[1.0, 0.25], [0.25, 2.25]], atol=1e-6)
    assert np.allclose(source_cov, cov, atol=1e-6)


def test_pixel_grid_transform_round_trips_cell_centers() -> None:
    grid = PixelGridTransform(image_width=640, image_height=480, grid_width=80, grid_height=60)
    grid_xy = np.asarray([[0.5, 0.5], [12.5, 7.5], [79.5, 59.5]], dtype=np.float64)

    pixel_xy = grid.grid_to_pixel_xy(grid_xy)
    round_trip = grid.pixel_to_grid_xy(pixel_xy)

    assert np.allclose(round_trip, grid_xy, atol=1e-6)
    assert np.allclose(pixel_xy[0], [4.0, 4.0], atol=1e-6)
