from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _xy_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[-1] != 2:
        raise ValueError("xy arrays must have final dimension 2")
    return arr


@dataclass(frozen=True)
class ImageResizeTransform:
    """Affine resize between two image pixel coordinate systems."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int

    @property
    def scale_x(self) -> float:
        return float(self.target_width) / float(self.source_width)

    @property
    def scale_y(self) -> float:
        return float(self.target_height) / float(self.source_height)

    @property
    def jacobian_to_target(self) -> np.ndarray:
        return np.asarray([[self.scale_x, 0.0], [0.0, self.scale_y]], dtype=np.float64)

    def to_target_xy(self, xy: np.ndarray) -> np.ndarray:
        arr = _xy_array(xy)
        scale = np.asarray([self.scale_x, self.scale_y], dtype=np.float64)
        return arr * scale

    def to_source_xy(self, xy: np.ndarray) -> np.ndarray:
        arr = _xy_array(xy)
        scale = np.asarray([self.scale_x, self.scale_y], dtype=np.float64)
        return arr / scale

    def covariance_to_target(self, cov_2x2: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov_2x2, dtype=np.float64).reshape(2, 2)
        jac = self.jacobian_to_target
        return jac @ cov @ jac.T

    def covariance_to_source(self, cov_2x2: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov_2x2, dtype=np.float64).reshape(2, 2)
        jac = np.linalg.inv(self.jacobian_to_target)
        return jac @ cov @ jac.T


@dataclass(frozen=True)
class PixelGridTransform:
    """Map between full-resolution pixels and continuous feature-grid cells."""

    image_width: int
    image_height: int
    grid_width: int
    grid_height: int

    @property
    def stride_x(self) -> float:
        return float(self.image_width) / float(self.grid_width)

    @property
    def stride_y(self) -> float:
        return float(self.image_height) / float(self.grid_height)

    def grid_to_pixel_xy(self, grid_xy: np.ndarray) -> np.ndarray:
        arr = _xy_array(grid_xy)
        return arr * np.asarray([self.stride_x, self.stride_y], dtype=np.float64)

    def pixel_to_grid_xy(self, pixel_xy: np.ndarray) -> np.ndarray:
        arr = _xy_array(pixel_xy)
        return arr / np.asarray([self.stride_x, self.stride_y], dtype=np.float64)
