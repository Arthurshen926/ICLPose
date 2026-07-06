from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _readonly_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(shape).copy()
    arr.setflags(write=False)
    return arr


def _optional_readonly_array(value: np.ndarray | None, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _readonly_array(value, shape, name)


def _probability(value: float, name: str) -> float:
    prob = float(value)
    if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return prob


@dataclass(frozen=True)
class SurfaceAnchor:
    """Immutable render-surface 3D anchor used by measurement_v1.

    Arrays are copied and marked read-only. Query-side fine predictions may
    create new QueryMeasurement instances but must never mutate world_xyz.
    """

    anchor_id: int
    token_index: int
    subanchor_index: int
    render_xy_px: np.ndarray
    world_xyz: np.ndarray
    cov_world_3x3: np.ndarray
    depth_m: float
    alpha: float
    quality: float
    normal_world: np.ndarray | None
    surface_id: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_id", int(self.anchor_id))
        object.__setattr__(self, "token_index", int(self.token_index))
        object.__setattr__(self, "subanchor_index", int(self.subanchor_index))
        object.__setattr__(self, "render_xy_px", _readonly_array(self.render_xy_px, (2,), "render_xy_px"))
        object.__setattr__(self, "world_xyz", _readonly_array(self.world_xyz, (3,), "world_xyz"))
        object.__setattr__(self, "cov_world_3x3", _readonly_array(self.cov_world_3x3, (3, 3), "cov_world_3x3"))
        object.__setattr__(self, "depth_m", float(self.depth_m))
        object.__setattr__(self, "alpha", _probability(float(self.alpha), "alpha"))
        object.__setattr__(self, "quality", _probability(float(self.quality), "quality"))
        object.__setattr__(self, "normal_world", _optional_readonly_array(self.normal_world, (3,), "normal_world"))
        if self.surface_id is not None:
            object.__setattr__(self, "surface_id", int(self.surface_id))
        if not np.isfinite(float(self.depth_m)) or float(self.depth_m) <= 0.0:
            raise ValueError("depth_m must be positive and finite")


@dataclass(frozen=True)
class QueryMeasurement:
    """Query-side observation likelihood attached to a SurfaceAnchor."""

    anchor_id: int
    query_xy_mean_px: np.ndarray
    cov_query_2x2: np.ndarray
    p_visible: float
    p_assignment: float
    local_log_likelihood: np.ndarray | None
    mode_probability: float
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_id", int(self.anchor_id))
        object.__setattr__(self, "query_xy_mean_px", _readonly_array(self.query_xy_mean_px, (2,), "query_xy_mean_px"))
        object.__setattr__(self, "cov_query_2x2", _readonly_array(self.cov_query_2x2, (2, 2), "cov_query_2x2"))
        object.__setattr__(self, "p_visible", _probability(float(self.p_visible), "p_visible"))
        object.__setattr__(self, "p_assignment", _probability(float(self.p_assignment), "p_assignment"))
        if self.local_log_likelihood is not None:
            ll = np.asarray(self.local_log_likelihood, dtype=np.float64).copy()
            ll.setflags(write=False)
            object.__setattr__(self, "local_log_likelihood", ll)
        object.__setattr__(self, "mode_probability", _probability(float(self.mode_probability), "mode_probability"))
        object.__setattr__(self, "diagnostics", {str(k): float(v) for k, v in dict(self.diagnostics).items()})
        eigvals = np.linalg.eigvalsh(np.asarray(self.cov_query_2x2, dtype=np.float64))
        if not np.all(np.isfinite(eigvals)) or float(np.min(eigvals)) <= 0.0:
            raise ValueError("cov_query_2x2 must be positive definite")

    @property
    def p_valid(self) -> float:
        return float(self.p_visible) * float(self.p_assignment)


@dataclass(frozen=True)
class PoseEstimate:
    """Pose estimate produced from measurement_v1 measurements."""

    pose_w2c: np.ndarray | None
    pose_cov_6x6: np.ndarray | None
    success: bool
    objective: float
    inlier_probability: np.ndarray
    hessian_condition: float
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.pose_w2c is not None:
            object.__setattr__(self, "pose_w2c", _readonly_array(self.pose_w2c, (4, 4), "pose_w2c"))
        if self.pose_cov_6x6 is not None:
            object.__setattr__(self, "pose_cov_6x6", _readonly_array(self.pose_cov_6x6, (6, 6), "pose_cov_6x6"))
        probs = np.asarray(self.inlier_probability, dtype=np.float64).reshape(-1).copy()
        if probs.size and (not np.all(np.isfinite(probs)) or np.min(probs) < 0.0 or np.max(probs) > 1.0):
            raise ValueError("inlier_probability must contain probabilities in [0, 1]")
        probs.setflags(write=False)
        object.__setattr__(self, "inlier_probability", probs)
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "hessian_condition", float(self.hessian_condition))
        object.__setattr__(self, "diagnostics", {str(k): float(v) for k, v in dict(self.diagnostics).items()})
