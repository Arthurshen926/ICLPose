"""Query-independent factorized local domains around retrieved pose basins.

The domain is a Cartesian product conceptually, but is kept factorized so a
backend can search position and orientation hierarchically without rendering
the full product.  It is a support construction, not a pose scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


FACTORIZED_POSE_BASIN_SEMANTICS = (
    "world_translation_grid_x_left_camera_axis_angle_orientation_product_v1"
)
CONTINUOUS_HIERARCHICAL_POSE_BASIN_SEMANTICS = (
    "continuous_world_center_box_x_left_camera_geodesic_ball_v2"
)


@dataclass(frozen=True)
class FactorizedPoseBasin:
    translation_offsets_world: np.ndarray
    rotation_offsets_left: np.ndarray
    semantics: str = FACTORIZED_POSE_BASIN_SEMANTICS

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_offsets_world)
        rotation = np.asarray(self.rotation_offsets_left)
        if translation.ndim != 2 or translation.shape[1] != 3:
            raise ValueError("translation offsets must have shape [N,3]")
        if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
            raise ValueError("rotation offsets must have shape [M,3,3]")
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            raise ValueError("factorized pose offsets must be finite")
        identity = np.eye(3, dtype=np.float64)
        if np.max(np.abs(rotation @ np.swapaxes(rotation, 1, 2) - identity)) > 1.0e-9:
            raise ValueError("rotation offsets must be orthonormal")
        if np.any(np.linalg.det(rotation) < 1.0 - 1.0e-9):
            raise ValueError("rotation offsets must be proper")
        if not np.any(np.linalg.norm(translation, axis=1) == 0.0):
            raise ValueError("translation domain must preserve its seed")
        if not np.any(np.max(np.abs(rotation - identity), axis=(1, 2)) == 0.0):
            raise ValueError("orientation domain must preserve its seed")

    @property
    def implicit_pose_count_per_seed(self) -> int:
        return int(self.translation_offsets_world.shape[0] * self.rotation_offsets_left.shape[0])


@dataclass(frozen=True)
class ContinuousHierarchicalPoseBasin:
    """A continuous local domain intended for lazy hierarchical expansion.

    The world-coordinate camera centre lives in a closed cube around the seed.
    The orientation lives in a closed left-camera SO(3) geodesic ball.  This
    class deliberately stores bounds rather than an enumerated pose grid.
    """

    translation_half_extent_m: float = 8.0
    rotation_radius_deg: float = 45.0
    semantics: str = CONTINUOUS_HIERARCHICAL_POSE_BASIN_SEMANTICS

    def __post_init__(self) -> None:
        extent = float(self.translation_half_extent_m)
        radius = float(self.rotation_radius_deg)
        if not np.isfinite(extent) or extent <= 0.0:
            raise ValueError("translation half extent must be finite and positive")
        if not np.isfinite(radius) or not 0.0 < radius < 180.0:
            raise ValueError("rotation radius must be finite and in (0,180) degrees")


def _so3_exp(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64).reshape(3)
    unit /= np.linalg.norm(unit)
    omega = unit * float(angle_rad)
    theta = float(np.linalg.norm(omega))
    x, y, z = omega
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) / theta * skew
        + (1.0 - np.cos(theta)) / theta**2 * (skew @ skew)
    )


def build_default_factorized_pose_basin() -> FactorizedPoseBasin:
    """Return the frozen P1 diagnostic support grid.

    Position uses a 0.75 m world-coordinate grid over +/-1.5 m.  Orientation
    uses all signed 26 cubic directions and 10/20/30/40 degree left-camera
    increments.  The grid is deliberately simple and query independent.
    """

    translation = np.asarray(
        list(product((-1.5, -0.75, 0.0, 0.75, 1.5), repeat=3)),
        dtype=np.float64,
    )
    axes = []
    for xyz in product((-1.0, 0.0, 1.0), repeat=3):
        value = np.asarray(xyz, dtype=np.float64)
        if np.linalg.norm(value) > 0.0:
            axes.append(value / np.linalg.norm(value))
    rotation = [np.eye(3, dtype=np.float64)]
    for angle_deg in (10.0, 20.0, 30.0, 40.0):
        for axis in axes:
            rotation.append(_so3_exp(axis, np.deg2rad(angle_deg)))
    return FactorizedPoseBasin(
        translation_offsets_world=translation,
        rotation_offsets_left=np.asarray(rotation, dtype=np.float64),
    )


def factorized_oracle_error(
    seed_poses_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
    domain: FactorizedPoseBasin,
) -> tuple[float, float, int, int, int]:
    """Return separable best errors and the common seed that minimises joint scale.

    GT is used only by this evaluator.  Domain generation above has no target
    input.  The returned position and orientation indices define a concrete
    pose in the implicit product for the selected seed.
    """

    seeds = np.asarray(seed_poses_w2c, dtype=np.float64)
    target = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    if seeds.ndim != 3 or seeds.shape[1:] != (4, 4) or seeds.shape[0] == 0:
        raise ValueError("seed poses must have shape [N,4,4]")
    if not np.isfinite(seeds).all() or not np.isfinite(target).all():
        raise ValueError("pose matrices must be finite")
    target_rotation = target[:3, :3]
    target_center = -target_rotation.T @ target[:3, 3]
    best = None
    for seed_index, seed in enumerate(seeds):
        rotation = seed[:3, :3]
        center = -rotation.T @ seed[:3, 3]
        position_error = np.linalg.norm(
            center[None] + domain.translation_offsets_world - target_center[None],
            axis=1,
        )
        position_index = int(np.argmin(position_error))
        proposed_rotation = domain.rotation_offsets_left @ rotation
        relative = proposed_rotation @ target_rotation.T
        cosine = np.clip(
            (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        rotation_error = np.degrees(np.arccos(cosine))
        rotation_index = int(np.argmin(rotation_error))
        row = (
            max(float(position_error[position_index]), float(rotation_error[rotation_index]) / 10.0),
            float(position_error[position_index]),
            float(rotation_error[rotation_index]),
            int(seed_index),
            position_index,
            rotation_index,
        )
        if best is None or row < best:
            best = row
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]


def continuous_factorized_oracle_error(
    seed_poses_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
    domain: ContinuousHierarchicalPoseBasin,
) -> tuple[float, float, int]:
    """Return residual error to the nearest continuous seed-centred domain.

    This is an acquisition/support oracle only.  It projects the target camera
    centre onto the seed cube and the relative target orientation onto the
    boundary of the geodesic ball.  It does not enumerate, rank, or search a
    pose and therefore cannot be reported as localization success.
    """

    seeds = np.asarray(seed_poses_w2c, dtype=np.float64)
    target = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    if seeds.ndim != 3 or seeds.shape[1:] != (4, 4) or seeds.shape[0] == 0:
        raise ValueError("seed poses must have shape [N,4,4]")
    if not np.isfinite(seeds).all() or not np.isfinite(target).all():
        raise ValueError("pose matrices must be finite")
    target_rotation = target[:3, :3]
    target_center = -target_rotation.T @ target[:3, 3]
    best = None
    extent = float(domain.translation_half_extent_m)
    radius = float(domain.rotation_radius_deg)
    for seed_index, seed in enumerate(seeds):
        seed_rotation = seed[:3, :3]
        seed_center = -seed_rotation.T @ seed[:3, 3]
        delta = target_center - seed_center
        projected_delta = np.clip(delta, -extent, extent)
        translation_residual = float(np.linalg.norm(delta - projected_delta))
        relative = seed_rotation @ target_rotation.T
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cosine)))
        rotation_residual = max(angle_deg - radius, 0.0)
        row = (
            max(translation_residual / 0.5, rotation_residual / 5.0),
            translation_residual,
            rotation_residual,
            int(seed_index),
        )
        if best is None or row < best:
            best = row
    assert best is not None
    return best[1], best[2], best[3]
