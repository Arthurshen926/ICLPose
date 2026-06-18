"""Deterministic synthetic source/target pose sampling for MATCHA-style pairs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    camera_center_from_pose_w2c,
    pose_w2c_from_center_rotation,
)
from feature_extract.vfm.render_pose_protocol import render_pose_error_fields

SYNTHETIC_PAIR_SOURCE = "2dgs_synthetic"
SYNTHETIC_RANDOM_PAIR_TYPE = "S2DGS_RANDOM"
SYNTHETIC_RANDOM_PAIR_TYPE_ID = 100
DEFAULT_SOURCE_TRANSLATION_RANGE_M = (0.0, 0.03)
DEFAULT_SOURCE_ROTATION_RANGE_DEG = (0.0, 1.0)
DEFAULT_TARGET_TRANSLATION_RANGE_M = (0.0, 0.25)
DEFAULT_TARGET_ROTATION_RANGE_DEG = (0.0, 6.0)
DEFAULT_MIN_SUPERVISION_COUNT = 128
DEFAULT_MIN_OVERLAP = 0.20


@dataclass(frozen=True)
class SyntheticPairSamplingConfig:
    source_translation_range_m: tuple[float, float] = DEFAULT_SOURCE_TRANSLATION_RANGE_M
    source_rotation_range_deg: tuple[float, float] = DEFAULT_SOURCE_ROTATION_RANGE_DEG
    target_translation_range_m: tuple[float, float] = DEFAULT_TARGET_TRANSLATION_RANGE_M
    target_rotation_range_deg: tuple[float, float] = DEFAULT_TARGET_ROTATION_RANGE_DEG
    min_supervision_count: int | None = DEFAULT_MIN_SUPERVISION_COUNT
    min_overlap: float | None = DEFAULT_MIN_OVERLAP

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_translation_range_m",
            _validate_range(self.source_translation_range_m, "source_translation_range_m"),
        )
        object.__setattr__(
            self,
            "source_rotation_range_deg",
            _validate_range(self.source_rotation_range_deg, "source_rotation_range_deg"),
        )
        object.__setattr__(
            self,
            "target_translation_range_m",
            _validate_range(self.target_translation_range_m, "target_translation_range_m"),
        )
        object.__setattr__(
            self,
            "target_rotation_range_deg",
            _validate_range(self.target_rotation_range_deg, "target_rotation_range_deg"),
        )
        if self.min_supervision_count is not None and int(self.min_supervision_count) < 0:
            raise ValueError("min_supervision_count must be non-negative")
        if self.min_supervision_count is not None:
            object.__setattr__(self, "min_supervision_count", int(self.min_supervision_count))
        if self.min_overlap is not None and float(self.min_overlap) < 0.0:
            raise ValueError("min_overlap must be in [0, 1]")
        if self.min_overlap is not None and float(self.min_overlap) > 1.0:
            raise ValueError("min_overlap must be in [0, 1]")
        if self.min_overlap is not None:
            object.__setattr__(self, "min_overlap", float(self.min_overlap))


@dataclass(frozen=True)
class SyntheticPairPose:
    source_pose_w2c: np.ndarray
    target_pose_w2c: np.ndarray
    source_anchor_translation_m: float
    source_anchor_rotation_deg: float
    target_source_translation_m: float
    target_source_rotation_deg: float
    pose_bin: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_pose_w2c", np.asarray(self.source_pose_w2c, dtype=np.float64).reshape(4, 4))
        object.__setattr__(self, "target_pose_w2c", np.asarray(self.target_pose_w2c, dtype=np.float64).reshape(4, 4))


def synthetic_config_from_metadata(metadata: Mapping[str, object]) -> SyntheticPairSamplingConfig:
    if metadata.get("pair_source") != SYNTHETIC_PAIR_SOURCE:
        raise ValueError(f"pair_source must be {SYNTHETIC_PAIR_SOURCE!r}")
    return SyntheticPairSamplingConfig(
        source_translation_range_m=_parse_range(
            metadata.get("synthetic_source_translation_range_m", DEFAULT_SOURCE_TRANSLATION_RANGE_M),
            "synthetic_source_translation_range_m",
        ),
        source_rotation_range_deg=_parse_range(
            metadata.get("synthetic_source_rotation_range_deg", DEFAULT_SOURCE_ROTATION_RANGE_DEG),
            "synthetic_source_rotation_range_deg",
        ),
        target_translation_range_m=_parse_range(
            metadata.get("synthetic_target_translation_range_m", DEFAULT_TARGET_TRANSLATION_RANGE_M),
            "synthetic_target_translation_range_m",
        ),
        target_rotation_range_deg=_parse_range(
            metadata.get("synthetic_target_rotation_range_deg", DEFAULT_TARGET_ROTATION_RANGE_DEG),
            "synthetic_target_rotation_range_deg",
        ),
        min_supervision_count=_optional_int(
            metadata.get("synthetic_min_supervision_count", DEFAULT_MIN_SUPERVISION_COUNT)
        ),
        min_overlap=_optional_float(metadata.get("synthetic_min_overlap", DEFAULT_MIN_OVERLAP)),
    )


def relative_pose_bin(translation_m: float, rotation_deg: float) -> str:
    translation = float(translation_m)
    rotation = float(rotation_deg)
    if translation <= 0.03 and rotation <= 1.0:
        return "micro"
    if translation <= 0.10 and rotation <= 3.0:
        return "small"
    if translation <= 0.25 and rotation <= 6.0:
        return "medium"
    if translation <= 0.50 and rotation <= 10.0:
        return "wide"
    return "out_of_range"


def sample_synthetic_pair_poses(
    anchor_pose_w2c: np.ndarray,
    *,
    config: SyntheticPairSamplingConfig,
    seed: int,
    key: str,
) -> SyntheticPairPose:
    anchor_pose = np.asarray(anchor_pose_w2c, dtype=np.float64).reshape(4, 4)
    rng = np.random.default_rng(_stable_seed(seed, key))
    source_pose = _sample_pose_around(
        anchor_pose,
        translation_range_m=config.source_translation_range_m,
        rotation_range_deg=config.source_rotation_range_deg,
        rng=rng,
    )
    target_pose = _sample_pose_around(
        source_pose,
        translation_range_m=config.target_translation_range_m,
        rotation_range_deg=config.target_rotation_range_deg,
        rng=rng,
    )
    source_t, source_r = render_pose_error_fields(source_pose, anchor_pose)
    target_t, target_r = render_pose_error_fields(target_pose, source_pose)
    return SyntheticPairPose(
        source_pose_w2c=source_pose,
        target_pose_w2c=target_pose,
        source_anchor_translation_m=source_t,
        source_anchor_rotation_deg=source_r,
        target_source_translation_m=target_t,
        target_source_rotation_deg=target_r,
        pose_bin=relative_pose_bin(target_t, target_r),
    )


def _sample_pose_around(
    pose_w2c: np.ndarray,
    *,
    translation_range_m: tuple[float, float],
    rotation_range_deg: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    translation = _sample_vector_with_magnitude(rng, translation_range_m)
    delta_rotation = _sample_axis_angle_rotation(rng, rotation_range_deg)
    center = camera_center_from_pose_w2c(pose_w2c) + translation
    rotation = delta_rotation @ pose_w2c[:3, :3]
    return pose_w2c_from_center_rotation(center, rotation)


def _sample_vector_with_magnitude(rng: np.random.Generator, value_range: tuple[float, float]) -> np.ndarray:
    magnitude = float(rng.uniform(value_range[0], value_range[1]))
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    return direction * magnitude


def _sample_axis_angle_rotation(rng: np.random.Generator, angle_range_deg: tuple[float, float]) -> np.ndarray:
    axis = _sample_unit_vector(rng)
    angle_deg = float(rng.uniform(angle_range_deg[0], angle_range_deg[1]))
    return _axis_angle_to_rotation(axis, angle_deg)


def _sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def _axis_angle_to_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(unit_axis))
    if norm <= 1e-12:
        unit_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        unit_axis = unit_axis / norm
    x, y, z = unit_axis.tolist()
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _parse_range(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"{name} must be a two-value range")
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return _validate_range((float(items[0]), float(items[1])), name)


def _validate_range(value: tuple[float, float], name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value range")
    minimum = float(value[0])
    maximum = float(value[1])
    if minimum < 0.0 or maximum < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if minimum > maximum:
        raise ValueError(f"{name} must be ordered min,max")
    return (minimum, maximum)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{str(key)}".encode("utf8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFF
