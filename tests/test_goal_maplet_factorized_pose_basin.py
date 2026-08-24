from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_basin import (
    ContinuousHierarchicalPoseBasin,
    build_default_factorized_pose_basin,
    continuous_factorized_oracle_error,
    factorized_oracle_error,
)


def _pose(rotation: np.ndarray | None = None, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.eye(3) if rotation is None else rotation
    value[:3, 3] = -value[:3, :3] @ np.asarray(center, dtype=np.float64)
    return value


def _rz(degrees: float) -> np.ndarray:
    value = np.deg2rad(float(degrees))
    return np.asarray([
        [np.cos(value), -np.sin(value), 0.0],
        [np.sin(value), np.cos(value), 0.0],
        [0.0, 0.0, 1.0],
    ])


def test_default_factorized_domain_is_fixed_and_preserves_seed():
    first = build_default_factorized_pose_basin()
    second = build_default_factorized_pose_basin()
    np.testing.assert_array_equal(first.translation_offsets_world, second.translation_offsets_world)
    np.testing.assert_array_equal(first.rotation_offsets_left, second.rotation_offsets_left)
    assert first.translation_offsets_world.shape == (125, 3)
    assert first.rotation_offsets_left.shape == (105, 3, 3)
    assert first.implicit_pose_count_per_seed == 13_125


def test_factorized_oracle_recovers_combined_fixed_position_and_rotation_offset():
    domain = build_default_factorized_pose_basin()
    seed = _pose()
    target = _pose(_rz(20.0), center=(0.75, -1.5, 0.0))
    translation, rotation, seed_row, position_row, rotation_row = factorized_oracle_error(
        seed[None], target, domain
    )
    assert translation < 1.0e-10
    assert rotation < 1.0e-5
    assert seed_row == 0
    np.testing.assert_allclose(domain.translation_offsets_world[position_row], [0.75, -1.5, 0.0])
    assert rotation_row != 0


def test_factorized_oracle_selects_one_common_seed_for_both_factors():
    domain = build_default_factorized_pose_basin()
    seeds = np.stack([_pose(center=(-10.0, 0.0, 0.0)), _pose(_rz(40.0), center=(1.0, 0.0, 0.0))])
    target = _pose(_rz(10.0), center=(1.0, 0.0, 0.0))
    translation, rotation, seed_row, _, _ = factorized_oracle_error(seeds, target, domain)
    assert seed_row == 1
    assert translation < 1.0e-10
    assert rotation < 1.0e-5


def test_continuous_hierarchical_domain_contains_interior_pose_without_enumeration():
    domain = ContinuousHierarchicalPoseBasin(
        translation_half_extent_m=8.0,
        rotation_radius_deg=45.0,
    )
    target = _pose(_rz(35.0), center=(7.5, -2.0, 3.0))
    translation, rotation, seed_row = continuous_factorized_oracle_error(
        _pose()[None], target, domain,
    )
    assert seed_row == 0
    assert translation < 1.0e-10
    assert rotation < 1.0e-5


def test_continuous_hierarchical_domain_reports_distance_beyond_closed_bounds():
    domain = ContinuousHierarchicalPoseBasin(
        translation_half_extent_m=8.0,
        rotation_radius_deg=45.0,
    )
    target = _pose(_rz(60.0), center=(9.0, 0.0, 0.0))
    translation, rotation, seed_row = continuous_factorized_oracle_error(
        _pose()[None], target, domain,
    )
    assert seed_row == 0
    np.testing.assert_allclose(translation, 1.0, atol=1.0e-10)
    np.testing.assert_allclose(rotation, 15.0, atol=1.0e-8)
