from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_map_density import (
    _compose_union_support,
    _midpoint_pose,
    _robust_refine,
    _spatial_balance_weights,
    _unique_token_inliers,
)


def test_union_support_counts_each_query_token_once() -> None:
    pose = np.eye(4, dtype=np.float64)
    # Both 3D points project to the same 2D token; only the better one votes.
    world = np.asarray([[0.0, 0.0, 2.0], [0.01, 0.0, 2.0]], np.float64)
    token = np.asarray([0, 0], np.int64)
    K = np.asarray([[1.0, 0.0, 0.566666667], [0.0, 1.0, 0.558823529], [0.0, 0.0, 1.0]])
    rows, residual = _unique_token_inliers(
        pose, world, token, K, 0.0, (68, 120), maximum_reprojection_error_px=4.0,
    )
    assert rows.tolist() == [0]
    assert len(residual) == 1


def test_union_support_rejects_points_behind_camera() -> None:
    rows, _ = _unique_token_inliers(
        np.eye(4), np.asarray([[0.0, 0.0, -2.0]]), np.asarray([0]),
        np.eye(3), 0.0, (68, 120), maximum_reprojection_error_px=100.0,
    )
    assert len(rows) == 0


def test_robust_refinement_is_finite_and_deterministic() -> None:
    tokens = np.asarray([0, 10, 20, 1200, 2400, 3600, 5000, 7000], np.int64)
    K = np.asarray([[100.0, 0.0, 127.5], [0.0, 100.0, 71.5], [0.0, 0.0, 1.0]])
    pixel = np.c_[
        (tokens % 120 + 0.5) * 256.0 / 120 - 0.5,
        (tokens // 120 + 0.5) * 144.0 / 68 - 0.5,
    ]
    depth = np.full(len(tokens), 3.0)
    world = np.c_[
        (pixel[:, 0] - K[0, 2]) * depth / K[0, 0],
        (pixel[:, 1] - K[1, 2]) * depth / K[1, 1],
        depth,
    ]
    initial = np.eye(4, dtype=np.float64)
    initial[0, 3] = 0.02
    rows = np.arange(len(tokens), dtype=np.int64)
    first = _robust_refine(initial, rows, world, tokens, K, 0.0, (68, 120))
    second = _robust_refine(initial, rows, world, tokens, K, 0.0, (68, 120))
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert abs(first[0, 3]) < abs(initial[0, 3])


def test_spatial_balance_equalizes_macrocell_mass_and_is_bounded() -> None:
    # Four tokens in the top-left macrocell and one in the bottom-right.
    tokens = np.asarray([0, 1, 120, 121, 67 * 120 + 119], np.int64)
    weights = _spatial_balance_weights(tokens, (68, 120), "macrocell_equal_mass_4x6")
    assert np.all(weights >= 0.5)
    assert np.all(weights <= 2.0)
    assert weights[-1] > weights[0]
    assert np.array_equal(
        _spatial_balance_weights(tokens, (68, 120), "none"),
        np.ones(len(tokens)),
    )
    for mode in (
        "macrocell_equal_mass_3x5", "macrocell_equal_mass_5x8",
        "macrocell_equal_mass_6x10", "macrocell_equal_mass_8x12",
    ):
        finer = _spatial_balance_weights(tokens, (68, 120), mode)
        assert np.all((finer >= 0.5) & (finer <= 2.0))
        assert finer[-1] > finer[0]


def test_staggered_macrocell_balance_changes_boundary_membership_and_is_bounded() -> None:
    width = 120
    # x=19 and x=20 straddle the unshifted 4x6 boundary at x=20, while the
    # half-cell x phase places both in the same clipped staggered cell.
    tokens = np.asarray([10 * width + 19, 10 * width + 20, 10 * width + 21, 60 * width + 110])
    base = _spatial_balance_weights(tokens, (68, width), "macrocell_equal_mass_4x6")
    shifted = _spatial_balance_weights(
        tokens, (68, width), "macrocell_equal_mass_4x6_shift_x",
    )
    assert not np.array_equal(base, shifted)
    assert np.all((shifted >= 0.5) & (shifted <= 2.0))

    shifted_xy = _spatial_balance_weights(
        tokens, (68, width), "macrocell_equal_mass_6x10_shift_xy",
    )
    assert np.all((shifted_xy >= 0.5) & (shifted_xy <= 2.0))


def test_local_density_balance_downweights_compact_match_cluster() -> None:
    width = 120
    compact = [10 * width + 10, 10 * width + 11, 11 * width + 10, 11 * width + 11]
    isolated = [40 * width + 80]
    weights = _spatial_balance_weights(
        np.asarray([*compact, *isolated]), (68, width), "local_density_radius2",
    )
    assert np.all(weights >= 0.5)
    assert np.all(weights <= 2.0)
    assert weights[-1] > weights[0]


def test_query_region_balance_equalizes_region_mass_and_requires_labels() -> None:
    tokens = np.asarray([1, 2, 3, 4, 100])
    weights = _spatial_balance_weights(
        tokens, (68, 120), "query_region_equal_mass",
        group_labels=np.asarray([0, 0, 0, 0, 1]),
    )
    assert weights[-1] > weights[0]
    assert np.all((weights >= 0.5) & (weights <= 2.0))
    with pytest.raises(ValueError, match="requires group labels"):
        _spatial_balance_weights(tokens, (68, 120), "query_region_equal_mass")

    combined = _spatial_balance_weights(
        tokens, (68, 120), "query_region_x_macrocell_4x6",
        group_labels=np.asarray([0, 0, 0, 0, 1]),
    )
    assert np.all((combined >= 0.5) & (combined <= 2.0))


def test_cross_density_consensus_averages_only_nearby_hypotheses() -> None:
    rows = [
        {"query_tokens": np.asarray([1, 2]),
         "world_points": np.asarray([[0.0, 0.0, 1.0], [4.0, 0.0, 1.0]])},
        {"query_tokens": np.asarray([1, 2]),
         "world_points": np.asarray([[0.2, 0.0, 1.0], [6.0, 0.0, 1.0]])},
    ]
    world, tokens, consensus, ambiguous = _compose_union_support(
        rows, mode="cross_density_consensus_average", consensus_radius_m=0.5,
    )
    assert tokens.tolist() == [1, 2, 2]
    assert np.allclose(world[0], [0.1, 0.0, 1.0])
    assert consensus == 1
    assert ambiguous == 1


def test_midpoint_pose_averages_camera_center_and_rotation() -> None:
    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    right[:3, :3] = Rotation.from_euler("z", 10.0, degrees=True).as_matrix()
    right_center = np.asarray([2.0, 0.0, 0.0])
    right[:3, 3] = -right[:3, :3] @ right_center
    midpoint = _midpoint_pose(left, right)
    center = -midpoint[:3, :3].T @ midpoint[:3, 3]
    angle = Rotation.from_matrix(midpoint[:3, :3]).magnitude() * 180.0 / np.pi
    assert np.allclose(center, [1.0, 0.0, 0.0])
    assert np.isclose(angle, 5.0)
