from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.evaluate_goal_maplet_parent_geometry_pose_proposal import (
    _select_minimum_parent_prefix,
)
from feature_extract.vfm.localization_goal_maplet.parent_geometry_pose_proposal import (
    ORIENTATION_COVER_RADIUS_DEG,
    analytic_orientation_codebook,
    build_parent_geometry_proposal_arrays,
    exact_parent_primitive_rectangle_aabbs,
    orientation_cover_certificate,
)


class _Physical(SimpleNamespace):
    def member_slice(self, row):
        return slice(int(self.membership_offsets[row]), int(self.membership_offsets[row + 1]))


def _physical(count=16):
    angle = np.deg2rad(45.0)
    tangent1 = np.tile(
        np.asarray([[np.cos(angle), np.sin(angle), 0.0]]), (count, 1),
    )
    tangent2 = np.tile(
        np.asarray([[-np.sin(angle), np.cos(angle), 0.0]]), (count, 1),
    )
    center = np.column_stack([
        4.0 * np.arange(count), np.zeros(count), np.zeros(count),
    ])
    return _Physical(
        maplet_ids=np.arange(100, 100 + count, dtype=np.int64),
        primitive_centers=center,
        primitive_tangent1=tangent1,
        primitive_tangent2=tangent2,
        primitive_scale1=np.full(count, 2.0),
        primitive_scale2=np.full(count, 1.0),
        membership_offsets=np.arange(count + 1, dtype=np.int64),
        membership_primitive_rows=np.arange(count, dtype=np.int64),
    )


def test_parent_aabb_uses_oriented_rectangle_axis_extent_not_max_scale():
    lower, upper = exact_parent_primitive_rectangle_aabbs(_physical(1))
    expected = 3.0 / np.sqrt(2.0)
    assert upper[0, 0] == pytest.approx(expected)
    assert upper[0, 1] == pytest.approx(expected)
    assert lower[0, 0] == pytest.approx(-expected)
    assert expected > 2.0  # center +/- max(scale) would under-cover this rectangle.


def test_parent_geometry_cells_are_nested_and_have_strict_2m_cell_cover():
    physical = _physical()
    ids = physical.maplet_ids[None]
    scores = np.linspace(2.0, 1.0, 16, dtype=np.float64)[None]
    arrays = build_parent_geometry_proposal_arrays(
        np.asarray(["seq10/frame.png"]), ids, scores, physical,
    )
    counts = arrays["unique_position_count_by_parent_prefix"][0]
    assert counts[0] <= counts[1] <= counts[2]
    assert arrays["implicit_pose_factor_count_by_parent_prefix"][0].tolist() == (
        counts.astype(np.int64) * 60
    ).tolist()
    first = arrays["cell_first_parent_rank"]
    cells = arrays["cell_indices_world"]
    origin = arrays["lattice_origin_world"]
    # Every point in an included 2 m cell is at most sqrt(3) m from its centre.
    index = cells[first <= 4][0]
    centre = origin + (index.astype(np.float64) + 0.5) * 2.0
    corner = origin + index.astype(np.float64) * 2.0
    assert np.linalg.norm(centre - corner) == pytest.approx(np.sqrt(3.0))
    assert np.linalg.norm(centre - corner) < 2.0


def test_regular_600_cell_is_an_analytic_full_so3_45_degree_cover():
    codebook = analytic_orientation_codebook()
    certificate = orientation_cover_certificate()
    assert codebook.shape == (60, 3, 3)
    assert certificate["so3_covering_radius_deg"] == pytest.approx(
        ORIENTATION_COVER_RADIUS_DEG
    )
    assert certificate["strictly_below_45_degrees"] is True
    assert certificate["monte_carlo_used_as_authority"] is False


def test_parent_prefix_gate_is_absolute_95_percent():
    rows = [
        {"parent_prefix_budget": 4,
         "raw_support": {"region_2m_45deg": {"joint_hits": 83}}},
        {"parent_prefix_budget": 8,
         "raw_support": {"region_2m_45deg": {"joint_hits": 84}}},
        {"parent_prefix_budget": 16,
         "raw_support": {"region_2m_45deg": {"joint_hits": 88}}},
    ]
    selected, required = _select_minimum_parent_prefix(rows, query_count=88)
    assert required == 84
    assert selected == 8
