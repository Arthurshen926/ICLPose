import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    child_surface_area_m2,
    select_fine_supports_under_area_budget,
    total_map_surface_area_m2,
)
from test_goal_maplet_pure_retrieval import _physical


def test_budgeted_selection_uses_all_token_mass_and_parent_filter():
    physical = _physical()
    child_count = int(physical.child_parent_rows.size)
    assert child_count >= 3
    rows = np.asarray(
        [[0, 1, 2], [0, 1, 2], [1, 2, -1], [1, 2, -1]], dtype=np.int64
    )
    probability = np.asarray(
        [[0.4, 0.3, 0.2], [0.4, 0.3, 0.2], [0.6, 0.3, 0.0], [0.6, 0.3, 0.0]],
        dtype=np.float32,
    )
    allowed_parent = int(
        physical.maplet_ids[int(physical.child_parent_rows[0])]
    )
    result = select_fine_supports_under_area_budget(
        rows,
        probability,
        np.asarray([allowed_parent], dtype=np.int64),
        physical,
        maximum_area_fraction=1.0,
        maximum_children=child_count,
    )
    expected_allowed = {
        row
        for row in range(child_count)
        if int(physical.child_parent_rows[row])
        == int(physical.child_parent_rows[0])
        and np.any(rows == row)
    }
    assert set(result.child_rows.tolist()) == expected_allowed
    mass = dict(zip(result.child_rows.tolist(), result.posterior_mass.tolist()))
    assert mass[0] == pytest.approx(0.8)
    if 1 in expected_allowed:
        assert mass[1] == pytest.approx(1.8)


def test_budget_and_count_are_hard_and_deterministic():
    physical = _physical()
    rows = np.tile(
        np.arange(min(4, physical.child_parent_rows.size), dtype=np.int64), (4, 1)
    )
    probability = np.tile(
        np.linspace(0.8, 0.2, rows.shape[1], dtype=np.float32), (4, 1)
    )
    parents = np.asarray(physical.maplet_ids, dtype=np.int64)
    left = select_fine_supports_under_area_budget(
        rows,
        probability,
        parents,
        physical,
        maximum_area_fraction=1.0,
        maximum_children=2,
    )
    right = select_fine_supports_under_area_budget(
        rows[::-1],
        probability[::-1],
        parents[::-1],
        physical,
        maximum_area_fraction=1.0,
        maximum_children=2,
    )
    assert left.child_rows.size <= 2
    assert left.selected_surface_area_m2 <= left.maximum_surface_area_m2
    np.testing.assert_array_equal(left.child_rows, right.child_rows)
    np.testing.assert_allclose(left.posterior_mass, right.posterior_mass)
    cached = select_fine_supports_under_area_budget(
        rows,
        probability,
        parents,
        physical,
        maximum_area_fraction=1.0,
        maximum_children=2,
        precomputed_child_surface_area_m2=child_surface_area_m2(physical),
        precomputed_total_map_surface_area_m2=total_map_surface_area_m2(physical),
    )
    np.testing.assert_array_equal(left.child_rows, cached.child_rows)

    adaptive = select_fine_supports_under_area_budget(
        rows,
        probability,
        parents,
        physical,
        maximum_area_fraction=1.0,
        maximum_children=rows.shape[1],
        target_candidate_posterior_mass_fraction=0.5,
    )
    assert adaptive.child_rows.size <= left.candidate_child_count
    assert adaptive.posterior_mass.sum() >= 0.5 * adaptive.eligible_posterior_mass


def test_budgeted_selection_rejects_invalid_probability():
    physical = _physical()
    with pytest.raises(ValueError, match="invalid fine-support budget input"):
        select_fine_supports_under_area_budget(
            np.asarray([[0]], dtype=np.int64),
            np.asarray([[1.1]], dtype=np.float32),
            physical.maplet_ids,
            physical,
            maximum_area_fraction=0.05,
        )
    with pytest.raises(ValueError, match="invalid fine-support budget input"):
        select_fine_supports_under_area_budget(
            np.asarray([[0]], dtype=np.int64),
            np.asarray([[0.5]], dtype=np.float32),
            physical.maplet_ids,
            physical,
            maximum_area_fraction=0.05,
            target_candidate_posterior_mass_fraction=0.0,
        )
    assert np.all(child_surface_area_m2(physical) > 0.0)
