from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_two_branch_seed_budget_sweep import (
    _select_minimum_budget,
)
from feature_extract.vfm.localization_goal_maplet.two_branch_seed_budget import (
    build_two_branch_seed_budget_arrays,
)


def _rotation(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _pool(*, layout: bool) -> dict[str, object]:
    details = []
    for rank in range(64):
        center_index = rank if not layout or rank >= 8 else rank
        if layout and 8 <= rank < 16:
            center_index = rank + 8
        center = np.asarray([2.0 * center_index, 0.0, 0.0])
        angle = float(rank if not layout or rank < 32 else rank + 32)
        rotation = _rotation(angle)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = -rotation @ center
        details.append({"rank": rank + 1, "pose_w2c": pose.tolist()})
    return {
        "maximum_modes": 64,
        "rows": [{
            "image_id": "seq10/frame.png",
            "mode_details": {"actual_parent_actual_child": details},
        }],
    }


def test_two_branch_seed_budget_keeps_branch_local_domains_and_prefix_counts():
    arrays = build_two_branch_seed_budget_arrays(
        _pool(layout=False), _pool(layout=True),
    )
    assert arrays["position_seed_budgets"].tolist() == [4, 8, 16]
    assert arrays["unique_position_seed_count_by_budget"].tolist() == [[4, 8, 24]]
    assert int(arrays["unique_orientation_factor_count"][0]) == 96
    implicit = arrays["implicit_lattice_pose_pair_count_by_budget"][0]
    assert np.all(np.diff(implicit) >= 0)
    # The stored domain is no larger than the two branch-local Cartesian
    # products and is not replaced by a cross-branch Cartesian hull.
    for budget, count in zip((4, 8, 16), implicit.tolist()):
        assert count <= 2 * budget * 605 * 64


def test_seed_budget_gate_is_absolute_95_percent_not_relative_retention():
    rows = [
        {"position_seed_budget_per_branch": 4,
         "union": {"region_2m_45deg": {"joint_hits": 83}}},
        {"position_seed_budget_per_branch": 8,
         "union": {"region_2m_45deg": {"joint_hits": 84}}},
        {"position_seed_budget_per_branch": 16,
         "union": {"region_2m_45deg": {"joint_hits": 88}}},
    ]
    selected, required = _select_minimum_budget(rows, query_count=88)
    assert required == 84
    assert selected == 8
    rows[-1]["union"]["region_2m_45deg"]["joint_hits"] = 83
    rows[1]["union"]["region_2m_45deg"]["joint_hits"] = 83
    assert _select_minimum_budget(rows, query_count=88)[0] is None
