from __future__ import annotations

from feature_extract.tools.vfm.eval_dense_depth_measurement_fusion import _rows_by_pose_group


def test_dense_depth_grouping_ignores_match_level_candidate_id_without_pose_id() -> None:
    rows = [
        {"query_id": "q0", "candidate_id": "match_0", "render_x": "1"},
        {"query_id": "q0", "candidate_id": "match_1", "render_x": "2"},
        {"query_id": "q1", "candidate_id": "match_0", "render_x": "3"},
    ]

    groups = _rows_by_pose_group(rows)

    assert sorted(groups) == [("q0", ""), ("q1", "")]
    assert len(groups[("q0", "")]) == 2


def test_dense_depth_grouping_uses_pose_level_render_pose_id_when_available() -> None:
    rows = [
        {"query_id": "q0", "candidate_id": "match_0", "render_pose_id": "pose_a"},
        {"query_id": "q0", "candidate_id": "match_1", "render_pose_id": "pose_a"},
        {"query_id": "q0", "candidate_id": "match_2", "render_pose_id": "pose_b"},
    ]

    groups = _rows_by_pose_group(rows)

    assert sorted(groups) == [("q0", "pose_a"), ("q0", "pose_b")]
    assert len(groups[("q0", "pose_a")]) == 2
