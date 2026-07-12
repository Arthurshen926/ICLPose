from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
    resolve_global_query_track_assignment,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


def _match(index: int, x: float, y: float, score: float, *, track: int | None = None) -> QueryTo3DMatch:
    return QueryTo3DMatch(
        token_index=index,
        xy=np.asarray([x, y], dtype=np.float64),
        track_id=index if track is None else int(track),
        xyz=np.asarray([float(index), 0.0, 5.0], dtype=np.float64),
        similarity=float(score),
        ratio=0.0,
        landmark_variance=0.0,
        source="test",
    )


def test_pose_safe_selection_resolves_track_conflicts_before_topk() -> None:
    matches = [
        _match(0, 5.0, 5.0, 0.9, track=7),
        _match(1, 95.0, 5.0, 0.8, track=7),
        _match(2, 5.0, 95.0, 0.7),
    ]
    selected = select_pose_safe_matches(
        matches,
        max_matches=2,
        image_width=100,
        image_height=100,
        mode="score_topk",
    )
    assert [match.token_index for match in selected] == [0, 2]


def test_spatial_round_robin_prevents_one_cell_from_filling_budget() -> None:
    matches = [
        _match(0, 5.0, 5.0, 0.99),
        _match(1, 6.0, 6.0, 0.98),
        _match(2, 7.0, 7.0, 0.97),
        _match(3, 95.0, 5.0, 0.80),
        _match(4, 5.0, 95.0, 0.70),
    ]
    selected = select_pose_safe_matches(
        matches,
        max_matches=3,
        image_width=100,
        image_height=100,
        mode="spatial_round_robin",
        grid_rows=2,
        grid_cols=2,
    )
    assert {match.token_index for match in selected} == {0, 3, 4}


def test_uniform_ransac_order_does_not_depend_on_network_score() -> None:
    first = [
        _match(2, 20.0, 20.0, 0.99, track=30),
        _match(0, 10.0, 10.0, 0.01, track=20),
        _match(1, 15.0, 15.0, 0.50, track=10),
    ]
    second = [
        _match(2, 20.0, 20.0, 0.01, track=30),
        _match(0, 10.0, 10.0, 0.99, track=20),
        _match(1, 15.0, 15.0, 0.20, track=10),
    ]

    assert [item.token_index for item in stable_uniform_ransac_order(first)] == [0, 1, 2]
    assert [item.token_index for item in stable_uniform_ransac_order(second)] == [0, 1, 2]


def test_confidence_threshold_adapts_count_with_a_minimum_floor() -> None:
    matches = [
        _match(index, float(index), float(index), score)
        for index, score in enumerate((0.9, 0.8, 0.7, 0.2, 0.1, 0.05))
    ]
    selected = select_pose_safe_matches(
        matches,
        max_matches=6,
        min_matches=2,
        min_confidence=0.6,
        image_width=100,
        image_height=100,
        mode="score_topk",
    )
    assert [match.token_index for match in selected] == [0, 1, 2]

    floor_selected = select_pose_safe_matches(
        matches,
        max_matches=6,
        min_matches=4,
        min_confidence=0.95,
        image_width=100,
        image_height=100,
        mode="score_topk",
    )
    assert [match.token_index for match in floor_selected] == [0, 1, 2, 3]


def test_confidence_threshold_arguments_are_validated() -> None:
    matches = [_match(0, 0.0, 0.0, 0.9)]
    for kwargs in (
        {"min_matches": 0},
        {"min_matches": 3},
        {"min_confidence": float("nan")},
    ):
        try:
            select_pose_safe_matches(
                matches,
                max_matches=2,
                image_width=100,
                image_height=100,
                mode="score_topk",
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid selection arguments: {kwargs}")


def test_global_assignment_can_use_second_choice_instead_of_dropping_conflict() -> None:
    tracks = np.asarray([[10, 20], [10, 30]], dtype=np.int64)
    scores = np.asarray([[0.90, 0.89], [0.88, 0.10]], dtype=np.float32)
    selected = resolve_global_query_track_assignment(tracks, scores)

    assert selected.tolist() == [1, 0]
    assert tracks[np.arange(2), selected].tolist() == [20, 10]


def test_global_assignment_uses_an_independent_dustbin_per_query() -> None:
    tracks = np.asarray([[10], [20], [30]], dtype=np.int64)
    scores = np.asarray([[0.9], [0.2], [0.1]], dtype=np.float32)
    selected = resolve_global_query_track_assignment(
        tracks,
        scores,
        dustbin_score=0.5,
    )

    assert selected.tolist() == [0, -1, -1]


def test_global_assignment_score_matrix_resolves_each_image_separately() -> None:
    tracks = np.asarray([[10], [10], [10], [10]], dtype=np.int64)
    scores = np.asarray([[0.9], [0.8], [0.7], [0.6]], dtype=np.float32)
    resolved, selected = global_assignment_score_matrix(
        tracks,
        scores,
        ["a", "a", "b", "b"],
    )

    assert selected.tolist() == [0, -1, 0, -1]
    assert np.isfinite(resolved[:, 0]).tolist() == [True, False, True, False]
