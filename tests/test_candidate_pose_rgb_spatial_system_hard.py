from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_system_hard import (
    rank_current_system_hard_wrong_modes,
    select_current_system_hard_repeat_edges,
    select_current_system_hard_repeat_edges_for_modes,
)


def _inputs() -> dict[str, np.ndarray]:
    # Two coherent-wrong modes, three P1 points, three top-L candidates, and
    # one support view. Candidate zero is the registered exact identity.
    return {
        "source_point_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "pair_ids": np.asarray([7, 3], dtype=np.int64),
        "wrong_pose_log_likelihood_ratios": np.asarray([0.2, 0.8], dtype=np.float32),
        "candidate_track_ids": np.asarray(
            [[100, 101, 102], [110, 111, 112], [120, 121, 122]], dtype=np.int64
        ),
        "candidate_probabilities": np.asarray(
            [[0.4, 0.35, 0.15], [0.4, 0.35, 0.15], [0.4, 0.35, 0.15]], dtype=np.float32
        ),
        "null_probabilities": np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        "observed_candidate_mask": np.asarray(
            [[True, False, False], [True, False, False], [True, False, False]]
        ),
        "correct_offsets_xy": np.zeros((3, 3, 2), dtype=np.float32),
        "correct_valid": np.ones((3, 3), dtype=bool),
        "correct_edge_usable": np.ones((3, 3, 1), dtype=bool),
        "wrong_offsets_xy": np.zeros((2, 3, 3, 2), dtype=np.float32),
        "wrong_valid": np.ones((2, 3, 3), dtype=bool),
        "wrong_candidate_log_likelihood_ratios": np.asarray(
            [
                [[0.0, 0.1, 0.0], [0.0, 0.1, 0.0], [0.0, 0.1, 0.0]],
                [[0.0, 0.2, 1.2], [0.0, 1.0, 0.1], [0.0, 0.1, 1.1]],
            ],
            dtype=np.float32,
        ),
        "wrong_edge_usable": np.ones((2, 3, 3, 1), dtype=bool),
    }


def test_system_hard_mining_uses_actual_highest_scoring_wrong_mode_and_posterior() -> None:
    result = select_current_system_hard_repeat_edges(
        **_inputs(),
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        minimum_negative_candidate_posterior=0.0,
    )
    assert result is not None
    # Mode 1 (pair 3) is harder than mode 0. Its posterior picks candidate 2
    # for rows 0 and 2, and candidate 1 for row 1.
    assert result.hardest_mode_index == 1
    assert result.hardest_pair_id == 3
    np.testing.assert_array_equal(result.positive_candidate_indices, [0, 0, 0])
    np.testing.assert_array_equal(result.negative_candidate_indices, [2, 1, 2])
    assert np.all(result.negative_candidate_posteriors > 0.0)


def test_system_hard_mining_preserves_target_free_top_h_mode_order() -> None:
    values = _inputs()
    ranked = rank_current_system_hard_wrong_modes(
        pair_ids=values["pair_ids"],
        wrong_pose_log_likelihood_ratios=values["wrong_pose_log_likelihood_ratios"],
        max_wrong_modes_per_query=2,
    )
    np.testing.assert_array_equal(ranked, [1, 0])
    selections = select_current_system_hard_repeat_edges_for_modes(
        **values,
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        max_wrong_modes_per_query=2,
    )
    assert len(selections) == 2
    assert all(selection is not None for selection in selections)
    assert [selection.hardest_mode_index for selection in selections if selection is not None] == [1, 0]
    assert [selection.hardest_pair_id for selection in selections if selection is not None] == [3, 7]
    assert [selection.selected_mode_rank for selection in selections if selection is not None] == [0, 1]

    legacy = select_current_system_hard_repeat_edges(
        **values,
        positive_radius_px=4.0,
        negative_radius_px=4.0,
    )
    assert legacy is not None
    assert selections[0] is not None
    np.testing.assert_array_equal(legacy.source_point_ids, selections[0].source_point_ids)
    np.testing.assert_array_equal(
        legacy.negative_candidate_indices, selections[0].negative_candidate_indices
    )


def test_system_hard_mining_does_not_backfill_an_ineligible_top_mode() -> None:
    values = _inputs()
    # Mode 1 remains target-free rank one, but it no longer has an eligible
    # wrong local projection. Top-H selection must retain this absence instead
    # of inspecting labels and silently promoting mode 0 into rank zero.
    values["wrong_offsets_xy"][1] = 9.0
    selections = select_current_system_hard_repeat_edges_for_modes(
        **values,
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        max_wrong_modes_per_query=2,
    )
    assert selections[0] is None
    assert selections[1] is not None
    assert selections[1].hardest_mode_index == 0
    assert selections[1].selected_mode_rank == 1


def test_system_hard_mining_rejects_nonpositive_top_h_count() -> None:
    with np.testing.assert_raises_regex(ValueError, "wrong-mode ranking"):
        rank_current_system_hard_wrong_modes(
            pair_ids=_inputs()["pair_ids"],
            wrong_pose_log_likelihood_ratios=_inputs()["wrong_pose_log_likelihood_ratios"],
            max_wrong_modes_per_query=0,
        )


def test_system_hard_mining_rejects_same_identity_and_low_posterior_candidates() -> None:
    values = _inputs()
    # Candidate one in row 0 aliases the registered physical track and must
    # not become a synthetic negative even though its score is high.
    values["candidate_track_ids"][0, 1] = values["candidate_track_ids"][0, 0]
    values["wrong_candidate_log_likelihood_ratios"][1, 0, 2] = -10.0
    result = select_current_system_hard_repeat_edges(
        **values,
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        minimum_negative_candidate_posterior=0.2,
    )
    assert result is not None
    assert 10 not in set(result.source_point_ids.tolist())
    assert np.all(result.negative_candidate_posteriors >= 0.2)


def test_system_hard_mining_returns_none_when_wrong_projection_is_not_local() -> None:
    values = _inputs()
    values["wrong_offsets_xy"][:] = 9.0
    result = select_current_system_hard_repeat_edges(
        **values,
        positive_radius_px=4.0,
        negative_radius_px=4.0,
    )
    assert result is None
