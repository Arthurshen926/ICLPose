from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization.frozen_lifted_loftr_map_to_query import (
    FrozenLiftedTrackModeBank,
    FrozenLiftedSupportViewModeBank,
    build_frozen_candidate_support_view_group_layout,
    build_frozen_candidate_group_layout,
    build_frozen_lifted_track_mode_bank,
    build_frozen_lifted_support_view_mode_bank,
    candidate_support_view_log_mixture_terms_from_support_view_log_ratios,
    deterministic_track_xyz_permutation,
    fixed_prior_group_log_ratios_from_track_log_ratios,
    fixed_prior_group_log_ratios_from_support_view_log_ratios,
    group_log_ratios_from_track_log_ratios,
    robust_track_log_ratio_statistics,
    select_loftr_modes_near_support_anchors,
    support_view_log_ratios_from_projected,
    track_mode_log_ratios_from_projected,
)
from feature_extract.tools.vfm.score_frozen_lifted_loftr_map_to_query_pose_evidence import (
    _parse_diagnostic_hypothesis_indices,
    _select_explicit_hypothesis_indices,
)


def test_support_anchor_modes_do_not_use_a_query_center_and_keep_distinct_modes() -> None:
    modes = select_loftr_modes_near_support_anchors(
        support_anchor_xy=np.asarray([[20.0, 20.0]], dtype=np.float32),
        matched_query_xy=np.asarray([[11.0, 90.0], [80.0, 10.0], [12.0, 90.0]], dtype=np.float32),
        matched_support_xy=np.asarray([[20.2, 20.0], [20.5, 20.0], [20.25, 20.0]], dtype=np.float32),
        match_confidence=np.asarray([0.7, 0.95, 0.8], dtype=np.float32),
        support_snap_radius_px=2.0,
        max_modes_per_anchor=2,
        query_mode_nms_radius_px=3.0,
    )
    assert modes.anchor_indices.tolist() == [0, 0]
    # Both query endpoints are far from the support anchor: only support-side
    # proximity participates in selecting them.
    np.testing.assert_allclose(modes.query_xy, [[80.0, 10.0], [12.0, 90.0]], atol=1e-6)


def test_cross_view_track_bank_preserves_two_modes_and_requires_two_views() -> None:
    bank = build_frozen_lifted_track_mode_bank(
        track_ids=np.asarray([7, 7, 7, 7, 9], dtype=np.int64),
        xyz=np.asarray(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        ),
        support_image_ids=np.asarray(["a.png", "b.png", "a.png", "b.png", "a.png"]),
        query_xy=np.asarray(
            [[10.0, 10.0], [12.0, 10.0], [100.0, 100.0], [102.0, 100.0], [50.0, 50.0]],
            dtype=np.float32,
        ),
        confidence=np.asarray([0.9, 0.8, 0.7, 0.6, 0.9], dtype=np.float32),
        support_distance_px=np.zeros((5,), dtype=np.float32),
        min_support_views=2,
        consensus_radius_px=4.0,
        max_modes_per_track=2,
    )
    assert bank.track_ids.tolist() == [7]
    assert bank.mode_offsets.tolist() == [0, 4]
    assert bank.mode_support_view_counts.tolist() == [2, 2, 2, 2]
    assert bank.mode_support_image_ids.tolist() == ["a.png", "b.png", "a.png", "b.png"]
    np.testing.assert_allclose(
        bank.mode_query_xy,
        [[10.0, 10.0], [12.0, 10.0], [100.0, 100.0], [102.0, 100.0]],
        atol=1e-5,
    )
    assert bank.mode_weights.sum() == pytest.approx(1.0)


def test_fixed_projection_scoring_rewards_correct_mode_and_penalizes_out_of_image() -> None:
    bank = FrozenLiftedTrackModeBank(
        track_ids=np.asarray([1], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        mode_offsets=np.asarray([0, 1], dtype=np.int64),
        mode_query_xy=np.asarray([[50.0, 50.0]], dtype=np.float32),
        mode_weights=np.asarray([1.0], dtype=np.float32),
        mode_support_image_ids=np.asarray(["a.png"]),
        mode_support_view_counts=np.asarray([2], dtype=np.int64),
        mode_confidence_sums=np.asarray([1.5], dtype=np.float32),
        track_reliabilities=np.asarray([0.75], dtype=np.float32),
        track_reference_xy=np.asarray([[50.0, 50.0]], dtype=np.float32),
    )
    values = track_mode_log_ratios_from_projected(
        projected_xy=torch.tensor([[[50.0, 50.0]], [[50.0, 50.0]]]),
        projection_valid=torch.tensor([[True], [False]]),
        bank=bank,
        image_width=100,
        image_height=100,
        sigma_px=4.0,
        out_of_image_ratio=0.01,
        max_log_ratio=8.0,
    )
    assert float(values[0, 0]) > 0.0
    assert float(values[1, 0]) < 0.0
    summary = robust_track_log_ratio_statistics(
        track_log_ratios=values,
        track_reference_xy=bank.track_reference_xy,
        image_width=100,
        image_height=100,
    )
    assert summary["spatial_median_of_means_2x2"].tolist() == pytest.approx(
        values[:, 0].tolist()
    )


def test_xyz_permutation_is_reproducible_and_has_no_fixed_track() -> None:
    tracks = np.asarray([11, 13, 17, 19], dtype=np.int64)
    first = deterministic_track_xyz_permutation(tracks, query_id="seq1/frame00051.png")
    second = deterministic_track_xyz_permutation(tracks, query_id="seq1/frame00051.png")
    assert first.tolist() == second.tolist()
    assert sorted(first.tolist()) == [0, 1, 2, 3]
    assert not np.any(first == np.arange(len(tracks)))


def test_xyz_permutation_repairs_a_large_seeded_fixed_point_case() -> None:
    tracks = np.arange(416, dtype=np.int64)
    permutation = deterministic_track_xyz_permutation(
        tracks, query_id="seq2/frame00079.png"
    )
    assert sorted(permutation.tolist()) == tracks.tolist()
    assert not np.any(permutation == np.arange(len(tracks)))


def test_candidate_group_mixture_retains_fixed_topl_denominator_and_null() -> None:
    bank = FrozenLiftedTrackModeBank(
        track_ids=np.asarray([10, 20, 30], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32),
        mode_offsets=np.asarray([0, 1, 2, 3], dtype=np.int64),
        mode_query_xy=np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        mode_weights=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        mode_support_image_ids=np.asarray(["a.png", "b.png", "c.png"]),
        mode_support_view_counts=np.asarray([2, 2, 2], dtype=np.int64),
        mode_confidence_sums=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        track_reliabilities=np.asarray([0.7, 0.7, 0.7], dtype=np.float32),
        track_reference_xy=np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
    )
    layout = build_frozen_candidate_group_layout(
        candidate_track_ids=np.asarray(
            [[10, 99], [20, 30], [98, 99]], dtype=np.int64
        ),
        bank=bank,
    )
    assert layout.candidate_track_indices.tolist() == [[0, -1], [1, 2], [-1, -1]]
    assert layout.active_group_mask.tolist() == [True, True, False]
    values = group_log_ratios_from_track_log_ratios(
        track_log_ratios=torch.log(torch.tensor([[4.0, 2.0, 0.5]])),
        layout=layout,
        null_mass=0.5,
        max_log_ratio=8.0,
    )
    # Group 0 keeps its unavailable candidate as a fixed unit-ratio term:
    # 0.5 null + 0.5 * mean(4, 1) = 1.75.
    # Group 1 is 0.5 + 0.5 * mean(2, 0.5) = 1.125.  The empty
    # group remains an exact fixed null-only ratio of one.
    np.testing.assert_allclose(values.exp().numpy(), [[1.75, 1.125, 1.0]], atol=1e-6)


def test_fixed_prior_group_mixture_keeps_null_and_missing_candidates_neutral() -> None:
    bank = FrozenLiftedTrackModeBank(
        track_ids=np.asarray([10, 20], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        mode_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        mode_query_xy=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        mode_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        mode_support_image_ids=np.asarray(["a.png", "b.png"]),
        mode_support_view_counts=np.asarray([2, 2], dtype=np.int64),
        mode_confidence_sums=np.asarray([1.0, 1.0], dtype=np.float32),
        track_reliabilities=np.asarray([0.7, 0.7], dtype=np.float32),
        track_reference_xy=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
    )
    layout = build_frozen_candidate_group_layout(
        candidate_track_ids=np.asarray([[10, 99], [20, 10]], dtype=np.int64),
        bank=bank,
        candidate_identity_probabilities=np.asarray(
            [[0.20, 0.30], [0.25, 0.50]], dtype=np.float32
        ),
        null_probabilities=np.asarray([0.50, 0.25], dtype=np.float32),
    )
    values = fixed_prior_group_log_ratios_from_track_log_ratios(
        track_log_ratios=torch.log(torch.tensor([[4.0, 2.0]])),
        layout=layout,
        max_log_ratio=8.0,
    )
    # Group 0: 0.5 null + 0.2 * 4 + 0.3 * 1 = 1.6.  The unavailable
    # candidate has retained fixed posterior mass but neutral evidence.
    # Group 1: 0.25 + 0.25 * 2 + 0.5 * 4 = 2.75.
    np.testing.assert_allclose(values.exp().numpy(), [[1.6, 2.75]], atol=1e-6)


def test_candidate_support_view_mixture_keeps_missing_track_and_missing_mode_neutral() -> None:
    bank = FrozenLiftedSupportViewModeBank(
        track_ids=np.asarray([10], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        support_slot_track_indices=np.asarray([0, 0], dtype=np.int64),
        support_slot_image_ids=np.asarray(["a.png", "b.png"]),
        mode_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        mode_query_xy=np.asarray([[10.0, 10.0], [90.0, 10.0]], dtype=np.float32),
        mode_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        mode_confidence_sums=np.asarray([1.0, 1.0], dtype=np.float32),
        support_reliabilities=np.asarray([1.0, 1.0], dtype=np.float32),
        support_reference_xy=np.asarray([[10.0, 10.0], [90.0, 10.0]], dtype=np.float32),
    )
    layout = build_frozen_candidate_support_view_group_layout(
        candidate_track_ids=np.asarray([[10, 99]], dtype=np.int64),
        candidate_identity_probabilities=np.asarray([[0.8, 0.1]], dtype=np.float32),
        null_probabilities=np.asarray([0.1], dtype=np.float32),
        candidate_support_image_ids=np.asarray(
            [[["a.png", "b.png"], ["c.png", "d.png"]]]
        ),
        support_view_probabilities=np.asarray(
            [[[0.75, 0.25], [0.5, 0.5]]], dtype=np.float32
        ),
        bank=bank,
    )
    assert layout.candidate_track_indices.tolist() == [[0, -1]]
    assert layout.candidate_support_slot_indices.tolist() == [[[0, 1], [-1, -1]]]
    values = fixed_prior_group_log_ratios_from_support_view_log_ratios(
        support_view_log_ratios=torch.log(torch.tensor([[4.0, 0.25]])),
        layout=layout,
        max_log_ratio=8.0,
    )
    # 0.1 null + 0.8 * (0.75 * 4 + 0.25 * 0.25) + 0.1 * 1.
    np.testing.assert_allclose(values.exp().numpy(), [[2.65]], atol=1e-6)


def test_candidate_support_view_diagnostic_terms_preserve_candidate_identity() -> None:
    bank = FrozenLiftedSupportViewModeBank(
        track_ids=np.asarray([10], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        support_slot_track_indices=np.asarray([0, 0], dtype=np.int64),
        support_slot_image_ids=np.asarray(["a.png", "b.png"]),
        mode_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        mode_query_xy=np.asarray([[10.0, 10.0], [90.0, 10.0]], dtype=np.float32),
        mode_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        mode_confidence_sums=np.asarray([1.0, 1.0], dtype=np.float32),
        support_reliabilities=np.asarray([1.0, 1.0], dtype=np.float32),
        support_reference_xy=np.asarray([[10.0, 10.0], [90.0, 10.0]], dtype=np.float32),
    )
    layout = build_frozen_candidate_support_view_group_layout(
        candidate_track_ids=np.asarray([[10, 99]], dtype=np.int64),
        candidate_identity_probabilities=np.asarray([[0.8, 0.1]], dtype=np.float32),
        null_probabilities=np.asarray([0.1], dtype=np.float32),
        candidate_support_image_ids=np.asarray(
            [[["a.png", "b.png"], ["c.png", "d.png"]]]
        ),
        support_view_probabilities=np.asarray(
            [[[0.75, 0.25], [0.5, 0.5]]], dtype=np.float32
        ),
        bank=bank,
    )
    candidate, group = candidate_support_view_log_mixture_terms_from_support_view_log_ratios(
        support_view_log_ratios=torch.log(torch.tensor([[4.0, 0.25]])),
        layout=layout,
        max_log_ratio=8.0,
    )
    # Candidate 0 preserves its explicit view mixture; unavailable candidate
    # 1 remains neutral before the fixed group mixture is formed.
    np.testing.assert_allclose(candidate.exp().numpy(), [[[3.0625, 1.0]]], atol=1e-6)
    np.testing.assert_allclose(group.exp().numpy(), [[2.65]], atol=1e-6)


def test_explicit_diagnostic_hypothesis_selection_preserves_immutable_order() -> None:
    exact = {
        "hypothesis_indices": np.asarray([8, 2, 19], dtype=np.int64),
        "poses_w2c": np.arange(3 * 4 * 4, dtype=np.float64).reshape(3, 4, 4),
        "query_ids": np.asarray(["q.png", "q.png", "q.png"]),
    }
    assert _parse_diagnostic_hypothesis_indices("19,2") == (19, 2)
    selected = _select_explicit_hypothesis_indices(exact, (19, 2))
    assert selected["hypothesis_indices"].tolist() == [19, 2]
    np.testing.assert_array_equal(selected["poses_w2c"], exact["poses_w2c"][[2, 1]])


def test_support_view_scoring_preserves_slots_without_cross_view_pooling() -> None:
    bank = build_frozen_lifted_support_view_mode_bank(
        track_ids=np.asarray([7, 7], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        support_image_ids=np.asarray(["a.png", "b.png"]),
        query_xy=np.asarray([[20.0, 20.0], [80.0, 20.0]], dtype=np.float32),
        confidence=np.asarray([1.0, 1.0], dtype=np.float32),
        support_distance_px=np.zeros((2,), dtype=np.float32),
        max_modes_per_support_view=1,
        base_support_reliability=1.0,
    )
    assert bank.support_slot_image_ids.tolist() == ["a.png", "b.png"]
    values = support_view_log_ratios_from_projected(
        projected_xy=torch.tensor([[[20.0, 20.0]]]),
        projection_valid=torch.tensor([[True]]),
        bank=bank,
        image_width=100,
        image_height=100,
        sigma_px=2.0,
        out_of_image_ratio=0.01,
        max_log_ratio=8.0,
    )
    assert float(values[0, 0]) > 0.0
    assert float(values[0, 1]) < 0.0
