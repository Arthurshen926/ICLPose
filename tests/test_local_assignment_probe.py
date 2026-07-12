import numpy as np

from feature_extract.vfm.local_maplet_matching import LocalMapletBank, LocalMapletSupportIndex
from feature_extract.vfm.localization.local_assignment_probe import (
    ProjectedSupportDescriptor,
    UniqueTrackCandidateSet,
    proposal_recall_summary,
    retrieve_unique_track_candidates_exact,
    score_candidate_support_feature_pooling,
    score_support_assignment_strategies,
    summarize_assignment_strategy,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _landmark_index():
    return LandmarkMapIndex(
        track_ids=np.asarray([1, 1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        features=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((4,), dtype=np.float32),
        observation_counts=np.full((4,), 2, dtype=np.int64),
        observation_image_ids=(("a",), ("b",), ("a",), ("b",)),
        prototype_ids=np.asarray([0, 1, 0, 0], dtype=np.int64),
    )


def _maplet_index():
    maplets = LocalMapletBank(
        neighbor_indices=np.asarray([[0], [1], [2]], dtype=np.int64),
        context_features=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        neighbor_counts=np.ones((3,), dtype=np.int64),
        context_radius=np.ones((3,), dtype=np.float32),
        context_feature_variance=np.zeros((3,), dtype=np.float32),
        covisibility_strength=np.ones((3,), dtype=np.float32),
        xyz_cov_eigvals=np.ones((3, 3), dtype=np.float32),
        neighbor_idf_mean=np.ones((3,), dtype=np.float32),
        maplet_type="test",
        maplet_k=1,
    )
    return LocalMapletSupportIndex(
        maplets=maplets,
        anchor_track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        neighbor_track_ids=np.asarray([[1], [2], [3]], dtype=np.int64),
        support_image_ids=("a", "b"),
        support_image_indices=np.asarray([[0, 1], [0, -1], [1, -1]], dtype=np.int64),
        support_coverage_counts=np.ones((3, 2), dtype=np.int64),
        candidate_k=1,
    )


def test_exact_candidates_collapse_multiple_prototypes_to_unique_tracks():
    candidates = retrieve_unique_track_candidates_exact(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _landmark_index(),
        top_l=3,
        device="cpu",
    )

    assert candidates.track_ids.tolist() == [[1, 2, 3]]
    assert candidates.prototype_ids[0, 0] == 0


def test_support_best_can_recover_correct_track_inside_fixed_proposals():
    candidates = UniqueTrackCandidateSet(
        bank_row_indices=np.asarray([[2, 0]], dtype=np.int64),
        track_ids=np.asarray([[2, 1]], dtype=np.int64),
        prototype_ids=np.asarray([[0, 0]], dtype=np.int64),
        coarse_scores=np.asarray([[0.9, 0.8]], dtype=np.float32),
    )
    support = {
        1: (ProjectedSupportDescriptor(1, "a", np.asarray([1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), 0.1),),
        2: (ProjectedSupportDescriptor(2, "a", np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0, 1.0]), 0.1),),
    }

    probe = score_support_assignment_strategies(
        query_descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_context_descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_viewing_rays=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        candidates=candidates,
        landmark_index=_landmark_index(),
        maplet_index=_maplet_index(),
        support_by_track=support,
    )

    coarse = summarize_assignment_strategy(
        candidates=candidates,
        correct_track_ids=[1],
        query_ids=["q"],
        scores=probe.strategy_scores["coarse_prototype"],
    )
    reranked = summarize_assignment_strategy(
        candidates=candidates,
        correct_track_ids=[1],
        query_ids=["q"],
        scores=probe.strategy_scores["maplet_support_best"],
    )

    assert coarse["recall_at_1"] == 0.0
    assert reranked["recall_at_1"] == 1.0
    assert reranked["pair_correct_average_precision"] == 1.0
    assert proposal_recall_summary(candidates, [1])["recall_at_5"] == 1.0


def test_generic_support_pooling_uses_candidate_track_groups():
    candidates = UniqueTrackCandidateSet(
        bank_row_indices=np.asarray([[2, 0]], dtype=np.int64),
        track_ids=np.asarray([[2, 1]], dtype=np.int64),
        prototype_ids=np.zeros((1, 2), dtype=np.int64),
        coarse_scores=np.asarray([[0.9, 0.8]], dtype=np.float32),
    )

    scores = score_candidate_support_feature_pooling(
        query_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        candidates=candidates,
        support_track_ids=np.asarray([2, 1, 1], dtype=np.int64),
        support_features=np.asarray([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        prefix="fine",
    )

    assert scores["fine_support_top2_mean"][0, 1] > scores["fine_support_top2_mean"][0, 0]
