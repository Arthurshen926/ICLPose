from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.landmark_retrieval_training import (
    LandmarkPrototypeMemoryBank,
    LandmarkRetrievalLossConfig,
    _merge_memory_negative_sources,
    landmark_retrieval_loss,
)
from feature_extract.vfm.matcha_joint_training import _sample_descriptor_rows_at_image_xy
from feature_extract.vfm.rendered_keypoint_matching import bilinear_sample_feature_map


def _bank() -> LandmarkPrototypeMemoryBank:
    return LandmarkPrototypeMemoryBank(capacity=8, descriptor_dim=3, device="cpu", momentum=0.5)


def test_landmark_retrieval_aggregates_same_track_support_rows() -> None:
    query = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]])
    support = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 1.0, 0.0]])
    track_ids = torch.tensor([10, 10, 20])

    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        track_ids,
        config=LandmarkRetrievalLossConfig(dustbin_logit=None),
    )

    assert loss is not None
    assert metrics["landmark_retrieval_track_count"] == 2
    assert metrics["landmark_retrieval_recall_at_1"] == pytest.approx(1.0)
    assert metrics["landmark_retrieval_mean_prototype_observation_count"] == pytest.approx(1.5)


def test_landmark_retrieval_deduplicates_heldout_query_but_keeps_all_support_views() -> None:
    query = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    support = torch.tensor(
        [[1.0, 0.1, 0.0], [0.9, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 0.9, 0.0]]
    )

    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10, 10, 20, 20]),
        query_group_ids=torch.tensor([100, 100, 101, 101]),
        query_image_group_ids=torch.tensor([7, 7, 7, 7]),
        config=LandmarkRetrievalLossConfig(dustbin_logit=None),
    )

    assert loss is not None
    assert metrics["landmark_retrieval_valid_count"] == 2
    assert metrics["landmark_retrieval_support_observation_count"] == 4
    assert metrics["landmark_retrieval_deduplicated_query_count"] == 2
    assert metrics["landmark_retrieval_mean_prototype_observation_count"] == pytest.approx(2.0)


def test_landmark_retrieval_excludes_current_track_history_from_negatives() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([10, 30]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]]),
        observation_counts=np.asarray([2, 1]),
        xyz=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([10, 20]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=4,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            dustbin_logit=None,
        ),
    )

    assert loss is not None
    assert metrics["landmark_retrieval_history_positive_fraction"] == pytest.approx(0.5)
    assert metrics["landmark_retrieval_memory_negative_count"] == 1
    assert metrics["landmark_retrieval_candidate_count"] == 3


def test_landmark_retrieval_geometry_hard_negative_uses_track_xyz() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30, 40]),
        descriptors=torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        observation_counts=np.asarray([1, 1]),
        xyz=torch.tensor([[0.2, 0.0, 0.0], [20.0, 0.0, 0.0]]),
    )

    _loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([10]),
        track_xyz=torch.tensor([[0.0, 0.0, 0.0]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=0,
            geometry_hard_negatives_per_track=1,
            random_negatives=0,
            dustbin_logit=0.0,
        ),
    )

    assert metrics["landmark_retrieval_geometry_hard_negative_count"] == 1
    assert metrics["landmark_retrieval_memory_negative_count"] == 1


def test_memory_negative_merge_balances_sources_under_shared_limit() -> None:
    selected, counts = _merge_memory_negative_sources(
        semantic=list(range(10)),
        geometry=list(range(100, 110)),
        random=list(range(200, 210)),
        limit=8,
        policy="source_balanced_round_robin",
    )
    legacy, legacy_counts = _merge_memory_negative_sources(
        semantic=list(range(10)),
        geometry=list(range(100, 110)),
        random=list(range(200, 210)),
        limit=8,
        policy="legacy_source_concat",
    )

    assert selected == [0, 1, 100, 200, 2, 3, 101, 201]
    assert counts == {"semantic": 4, "geometry": 2, "random": 2}
    assert legacy == list(range(8))
    assert legacy_counts == {"semantic": 8, "geometry": 0, "random": 0}


def test_rank_major_semantic_mining_admits_each_query_top_confuser() -> None:
    bank = LandmarkPrototypeMemoryBank(
        capacity=8,
        descriptor_dim=4,
        device="cpu",
        momentum=0.0,
    )
    bank.update(
        track_ids=np.arange(100, 108, dtype=np.int64),
        descriptors=torch.cat([torch.eye(4), 0.9 * torch.eye(4)], dim=0),
        observation_counts=np.ones((8,), dtype=np.int64),
        xyz=torch.arange(24, dtype=torch.float32).reshape(8, 3),
    )
    query = torch.eye(4)
    support = torch.eye(4)
    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.arange(4),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            memory_candidate_pool_size=0,
            semantic_hard_negatives_per_query=2,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            max_memory_negatives=4,
            memory_negative_merge_policy="source_balanced_round_robin",
            dustbin_logit=None,
        ),
    )

    assert loss is not None
    assert metrics["landmark_retrieval_global_exact_semantic_mining"] == 1
    assert metrics["landmark_retrieval_memory_candidate_pool_count"] == 8
    assert metrics["landmark_retrieval_semantic_top1_admitted_fraction"] == pytest.approx(1.0)
    assert metrics["landmark_retrieval_semantic_topk_admitted_per_query_mean"] == pytest.approx(1.0)


def test_balanced_merge_does_not_let_semantic_negatives_starve_geometry() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30, 40, 50, 60]),
        descriptors=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.7, 0.3, 0.0],
            ]
        ),
        observation_counts=np.ones((4,), dtype=np.int64),
        xyz=torch.tensor(
            [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        ),
    )
    _loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]),
        torch.tensor([10, 20]),
        track_xyz=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=4,
            geometry_hard_negatives_per_track=1,
            random_negatives=0,
            max_memory_negatives=3,
            memory_negative_merge_policy="source_balanced_round_robin",
        ),
    )

    assert metrics["landmark_retrieval_semantic_hard_negative_count"] == 2
    assert metrics["landmark_retrieval_geometry_hard_negative_count"] == 1
    assert metrics["landmark_retrieval_geometry_hard_negative_raw_unique_count"] >= 1


def test_system_confuser_margin_contributes_finite_gradients() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0]]),
        observation_counts=np.asarray([2]),
    )
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    support = torch.tensor([[0.8, 0.6, 0.0]], requires_grad=True)
    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=1,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            system_hard_negative_margin=0.1,
            system_hard_negative_margin_weight=1.0,
            dustbin_logit=None,
        ),
    )
    assert loss is not None
    loss.backward()

    assert metrics["landmark_retrieval_system_hard_negative_margin_count"] == 1
    assert metrics["landmark_retrieval_system_hard_negative_margin_loss"] > 0.0
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert support.grad is not None and torch.isfinite(support.grad).all()


def test_explicit_coherent_wrong_pose_tracks_drive_margin_from_frozen_bank() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30, 40]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        observation_counts=np.asarray([3, 2]),
    )
    bank.frozen = True
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    support = torch.tensor([[0.8, 0.6, 0.0]], requires_grad=True)

    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10]),
        coherent_hard_negative_track_ids=torch.tensor([[30, 999, -1]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=0,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            coherent_hard_negative_margin=0.1,
            coherent_hard_negative_margin_weight=1.0,
            dustbin_logit=None,
        ),
    )

    assert loss is not None
    loss.backward()
    assert metrics["landmark_retrieval_coherent_hard_negative_margin_count"] == 1
    assert metrics["landmark_retrieval_coherent_hard_negative_margin_loss"] > 0.0
    assert metrics["landmark_retrieval_coherent_hard_negative_missing_count"] == 1
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_coherent_mode_uses_whole_configuration_margin() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([10, 30]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        observation_counts=np.asarray([8, 6]),
    )
    bank.frozen = True
    query = torch.tensor([[1.0, 0.0, 0.0]] * 4, requires_grad=True)

    loss, metrics = landmark_retrieval_loss(
        query,
        torch.tensor([[1.0, 0.0, 0.0]] * 4),
        torch.tensor([10, 10, 10, 10]),
        coherent_hard_negative_track_ids=torch.tensor([[30], [30], [30], [30]]),
        coherent_hard_negative_mode_ids=torch.tensor([[7], [7], [7], [7]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            positive_prototype_source="query_disjoint_frozen_bank",
            prototype_min_support_observations=2,
            semantic_hard_negatives_per_query=0,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            coherent_hard_negative_margin=0.1,
            coherent_hard_negative_margin_weight=1.0,
            coherent_hard_negative_min_mode_rows=4,
            dustbin_logit=None,
        ),
    )

    assert loss is not None
    loss.backward()
    assert metrics["landmark_retrieval_coherent_configuration_mode_count"] == 1
    assert metrics["landmark_retrieval_coherent_configuration_row_count"] == 4
    assert metrics["landmark_retrieval_coherent_hard_negative_margin_loss"] > 0.0
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_coherent_mode_hard_rows_are_not_cancelled_by_easy_rows() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([10, 30]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        observation_counts=np.asarray([8, 6]),
    )
    bank.frozen = True
    query = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        requires_grad=True,
    )

    loss, metrics = landmark_retrieval_loss(
        query,
        torch.tensor([[1.0, 0.0, 0.0]] * 4),
        torch.tensor([10, 10, 10, 10]),
        coherent_hard_negative_track_ids=torch.tensor([[30], [30], [30], [30]]),
        coherent_hard_negative_mode_ids=torch.tensor([[7], [7], [7], [7]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            positive_prototype_source="query_disjoint_frozen_bank",
            prototype_min_support_observations=2,
            semantic_hard_negatives_per_query=0,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            coherent_hard_negative_margin=0.1,
            coherent_hard_negative_margin_weight=1.0,
            coherent_hard_negative_min_mode_rows=4,
            dustbin_logit=None,
        ),
    )

    assert loss is not None
    assert metrics["landmark_retrieval_coherent_configuration_margin_loss"] == 0.0
    assert metrics[
        "landmark_retrieval_coherent_configuration_row_margin_loss"
    ] > 0.0
    assert metrics[
        "landmark_retrieval_coherent_configuration_row_violation_fraction"
    ] == pytest.approx(0.25)
    assert metrics["landmark_retrieval_coherent_hard_negative_margin_loss"] > 0.0


def test_explicit_coherent_wrong_pose_tracks_cannot_include_ambiguity_positive() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0]]),
        observation_counts=np.asarray([3]),
    )
    bank.frozen = True
    loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True),
        torch.tensor([[0.8, 0.6, 0.0]], requires_grad=True),
        torch.tensor([10]),
        known_positive_track_ids=torch.tensor([[10, 30]]),
        coherent_hard_negative_track_ids=torch.tensor([[30]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=0,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            coherent_hard_negative_margin_weight=1.0,
            dustbin_logit=None,
        ),
    )

    assert loss is not None
    assert metrics["landmark_retrieval_coherent_hard_negative_margin_count"] == 0
    assert metrics[
        "landmark_retrieval_coherent_hard_negative_excluded_positive_count"
    ] == 1


def test_coherent_wrong_pose_rows_follow_eligible_track_filtering() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([10, 30, 40, 50]),
        descriptors=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        observation_counts=np.asarray([7, 3, 3, 3]),
    )
    bank.frozen = True

    loss, metrics = landmark_retrieval_loss(
        torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            requires_grad=True,
        ),
        torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            requires_grad=True,
        ),
        torch.tensor([10, 10, 20]),
        query_image_group_ids=torch.tensor([0, 0, 0]),
        coherent_hard_negative_track_ids=torch.tensor([[30], [40], [50]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            positive_prototype_source="query_disjoint_frozen_bank",
            prototype_min_support_observations=2,
            semantic_hard_negatives_per_query=0,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            coherent_hard_negative_margin=0.1,
            coherent_hard_negative_margin_weight=1.0,
            dustbin_logit=None,
        ),
    )

    assert loss is not None
    assert metrics["landmark_retrieval_dropped_singleton_track_count"] == 1
    assert metrics["landmark_retrieval_deduplicated_query_count"] == 1
    assert metrics["landmark_retrieval_coherent_hard_negative_margin_count"] == 1
    assert metrics["landmark_retrieval_coherent_hard_negative_margin_loss"] > 0.0


def test_global_mining_excludes_full_sfm_same_cell_tracks_from_negatives() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30, 40]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        observation_counts=np.asarray([2, 2]),
    )
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    support = torch.tensor([[0.8, 0.6, 0.0]], requires_grad=True)
    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10]),
        known_positive_track_ids=torch.tensor([[10, 30, -1]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=1,
            geometry_hard_negatives_per_track=0,
            random_negatives=2,
            max_memory_negatives=2,
            system_hard_negative_margin=0.1,
            system_hard_negative_margin_weight=1.0,
            dustbin_logit=None,
        ),
    )
    assert loss is not None
    loss.backward()

    assert metrics["landmark_retrieval_known_positive_raw_top1_fraction"] == pytest.approx(1.0)
    assert metrics["landmark_retrieval_known_positive_memory_candidate_count"] == 1
    assert metrics["landmark_retrieval_known_positive_selected_negative_count"] == 1
    assert metrics["landmark_retrieval_system_hard_negative_margin_loss"] == pytest.approx(0.0)
    assert metrics["landmark_retrieval_hardest_negative_score_mean"] == pytest.approx(0.0)
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_strict_spatial_bank_track_is_excluded_not_used_as_positive() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30, 40]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        observation_counts=np.asarray([2, 2]),
    )
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    support = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)

    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10]),
        known_positive_track_ids=torch.tensor([[10, 30, -1]]),
        strict_positive_track_ids=torch.tensor([[10, 30, -1]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            semantic_hard_negatives_per_query=1,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            max_memory_negatives=1,
            dustbin_logit=None,
        ),
    )
    assert loss is not None
    loss.backward()

    assert metrics["landmark_retrieval_memory_negative_count"] == 1
    assert metrics["landmark_retrieval_strict_ambiguity_memory_pair_count"] == 1
    assert metrics["landmark_retrieval_memory_strict_ambiguity_candidate_count"] == 0
    assert metrics["landmark_retrieval_strict_ambiguity_selected_pair_count"] == 0
    assert metrics["landmark_retrieval_candidate_count"] == 2
    assert float(loss.detach()) == pytest.approx(np.log(2.0))
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_strict_spatial_support_prototype_enters_set_valued_numerator() -> None:
    query = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True
    )
    support = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True
    )
    track_ids = torch.tensor([10, 20])
    strict = torch.tensor([[10, 20], [20, -1]])

    exact_loss, _ = landmark_retrieval_loss(
        query,
        support,
        track_ids,
        config=LandmarkRetrievalLossConfig(
            set_valued_cell_positives=False,
            dustbin_logit=None,
        ),
    )
    strict_loss, metrics = landmark_retrieval_loss(
        query,
        support,
        track_ids,
        strict_positive_track_ids=strict,
        config=LandmarkRetrievalLossConfig(dustbin_logit=None),
    )

    assert exact_loss is not None and strict_loss is not None
    assert strict_loss < exact_loss
    assert metrics["landmark_retrieval_strict_valid_recall_at_1"] == pytest.approx(1.0)


def test_query_disjoint_frozen_bank_supplies_positive_prototype() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([10, 30]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        observation_counts=np.asarray([7, 5]),
    )
    bank.frozen = True
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    deliberately_wrong_episode_support = torch.tensor(
        [[0.0, 1.0, 0.0]], requires_grad=True
    )

    loss, metrics = landmark_retrieval_loss(
        query,
        deliberately_wrong_episode_support,
        torch.tensor([10]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            positive_prototype_source="query_disjoint_frozen_bank",
            semantic_hard_negatives_per_query=1,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            dustbin_logit=None,
        ),
    )
    assert loss is not None
    loss.backward()

    assert float(loss.detach()) < 1e-4
    assert metrics["landmark_retrieval_positive_source_query_disjoint_frozen_bank"] == 1.0
    assert metrics["landmark_retrieval_mean_prototype_observation_count"] == pytest.approx(7.0)
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert deliberately_wrong_episode_support.grad is None


def test_query_disjoint_frozen_bank_drops_missing_positive_track() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30]),
        descriptors=torch.tensor([[0.0, 1.0, 0.0]]),
        observation_counts=np.asarray([5]),
    )
    bank.frozen = True

    loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([10]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            positive_prototype_source="query_disjoint_frozen_bank",
            dustbin_logit=None,
        ),
    )

    assert loss is None
    assert metrics["landmark_retrieval_valid_count"] == 0.0
    assert metrics["landmark_retrieval_frozen_bank_missing_positive_track_count"] == 1.0
    assert metrics["landmark_retrieval_frozen_bank_dropped_query_count"] == 1.0


def test_query_disjoint_frozen_bank_filters_missing_track_rows_and_relabels() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([10, 30]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        observation_counts=np.asarray([7, 5]),
    )
    bank.frozen = True

    loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.tensor([10, 20]),
        query_image_group_ids=torch.tensor([0, 0]),
        coherent_hard_negative_track_ids=torch.tensor([[30], [30]]),
        coherent_hard_negative_mode_ids=torch.tensor([[1], [1]]),
        memory_bank=bank,
        config=LandmarkRetrievalLossConfig(
            positive_prototype_source="query_disjoint_frozen_bank",
            semantic_hard_negatives_per_query=1,
            geometry_hard_negatives_per_track=0,
            random_negatives=0,
            coherent_hard_negative_margin_weight=0.5,
            coherent_hard_negative_min_mode_rows=2,
            dustbin_logit=None,
        ),
    )

    assert loss is not None and torch.isfinite(loss)
    assert metrics["landmark_retrieval_valid_count"] == 1.0
    assert metrics["landmark_retrieval_track_count"] == 1.0
    assert metrics["landmark_retrieval_frozen_bank_missing_positive_track_count"] == 1.0
    assert metrics["landmark_retrieval_frozen_bank_dropped_query_count"] == 1.0


def test_landmark_memory_update_during_loss_does_not_break_backward() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([30]),
        descriptors=torch.tensor([[0.9, 0.1, 0.0]]),
        observation_counts=np.asarray([1]),
    )
    query = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    support = torch.tensor([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0]], requires_grad=True)

    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10, 20]),
        memory_bank=bank,
        update_memory=True,
        config=LandmarkRetrievalLossConfig(geometry_hard_negatives_per_track=0),
    )
    assert loss is not None
    loss.backward()

    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(support.grad).all()
    assert len(bank) == 3
    assert metrics["landmark_retrieval_memory_size_before"] == 1
    assert metrics["landmark_retrieval_memory_size_after"] == 3


def test_landmark_memory_ema_accumulates_observation_count() -> None:
    bank = _bank()
    bank.update(
        track_ids=np.asarray([10]),
        descriptors=torch.tensor([[1.0, 0.0, 0.0]]),
        observation_counts=np.asarray([2]),
    )
    bank.update(
        track_ids=np.asarray([10]),
        descriptors=torch.tensor([[0.0, 1.0, 0.0]]),
        observation_counts=np.asarray([3]),
    )

    descriptor, found, counts, _xyz = bank.lookup(np.asarray([10]))
    assert found.tolist() == [True]
    assert counts.tolist() == [5]
    np.testing.assert_allclose(descriptor.numpy(), [[2**-0.5, 2**-0.5, 0.0]], atol=1e-6)


def test_frozen_landmark_snapshot_is_immutable_and_global(tmp_path) -> None:
    path = tmp_path / "projected.npz"
    np.savez(
        path,
        track_ids=np.asarray([10, 20], dtype=np.int64),
        features=np.asarray([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32),
        xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        observation_counts=np.asarray([2, 3], dtype=np.int64),
        metadata_json=np.asarray('{"descriptor_space_id":"space"}'),
    )
    bank = LandmarkPrototypeMemoryBank.from_projected_landmark_npz(
        path,
        device="cpu",
        expected_descriptor_dim=3,
    )
    before, _found, before_counts, _xyz = bank.lookup([10])
    bank.update(
        track_ids=[10],
        descriptors=torch.tensor([[0.0, 1.0, 0.0]]),
        observation_counts=[100],
    )
    after, _found, after_counts, _xyz = bank.lookup([10])

    assert bank.frozen is True
    assert bank.source_path == str(path)
    np.testing.assert_allclose(before.numpy(), after.numpy())
    np.testing.assert_array_equal(before_counts, after_counts)
    assert len(bank) == 2


def test_landmark_retrieval_returns_no_loss_without_valid_track_ids() -> None:
    loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([-1]),
    )

    assert loss is None
    assert metrics["landmark_retrieval_valid_count"] == 0


def test_landmark_retrieval_learns_query_dependent_dustbin_from_unmatched_rows() -> None:
    valid_dustbin = torch.tensor([-2.0], requires_grad=True)
    unmatched_dustbin = torch.tensor([8.0], requires_grad=True)
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    support = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    unmatched = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)

    loss, metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10]),
        dustbin_logits=valid_dustbin,
        unmatched_query_descriptors=unmatched,
        unmatched_dustbin_logits=unmatched_dustbin,
        config=LandmarkRetrievalLossConfig(dustbin_logit=None, dustbin_loss_weight=1.0),
    )
    assert loss is not None
    loss.backward()

    assert metrics["landmark_retrieval_dustbin_positive_count"] == 1
    assert metrics["landmark_retrieval_dustbin_recall_0p5"] == pytest.approx(1.0)
    assert metrics["landmark_retrieval_valid_accept_rate_0p5"] == pytest.approx(1.0)
    assert valid_dustbin.grad is not None and torch.isfinite(valid_dustbin.grad).all()
    assert unmatched_dustbin.grad is not None and torch.isfinite(unmatched_dustbin.grad).all()
    assert unmatched.grad is not None and torch.isfinite(unmatched.grad).all()


def test_landmark_dustbin_can_train_without_moving_descriptor_space() -> None:
    query = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    support = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    unmatched = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
    valid_dustbin = torch.tensor([-2.0], requires_grad=True)
    unmatched_dustbin = torch.tensor([2.0], requires_grad=True)

    loss, _metrics = landmark_retrieval_loss(
        query,
        support,
        torch.tensor([10]),
        dustbin_logits=valid_dustbin,
        unmatched_query_descriptors=unmatched,
        unmatched_dustbin_logits=unmatched_dustbin,
        config=LandmarkRetrievalLossConfig(
            dustbin_logit=None,
            dustbin_loss_weight=1.0,
            dustbin_detach_descriptors=True,
        ),
    )
    assert loss is not None
    loss.backward()

    assert unmatched.grad is None
    assert unmatched_dustbin.grad is not None


def test_landmark_retrieval_masks_same_query_cell_tracks_as_false_negatives() -> None:
    query = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    support = torch.tensor([[1.0, 0.0, 0.0], [0.98, 0.02, 0.0]])
    track_ids = torch.tensor([10, 20])
    config = LandmarkRetrievalLossConfig(dustbin_logit=None)

    unmasked_loss, _unmasked_metrics = landmark_retrieval_loss(query, support, track_ids, config=config)
    masked_loss, masked_metrics = landmark_retrieval_loss(
        query,
        support,
        track_ids,
        query_group_ids=torch.tensor([7, 7]),
        config=config,
    )

    assert unmasked_loss is not None and masked_loss is not None
    assert masked_loss < unmasked_loss
    assert masked_metrics["landmark_retrieval_same_cell_false_negatives_excluded_mean"] == pytest.approx(1.0)
    assert masked_metrics["landmark_retrieval_same_cell_valid_recall_at_1"] == pytest.approx(1.0)


def test_torch_observation_sampling_matches_projected_bank_bilinear_sampling() -> None:
    feature_map = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    xy = np.asarray([[0.0, 0.0], [3.25, 2.5], [7.0, 5.0]], dtype=np.float32)
    expected, valid = bilinear_sample_feature_map(
        feature_map,
        xy,
        image_width=8,
        image_height=6,
    )

    actual = _sample_descriptor_rows_at_image_xy(
        torch.from_numpy(feature_map[None]),
        torch.zeros((xy.shape[0],), dtype=torch.long),
        torch.from_numpy(xy),
        image_width=8,
        image_height=6,
    )

    assert valid.tolist() == [True, True, True]
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_torch_observation_sampling_groups_rows_without_breaking_gradients() -> None:
    feature_maps = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    feature_maps.requires_grad_(True)
    pair_indices = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    xy = torch.tensor(
        [[0.0, 0.0], [4.0, 3.0], [7.0, 5.0], [1.0, 1.0], [5.0, 4.0], [7.0, 5.0]],
        dtype=torch.float32,
    )

    sampled = _sample_descriptor_rows_at_image_xy(
        feature_maps,
        pair_indices,
        xy,
        image_width=torch.full((6,), 8.0),
        image_height=torch.full((6,), 6.0),
    )
    sampled.sum().backward()

    assert sampled.shape == (6, 3)
    assert feature_maps.grad is not None
    assert torch.isfinite(feature_maps.grad).all()
