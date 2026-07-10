from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.landmark_retrieval_training import (
    LandmarkPrototypeMemoryBank,
    LandmarkRetrievalLossConfig,
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


def test_landmark_retrieval_returns_no_loss_without_valid_track_ids() -> None:
    loss, metrics = landmark_retrieval_loss(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([-1]),
    )

    assert loss is None
    assert metrics["landmark_retrieval_valid_count"] == 0


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
