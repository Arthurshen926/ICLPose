from __future__ import annotations

import numpy as np
import torch

from feature_extract.tools.vfm.audit_candidate_maplet_permutation import (
    _non_anchor_orders,
    _reorder_inference_batch,
    _restore_group_output,
    _restore_pair_axis,
    _restore_query_probabilities,
    _stratified_group_sample,
)
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    CandidateMapletBatch,
)


def _batch() -> CandidateMapletBatch:
    return CandidateMapletBatch(
        query_features=torch.tensor(
            [[[10.0], [11.0], [12.0], [0.0]], [[20.0], [21.0], [0.0], [0.0]]]
        ),
        query_mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
        support_features=torch.tensor(
            [[[30.0], [31.0], [32.0], [0.0]], [[40.0], [41.0], [42.0], [0.0]]]
        ),
        support_mask=torch.tensor([[True, True, True, False], [True, True, True, False]]),
        static_features=torch.zeros((2, 1)),
        target_track_indices=None,
        candidate_labels=None,
        edge_indices=torch.tensor([7, 8]),
    )


def test_non_anchor_orders_preserve_anchor_and_padded_suffixes() -> None:
    mask = _batch().query_mask
    order = _non_anchor_orders(mask, seed=3)

    assert torch.equal(order[:, 0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(order[0, :3].sort().values, torch.tensor([0, 1, 2]))
    assert torch.equal(order[0, 3:], torch.tensor([3]))
    assert torch.equal(order[1], torch.tensor([0, 1, 2, 3]))


def test_node_reorder_and_inverse_restore_pair_and_assignment_axes() -> None:
    batch = _batch()
    query_order = _non_anchor_orders(batch.query_mask, seed=0)
    support_order = _non_anchor_orders(batch.support_mask, seed=2)
    permuted = _reorder_inference_batch(
        batch, query_order=query_order, support_order=support_order
    )

    assert torch.equal(permuted.query_features[:, 0], batch.query_features[:, 0])
    assert torch.equal(permuted.support_features[:, 0], batch.support_features[:, 0])
    assert torch.equal(permuted.query_mask[:, 0], torch.tensor([True, True]))
    assert torch.equal(permuted.support_mask[:, 0], torch.tensor([True, True]))

    pair = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    probability = torch.cat([pair, torch.full((2, 4, 1), 99.0)], dim=2)
    permuted_pair = torch.gather(
        torch.gather(
            pair,
            1,
            query_order[:, :, None].expand(-1, -1, pair.shape[2]),
        ),
        2,
        support_order[:, None, :].expand(-1, pair.shape[1], -1),
    )
    permuted_probability = torch.cat([permuted_pair, probability[:, :, -1:]], dim=2)
    query_inverse = torch.argsort(query_order, dim=1)
    support_inverse = torch.argsort(support_order, dim=1)

    torch.testing.assert_close(
        _restore_pair_axis(
            permuted_pair,
            query_inverse=query_inverse,
            support_inverse=support_inverse,
        ),
        pair,
    )
    torch.testing.assert_close(
        _restore_query_probabilities(
            permuted_probability,
            query_inverse=query_inverse,
            support_inverse=support_inverse,
        ),
        probability,
    )


def test_stratified_group_sample_covers_queries_before_extra_groups() -> None:
    query_ids = np.asarray(["a", "a", "b", "b", "c", "c", "d"])
    selected = _stratified_group_sample(query_ids, sample_count=5, seed=9)

    assert len(selected) == 5
    assert len(np.unique(selected)) == 5
    assert set(query_ids[selected]) == {"a", "b", "c", "d"}


def test_restore_group_output_restores_support_view_prior_axis() -> None:
    original = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]])
    order = torch.tensor([[[1, 0], [1, 0]]])
    inverse = torch.argsort(order, dim=2)
    permuted = torch.gather(original, 2, order)

    torch.testing.assert_close(
        _restore_group_output(
            "support_view_prior_probabilities",
            permuted,
            candidate_inverse=None,
            view_inverse=inverse,
        ),
        original,
    )
