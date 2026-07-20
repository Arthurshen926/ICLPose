from __future__ import annotations

import pytest
import torch

from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance import (
    aligned_patch_ncc,
    coverage_weighted_view_appearance,
)


def test_aligned_patch_ncc_preserves_aligned_cosine_and_masks_edges() -> None:
    query = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
    support = torch.zeros_like(query)
    query[:, 0] = 1.0
    support[0, 0] = 1.0
    support[1, 0] = -1.0
    query_valid = torch.ones((2, 3, 3), dtype=torch.bool)
    support_valid = query_valid.clone()
    support_valid[1, 0, :] = False

    result = aligned_patch_ncc(
        query_patches=query,
        support_patches=support,
        query_valid=query_valid,
        support_valid=support_valid,
        minimum_support_fraction=0.5,
        minimum_overlap_fraction=0.5,
    )

    assert result.usable.tolist() == [True, True]
    assert torch.allclose(result.score, torch.tensor([1.0, -1.0]))
    assert torch.allclose(result.overlap_fraction, torch.tensor([1.0, 6.0 / 9.0]))
    assert torch.allclose(result.support_fraction, torch.tensor([1.0, 6.0 / 9.0]))


def test_aligned_patch_ncc_marks_low_coverage_unknown_not_negative() -> None:
    query = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    support = query.clone()
    query_valid = torch.ones((1, 3, 3), dtype=torch.bool)
    support_valid = torch.zeros((1, 3, 3), dtype=torch.bool)
    support_valid[:, 1, 1] = True

    result = aligned_patch_ncc(
        query_patches=query,
        support_patches=support,
        query_valid=query_valid,
        support_valid=support_valid,
        minimum_support_fraction=0.5,
        minimum_overlap_fraction=0.5,
    )

    assert result.usable.tolist() == [False]
    assert torch.isnan(result.score).all()


def test_coverage_weighted_view_appearance_keeps_views_separate_until_marginalization() -> None:
    scores = torch.tensor(
        [
            [[0.2, 0.9], [0.8, 0.4], [0.7, 0.6]],
            [[0.1, 0.3], [0.9, 0.2], [0.5, 0.7]],
        ],
        dtype=torch.float32,
    )
    usable = torch.tensor(
        [
            [[True, True], [True, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ]
    )
    weights = torch.tensor([[0.25, 0.75, 0.0], [0.4, 0.6, 0.0]], dtype=torch.float32)

    mean, maximum, mass = coverage_weighted_view_appearance(
        view_scores=scores,
        view_usable=usable,
        view_weights=weights,
    )

    assert torch.allclose(mean[0], torch.tensor([0.65, 0.9]))
    assert torch.allclose(maximum[0], torch.tensor([0.8, 0.9]))
    assert torch.allclose(mass[0], torch.tensor([1.0, 0.25]))
    assert torch.isnan(mean[1]).all()
    assert torch.isnan(maximum[1]).all()
    assert torch.equal(mass[1], torch.zeros(2))


def test_appearance_rejects_unmasked_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="invalid"):
        aligned_patch_ncc(
            query_patches=torch.tensor([[[[float("nan")]]]]),
            support_patches=torch.ones((1, 1, 1, 1)),
            query_valid=torch.ones((1, 1, 1), dtype=torch.bool),
            support_valid=torch.ones((1, 1, 1), dtype=torch.bool),
            minimum_support_fraction=1.0,
            minimum_overlap_fraction=1.0,
        )
