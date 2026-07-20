from __future__ import annotations

import torch

from feature_extract.vfm.localization.fulltrack_aligned_layout_probe import (
    aligned_spatial_dct_features,
)


def test_aligned_layout_dct_keeps_mean_and_has_zero_phase_for_uniform_agreement() -> None:
    query = torch.zeros((2, 2, 5, 5), dtype=torch.float32)
    query[:, 0] = 1.0
    values, usable = aligned_spatial_dct_features(
        query_patches=query,
        support_patches=query.clone(),
        query_valid=torch.ones((2, 5, 5), dtype=torch.bool),
        support_valid=torch.ones((2, 5, 5), dtype=torch.bool),
        dct_size=3,
        minimum_support_fraction=0.75,
        minimum_overlap_fraction=0.75,
    )
    assert usable.tolist() == [True, True]
    torch.testing.assert_close(values[:, :1], torch.ones((2, 1)))
    torch.testing.assert_close(values[:, 1:], torch.zeros((2, 8)), atol=1e-6, rtol=0.0)


def test_aligned_layout_dct_retains_relative_crop_phase() -> None:
    query = torch.zeros((2, 2, 5, 5), dtype=torch.float32)
    query[:, 0] = 1.0
    support = query.clone()
    # The two candidates have equal mean agreement (0.6), but the mismatched
    # cells occupy different crop locations.  A scalar NCC cannot distinguish
    # them, whereas at least one retained phase coefficient must.
    support[0, :, :2, :2] = torch.tensor([0.0, 1.0])[:, None, None]
    support[1, :, 3:, 3:] = torch.tensor([0.0, 1.0])[:, None, None]
    values, usable = aligned_spatial_dct_features(
        query_patches=query,
        support_patches=support,
        query_valid=torch.ones((2, 5, 5), dtype=torch.bool),
        support_valid=torch.ones((2, 5, 5), dtype=torch.bool),
        dct_size=3,
        minimum_support_fraction=0.75,
        minimum_overlap_fraction=0.75,
    )
    assert usable.tolist() == [True, True]
    torch.testing.assert_close(values[0, :1], values[1, :1], atol=1e-6, rtol=0.0)
    assert not torch.allclose(values[0, 1:], values[1, 1:])


def test_aligned_layout_dct_emits_neutral_placeholder_for_insufficient_coverage() -> None:
    patches = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
    patches[:, 0] = 1.0
    query_valid = torch.zeros((1, 5, 5), dtype=torch.bool)
    query_valid[:, :2, :2] = True
    values, usable = aligned_spatial_dct_features(
        query_patches=patches,
        support_patches=patches.clone(),
        query_valid=query_valid,
        support_valid=torch.ones((1, 5, 5), dtype=torch.bool),
        dct_size=3,
        minimum_support_fraction=0.75,
        minimum_overlap_fraction=0.75,
    )
    assert usable.tolist() == [False]
    torch.testing.assert_close(values, torch.zeros_like(values))
