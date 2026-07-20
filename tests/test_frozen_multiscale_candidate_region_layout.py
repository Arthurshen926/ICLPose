from __future__ import annotations

import torch

from feature_extract.vfm.localization.frozen_multiscale_candidate_region_layout import (
    REGION_LAYOUT_STATISTIC_NAMES,
    region_layout_similarity,
)


def test_region_layout_similarity_retains_partial_crop_as_explicit_availability() -> None:
    query = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
    support = torch.zeros_like(query)
    query[:, 0] = 1.0
    support[0, 0] = 1.0
    support[1, 0] = -1.0
    query_valid = torch.ones((2, 3, 3), dtype=torch.bool)
    support_valid = torch.ones((2, 3, 3), dtype=torch.bool)
    support_valid[1, 2, 2] = False

    result = region_layout_similarity(
        query_patches=query,
        support_patches=support,
        query_valid=query_valid,
        support_valid=support_valid,
        region_grid_size=3,
    )

    assert REGION_LAYOUT_STATISTIC_NAMES[0] == "global_pool_cosine"
    assert result.usable.tolist() == [[True, True, True, True], [True, True, True, True]]
    assert torch.allclose(result.scores[0], torch.ones(4))
    assert torch.allclose(result.scores[1], -torch.ones(4))
    assert result.region_pair_count.tolist() == [9, 8]


def test_region_layout_similarity_marks_empty_overlap_unknown() -> None:
    patches = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    query_valid = torch.ones((1, 3, 3), dtype=torch.bool)
    support_valid = torch.zeros((1, 3, 3), dtype=torch.bool)
    result = region_layout_similarity(
        query_patches=patches,
        support_patches=patches,
        query_valid=query_valid,
        support_valid=support_valid,
    )
    assert result.usable.tolist() == [[False, False, False, False]]
    assert torch.isnan(result.scores).all()
    assert result.region_pair_count.tolist() == [0]
