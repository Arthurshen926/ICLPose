from __future__ import annotations

import numpy as np
import torch

from feature_extract.tools.vfm.score_mast3r_pairwise_support_alignment import (
    _build_pair_maplet_context,
    _fixed_support_image_priors,
    _resample_pair_descriptor_grid,
    _score_pair_maplet_context,
)


def test_pair_descriptor_resampling_is_square_and_normalized() -> None:
    descriptors = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) + 1.0
    output = _resample_pair_descriptor_grid(
        descriptors, grid_size=5, device=torch.device("cpu")
    )
    assert output.shape == (5, 5, 4)
    assert torch.allclose(torch.linalg.vector_norm(output, dim=2), torch.ones((5, 5)))


def test_pair_maplet_context_is_spatial_and_neutral_when_geometry_is_unknown() -> None:
    query = torch.eye(49, dtype=torch.float32).reshape(7, 7, 49)
    support = query[2:5, 2:5].permute(2, 0, 1).unsqueeze(0)
    context = _build_pair_maplet_context(
        query_grid=query,
        support_patches=support,
        support_patch_valid=torch.ones((1, 3, 3), dtype=torch.bool),
        support_image_sizes=torch.tensor([[7.0, 7.0]], dtype=torch.float32),
        support_grid_size=7,
        temperature=0.1,
        minimum_support_fraction=1.0,
    )
    scores, active = _score_pair_maplet_context(
        context=context,
        affine_matrices=torch.eye(2, dtype=torch.float32)
        .reshape(1, 1, 2, 2)
        .repeat(3, 1, 1, 1),
        projected_anchor_xy=torch.tensor(
            [[[3.0, 3.0]], [[1.0, 1.0]], [[3.0, 3.0]]], dtype=torch.float32
        ),
        maplet_geometry_valid=torch.tensor([[True], [True], [False]]),
        image_width=7,
        image_height=7,
        temperature=0.1,
    )
    assert scores[0, 0] > scores[1, 0] + 1.0
    assert scores[2, 0].item() == 0.0
    assert active.tolist() == [[True], [True], [False]]


def test_fixed_support_image_priors_preserve_layout_mass() -> None:
    images, priors = _fixed_support_image_priors(
        support_ids=np.asarray(["b", "a", "b", "a"]),
        support_scores=np.asarray([2.0, 1.0, 2.0, 1.0], dtype=np.float32),
    )
    assert images.tolist() == ["a", "b"]
    assert np.allclose(priors, [1.0 / 3.0, 2.0 / 3.0])
