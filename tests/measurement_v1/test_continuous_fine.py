from __future__ import annotations

import torch

from feature_extract.vfm.measurement_v1.continuous_fine import (
    continuous_fine_nll,
    local_map_refined_xy,
)


def test_continuous_fine_nll_uses_bilinear_probability_at_subcell_target() -> None:
    logits = torch.full((1, 64), -8.0)
    for idx in (18, 19, 26, 27):
        logits[0, idx] = 2.0
    target_xy = torch.tensor([[2.5, 2.5]], dtype=torch.float32)

    loss, metrics = continuous_fine_nll(logits, target_xy)

    assert 1.3 < float(loss.item()) < 1.5
    assert metrics["valid_count"] == 1.0
    assert 1.3 < metrics["nll"] < 1.5


def test_local_map_refined_xy_does_not_average_across_distant_modes() -> None:
    logits = torch.full((1, 64), -10.0)
    logits[0, 0] = 8.0
    logits[0, 63] = 7.5

    xy = local_map_refined_xy(logits, radius=1)

    assert torch.linalg.norm(xy[0] - torch.tensor([0.5, 0.5])).item() < 0.2
    assert torch.linalg.norm(xy[0] - torch.tensor([4.0, 4.0])).item() > 3.0
