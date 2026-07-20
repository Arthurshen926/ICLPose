from __future__ import annotations

import math

import pytest
import torch

from feature_extract.vfm.localization.frozen_multiscale_pose_evidence import (
    LocalContextModeRatios,
    fixed_candidate_point_log_ratios,
    local_context_mode_ratios,
    materialize_dense_local_context_mode_maps,
    sample_dense_local_context_mode_ratios,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ContextPositionLikelihoodMaps,
)


def _maps(*, usable: torch.Tensor | None = None) -> ContextPositionLikelihoodMaps:
    logits = torch.zeros((2, 5, 5), dtype=torch.float32)
    logits[0, 2, 2] = 4.0
    valid = torch.ones_like(logits, dtype=torch.bool)
    if usable is None:
        usable = torch.tensor([True, False])
    return ContextPositionLikelihoodMaps(
        logits=logits,
        valid_cells=valid & usable[:, None, None],
        log_uniform_normalizers=torch.zeros((2,), dtype=torch.float32),
        template_usable=usable,
    )


def test_local_context_modes_are_normalized_against_their_own_uniform_null() -> None:
    modes = local_context_mode_ratios(
        maps=_maps(),
        anchor_xy=torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
        image_width=5,
        image_height=5,
        local_window_size=3,
    )

    valid = modes.valid_cells[0]
    assert int(valid.sum()) == 9
    assert torch.mean(torch.exp(modes.log_ratios[0, valid])).item() == pytest.approx(1.0)
    assert modes.log_ratios[0, 4] > modes.log_ratios[0, 0]
    assert not bool(modes.template_usable[1])
    assert not bool(modes.valid_cells[1].any())


def test_dense_modes_distinguish_geometric_miss_from_unavailable_template() -> None:
    modes = local_context_mode_ratios(
        maps=_maps(),
        anchor_xy=torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
        image_width=5,
        image_height=5,
        local_window_size=3,
    )
    dense = materialize_dense_local_context_mode_maps(modes=modes, grid_size=5)
    values, available, in_window = sample_dense_local_context_mode_ratios(
        maps=dense,
        projected_xy=torch.tensor(
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[0.0, 0.0], [2.0, 2.0]],
            ]
        ),
        projection_valid=torch.tensor([[True, True], [True, True]]),
        image_width=5,
        image_height=5,
    )

    assert available.tolist() == [[True, False], [True, False]]
    assert in_window.tolist() == [[True, False], [False, False]]
    assert values[0, 0] > 0.0
    assert values[1, 0].item() == 0.0


def test_dense_mode_materialization_rejects_out_of_bounds_sparse_cells() -> None:
    modes = LocalContextModeRatios(
        log_ratios=torch.zeros((1, 1)),
        valid_cells=torch.ones((1, 1), dtype=torch.bool),
        grid_rows=torch.tensor([[5]]),
        grid_columns=torch.tensor([[0]]),
        template_usable=torch.tensor([True]),
    )
    with pytest.raises(ValueError, match="exceed"):
        materialize_dense_local_context_mode_maps(modes=modes, grid_size=5)


def test_fixed_candidate_view_marginal_preserves_null_and_unknown_view_mass() -> None:
    logs = torch.tensor(
        [[[[math.log(4.0), math.log(1.0)], [math.log(9.0), 0.0]]]],
        dtype=torch.float32,
    )
    available = torch.tensor([[[[True, True], [True, False]]]])
    in_window = torch.tensor([[[[True, True], [True, False]]]])
    weights = torch.tensor([[[0.75, 0.25], [1.0, 0.0]]], dtype=torch.float32)
    probabilities = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
    null = torch.tensor([0.3], dtype=torch.float32)

    point_logs, candidate_ratios, contributed = fixed_candidate_point_log_ratios(
        view_log_ratios=logs,
        view_available=available,
        view_geometric_in_window=in_window,
        candidate_view_weights=weights,
        candidate_probabilities=probabilities,
        null_probabilities=null,
    )

    # Candidate 0 is 0.75 * 4 + 0.25 * 1, candidate 1 is 9.  The missing
    # padded view stays neutral but retains zero fixed support mass.
    assert candidate_ratios[0, 0, 0].item() == pytest.approx(3.25)
    assert candidate_ratios[0, 0, 1].item() == pytest.approx(9.0)
    assert torch.exp(point_logs)[0, 0].item() == pytest.approx(0.3 + 0.4 * 3.25 + 0.3 * 9.0)
    assert contributed[0, 0, 0].item() == pytest.approx(1.0)
    assert contributed[0, 0, 1].item() == pytest.approx(1.0)


def test_usable_view_outside_its_local_window_is_zero_not_neutral() -> None:
    logs = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    point_logs, candidate_ratios, _contributed = fixed_candidate_point_log_ratios(
        view_log_ratios=logs,
        view_available=torch.ones((1, 1, 1, 1), dtype=torch.bool),
        view_geometric_in_window=torch.zeros((1, 1, 1, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 1, 1), dtype=torch.float32),
        candidate_probabilities=torch.ones((1, 1), dtype=torch.float32),
        null_probabilities=torch.zeros((1,), dtype=torch.float32),
    )
    assert candidate_ratios.item() == 0.0
    assert torch.exp(point_logs).item() < 1e-30
