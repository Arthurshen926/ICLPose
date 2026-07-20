from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    CONTEXT_ATTENTION_SCALES,
    CandidateContextAttentionProbe,
    ContextAttentionRuntimeArrays,
    context_valid_mask,
    crop_anchor_aligned_grid_absolute_coordinates,
    masked_view_log_mean,
)


def _unit_grid(image_count: int, grid_size: int, dimension: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17 + grid_size)
    values = torch.randn(
        image_count, grid_size, grid_size, dimension, generator=generator, dtype=torch.float32
    )
    return torch.nn.functional.normalize(values, p=2, dim=-1)


def _model(
    *,
    family: str = "context_attention_multiscale_context_only",
    position_encoding: str | None = None,
) -> CandidateContextAttentionProbe:
    sources = {
        scale.name: _unit_grid(3, scale.grid_size, 4) for scale in CONTEXT_ATTENTION_SCALES
    }
    runtime = ContextAttentionRuntimeArrays(
        query_image_indices=np.asarray([0], dtype=np.int64),
        support_image_indices=np.asarray([[[1, 2], [1, 2]]], dtype=np.int64),
        support_xy=np.asarray(
            [[[[50.0, 50.0], [50.0, 50.0]], [[50.0, 50.0], [50.0, 50.0]]]],
            dtype=np.float32,
        ),
        view_valid=np.asarray([[[True, True], [True, True]]]),
    )
    model = CandidateContextAttentionProbe(
        family=family,
        sources=sources,
        image_sizes=torch.as_tensor([[100.0, 100.0]] * 3),
        runtime=runtime,
        query_xy=np.asarray([[50.0, 50.0]], dtype=np.float32),
        base_candidate_probabilities=np.asarray([[0.4, 0.5]], dtype=np.float32),
        base_null_probabilities=np.asarray([0.1], dtype=np.float32),
        hidden_dim=8,
        heads=2,
        dropout=0.0,
        **({"position_encoding": position_encoding} if position_encoding is not None else {}),
    )
    # Make outputs depend on the context encoder rather than the intentionally
    # zero-initialized residual head.
    with torch.no_grad():
        model.view_head[-1].weight.fill_(0.05)
        model.view_head[-1].bias.zero_()
    return model.eval()


def test_context_mask_removes_the_entire_anchor_neighbourhood() -> None:
    valid = torch.ones((1, 25), dtype=torch.bool)
    masked = context_valid_mask(valid, window_size=5, center_mask_radius=1)

    assert int(masked.sum()) == 16
    assert not bool(masked[0, 2 * 5 + 2])
    assert not bool(masked[0, 1 * 5 + 1])
    assert bool(masked[0, 0])


def test_context_only_family_is_invariant_to_center_descriptor_changes() -> None:
    model = _model()
    rows = torch.as_tensor([0], dtype=torch.long)
    before, null_before, _views, _logits = model(rows)

    with torch.no_grad():
        for scale in CONTEXT_ATTENTION_SCALES:
            grid = model._grid(scale.name)
            center = scale.grid_size // 2
            replacement = torch.nn.functional.normalize(
                torch.tensor([7.0, 3.0, -5.0, 11.0], dtype=torch.float32), p=2, dim=0
            ).to(dtype=grid.dtype)
            grid[:, center, center] = replacement
    after, null_after, _views, _logits = model(rows)

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_per_view_marginal_is_permutation_invariant_and_handles_padding() -> None:
    logits = torch.as_tensor([[[0.2, 1.1, 3.0], [-4.0, 2.0, 7.0]]])
    valid = torch.as_tensor([[[True, True, False], [False, True, False]]])
    original = masked_view_log_mean(logits, valid)
    permutation = torch.as_tensor([1, 0, 2])
    permuted = masked_view_log_mean(logits[:, :, permutation], valid[:, :, permutation])

    torch.testing.assert_close(original, permuted)
    expected = torch.logsumexp(torch.as_tensor([0.2, 1.1]), dim=0) - np.log(2.0)
    torch.testing.assert_close(original[0, 0], expected)
    torch.testing.assert_close(original[0, 1], torch.as_tensor(2.0))


def test_context_probe_probability_mass_is_conserved() -> None:
    model = _model()
    candidate, null, view_logits, _logits = model(torch.as_tensor([0], dtype=torch.long))

    torch.testing.assert_close(candidate.sum(dim=1) + null, torch.ones((1,)))
    assert view_logits.shape == (1, 2, 2)


def test_absolute_crop_coordinates_preserve_full_image_phase() -> None:
    grid = torch.ones((1, 5, 5, 2), dtype=torch.float32)
    coordinates = crop_anchor_aligned_grid_absolute_coordinates(
        image_grids=grid,
        image_sizes=torch.as_tensor([[100.0, 100.0]]),
        image_indices=torch.as_tensor([0, 0]),
        xy=torch.as_tensor([[10.0, 10.0], [90.0, 90.0]]),
        window_size=3,
    )

    centre = 4
    torch.testing.assert_close(coordinates[0, centre], torch.as_tensor([0.0, 0.0]))
    torch.testing.assert_close(coordinates[1, centre], torch.as_tensor([1.0, 1.0]))


def test_absolute_position_only_control_is_descriptor_invariant() -> None:
    model = _model(
        family="absolute_phase_attention_position_only",
        position_encoding=ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    )
    rows = torch.as_tensor([0], dtype=torch.long)
    before, null_before, _views, _logits = model(rows)

    with torch.no_grad():
        for scale in CONTEXT_ATTENTION_SCALES:
            model._grid(scale.name).normal_()
    after, null_after, _views, _logits = model(rows)

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_context_family_cannot_silently_mix_position_spaces() -> None:
    with pytest.raises(ValueError, match="family and position encoding differ"):
        _model(position_encoding=ABSOLUTE_DUAL_FRAME_POSITION_ENCODING)
