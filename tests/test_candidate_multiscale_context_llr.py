from __future__ import annotations

import inspect

import torch

from feature_extract.vfm.localization.candidate_multiscale_context_llr import (
    CandidateMultiscaleContextLLR,
    fixed_support_view_mixture_log_likelihood_ratio,
    score_fixed_global_topl_phase_pose,
    source_candidate_log_likelihood_ratios,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_candidate_slots,
    permute_runtime_support_image_appearance_only,
    permute_runtime_support_appearance,
)


def _sources() -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    values: dict[str, torch.Tensor] = {}
    for name, side, dim in (
        ("radio_final", 7, 8),
        ("radio_intermediate", 11, 10),
        ("alike", 15, 6),
    ):
        values[name] = torch.nn.functional.normalize(torch.randn(4, side, side, dim), dim=-1)
    return values


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[45.0, 35.0], [53.0, 42.0]]),
        support_image_indices=torch.tensor(
            [
                [[2, 3], [2, 3]],
                [[2, 3], [2, 3]],
            ]
        ),
        support_xy=torch.tensor(
            [
                [[[42.0, 33.0], [47.0, 37.0]], [[62.0, 50.0], [64.0, 53.0]]],
                [[[49.0, 39.0], [54.0, 44.0]], [[65.0, 51.0], [69.0, 55.0]]],
            ]
        ),
        support_view_valid=torch.ones((2, 2, 2), dtype=torch.bool),
        candidate_view_weights=torch.full((2, 2, 2), 0.5),
        candidate_probabilities=torch.tensor([[0.45, 0.35], [0.60, 0.30]]),
        null_probabilities=torch.tensor([0.20, 0.10]),
    )


def _model() -> CandidateMultiscaleContextLLR:
    return CandidateMultiscaleContextLLR(
        sources=_sources(),
        image_sizes=torch.tensor([[96.0, 72.0]]).repeat(4, 1),
        hidden_dim=8,
        edge_chunk_size=3,
    )


def _randomize_heads(model: CandidateMultiscaleContextLLR) -> None:
    with torch.no_grad():
        for head in model.heads.values():
            final = head.network[-1]
            assert isinstance(final, torch.nn.Linear)
            final.weight.normal_(mean=0.0, std=0.2)


def test_context_phase_llr_is_target_free_and_per_source() -> None:
    signature = inspect.signature(CandidateMultiscaleContextLLR.forward)
    assert "pose" not in signature.parameters
    assert "target" not in signature.parameters
    model = _model().eval()
    _randomize_heads(model)
    prediction = model(runtime=_runtime())
    assert set(prediction.source_edge_log_likelihood_ratios) == {
        "radio_final",
        "radio_intermediate",
        "alike",
    }
    assert prediction.shape == (2, 2, 2)
    assert torch.max(torch.abs(prediction.source_edge_log_likelihood_ratios["alike"])) <= 4.0


def test_context_phase_llr_zero_visual_control_is_exactly_neutral() -> None:
    model = _model().eval()
    _randomize_heads(model)
    prediction = model(
        runtime=_runtime(),
        visual_source_scales={"radio_final": 0.0, "radio_intermediate": 0.0, "alike": 0.0},
    )
    for value in prediction.source_edge_log_likelihood_ratios.values():
        assert torch.equal(value, torch.zeros_like(value))


def test_context_phase_llr_candidate_slot_permutation_is_equivariant() -> None:
    model = _model().eval()
    _randomize_heads(model)
    runtime = _runtime()
    order = torch.tensor([[1, 0], [1, 0]])
    original = model(runtime=runtime)
    permuted = model(runtime=permute_runtime_candidate_slots(runtime, permutations=order))
    for name in original.source_edge_log_likelihood_ratios:
        assert torch.allclose(
            permuted.source_edge_log_likelihood_ratios[name],
            original.source_edge_log_likelihood_ratios[name].gather(
                1, order[:, :, None].expand(-1, -1, runtime.support_view_count)
            ),
            atol=1e-5,
        )


def test_context_phase_llr_support_permutation_changes_only_appearance_evidence() -> None:
    model = _model().eval()
    _randomize_heads(model)
    runtime = _runtime()
    normal = model(runtime=runtime)
    permuted = model(runtime=permute_runtime_support_appearance(runtime, shift=1))
    differences = [
        torch.max(torch.abs(normal.source_edge_log_likelihood_ratios[name] - permuted.source_edge_log_likelihood_ratios[name])).item()
        for name in normal.source_edge_log_likelihood_ratios
    ]
    assert max(differences) > 1e-5
    assert torch.equal(runtime.query_xy, permute_runtime_support_appearance(runtime, shift=1).query_xy)


def test_context_phase_llr_image_only_support_control_preserves_geometry_and_availability() -> None:
    model = _model().eval()
    _randomize_heads(model)
    runtime = _runtime()
    permuted_runtime = permute_runtime_support_image_appearance_only(runtime, shift=1)
    assert not torch.equal(runtime.support_image_indices, permuted_runtime.support_image_indices)
    assert torch.equal(runtime.support_xy, permuted_runtime.support_xy)
    assert torch.equal(runtime.support_view_valid, permuted_runtime.support_view_valid)
    normal = model(runtime=runtime)
    permuted = model(runtime=permuted_runtime)
    for name in normal.source_edge_usable:
        assert torch.equal(normal.source_edge_usable[name], permuted.source_edge_usable[name])


def test_fixed_view_mixture_preserves_duplicate_view_mass() -> None:
    base = fixed_support_view_mixture_log_likelihood_ratio(
        edge_log_likelihood_ratios=torch.tensor([[[torch.log(torch.tensor(3.0)), 0.0]]]),
        edge_usable=torch.tensor([[[True, True]]]),
        candidate_view_weights=torch.tensor([[[0.5, 0.5]]]),
    )
    duplicated = fixed_support_view_mixture_log_likelihood_ratio(
        edge_log_likelihood_ratios=torch.tensor([[[torch.log(torch.tensor(3.0)), torch.log(torch.tensor(3.0)), 0.0, 0.0]]]),
        edge_usable=torch.tensor([[[True, True, True, True]]]),
        candidate_view_weights=torch.tensor([[[0.25, 0.25, 0.25, 0.25]]]),
    )
    assert torch.allclose(base, duplicated, atol=1e-6)


def test_pose_scoring_uses_neutral_out_of_window_evidence_and_fixed_null() -> None:
    model = _model().eval()
    _randomize_heads(model)
    runtime = _runtime()
    prediction = model(runtime=runtime)
    candidate = source_candidate_log_likelihood_ratios(prediction=prediction, runtime=runtime)
    offsets = torch.tensor(
        [
            [[[0.0, 0.0], [30.0, 0.0]], [[0.0, 0.0], [30.0, 0.0]]],
            [[[30.0, 0.0], [30.0, 0.0]], [[30.0, 0.0], [30.0, 0.0]]],
        ]
    )
    score = score_fixed_global_topl_phase_pose(
        prediction=prediction,
        runtime=runtime,
        candidate_projection_offsets_xy=offsets,
        candidate_projection_valid=torch.ones(offsets.shape[:-1], dtype=torch.bool),
        local_radius_px=8.0,
        source_weights={"radio_final": 1.0, "radio_intermediate": 0.0, "alike": 0.0},
    )
    assert score.candidate_projection_compatible.shape == (2, 2, 2)
    assert torch.allclose(score.candidate_log_likelihood_ratios[1], torch.zeros_like(score.candidate_log_likelihood_ratios[1]))
    assert torch.allclose(score.source_candidate_log_likelihood_ratios["radio_final"], candidate["radio_final"])
