from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ContextAttentionRuntimeArrays,
)
from feature_extract.vfm.localization.frozen_absolute_phase_probe import (
    AbsolutePhaseProfile,
    FrozenCandidateMultiscaleCropRuntime,
    PairwiseDescriptorCrops,
    deterministic_support_channel_permutation,
    evaluate_absolute_phase_profile,
)


def _phase_profile(window_size: int = 5) -> AbsolutePhaseProfile:
    return AbsolutePhaseProfile(
        name="phase",
        source_name="radio_final",
        window_size=window_size,
        score_kind="translation_phase",
        maximum_translation_cells=min(2, window_size // 2),
        temperature=0.05,
        alignment_sigma_cells=0.75,
    )


def _crops(*, support_shift: int = 0, valid: bool = True) -> PairwiseDescriptorCrops:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn((1, 25, 32), generator=generator)
    query = torch.nn.functional.normalize(query, dim=2)
    support = query.reshape(1, 5, 5, 32).roll(shifts=(0, support_shift), dims=(1, 2))
    return PairwiseDescriptorCrops(
        query_descriptors=query,
        query_valid=torch.ones((1, 25), dtype=torch.bool),
        support_descriptors=support.reshape(1, 25, 32),
        support_valid=torch.full((1, 25), valid, dtype=torch.bool),
        edge_valid=torch.as_tensor([valid]),
    )


def test_translation_phase_prefers_the_declared_zero_relative_phase() -> None:
    aligned = evaluate_absolute_phase_profile(_crops(), profile=_phase_profile())
    shifted = evaluate_absolute_phase_profile(_crops(support_shift=1), profile=_phase_profile())

    assert bool(aligned.available[0])
    assert bool(shifted.available[0])
    assert float(aligned.log_ratios[0]) > float(shifted.log_ratios[0])
    assert float(aligned.zero_shift_coverage[0]) == 1.0


def test_local_bipartite_field_prefers_aligned_unique_structure() -> None:
    profile = AbsolutePhaseProfile(
        name="local_bipartite",
        source_name="radio_final",
        window_size=5,
        score_kind="local_bipartite",
        maximum_translation_cells=2,
        temperature=0.05,
        alignment_sigma_cells=0.75,
    )
    aligned = evaluate_absolute_phase_profile(_crops(), profile=profile)
    shifted = evaluate_absolute_phase_profile(_crops(support_shift=1), profile=profile)

    assert bool(aligned.available[0])
    assert bool(shifted.available[0])
    assert float(aligned.log_ratios[0]) > float(shifted.log_ratios[0])


def test_local_bipartite_field_is_bidirectional_and_descriptor_sensitive() -> None:
    profile = AbsolutePhaseProfile(
        name="local_bipartite",
        source_name="radio_final",
        window_size=5,
        score_kind="local_bipartite",
        maximum_translation_cells=2,
        temperature=0.05,
        alignment_sigma_cells=0.75,
    )
    crops = _crops()
    permutation = deterministic_support_channel_permutation(
        source_name="radio_final", descriptor_dim=32, device=torch.device("cpu")
    )
    visual = evaluate_absolute_phase_profile(crops, profile=profile)
    control = evaluate_absolute_phase_profile(
        crops, profile=profile, support_channel_permutation=permutation
    )

    assert float(visual.log_ratios[0]) > float(control.log_ratios[0])


def test_channel_permutation_is_a_descriptor_destroying_paired_control() -> None:
    crops = _crops()
    profile = AbsolutePhaseProfile(
        name="center",
        source_name="radio_final",
        window_size=1,
        score_kind="center_cosine",
        maximum_translation_cells=0,
        temperature=0.10,
        alignment_sigma_cells=0.0,
    )
    permutation = deterministic_support_channel_permutation(
        source_name="radio_final", descriptor_dim=32, device=torch.device("cpu")
    )
    visual = evaluate_absolute_phase_profile(crops, profile=profile)
    control = evaluate_absolute_phase_profile(
        crops, profile=profile, support_channel_permutation=permutation
    )

    assert not torch.equal(permutation, torch.arange(32))
    assert float(visual.log_ratios[0]) > float(control.log_ratios[0])


def test_missing_support_is_unknown_not_synthetic_negative_evidence() -> None:
    evidence = evaluate_absolute_phase_profile(
        _crops(valid=False), profile=_phase_profile()
    )

    assert not bool(evidence.available[0])
    torch.testing.assert_close(evidence.log_ratios, torch.zeros((1,)))
    torch.testing.assert_close(evidence.zero_shift_coverage, torch.zeros((1,)))


def _runtime(*, candidate_permutation: tuple[int, int] = (0, 1)) -> FrozenCandidateMultiscaleCropRuntime:
    generator = torch.Generator().manual_seed(23)
    sources = {
        "radio_final": torch.randn((3, 5, 5, 8), generator=generator),
        "radio_intermediate": torch.randn((3, 5, 5, 8), generator=generator),
        "alike": torch.randn((3, 5, 5, 8), generator=generator),
    }
    support_indices = np.asarray([[[1], [2]]], dtype=np.int64)[:, candidate_permutation]
    support_xy = np.asarray([[[[50.0, 50.0]], [[50.0, 50.0]]]], dtype=np.float32)[
        :, candidate_permutation
    ]
    runtime = ContextAttentionRuntimeArrays(
        query_image_indices=np.asarray([0], dtype=np.int64),
        support_image_indices=support_indices,
        support_xy=support_xy,
        view_valid=np.ones((1, 2, 1), dtype=bool),
    )
    return FrozenCandidateMultiscaleCropRuntime(
        sources=sources,
        image_sizes=torch.full((3, 2), 100.0),
        runtime=runtime,
    )


def test_dynamic_crops_are_candidate_permutation_equivariant() -> None:
    baseline = _runtime()
    permuted = _runtime(candidate_permutation=(1, 0))
    rows = torch.as_tensor([0])
    xy = torch.as_tensor([[[50.0, 50.0], [50.0, 50.0]]])
    base, _views, _projection = baseline.pairwise_crops_at_query_xy(
        rows, xy, source_windows={"radio_final": 3}
    )
    changed, _views, _projection = permuted.pairwise_crops_at_query_xy(
        rows, xy[:, [1, 0]], source_windows={"radio_final": 3}
    )
    profile = AbsolutePhaseProfile(
        name="center",
        source_name="radio_final",
        window_size=1,
        score_kind="center_cosine",
        maximum_translation_cells=0,
        temperature=0.10,
        alignment_sigma_cells=0.0,
    )
    base_score = evaluate_absolute_phase_profile(base["radio_final"], profile=profile)
    changed_score = evaluate_absolute_phase_profile(changed["radio_final"], profile=profile)

    torch.testing.assert_close(
        changed_score.log_ratios.reshape(1, 2, 1)[:, [1, 0]],
        base_score.log_ratios.reshape(1, 2, 1),
    )
