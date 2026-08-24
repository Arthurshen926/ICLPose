from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization_goal_maplet.fulltoken_surface_pose_energy import (
    child_gated_fulltoken_surface_pose_energy,
    conditional_fulltoken_phase_pose_energy_control,
    conservative_fulltoken_phase_pose_energy,
    fulltoken_surface_pose_energy,
    parent_gated_fulltoken_surface_pose_energy,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_fulltoken_pattern_search import (
    _wide_oracle_candidate_index,
)


def _xy() -> np.ndarray:
    return np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)


def test_identical_fulltoken_surface_is_above_mismatched_surface():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    target = query[:, None, :].clone()
    mass = torch.ones(4, 1)
    valid = torch.ones(4, 1, dtype=torch.bool)
    good = fulltoken_surface_pose_energy(query, torch.ones(4), _xy(), target, mass, valid, local_radius_tokens=0)
    bad = fulltoken_surface_pose_energy(query, torch.ones(4), _xy(), -target, mass, valid, local_radius_tokens=0)
    assert float(good.score) == pytest.approx(1.0)
    assert float(bad.score) == pytest.approx(-1.0)


def test_disappearing_target_mass_cannot_improve_fixed_denominator_score():
    query = torch.tensor([[1.0, 0.0]] * 4)
    target = query[:, None, :].clone()
    valid = torch.ones(4, 1, dtype=torch.bool)
    present = fulltoken_surface_pose_energy(query, torch.ones(4), _xy(), target, torch.ones(4, 1), valid)
    partial = fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), target,
        torch.tensor([[1.0], [0.0], [1.0], [0.0]]), valid,
    )
    absent = fulltoken_surface_pose_energy(query, torch.ones(4), _xy(), target, torch.zeros(4, 1), valid)
    assert float(present.score) >= float(partial.score) >= float(absent.score)
    assert float(absent.score) == pytest.approx(-1.0)


def test_local_radius_recovers_one_token_shift_without_changing_denominator():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    target = torch.roll(query, shifts=1, dims=0)[:, None, :]
    mass = torch.ones(4, 1)
    valid = torch.ones(4, 1, dtype=torch.bool)
    exact = fulltoken_surface_pose_energy(query, torch.ones(4), _xy(), target, mass, valid, local_radius_tokens=0)
    tolerant = fulltoken_surface_pose_energy(query, torch.ones(4), _xy(), target, mass, valid, local_radius_tokens=1)
    assert float(tolerant.score) >= float(exact.score)


def test_target_mass_overflow_and_duplicate_coordinates_fail_closed():
    query = torch.ones(4, 2)
    target = torch.ones(4, 1, 2)
    with pytest.raises(ValueError, match="exceeds one"):
        fulltoken_surface_pose_energy(query, torch.ones(4), _xy(), target, torch.full((4, 1), 1.1), torch.ones(4, 1, dtype=torch.bool))
    bad_xy = _xy(); bad_xy[1] = bad_xy[0]
    with pytest.raises(ValueError, match="unique"):
        fulltoken_surface_pose_energy(query, torch.ones(4), bad_xy, target, torch.ones(4, 1), torch.ones(4, 1, dtype=torch.bool))


def test_parent_gated_energy_requires_both_parent_and_appearance_evidence():
    query = torch.tensor([[1.0, 0.0]] * 4)
    target = query[:, None, :].clone()
    mass = torch.ones(4, 1)
    valid = torch.ones(4, 1, dtype=torch.bool)
    query_parent = torch.full((4, 1), 7)
    probability = torch.ones(4, 1)
    present = parent_gated_fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), query_parent, probability,
        target, mass, valid, torch.full((4, 1), 7),
    )
    wrong_parent = parent_gated_fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), query_parent, probability,
        target, mass, valid, torch.full((4, 1), 8),
    )
    missing_appearance = parent_gated_fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), query_parent, probability,
        target, torch.zeros_like(mass), valid, torch.full((4, 1), 7),
    )
    assert float(present.score) == pytest.approx(1.0)
    assert float(wrong_parent.score) == pytest.approx(-1.0)
    assert float(missing_appearance.score) == pytest.approx(-1.0)


def test_contrastive_cosine_floor_removes_unrelated_mass_without_breaking_missingness():
    query = torch.tensor([[1.0, 0.0]] * 4)
    unrelated = torch.tensor([[[0.0, 1.0]]] * 4)
    valid = torch.ones(4, 1, dtype=torch.bool)
    legacy = fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), unrelated, torch.ones(4, 1), valid,
        local_radius_tokens=0,
    )
    contrastive = fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), unrelated, torch.ones(4, 1), valid,
        local_radius_tokens=0, minimum_cosine_evidence=0.25,
    )
    missing = fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), unrelated, torch.zeros(4, 1), valid,
        local_radius_tokens=0, minimum_cosine_evidence=0.25,
    )
    assert float(legacy.score) == pytest.approx(0.0)
    assert float(contrastive.score) == pytest.approx(-1.0)
    assert float(missing.score) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="minimum_cosine_evidence"):
        fulltoken_surface_pose_energy(
            query, torch.ones(4), _xy(), unrelated, torch.ones(4, 1), valid,
            minimum_cosine_evidence=1.0,
        )


def test_child_gate_uses_soft_identity_without_hard_correspondence():
    query = torch.tensor([[1.0, 0.0]] * 4)
    target = query[:, None, :].clone()
    query_child = torch.tensor([[3, 7]] * 4)
    probability = torch.tensor([[0.75, 0.25]] * 4)
    shared = child_gated_fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), query_child, probability,
        target, torch.ones(4, 1), torch.ones(4, 1, dtype=torch.bool),
        torch.full((4, 1), 3),
    )
    secondary = child_gated_fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), query_child, probability,
        target, torch.ones(4, 1), torch.ones(4, 1, dtype=torch.bool),
        torch.full((4, 1), 7),
    )
    missing = child_gated_fulltoken_surface_pose_energy(
        query, torch.ones(4), _xy(), query_child, probability,
        target, torch.zeros(4, 1), torch.ones(4, 1, dtype=torch.bool),
        torch.full((4, 1), 3),
    )
    assert float(shared.score) > float(secondary.score) > float(missing.score)


def test_conservative_phase_requires_both_axes_and_missing_edges_do_not_improve():
    query = torch.tensor([
        [1.0, 0.0], [0.8, 0.2], [0.5, 0.5],
        [0.9, 0.1], [0.6, 0.4], [0.2, 0.8],
        [0.7, 0.3], [0.3, 0.7], [0.0, 1.0],
    ])
    target = query[:, None, :].clone()
    mass = torch.ones(9, 1)
    valid = torch.ones(9, 1, dtype=torch.bool)
    exact = conservative_fulltoken_phase_pose_energy(
        query, target, mass, valid, height=3, width=3
    )
    missing_mass = mass.clone(); missing_mass[4] = 0.0
    missing = conservative_fulltoken_phase_pose_energy(
        query, target, missing_mass, valid, height=3, width=3
    )
    vertical_bad = target.reshape(3, 3, 1, 2).flip(0).reshape(9, 1, 2)
    one_axis_wrong = conservative_fulltoken_phase_pose_energy(
        query, vertical_bad, mass, valid, height=3, width=3
    )
    assert float(exact.score) == pytest.approx(1.0, abs=1e-6)
    assert float(missing.score) <= float(exact.score)
    assert float(one_axis_wrong.score) < float(exact.score)
    assert float(one_axis_wrong.score) == pytest.approx(
        min(float(one_axis_wrong.horizontal_score), float(one_axis_wrong.vertical_score))
    )


def test_shift_tolerant_phase_recovers_joint_shift_and_preserves_missing_monotonicity():
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(5, 6, 8, generator=generator)
    # Shift the entire target field by one column; the empty border is unknown.
    target_grid = torch.zeros_like(query)
    target_grid[:, 1:] = query[:, :-1]
    target = target_grid.reshape(30, 1, 8)
    mass = torch.ones(30, 1)
    mass.reshape(5, 6, 1)[:, 0] = 0.0
    valid = torch.ones(30, 1, dtype=torch.bool)
    exact = conservative_fulltoken_phase_pose_energy(
        query.reshape(30, 8), target, mass, valid, height=5, width=6
    )
    tolerant = conservative_fulltoken_phase_pose_energy(
        query.reshape(30, 8), target, mass, valid, height=5, width=6,
        maximum_shift_tokens=1,
    )
    more_missing_mass = mass.clone()
    more_missing_mass.reshape(5, 6, 1)[:, 3] = 0.0
    missing = conservative_fulltoken_phase_pose_energy(
        query.reshape(30, 8), target, more_missing_mass, valid, height=5, width=6,
        maximum_shift_tokens=1,
    )
    assert float(tolerant.score) > float(exact.score)
    assert float(missing.score) <= float(tolerant.score) + 1.0e-7
    with pytest.raises(ValueError, match="maximum_shift_tokens"):
        conservative_fulltoken_phase_pose_energy(
            query.reshape(30, 8), target, mass, valid, height=5, width=6,
            maximum_shift_tokens=5,
        )


def test_additive_phase_fixes_conditional_mixture_disappearance_counterexample():
    query = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [-1.0, 0.0], [0.0, -1.0],
    ])
    correct = query[:, None, :]
    distractor = torch.tensor([
        [0.0, 1.0], [-1.0, 0.0],
        [0.0, -1.0], [1.0, 0.0],
    ])[:, None, :]
    target = torch.cat((correct, distractor), dim=1)
    mass = torch.full((4, 2), 0.5)
    valid = torch.ones(4, 2, dtype=torch.bool)
    disappeared = mass.clone()
    disappeared[:, 1] = 0.0

    conditional_before = conditional_fulltoken_phase_pose_energy_control(
        query, target, mass, valid, height=2, width=2
    )
    conditional_after = conditional_fulltoken_phase_pose_energy_control(
        query, target, disappeared, valid, height=2, width=2
    )
    additive_before = conservative_fulltoken_phase_pose_energy(
        query, target, mass, valid, height=2, width=2
    )
    additive_after = conservative_fulltoken_phase_pose_energy(
        query, target, disappeared, valid, height=2, width=2
    )
    assert float(conditional_after.score) > float(conditional_before.score)
    assert float(conditional_after.conditional_content_score) > float(
        conditional_before.conditional_content_score
    )
    assert float(conditional_after.mass_observability) < float(
        conditional_before.mass_observability
    )
    assert float(additive_after.score) <= float(additive_before.score) + 1.0e-7


@pytest.mark.parametrize("maximum_shift_tokens", [0, 1, 2])
@pytest.mark.parametrize("slot_pair_reduction", ["additive_product", "maximum_bottleneck"])
def test_additive_phase_random_fractional_mass_and_validity_disappearance_is_monotone(
    maximum_shift_tokens: int,
    slot_pair_reduction: str,
):
    generator = torch.Generator().manual_seed(101 + maximum_shift_tokens)
    height, width, slots, channels = 5, 6, 3, 9
    query = torch.randn(height * width, channels, generator=generator)
    target = torch.randn(height * width, slots, channels, generator=generator)
    mass = torch.rand(height * width, slots, generator=generator)
    mass = 0.9 * mass / mass.sum(dim=1, keepdim=True)
    valid = torch.rand(height * width, slots, generator=generator) > 0.1
    reduced_mass = mass * torch.rand(height * width, slots, generator=generator)
    reduced_valid = valid & (torch.rand(height * width, slots, generator=generator) > 0.3)
    before = conservative_fulltoken_phase_pose_energy(
        query, target, mass, valid, height=height, width=width,
        maximum_shift_tokens=maximum_shift_tokens,
        slot_pair_reduction=slot_pair_reduction,
    )
    after = conservative_fulltoken_phase_pose_energy(
        query, target, reduced_mass, reduced_valid, height=height, width=width,
        maximum_shift_tokens=maximum_shift_tokens,
        slot_pair_reduction=slot_pair_reduction,
    )
    assert float(after.score) <= float(before.score) + 2.0e-7
    assert float(after.horizontal_observability) <= float(before.horizontal_observability) + 2.0e-7
    assert float(after.vertical_observability) <= float(before.vertical_observability) + 2.0e-7


def test_wide_oracle_diagnostic_excludes_gt_anchor_and_uses_joint_scale():
    selected = _wide_oracle_candidate_index(
        np.asarray([True, True, True, True]),
        np.asarray([0.0, 0.4, 0.9, 1.5]),
        np.asarray([0.0, 30.0, 8.0, 10.0]),
    )
    assert selected == 2
