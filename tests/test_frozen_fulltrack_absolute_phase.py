from __future__ import annotations

import torch

from feature_extract.vfm.localization.frozen_fulltrack_absolute_phase import (
    ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE,
    ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE,
    ABSOLUTE_PHASE_VISUAL_PROFILES,
    FULLTRACK_ABSOLUTE_PHASE_POSITION_FEATURE_NAMES,
    FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES,
    FULLTRACK_ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES,
    absolute_phase_anchor_mask,
    absolute_phase_visual_feature_names,
    batched_absolute_phase_position_control_features,
    batched_absolute_phase_region_transport_features,
    profile_feature_slices,
)


def test_absolute_phase_region_transport_preserves_nonrecentred_image_phase() -> None:
    # Every 6x6 cell has a distinct descriptor.  Swapping the left and right
    # image thirds changes only absolute support-region placement; a centred
    # crop would not retain this signal.
    query = torch.eye(36, dtype=torch.float32).reshape(1, 6, 6, 36)
    matched = query.clone()
    shifted = query.clone()
    shifted[:, :, :2] = query[:, :, 4:]
    shifted[:, :, 4:] = query[:, :, :2]
    valid = torch.ones((1, 6, 6), dtype=torch.bool)
    reference = batched_absolute_phase_region_transport_features(
        query, valid, matched, valid, region_grid_size=3, temperature=0.01
    )[0]
    moved = batched_absolute_phase_region_transport_features(
        query, valid, shifted, valid, region_grid_size=3, temperature=0.01
    )[0]
    names = absolute_phase_visual_feature_names("unit", region_grid_size=3)
    q_left_to_left = names.index("unit_qregion_r0_c0_sregion_r0_c0_attention_mass")
    q_left_to_right = names.index("unit_qregion_r0_c0_sregion_r0_c2_attention_mass")
    assert torch.isfinite(reference).all()
    assert torch.isfinite(moved).all()
    assert reference[q_left_to_left] > reference[q_left_to_right]
    assert moved[q_left_to_right] > moved[q_left_to_left]


def test_absolute_phase_visual_columns_exclude_mask_geometry_control() -> None:
    assert FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES == (
        *FULLTRACK_ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES,
        *FULLTRACK_ABSOLUTE_PHASE_POSITION_FEATURE_NAMES,
    )
    assert not any(
        "coverage" in name or "count" in name or "availability" in name
        for name in FULLTRACK_ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES
    )
    assert all("coverage_control" in name for name in FULLTRACK_ABSOLUTE_PHASE_POSITION_FEATURE_NAMES)
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES:
        assert ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE[profile.name]
        assert ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE[profile.name]
    slices = profile_feature_slices()
    assert max(item.stop for item in slices.values()) == len(
        FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES
    )


def test_absolute_phase_position_control_only_observes_anchor_mask_geometry() -> None:
    sizes = torch.tensor([[100.0, 100.0]])
    indices = torch.tensor([0])
    center = absolute_phase_anchor_mask(
        image_sizes=sizes,
        image_indices=indices,
        xy=torch.tensor([[50.0, 50.0]]),
        grid_size=16,
    )
    corner = absolute_phase_anchor_mask(
        image_sizes=sizes,
        image_indices=indices,
        xy=torch.tensor([[2.0, 2.0]]),
        grid_size=16,
    )
    center_control = batched_absolute_phase_position_control_features(
        center, center, region_grid_size=3
    )
    corner_control = batched_absolute_phase_position_control_features(
        corner, corner, region_grid_size=3
    )
    assert center.sum().item() == 16 * 16 - 9
    assert corner.sum().item() == 16 * 16 - 4
    assert torch.isfinite(center_control).all()
    assert torch.isfinite(corner_control).all()
    assert not torch.equal(center_control, corner_control)
