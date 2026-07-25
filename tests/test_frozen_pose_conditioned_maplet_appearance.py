from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization.frozen_pose_conditioned_maplet_appearance import (
    FrozenCandidateMapletEvidenceLayout,
    FrozenMapletAppearanceProfile,
    build_frozen_candidate_maplet_profile_layout,
    deterministic_support_descriptor_derangement,
    fixed_candidate_maplet_group_log_ratios,
    pool_maplet_neighbor_log_ratios,
    select_maplet_slots,
    summarize_frozen_group_log_ratios,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)


def _geometry() -> SupportObservationGeometryIndex:
    return SupportObservationGeometryIndex(
        image_ids=("support.png",),
        image_offsets=np.asarray([0, 5], dtype=np.int64),
        source_row_indices=np.arange(5, dtype=np.int64),
        track_ids=np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
        xy=np.asarray(
            [[10.0, 10.0], [5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]],
            dtype=np.float32,
        ),
        viewing_rays=np.ones((5, 3), dtype=np.float32),
        reprojection_errors=np.zeros((5,), dtype=np.float32),
    )


def _evidence() -> FrozenCandidateMapletEvidenceLayout:
    return FrozenCandidateMapletEvidenceLayout(
        verification_source_rows=np.asarray([17], dtype=np.int64),
        verification_xy=np.asarray([[50.0, 50.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[10]], dtype=np.int64),
        candidate_probabilities=np.asarray([[0.8]], dtype=np.float32),
        null_probabilities=np.asarray([0.2], dtype=np.float32),
        support_view_probabilities=np.asarray([[[1.0]]], dtype=np.float32),
        support_image_ids=np.asarray([[["support.png"]]], dtype=np.str_),
        support_view_valid=np.asarray([[[True]]]),
    )


def test_maplet_layout_keeps_center_excluded_sfm_neighbors() -> None:
    topology = np.full((5, 4, 2), -1, dtype=np.int64)
    topology[0, :, 0] = np.asarray([1, 2, 3, 4], dtype=np.int64)
    profile = FrozenMapletAppearanceProfile("p", "radio_final", "grid16_radius4", 16, 4)
    layout = build_frozen_candidate_maplet_profile_layout(
        evidence=_evidence(),
        profile=profile,
        neighbor_topology=topology,
        support_geometry=_geometry(),
        canonical_track_ids=np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
        canonical_xyz=np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    assert layout.anchor_geometry_rows.tolist() == [[[0]]]
    assert layout.neighbor_valid.sum() == 4
    assert 10 not in layout.neighbor_track_ids[layout.neighbor_valid].tolist()
    narrowed = select_maplet_slots(layout, slots_per_quadrant=1)
    assert narrowed.neighbor_valid.shape == (1, 1, 1, 4, 1)
    with pytest.raises(ValueError):
        select_maplet_slots(layout, slots_per_quadrant=3)


def test_maplet_pool_and_fixed_candidate_null_keep_missing_view_mass() -> None:
    logs = torch.log(torch.tensor([[[[[[4.0], [2.0], [1.0], [0.5]]]]]], dtype=torch.float32))
    # Layout is [batch, point, candidate, view, quadrant, slot].
    logs = logs.reshape(1, 1, 1, 1, 4, 1)
    active = torch.ones_like(logs, dtype=torch.bool)
    present = torch.ones((1, 1, 1, 4, 1), dtype=torch.bool)
    view_logs, view_usable, active_quadrants = pool_maplet_neighbor_log_ratios(
        neighbor_log_ratios=logs,
        neighbor_active=active,
        neighbor_present=present,
        minimum_active_neighbors_per_quadrant=1,
        minimum_active_quadrants=3,
        quadrant_reduction="mean",
    )
    assert active_quadrants.tolist() == [[[[4]]]]
    assert view_usable.tolist() == [[[[True]]]]
    np.testing.assert_allclose(view_logs.exp().numpy(), [[[[np.sqrt(2.0)]]]], rtol=1e-5)
    group, candidate = fixed_candidate_maplet_group_log_ratios(
        view_log_ratios=view_logs,
        view_usable=view_usable,
        support_view_probabilities=torch.tensor([[[1.0]]]),
        candidate_probabilities=torch.tensor([[0.8]]),
        null_probabilities=torch.tensor([0.2]),
        missing_view_ratio=0.25,
        max_log_ratio=12.0,
    )
    np.testing.assert_allclose(candidate.numpy(), [[[np.sqrt(2.0)]]], rtol=1e-5)
    np.testing.assert_allclose(group.exp().numpy(), [[0.2 + 0.8 * np.sqrt(2.0)]], rtol=1e-5)
    missing_group, _ = fixed_candidate_maplet_group_log_ratios(
        view_log_ratios=view_logs,
        view_usable=torch.zeros_like(view_usable),
        support_view_probabilities=torch.tensor([[[1.0]]]),
        candidate_probabilities=torch.tensor([[0.8]]),
        null_probabilities=torch.tensor([0.2]),
        missing_view_ratio=0.25,
        max_log_ratio=12.0,
    )
    # Missing view evidence cannot disappear from the candidate mixture.
    np.testing.assert_allclose(missing_group.exp().numpy(), [[0.2 + 0.8 * 0.25]], atol=1e-6)


def test_maplet_summary_and_descriptor_control_are_deterministic() -> None:
    summary = summarize_frozen_group_log_ratios(
        group_log_ratios=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        query_xy=np.asarray([[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]], dtype=np.float32),
        image_width=100,
        image_height=100,
    )
    assert float(summary["spatial_median_of_means_2x2"][0]) in {2.0, 3.0}
    mapping = deterministic_support_descriptor_derangement(
        image_ids=np.asarray(["a", "b", "c", "d"], dtype=np.str_),
        image_sizes=np.asarray([[100, 80], [100, 80], [120, 90], [120, 90]], dtype=np.int64),
    )
    assert mapping == {"a": "b", "b": "a", "c": "d", "d": "c"}
    with pytest.raises(ValueError):
        deterministic_support_descriptor_derangement(
            image_ids=np.asarray(["a", "b"], dtype=np.str_),
            image_sizes=np.asarray([[100, 80], [120, 90]], dtype=np.int64),
        )
