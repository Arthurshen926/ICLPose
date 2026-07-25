from __future__ import annotations

import numpy as np
import torch

from feature_extract.tools.vfm.score_frozen_absolute_phase_pose_evidence import (
    _lookup_source_tensor,
    _masked_statistics,
    _score_hypotheses,
)
from feature_extract.vfm.localization.frozen_absolute_phase_probe import (
    FROZEN_ABSOLUTE_PHASE_PROFILES,
)


def test_masked_statistics_do_not_turn_missing_points_into_infinite_tail() -> None:
    values = torch.as_tensor([[2.0, -3.0, 5.0], [1.0, 2.0, 3.0]])
    active = torch.as_tensor([[False, False, False], [True, False, True]])
    masses = torch.as_tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    summary = _masked_statistics(
        values=values,
        active=active,
        point_view_masses=masses,
        query_xy=np.asarray([[10.0, 10.0], [60.0, 10.0], [60.0, 60.0]], dtype=np.float32),
        image_width=100,
        image_height=100,
    )

    for value in summary.values():
        assert bool(torch.isfinite(value).all())
    torch.testing.assert_close(summary["means"][0], torch.tensor(0.0))
    torch.testing.assert_close(summary["effective_point_counts"], torch.tensor([0, 2]))
    torch.testing.assert_close(summary["effective_view_masses"], torch.tensor([0.0, 3.0]))


def test_lookup_uses_the_same_floor_cell_rule_as_dynamic_crops() -> None:
    lookup = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1)
    projected = torch.as_tensor([[[[0.0, 0.0]], [[99.0, 99.0]]]])
    output = _lookup_source_tensor(
        lookup=lookup,
        projected_xy=projected,
        image_width=100,
        image_height=100,
        grid_size=2,
    )

    torch.testing.assert_close(output[:, 0, 0, 0], torch.tensor([0.0]))
    torch.testing.assert_close(output[:, 1, 0, 0], torch.tensor([7.0]))


class _Camera:
    model_id = 2
    params = (1.0, 10.0, 10.0, 0.0)


def test_score_hypotheses_keeps_profile_evidence_per_source_and_view() -> None:
    profile_lookups = {
        profile.name: {
            "raw": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "available": torch.ones((1, 1, 1, 1), dtype=torch.bool),
            "coverage": torch.ones((1, 1, 1, 1), dtype=torch.float32),
        }
        for profile in FROZEN_ABSOLUTE_PHASE_PROFILES
    }
    source_rows = {
        "radio_final": np.asarray([0], dtype=np.int64),
        "radio_intermediate": np.asarray([1], dtype=np.int64),
        "alike": np.asarray([2], dtype=np.int64),
    }
    statistics, sidecar = _score_hypotheses(
        profile_lookups=profile_lookups,
        source_rows=source_rows,
        poses_w2c=np.eye(4, dtype=np.float64)[None],
        candidate_xyz=np.asarray([[[0.0, 0.0, 1.0]]] * 3, dtype=np.float32),
        candidate_probabilities=np.full((3, 1), 0.9, dtype=np.float32),
        null_probabilities=np.full((3,), 0.1, dtype=np.float32),
        candidate_view_weights=np.ones((3, 1, 1), dtype=np.float32),
        query_xy=np.asarray([[2.0, 2.0], [10.0, 2.0], [2.0, 10.0]], dtype=np.float32),
        camera=_Camera(),
        image_width=20,
        image_height=20,
        hypothesis_batch_size=1,
        device=torch.device("cpu"),
    )

    assert statistics["means"].shape[0] == 1
    assert sidecar["point_log_ratios"].shape[:2] == (1, 3)
    assert bool(np.isfinite(statistics["means"]).all())
    assert bool(np.isfinite(sidecar["point_zero_shift_coverages"]).all())
    # A zero log ratio with a 0.9 candidate plus 0.1 null is exactly neutral.
    np.testing.assert_allclose(statistics["means"], 0.0, atol=1e-6)
