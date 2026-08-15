import pytest

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _fixed_evidence_is_usable,
    _surface_pose_gate_passes,
)
from feature_extract.vfm.localization.surface_localization import SurfacePoseConfig


def test_pose_gate_requires_absolute_mass_in_addition_to_conditional_confidence():
    diagnostics = {
        "conditionally_confident_group_count": 16,
        # Conditional identity can be high even when absolute evidence is tiny.
        "mean_null_probability": 0.99,
    }
    passed, reason = _surface_pose_gate_passes(
        diagnostics,
        minimum_confident_groups=12,
        maximum_mean_null_probability=0.85,
    )
    assert not passed
    assert reason == "absolute_null_mass_gate"


def test_pose_gate_accepts_both_required_evidence_conditions():
    diagnostics = {
        "conditionally_confident_group_count": 12,
        "mean_null_probability": 0.84,
    }
    assert _surface_pose_gate_passes(
        diagnostics,
        minimum_confident_groups=12,
        maximum_mean_null_probability=0.85,
    ) == (True, "passed")


def test_pose_gate_reports_missing_conditional_groups_first():
    diagnostics = {
        "conditionally_confident_group_count": 11,
        "mean_null_probability": 0.1,
    }
    assert _surface_pose_gate_passes(
        diagnostics,
        minimum_confident_groups=12,
        maximum_mean_null_probability=0.85,
    ) == (False, "insufficient_conditionally_confident_groups")


def test_pose_gate_rejects_nonfinite_absolute_mass():
    diagnostics = {
        "conditionally_confident_group_count": 16,
        "mean_null_probability": float("nan"),
    }
    assert _surface_pose_gate_passes(
        diagnostics,
        minimum_confident_groups=12,
        maximum_mean_null_probability=0.85,
    ) == (False, "nonfinite_mean_null_probability")


def test_pose_gate_rejects_invalid_limits():
    with pytest.raises(ValueError):
        _surface_pose_gate_passes(
            {}, minimum_confident_groups=0, maximum_mean_null_probability=0.85
        )
    with pytest.raises(ValueError):
        _surface_pose_gate_passes(
            {}, minimum_confident_groups=12, maximum_mean_null_probability=1.1
        )


def test_fixed_map_null_only_evidence_cannot_promote_a_generated_pose():
    config = SurfacePoseConfig(minimum_selected_inliers=8)
    assert not _fixed_evidence_is_usable(float("-inf"), 100, config)
    assert not _fixed_evidence_is_usable(-0.1, 7, config)
    assert _fixed_evidence_is_usable(-0.1, 8, config)
