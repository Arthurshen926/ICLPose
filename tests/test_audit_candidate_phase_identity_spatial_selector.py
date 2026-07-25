from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_candidate_phase_identity_spatial_selector import (
    parse_selector_topk,
    parse_uniform_mixture_fractions,
    summarize_pose_gap_distribution,
)


def test_selector_topk_parser_requires_unique_positive_values() -> None:
    assert parse_selector_topk("8, 16,32") == (8, 16, 32)
    with pytest.raises(ValueError, match="unique positive"):
        parse_selector_topk("8,8")
    with pytest.raises(ValueError, match="unique positive"):
        parse_selector_topk("0")


def test_uniform_mixture_parser_requires_unique_interior_fractions() -> None:
    assert parse_uniform_mixture_fractions("0.25, 0.5") == (0.25, 0.5)
    assert parse_uniform_mixture_fractions("") == ()
    with pytest.raises(ValueError, match="unique values"):
        parse_uniform_mixture_fractions("0.5,0.5")
    with pytest.raises(ValueError, match="unique values"):
        parse_uniform_mixture_fractions("1.0")


def test_pose_gap_summary_reports_paired_tail_regressions() -> None:
    summary = summarize_pose_gap_distribution(
        normal_pose_gaps=np.asarray([0.20, -0.10, 0.30]),
        normal_minus_rgb_deranged_pose_gaps=np.asarray([0.10, 0.20, 0.30]),
        uniform_normal_pose_gaps=np.asarray([0.10, 0.10, 0.20]),
    )
    assert summary["negative_normal_pose_gap_count"] == 1
    assert summary["paired_vs_uniform_win_count"] == 2
    assert summary["paired_vs_uniform_loss_count"] == 1
    assert summary["uniform_win_to_selector_loss_count"] == 1
    assert summary["paired_vs_uniform_minimum_gap_delta"] == pytest.approx(-0.20)


def test_pose_gap_summary_rejects_misaligned_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="distribution"):
        summarize_pose_gap_distribution(
            normal_pose_gaps=np.asarray([0.1]),
            normal_minus_rgb_deranged_pose_gaps=np.asarray([0.1, 0.2]),
            uniform_normal_pose_gaps=np.asarray([0.1]),
        )
