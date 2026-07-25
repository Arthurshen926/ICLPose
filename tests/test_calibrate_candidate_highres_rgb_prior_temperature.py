from __future__ import annotations

import argparse

import pytest

from feature_extract.tools.vfm.calibrate_candidate_highres_rgb_prior_temperature import (
    parse_temperature_grid,
    select_temperature,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        minimum_pose_gap=0.05,
        minimum_pose_win_fraction=0.55,
        minimum_effective_candidate_count_median=2.5,
        minimum_effective_candidate_count_p10=1.1,
    )


def test_temperature_grid_is_positive_sorted_and_unique() -> None:
    assert parse_temperature_grid("1.0,0.5,0.35") == (0.35, 0.5, 1.0)
    with pytest.raises(ValueError):
        parse_temperature_grid("0.5,0.5")
    with pytest.raises(ValueError):
        parse_temperature_grid("0")


def test_train_only_temperature_selection_respects_ess_safeguards() -> None:
    metrics = {
        0.35: {
            "normal_pose_gap": 0.10,
            "normal_pose_win_fraction": 0.70,
            "normal_minus_permuted_pose_gap": 0.07,
        },
        0.25: {
            "normal_pose_gap": 0.20,
            "normal_pose_win_fraction": 0.80,
            "normal_minus_permuted_pose_gap": 0.20,
        },
    }
    ess = {
        0.35: {
            "effective_candidate_count_median": 2.9,
            "effective_candidate_count_p10": 1.2,
        },
        0.25: {
            "effective_candidate_count_median": 2.0,
            "effective_candidate_count_p10": 1.02,
        },
    }
    assert select_temperature(
        temperatures=(0.25, 0.35), train_metrics=metrics, effective_counts=ess, args=_args()
    ) == 0.35
