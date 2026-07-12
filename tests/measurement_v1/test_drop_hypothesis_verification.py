from __future__ import annotations

from feature_extract.tools.vfm.eval_selected_drop_hypothesis_verification import (
    should_choose_drop_hypothesis,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    HypothesisVerification,
)


def _verification(strict: int, loose: int) -> HypothesisVerification:
    return HypothesisVerification(
        verification_count=20,
        finite_count=20,
        positive_depth_count=20,
        positive_depth_ratio=1.0,
        strict_inlier_count=strict,
        loose_inlier_count=loose,
        strict_grid_cell_count=4,
        loose_grid_cell_count=6,
        soft_consensus=float(strict),
        clipped_median_residual_px=1.0,
        depth_range_m=5.0,
    )


def test_drop_hypothesis_requires_strict_gain_and_better_rank() -> None:
    baseline = _verification(8, 12)
    drop = _verification(10, 13)

    assert should_choose_drop_hypothesis(
        drop_solver_success=True,
        baseline_verification=baseline,
        drop_verification=drop,
        minimum_strict_inlier_gain=2,
    )
    assert not should_choose_drop_hypothesis(
        drop_solver_success=True,
        baseline_verification=baseline,
        drop_verification=drop,
        minimum_strict_inlier_gain=3,
    )
    assert not should_choose_drop_hypothesis(
        drop_solver_success=False,
        baseline_verification=baseline,
        drop_verification=drop,
        minimum_strict_inlier_gain=0,
    )
