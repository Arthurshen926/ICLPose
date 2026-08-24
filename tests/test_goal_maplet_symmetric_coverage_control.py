from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_symmetric_coverage_control import (
    _propagated_reliability,
    _symmetric_coverage_scores,
)


def _score_case(
    *, matched_component: float, target_mass: float,
):
    # Fine has an identity spatial kernel.  Six unit weights make the existing
    # recall component matched_component / 6 and M half of that for relsum=1.
    statistics = np.zeros((1, 6), dtype=np.float64)
    statistics[0, 0] = matched_component
    target = np.zeros((1, 2304, 1), dtype=np.float64)
    target[0, 0, 0] = target_mass
    return _symmetric_coverage_scores(
        statistics,
        np.ones(6, dtype=np.float64),
        np.pad(np.asarray([[0.5]], dtype=np.float64), ((0, 2303), (0, 0))),
        np.pad(np.asarray([1.0], dtype=np.float64), (0, 2303)),
        target,
        np.asarray([True]),
        stage="fine",
    )


def test_propagated_reliability_uses_nonrenormalized_boundary_kernel() -> None:
    reliability = np.zeros(36 * 64, dtype=np.float64)
    reliability[0] = 1.0
    propagated = _propagated_reliability(reliability, stage="medium").reshape(36, 64)
    assert propagated[0, 0] == 1.0 / 25.0
    assert propagated[2, 2] == 1.0 / 25.0
    assert np.isclose(propagated.sum(), 9.0 / 25.0)


def test_symmetric_coverage_is_bounded_and_all_missing_is_minus_one() -> None:
    recall, symmetric, audit = _score_case(
        matched_component=0.0, target_mass=0.0,
    )
    assert recall[0] == -1.0
    assert symmetric[0] == -1.0
    assert audit["all_missing_target_score"] == -1.0
    assert -1.0 <= audit["minimum_f_score"] <= audit["maximum_f_score"] <= 1.0


def test_deleting_matched_evidence_or_invalidating_it_cannot_improve() -> None:
    _, full, _ = _score_case(matched_component=0.6, target_mass=0.4)
    _, deleted, _ = _score_case(matched_component=0.3, target_mass=0.4)
    # Invalid/unresolved evidence keeps its planned target residual R while M
    # vanishes; it is not treated as candidate geometry disappearing.
    _, invalid, _ = _score_case(matched_component=0.0, target_mass=0.4)
    assert deleted[0] < full[0]
    assert invalid[0] <= deleted[0]


def test_adding_true_unmatched_target_geometry_lowers_precision_score() -> None:
    _, smaller, _ = _score_case(matched_component=0.3, target_mass=0.2)
    _, larger, _ = _score_case(matched_component=0.3, target_mass=0.4)
    assert larger[0] < smaller[0]
