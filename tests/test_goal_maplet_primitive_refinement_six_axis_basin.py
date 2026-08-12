import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_primitive_refinement_six_axis_basin import (
    _score_history_is_monotonic,
    _trial_definitions,
)


def test_six_axis_trials_cover_both_signs_without_mixed_axes():
    trials = _trial_definitions([0.1, 0.5], [2.0], include_zero=True)
    assert len(trials) == 1 + 3 * 2 * 2 + 3 * 1 * 2
    assert {trial["axis"] for trial in trials} == {
        "zero", "rx", "ry", "rz", "tx", "ty", "tz"
    }
    for trial in trials[1:]:
        assert np.count_nonzero(trial["delta"]) == 1


def test_exact_score_monotonicity_audit_detects_regression():
    assert _score_history_is_monotonic(
        0.1, [{"score": 0.1}, {"score": 0.2}], 0.2
    )
    assert not _score_history_is_monotonic(
        0.1, [{"score": 0.2}, {"score": 0.19}], 0.19
    )
