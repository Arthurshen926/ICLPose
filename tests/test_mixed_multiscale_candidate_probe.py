import numpy as np

from feature_extract.vfm.localization.mixed_multiscale_candidate_probe import (
    MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT,
    candidate_probe_gate,
    geometric_membership_from_residuals,
)
from feature_extract.tools.vfm.fit_mixed_multiscale_candidate_probe import _resolve_families


def test_geometric_membership_retains_all_valid_positives_and_null() -> None:
    membership = geometric_membership_from_residuals(
        residuals=np.asarray([[1.0, 4.0, np.inf], [8.0, np.inf, np.inf]], dtype=np.float32),
        candidate_valid=np.asarray([[True, True, False], [True, False, False]]),
        threshold_px=5.0,
    )
    assert membership.shape == (2, 4)
    assert membership[0].tolist() == [True, True, False, False]
    assert membership[1].tolist() == [False, False, False, True]


def test_candidate_gate_requires_all_four_conditions() -> None:
    baseline = {"group_target_nll": 1.0, "top1_geometry_valid_rate": 0.4}
    probe = {"group_target_nll": 0.8, "top1_geometry_valid_rate": 0.5}
    passed = candidate_probe_gate(
        baseline,
        probe,
        {
            "rank_win_count": 10,
            "rank_loss_count": 2,
            "top1_rescue_count": 5,
            "top1_harm_count": 1,
        },
    )
    assert passed["passed"] is True
    failed = candidate_probe_gate(
        baseline,
        probe,
        {
            "rank_win_count": 10,
            "rank_loss_count": 2,
            "top1_rescue_count": 1,
            "top1_harm_count": 5,
        },
    )
    assert failed["passed"] is False


def test_mixed_family_selection_is_bound_to_feature_artifact_format() -> None:
    local_format, absolute_format = tuple(MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT)
    absolute_family = MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT[absolute_format][0]
    assert _resolve_families("all", feature_format=absolute_format) == (
        MIXED_PROBE_FAMILIES_BY_FEATURE_FORMAT[absolute_format]
    )
    try:
        _resolve_families(absolute_family, feature_format=local_format)
    except ValueError as error:
        assert "not declared" in str(error)
    else:
        raise AssertionError("absolute transport family was accepted for local-region features")
