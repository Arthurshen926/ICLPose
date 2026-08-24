from types import SimpleNamespace

import pytest

from feature_extract.tools.vfm.calibrate_goal_maplet_validity import (
    _calibration_split_audit,
)
from feature_extract.tools.vfm.retrieve_goal_maplet_pure_radio import (
    _query_split_audit,
    _validate_promotion_contract,
)


def _artifact(*, eligible: bool) -> SimpleNamespace:
    return SimpleNamespace(metadata={"promotion_eligible": eligible})


def test_retrieval_promotion_contract_accepts_fully_promoted_chain() -> None:
    eligible, blockers = _validate_promotion_contract(
        _artifact(eligible=True),
        _artifact(eligible=True),
        allow_unpromoted_mapper_control=False,
    )
    assert eligible is True
    assert blockers == []


def test_calibration_split_requires_route_unseen_by_mapper_and_canonical() -> None:
    field = {
        "mapping_trajectory_ids": ["seq1", "seq9"],
        "excluded_trajectory_ids": ["seq10", "seq12", "seq14"],
        "route_exclusion_applied_before_opening_contributor_archives": True,
    }
    mapper = {
        "training_trajectory_ids": ["seq1"],
        "validation_trajectory_ids": ["seq9"],
        "strict_holdout_trajectory_ids": ["seq10", "seq12", "seq14"],
    }
    assert _calibration_split_audit({"seq10"}, field, mapper)["disjoint"] is True
    bad = _calibration_split_audit({"seq9"}, field, mapper)
    assert bad["disjoint"] is False
    assert "calibration_route_present_in_canonical_fusion" in bad["blockers"]
    assert "calibration_route_used_for_mapper_fit_or_selection" in bad["blockers"]


def test_query_split_requires_route_unseen_by_entire_fitting_chain() -> None:
    field = {
        "mapping_trajectory_ids": ["seq1", "seq9"],
        "excluded_trajectory_ids": ["seq10", "seq12", "seq14"],
    }
    mapper = {
        "training_trajectory_ids": ["seq1"],
        "validation_trajectory_ids": ["seq9"],
        "strict_holdout_trajectory_ids": ["seq10", "seq12", "seq14"],
    }
    calibration = {"fit_trajectory_ids": ["seq10"]}
    assert _query_split_audit(
        {"seq12", "seq14"}, field, mapper, calibration
    )["disjoint"] is True
    bad = _query_split_audit({"seq10"}, field, mapper, calibration)
    assert bad["disjoint"] is False
    assert "query_route_used_for_validity_calibration" in bad["blockers"]


@pytest.mark.parametrize(
    ("field_eligible", "calibration_eligible"),
    ((False, True), (True, False), (False, False)),
)
def test_retrieval_promotion_contract_fails_closed(
    field_eligible: bool,
    calibration_eligible: bool,
) -> None:
    with pytest.raises(ValueError, match="unpromoted"):
        _validate_promotion_contract(
            _artifact(eligible=field_eligible),
            _artifact(eligible=calibration_eligible),
            allow_unpromoted_mapper_control=False,
        )


def test_retrieval_promotion_contract_explicit_control_stays_unpromoted() -> None:
    eligible, blockers = _validate_promotion_contract(
        _artifact(eligible=False),
        _artifact(eligible=False),
        allow_unpromoted_mapper_control=True,
    )
    assert eligible is False
    assert blockers == [
        "canonical_field_not_promotion_eligible",
        "validity_calibration_not_promotion_eligible",
    ]
