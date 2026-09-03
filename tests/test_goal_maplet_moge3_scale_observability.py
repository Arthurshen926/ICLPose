from feature_extract.tools.vfm.audit_goal_maplet_moge3_scale_observability import _summary


def test_scale_observability_summary_is_fail_closed_and_acceptance_aware():
    rows = [
        {"conditional_scale_data_to_prior_information_ratio": 2.0, "refinement_accepted": True},
        {"conditional_scale_data_to_prior_information_ratio": 0.5, "refinement_accepted": True},
        {"refinement_accepted": False},
    ]
    result = _summary(rows)
    assert result["diagnosed_count"] == 2
    assert result["data_at_least_prior_observable_count"] == 1
    assert result["accepted_and_data_at_least_prior_observable_count"] == 1
