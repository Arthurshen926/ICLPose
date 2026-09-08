from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import (
    _failure_category,
)


def _category(**updates):
    values = dict(
        selected_hit=False,
        oracle_row_count=20,
        oracle_plane_count=3,
        oracle_exact_hit=True,
        primary_pnp_hit=False,
        alternate_pnp_hit=False,
        primary_final_hit=False,
        alternate_final_hit=False,
    )
    values.update(updates)
    return _failure_category(**values)


def test_failure_categories_follow_causal_stage_order():
    assert _category(selected_hit=True) == "selected_pose_coarse_success"
    assert _category(oracle_row_count=5) == "retrieved_chart_or_uv_support_insufficient"
    assert _category(oracle_plane_count=1) == "retrieved_candidate_geometry_degenerate"
    assert _category(oracle_exact_hit=False) == "retrieved_candidate_geometry_degenerate"
    assert _category() == "hard_coordinate_or_pnp_initialization_failure"
    assert _category(primary_pnp_hit=True) == (
        "pnp_candidate_collapse_or_post_pnp_refinement_regression"
    )
    assert _category(primary_final_hit=True) == "final_v5_v11_selector_failure"


def test_selector_category_requires_a_valid_final_arm():
    assert _category(alternate_pnp_hit=True, alternate_final_hit=True) == (
        "final_v5_v11_selector_failure"
    )
