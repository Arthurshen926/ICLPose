from feature_extract.tools.vfm.evaluate_goal_maplet_refinement_k_curve import (
    _prefix_selection,
    _wilson,
)


def _candidate(index, rank, score, validation, error):
    return {
        "union_candidate_index": index,
        "common_initial_rank": rank,
        "final_score": score,
        "validation_score": validation,
        "final_translation_m": error,
        "final_rotation_deg": error,
    }


def test_prefix_selection_always_retains_real_deployed_baseline():
    row = {
        "baseline_null_threshold": 0.0,
        "candidate_refinements": [
            _candidate(5, 1, 0.9, 0.9, 9.0),
            _candidate(0, 7, 0.8, 0.8, 0.1),
        ],
    }
    selected, decision, count = _prefix_selection(
        row,
        refine_k=1,
        validation_enabled=True,
        require_cross_splat_winner_consistency=True,
    )
    assert selected["union_candidate_index"] == 0
    assert decision == "baseline_explained_query_control_limit"
    assert count == 2


def test_cross_splat_disagreement_falls_back_to_union_candidate_zero():
    row = {
        "baseline_null_threshold": 1.0,
        "candidate_refinements": [
            _candidate(3, 1, 0.9, 0.5, 9.0),
            _candidate(4, 2, 0.7, 0.95, 8.0),
            _candidate(0, 8, 0.4, 0.4, 0.1),
        ],
    }
    selected, decision, _ = _prefix_selection(
        row,
        refine_k=2,
        validation_enabled=True,
        require_cross_splat_winner_consistency=True,
    )
    assert selected["union_candidate_index"] == 0
    assert decision == "cross_discretization_disagreement_baseline_fallback"


def test_wilson_interval_contains_observed_rate():
    low, high = _wilson(7, 17)
    assert low < 7 / 17 < high
