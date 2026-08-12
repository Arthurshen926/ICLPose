from feature_extract.tools.vfm.evaluate_goal_maplet_failure_taxonomy import (
    _classify,
)


def _candidate(*, raw: int, exact: int, retained: int, top1=(3.0, 30.0)):
    stage = lambda count: {
        "within_0_5m_5deg_count": count,
        "within_1m_10deg_count": count,
    }
    return {
        "mode_details": {
            "actual_parent_actual_child": [{
                "translation_m": top1[0], "rotation_deg": top1[1]
            }]
        },
        "proposal_diagnostics": {
            "actual_parent_actual_child": {
                "retained_seed_heap": stage(raw),
                "structural_equivalence_prescreen": stage(raw),
                "full_map_sparse_seed_vfm_likelihood": stage(raw),
                "exact_verification_pool_selection": stage(exact),
                "full_surface_ranked": stage(exact),
                "topn_after_nms": stage(retained),
            }
        },
    }


def _class(candidate, final=(3.0, 30.0)):
    return _classify(
        candidate,
        {"final_translation_m": final[0], "final_rotation_deg": final[1]},
        translation_m=0.5,
        rotation_deg=5.0,
        count_field="within_0_5m_5deg_count",
    )[0]


def test_failure_taxonomy_separates_pipeline_failure_stages():
    assert _class(_candidate(raw=0, exact=0, retained=0)) == "G_generator_absence"
    assert _class(_candidate(raw=1, exact=0, retained=0)) == (
        "S_sparse_screen_or_exact_pool_pruning"
    )
    assert _class(_candidate(raw=1, exact=1, retained=0)) == (
        "V_exact_verifier_ranking_or_nms"
    )
    assert _class(_candidate(raw=1, exact=1, retained=1)) == (
        "R_refinement_or_final_selection_failure"
    )


def test_failure_taxonomy_does_not_call_prescreen_loss_generator_absence():
    candidate = _candidate(raw=1, exact=0, retained=0)
    diagnostics = candidate["proposal_diagnostics"]["actual_parent_actual_child"]
    diagnostics["structural_equivalence_prescreen"]["within_0_5m_5deg_count"] = 0
    assert _class(candidate) == "S_structural_prescreen_pruning"


def test_failure_taxonomy_detects_gate_regression_and_success():
    candidate = _candidate(raw=1, exact=1, retained=1, top1=(0.2, 2.0))
    assert _class(candidate) == "R_refinement_or_gate_regression"
    assert _class(candidate, final=(0.1, 1.0)) == "success"
