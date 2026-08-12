import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_adaptive_refinement_oof import (
    _configuration_key,
    _nested_route_crossfit_selection,
    _policy_subset,
    _replay_row,
)


def _pose(x):
    pose = np.eye(4)
    pose[0, 3] = -x
    return pose.tolist()


def _candidate(index, rank, score, anchor, x):
    return {
        "union_candidate_index": index,
        "common_initial_rank": rank,
        "source_report_index": 0,
        "source_mode_rank": index + 1,
        "mapping_view_anchor_label": anchor,
        "initial_score": score,
        "initial_pose_w2c": _pose(x),
        "pose_w2c": _pose(x),
        "final_score": score,
        "validation_score": score,
        "final_translation_m": x,
        "final_rotation_deg": 0.0,
        "rendered_coverage": 0.2 + 0.1 * index,
        "feature_coverage": 0.1 + 0.1 * index,
    }


def test_anchor_basin_policy_preserves_structure_and_real_baseline():
    row = {"candidate_refinements": [
        _candidate(0, 4, 0.6, 0, 0.0),
        _candidate(1, 1, 0.9, 1, 2.0),
        _candidate(2, 2, 0.8, 1, 2.1),
        _candidate(3, 3, 0.7, 2, 4.0),
    ]}
    selected, diagnostic = _policy_subset(
        row, policy="anchor_basin_cover", budget=2, score_margin=0.01
    )
    assert {value["union_candidate_index"] for value in selected} == {0, 1, 3}
    assert diagnostic["baseline_guard_added"]
    assert diagnostic["selected_pose_basin_count"] == 2


def test_policy_selection_is_catastrophe_first_then_recall_then_compute():
    safe = {"summary": {
        "catastrophic_count": 0, "strict_count": 1, "loose_count": 1,
        "translation_p90_m": 1.0, "rotation_p90_deg": 1.0,
        "mean_refined_candidate_count": 32.0,
    }}
    risky = {"summary": {
        "catastrophic_count": 1, "strict_count": 100, "loose_count": 100,
        "translation_p90_m": 0.1, "rotation_p90_deg": 0.1,
        "mean_refined_candidate_count": 1.0,
    }}
    assert _configuration_key(safe) < _configuration_key(risky)


def test_adaptive_replay_moves_selected_pose_coverage_with_the_winner():
    row = {
        "image_id": "seq0/a.png",
        "candidate_refinements": [
            _candidate(0, 2, 0.5, 0, 0.0),
            _candidate(1, 1, 0.9, 1, 1.0),
        ],
        "postselection_source_evidence": {
            "selected_exact_rendered_coverage": 0.2,
            "selected_exact_feature_coverage": 0.1,
        },
    }
    replay = _replay_row(
        row, policy="score_topk", budget=1, score_margin=-1.0,
        validation_enabled=True, require_cross_splat_winner_consistency=True,
    )
    assert replay["selected_union_candidate_index"] == 1
    assert np.isclose(
        replay["postselection_source_evidence"][
            "selected_exact_rendered_coverage"
        ],
        0.3,
    )


def _outcome(image_id, error):
    return {
        "image_id": image_id,
        "final_translation_m": error,
        "final_rotation_deg": error * 10.0,
        "refined_mode_count": 1,
    }


def test_nested_policy_selection_never_uses_held_route_outcomes():
    configurations = [
        {"configuration_id": "policy_a", "policy": "score_topk"},
        {"configuration_id": "policy_b", "policy": "anchor_basin_cover"},
    ]
    replay_cache = {
        "policy_a": [_outcome("seq0/a.png", 0.1), _outcome("seq1/b.png", 3.0)],
        "policy_b": [_outcome("seq0/a.png", 3.0), _outcome("seq1/b.png", 0.1)],
    }
    rows, folds = _nested_route_crossfit_selection(
        configurations=configurations,
        replay_cache=replay_cache,
        folds=[
            {"fold_id": "fold0", "held_query_trajectories": ["seq0"]},
            {"fold_id": "fold1", "held_query_trajectories": ["seq1"]},
        ],
        image_ids=["seq0/a.png", "seq1/b.png"],
    )
    # Holding seq0 leaves only seq1, which selects policy_b; vice versa selects A.
    assert folds[0]["selected_configuration"]["configuration_id"] == "policy_b"
    assert folds[1]["selected_configuration"]["configuration_id"] == "policy_a"
    assert rows[0]["adaptive_policy_crossfit"]["configuration_id"] == "policy_b"
    assert rows[1]["adaptive_policy_crossfit"]["configuration_id"] == "policy_a"
    assert all(
        row["adaptive_policy_crossfit"][
            "policy_selected_without_this_query_or_trajectory"
        ]
        for row in rows
    )
