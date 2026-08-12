import numpy as np
import pytest

from feature_extract.tools.vfm.evaluate_goal_maplet_primitive_refinement_gate import (
    _refinement_uses_candidate_report,
)
from feature_extract.vfm.localization_goal_maplet.primitive_pose_refiner import (
    refine_pose_with_primitive_vfm_score,
)
from feature_extract.tools.vfm.refine_goal_maplet_pose_modes_with_primitive_vfm import (
    _deployed_baseline_first,
    _baseline_refinement_index,
    _load_deployed_baseline_selection,
    _select_refinement_candidate_indices,
)


def test_refinement_gate_accepts_single_and_multi_source_lineage():
    assert _refinement_uses_candidate_report(
        {"candidate_report_sha256": "abc"}, "abc"
    )
    assert _refinement_uses_candidate_report(
        {"candidate_report_sha256": ["abc", "def"]}, "def"
    )
    assert not _refinement_uses_candidate_report(
        {"candidate_report_sha256": ["abc", "def"]}, "xyz"
    )


def test_trust_region_improves_pose_without_regressing_objective():
    initial = np.eye(4, dtype=np.float64)
    initial[0, 3] = 0.55

    def score(poses):
        poses = np.asarray(poses)
        return -np.square(poses[:, 0, 3] - 0.05)

    result = refine_pose_with_primitive_vfm_score(
        initial,
        score,
        translation_steps_m=(0.25, 0.10),
        rotation_steps_deg=(2.0, 1.0),
        iterations_per_scale=2,
    )
    assert result.final_score > result.initial_score
    assert abs(result.pose_w2c[0, 3] - 0.05) < abs(initial[0, 3] - 0.05)
    assert result.accepted_steps > 0
    assert all(
        later["score"] >= earlier["score"]
        for earlier, later in zip(result.history, result.history[1:])
    )


def test_trust_region_keeps_pose_on_flat_score():
    initial = np.eye(4, dtype=np.float64)
    result = refine_pose_with_primitive_vfm_score(
        initial,
        lambda poses: np.ones((len(poses),), dtype=np.float64),
        translation_steps_m=(0.1,),
        rotation_steps_deg=(1.0,),
    )
    np.testing.assert_allclose(result.pose_w2c, initial)
    assert result.accepted_steps == 0


def test_trust_region_rejects_mismatched_schedule():
    with pytest.raises(ValueError, match="schedules differ"):
        refine_pose_with_primitive_vfm_score(
            np.eye(4), lambda poses: np.zeros((len(poses),)),
            translation_steps_m=(0.1, 0.05), rotation_steps_deg=(1.0,),
        )


def test_deployed_baseline_selection_inverts_rank_permutation_without_error_labels():
    selection = {
        "transfer_evaluation": [{
            "report_sha256": "abc",
            "predictions": [{
                "image_id": "seq12/frame00097.png",
                "selected_index": 7,
                # Evaluation-only values must not participate in resolution.
                "translation_m": 999.0,
                "rotation_deg": 999.0,
            }],
        }],
    }
    selected = _load_deployed_baseline_selection(selection, "abc")
    assert selected == {"seq12/frame00097.png": 7}
    source = {
        "mode_details": {"mode": [{"id": "rank1"}, {"id": "deployed"}]},
        "ranking_diagnostics": {
            "mode": {"surface_alignment_original_indices": [9, 7]},
        },
    }
    reordered = _deployed_baseline_first(source, "mode", selected["seq12/frame00097.png"])
    assert [value["id"] for value in reordered] == ["deployed", "rank1"]


def test_deployed_baseline_selection_rejects_report_lineage_mismatch():
    with pytest.raises(ValueError, match="exactly one transfer block"):
        _load_deployed_baseline_selection(
            {"transfer_evaluation": [{"report_sha256": "wrong", "predictions": []}]},
            "expected",
        )


def _pose(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def test_adaptive_refinement_reserves_distinct_structural_anchors():
    scores = np.asarray([1.0, 0.99, 0.98, 0.80])
    poses = np.stack([_pose(0.0), _pose(0.01), _pose(1.0), _pose(2.0)])
    details = [
        (0, 1, {"mapping_view_anchor_label": 1}),
        (0, 2, {"mapping_view_anchor_label": 1}),
        (0, 3, {"mapping_view_anchor_label": 2}),
        (0, 4, {"mapping_view_anchor_label": 3}),
    ]
    selected, diagnostic = _select_refinement_candidate_indices(
        common_order=np.arange(4),
        common_initial_scores=scores,
        candidate_poses=poses,
        candidate_details=details,
        maximum_count=3,
        policy="anchor_basin_cover",
        score_margin=0.05,
        minimum_count=1,
        translation_radius_m=0.5,
        rotation_radius_deg=5.0,
    )
    assert selected == [0, 2, 3]
    assert diagnostic["selected_anchor_count"] == 3
    assert diagnostic["selected_pose_basin_count"] == 3


def test_score_margin_makes_topk_workload_query_adaptive():
    scores = np.asarray([1.0, 0.99, 0.50])
    poses = np.stack([_pose(0.0), _pose(1.0), _pose(2.0)])
    details = [(0, index + 1, {}) for index in range(3)]
    selected, diagnostic = _select_refinement_candidate_indices(
        common_order=np.arange(3),
        common_initial_scores=scores,
        candidate_poses=poses,
        candidate_details=details,
        maximum_count=3,
        policy="score_topk",
        score_margin=0.02,
        minimum_count=1,
        translation_radius_m=0.5,
        rotation_radius_deg=5.0,
    )
    assert selected == [0, 1]
    assert diagnostic["selected_count_before_baseline_guard"] == 2


def test_baseline_guard_resolves_union_candidate_zero_not_report_zero():
    refinements = [
        {"union_candidate_index": 7, "source_report_index": 0},
        {"union_candidate_index": 0, "source_report_index": 0},
    ]
    assert _baseline_refinement_index(refinements) == 1
