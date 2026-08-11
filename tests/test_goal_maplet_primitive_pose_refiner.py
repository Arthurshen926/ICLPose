import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.primitive_pose_refiner import (
    refine_pose_with_primitive_vfm_score,
)
from feature_extract.tools.vfm.refine_goal_maplet_pose_modes_with_primitive_vfm import (
    _deployed_baseline_first,
    _load_deployed_baseline_selection,
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
