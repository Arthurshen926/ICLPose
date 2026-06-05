from __future__ import annotations

import numpy as np

from feature_extract.vfm.superpoint_keypoint_abcd import (
    best_action_label,
    select_softmax_candidate,
    snap_improvement_summary,
    train_no_snap_softmax_selector,
    train_ridge_residual_model,
    train_snap_improvement_gate,
)


def _candidate(
    query: str,
    match_index: int,
    candidate_index: int,
    *,
    descriptor_similarity: float,
    selector_score: float,
    center_error: float,
    candidate_error: float,
    support_delta=(0.0, 0.0),
):
    return {
        "query_id": query,
        "match_index": match_index,
        "action": "snap",
        "candidate_index": candidate_index,
        "candidate_rank": candidate_index + 1,
        "selector_score": selector_score,
        "selector_margin": selector_score,
        "descriptor_similarity": descriptor_similarity,
        "query_keypoint_score": 0.5,
        "support_keypoint_score": 0.5,
        "distance_to_center_px": 2.0,
        "distance_to_center_norm": 0.25,
        "support_bias_px": 1.0,
        "support_bias_norm": 0.1,
        "match_similarity": 0.8,
        "match_ratio": 0.0,
        "similarity_margin": 0.1,
        "landmark_variance": 0.01,
        "landmark_reprojection_error": 0.2,
        "landmark_quality": 1.0,
        "landmark_ambiguity": 0.0,
        "observation_count": 5,
        "visibility_count": 5,
        "baseline_reproj_residual_px": 3.0,
        "support_delta_xy": list(support_delta),
        "candidate_xy": [10.0, 10.0],
        "gt_xy": [10.0 + candidate_error, 10.0],
        "center_error_px": center_error,
        "candidate_error_px": candidate_error,
        "snap_improvement_px": center_error - candidate_error,
    }


def test_snap_gate_learns_higher_scores_for_useful_candidates() -> None:
    rows = []
    for idx in range(12):
        rows.append(_candidate("q", idx, 0, descriptor_similarity=0.95, selector_score=1.0, center_error=14.0, candidate_error=3.0))
        rows.append(_candidate("q", idx + 100, 0, descriptor_similarity=0.10, selector_score=0.1, center_error=6.0, candidate_error=12.0))

    model, metrics = train_snap_improvement_gate(rows, iterations=250, learning_rate=0.2)
    probs = model.predict_proba([rows[0], rows[1]])

    assert metrics["auc"] > 0.95
    assert probs[0] > probs[1]


def test_no_snap_softmax_selector_can_choose_snap_or_no_snap() -> None:
    rows = [
        _candidate("q", 0, 0, descriptor_similarity=0.95, selector_score=1.0, center_error=14.0, candidate_error=3.0),
        _candidate("q", 1, 0, descriptor_similarity=0.05, selector_score=0.1, center_error=5.0, candidate_error=12.0),
    ]

    assert best_action_label([rows[0]]) == 0
    assert best_action_label([rows[1]]) == 1

    model = train_no_snap_softmax_selector(rows, iterations=120, learning_rate=0.2, seed=0)
    selected_good = select_softmax_candidate([rows[0]], model)
    selected_bad = select_softmax_candidate([rows[1]], model)

    assert selected_good is not None
    assert selected_good["candidate_index"] == 0
    assert selected_bad is None


def test_ridge_residual_model_predicts_candidate_to_gt_residual() -> None:
    rows = []
    for idx in range(8):
        rows.append(
            _candidate(
                "q",
                idx,
                0,
                descriptor_similarity=0.9,
                selector_score=1.0,
                center_error=12.0,
                candidate_error=2.0,
                support_delta=(2.0, -1.0),
            )
        )
        rows[-1]["gt_xy"] = [12.0, 9.0]
        rows[-1]["candidate_xy"] = [10.0, 10.0]

    model = train_ridge_residual_model(rows, positive_px=4.0, l2=1e-4)
    pred = model.predict([rows[0]])[0]

    np.testing.assert_allclose(pred, [2.0, -1.0], atol=1e-3)


def test_snap_improvement_summary_uses_heuristic_candidate() -> None:
    rows = [
        {**_candidate("q", 0, 0, descriptor_similarity=0.9, selector_score=1.0, center_error=10.0, candidate_error=3.0), "heuristic_applied": True},
        {**_candidate("q", 1, 0, descriptor_similarity=0.2, selector_score=0.2, center_error=5.0, candidate_error=9.0), "heuristic_applied": True},
    ]

    summary = snap_improvement_summary(rows)

    assert summary["selected_count"] == 2.0
    assert summary["snap_improve_ratio"] == 0.5
    assert summary["snap_worsen_ratio"] == 0.5
