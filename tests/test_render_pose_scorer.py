from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence
from feature_extract.vfm.render_pose_scorer import (
    PairwisePoseRanker,
    fit_pairwise_pose_ranker,
    label_pose_row,
    pose_candidate_row_from_eval_candidate,
    pose_candidate_selection_report,
    score_pose_candidates_with_model,
    vectorize_pose_candidate_rows,
    vectorize_pose_rows,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch
from feature_extract.vfm.rendered_pose_scoring import PoseHypothesisScore
from feature_extract.tools.vfm.train_render_pose_scorer import main as train_pose_scorer_main


def test_pose_scorer_labels_rows_by_localization_success_thresholds() -> None:
    positive = {"translation_error_m": 0.08, "rotation_error_deg": 2.0}
    negative = {"translation_error_m": 0.30, "rotation_error_deg": 2.0}

    assert label_pose_row(positive, translation_threshold_m=0.10, rotation_threshold_deg=5.0) == 1
    assert label_pose_row(negative, translation_threshold_m=0.10, rotation_threshold_deg=5.0) == 0


def test_pose_scorer_vectorizes_rows_without_gt_pose_error_features() -> None:
    rows = [
        {
            "pnp_success": True,
            "pnp_inlier_count": 40,
            "pnp_inlier_ratio": 0.8,
            "pose_candidate_match_count": 80,
            "pose_update_selected_score": 0.7,
            "pose_update_selected_alignment_score": 0.6,
            "pnp_match_confidence_mean": 0.9,
            "translation_error_m": 0.04,
            "rotation_error_deg": 1.0,
        },
        {
            "pnp_success": False,
            "pnp_inlier_count": 0,
            "pnp_inlier_ratio": 0.0,
            "pose_candidate_match_count": 12,
            "pose_update_selected_score": -1.0,
            "pose_update_selected_alignment_score": 0.1,
            "pnp_match_confidence_mean": 0.2,
            "translation_error_m": 1.0,
            "rotation_error_deg": 20.0,
        },
    ]

    features, labels, names = vectorize_pose_rows(rows)

    assert features.shape[0] == 2
    assert labels.tolist() == [1, 0]
    assert "translation_error_m" not in names
    assert "rotation_error_deg" not in names
    assert "pnp_inlier_count" in names


def test_pose_scorer_uses_match_validity_aggregates_without_gt_validity_labels() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([10.0, 10.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.0,
            landmark_variance=0.1,
            pnp_soft_score=0.9,
        ),
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([20.0, 10.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([1.0, 0.0, 4.0], dtype=np.float64),
            similarity=0.2,
            ratio=0.0,
            landmark_variance=0.1,
            pnp_soft_score=0.1,
        ),
    ]
    candidate = {
        "pnp": SimpleNamespace(success=True, inlier_count=1, inlier_ratio=0.5, inlier_mask=np.asarray([True, False])),
        "pose_pnp_matches": matches,
        "unfiltered_pnp_match_count": 2,
        "iteration_pose_score": PoseHypothesisScore(0.8, 1, 2.0, 0.5, 0.5, 0.0),
    }

    row = pose_candidate_row_from_eval_candidate(candidate)
    features, names = vectorize_pose_candidate_rows([row])

    assert row["match_validity_probability_mean"] == pytest.approx(0.5)
    assert row["match_validity_inlier_outlier_gap"] == pytest.approx(0.8)
    assert "match_validity_probability_mean" in names
    assert "match_validity_inlier_outlier_gap" in names
    assert "match_validity_rate_5px" not in names
    assert features.shape[1] == len(names)


def test_pose_scorer_scores_eval_candidate_dicts_with_trained_model() -> None:
    train_rows = [
        {"pnp_success": True, "pnp_inlier_count": 50, "pnp_inlier_ratio": 0.9, "pose_candidate_match_count": 90, "translation_error_m": 0.04, "rotation_error_deg": 1.0},
        {"pnp_success": True, "pnp_inlier_count": 45, "pnp_inlier_ratio": 0.8, "pose_candidate_match_count": 75, "translation_error_m": 0.06, "rotation_error_deg": 2.0},
        {"pnp_success": False, "pnp_inlier_count": 2, "pnp_inlier_ratio": 0.05, "pose_candidate_match_count": 30, "translation_error_m": 1.0, "rotation_error_deg": 12.0},
        {"pnp_success": False, "pnp_inlier_count": 4, "pnp_inlier_ratio": 0.1, "pose_candidate_match_count": 20, "translation_error_m": 0.7, "rotation_error_deg": 15.0},
    ]
    features, labels, names = vectorize_pose_rows(train_rows)
    model = CalibratedLogisticConfidence(max_iter=30).fit(features, labels)
    high = {
        "pnp": SimpleNamespace(success=True, inlier_count=60, inlier_ratio=0.9),
        "pose_pnp_matches": [object()] * 90,
        "iteration_pose_score": PoseHypothesisScore(0.8, 60, 1.2, 0.9, 0.5, 0.0),
        "iteration_alignment_score": 0.7,
        "initial_render_index": 0,
    }
    low = {
        "pnp": SimpleNamespace(success=False, inlier_count=1, inlier_ratio=0.02),
        "pose_pnp_matches": [object()] * 20,
        "iteration_pose_score": PoseHypothesisScore(-2.0, 1, 10.0, 0.1, 0.1, 0.8),
        "iteration_alignment_score": 0.0,
        "initial_render_index": 4,
    }

    scored = score_pose_candidates_with_model([low, high], model, feature_names=names)

    assert scored[0]["learned_pose_probability"] > scored[1]["learned_pose_probability"]
    assert scored[0] is high
    assert scored[0]["iteration_pose_score"].learned_probability == pytest.approx(
        scored[0]["learned_pose_probability"]
    )


def test_pose_scorer_selection_report_evaluates_query_level_candidate_choice() -> None:
    rows = [
        {"query_id": "q0", "candidate_rank": 0, "pose_update_selected_score": 0.9, "translation_error_m": 0.6, "rotation_error_deg": 2.0},
        {"query_id": "q0", "candidate_rank": 1, "pose_update_selected_score": 0.1, "translation_error_m": 0.03, "rotation_error_deg": 1.0},
        {"query_id": "q1", "candidate_rank": 0, "pose_update_selected_score": 0.8, "translation_error_m": 0.7, "rotation_error_deg": 8.0},
        {"query_id": "q1", "candidate_rank": 1, "pose_update_selected_score": 0.2, "translation_error_m": 0.08, "rotation_error_deg": 2.0},
    ]

    report = pose_candidate_selection_report(rows, learned_scores=[0.1, 0.9, 0.1, 0.8])

    assert report["query_count"] == 2
    assert report["oracle_success_rate"] == pytest.approx(1.0)
    assert report["rank0_success_rate"] == pytest.approx(0.0)
    assert report["pose_score_success_rate"] == pytest.approx(0.0)
    assert report["learned_success_rate"] == pytest.approx(1.0)
    assert report["learned_median_translation_error_m"] == pytest.approx(0.055)


def test_pairwise_pose_ranker_optimizes_query_level_candidate_ordering() -> None:
    rows = [
        {"query_id": "q0", "candidate_rank": 0, "pnp_success": True, "pnp_inlier_count": 1, "pose_candidate_match_count": 20, "translation_error_m": 0.8, "rotation_error_deg": 3.0},
        {"query_id": "q0", "candidate_rank": 1, "pnp_success": True, "pnp_inlier_count": 50, "pose_candidate_match_count": 90, "translation_error_m": 0.03, "rotation_error_deg": 1.0},
        {"query_id": "q1", "candidate_rank": 0, "pnp_success": False, "pnp_inlier_count": 0, "pose_candidate_match_count": 10, "translation_error_m": 0.7, "rotation_error_deg": 8.0},
        {"query_id": "q1", "candidate_rank": 1, "pnp_success": True, "pnp_inlier_count": 45, "pose_candidate_match_count": 70, "translation_error_m": 0.06, "rotation_error_deg": 2.0},
    ]

    model, summary = fit_pairwise_pose_ranker(rows, max_iter=120, learning_rate=0.1)
    scores = model.predict_scores_from_rows(rows)
    report = pose_candidate_selection_report(rows, learned_scores=scores)

    assert isinstance(model, PairwisePoseRanker)
    assert summary["pair_count"] == 2
    assert report["learned_success_rate"] == pytest.approx(1.0)


def test_train_render_pose_scorer_cli_writes_model_and_calibration_summary(tmp_path) -> None:
    rows_csv = tmp_path / "rows.csv"
    rows_csv.write_text(
        "\n".join(
            [
                "query_id,pnp_success,pnp_inlier_count,pnp_inlier_ratio,pose_candidate_match_count,pose_update_selected_score,translation_error_m,rotation_error_deg",
                "q0,True,40,0.8,80,0.7,0.04,1.0",
                "q1,True,35,0.7,70,0.6,0.05,2.0",
                "q2,False,2,0.1,20,-1.0,1.0,20.0",
                "q3,False,1,0.1,15,-1.2,0.8,15.0",
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "pose_scorer"

    train_pose_scorer_main(["--rows_csv", str(rows_csv), "--output_dir", str(output_dir), "--eval_on_train"])

    assert (output_dir / "pose_scorer_model.json").exists()
    summary = (output_dir / "pose_scorer_summary.json").read_text()
    assert '"brier"' in summary
    assert '"ece"' in summary
    assert '"eval_selection"' in summary


def test_train_render_pose_scorer_cli_supports_pairwise_ranking_objective(tmp_path) -> None:
    rows_csv = tmp_path / "rows.csv"
    rows_csv.write_text(
        "\n".join(
            [
                "query_id,candidate_rank,pnp_success,pnp_inlier_count,pnp_inlier_ratio,pose_candidate_match_count,pose_update_selected_score,translation_error_m,rotation_error_deg",
                "q0,0,True,1,0.1,20,0.9,0.8,3.0",
                "q0,1,True,50,0.9,90,0.1,0.03,1.0",
                "q1,0,False,0,0.0,10,0.8,0.7,8.0",
                "q1,1,True,45,0.8,70,0.2,0.06,2.0",
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "pose_ranker"

    train_pose_scorer_main(
        [
            "--rows_csv",
            str(rows_csv),
            "--output_dir",
            str(output_dir),
            "--eval_on_train",
            "--objective",
            "pairwise_rank",
        ]
    )

    model_json = (output_dir / "pose_scorer_model.json").read_text()
    summary = (output_dir / "pose_scorer_summary.json").read_text()
    assert '"model_type": "pairwise_pose_ranker"' in model_json
    assert '"objective": "pairwise_rank"' in summary
    assert '"pair_count"' in summary


def test_train_render_pose_scorer_cli_supports_query_kfold_validation_split(tmp_path) -> None:
    rows_csv = tmp_path / "rows.csv"
    rows_csv.write_text(
        "\n".join(
            [
                "query_id,candidate_rank,pnp_success,pnp_inlier_count,pnp_inlier_ratio,pose_candidate_match_count,pose_update_selected_score,translation_error_m,rotation_error_deg,match_validity_probability_mean",
                "q0,0,True,50,0.9,90,0.8,0.03,1.0,0.9",
                "q0,1,False,2,0.1,20,0.1,0.8,12.0,0.1",
                "q1,0,True,45,0.8,80,0.7,0.04,1.0,0.8",
                "q1,1,False,2,0.1,20,0.1,0.9,15.0,0.2",
                "q2,0,True,40,0.8,70,0.6,0.05,2.0,0.8",
                "q2,1,False,1,0.0,10,0.1,1.0,20.0,0.1",
                "q3,0,True,42,0.8,72,0.6,0.06,2.0,0.8",
                "q3,1,False,1,0.0,10,0.1,1.1,20.0,0.1",
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "pose_ranker_kfold"

    train_pose_scorer_main(
        [
            "--rows_csv",
            str(rows_csv),
            "--output_dir",
            str(output_dir),
            "--objective",
            "pairwise_rank",
            "--split_mode",
            "kfold",
            "--fold_count",
            "2",
            "--fold_index",
            "0",
            "--min_train_queries",
            "1",
            "--min_eval_queries",
            "1",
        ]
    )

    summary = json.loads((output_dir / "pose_scorer_summary.json").read_text())
    assert summary["split"]["mode"] == "kfold"
    assert summary["split"]["fold_count"] == 2
    assert summary["train_query_count"] >= 1
    assert summary["eval_query_count"] >= 1
