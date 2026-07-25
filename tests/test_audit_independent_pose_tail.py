from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.audit_independent_pose_tail import (
    SCORE_FORMAT,
    TARGET_FORMAT,
    audit_independent_pose_tail,
)


def _write_score(path, *, rows, scores, coverage, spatial_count=0) -> None:
    count = len(rows)
    arrays = {
        "query_ids": np.asarray([item[0] for item in rows]),
        "split_names": np.asarray([item[1] for item in rows]),
        "evaluation_labels": np.asarray([item[2] for item in rows]),
        "hypothesis_indices": np.asarray([item[3] for item in rows], dtype=np.int64),
        "independent_selection_scores": np.asarray(scores, dtype=np.float64),
        "independent_log_likelihood_means": np.asarray(scores, dtype=np.float64),
        "independent_log_likelihood_medians": np.asarray(scores, dtype=np.float64),
        "independent_log_likelihood_worst_quartile_means": np.asarray(scores, dtype=np.float64),
        "independent_log_likelihood_lcb95s": np.asarray(scores, dtype=np.float64),
        "independent_spatial_median_of_means_2x2": np.asarray(scores, dtype=np.float64),
        "independent_effective_point_counts": np.full((count,), 96, dtype=np.int64),
        "independent_evidence_coverages": np.asarray(coverage, dtype=np.float64),
        "verification_point_counts": np.full((count,), 192, dtype=np.int64),
        "candidate_spatial_materialized_verification_point_counts": np.full((count,), spatial_count, dtype=np.int64),
        "candidate_spatial_materialized_candidate_view_counts": np.full((count,), spatial_count, dtype=np.int64),
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps({"format": SCORE_FORMAT, "contains_target_fields": False, "pose_or_ground_truth_used_for_scoring": False, "inputs": {"candidate_spatial_likelihood": []}})),
    )


def _write_target(path, *, rows, errors) -> None:
    np.savez_compressed(
        path,
        query_ids=np.asarray([item[0] for item in rows]),
        split_names=np.asarray([item[1] for item in rows]),
        evaluation_labels=np.asarray([item[2] for item in rows]),
        hypothesis_indices=np.asarray([item[3] for item in rows], dtype=np.int64),
        translation_errors_m=np.asarray(errors, dtype=np.float64),
        rotation_errors_deg=np.zeros((len(rows),), dtype=np.float64),
        metadata_json=np.asarray(json.dumps({"format": TARGET_FORMAT, "contains_target_fields": True})),
    )


def test_tail_audit_identifies_a_high_coverage_self_consistent_tail(tmp_path) -> None:
    rows = [
        ("q0", "validation", "label", 0),
        ("q0", "validation", "label", 1),
        ("q1", "validation", "label", 0),
        ("q1", "validation", "label", 1),
    ]
    score = tmp_path / "score.npz"
    target = tmp_path / "target.npz"
    _write_score(score, rows=rows, scores=[3.0, 1.0, 2.0, 1.0], coverage=[0.8, 0.7, 0.2, 0.7])
    _write_target(target, rows=rows, errors=[1.2, 0.04, 0.2, 0.05])

    report = audit_independent_pose_tail(
        score_artifacts=[score], target_artifact=target, low_coverage_threshold=0.5
    )

    assert report["summary"]["catastrophic_selected_count"] == 1
    assert report["summary"]["catastrophic_with_good_oracle_count"] == 1
    tail = report["tail_queries_TARGET_ONLY"][0]
    assert tail["query_id"] == "q0"
    assert tail["oracle_score_rank_TARGET_ONLY"] == 2
    assert "self_consistent_wrong_pose_under_aggregate_evidence" in tail["tail_categories"]
    assert "rgb_spatial_modes_not_present" in tail["tail_categories"]
    assert report["mode_status"]["spatial_modes_available_for_this_score"] is False


def test_tail_audit_reports_materialized_spatial_modes_without_claiming_block_attribution(tmp_path) -> None:
    rows = [("q0", "validation", "label", 0), ("q0", "validation", "label", 1)]
    score = tmp_path / "score.npz"
    target = tmp_path / "target.npz"
    _write_score(score, rows=rows, scores=[0.0, 1.0], coverage=[0.8, 0.8], spatial_count=32)
    _write_target(target, rows=rows, errors=[0.1, 0.2])

    report = audit_independent_pose_tail(score_artifacts=[score], target_artifact=target)

    assert report["mode_status"]["spatial_modes_available_for_this_score"] is True
    assert report["mode_status"]["per_point_or_block_sidecar_available"] is False
    assert report["all_queries_TARGET_ONLY"][0]["tail_categories"] == ["non_tail"]
