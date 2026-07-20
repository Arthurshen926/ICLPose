import json

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_candidate_spatial_rank_coverage import (
    audit_candidate_spatial_rank_coverage,
)


def test_rank_coverage_audit_preserves_omitted_candidate_mass(tmp_path) -> None:
    evidence_path = tmp_path / "candidate_evidence.npz"
    np.savez(
        evidence_path,
        selected_rows=np.asarray([17], dtype=np.int64),
        split_names=np.asarray(["validation"]),
        candidate_valid=np.asarray([[True, True, True]]),
        candidate_track_ids=np.asarray([[10, 20, 30]], dtype=np.int64),
        candidate_prototype_ids=np.asarray([[0, 0, 0]], dtype=np.int64),
        candidate_prior_probabilities=np.asarray([[0.2, 0.3, 0.1]], dtype=np.float32),
        candidate_score_ranks=np.asarray([[1, 6, 11]], dtype=np.int64),
        unknown_probability=np.asarray([0.4], dtype=np.float32),
        candidate_target_gt_residuals_px=np.asarray([[1.0, 3.0, 1.5]], dtype=np.float32),
        metadata_json=np.asarray(json.dumps({"format": "candidate_evidence_v3"})),
    )
    spatial_path = tmp_path / "spatial.npz"
    np.savez(
        spatial_path,
        source_query_rows=np.asarray([17, 17], dtype=np.int64),
        candidate_measurement_ranks=np.asarray([1, 11], dtype=np.int64),
        candidate_track_ids=np.asarray([10, 30], dtype=np.int64),
        candidate_prototype_ids=np.asarray([0, 0], dtype=np.int64),
        support_view_ranks=np.asarray([0, 0], dtype=np.int64),
        support_view_probabilities=np.asarray([1.0, 1.0], dtype=np.float32),
        local_log_probabilities=np.log(
            np.asarray([[0.75, 0.25], [0.5, 0.5]], dtype=np.float32)
        ),
        dustbin_probabilities=np.asarray([0.2, 0.7], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_spatial_likelihood_v7",
                    "contains_ground_truth_arrays": False,
                    "ground_truth_loaded_by_inference_process": False,
                    "pose_or_ground_truth_used_for_inference": False,
                    "prediction_frozen_before_target_join": True,
                }
            )
        ),
    )

    report = audit_candidate_spatial_rank_coverage(
        candidate_evidence_path=evidence_path,
        candidate_spatial_likelihood_paths=(spatial_path,),
        output_path=tmp_path / "report.json",
    )

    summary = report["splits"]["validation"]
    assert summary["rank_1_5"]["spatial_candidate_coverage_rate"] == 1.0
    assert summary["rank_6_10"]["spatial_candidate_coverage_rate"] == 0.0
    assert summary["rank_11_20"]["correct_candidate_occurrence_rate_2px"] == 1.0
    conservation = summary["topk_mass_conservation"]
    assert conservation["materialized_candidate_mass_mean"] == pytest.approx(0.3)
    assert conservation["unmaterialized_candidate_mass_mean"] == pytest.approx(0.3)
    assert conservation["effective_null_mass_mean"] == pytest.approx(0.7)
    assert conservation["materialized_plus_effective_null_max_abs_error"] < 1e-6
