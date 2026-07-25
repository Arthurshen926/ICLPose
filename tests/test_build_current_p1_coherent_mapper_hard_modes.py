from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_current_p1_coherent_mapper_hard_modes import (
    P1_HYPOTHESIS_FORMAT,
    _candidate_pool_from_proposal_rows,
    _proposal_arrays_from_layout,
    _validate_bank_alignment,
    _validate_crossfit_partition,
    _validate_target_free_hypothesis_payload,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    GroupedCandidateCrossfitPools,
)


class _Layout:
    row_count = 3
    source_point_ids = np.asarray([50, 51, 52], dtype=np.int64)
    query_ids = np.asarray(["train/a.png", "train/a.png", "val/a.png"])
    xy = np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
    candidate_track_ids = np.asarray([[11, 12], [21, 22], [31, 32]], dtype=np.int64)
    candidate_bank_rows = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    candidate_prior_probabilities = np.asarray(
        [[0.6, 0.3], [0.55, 0.35], [0.6, 0.3]], dtype=np.float32
    )
    null_probabilities = np.asarray([0.1, 0.1, 0.1], dtype=np.float32)


def test_p1_proposal_contract_preserves_fixed_null_mass_and_excludes_validation() -> None:
    arrays = _proposal_arrays_from_layout(_Layout(), np.asarray([0, 1]))

    np.testing.assert_array_equal(arrays["layout_row_indices"], np.asarray([0, 1]))
    np.testing.assert_array_equal(
        arrays["query_ids"], np.asarray(["train/a.png", "train/a.png"])
    )
    np.testing.assert_allclose(
        arrays["candidate_prior_probabilities"].sum(axis=1)
        + arrays["null_probabilities"],
        1.0,
    )
    assert not any("target" in name for name in arrays)


def test_candidate_pool_rejects_bank_track_mismatch() -> None:
    arrays = _proposal_arrays_from_layout(_Layout(), np.asarray([0, 1]))
    with pytest.raises(ValueError, match="different physical tracks"):
        _validate_bank_alignment(
            arrays,
            bank_track_ids=np.asarray([11, 999, 21, 22], dtype=np.int64),
        )


def test_candidate_pool_uses_explicit_prior_and_null_without_recalibration() -> None:
    arrays = _proposal_arrays_from_layout(_Layout(), np.asarray([0, 1]))
    pool = _candidate_pool_from_proposal_rows(
        arrays,
        np.asarray([0], dtype=np.int64),
        bank_xyz=np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )

    np.testing.assert_allclose(pool.descriptor_scores, np.asarray([[0.6, 0.3]]))
    np.testing.assert_allclose(pool.null_scores, np.asarray([0.1]))
    np.testing.assert_array_equal(pool.token_indices, np.asarray([0]))


def test_target_free_hypothesis_payload_rejects_target_field() -> None:
    arrays = {
        "query_ids": np.asarray(["train/a.png"]),
        "split_names": np.asarray(["train"]),
        "evaluation_labels": np.asarray(["p1_train_crossfit"]),
        "hypothesis_indices": np.asarray([0], dtype=np.int64),
        "poses_w2c": np.eye(4, dtype=np.float64)[None],
        "verification_log_likelihood_means": np.asarray([0.0]),
        "audit_log_likelihood_means": np.asarray([np.nan]),
        "sample_track_ids": np.asarray([[1, -1]], dtype=np.int64),
        "sample_token_indices": np.asarray([[2, -1]], dtype=np.int64),
        "generation_profiles": np.asarray(["coarse"]),
        "metadata_json": np.asarray("{}"),
        "ground_truth_residuals": np.asarray([0.0]),
    }
    metadata = {
        "format": P1_HYPOTHESIS_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_denominator": "fixed_global_top20_coarse_prior_with_explicit_null",
        "ranking_pool": "crossfit_verification",
        "audit_pool": "crossfit_audit_diagnostic_only",
    }
    with pytest.raises(ValueError, match="target-free hypothesis artifact exposes"):
        _validate_target_free_hypothesis_payload(arrays, metadata)


def test_target_free_hypothesis_payload_allows_fixed_evaluation_label() -> None:
    arrays = {
        "query_ids": np.asarray(["train/a.png"]),
        "split_names": np.asarray(["train"]),
        "evaluation_labels": np.asarray(["p1_train_crossfit"]),
        "hypothesis_indices": np.asarray([0], dtype=np.int64),
        "poses_w2c": np.eye(4, dtype=np.float64)[None],
        "verification_log_likelihood_means": np.asarray([0.0]),
        "audit_log_likelihood_means": np.asarray([np.nan]),
        "sample_track_ids": np.asarray([[1, -1]], dtype=np.int64),
        "sample_token_indices": np.asarray([[2, -1]], dtype=np.int64),
        "generation_profiles": np.asarray(["coarse"]),
        "metadata_json": np.asarray("{}"),
    }
    metadata = {
        "format": P1_HYPOTHESIS_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_denominator": "fixed_global_top20_coarse_prior_with_explicit_null",
        "ranking_pool": "crossfit_verification",
        "audit_pool": "crossfit_audit_diagnostic_only",
    }
    _validate_target_free_hypothesis_payload(arrays, metadata)


class _Pool:
    pass


def test_crossfit_validation_rejects_overlapping_token_roles() -> None:
    partition = GroupedCandidateCrossfitPools(
        fit=_Pool(),
        shortlist=_Pool(),
        verification=_Pool(),
        audit=_Pool(),
        fit_tokens=(1, 2),
        shortlist_tokens=(3,),
        verification_tokens=(2, 3),
        audit_tokens=(4,),
        partition_audit={"strict_track_disjoint": True},
    )
    with pytest.raises(RuntimeError, match="token roles overlap"):
        _validate_crossfit_partition(partition, require_track_purge=True)


def test_crossfit_validation_requires_track_purge() -> None:
    partition = GroupedCandidateCrossfitPools(
        fit=_Pool(),
        shortlist=_Pool(),
        verification=_Pool(),
        audit=_Pool(),
        fit_tokens=(1, 2),
        shortlist_tokens=(3,),
        verification_tokens=(3,),
        audit_tokens=(4,),
        partition_audit={"strict_track_disjoint": False},
    )
    with pytest.raises(RuntimeError, match="track disjointness"):
        _validate_crossfit_partition(partition, require_track_purge=True)
