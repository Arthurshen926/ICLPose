import numpy as np
import pytest

from feature_extract.tools.vfm.audit_grouped_pose_failure_modes import (
    _actual_sample_candidates,
    _classify,
    _coherent_shift,
)


def test_coherent_shift_requires_repeated_consistent_3d_displacement() -> None:
    coherent, norm, dispersion = _coherent_shift(
        np.asarray([[0.5, 0.0, 0.0], [0.52, 0.01, 0.0], [0.48, -0.01, 0.0]]),
        minimum_pairs=3,
        minimum_norm_m=0.1,
        maximum_dispersion_m=0.2,
        maximum_relative_dispersion=0.35,
    )

    assert coherent
    assert np.isclose(norm, 0.5)
    assert dispersion < 0.03


def test_failure_class_prioritizes_missing_proposals_before_identity_modes() -> None:
    category = _classify(
        translation_error_m=0.3,
        rotation_error_deg=0.1,
        success_threshold_m=0.1,
        success_rotation_threshold_deg=5.0,
        available_fraction=0.25,
        minimum_available_fraction=0.5,
        identity_fraction=0.75,
        minimum_identity_fraction=0.5,
        degenerate=True,
        coherent_shift=True,
        maplet_mismatch_fraction=1.0,
    )

    assert category == "E_correct_candidate_absent_from_frozen_topL_pool"


def test_failure_class_requires_rotation_for_10cm_success() -> None:
    category = _classify(
        translation_error_m=0.05,
        rotation_error_deg=6.0,
        success_threshold_m=0.1,
        success_rotation_threshold_deg=5.0,
        available_fraction=1.0,
        minimum_available_fraction=0.5,
        identity_fraction=1.0,
        minimum_identity_fraction=0.5,
        degenerate=False,
        coherent_shift=False,
        maplet_mismatch_fraction=0.0,
    )

    assert category == "C_identity_right_spatial_or_pose_wrong"


def test_sample_candidates_follow_frozen_candidate_layout_and_query() -> None:
    tracks, residuals = _actual_sample_candidates(
        proposal_row=1,
        selected_track=21,
        expected_query_id="q",
        proposal_query_ids=np.asarray(["other", "q"]),
        candidate_track_ids=np.asarray([[1, 2, 3], [20, 21, 22]]),
        candidate_residuals=np.asarray([[9.0, 9.0, 9.0], [8.0, 1.0, 4.0]]),
        candidate_position_by_proposal_row=np.asarray([-1, 0]),
        candidate_selected_columns=np.asarray([[1, 0]]),
    )

    assert np.array_equal(tracks, np.asarray([21, 20]))
    assert np.array_equal(residuals, np.asarray([1.0, 8.0]))


def test_sample_candidates_reject_wrong_query_or_missing_layout_row() -> None:
    common = {
        "proposal_row": 1,
        "selected_track": 21,
        "proposal_query_ids": np.asarray(["other", "q"]),
        "candidate_track_ids": np.asarray([[1, 2], [20, 21]]),
        "candidate_residuals": np.asarray([[9.0, 9.0], [8.0, 1.0]]),
        "candidate_position_by_proposal_row": np.asarray([-1, -1]),
        "candidate_selected_columns": np.asarray([[1, 0]]),
    }
    with pytest.raises(ValueError, match="different query"):
        _actual_sample_candidates(expected_query_id="other", **common)
    with pytest.raises(ValueError, match="absent from the candidate artifact"):
        _actual_sample_candidates(expected_query_id="q", **common)
