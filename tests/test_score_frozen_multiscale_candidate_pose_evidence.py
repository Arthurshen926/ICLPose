from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _fixed_candidate_views,
    _load_npz_allowlist,
    _parse_profiles,
    _select_s0_verification_rows,
)


def test_profile_parser_requires_declared_source_geometry() -> None:
    profiles = _parse_profiles(
        "final:radio_final:3:9;alike:alike:5:13",
        source_grid_sizes={"radio_final": 16, "alike": 32},
    )
    assert [(item.name, item.source_name) for item in profiles] == [
        ("final", "radio_final"),
        ("alike", "alike"),
    ]
    with pytest.raises(ValueError, match="exceeds"):
        _parse_profiles("too_wide:radio_final:17:9", source_grid_sizes={"radio_final": 16})


def test_allowlist_can_read_legacy_proposals_without_loading_target_field(tmp_path) -> None:
    path = tmp_path / "proposal.npz"
    np.savez(
        path,
        query_ids=np.asarray(["q"], dtype=np.str_),
        candidate_gt_residuals_px=np.asarray([[1.0]], dtype=np.float32),
    )
    arrays, metadata, names = _load_npz_allowlist(
        path,
        ("query_ids",),
        metadata_required=False,
    )
    assert arrays["query_ids"].tolist() == ["q"]
    assert metadata == {}
    assert "candidate_gt_residuals_px" in names
    with pytest.raises(ValueError, match="target-bearing"):
        _load_npz_allowlist(path, ("candidate_gt_residuals_px",), metadata_required=False)


def test_s0_verification_selector_is_heldout_and_stably_merit_ranked() -> None:
    detector = {
        "image_ids": np.asarray(["q"], dtype=np.str_),
        "offsets": np.asarray([0, 5], dtype=np.int64),
        "xy": np.asarray([[0.0, 0.0]] * 5, dtype=np.float32),
        "detector_scores": np.asarray([0.1, 0.9, 0.5, 0.8, 0.7], dtype=np.float32),
    }
    proposals = {
        "query_ids": np.asarray(["q"] * 5, dtype=np.str_),
        "candidate_track_ids": np.arange(10, dtype=np.int64).reshape(5, 2),
        "coarse_scores": np.asarray(
            [[0.1, 0.0], [0.3, 0.0], [0.3, 0.0], [0.2, 0.0], [0.25, 0.0]],
            dtype=np.float32,
        ),
    }
    rows, audit = _select_s0_verification_rows(
        "q",
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0], dtype=np.int64),
        point_count=3,
        detector_log_merit_weight=0.0,
    )
    # Equal coarse scores retain source-row order through mergesort.
    assert rows.tolist() == [1, 2, 4]
    assert audit == {
        "fit_query_point_count": 1,
        "available_unused_query_point_count": 4,
        "selected_verification_point_count": 3,
    }


def test_zero_posterior_candidate_retains_support_layout_but_not_weight() -> None:
    views = _fixed_candidate_views(
        candidate_track_ids=np.asarray([[11, 12]], dtype=np.int64),
        candidate_probabilities=np.asarray([[0.7, 0.0]], dtype=np.float32),
        maplet_track_ids=np.asarray([11, 12], dtype=np.int64),
        support_image_ids=("a", "b"),
        support_image_indices=np.asarray([[0, 1], [1, -1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[3, 1], [2, 0]], dtype=np.int64),
    )
    assert views.valid.tolist() == [[[True, True], [True, False]]]
    assert np.allclose(views.weights[0, 0], [0.75, 0.25])
    assert np.allclose(views.weights[0, 1], [0.0, 0.0])
