from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _fixed_candidate_views,
    _load_exact_hypotheses,
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


def test_exact_hypothesis_loader_selects_a_query_before_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence as module

    baseline = {
        "query_ids": np.asarray(["query-a", "query-b", "query-b"], dtype=np.str_),
        "split_names": np.asarray(["train", "train", "train"], dtype=np.str_),
        "evaluation_labels": np.asarray(["a0", "b0", "b1"], dtype=np.str_),
        "hypothesis_indices": np.asarray([3, 7, 8], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray([False, True, False]),
        "independent_score_top1": np.asarray([False, True, False]),
        "independent_selection_scores": np.asarray([0.1, 0.7, 0.2]),
    }
    # Deliberately reorder source rows: exact identity, not artifact order,
    # must select the matching frozen pose.
    hypotheses = {
        "query_ids": np.asarray(["query-b", "query-a", "query-b"], dtype=np.str_),
        "split_names": np.asarray(["train", "train", "train"], dtype=np.str_),
        "evaluation_labels": np.asarray(["b1", "a0", "b0"], dtype=np.str_),
        "hypothesis_indices": np.asarray([8, 3, 7], dtype=np.int64),
        "poses_w2c": np.asarray(
            [
                [[1.0, 0.0, 0.0, 80.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 30.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 70.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        ),
    }
    monkeypatch.setattr(
        module,
        "_load_npz_allowlist",
        lambda *_args, **_kwargs: (baseline, {}, tuple(baseline)),
    )
    monkeypatch.setattr(module, "_validate_baseline_score_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "load_inference_artifact_fields",
        lambda *_args, **_kwargs: (hypotheses, {}),
    )

    exact, _hypothesis_metadata, _baseline_metadata = _load_exact_hypotheses(
        hypothesis_path=Path("hypotheses.npz"),
        baseline_path=Path("baseline.npz"),
        detector_path=Path("detector.npz"),
        proposals_path=Path("proposals.npz"),
        candidate_path=Path("candidate.npz"),
        prior_path=Path("prior.npz"),
        fixed_candidate_top_k=20,
        query_id="query-b",
    )

    assert exact["query_ids"].tolist() == ["query-b", "query-b"]
    assert exact["hypothesis_indices"].tolist() == [7, 8]
    np.testing.assert_allclose(exact["poses_w2c"][:, 0, 3], [70.0, 80.0])
