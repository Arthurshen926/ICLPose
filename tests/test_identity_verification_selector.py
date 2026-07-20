import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_identity_verification_selector import (
    build_identity_verification_selector,
    select_identity_confidence_spatial_quota,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_verification_point_selection_artifact,
)
from feature_extract.vfm.artifacts import file_sha256_short


def test_identity_selector_uses_each_grid_cell_before_global_fill() -> None:
    rows = np.arange(8, dtype=np.int64)
    xy = np.asarray(
        [
            [5.0, 5.0],
            [10.0, 10.0],
            [95.0, 5.0],
            [90.0, 10.0],
            [5.0, 95.0],
            [10.0, 90.0],
            [95.0, 95.0],
            [90.0, 90.0],
        ],
        dtype=np.float32,
    )
    posterior = np.asarray(
        [[0.1, 0.0], [0.9, 0.0], [0.2, 0.0], [0.8, 0.0],
         [0.3, 0.0], [0.7, 0.0], [0.4, 0.0], [0.6, 0.0]],
        dtype=np.float32,
    )

    selected, scores = select_identity_confidence_spatial_quota(
        source_rows=rows,
        xy=xy,
        candidate_probabilities=posterior,
        point_count=4,
        grid_rows=2,
        grid_columns=2,
        points_per_cell=1,
        image_width=100,
        image_height=100,
    )

    np.testing.assert_array_equal(selected, [1, 3, 5, 7])
    np.testing.assert_allclose(scores, [0.9, 0.8, 0.7, 0.6])


def test_identity_selector_falls_back_globally_when_grid_has_empty_cells() -> None:
    selected, _scores = select_identity_confidence_spatial_quota(
        source_rows=np.asarray([10, 11, 12], dtype=np.int64),
        xy=np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        candidate_probabilities=np.asarray(
            [[0.1], [0.9], [0.8]], dtype=np.float32
        ),
        point_count=3,
        grid_rows=2,
        grid_columns=2,
        points_per_cell=1,
        image_width=100,
        image_height=100,
    )

    np.testing.assert_array_equal(selected, [11, 12, 10])


def test_built_selector_is_target_free_and_fit_row_disjoint(tmp_path) -> None:
    detector_path = tmp_path / "detector.npz"
    proposals_path = tmp_path / "proposals.npz"
    candidate_path = tmp_path / "candidate.npz"
    overlay_path = tmp_path / "overlay.npz"
    output_path = tmp_path / "selector.npz"
    summary_path = tmp_path / "summary.json"
    tracks = np.asarray([[10], [11], [12], [13], [14], [15]], dtype=np.int64)
    np.savez(
        detector_path,
        image_ids=np.asarray(["query.png"]),
        offsets=np.asarray([0, 6], dtype=np.int64),
        xy=np.asarray(
            [[10.0, 10.0], [20.0, 20.0], [510.0, 10.0], [520.0, 20.0], [10.0, 400.0], [520.0, 400.0]],
            dtype=np.float32,
        ),
        metadata_json=np.asarray("{}"),
    )
    np.savez(
        proposals_path,
        query_ids=np.asarray(["query.png"] * 6),
        candidate_track_ids=tracks,
    )
    np.savez(
        candidate_path,
        selected_rows=np.asarray([0, 1], dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "contains_ground_truth": False,
                    "supervision_mode": "none_inference_only",
                    "detector_query_cache_sha256": file_sha256_short(detector_path),
                    "proposals_sha256": file_sha256_short(proposals_path),
                }
            )
        ),
    )
    np.savez(
        overlay_path,
        candidate_track_ids=tracks,
        candidate_probabilities=np.asarray(
            [[0.1], [0.2], [0.9], [0.8], [0.7], [0.6]], dtype=np.float32
        ),
        null_probabilities=np.asarray([0.9, 0.8, 0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "contains_ground_truth": False,
                    "contains_target_errors": False,
                    "probability_semantics": (
                        "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
                    ),
                    "proposals_sha256": file_sha256_short(proposals_path),
                }
            )
        ),
    )

    summary = build_identity_verification_selector(
        detector_path=detector_path,
        proposals_path=proposals_path,
        candidate_path=candidate_path,
        identity_overlay_path=overlay_path,
        output_path=output_path,
        summary_path=summary_path,
        point_count=4,
        grid_rows=2,
        grid_columns=2,
        points_per_cell=1,
        image_width=1024,
        image_height=576,
    )

    assert summary["protocol"]["target_free"] is True
    with np.load(output_path, allow_pickle=False) as data:
        rows = np.asarray(data["source_row_indices"], dtype=np.int64)
    assert not np.any(np.isin(rows, [0, 1]))
    detector = {
        "image_ids": np.asarray(["query.png"]),
        "offsets": np.asarray([0, 6], dtype=np.int64),
    }
    proposals = {"query_ids": np.asarray(["query.png"] * 6)}
    loaded, metadata = _load_verification_point_selection_artifact(
        output_path,
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0, 1], dtype=np.int64),
        detector_path=detector_path,
        proposals_path=proposals_path,
        candidate_path=candidate_path,
    )
    assert metadata["contains_ground_truth"] is False
    np.testing.assert_array_equal(loaded["query.png"], rows)


def test_identity_selector_rejects_over_budget() -> None:
    with pytest.raises(ValueError, match="inputs are invalid"):
        select_identity_confidence_spatial_quota(
            source_rows=np.asarray([0], dtype=np.int64),
            xy=np.asarray([[1.0, 1.0]], dtype=np.float32),
            candidate_probabilities=np.asarray([[1.0]], dtype=np.float32),
            point_count=2,
            grid_rows=1,
            grid_columns=1,
            points_per_cell=1,
            image_width=10,
            image_height=10,
        )
