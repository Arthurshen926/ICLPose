from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_current_v3_frozen_candidate_layout import (
    ARTIFACT_FORMAT,
    build_current_v3_frozen_candidate_layout,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _write_maplet(path) -> None:
    np.savez(
        path,
        anchor_track_ids=np.asarray([11, 12], dtype=np.int64),
        support_image_ids=np.asarray(["s1.png", "s2.png", "s3.png"]),
        support_image_indices=np.asarray([[0, 1], [2, -1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[5, 3], [4, 0]], dtype=np.int32),
        metadata_json=np.asarray(
            json.dumps({"format": "local_maplet_support_index_npz", "max_support_views": 2})
        ),
    )


def _write_evidence(path, *, maplet_sha: str, second_track: int = 12) -> None:
    selected_rows = np.asarray([4, 5, 6], dtype=np.int64)
    tracks = np.asarray([[11, second_track], [11, second_track], [11, second_track]])
    valid = np.ones_like(tracks, dtype=bool)
    np.savez(
        path,
        selected_rows=selected_rows,
        query_ids=np.asarray(["q_train.png", "q_validation.png", "q_test.png"]),
        query_xy=np.asarray([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]], dtype=np.float32),
        split_names=np.asarray(["train", "validation", "test"]),
        candidate_valid=valid,
        candidate_track_ids=tracks,
        candidate_bank_rows=np.asarray([[0, 1], [0, 1], [0, 1]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray(
            [[0.8, 0.4], [0.7, 0.3], [0.6, 0.2]], dtype=np.float32
        ),
        # Its presence proves that the target-bearing source artifact is not
        # copied into the target-free train/validation layout.
        candidate_target_gt_residuals_px=np.asarray(
            [[1.0, 9.0], [2.0, 8.0], [3.0, 7.0]], dtype=np.float32
        ),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_evidence_v3",
                    "candidate_probability_semantics": "factorized_top_l_availability_times_conditional_identity",
                    "pose_used_for_selection": False,
                    "image_retrieval": False,
                    "render": False,
                    "maplet_support_index_sha256": maplet_sha,
                    "proposals_sha256": "proposal-sha",
                    "descriptor_space_id": "space-test",
                    "descriptor_space_manifest": {"checkpoint_sha256": "radio-test"},
                }
            )
        ),
    )


def test_current_v3_layout_excludes_test_and_target_arrays(tmp_path) -> None:
    maplet = tmp_path / "maplet.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "layout.npz"
    summary = tmp_path / "summary.json"
    _write_maplet(maplet)
    _write_evidence(evidence, maplet_sha=file_sha256_short(maplet))

    result = build_current_v3_frozen_candidate_layout(
        candidate_evidence=evidence,
        maplet_support_index=maplet,
        output=output,
        summary_json=summary,
    )

    assert result["protocol"]["test_rows_materialized"] is False
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        assert data["split_names"].tolist() == ["train", "validation"]
        assert data["source_row_indices"].tolist() == [4, 5]
        assert "candidate_target_gt_residuals_px" not in data.files
        assert data["feature_names"].tolist() == ["radio_final_anchor_cosine"]
        features = np.asarray(data["candidate_features"], dtype=np.float32)
        views = np.asarray(data["candidate_view_valid"], dtype=bool)
        assert features.shape == (2, 2, 2, 1)
        np.testing.assert_allclose(features[0, 0, :2, 0], [0.8, 0.8])
        np.testing.assert_allclose(features[0, 1, 0, 0], 0.4)
        assert np.isnan(features[0, 1, 1, 0])
        assert views.tolist() == [
            [[True, True], [True, False]],
            [[True, True], [True, False]],
        ]
        assert metadata["format"] == ARTIFACT_FORMAT
        assert metadata["source_target_arrays_read"] is False
        assert metadata["source_test_rows_materialized"] is False


def test_current_v3_layout_rejects_candidate_maplet_identity_mismatch(tmp_path) -> None:
    maplet = tmp_path / "maplet.npz"
    evidence = tmp_path / "evidence.npz"
    _write_maplet(maplet)
    _write_evidence(
        evidence,
        maplet_sha=file_sha256_short(maplet),
        second_track=99,
    )

    with pytest.raises(ValueError, match="differ from the maplet"):
        build_current_v3_frozen_candidate_layout(
            candidate_evidence=evidence,
            maplet_support_index=maplet,
            output=tmp_path / "layout.npz",
            summary_json=tmp_path / "summary.json",
        )
