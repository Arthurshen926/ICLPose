from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.materialize_candidate_maplet_inference_subset import (
    SUBSET_SELECTION_POLICY,
    materialize_candidate_maplet_inference_subset,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    proposals = root / "proposals.npz"
    query_ids = np.asarray(["q0"] * 4 + ["q1"] * 4)
    xy = np.asarray(
        [
            [10, 10],
            [90, 10],
            [10, 90],
            [90, 90],
            [15, 15],
            [85, 15],
            [15, 85],
            [85, 85],
        ],
        dtype=np.float32,
    )
    np.savez(
        proposals,
        query_ids=query_ids,
        xy=xy,
        pose_keep_mask=np.ones((8,), dtype=bool),
        candidate_gt_residuals_px=np.zeros((8, 2), dtype=np.float32),
    )
    detector = root / "detector.npz"
    np.savez(
        detector,
        image_ids=np.asarray(["q0", "q1"]),
        offsets=np.asarray([0, 4, 8], dtype=np.int64),
        xy=xy,
        detector_scores=np.asarray(
            [0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.75, 0.65],
            dtype=np.float32,
        ),
        metadata_json=np.asarray(
            json.dumps({"detector_selection_version": "source_v1"})
        ),
    )
    feature = root / "features.npz"
    metadata = {
        "format": "detector_maplet_geometry_features_v1",
        "proposals_sha256": file_sha256_short(proposals),
        "projected_landmark_bank_sha256": "bank",
        "maplet_support_index_sha256": "maplet",
        "support_geometry_index_sha256": "geometry",
        "query_point_selection": "full_target_free",
        "supervision_mode": "none_inference_only",
    }
    features = np.arange(8 * 2 * 3, dtype=np.float32).reshape(8, 2, 3)
    np.savez(
        feature,
        selected_rows=np.arange(8, dtype=np.int64),
        selected_columns=np.tile(np.asarray([[0, 1]], dtype=np.int64), (8, 1)),
        features=features,
        valid_edges=np.ones((8, 2), dtype=bool),
        feature_names=np.asarray(["a", "b", "c"]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return proposals, detector, feature


def test_materializes_target_free_detector_subset(tmp_path: Path) -> None:
    proposals, detector, feature = _write_inputs(tmp_path)
    output = tmp_path / "subset.npz"
    summary = materialize_candidate_maplet_inference_subset(
        proposals_path=proposals,
        detector_query_cache_path=detector,
        inference_feature_artifact_path=feature,
        output_path=output,
        query_points_per_image=2,
        image_width=100,
        image_height=100,
        grid_rows=2,
        grid_cols=2,
    )
    with np.load(output, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "selected_rows",
            "selected_columns",
            "features",
            "valid_edges",
            "feature_names",
            "metadata_json",
        }
        assert payload["selected_rows"].shape == (4,)
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["query_point_selection"] == SUBSET_SELECTION_POLICY
    assert metadata["supervision_mode"] == "none_inference_only"
    assert metadata["contains_ground_truth"] is False
    assert metadata["contains_pose_derived_selection"] is False
    assert summary["protocol"]["ground_truth_loaded"] is False
    assert summary["selection"]["per_image_counts"] == {"q0": 2, "q1": 2}


def test_rejects_source_feature_artifact_with_supervision(tmp_path: Path) -> None:
    proposals, detector, feature = _write_inputs(tmp_path)
    with np.load(feature, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    arrays["labels"] = np.zeros((8, 2), dtype=bool)
    supervised = tmp_path / "supervised.npz"
    np.savez(supervised, **arrays)
    with pytest.raises(ValueError, match="supervision fields"):
        materialize_candidate_maplet_inference_subset(
            proposals_path=proposals,
            detector_query_cache_path=detector,
            inference_feature_artifact_path=supervised,
            output_path=tmp_path / "out.npz",
            query_points_per_image=2,
            image_width=100,
            image_height=100,
            grid_rows=2,
            grid_cols=2,
        )
