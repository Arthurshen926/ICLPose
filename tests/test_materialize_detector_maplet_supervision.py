from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.materialize_detector_maplet_supervision import (
    materialize_supervision,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _write_proposals(path: Path) -> None:
    np.savez(
        path,
        query_ids=np.asarray(["q0", "q1", "q2"]),
        candidate_gt_residuals_px=np.asarray(
            [[1.0, 3.0, np.inf], [2.0, 2.01, 0.5], [8.0, 1.5, 4.0]],
            dtype=np.float32,
        ),
    )


def _write_features(
    path: Path,
    proposals: Path,
    *,
    rows: np.ndarray | None = None,
    include_labels: bool = False,
    proposal_hash: str | None = None,
    legacy_pose_field: bool = False,
) -> None:
    selected_rows = np.arange(3, dtype=np.int64) if rows is None else rows
    selected_columns = np.asarray([[2, 0], [1, 2], [0, 1]], dtype=np.int64)[
        selected_rows
    ]
    arrays: dict[str, np.ndarray] = {
        "selected_rows": selected_rows,
        "selected_columns": selected_columns,
        "features": np.zeros((len(selected_rows), 2, 2), dtype=np.float32),
        "valid_edges": np.ones((len(selected_rows), 2), dtype=bool),
    }
    if legacy_pose_field:
        arrays["selected_from_pose_keep"] = np.zeros(len(selected_rows), dtype=bool)
    if include_labels:
        arrays["labels"] = np.zeros((len(selected_rows), 2), dtype=bool)
    metadata = {
        "format": "detector_maplet_geometry_features_v1",
        "supervision_mode": "none_inference_only",
        "positive_threshold_px": 2.0,
        "proposals_sha256": (
            file_sha256_short(proposals) if proposal_hash is None else proposal_hash
        ),
    }
    np.savez(
        path,
        **arrays,
        feature_names=np.asarray(["a", "b"]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_materialize_supervision_maps_only_selected_candidates(tmp_path: Path) -> None:
    proposals = tmp_path / "proposals.npz"
    features = tmp_path / "features.npz"
    _write_proposals(proposals)
    _write_features(features, proposals)

    arrays, names, metadata = materialize_supervision(
        features, proposals, positive_threshold_px=2.0
    )

    np.testing.assert_array_equal(
        arrays["labels"],
        np.asarray([[False, True], [False, True], [False, True]]),
    )
    assert names == ("a", "b")
    assert metadata["supervision_full_proposal_row_coverage"] is True
    assert metadata["supervision_positive_edge_count"] == 3
    assert metadata["source_inference_feature_artifact_sha256"] == file_sha256_short(
        features
    )
    assert "selected_from_pose_keep" not in arrays


def test_materialize_supervision_rejects_non_target_free_source(
    tmp_path: Path,
) -> None:
    proposals = tmp_path / "proposals.npz"
    features = tmp_path / "features.npz"
    _write_proposals(proposals)
    _write_features(features, proposals, include_labels=True)

    with pytest.raises(ValueError, match="target-free contract"):
        materialize_supervision(features, proposals, positive_threshold_px=2.0)


def test_materialize_supervision_removes_legacy_pose_audit_field(
    tmp_path: Path,
) -> None:
    proposals = tmp_path / "proposals.npz"
    features = tmp_path / "features.npz"
    _write_proposals(proposals)
    _write_features(features, proposals, legacy_pose_field=True)

    arrays, _names, metadata = materialize_supervision(
        features, proposals, positive_threshold_px=2.0
    )

    assert "selected_from_pose_keep" not in arrays
    assert metadata["legacy_pose_keep_audit_field_removed"] is True


def test_materialize_supervision_rejects_partial_or_stale_source(
    tmp_path: Path,
) -> None:
    proposals = tmp_path / "proposals.npz"
    features = tmp_path / "features.npz"
    _write_proposals(proposals)
    _write_features(features, proposals, rows=np.asarray([0, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="full proposal-row coverage"):
        materialize_supervision(features, proposals, positive_threshold_px=2.0)

    _write_features(features, proposals, proposal_hash="stale")
    with pytest.raises(ValueError, match="different proposals"):
        materialize_supervision(features, proposals, positive_threshold_px=2.0)
