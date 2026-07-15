from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.merge_detector_maplet_feature_shards import (
    merge_feature_shards,
)


def _write_shard(
    path: Path, *, index: int, rows: list[int], legacy_pose_field: bool = False
) -> None:
    metadata = {
        "format": "detector_maplet_geometry_features_v1",
        "supervision_mode": "none_inference_only",
        "query_shard_count": 2,
        "query_shard_index": index,
        "candidate_top_k": 2,
    }
    count = len(rows)
    arrays = {
        "selected_rows": np.asarray(rows, dtype=np.int64),
        "selected_columns": np.zeros((count, 2), dtype=np.int64),
        "features": np.zeros((count, 2, 3), dtype=np.float32),
        "valid_edges": np.ones((count, 2), dtype=bool),
    }
    if legacy_pose_field:
        arrays["selected_from_pose_keep"] = np.zeros((count,), dtype=bool)
    np.savez(
        path,
        **arrays,
        feature_names=np.asarray(["a", "b", "c"]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_merge_inference_feature_shards_requires_exact_row_coverage(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(first, index=0, rows=[0, 2])
    _write_shard(second, index=1, rows=[1, 3])

    arrays, names, metadata = merge_feature_shards(
        [first, second], expected_row_count=4
    )

    np.testing.assert_array_equal(arrays["selected_rows"], np.arange(4))
    assert names == ("a", "b", "c")
    assert metadata["merged_query_shard_count"] == 2
    assert "selected_from_pose_keep" not in arrays

    _write_shard(second, index=1, rows=[0, 3])
    with pytest.raises(ValueError, match="overlap"):
        merge_feature_shards([first, second], expected_row_count=4)


def test_merge_legacy_inference_shards_sanitizes_only_after_full_coverage(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(first, index=0, rows=[0, 2], legacy_pose_field=True)
    _write_shard(second, index=1, rows=[1, 3], legacy_pose_field=True)

    with pytest.raises(ValueError, match="exact full-row coverage"):
        merge_feature_shards([first, second])
    arrays, _names, metadata = merge_feature_shards(
        [first, second], expected_row_count=4
    )
    assert "selected_from_pose_keep" not in arrays
    assert metadata["legacy_pose_keep_audit_field_removed"] is True
    assert (
        metadata["query_point_selection"]
        == "full_proposal_row_coverage_target_free_sanitized_v1"
    )
