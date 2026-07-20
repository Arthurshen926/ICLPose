from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest

from feature_extract.tools.vfm.fit_multiscale_candidate_probe import _load_features
from feature_extract.tools.vfm.merge_structured_multiscale_candidate_probe_shards import (
    merge_structured_multiscale_candidate_probe_shards,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:16]


def _write_frozen_layout(tmp_path) -> object:
    rows = np.asarray([100, 101, 102, 103], dtype=np.int64)
    tracks = np.ones((4, 1), dtype=np.int64)
    support_ids = np.asarray([["support.png"]] * 4)
    path = tmp_path / "frozen.npz"
    metadata = {
        "format": "multiscale_candidate_probe_features_v1",
        "contains_ground_truth": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "source_row_selection": "heldout_detector_merit_after_target_free_fit_rows_v1",
        "verification_point_count": 192,
        "support_view_selection": "fixed_maplet_coverage_rank_with_real_observation_fallback_v1",
        "support_view_count": 2,
        "candidate_fit_rows_sha256": "fitrows",
        "detector_log_merit_weight": 0.01,
    }
    np.savez(
        path,
        source_row_indices=rows,
        query_ids=np.asarray([f"q{value}" for value in range(4)]),
        split_names=np.asarray(["train"] * 4),
        xy=np.zeros((4, 2), dtype=np.float32),
        candidate_track_ids=tracks,
        candidate_canonical_rows=np.zeros((4, 1), dtype=np.int64),
        candidate_view_valid=np.ones((4, 1, 1), dtype=bool),
        candidate_support_image_ids=support_ids,
        candidate_support_coverage_counts=np.ones((4, 1, 1), dtype=np.int32),
        feature_names=np.asarray(
            STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES, dtype=np.str_
        ),
        metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
    )
    return path


def _write_shard(
    tmp_path,
    *,
    frozen_path,
    index: int,
    positions: list[int],
    feature_names=STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    artifact_format: str = "structured_multiscale_candidate_probe_features_v2",
) -> object:
    rows = np.asarray(positions, dtype=np.int64)
    count = len(rows)
    feature_count = len(feature_names)
    features = np.zeros((count, 1, 1, feature_count), dtype=np.float16)
    features[:, 0, 0, 0] = rows.astype(np.float16)
    all_rows = np.asarray([100, 101, 102, 103], dtype=np.int64)
    all_tracks = np.ones((4, 1), dtype=np.int64)
    all_support_ids = np.asarray([["support.png"]] * 4)

    metadata = {
        "format": artifact_format,
        "contains_ground_truth": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "diagnostic_max_rows": 0,
        "is_complete_frozen_layout": False,
        "layout_shard_count": 2,
        "layout_shard_index": index,
        "layout_position_count": count,
        "full_frozen_layout_row_count": 4,
        "full_frozen_source_rows_sha256": _digest(all_rows),
        "full_frozen_candidate_tracks_sha256": _digest(all_tracks),
        "full_frozen_support_view_ids_sha256": _digest(all_support_ids),
        "frozen_layout_features": str(frozen_path),
        "frozen_layout_features_sha256": _digest(np.frombuffer(frozen_path.read_bytes(), dtype=np.uint8)),
        "context": {"radius": 1.0},
    }
    path = tmp_path / f"shard{index}.npz"
    np.savez(
        path,
        layout_positions=rows,
        source_row_indices=rows + 100,
        query_ids=np.asarray([f"q{value}" for value in rows]),
        split_names=np.asarray(["train"] * count),
        xy=np.zeros((count, 2), dtype=np.float32),
        candidate_track_ids=np.ones((count, 1), dtype=np.int64),
        candidate_canonical_rows=np.zeros((count, 1), dtype=np.int64),
        candidate_features=features,
        candidate_view_valid=np.ones((count, 1, 1), dtype=bool),
        candidate_support_image_ids=np.asarray([["support.png"]] * count),
        candidate_support_coverage_counts=np.ones((count, 1, 1), dtype=np.int32),
        feature_names=np.asarray(
            feature_names, dtype=np.str_
        ),
        metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
    )
    return path


def test_merge_restores_frozen_layout_order_and_requires_full_coverage(tmp_path) -> None:
    # Deliberately pass shards in reverse order: output must be ordered by the
    # original frozen layout positions, never by source-row value or CLI order.
    frozen = _write_frozen_layout(tmp_path)
    shard0 = _write_shard(tmp_path, frozen_path=frozen, index=0, positions=[0, 1])
    shard1 = _write_shard(tmp_path, frozen_path=frozen, index=1, positions=[2, 3])
    arrays, names, metadata = merge_structured_multiscale_candidate_probe_shards(
        [shard1, shard0]
    )
    assert names == STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES
    np.testing.assert_array_equal(arrays["source_row_indices"], [100, 101, 102, 103])
    np.testing.assert_array_equal(arrays["candidate_features"][:, 0, 0, 0], [0, 1, 2, 3])
    assert metadata["is_complete_frozen_layout"] is True
    assert metadata["merged_layout_shard_count"] == 2


def test_fit_loader_rejects_unmerged_structured_shard(tmp_path) -> None:
    frozen = _write_frozen_layout(tmp_path)
    shard = _write_shard(tmp_path, frozen_path=frozen, index=0, positions=[0, 1])
    with pytest.raises(ValueError, match="fully merged frozen layout"):
        _load_features(shard)


def test_merge_accepts_cost_volume_schema_without_relaxing_layout_checks(tmp_path) -> None:
    frozen = _write_frozen_layout(tmp_path)
    kwargs = {
        "feature_names": COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
        "artifact_format": "cost_volume_multiscale_candidate_probe_features_v1",
    }
    shard0 = _write_shard(
        tmp_path, frozen_path=frozen, index=0, positions=[0, 1], **kwargs
    )
    shard1 = _write_shard(
        tmp_path, frozen_path=frozen, index=1, positions=[2, 3], **kwargs
    )
    arrays, names, metadata = merge_structured_multiscale_candidate_probe_shards(
        [shard0, shard1]
    )
    assert names == COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES
    assert metadata["format"] == "cost_volume_multiscale_candidate_probe_features_v1"
    assert metadata["is_complete_frozen_layout"] is True
    assert arrays["candidate_features"].shape[-1] == len(names)


def test_merge_accepts_wide_full_correlation_schema_without_relaxing_layout_checks(
    tmp_path,
) -> None:
    frozen = _write_frozen_layout(tmp_path)
    kwargs = {
        "feature_names": WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
        "artifact_format": "wide_full_correlation_multiscale_candidate_probe_features_v1",
    }
    shard0 = _write_shard(
        tmp_path, frozen_path=frozen, index=0, positions=[0, 1], **kwargs
    )
    shard1 = _write_shard(
        tmp_path, frozen_path=frozen, index=1, positions=[2, 3], **kwargs
    )
    arrays, names, metadata = merge_structured_multiscale_candidate_probe_shards(
        [shard0, shard1]
    )
    assert names == WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES
    assert metadata["format"] == "wide_full_correlation_multiscale_candidate_probe_features_v1"
    assert metadata["is_complete_frozen_layout"] is True
    assert arrays["candidate_features"].shape[-1] == len(names)
