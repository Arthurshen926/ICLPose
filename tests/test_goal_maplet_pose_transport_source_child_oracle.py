from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_pose_transport_source_child_oracle import (
    _replace_source_evidence,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "source_child_rows": np.asarray([[[1, 2], [3, -1]]], dtype=np.int32),
        "source_child_probabilities": np.asarray(
            [[[0.4, 0.2], [0.5, 0.0]]], dtype=np.float32
        ),
        "target_child_rows": np.asarray([[[[1], [3]]]], dtype=np.int32),
        "candidate_poses_w2c": np.eye(4, dtype=np.float64)[None, None],
    }


def test_oracle_intervention_changes_only_source_evidence() -> None:
    parent = _arrays()
    rows = np.asarray([[[9, -1], [8, 7]]], dtype=np.int32)
    probabilities = np.asarray([[[0.8, 0.0], [0.4, 0.3]]], dtype=np.float32)
    result = _replace_source_evidence(
        parent, source_rows=rows, source_probabilities=probabilities
    )
    assert np.array_equal(result["source_child_rows"], rows)
    assert np.array_equal(result["source_child_probabilities"], probabilities)
    for name in ("target_child_rows", "candidate_poses_w2c"):
        assert np.array_equal(result[name], parent[name])
        assert not np.shares_memory(result[name], parent[name])


def test_oracle_intervention_rejects_shape_and_capacity_drift() -> None:
    parent = _arrays()
    with pytest.raises(ValueError, match="shape differs"):
        _replace_source_evidence(
            parent,
            source_rows=np.zeros((1, 2, 1), dtype=np.int32),
            source_probabilities=np.zeros((1, 2, 1), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="exceed token capacity"):
        _replace_source_evidence(
            parent,
            source_rows=parent["source_child_rows"],
            source_probabilities=np.ones((1, 2, 2), dtype=np.float32),
        )
