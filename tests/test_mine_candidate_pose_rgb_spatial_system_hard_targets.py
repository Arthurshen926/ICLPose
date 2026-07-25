from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.mine_candidate_pose_rgb_spatial_system_hard_targets import (
    _repeat_selected_query_ids,
    _validate_checkpoint_partition_for_mining,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    current_inner_gate_evaluator_manifest,
    train_query_partition_manifest,
)


def test_repeat_selected_query_ids_preserves_full_paths() -> None:
    rows = [
        {"query_id": "seq1/frame00031.png", "selected_hard_edge_count": 2},
        {"query_id": "seq10/frame00191.png", "selected_hard_edge_count": 1},
    ]
    query_ids = _repeat_selected_query_ids(rows)
    assert query_ids.tolist() == [
        "seq1/frame00031.png",
        "seq1/frame00031.png",
        "seq10/frame00191.png",
    ]
    assert query_ids.dtype.itemsize >= len("seq10/frame00191.png") * np.dtype("U1").itemsize


def test_current_system_miner_requires_gate_approved_exact_fold_partition() -> None:
    expected = train_query_partition_manifest(
        all_query_ids=("q/a.png", "q/b.png"),
        inner_train_query_ids=("q/a.png",),
        inner_validation_query_ids=("q/b.png",),
        fold_count=2,
        fold_index=1,
    )
    metadata = {
        "train_only_inner_gate_passed": True,
        "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
        "training": {"inner_validation": {"query_partition": expected}},
    }
    assert _validate_checkpoint_partition_for_mining(
        checkpoint_metadata=metadata,
        expected_partition=expected,
    ) == expected

    with pytest.raises(ValueError, match="did not pass"):
        _validate_checkpoint_partition_for_mining(
            checkpoint_metadata={**metadata, "train_only_inner_gate_passed": False},
            expected_partition=expected,
        )

    with pytest.raises(ValueError, match="manifest is stale or missing"):
        _validate_checkpoint_partition_for_mining(
            checkpoint_metadata={
                **metadata,
                "inner_gate_evaluator_manifest": {"version": "legacy"},
            },
            expected_partition=expected,
        )

    other_fold = train_query_partition_manifest(
        all_query_ids=("q/a.png", "q/b.png"),
        inner_train_query_ids=("q/b.png",),
        inner_validation_query_ids=("q/a.png",),
        fold_count=2,
        fold_index=0,
    )
    with pytest.raises(ValueError, match="different inner validation fold"):
        _validate_checkpoint_partition_for_mining(
            checkpoint_metadata={
                "train_only_inner_gate_passed": True,
                "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
                "training": {"inner_validation": {"query_partition": other_fold}},
            },
            expected_partition=expected,
        )
