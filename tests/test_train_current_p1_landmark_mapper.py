from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_current_p1_landmark_mapper import (
    DistributedState,
    QueryGroup,
    _batched_training_groups,
    active_coherent_query_ids,
    direct_target_coverage_audit,
    split_train_only_inner_validation,
    validate_distributed_evaluation_cardinality,
)
from feature_extract.vfm.localization.current_p1_mapper_direct import (
    CurrentP1MapperDirectTargets,
    CurrentP1MapperRuntime,
)


def _runtime_and_targets() -> tuple[CurrentP1MapperRuntime, CurrentP1MapperDirectTargets]:
    runtime = CurrentP1MapperRuntime(
        source_point_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        query_ids=np.asarray(["query/a.png", "query/a.png", "query/b.png", "query/b.png"]),
        xy=np.asarray([[2.0, 2.0], [5.0, 5.0], [2.0, 2.0], [5.0, 5.0]], dtype=np.float32),
        candidate_bank_rows=np.tile(np.asarray([[0, 1, 2]], dtype=np.int64), (4, 1)),
        candidate_prior_probabilities=np.tile(
            np.asarray([[0.45, 0.30, 0.15]], dtype=np.float32), (4, 1)
        ),
        null_probabilities=np.full((4,), 0.10, dtype=np.float32),
        metadata={},
    )
    targets = CurrentP1MapperDirectTargets(
        positive_mask=np.asarray(
            [[True, False, False], [True, False, False], [True, False, False], [True, False, False]],
            dtype=bool,
        ),
        hard_negative_mask=np.asarray(
            [[False, True, False], [False, True, False], [False, False, False], [False, False, False]],
            dtype=bool,
        ),
        group_hard_mask=np.asarray([True, True, False, False], dtype=bool),
        hard_mode_ids=np.asarray([[17], [17], [-1], [-1]], dtype=np.int64),
        hard_mode_candidate_mask=np.asarray(
            [
                [[False, True, False]],
                [[False, True, False]],
                [[False, False, False]],
                [[False, False, False]],
            ],
            dtype=bool,
        ),
        metadata={},
    )
    return runtime, targets


def test_direct_target_coverage_returns_audit_for_exact_p1_modes() -> None:
    _runtime, targets = _runtime_and_targets()
    audit = direct_target_coverage_audit(targets=targets, minimum_mode_rows=2)
    assert audit["coherent_mode_count"] == 1
    assert audit["coherent_mode_active_count"] == 1
    assert audit["coherent_mode_rows_with_positive"] == 2


def test_sampler_guarantees_coherent_query_in_each_rank_batch(tmp_path: Path) -> None:
    runtime, targets = _runtime_and_targets()
    token = tmp_path / "token.npz"
    np.savez(token, radio_final=np.zeros((2, 2, 2), dtype=np.float32))
    groups = {
        "query/a.png": QueryGroup("query/a.png", np.asarray([0, 1]), token, (10, 10)),
        "query/b.png": QueryGroup("query/b.png", np.asarray([2, 3]), token, (10, 10)),
    }
    active = active_coherent_query_ids(
        runtime=runtime,
        targets=targets,
        groups_by_id=groups,
        minimum_mode_rows=2,
    )
    assert active == ["query/a.png"]
    batches = _batched_training_groups(
        list(groups.values()),
        [groups[query_id] for query_id in active],
        state=DistributedState(0, 1, 0, torch.device("cpu"), False),
        seed=7,
        batch_size=1,
        coherent_per_batch=1,
    )
    assert next(batches)[0].query_id == "query/a.png"


def test_inner_validation_split_is_query_disjoint() -> None:
    train, validation = split_train_only_inner_validation(
        ["q0", "q1", "q2", "q3", "q4"], fraction=0.2, seed=5
    )
    assert len(train) == 4
    assert len(validation) == 1
    assert not set(train).intersection(validation)


def test_distributed_evaluation_rejects_an_empty_rank_diagnostic() -> None:
    distributed = DistributedState(0, 2, 0, torch.device("cpu"), True)
    with pytest.raises(ValueError, match="at least one query per rank"):
        validate_distributed_evaluation_cardinality(query_count=1, state=distributed)
    validate_distributed_evaluation_cardinality(query_count=2, state=distributed)
