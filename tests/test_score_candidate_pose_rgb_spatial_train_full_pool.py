from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_train_full_pool import (
    _empty_score_rows,
    _merge_rank_rows,
    _select_scored_train_query_ids,
    load_frozen_train_full_pool_queries,
)


def _hypothesis_metadata() -> str:
    return json.dumps(
        {
            "format": "grouped_pose_hypotheses_inference_only_v1",
            "contains_target_fields": False,
            "pose_or_ground_truth_used_for_generation": False,
            "candidate_pose_evidence_version": "v1",
            "inputs": {"candidate_artifact_sha256": "candidate"},
            "grouped_config": {"latent_em_enabled": True},
        }
    )


def test_load_train_full_pool_queries_excludes_nontrain_rows(tmp_path) -> None:
    path = tmp_path / "frozen.npz"
    np.savez_compressed(
        path,
        query_ids=np.asarray(["train/a.png", "validation/b.png"]),
        split_names=np.asarray(["train", "validation"]),
        evaluation_labels=np.asarray(["frozen", "frozen"]),
        hypothesis_indices=np.asarray([7, 9], dtype=np.int64),
        poses_w2c=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
        metadata_json=np.asarray(_hypothesis_metadata()),
    )
    queries, lineage = load_frozen_train_full_pool_queries(
        hypothesis_artifacts=[path],
        evaluation_label="frozen",
        expected_train_query_ids={"train/a.png"},
    )

    assert set(queries) == {"train/a.png"}
    assert queries["train/a.png"].source_row_indices.tolist() == [0]
    assert lineage["evaluation_label"] == "frozen"


def test_load_train_full_pool_queries_rejects_coverage_drift(tmp_path) -> None:
    path = tmp_path / "frozen.npz"
    np.savez_compressed(
        path,
        query_ids=np.asarray(["train/a.png"]),
        split_names=np.asarray(["train"]),
        evaluation_labels=np.asarray(["frozen"]),
        hypothesis_indices=np.asarray([7], dtype=np.int64),
        poses_w2c=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
        metadata_json=np.asarray(_hypothesis_metadata()),
    )
    with pytest.raises(ValueError, match="coverage differs"):
        load_frozen_train_full_pool_queries(
            hypothesis_artifacts=[path],
            evaluation_label="frozen",
            expected_train_query_ids={"train/a.png", "train/missing.png"},
        )


def test_merge_rank_rows_orders_by_frozen_source_identity() -> None:
    def rows(artifact: int, source_rows: list[int]) -> dict[str, np.ndarray]:
        count = len(source_rows)
        return {
            "source_artifact_indices": np.full(count, artifact, dtype=np.int64),
            "source_row_indices": np.asarray(source_rows, dtype=np.int64),
            "query_ids": np.asarray(["q"] * count),
            "split_names": np.asarray(["train"] * count),
            "evaluation_labels": np.asarray(["frozen"] * count),
            "hypothesis_indices": np.asarray(source_rows, dtype=np.int64),
            "pose_log_likelihood_ratios": np.asarray(source_rows, dtype=np.float32),
        }

    merged = _merge_rank_rows([rows(1, [4]), rows(0, [9, 2])])
    assert merged["source_artifact_indices"].tolist() == [0, 0, 1]
    assert merged["source_row_indices"].tolist() == [2, 9, 4]


def test_empty_rank_rows_preserve_string_and_numeric_schemas() -> None:
    rows = _empty_score_rows()
    assert rows["source_artifact_indices"].dtype == np.int64
    assert rows["hypothesis_indices"].dtype == np.int64
    assert rows["pose_log_likelihood_ratios"].dtype == np.float32
    assert rows["query_ids"].dtype.kind == "U"
    assert rows["evaluation_labels"].dtype.kind == "U"


def test_limited_scoring_query_selection_is_deterministic() -> None:
    assert _select_scored_train_query_ids(
        ["train/c.png", "train/a.png", "train/b.png"], query_limit=2
    ) == ("train/a.png", "train/b.png")
    assert _select_scored_train_query_ids(
        ["train/c.png", "train/a.png"], query_limit=0
    ) == ("train/a.png", "train/c.png")
