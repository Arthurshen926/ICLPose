from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_p1_candidate_edge_group_linear_crossfit import (
    _group_linear_batch,
    _load_feature_audit_contract,
    crossfit_group_profile,
)
from feature_extract.tools.vfm.audit_p1_candidate_edge_representation_crossfit import (
    FEATURE_SOURCES,
    CandidateEdgeProbeQueryFeatures,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatQueryTargets,
)


def _query(query_id: str, *, positive: float = 1.0) -> CandidateEdgeProbeQueryFeatures:
    point_count = 4
    candidate_count = 2
    weights = np.full((point_count, candidate_count, 2), 0.5, dtype=np.float32)
    normal: dict[str, np.ndarray] = {}
    for name in FEATURE_SOURCES:
        values = np.zeros((point_count, candidate_count, 2, 3), dtype=np.float32)
        values[:, 0, :, 0] = float(positive)
        normal[name] = values
    zero = {name: np.zeros_like(values) for name, values in normal.items()}
    masks = np.ones((point_count, candidate_count, 2), dtype=bool)
    return CandidateEdgeProbeQueryFeatures(
        query_id=query_id,
        source_point_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        candidate_view_weights=weights,
        normal_features=normal,
        permuted_features=zero,
        position_features=zero,
        normal_context_usable=masks,
        normal_rgb_usable=masks,
        normal_edge_usable=masks,
        permuted_context_usable=masks,
        permuted_rgb_usable=masks,
        permuted_edge_usable=masks,
        position_context_usable=masks,
        position_rgb_usable=masks,
        position_edge_usable=masks,
    )


def _targets(query_id: str, *, duplicate_pose: bool = False) -> HardRepeatQueryTargets:
    source = np.asarray([10, 11, 12, 13], dtype=np.int64)
    if duplicate_pose:
        source = np.tile(source, 2)
        pair_ids = np.asarray([7] * 4 + [9] * 4, dtype=np.int64)
    else:
        pair_ids = np.asarray([7] * 4, dtype=np.int64)
    return HardRepeatQueryTargets(
        query_id=query_id,
        source_point_ids=source,
        pair_ids=pair_ids,
        positive_candidate_indices=np.zeros((len(source),), dtype=np.int64),
        negative_candidate_indices=np.ones((len(source),), dtype=np.int64),
        positive_offsets_xy=np.zeros((len(source), 2), dtype=np.float32),
        negative_offsets_xy=np.zeros((len(source), 2), dtype=np.float32),
    )


def test_group_batch_keeps_dense_pose_indices_for_multi_point_groups() -> None:
    query = _query("q.png")
    batch, stats = _group_linear_batch(
        queries={query.query_id: query},
        hard_targets={query.query_id: _targets(query.query_id, duplicate_pose=True)},
        query_ids=[query.query_id],
        profile="radio_final",
        minimum_points=4,
    )
    assert batch.pose_count == 2
    assert batch.point_count == 8
    assert np.array_equal(batch.point_to_pose.numpy(), np.asarray([0] * 4 + [1] * 4))
    assert stats["train_pose_group_count"] == 2


def test_group_crossfit_scores_held_queries_with_same_exact_hard_min_metric() -> None:
    queries = {f"q{index}.png": _query(f"q{index}.png", positive=1.0 + index * 0.1) for index in range(4)}
    targets = {query_id: _targets(query_id) for query_id in queries}
    rows, fits = crossfit_group_profile(
        queries=queries,
        hard_targets=targets,
        profile="radio_final",
        fold_count=2,
        minimum_points=4,
        device="cpu",
        epochs=24,
        learning_rate=0.05,
        weight_decay=1e-3,
        softmin_temperature=0.1,
        train_margin=0.0,
        seed=7,
    )
    assert len(rows) == 4
    assert len(fits) == 2
    assert all(bool(row["eligible"]) for row in rows)
    assert all(float(row["normal_gap"]) > 0.0 for row in rows)
    assert all(abs(float(row["permuted_gap"])) < 1e-8 for row in rows)
    assert all(abs(float(row["position_gap"])) < 1e-8 for row in rows)
    assert all(int(fit["train_pose_group_count"]) == 2 for fit in fits)


def test_feature_audit_contract_rejects_target_bearing_upstream_artifact(tmp_path) -> None:
    payload = {
        "format": "p1_candidate_edge_representation_crossfit_audit_v1",
        "query_count": 1,
        "query_ids": ["q.png"],
        "feature_lineage": {
            "layout_sha256": "layout",
            "identity_checkpoint_sha256": "checkpoint",
            "candidate_count": 20,
            "support_view_count": 2,
            "feature_sources": ["radio_final"],
            "source_lineage": {},
            "visual_content_controls": {},
        },
        "protocol": {
            "diagnostic_only": True,
            "runtime_layout_target_free": True,
            "raw_feature_artifacts_contain_targets": False,
            "target_join_after_visual_inference": True,
            "runtime_scorer_must_not_load_feature_artifacts": True,
            "train_query_only": True,
            "heldout_validation_or_test_not_run": True,
            "pnp_or_pose_estimation_run": False,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload))
    loaded, lineage = _load_feature_audit_contract(path)
    assert loaded["query_ids"] == ["q.png"]
    assert lineage["layout_sha256"] == "layout"
    payload["protocol"]["raw_feature_artifacts_contain_targets"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract"):
        _load_feature_audit_contract(path)

