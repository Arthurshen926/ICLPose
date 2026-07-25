from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_p1_candidate_edge_representation_crossfit import (
    FEATURE_SOURCES,
    CandidateEdgeProbeQueryFeatures,
    _feature_filename,
    _load_training_feature,
    _write_training_feature,
    aggregate_fixed_support_view_features,
    aggregate_hard_pose_group_gaps,
    crossfit_profile,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatQueryTargets,
)


def _feature_tensor(*, point_count: int, candidate_count: int, dimension: int, positive: float) -> np.ndarray:
    values = np.zeros((point_count, candidate_count, 2, dimension), dtype=np.float32)
    values[:, 0, :, 0] = float(positive)
    return values


def _query_features(query_id: str, *, positive: float = 1.0) -> CandidateEdgeProbeQueryFeatures:
    point_count = 4
    candidate_count = 2
    normal = {
        name: _feature_tensor(
            point_count=point_count,
            candidate_count=candidate_count,
            dimension=4 if name == "candidate_relative" else 3,
            positive=positive,
        )
        for name in FEATURE_SOURCES
    }
    zero = {name: np.zeros_like(value) for name, value in normal.items()}
    masks = np.ones((point_count, candidate_count, 2), dtype=bool)
    return CandidateEdgeProbeQueryFeatures(
        query_id=query_id,
        source_point_ids=np.asarray([10, 11, 12, 13]),
        candidate_view_weights=np.full((point_count, candidate_count, 2), 0.5, dtype=np.float32),
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


def _targets(query_id: str) -> HardRepeatQueryTargets:
    return HardRepeatQueryTargets(
        query_id=query_id,
        source_point_ids=np.asarray([10, 11, 12, 13]),
        pair_ids=np.asarray([7, 7, 7, 7]),
        positive_candidate_indices=np.zeros((4,), dtype=np.int64),
        negative_candidate_indices=np.ones((4,), dtype=np.int64),
        positive_offsets_xy=np.zeros((4, 2), dtype=np.float32),
        negative_offsets_xy=np.zeros((4, 2), dtype=np.float32),
    )


def test_fixed_support_aggregation_does_not_renormalize_missing_view_mass() -> None:
    features = np.asarray([[[[4.0], [100.0]]]], dtype=np.float32)
    weights = np.asarray([[[0.25, 0.75]]], dtype=np.float32)
    usable = np.asarray([[[True, False]]])
    aggregated, available = aggregate_fixed_support_view_features(
        features=features, candidate_view_weights=weights, usable=usable
    )
    # Only the fixed 0.25 support mass remains; the missing 0.75 mass is a
    # neutral zero and must not be reassigned to the surviving feature.
    assert aggregated == pytest.approx(np.asarray([[[1.0]]], dtype=np.float32))
    assert np.array_equal(available, np.asarray([[True]]))


def test_hard_pose_group_uses_strongest_wrong_candidate_per_point() -> None:
    targets = HardRepeatQueryTargets(
        query_id="q.png",
        source_point_ids=np.asarray([10, 10, 11, 12]),
        pair_ids=np.asarray([9, 9, 9, 9]),
        positive_candidate_indices=np.asarray([0, 0, 0, 0]),
        negative_candidate_indices=np.asarray([1, 2, 1, 1]),
        positive_offsets_xy=np.zeros((4, 2), dtype=np.float32),
        negative_offsets_xy=np.zeros((4, 2), dtype=np.float32),
    )
    normal, permuted, position, active_points = aggregate_hard_pose_group_gaps(
        normal_margins=np.asarray([0.7, -0.2, 0.8, 0.9]),
        permuted_margins=np.asarray([0.1, -0.4, 0.2, 0.3]),
        position_margins=np.zeros((4,)),
        common_active=np.ones((4,), dtype=bool),
        targets=targets,
        minimum_points=3,
    )
    # Source point 10 has two wrong identities, so the -0.2 margin must win
    # the min reduction before the coherent-pose average is formed.
    assert normal == pytest.approx(np.asarray([0.5]))
    assert permuted == pytest.approx(np.asarray([1.0 / 30.0]))
    assert position == pytest.approx(np.asarray([0.0]))
    assert active_points == 3


def test_query_grouped_crossfit_scores_held_queries_without_control_signal() -> None:
    queries = {
        f"q{index}.png": _query_features(f"q{index}.png", positive=1.0 + 0.1 * index)
        for index in range(4)
    }
    targets = {query_id: _targets(query_id) for query_id in queries}
    rows, fits = crossfit_profile(
        queries=queries,
        hard_targets=targets,
        profile="radio_final",
        fold_count=2,
        ridge_lambda=0.1,
        minimum_points=4,
    )
    assert len(rows) == 4
    assert len(fits) == 2
    assert all(row["eligible"] for row in rows)
    assert all(float(row["normal_gap"]) > 0.0 for row in rows)
    assert all(abs(float(row["permuted_gap"])) < 1e-8 for row in rows)
    assert all(abs(float(row["position_gap"])) < 1e-8 for row in rows)
    assert all(int(fit["train_query_count"]) == 2 for fit in fits)


def test_raw_feature_artifact_is_runtime_ineligible_and_lineage_bound(tmp_path) -> None:
    features = _query_features("q/a.png")
    lineage = {"layout_sha256": "layout", "identity_checkpoint_sha256": "checkpoint"}
    path = tmp_path / _feature_filename(features.query_id)
    _write_training_feature(path=path, features=features, lineage=lineage)
    loaded = _load_training_feature(
        path=path, expected_query_id=features.query_id, expected_lineage=lineage
    )
    assert loaded.query_id == features.query_id
    assert set(loaded.normal_features) == set(FEATURE_SOURCES)
    with pytest.raises(ValueError, match="contract"):
        _load_training_feature(
            path=path,
            expected_query_id=features.query_id,
            expected_lineage={"layout_sha256": "different"},
        )
