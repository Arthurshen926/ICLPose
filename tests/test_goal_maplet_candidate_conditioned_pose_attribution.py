from __future__ import annotations

from dataclasses import replace
import numpy as np

from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    CandidatePoseTransportObservation,
    CandidatePoseFieldObservation,
    PoseTransportHierarchy,
    QueryPoseTransportObservation,
    candidate_conditioned_pose_attribution,
    pose_transport_hierarchy_content_sha256,
    reference_pose_transport_model_content_sha256,
    sparse_candidate_conditioned_pose_transport,
)
from test_goal_maplet_soft_surface_pose_energy import _retrieval


def _candidate(*, correct: bool = True, valid: bool = True):
    rows = np.asarray([[0], [1], [0], [1]], dtype=np.int64)
    code = np.zeros((4, 1, 4), dtype=np.float32)
    code[..., 0 if correct else 1] = 1.0
    normal = np.zeros((4, 1, 3), dtype=np.float32)
    normal[..., 2 if correct else 1] = 1.0
    return CandidatePoseFieldObservation(
        child_rows=rows, child_weights=np.full((4, 1), 0.9, dtype=np.float32),
        pose_codes=code, normals=normal,
        relative_depth=np.full((4, 1), 0.2 if correct else 1.5, dtype=np.float32),
        boundary=np.full((4, 1), 0.25 if correct else 1.0, dtype=np.float32),
        valid=np.full((4, 1), valid, dtype=bool),
    )


def _query():
    code = np.zeros((4, 4), dtype=np.float32)
    code[:, 0] = 1.0
    normal = np.zeros((4, 3), dtype=np.float32)
    normal[:, 2] = 1.0
    return code, normal, np.full(4, 0.2), np.full(4, 0.25), np.ones(4, dtype=bool)


def test_candidate_conditioned_pose_field_ranks_same_retrieval_by_local_measurement():
    arguments = (_retrieval(), *_query())
    correct = candidate_conditioned_pose_attribution(*arguments, _candidate(correct=True))
    wrong = candidate_conditioned_pose_attribution(*arguments, _candidate(correct=False))
    assert correct.combined_score > wrong.combined_score
    assert correct.production_eligible is False
    assert np.all(correct.unmatched_probability < wrong.unmatched_probability)


def test_missing_candidate_evidence_moves_mass_only_to_explicit_unmatched():
    arguments = (_retrieval(), *_query())
    present = candidate_conditioned_pose_attribution(*arguments, _candidate())
    missing = candidate_conditioned_pose_attribution(*arguments, _candidate(valid=False))
    np.testing.assert_array_equal(missing.unmatched_probability, np.ones(4))
    np.testing.assert_array_equal(missing.token_score, -np.ones(4))
    assert missing.combined_score < present.combined_score


def test_pose_attribution_is_invariant_to_map_slot_permutation():
    query = _retrieval()
    code, normal, depth, boundary, valid = _query()
    candidate = _candidate()
    doubled = CandidatePoseFieldObservation(
        child_rows=np.concatenate([candidate.child_rows, np.full_like(candidate.child_rows, -1)], axis=1),
        child_weights=np.concatenate([candidate.child_weights, np.zeros_like(candidate.child_weights)], axis=1),
        pose_codes=np.concatenate([candidate.pose_codes, candidate.pose_codes], axis=1),
        normals=np.concatenate([candidate.normals, candidate.normals], axis=1),
        relative_depth=np.concatenate([candidate.relative_depth, candidate.relative_depth], axis=1),
        boundary=np.concatenate([candidate.boundary, candidate.boundary], axis=1),
        valid=np.concatenate([candidate.valid, np.zeros_like(candidate.valid)], axis=1),
    )
    first = candidate_conditioned_pose_attribution(query, code, normal, depth, boundary, valid, doubled)
    permutation = [1, 0]
    second = candidate_conditioned_pose_attribution(
        query, code, normal, depth, boundary, valid,
        replace(doubled,
            child_rows=doubled.child_rows[:, permutation],
            child_weights=doubled.child_weights[:, permutation],
            pose_codes=doubled.pose_codes[:, permutation],
            normals=doubled.normals[:, permutation],
            relative_depth=doubled.relative_depth[:, permutation],
            boundary=doubled.boundary[:, permutation],
            valid=doubled.valid[:, permutation],
        ),
    )
    np.testing.assert_allclose(first.child_probabilities, second.child_probabilities)


def test_retrieval_canonical_field_cannot_masquerade_as_pose_field():
    with np.testing.assert_raises_regex(ValueError, "dedicated pose-equivariant"):
        candidate_conditioned_pose_attribution(
            _retrieval(), *_query(), replace(_candidate(), field_semantics="canonical_retrieval_field")
        )


def _hierarchy():
    parent = np.asarray([7, 7, 7])
    support = np.asarray([3, 3, 9])
    offsets = np.asarray([0, 1, 2, 2])
    adjacency = np.asarray([1, 0])
    return PoseTransportHierarchy(
        child_parent_ids=parent, child_support_ids=support,
        adjacency_offsets=offsets, adjacency_child_rows=adjacency,
        content_sha256=pose_transport_hierarchy_content_sha256(
            parent, support, offsets, adjacency
        ),
    )


def _transport_query(depth_semantics="centered_log_depth_v1", *, zero_code=False):
    code = np.zeros((4, 4), dtype=np.float32)
    if not zero_code:
        code[:, 0] = 1.0
    normal = np.zeros((4, 3), dtype=np.float32)
    normal[:, 2] = 1.0
    valid = np.ones(4, dtype=bool)
    confidence = np.ones(4, dtype=np.float32)
    return QueryPoseTransportObservation(
        query_id="q", radio_content_sha256="a" * 64,
        readout_content_sha256="b" * 64, pose_codes=code,
        normals_camera=normal, relative_depth=np.zeros(4),
        boundary=np.full(4, 0.25), pose_code_valid=valid.copy(),
        normal_valid=valid.copy(), depth_valid=valid.copy(),
        boundary_valid=valid.copy(), pose_code_confidence=confidence.copy(),
        normal_confidence=confidence.copy(), depth_confidence=confidence.copy(),
        boundary_confidence=confidence.copy(), depth_semantics=depth_semantics,
    )


def _transport_candidate(
    rows=None, depth_semantics="centered_log_depth_v1", *, valid=True,
    normal_sign=1.0,
):
    child = np.asarray([[0], [1], [0], [1]], dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    slots = child.shape[1]
    code = np.zeros((4, slots, 4), dtype=np.float32)
    code[..., 0] = 1.0
    normal = np.zeros((4, slots, 3), dtype=np.float32)
    normal[..., 2] = float(normal_sign)
    mask = np.full((4, slots), valid, dtype=bool)
    confidence = np.ones((4, slots), dtype=np.float32)
    return CandidatePoseTransportObservation(
        basin_id="b", pose_field_content_sha256="c" * 64,
        child_rows=child, child_weights=np.full(child.shape, 0.8, dtype=np.float32),
        pose_codes=code, normals_camera=normal,
        double_sided=np.zeros(child.shape, dtype=bool),
        relative_depth=np.zeros(child.shape), boundary=np.full(child.shape, 0.25),
        pose_code_valid=mask.copy(), normal_valid=mask.copy(),
        depth_valid=mask.copy(), boundary_valid=mask.copy(),
        pose_code_confidence=confidence.copy(), normal_confidence=confidence.copy(),
        depth_confidence=confidence.copy(), boundary_confidence=confidence.copy(),
        depth_semantics=depth_semantics,
    )


def test_sparse_transport_conserves_each_retrieval_source_with_unmatched_sink():
    result = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(), _transport_candidate(), _hierarchy(),
        stage="medium",
    )
    np.testing.assert_allclose(
        result.matched_source_probability + result.unmatched_source_probability,
        result.source_child_probabilities, atol=2e-7,
    )
    assert result.edge_count > 0
    assert result.production_eligible is False
    assert result.query_id == "q"
    assert result.basin_id == "b"
    assert result.transport_model_content_sha256 == reference_pose_transport_model_content_sha256(
        "medium"
    )


def test_coarse_transport_can_reattribute_same_parent_child_but_fine_cannot():
    same_parent = np.full((4, 1), 2, dtype=np.int64)
    coarse = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query("ordinal_depth_v1"),
        _transport_candidate(same_parent, "ordinal_depth_v1"), _hierarchy(),
        stage="coarse",
    )
    fine = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query("metric_log_depth_with_uncertainty_v1"),
        _transport_candidate(same_parent, "metric_log_depth_with_uncertainty_v1"),
        _hierarchy(), stage="fine",
    )
    assert coarse.edge_count > fine.edge_count
    assert coarse.combined_score > fine.combined_score


def test_invalidating_negative_modality_cannot_increase_transport_score():
    negative = _transport_candidate(normal_sign=-1.0)
    present = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(), negative, _hierarchy(), stage="medium"
    )
    missing = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(),
        replace(negative, normal_valid=np.zeros((4, 1), dtype=bool)),
        _hierarchy(), stage="medium",
    )
    assert present.combined_score >= missing.combined_score


def test_disappearing_transport_edges_cannot_improve_score():
    present = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(), _transport_candidate(), _hierarchy(),
        stage="medium",
    )
    missing = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(), _transport_candidate(valid=False),
        _hierarchy(), stage="medium",
    )
    # Invalid modalities leave hierarchy/layout edges active; removing target
    # mass removes the edges entirely and must move all source mass to sink.
    no_mass = replace(
        _transport_candidate(valid=False),
        child_weights=np.zeros((4, 1), dtype=np.float32),
    )
    absent = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(), no_mass, _hierarchy(), stage="medium",
    )
    assert present.combined_score >= missing.combined_score
    assert absent.combined_score == -1.0


def test_disappearing_any_modality_cannot_improve_score():
    query = _transport_query()
    candidate = _transport_candidate()
    present = sparse_candidate_conditioned_pose_transport(
        _retrieval(), query, candidate, _hierarchy(), stage="medium"
    )
    for field in (
        "pose_code_valid", "normal_valid", "depth_valid", "boundary_valid"
    ):
        missing = sparse_candidate_conditioned_pose_transport(
            _retrieval(), query,
            replace(candidate, **{field: np.zeros((4, 1), dtype=bool)}),
            _hierarchy(), stage="medium",
        )
        assert present.combined_score >= missing.combined_score


def test_transport_supports_different_query_and_map_slot_counts():
    candidate = _transport_candidate(
        np.asarray([[0, 2], [1, 2], [0, 2], [1, 2]], dtype=np.int64)
    )
    candidate = replace(
        candidate, child_weights=np.full((4, 2), 0.4, dtype=np.float32)
    )
    result = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(), candidate, _hierarchy(), stage="medium"
    )
    assert result.edge_count > 0


def test_zero_pose_code_is_invalidated_without_disabling_valid_normal():
    positive = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(zero_code=True), _transport_candidate(),
        _hierarchy(), stage="medium",
    )
    negative = sparse_candidate_conditioned_pose_transport(
        _retrieval(), _transport_query(zero_code=True),
        _transport_candidate(normal_sign=-1.0), _hierarchy(), stage="medium",
    )
    assert positive.combined_score > negative.combined_score


def test_sparse_transport_rejects_duplicate_child_and_schema_drift():
    duplicate = _transport_candidate(
        np.asarray([[0, 0], [1, -1], [0, -1], [1, -1]], dtype=np.int64)
    )
    with np.testing.assert_raises_regex(ValueError, "duplicate child"):
        sparse_candidate_conditioned_pose_transport(
            _retrieval(), _transport_query(), duplicate, _hierarchy(), stage="medium"
        )
    with np.testing.assert_raises_regex(ValueError, "depth semantics"):
        sparse_candidate_conditioned_pose_transport(
            _retrieval(), _transport_query("ordinal_depth_v1"),
            _transport_candidate(), _hierarchy(), stage="medium",
        )
    with np.testing.assert_raises_regex(ValueError, "camera frame"):
        sparse_candidate_conditioned_pose_transport(
            _retrieval(), replace(_transport_query(), normal_frame="world"),
            _transport_candidate(), _hierarchy(), stage="medium",
        )
    with np.testing.assert_raises_regex(ValueError, "content hash"):
        sparse_candidate_conditioned_pose_transport(
            _retrieval(), _transport_query(), _transport_candidate(),
            replace(_hierarchy(), child_support_ids=np.asarray([3, 8, 9])),
            stage="medium",
        )
