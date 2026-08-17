from __future__ import annotations

from dataclasses import replace
import numpy as np

from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    CandidatePoseFieldObservation,
    candidate_conditioned_pose_attribution,
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
