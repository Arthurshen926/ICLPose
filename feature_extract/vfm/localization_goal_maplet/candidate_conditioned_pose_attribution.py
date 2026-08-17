"""Candidate-conditioned soft pose attribution without point correspondences.

Retrieval child probabilities are a pose-free prior.  A basin-specific render
supplies a separate low-dimensional pose field plus normal, relative-depth and
boundary observations.  The kernel conservatively transfers prior mass to a
matched child; every unsupported or incompatible unit becomes explicit
unmatched mass.  There is no per-candidate renormalization that can reward
disappearing evidence, no hard correspondence, and no PnP interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pure_retrieval import PureRadioPhysicalRetrieval


POSE_FIELD_SEMANTICS = "candidate_conditioned_lowd_pose_equivariant_surface_field_v1"


@dataclass(frozen=True)
class CandidatePoseFieldObservation:
    child_rows: np.ndarray
    child_weights: np.ndarray
    pose_codes: np.ndarray
    normals: np.ndarray
    relative_depth: np.ndarray
    boundary: np.ndarray
    valid: np.ndarray
    field_semantics: str = POSE_FIELD_SEMANTICS


@dataclass(frozen=True)
class CandidateConditionedPoseAttribution:
    child_rows: np.ndarray
    child_probabilities: np.ndarray
    unmatched_probability: np.ndarray
    token_matched_probability: np.ndarray
    token_score: np.ndarray
    combined_score: float
    field_semantics: str
    production_eligible: bool


def _unit(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def candidate_conditioned_pose_attribution(
    retrieval: PureRadioPhysicalRetrieval,
    query_pose_codes: np.ndarray,
    query_normals: np.ndarray,
    query_relative_depth: np.ndarray,
    query_boundary: np.ndarray,
    query_valid: np.ndarray,
    candidate: CandidatePoseFieldObservation,
    *,
    depth_scale: float = 0.25,
    feature_power: float = 1.0,
    normal_power: float = 0.5,
    depth_power: float = 0.5,
    boundary_power: float = 0.25,
) -> CandidateConditionedPoseAttribution:
    """Convert retrieval prior into a conservative basin-conditioned posterior.

    All compatibility factors lie in ``[0,1]``.  For query child ``c`` the
    matched probability is ``q_ret(c) * sum_l map_mass(l) * compatibility(l)``
    over slots with the same child.  The remainder of unit mass is unmatched.
    Thus removing map support or invalidating either side cannot improve the
    token score ``2 * matched - 1``.
    """

    if str(candidate.field_semantics) != POSE_FIELD_SEMANTICS:
        raise ValueError("candidate observation is not the dedicated pose-equivariant field")
    rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    prior = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    token_count, query_slots = rows.shape
    query_code = np.asarray(query_pose_codes, dtype=np.float64)
    query_normal = np.asarray(query_normals, dtype=np.float64)
    query_depth = np.asarray(query_relative_depth, dtype=np.float64).reshape(-1)
    query_edge = np.asarray(query_boundary, dtype=np.float64).reshape(-1)
    query_mask = np.asarray(query_valid, dtype=bool).reshape(-1)
    map_rows = np.asarray(candidate.child_rows, dtype=np.int64)
    map_mass = np.asarray(candidate.child_weights, dtype=np.float64)
    map_code = np.asarray(candidate.pose_codes, dtype=np.float64)
    map_normal = np.asarray(candidate.normals, dtype=np.float64)
    map_depth = np.asarray(candidate.relative_depth, dtype=np.float64)
    map_edge = np.asarray(candidate.boundary, dtype=np.float64)
    map_valid = np.asarray(candidate.valid, dtype=bool)
    if query_code.ndim != 2:
        raise ValueError("query pose code must have shape [token,dimension]")
    dimension = int(query_code.shape[1])
    if (
        prior.shape != rows.shape or query_code.shape[0] != token_count
        or query_normal.shape != (token_count, 3)
        or query_depth.shape != (token_count,) or query_edge.shape != (token_count,)
        or query_mask.shape != (token_count,) or map_rows.ndim != 2
        or map_rows.shape[0] != token_count or map_mass.shape != map_rows.shape
        or map_code.shape != map_rows.shape + (dimension,)
        or map_normal.shape != map_rows.shape + (3,)
        or map_depth.shape != map_rows.shape or map_edge.shape != map_rows.shape
        or map_valid.shape != map_rows.shape
    ):
        raise ValueError("candidate-conditioned pose observation arrays differ")
    powers = np.asarray(
        [feature_power, normal_power, depth_power, boundary_power], dtype=np.float64
    )
    if (
        dimension < 2 or dimension > 64 or np.any(~np.isfinite(powers))
        or np.any(powers < 0.0) or float(np.sum(powers)) <= 0.0
        or not np.isfinite(depth_scale) or float(depth_scale) <= 0.0
        or any(np.any(~np.isfinite(value)) for value in (
            prior, query_code, query_normal, query_depth, query_edge,
            map_mass, map_code, map_normal, map_depth, map_edge,
        ))
        or np.any(prior < 0.0) or np.any(map_mass < 0.0)
        or np.any(np.sum(prior, axis=1) > 1.0 + 2e-5)
        or np.any(np.sum(map_mass, axis=1) > 1.0 + 2e-5)
        or np.any((query_edge < 0.0) | (query_edge > 1.0))
        or np.any((map_edge < 0.0) | (map_edge > 1.0))
    ):
        raise ValueError("invalid candidate-conditioned pose evidence")
    q_code = _unit(query_code)
    m_code = _unit(map_code)
    q_normal = _unit(query_normal)
    m_normal = _unit(map_normal)
    feature = np.clip(
        (np.einsum("td,tld->tl", q_code, m_code) + 1.0) * 0.5, 0.0, 1.0
    )
    normal = np.clip(
        (np.einsum("td,tld->tl", q_normal, m_normal) + 1.0) * 0.5, 0.0, 1.0
    )
    depth = np.exp(-np.abs(query_depth[:, None] - map_depth) / float(depth_scale))
    boundary = np.clip(1.0 - np.abs(query_edge[:, None] - map_edge), 0.0, 1.0)
    components = np.stack([feature, normal, depth, boundary], axis=-1)
    compatibility = np.exp(
        np.sum(powers * np.log(np.maximum(components, 1e-12)), axis=-1)
        / float(np.sum(powers))
    )
    compatibility *= query_mask[:, None] * map_valid
    match = rows[:, :, None] == map_rows[:, None, :]
    match &= (rows[:, :, None] >= 0) & (map_rows[:, None, :] >= 0)
    transferred = prior[:, :, None] * map_mass[:, None, :] * compatibility[:, None, :] * match
    child_probability = np.sum(transferred, axis=2)
    matched = np.clip(np.sum(child_probability, axis=1), 0.0, 1.0)
    unmatched = np.clip(1.0 - matched, 0.0, 1.0)
    token_score = 2.0 * matched - 1.0
    return CandidateConditionedPoseAttribution(
        child_rows=rows.copy(),
        child_probabilities=child_probability.astype(np.float32),
        unmatched_probability=unmatched.astype(np.float32),
        token_matched_probability=matched.astype(np.float32),
        token_score=token_score.astype(np.float32),
        combined_score=float(np.mean(token_score)),
        field_semantics=POSE_FIELD_SEMANTICS,
        # This is the frozen mathematical/reference seam.  A trained OOF
        # query readout and map pose-field artifact are still required.
        production_eligible=False,
    )
