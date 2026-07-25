from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_bridge import (
    CandidatePoseRGBSpatialBridgeQueryEvidence,
    CandidatePoseRGBSpatialBridgeWeights,
    bridge_pose_log_likelihood_ratios,
    crossfit_bridge_profiles,
    reweight_candidate_probabilities,
)


def _evidence(query_id: str, *, spatial_sign: float = 1.0) -> CandidatePoseRGBSpatialBridgeQueryEvidence:
    # First pose row is correct.  Identity prefers candidate zero, while the
    # spatial term determines whether this query needs a positive source weight.
    spatial = np.asarray(
        [
            [[1.0 * spatial_sign, -1.0], [1.0 * spatial_sign, -1.0]],
            [[-1.0 * spatial_sign, 1.0], [-1.0 * spatial_sign, 1.0]],
        ],
        dtype=np.float64,
    )
    return CandidatePoseRGBSpatialBridgeQueryEvidence(
        query_id=query_id,
        spatial_candidate_llrs=spatial,
        control_spatial_candidate_llrs=np.zeros_like(spatial),
        identity_candidate_llrs=np.asarray([[0.5, -0.5], [0.5, -0.5]]),
        control_identity_candidate_llrs=np.zeros((2, 2)),
        candidate_probabilities=np.asarray([[0.60, 0.30], [0.60, 0.30]]),
        null_probabilities=np.asarray([0.10, 0.10]),
    )


def test_prior_reweight_preserves_explicit_null_and_zero_slots() -> None:
    candidates = np.asarray([[0.60, 0.20, 0.0], [0.0, 0.0, 0.0]])
    null = np.asarray([0.20, 1.0])
    identity = reweight_candidate_probabilities(
        candidate_probabilities=candidates,
        null_probabilities=null,
        prior_exponent=1.0,
    )
    uniform = reweight_candidate_probabilities(
        candidate_probabilities=candidates,
        null_probabilities=null,
        prior_exponent=0.0,
    )
    assert identity == pytest.approx(candidates)
    assert uniform[0] == pytest.approx([0.4, 0.4, 0.0])
    assert uniform[1] == pytest.approx([0.0, 0.0, 0.0])
    assert np.allclose(uniform.sum(axis=1) + null, 1.0)


def test_bridge_identity_weight_zero_ignores_identity_tensor() -> None:
    evidence = _evidence("q")
    first = bridge_pose_log_likelihood_ratios(
        spatial_candidate_llrs=evidence.spatial_candidate_llrs,
        identity_candidate_llrs=evidence.identity_candidate_llrs,
        candidate_probabilities=evidence.candidate_probabilities,
        null_probabilities=evidence.null_probabilities,
        weights=CandidatePoseRGBSpatialBridgeWeights(1.0, 0.0, 1.0),
    )
    second = bridge_pose_log_likelihood_ratios(
        spatial_candidate_llrs=evidence.spatial_candidate_llrs,
        identity_candidate_llrs=-100.0 * evidence.identity_candidate_llrs,
        candidate_probabilities=evidence.candidate_probabilities,
        null_probabilities=evidence.null_probabilities,
        weights=CandidatePoseRGBSpatialBridgeWeights(1.0, 0.0, 1.0),
    )
    assert first == pytest.approx(second)
    assert first[0] > first[1]


def test_crossfit_weight_selection_does_not_read_held_query_scores() -> None:
    records = [_evidence(f"q{index}") for index in range(4)]
    options = (
        CandidatePoseRGBSpatialBridgeWeights(0.0, 0.0, 1.0),
        CandidatePoseRGBSpatialBridgeWeights(1.0, 0.0, 1.0),
    )
    _, selections = crossfit_bridge_profiles(
        evidences=records, candidates=options, fold_count=2
    )
    # q0 is held in fold zero.  Changing only its target-ordered score tensor
    # cannot alter that fold's fit-time selected profile.
    changed = list(records)
    changed[0] = _evidence("q0", spatial_sign=-1.0)
    _, changed_selections = crossfit_bridge_profiles(
        evidences=changed, candidates=options, fold_count=2
    )
    assert selections[0]["weights"] == changed_selections[0]["weights"]
    assert selections[0]["fit_query_ids"] == changed_selections[0]["fit_query_ids"]
