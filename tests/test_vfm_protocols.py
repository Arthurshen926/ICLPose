import pytest

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost, assign_basin_labels
from feature_extract.vfm.protocols import EvaluationProtocol, ProtocolKind, validate_no_leakage


def test_protocol_rejects_training_input_leakage():
    protocol = EvaluationProtocol(
        name="oldhospital_real_top20",
        kind=ProtocolKind.REAL_RETRIEVAL,
        split="test",
        candidate_generator="hloc_retrieval_top20",
        allowed_training_inputs=("query_tokens", "candidate_tokens", "candidate_prior"),
        candidate_uses_gt=False,
        solver_conditioned=False,
    )

    validate_no_leakage(
        protocol,
        observed_training_inputs=("query_tokens", "candidate_tokens", "candidate_prior"),
    )

    with pytest.raises(ValueError, match="leakage"):
        validate_no_leakage(
            protocol,
            observed_training_inputs=("query_tokens", "candidate_tokens", "gt_pose"),
        )


def test_protocol_fingerprint_is_stable_and_sensitive():
    protocol = EvaluationProtocol(
        name="oldhospital_real_top20",
        kind=ProtocolKind.REAL_RETRIEVAL,
        split="test",
        candidate_generator="hloc_retrieval_top20",
        allowed_training_inputs=("query_tokens", "candidate_tokens"),
        candidate_uses_gt=False,
        solver_conditioned=False,
    )
    same = EvaluationProtocol(**protocol.to_dict())
    changed = EvaluationProtocol(
        **{
            **protocol.to_dict(),
            "candidate_generator": "different_generator",
        }
    )

    assert protocol.fingerprint() == same.fingerprint()
    assert protocol.fingerprint() != changed.fingerprint()


def test_controlled_lattice_protocol_must_disclose_gt_candidate_generation():
    with pytest.raises(ValueError, match="GT-centered"):
        EvaluationProtocol(
            name="oldhospital_controlled_lattice",
            kind=ProtocolKind.CONTROLLED_LATTICE,
            split="val",
            candidate_generator="gt_centered_local_lattice",
            allowed_training_inputs=("query_tokens", "candidate_tokens"),
            candidate_uses_gt=False,
            solver_conditioned=False,
        )


def test_candidate_hypothesis_cost_and_basin_labels():
    hypotheses = [
        CandidateHypothesis(
            candidate_id="near",
            candidate_type="rendered_pose",
            pose_error=PoseCost(translation_m=0.12, rotation_deg=2.0),
            prior_score=0.6,
        ),
        CandidateHypothesis(
            candidate_id="far",
            candidate_type="reference_pose",
            pose_error=PoseCost(translation_m=1.4, rotation_deg=12.0),
            prior_score=0.8,
        ),
    ]

    labels = assign_basin_labels(hypotheses, translation_threshold_m=0.25, rotation_threshold_deg=5.0)

    assert hypotheses[0].pose_error.combined(max_translation_m=2.0, max_rotation_deg=20.0) < 0.2
    assert labels == {"near": True, "far": False}
