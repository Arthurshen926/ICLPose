import numpy as np
import pytest

from feature_extract.vfm.configs import load_protocol_config
from feature_extract.vfm.hypothesis_io import (
    CandidateHypothesisBank,
    candidate_from_dict,
    candidate_to_dict,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.metrics import (
    calibration_ece,
    catastrophic_failure_rate,
    kendall_tau,
    risk_coverage_auc,
)
from feature_extract.vfm.protocols import ProtocolKind


def test_protocol_yaml_loader_validates_vfm_configs():
    protocol = load_protocol_config("feature_extract/configs/vfm/protocol/map_conditioned_verifier.yaml")

    assert protocol.kind == ProtocolKind.REAL_RETRIEVAL
    assert protocol.candidate_uses_gt is False
    assert "query_tokens" in protocol.allowed_training_inputs
    assert "pose_error" not in protocol.allowed_training_inputs


def test_protocol_yaml_loader_rejects_undisclosed_controlled_lattice(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
protocol:
  name: bad
  kind: controlled_lattice
  split: val
  candidate_generator: local_lattice
  candidate_uses_gt: false
  solver_conditioned: false
  allowed_training_inputs: [query_tokens]
"""
    )

    with pytest.raises(ValueError, match="GT-centered"):
        load_protocol_config(path)


def test_candidate_bank_jsonl_roundtrip_allows_inference_without_gt_cost(tmp_path):
    bank = CandidateHypothesisBank(
        protocol_name="real_top20",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        protocol_fingerprint="abc123",
        candidates=[
            CandidateHypothesis(
                candidate_id="c0",
                candidate_type="reference_pose",
                pose_error=None,
                prior_score=0.7,
                reference_image="seq1/frame0001.png",
            ),
            CandidateHypothesis(
                candidate_id="c1",
                candidate_type="rendered_pose",
                pose_error=PoseCost(0.12, 3.0),
                prior_score=0.4,
                hard_case_type="near_identity_false_positive",
            ),
        ],
    )
    path = tmp_path / "bank.jsonl"

    bank.to_jsonl(path)
    loaded = CandidateHypothesisBank.from_jsonl(path)

    assert loaded.protocol_kind == ProtocolKind.REAL_RETRIEVAL
    assert loaded.protocol_fingerprint == "abc123"
    assert loaded.candidates[0].pose_error is None
    assert loaded.candidates[1].basin_label(0.25, 5.0) is True
    assert candidate_from_dict(candidate_to_dict(loaded.candidates[1])) == loaded.candidates[1]


def test_calibration_and_risk_metrics_have_expected_direction():
    scores = np.array([0.95, 0.80, 0.30, 0.10], dtype=np.float32)
    success = np.array([True, True, False, False])
    risks = 1.0 - scores
    costs = np.array([0.05, 0.08, 2.0, 0.40], dtype=np.float32)

    assert kendall_tau(scores, -costs) > 0.6
    assert calibration_ece(scores, success, bins=2) < 0.2
    assert risk_coverage_auc(risks, success) > 0.8
    assert catastrophic_failure_rate(costs, threshold_m=1.0) == pytest.approx(0.25)
