import pytest

from feature_extract.vfm.candidate_scoring import score_candidate_bank_by_metadata
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import evaluate_score_table


def test_score_candidate_bank_by_retrieval_order():
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="reference",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="q0",
                candidate_id="c0",
                candidate_type="reference_pose",
                pose_error=PoseCost(0.3, 4.0),
                metadata={"retrieval_rank": 2},
            ),
            CandidateHypothesis(
                query_id="q0",
                candidate_id="c1",
                candidate_type="reference_pose",
                pose_error=PoseCost(0.1, 2.0),
                metadata={"retrieval_rank": 1},
            ),
        ],
    )

    rows = score_candidate_bank_by_metadata(
        bank,
        method="retrieval_order",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
    )
    report = evaluate_score_table(rows)

    assert rows[1].score > rows[0].score
    assert report.mean_top1_acc == 1.0
    assert report.mean_pred_cost_m == pytest.approx(0.1)
