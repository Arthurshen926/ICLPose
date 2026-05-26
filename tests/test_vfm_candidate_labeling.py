from feature_extract.vfm.candidate_labeling import label_candidate_bank_from_score_table
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind


def test_label_candidate_bank_from_score_table_matches_query_and_retrieval_rank():
    unlabeled = CandidateHypothesisBank.from_candidates(
        protocol_name="real_top20",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="q:retrieval:000",
                candidate_type="real_retrieval",
                reference_image="ref0.png",
                metadata={"retrieval_rank": 1},
            ),
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="q:retrieval:001",
                candidate_type="real_retrieval",
                reference_image="ref1.png",
                metadata={"retrieval_rank": 2},
            ),
        ],
    )
    score_table = CandidateHypothesisBank.from_candidates(
        protocol_name="score_table",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="q:score:001",
                candidate_type="score_table_candidate",
                pose_error=PoseCost(0.2, 2.0),
                prior_score=-2.0,
                metadata={"retrieval_rank": 2, "pnp_inliers": 20},
            ),
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="q:score:000",
                candidate_type="score_table_candidate",
                pose_error=PoseCost(0.1, 1.0),
                prior_score=-1.0,
                metadata={"retrieval_rank": 1, "pnp_inliers": 10},
            ),
        ],
    )

    labeled = label_candidate_bank_from_score_table(unlabeled, score_table, protocol_name="labeled_real_top20")

    assert labeled.protocol_name == "labeled_real_top20"
    assert labeled.candidates[0].reference_image == "ref0.png"
    assert labeled.candidates[0].pose_error == PoseCost(0.1, 1.0)
    assert labeled.candidates[0].prior_score == -1.0
    assert labeled.candidates[0].metadata["pnp_inliers"] == 10
    assert labeled.candidates[1].pose_error == PoseCost(0.2, 2.0)


def test_label_candidate_bank_from_score_table_skips_unmatched_candidates():
    unlabeled = CandidateHypothesisBank.from_candidates(
        protocol_name="real_top20",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="q:retrieval:000",
                candidate_type="real_retrieval",
                reference_image="ref0.png",
                metadata={"retrieval_rank": 1},
            )
        ],
    )
    score_table = CandidateHypothesisBank.from_candidates(
        protocol_name="score_table",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="q:score:001",
                candidate_type="score_table_candidate",
                pose_error=PoseCost(0.2, 2.0),
                metadata={"retrieval_rank": 2},
            )
        ],
    )

    labeled = label_candidate_bank_from_score_table(unlabeled, score_table, protocol_name="labeled")

    assert labeled.candidates == []
