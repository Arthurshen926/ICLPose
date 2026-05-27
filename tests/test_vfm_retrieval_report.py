import json
import subprocess
import sys

import pytest

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.retrieval_report import summarize_descriptor_retrieval_bank


def _bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="raw_vfm_retrieval",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                candidate_id="q0:1",
                candidate_type="descriptor_retrieval_reference_pose",
                query_id="q0.png",
                pose_error=PoseCost(0.30, 2.0),
                metadata={"retrieval_rank": 1},
            ),
            CandidateHypothesis(
                candidate_id="q0:2",
                candidate_type="descriptor_retrieval_reference_pose",
                query_id="q0.png",
                pose_error=PoseCost(0.05, 1.0),
                metadata={"retrieval_rank": 2},
            ),
            CandidateHypothesis(
                candidate_id="q1:1",
                candidate_type="descriptor_retrieval_reference_pose",
                query_id="q1.png",
                pose_error=PoseCost(2.00, 30.0),
                metadata={"retrieval_rank": 1},
            ),
            CandidateHypothesis(
                candidate_id="q1:2",
                candidate_type="descriptor_retrieval_reference_pose",
                query_id="q1.png",
                pose_error=PoseCost(1.00, 5.0),
                metadata={"retrieval_rank": 2},
            ),
        ],
    )


def test_summarize_descriptor_retrieval_bank_reports_recall_at_k():
    report = summarize_descriptor_retrieval_bank(
        _bank(),
        thresholds=((0.25, 10.0), (1.0, 10.0)),
        recall_at=(1, 2),
    )

    assert report["query_count"] == 2
    assert report["median_top1_translation_m"] == pytest.approx(1.15)
    assert report["median_oracle_translation_m"] == pytest.approx(0.525)
    assert report["thresholds"]["0.25m_10deg"]["recall_at_1"] == pytest.approx(0.0)
    assert report["thresholds"]["0.25m_10deg"]["recall_at_2"] == pytest.approx(0.5)
    assert report["thresholds"]["1m_10deg"]["recall_at_2"] == pytest.approx(1.0)


def test_report_descriptor_retrieval_cli_writes_json(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    output = tmp_path / "report.json"
    _bank().to_jsonl(bank_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.report_descriptor_retrieval",
            "--candidate_bank",
            str(bank_path),
            "--threshold",
            "0.25,10",
            "--recall_at",
            "1",
            "2",
            "--output_json",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["thresholds"]["0.25m_10deg"]["recall_at_2"] == pytest.approx(0.5)
