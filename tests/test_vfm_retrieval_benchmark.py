import json
import subprocess
import sys

import pytest

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.retrieval_benchmark import summarize_reference_pose_retrieval
from feature_extract.vfm.retrieval_fusion import fuse_reference_pose_banks


def _candidate(query_id, rank, translation_m, rotation_deg=0.0, score=None):
    return CandidateHypothesis(
        query_id=query_id,
        candidate_id=f"{query_id}:c{rank}",
        candidate_type="descriptor_retrieval_reference_pose",
        reference_image=f"r{rank}.png",
        pose_error=PoseCost(translation_m=translation_m, rotation_deg=rotation_deg),
        prior_score=score,
        metadata={"retrieval_rank": rank},
    )


def _source_candidate(query_id, source, reference_image, rank, score, translation_m):
    return CandidateHypothesis(
        query_id=query_id,
        candidate_id=f"{query_id}:{source}:{rank}",
        candidate_type=f"{source}_reference_pose",
        reference_image=reference_image,
        pose_error=PoseCost(translation_m=translation_m, rotation_deg=0.0),
        prior_score=score,
        metadata={"retrieval_rank": rank, "source": source},
    )


def test_reference_pose_retrieval_summary_reports_recall_and_topk_pose_cost() -> None:
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="radio_vpr",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            _candidate("q0.png", 1, 2.0, 20.0),
            _candidate("q0.png", 2, 0.2, 2.0),
            _candidate("q1.png", 1, 0.8, 1.0),
            _candidate("q1.png", 2, 0.6, 1.0),
        ],
    )

    summary = summarize_reference_pose_retrieval(
        bank,
        top_ks=(1, 2),
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        rot_cost_weight=0.1,
    )

    assert summary["query_count"] == 2
    assert summary["candidate_count"] == 4
    assert summary["recall_at_1_25cm_5deg"] == 0.0
    assert summary["recall_at_2_25cm_5deg"] == 0.5
    assert summary["top1_median_translation_m"] == pytest.approx(1.4)
    assert summary["top2_best_median_translation_m"] == pytest.approx(0.4)
    assert summary["top2_best_median_pose_cost_m"] == pytest.approx(0.55)


def test_reference_pose_retrieval_benchmark_cli_writes_summary(tmp_path) -> None:
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="radio_vpr",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            _candidate("q0.png", 1, 1.0, 10.0),
            _candidate("q0.png", 2, 0.1, 1.0),
        ],
    )
    bank_path = tmp_path / "bank.jsonl"
    output = tmp_path / "summary.json"
    bank.to_jsonl(bank_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.benchmark_reference_pose_retrieval",
            "--candidate_bank",
            str(bank_path),
            "--top_ks",
            "1,2",
            "--translation_threshold_m",
            "0.25",
            "--rotation_threshold_deg",
            "5",
            "--output",
            str(output),
        ],
        check=True,
    )

    summary = json.loads(output.read_text())
    assert summary["recall_at_1_25cm_5deg"] == 0.0
    assert summary["recall_at_2_25cm_5deg"] == 1.0


def test_reference_pose_retrieval_benchmark_cli_supports_decimeter_preset(tmp_path) -> None:
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="radio_vpr",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            _candidate("q0.png", 1, 0.3, 4.0),
            _candidate("q0.png", 2, 0.2, 4.0),
        ],
    )
    bank_path = tmp_path / "bank.jsonl"
    output = tmp_path / "summary.json"
    bank.to_jsonl(bank_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.benchmark_reference_pose_retrieval",
            "--candidate_bank",
            str(bank_path),
            "--top_ks",
            "1,2",
            "--target_preset",
            "decimeter_coarse",
            "--output",
            str(output),
        ],
        check=True,
    )

    summary = json.loads(output.read_text())
    assert summary["translation_threshold_m"] == 0.25
    assert summary["rotation_threshold_deg"] == 5.0
    assert summary["recall_at_1_25cm_5deg"] == 0.0
    assert summary["recall_at_2_25cm_5deg"] == 1.0


def test_reference_pose_rrf_fusion_reranks_without_pose_cost() -> None:
    bank_a = CandidateHypothesisBank.from_candidates(
        protocol_name="hloc_netvlad",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            _source_candidate("q0.png", "hloc", "bad.png", 1, 0.9, 5.0),
            _source_candidate("q0.png", "hloc", "good.png", 2, 0.8, 0.1),
        ],
    )
    bank_b = CandidateHypothesisBank.from_candidates(
        protocol_name="radio_gem",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            _source_candidate("q0.png", "radio", "good.png", 1, 0.7, 0.1),
            _source_candidate("q0.png", "radio", "bad.png", 5, 0.6, 5.0),
        ],
    )

    fused = fuse_reference_pose_banks([bank_a, bank_b], protocol_name="rrf_vpr", top_k=2, rrf_k=10)

    assert fused.protocol_kind == ProtocolKind.REFERENCE_POSE
    assert [candidate.reference_image for candidate in fused.candidates] == ["good.png", "bad.png"]
    assert [candidate.metadata["retrieval_rank"] for candidate in fused.candidates] == [1, 2]
    assert fused.candidates[0].candidate_type == "rrf_reference_pose"
    assert fused.candidates[0].metadata["candidate_generator"] == "reference_pose_rrf_fusion"
    assert fused.candidates[0].metadata["candidate_uses_gt"] is False
    assert fused.candidates[0].metadata["source_ranks"] == {"hloc_netvlad": 2, "radio_gem": 1}
    assert fused.candidates[0].prior_score > fused.candidates[1].prior_score


def test_reference_pose_rrf_fusion_cli_writes_bank(tmp_path) -> None:
    bank_a = CandidateHypothesisBank.from_candidates(
        protocol_name="hloc_netvlad",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            _source_candidate("q0.png", "hloc", "bad.png", 1, 0.9, 5.0),
            _source_candidate("q0.png", "hloc", "good.png", 2, 0.8, 0.1),
        ],
    )
    bank_b = CandidateHypothesisBank.from_candidates(
        protocol_name="radio_gem",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[_source_candidate("q0.png", "radio", "good.png", 1, 0.7, 0.1)],
    )
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    output = tmp_path / "fused.jsonl"
    bank_a.to_jsonl(path_a)
    bank_b.to_jsonl(path_b)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.fuse_reference_pose_banks",
            "--candidate_banks",
            str(path_a),
            str(path_b),
            "--protocol_name",
            "rrf_vpr",
            "--top_k",
            "2",
            "--rrf_k",
            "10",
            "--output",
            str(output),
        ],
        check=True,
    )

    fused = CandidateHypothesisBank.from_jsonl(output)
    assert [candidate.reference_image for candidate in fused.candidates] == ["good.png", "bad.png"]
