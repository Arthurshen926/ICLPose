import json
import subprocess
import sys

import numpy as np

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.token_maxsim_reranking import rerank_candidate_bank_by_token_maxsim
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def _record(tmp_path, image_id, array, split="test"):
    path = tmp_path / f"{image_id.replace('/', '__')}.npz"
    write_npz_token_record(path, {"radio_final": np.asarray(array, dtype=np.float16)})
    return TokenBankRecord(
        image_id=image_id,
        token_path=path,
        layers=(TokenLayerSpec("radio_final", "c-radio_v4-h", "final", array.shape[0], 16),),
        split=split,
        scene="Synthetic",
        checksum=compute_file_sha256(path),
    )


def _candidate_bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="stage1_vlad",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="bad",
                candidate_type="descriptor_retrieval_reference_pose",
                reference_image="bad.png",
                pose_error=PoseCost(translation_m=1.0, rotation_deg=20.0),
                metadata={"retrieval_rank": 1},
            ),
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="good",
                candidate_type="descriptor_retrieval_reference_pose",
                reference_image="good.png",
                pose_error=PoseCost(translation_m=0.1, rotation_deg=2.0),
                metadata={"retrieval_rank": 2},
            ),
        ],
    )


def test_token_maxsim_reranker_promotes_patch_consistent_candidate(tmp_path):
    query_manifest = TokenBankManifest(
        records=(_record(tmp_path, "q.png", np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]])),)
    )
    map_manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "bad.png", np.asarray([[[-1.0, 0.0]], [[0.0, -1.0]]])),
            _record(tmp_path, "good.png", np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]])),
        )
    )

    reranked = rerank_candidate_bank_by_token_maxsim(
        bank=_candidate_bank(),
        query_manifest=query_manifest,
        map_manifest=map_manifest,
        layer_name="radio_final",
        top_k_per_query=2,
    )

    assert [candidate.candidate_id for candidate in reranked.candidates] == ["good", "bad"]
    assert reranked.candidates[0].prior_score > reranked.candidates[1].prior_score
    assert reranked.candidates[0].metadata["reranker"] == "token_maxsim"
    assert reranked.candidates[0].metadata["rerank_rank"] == 1


def test_token_maxsim_reranker_can_sample_tokens_deterministically(tmp_path):
    query_manifest = TokenBankManifest(
        records=(_record(tmp_path, "q.png", np.asarray([[[1.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])),)
    )
    map_manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "bad.png", np.asarray([[[0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0]]])),
            _record(tmp_path, "good.png", np.asarray([[[1.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])),
        )
    )

    reranked_a = rerank_candidate_bank_by_token_maxsim(
        bank=_candidate_bank(),
        query_manifest=query_manifest,
        map_manifest=map_manifest,
        layer_name="radio_final",
        top_k_per_query=2,
        max_tokens_per_image=2,
        seed=7,
    )
    reranked_b = rerank_candidate_bank_by_token_maxsim(
        bank=_candidate_bank(),
        query_manifest=query_manifest,
        map_manifest=map_manifest,
        layer_name="radio_final",
        top_k_per_query=2,
        max_tokens_per_image=2,
        seed=7,
    )

    assert [candidate.candidate_id for candidate in reranked_a.candidates] == [
        candidate.candidate_id for candidate in reranked_b.candidates
    ]
    assert reranked_a.candidates[0].metadata["token_maxsim_max_tokens_per_image"] == 2


def test_token_maxsim_reranker_only_scores_requested_prefix(tmp_path):
    query_manifest = TokenBankManifest(
        records=(_record(tmp_path, "q.png", np.asarray([[[1.0]], [[0.0]]])),)
    )
    map_manifest = TokenBankManifest(
        records=(_record(tmp_path, "bad.png", np.asarray([[[0.0]], [[1.0]]])),)
    )

    reranked = rerank_candidate_bank_by_token_maxsim(
        bank=_candidate_bank(),
        query_manifest=query_manifest,
        map_manifest=map_manifest,
        layer_name="radio_final",
        top_k_per_query=1,
    )

    assert [candidate.candidate_id for candidate in reranked.candidates] == ["bad"]


def test_token_maxsim_reranker_cli_writes_reranked_bank(tmp_path):
    query_manifest = TokenBankManifest(
        records=(_record(tmp_path, "q.png", np.asarray([[[1.0]], [[0.0]]])),)
    )
    map_manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "bad.png", np.asarray([[[0.0]], [[1.0]]])),
            _record(tmp_path, "good.png", np.asarray([[[1.0]], [[0.0]]])),
        )
    )
    query_manifest_path = tmp_path / "query_manifest.json"
    map_manifest_path = tmp_path / "map_manifest.json"
    bank_path = tmp_path / "bank.jsonl"
    output = tmp_path / "reranked.jsonl"
    query_manifest.to_json(query_manifest_path)
    map_manifest.to_json(map_manifest_path)
    _candidate_bank().to_jsonl(bank_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.rerank_candidate_bank_token_maxsim",
            "--candidate_bank",
            str(bank_path),
            "--query_manifest",
            str(query_manifest_path),
            "--map_manifest",
            str(map_manifest_path),
            "--layer_name",
            "radio_final",
            "--top_k_per_query",
            "2",
            "--max_tokens_per_image",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["protocol_kind"] == "reference_pose"
    assert lines[1]["candidate_id"] == "good"
