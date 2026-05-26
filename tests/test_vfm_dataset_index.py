import numpy as np
import pytest

from feature_extract.vfm.datasets import build_token_candidate_index
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def _manifest(tmp_path):
    token_path = tmp_path / "q0.npz"
    write_npz_token_record(token_path, {"feat": np.ones((2, 1, 1), dtype=np.float32)})
    layer = TokenLayerSpec("feat", "synthetic", "final", 2, 16)
    return TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q0",
                token_path=token_path,
                layers=(layer,),
                split="train",
                scene="Synthetic",
                checksum=compute_file_sha256(token_path),
            ),
        )
    )


def test_token_candidate_index_groups_by_query(tmp_path):
    manifest = _manifest(tmp_path)
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="train",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="q0",
                candidate_id="c0",
                candidate_type="reference_pose",
                pose_error=PoseCost(0.1, 2.0),
            ),
            CandidateHypothesis(
                query_id="q0",
                candidate_id="c1",
                candidate_type="reference_pose",
                pose_error=PoseCost(0.5, 6.0),
            ),
        ],
    )

    index = build_token_candidate_index(manifest, bank, require_pose_labels=True)

    assert index.query_count == 1
    assert index.examples[0].query_id == "q0"
    assert len(index.examples[0].candidates) == 2


def test_token_candidate_index_rejects_missing_query_id(tmp_path):
    manifest = _manifest(tmp_path)
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="train",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[CandidateHypothesis(candidate_id="c0", candidate_type="reference_pose")],
    )

    with pytest.raises(ValueError, match="query_id"):
        build_token_candidate_index(manifest, bank)
