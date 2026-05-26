import numpy as np
import pytest

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.token_candidate_scoring import score_candidate_bank_by_token_cosine
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)
from feature_extract.vfm.score_table import evaluate_score_table


def _record(tmp_path, image_id, vector):
    path = tmp_path / f"{image_id.replace('/', '__')}.npz"
    write_npz_token_record(path, {"radio_final": np.asarray(vector, dtype=np.float16).reshape(2, 1, 1)})
    return TokenBankRecord(
        image_id=image_id,
        token_path=path,
        layers=(TokenLayerSpec("radio_final", "c-radio_v4-h", "final", 2, 16),),
        split="test",
        scene="Synthetic",
        checksum=compute_file_sha256(path),
    )


def test_score_candidate_bank_by_token_cosine_prefers_matching_reference(tmp_path):
    query_manifest = TokenBankManifest(
        records=(_record(tmp_path, "q.png", [1.0, 0.0]),)
    )
    map_manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "good.png", [1.0, 0.0]),
            _record(tmp_path, "bad.png", [0.0, 1.0]),
        )
    )
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="synthetic",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="bad",
                candidate_type="reference_pose",
                reference_image="bad.png",
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="good",
                candidate_type="reference_pose",
                reference_image="good.png",
                pose_error=PoseCost(0.1, 2.0),
            ),
        ],
    )

    rows = score_candidate_bank_by_token_cosine(
        bank=bank,
        query_manifest=query_manifest,
        map_manifest=map_manifest,
        layer_name="radio_final",
        method="raw_radio_cosine",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
    )
    report = evaluate_score_table(rows)

    assert rows[1].score > rows[0].score
    assert report.mean_top1_acc == pytest.approx(1.0)
