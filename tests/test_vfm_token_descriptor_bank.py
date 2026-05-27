import numpy as np
import pytest

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    build_token_descriptor_bank,
    score_candidate_bank_by_descriptor_cosine,
)
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


def test_build_token_descriptor_bank_mean_pools_and_round_trips(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _record(tmp_path, "a.png", np.array([[[1.0, 3.0]], [[0.0, 0.0]]])),
            _record(tmp_path, "b.png", np.array([[[0.0, 0.0]], [[2.0, 2.0]]])),
        )
    )

    bank = build_token_descriptor_bank(manifest, layer_name="radio_final")

    assert bank.image_ids == ("a.png", "b.png")
    assert bank.descriptors.shape == (2, 2)
    assert np.linalg.norm(bank.descriptors, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert bank.get("a.png").tolist() == pytest.approx([1.0, 0.0])
    assert bank.get("b.png").tolist() == pytest.approx([0.0, 1.0])

    path = tmp_path / "descriptors.npz"
    bank.to_npz(path)
    loaded = TokenDescriptorBank.from_npz(path)

    assert loaded.image_ids == bank.image_ids
    assert loaded.layer_name == "radio_final"
    np.testing.assert_allclose(loaded.descriptors, bank.descriptors)


def test_build_token_descriptor_bank_signed_gem_handles_negative_features(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _record(
                tmp_path,
                "a.png",
                np.array(
                    [
                        [[1.0, 8.0]],
                        [[-1.0, -8.0]],
                    ]
                ),
            ),
        )
    )

    bank = build_token_descriptor_bank(
        manifest,
        layer_name="radio_final",
        pooling="gem",
        gem_power=3.0,
        normalize=False,
    )

    expected = ((np.array([1.0, 512.0], dtype=np.float32).mean()) ** (1.0 / 3.0))
    assert bank.pooling == "gem"
    assert bank.metadata["gem_power"] == 3.0
    np.testing.assert_allclose(
        bank.descriptors,
        np.asarray([[expected, -expected]], dtype=np.float32),
    )


def test_build_token_descriptor_bank_can_normalize_tokens_before_pooling(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _record(
                tmp_path,
                "a.png",
                np.array(
                    [
                        [[3.0, 0.0]],
                        [[4.0, 2.0]],
                    ]
                ),
            ),
        )
    )

    bank = build_token_descriptor_bank(
        manifest,
        layer_name="radio_final",
        pooling="mean",
        normalize_tokens=True,
        normalize=False,
    )

    np.testing.assert_allclose(
        bank.descriptors,
        np.asarray([[(3.0 / 5.0 + 0.0) / 2.0, (4.0 / 5.0 + 1.0) / 2.0]], dtype=np.float32),
    )
    assert bank.metadata["normalize_tokens"] is True


def test_score_candidate_bank_by_descriptor_cosine_prefers_matching_reference(tmp_path):
    query_bank = TokenDescriptorBank(
        image_ids=("q.png",),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
        normalized=True,
    )
    map_bank = TokenDescriptorBank(
        image_ids=("good.png", "bad.png"),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
        normalized=True,
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

    rows = score_candidate_bank_by_descriptor_cosine(
        bank=bank,
        query_descriptors=query_bank,
        map_descriptors=map_bank,
        method="raw_radio_descriptor_cosine",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
    )
    report = evaluate_score_table(rows)

    assert rows[1].score > rows[0].score
    assert report.mean_top1_acc == pytest.approx(1.0)
