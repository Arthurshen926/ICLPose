import subprocess
import sys

import numpy as np
import pytest

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    apply_pca_whitening_transform,
    build_token_descriptor_bank,
    combine_token_descriptor_banks,
    fit_pca_whitening_transform,
    fit_vlad_codebook_from_manifests,
    load_pca_whitening_transform_npz,
    save_pca_whitening_transform_npz,
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


def test_build_token_descriptor_bank_applies_descriptor_power_norm_before_l2(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _record(
                tmp_path,
                "a.png",
                np.array(
                    [
                        [[4.0]],
                        [[1.0]],
                    ]
                ),
            ),
        )
    )

    bank = build_token_descriptor_bank(
        manifest,
        layer_name="radio_final",
        pooling="mean",
        descriptor_power=0.5,
        normalize=False,
    )

    np.testing.assert_allclose(bank.descriptors, np.asarray([[2.0, 1.0]], dtype=np.float32))
    assert bank.metadata["descriptor_power"] == 0.5


def test_pca_whitening_transform_reduces_and_round_trips(tmp_path):
    descriptors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    transform = fit_pca_whitening_transform(descriptors, output_dim=2, whitening_epsilon=1e-6)
    projected = apply_pca_whitening_transform(descriptors, transform, normalize=True)
    path = tmp_path / "pca.npz"
    save_pca_whitening_transform_npz(path, transform)
    loaded = load_pca_whitening_transform_npz(path)
    loaded_projected = apply_pca_whitening_transform(descriptors, loaded, normalize=True)

    assert projected.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), np.ones(4), atol=1e-5)
    np.testing.assert_allclose(loaded_projected, projected, atol=1e-6)


def test_fit_vlad_codebook_from_manifests_uses_mixed_token_sources(tmp_path):
    manifest_a = TokenBankManifest(
        records=(
            _record(tmp_path, "real_a.png", np.array([[[1.0, 1.1]], [[0.0, 0.0]]])),
        )
    )
    manifest_b = TokenBankManifest(
        records=(
            _record(tmp_path, "render_b.png", np.array([[[0.0, 0.0]], [[1.0, 1.1]]])),
        )
    )

    codebook = fit_vlad_codebook_from_manifests(
        (manifest_a, manifest_b),
        layer_name="radio_final",
        clusters=2,
        max_tokens=8,
        seed=0,
    )

    assert codebook.shape == (2, 2)


def test_fit_vlad_codebook_from_manifests_can_subsample_training_images_and_tokens(tmp_path):
    manifest_a = TokenBankManifest(
        records=(
            _record(tmp_path, "real_a.png", np.array([[[1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0]]])),
            _record(tmp_path, "real_b.png", np.array([[[0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0]]])),
        )
    )
    manifest_b = TokenBankManifest(
        records=(
            _record(tmp_path, "render_a.png", np.array([[[4.0, 5.0, 6.0]], [[0.0, 0.0, 0.0]]])),
            _record(tmp_path, "render_b.png", np.array([[[0.0, 0.0, 0.0]], [[4.0, 5.0, 6.0]]])),
        )
    )

    codebook = fit_vlad_codebook_from_manifests(
        (manifest_a, manifest_b),
        layer_name="radio_final",
        clusters=1,
        iterations=1,
        max_tokens=4,
        max_tokens_per_image=1,
        max_images=2,
        seed=0,
    )

    assert codebook.shape == (1, 2)


def test_combine_token_descriptor_banks_concatenates_matching_image_ids():
    bank_a = TokenDescriptorBank(
        image_ids=("a.png", "b.png"),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="vlad",
    )
    bank_b = TokenDescriptorBank(
        image_ids=("b.png", "a.png"),
        descriptors=np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="vlad",
    )

    combined = combine_token_descriptor_banks((bank_a, bank_b), mode="concat", normalize=False)

    assert combined.image_ids == ("a.png", "b.png")
    np.testing.assert_allclose(combined.descriptors, np.asarray([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]))
    assert combined.pooling == "concat:vlad+vlad"


def test_fuse_token_descriptor_banks_cli_writes_combined_bank(tmp_path):
    bank_a_path = tmp_path / "a.npz"
    bank_b_path = tmp_path / "b.npz"
    output = tmp_path / "combined.npz"
    TokenDescriptorBank(
        image_ids=("a.png",),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="vlad",
    ).to_npz(bank_a_path)
    TokenDescriptorBank(
        image_ids=("a.png",),
        descriptors=np.asarray([[0.0, 1.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="vlad",
    ).to_npz(bank_b_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.fuse_token_descriptor_banks",
            "--descriptor_banks",
            str(bank_a_path),
            str(bank_b_path),
            "--mode",
            "concat",
            "--no_normalize",
            "--output",
            str(output),
        ],
        check=True,
    )

    combined = TokenDescriptorBank.from_npz(output)
    assert combined.pooling == "concat:vlad+vlad"
    np.testing.assert_allclose(combined.descriptors, np.asarray([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32))


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
