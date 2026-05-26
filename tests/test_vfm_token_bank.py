import numpy as np
import pytest

from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def test_token_record_manifest_roundtrip_with_checksum(tmp_path):
    token_path = tmp_path / "q0.npz"
    write_npz_token_record(
        token_path,
        {
            "radio_final": np.ones((4, 2, 2), dtype=np.float32),
            "dinov2_final": np.zeros((3, 2, 2), dtype=np.float32),
        },
    )
    record = TokenBankRecord(
        image_id="q0",
        token_path=token_path,
        layers=(
            TokenLayerSpec("radio_final", "c-radio_v4-h", "final", 4, 16),
            TokenLayerSpec("dinov2_final", "dinov2", "final", 3, 14),
        ),
        split="test",
        scene="OldHospital",
        checksum=compute_file_sha256(token_path),
    )
    manifest = TokenBankManifest(records=(record,))
    path = tmp_path / "manifest.json"

    manifest.validate()
    manifest.to_json(path)
    loaded = TokenBankManifest.from_json(path)

    assert loaded.records[0] == record


def test_token_manifest_rejects_bad_layer_and_checksum(tmp_path):
    token_path = tmp_path / "q0.npz"
    write_npz_token_record(token_path, {"radio_final": np.ones((1, 1, 1), dtype=np.float32)})
    record = TokenBankRecord(
        image_id="q0",
        token_path=token_path,
        layers=(TokenLayerSpec("bad", "radio", "final", 0, 16),),
        split="test",
        scene="OldHospital",
        checksum="bad",
    )

    with pytest.raises(ValueError, match="invalid token layer"):
        record.require_raw_tokens()

    fixed_record = TokenBankRecord(
        image_id="q0",
        token_path=token_path,
        layers=(TokenLayerSpec("radio_final", "radio", "final", 1, 16),),
        split="test",
        scene="OldHospital",
        checksum="bad",
    )
    with pytest.raises(ValueError, match="checksum"):
        TokenBankManifest(records=(fixed_record,)).validate()

    TokenBankManifest(records=(fixed_record,)).validate(verify_checksums=False)
