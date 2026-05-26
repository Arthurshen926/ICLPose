import numpy as np

from feature_extract.vfm.token_manifest_summary import summarize_token_manifest
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def test_summarize_token_manifest_reads_shapes_and_storage(tmp_path):
    path = tmp_path / "q0.npz"
    write_npz_token_record(path, {"radio_final": np.ones((4, 2, 2), dtype=np.float16)})
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q0",
                token_path=path,
                layers=(TokenLayerSpec("radio_final", "c-radio_v4-h", "final", 4, 16),),
                split="test",
                scene="Synthetic",
                checksum=compute_file_sha256(path),
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)

    summary = summarize_token_manifest(manifest_path)

    assert summary.record_count == 1
    assert summary.layers["radio_final"]["shape"] == (4, 2, 2)
    assert summary.layers["radio_final"]["dtype"] == "float16"
    assert summary.total_bytes > 0
