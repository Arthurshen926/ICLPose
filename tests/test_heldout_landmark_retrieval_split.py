import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_heldout_landmark_retrieval_split import (
    build_heldout_landmark_retrieval_split,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def test_heldout_split_removes_query_images_from_support(tmp_path: Path) -> None:
    records = []
    for image_id in ("q.png", "support.png"):
        token_path = tmp_path / f"{image_id}.npz"
        np.savez(token_path, radio_final=np.ones((2, 1, 1), dtype=np.float32))
        records.append(
            TokenBankRecord(
                image_id=image_id,
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "c-radio_v4-h", "final", 2, 16),),
                split="train",
                scene="scene",
            )
        )
    token_manifest = tmp_path / "tokens.json"
    TokenBankManifest(records=tuple(records)).to_json(token_manifest)
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text(
        json.dumps({"track_id": 1, "image_id": "q.png"})
        + "\n"
        + json.dumps({"track_id": 1, "image_id": "support.png"})
        + "\n"
    )
    referenced = tmp_path / "val.json"
    referenced.write_text(json.dumps({"records": [{"query_id": "q.png", "reference_image_id": "support.png"}]}))
    support_output = tmp_path / "support.jsonl"
    query_output = tmp_path / "query_tokens.json"
    support_tokens_output = tmp_path / "support_tokens.json"

    summary = build_heldout_landmark_retrieval_split(
        track_observations_jsonl=tracks,
        token_manifest=token_manifest,
        query_referenced_manifest=referenced,
        output_support_observations_jsonl=support_output,
        output_query_token_manifest=query_output,
        output_support_token_manifest=support_tokens_output,
    )

    assert summary["heldout_query_count"] == 1
    assert summary["excluded_query_observation_count"] == 1
    assert '"image_id": "q.png"' not in support_output.read_text()
    assert [record.image_id for record in TokenBankManifest.from_json(query_output).records] == ["q.png"]
    assert [record.image_id for record in TokenBankManifest.from_json(support_tokens_output).records] == ["support.png"]
    assert summary["outputs"]["support_token_manifest"]["record_count"] == 1
