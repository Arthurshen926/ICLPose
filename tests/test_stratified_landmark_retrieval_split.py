import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.build_stratified_landmark_retrieval_split import (
    SPLIT_FORMAT,
    _temporal_split_labels,
    build_stratified_landmark_retrieval_split,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _token_manifest(tmp_path: Path) -> Path:
    records = []
    for sequence in ("seq1", "seq2"):
        for frame in range(1, 11):
            image_id = f"{sequence}/frame{frame:05d}.png"
            token_path = tmp_path / f"{sequence}_{frame}.npz"
            np.savez(token_path, radio_final=np.ones((2, 1, 1), dtype=np.float32))
            records.append(
                TokenBankRecord(
                    image_id=image_id,
                    token_path=token_path,
                    layers=(TokenLayerSpec("radio_final", "radio", "final", 2, 16),),
                    split="train",
                    scene="scene",
                )
            )
    path = tmp_path / "tokens.json"
    TokenBankManifest(records=tuple(records)).to_json(path)
    return path


def _track_observations(tmp_path: Path) -> Path:
    path = tmp_path / "tracks.jsonl"
    with path.open("w") as handle:
        for sequence in ("seq1", "seq2"):
            for frame in range(1, 11):
                for track_id in range(3):
                    handle.write(
                        json.dumps(
                            {
                                "track_id": track_id,
                                "image_id": f"{sequence}/frame{frame:05d}.png",
                            }
                        )
                        + "\n"
                    )
    return path


def test_temporal_split_labels_are_balanced_and_disjoint() -> None:
    labels = _temporal_split_labels(
        15,
        train_count=9,
        validation_count=3,
        test_count=3,
    )

    assert labels.count("train") == 9
    assert labels.count("validation") == 3
    assert labels.count("test") == 3
    assert set(index for index, value in enumerate(labels) if value == "validation") == {2, 7, 12}


def test_stratified_split_balances_sequences_and_filters_all_query_observations(
    tmp_path: Path,
) -> None:
    token_manifest = _token_manifest(tmp_path)
    tracks = _track_observations(tmp_path)
    query_tokens = tmp_path / "query.json"
    support_tokens = tmp_path / "support.json"
    support_tracks = tmp_path / "support.jsonl"
    split_path = tmp_path / "split.json"

    summary = build_stratified_landmark_retrieval_split(
        track_observations_jsonl=tracks,
        token_manifest=token_manifest,
        output_support_observations_jsonl=support_tracks,
        output_query_token_manifest=query_tokens,
        output_support_token_manifest=support_tokens,
        output_query_split_json=split_path,
        train_per_sequence=2,
        validation_per_sequence=1,
        test_per_sequence=1,
        min_query_observations=2,
    )

    split = json.loads(split_path.read_text())
    assert split["format"] == SPLIT_FORMAT
    assert summary["split_counts"] == {"train": 4, "validation": 2, "test": 2}
    query_ids = split["train"] + split["validation"] + split["test"]
    assert len(query_ids) == len(set(query_ids)) == 8
    assert {value.split("/")[0] for value in split["train"]} == {"seq1", "seq2"}
    assert {value.split("/")[0] for value in split["validation"]} == {"seq1", "seq2"}
    assert {value.split("/")[0] for value in split["test"]} == {"seq1", "seq2"}
    assert [record.image_id for record in TokenBankManifest.from_json(query_tokens).records] == query_ids
    assert not set(query_ids) & {
        record.image_id for record in TokenBankManifest.from_json(support_tokens).records
    }
    remaining_observations = [json.loads(line) for line in support_tracks.read_text().splitlines()]
    assert not set(query_ids) & {str(item["image_id"]) for item in remaining_observations}
    assert summary["excluded_query_observation_count"] == 8 * 3


def test_stratified_split_rejects_sequence_with_too_few_eligible_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eligible images"):
        build_stratified_landmark_retrieval_split(
            track_observations_jsonl=_track_observations(tmp_path),
            token_manifest=_token_manifest(tmp_path),
            output_support_observations_jsonl=tmp_path / "support.jsonl",
            output_query_token_manifest=tmp_path / "query.json",
            output_support_token_manifest=tmp_path / "support.json",
            output_query_split_json=tmp_path / "split.json",
            train_per_sequence=8,
            validation_per_sequence=2,
            test_per_sequence=1,
            min_query_observations=2,
        )
