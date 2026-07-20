from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_frozen_multiscale_candidate_appearance import (
    _load_baseline_query_identity,
    parse_profiles,
)


def test_parse_profiles_requires_known_distinct_in_grid_profiles() -> None:
    profiles = parse_profiles(
        "final:radio_final:5;alike:alike:9",
        source_grid_sizes={"radio_final": 16, "alike": 32},
    )
    assert [(item.name, item.source_name, item.window_size) for item in profiles] == [
        ("final", "radio_final", 5),
        ("alike", "alike", 9),
    ]
    with pytest.raises(ValueError, match="unknown source"):
        parse_profiles("bad:missing:1", source_grid_sizes={"radio_final": 16})
    with pytest.raises(ValueError, match="unique"):
        parse_profiles(
            "same:radio_final:1;same:radio_final:3",
            source_grid_sizes={"radio_final": 16},
        )
    with pytest.raises(ValueError, match="exceeds"):
        parse_profiles("wide:radio_final:17", source_grid_sizes={"radio_final": 16})


def test_load_baseline_query_identity_requires_explicit_train_shard_query(tmp_path) -> None:
    path = tmp_path / "scores.npz"
    np.savez_compressed(
        path,
        query_ids=np.asarray(["train-a.png", "train-a.png", "train-b.png"]),
        split_names=np.asarray(["train", "train", "train"]),
        metadata_json=np.asarray(json.dumps({"format": "fixture"})),
    )
    with pytest.raises(ValueError, match="requires --query_id"):
        _load_baseline_query_identity(path)
    query_id, split_name, metadata = _load_baseline_query_identity(
        path, requested_query_id="train-b.png"
    )
    assert query_id == "train-b.png"
    assert split_name == "train"
    assert metadata["format"] == "fixture"
    with pytest.raises(ValueError, match="absent"):
        _load_baseline_query_identity(path, requested_query_id="validation.png")
