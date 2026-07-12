from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.infer_candidate_maplet_matcher import (
    _validate_radio_projection_lineage,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _metadata_cache(path: Path, metadata: dict[str, object]) -> None:
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


def test_radio_lineage_accepts_same_training_cache(tmp_path: Path) -> None:
    path = tmp_path / "radio.npz"
    _metadata_cache(path, {"format": "radio_intermediate_observation_context_cache_v1"})
    digest = file_sha256_short(path)

    lineage = _validate_radio_projection_lineage(
        {"radio_intermediate_cache_sha256": digest}, path
    )

    assert lineage["mode"] == "same_cache"


def test_radio_lineage_requires_explicit_training_projection_source(tmp_path: Path) -> None:
    path = tmp_path / "external.npz"
    _metadata_cache(
        path,
        {
            "projection_source_cache_sha256": "training-hash",
            "support_descriptor_source": "reused_projection_source_cache",
        },
    )

    lineage = _validate_radio_projection_lineage(
        {"radio_intermediate_cache_sha256": "training-hash"}, path
    )
    assert lineage["mode"] == "query_only_cache_with_training_projection_lineage"

    with pytest.raises(ValueError, match="projection_source_cache"):
        _validate_radio_projection_lineage(
            {"radio_intermediate_cache_sha256": "other-training-hash"}, path
        )
