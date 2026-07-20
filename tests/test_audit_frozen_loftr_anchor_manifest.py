from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_frozen_loftr_anchor_manifest import (
    FrozenQueryArtifact,
    assert_frozen_anchor_aligned,
)


def _artifact(*, anchor: bool, changed_field: str | None = None) -> FrozenQueryArtifact:
    count = 192
    arrays = {
        "verification_query_ids": np.full((count,), "seq1/frame00011.png"),
        "split_names": np.full((count,), "train"),
        "verification_source_row_indices": np.arange(count, dtype=np.int64),
        "verification_xy": np.zeros((count, 2), dtype=np.float32),
        "candidate_track_ids": np.tile(np.arange(20, dtype=np.int64), (count, 1)),
        "candidate_probabilities": np.full((count, 20), 0.025, dtype=np.float32),
        "null_probabilities": np.full((count,), 0.5, dtype=np.float32),
        "candidate_view_weights": np.full((count, 20, 1), 1.0, dtype=np.float32),
    }
    # Keep the synthetic posterior normalized while preserving all candidate view mass.
    arrays["candidate_probabilities"][:] = 0.025
    arrays["null_probabilities"][:] = 0.5
    if changed_field is not None:
        arrays[changed_field] = arrays[changed_field].copy()
        arrays[changed_field].flat[0] += 1
    metadata = {
        "inputs": {"appearance_artifact": {"sha256": "direct-sha"}},
    }
    return FrozenQueryArtifact(
        path="/tmp/anchor.npz" if anchor else "/tmp/direct.npz",
        sha256="anchor-sha" if anchor else "direct-sha",
        query_id="seq1/frame00011.png",
        split_name="train",
        metadata=MappingProxyType(metadata),
        arrays=MappingProxyType(arrays),
    )


def test_anchor_alignment_requires_byte_identical_immutable_s0_arrays() -> None:
    assert_frozen_anchor_aligned(direct=_artifact(anchor=False), anchor=_artifact(anchor=True))


def test_anchor_alignment_rejects_candidate_identity_change() -> None:
    with pytest.raises(ValueError, match="candidate_track_ids"):
        assert_frozen_anchor_aligned(
            direct=_artifact(anchor=False),
            anchor=_artifact(anchor=True, changed_field="candidate_track_ids"),
        )
