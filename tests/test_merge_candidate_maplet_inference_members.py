from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.merge_candidate_maplet_inference_members import (
    _finite_mean,
    _merge_members,
)


def _member(checkpoint_hash: str, shift: float = 0.0):
    summary = {
        "protocol": {"inference_only": True},
        "data_manifest": {"schema": "same"},
        "query_set": {"query_point_count": 2, "candidate_top_k": 2},
        "model_config": {"hidden_dim": 8},
        "checkpoints": [{"sha256": checkpoint_hash}],
        "radio_projection_lineages": [{"mode": "same_cache"}],
    }
    values = np.asarray([[0.1, -np.inf], [0.3, 0.4]], dtype=np.float32) + shift
    scores = {
        "member_0__probability": values.copy(),
        "ensemble__probability": values.copy(),
        "baseline_scores": np.asarray([[2.0, -np.inf], [0.2, 0.1]], dtype=np.float32),
        "selected_columns": np.asarray([0, 0], dtype=np.int64),
    }
    return summary, scores


def test_finite_mean_preserves_missing_candidate_mask() -> None:
    result = _finite_mean(
        [
            np.asarray([[0.2, -np.inf]], dtype=np.float32),
            np.asarray([[0.4, -np.inf]], dtype=np.float32),
        ]
    )
    np.testing.assert_allclose(result[:, :1], [[0.3]], atol=1e-6)
    assert np.isneginf(result[0, 1])


def test_merge_members_exports_distinct_members_and_mean() -> None:
    arrays, summary = _merge_members([_member("a"), _member("b", 0.2)])

    np.testing.assert_allclose(
        arrays["ensemble__probability"],
        np.asarray([[0.2, -np.inf], [0.4, 0.5]], dtype=np.float32),
        atol=1e-6,
    )
    assert summary["member_score_prefixes"] == ["member_0", "member_1"]
    assert [row["sha256"] for row in summary["checkpoints"]] == ["a", "b"]


def test_merge_members_rejects_contract_or_baseline_mismatch() -> None:
    first = _member("a")
    contract = _member("b")
    contract[0]["model_config"] = {"hidden_dim": 16}
    with pytest.raises(ValueError, match="contract differs"):
        _merge_members([first, contract])

    baseline = _member("b")
    baseline[1]["baseline_scores"][0, 0] = 9.0
    with pytest.raises(ValueError, match="baseline scores differ"):
        _merge_members([first, baseline])
