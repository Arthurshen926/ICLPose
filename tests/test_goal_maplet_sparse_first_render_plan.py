from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_sparse_first_render_plan import (
    ARTIFACT_TYPE,
    _dense_render_required,
    _load_sparse_first_plan,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
)


def test_dense_render_required_is_exact_sparse_gate() -> None:
    objective = np.asarray([[2.0, 1.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0]])
    usable = np.asarray([[1, 1], [1, 1], [1, 1], [1, 0]], bool)
    np.testing.assert_array_equal(
        _dense_render_required(objective, usable), [True, False, False, False],
    )


def test_sparse_render_plan_loader_rejects_mask_tampering(tmp_path) -> None:
    arrays = {
        "names": np.asarray(["a", "b"]),
        "dense_render_required": np.asarray([True, False]),
    }
    metadata = {
        "artifact_type": ARTIFACT_TYPE,
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": 2,
        "dense_render_query_count": 1,
        "query_pose_or_ground_truth_read": False,
        "selection_rule_changed": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    path = tmp_path / "plan.npz"
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _load_sparse_first_plan(path)
    arrays["dense_render_required"][1] = True
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="plan differs"):
        _load_sparse_first_plan(path)
