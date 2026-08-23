from __future__ import annotations

import numpy as np
import json

from feature_extract.tools.vfm.build_goal_maplet_pose_free_visibility_candidate_pool import (
    _select_pose_rows,
)
from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_token_inventory,
)


class _Atlas:
    def __init__(self) -> None:
        self.poses_w2c = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
        self.poses_w2c[:, 0, 3] = np.asarray([0.0, -0.2, -1.0, -2.0])


def test_global_selection_uses_pose_free_global_score_and_physical_nms():
    selected = _select_pose_rows(
        _Atlas(), np.asarray([99.0, 98.0, 97.0, 96.0]),
        np.asarray([4.0, 3.0, 2.0, 1.0]), np.zeros(4),
        semantics="global_physical_nms_v1", maximum_modes=3,
        translation_nms_m=0.5, rotation_nms_deg=5.0,
        orientations_per_location=2, location_radius_m=2.0,
        orientation_nms_deg=10.0,
    )
    assert selected.tolist() == [0, 2, 3]


def test_unknown_candidate_semantics_fail_closed():
    try:
        _select_pose_rows(
            _Atlas(), np.zeros(4), np.zeros(4), np.zeros(4), semantics="unknown",
            maximum_modes=2, translation_nms_m=0.5, rotation_nms_deg=5.0,
            orientations_per_location=2, location_radius_m=2.0,
            orientation_nms_deg=10.0,
        )
    except ValueError as error:
        assert "unsupported candidate semantics" in str(error)
    else:
        raise AssertionError("unknown candidate semantics should be rejected")


def test_token_inventory_merges_disjoint_shards_and_rejects_duplicates(tmp_path):
    token_a = tmp_path / "a.npz"
    token_b = tmp_path / "b.npz"
    token_a.write_bytes(b"a")
    token_b.write_bytes(b"b")
    shard_a = tmp_path / "shard_a.json"
    shard_b = tmp_path / "shard_b.json"
    shard_a.write_text(json.dumps({"records": [{
        "image_id": "seq13/a.png", "token_path": str(token_a),
    }]}))
    shard_b.write_text(json.dumps({"records": [{
        "image_id": "seq13/b.png", "token_path": str(token_b),
    }]}))
    merged = _load_token_inventory([shard_a, shard_b], artifact_root=tmp_path)
    assert sorted(merged) == ["seq13/a.png", "seq13/b.png"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"records": [{
        "image_id": "seq13/a.png", "token_path": str(token_b),
    }]}))
    try:
        _load_token_inventory([shard_a, duplicate], artifact_root=tmp_path)
    except ValueError as error:
        assert "duplicated" in str(error)
    else:
        raise AssertionError("duplicate token IDs across shards should be rejected")
