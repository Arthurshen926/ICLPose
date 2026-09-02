from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _camera_inventory
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256


def test_camera_inventory_replays_without_pose(tmp_path) -> None:
    arrays = {
        "names": np.asarray(["seq14__frame00001.png.npz"]),
        "camera_model_id": np.asarray([2], np.int32),
        "camera_width": np.asarray([1024], np.int32),
        "camera_height": np.asarray([576], np.int32),
        "camera_params": np.asarray([[880.0, 512.0, 288.0, 0.04]]),
        "source_contributor_file_sha256": np.asarray(["a" * 64]),
    }
    metadata = {
        "artifact_type": "goal_maplet_query_camera_only_inventory_v1",
        "query_count": 1,
        "pose_or_ground_truth_member_read": False,
        "source_archives_are_pose_bearing": True,
        "consumer_may_use_before_pose_freeze": True,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    path = tmp_path / "camera.npz"
    np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    rows, loaded = _camera_inventory(path)
    assert loaded == metadata
    assert rows["seq14__frame00001.png.npz"][0:3] == (2, 1024, 576)
