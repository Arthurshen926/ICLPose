from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_direct_plane_pnp_map_density import main
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256


def _write_pose(path: Path, names: list[str], candidates: list[int], inliers: list[int]) -> None:
    count = len(names)
    arrays = {
        "names": np.asarray(names),
        "pose_w2c": np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0),
        "usable": np.ones(count, bool),
        "candidate_correspondence_count": np.asarray(candidates, np.int64),
        "pnp_inlier_count": np.asarray(inliers, np.int64),
    }
    metadata = {
        "artifact_type": "goal_maplet_frozen_direct_plane_pnp_pose_inventory_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "content_sha256": path.stem,
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_camera_only_inventory_file_sha256": "camera-bytes",
    }
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_map_density_selector_uses_strictly_higher_inlier_ratio_and_ties_sparse(
    tmp_path: Path, monkeypatch,
) -> None:
    sparse = tmp_path / "sparse.npz"
    dense = tmp_path / "dense.npz"
    output = tmp_path / "selected.npz"
    _write_pose(sparse, ["a", "b"], [100, 100], [50, 50])
    _write_pose(dense, ["a", "b"], [100, 100], [60, 50])
    monkeypatch.setattr(sys, "argv", [
        "select", "--sparse_pose_inventory", str(sparse),
        "--dense_pose_inventory", str(dense), "--sparse_view_count", "240",
        "--dense_view_count", "480", "--output", str(output),
    ])
    main()
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        assert data["selected_branch"].tolist() == [480, 240]
        assert metadata["selected_dense_count"] == 1
        assert metadata["query_pose_or_ground_truth_read"] is False
