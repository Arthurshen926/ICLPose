from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus import (
    OUTPUT_ARTIFACT,
    _load_selected,
    _normal_good_ray_score,
    _select_geometry_consensus,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
)


def test_dense_normal_score_counts_missing_render_as_failure() -> None:
    assert _normal_good_ray_score({
        "query_valid_pixel_count": 100,
        "common_valid_pixel_count": 50,
        "normal_within_20deg": 0.8,
    }) == 0.4


def test_dense_normal_score_can_use_observed_plane_domain() -> None:
    row = {
        "query_valid_pixel_count": 100,
        "common_valid_pixel_count": 50,
        "normal_within_20deg": 0.8,
        "planar_query_valid_pixel_count": 40,
        "planar_common_valid_pixel_count": 30,
        "planar_normal_within_20deg": 0.6,
    }
    assert _normal_good_ray_score(row) == 0.4
    assert _normal_good_ray_score(row, "planar_") == 0.45
    with pytest.raises(ValueError, match="domain differs"):
        _normal_good_ray_score(row, "other_")


def test_geometry_consensus_requires_both_sparse_and_dense_improvement() -> None:
    objective = np.asarray([[2.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    normal = np.asarray([[0.4, 0.5], [0.5, 0.4], [0.4, 0.5]])
    usable = np.ones((3, 2), bool)
    np.testing.assert_array_equal(
        _select_geometry_consensus(objective, normal, usable), [1, 0, 0],
    )
    usable[2, 0] = False
    np.testing.assert_array_equal(
        _select_geometry_consensus(objective, normal, usable), [1, 0, 1],
    )


def test_selected_pose_loader_rejects_score_tampering(tmp_path) -> None:
    arrays = {
        "names": np.asarray(["q"]),
        "pose_w2c": np.eye(4)[None],
        "usable": np.ones(1, bool),
        "selected_branch": np.zeros(1, np.int8),
        "candidate_plane_geometry_objective": np.ones((1, 2)),
        "candidate_dense_normal_good_ray_recall": np.ones((1, 2)),
        "candidate_moge3_depth_scale": np.ones((1, 2)),
    }
    metadata = {
        "artifact_type": OUTPUT_ARTIFACT,
        "arrays_sha256": arrays_sha256(arrays),
        "query_pose_or_ground_truth_read": False,
        "selection_has_continuous_fusion_weight": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    path = tmp_path / "selected.npz"
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _load_selected(path)
    arrays["candidate_dense_normal_good_ray_recall"][0, 1] = 0.0
    np.savez_compressed(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="contract differs"):
        _load_selected(path)
