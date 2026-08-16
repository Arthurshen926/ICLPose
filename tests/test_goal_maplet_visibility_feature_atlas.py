from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.visibility_feature_atlas import (
    CanonicalFeatureVisibilityPoseAtlas,
    build_canonical_feature_visibility_pose_atlas,
    pool_normalized_feature_grid,
    score_canonical_feature_visibility_pose_atlas,
)
from test_goal_maplet_pure_retrieval import _physical


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = -float(x)
    return pose


def test_fixed_grid_pooling_keeps_spatial_cells_separate():
    feature = np.zeros((2, 4, 4), dtype=np.float32)
    feature[0, :2] = 1.0
    feature[1, 2:] = 1.0
    pooled, valid = pool_normalized_feature_grid(
        feature, np.ones((4, 4), dtype=bool), output_rows=2, output_cols=2
    )
    np.testing.assert_allclose(pooled[:, 0], np.asarray([[1.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(pooled[:, 1], np.asarray([[0.0, 0.0], [1.0, 1.0]]))
    assert np.all(valid)


def test_feature_atlas_scores_matching_layout_and_missing_floor():
    feature = np.zeros((2, 2, 2, 2), dtype=np.float16)
    feature[0, 0] = 1.0
    feature[1, 1] = 1.0
    atlas = CanonicalFeatureVisibilityPoseAtlas(
        poses_w2c=np.stack([_pose(0), _pose(1)]),
        features=feature,
        valid=np.ones((2, 2, 2), dtype=bool),
        physical_map_sha256="p",
        canonical_field_sha256="f",
        metadata={},
    )
    query = np.zeros((2, 4, 4), dtype=np.float32)
    query[0] = 1.0
    score = score_canonical_feature_visibility_pose_atlas(atlas, query)
    assert score[0] == pytest.approx(1.0)
    assert score[1] == pytest.approx(0.0)
    missing = CanonicalFeatureVisibilityPoseAtlas(
        **{**atlas.__dict__, "valid": np.zeros_like(atlas.valid)}
    )
    np.testing.assert_allclose(
        score_canonical_feature_visibility_pose_atlas(missing, query), -1.0
    )


def test_feature_atlas_builder_uses_canonical_codes_not_mapping_rgb(tmp_path):
    physical = _physical()
    codes = np.zeros((physical.primitive_ids.size, 2), dtype=np.float32)
    codes[:, 0] = 1.0
    field = SimpleNamespace(
        physical_map_sha256=physical.content_sha256,
        primitive_rows=np.arange(physical.primitive_ids.size, dtype=np.int64),
        codes=codes,
        feature_dim=2,
        content_sha256="field",
    )
    path = tmp_path / "view.npz"
    ids = np.full((144, 256, 1), int(physical.primitive_ids[0]), dtype=np.int32)
    np.savez_compressed(
        path, topk_ids=ids, topk_weights=np.ones_like(ids, dtype=np.float16),
        pose_w2c=_pose(0),
    )
    atlas = build_canonical_feature_visibility_pose_atlas(
        physical, field, [path], output_rows=9, output_cols=16
    )
    assert atlas.features.shape == (1, 2, 9, 16)
    assert np.all(atlas.valid)
    np.testing.assert_allclose(atlas.features[:, 0], 1.0)
    assert atlas.metadata["stores_mapping_rgb"] is False
    assert atlas.metadata["uses_pnp"] is False
