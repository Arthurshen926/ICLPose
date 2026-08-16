from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    SCENE_AGGREGATION,
    PureRadioPhysicalRetrieval,
    all_radio_token_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
    build_child_visibility_pose_atlas,
    diverse_pose_rows,
    score_visibility_pose_atlas,
)
from test_goal_maplet_pure_retrieval import _physical


def _pose(x: float, yaw_degrees: float = 0.0) -> np.ndarray:
    angle = np.radians(yaw_degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ np.asarray([x, 0.0, 0.0])
    return pose


def _contributor(path: Path, physical, child: int, pose: np.ndarray) -> None:
    start, end = physical.child_member_offsets[child : child + 2]
    primitive_rows = physical.child_member_primitive_rows[start:end]
    primitive_ids = physical.primitive_ids[primitive_rows]
    ids = np.full((8, 8, 1), int(primitive_ids[0]), dtype=np.int32)
    np.savez_compressed(
        path,
        topk_ids=ids,
        topk_weights=np.ones(ids.shape, dtype=np.float16),
        pose_w2c=pose,
    )


def _retrieval(physical, child: int, *, selected_child: int | None = None):
    xy = all_radio_token_coordinates(36, 64)
    rows = np.full((xy.shape[0], 1), int(child), dtype=np.int64)
    parent = int(physical.maplet_ids[int(physical.child_parent_rows[child])])
    chosen = int(child if selected_child is None else selected_child)
    return PureRadioPhysicalRetrieval(
        image_id="seq/query.png",
        token_xy=xy,
        token_parent_ids=np.full(rows.shape, parent, dtype=np.int64),
        token_parent_probabilities=np.full(rows.shape, 0.9, dtype=np.float32),
        token_out_of_map_probabilities=np.full(xy.shape[0], 0.05, dtype=np.float32),
        token_in_map_tail_probabilities=np.full(xy.shape[0], 0.05, dtype=np.float32),
        token_child_rows=rows,
        token_child_probabilities=np.full(rows.shape, 0.9, dtype=np.float32),
        scene_parent_ids=np.asarray([parent], dtype=np.int64),
        scene_parent_scores=np.asarray([1.0], dtype=np.float32),
        scene_child_rows=np.asarray([chosen], dtype=np.int64),
        scene_child_scores=np.asarray([1.0], dtype=np.float32),
        physical_map_sha256=physical.content_sha256,
        metadata={
            "artifact_type": "goal_maplet_pure_radio_physical_retrieval_v1",
            "token_height": 36,
            "token_width": 64,
            "scene_aggregation": SCENE_AGGREGATION,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "uses_alike": False,
            "uses_pnp": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_mapping_rgb": False,
            "uses_image_retrieval": False,
        },
    )


def test_child_visibility_atlas_retrieves_covisible_child_pose(tmp_path):
    physical = _physical()
    paths = []
    for index, child in enumerate((0, 1, 2)):
        path = tmp_path / f"view{index}.npz"
        _contributor(path, physical, child, _pose(float(index) * 3.0))
        paths.append(path)
    atlas = build_child_visibility_pose_atlas(
        physical, paths, grid_rows=2, grid_cols=2,
        maximum_global_children=3, maximum_children_per_cell=3,
    )
    retrieval = _retrieval(physical, 1)
    score, global_score, layout_score = score_visibility_pose_atlas(
        atlas, retrieval, layout_weight=0.5
    )
    assert int(np.argmax(score)) == 1
    assert global_score[1] == pytest.approx(1.0)
    assert layout_score[1] == pytest.approx(1.0)
    assert score[0] == pytest.approx(0.0)


def test_visibility_handoff_respects_selected_set_and_candidate_ceiling(tmp_path):
    physical = _physical()
    paths = []
    for index, child in enumerate((0, 1)):
        path = tmp_path / f"view{index}.npz"
        _contributor(path, physical, child, _pose(float(index)))
        paths.append(path)
    atlas = build_child_visibility_pose_atlas(physical, paths)
    retrieval = _retrieval(physical, 1, selected_child=0)
    selected_score = score_visibility_pose_atlas(
        atlas, retrieval, selected_children_only=True
    )[0]
    candidate_score = score_visibility_pose_atlas(
        atlas, retrieval, selected_children_only=False
    )[0]
    np.testing.assert_allclose(selected_score, 0.0)
    assert int(np.argmax(candidate_score)) == 1
    assert candidate_score[1] > 0.99


def test_visibility_atlas_roundtrip_and_forbidden_claim(tmp_path):
    physical = _physical()
    contributor = tmp_path / "view.npz"
    _contributor(contributor, physical, 0, _pose(0.0))
    atlas = build_child_visibility_pose_atlas(physical, [contributor])
    path = tmp_path / "atlas.npz"
    atlas.save_npz(path)
    loaded = ChildVisibilityPoseAtlas.load_npz(path)
    assert loaded.content_sha256 == atlas.content_sha256
    with pytest.raises(ValueError, match="pose-free handoff"):
        ChildVisibilityPoseAtlas(
            **{**atlas.__dict__, "metadata": {**atlas.metadata, "uses_pnp": True}}
        )


def test_pose_nms_keeps_distinct_basins_and_stable_ties():
    poses = np.stack([_pose(0.0), _pose(0.1, 1.0), _pose(2.0), _pose(4.0)])
    rows = diverse_pose_rows(
        poses,
        np.asarray([1.0, 1.0, 0.8, 0.7]),
        maximum_modes=3,
        translation_nms_m=0.5,
        rotation_nms_deg=5.0,
    )
    assert rows.tolist() == [0, 2, 3]
