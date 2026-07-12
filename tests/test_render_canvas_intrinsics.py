from __future__ import annotations

import pytest
import numpy as np

from feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose import (
    _match_table_rows_for_query,
    _render_canvas_camera_from_base,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


def test_render_canvas_keeps_focal_and_shifts_principal_point() -> None:
    base = ColmapCamera(
        camera_id=1,
        model_id=1,
        width=1280,
        height=720,
        params=(1000.0, 1000.0, 640.0, 360.0),
    )

    canvas = _render_canvas_camera_from_base(base, canvas_width=1600, canvas_height=900)

    assert canvas.width == 1600
    assert canvas.height == 900
    assert canvas.params == (1000.0, 1000.0, 800.0, 450.0)


def test_render_canvas_rejects_smaller_canvas() -> None:
    base = ColmapCamera(
        camera_id=1,
        model_id=1,
        width=1280,
        height=720,
        params=(1000.0, 1000.0, 640.0, 360.0),
    )

    with pytest.raises(ValueError, match="at least as large"):
        _render_canvas_camera_from_base(base, canvas_width=1200, canvas_height=720)


def test_match_table_rows_store_gt_error_confidence_and_inlier_label() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=7,
            xy=np.asarray([10.0, 20.0], dtype=np.float64),
            track_id=3,
            xyz=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            similarity=0.75,
            ratio=0.2,
            landmark_variance=0.0,
            pnp_soft_score=0.9,
            patch_offset_norm_px=1.5,
        )
    ]

    rows = _match_table_rows_for_query(
        query_id="seq/frame.png",
        matches=matches,
        gt_errors=np.asarray([12.0], dtype=np.float64),
        gt_stride_px=16.0,
        inlier_mask=np.asarray([True]),
        baseline_reproj_errors=np.asarray([4.25], dtype=np.float64),
        render_xy_by_match={3: np.asarray([30.0, 40.0], dtype=np.float64)},
    )

    assert rows == [
        {
            "query_id": "seq/frame.png",
            "match_index": 0,
            "query_index": 7,
            "render_index": 3,
            "query_x": 10.0,
            "query_y": 20.0,
            "query_gt_x": None,
            "query_gt_y": None,
            "render_x": 30.0,
            "render_y": 40.0,
            "xy": [10.0, 20.0],
            "world_x": 1.0,
            "world_y": 2.0,
            "world_z": 3.0,
            "similarity": 0.75,
            "similarity_margin": None,
            "match_rank": 0,
            "token_match_rank": None,
            "base_render_index": None,
            "candidate_render_index": None,
            "candidate_id": None,
            "coarse_rank": None,
            "coarse_score": None,
            "coarse_score_gap": None,
            "mutual_rank": None,
            "cell_delta_x": None,
            "cell_delta_y": None,
            "confidence": 0.9,
            "gt_reproj_error_px": 12.0,
            "gt_reproj_error_stride": 0.75,
            "gt_correct_5px": False,
            "gt_correct_10px": False,
            "gt_correct_8px": False,
            "gt_correct_16px": True,
            "gt_correct_24px": True,
            "patch_correct": True,
            "patch_positive_label": True,
            "strong_positive_label": True,
            "weak_positive_label": False,
            "hard_negative_label": False,
            "ignore_label": False,
            "pose_usable_label": True,
            "pnp_inlier": True,
            "baseline_reproj_residual_px": 4.25,
            "patch_offset_norm_px": 1.5,
            "anchor_xyz_change_m": None,
            "render_depth_change_m": None,
            "surface_switch_flag": None,
            "render_depth_gradient": None,
            "render_depth": None,
            "render_alpha": None,
        }
    ]


def test_match_table_rows_mark_self_consistent_wrong_match_as_hard_negative() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=9,
            xy=np.asarray([32.0, 48.0], dtype=np.float64),
            track_id=4,
            xyz=np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
            similarity=0.85,
            ratio=0.1,
            landmark_variance=0.0,
        )
    ]

    rows = _match_table_rows_for_query(
        query_id="seq/frame.png",
        matches=matches,
        gt_errors=np.asarray([40.0], dtype=np.float64),
        gt_stride_px=16.0,
        inlier_mask=np.asarray([True]),
        baseline_reproj_errors=np.asarray([2.0], dtype=np.float64),
        render_xy_by_match={4: np.asarray([64.0, 80.0], dtype=np.float64)},
    )

    assert rows[0]["gt_reproj_error_stride"] == 2.5
    assert rows[0]["patch_positive_label"] is False
    assert rows[0]["ignore_label"] is False
    assert rows[0]["pnp_inlier"] is True
    assert rows[0]["baseline_reproj_residual_px"] == 2.0
    assert rows[0]["hard_negative_label"] is True


def test_match_table_rows_prefer_per_match_render_xy_over_render_index_lookup() -> None:
    match = QueryTo3DMatch(
        token_index=1,
        xy=np.asarray([10.0, 12.0], dtype=np.float64),
        track_id=5,
        xyz=np.asarray([0.0, 0.0, 3.0], dtype=np.float64),
        similarity=0.5,
        ratio=0.0,
        landmark_variance=0.0,
        render_xy=np.asarray([40.0, 44.0], dtype=np.float64),
        base_render_index=4,
        candidate_render_index=5,
        candidate_id=12,
        cell_delta_x=1,
        cell_delta_y=0,
    )

    rows = _match_table_rows_for_query(
        query_id="seq/frame.png",
        matches=[match],
        gt_errors=np.asarray([4.0], dtype=np.float64),
        gt_stride_px=16.0,
        inlier_mask=None,
        baseline_reproj_errors=None,
        render_xy_by_match={5: np.asarray([999.0, 999.0], dtype=np.float64)},
    )

    assert rows[0]["render_x"] == 40.0
    assert rows[0]["render_y"] == 44.0
    assert rows[0]["base_render_index"] == 4
    assert rows[0]["candidate_render_index"] == 5
    assert rows[0]["candidate_id"] == 12
    assert rows[0]["cell_delta_x"] == 1
    assert rows[0]["cell_delta_y"] == 0
