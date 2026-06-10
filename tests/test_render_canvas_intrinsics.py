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
        inlier_mask=np.asarray([True]),
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
            "render_x": 30.0,
            "render_y": 40.0,
            "world_x": 1.0,
            "world_y": 2.0,
            "world_z": 3.0,
            "similarity": 0.75,
            "similarity_margin": None,
            "confidence": 0.9,
            "gt_reproj_error_px": 12.0,
            "gt_correct_8px": False,
            "gt_correct_16px": True,
            "gt_correct_24px": True,
            "pnp_inlier": True,
            "patch_offset_norm_px": 1.5,
            "render_depth": None,
            "render_alpha": None,
        }
    ]
