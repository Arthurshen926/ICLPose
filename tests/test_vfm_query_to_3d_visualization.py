import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch
from feature_extract.vfm.query_to_3d_visualization import (
    render_query_to_3d_match_overlay,
    render_query_to_projected_map_correspondence,
)


def test_render_query_to_3d_match_overlay_marks_correct_and_false_matches() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    matches = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([50.0, 50.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.1,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([10.0, 10.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.8,
            ratio=0.2,
            landmark_variance=0.0,
        ),
    ]

    overlay, summary = render_query_to_3d_match_overlay(
        image,
        matches,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=camera,
        inlier_mask=np.asarray([True, False]),
        reprojection_threshold_px=4.0,
        max_draw=10,
    )

    assert overlay.shape == image.shape
    assert int(overlay.sum()) > 0
    assert summary["drawn_match_count"] == 2
    assert summary["gt_inlier_count"] == 1
    assert summary["gt_precision"] == 0.5
    assert summary["pnp_inlier_count"] == 1


def test_render_query_to_projected_map_correspondence_builds_side_by_side_view() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    query_feature_map = np.zeros((4, 2, 2), dtype=np.float32)
    query_feature_map[0, 0, 0] = 1.0
    query_feature_map[1, 1, 1] = 1.0
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.5, 0.5, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64),
        observation_image_ids=(("ref.png",), ("ref.png",)),
    )
    matches = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([50.0, 50.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.1,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=3,
            xy=np.asarray([99.0, 99.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([0.5, 0.5, 5.0], dtype=np.float64),
            similarity=0.8,
            ratio=0.2,
            landmark_variance=0.0,
        ),
    ]

    rgb_view, rgb_summary = render_query_to_projected_map_correspondence(
        image,
        query_feature_map,
        index,
        matches,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=camera,
        mode="rgb",
        max_draw=10,
    )
    feature_view, feature_summary = render_query_to_projected_map_correspondence(
        image,
        query_feature_map,
        index,
        matches,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=camera,
        mode="feature",
        max_draw=10,
    )

    assert rgb_view.shape[0] == 100
    assert rgb_view.shape[1] > 200
    assert int(rgb_view.sum()) > 0
    assert feature_view.shape == rgb_view.shape
    assert rgb_summary["projected_landmark_count"] == 2
    assert rgb_summary["drawn_match_count"] == 2
    assert feature_summary["mode"] == "feature"
