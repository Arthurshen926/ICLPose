import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import estimate_pose_pnp_ransac, pnp_pose_error, token_grid_xy
from feature_extract.vfm.query_to_render_matching import (
    QueryToRenderMatchingConfig,
    match_query_tokens_to_rendered_map,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _xyz_from_xy(xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (xy[:, 0] - 50.0) / 80.0 * z
    y = (xy[:, 1] - 50.0) / 80.0 * z
    return np.stack([x, y, z], axis=1).astype(np.float64)


def test_query_tokens_match_rendered_dense_map_and_support_pnp() -> None:
    dim = 8
    height = 4
    width = 4
    grid_xy = token_grid_xy(width, height, image_width=100, image_height=100)
    token_indices = np.asarray([0, 3, 5, 6, 9, 12], dtype=np.int64)
    xy = grid_xy[token_indices]
    xyz = _xyz_from_xy(xy, np.asarray([4.0, 4.6, 5.2, 5.8, 6.4, 7.0], dtype=np.float64))

    query_map = np.zeros((dim, height, width), dtype=np.float32)
    rendered_map = np.zeros((dim, height, width), dtype=np.float32)
    rendered_xyz = np.zeros((height, width, 3), dtype=np.float32)
    visibility = np.zeros((height, width), dtype=bool)
    for local_idx, token_idx in enumerate(token_indices):
        feature = np.zeros((dim,), dtype=np.float32)
        feature[local_idx] = 1.0
        y_idx, x_idx = divmod(int(token_idx), width)
        query_map[:, y_idx, x_idx] = feature
        rendered_map[:, y_idx, x_idx] = feature
        rendered_xyz[y_idx, x_idx] = xyz[local_idx].astype(np.float32)
        visibility[y_idx, x_idx] = True

    matches = match_query_tokens_to_rendered_map(
        query_map,
        rendered_map,
        rendered_xyz,
        visibility,
        QueryToRenderMatchingConfig(top_k=2, ratio_threshold=0.8, min_similarity=0.5, mutual=True),
        image_width=100,
        image_height=100,
    )

    assert len(matches) == len(token_indices)
    result = estimate_pose_pnp_ransac(matches, _camera(), reprojection_error_px=2.0, iterations=200)
    assert result.success
    assert result.inlier_count >= 5
    error = pnp_pose_error(result.pose_w2c, np.eye(4, dtype=np.float64))
    assert error.translation_m < 1e-4
    assert error.rotation_deg < 1e-3


def test_rendered_matching_ignores_invisible_pixels_and_applies_ratio() -> None:
    query_map = np.zeros((4, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    rendered_map = np.zeros((4, 1, 4), dtype=np.float32)
    rendered_map[:, 0, 0] = np.asarray([0.95, 0.3122499, 0.0, 0.0], dtype=np.float32)
    rendered_map[:, 0, 1] = np.asarray([0.94, 0.341174, 0.0, 0.0], dtype=np.float32)
    rendered_map[:, 0, 2] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    rendered_map[:, 0, 3] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rendered_xyz = np.zeros((1, 4, 3), dtype=np.float32)
    rendered_xyz[0, :, 2] = 3.0
    visibility = np.asarray([[True, True, True, False]], dtype=bool)

    matches = match_query_tokens_to_rendered_map(
        query_map,
        rendered_map,
        rendered_xyz,
        visibility,
        QueryToRenderMatchingConfig(top_k=2, ratio_threshold=0.8, min_similarity=0.5, mutual=True),
        image_width=20,
        image_height=10,
    )

    assert len(matches) == 1
    assert matches[0].track_id == 2
