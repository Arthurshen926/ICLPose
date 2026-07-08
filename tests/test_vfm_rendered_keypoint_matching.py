from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.rendered_keypoint_matching import (
    KeypointFeatureMatch,
    backproject_depth_to_world,
    bilinear_sample_feature_map,
    dual_softmax_keypoint_matches,
    keypoint_feature_matches_to_pnp_matches,
    mutual_nn_keypoint_matches,
    refine_render_keypoint_matches_by_local_correlation,
    render_anchor_topk_keypoint_matches,
)
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import _geometry_row_fields
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import _render_keypoint_detector_image
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianRGBSource,
    GaussianVFMRenderConfig,
    GaussianVFMRenderResult,
)


def test_bilinear_sample_feature_map_uses_image_coordinates() -> None:
    feature_map = np.zeros((1, 2, 2), dtype=np.float32)
    feature_map[0] = np.asarray([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    values, valid = bilinear_sample_feature_map(
        feature_map,
        np.asarray([[5.0, 5.0], [11.0, 5.0]], dtype=np.float64),
        image_width=11,
        image_height=11,
    )

    assert valid.tolist() == [True, False]
    assert np.allclose(values[0], [3.0], atol=1e-6)


def test_mutual_nn_keypoint_matches_enforces_ratio_and_mutuality() -> None:
    query_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    render_xy = np.asarray([[1.0, 0.0], [11.0, 0.0], [20.0, 0.0]], dtype=np.float64)
    query_desc = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    render_desc = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)

    matches = mutual_nn_keypoint_matches(
        query_xy,
        query_desc,
        render_xy,
        render_desc,
        ratio_threshold=0.8,
        min_similarity=0.0,
    )

    assert [(m.query_index, m.render_index) for m in matches] == [(0, 0), (1, 1)]
    assert all(isinstance(match, KeypointFeatureMatch) for match in matches)


def test_mutual_nn_keypoint_matches_can_filter_by_dual_softmax_confidence() -> None:
    query_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    render_xy = np.asarray([[1.0, 0.0], [11.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    query_desc = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    render_desc = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.05]], dtype=np.float32)

    matches = mutual_nn_keypoint_matches(
        query_xy,
        query_desc,
        render_xy,
        render_desc,
        ratio_threshold=None,
        min_similarity=-1.0,
        dual_softmax_logit_scale=12.0,
        min_dual_softmax_confidence=0.75,
    )

    assert [(m.query_index, m.render_index) for m in matches] == [(1, 1)]
    assert matches[0].dual_softmax_confidence is not None


def test_dual_softmax_keypoint_matches_uses_symmetric_confidence() -> None:
    query_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    render_xy = np.asarray([[1.0, 0.0], [11.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    query_desc = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    render_desc = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.05]], dtype=np.float32)

    matches = dual_softmax_keypoint_matches(
        query_xy,
        query_desc,
        render_xy,
        render_desc,
        logit_scale=12.0,
        min_confidence=0.0,
    )

    assert [(m.query_index, m.render_index) for m in matches] == [(1, 1), (0, 0)]
    assert all(m.dual_softmax_confidence is not None for m in matches)
    assert matches[0].dual_softmax_confidence >= matches[1].dual_softmax_confidence


def test_render_anchor_topk_keypoint_matches_keeps_multiple_query_candidates_per_render_anchor() -> None:
    query_xy = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
    render_xy = np.asarray([[1.0, 0.0], [21.0, 0.0]], dtype=np.float64)
    query_desc = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    render_desc = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    matches = render_anchor_topk_keypoint_matches(
        query_xy,
        query_desc,
        render_xy,
        render_desc,
        top_l=2,
        min_similarity=-1.0,
    )

    by_render = {}
    for match in matches:
        by_render.setdefault(match.render_index, []).append(match)
    assert [match.query_index for match in by_render[0]] == [0, 1]
    assert [match.coarse_rank for match in by_render[0]] == [0, 1]
    assert [match.query_index for match in by_render[1]][0] == 2
    assert all(match.base_render_index == match.render_index for match in matches)


def test_refine_render_keypoint_matches_by_local_correlation_moves_render_point() -> None:
    render_feature = np.zeros((2, 10, 10), dtype=np.float32)
    render_feature[:, 5, 5] = np.asarray([1.0, 0.0], dtype=np.float32)
    render_feature[:, 5, 7] = np.asarray([0.0, 1.0], dtype=np.float32)
    query_desc = np.asarray([[0.0, 1.0]], dtype=np.float32)
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([3.0, 3.0], dtype=np.float64),
            render_xy=np.asarray([5.0, 5.0], dtype=np.float64),
            similarity=0.0,
            ratio=1.0,
        )
    ]

    refined = refine_render_keypoint_matches_by_local_correlation(
        matches,
        query_desc,
        render_feature,
        image_width=10,
        image_height=10,
        search_radius_px=3.0,
        step_px=1.0,
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].render_xy, [7.0, 5.0], atol=1e-6)
    assert refined[0].similarity > 0.99


def test_backproject_depth_to_world_handles_pinhole_pose() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(100.0, 100.0, 50.0, 50.0))
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 1.0
    xyz, valid = backproject_depth_to_world(
        np.asarray([[50.0, 50.0], [60.0, 50.0]], dtype=np.float64),
        np.asarray([5.0, 5.0], dtype=np.float64),
        camera,
        pose,
    )

    assert valid.tolist() == [True, True]
    assert np.allclose(xyz[0], [-1.0, 0.0, 5.0], atol=1e-6)
    assert np.allclose(xyz[1], [-0.5, 0.0, 5.0], atol=1e-6)


def test_keypoint_feature_matches_to_pnp_matches_samples_depth_and_xyz() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=20, height=20, params=(10.0, 10.0, 10.0, 10.0))
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([10.0, 10.0], dtype=np.float64),
            render_xy=np.asarray([10.0, 10.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.1,
            similarity_margin=0.8,
            dual_softmax_confidence=0.25,
        )
    ]
    depth = np.full((20, 20), 4.0, dtype=np.float32)
    pnp_matches = keypoint_feature_matches_to_pnp_matches(
        matches,
        depth,
        camera,
        np.eye(4, dtype=np.float64),
    )

    assert len(pnp_matches) == 1
    assert np.allclose(pnp_matches[0].xyz, [0.0, 0.0, 4.0], atol=1e-6)
    assert np.allclose(pnp_matches[0].xy, [10.0, 10.0], atol=1e-6)
    assert pnp_matches[0].pnp_soft_score == 0.25


def test_keypoint_feature_matches_to_pnp_matches_can_guard_render_offset_with_alpha_fallback() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=20, height=20, params=(10.0, 10.0, 10.0, 10.0))
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([10.0, 10.0], dtype=np.float64),
            render_xy=np.asarray([15.0, 15.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.1,
            similarity_margin=0.8,
            dual_softmax_confidence=0.25,
        )
    ]
    depth = np.full((20, 20), 4.0, dtype=np.float32)
    alpha = np.ones((20, 20), dtype=np.float32)
    alpha[15, 15] = 0.05

    pnp_matches = keypoint_feature_matches_to_pnp_matches(
        matches,
        depth,
        camera,
        np.eye(4, dtype=np.float64),
        rendered_alpha=alpha,
        render_grid_width=2,
        render_grid_height=2,
        min_render_alpha=0.2,
        fallback_to_cell_center=True,
    )

    assert len(pnp_matches) == 1
    assert np.allclose(pnp_matches[0].xyz, [-2.0, -2.0, 4.0], atol=1e-6)
    assert pnp_matches[0].patch_offset_applied is False
    assert pnp_matches[0].render_alpha == 1.0


def test_keypoint_feature_matches_to_pnp_matches_can_freeze_render_anchor_xyz() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=20, height=20, params=(10.0, 10.0, 10.0, 10.0))
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([10.0, 10.0], dtype=np.float64),
            render_xy=np.asarray([15.0, 15.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.1,
        )
    ]
    depth = np.full((20, 20), 4.0, dtype=np.float32)
    depth[15, 15] = 8.0

    dynamic = keypoint_feature_matches_to_pnp_matches(
        matches,
        depth,
        camera,
        np.eye(4, dtype=np.float64),
        render_grid_width=2,
        render_grid_height=2,
    )
    fixed = keypoint_feature_matches_to_pnp_matches(
        matches,
        depth,
        camera,
        np.eye(4, dtype=np.float64),
        render_grid_width=2,
        render_grid_height=2,
        fixed_render_anchor=True,
    )

    assert len(dynamic) == len(fixed) == 1
    assert np.allclose(dynamic[0].xyz, [4.0, 4.0, 8.0], atol=1e-6)
    assert np.allclose(fixed[0].xyz, [-2.0, -2.0, 4.0], atol=1e-6)
    assert fixed[0].anchor_xyz_change_m == 0.0
    assert fixed[0].surface_switch_flag is False
    assert fixed[0].render_depth_change_m == 4.0


def test_keypoint_feature_matches_requires_grid_for_depth_delta_guard() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=20, height=20, params=(10.0, 10.0, 10.0, 10.0))
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([10.0, 10.0], dtype=np.float64),
            render_xy=np.asarray([10.0, 10.0], dtype=np.float64),
            similarity=0.9,
            ratio=0.1,
        )
    ]
    depth = np.full((20, 20), 4.0, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        keypoint_feature_matches_to_pnp_matches(
            matches,
            depth,
            camera,
            np.eye(4, dtype=np.float64),
            max_render_depth_delta_m=0.1,
        )


def test_geometry_row_fields_uses_reprojection_stats_key_names() -> None:
    fields = _geometry_row_fields(
        {
            "gt_precision_5px": 0.25,
            "gt_precision_16px": 0.5,
            "gt_precision_32px": 0.875,
            "gt_reproj_median_px": 18.0,
            "pnp_inlier_gt_precision_16px": 0.75,
            "pnp_inlier_gt_precision_32px": 1.0,
            "pnp_inlier_gt_reproj_median_px": 6.0,
            "match_count": 8,
        }
    )

    assert fields == {
        "gt_precision_5px": 0.25,
        "gt_precision_16px": 0.5,
        "gt_precision_32px": 0.875,
        "gt_reproj_median_px": 18.0,
        "pnp_inlier_gt_precision_16px": 0.75,
        "pnp_inlier_gt_precision_32px": 1.0,
        "pnp_inlier_gt_reproj_median_px": 6.0,
    }


def test_render_keypoint_detector_image_prefers_true_rgb_source() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=20, height=20, params=(10.0, 10.0, 10.0, 10.0))
    config = GaussianVFMRenderConfig(width=20, height=20, radius_px=2.0, depth_epsilon=0.01)
    rendered = GaussianVFMRenderResult(
        feature_map=np.zeros((2, 20, 20), dtype=np.float32),
        xyz_map=np.zeros((20, 20, 3), dtype=np.float32),
        visibility_mask=np.ones((20, 20), dtype=bool),
        depth=np.ones((20, 20), dtype=np.float32),
        weight_sum=np.ones((20, 20), dtype=np.float32),
        dominant_gaussian_index=np.zeros((20, 20), dtype=np.int64),
    )
    source = GaussianRGBSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        rgb=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        opacity=np.asarray([1.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        gaussian_indices=np.asarray([0], dtype=np.int64),
    )

    rgb = _render_keypoint_detector_image(
        rendered,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=camera,
        config=config,
        rgb_source=source,
        renderer="soft",
        device="cpu",
    )

    assert rgb.dtype == np.uint8
    assert rgb.shape == (20, 20, 3)
    assert int(np.max(rgb[..., 0])) > int(np.max(rgb[..., 1]))
