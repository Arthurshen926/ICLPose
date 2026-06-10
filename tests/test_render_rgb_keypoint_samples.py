from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose import (
    _project_feature_pair_for_render_rgb_eval,
    parse_args as parse_render_rgb_eval_args,
)
from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import (
    _extract_radio_feature_from_rgb,
    _load_or_render_rgb_depth_cache,
    _render_token_cache_path,
    _resolve_render_size,
    _select_records,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.render_rgb_keypoint_samples import (
    build_render_rgb_keypoint_adapter_samples,
    render_rgb_keypoint_pair_diagnostics,
)
from feature_extract.vfm.rendered_keypoint_selector_samples import RenderedKeypointSelectorSampleConfig


def test_render_rgb_keypoint_pair_diagnostics_reports_top1_geometry() -> None:
    query_desc = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    render_desc = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    errors = np.asarray(
        [
            [2.0, 64.0, 80.0],
            [50.0, 4.0, 72.0],
        ],
        dtype=np.float32,
    )

    diag = render_rgb_keypoint_pair_diagnostics(
        query_desc,
        render_desc,
        errors,
        positive_threshold_px=16.0,
        negative_threshold_px=32.0,
    )

    assert diag.query_keypoint_count == 2
    assert diag.render_keypoint_count == 3
    assert diag.positive_pair_count == 2
    assert diag.raw_top1_gt16 == 1.0
    assert np.isclose(diag.raw_top1_median_reprojection_px, 3.0)
    assert diag.positive_similarity_mean > diag.negative_similarity_mean


def test_build_render_rgb_keypoint_adapter_samples_attaches_diagnostics() -> None:
    query_desc = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    render_desc = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    errors = np.asarray([[2.0, 64.0, 80.0], [70.0, 3.0, 60.0]], dtype=np.float32)

    samples = build_render_rgb_keypoint_adapter_samples(
        query_desc,
        render_desc,
        errors,
        RenderedKeypointSelectorSampleConfig(
            positive_threshold_px=16.0,
            negative_threshold_px=32.0,
            hard_negatives_per_keypoint=1,
            max_positives_per_keypoint=1,
            keypoint_stride_px=16.0,
        ),
    )

    assert samples.sample_count == 2
    assert samples.metadata["source"] == "render_rgb_radio_keypoints"
    assert samples.metadata["diagnostics"]["raw_top1_gt16"] == 1.0


def test_extract_radio_feature_from_rgb_uses_extractor_tensor_contract() -> None:
    class FakeExtractor:
        def extract(self, tensor):
            assert tuple(tensor.shape) == (1, 3, 2, 2)
            assert float(tensor.max()) <= 1.0
            local = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            import torch

            return {"local": torch.from_numpy(local)}

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 128, 0]
    feature = _extract_radio_feature_from_rgb(rgb, FakeExtractor())

    assert feature.shape == (2, 2, 2)
    assert feature.dtype == np.float32


def test_render_rgb_eval_args_do_not_require_precomputed_feature_field() -> None:
    args = parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "manifest.json",
            "--query_pose_file",
            "poses.txt",
            "--image_root",
            "images",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "out",
        ]
    )

    assert args.gaussian_rgb_ply == "point_cloud.ply"
    assert not hasattr(args, "field")


def test_project_feature_pair_for_render_rgb_eval_uses_same_selector(monkeypatch) -> None:
    calls = []

    def fake_project(feature, selector_checkpoint, device):
        calls.append((feature.shape, selector_checkpoint, device))
        return feature[:1] * 2.0

    monkeypatch.setattr(
        "feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose._project_query_feature",
        fake_project,
    )
    query = np.ones((2, 3, 4), dtype=np.float32)
    render = np.full((2, 5, 6), 3.0, dtype=np.float32)

    query_projected, render_projected = _project_feature_pair_for_render_rgb_eval(
        query,
        render,
        selector_checkpoint="selector.pt",
        device="cpu",
    )

    assert query_projected.shape == (1, 3, 4)
    assert render_projected.shape == (1, 5, 6)
    assert np.allclose(query_projected, 2.0)
    assert np.allclose(render_projected, 6.0)
    assert calls == [((2, 3, 4), "selector.pt", "cpu"), ((2, 5, 6), "selector.pt", "cpu")]


def test_select_records_supports_start_index_before_limit() -> None:
    records = list(range(10))

    selected = _select_records(records, max_queries=3, mode="prefix", start_index=4)

    assert selected == [4, 5, 6]


def test_resolve_render_size_uses_camera_resolution_for_zero_values() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=1920, height=1080, params=(1000.0, 1000.0, 960.0, 540.0))

    assert _resolve_render_size(camera, 0, 0) == (1920, 1080)
    assert _resolve_render_size(camera, 640, 0) == (640, 1080)
    assert _resolve_render_size(camera, 0, 360) == (1920, 360)
    assert _resolve_render_size(camera, 320, 180) == (320, 180)


def test_load_or_render_rgb_depth_cache_reuses_cached_render(tmp_path) -> None:
    calls = {"count": 0}

    def render_fn():
        calls["count"] += 1
        return (
            np.full((3, 4, 3), 17, dtype=np.uint8),
            np.full((3, 4), 2.5, dtype=np.float32),
            np.full((3, 4), 0.75, dtype=np.float32),
        )

    cache_path = tmp_path / "render_rgb_depth.npz"
    first_rgb, first_depth, first_alpha = _load_or_render_rgb_depth_cache(
        cache_path=cache_path,
        render_fn=render_fn,
        skip_existing=True,
    )
    second_rgb, second_depth, second_alpha = _load_or_render_rgb_depth_cache(
        cache_path=cache_path,
        render_fn=render_fn,
        skip_existing=True,
    )

    assert calls["count"] == 1
    assert np.array_equal(first_rgb, second_rgb)
    assert np.allclose(first_depth, second_depth)
    assert np.allclose(first_alpha, second_alpha)
    assert first_rgb.dtype == np.uint8


def test_render_token_cache_path_includes_resolution(tmp_path) -> None:
    path_320 = _render_token_cache_path(tmp_path, "seq9/frame00001.png", 320, 180)
    path_1920 = _render_token_cache_path(tmp_path, "seq9/frame00001.png", 1920, 1080)

    assert path_320.name == "seq9__frame00001.png_320x180.npz"
    assert path_1920.name == "seq9__frame00001.png_1920x1080.npz"
    assert path_320 != path_1920
