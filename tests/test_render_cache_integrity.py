from __future__ import annotations

import numpy as np
import torch

from feature_extract.tools.vfm.build_matcha_joint_cache import _load_or_extract_matcha_joint_feature
from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _load_or_render_rgb_depth_cache


def test_rgb_depth_cache_recovers_from_corrupt_npz(tmp_path) -> None:
    cache_path = tmp_path / "rgb_depth.npz"
    cache_path.write_bytes(b"not a zip")
    calls = {"count": 0}

    def render_fn():
        calls["count"] += 1
        return (
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.ones((4, 5), dtype=np.float32),
            np.ones((4, 5), dtype=np.float32),
        )

    rgb, depth, alpha = _load_or_render_rgb_depth_cache(
        cache_path=cache_path,
        render_fn=render_fn,
        skip_existing=True,
    )

    assert calls["count"] == 1
    assert rgb.shape == (4, 5, 3)
    assert depth.shape == (4, 5)
    assert alpha.shape == (4, 5)
    with np.load(cache_path) as data:
        assert set(data.files) == {"alpha", "depth", "rgb"}


def test_matcha_joint_feature_cache_recovers_from_corrupt_npz(tmp_path) -> None:
    cache_path = tmp_path / "feature.npz"
    cache_path.write_bytes(b"not a zip")

    class _Extractor:
        def extract_dual(self, tensor, *, fine_intermediate_index, coarse_source, coarse_intermediate_index):
            return {"dual": torch.ones((3, 2, 2), dtype=torch.float32)}

    feature = _load_or_extract_matcha_joint_feature(
        cache_path=cache_path,
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        extractor=_Extractor(),
        layer_name="radio_dual",
        feature_mode="radio_dual",
        fine_intermediate_index=9,
        coarse_source="radio_intermediate",
        coarse_intermediate_index=23,
        skip_existing=True,
    )

    assert feature.shape == (3, 2, 2)
    with np.load(cache_path) as data:
        np.testing.assert_allclose(data["radio_dual"], np.ones((3, 2, 2), dtype=np.float32))
