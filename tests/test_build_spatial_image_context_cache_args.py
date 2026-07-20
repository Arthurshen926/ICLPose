from __future__ import annotations

from feature_extract.tools.vfm.build_alike_image_spatial_context_cache import (
    _load_alike_input,
    parse_args as parse_alike_args,
)
from feature_extract.tools.vfm.build_radio_intermediate_image_context_cache import (
    parse_args as parse_radio_args,
)


def test_spatial_image_cache_builders_expose_fixed_grid_configuration() -> None:
    radio = parse_radio_args(
        ["--colmap_model_dir", "model", "--image_root", "images", "--output", "cache.npz"]
    )
    alike = parse_alike_args(
        ["--colmap_model_dir", "model", "--image_root", "images", "--output", "cache.npz"]
    )

    assert radio.spatial_grid_sizes == "16"
    assert radio.intermediate_index == -6
    assert alike.spatial_grid_sizes == "16,32"


def test_alike_input_resizes_to_colmap_pixels_and_uses_short_file_hash(tmp_path) -> None:
    import cv2
    import numpy as np

    path = tmp_path / "rgb.png"
    bgr = np.zeros((6, 10, 3), dtype=np.uint8)
    bgr[..., 1] = 127
    assert cv2.imwrite(str(path), bgr)

    image, digest = _load_alike_input(path, width=5, height=3)

    assert tuple(image.shape) == (3, 3, 5)
    assert len(digest) == 16
