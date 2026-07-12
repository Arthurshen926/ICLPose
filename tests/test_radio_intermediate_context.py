from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from feature_extract.tools.vfm.build_radio_intermediate_context_cache import (
    _coordinate_space,
    _fit_pca_from_rgb,
)
from feature_extract.vfm.colmap_tracks import (
    ColmapCamera,
    ColmapImageObservation,
    write_colmap_cameras_binary,
    write_colmap_images_binary,
)
from feature_extract.vfm.localization.radio_intermediate_context import (
    RadioIntermediateContextCache,
    load_radio_intermediate_context_cache,
    project_and_sample_radio_map,
    save_radio_intermediate_context_cache,
)


def test_projected_radio_map_uses_endpoint_pixel_sampling() -> None:
    feature_map = np.asarray(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.1, 0.1], [0.1, 0.1]],
            [[0.2, 0.2], [0.2, 0.2]],
        ],
        dtype=np.float32,
    )
    sampled = project_and_sample_radio_map(
        feature_map,
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        image_width=2,
        image_height=2,
        pca_mean=np.zeros((4,), dtype=np.float32),
        pca_components=np.eye(4, dtype=np.float32)[:2],
        device="cpu",
    )

    np.testing.assert_allclose(sampled, np.eye(2, dtype=np.float32), atol=1e-6)


def test_radio_intermediate_cache_roundtrip_and_stale_rejection(tmp_path: Path) -> None:
    cache = RadioIntermediateContextCache(
        support_descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_anchor_descriptors=np.asarray([[0.0, 1.0]], dtype=np.float32),
        query_context_descriptors=np.asarray([[2**-0.5, 2**-0.5]], dtype=np.float32),
        pca_mean=np.zeros((4,), dtype=np.float32),
        pca_components=np.eye(4, dtype=np.float32)[:2],
        metadata={"source_sha256": "abc"},
    )
    path = tmp_path / "radio_context.npz"
    save_radio_intermediate_context_cache(cache, path, cache_dtype="float16")

    loaded = load_radio_intermediate_context_cache(
        path, expected_metadata={"source_sha256": "abc"}
    )
    assert loaded.descriptor_dim == 2
    np.testing.assert_allclose(loaded.support_descriptors, cache.support_descriptors)
    with pytest.raises(ValueError, match="stale"):
        load_radio_intermediate_context_cache(
            path, expected_metadata={"source_sha256": "different"}
        )


def test_radio_coordinate_space_uses_feature_xy_model_not_source_rgb_size(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (4, 2)).save(image_root / "query.png")
    model_dir = tmp_path / "model"
    camera = ColmapCamera(
        camera_id=1,
        model_id=0,
        width=2,
        height=1,
        params=(1.0, 1.0, 0.5),
    )
    image = ColmapImageObservation(
        image_id=1,
        image_name="query.png",
        camera_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros((3,), dtype=np.float64),
        xys=np.zeros((0, 2), dtype=np.float64),
        point3d_ids=np.zeros((0,), dtype=np.int64),
    )
    write_colmap_cameras_binary({1: camera}, model_dir / "cameras.bin")
    write_colmap_images_binary({1: image}, model_dir / "images.bin")

    model_sizes, model_metadata = _coordinate_space(
        model_dir, ["query.png"], image_root=image_root
    )
    rgb_sizes, rgb_metadata = _coordinate_space(
        None, ["query.png"], image_root=image_root
    )

    assert model_sizes["query.png"] == (2, 1)
    assert rgb_sizes["query.png"] == (4, 2)
    assert model_metadata["coordinate_space_id"] != rgb_metadata["coordinate_space_id"]


def test_fresh_radio_pca_uses_explicit_checkpoint_and_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index in range(2):
        Image.new("RGB", (4, 4), color=(index * 40, 10, 20)).save(
            image_root / f"image{index}.png"
        )
    checkpoint = tmp_path / "radio.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    class FakeExtractor:
        def __init__(self, *, version, device, radio_repo):
            calls.append((version, device, radio_repo))
            self.index = 0

        def extract_intermediate_batch(self, tensor, **kwargs):
            self.index += 1
            values = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
            values = values + torch.tensor(
                [0.0, float(self.index), 2.0 * self.index, 3.0],
                dtype=torch.float32,
            ).reshape(1, 4, 1, 1)
            return values

    monkeypatch.setattr(
        "feature_extract.tools.vfm.build_radio_intermediate_context_cache.RADIOFeatureExtractor",
        FakeExtractor,
    )

    mean, components, metadata = _fit_pca_from_rgb(
        ["image0.png", "image1.png"],
        image_root=image_root,
        source_image_count=2,
        samples_per_image=4,
        output_dim=2,
        seed=0,
        device="cpu",
        radio_repo="radio-repo",
        checkpoint_path=checkpoint,
        intermediate_index=-6,
    )

    assert calls == [(str(checkpoint.resolve()), "cpu", "radio-repo")]
    assert mean.shape == (4,)
    assert components.shape == (2, 4)
    assert metadata["source"] == "fresh_rgb_explicit_radio_checkpoint_v1"
