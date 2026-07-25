from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.frozen_loftr_coordinate_contract import (
    FrozenLoFTRColmapCoordinateContract,
    validate_frozen_loftr_colmap_coordinate_metadata,
)
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    loftr_source_to_model_pixel_centers_by_image,
    model_to_loftr_source_pixel_centers_by_image,
    shared_source_size_from_loftr_cache_metadata,
)


def _cache_metadata() -> dict[str, object]:
    return {
        "strict_global_pair_contract": {
            "coordinate_space": "source_pixel_centers_half_pixel_resize_inverse_v1",
        },
        "image_source_contract": {
            "query": {"source_image_dimensions": {"1920x1080": 1}},
            "mapping_support": {"source_image_dimensions": {"1920x1080": 2}},
        },
    }


def test_model_to_source_conversion_is_per_image_and_half_pixel_invertible() -> None:
    model_xy = np.asarray([[0.0, 0.0], [959.0, 539.0]], dtype=np.float32)
    image_ids = np.asarray(["support-a.png", "support-b.png"])
    model_sizes = {
        "support-a.png": (1024, 576),
        "support-b.png": (960, 540),
    }
    source_xy = model_to_loftr_source_pixel_centers_by_image(
        model_xy,
        image_ids=image_ids,
        model_sizes_by_image=model_sizes,
        source_size=(1920, 1080),
    )
    # These values differ from an origin-preserving multiply; that is exactly
    # why a direct 1024-grid vs. 1920-grid comparison is invalid.
    assert source_xy[0].tolist() == pytest.approx([0.4375, 0.4375])
    assert source_xy[1].tolist() == pytest.approx([1918.5, 1078.5])
    restored = loftr_source_to_model_pixel_centers_by_image(
        source_xy,
        image_ids=image_ids,
        model_sizes_by_image=model_sizes,
        source_size=(1920, 1080),
    )
    np.testing.assert_allclose(restored, model_xy, atol=1e-5)


def test_cache_coordinate_contract_rejects_ambiguous_or_mixed_source_grid() -> None:
    assert shared_source_size_from_loftr_cache_metadata(_cache_metadata()) == (1920, 1080)
    ambiguous = _cache_metadata()
    del ambiguous["strict_global_pair_contract"]
    with pytest.raises(ValueError, match="source-pixel coordinate space"):
        shared_source_size_from_loftr_cache_metadata(ambiguous)

    mixed = _cache_metadata()
    mixed["image_source_contract"]["mapping_support"]["source_image_dimensions"] = {
        "1024x576": 2
    }
    with pytest.raises(ValueError, match="source grids differ"):
        shared_source_size_from_loftr_cache_metadata(mixed)


def test_v2_coordinate_metadata_binds_transform_and_refuses_missing_contract() -> None:
    contract = FrozenLoFTRColmapCoordinateContract(
        source_size=(1920, 1080),
        query_id="query.png",
        query_model_size=(1024, 576),
        support_image_ids=("support-a.png", "support-b.png"),
        model_sizes_by_image={
            "query.png": (1024, 576),
            "support-a.png": (1024, 576),
            "support-b.png": (1024, 576),
        },
        colmap_model_dir="/tmp/model",
        cameras_sha256="camera-hash",
        images_camera_ownership_sha256="ownership-hash",
    )
    metadata = {"coordinate_contract": contract.metadata()}
    validate_frozen_loftr_colmap_coordinate_metadata(metadata)
    with pytest.raises(ValueError, match="no explicit"):
        validate_frozen_loftr_colmap_coordinate_metadata({})
