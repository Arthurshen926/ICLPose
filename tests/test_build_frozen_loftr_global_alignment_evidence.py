from __future__ import annotations

import pytest

from feature_extract.tools.vfm.build_frozen_loftr_global_alignment_evidence import (
    _source_size_from_cache_metadata,
)


def test_source_size_requires_one_shared_declared_pixel_grid() -> None:
    metadata = {
        "image_source_contract": {
            "query": {"source_image_dimensions": {"1920x1080": 1}},
            "mapping_support": {"source_image_dimensions": {"1920x1080": 790}},
        }
    }
    assert _source_size_from_cache_metadata(metadata) == (1920, 1080)

    metadata["image_source_contract"]["mapping_support"]["source_image_dimensions"] = {
        "1024x768": 790
    }
    with pytest.raises(ValueError, match="grids differ"):
        _source_size_from_cache_metadata(metadata)
