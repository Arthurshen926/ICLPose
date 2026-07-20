from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    FROZEN_LOFTR_PAIR_CACHE_FORMAT,
    load_frozen_loftr_pair_cache,
    resize_pixel_centers_half_pixel,
    restore_pixel_centers_half_pixel,
    split_batched_loftr_matches,
)


def _metadata() -> dict[str, object]:
    return {
        "format": FROZEN_LOFTR_PAIR_CACHE_FORMAT,
        "query_id": "query.png",
        "contains_target_fields": False,
        "supervision_arrays_loaded": False,
        "strict_global_pair_contract": {
            "all_mapping_manifest_images_processed": True,
            "image_level_selection": False,
            "max_num_matches": None,
            "pose_or_ground_truth_used": False,
            "render": False,
        },
    }


def test_half_pixel_coordinate_round_trip_is_exact_up_to_float_tolerance() -> None:
    xy = np.asarray([[0.0, 0.0], [1919.0, 1079.0], [631.25, 412.75]], dtype=np.float32)
    resized = resize_pixel_centers_half_pixel(
        xy, source_size=(1920, 1080), resized_size=(960, 540)
    )
    restored = restore_pixel_centers_half_pixel(
        resized, source_size=(1920, 1080), resized_size=(960, 540)
    )
    assert np.allclose(restored, xy, atol=1e-5)


def test_split_batched_loftr_matches_preserves_pair_order_and_rejects_truncation() -> None:
    split = split_batched_loftr_matches(
        query_xy=np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        support_xy=np.asarray([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        confidence=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        batch_indices=np.asarray([1, 0, 1], dtype=np.int64),
        batch_size=2,
    )
    assert split[0][0].tolist() == [[3.0, 4.0]]
    assert split[1][2].tolist() == pytest.approx([0.9, 0.7])
    with pytest.raises(ValueError, match="truncated"):
        split_batched_loftr_matches(
            query_xy=np.zeros((2, 2), dtype=np.float32),
            support_xy=np.zeros((2, 2), dtype=np.float32),
            confidence=np.ones((2,), dtype=np.float32),
            batch_indices=np.asarray([0, 1, 1], dtype=np.int64),
            batch_size=2,
        )


def test_cache_loader_preserves_csr_support_pairs(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        query_id=np.asarray(["query.png"]),
        support_image_ids=np.asarray(["support-a.png", "support-b.png"]),
        match_offsets=np.asarray([0, 1, 3], dtype=np.int64),
        query_match_xy=np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        support_match_xy=np.asarray([[7, 8], [9, 10], [11, 12]], dtype=np.float32),
        match_confidence=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(_metadata())),
    )
    cache = load_frozen_loftr_pair_cache(path)
    assert cache.image_index("support-b.png") == 1
    query, support, score = cache.matches_for_index(1)
    assert query.tolist() == [[3.0, 4.0], [5.0, 6.0]]
    assert support.tolist() == [[9.0, 10.0], [11.0, 12.0]]
    assert score.tolist() == pytest.approx([0.8, 0.7])
