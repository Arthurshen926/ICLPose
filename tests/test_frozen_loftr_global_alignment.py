from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization.frozen_loftr_global_alignment import (
    LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES,
    fit_frozen_loftr_support_to_query_homography,
    frozen_loftr_global_alignment_view_features,
    loftr_global_alignment_pair_features,
)
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    FROZEN_LOFTR_PAIR_CACHE_FORMAT,
    FrozenLoFTRPairCache,
)


def _transform(points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(
        [[1.05, 0.03, 11.0], [-0.02, 0.98, 7.0], [0.0001, -0.00005, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ matrix.T
    return projected[:, :2] / projected[:, 2:]


def _cache() -> FrozenLoFTRPairCache:
    support = np.asarray(
        [[x, y] for x in (10.0, 40.0, 80.0, 130.0) for y in (15.0, 50.0, 100.0, 150.0)],
        dtype=np.float32,
    )
    query = _transform(support).astype(np.float32)
    return FrozenLoFTRPairCache(
        path=Path("/tmp/loftr-global-cache.npz"),
        query_id="query.png",
        support_image_ids=np.asarray(["support-a.png", "support-b.png"]),
        match_offsets=np.asarray([0, len(support), len(support)], dtype=np.int64),
        query_match_xy=query,
        support_match_xy=support,
        match_confidence=np.full((len(support),), 0.9, dtype=np.float32),
        metadata={
            "format": FROZEN_LOFTR_PAIR_CACHE_FORMAT,
            "contains_target_fields": False,
            "supervision_arrays_loaded": False,
            "strict_global_pair_contract": {
                "all_mapping_manifest_images_processed": True,
                "image_level_selection": False,
                "max_num_matches": None,
                "pose_or_ground_truth_used": False,
                "render": False,
            },
        },
    )


def test_global_homography_anchor_error_distinguishes_global_phase() -> None:
    support = np.asarray(
        [[x, y] for x in (10.0, 40.0, 80.0, 130.0) for y in (15.0, 50.0, 100.0, 150.0)],
        dtype=np.float32,
    )
    query = _transform(support).astype(np.float32)
    model = fit_frozen_loftr_support_to_query_homography(
        matched_query_xy=query,
        matched_support_xy=support,
        match_confidence=np.full((len(support),), 0.9, dtype=np.float32),
        support_image_id="support-a.png",
        source_size=(1920, 1080),
    )
    assert model is not None
    assert model.inlier_count == len(support)
    values, usable = loftr_global_alignment_pair_features(
        model=model,
        query_xy=query[[3, 7]],
        support_xy=np.asarray([support[3], support[7] + [40.0, 0.0]], dtype=np.float32),
    )
    assert usable.tolist() == [True, True]
    assert values.shape == (2, len(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES))
    assert values[0, 0] < 1e-3
    assert values[0, 1] < 1e-3
    assert values[1, 0] > 20.0
    assert values[0, 3] == pytest.approx(1.0)


def test_global_alignment_keeps_empty_pair_as_unknown() -> None:
    cache = _cache()
    values, usable, counts, model_valid = frozen_loftr_global_alignment_view_features(
        cache=cache,
        query_xy=_transform(np.asarray([[10.0, 15.0]], dtype=np.float32)).astype(np.float32),
        candidate_support_xy=np.asarray([[[[10.0, 15.0], [20.0, 20.0]]]], dtype=np.float32),
        candidate_support_image_ids=np.asarray([[["support-a.png", "support-b.png"]]]),
        candidate_view_valid=np.asarray([[[True, True]]]),
        source_size=(1920, 1080),
    )
    assert model_valid.tolist() == [[[True, False]]]
    assert usable.tolist() == [[[True, False]]]
    assert counts.tolist() == [[[16, 0]]]
    assert values[0, 0, 0, 0] < 1e-3
    assert np.isnan(values[0, 0, 1]).all()
