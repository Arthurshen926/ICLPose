from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization.frozen_loftr_candidate_anchor_evidence import (
    LOFTR_ANCHOR_FEATURE_NAMES,
    frozen_loftr_candidate_view_features,
    loftr_anchor_pair_features,
)
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    FROZEN_LOFTR_PAIR_CACHE_FORMAT,
    FrozenLoFTRPairCache,
)


def _cache() -> FrozenLoFTRPairCache:
    return FrozenLoFTRPairCache(
        path=Path("/tmp/loftr-cache.npz"),
        query_id="query.png",
        support_image_ids=np.asarray(["support-a.png", "support-b.png"]),
        match_offsets=np.asarray([0, 2, 2], dtype=np.int64),
        query_match_xy=np.asarray([[10, 10], [90, 90]], dtype=np.float32),
        support_match_xy=np.asarray([[20, 20], [80, 80]], dtype=np.float32),
        match_confidence=np.asarray([1.0, 0.5], dtype=np.float32),
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


def test_joint_anchor_features_require_both_query_and_support_proximity() -> None:
    features, usable = loftr_anchor_pair_features(
        query_xy=np.asarray([[10, 10], [10, 10]], dtype=np.float32),
        support_xy=np.asarray([[20, 20], [80, 80]], dtype=np.float32),
        matched_query_xy=np.asarray([[10, 10]], dtype=np.float32),
        matched_support_xy=np.asarray([[20, 20]], dtype=np.float32),
        match_confidence=np.asarray([1.0], dtype=np.float32),
        chunk_size=2,
        device=torch.device("cpu"),
    )
    assert usable.tolist() == [True, True]
    assert features.shape == (2, len(LOFTR_ANCHOR_FEATURE_NAMES))
    assert features[0, 0] == pytest.approx(1.0)
    assert features[0, 4] == pytest.approx(0.0)
    assert features[1, 0] < 1e-6
    assert features[1, 6] == pytest.approx(60.0 * 2.0**0.5)


def test_candidate_view_features_keep_empty_pair_as_unknown() -> None:
    values, usable, counts = frozen_loftr_candidate_view_features(
        cache=_cache(),
        query_xy=np.asarray([[10, 10]], dtype=np.float32),
        candidate_support_xy=np.asarray([[[[20, 20], [10, 10]]]], dtype=np.float32),
        candidate_support_image_ids=np.asarray([[["support-a.png", "support-b.png"]]]),
        candidate_view_valid=np.asarray([[[True, True]]]),
        chunk_size=4,
        device=torch.device("cpu"),
    )
    assert usable.tolist() == [[[True, False]]]
    assert counts.tolist() == [[[2, 0]]]
    assert values[0, 0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(values[0, 0, 1]).all()
