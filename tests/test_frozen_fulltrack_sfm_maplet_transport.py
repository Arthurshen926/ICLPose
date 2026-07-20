from __future__ import annotations

import torch

from feature_extract.vfm.localization.frozen_fulltrack_sfm_maplet_transport import (
    SFM_MAPLET_TRANSPORT_FEATURE_NAMES,
    SFM_MAPLET_TRANSPORT_PROFILES,
    batched_sfm_maplet_quadrant_transport_features,
    pool_center_excluded_query_quadrants,
    pool_sparse_support_quadrants,
)
from feature_extract.vfm.localization.frozen_fulltrack_multiscale_translation_mode import (
    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES,
    MULTISCALE_TRANSLATION_MODE_PROFILES,
)


def test_query_quadrant_pool_excludes_center_descriptor() -> None:
    patches = torch.zeros((1, 4, 5, 5), dtype=torch.float32)
    for quadrant, (row_sign, column_sign) in enumerate(
        ((-1, -1), (-1, 1), (1, -1), (1, 1))
    ):
        rows = slice(0, 2) if row_sign < 0 else slice(3, 5)
        columns = slice(0, 2) if column_sign < 0 else slice(3, 5)
        patches[0, quadrant, rows, columns] = 1.0
    # A strong center vector would dominate every quadrant if the core local
    # descriptor were accidentally retained by the structural feature.
    patches[0, :, 2, 2] = 100.0
    pooled, usable, counts = pool_center_excluded_query_quadrants(
        patches=patches,
        valid=torch.ones((1, 5, 5), dtype=torch.bool),
        minimum_fraction=1.0,
    )
    assert usable.tolist() == [[True, True, True, True]]
    assert counts.tolist() == [[4, 4, 4, 4]]
    assert torch.argmax(pooled[0], dim=1).tolist() == [0, 1, 2, 3]


def test_quadrant_transport_is_finite_and_requires_real_support_quadrants() -> None:
    query = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    support, support_valid, counts = pool_sparse_support_quadrants(
        descriptors=query[:, :, None, :],
        present=torch.ones((1, 4, 1), dtype=torch.bool),
        minimum_neighbors=1,
    )
    assert support_valid.all()
    assert counts.tolist() == [[1, 1, 1, 1]]
    features = batched_sfm_maplet_quadrant_transport_features(
        query_quadrants=query,
        query_valid=torch.ones((1, 4), dtype=torch.bool),
        support_quadrants=support,
        support_valid=support_valid,
        temperature=0.07,
    )
    assert features.shape == (1, len(SFM_MAPLET_TRANSPORT_FEATURE_NAMES) // len(SFM_MAPLET_TRANSPORT_PROFILES))
    # The zero relative shift is the center of the 3x3 Hough lattice.
    assert torch.isclose(features[0, 4], torch.tensor(1.0), atol=1e-6)
    assert torch.isfinite(features).all()


def test_transport_features_do_not_expose_availability_or_neighbor_count_fields() -> None:
    assert len(SFM_MAPLET_TRANSPORT_FEATURE_NAMES) > 0
    assert not any(
        "coverage" in name or "count" in name or "entropy" in name
        for name in SFM_MAPLET_TRANSPORT_FEATURE_NAMES
    )


def test_multiscale_translation_mode_profiles_are_dense_and_coverage_free() -> None:
    assert {profile.source_name for profile in MULTISCALE_TRANSLATION_MODE_PROFILES} == {
        "radio_final",
        "radio_intermediate_pca256",
        "alike_fpn",
    }
    assert all(profile.window_size <= profile.grid_size for profile in MULTISCALE_TRANSLATION_MODE_PROFILES)
    assert not any(
        "coverage" in name or "count" in name or "fraction" in name
        for name in MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES
    )
