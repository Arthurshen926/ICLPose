import numpy as np

from feature_extract.vfm.local_maplet_matching import (
    ContextualLandmarkMatchingConfig,
    build_covisibility_maplets,
    build_knn_maplets,
    compute_query_context_descriptors,
    match_query_patches_to_contextual_landmarks,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _toy_index() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([20, 21, 10, 11], dtype=np.int64),
        xyz=np.asarray(
            [
                [10.0, 0.0, 4.0],
                [10.1, 0.0, 4.0],
                [0.0, 0.0, 4.0],
                [0.1, 0.0, 4.0],
            ],
            dtype=np.float64,
        ),
        features=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mean_variances=np.asarray([0.01, 0.02, 0.01, 0.02], dtype=np.float32),
        observation_counts=np.asarray([5, 4, 5, 4], dtype=np.int64),
        observation_image_ids=(
            ("ref_wrong_a.png", "ref_shared.png"),
            ("ref_wrong_a.png",),
            ("ref_correct_a.png", "ref_shared.png"),
            ("ref_correct_a.png",),
        ),
        reprojection_errors=np.asarray([0.2, 0.3, 0.2, 0.3], dtype=np.float32),
    )


def test_knn_maplets_use_nearby_3d_landmarks_as_context() -> None:
    bank = build_knn_maplets(_toy_index(), maplet_k=1, context_pool="mean")

    assert bank.maplet_type == "knn"
    assert bank.neighbor_indices.shape == (4, 1)
    assert int(bank.neighbor_indices[0, 0]) == 1
    assert int(bank.neighbor_indices[2, 0]) == 3
    assert np.allclose(bank.context_features[2], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))


def test_covisibility_maplets_use_shared_reference_views() -> None:
    bank = build_covisibility_maplets(_toy_index(), reference_image_ids=["ref_correct_a.png"], maplet_k=2)

    assert bank.maplet_type == "covis"
    assert bank.neighbor_counts.tolist() == [0, 0, 1, 1]
    assert int(bank.neighbor_indices[2, 0]) == 3
    assert int(bank.neighbor_indices[3, 0]) == 2
    assert bank.covisibility_strength[2] > 0.0


def test_query_context_descriptors_average_neighboring_tokens() -> None:
    feature_map = np.zeros((2, 3, 3), dtype=np.float32)
    feature_map[0, :, :] = 1.0
    feature_map[1, 1, 1] = 2.0

    context = compute_query_context_descriptors(feature_map, context="3x3")

    center = context[:, 1, 1]
    expected = np.asarray([1.0, 2.0 / 9.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(center, expected, atol=1e-6)


def test_contextual_matching_keeps_single_anchor_but_scores_with_local_context() -> None:
    index = _toy_index()
    maplets = build_knn_maplets(index, maplet_k=1, context_pool="mean")
    query = np.zeros((3, 3, 3), dtype=np.float32)
    query[1, :, :] = 1.0
    query[:, 1, 1] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    matches = match_query_patches_to_contextual_landmarks(
        query,
        index,
        maplets,
        ContextualLandmarkMatchingConfig(
            query_context="3x3",
            context_weight=1.0,
            match_mode="nn",
            top_k=1,
            min_similarity=-1.0,
            max_matches=None,
        ),
        image_width=30,
        image_height=30,
    )

    center_matches = [match for match in matches if match.token_index == 4]
    assert len(center_matches) == 1
    assert center_matches[0].track_id == 10
    assert center_matches[0].source == "contextual_landmark_knn_k1"
