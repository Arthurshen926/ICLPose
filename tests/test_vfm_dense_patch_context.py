import numpy as np

from feature_extract.vfm.dense_patch_context import (
    DensePatchContextConfig,
    build_dense_patch_context_bank,
    dense_context_scores,
    rerank_anchor_candidates_with_dense_context,
)
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMSource
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorConfig, build_sfm_guided_semidense_anchor_map


def _toy_sparse() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 11], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.01, 0.02], dtype=np.float32),
        observation_counts=np.asarray([5, 4], dtype=np.int64),
        observation_image_ids=(("a.png", "b.png"), ("a.png",)),
        reprojection_errors=np.asarray([0.2, 0.3], dtype=np.float32),
    )


def _toy_semidense():
    gaussians = GaussianVFMSource(
        xyz=np.asarray(
            [[0.03, 0.0, 4.0], [0.06, 0.0, 4.0], [1.03, 0.0, 4.0], [1.06, 0.0, 4.0]],
            dtype=np.float64,
        ),
        opacity=np.asarray([0.9, 0.8, 0.9, 0.8], dtype=np.float32),
        scale=np.asarray([0.02, 0.02, 0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([100, 101, 102, 103], dtype=np.int64),
    )
    return build_sfm_guided_semidense_anchor_map(
        _toy_sparse(),
        gaussians,
        SemiDenseAnchorConfig(max_distance=0.10, k_neighbors=1, min_opacity=0.1),
    )


def test_dense_patch_context_bank_aggregates_gaussian_support_per_sparse_anchor() -> None:
    bank = build_dense_patch_context_bank(
        _toy_sparse(),
        _toy_semidense(),
        DensePatchContextConfig(max_radius_m=0.12, max_support=4, prototype_count=2),
    )

    assert bank.track_ids.tolist() == [10, 11]
    assert bank.support_counts.tolist() == [2, 2]
    assert bank.prototype_features.shape == (2, 2, 2)
    assert np.allclose(bank.mean_features[0], np.asarray([1.0, 0.0], dtype=np.float32), atol=1e-5)
    assert np.all(bank.reliability_scores >= 0.0)
    assert np.all(bank.reliability_scores <= 1.0)


def test_dense_context_scores_support_mean_medoid_and_prototype_modes() -> None:
    bank = build_dense_patch_context_bank(
        _toy_sparse(),
        _toy_semidense(),
        DensePatchContextConfig(max_radius_m=0.12, max_support=4, prototype_count=2),
    )

    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = dense_context_scores(query, bank, np.asarray([0, 1], dtype=np.int64), mode="max_proto")

    assert scores.shape == (2,)
    assert np.all(scores > 0.99)


def test_dense_context_rerank_changes_only_topm_sparse_anchor_candidates() -> None:
    bank = build_dense_patch_context_bank(
        _toy_sparse(),
        _toy_semidense(),
        DensePatchContextConfig(max_radius_m=0.12, max_support=4, prototype_count=2),
    )
    candidates = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([10.0, 10.0]),
            track_id=10,
            xyz=np.asarray([0.0, 0.0, 4.0]),
            similarity=0.61,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([10.0, 10.0]),
            track_id=11,
            xyz=np.asarray([1.0, 0.0, 4.0]),
            similarity=0.60,
            ratio=0.0,
            landmark_variance=0.0,
        ),
    ]
    query_features = np.asarray([[0.0, 1.0]], dtype=np.float32)

    reranked = rerank_anchor_candidates_with_dense_context(
        candidates,
        query_features_by_token=query_features,
        context_bank=bank,
        context_weight=0.2,
        context_mode="max_proto",
        top_m_per_token=2,
    )

    assert [match.track_id for match in reranked] == [11, 10]
    assert all(match.xyz.shape == (3,) for match in reranked)
