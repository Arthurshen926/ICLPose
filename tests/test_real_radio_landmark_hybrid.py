from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization.landmark_hybrid import (
    LandmarkOwnerObservationIndex,
    LandmarkRetrievalConfig,
    LandmarkSearchIndexCache,
    _submap_cache_key,
    build_projected_observation_landmark_index,
    landmark_match_measurement_quality,
    load_landmark_index_npz,
    match_query_tokens_to_landmarks_ann,
    project_landmark_index_features,
    refine_landmark_match_batches_with_measurement,
    refine_landmark_matches_with_measurement,
    rescore_landmark_matches_for_measurement,
    save_landmark_index_npz,
    select_measurement_candidates_from_inlier_mask,
    select_quality_spatial_landmark_submap,
    select_measurement_candidates_spatially,
    select_query_tokens_for_landmark_retrieval,
    select_submap_with_global_fallback,
)
from feature_extract.vfm.localization.schemas import MappedFeatureMap
from feature_extract.vfm.localization.schemas import MeasurementResult
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec, write_npz_token_record


def _track_observation(track_id: int, image_id: str, xy: tuple[float, float]) -> ColmapTrackObservation:
    return ColmapTrackObservation(
        track_id=track_id,
        image_id=image_id,
        point2d_idx=track_id,
        xy=xy,
        xyz=np.asarray([float(track_id), 1.0, 4.0], dtype=np.float64),
        track_length=5,
        reprojection_error=0.5,
        camera_id=1,
        image_width=100,
        image_height=50,
    )


def _token_record(tmp_path, image_id: str, feature_map: np.ndarray) -> TokenBankRecord:
    path = tmp_path / f"{image_id}.npz"
    write_npz_token_record(path, {"radio_final": np.asarray(feature_map, dtype=np.float32)})
    return TokenBankRecord(
        image_id=image_id,
        token_path=path,
        layers=(TokenLayerSpec(name="radio_final", model="radio", layer="final", channels=int(feature_map.shape[0]), stride=8),),
        split="train",
        scene="unit",
    )


class _ScaleMapper:
    def __init__(self, scale: float) -> None:
        self.scale = float(scale)
        self.call_count = 0

    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        self.call_count += 1
        projected = np.asarray(feature_map, dtype=np.float32) * self.scale
        return MappedFeatureMap(
            coarse_descriptors=projected,
            measurement_context=projected,
            offset_logits=None,
            heatmap=np.ones(projected.shape[1:], dtype=np.float32),
        )


def test_owner_observation_index_scales_xy_to_target_image_size() -> None:
    index = LandmarkOwnerObservationIndex(
        [_track_observation(7, "seq/r.png", (25.0, 10.0))],
        target_image_sizes={"seq/r.png": (200, 150)},
    )

    owner = index.select(7)

    assert owner is not None
    assert owner.track_id == 7
    assert owner.image_id == "seq/r.png"
    assert owner.image_size == (200, 150)
    np.testing.assert_allclose(owner.xy, [50.0, 30.0])


def test_full_bank_faiss_cache_key_is_shared_across_queries() -> None:
    key_a = _submap_cache_key(
        query_id="seq/q0.png",
        references=[],
        selection_mode="quality_spatial",
        max_submap_landmarks=None,
        grid_rows=8,
        grid_cols=8,
    )
    key_b = _submap_cache_key(
        query_id="seq/q1.png",
        references=[],
        selection_mode="quality_spatial",
        max_submap_landmarks=None,
        grid_rows=8,
        grid_cols=8,
    )
    key_c = _submap_cache_key(
        query_id="seq/q1.png",
        references=["seq/r.png"],
        selection_mode="quality_spatial",
        max_submap_landmarks=None,
        grid_rows=8,
        grid_cols=8,
    )

    assert key_a == key_b
    assert key_a != key_c


def test_project_landmark_index_with_identity_mapper_preserves_metadata() -> None:
    bank = SelectedTrackFeatureBank(
        feature_dim=2,
        tracks={
            2: TrackFeature(
                track_id=2,
                mean_feature=np.asarray([1.0, 2.0], dtype=np.float32),
                variance=np.asarray([0.1, 0.2], dtype=np.float32),
                observation_count=3,
                mean_utility=1.0,
                observation_image_ids=("a.png",),
            )
        },
    )
    index = LandmarkMapIndex.from_track_bank(
        bank,
        xyz_by_track={2: np.asarray([2.0, 0.0, 5.0], dtype=np.float64)},
        reprojection_error_by_track={2: 0.25},
    )

    projected = project_landmark_index_features(index, lambda values: values * 2.0)

    np.testing.assert_array_equal(projected.track_ids, index.track_ids)
    np.testing.assert_allclose(projected.xyz, index.xyz)
    np.testing.assert_allclose(projected.features, [[2.0, 4.0]])
    np.testing.assert_allclose(projected.mean_variances, index.mean_variances)
    np.testing.assert_array_equal(projected.observation_counts, index.observation_counts)
    assert projected.observation_image_ids == index.observation_image_ids


def test_projected_observation_landmark_index_samples_full_map_mapper_outputs(tmp_path) -> None:
    manifest = TokenBankManifest(
        records=(
            _token_record(
                tmp_path,
                "a.png",
                np.asarray(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[10.0, 20.0], [30.0, 40.0]],
                    ],
                    dtype=np.float32,
                ),
            ),
            _token_record(
                tmp_path,
                "b.png",
                np.asarray(
                    [
                        [[5.0, 6.0], [7.0, 8.0]],
                        [[50.0, 60.0], [70.0, 80.0]],
                    ],
                    dtype=np.float32,
                ),
            ),
        )
    )
    observations = [
        _track_observation(3, "a.png", (99.0, 49.0)),
        _track_observation(3, "b.png", (99.0, 49.0)),
    ]
    mapper = _ScaleMapper(scale=2.0)

    index, metadata = build_projected_observation_landmark_index(
        observations,
        manifest,
        mapper,
        feature_key="radio_final",
        min_observations=2,
        aggregation_method="mean",
        sample_mode="nearest",
    )

    assert mapper.call_count == 2
    assert metadata["projection_mode"] == "full_map_projected_observations"
    assert list(index.track_ids) == [3]
    np.testing.assert_allclose(index.xyz, [[3.0, 1.0, 4.0]])
    np.testing.assert_allclose(index.features, [[12.0, 120.0]])
    np.testing.assert_array_equal(index.observation_counts, [2])
    assert index.observation_image_ids == (("a.png", "b.png"),)


def test_landmark_index_npz_roundtrip_preserves_metadata(tmp_path) -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([2, 5], dtype=np.int64),
        xyz=np.asarray([[2.0, 0.0, 5.0], [5.0, 1.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.1, 0.2], dtype=np.float32),
        observation_counts=np.asarray([3, 4], dtype=np.int64),
        observation_image_ids=(("a.png", "b.png"), ("c.png",)),
        reprojection_errors=np.asarray([0.25, 0.5], dtype=np.float32),
        feature_ambiguities=np.asarray([0.05, 0.1], dtype=np.float32),
    )
    path = tmp_path / "projected_landmarks.npz"

    save_landmark_index_npz(index, path, metadata={"projection": "joint"})
    loaded, metadata = load_landmark_index_npz(path)

    np.testing.assert_array_equal(loaded.track_ids, index.track_ids)
    np.testing.assert_allclose(loaded.xyz, index.xyz)
    np.testing.assert_allclose(loaded.features, index.features)
    np.testing.assert_allclose(loaded.mean_variances, index.mean_variances)
    np.testing.assert_array_equal(loaded.observation_counts, index.observation_counts)
    assert loaded.observation_image_ids == index.observation_image_ids
    np.testing.assert_allclose(loaded.reprojection_errors, index.reprojection_errors)
    np.testing.assert_allclose(loaded.feature_ambiguities, index.feature_ambiguities)
    assert metadata["projection"] == "joint"


def test_ann_landmark_matcher_returns_nearest_tracks() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([11, 22], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.1, 0.1], dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("r0.png",), ("r1.png",)),
        reprojection_errors=np.asarray([0.2, 0.2], dtype=np.float32),
        feature_ambiguities=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    query = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    config = LandmarkRetrievalConfig(
        backend="exact",
        top_k=2,
        ratio_threshold=None,
        min_similarity=0.5,
        query_token_step=1,
        max_matches=2,
    )

    matches, metadata = match_query_tokens_to_landmarks_ann(
        query,
        index,
        config,
        image_width=20,
        image_height=10,
    )

    assert metadata["backend"] == "exact"
    assert [match.track_id for match in matches] == [11, 22]
    assert [match.source for match in matches] == ["landmark_exact", "landmark_exact"]


def test_ann_landmark_matcher_proposal_top_l_keeps_multiple_landmarks_per_query_token() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([11, 22, 33], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [2.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        observation_counts=np.asarray([3, 3, 3], dtype=np.int64),
        observation_image_ids=(("r0.png",), ("r1.png",), ("r2.png",)),
        reprojection_errors=np.asarray([0.2, 0.2, 0.2], dtype=np.float32),
        feature_ambiguities=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    query = np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)
    config = LandmarkRetrievalConfig(
        backend="exact",
        top_k=2,
        nn_search_k_for_ratio=3,
        proposal_top_l=2,
        ratio_threshold=None,
        min_similarity=0.1,
        query_token_step=1,
        max_matches=4,
        deduplicate_tracks=False,
    )

    matches, metadata = match_query_tokens_to_landmarks_ann(
        query,
        index,
        config,
        image_width=20,
        image_height=10,
    )

    assert [match.track_id for match in matches[:2]] == [11, 22]
    assert [match.coarse_rank for match in matches[:2]] == [1, 2]
    assert metadata["nn_search_k_for_ratio"] == 3
    assert metadata["proposal_top_l"] == 2


def test_heatmap_query_token_selector_uses_nms_and_spatial_quota() -> None:
    feature_map = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[0, 0] = 0.99
    heatmap[0, 1] = 0.98
    heatmap[3, 3] = 0.60
    config = LandmarkRetrievalConfig(
        backend="exact",
        query_token_selection="heatmap",
        query_heatmap_top_k=2,
        query_heatmap_nms_radius=1,
        query_heatmap_grid_rows=1,
        query_heatmap_grid_cols=2,
    )

    _features, _xy, token_indices, metadata = select_query_tokens_for_landmark_retrieval(
        feature_map,
        image_width=40,
        image_height=40,
        config=config,
        query_heatmap=heatmap,
    )

    assert token_indices.tolist() == [0, 15]
    assert metadata["query_token_selection"] == "heatmap"
    assert metadata["selected_query_token_count"] == 2


def test_faiss_landmark_matcher_matches_exact_backend_when_available() -> None:
    pytest.importorskip("faiss")
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [2.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32),
        mean_variances=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        observation_counts=np.asarray([3, 3, 3], dtype=np.int64),
        observation_image_ids=(("r0.png",), ("r1.png",), ("r2.png",)),
        reprojection_errors=np.asarray([0.2, 0.2, 0.2], dtype=np.float32),
        feature_ambiguities=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    query = np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)
    common = dict(top_k=2, ratio_threshold=None, min_similarity=0.1, query_token_step=1, max_matches=1)

    exact, _ = match_query_tokens_to_landmarks_ann(
        query,
        index,
        LandmarkRetrievalConfig(backend="exact", **common),
        image_width=20,
        image_height=10,
    )
    faiss_matches, metadata = match_query_tokens_to_landmarks_ann(
        query,
        index,
        LandmarkRetrievalConfig(backend="faiss", **common),
        image_width=20,
        image_height=10,
    )

    assert metadata["backend"] == "faiss_flat_ip"
    assert [match.track_id for match in faiss_matches] == [match.track_id for match in exact]


def test_landmark_search_index_cache_reuses_same_submap_key() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.1, 0.1], dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("r0.png",), ("r1.png",)),
        reprojection_errors=np.asarray([0.2, 0.2], dtype=np.float32),
        feature_ambiguities=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    query = np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)
    cache = LandmarkSearchIndexCache(max_entries=2)
    config = LandmarkRetrievalConfig(
        backend="exact",
        top_k=1,
        ratio_threshold=None,
        min_similarity=0.1,
        query_token_step=1,
        max_matches=1,
    )

    _matches, first = match_query_tokens_to_landmarks_ann(
        query,
        index,
        config,
        image_width=20,
        image_height=10,
        index_cache=cache,
        cache_key=("refs", "r0", "r1"),
    )
    _matches, second = match_query_tokens_to_landmarks_ann(
        query,
        index,
        config,
        image_width=20,
        image_height=10,
        index_cache=cache,
        cache_key=("refs", "r0", "r1"),
    )

    assert first["index_cache_hit"] is False
    assert second["index_cache_hit"] is True
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 1


def test_quality_spatial_submap_selection_keeps_good_landmarks_across_cells() -> None:
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.01, 0.01, 0.05], dtype=np.float32),
        observation_counts=np.asarray([10, 9, 3], dtype=np.int64),
        observation_image_ids=(("a.png",), ("a.png",), ("b.png",)),
        reprojection_errors=np.asarray([0.1, 0.1, 0.5], dtype=np.float32),
        feature_ambiguities=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )

    selected, metadata = select_quality_spatial_landmark_submap(
        index,
        max_landmarks=2,
        grid_rows=1,
        grid_cols=2,
    )

    assert list(selected.track_ids) == [1, 3]
    assert metadata["input_landmark_count"] == 3
    assert metadata["output_landmark_count"] == 2
    assert metadata["spatial_cell_count"] == 2


def test_submap_global_fallback_adds_landmarks_when_geometry_coverage_is_low() -> None:
    full = LandmarkMapIndex(
        track_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.01, 0.01, 0.05], dtype=np.float32),
        observation_counts=np.asarray([10, 9, 3], dtype=np.int64),
        observation_image_ids=(("a.png",), ("a.png",), ("b.png",)),
        reprojection_errors=np.asarray([0.1, 0.1, 0.5], dtype=np.float32),
        feature_ambiguities=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    retrieval_only = full.subset([0, 1])

    selected, metadata = select_submap_with_global_fallback(
        retrieval_only,
        full,
        max_landmarks=2,
        selection_mode="quality_spatial",
        grid_rows=1,
        grid_cols=2,
        min_landmarks=2,
        min_spatial_cells=2,
        fallback_fraction=0.5,
    )

    assert list(selected.track_ids) == [1, 3]
    assert metadata["fallback_applied"] is True
    assert metadata["fallback_added_landmark_count"] == 1
    assert metadata["selected_spatial_cell_count"] == 2


def test_spatial_measurement_candidate_selection_covers_multiple_cells() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=idx,
            xy=np.asarray(xy, dtype=np.float64),
            track_id=idx,
            xyz=np.asarray([float(idx), 0.0, 5.0], dtype=np.float64),
            similarity=score,
            ratio=1.0,
            landmark_variance=0.1,
        )
        for idx, xy, score in [
            (1, (5.0, 5.0), 0.99),
            (2, (8.0, 6.0), 0.98),
            (3, (90.0, 6.0), 0.5),
        ]
    ]

    selected, skipped = select_measurement_candidates_spatially(
        matches,
        image_width=100,
        image_height=20,
        max_count=2,
        grid_rows=1,
        grid_cols=2,
    )

    assert [match.track_id for match in selected] == [1, 3]
    assert [match.track_id for match in skipped] == [2]


def test_measurement_candidate_scoring_combines_heatmap_quality_and_measurement_confidence() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([5.0, 5.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([1.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.99,
            ratio=1.0,
            landmark_variance=0.5,
            observation_count=2,
            landmark_reprojection_error=5.0,
            query_heatmap_score=0.2,
            patch_offset_confidence=0.2,
            measurement_sigma_px=8.0,
        ),
        QueryTo3DMatch(
            token_index=2,
            xy=np.asarray([90.0, 5.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([2.0, 0.0, 5.0], dtype=np.float64),
            similarity=0.70,
            ratio=1.0,
            landmark_variance=0.01,
            observation_count=12,
            landmark_reprojection_error=0.1,
            query_heatmap_score=0.95,
            patch_offset_confidence=0.9,
            measurement_sigma_px=0.5,
        ),
    ]

    rescored = rescore_landmark_matches_for_measurement(matches)
    selected, skipped = select_measurement_candidates_spatially(
        rescored,
        image_width=100,
        image_height=20,
        max_count=1,
        grid_rows=1,
        grid_cols=1,
        score_mode="measurement_quality",
    )

    assert selected[0].track_id == 2
    assert skipped[0].track_id == 1
    assert selected[0].pnp_soft_score is not None
    assert selected[0].pnp_soft_score > skipped[0].pnp_soft_score


class _FakeMeasurementAdapter:
    def measure(self, query_rgb, reference_rgb, proposals, **_kwargs):
        return [
            MeasurementResult(
                proposal=proposal,
                measured_query_xy=proposal.query_xy + np.asarray([1.5, -0.5], dtype=np.float32),
                measured_reference_xy=proposal.reference_xy,
                confidence=0.8,
                uncertainty_px=0.25,
            )
            for proposal in proposals
        ]


class _FakeGroupedMeasurementAdapter:
    def __init__(self) -> None:
        self.measure_called = False
        self.grouped_reference_ids: list[str] = []

    def measure(self, query_rgb, reference_rgb, proposals, **_kwargs):
        del query_rgb, reference_rgb, proposals, _kwargs
        self.measure_called = True
        raise AssertionError("refine should prefer measure_by_reference when available")

    def measure_by_reference(self, query_rgb, reference_rgb_by_id, proposals_by_reference):
        del query_rgb, reference_rgb_by_id
        self.grouped_reference_ids = list(proposals_by_reference)
        return {
            image_id: [
                MeasurementResult(
                    proposal=proposal,
                    measured_query_xy=proposal.query_xy + np.asarray([0.25, 0.5], dtype=np.float32),
                    measured_reference_xy=proposal.reference_xy,
                    confidence=0.9,
                    uncertainty_px=0.5,
                )
                for proposal in proposals
            ]
            for image_id, proposals in proposals_by_reference.items()
        }


class _FakeManyMeasurementAdapter:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def measure_many_by_reference(self, query_rgb_by_id, reference_rgb_by_id, proposals_by_query_reference):
        del query_rgb_by_id, reference_rgb_by_id
        self.pairs = list(proposals_by_query_reference)
        return {
            pair: [
                MeasurementResult(
                    proposal=proposal,
                    measured_query_xy=proposal.query_xy + np.asarray([0.75, -0.25], dtype=np.float32),
                    measured_reference_xy=proposal.reference_xy,
                    confidence=0.85,
                    uncertainty_px=0.75,
                )
                for proposal in proposals
            ]
            for pair, proposals in proposals_by_query_reference.items()
        }


class _LowGeometryProbabilityModel:
    def predict_match(self, match) -> float:
        return 0.1 if int(match.track_id) == 9 else 0.9


def test_refine_matches_with_measurement_updates_query_xy_and_preserves_xyz() -> None:
    xyz = np.asarray([3.0, 4.0, 5.0], dtype=np.float64)
    match = QueryTo3DMatch(
        token_index=4,
        xy=np.asarray([10.0, 20.0], dtype=np.float64),
        track_id=9,
        xyz=xyz,
        similarity=0.7,
        ratio=1.0,
        landmark_variance=0.1,
        pnp_soft_score=0.7,
    )
    owner_index = LandmarkOwnerObservationIndex(
        [_track_observation(9, "seq/r.png", (5.0, 6.0))],
        target_image_sizes={"seq/r.png": (100, 50)},
    )
    images = {"seq/r.png": np.zeros((3, 50, 100), dtype=np.float32)}

    refined, rows = refine_landmark_matches_with_measurement(
        query_id="seq/q.png",
        query_rgb=np.zeros((3, 50, 100), dtype=np.float32),
        matches=[match],
        owner_index=owner_index,
        measurement_adapter=_FakeMeasurementAdapter(),
        reference_image_loader=lambda image_id: images[image_id],
    )

    assert len(refined) == 1
    assert len(rows) == 1
    assert rows[0]["measurement_status"] == "measured"
    np.testing.assert_allclose(refined[0].xy, [11.5, 19.5])
    np.testing.assert_allclose(refined[0].xyz, xyz)
    assert refined[0].track_id == 9
    assert refined[0].pnp_soft_score == pytest.approx(landmark_match_measurement_quality(refined[0]))
    assert refined[0].pnp_soft_score != pytest.approx(refined[0].patch_offset_confidence)
    assert refined[0].patch_offset_confidence == 0.8
    assert refined[0].measurement_sigma_px == 0.25


def test_refine_matches_with_measurement_can_drop_low_geometry_probability() -> None:
    match = QueryTo3DMatch(
        token_index=4,
        xy=np.asarray([10.0, 20.0], dtype=np.float64),
        track_id=9,
        xyz=np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
        similarity=0.7,
        ratio=1.0,
        landmark_variance=0.1,
        pnp_soft_score=0.7,
    )
    owner_index = LandmarkOwnerObservationIndex(
        [_track_observation(9, "seq/r.png", (5.0, 6.0))],
        target_image_sizes={"seq/r.png": (100, 50)},
    )

    refined, rows = refine_landmark_matches_with_measurement(
        query_id="seq/q.png",
        query_rgb=np.zeros((3, 50, 100), dtype=np.float32),
        matches=[match],
        owner_index=owner_index,
        measurement_adapter=_FakeMeasurementAdapter(),
        reference_image_loader=lambda _image_id: np.zeros((3, 50, 100), dtype=np.float32),
        measurement_geometry_model=_LowGeometryProbabilityModel(),
        min_measurement_geometry_probability=0.5,
        drop_rejected_measurements=True,
    )

    assert refined == []
    assert rows[0]["measurement_status"] == "rejected_low_geometry_probability"
    assert rows[0]["geometry_probability"] == pytest.approx(0.1)
    assert rows[0]["measurement_kept"] is False


def test_refine_matches_with_measurement_uses_grouped_reference_measurement() -> None:
    matches = [
        QueryTo3DMatch(
            token_index=index,
            xy=np.asarray([10.0 + index, 20.0], dtype=np.float64),
            track_id=track_id,
            xyz=np.asarray([float(track_id), 0.0, 5.0], dtype=np.float64),
            similarity=0.7,
            ratio=1.0,
            landmark_variance=0.1,
            pnp_soft_score=0.7,
        )
        for index, track_id in enumerate((9, 10))
    ]
    owner_index = LandmarkOwnerObservationIndex(
        [
            _track_observation(9, "seq/r0.png", (5.0, 6.0)),
            _track_observation(10, "seq/r1.png", (7.0, 8.0)),
        ],
        target_image_sizes={"seq/r0.png": (100, 50), "seq/r1.png": (100, 50)},
    )
    images = {
        "seq/r0.png": np.zeros((3, 50, 100), dtype=np.float32),
        "seq/r1.png": np.zeros((3, 50, 100), dtype=np.float32),
    }
    adapter = _FakeGroupedMeasurementAdapter()

    refined, rows = refine_landmark_matches_with_measurement(
        query_id="seq/q.png",
        query_rgb=np.zeros((3, 50, 100), dtype=np.float32),
        matches=matches,
        owner_index=owner_index,
        measurement_adapter=adapter,
        reference_image_loader=lambda image_id: images[image_id],
    )

    assert not adapter.measure_called
    assert adapter.grouped_reference_ids == ["seq/r0.png", "seq/r1.png"]
    assert len(refined) == 2
    assert len(rows) == 2
    np.testing.assert_allclose(refined[0].xy, [10.25, 20.5])
    np.testing.assert_allclose(refined[1].xy, [11.25, 20.5])


def test_refine_match_batches_with_measurement_uses_cross_query_adapter() -> None:
    queries = []
    for query_id, track_id, x in [("seq/q0.png", 9, 10.0), ("seq/q1.png", 10, 20.0)]:
        match = QueryTo3DMatch(
            token_index=track_id,
            xy=np.asarray([x, 30.0], dtype=np.float64),
            track_id=track_id,
            xyz=np.asarray([float(track_id), 0.0, 5.0], dtype=np.float64),
            similarity=0.7,
            ratio=1.0,
            landmark_variance=0.1,
            pnp_soft_score=0.7,
        )
        queries.append((query_id, np.zeros((3, 50, 100), dtype=np.float32), [match]))
    owner_index = LandmarkOwnerObservationIndex(
        [
            _track_observation(9, "seq/r0.png", (5.0, 6.0)),
            _track_observation(10, "seq/r1.png", (7.0, 8.0)),
        ],
        target_image_sizes={"seq/r0.png": (100, 50), "seq/r1.png": (100, 50)},
    )
    images = {
        "seq/r0.png": np.zeros((3, 50, 100), dtype=np.float32),
        "seq/r1.png": np.zeros((3, 50, 100), dtype=np.float32),
    }
    adapter = _FakeManyMeasurementAdapter()

    refined = refine_landmark_match_batches_with_measurement(
        queries,
        owner_index=owner_index,
        measurement_adapter=adapter,
        reference_image_loader=lambda image_id: images[image_id],
    )

    assert adapter.pairs == [("seq/q0.png", "seq/r0.png"), ("seq/q1.png", "seq/r1.png")]
    assert set(refined) == {"seq/q0.png", "seq/q1.png"}
    np.testing.assert_allclose(refined["seq/q0.png"][0][0].xy, [10.75, 29.75])
    np.testing.assert_allclose(refined["seq/q1.png"][0][0].xy, [20.75, 29.75])
    assert refined["seq/q0.png"][0][0].patch_offset_confidence == 0.85
    assert refined["seq/q0.png"][0][0].pnp_soft_score == pytest.approx(
        landmark_match_measurement_quality(refined["seq/q0.png"][0][0])
    )


def test_refine_match_batches_with_measurement_preserves_missing_owner_matches() -> None:
    match = QueryTo3DMatch(
        token_index=4,
        xy=np.asarray([10.0, 30.0], dtype=np.float64),
        track_id=99,
        xyz=np.asarray([99.0, 0.0, 5.0], dtype=np.float64),
        similarity=0.7,
        ratio=1.0,
        landmark_variance=0.1,
        pnp_soft_score=0.7,
    )
    owner_index = LandmarkOwnerObservationIndex([], target_image_sizes={})
    adapter = _FakeManyMeasurementAdapter()

    refined = refine_landmark_match_batches_with_measurement(
        [("seq/q0.png", np.zeros((3, 50, 100), dtype=np.float32), [match])],
        owner_index=owner_index,
        measurement_adapter=adapter,
        reference_image_loader=lambda _image_id: np.zeros((3, 50, 100), dtype=np.float32),
    )

    assert adapter.pairs == []
    assert len(refined["seq/q0.png"][0]) == 1
    assert refined["seq/q0.png"][0][0] is match
    assert refined["seq/q0.png"][1][0]["measurement_status"] == "missing_owner_observation"


def test_select_measurement_candidates_from_inlier_mask_prioritizes_pnp_inliers() -> None:
    matches = []
    for index, (x, score) in enumerate([(5.0, 0.99), (25.0, 0.7), (45.0, 0.98), (65.0, 0.65)]):
        matches.append(
            QueryTo3DMatch(
                token_index=index,
                xy=np.asarray([x, 10.0], dtype=np.float64),
                track_id=index,
                xyz=np.asarray([float(index), 0.0, 5.0], dtype=np.float64),
                similarity=score,
                ratio=1.0,
                landmark_variance=0.1,
                pnp_soft_score=score,
            )
        )

    selected, skipped, metadata = select_measurement_candidates_from_inlier_mask(
        matches,
        inlier_mask=np.asarray([False, True, False, True]),
        image_width=100,
        image_height=20,
        max_count=3,
        grid_rows=1,
        grid_cols=4,
        score_mode="match",
    )

    assert [match.track_id for match in selected[:2]] == [1, 3]
    assert selected[2].track_id == 0
    assert {match.track_id for match in skipped} == {2}
    assert metadata["coarse_pnp_inlier_count"] == 2
    assert metadata["measurement_filled_from_non_inliers"] == 1
