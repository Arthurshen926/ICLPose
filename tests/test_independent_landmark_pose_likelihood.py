import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.independent_landmark_pose_likelihood import (
    IndependentPoseRefinementConfig,
    IndependentLandmarkPoseLikelihoodConfig,
    IndependentLandmarkPoseVerifier,
    IndependentVerificationPoints,
    LandmarkObservationViewIndex,
    LandmarkPrototypeViewIndex,
    deterministic_identity_folds,
    spatially_balanced_point_folds,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _bank() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([7], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((1,), dtype=np.float32),
        observation_counts=np.asarray([3], dtype=np.int64),
        observation_image_ids=(("support.png",),),
    )


def _camera() -> ColmapCamera:
    return ColmapCamera(
        camera_id=1,
        model_id=1,
        width=100,
        height=100,
        params=(80.0, 80.0, 50.0, 50.0),
    )


def _points() -> IndependentVerificationPoints:
    return IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([0.0], dtype=np.float32),
        source_row_indices=np.asarray([11], dtype=np.int64),
    )


def test_view_consistent_landmark_produces_high_likelihood() -> None:
    bank = _bank()
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids,
        np.asarray([7], dtype=np.int64),
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    verifier = IndependentLandmarkPoseVerifier(bank, views)

    score = verifier.score_pose(np.eye(4), _camera(), _points())

    assert score.verification_point_count == 1
    assert score.effective_point_count == 1
    assert score.evidence_coverage == pytest.approx(1.0)
    assert score.log_likelihood_mean > -1e-6


def test_view_inconsistent_landmark_cannot_explain_point() -> None:
    bank = _bank()
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids,
        np.asarray([7], dtype=np.int64),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    config = IndependentLandmarkPoseLikelihoodConfig(
        maximum_view_angle_deg=15.0
    )
    verifier = IndependentLandmarkPoseVerifier(bank, views, config)

    score = verifier.score_pose(np.eye(4), _camera(), _points())

    assert score.effective_point_count == 0
    assert score.log_likelihood_mean == pytest.approx(np.log(0.01))


def test_excluded_fit_track_cannot_leak_into_verification() -> None:
    bank = _bank()
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids,
        np.asarray([7], dtype=np.int64),
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    verifier = IndependentLandmarkPoseVerifier(bank, views)
    eligible = verifier.eligible_mask_excluding_tracks(
        np.asarray([7], dtype=np.int64)
    )

    score = verifier.score_pose(
        np.eye(4), _camera(), _points(), eligible_landmark_mask=eligible
    )

    assert not np.any(eligible)
    assert score.effective_point_count == 0
    assert score.log_likelihood_mean == pytest.approx(np.log(0.01))


def test_fixed_global_candidates_remove_pose_local_look_elsewhere_match() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 8], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("support.png",), ("support.png",)),
    )
    rays = bank.xyz / np.linalg.norm(bank.xyz, axis=1, keepdims=True)
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids, bank.track_ids, rays.astype(np.float32)
    )
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        source_row_indices=np.asarray([11], dtype=np.int64),
        candidate_track_ids=np.asarray([[8]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[1.0]], dtype=np.float32),
    )
    local = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(candidate_mode="pose_local_knn"),
    ).score_pose(np.eye(4), _camera(), points)
    fixed = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(candidate_mode="fixed_global_topl"),
    ).score_pose(np.eye(4), _camera(), points)

    assert local.effective_point_count == 1
    assert local.log_likelihood_mean > -1e-6
    assert fixed.effective_point_count == 0
    assert fixed.log_likelihood_mean == pytest.approx(np.log(0.01))


def test_fixed_global_candidate_denominator_keeps_all_identity_modes() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 8], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("support.png",), ("support.png",)),
    )
    rays = bank.xyz / np.linalg.norm(bank.xyz, axis=1, keepdims=True)
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids, bank.track_ids, rays.astype(np.float32)
    )
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        source_row_indices=np.asarray([11], dtype=np.int64),
        candidate_track_ids=np.asarray([[7, 8]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    score = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(candidate_mode="fixed_global_topl"),
    ).score_pose(np.eye(4), _camera(), points)

    assert score.effective_point_count == 1
    assert score.point_evidence[0] == pytest.approx(0.5, abs=1e-6)


def test_learned_candidate_posterior_preserves_null_probability_mass() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 8], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("support.png",), ("support.png",)),
    )
    rays = bank.xyz / np.linalg.norm(bank.xyz, axis=1, keepdims=True)
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids, bank.track_ids, rays.astype(np.float32)
    )
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        source_row_indices=np.asarray([11], dtype=np.int64),
        candidate_track_ids=np.asarray([[7, 8]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[0.2, 0.3]], dtype=np.float32),
        candidate_null_probabilities=np.asarray([0.5], dtype=np.float32),
    )
    score = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(
            candidate_mode="fixed_global_topl",
            fixed_candidate_prior_source="learned_probability",
        ),
    ).score_pose(np.eye(4), _camera(), points)

    assert score.point_evidence[0] == pytest.approx(0.2, abs=1e-6)
    assert score.log_likelihood_mean == pytest.approx(
        np.log(0.01 + 0.99 * 0.2), abs=1e-6
    )


def test_fixed_candidate_fold_does_not_renormalize_removed_identity_mass() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 8], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("support.png",), ("support.png",)),
    )
    rays = bank.xyz / np.linalg.norm(bank.xyz, axis=1, keepdims=True)
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids, bank.track_ids, rays.astype(np.float32)
    )
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        source_row_indices=np.asarray([11], dtype=np.int64),
        candidate_track_ids=np.asarray([[7, 8]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    verifier = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(candidate_mode="fixed_global_topl"),
    )

    score = verifier.score_pose(
        np.eye(4),
        _camera(),
        points,
        eligible_landmark_mask=np.asarray([True, False]),
    )

    assert score.point_evidence[0] == pytest.approx(0.5, abs=1e-6)


def test_prototype_identity_conditional_can_reuse_learned_null_mass() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 8], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("support.png",), ("support.png",)),
    )
    rays = bank.xyz / np.linalg.norm(bank.xyz, axis=1, keepdims=True)
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids, bank.track_ids, rays.astype(np.float32)
    )
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        candidate_track_ids=np.asarray([[7, 8]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[0.2, 0.3]], dtype=np.float32),
        candidate_null_probabilities=np.asarray([0.5], dtype=np.float32),
    )
    score = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(
            candidate_mode="fixed_global_topl",
            fixed_candidate_prior_source=(
                "prototype_similarity_with_learned_null"
            ),
        ),
    ).score_pose(np.eye(4), _camera(), points)

    assert score.point_evidence[0] == pytest.approx(0.25, abs=1e-6)


def test_candidate_posterior_requires_normalized_explicit_null() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        IndependentVerificationPoints(
            xy=np.asarray([[0.0, 0.0]], dtype=np.float64),
            descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            descriptor_reference_scores=np.asarray([0.0], dtype=np.float32),
            candidate_track_ids=np.asarray([[7, 8]], dtype=np.int64),
            candidate_descriptor_scores=np.asarray([[0.2, 0.3]], dtype=np.float32),
            candidate_null_probabilities=np.asarray([0.4], dtype=np.float32),
        )


def test_view_index_rejects_multi_prototype_track_ambiguity() -> None:
    with pytest.raises(ValueError, match="one bank row per physical track"):
        LandmarkObservationViewIndex.from_track_observations(
            np.asarray([7, 7], dtype=np.int64),
            np.asarray([7], dtype=np.int64),
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        )


def test_subset_view_gates_are_exactly_equal_to_full_bank_gates() -> None:
    rows = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    rays = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.8, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    observation = LandmarkObservationViewIndex(rows, rays, landmark_count=3)
    query_rays = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    selected = np.asarray([2, 0], dtype=np.int64)
    assert np.allclose(
        observation.minimum_view_angles_deg_for_rows(
            selected, query_rays[selected]
        ),
        observation.minimum_view_angles_deg(query_rays)[selected],
        rtol=0.0,
        atol=1e-6,
    )

    prototype = LandmarkPrototypeViewIndex(
        track_ids=np.asarray([7, 8, 9], dtype=np.int64),
        prototype_ids=np.zeros((3,), dtype=np.int64),
        mean_viewing_rays=rays[[0, 2, 3]],
        viewing_ray_concentrations=np.ones((3,), dtype=np.float32),
        viewing_angle_p90_deg=np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        valid_mask=np.ones((3,), dtype=bool),
    )
    assert np.allclose(
        prototype.minimum_view_angles_deg_for_rows(
            selected, query_rays[selected]
        ),
        prototype.minimum_view_angles_deg(query_rays)[selected],
        rtol=0.0,
        atol=1e-6,
    )


def test_verification_points_require_unique_source_rows() -> None:
    with pytest.raises(ValueError, match="source rows must be unique"):
        IndependentVerificationPoints(
            xy=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            descriptors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            descriptor_reference_scores=np.zeros((2,), dtype=np.float32),
            source_row_indices=np.asarray([3, 3], dtype=np.int64),
        )


def test_prototype_view_index_keeps_duplicate_track_descriptor_modes_separate() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 7], dtype=np.int64),
        prototype_ids=np.asarray([0, 1], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([2, 2], dtype=np.int64),
        observation_image_ids=(("front.png",), ("side.png",)),
    )
    views = LandmarkPrototypeViewIndex(
        track_ids=bank.track_ids,
        prototype_ids=bank.prototype_ids,
        mean_viewing_rays=np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        viewing_ray_concentrations=np.asarray([1.0, 1.0], dtype=np.float32),
        viewing_angle_p90_deg=np.asarray([0.0, 0.0], dtype=np.float32),
        valid_mask=np.asarray([True, True]),
    )
    verifier = IndependentLandmarkPoseVerifier(bank, views)

    score = verifier.score_pose(np.eye(4), _camera(), _points())

    assert score.projected_landmark_count == 1
    assert score.evidence_coverage == pytest.approx(1.0)


def test_crossfit_folds_keep_physical_tracks_together_and_balance_cells() -> None:
    track_folds = deterministic_identity_folds(
        np.asarray([7, 7, 8, 9, 9, 10], dtype=np.int64),
        fold_count=3,
        seed=17,
    )
    assert track_folds[0] == track_folds[1]
    assert track_folds[3] == track_folds[4]

    points = IndependentVerificationPoints(
        xy=np.asarray(
            [[10.0 + index, 10.0] for index in range(9)], dtype=np.float64
        ),
        descriptors=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (9, 1)),
        descriptor_reference_scores=np.zeros((9,), dtype=np.float32),
        source_row_indices=np.arange(100, 109, dtype=np.int64),
    )
    point_folds = spatially_balanced_point_folds(
        points,
        image_width=100,
        image_height=100,
        fold_count=3,
        seed=23,
    )
    counts = np.bincount(point_folds, minlength=3)
    assert int(np.max(counts) - np.min(counts)) <= 1
    assert np.array_equal(
        point_folds,
        spatially_balanced_point_folds(
            points,
            image_width=100,
            image_height=100,
            fold_count=3,
            seed=23,
        ),
    )


def test_pose_conditioned_refinement_recovers_small_local_pose_error() -> None:
    xyz = np.asarray(
        [
            [-1.0, -0.8, 4.0],
            [-0.3, -0.7, 4.5],
            [0.4, -0.6, 5.0],
            [1.0, -0.5, 5.5],
            [-0.9, 0.5, 4.2],
            [-0.2, 0.6, 4.8],
            [0.5, 0.7, 5.3],
            [1.1, 0.8, 5.8],
        ],
        dtype=np.float64,
    )
    features = np.eye(len(xyz), dtype=np.float32)
    bank = LandmarkMapIndex(
        track_ids=np.arange(100, 108, dtype=np.int64),
        xyz=xyz,
        features=features,
        mean_variances=np.zeros((len(xyz),), dtype=np.float32),
        observation_counts=np.full((len(xyz),), 3, dtype=np.int64),
        observation_image_ids=tuple((f"support_{index}.png",) for index in range(8)),
    )
    rays = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids, bank.track_ids, rays.astype(np.float32)
    )
    camera = ColmapCamera(
        camera_id=1,
        model_id=1,
        width=200,
        height=160,
        params=(120.0, 120.0, 100.0, 80.0),
    )
    true_xy = np.column_stack(
        [
            120.0 * xyz[:, 0] / xyz[:, 2] + 100.0,
            120.0 * xyz[:, 1] / xyz[:, 2] + 80.0,
        ]
    )
    points = IndependentVerificationPoints(
        xy=true_xy,
        descriptors=features,
        descriptor_reference_scores=np.full((len(xyz),), 0.9, dtype=np.float32),
        source_row_indices=np.arange(len(xyz), dtype=np.int64),
    )
    verifier = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(
            nearest_landmarks=4,
            maximum_reprojection_distance_px=8.0,
            spatial_sigma_px=3.0,
            descriptor_temperature=0.04,
            maximum_view_angle_deg=20.0,
        ),
    )
    initial = np.eye(4, dtype=np.float64)
    initial[0, 3] = 0.05
    config = IndependentPoseRefinementConfig(
        nearest_landmarks=4,
        maximum_reprojection_distance_px=8.0,
        spatial_sigma_px=3.0,
        minimum_match_evidence=0.05,
        minimum_correspondences=8,
        iterations=2,
        maximum_translation_step_m=0.2,
        maximum_rotation_step_deg=2.0,
    )

    correspondences = verifier.pose_conditioned_correspondences(
        initial, camera, points, config
    )
    result = verifier.refine_pose(initial, camera, points, config)

    assert len(correspondences) == 8
    assert len(np.unique(correspondences.track_ids)) == 8
    assert result.success
    assert result.fit_log_likelihood_after >= result.fit_log_likelihood_before
    assert np.linalg.norm(result.pose_w2c[:3, 3]) < 1e-4
    assert result.translation_step_m == pytest.approx(0.05, abs=1e-4)
