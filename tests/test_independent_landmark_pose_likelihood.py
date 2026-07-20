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
    _normalized_spatial_mixture_density_from_regular_grid,
    _normalized_spatial_mixture_log_density,
    _regular_offset_grid_background_density,
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


def test_candidate_specific_spatial_mode_scores_its_offset_not_query_center() -> None:
    bank = _bank()
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids,
        np.asarray([7], dtype=np.int64),
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        source_row_indices=np.asarray([11], dtype=np.int64),
        candidate_track_ids=np.asarray([[7]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[1.0]], dtype=np.float32),
        candidate_null_probabilities=np.asarray([0.0], dtype=np.float32),
        candidate_spatial_offsets_xy=np.asarray(
            [[0.0, 0.0], [4.0, 0.0]], dtype=np.float32
        ),
        candidate_spatial_log_probabilities=np.log(
            np.asarray([[[[0.01, 0.99]]]], dtype=np.float32)
        ),
        candidate_spatial_dustbin_probabilities=np.zeros(
            (1, 1, 1), dtype=np.float32
        ),
        candidate_support_view_probabilities=np.ones(
            (1, 1, 1), dtype=np.float32
        ),
        candidate_spatial_valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    verifier = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(
            candidate_mode="fixed_global_topl",
            fixed_candidate_prior_source="learned_probability",
            spatial_sigma_px=0.75,
        ),
    )
    center = verifier.score_pose(np.eye(4), _camera(), points)
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.25
    shifted = verifier.score_pose(shifted_pose, _camera(), points)

    assert shifted.point_evidence[0] > 0.4
    assert center.point_evidence[0] < 0.05
    assert shifted.log_likelihood_mean > center.log_likelihood_mean


def test_candidate_spatial_tensor_without_valid_modes_is_pose_independent_unknown() -> None:
    bank = _bank()
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids,
        np.asarray([7], dtype=np.int64),
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    common = {
        "xy": np.asarray([[50.0, 50.0]], dtype=np.float64),
        "descriptors": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "descriptor_reference_scores": np.asarray([1.0], dtype=np.float32),
        "source_row_indices": np.asarray([11], dtype=np.int64),
        "candidate_track_ids": np.asarray([[7]], dtype=np.int64),
        "candidate_descriptor_scores": np.asarray([[1.0]], dtype=np.float32),
        "candidate_null_probabilities": np.asarray([0.0], dtype=np.float32),
    }
    no_modes = IndependentVerificationPoints(
        **common,
        candidate_spatial_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
        candidate_spatial_log_probabilities=np.zeros(
            (1, 1, 1, 1), dtype=np.float32
        ),
        candidate_spatial_dustbin_probabilities=np.ones(
            (1, 1, 1), dtype=np.float32
        ),
        candidate_support_view_probabilities=np.zeros(
            (1, 1, 1), dtype=np.float32
        ),
        candidate_spatial_valid_mask=np.zeros((1, 1, 1), dtype=bool),
    )
    verifier = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(
            candidate_mode="fixed_global_topl",
            fixed_candidate_prior_source="learned_probability",
        ),
    )

    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.25
    center_score = verifier.score_pose(np.eye(4), _camera(), no_modes)
    shifted_score = verifier.score_pose(shifted_pose, _camera(), no_modes)

    behind_camera = np.eye(4)
    behind_camera[2, 3] = -10.0
    behind_score = verifier.score_pose(behind_camera, _camera(), no_modes)

    assert center_score.point_evidence[0] == pytest.approx(1.0, abs=1e-6)
    assert shifted_score.point_evidence[0] == pytest.approx(1.0, abs=1e-6)
    assert behind_score.point_evidence[0] == pytest.approx(1.0, abs=1e-6)
    assert shifted_score.log_likelihood_mean == pytest.approx(
        center_score.log_likelihood_mean
    )
    assert behind_score.log_likelihood_mean == pytest.approx(
        center_score.log_likelihood_mean
    )


def test_incomplete_candidate_spatial_artifact_is_rejected() -> None:
    with pytest.raises(ValueError, match="candidate spatial modes require"):
        IndependentVerificationPoints(
            xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
            descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
            candidate_track_ids=np.asarray([[7]], dtype=np.int64),
            candidate_descriptor_scores=np.asarray([[1.0]], dtype=np.float32),
            candidate_null_probabilities=np.asarray([0.0], dtype=np.float32),
            candidate_spatial_offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
        )


def _offset_grid(values: tuple) -> np.ndarray:
    return np.stack(np.meshgrid(values, values), axis=-1).reshape(-1, 2).astype(
        np.float32
    )


def _spatial_verifier(bank: LandmarkMapIndex) -> IndependentLandmarkPoseVerifier:
    rays = bank.xyz / np.linalg.norm(bank.xyz, axis=1, keepdims=True)
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids, bank.track_ids, rays.astype(np.float32)
    )
    return IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(
            candidate_mode="fixed_global_topl",
            fixed_candidate_prior_source="learned_probability",
            maximum_view_angle_deg=90.0,
            spatial_sigma_px=0.75,
        ),
    )


def test_normalized_spatial_mixture_density_integrates_to_one() -> None:
    axis = np.linspace(-6.0, 6.0, 241, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    deltas = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)[:, None, :]
    log_density = _normalized_spatial_mixture_log_density(
        delta_xy=deltas,
        offsets_xy=np.asarray([[0.0, 0.0]], dtype=np.float64),
        log_probabilities=np.zeros((len(deltas), 1, 1, 1), dtype=np.float64),
        sigma_px=0.75,
    )[:, 0, 0]
    step = float(axis[1] - axis[0])
    assert np.sum(np.exp(log_density)) * step * step == pytest.approx(
        1.0, abs=2e-4
    )


def test_spatial_mode_density_does_not_depend_on_offset_grid_resolution() -> None:
    coarse = _offset_grid((-2.0, 0.0, 2.0))
    fine = _offset_grid((-2.0, -1.0, 0.0, 1.0, 2.0))

    def density(offsets: np.ndarray) -> float:
        logits = np.full((1, 1, 1, len(offsets)), -30.0, dtype=np.float64)
        center = int(np.flatnonzero(np.all(offsets == 0.0, axis=1))[0])
        logits[..., center] = 0.0
        return float(
            np.exp(
                _normalized_spatial_mixture_log_density(
                    delta_xy=np.asarray([[[0.7, -0.4]]], dtype=np.float64),
                    offsets_xy=offsets,
                    log_probabilities=logits,
                    sigma_px=0.75,
                )[0, 0, 0]
            )
        )

    assert density(coarse) == pytest.approx(density(fine), rel=1e-8)
    assert _regular_offset_grid_background_density(_offset_grid((-2.0, 0.0, 2.0))) > 0


def test_separable_spatial_mixture_matches_logsumexp_density() -> None:
    offsets = _offset_grid((-2.0, 0.0, 2.0)).astype(np.float64)
    logits = np.linspace(-2.0, 1.0, len(offsets), dtype=np.float64)[None, None, None]
    delta = np.asarray([[[0.6, -0.4]]], dtype=np.float64)
    log_density = _normalized_spatial_mixture_log_density(
        delta_xy=delta,
        offsets_xy=offsets,
        log_probabilities=logits,
        sigma_px=0.75,
    )[0, 0, 0]
    probability_grid = np.exp(logits - np.logaddexp.reduce(logits, axis=-1)[..., None])
    density = _normalized_spatial_mixture_density_from_regular_grid(
        delta_xy=delta.reshape(-1, 2),
        probability_grids=probability_grid.reshape(1, 3, 3),
        x_values=np.asarray([-2.0, 0.0, 2.0], dtype=np.float64),
        y_values=np.asarray([-2.0, 0.0, 2.0], dtype=np.float64),
        sigma_px=0.75,
    )[0]

    assert np.log(density) == pytest.approx(log_density, abs=2e-6)


def test_spatial_dustbin_is_pose_independent() -> None:
    offsets = _offset_grid((-2.0, 0.0, 2.0))
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        candidate_track_ids=np.asarray([[7]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[1.0]], dtype=np.float32),
        candidate_null_probabilities=np.asarray([0.0], dtype=np.float32),
        candidate_spatial_offsets_xy=offsets,
        candidate_spatial_log_probabilities=np.zeros(
            (1, 1, 1, len(offsets)), dtype=np.float32
        ),
        candidate_spatial_dustbin_probabilities=np.ones(
            (1, 1, 1), dtype=np.float32
        ),
        candidate_support_view_probabilities=np.ones(
            (1, 1, 1), dtype=np.float32
        ),
        candidate_spatial_valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    verifier = _spatial_verifier(_bank())
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.25
    behind_camera = np.eye(4)
    behind_camera[2, 3] = -10.0

    center = verifier.score_pose(np.eye(4), _camera(), points)
    shifted = verifier.score_pose(shifted_pose, _camera(), points)
    behind = verifier.score_pose(behind_camera, _camera(), points)

    assert center.point_evidence[0] == pytest.approx(1.0, abs=1e-6)
    assert shifted.point_evidence[0] == pytest.approx(1.0, abs=1e-6)
    assert behind.point_evidence[0] == pytest.approx(1.0, abs=1e-6)
    assert shifted.log_likelihood_mean == pytest.approx(center.log_likelihood_mean)
    assert behind.log_likelihood_mean == pytest.approx(center.log_likelihood_mean)


def test_unique_track_assignment_rejects_repeated_track_explanations() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 8], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.05, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(('support_7.png',), ('support_8.png',)),
    )
    offsets = _offset_grid((-2.0, 0.0, 2.0))
    center = int(np.flatnonzero(np.all(offsets == 0.0, axis=1))[0])
    log_maps = np.full((2, 2, 1, len(offsets)), -20.0, dtype=np.float32)
    log_maps[..., center] = 0.0
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0], [50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        source_row_indices=np.asarray([11, 12], dtype=np.int64),
        candidate_track_ids=np.asarray([[7, 8], [7, 8]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray(
            [[0.7, 0.3], [0.7, 0.3]], dtype=np.float32
        ),
        candidate_null_probabilities=np.zeros((2,), dtype=np.float32),
        candidate_spatial_offsets_xy=offsets,
        candidate_spatial_log_probabilities=log_maps,
        candidate_spatial_dustbin_probabilities=np.zeros((2, 2, 1), dtype=np.float32),
        candidate_support_view_probabilities=np.ones((2, 2, 1), dtype=np.float32),
        candidate_spatial_valid_mask=np.ones((2, 2, 1), dtype=bool),
    )
    verifier = _spatial_verifier(bank)

    baseline = verifier.score_pose(np.eye(4), _camera(), points)
    diagnostic_score = verifier.score_pose(
        np.eye(4),
        _camera(),
        points,
        emit_unique_track_assignment_diagnostic=True,
    )
    diagnostic = diagnostic_score.unique_track_assignment

    assert diagnostic is not None
    np.testing.assert_allclose(
        diagnostic_score.point_evidence, baseline.point_evidence, rtol=0.0, atol=0.0
    )
    assert diagnostic.eligible_edge_count == 4
    assert diagnostic.selected_candidate_count == 2
    assert diagnostic.selected_unique_track_count == 2
    assert diagnostic.collision_penalty > 0.0
    selected = diagnostic.selected_candidate_columns
    assert sorted(points.candidate_track_ids[np.arange(2), selected].tolist()) == [7, 8]


def test_unique_track_assignment_keeps_dustbin_and_missing_modes_neutral() -> None:
    offsets = _offset_grid((-2.0, 0.0, 2.0))
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        candidate_track_ids=np.asarray([[7]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[1.0]], dtype=np.float32),
        candidate_null_probabilities=np.asarray([0.0], dtype=np.float32),
        candidate_spatial_offsets_xy=offsets,
        candidate_spatial_log_probabilities=np.zeros(
            (1, 1, 1, len(offsets)), dtype=np.float32
        ),
        candidate_spatial_dustbin_probabilities=np.ones((1, 1, 1), dtype=np.float32),
        candidate_support_view_probabilities=np.ones((1, 1, 1), dtype=np.float32),
        candidate_spatial_valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    verifier = _spatial_verifier(_bank())
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.25
    behind_camera = np.eye(4)
    behind_camera[2, 3] = -10.0

    diagnostics = [
        verifier.score_pose(
            pose,
            _camera(),
            points,
            emit_unique_track_assignment_diagnostic=True,
        ).unique_track_assignment
        for pose in (np.eye(4), shifted_pose, behind_camera)
    ]

    assert all(value is not None for value in diagnostics)
    for diagnostic in diagnostics:
        assert diagnostic is not None
        assert diagnostic.active_point_count == 0
        assert diagnostic.eligible_edge_count == 0
        assert diagnostic.selected_candidate_count == 0
        assert diagnostic.log_gain_over_null == pytest.approx(0.0, abs=1e-12)
        assert diagnostic.independent_log_gain_over_null == pytest.approx(
            0.0, abs=1e-12
        )
    assert diagnostics[0].log_joint == pytest.approx(diagnostics[1].log_joint)
    assert diagnostics[0].log_joint == pytest.approx(diagnostics[2].log_joint)


def test_unique_track_assignment_requires_candidate_specific_rgb_modes() -> None:
    bank = _bank()
    views = LandmarkObservationViewIndex.from_track_observations(
        bank.track_ids,
        np.asarray([7], dtype=np.int64),
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    verifier = IndependentLandmarkPoseVerifier(
        bank,
        views,
        IndependentLandmarkPoseLikelihoodConfig(candidate_mode="fixed_global_topl"),
    )
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        candidate_track_ids=np.asarray([[7]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[1.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="candidate-specific RGB modes"):
        verifier.score_pose(
            np.eye(4),
            _camera(),
            points,
            emit_unique_track_assignment_diagnostic=True,
        )


def test_spatial_missing_view_and_omitted_candidate_mass_are_neutral() -> None:
    bank = LandmarkMapIndex(
        track_ids=np.asarray([7, 8], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.5, 0.0, 5.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([3, 3], dtype=np.int64),
        observation_image_ids=(("support_7.png",), ("support_8.png",)),
    )
    offsets = _offset_grid((-2.0, 0.0, 2.0))
    points = IndependentVerificationPoints(
        xy=np.asarray([[50.0, 50.0]], dtype=np.float64),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_reference_scores=np.asarray([1.0], dtype=np.float32),
        candidate_track_ids=np.asarray([[7, 8]], dtype=np.int64),
        candidate_descriptor_scores=np.asarray([[0.2, 0.3]], dtype=np.float32),
        candidate_null_probabilities=np.asarray([0.5], dtype=np.float32),
        candidate_spatial_offsets_xy=offsets,
        candidate_spatial_log_probabilities=np.zeros(
            (1, 2, 1, len(offsets)), dtype=np.float32
        ),
        candidate_spatial_dustbin_probabilities=np.ones(
            (1, 2, 1), dtype=np.float32
        ),
        candidate_support_view_probabilities=np.asarray(
            [[[1.0], [0.0]]], dtype=np.float32
        ),
        candidate_spatial_valid_mask=np.asarray(
            [[[True], [False]]], dtype=bool
        ),
    )
    verifier = _spatial_verifier(bank)
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.2

    center = verifier.score_pose(np.eye(4), _camera(), points)
    shifted = verifier.score_pose(shifted_pose, _camera(), points)

    # 0.5 explicit null + 0.2 dustbin + 0.3 missing candidate = one.
    assert center.point_evidence[0] == pytest.approx(1.0, abs=1e-6)
    assert shifted.point_evidence[0] == pytest.approx(1.0, abs=1e-6)


def test_spatial_support_view_permutation_preserves_mixture_likelihood() -> None:
    offsets = _offset_grid((-4.0, 0.0, 4.0))
    center = int(np.flatnonzero(np.all(offsets == 0.0, axis=1))[0])
    right = int(
        np.flatnonzero(np.all(offsets == np.asarray([4.0, 0.0]), axis=1))[0]
    )
    log_maps = np.full((1, 1, 2, len(offsets)), -20.0, dtype=np.float32)
    log_maps[0, 0, 0, center] = 0.0
    log_maps[0, 0, 1, right] = 0.0
    common = {
        "xy": np.asarray([[50.0, 50.0]], dtype=np.float64),
        "descriptors": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "descriptor_reference_scores": np.asarray([1.0], dtype=np.float32),
        "candidate_track_ids": np.asarray([[7]], dtype=np.int64),
        "candidate_descriptor_scores": np.asarray([[1.0]], dtype=np.float32),
        "candidate_null_probabilities": np.asarray([0.0], dtype=np.float32),
        "candidate_spatial_offsets_xy": offsets,
        "candidate_spatial_dustbin_probabilities": np.zeros(
            (1, 1, 2), dtype=np.float32
        ),
        "candidate_spatial_valid_mask": np.ones((1, 1, 2), dtype=bool),
    }
    original = IndependentVerificationPoints(
        **common,
        candidate_spatial_log_probabilities=log_maps,
        candidate_support_view_probabilities=np.asarray(
            [[[0.25, 0.75]]], dtype=np.float32
        ),
    )
    swapped = IndependentVerificationPoints(
        **common,
        candidate_spatial_log_probabilities=log_maps[:, :, ::-1],
        candidate_support_view_probabilities=np.asarray(
            [[[0.75, 0.25]]], dtype=np.float32
        ),
    )
    pose = np.eye(4)
    pose[0, 3] = 0.25
    verifier = _spatial_verifier(_bank())

    first = verifier.score_pose(pose, _camera(), original)
    second = verifier.score_pose(pose, _camera(), swapped)

    assert first.point_evidence[0] == pytest.approx(
        second.point_evidence[0], abs=1e-12
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
