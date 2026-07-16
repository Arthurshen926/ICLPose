from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.latent_correspondence_pnp import (
    LatentEMConfig,
    _bounded_joint_update,
    _fractional_latent_edges,
    latent_correspondence_responsibilities,
    refine_pose_latent_em,
)
from feature_extract.vfm.localization.candidate_pose_evidence import (
    candidate_pose_evidence,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    CandidateSpatialLikelihood,
    PoseVerificationCandidatePool,
    fixed_posterior_pose_log_likelihood,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _camera() -> ColmapCamera:
    return ColmapCamera(
        camera_id=1,
        model_id=1,
        width=640,
        height=480,
        params=(500.0, 500.0, 320.0, 240.0),
    )


def _pool(*, shared_wrong_track: bool = False) -> PoseVerificationCandidatePool:
    rng = np.random.default_rng(41)
    count = 32
    correct_xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(6.0, 14.0, count),
        ]
    )
    wrong_xyz = np.roll(correct_xyz, shift=9, axis=0)
    xy = np.column_stack(
        [
            500.0 * correct_xyz[:, 0] / correct_xyz[:, 2] + 320.0,
            500.0 * correct_xyz[:, 1] / correct_xyz[:, 2] + 240.0,
        ]
    )
    wrong_tracks = (
        np.full((count,), 9000, dtype=np.int64)
        if shared_wrong_track
        else np.arange(2000, 2000 + count, dtype=np.int64)
    )
    return PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [
                np.arange(1000, 1000 + count, dtype=np.int64),
                wrong_tracks,
            ]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([correct_xyz, wrong_xyz], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.35, 0.55]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.10, dtype=np.float64),
    )


def test_latent_responsibilities_preserve_group_null_and_track_constraints() -> None:
    pool = _pool(shared_wrong_track=True)
    config = LatentEMConfig(
        null_mass_floor=0.05,
        max_responsibility_change=0.15,
        residual_sigma_px=3.0,
    )
    state = latent_correspondence_responsibilities(
        pool, np.eye(4, dtype=np.float64), _camera(), config
    )

    np.testing.assert_allclose(
        np.sum(state.candidate_responsibilities, axis=1)
        + state.null_responsibilities,
        1.0,
        rtol=0.0,
        atol=1e-10,
    )
    assert np.all(state.null_responsibilities >= 0.05 - 1e-10)
    assert state.max_track_mass <= 1.0 + 1e-10
    assert np.mean(state.candidate_responsibilities[:, 0]) > np.mean(
        state.candidate_responsibilities[:, 1]
    )
    assert np.mean(state.candidate_inlier_probabilities[:, 0]) > np.mean(
        state.candidate_inlier_probabilities[:, 1]
    )

    displaced = np.eye(4, dtype=np.float64)
    displaced[:3, 3] = np.asarray([0.4, -0.2, 0.1])
    updated = latent_correspondence_responsibilities(
        pool, displaced, _camera(), config, previous=state
    )
    previous_joint = np.concatenate(
        [state.candidate_responsibilities, state.null_responsibilities[:, None]],
        axis=1,
    )
    updated_joint = np.concatenate(
        [updated.candidate_responsibilities, updated.null_responsibilities[:, None]],
        axis=1,
    )
    assert float(np.max(np.abs(updated_joint - previous_joint))) <= 0.15 + 1e-10
    assert updated.max_track_mass <= 1.0 + 1e-10


def test_bounded_update_preserves_constraints_that_couple_query_rows() -> None:
    previous_candidate = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    previous_null = np.zeros((2,), dtype=np.float64)
    proposed_candidate = np.asarray([[0.4, 0.0], [0.6, 0.4]])
    proposed_null = np.asarray([0.6, 0.0])

    candidate, null = _bounded_joint_update(
        previous_candidate,
        previous_null,
        proposed_candidate,
        proposed_null,
        0.25,
    )

    np.testing.assert_allclose(candidate.sum(axis=1) + null, 1.0, atol=1e-12)
    assert float(np.sum(candidate[:, 0])) <= 1.0 + 1e-12
    assert float(
        np.max(
            np.abs(
                np.concatenate([candidate, null[:, None]], axis=1)
                - np.concatenate(
                    [previous_candidate, previous_null[:, None]], axis=1
                )
            )
        )
    ) <= 0.25 + 1e-12


def test_latent_em_does_not_treat_outlier_floor_as_geometric_support() -> None:
    pool = _pool()
    initial = np.eye(4, dtype=np.float64)
    initial[:3, 3] = np.asarray([8.0, -6.0, 0.0])
    config = LatentEMConfig(
        iterations=2,
        residual_sigma_px=2.0,
        min_effective_group_mass=0.05,
        min_effective_groups=8,
    )

    state = latent_correspondence_responsibilities(pool, initial, _camera(), config)
    assert float(np.mean(np.sum(state.candidate_responsibilities, axis=1))) > 0.1
    assert float(np.max(state.candidate_inlier_probabilities)) < 1e-10
    assert state.effective_group_count == 0

    result = refine_pose_latent_em(pool, initial, _camera(), config)
    assert not result.success
    assert result.failure_reason == "insufficient_effective_groups"
    np.testing.assert_array_equal(result.pose_w2c, initial)


def test_fractional_m_step_retains_mutually_exclusive_candidate_mass() -> None:
    source = _pool()
    count = len(source.token_indices)
    pool = PoseVerificationCandidatePool(
        token_indices=source.token_indices,
        xy=source.xy,
        track_ids=np.column_stack(
            [
                np.arange(3000, 3000 + count, dtype=np.int64),
                np.arange(4000, 4000 + count, dtype=np.int64),
            ]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.repeat(source.xyz[:, :1], 2, axis=1),
        descriptor_scores=np.tile(np.asarray([[0.45, 0.45]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.10, dtype=np.float64),
    )
    config = LatentEMConfig(min_candidate_weight=1e-3)
    state = latent_correspondence_responsibilities(
        pool, np.eye(4, dtype=np.float64), _camera(), config
    )

    _xyz, _xy, weights, rows, _track_ids = _fractional_latent_edges(
        pool, state, config
    )
    assert len(weights) == count * 2
    assert np.all(np.bincount(rows, minlength=count) == 2)
    row_mass = np.bincount(rows, weights=weights, minlength=count)
    assert np.all(row_mass <= 1.0 + 1e-10)


def test_latent_em_improves_nearby_pose_without_destroying_seed() -> None:
    pool = _pool()
    initial = np.eye(4, dtype=np.float64)
    initial[:3, 3] = np.asarray([0.08, -0.04, 0.03])
    config = LatentEMConfig(
        iterations=4,
        residual_sigma_px=6.0,
        max_responsibility_change=0.25,
        robust_f_scale_px=2.0,
        max_translation_step_m=0.5,
        max_rotation_step_deg=5.0,
    )
    result = refine_pose_latent_em(pool, initial, _camera(), config)
    initial_error = pnp_pose_error(initial, np.eye(4, dtype=np.float64))
    refined_error = pnp_pose_error(result.pose_w2c, np.eye(4, dtype=np.float64))

    assert result.success
    assert result.accepted_iterations >= 1
    assert result.final_log_likelihood_sum >= result.initial_log_likelihood_sum - 1e-5
    assert refined_error.translation_m < initial_error.translation_m
    assert refined_error.rotation_deg < 0.1


def test_latent_and_fixed_verification_share_calibrated_spatial_evidence() -> None:
    count = 8
    xyz = np.column_stack(
        [
            np.linspace(-1.5, 1.5, count),
            np.linspace(-0.5, 0.5, count),
            np.full((count,), 8.0),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-2.0, 0.0, 2.0]),
            np.asarray([-2.0, 0.0, 2.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((count, 1, 1, 9), 0.025, dtype=np.float64)
    probabilities[:, 0, 0, 4] = 0.8
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities),
        view_probabilities=np.ones((count, 1, 1), dtype=np.float64),
        dustbin_probabilities=np.ones((count, 1, 1), dtype=np.float64),
        valid_mask=np.ones((count, 1, 1), dtype=bool),
    )
    kwargs = dict(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(count, dtype=np.int64).reshape(-1, 1),
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz.reshape(count, 1, 3),
        descriptor_scores=np.full((count, 1), 0.9, dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        measurement_geometry_probabilities=np.ones((count, 1), dtype=np.float64),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
        spatial_likelihood=spatial,
    )
    raw = PoseVerificationCandidatePool(**kwargs)
    calibrated = PoseVerificationCandidatePool(
        **kwargs, spatial_geometry_calibration_weight=1.0
    )
    config = LatentEMConfig(
        residual_sigma_px=2.0,
        outlier_likelihood=1e-4,
        spatial_evidence_weight=1.0,
    )
    pose = np.eye(4, dtype=np.float64)

    raw_state = latent_correspondence_responsibilities(
        raw, pose, _camera(), config
    )
    calibrated_state = latent_correspondence_responsibilities(
        calibrated, pose, _camera(), config
    )
    raw_fixed = fixed_posterior_pose_log_likelihood(
        raw, pose, _camera(), residual_sigma_px=2.0
    )
    calibrated_fixed = fixed_posterior_pose_log_likelihood(
        calibrated, pose, _camera(), residual_sigma_px=2.0
    )

    np.testing.assert_allclose(raw_state.candidate_likelihoods, 1.0, atol=1e-12)
    expected_raw = np.log(
        raw.null_scores
        + np.sum(raw.descriptor_scores * raw_state.candidate_likelihoods, axis=1)
    )
    expected_calibrated = np.log(
        calibrated.null_scores
        + np.sum(
            calibrated.descriptor_scores
            * calibrated_state.candidate_likelihoods,
            axis=1,
        )
    )
    np.testing.assert_allclose(
        raw_fixed["log_likelihood_mean"], np.mean(expected_raw), atol=1e-12
    )
    np.testing.assert_allclose(
        calibrated_fixed["log_likelihood_mean"],
        np.mean(expected_calibrated),
        atol=1e-12,
    )
    # A categorical RGB distribution has non-zero uncertainty, so its peak
    # density is lower than the zero-offset point-mass base branch. It must not
    # receive the artificial >1 boost produced by a one-sided likelihood-ratio
    # normalization.
    assert np.mean(calibrated_state.candidate_likelihoods) < np.mean(
        raw_state.candidate_likelihoods
    )
    assert calibrated_fixed["log_likelihood_mean"] < raw_fixed[
        "log_likelihood_mean"
    ]


def test_latent_evidence_rejects_candidates_projected_outside_image() -> None:
    pool = _pool()
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 20.0
    config = LatentEMConfig(
        residual_sigma_px=2.0,
        outlier_likelihood=1e-3,
    )

    state = latent_correspondence_responsibilities(pool, pose, _camera(), config)

    np.testing.assert_allclose(
        state.candidate_likelihoods[pool.valid_mask], 1e-3, atol=1e-12
    )
    np.testing.assert_array_equal(
        state.candidate_inlier_probabilities[pool.valid_mask], 0.0
    )
    assert state.effective_group_count == 0


def test_low_rgb_reliability_keeps_latent_coordinate_near_base_measurement() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-2.0, 0.0, 2.0]),
            np.asarray([-2.0, 0.0, 2.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((1, 1, 1, 9), 1e-8, dtype=np.float64)
    probabilities[0, 0, 0, -1] = 1.0 - 8e-8
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities),
        view_probabilities=np.ones((1, 1, 1), dtype=np.float64),
        dustbin_probabilities=np.full((1, 1, 1), 0.99, dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0]),
        xy=np.asarray([[320.0, 240.0]]),
        track_ids=np.asarray([[1]]),
        prototype_ids=np.asarray([[0]]),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]]),
        descriptor_scores=np.asarray([[0.9]]),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1]),
        spatial_likelihood=spatial,
    )

    evidence = candidate_pose_evidence(
        pool,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=2.0,
        outlier_likelihood=1e-4,
    )

    shift = np.linalg.norm(evidence.candidate_xy[0, 0] - pool.xy[0])
    assert 0.0 < shift < 0.1


def test_missing_view_mass_keeps_candidate_coordinate_near_base() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-2.0, 0.0, 2.0]),
            np.asarray([-2.0, 0.0, 2.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((1, 1, 1, 9), 1e-8, dtype=np.float64)
    probabilities[0, 0, 0, -1] = 1.0 - 8e-8
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities),
        view_probabilities=np.asarray([[[0.01]]], dtype=np.float64),
        dustbin_probabilities=np.zeros((1, 1, 1), dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0]),
        xy=np.asarray([[320.0, 240.0]]),
        track_ids=np.asarray([[1]]),
        prototype_ids=np.asarray([[0]]),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]]),
        descriptor_scores=np.asarray([[0.9]]),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1]),
        spatial_likelihood=spatial,
    )

    evidence = candidate_pose_evidence(
        pool,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=2.0,
        outlier_likelihood=1e-4,
    )

    shift = np.linalg.norm(evidence.candidate_xy[0, 0] - pool.xy[0])
    assert 0.0 < shift < 0.1


def test_concentrated_map_does_not_average_ambiguous_spatial_modes() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([0.0, 2.0, 4.0]),
            np.asarray([-2.0, 0.0, 2.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((len(offsets),), 1e-8, dtype=np.float64)
    probabilities[np.all(offsets == np.asarray([0.0, 0.0]), axis=1)] = 0.5
    probabilities[np.all(offsets == np.asarray([4.0, 0.0]), axis=1)] = 0.5
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities).reshape(1, 1, 1, -1),
        view_probabilities=np.ones((1, 1, 1), dtype=np.float64),
        dustbin_probabilities=np.zeros((1, 1, 1), dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0]),
        xy=np.asarray([[318.0, 240.0]]),
        track_ids=np.asarray([[1]]),
        prototype_ids=np.asarray([[0]]),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]]),
        descriptor_scores=np.asarray([[0.9]]),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1]),
        spatial_likelihood=spatial,
    )

    evidence = candidate_pose_evidence(
        pool,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=3.0,
        outlier_likelihood=1e-4,
        coordinate_update_policy="concentrated_map",
        minimum_coordinate_mode_probability=0.6,
    )

    np.testing.assert_allclose(evidence.candidate_xy[0, 0], pool.xy[0])
    assert evidence.candidate_coordinate_mode_probabilities[0, 0] < 0.6
    assert not evidence.candidate_coordinate_updated_mask[0, 0]


def test_calibrated_mixture_map_uses_frozen_target_free_action_gate() -> None:
    offsets = np.asarray(
        [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]],
        dtype=np.float64,
    )
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(
            np.asarray(
                [[[[0.05, 0.85, 0.05, 0.05]], [[0.05, 0.85, 0.05, 0.05]]]],
                dtype=np.float64,
            )
        ),
        view_probabilities=np.ones((1, 2, 1), dtype=np.float64),
        dustbin_probabilities=np.zeros((1, 2, 1), dtype=np.float64),
        valid_mask=np.ones((1, 2, 1), dtype=bool),
    )
    base_xy = np.asarray([[318.0, 240.0]], dtype=np.float64)
    refined_xy = np.asarray([[[322.0, 240.0], [314.0, 240.0]]])
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0]),
        xy=base_xy,
        track_ids=np.asarray([[1, 2]]),
        prototype_ids=np.asarray([[0, 0]]),
        xyz=np.asarray([[[0.0, 0.0, 8.0], [0.1, 0.0, 8.0]]]),
        descriptor_scores=np.asarray([[0.45, 0.45]]),
        valid_mask=np.ones((1, 2), dtype=bool),
        null_scores=np.asarray([0.1]),
        spatial_likelihood=spatial,
        candidate_update_probabilities=np.asarray([[0.81, 0.79]]),
        candidate_refined_xy=refined_xy,
        candidate_update_threshold=0.8,
    )

    evidence = candidate_pose_evidence(
        pool,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=2.0,
        outlier_likelihood=1e-4,
        coordinate_update_policy="calibrated_mixture_map",
    )

    np.testing.assert_allclose(evidence.candidate_xy[0, 0], refined_xy[0, 0])
    np.testing.assert_allclose(evidence.candidate_xy[0, 1], base_xy[0])
    np.testing.assert_array_equal(
        evidence.candidate_coordinate_updated_mask,
        np.asarray([[True, False]]),
    )
    np.testing.assert_allclose(
        evidence.candidate_coordinate_mode_probabilities,
        np.asarray([[0.81, 0.79]]),
    )


def test_spatial_mixture_uses_one_normalizer_for_likelihood_inlier_and_xy() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-2.0, 0.0, 2.0]),
            np.asarray([-2.0, 0.0, 2.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((len(offsets),), 0.01, dtype=np.float64)
    target_index = int(
        np.flatnonzero(np.all(offsets == np.asarray([2.0, 0.0]), axis=1))[0]
    )
    probabilities[target_index] = 1.0 - float(np.sum(probabilities)) + 0.01
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities).reshape(1, 1, 1, -1),
        view_probabilities=np.asarray([[[0.4]]], dtype=np.float64),
        dustbin_probabilities=np.asarray([[[0.75]]], dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    base_xy = np.asarray([318.0, 240.0], dtype=np.float64)
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0]),
        xy=base_xy.reshape(1, 2),
        track_ids=np.asarray([[1]]),
        prototype_ids=np.asarray([[0]]),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]]),
        descriptor_scores=np.asarray([[0.9]]),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1]),
        spatial_likelihood=spatial,
    )
    sigma = 1.0
    outlier = 0.1

    evidence = candidate_pose_evidence(
        pool,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=sigma,
        outlier_likelihood=outlier,
    )

    projected_xy = np.asarray([320.0, 240.0], dtype=np.float64)
    base_geometric = float(np.exp(-0.5 * (2.0 / sigma) ** 2))
    base_inlier_mass = (1.0 - outlier) * base_geometric
    base_likelihood = outlier + base_inlier_mass
    mode_xy = base_xy[None, :] + offsets
    mode_geometric = np.exp(
        -0.5
        * np.square(np.linalg.norm(mode_xy - projected_xy[None, :], axis=1) / sigma)
    )
    mode_evidence = float(np.sum(probabilities * mode_geometric))
    mode_inlier_mass = (1.0 - outlier) * mode_evidence
    mode_likelihood = outlier + mode_inlier_mass
    mode_mean_xy = np.sum(
        (probabilities * mode_geometric)[:, None] * mode_xy, axis=0
    ) / mode_evidence
    view_mass = 0.4
    missing_view_mass = 0.6
    reliability = 0.25
    expected_likelihood = missing_view_mass * base_likelihood + view_mass * (
        (1.0 - reliability) * base_likelihood
        + reliability * mode_likelihood
    )
    expected_inlier_mass = missing_view_mass * base_inlier_mass + view_mass * (
        (1.0 - reliability) * base_inlier_mass
        + reliability * mode_inlier_mass
    )
    expected_coordinate = (
        missing_view_mass * base_inlier_mass * base_xy
        + view_mass
        * (
            (1.0 - reliability) * base_inlier_mass * base_xy
            + reliability * mode_inlier_mass * mode_mean_xy
        )
    ) / expected_inlier_mass

    np.testing.assert_allclose(
        evidence.candidate_likelihoods[0, 0], expected_likelihood, atol=1e-12
    )
    np.testing.assert_allclose(
        evidence.candidate_inlier_probabilities[0, 0],
        expected_inlier_mass / expected_likelihood,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        evidence.candidate_xy[0, 0], expected_coordinate, atol=1e-12
    )
