from __future__ import annotations

from dataclasses import replace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    CandidateSpatialLikelihood,
    GroupedCandidatePnPConfig,
    HypothesisVerification,
    PoseVerificationCandidatePool,
    VerifiedPnPResult,
    VerifiedPnPConfig,
    _apply_selected_candidate_coordinate_updates,
    _accept_grouped_final_refine,
    _sample_grouped_candidate_assignment,
    _select_fit_matches,
    deterministic_spatial_holdout,
    deterministic_spatial_partitions,
    estimate_pose_from_grouped_candidate_pool,
    estimate_pose_with_heldout_verification,
    fixed_posterior_pose_log_likelihood,
    pose_information_diagnostics,
    select_geometry_diverse_matches,
    resolve_pose_guided_candidate_pool,
    select_geometry_guided_generation_with_immutable_baseline,
    verify_pose_candidate_pool,
    verify_pose_hypothesis,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch, pnp_pose_error


def _camera() -> ColmapCamera:
    return ColmapCamera(
        camera_id=1,
        model_id=1,
        width=640,
        height=480,
        params=(500.0, 500.0, 320.0, 240.0),
    )


def _match(
    index: int,
    xy: tuple[float, float],
    xyz: tuple[float, float, float],
    score: float = 1.0,
) -> QueryTo3DMatch:
    return QueryTo3DMatch(
        token_index=index,
        xy=np.asarray(xy, dtype=np.float64),
        track_id=1000 + index,
        xyz=np.asarray(xyz, dtype=np.float64),
        similarity=score,
        ratio=0.0,
        landmark_variance=0.0,
    )


def test_deterministic_spatial_holdout_is_disjoint_and_repeatable() -> None:
    matches = [
        _match(
            index,
            (40.0 + 140.0 * (index % 4), 40.0 + 100.0 * ((index // 4) % 4)),
            (float(index), 0.0, 5.0),
        )
        for index in range(64)
    ]

    first_fit, first_verify = deterministic_spatial_holdout(
        matches, image_width=640, image_height=480, folds=4, fold=0, salt=7
    )
    second_fit, second_verify = deterministic_spatial_holdout(
        matches, image_width=640, image_height=480, folds=4, fold=0, salt=7
    )

    np.testing.assert_array_equal(first_fit, second_fit)
    np.testing.assert_array_equal(first_verify, second_verify)
    assert set(first_fit.tolist()).isdisjoint(first_verify.tolist())
    assert sorted(np.concatenate([first_fit, first_verify]).tolist()) == list(range(64))
    assert len(first_verify) == 16


def test_deterministic_spatial_partitions_are_disjoint_and_complete() -> None:
    matches = [
        _match(
            index,
            (40.0 + 140.0 * (index % 4), 40.0 + 100.0 * ((index // 4) % 4)),
            (float(index), 0.0, 5.0),
        )
        for index in range(64)
    ]

    fit, verify, audit = deterministic_spatial_partitions(
        matches,
        image_width=640,
        image_height=480,
        folds=4,
        verification_fold=0,
        final_audit_fold=1,
        salt=7,
    )

    assert set(fit.tolist()).isdisjoint(verify.tolist())
    assert set(fit.tolist()).isdisjoint(audit.tolist())
    assert set(verify.tolist()).isdisjoint(audit.tolist())
    assert sorted(np.concatenate([fit, verify, audit]).tolist()) == list(range(64))
    assert len(verify) == 16
    assert len(audit) == 16


def test_multiple_verification_folds_keep_final_audit_independent() -> None:
    matches = [
        _match(
            index,
            (40.0 + 140.0 * (index % 4), 40.0 + 100.0 * ((index // 4) % 4)),
            (float(index), 0.0, 5.0),
        )
        for index in range(80)
    ]

    fit, verify, audit = deterministic_spatial_partitions(
        matches,
        image_width=640,
        image_height=480,
        folds=5,
        verification_fold=0,
        verification_fold_count=2,
        final_audit_fold=2,
        salt=7,
    )

    assert set(fit.tolist()).isdisjoint(verify.tolist())
    assert set(fit.tolist()).isdisjoint(audit.tolist())
    assert set(verify.tolist()).isdisjoint(audit.tolist())
    assert sorted(np.concatenate([fit, verify, audit]).tolist()) == list(range(80))
    assert len(verify) == 32
    assert len(audit) == 16


def test_geometry_diverse_selection_keeps_bearing_and_xyz_coverage() -> None:
    clustered = [
        _match(
            index,
            (300.0 + index, 230.0 + index),
            (0.01 * index, 0.0, 5.0),
            1.0 - 0.01 * index,
        )
        for index in range(8)
    ]
    diverse = [
        _match(8, (40.0, 40.0), (-3.0, -2.0, 6.0), 0.90),
        _match(9, (600.0, 50.0), (3.0, -2.0, 7.0), 0.89),
        _match(10, (50.0, 430.0), (-3.0, 2.0, 8.0), 0.88),
        _match(11, (590.0, 420.0), (3.0, 2.0, 9.0), 0.87),
    ]

    selected = select_geometry_diverse_matches(
        [*clustered, *diverse], _camera(), max_matches=6, prefilter_multiplier=2
    )

    assert selected[0].track_id == 1000
    assert len({match.track_id for match in selected if match.token_index >= 8}) >= 3


def test_pose_verification_uses_camera_depth_not_world_z() -> None:
    # Camera z is world x after this rotation, so camera depth spans 3 m while
    # all world-coordinate z values are exactly zero.
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    points = [(5.0, -1.0, 0.0), (6.0, 0.0, 0.0), (8.0, 1.0, 0.0)]
    matches = []
    for index, xyz in enumerate(points):
        camera_xyz = pose[:3, :3] @ np.asarray(xyz, dtype=np.float64)
        xy = (
            500.0 * camera_xyz[0] / camera_xyz[2] + 320.0,
            500.0 * camera_xyz[1] / camera_xyz[2] + 240.0,
        )
        matches.append(_match(index, xy, xyz))

    verification = verify_pose_hypothesis(pose, matches, _camera())

    assert verification is not None
    assert verification.strict_inlier_count == 3
    assert np.isclose(verification.depth_range_m, 3.0)


def test_pose_information_diagnostics_reports_observable_geometry() -> None:
    pose = np.eye(4, dtype=np.float64)
    points = [
        (-3.0, -2.0, 5.0),
        (3.0, -2.0, 6.0),
        (-3.0, 2.0, 8.0),
        (3.0, 2.0, 10.0),
        (0.0, 0.0, 12.0),
        (1.0, -1.0, 7.0),
    ]
    matches = []
    for index, xyz in enumerate(points):
        xy = (
            500.0 * xyz[0] / xyz[2] + 320.0,
            500.0 * xyz[1] / xyz[2] + 240.0,
        )
        matches.append(_match(index, xy, xyz))

    diagnostics = pose_information_diagnostics(
        pose, matches, _camera(), residual_sigma_px=2.0
    )

    assert diagnostics["information_match_count"] == len(points)
    assert np.isclose(diagnostics["camera_depth_span_m"], 7.0)
    assert diagnostics["bearing_max_angle_deg"] > 20.0
    assert diagnostics["translation_information_min_eigenvalue"] > 0.0
    assert diagnostics["rotation_information_min_eigenvalue"] > 0.0
    assert diagnostics["joint_information_min_eigenvalue"] > 0.0
    assert diagnostics["xyz_second_singular_ratio"] > 0.0
    assert diagnostics["xyz_third_singular_ratio"] > 0.0


def test_heldout_pose_verifier_recovers_pose_with_outliers() -> None:
    rng = np.random.default_rng(4)
    xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, 80),
            rng.uniform(-2.0, 2.0, 80),
            rng.uniform(5.0, 12.0, 80),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    xy += rng.normal(0.0, 0.25, xy.shape)
    # Keep the 2D locations but replace a deterministic subset of 3D tracks.
    corrupted_xyz = xyz.copy()
    outliers = np.arange(0, len(xyz), 5)
    corrupted_xyz[outliers] = np.roll(xyz[outliers], shift=3, axis=0)
    matches = [
        _match(
            index,
            (float(xy[index, 0]), float(xy[index, 1])),
            tuple(float(value) for value in corrupted_xyz[index]),
            score=1.0 - 0.005 * index,
        )
        for index in range(len(xyz))
    ]

    selector_calls = []

    def legacy_selector(records, _poses, eligible):
        selector_calls.append(tuple(eligible))
        return max(
            eligible,
            key=lambda index: (
                records[index].verification.rank_key(),
                records[index].fit_inlier_count,
                -index,
            ),
        )

    result = estimate_pose_with_heldout_verification(
        matches,
        _camera(),
        config=VerifiedPnPConfig(
            fit_match_counts=(32, 48),
            selection_modes=("score_topk", "spatial_round_robin", "geometry_diverse"),
            ransac_thresholds_px=(2.0, 4.0),
            rng_seed_offsets=(0,),
            ransac_iterations=1000,
            min_final_inliers=6,
        ),
        query_seed=11,
        hypothesis_selector=legacy_selector,
    )

    assert result.success
    assert result.pose_w2c is not None
    error = pnp_pose_error(result.pose_w2c, np.eye(4, dtype=np.float64))
    assert error.translation_m < 0.03
    assert error.rotation_deg < 0.3
    assert result.inlier_count >= 55
    assert result.chosen_hypothesis_index is not None
    assert len(selector_calls) == 1
    assert result.final_audit_count > 0
    assert result.pre_refine_final_audit_verification is not None


def test_pose_guided_candidate_pool_recovers_topl_and_enforces_unique_tracks() -> None:
    xy = np.asarray([[270.0, 240.0], [370.0, 240.0]], dtype=np.float64)
    # Track 10 is geometrically plausible for both rows. Row 1 has its own
    # exact track 20, so global one-to-one assignment must not duplicate 10.
    xyz = np.asarray(
        [
            [[-0.5, 0.0, 5.0], [2.0, 2.0, 5.0]],
            [[-0.5, 0.0, 5.0], [0.5, 0.0, 5.0]],
        ],
        dtype=np.float64,
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0, 1], dtype=np.int64),
        xy=xy,
        track_ids=np.asarray([[10, 11], [10, 20]], dtype=np.int64),
        prototype_ids=np.zeros((2, 2), dtype=np.int64),
        xyz=xyz,
        descriptor_scores=np.asarray([[0.8, 0.9], [0.9, 0.7]], dtype=np.float64),
        valid_mask=np.ones((2, 2), dtype=bool),
        measurement_geometry_probabilities=np.asarray(
            [[0.9, np.nan], [0.8, np.nan]], dtype=np.float64
        ),
        measurement_verification_threshold=0.6,
    )

    matches, selected, residuals = resolve_pose_guided_candidate_pool(
        pool,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=2.0,
        hard_threshold_px=120.0,
        descriptor_rank_weight=0.02,
    )

    assert selected.tolist() == [0, 1]
    assert [match.track_id for match in matches] == [10, 20]
    assert np.allclose(residuals, 0.0, atol=1e-6)

    verification = verify_pose_candidate_pool(
        np.eye(4, dtype=np.float64),
        pool,
        _camera(),
        hard_threshold_px=120.0,
    )
    assert verification is not None
    assert verification.selected_candidate_count == 2
    assert verification.selected_candidate_fraction == 1.0
    assert verification.selected_descriptor_score_mean is not None
    assert verification.selected_descriptor_rank_score_mean is not None
    assert verification.selected_reprojection_mean_px == 0.0
    assert verification.measurement_evidence_count == 2
    assert verification.measurement_evidence_fraction == 1.0
    assert np.isclose(verification.measurement_probability_mean, 0.85)
    assert np.isclose(
        verification.measurement_strict_probability_mass_fraction,
        0.9 / 1.7,
    )
    assert verification.measurement_high_confidence_strict_fraction == 0.5
    assert verification.measurement_high_confidence_loose_fraction == 0.5
    assert verification.measurement_high_confidence_contradiction_fraction == 0.5


def test_measurement_verified_refined_mode_changes_only_approved_xy() -> None:
    matches = []
    for index in range(8):
        match = _match(
            index,
            (80.0 + 140.0 * (index % 4), 80.0 + 220.0 * (index // 4)),
            (float(index) * 0.1, 0.0, 5.0),
        )
        matches.append(
            QueryTo3DMatch(
                **{
                    **match.__dict__,
                    "geometry_probability": 0.9,
                    "measurement_refined_xy": (
                        np.asarray(match.xy) + np.asarray([0.5, -0.25])
                        if index < 3
                        else None
                    ),
                }
            )
        )
    config = VerifiedPnPConfig(
        selection_modes=("measurement_verified_refined",),
        measurement_verified_min_matches=8,
        measurement_verified_min_grid_cells=4,
    )

    selected = _select_fit_matches(
        matches,
        _camera(),
        max_matches=8,
        mode="measurement_verified_refined",
        config=config,
    )
    by_token = {match.token_index: match for match in selected}

    assert len(selected) == 8
    for index in range(3):
        assert np.allclose(by_token[index].xy, matches[index].measurement_refined_xy)
    for index in range(3, 8):
        assert np.allclose(by_token[index].xy, matches[index].xy)


def test_grouped_assignment_respects_null_and_unique_physical_tracks() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0, 1, 2], dtype=np.int64),
        xy=np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
        track_ids=np.asarray([[10, 11], [10, 12], [13, 14]], dtype=np.int64),
        prototype_ids=np.zeros((3, 2), dtype=np.int64),
        xyz=np.asarray(
            [
                [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]],
                [[0.0, 0.0, 5.0], [0.0, 1.0, 5.0]],
                [[1.0, 1.0, 5.0], [2.0, 1.0, 5.0]],
            ],
            dtype=np.float64,
        ),
        descriptor_scores=np.asarray(
            [[0.8, 0.1], [0.7, 0.2], [0.1, 0.05]], dtype=np.float64
        ),
        valid_mask=np.ones((3, 2), dtype=bool),
        null_scores=np.asarray([0.1, 0.1, 0.9], dtype=np.float64),
    )

    matches, null_count = _sample_grouped_candidate_assignment(
        pool,
        candidate_limit=2,
        mode="null_argmax",
        temperature=1.0,
        seed=3,
    )

    assert len(matches) == 1
    assert matches[0].track_id == 10
    assert null_count == 2
    assert len({match.token_index for match in matches}) == len(matches)
    assert len({match.track_id for match in matches}) == len(matches)


def test_geometry_generation_mix_changes_sampling_but_keeps_forced_top1() -> None:
    kwargs = dict(
        token_indices=np.asarray([0], dtype=np.int64),
        xy=np.asarray([[20.0, 20.0]], dtype=np.float64),
        track_ids=np.asarray([[10, 11]], dtype=np.int64),
        prototype_ids=np.zeros((1, 2), dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]]),
        descriptor_scores=np.asarray([[0.9, 0.05]], dtype=np.float64),
        valid_mask=np.ones((1, 2), dtype=bool),
        measurement_geometry_probabilities=np.asarray(
            [[0.01, 0.99]], dtype=np.float64
        ),
        null_scores=np.asarray([0.05], dtype=np.float64),
    )
    baseline = PoseVerificationCandidatePool(**kwargs)
    mixed = PoseVerificationCandidatePool(
        **kwargs, geometry_generation_mix_weight=1.0
    )

    baseline_second = 0
    mixed_second = 0
    for seed in range(100):
        baseline_matches, _ = _sample_grouped_candidate_assignment(
            baseline,
            candidate_limit=2,
            mode="posterior_sample",
            temperature=1.0,
            seed=seed,
        )
        mixed_matches, _ = _sample_grouped_candidate_assignment(
            mixed,
            candidate_limit=2,
            mode="posterior_sample",
            temperature=1.0,
            seed=seed,
        )
        baseline_second += bool(
            baseline_matches and baseline_matches[0].track_id == 11
        )
        mixed_second += bool(mixed_matches and mixed_matches[0].track_id == 11)

    forced, _ = _sample_grouped_candidate_assignment(
        mixed,
        candidate_limit=2,
        mode="forced_top1",
        temperature=1.0,
        seed=0,
    )
    geometry_argmax, _ = _sample_grouped_candidate_assignment(
        mixed,
        candidate_limit=2,
        mode="geometry_mixed_argmax",
        temperature=1.0,
        seed=0,
    )
    assert baseline_second < 15
    assert mixed_second > 80
    assert forced[0].track_id == 10
    assert geometry_argmax[0].track_id == 11


def test_grouped_candidate_pnp_recovers_when_top1_identity_is_wrong() -> None:
    rng = np.random.default_rng(18)
    count = 96
    correct_xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(5.0, 12.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * correct_xyz[:, 0] / correct_xyz[:, 2] + 320.0,
            500.0 * correct_xyz[:, 1] / correct_xyz[:, 2] + 240.0,
        ]
    )
    xy += rng.normal(0.0, 0.15, xy.shape)
    wrong_xyz = np.roll(correct_xyz, shift=17, axis=0)
    xyz = np.stack([wrong_xyz, correct_xyz], axis=1)
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [
                np.arange(2000, 2000 + count, dtype=np.int64),
                np.arange(1000, 1000 + count, dtype=np.int64),
            ]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=xyz,
        descriptor_scores=np.tile(
            np.asarray([[0.55, 0.45]], dtype=np.float64), (count, 1)
        ),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.zeros((count,), dtype=np.float64),
    )

    result = estimate_pose_from_grouped_candidate_pool(
        pool,
        _camera(),
        config=GroupedCandidatePnPConfig(
            candidate_limits=(1, 2),
            samples_per_limit=8,
            sampling_temperatures=(1.0,),
            fit_match_counts=(64,),
            ransac_thresholds_px=(3.0,),
            ransac_iterations=1500,
            min_fit_matches=12,
            min_fit_grid_cells=4,
            min_final_inliers=8,
        ),
        query_seed=23,
    )

    assert result.success
    assert result.pose_w2c is not None
    error = pnp_pose_error(result.pose_w2c, np.eye(4, dtype=np.float64))
    assert error.translation_m < 0.03
    assert error.rotation_deg < 0.3
    assert result.chosen_hypothesis_index is not None
    assert any(
        "posterior_sample_L2" in record.selection_mode
        for record in result.hypotheses
        if record.solver_success
    )


def test_fixed_posterior_likelihood_does_not_pose_condition_identity_prior() -> None:
    count = 12
    xyz = np.column_stack(
        [
            np.linspace(-2.0, 2.0, count),
            np.sin(np.linspace(0.0, 3.0, count)),
            np.linspace(5.0, 10.0, count),
        ]
    )
    xy = np.column_stack(
        [500.0 * xyz[:, 0] / xyz[:, 2] + 320.0, 500.0 * xyz[:, 1] / xyz[:, 2] + 240.0]
    )
    wrong_xyz = np.roll(xyz, shift=4, axis=0)
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [np.arange(count), np.arange(100, 100 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([xyz, wrong_xyz], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.7, 0.2]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    wrong_pose = np.eye(4, dtype=np.float64)
    wrong_pose[0, 3] = 0.5

    correct = fixed_posterior_pose_log_likelihood(
        pool, np.eye(4, dtype=np.float64), _camera()
    )
    wrong = fixed_posterior_pose_log_likelihood(pool, wrong_pose, _camera())

    assert correct["log_likelihood_mean"] > wrong["log_likelihood_mean"]
    assert correct["effective_group_count"] == count
    assert correct["mass_max_abs_error"] < 1e-12


def test_candidate_coordinate_update_changes_only_selected_approved_identity() -> None:
    xyz = np.asarray(
        [
            [[-1.0, 0.0, 6.0], [1.0, 0.0, 6.0]],
            [[0.0, -1.0, 7.0], [0.0, 1.0, 7.0]],
        ],
        dtype=np.float64,
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([10, 20]),
        xy=np.asarray([[100.0, 120.0], [300.0, 320.0]]),
        track_ids=np.asarray([[101, 102], [201, 202]]),
        prototype_ids=np.zeros((2, 2), dtype=np.int64),
        xyz=xyz,
        descriptor_scores=np.asarray([[0.6, 0.3], [0.5, 0.4]]),
        valid_mask=np.ones((2, 2), dtype=bool),
        null_scores=np.asarray([0.1, 0.1]),
        candidate_update_probabilities=np.asarray([[0.95, 0.1], [0.2, 0.9]]),
        candidate_refined_xy=np.asarray(
            [
                [[101.0, 119.0], [98.0, 121.0]],
                [[301.0, 319.0], [302.0, 318.0]],
            ]
        ),
        candidate_update_threshold=0.8,
    )
    matches = [
        QueryTo3DMatch(
            token_index=10,
            xy=pool.xy[0],
            track_id=101,
            xyz=pool.xyz[0, 0],
            similarity=0.6,
            ratio=0.0,
            landmark_variance=0.0,
            prototype_id=0,
        ),
        QueryTo3DMatch(
            token_index=20,
            xy=pool.xy[1],
            track_id=201,
            xyz=pool.xyz[1, 0],
            similarity=0.5,
            ratio=0.0,
            landmark_variance=0.0,
            prototype_id=0,
        ),
    ]

    updated, count = _apply_selected_candidate_coordinate_updates(
        matches, np.asarray([0, 0]), pool
    )

    assert count == 1
    np.testing.assert_allclose(updated[0].xy, [101.0, 119.0])
    np.testing.assert_allclose(updated[1].xy, pool.xy[1])
    assert updated[0].track_id == matches[0].track_id
    assert updated[1].track_id == matches[1].track_id


def test_geometry_probability_reweights_only_retained_candidate_mass() -> None:
    count = 12
    xyz = np.column_stack(
        [
            np.linspace(-2.0, 2.0, count),
            np.sin(np.linspace(0.0, 3.0, count)),
            np.linspace(5.0, 10.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    kwargs = dict(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [np.arange(count), np.arange(100, 100 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([xyz, np.roll(xyz, shift=4, axis=0)], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.2, 0.7]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        measurement_geometry_probabilities=np.tile(
            np.asarray([[0.9, 0.1]]), (count, 1)
        ),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    zero_mix = PoseVerificationCandidatePool(**kwargs)
    full_mix = PoseVerificationCandidatePool(
        **kwargs, geometry_prior_mix_weight=1.0
    )

    zero_result = fixed_posterior_pose_log_likelihood(
        zero_mix, np.eye(4, dtype=np.float64), _camera()
    )
    mixed_result = fixed_posterior_pose_log_likelihood(
        full_mix, np.eye(4, dtype=np.float64), _camera()
    )

    assert zero_result["mass_max_abs_error"] < 1e-12
    assert mixed_result["mass_max_abs_error"] < 1e-12
    assert zero_result["geometry_prior_mix_weight"] == 0.0
    assert mixed_result["geometry_prior_evidence_count"] == 2 * count
    assert mixed_result["log_likelihood_mean"] > zero_result["log_likelihood_mean"]


def test_candidate_spatial_likelihood_is_bilinear_and_missing_rgb_is_neutral() -> None:
    count = 12
    xyz = np.column_stack(
        [
            np.linspace(-2.0, 2.0, count),
            np.sin(np.linspace(0.0, 3.0, count)),
            np.full((count,), 8.0),
        ]
    )
    xy = np.column_stack(
        [500.0 * xyz[:, 0] / xyz[:, 2] + 320.0, 500.0 * xyz[:, 1] / xyz[:, 2] + 240.0]
    )
    offsets = np.stack(
        np.meshgrid(np.asarray([-2.0, 0.0, 2.0]), np.asarray([-2.0, 0.0, 2.0])),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((count, 2, 1, 9), 0.025, dtype=np.float64)
    probabilities[:, 0, 0, 4] = 0.8
    probabilities[:, 1, 0] = 1.0 / 9.0
    spatial_valid = np.zeros((count, 2, 1), dtype=bool)
    spatial_valid[:, 0, 0] = True
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities),
        view_probabilities=np.ones((count, 2, 1), dtype=np.float64),
        dustbin_probabilities=np.zeros((count, 2, 1), dtype=np.float64),
        valid_mask=spatial_valid,
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [np.arange(count), np.arange(100, 100 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([xyz, np.roll(xyz, shift=3, axis=0)], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.7, 0.2]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
        spatial_likelihood=spatial,
    )
    shifted_pose = np.eye(4, dtype=np.float64)
    shifted_pose[0, 3] = 0.03

    correct = fixed_posterior_pose_log_likelihood(
        pool, np.eye(4, dtype=np.float64), _camera()
    )
    shifted = fixed_posterior_pose_log_likelihood(pool, shifted_pose, _camera())

    assert correct["log_likelihood_mean"] > shifted["log_likelihood_mean"]
    assert correct["spatial_candidate_count"] == count


def test_zero_spatial_evidence_weight_exactly_matches_gaussian_baseline() -> None:
    count = 8
    xyz = np.column_stack(
        [
            np.linspace(-1.5, 1.5, count),
            np.linspace(-0.8, 0.8, count),
            np.linspace(5.0, 8.0, count),
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
            np.arange(-1.0, 2.0),
            np.arange(-1.0, 2.0),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(
            np.full(
                (count, 2, 1, len(offsets)),
                1.0 / len(offsets),
                dtype=np.float64,
            )
        ),
        view_probabilities=np.ones((count, 2, 1), dtype=np.float64),
        dustbin_probabilities=np.zeros((count, 2, 1), dtype=np.float64),
        valid_mask=np.ones((count, 2, 1), dtype=bool),
        log_evidence_weight=0.0,
    )
    kwargs = dict(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [np.arange(count), np.arange(100, 100 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([xyz, np.roll(xyz, shift=2, axis=0)], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.7, 0.2]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    baseline = PoseVerificationCandidatePool(**kwargs)
    weighted = PoseVerificationCandidatePool(**kwargs, spatial_likelihood=spatial)

    baseline_result = fixed_posterior_pose_log_likelihood(
        baseline, np.eye(4, dtype=np.float64), _camera()
    )
    weighted_result = fixed_posterior_pose_log_likelihood(
        weighted, np.eye(4, dtype=np.float64), _camera()
    )

    assert weighted_result["log_likelihood_sum"] == baseline_result[
        "log_likelihood_sum"
    ]
    assert weighted_result["log_likelihood_mean"] == baseline_result[
        "log_likelihood_mean"
    ]


def test_calibrated_geometry_controls_spatial_reliability_without_mass_change() -> None:
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
        # Raw dustbin says the map is always uninformative.
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
    correct_pose = np.eye(4, dtype=np.float64)

    raw_result = fixed_posterior_pose_log_likelihood(
        raw, correct_pose, _camera()
    )
    calibrated_result = fixed_posterior_pose_log_likelihood(
        calibrated, correct_pose, _camera()
    )

    assert raw_result["mass_max_abs_error"] < 1e-12
    assert calibrated_result["mass_max_abs_error"] < 1e-12
    assert calibrated_result["spatial_calibrated_candidate_count"] == count
    assert calibrated_result["spatial_geometry_calibration_weight"] == 1.0
    assert calibrated_result["log_likelihood_mean"] > raw_result[
        "log_likelihood_mean"
    ]


def _selection_result(*, success: bool, strict_grid_cells: int) -> VerifiedPnPResult:
    verification = HypothesisVerification(
        verification_count=12,
        finite_count=12,
        positive_depth_count=12,
        positive_depth_ratio=1.0,
        strict_inlier_count=8,
        loose_inlier_count=10,
        strict_grid_cell_count=int(strict_grid_cells),
        loose_grid_cell_count=int(strict_grid_cells),
        soft_consensus=7.5,
        clipped_median_residual_px=1.0,
        depth_range_m=5.0,
    )
    pose = np.eye(4, dtype=np.float64) if success else None
    return VerifiedPnPResult(
        success=bool(success),
        pose_w2c=pose,
        inlier_mask=np.ones((12,), dtype=bool) if success else np.zeros((12,), dtype=bool),
        match_count=12,
        inlier_count=8 if success else 0,
        fit_count=12,
        verification_count=12,
        final_audit_count=12,
        chosen_hypothesis_index=None,
        hypotheses=(),
        hypothesis_poses_w2c=(),
        pre_refine_pose_w2c=pose,
        pre_refine_verification=verification if success else None,
        pre_refine_final_audit_verification=verification if success else None,
        final_verification=verification if success else None,
    )


def test_geometry_guided_generation_fallback_preserves_immutable_baseline() -> None:
    baseline = _selection_result(success=True, strict_grid_cells=9)
    regressed = _selection_result(success=True, strict_grid_cells=8)
    tied = _selection_result(success=True, strict_grid_cells=9)

    selected_regressed, regressed_audit = (
        select_geometry_guided_generation_with_immutable_baseline(
            baseline, regressed
        )
    )
    selected_tied, tied_audit = (
        select_geometry_guided_generation_with_immutable_baseline(baseline, tied)
    )

    assert selected_regressed is baseline
    assert not regressed_audit["promoted"]
    assert regressed_audit["fallback_reason"] == "strict_grid_coverage_regressed"
    assert selected_tied is tied
    assert tied_audit["promoted"]


def test_geometry_guided_generation_fallback_never_replaces_working_baseline_with_failure() -> None:
    baseline = _selection_result(success=True, strict_grid_cells=9)
    failed = _selection_result(success=False, strict_grid_cells=99)

    selected, audit = select_geometry_guided_generation_with_immutable_baseline(
        baseline, failed
    )

    assert selected is baseline
    assert not audit["promoted"]
    assert audit["fallback_reason"] == "optional_failed"


def test_strict_grouped_refine_policy_requires_gain_without_coverage_loss() -> None:
    baseline = _selection_result(
        success=True, strict_grid_cells=9
    ).pre_refine_verification
    assert baseline is not None
    strict_gain = replace(
        baseline, strict_inlier_count=baseline.strict_inlier_count + 1
    )
    coverage_loss = replace(
        strict_gain,
        strict_grid_cell_count=baseline.strict_grid_cell_count - 1,
    )

    assert _accept_grouped_final_refine(
        baseline,
        strict_gain,
        policy="strict_count_gain_with_grid_nondecrease",
    )
    assert not _accept_grouped_final_refine(
        baseline,
        baseline,
        policy="strict_count_gain_with_grid_nondecrease",
    )
    assert not _accept_grouped_final_refine(
        baseline,
        coverage_loss,
        policy="strict_count_gain_with_grid_nondecrease",
    )
