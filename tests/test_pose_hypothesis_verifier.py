from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    PoseVerificationCandidatePool,
    VerifiedPnPConfig,
    _select_fit_matches,
    deterministic_spatial_holdout,
    deterministic_spatial_partitions,
    estimate_pose_with_heldout_verification,
    select_geometry_diverse_matches,
    resolve_pose_guided_candidate_pool,
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
