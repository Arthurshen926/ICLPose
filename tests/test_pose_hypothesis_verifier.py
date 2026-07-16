from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.candidate_pose_evidence import (
    pose_conditioned_view_probabilities,
)
from feature_extract.vfm.localization.candidate_relation_features import (
    CandidateModeMixture,
    build_query_knn_relation_graph,
    relation_residual_histograms,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    CandidateSpatialLikelihood,
    GroupedCandidatePnPConfig,
    GroupedGeneratedPose,
    GroupedProsacProfile,
    GroupedMinimalSet,
    HypothesisVerification,
    PoseHypothesisRecord,
    PoseVerificationCandidatePool,
    VerifiedPnPResult,
    VerifiedPnPConfig,
    _empty_result,
    _generated_pose_observability_failures,
    _apply_selected_candidate_coordinate_updates,
    _accept_grouped_final_refine,
    _deterministic_component_partition_plan,
    _prosac_group_order_and_quality,
    _grouped_minimal_set_signature,
    _marginal_information_block,
    _sample_grouped_candidate_assignment,
    _sample_candidate_spatial_xy,
    _select_configured_grouped_shortlist,
    _select_fit_matches,
    _select_latent_em_seeds_from_ranked,
    _validate_generation_candidate_pool_compatibility,
    candidate_relation_neighbor_edges,
    candidate_pool_likelihood_manifest_sha256,
    deterministic_component_partitions,
    deterministic_spatial_holdout,
    deterministic_spatial_partitions,
    estimate_pose_from_grouped_candidate_pool,
    estimate_pose_with_heldout_verification,
    fixed_posterior_group_log_likelihood_statistics,
    fixed_posterior_pose_log_likelihood,
    fixed_posterior_pairwise_relation_log_likelihood,
    generate_grouped_prosac_hypotheses,
    grouped_pose_observability_gate_failures,
    mass_preserving_tempered_identity_probabilities,
    partition_grouped_candidate_pool,
    pose_information_diagnostics,
    reverify_grouped_result_on_shared_denominator,
    sample_grouped_minimal_set,
    select_grouped_hypothesis_shortlist,
    select_geometry_diverse_matches,
    resolve_pose_guided_candidate_pool,
    select_crossfit_likelihood_with_immutable_baseline,
    select_geometry_guided_generation_with_immutable_baseline,
    verify_pose_candidate_pool,
    verify_pose_hypothesis,
    wrap_immutable_pose_on_grouped_denominator,
)
from feature_extract.vfm.query_to_3d_matching import (
    PnPResult,
    QueryTo3DMatch,
    pnp_pose_error,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(
        camera_id=1,
        model_id=1,
        width=640,
        height=480,
        params=(500.0, 500.0, 320.0, 240.0),
    )


def test_grouped_config_with_latent_em_is_json_serializable() -> None:
    json.dumps(asdict(GroupedCandidatePnPConfig(latent_em_enabled=True)))


def test_legacy_hypothesis_rank_key_is_explicitly_diagnostic() -> None:
    common = {
        "verification_count": 12,
        "finite_count": 12,
        "positive_depth_count": 12,
        "positive_depth_ratio": 1.0,
        "loose_inlier_count": 10,
        "loose_grid_cell_count": 4,
        "soft_consensus": 7.0,
        "clipped_median_residual_px": 1.0,
        "depth_range_m": 4.0,
        "fixed_posterior_log_likelihood_mean": -0.5,
    }
    weaker = HypothesisVerification(
        **common, strict_inlier_count=6, strict_grid_cell_count=3
    )
    self_consistent = HypothesisVerification(
        **common, strict_inlier_count=9, strict_grid_cell_count=4
    )

    assert weaker.fixed_posterior_rank_key() == self_consistent.fixed_posterior_rank_key()
    assert (
        weaker.legacy_fixed_posterior_rank_key_diagnostic_only()
        < self_consistent.legacy_fixed_posterior_rank_key_diagnostic_only()
    )


def test_fixed_posterior_group_statistics_are_robust_and_order_invariant() -> None:
    values = np.asarray([0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 100.0, 100.0])
    xy = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 0.0],
            [11.0, 1.0],
            [0.0, 10.0],
            [1.0, 11.0],
            [10.0, 10.0],
            [11.0, 11.0],
        ]
    )
    statistics = fixed_posterior_group_log_likelihood_statistics(values, xy)
    permutation = np.asarray([6, 1, 4, 3, 0, 7, 2, 5], dtype=np.int64)
    permuted = fixed_posterior_group_log_likelihood_statistics(
        values[permutation], xy[permutation]
    )

    assert statistics == permuted
    assert statistics["log_likelihood_median"] == 3.0
    assert statistics["log_likelihood_trimmed_mean_10"] == 26.5
    assert statistics["log_likelihood_worst_quartile_mean"] == 0.0
    assert statistics["spatial_median_of_means_2x2"] == 3.0
    assert statistics["spatial_mom_cell_count"] == 4
    expected_std = float(np.std(values, ddof=1))
    assert statistics["log_likelihood_std"] == pytest.approx(expected_std)
    assert statistics["log_likelihood_standard_error"] == pytest.approx(
        expected_std / np.sqrt(len(values))
    )
    assert statistics["log_likelihood_lcb95"] == pytest.approx(
        np.mean(values) - 1.96 * expected_std / np.sqrt(len(values))
    )


def test_fixed_posterior_group_statistics_reject_nonfinite_coordinates() -> None:
    with pytest.raises(ValueError, match="finite inputs"):
        fixed_posterior_group_log_likelihood_statistics(
            np.asarray([-1.0, -2.0]),
            np.asarray([[0.0, 0.0], [np.nan, 1.0]]),
        )


@pytest.mark.parametrize(
    ("policy", "field_name"),
    (
        (
            "fixed_posterior_median_DIAGNOSTIC_ONLY",
            "fixed_posterior_log_likelihood_median",
        ),
        (
            "fixed_posterior_trimmed_mean_10_DIAGNOSTIC_ONLY",
            "fixed_posterior_log_likelihood_trimmed_mean_10",
        ),
        (
            "fixed_posterior_worst_quartile_mean_DIAGNOSTIC_ONLY",
            "fixed_posterior_log_likelihood_worst_quartile_mean",
        ),
        (
            "fixed_posterior_lcb95_DIAGNOSTIC_ONLY",
            "fixed_posterior_log_likelihood_lcb95",
        ),
        (
            "fixed_posterior_spatial_mom_2x2_DIAGNOSTIC_ONLY",
            "fixed_posterior_spatial_median_of_means_2x2",
        ),
    ),
)
def test_robust_hypothesis_rank_keys_exclude_self_consistency(
    policy: str,
    field_name: str,
) -> None:
    common = {
        "verification_count": 12,
        "finite_count": 12,
        "positive_depth_count": 12,
        "positive_depth_ratio": 1.0,
        "loose_inlier_count": 10,
        "loose_grid_cell_count": 4,
        "soft_consensus": 7.0,
        "clipped_median_residual_px": 1.0,
        "depth_range_m": 4.0,
        field_name: -0.5,
    }
    weak = HypothesisVerification(
        **common, strict_inlier_count=2, strict_grid_cell_count=1
    )
    self_consistent = HypothesisVerification(
        **common, strict_inlier_count=11, strict_grid_cell_count=12
    )

    assert weak.hypothesis_selection_rank_key(policy) == (-0.5,)
    assert (
        weak.hypothesis_selection_rank_key(policy)
        == self_consistent.hypothesis_selection_rank_key(policy)
    )


def test_robust_hypothesis_rank_key_does_not_silently_fallback() -> None:
    verification = HypothesisVerification(
        verification_count=4,
        finite_count=4,
        positive_depth_count=4,
        positive_depth_ratio=1.0,
        strict_inlier_count=4,
        loose_inlier_count=4,
        strict_grid_cell_count=4,
        loose_grid_cell_count=4,
        soft_consensus=4.0,
        clipped_median_residual_px=0.1,
        depth_range_m=1.0,
    )
    with pytest.raises(ValueError, match="lacks"):
        verification.hypothesis_selection_rank_key(
            "fixed_posterior_median_DIAGNOSTIC_ONLY"
        )


@pytest.mark.parametrize(
    "policy",
    (
        "fixed_posterior_median_DIAGNOSTIC_ONLY",
        "fixed_posterior_trimmed_mean_10_DIAGNOSTIC_ONLY",
        "fixed_posterior_worst_quartile_mean_DIAGNOSTIC_ONLY",
        "fixed_posterior_lcb95_DIAGNOSTIC_ONLY",
        "fixed_posterior_spatial_mom_2x2_DIAGNOSTIC_ONLY",
    ),
)
def test_grouped_config_accepts_explicit_robust_diagnostic_policy(
    policy: str,
) -> None:
    assert GroupedCandidatePnPConfig(
        hypothesis_selection_policy=policy
    ).hypothesis_selection_policy == policy


def test_grouped_config_rejects_unknown_crossfit_role_assignment() -> None:
    with pytest.raises(ValueError, match="role assignment"):
        GroupedCandidatePnPConfig(crossfit_role_assignment="unknown")


def test_independent_shortlist_requires_strict_maplet_purging_and_two_folds() -> None:
    with pytest.raises(ValueError, match="strict token/track/maplet"):
        GroupedCandidatePnPConfig(independent_shortlist_pool=True)
    with pytest.raises(ValueError, match="at least two rank folds"):
        GroupedCandidatePnPConfig(
            crossfit_mode="token_spatial_track_maplet_purged",
            independent_shortlist_pool=True,
        )
    with pytest.raises(ValueError, match="balanced spatial folds"):
        GroupedCandidatePnPConfig(
            verification_fold_count=2,
            final_audit_fold=2,
            crossfit_mode="token_spatial_track_maplet_purged",
            independent_shortlist_pool=True,
        )


def test_grouped_config_rejects_unknown_shortlist_evidence_mode() -> None:
    with pytest.raises(ValueError, match="shortlist evidence mode"):
        GroupedCandidatePnPConfig(prosac_shortlist_evidence_mode="unknown")


def test_grouped_config_rejects_unknown_latent_em_seed_evidence_mode() -> None:
    with pytest.raises(ValueError, match="latent EM seed evidence mode"):
        GroupedCandidatePnPConfig(latent_em_seed_evidence_mode="unknown")


def test_grouped_config_rejects_unknown_observability_evidence_mode() -> None:
    with pytest.raises(ValueError, match="observability evidence mode"):
        GroupedCandidatePnPConfig(
            prosac_observability_evidence_mode="unknown"
        )


def test_grouped_spatial_rescore_requires_selective_covering_shortlist() -> None:
    with pytest.raises(ValueError, match="selective shortlist"):
        GroupedCandidatePnPConfig(prosac_spatial_rescore_top_k=64)
    with pytest.raises(ValueError, match="cover the verification"):
        GroupedCandidatePnPConfig(
            prosac_verification_top_k=64,
            prosac_shortlist_evidence_mode="base_coordinate_then_full_spatial",
            prosac_spatial_rescore_top_k=32,
        )
    GroupedCandidatePnPConfig(
        prosac_verification_top_k=64,
        prosac_shortlist_evidence_mode="base_coordinate_then_full_spatial",
        prosac_spatial_rescore_top_k=256,
    )


def test_grouped_diverse_shortlist_config_requires_explicit_budget() -> None:
    with pytest.raises(ValueError, match="require profile_pose_diverse"):
        GroupedCandidatePnPConfig(prosac_shortlist_diverse_count=8)
    with pytest.raises(ValueError, match="finite verification top-k"):
        GroupedCandidatePnPConfig(
            prosac_shortlist_selection_mode="profile_pose_diverse",
            prosac_shortlist_diverse_count=8,
        )
    with pytest.raises(ValueError, match="in verification top-k"):
        GroupedCandidatePnPConfig(
            prosac_verification_top_k=8,
            prosac_shortlist_selection_mode="profile_pose_diverse",
            prosac_shortlist_diverse_count=9,
        )


def test_grouped_latent_seed_profile_minimum_is_non_negative() -> None:
    with pytest.raises(ValueError, match="seed minimum per profile"):
        GroupedCandidatePnPConfig(latent_em_seed_min_per_profile=-1)


def test_prosac_group_quality_counts_factorized_availability_once() -> None:
    conditional = np.asarray([0.75, 0.25], dtype=np.float64)
    availability = np.asarray([0.8, 0.4], dtype=np.float64)
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([10, 20], dtype=np.int64),
        xy=np.asarray([[100.0, 100.0], [200.0, 200.0]], dtype=np.float64),
        track_ids=np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        prototype_ids=np.zeros((2, 2), dtype=np.int64),
        xyz=np.asarray(
            [
                [[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0]],
                [[0.0, -1.0, 6.0], [0.0, 1.0, 6.0]],
            ],
            dtype=np.float64,
        ),
        descriptor_scores=availability[:, None] * conditional[None, :],
        valid_mask=np.ones((2, 2), dtype=bool),
        null_scores=1.0 - availability,
    )

    order, quality = _prosac_group_order_and_quality(
        pool,
        candidate_limit=2,
        temperature=1.0,
        group_probability_power=1.0,
        candidate_probability_power=0.5,
    )
    conditional_quality = float(np.sum(np.sqrt(conditional)))
    assert np.array_equal(order, [0, 1])
    assert np.allclose(quality, availability * conditional_quality)

    _order_without_q, quality_without_q = _prosac_group_order_and_quality(
        pool,
        candidate_limit=2,
        temperature=1.0,
        group_probability_power=0.0,
        candidate_probability_power=0.5,
    )
    assert np.allclose(quality_without_q, conditional_quality)


def test_candidate_masking_transfers_removed_identity_mass_to_null() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([10, 20], dtype=np.int64),
        xy=np.asarray([[100.0, 100.0], [200.0, 200.0]], dtype=np.float64),
        track_ids=np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        prototype_ids=np.zeros((2, 2), dtype=np.int64),
        xyz=np.asarray(
            [
                [[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0]],
                [[0.0, -1.0, 6.0], [0.0, 1.0, 6.0]],
            ],
            dtype=np.float64,
        ),
        descriptor_scores=np.asarray([[0.4, 0.3], [0.2, 0.5]]),
        valid_mask=np.ones((2, 2), dtype=bool),
        null_scores=np.asarray([0.3, 0.3], dtype=np.float64),
        maplet_cluster_ids=np.asarray([[7, 8], [9, 10]], dtype=np.int64),
    )
    keep = np.asarray([[True, False], [False, True]], dtype=bool)

    masked = pool.mask_candidates_to_null(keep)

    assert np.array_equal(masked.valid_mask, keep)
    assert np.allclose(masked.descriptor_scores, [[0.4, 0.0], [0.0, 0.5]])
    assert np.allclose(masked.null_scores, [0.6, 0.5])
    assert np.allclose(
        np.sum(masked.descriptor_scores, axis=1) + masked.null_scores,
        1.0,
    )
    assert np.array_equal(masked.maplet_cluster_ids, [[7, -1], [-1, 10]])


def test_candidate_topology_is_masked_and_preserved_with_identity_pool() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([10, 20]),
        xy=np.asarray([[100.0, 100.0], [200.0, 200.0]]),
        track_ids=np.asarray([[1, 2], [3, 4]]),
        prototype_ids=np.zeros((2, 2), dtype=np.int64),
        xyz=np.ones((2, 2, 3), dtype=np.float64),
        descriptor_scores=np.asarray([[0.4, 0.3], [0.2, 0.5]]),
        valid_mask=np.ones((2, 2), dtype=bool),
        null_scores=np.asarray([0.3, 0.3]),
        topology_neighbor_track_ids=np.asarray(
            [[[3, -1], [4, -1]], [[1, -1], [2, -1]]]
        ),
        topology_support_image_indices=np.asarray(
            [[[7, -1], [8, -1]], [[7, -1], [9, -1]]]
        ),
        topology_support_coverage_counts=np.asarray(
            [[[8, 0], [6, 0]], [[7, 0], [5, 0]]]
        ),
    )
    masked = pool.mask_candidates_to_null(
        np.asarray([[True, False], [True, False]])
    )
    assert masked.has_explicit_topology
    assert np.array_equal(masked.topology_neighbor_track_ids[0, 0], [3, -1])
    assert np.all(masked.topology_neighbor_track_ids[0, 1] == -1)
    assert np.array_equal(masked.topology_support_image_indices[1, 0], [7, -1])


def test_relation_feature_prioritizes_explicit_sfm_neighbor_topology() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([10, 20]),
        xy=np.asarray([[320.0, 240.0], [420.0, 240.0]]),
        track_ids=np.asarray([[1], [3]]),
        prototype_ids=np.zeros((2, 1), dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 5.0]], [[1.0, 0.0, 5.0]]]),
        descriptor_scores=np.ones((2, 1), dtype=np.float64),
        valid_mask=np.ones((2, 1), dtype=bool),
        null_scores=np.zeros((2,), dtype=np.float64),
        maplet_cluster_ids=np.asarray([[7], [7]]),
        topology_neighbor_track_ids=np.asarray([[[3, -1]], [[1, -1]]]),
        topology_support_image_indices=np.asarray([[[5, -1]], [[5, -1]]]),
        topology_support_coverage_counts=np.asarray([[[8, 0]], [[8, 0]]]),
    )
    modes = CandidateModeMixture(
        pool.xy[:, None, None, :],
        np.ones((2, 1, 1), dtype=np.float64),
        np.ones((2, 1, 1), dtype=bool),
    )
    features = relation_residual_histograms(
        pool,
        np.eye(4),
        _camera(),
        build_query_knn_relation_graph(pool.xy, neighbor_k=1),
        modes,
        bin_edges_px=np.asarray([0.0, 1.0, 8.0]),
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
    )
    assert np.isclose(features.histograms[0, 24, 0], 1.0)
    assert np.isclose(np.sum(features.histograms[0, :24]), 0.0)


def test_candidate_masking_canonicalizes_float32_simplex_drift() -> None:
    source_scores = np.asarray(
        [[0.003, 0.002263926490442827]], dtype=np.float64
    )
    source_null = np.asarray([0.9947363138198853], dtype=np.float64)
    assert float(np.sum(source_scores) + source_null[0]) > 1.0
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([10], dtype=np.int64),
        xy=np.asarray([[100.0, 100.0]], dtype=np.float64),
        track_ids=np.asarray([[1, 2]], dtype=np.int64),
        prototype_ids=np.zeros((1, 2), dtype=np.int64),
        xyz=np.asarray([[[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0]]]),
        descriptor_scores=source_scores,
        valid_mask=np.ones((1, 2), dtype=bool),
        null_scores=source_null,
    )
    assert float(np.sum(pool.descriptor_scores) + pool.null_scores[0]) == 1.0

    masked = pool.mask_candidates_to_null(np.zeros((1, 2), dtype=bool))

    assert masked.null_scores[0] == 1.0
    assert np.sum(masked.descriptor_scores) == 0.0


def test_explicit_null_pool_rejects_non_normalized_probability_mass() -> None:
    kwargs = dict(
        token_indices=np.asarray([10], dtype=np.int64),
        xy=np.asarray([[100.0, 100.0]], dtype=np.float64),
        track_ids=np.asarray([[1, 2]], dtype=np.int64),
        prototype_ids=np.zeros((1, 2), dtype=np.int64),
        xyz=np.asarray([[[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0]]]),
        valid_mask=np.ones((1, 2), dtype=bool),
        null_scores=np.asarray([0.2], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="must sum to one"):
        PoseVerificationCandidatePool(
            **kwargs,
            descriptor_scores=np.asarray([[0.3, 0.3]], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="probabilities in \\[0, 1\\]"):
        PoseVerificationCandidatePool(
            **kwargs,
            descriptor_scores=np.asarray([[-0.1, 0.9]], dtype=np.float64),
        )


def test_identity_prior_temperature_preserves_retained_and_null_mass() -> None:
    probabilities = np.asarray(
        [[0.45, 0.15, 0.0], [0.1, 0.2, 0.4]], dtype=np.float64
    )
    valid = np.asarray(
        [[True, True, False], [True, True, True]], dtype=bool
    )

    tempered = mass_preserving_tempered_identity_probabilities(
        probabilities,
        valid,
        temperature=4.0,
    )

    assert np.allclose(np.sum(tempered, axis=1), np.sum(probabilities, axis=1))
    assert tempered[0, 2] == 0.0
    assert tempered[0, 0] < probabilities[0, 0]
    assert tempered[0, 1] > probabilities[0, 1]
    with pytest.raises(ValueError, match="temperature"):
        mass_preserving_tempered_identity_probabilities(
            probabilities, valid, temperature=0.0
        )


def test_tempered_identity_prior_can_rescue_consistent_low_rank_modes() -> None:
    count = 12
    correct_xyz = np.column_stack(
        [
            np.linspace(-2.0, 2.0, count),
            np.sin(np.linspace(0.0, 3.0, count)),
            np.full((count,), 8.0),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * correct_xyz[:, 0] / correct_xyz[:, 2] + 320.0,
            500.0 * correct_xyz[:, 1] / correct_xyz[:, 2] + 240.0,
        ]
    )
    wrong_pose = np.eye(4, dtype=np.float64)
    wrong_pose[0, 3] = 0.5
    wrong_xyz = correct_xyz.copy()
    wrong_xyz[:, 0] -= wrong_pose[0, 3]
    kwargs = dict(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [
                np.arange(100, 100 + count),
                np.arange(200, 200 + count),
                np.arange(300, 300 + count),
            ]
        ),
        prototype_ids=np.zeros((count, 3), dtype=np.int64),
        xyz=np.stack([wrong_xyz, correct_xyz, correct_xyz], axis=1),
        descriptor_scores=np.tile(
            np.asarray([[0.7, 0.1, 0.1]], dtype=np.float64), (count, 1)
        ),
        valid_mask=np.ones((count, 3), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    sharp = PoseVerificationCandidatePool(**kwargs)
    tempered = PoseVerificationCandidatePool(
        **kwargs, identity_prior_temperature=4.0
    )

    sharp_correct = fixed_posterior_pose_log_likelihood(
        sharp, np.eye(4, dtype=np.float64), _camera()
    )
    sharp_wrong = fixed_posterior_pose_log_likelihood(
        sharp, wrong_pose, _camera()
    )
    tempered_correct = fixed_posterior_pose_log_likelihood(
        tempered, np.eye(4, dtype=np.float64), _camera()
    )
    tempered_wrong = fixed_posterior_pose_log_likelihood(
        tempered, wrong_pose, _camera()
    )

    assert sharp_wrong["log_likelihood_mean"] > sharp_correct["log_likelihood_mean"]
    assert tempered_correct["log_likelihood_mean"] > tempered_wrong[
        "log_likelihood_mean"
    ]
    assert tempered_correct["identity_prior_temperature"] == 4.0
    assert tempered_correct["identity_prior_effective_candidate_count_mean"] > (
        sharp_correct["identity_prior_effective_candidate_count_mean"]
    )


def test_spatial_track_maplet_purging_is_strict_and_mass_preserving() -> None:
    count = 24
    xy = np.asarray(
        [
            [80.0 + 96.0 * (index % 6), 70.0 + 105.0 * (index // 6)]
            for index in range(count)
        ],
        dtype=np.float64,
    )
    track_ids = np.asarray(
        [[index % 8, 100 + (index % 8)] for index in range(count)],
        dtype=np.int64,
    )
    maplet_ids = np.asarray(
        [[index % 6, (index + 2) % 6] for index in range(count)],
        dtype=np.int64,
    )
    xyz = np.zeros((count, 2, 3), dtype=np.float64)
    xyz[:, 0] = np.column_stack(
        [np.linspace(-2.0, 2.0, count), np.zeros(count), np.full(count, 6.0)]
    )
    xyz[:, 1] = xyz[:, 0] + np.asarray([0.2, 0.1, 1.0])
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=track_ids,
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=xyz,
        descriptor_scores=np.full((count, 2), 0.35, dtype=np.float64),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.3, dtype=np.float64),
        maplet_cluster_ids=maplet_ids,
    )
    config = GroupedCandidatePnPConfig(
        crossfit_mode="token_spatial_track_maplet_purged"
    )

    partitions = partition_grouped_candidate_pool(
        pool, _camera(), config=config, query_seed=17
    )

    role_pools = (partitions.fit, partitions.verification, partitions.audit)
    for role_pool in role_pools:
        assert np.allclose(
            np.sum(role_pool.descriptor_scores, axis=1) + role_pool.null_scores,
            1.0,
        )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        first_tracks = set(
            role_pools[first].track_ids[role_pools[first].valid_mask].tolist()
        )
        second_tracks = set(
            role_pools[second].track_ids[role_pools[second].valid_mask].tolist()
        )
        first_maplets = set(
            role_pools[first].maplet_cluster_ids[
                role_pools[first].valid_mask
            ].tolist()
        )
        second_maplets = set(
            role_pools[second].maplet_cluster_ids[
                role_pools[second].valid_mask
            ].tolist()
        )
        assert not first_tracks & second_tracks
        assert not first_maplets & second_maplets
    audit = partitions.partition_audit
    assert audit["strict_track_disjoint"] is True
    assert audit["strict_maplet_disjoint"] is True
    assert 0 < audit["retained_candidate_count"] < audit["original_candidate_count"]


def test_independent_shortlist_partition_is_token_track_maplet_disjoint() -> None:
    count = 24
    xy = np.asarray(
        [
            [80.0 + 96.0 * (index % 6), 70.0 + 105.0 * (index // 6)]
            for index in range(count)
        ],
        dtype=np.float64,
    )
    track_ids = np.asarray(
        [[index % 8, 100 + (index % 8)] for index in range(count)],
        dtype=np.int64,
    )
    maplet_ids = np.asarray(
        [[index % 6, (index + 2) % 6] for index in range(count)],
        dtype=np.int64,
    )
    xyz = np.zeros((count, 2, 3), dtype=np.float64)
    xyz[:, 0] = np.column_stack(
        [np.linspace(-2.0, 2.0, count), np.zeros(count), np.full(count, 6.0)]
    )
    xyz[:, 1] = xyz[:, 0] + np.asarray([0.2, 0.1, 1.0])
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=track_ids,
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=xyz,
        descriptor_scores=np.full((count, 2), 0.35, dtype=np.float64),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.3, dtype=np.float64),
        maplet_cluster_ids=maplet_ids,
    )
    partitions = partition_grouped_candidate_pool(
        pool,
        _camera(),
        config=GroupedCandidatePnPConfig(
            holdout_folds=6,
            verification_fold=0,
            verification_fold_count=2,
            final_audit_fold=2,
            crossfit_mode="token_spatial_track_maplet_purged",
            crossfit_spatial_fold_policy="cell_rotated_balanced",
            independent_shortlist_pool=True,
        ),
        query_seed=17,
    )

    role_pools = (
        partitions.fit,
        partitions.shortlist,
        partitions.verification,
        partitions.audit,
    )
    role_tokens = (
        set(partitions.fit_tokens),
        set(partitions.shortlist_tokens),
        set(partitions.verification_tokens),
        set(partitions.audit_tokens),
    )
    for role_pool in role_pools:
        assert np.allclose(
            np.sum(role_pool.descriptor_scores, axis=1) + role_pool.null_scores,
            1.0,
        )
    for first in range(len(role_pools)):
        for second in range(first + 1, len(role_pools)):
            assert not role_tokens[first] & role_tokens[second]
            first_tracks = set(
                role_pools[first].track_ids[role_pools[first].valid_mask].tolist()
            )
            second_tracks = set(
                role_pools[second].track_ids[role_pools[second].valid_mask].tolist()
            )
            first_maplets = set(
                role_pools[first].maplet_cluster_ids[
                    role_pools[first].valid_mask
                ].tolist()
            )
            second_maplets = set(
                role_pools[second].maplet_cluster_ids[
                    role_pools[second].valid_mask
                ].tolist()
            )
            assert not first_tracks & second_tracks
            assert not first_maplets & second_maplets
    audit = partitions.partition_audit
    assert audit["independent_shortlist_pool"] is True
    assert audit["shortlist_count"] == 4
    assert audit["verification_count"] == 4
    assert audit["audit_count"] == 4
    assert len(audit["identity_overlap_role_pairs"]) == 6


def _generated_shortlist_pose(profile: str, translation_x: float) -> GroupedGeneratedPose:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(translation_x)
    return GroupedGeneratedPose(
        pose_w2c=pose,
        sample=GroupedMinimalSet(
            matches=(),
            row_indices=(),
            candidate_columns=(),
            prosac_prefix_size=4,
            sampling_log_probability=0.0,
            generation_profile=profile,
        ),
        solver_success=True,
    )


def test_grouped_diverse_shortlist_preserves_head_and_distinct_profiles() -> None:
    generated = (
        _generated_shortlist_pose("a", 0.0),
        _generated_shortlist_pose("a", 0.001),
        _generated_shortlist_pose("a", 1.0),
        _generated_shortlist_pose("b", 0.002),
        _generated_shortlist_pose("b", 2.0),
        _generated_shortlist_pose("c", 3.0),
    )
    scores = tuple(
        (float(10 - index), 0.0, index) for index in range(len(generated))
    )

    assert select_grouped_hypothesis_shortlist(
        generated, scores, top_k=4
    ) == (0, 1, 2, 3)
    selected = select_grouped_hypothesis_shortlist(
        generated,
        scores,
        top_k=4,
        selection_mode="profile_pose_diverse",
        diverse_count=3,
        min_per_profile=1,
        translation_diversity_m=0.05,
        rotation_diversity_deg=0.5,
    )
    assert selected == (0, 4, 5, 2)
    assert len(selected) == len(set(selected)) == 4


def test_latent_em_seed_selector_follows_crossfit_ranked_evidence() -> None:
    generated = tuple(
        _generated_shortlist_pose("profile", float(index)) for index in range(5)
    )
    crossfit_scores = (
        (7.0, 0.0, 3),
        (6.0, 0.0, 1),
        (5.0, 0.0, 4),
        (4.0, 0.0, 0),
        (3.0, 0.0, 2),
    )
    selected = _select_latent_em_seeds_from_ranked(
        generated,
        crossfit_scores,
        GroupedCandidatePnPConfig(
            latent_em_seed_count=2,
            latent_em_translation_diversity_m=0.01,
        ),
    )

    assert selected == (3, 1)


def test_optional_em_variants_cannot_evict_frozen_raw_shortlist() -> None:
    raw = tuple(
        _generated_shortlist_pose("raw", float(index)) for index in range(4)
    )
    raw_scores = tuple(
        (float(10 - index), 0.0, index) for index in range(len(raw))
    )
    config = GroupedCandidatePnPConfig(
        prosac_verification_top_k=2,
        prosac_shortlist_selection_mode="profile_pose_diverse",
        prosac_shortlist_diverse_count=2,
    )
    frozen = _select_configured_grouped_shortlist(
        raw, raw_scores, config, top_k=2
    )
    optional = tuple(
        replace(
            _generated_shortlist_pose("em", 10.0 + float(index)),
            latent_em_applied=True,
        )
        for index in range(8)
    )
    extended = raw + optional

    assert _select_configured_grouped_shortlist(
        extended, raw_scores, config, top_k=2
    ) == frozen
    assert frozen == (0, 1)


def test_grouped_latent_and_fixed_likelihood_parameters_must_match() -> None:
    with pytest.raises(ValueError, match="share outlier likelihood"):
        GroupedCandidatePnPConfig(
            latent_em_enabled=True,
            candidate_pose_outlier_likelihood=1e-2,
        )
    with pytest.raises(ValueError, match="share null likelihood"):
        GroupedCandidatePnPConfig(
            latent_em_enabled=True,
            candidate_pose_null_likelihood=1e-2,
        )


def test_latent_final_refine_requires_latent_em() -> None:
    with pytest.raises(ValueError, match="latent final refine requires latent EM"):
        GroupedCandidatePnPConfig(final_refine_mode="latent_em")


def test_grouped_minimal_signature_preserves_spatial_mode_hypotheses() -> None:
    first_match = _match(0, (100.0, 120.0), (0.0, 0.0, 8.0))
    second_match = replace(first_match, xy=np.asarray([102.0, 120.0]))
    first = GroupedMinimalSet(
        matches=(first_match,),
        row_indices=(0,),
        candidate_columns=(0,),
        prosac_prefix_size=4,
        sampling_log_probability=-1.0,
    )
    second = replace(first, matches=(second_match,))

    assert _grouped_minimal_set_signature(first) != _grouped_minimal_set_signature(
        second
    )


def test_candidate_pool_evidence_is_immutable_across_sampling_and_scoring() -> None:
    count = 12
    xyz = np.column_stack(
        [
            np.linspace(-1.5, 1.5, count),
            np.linspace(-0.75, 0.75, count),
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
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.tile(
            np.linspace(-7.0, 1.0, len(offsets)), (count, 1, 1, 1)
        ),
        view_probabilities=np.ones((count, 1, 1), dtype=np.float64),
        dustbin_probabilities=np.full((count, 1, 1), 0.1, dtype=np.float64),
        valid_mask=np.ones((count, 1, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(count, dtype=np.int64).reshape(-1, 1),
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz.reshape(count, 1, 3),
        descriptor_scores=np.full((count, 1), 0.9, dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
        spatial_likelihood=spatial,
    )
    before = candidate_pool_likelihood_manifest_sha256(
        pool, _camera(), residual_sigma_px=8.0
    )

    _sample_candidate_spatial_xy(
        pool, 0, 0, np.random.default_rng(5), enabled=True
    )
    fixed_posterior_pose_log_likelihood(
        pool, np.eye(4, dtype=np.float64), _camera(), residual_sigma_px=8.0
    )
    after = candidate_pool_likelihood_manifest_sha256(
        pool, _camera(), residual_sigma_px=8.0
    )

    assert after == before
    assert not pool.xy.flags.writeable
    assert not spatial.local_log_probabilities.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        pool.xy[0, 0] = 0.0


def test_likelihood_scale_changes_immutable_denominator_hash() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0], dtype=np.int64),
        xy=np.asarray([[320.0, 240.0]], dtype=np.float64),
        track_ids=np.asarray([[1]], dtype=np.int64),
        prototype_ids=np.asarray([[0]], dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]], dtype=np.float64),
        descriptor_scores=np.asarray([[0.9]], dtype=np.float64),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1], dtype=np.float64),
    )
    baseline = candidate_pool_likelihood_manifest_sha256(
        pool,
        _camera(),
        residual_sigma_px=2.0,
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
    )
    changed_outlier = candidate_pool_likelihood_manifest_sha256(
        pool,
        _camera(),
        residual_sigma_px=2.0,
        candidate_outlier_likelihood=1e-2,
        null_likelihood=1e-3,
    )
    changed_null = candidate_pool_likelihood_manifest_sha256(
        pool,
        _camera(),
        residual_sigma_px=2.0,
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-2,
    )

    assert len({baseline, changed_outlier, changed_null}) == 3


def test_relation_graph_changes_immutable_denominator_hash() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0], dtype=np.int64),
        xy=np.asarray([[320.0, 240.0]], dtype=np.float64),
        track_ids=np.asarray([[1]], dtype=np.int64),
        prototype_ids=np.asarray([[0]], dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]], dtype=np.float64),
        descriptor_scores=np.asarray([[0.9]], dtype=np.float64),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1], dtype=np.float64),
    )
    common = dict(
        residual_sigma_px=2.0,
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
    )
    baseline = candidate_pool_likelihood_manifest_sha256(
        pool, _camera(), **common
    )
    first = candidate_pool_likelihood_manifest_sha256(
        pool,
        _camera(),
        relation_feature_manifest=("v1", "graph-a", 1, (0.0, 1.0)),
        **common,
    )
    second = candidate_pool_likelihood_manifest_sha256(
        pool,
        _camera(),
        relation_feature_manifest=("v1", "graph-b", 1, (0.0, 1.0)),
        **common,
    )
    assert len({baseline, first, second}) == 3


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


def test_cell_rotated_spatial_folds_balance_nondivisible_cell_counts() -> None:
    matches = [
        _match(
            index,
            (40.0 + 140.0 * (index % 4), 40.0 + 100.0 * ((index // 4) % 4)),
            (float(index), 0.0, 5.0),
        )
        for index in range(128)
    ]

    fold_rows = []
    for fold in range(6):
        _fit, heldout = deterministic_spatial_holdout(
            matches,
            image_width=640,
            image_height=480,
            folds=6,
            fold=fold,
            salt=7,
            fold_policy="cell_rotated_balanced",
        )
        fold_rows.append(heldout)

    counts = [len(rows) for rows in fold_rows]
    assert max(counts) - min(counts) <= 1
    assert sorted(np.concatenate(fold_rows).tolist()) == list(range(128))


def test_cell_rotated_multiple_verification_folds_preserve_role_balance() -> None:
    matches = [
        _match(
            index,
            (40.0 + 140.0 * (index % 4), 40.0 + 100.0 * ((index // 4) % 4)),
            (float(index), 0.0, 5.0),
        )
        for index in range(128)
    ]

    fit, verify, audit = deterministic_spatial_partitions(
        matches,
        image_width=640,
        image_height=480,
        folds=6,
        verification_fold=0,
        verification_fold_count=2,
        final_audit_fold=2,
        salt=7,
        fold_policy="cell_rotated_balanced",
    )

    assert set(fit.tolist()).isdisjoint(verify.tolist())
    assert set(fit.tolist()).isdisjoint(audit.tolist())
    assert set(verify.tolist()).isdisjoint(audit.tolist())
    assert sorted(np.concatenate([fit, verify, audit]).tolist()) == list(range(128))
    assert abs(len(fit) - 64) <= 1
    assert 42 <= len(verify) <= 44
    assert 21 <= len(audit) <= 22


def test_component_partitions_are_track_and_voxel_disjoint() -> None:
    count = 24
    columns = 2
    token_indices = np.arange(count, dtype=np.int64)
    xy = np.column_stack(
        [
            40.0 + 140.0 * (token_indices % 4),
            40.0 + 100.0 * ((token_indices // 4) % 4),
        ]
    )
    track_ids = np.column_stack(
        [
            1000 + token_indices // 2,
            2000 + token_indices,
        ]
    )
    xyz = np.empty((count, columns, 3), dtype=np.float64)
    for row in range(count):
        component = row // 2
        xyz[row, 0] = np.asarray([component * 2.0, 0.0, 6.0])
        xyz[row, 1] = np.asarray([component * 2.0, 0.1, 6.1])
    pool = PoseVerificationCandidatePool(
        token_indices=token_indices,
        xy=xy,
        track_ids=track_ids,
        prototype_ids=np.zeros((count, columns), dtype=np.int64),
        xyz=xyz,
        descriptor_scores=np.full((count, columns), 0.45, dtype=np.float64),
        valid_mask=np.ones((count, columns), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )

    first = deterministic_component_partitions(
        pool,
        image_width=640,
        image_height=480,
        folds=4,
        verification_fold=0,
        final_audit_fold=1,
        maplet_voxel_size_m=0.5,
        salt=17,
    )
    second = deterministic_component_partitions(
        pool,
        image_width=640,
        image_height=480,
        folds=4,
        verification_fold=0,
        final_audit_fold=1,
        maplet_voxel_size_m=0.5,
        salt=17,
    )
    for first_indices, second_indices in zip(first, second):
        np.testing.assert_array_equal(first_indices, second_indices)
    assert sorted(np.concatenate(first).tolist()) == list(range(count))

    partition_tracks: list[set[int]] = []
    partition_voxels: list[set[tuple[int, int, int]]] = []
    for indices in first:
        partition_tracks.append(set(track_ids[indices].reshape(-1).tolist()))
        partition_voxels.append(
            {
                tuple(value)
                for value in np.floor(xyz[indices] / 0.5)
                .astype(np.int64)
                .reshape(-1, 3)
                .tolist()
            }
        )
    for left in range(3):
        for right in range(left + 1, 3):
            assert partition_tracks[left].isdisjoint(partition_tracks[right])
            assert partition_voxels[left].isdisjoint(partition_voxels[right])


def test_component_partitions_are_explicit_maplet_disjoint() -> None:
    count = 24
    token_indices = np.arange(count, dtype=np.int64)
    track_ids = np.column_stack(
        [1000 + 2 * token_indices, 1001 + 2 * token_indices]
    )
    maplet_ids = np.column_stack(
        [5000 + token_indices // 2, 6000 + token_indices // 2]
    )
    pool = PoseVerificationCandidatePool(
        token_indices=token_indices,
        xy=np.column_stack(
            [40.0 + 140.0 * (token_indices % 4), 40.0 + 90.0 * (token_indices // 4)]
        ),
        track_ids=track_ids,
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack(
            [
                np.column_stack([token_indices, np.zeros(count), np.full(count, 6.0)]),
                np.column_stack([token_indices, np.ones(count), np.full(count, 7.0)]),
            ],
            axis=1,
        ),
        descriptor_scores=np.full((count, 2), 0.45, dtype=np.float64),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
        maplet_cluster_ids=maplet_ids,
    )

    partitions = deterministic_component_partitions(
        pool,
        image_width=640,
        image_height=480,
        folds=4,
        verification_fold=0,
        final_audit_fold=1,
        use_explicit_maplet_clusters=True,
        salt=19,
    )

    partition_maplets = [
        set(maplet_ids[indices].reshape(-1).tolist()) for indices in partitions
    ]
    for left in range(3):
        for right in range(left + 1, 3):
            assert partition_maplets[left].isdisjoint(partition_maplets[right])


def test_component_partition_adaptive_roles_are_balanced_and_order_invariant() -> None:
    component_sizes = (12, 6, 4, 2)
    component_ids = np.concatenate(
        [np.full((size,), index, dtype=np.int64) for index, size in enumerate(component_sizes)]
    )
    count = int(component_ids.size)
    tokens = np.arange(100, 100 + count, dtype=np.int64)

    def build(order: np.ndarray) -> PoseVerificationCandidatePool:
        ordered_tokens = tokens[order]
        ordered_components = component_ids[order]
        return PoseVerificationCandidatePool(
            token_indices=ordered_tokens,
            xy=np.column_stack(
                [
                    30.0 + 20.0 * (ordered_tokens % 20),
                    30.0 + 20.0 * ((ordered_tokens // 3) % 16),
                ]
            ),
            track_ids=(1000 + ordered_components)[:, None],
            prototype_ids=np.zeros((count, 1), dtype=np.int64),
            xyz=np.column_stack(
                [ordered_components, ordered_tokens * 0.01, np.full(count, 8.0)]
            )[:, None, :],
                descriptor_scores=np.full((count, 1), 0.9, dtype=np.float64),
            valid_mask=np.ones((count, 1), dtype=bool),
            null_scores=np.full((count,), 0.1, dtype=np.float64),
            maplet_cluster_ids=(5000 + ordered_components)[:, None],
        )

    first_pool = build(np.arange(count, dtype=np.int64))
    second_pool = build(np.arange(count - 1, -1, -1, dtype=np.int64))
    first = _deterministic_component_partition_plan(
        first_pool,
        image_width=640,
        image_height=480,
        folds=4,
        verification_fold=0,
        final_audit_fold=1,
        use_explicit_maplet_clusters=True,
        role_assignment="adaptive_balanced",
        salt=23,
    )
    second = _deterministic_component_partition_plan(
        second_pool,
        image_width=640,
        image_height=480,
        folds=4,
        verification_fold=0,
        final_audit_fold=1,
        use_explicit_maplet_clusters=True,
        role_assignment="adaptive_balanced",
        salt=23,
    )

    assert first.audit["fold_sizes"] == second.audit["fold_sizes"]
    assert first.audit["component_sizes_desc"] == [12, 6, 4, 2]
    assert first.audit["fit_count"] == 12
    assert first.audit["verification_count"] == 6
    assert first.audit["audit_count"] == 6
    assert first.audit["all_folds_nonempty"] is True
    assert first.audit[
        "maplet_overlap_counts_fit_verify_fit_audit_verify_audit"
    ] == [0, 0, 0]
    assert first.audit["role_manifest_sha256"] == second.audit["role_manifest_sha256"]

    def role_tokens(
        pool: PoseVerificationCandidatePool,
        plan: object,
    ) -> tuple[set[int], set[int], set[int]]:
        return tuple(
            set(int(value) for value in pool.token_indices[indices].tolist())
            for indices in (
                plan.fit_indices,
                plan.verification_indices,
                plan.audit_indices,
            )
        )

    assert role_tokens(first_pool, first) == role_tokens(second_pool, second)


def test_explicit_maplet_partition_requires_cluster_ids() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(8, dtype=np.int64),
        xy=np.column_stack([np.arange(8) * 20.0, np.arange(8) * 10.0]),
        track_ids=np.arange(16, dtype=np.int64).reshape(8, 2),
        prototype_ids=np.zeros((8, 2), dtype=np.int64),
        xyz=np.ones((8, 2, 3), dtype=np.float64),
        descriptor_scores=np.full((8, 2), 0.5, dtype=np.float64),
        valid_mask=np.ones((8, 2), dtype=bool),
    )

    with np.testing.assert_raises_regex(ValueError, "requires cluster ids"):
        deterministic_component_partitions(
            pool,
            image_width=640,
            image_height=480,
            use_explicit_maplet_clusters=True,
        )


def test_component_partitions_reject_percolated_candidate_graph() -> None:
    count = 12
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=np.column_stack(
            [np.linspace(20.0, 620.0, count), np.linspace(20.0, 460.0, count)]
        ),
        track_ids=np.column_stack(
            [np.full((count,), 100), np.arange(200, 200 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.zeros((count, 2, 3), dtype=np.float64),
        descriptor_scores=np.full((count, 2), 0.45, dtype=np.float64),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )

    with np.testing.assert_raises_regex(ValueError, "fewer connected components"):
        deterministic_component_partitions(
            pool,
            image_width=640,
            image_height=480,
            folds=4,
            maplet_voxel_size_m=None,
        )


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


def test_marginal_information_removes_nuisance_explainable_signal() -> None:
    primary = 2.0 * np.eye(3, dtype=np.float64)
    nuisance = np.eye(3, dtype=np.float64)
    cross = np.eye(3, dtype=np.float64)

    marginalized = _marginal_information_block(primary, cross, nuisance)

    np.testing.assert_allclose(marginalized, np.eye(3), atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(marginalized))) < float(
        np.min(np.linalg.eigvalsh(primary))
    )


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
            [[0.8, 0.1], [0.7, 0.2], [0.06, 0.04]], dtype=np.float64
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
            enable_final_refine=True,
        ),
        query_seed=23,
    )

    assert result.success
    assert result.pose_w2c is not None
    error = pnp_pose_error(result.pose_w2c, np.eye(4, dtype=np.float64))
    assert error.translation_m < 0.03
    assert error.rotation_deg < 0.3
    assert result.chosen_hypothesis_index is not None
    assert result.final_refine_mode_used == "hard_assignment"
    assert any(
        "posterior_sample_L2" in record.selection_mode
        for record in result.hypotheses
        if record.solver_success
    )


def test_grouped_fallback_wraps_source_pose_bit_exact_without_resolving() -> None:
    rng = np.random.default_rng(28)
    count = 32
    xyz = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(-1.5, 1.5, count),
            rng.uniform(6.0, 10.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(1000, 1000 + count, dtype=np.int64).reshape(-1, 1),
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz.reshape(count, 1, 3),
        descriptor_scores=np.full((count, 1), 0.9, dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    source_pose = np.eye(4, dtype=np.float64)
    source_pose[0, 3] = np.float64(0.0123456789012345)
    source = PnPResult(
        success=True,
        pose_w2c=source_pose,
        inlier_mask=np.ones((count,), dtype=bool),
        match_count=count,
        inlier_count=count,
    )

    wrapped = wrap_immutable_pose_on_grouped_denominator(
        source,
        pool,
        _camera(),
        config=GroupedCandidatePnPConfig(
            holdout_folds=4,
            verification_fold=0,
            final_audit_fold=1,
            crossfit_mode="token_spatial",
            enable_final_refine=False,
        ),
        query_seed=9,
    )

    np.testing.assert_array_equal(wrapped.pose_w2c, source_pose)
    np.testing.assert_array_equal(wrapped.pre_refine_pose_w2c, source_pose)
    assert wrapped.hypotheses == ()
    assert wrapped.final_verification is not None
    assert wrapped.verification_denominator_sha256 is not None
    assert wrapped.final_audit_denominator_sha256 is not None


def test_candidate_pool_generation_weight_copy_preserves_absent_maplet_state() -> None:
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0, 1], dtype=np.int64),
        xy=np.asarray([[20.0, 30.0], [40.0, 50.0]], dtype=np.float64),
        track_ids=np.asarray([[10], [11]], dtype=np.int64),
        prototype_ids=np.zeros((2, 1), dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 5.0]], [[1.0, 0.0, 6.0]]]),
        descriptor_scores=np.asarray([[0.8], [0.7]], dtype=np.float64),
        valid_mask=np.ones((2, 1), dtype=bool),
        geometry_generation_mix_weight=0.5,
    )

    copied = pool.with_geometry_generation_mix_weight(0.0)

    assert copied.has_explicit_maplet_clusters is False
    assert copied.geometry_generation_mix_weight == 0.0
    np.testing.assert_array_equal(copied.descriptor_scores, pool.descriptor_scores)
    np.testing.assert_array_equal(copied.maplet_cluster_ids, pool.maplet_cluster_ids)


def test_grouped_fallback_abstains_when_heldout_component_denominator_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 8
    xyz = np.column_stack(
        [
            np.linspace(-1.0, 1.0, count),
            np.linspace(-0.5, 0.5, count),
            np.linspace(6.0, 8.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(1000, 1000 + count, dtype=np.int64).reshape(-1, 1),
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz.reshape(count, 1, 3),
        descriptor_scores=np.full((count, 1), 0.9, dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    source_pose = np.eye(4, dtype=np.float64)
    source_pose[1, 3] = np.float64(0.0123456789012345)
    source = PnPResult(
        success=True,
        pose_w2c=source_pose,
        inlier_mask=np.ones((count,), dtype=bool),
        match_count=count,
        inlier_count=count,
    )
    config = GroupedCandidatePnPConfig(
        holdout_folds=4,
        verification_fold=0,
        final_audit_fold=1,
        crossfit_mode="token_track_component",
        enable_final_refine=False,
    )
    monkeypatch.setattr(
        "feature_extract.vfm.localization.pose_hypothesis_verifier.verify_pose_candidate_pool",
        lambda *args, **kwargs: None,
    )

    baseline = wrap_immutable_pose_on_grouped_denominator(
        source, pool, _camera(), config=config, query_seed=9
    )
    optional = wrap_immutable_pose_on_grouped_denominator(
        source, pool, _camera(), config=config, query_seed=9
    )
    selected, audit = select_crossfit_likelihood_with_immutable_baseline(
        baseline, optional, min_effective_group_count=1
    )

    np.testing.assert_array_equal(selected.pose_w2c, source_pose)
    assert baseline.pre_refine_verification is None or baseline.final_verification is None
    assert baseline.verification_denominator_sha256 is not None
    assert baseline.final_audit_denominator_sha256 is not None
    assert audit["promoted"] is False
    assert audit["abstained"] is True
    assert audit["fallback_reason"] == "fixed_posterior_likelihood_missing"


def test_grouped_minimal_set_preserves_groups_tracks_and_null_mass() -> None:
    rng = np.random.default_rng(8)
    count = 24
    xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(5.0, 10.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    null = np.concatenate(
        [np.full((12,), 0.98), np.full((12,), 0.02)]
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [np.arange(count), np.arange(100, 100 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([xyz, xyz + np.asarray([0.2, 0.1, 0.3])], axis=1),
        descriptor_scores=np.concatenate(
            [
                np.tile(np.asarray([[0.011, 0.009]]), (12, 1)),
                np.tile(np.asarray([[0.55, 0.43]]), (12, 1)),
            ],
            axis=0,
        ),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=null,
    )

    sample = sample_grouped_minimal_set(
        pool,
        _camera(),
        candidate_limit=2,
        sample_size=6,
        iteration=0,
        iteration_count=64,
        temperature=1.0,
        group_probability_power=1.0,
        candidate_probability_power=0.5,
        min_grid_cells=3,
        grid_rows=4,
        grid_cols=4,
        min_xyz_second_singular_ratio=1e-4,
        min_bearing_span_deg=2.0,
        max_attempts=128,
        use_spatial_modes=False,
        seed=3,
    )

    assert sample is not None
    assert len(sample.matches) == 6
    assert len({match.token_index for match in sample.matches}) == 6
    assert len({match.track_id for match in sample.matches}) == 6
    # The first PROSAC prefix contains only the low-null half.
    assert all(match.token_index >= 12 for match in sample.matches)


def test_spatial_mode_sampling_does_not_change_minimal_set_identities() -> None:
    rng = np.random.default_rng(81)
    count = 24
    xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(5.0, 10.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    offsets = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    spatial_logits = np.full((count, 2, 1, len(offsets)), -100.0)
    spatial_logits[..., -1] = 0.0
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=spatial_logits,
        view_probabilities=np.ones((count, 2, 1), dtype=np.float64),
        dustbin_probabilities=np.zeros((count, 2, 1), dtype=np.float64),
        valid_mask=np.ones((count, 2, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [np.arange(count), np.arange(100, 100 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([xyz, xyz + np.asarray([0.2, 0.1, 0.3])], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.55, 0.43]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.02, dtype=np.float64),
        spatial_likelihood=spatial,
    )
    kwargs = dict(
        candidate_limit=2,
        sample_size=6,
        iteration=17,
        iteration_count=64,
        temperature=1.0,
        group_probability_power=1.0,
        candidate_probability_power=0.5,
        min_grid_cells=3,
        grid_rows=4,
        grid_cols=4,
        min_xyz_second_singular_ratio=1e-4,
        min_bearing_span_deg=2.0,
        max_attempts=128,
        seed=31,
    )

    base = sample_grouped_minimal_set(
        pool, _camera(), use_spatial_modes=False, **kwargs
    )
    measured = sample_grouped_minimal_set(
        pool, _camera(), use_spatial_modes=True, **kwargs
    )

    assert base is not None and measured is not None
    assert base.row_indices == measured.row_indices
    assert base.candidate_columns == measured.candidate_columns
    assert [match.track_id for match in base.matches] == [
        match.track_id for match in measured.matches
    ]
    assert measured.spatial_mode_count == len(measured.matches)
    assert any(
        not np.array_equal(base_match.xy, measured_match.xy)
        for base_match, measured_match in zip(base.matches, measured.matches)
    )


def test_spatial_profile_preserves_a_base_coordinate_hypothesis_family() -> None:
    rng = np.random.default_rng(82)
    count = 32
    xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(5.0, 11.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    offsets = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    spatial_logits = np.full((count, 1, 1, len(offsets)), -100.0)
    spatial_logits[..., -1] = 0.0
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=spatial_logits,
        view_probabilities=np.ones((count, 1, 1), dtype=np.float64),
        dustbin_probabilities=np.zeros((count, 1, 1), dtype=np.float64),
        valid_mask=np.ones((count, 1, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(count, dtype=np.int64).reshape(-1, 1),
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz.reshape(count, 1, 3),
        descriptor_scores=np.full((count, 1), 0.98, dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        null_scores=np.full((count,), 0.02, dtype=np.float64),
        spatial_likelihood=spatial,
    )
    config = GroupedCandidatePnPConfig(
        candidate_limits=(1,),
        sampling_temperatures=(1.0,),
        generation_mode="grouped_prosac",
        min_fit_grid_cells=2,
        min_xyz_second_singular_ratio=1e-4,
        prosac_min_bearing_span_deg=1.0,
        prosac_profiles=(
            GroupedProsacProfile(
                name="measured",
                hypotheses_per_limit=8,
                minimal_set_sizes=(4,),
                use_spatial_modes=True,
            ),
        ),
    )

    generated = generate_grouped_prosac_hypotheses(
        pool, _camera(), config=config, query_seed=19
    )
    profiles = {item.sample.generation_profile for item in generated}

    assert "measured__base_coordinate" in profiles
    assert "measured" in profiles
    base_identities = {
        (item.sample.row_indices, item.sample.candidate_columns)
        for item in generated
        if item.sample.generation_profile == "measured__base_coordinate"
    }
    measured_identities = {
        (item.sample.row_indices, item.sample.candidate_columns)
        for item in generated
        if item.sample.generation_profile == "measured"
    }
    assert base_identities
    assert measured_identities.issubset(base_identities)


def test_grouped_prosac_oracle_recovers_rank2_repeated_structure() -> None:
    rng = np.random.default_rng(31)
    count = 72
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
    xy += rng.normal(0.0, 0.1, xy.shape)
    wrong_xyz = np.roll(correct_xyz, shift=11, axis=0)
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
        xyz=np.stack([wrong_xyz, correct_xyz], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.55, 0.45]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.zeros((count,), dtype=np.float64),
    )
    config = GroupedCandidatePnPConfig(
        candidate_limits=(2,),
        sampling_temperatures=(1.0,),
        generation_mode="grouped_prosac",
        prosac_hypotheses_per_limit=256,
        prosac_minimal_set_size=6,
        prosac_candidate_probability_power=0.1,
        prosac_local_optimization=True,
        prosac_local_consensus_px=3.0,
        prosac_local_min_matches=8,
        min_fit_grid_cells=3,
        min_xyz_second_singular_ratio=1e-4,
    )

    generated = generate_grouped_prosac_hypotheses(
        pool, _camera(), config=config, query_seed=17
    )
    errors = [
        pnp_pose_error(item.pose_w2c, np.eye(4, dtype=np.float64))
        for item in generated
        if item.solver_success and item.pose_w2c is not None
    ]

    assert errors
    assert min(error.translation_m for error in errors) < 0.03
    assert min(error.rotation_deg for error in errors) < 0.3
    assert any(
        all(1000 <= match.track_id < 2000 for match in item.sample.matches)
        for item in generated
    )
    assert any(not item.local_optimization_applied for item in generated)
    assert any(item.local_optimization_applied for item in generated)

    multi_size = generate_grouped_prosac_hypotheses(
        pool,
        _camera(),
        config=replace(
            config,
            prosac_hypotheses_per_limit=24,
            prosac_minimal_set_sizes=(4, 5, 6),
            prosac_local_optimization=False,
        ),
        query_seed=19,
    )
    assert {len(item.sample.matches) for item in multi_size} == {4, 5, 6}

    profiled = generate_grouped_prosac_hypotheses(
        pool,
        _camera(),
        config=replace(
            config,
            prosac_profiles=(
                GroupedProsacProfile(
                    name="raw6",
                    hypotheses_per_limit=128,
                    minimal_set_sizes=(6,),
                    candidate_probability_power=0.1,
                ),
                GroupedProsacProfile(
                    name="local6",
                    hypotheses_per_limit=128,
                    minimal_set_sizes=(6,),
                    candidate_probability_power=0.1,
                    local_optimization=True,
                ),
            ),
        ),
        query_seed=23,
    )
    assert {item.sample.generation_profile for item in profiled} == {
        "raw6",
        "local6",
    }
    assert any(not item.local_optimization_applied for item in profiled)
    assert any(item.local_optimization_applied for item in profiled)


def test_grouped_prosac_uniform_candidate_mix_explores_low_mass_identity() -> None:
    rng = np.random.default_rng(37)
    count = 32
    xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(5.0, 12.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
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
        xyz=np.stack([np.roll(xyz, shift=7, axis=0), xyz], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.999, 0.001]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.zeros((count,), dtype=np.float64),
    )

    sampled_tracks: set[int] = set()
    for seed in range(16):
        sample = sample_grouped_minimal_set(
            pool,
            _camera(),
            candidate_limit=2,
            sample_size=4,
            iteration=15,
            iteration_count=16,
            temperature=1.0,
            group_probability_power=1.0,
            candidate_probability_power=1.0,
            min_grid_cells=2,
            grid_rows=4,
            grid_cols=4,
            min_xyz_second_singular_ratio=1e-4,
            min_bearing_span_deg=1.0,
            max_attempts=64,
            use_spatial_modes=False,
            seed=seed,
            candidate_uniform_mix=1.0,
        )
        assert sample is not None
        sampled_tracks.update(match.track_id for match in sample.matches)

    assert any(1000 <= track_id < 2000 for track_id in sampled_tracks)


def test_grouped_prosac_profile_rejects_invalid_uniform_mix() -> None:
    with pytest.raises(ValueError, match="uniform mix"):
        GroupedProsacProfile(
            name="invalid",
            hypotheses_per_limit=1,
            minimal_set_sizes=(4,),
            candidate_uniform_mix=1.1,
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


def test_pairwise_relation_likelihood_marginalizes_topl_candidates() -> None:
    count = 10
    xyz = np.column_stack(
        [
            np.linspace(-2.0, 2.0, count),
            np.sin(np.linspace(0.0, 4.0, count)),
            np.linspace(5.0, 11.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    wrong_xyz = np.roll(xyz, shift=3, axis=0)
    kwargs = dict(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.column_stack(
            [np.arange(count), np.arange(100, 100 + count)]
        ),
        prototype_ids=np.zeros((count, 2), dtype=np.int64),
        xyz=np.stack([xyz, wrong_xyz], axis=1),
        descriptor_scores=np.tile(np.asarray([[0.65, 0.25]]), (count, 1)),
        valid_mask=np.ones((count, 2), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    pool = PoseVerificationCandidatePool(**kwargs)
    edges = candidate_relation_neighbor_edges(pool, neighbor_k=3)
    wrong_pose = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(5.0)
    wrong_pose[:3, :3] = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )

    correct = fixed_posterior_pairwise_relation_log_likelihood(
        pool,
        np.eye(4, dtype=np.float64),
        _camera(),
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
        neighbor_edges=edges,
    )
    wrong = fixed_posterior_pairwise_relation_log_likelihood(
        pool,
        wrong_pose,
        _camera(),
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
        neighbor_edges=edges,
    )

    assert correct["pair_count"] == len(edges)
    assert correct["effective_pair_count"] == len(edges)
    assert correct["log_likelihood_ratio_mean"] > wrong[
        "log_likelihood_ratio_mean"
    ]
    assert correct["mass_max_abs_error"] < 1e-12

    permuted = PoseVerificationCandidatePool(
        **{
            **kwargs,
            "track_ids": kwargs["track_ids"][:, ::-1],
            "prototype_ids": kwargs["prototype_ids"][:, ::-1],
            "xyz": kwargs["xyz"][:, ::-1],
            "descriptor_scores": kwargs["descriptor_scores"][:, ::-1],
            "valid_mask": kwargs["valid_mask"][:, ::-1],
        }
    )
    permuted_result = fixed_posterior_pairwise_relation_log_likelihood(
        permuted,
        np.eye(4, dtype=np.float64),
        _camera(),
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
        neighbor_edges=edges,
    )
    assert permuted_result["log_likelihood_ratio_mean"] == pytest.approx(
        correct["log_likelihood_ratio_mean"], abs=1e-12
    )


def test_pairwise_relation_diagnostic_cannot_resolve_coherent_image_shift() -> None:
    count = 8
    xyz = np.column_stack(
        [
            np.linspace(-2.0, 2.0, count),
            np.linspace(-1.0, 1.0, count),
            np.full((count,), 8.0),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(count, dtype=np.int64).reshape(-1, 1),
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz.reshape(count, 1, 3),
        descriptor_scores=np.ones((count, 1), dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        null_scores=np.zeros((count,), dtype=np.float64),
    )
    shifted_pose = np.eye(4, dtype=np.float64)
    shifted_pose[0, 3] = 0.2

    correct = fixed_posterior_pairwise_relation_log_likelihood(
        pool, np.eye(4, dtype=np.float64), _camera(), neighbor_k=3
    )
    shifted = fixed_posterior_pairwise_relation_log_likelihood(
        pool, shifted_pose, _camera(), neighbor_k=3
    )

    assert shifted["log_likelihood_ratio_mean"] == pytest.approx(
        correct["log_likelihood_ratio_mean"], abs=1e-12
    )


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


def test_missing_support_view_mass_falls_back_without_renormalization() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-1.0, 0.0, 1.0]),
            np.asarray([-1.0, 0.0, 1.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((1, 1, 1, len(offsets)), 1e-8, dtype=np.float64)
    probabilities[0, 0, 0, -1] = 1.0 - (len(offsets) - 1) * 1e-8
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities),
        view_probabilities=np.asarray([[[0.2]]], dtype=np.float64),
        dustbin_probabilities=np.zeros((1, 1, 1), dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0], dtype=np.int64),
        xy=np.asarray([[320.0, 240.0]], dtype=np.float64),
        track_ids=np.asarray([[1]], dtype=np.int64),
        prototype_ids=np.asarray([[0]], dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]], dtype=np.float64),
        descriptor_scores=np.asarray([[0.9]], dtype=np.float64),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1], dtype=np.float64),
        spatial_likelihood=spatial,
    )
    rng = np.random.default_rng(23)
    used = [
        _sample_candidate_spatial_xy(pool, 0, 0, rng, enabled=True)[1]
        for _ in range(4000)
    ]

    assert 0.17 < float(np.mean(used)) < 0.23


def test_spatial_support_view_probability_mass_cannot_exceed_one() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-1.0, 1.0]),
            np.asarray([-1.0, 1.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)

    with pytest.raises(ValueError, match="probability mass exceeds one"):
        CandidateSpatialLikelihood(
            offsets_xy=offsets,
            local_log_probabilities=np.zeros((1, 1, 2, len(offsets))),
            view_probabilities=np.asarray([[[0.7, 0.6]]]),
            dustbin_probabilities=np.zeros((1, 1, 2)),
            valid_mask=np.ones((1, 1, 2), dtype=bool),
        )


def test_pose_conditioned_view_probabilities_preserve_mass_and_zero_views() -> None:
    probabilities = np.asarray([0.3, 0.0, 0.5], dtype=np.float64)
    support_centers = np.asarray(
        [[0.0, 0.0, -4.0], [0.0, 4.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    conditioned = pose_conditioned_view_probabilities(
        probabilities,
        support_centers,
        track_xyz=np.zeros((3,), dtype=np.float64),
        query_camera_center=np.asarray([0.0, 0.0, -5.0]),
        sigma_deg=15.0,
    )

    assert np.isclose(np.sum(conditioned), np.sum(probabilities))
    assert conditioned[0] > probabilities[0]
    assert conditioned[1] == 0.0
    assert conditioned[2] < probabilities[2]


def test_pose_conditioned_view_probabilities_are_permutation_equivariant() -> None:
    probabilities = np.asarray([0.2, 0.3, 0.1], dtype=np.float64)
    support_centers = np.asarray(
        [[0.0, 0.0, -3.0], [2.0, 0.0, -2.0], [-2.0, 0.0, -2.0]],
        dtype=np.float64,
    )
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    kwargs = {
        "track_xyz": np.zeros((3,), dtype=np.float64),
        "query_camera_center": np.asarray([0.0, 0.0, -5.0]),
        "sigma_deg": 20.0,
    }

    original = pose_conditioned_view_probabilities(
        probabilities, support_centers, **kwargs
    )
    permuted = pose_conditioned_view_probabilities(
        probabilities[permutation], support_centers[permutation], **kwargs
    )

    assert np.allclose(permuted, original[permutation], rtol=0.0, atol=1e-12)


def test_pose_view_geometry_requires_finite_aligned_support_centers() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-1.0, 1.0]),
            np.asarray([-1.0, 1.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    arrays = {
        "offsets_xy": offsets,
        "local_log_probabilities": np.zeros((1, 1, 1, len(offsets))),
        "view_probabilities": np.ones((1, 1, 1)),
        "dustbin_probabilities": np.zeros((1, 1, 1)),
        "valid_mask": np.ones((1, 1, 1), dtype=bool),
    }

    with pytest.raises(ValueError, match="requires support camera centers"):
        CandidateSpatialLikelihood(
            **arrays,
            pose_view_geometry_sigma_deg=15.0,
        )
    with pytest.raises(ValueError, match="must have shape"):
        CandidateSpatialLikelihood(
            **arrays,
            support_camera_centers=np.zeros((1, 1, 3)),
            pose_view_geometry_sigma_deg=15.0,
        )
    with pytest.raises(ValueError, match="must be finite"):
        CandidateSpatialLikelihood(
            **arrays,
            support_camera_centers=np.full((1, 1, 1, 3), np.nan),
            pose_view_geometry_sigma_deg=15.0,
        )


def test_pose_view_geometry_only_reweights_candidate_spatial_view_mixture() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-4.0, 0.0, 4.0]),
            np.asarray([-4.0, 0.0, 4.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((1, 1, 2, len(offsets)), 1e-6)
    center_bin = int(np.flatnonzero(np.all(offsets == [0.0, 0.0], axis=1))[0])
    wrong_bin = int(np.flatnonzero(np.all(offsets == [4.0, 0.0], axis=1))[0])
    probabilities[0, 0, 0, center_bin] = 1.0
    probabilities[0, 0, 1, wrong_bin] = 1.0
    probabilities /= np.sum(probabilities, axis=3, keepdims=True)
    spatial_kwargs = {
        "offsets_xy": offsets,
        "local_log_probabilities": np.log(probabilities),
        "view_probabilities": np.asarray([[[0.1, 0.9]]]),
        "dustbin_probabilities": np.zeros((1, 1, 2)),
        "valid_mask": np.ones((1, 1, 2), dtype=bool),
    }
    pool_kwargs = {
        "token_indices": np.asarray([0], dtype=np.int64),
        "xy": np.asarray([[320.0, 240.0]], dtype=np.float64),
        "track_ids": np.asarray([[1]], dtype=np.int64),
        "prototype_ids": np.asarray([[0]], dtype=np.int64),
        "xyz": np.asarray([[[0.0, 0.0, 8.0]]], dtype=np.float64),
        "descriptor_scores": np.asarray([[0.8]], dtype=np.float64),
        "valid_mask": np.ones((1, 1), dtype=bool),
        "null_scores": np.asarray([0.2], dtype=np.float64),
    }
    unconditioned = PoseVerificationCandidatePool(
        **pool_kwargs,
        spatial_likelihood=CandidateSpatialLikelihood(**spatial_kwargs),
    )
    conditioned = PoseVerificationCandidatePool(
        **pool_kwargs,
        spatial_likelihood=CandidateSpatialLikelihood(
            **spatial_kwargs,
            support_camera_centers=np.asarray(
                [[[[0.0, 0.0, 0.0], [8.0, 0.0, 8.0]]]],
                dtype=np.float64,
            ),
            pose_view_geometry_sigma_deg=15.0,
        ),
    )

    baseline = fixed_posterior_pose_log_likelihood(
        unconditioned,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=2.0,
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
    )
    geometry = fixed_posterior_pose_log_likelihood(
        conditioned,
        np.eye(4, dtype=np.float64),
        _camera(),
        residual_sigma_px=2.0,
        candidate_outlier_likelihood=1e-3,
        null_likelihood=1e-3,
    )

    assert baseline["mass_max_abs_error"] == geometry["mass_max_abs_error"]
    assert geometry["log_likelihood_mean"] > baseline["log_likelihood_mean"]


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


def test_frozen_generation_pool_may_differ_only_in_spatial_likelihood() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)

    def spatial(probability: float) -> CandidateSpatialLikelihood:
        probabilities = np.full((2, 1, 1, 4), (1.0 - probability) / 3.0)
        probabilities[..., 0] = probability
        return CandidateSpatialLikelihood(
            offsets_xy=offsets,
            local_log_probabilities=np.log(probabilities),
            view_probabilities=np.ones((2, 1, 1), dtype=np.float64),
            dustbin_probabilities=np.full((2, 1, 1), 0.1, dtype=np.float64),
            valid_mask=np.ones((2, 1, 1), dtype=bool),
        )

    kwargs = dict(
        token_indices=np.arange(2, dtype=np.int64),
        xy=np.asarray([[100.0, 120.0], [200.0, 220.0]], dtype=np.float64),
        track_ids=np.asarray([[10], [11]], dtype=np.int64),
        prototype_ids=np.zeros((2, 1), dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 8.0]], [[1.0, 0.0, 8.0]]]),
        descriptor_scores=np.full((2, 1), 0.8, dtype=np.float64),
        valid_mask=np.ones((2, 1), dtype=bool),
        null_scores=np.full((2,), 0.2, dtype=np.float64),
    )
    scoring = PoseVerificationCandidatePool(
        **kwargs, spatial_likelihood=spatial(0.8)
    )
    generation = PoseVerificationCandidatePool(
        **kwargs, spatial_likelihood=spatial(0.2)
    )

    _validate_generation_candidate_pool_compatibility(scoring, generation)

    # Frozen dataclass arrays are read-only. Partition purging must accept the
    # pool's own valid mask without trying to modify it in place.
    masked_generation = generation.mask_candidates_to_null(generation.valid_mask)
    np.testing.assert_array_equal(
        masked_generation.valid_mask,
        generation.valid_mask,
    )
    np.testing.assert_allclose(
        masked_generation.descriptor_scores,
        generation.descriptor_scores,
    )
    np.testing.assert_allclose(
        masked_generation.null_scores,
        generation.null_scores,
    )
    assert scoring.spatial_likelihood is not None
    swapped_spatial = masked_generation.with_spatial_likelihood(
        scoring.spatial_likelihood.mask_candidates(masked_generation.valid_mask)
    )
    assert swapped_spatial.descriptor_scores is masked_generation.descriptor_scores
    assert swapped_spatial.null_scores is masked_generation.null_scores
    assert swapped_spatial.valid_mask is masked_generation.valid_mask

    incompatible_tracks = np.asarray(generation.track_ids).copy()
    incompatible_tracks[0, 0] = 99
    incompatible = PoseVerificationCandidatePool(
        **{**kwargs, "track_ids": incompatible_tracks},
        spatial_likelihood=spatial(0.2),
    )
    with pytest.raises(ValueError, match="track_ids"):
        _validate_generation_candidate_pool_compatibility(scoring, incompatible)


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
    no_rgb = PoseVerificationCandidatePool(
        **{key: value for key, value in kwargs.items() if key != "spatial_likelihood"}
    )
    correct_pose = np.eye(4, dtype=np.float64)

    no_rgb_result = fixed_posterior_pose_log_likelihood(
        no_rgb, correct_pose, _camera()
    )
    raw_result = fixed_posterior_pose_log_likelihood(
        raw, correct_pose, _camera()
    )
    selective_base_result = fixed_posterior_pose_log_likelihood(
        raw,
        correct_pose,
        _camera(),
        spatial_evidence_weight=0.0,
    )
    calibrated_result = fixed_posterior_pose_log_likelihood(
        calibrated, correct_pose, _camera()
    )

    assert raw_result["mass_max_abs_error"] < 1e-12
    assert calibrated_result["mass_max_abs_error"] < 1e-12
    assert raw_result["log_likelihood_sum"] == no_rgb_result["log_likelihood_sum"]
    assert raw_result["log_likelihood_mean"] == no_rgb_result["log_likelihood_mean"]
    assert selective_base_result["log_likelihood_sum"] == no_rgb_result[
        "log_likelihood_sum"
    ]
    assert calibrated_result["spatial_calibrated_candidate_count"] == count
    assert calibrated_result["spatial_geometry_calibration_weight"] == 1.0
    # Enabling an uncertain RGB map changes only spatial evidence. Its peak is
    # correctly lower than the exact zero-offset base measurement; it does not
    # manufacture a likelihood-ratio reward or modify identity/null mass.
    assert calibrated_result["log_likelihood_mean"] < raw_result[
        "log_likelihood_mean"
    ]


def test_measurement_utility_cannot_gate_spatial_pose_likelihood() -> None:
    kwargs = dict(
        token_indices=np.asarray([0], dtype=np.int64),
        xy=np.asarray([[320.0, 240.0]], dtype=np.float64),
        track_ids=np.asarray([[1]], dtype=np.int64),
        prototype_ids=np.asarray([[0]], dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]], dtype=np.float64),
        descriptor_scores=np.asarray([[0.8]], dtype=np.float64),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.2], dtype=np.float64),
        candidate_update_probabilities=np.asarray([[0.99]], dtype=np.float64),
        candidate_refined_xy=np.asarray([[[321.0, 240.0]]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="action posterior"):
        PoseVerificationCandidatePool(**kwargs, spatial_utility_gate_weight=1.0)


def test_grouped_spatial_mode_sampling_uses_calibrated_reliability() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-2.0, 0.0, 2.0]),
            np.asarray([-2.0, 0.0, 2.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    probabilities = np.full((1, 1, 1, 9), 1e-12, dtype=np.float64)
    probabilities[0, 0, 0, -1] = 1.0 - 8e-12
    spatial = CandidateSpatialLikelihood(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probabilities),
        view_probabilities=np.ones((1, 1, 1), dtype=np.float64),
        dustbin_probabilities=np.ones((1, 1, 1), dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    kwargs = dict(
        token_indices=np.asarray([0], dtype=np.int64),
        xy=np.asarray([[100.0, 120.0]], dtype=np.float64),
        track_ids=np.asarray([[1]], dtype=np.int64),
        prototype_ids=np.asarray([[0]], dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]], dtype=np.float64),
        descriptor_scores=np.asarray([[0.9]], dtype=np.float64),
        valid_mask=np.ones((1, 1), dtype=bool),
        measurement_geometry_probabilities=np.ones((1, 1), dtype=np.float64),
        null_scores=np.asarray([0.1], dtype=np.float64),
        spatial_likelihood=spatial,
    )
    raw = PoseVerificationCandidatePool(**kwargs)
    calibrated = PoseVerificationCandidatePool(
        **kwargs, spatial_geometry_calibration_weight=1.0
    )

    raw_xy, raw_used = _sample_candidate_spatial_xy(
        raw, 0, 0, np.random.default_rng(3), enabled=True
    )
    calibrated_xy, calibrated_used = _sample_candidate_spatial_xy(
        calibrated, 0, 0, np.random.default_rng(3), enabled=True
    )

    np.testing.assert_array_equal(raw_xy, raw.xy[0])
    assert not raw_used
    np.testing.assert_allclose(calibrated_xy, calibrated.xy[0] + offsets[-1])
    assert calibrated_used


def test_grouped_spatial_mode_sampling_respects_reliability_mass() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-1.0, 0.0, 1.0]),
            np.asarray([-1.0, 0.0, 1.0]),
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
        dustbin_probabilities=np.full((1, 1, 1), 0.8, dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.asarray([0], dtype=np.int64),
        xy=np.asarray([[100.0, 120.0]], dtype=np.float64),
        track_ids=np.asarray([[1]], dtype=np.int64),
        prototype_ids=np.asarray([[0]], dtype=np.int64),
        xyz=np.asarray([[[0.0, 0.0, 8.0]]], dtype=np.float64),
        descriptor_scores=np.asarray([[0.9]], dtype=np.float64),
        valid_mask=np.ones((1, 1), dtype=bool),
        null_scores=np.asarray([0.1], dtype=np.float64),
        spatial_likelihood=spatial,
    )
    rng = np.random.default_rng(31)
    used = [
        _sample_candidate_spatial_xy(pool, 0, 0, rng, enabled=True)[1]
        for _ in range(4000)
    ]

    assert 0.17 < float(np.mean(used)) < 0.23


def test_spatial_likelihood_is_invariant_to_logit_constant_shift() -> None:
    offsets = np.stack(
        np.meshgrid(
            np.asarray([-1.0, 0.0, 1.0]),
            np.asarray([-1.0, 0.0, 1.0]),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    logits = np.linspace(-4.0, 3.0, len(offsets), dtype=np.float64).reshape(
        1, 1, 1, -1
    )
    kwargs = dict(
        offsets_xy=offsets,
        view_probabilities=np.ones((1, 1, 1), dtype=np.float64),
        dustbin_probabilities=np.zeros((1, 1, 1), dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    baseline = CandidateSpatialLikelihood(
        **kwargs, local_log_probabilities=logits
    )
    shifted = CandidateSpatialLikelihood(
        **kwargs, local_log_probabilities=logits + 1000.0
    )

    np.testing.assert_allclose(
        baseline._probability_maps,
        shifted._probability_maps,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.sum(baseline._probability_maps, axis=(-2, -1)),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


def _selection_result(
    *,
    success: bool,
    strict_grid_cells: int,
    likelihood: float = -1.0,
    denominator_sha256: str = "same-denominator",
    final_likelihood: float | None = None,
    final_audit_denominator_sha256: str | None = None,
    effective_groups: int = 12,
    translation_information_min_eigenvalue: float = 100.0,
    translation_information_condition: float = 10.0,
    joint_information_condition: float = 100.0,
) -> VerifiedPnPResult:
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
        fixed_posterior_log_likelihood_sum=float(likelihood * 12),
        fixed_posterior_log_likelihood_mean=float(likelihood),
        fixed_posterior_effective_group_count=int(effective_groups),
        information_match_count=12,
        translation_information_min_eigenvalue=float(
            translation_information_min_eigenvalue
        ),
        translation_information_condition=float(
            translation_information_condition
        ),
        joint_information_condition=float(joint_information_condition),
        bearing_max_angle_deg=20.0,
        camera_depth_span_ratio=0.5,
        xyz_second_singular_ratio=0.4,
        xyz_third_singular_ratio=0.2,
    )
    final_verification = replace(
        verification,
        fixed_posterior_log_likelihood_sum=float(
            (likelihood if final_likelihood is None else final_likelihood) * 12
        ),
        fixed_posterior_log_likelihood_mean=float(
            likelihood if final_likelihood is None else final_likelihood
        ),
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
        final_verification=final_verification if success else None,
        verification_denominator_sha256=(
            str(denominator_sha256) if success else None
        ),
        final_audit_denominator_sha256=(
            str(
                denominator_sha256
                if final_audit_denominator_sha256 is None
                else final_audit_denominator_sha256
            )
            if success
            else None
        ),
    )


def test_fixed_posterior_rank_key_excludes_pose_self_consistency_tiebreaks() -> None:
    baseline = _selection_result(success=True, strict_grid_cells=2, likelihood=-0.5)
    self_consistent = _selection_result(
        success=True,
        strict_grid_cells=12,
        likelihood=-0.5,
    )

    assert baseline.final_verification is not None
    assert self_consistent.final_verification is not None
    assert baseline.final_verification.rank_key() != self_consistent.final_verification.rank_key()
    assert baseline.final_verification.fixed_posterior_rank_key() == (-0.5,)
    assert (
        baseline.final_verification.fixed_posterior_rank_key()
        == self_consistent.final_verification.fixed_posterior_rank_key()
    )


def test_fixed_posterior_final_refine_requires_strict_likelihood_gain() -> None:
    from feature_extract.vfm.localization.pose_hypothesis_verifier import (
        _accept_grouped_final_refine,
    )

    baseline = _selection_result(success=True, strict_grid_cells=2, likelihood=-0.5)
    tied = _selection_result(success=True, strict_grid_cells=12, likelihood=-0.5)
    improved = _selection_result(success=True, strict_grid_cells=1, likelihood=-0.4)

    assert baseline.final_verification is not None
    assert tied.final_verification is not None
    assert improved.final_verification is not None
    assert not _accept_grouped_final_refine(
        baseline.final_verification,
        tied.final_verification,
        policy="fixed_posterior_likelihood_gain",
    )
    assert _accept_grouped_final_refine(
        baseline.final_verification,
        improved.final_verification,
        policy="fixed_posterior_likelihood_gain",
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


def test_crossfit_likelihood_gate_promotes_only_on_fixed_denominator_gain() -> None:
    baseline = _selection_result(
        success=True, strict_grid_cells=9, likelihood=-0.8
    )
    stronger = _selection_result(
        success=True, strict_grid_cells=3, likelihood=-0.7
    )
    weaker = _selection_result(
        success=True, strict_grid_cells=16, likelihood=-0.9
    )

    selected_stronger, promoted = (
        select_crossfit_likelihood_with_immutable_baseline(
            baseline,
            stronger,
            min_log_likelihood_mean_delta=0.05,
            min_effective_group_count=8,
        )
    )
    selected_weaker, abstained = (
        select_crossfit_likelihood_with_immutable_baseline(
            baseline,
            weaker,
            min_log_likelihood_mean_delta=0.05,
            min_effective_group_count=8,
        )
    )

    assert selected_stronger is stronger
    assert promoted["promoted"]
    assert selected_weaker is baseline
    assert abstained["abstained"]
    assert abstained["fallback_reason"] == "crossfit_likelihood_ratio_gate_abstained"


def test_crossfit_gate_scores_actual_final_pose_on_independent_audit() -> None:
    baseline = _selection_result(
        success=True,
        strict_grid_cells=9,
        likelihood=-0.9,
        final_likelihood=-0.7,
    )
    optional = _selection_result(
        success=True,
        strict_grid_cells=9,
        likelihood=-0.5,
        final_likelihood=-0.9,
    )

    selected, audit = select_crossfit_likelihood_with_immutable_baseline(
        baseline,
        optional,
        min_log_likelihood_mean_delta=0.05,
    )

    assert selected is baseline
    assert audit["evidence_partition"] == "final_audit"
    assert audit["baseline_log_likelihood_mean"] == -0.7
    assert audit["optional_log_likelihood_mean"] == -0.9


def test_crossfit_likelihood_gate_rejects_denominator_mismatch() -> None:
    baseline = _selection_result(success=True, strict_grid_cells=9)
    mismatched = _selection_result(
        success=True,
        strict_grid_cells=9,
        denominator_sha256="different-denominator",
    )

    with np.testing.assert_raises_regex(ValueError, "different held-out"):
        select_crossfit_likelihood_with_immutable_baseline(baseline, mismatched)


def test_crossfit_likelihood_abstain_returns_baseline_bit_exact() -> None:
    baseline = _selection_result(
        success=True, strict_grid_cells=9, likelihood=-0.8
    )
    optional = _selection_result(
        success=True, strict_grid_cells=9, likelihood=-0.79
    )

    selected, audit = select_crossfit_likelihood_with_immutable_baseline(
        baseline,
        optional,
        min_log_likelihood_mean_delta=0.05,
    )

    assert selected is baseline
    assert selected.pose_w2c is baseline.pose_w2c
    assert audit["abstained"]


def test_crossfit_likelihood_observability_veto_rejects_weak_translation() -> None:
    baseline = _selection_result(
        success=True, strict_grid_cells=9, likelihood=-0.8
    )
    optional = _selection_result(
        success=True,
        strict_grid_cells=9,
        likelihood=-0.6,
        translation_information_min_eigenvalue=0.01,
    )

    selected, audit = select_crossfit_likelihood_with_immutable_baseline(
        baseline,
        optional,
        min_translation_information_eigenvalue=1.0,
    )

    assert selected is baseline
    assert audit["fallback_reason"] == "observability_gate_veto"
    assert audit["observability_gate_failures"] == [
        "translation_information_min_eigenvalue"
    ]


def test_grouped_generation_observability_gate_rejects_planar_geometry() -> None:
    xyz = np.asarray(
        [
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [0.0, -0.5, 5.0],
            [0.5, 0.5, 5.0],
        ],
        dtype=np.float64,
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    matches = [
        _match(index, tuple(xy[index]), tuple(xyz[index]))
        for index in range(len(xyz))
    ]

    failures = grouped_pose_observability_gate_failures(
        np.eye(4, dtype=np.float64),
        matches,
        _camera(),
        GroupedCandidatePnPConfig(
            prosac_min_xyz_third_singular_ratio=0.01
        ),
    )

    assert failures == ("xyz_third_singular_ratio",)


def test_grouped_observability_can_use_resolved_fit_consensus() -> None:
    xyz = np.asarray(
        [
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [-1.5, -0.5, 6.0],
            [1.5, -0.5, 7.0],
            [-0.5, 1.5, 8.0],
            [0.5, 1.5, 9.0],
        ],
        dtype=np.float64,
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    sample = [
        _match(index, tuple(xy[index]), tuple(xyz[index]))
        for index in range(4)
    ]
    count = len(xyz)
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(count, dtype=np.int64)[:, None],
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz[:, None, :],
        descriptor_scores=np.ones((count, 1), dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        null_scores=np.zeros((count,), dtype=np.float64),
    )
    config = GroupedCandidatePnPConfig(
        prosac_min_xyz_third_singular_ratio=0.01,
        prosac_observability_evidence_mode="resolved_fit_consensus",
    )

    assert _generated_pose_observability_failures(
        np.eye(4, dtype=np.float64), sample, pool, _camera(), config
    ) == ()
    assert _generated_pose_observability_failures(
        np.eye(4, dtype=np.float64),
        sample,
        pool,
        _camera(),
        replace(config, prosac_observability_evidence_mode="minimal_sample"),
    ) == ("xyz_third_singular_ratio",)


def test_empty_result_preserves_generated_pose_audit() -> None:
    record = PoseHypothesisRecord(
        fit_match_count_limit=4,
        fit_match_count=4,
        selection_mode="raw",
        ransac_threshold_px=2.0,
        rng_seed_offset=0,
        solver_success=True,
        fit_inlier_count=4,
        verification=None,
    )
    pose = np.eye(4, dtype=np.float64)

    result = _empty_result(
        8,
        fit_count=4,
        verification_count=4,
        hypotheses=(record,),
        hypothesis_poses_w2c=(pose,),
    )

    assert result.hypothesis_poses_w2c[0] is pose
    with pytest.raises(ValueError, match="equal length"):
        _empty_result(
            8,
            fit_count=4,
            verification_count=4,
            hypotheses=(record,),
            hypothesis_poses_w2c=(),
        )


def test_crossfit_observability_veto_also_applies_when_baseline_failed() -> None:
    baseline = _selection_result(success=False, strict_grid_cells=0)
    optional = _selection_result(
        success=True,
        strict_grid_cells=9,
        likelihood=-0.6,
        translation_information_min_eigenvalue=0.01,
    )

    selected, audit = select_crossfit_likelihood_with_immutable_baseline(
        baseline,
        optional,
        min_translation_information_eigenvalue=1.0,
    )

    assert selected is baseline
    assert not audit["promoted"]
    assert audit["fallback_reason"] == "observability_gate_veto"


def test_shared_denominator_reverification_keeps_baseline_pose_bit_exact() -> None:
    rng = np.random.default_rng(73)
    count = 24
    xyz = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(5.0, 10.0, count),
        ]
    )
    xy = np.column_stack(
        [
            500.0 * xyz[:, 0] / xyz[:, 2] + 320.0,
            500.0 * xyz[:, 1] / xyz[:, 2] + 240.0,
        ]
    )
    pool = PoseVerificationCandidatePool(
        token_indices=np.arange(count, dtype=np.int64),
        xy=xy,
        track_ids=np.arange(1000, 1000 + count, dtype=np.int64)[:, None],
        prototype_ids=np.zeros((count, 1), dtype=np.int64),
        xyz=xyz[:, None, :],
        descriptor_scores=np.full((count, 1), 0.9, dtype=np.float64),
        valid_mask=np.ones((count, 1), dtype=bool),
        null_scores=np.full((count,), 0.1, dtype=np.float64),
    )
    baseline = _selection_result(success=True, strict_grid_cells=4)
    config = GroupedCandidatePnPConfig(
        crossfit_mode="token_track_component",
        min_fit_matches=4,
    )

    rescored = reverify_grouped_result_on_shared_denominator(
        baseline,
        pool,
        _camera(),
        config=config,
        query_seed=19,
    )

    assert rescored.pose_w2c is baseline.pose_w2c
    assert rescored.pre_refine_pose_w2c is baseline.pre_refine_pose_w2c
    np.testing.assert_array_equal(rescored.pose_w2c, baseline.pose_w2c)
    assert rescored.verification_denominator_sha256 not in {
        None,
        baseline.verification_denominator_sha256,
    }
    assert rescored.final_audit_denominator_sha256 not in {
        None,
        baseline.final_audit_denominator_sha256,
    }
    assert rescored.pre_refine_verification is not None
    assert rescored.final_verification is not None
    assert rescored.pre_refine_verification.information_match_count >= 4


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
