from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.eval_pose_hypothesis_verification import (
    _hypothesis_is_chosen_for_audit,
    _load_immutable_baseline_pose_artifact,
    _pose_summary,
    _query_execution_shard,
    _resolve_spatial_view_mixture_probabilities,
    _train_grouped_export_requires_geometry,
    _write_selected_pose_artifact,
    main as eval_pose_hypothesis_main,
)
from feature_extract.tools.vfm.merge_pose_hypothesis_shards import (
    _merge_row_sections,
    _merge_selected_pose_rows,
)


def test_pose_summary_reports_full_accuracy_and_uncertainty_contract() -> None:
    rows = [
        {
            "query_id": "q0",
            "success": True,
            "translation_m": 0.02,
            "rotation_deg": 0.2,
            "match_count": 10,
            "inlier_count": 8,
        },
        {
            "query_id": "q1",
            "success": True,
            "translation_m": 0.08,
            "rotation_deg": 0.4,
            "match_count": 12,
            "inlier_count": 9,
        },
        {
            "query_id": "q2",
            "success": False,
            "match_count": 3,
            "inlier_count": 0,
        },
    ]

    summary = _pose_summary(rows)

    assert summary["recall_3cm_5deg"] == 1.0 / 3.0
    assert summary["p90_rotation_deg_success"] == 0.38
    assert len(summary["median_translation_m_success_bootstrap95_ci"]) == 2
    assert len(summary["p90_translation_m_success_bootstrap95_ci"]) == 2
    assert len(summary["median_rotation_deg_success_bootstrap95_ci"]) == 2
    assert len(summary["p90_rotation_deg_success_bootstrap95_ci"]) == 2


def test_query_execution_shards_are_balanced_complete_and_disjoint() -> None:
    query_ids = tuple(f"query_{index}" for index in range(11))
    shards = [
        _query_execution_shard(query_ids, shard_count=4, shard_index=index)
        for index in range(4)
    ]

    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert set().union(*(set(shard) for shard in shards)) == set(query_ids)
    assert sum(map(len, shards)) == len(query_ids)
    assert shards[0] == ("query_0", "query_4", "query_8")
    with pytest.raises(ValueError, match="query_shard_index"):
        _query_execution_shard(query_ids, shard_count=4, shard_index=4)


def test_train_grouped_export_only_requires_geometry_when_consumed() -> None:
    base = {
        "export_train_grouped_hypotheses": True,
        "candidate_geometry_prior_mix_weight": 0.0,
        "candidate_geometry_generation_mix_weight": 0.0,
        "candidate_spatial_geometry_calibration_weight": 0.0,
    }

    assert not _train_grouped_export_requires_geometry(SimpleNamespace(**base))
    for field in (
        "candidate_geometry_prior_mix_weight",
        "candidate_geometry_generation_mix_weight",
        "candidate_spatial_geometry_calibration_weight",
    ):
        configured = dict(base)
        configured[field] = 0.25
        assert _train_grouped_export_requires_geometry(
            SimpleNamespace(**configured)
        )
    disabled = dict(base)
    disabled["export_train_grouped_hypotheses"] = False
    disabled["candidate_geometry_generation_mix_weight"] = 1.0
    assert not _train_grouped_export_requires_geometry(SimpleNamespace(**disabled))


@pytest.mark.parametrize(
    "policy",
    (
        "fixed_posterior_median_DIAGNOSTIC_ONLY",
        "fixed_posterior_trimmed_mean_10_DIAGNOSTIC_ONLY",
        "fixed_posterior_worst_quartile_mean_DIAGNOSTIC_ONLY",
        "fixed_posterior_lcb95_DIAGNOSTIC_ONLY",
        "fixed_posterior_spatial_mom_2x2_DIAGNOSTIC_ONLY",
        "legacy_self_consistency_DIAGNOSTIC_ONLY",
    ),
)
def test_robust_and_legacy_selectors_are_rejected_for_untouched_test(
    policy: str,
) -> None:
    with pytest.raises(ValueError, match="development-only"):
        eval_pose_hypothesis_main(
            [
                "--proposals",
                "unused-proposals.npz",
                "--candidate_artifact",
                "unused-candidates.npz",
                "--score_artifact",
                "unused-scores.npz",
                "--projected_landmark_bank",
                "unused-bank.npz",
                "--colmap_model_dir",
                "unused-model",
                "--split_json",
                "unused-split.json",
                "--output_dir",
                "unused-output",
                "--evaluation_role",
                "untouched_test",
                "--grouped_hypothesis_selection_policy",
                policy,
            ]
        )


def test_selected_pose_artifact_replays_pose_bit_exact_and_checks_scene(
    tmp_path,
) -> None:
    source_pose = np.eye(4, dtype=np.float64)
    source_pose[0, 3] = np.float64(0.0123456789012345)
    source_manifest = {
        "inputs": {
            "colmap_cameras_bin_sha256": "camera-hash",
            "colmap_images_bin_sha256": "image-hash",
            "score_artifact_sha256": "source-score-hash",
        },
        "protocol": {"ground_truth_available_to_pose_selector": False},
        "config": {"generation": "frozen"},
        "execution": {"query_ids": {"validation": ["q0"]}},
    }
    artifact_path = tmp_path / "selected_pose.npz"
    _write_selected_pose_artifact(
        artifact_path,
        [
            {
                "query_id": "q0",
                "split_name": "validation",
                "evaluation_label": "grouped_candidate_pool__score",
                "success": True,
                "pose_w2c": source_pose,
                "match_count": 32,
                "inlier_count": 19,
            }
        ],
        source_manifest=source_manifest,
    )

    loaded = _load_immutable_baseline_pose_artifact(
        artifact_path,
        expected_colmap_cameras_sha256="camera-hash",
        expected_colmap_images_sha256="image-hash",
    )
    replay = loaded["records"][("validation", "q0")]

    np.testing.assert_array_equal(replay.pose_w2c, source_pose)
    assert replay.match_count == 32
    assert replay.inlier_count == 19
    assert loaded["evaluation_label"] == "grouped_candidate_pool__score"
    with pytest.raises(ValueError, match="different COLMAP scene"):
        _load_immutable_baseline_pose_artifact(
            artifact_path,
            expected_colmap_cameras_sha256="wrong-camera-hash",
            expected_colmap_images_sha256="image-hash",
        )


def test_selected_pose_artifact_requires_explicit_policy_when_ambiguous(
    tmp_path,
) -> None:
    source_manifest = {
        "inputs": {
            "colmap_cameras_bin_sha256": "camera-hash",
            "colmap_images_bin_sha256": "image-hash",
        }
    }
    rows = []
    for label in ("policy_a", "policy_b"):
        rows.append(
            {
                "query_id": "q0",
                "split_name": "validation",
                "evaluation_label": label,
                "success": True,
                "pose_w2c": np.eye(4, dtype=np.float64),
                "match_count": 8,
                "inlier_count": 6,
            }
        )
    artifact_path = tmp_path / "selected_pose.npz"
    _write_selected_pose_artifact(
        artifact_path, rows, source_manifest=source_manifest
    )

    with pytest.raises(ValueError, match="multiple policies"):
        _load_immutable_baseline_pose_artifact(
            artifact_path,
            expected_colmap_cameras_sha256="camera-hash",
            expected_colmap_images_sha256="image-hash",
        )
    loaded = _load_immutable_baseline_pose_artifact(
        artifact_path,
        expected_colmap_cameras_sha256="camera-hash",
        expected_colmap_images_sha256="image-hash",
        evaluation_label="policy_b",
    )
    assert loaded["evaluation_label"] == "policy_b"


def test_optional_hypothesis_chosen_marker_does_not_depend_on_fallback_object() -> None:
    optional = SimpleNamespace(chosen_hypothesis_index=7)

    assert _hypothesis_is_chosen_for_audit(optional, 7)
    assert not _hypothesis_is_chosen_for_audit(optional, 6)


def test_uniform_spatial_view_mixture_uses_every_measured_view_only() -> None:
    frozen = np.asarray(
        [[[0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    valid = np.asarray(
        [[[True, True, True, False], [False, True, False, True]]],
        dtype=bool,
    )

    resolved = _resolve_spatial_view_mixture_probabilities(
        frozen,
        valid,
        policy="uniform_all_measured_views",
    )

    np.testing.assert_allclose(resolved[0, 0], [1.0 / 3.0] * 3 + [0.0])
    np.testing.assert_allclose(resolved[0, 1], [0.0, 0.5, 0.0, 0.5])


def test_frozen_spatial_view_mixture_preserves_missing_probability_mass() -> None:
    frozen = np.asarray([[[0.4, 0.3, 0.0]]], dtype=np.float64)
    valid = np.ones_like(frozen, dtype=bool)

    resolved = _resolve_spatial_view_mixture_probabilities(
        frozen,
        valid,
        policy="frozen_candidate_posterior",
    )

    np.testing.assert_array_equal(resolved, frozen.astype(np.float32))
    assert float(np.sum(resolved)) < 1.0


def test_artifact_pose_view_mixture_requires_unit_mass_per_measured_candidate() -> None:
    learned = np.asarray(
        [[[0.1, 0.2, 0.7, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    valid = learned > 0.0

    resolved = _resolve_spatial_view_mixture_probabilities(
        learned,
        valid,
        policy="artifact_pose_view_posterior",
    )

    np.testing.assert_allclose(resolved, learned.astype(np.float32))
    invalid = learned.copy()
    invalid[0, 0, 2] = 0.6
    with pytest.raises(ValueError, match="must equal one"):
        _resolve_spatial_view_mixture_probabilities(
            invalid,
            valid,
            policy="artifact_pose_view_posterior",
        )


def test_merge_pose_rows_rejects_duplicate_queries_and_restores_split_order() -> None:
    def row(query_id: str, translation: float) -> dict[str, object]:
        return {
            "query_id": query_id,
            "success": True,
            "translation_m": translation,
            "rotation_deg": 0.1,
            "match_count": 8,
            "inlier_count": 8,
        }

    shards = [
        {
            "train_oof_geometry": {},
            "validation": {"policy": {"verified": [row("q0", 0.1)]}},
            "late_development": {},
        },
        {
            "train_oof_geometry": {},
            "validation": {"policy": {"verified": [row("q1", 0.2)]}},
            "late_development": {},
        },
    ]
    split = {"train": ["t"], "validation": ["q0", "q1"], "test": ["x"]}

    merged, metrics = _merge_row_sections(shards, split=split)

    rows = merged["validation"]["policy"]["verified"]
    assert [item["query_id"] for item in rows] == ["q0", "q1"]
    assert metrics["validation"]["policy"]["verified"]["query_count"] == 2

    shards[1]["validation"]["policy"]["verified"][0]["query_id"] = "q0"
    with pytest.raises(ValueError, match="duplicate query"):
        _merge_row_sections(shards, split=split)


def test_merge_selected_pose_rows_restores_order_and_rejects_partial_export() -> None:
    def row(query_id: str, split_name: str) -> dict[str, object]:
        return {
            "query_id": query_id,
            "split_name": split_name,
            "evaluation_label": "policy",
            "success": True,
            "pose_w2c": np.eye(4, dtype=np.float64),
            "match_count": 8,
            "inlier_count": 6,
        }

    split = {
        "train": ["t0", "t1"],
        "validation": ["q0", "q1"],
        "test": ["x0", "x1"],
    }
    shards = [
        [row("q0", "validation"), row("x0", "test")],
        [row("q1", "validation"), row("x1", "test")],
    ]

    merged = _merge_selected_pose_rows(shards, split=split)

    assert merged is not None
    assert [item["query_id"] for item in merged] == ["q0", "q1", "x0", "x1"]
    with pytest.raises(ValueError, match="only some shards"):
        _merge_selected_pose_rows([shards[0], None], split=split)


def test_pose_summary_reports_generation_profiles_separately() -> None:
    def row(query_id: str, oracle: float, all_correct: int) -> dict[str, object]:
        return {
            "query_id": query_id,
            "success": True,
            "translation_m": 0.2,
            "rotation_deg": 0.4,
            "match_count": 8,
            "inlier_count": 8,
            "hypothesis_profile_audit_TARGET_ONLY": {
                "profile_a": {
                    "valid_hypothesis_count": 10,
                    "raw_hypothesis_count": 8,
                    "latent_em_hypothesis_count": 2,
                    "oracle_translation_m": oracle,
                    "oracle_rotation_deg": 0.2,
                    "oracle_3cm_5deg": oracle <= 0.03,
                    "oracle_5cm_5deg": oracle <= 0.05,
                    "oracle_10cm_5deg": oracle <= 0.10,
                    "oracle_25cm_2deg": oracle <= 0.25,
                    "catastrophic_hypothesis_count": 1,
                    "minimal_sample_count": 4,
                    "minimal_sample_pair_count": 16,
                    "minimal_sample_correct_2px_pair_count": 8,
                    "minimal_sample_correct_5px_pair_count": 12,
                    "minimal_sample_all_correct_2px_count": all_correct,
                    "minimal_sample_all_correct_5px_count": 2,
                }
            },
        }

    summary = _pose_summary([row("q0", 0.02, 1), row("q1", 0.08, 0)])
    profile = summary["hypothesis_profile_audit_TARGET_ONLY"]["profile_a"]

    assert profile["oracle_median_translation_m"] == 0.05
    assert profile["oracle_recall_3cm_5deg"] == 0.5
    assert profile["minimal_sample_all_correct_2px_rate"] == 0.125
    assert profile["minimal_sample_correct_5px_pair_rate"] == 0.75


def test_pose_summary_reports_latent_em_against_its_parent_seed() -> None:
    rows = []
    for index, values in enumerate(
        (
            (4, 3, 1, -0.02, 0.03, 8, 0.06, 12, -0.01),
            (2, 0, 2, 0.04, 0.10, 4, 0.14, 31, 0.03),
        )
    ):
        (
            pair_count,
            wins,
            losses,
            median_delta,
            max_delta,
            seed_count,
            seed_oracle,
            seed_rank,
            refinement_delta,
        ) = values
        rows.append(
            {
                "query_id": f"q{index}",
                "success": True,
                "translation_m": 0.2,
                "rotation_deg": 0.4,
                "match_count": 8,
                "inlier_count": 8,
                "latent_em_parent_pair_count_TARGET_ONLY": pair_count,
                "latent_em_parent_win_count_TARGET_ONLY": wins,
                "latent_em_parent_loss_count_TARGET_ONLY": losses,
                "latent_em_parent_median_translation_delta_m_TARGET_ONLY": median_delta,
                "latent_em_parent_max_translation_delta_m_TARGET_ONLY": max_delta,
                "raw_hypothesis_oracle_translation_m_TARGET_ONLY": 0.04,
                "latent_em_seed_parent_count_TARGET_ONLY": seed_count,
                "latent_em_seed_parent_oracle_translation_m_TARGET_ONLY": seed_oracle,
                "latent_em_seed_parent_raw_rank_TARGET_ONLY": seed_rank,
                "latent_em_refined_minus_seed_oracle_translation_m_TARGET_ONLY": refinement_delta,
            }
        )

    summary = _pose_summary(rows)

    assert summary["latent_em_parent_pair_count_TARGET_ONLY"] == 6
    assert summary["latent_em_parent_win_rate_TARGET_ONLY"] == 0.5
    assert summary[
        "latent_em_parent_median_query_median_translation_delta_m_TARGET_ONLY"
    ] == pytest.approx(0.01)
    assert summary[
        "latent_em_parent_worst_translation_regression_m_TARGET_ONLY"
    ] == 0.10
    assert summary["latent_em_seed_parent_count_median_TARGET_ONLY"] == 6.0
    assert summary[
        "latent_em_seed_parent_oracle_median_translation_m_TARGET_ONLY"
    ] == pytest.approx(0.10)
    assert summary[
        "latent_em_seed_parent_raw_rank_median_TARGET_ONLY"
    ] == pytest.approx(21.5)
    assert summary[
        "latent_em_seed_selection_oracle_gap_median_m_TARGET_ONLY"
    ] == pytest.approx(0.06)
    assert summary[
        "latent_em_refined_minus_seed_oracle_median_translation_m_TARGET_ONLY"
    ] == pytest.approx(0.01)
    assert summary[
        "latent_em_refined_beats_seed_oracle_rate_TARGET_ONLY"
    ] == pytest.approx(0.5)


def test_pose_summary_reports_raw_generation_without_latent_hypotheses() -> None:
    rows = []
    for index, (translation, rotation) in enumerate(
        ((0.02, 0.2), (0.08, 0.4), (0.30, 1.0))
    ):
        rows.append(
            {
                "query_id": f"q{index}",
                "success": True,
                "translation_m": 0.5,
                "rotation_deg": 2.0,
                "match_count": 8,
                "inlier_count": 8,
                "raw_hypothesis_oracle_translation_m_TARGET_ONLY": translation,
                "raw_hypothesis_oracle_rotation_deg_TARGET_ONLY": rotation,
                "latent_em_hypothesis_oracle_translation_m_TARGET_ONLY": None,
            }
        )

    summary = _pose_summary(rows)

    assert summary["raw_hypothesis_oracle_query_count"] == 3
    assert summary[
        "raw_hypothesis_oracle_median_translation_m_TARGET_ONLY"
    ] == 0.08
    assert summary[
        "raw_hypothesis_oracle_p90_translation_m_TARGET_ONLY"
    ] == pytest.approx(0.256)
    assert summary[
        "raw_hypothesis_oracle_recall_3cm_5deg_TARGET_ONLY"
    ] == pytest.approx(1.0 / 3.0)
    assert summary[
        "raw_hypothesis_oracle_recall_10cm_5deg_TARGET_ONLY"
    ] == pytest.approx(2.0 / 3.0)
    assert summary[
        "raw_hypothesis_oracle_recall_25cm_2deg_TARGET_ONLY"
    ] == pytest.approx(2.0 / 3.0)
    assert "latent_em_only_oracle_median_translation_m_TARGET_ONLY" not in summary


def test_pose_summary_separates_generation_shortlist_and_final_selection() -> None:
    rows = []
    for index, values in enumerate(((0.02, 0.05, 0.10), (0.04, 0.08, 0.20))):
        full_oracle, verified_oracle, selected = values
        rows.append(
            {
                "query_id": f"q{index}",
                "success": True,
                "translation_m": selected,
                "rotation_deg": 0.4,
                "match_count": 8,
                "inlier_count": 8,
                "hypothesis_oracle_translation_m": full_oracle,
                "hypothesis_oracle_rotation_deg": 0.2,
                "hypothesis_oracle_3cm_5deg": full_oracle <= 0.03,
                "hypothesis_oracle_5cm_5deg": full_oracle <= 0.05,
                "hypothesis_oracle_10cm_5deg": True,
                "hypothesis_oracle_25cm_2deg": True,
                "chosen_hypothesis_translation_rank": 2,
                "valid_hypothesis_count": 100,
                "hypothesis_3cm_5deg_count": 1,
                "hypothesis_10cm_5deg_count": 5,
                "hypothesis_25cm_2deg_count": 10,
                "catastrophic_hypothesis_count": 2,
                "verified_hypothesis_count": 16,
                "verified_hypothesis_oracle_translation_m_TARGET_ONLY": verified_oracle,
                "verified_hypothesis_oracle_rotation_deg_TARGET_ONLY": 0.3,
                "verified_hypothesis_oracle_3cm_5deg_TARGET_ONLY": (
                    verified_oracle <= 0.03
                ),
                "verified_hypothesis_oracle_5cm_5deg_TARGET_ONLY": (
                    verified_oracle <= 0.05
                ),
                "verified_hypothesis_oracle_10cm_5deg_TARGET_ONLY": (
                    verified_oracle <= 0.10
                ),
            }
        )

    summary = _pose_summary(rows)

    assert summary[
        "verified_hypothesis_oracle_median_translation_m_TARGET_ONLY"
    ] == pytest.approx(0.065)
    assert summary["median_shortlist_oracle_gap_m_TARGET_ONLY"] == pytest.approx(
        0.035
    )
    assert summary["p90_shortlist_oracle_gap_m_TARGET_ONLY"] == pytest.approx(
        0.039
    )
    assert summary["max_shortlist_oracle_gap_m_TARGET_ONLY"] == pytest.approx(
        0.04
    )
    assert summary["shortlist_oracle_degradation_rate_TARGET_ONLY"] == 1.0
    assert summary[
        "median_verified_selection_regret_m_TARGET_ONLY"
    ] == pytest.approx(0.085)
    assert summary[
        "p90_verified_selection_regret_m_TARGET_ONLY"
    ] == pytest.approx(0.113)
    assert summary["p90_hypothesis_selection_regret_m"] == pytest.approx(0.152)
