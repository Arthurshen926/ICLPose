import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _assignment_identity_gate,
    _balanced_train_groups,
    _build_query_split,
    _candidate_data_manifest_mismatches,
    _fit_static_feature_normalization,
    _global_assignment_strategy_scores,
    _identity_transition_report,
    _load_query_split_manifest,
    _load_frozen_global_baseline_policy,
    _load_refit_selection,
    _load_system_hard_score_artifact,
    _ordered_prefetch,
    _preserve_prior_row_confidence,
    _recalibrate_set_posterior,
    _system_hard_candidate_group_mask,
    _validate_frozen_baseline_pose,
)
from feature_extract.vfm.localization.candidate_maplet_data import (
    CandidateMapletEpisodeArrays,
    _load_candidate_proposal_arrays,
    build_candidate_maplet_assignment_targets,
)
from feature_extract.vfm.localization.candidate_maplet_schema import (
    CANDIDATE_MAPLET_DEPLOYABLE_FEATURE_NAMES,
    CANDIDATE_MAPLET_INFERENCE_COMPATIBILITY_KEYS,
    CANDIDATE_MAPLET_STATIC_FEATURE_NAMES,
    candidate_maplet_inference_manifest_mismatches,
    validate_candidate_maplet_static_feature_names,
)


def test_inference_proposal_loader_is_allow_listed(tmp_path) -> None:
    path = tmp_path / "proposals.npz"
    np.savez(
        path,
        query_ids=np.asarray(["query.png"]),
        xy=np.asarray([[12.0, 24.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[7, 8]], dtype=np.int64),
        candidate_prototype_ids=np.asarray([[0, 0]], dtype=np.int64),
        coarse_scores=np.asarray([[0.8, 0.7]], dtype=np.float32),
        strategy__alike_support_top2_mean=np.asarray(
            [[0.7, 0.6]], dtype=np.float32
        ),
        candidate_gt_residuals_px=np.asarray([[1.0, 9.0]], dtype=np.float32),
        nearest_visible_track_ids=np.asarray([7], dtype=np.int64),
        nearest_visible_residuals_px=np.asarray([0.5], dtype=np.float32),
        pose_keep_mask=np.asarray([True]),
        future_target_only_field=np.asarray([123.0], dtype=np.float32),
    )

    inference = _load_candidate_proposal_arrays(path, load_supervision=False)
    target_free = {
        "query_ids",
        "xy",
        "candidate_track_ids",
        "candidate_prototype_ids",
        "coarse_scores",
        "strategy__alike_support_top2_mean",
    }
    assert set(inference) == target_free

    supervised = _load_candidate_proposal_arrays(path, load_supervision=True)
    assert set(supervised) == target_free | {
        "candidate_gt_residuals_px",
        "nearest_visible_track_ids",
        "nearest_visible_residuals_px",
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_ordered_prefetch_preserves_input_order(enabled: bool) -> None:
    built = []

    def build(value: int) -> int:
        built.append(value)
        return value * 2

    assert list(_ordered_prefetch(range(5), build, enabled=enabled)) == [0, 2, 4, 6, 8]
    assert built == [0, 1, 2, 3, 4]
    assert list(_ordered_prefetch([], build, enabled=enabled)) == []


def test_system_hard_group_mining_targets_real_inference_failures() -> None:
    labels = np.asarray(
        [
            [True, False, False, False],
            [False, False, True, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    scores = np.asarray(
        [
            [0.90, 0.40, 0.30, 0.20],
            [0.90, 0.70, 0.60, 0.20],
            [0.60, 0.58, 0.20, 0.10],
            [0.90, 0.50, 0.30, 0.20],
            [0.40, 0.30, 0.20, 0.10],
            [0.99, 0.80, 0.70, 0.60],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(labels, dtype=bool)

    hard, audit = _system_hard_candidate_group_mask(
        labels,
        scores,
        valid,
        np.arange(5, dtype=np.int64),
        ambiguous_margin=0.05,
        no_match_quantile=0.75,
    )

    assert hard.tolist() == [False, True, True, True, False, False]
    assert audit["rank2_to_l_positive_group_count"] == 1
    assert audit["ambiguous_wrong_competitor_group_count"] == 2
    assert audit["high_score_no_match_group_count"] == 1

    store = SimpleNamespace(
        candidate_top_k=4,
        selected_rows=np.arange(6, dtype=np.int64),
        labels=labels,
    )
    train_edges = np.arange(5 * 4, dtype=np.int64)
    sampled = _balanced_train_groups(
        store,
        train_edges,
        no_match_ratio=1.0,
        max_groups=0,
        rng=np.random.default_rng(7),
        system_hard_group_mask=hard,
        system_hard_oversample_factor=2.0,
    )
    counts = np.bincount(sampled, minlength=6)
    assert counts.tolist() == [1, 2, 2, 2, 1, 0]


def test_identity_transition_report_separates_rescue_from_wrong_switch() -> None:
    labels = np.asarray(
        [
            [True, False, False],
            [False, False, True],
            [True, False, False],
            [False, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    valid = np.ones_like(labels, dtype=bool)
    baseline = np.asarray(
        [
            [0.9, 0.2, 0.1],
            [0.9, 0.5, 0.4],
            [0.9, 0.8, 0.1],
            [0.9, 0.7, 0.1],
            [0.9, 0.5, 0.1],
            [0.9, 0.4, 0.8],
        ],
        dtype=np.float32,
    )
    candidate = np.asarray(
        [
            [0.9, 0.2, 0.1],
            [0.3, 0.2, 0.9],
            [0.7, 0.9, 0.1],
            [0.7, 0.9, 0.1],
            [0.9, 0.5, 0.1],
            [0.8, 0.4, 0.9],
        ],
        dtype=np.float32,
    )

    report = _identity_transition_report(labels, valid, baseline, candidate)

    assert report["mappable_group_count"] == 5
    assert report["baseline_correct_count"] == 3
    assert report["candidate_correct_count"] == 3
    assert report["rank2_to_l_rescue_eligible_count"] == 2
    assert report["rank2_to_l_rescue_count"] == 1
    assert report["rank2_to_l_rescue_rate"] == pytest.approx(0.5)
    assert report["baseline_correct_corruption_count"] == 1
    assert report["wrong_switch_rate"] == pytest.approx(1.0 / 3.0)
    assert report["switch_count"] == 4
    assert report["beneficial_switch_count"] == 1
    assert report["harmful_switch_count"] == 1
    assert report["beneficial_switch_precision"] == pytest.approx(0.25)
    assert report["no_match_group_switch_count"] == 1
    assert report["unchanged_wrong_mappable_count"] == 1


def test_system_hard_score_artifact_requires_same_candidate_store(tmp_path) -> None:
    manifest = {
        "proposals_sha256": "proposal",
        "detector_query_cache_sha256": "detector",
        "query_context_detector_cache_sha256": "context",
        "support_feature_cache_sha256": "support",
        "support_geometry_index_sha256": "geometry",
        "projected_landmark_bank_sha256": "bank",
        "maplet_support_index_sha256": "maplet",
        "feature_artifact_sha256": "features",
        "query_split_manifest_sha256": "split",
        "global_assignment_baseline_summary_sha256": "baseline",
        "radio_intermediate_cache_sha256": "radio",
        "query_input_dim": 8,
        "support_input_dim": 9,
        "static_input_dim": 4,
        "candidate_top_k": 3,
        "static_feature_names": ["a", "b", "c", "d"],
        "positive_threshold_px": 2.0,
        "assignment_threshold_px": 5.0,
        "support_view_count": 2,
        "query_radius_px": 96.0,
        "max_query_nodes": 48,
        "max_support_tracks": 33,
    }
    path = tmp_path / "scores.npz"
    np.savez(
        path,
        selected_identity_scores=np.asarray(
            [[0.9, 0.2, -np.inf], [0.1, 0.8, 0.4]], dtype=np.float32
        ),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_maplet_ensemble_scores_v2",
                    "data_manifest": manifest,
                }
            )
        ),
    )

    scores, audit = _load_system_hard_score_artifact(
        path,
        score_key="selected_identity_scores",
        expected_manifest=manifest,
        expected_shape=(2, 3),
    )
    assert scores.shape == (2, 3)
    assert audit["score_key"] == "selected_identity_scores"
    assert audit["finite_score_fraction"] == pytest.approx(5.0 / 6.0)

    stale = {**manifest, "proposals_sha256": "other"}
    with pytest.raises(ValueError, match="stale or from a different"):
        _load_system_hard_score_artifact(
            path,
            score_key="selected_identity_scores",
            expected_manifest=stale,
            expected_shape=(2, 3),
        )


def test_runtime_manifest_ignores_legacy_training_curriculum_fields() -> None:
    runtime = {
        "proposals_sha256": "proposal",
        "projected_landmark_bank_sha256": "bank",
    }
    legacy_checkpoint = {
        **runtime,
        "system_hard_score_artifact_sha256": "hard-scores",
        "system_hard_score_key": "ensemble__candidate_probability",
    }

    assert not _candidate_data_manifest_mismatches(
        legacy_checkpoint,
        runtime,
        allow_legacy_missing_colmap_provenance=False,
    )
    assert _candidate_data_manifest_mismatches(
        legacy_checkpoint,
        {**runtime, "projected_landmark_bank_sha256": "other-bank"},
        allow_legacy_missing_colmap_provenance=False,
    ) == {
        "projected_landmark_bank_sha256": {
            "checkpoint": "bank",
            "current": "other-bank",
        }
    }


def test_global_assignment_set_probability_uses_candidate_vs_null_log_odds() -> None:
    predictions = {
        "set_candidate_probability": np.asarray(
            [0.2, 0.3, -np.inf, -np.inf], dtype=np.float32
        ),
        "set_dustbin_probability_DIAGNOSTIC_ONLY": np.asarray(
            [0.5, 0.5, -np.inf, -np.inf], dtype=np.float32
        ),
    }

    scores, dustbins, score_space = _global_assignment_strategy_scores(
        predictions,
        "set_candidate_probability",
        group_count=2,
        candidate_top_k=2,
    )

    np.testing.assert_allclose(
        scores[0],
        np.log(np.asarray([0.2, 0.3]) / 0.5),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.all(np.isneginf(scores[1]))
    np.testing.assert_array_equal(dustbins, np.zeros((2,), dtype=np.float32))
    assert score_space == "joint_set_posterior_candidate_vs_null_log_odds"

    invalid = dict(predictions)
    invalid["set_candidate_probability"] = np.asarray(
        [0.2, 0.2, -np.inf, -np.inf], dtype=np.float32
    )
    with pytest.raises(ValueError, match="conserve mass"):
        _global_assignment_strategy_scores(
            invalid,
            "set_candidate_probability",
            group_count=2,
            candidate_top_k=2,
        )


def test_global_assignment_factorized_probability_conserves_mass() -> None:
    predictions = {
        "factorized_set_candidate_probability": np.asarray(
            [0.12, 0.28, 0.10], dtype=np.float32
        ),
        "factorized_set_dustbin_probability_DIAGNOSTIC_ONLY": np.asarray(
            [0.50, 0.50, 0.50], dtype=np.float32
        ),
    }

    scores, dustbins, score_space = _global_assignment_strategy_scores(
        predictions,
        "factorized_set_candidate_probability",
        group_count=1,
        candidate_top_k=3,
    )

    np.testing.assert_allclose(
        scores[0], np.log(np.asarray([0.12, 0.28, 0.10]) / 0.50)
    )
    np.testing.assert_array_equal(dustbins, np.zeros((1,), dtype=np.float32))
    assert score_space == "joint_set_posterior_candidate_vs_null_log_odds"


def test_set_posterior_recalibration_matches_direct_logit_rescaling() -> None:
    prior = np.asarray([[0.8, 0.4, 0.1], [0.7, 0.6, 0.2]], dtype=np.float32)
    evidence = np.asarray([[0.3, -0.2, 0.1], [-0.4, 0.5, 0.2]], dtype=np.float64)
    dustbin_logits = np.asarray([0.2, -0.1], dtype=np.float64)
    source_scale = 4.0
    centered = prior - prior.mean(axis=1, keepdims=True)
    source_logits = np.concatenate(
        [evidence + source_scale * centered, dustbin_logits[:, None]], axis=1
    )
    source_probability = np.exp(source_logits - source_logits.max(axis=1, keepdims=True))
    source_probability /= source_probability.sum(axis=1, keepdims=True)

    candidate, dustbin = _recalibrate_set_posterior(
        source_probability[:, :-1],
        source_probability[:, -1],
        prior,
        source_prior_scale=source_scale,
        target_prior_scale=1.5,
        dustbin_logit_bias=-0.7,
    )

    expected_logits = np.concatenate(
        [evidence + 1.5 * centered, (dustbin_logits - 0.7)[:, None]], axis=1
    )
    expected = np.exp(expected_logits - expected_logits.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(candidate, expected[:, :-1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(dustbin, expected[:, -1], atol=1e-6, rtol=1e-6)


def test_global_assignment_uses_calibrated_variant_dustbin() -> None:
    strategy = "set_candidate_probability_cal_ps2_dbm1"
    predictions = {
        strategy: np.asarray([0.2, 0.3], dtype=np.float32),
        "set_dustbin_probability_cal_ps2_dbm1_DIAGNOSTIC_ONLY": np.asarray(
            [0.5, 0.5], dtype=np.float32
        ),
    }

    scores, dustbins, _score_space = _global_assignment_strategy_scores(
        predictions,
        strategy,
        group_count=1,
        candidate_top_k=2,
    )

    np.testing.assert_allclose(scores[0], np.log(np.asarray([0.2, 0.3]) / 0.5))
    np.testing.assert_array_equal(dustbins, np.zeros((1,), dtype=np.float32))


def test_candidate_maplet_targets_force_anchor_then_assign_unique_neighbors() -> None:
    residuals = np.asarray(
        [
            [1.5, 0.5, 9.0],
            [0.3, 4.0, 0.7],
            [5.0, 0.2, 0.4],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    targets = build_candidate_maplet_assignment_targets(
        residuals,
        threshold_px=2.0,
        candidate_label=True,
    )
    assert targets.tolist() == [0, 2, 1, 3]


def test_inference_episode_accepts_no_pose_derived_supervision() -> None:
    episode = CandidateMapletEpisodeArrays(
        query_features=np.ones((2, 4), dtype=np.float32),
        query_xy=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        support_features=np.ones((3, 5), dtype=np.float32),
        support_track_ids=np.asarray([10, 11, 12], dtype=np.int64),
        support_xyz=np.ones((3, 3), dtype=np.float32),
        static_features=np.ones((2,), dtype=np.float32),
        target_track_indices=None,
        candidate_label=None,
        anchor_residual_px=None,
        candidate_visible=None,
        edge_index=0,
        query_id="query.png",
        support_image_id="support.png",
    )

    assert episode.target_track_indices is None
    assert episode.anchor_residual_px is None


def test_candidate_maplet_targets_reject_label_geometry_mismatch() -> None:
    with pytest.raises(ValueError, match="disagree"):
        build_candidate_maplet_assignment_targets(
            np.asarray([[0.5, 3.0], [4.0, 0.2]], dtype=np.float32),
            threshold_px=2.0,
            candidate_label=False,
        )


def test_candidate_anchor_remains_dustbin_under_looser_context_threshold() -> None:
    targets = build_candidate_maplet_assignment_targets(
        np.asarray(
            [
                [3.0, 0.5],
                [0.4, 8.0],
                [9.0, 0.3],
            ],
            dtype=np.float32,
        ),
        threshold_px=5.0,
        anchor_threshold_px=2.0,
        candidate_label=False,
    )
    assert targets.tolist() == [2, 0, 1]


def test_interleaved_development_split_keeps_final_test_block_fixed() -> None:
    query_ids = [f"frame{index:03d}" for index in range(90)]
    split = _build_query_split(
        query_ids,
        strategy="interleaved_development_v1",
        train_count=60,
        validation_count=15,
    )

    assert len(split["train"]) == 60
    assert split["validation"] == [f"frame{index:03d}" for index in range(0, 75, 5)]
    assert split["test"] == query_ids[75:]
    assert not (set(split["train"]) & set(split["validation"]))


def test_explicit_query_split_requires_an_exact_disjoint_query_partition(tmp_path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "format": "stratified_landmark_query_split_v1",
                "strategy": "sequence_balanced_temporal_coverage_v1",
                "train": ["seq1/a", "seq2/a"],
                "validation": ["seq1/b"],
                "test": ["seq2/b"],
            }
        )
    )

    split = _load_query_split_manifest(
        path,
        query_ids=["seq2/b", "seq1/a", "seq1/b", "seq2/a"],
    )

    assert split["strategy"] == "explicit_query_split_manifest_v1"
    assert split["train"] == ["seq1/a", "seq2/a"]
    assert split["validation"] == ["seq1/b"]
    assert split["test"] == ["seq2/b"]
    assert len(split["source_sha256"]) == 16


def test_explicit_query_split_rejects_overlap_and_missing_queries(tmp_path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "format": "stratified_landmark_query_split_v1",
                "train": ["a"],
                "validation": ["b"],
                "test": ["b"],
            }
        )
    )

    with pytest.raises(ValueError, match="overlap"):
        _load_query_split_manifest(path, query_ids=["a", "b", "c"])


def test_frozen_global_baseline_policy_validates_inputs_and_overrides_budget(tmp_path) -> None:
    path = tmp_path / "baseline.json"
    validation_pose = {
        "query_count": 2,
        "success_count": 2,
        "success_rate": 1.0,
        "median_translation_m_success": 0.2,
        "p90_translation_m_success": 0.5,
        "median_rotation_deg_success": 0.4,
        "recall_25cm_2deg": 0.5,
        "recall_10cm_5deg": 0.0,
        "recall_5cm_5deg": 0.0,
    }
    path.write_text(
        json.dumps(
            {
                "stage": "whole_image_global_partial_assignment_audit",
                "protocol": {"policy_selected_on_validation_only": True},
                "inputs": {
                    "proposals_sha256": "proposals",
                    "candidate_artifact_sha256": "features",
                    "projected_landmark_bank_sha256": "bank",
                    "split_json_sha256": "split",
                    "baseline_score_key": "strategy__alike_support_top2_mean",
                },
                "baseline": {
                    "frozen_validation_policy": {
                        "policy_key": "baseline_global_max128_score_topk",
                        "max_matches": 128,
                        "selection_mode": "score_topk",
                        "pose": validation_pose,
                    }
                },
            }
        )
    )
    args = Namespace(
        baseline_strategy="alike_support_top2_mean",
        global_assignment_match_count=48,
        global_assignment_selection_mode="spatial_round_robin",
    )

    source = _load_frozen_global_baseline_policy(
        path,
        args=args,
        data_manifest={
            "proposals_sha256": "proposals",
            "feature_artifact_sha256": "features",
            "projected_landmark_bank_sha256": "bank",
            "query_split_manifest_sha256": "split",
        },
    )

    assert args.global_assignment_match_count == 128
    assert args.global_assignment_selection_mode == "score_topk"
    _validate_frozen_baseline_pose(validation_pose, source)
    with pytest.raises(ValueError, match="baseline replay differs"):
        _validate_frozen_baseline_pose(
            {**validation_pose, "p90_translation_m_success": 0.6},
            source,
        )


def test_frozen_global_baseline_policy_rejects_stale_candidate_artifact(tmp_path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "stage": "whole_image_global_partial_assignment_audit",
                "protocol": {"policy_selected_on_validation_only": True},
                "inputs": {
                    "proposals_sha256": "proposals",
                    "candidate_artifact_sha256": "old-features",
                    "projected_landmark_bank_sha256": "bank",
                    "split_json_sha256": "split",
                    "baseline_score_key": "strategy__alike_support_top2_mean",
                },
                "baseline": {
                    "frozen_validation_policy": {
                        "max_matches": 128,
                        "selection_mode": "score_topk",
                        "pose": {},
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="different training inputs"):
        _load_frozen_global_baseline_policy(
            path,
            args=Namespace(baseline_strategy="alike_support_top2_mean"),
            data_manifest={
                "proposals_sha256": "proposals",
                "feature_artifact_sha256": "new-features",
                "projected_landmark_bank_sha256": "bank",
                "query_split_manifest_sha256": "split",
            },
        )


def test_refit_reuses_fixed_validation_epoch_and_policy(tmp_path) -> None:
    path = tmp_path / "selected.pt"
    manifest = {"proposals_sha256": "abc"}
    split = {"strategy": "interleaved_development_v1", "train": ["a"], "validation": ["b"], "test": ["c"]}
    model_config = {"model_dim": 16}
    selection = {
        "strategy": "set_candidate_probability",
        "mode": "unconditional",
        "margin_threshold": None,
        "validation_gate_passed": True,
    }
    torch.save(
        {
            "format": "candidate_maplet_matcher_checkpoint_v5",
            "data_manifest": manifest,
            "split": split,
            "model_config": model_config,
            "epoch": 3,
            "selection": selection,
            "seed": 7,
        },
        path,
    )

    loaded, metadata = _load_refit_selection(
        path,
        data_manifest=manifest,
        split=split,
        model_config=model_config,
        epochs=4,
    )
    assert loaded == selection
    assert metadata["source_epoch"] == 3
    assert metadata["source_seed"] == 7
    with pytest.raises(ValueError, match="selected epoch"):
        _load_refit_selection(
            path,
            data_manifest=manifest,
            split=split,
            model_config=model_config,
            epochs=5,
        )


def test_candidate_static_feature_schema_rejects_same_dim_reordering() -> None:
    names = tuple(CANDIDATE_MAPLET_STATIC_FEATURE_NAMES)
    assert validate_candidate_maplet_static_feature_names(names, count=len(names)) == names
    reordered = (names[1], names[0], *names[2:])
    with pytest.raises(ValueError, match="field order"):
        validate_candidate_maplet_static_feature_names(reordered, count=len(reordered))
    extended = tuple(CANDIDATE_MAPLET_DEPLOYABLE_FEATURE_NAMES)
    assert validate_candidate_maplet_static_feature_names(
        extended, count=len(extended)
    ) == extended
    with pytest.raises(ValueError, match="field order"):
        validate_candidate_maplet_static_feature_names(
            (*names, "query_gt_pose_residual"), count=len(names) + 1
        )


def test_static_feature_normalization_uses_only_training_rows() -> None:
    features = np.asarray(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[1001.0, 1010.0], [1003.0, 1014.0]],
        ],
        dtype=np.float32,
    )
    mean, scale = _fit_static_feature_normalization(
        features,
        row_mask=np.asarray([True, False]),
        valid_edges=np.ones((2, 2), dtype=bool),
    )
    np.testing.assert_allclose(mean, [2.0, 12.0])
    np.testing.assert_allclose(scale, [1.0, 2.0])


def test_assignment_gate_rejects_pose_luck_when_identity_regresses() -> None:
    baseline = {
        "geometry": {
            "thresholds_px": {
                "1": {"recall_at_1_given_mappable": 0.10},
                "2": {"recall_at_1_given_mappable": 0.20},
            }
        },
        "pair_positive_average_precision": 0.30,
        "wrong_pool_rejection_average_precision": 0.40,
    }
    improved = {
        "geometry": {
            "thresholds_px": {
                "1": {"recall_at_1_given_mappable": 0.11},
                "2": {"recall_at_1_given_mappable": 0.21},
            }
        },
        "pair_positive_average_precision": 0.31,
        "wrong_pool_rejection_average_precision": 0.41,
    }
    regressed = {
        **improved,
        "pair_positive_average_precision": 0.29,
    }

    assert _assignment_identity_gate(improved, baseline)
    assert not _assignment_identity_gate(regressed, baseline)


def test_prior_row_confidence_preserves_rejection_order_and_learned_rank() -> None:
    learned = np.asarray([[0.2, 0.8, 0.4], [0.9, 0.1, -np.inf]], dtype=np.float32)
    prior = np.asarray([[0.7, 0.6, 0.5], [0.3, 0.2, 0.1]], dtype=np.float32)
    fused = _preserve_prior_row_confidence(learned, prior)

    np.testing.assert_allclose(np.max(fused, axis=1), np.max(prior, axis=1))
    assert int(np.argmax(fused[0])) == 1
    assert int(np.argmax(fused[1])) == 0
    assert np.isneginf(fused[1, 2])


def test_external_query_manifest_allows_query_hashes_but_not_semantic_drift() -> None:
    checkpoint = {
        key: ["f0", "f1"] if key == "static_feature_names" else index
        for index, key in enumerate(CANDIDATE_MAPLET_INFERENCE_COMPATIBILITY_KEYS)
    }
    checkpoint.update(
        {
            "proposals_sha256": "training-proposals",
            "feature_artifact_sha256": "training-features",
            "detector_query_cache_sha256": "training-query",
        }
    )
    inference = {
        **checkpoint,
        "proposals_sha256": "new-proposals",
        "feature_artifact_sha256": "new-features",
        "detector_query_cache_sha256": "new-query",
    }

    assert not candidate_maplet_inference_manifest_mismatches(
        checkpoint, inference
    )
    inference["projected_landmark_bank_sha256"] = "different-bank"
    assert candidate_maplet_inference_manifest_mismatches(
        checkpoint, inference
    ) == {
        "projected_landmark_bank_sha256": {
            "checkpoint": checkpoint["projected_landmark_bank_sha256"],
            "inference": "different-bank",
        }
    }
