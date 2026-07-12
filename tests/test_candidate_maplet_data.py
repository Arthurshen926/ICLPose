import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _assignment_identity_gate,
    _build_query_split,
    _fit_static_feature_normalization,
    _load_query_split_manifest,
    _load_frozen_global_baseline_policy,
    _load_refit_selection,
    _preserve_prior_row_confidence,
    _validate_frozen_baseline_pose,
)
from feature_extract.vfm.localization.candidate_maplet_data import (
    CandidateMapletEpisodeArrays,
    build_candidate_maplet_assignment_targets,
)
from feature_extract.vfm.localization.candidate_maplet_schema import (
    CANDIDATE_MAPLET_INFERENCE_COMPATIBILITY_KEYS,
    CANDIDATE_MAPLET_STATIC_FEATURE_NAMES,
    candidate_maplet_inference_manifest_mismatches,
    validate_candidate_maplet_static_feature_names,
)


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
    extended = (*names, "maplet_probe_summary")
    assert validate_candidate_maplet_static_feature_names(
        extended, count=len(extended)
    ) == extended


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
