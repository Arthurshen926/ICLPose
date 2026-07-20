from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
    FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES,
    FrozenFulltrackAppearanceFeatures,
    SUMMARY_TOP4_BALANCED_ARCHITECTURE,
    SUMMARY_TOP4_FEATURE_GRANULARITY,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_pose_overlay import (
    IDENTITY_PROBABILITY_SEMANTICS,
)
from feature_extract.vfm.localization.frozen_fulltrack_summary_pose_overlay import (
    FROZEN_FULLTRACK_SUMMARY_TOP4_POSE_OVERLAY_FORMAT,
    build_frozen_fulltrack_summary_top4_validation_pose_overlay,
)


FAMILY = (
    "fixedprior_fulltrack_summarytop4_positive_uplift_multiscale_tanh_cap_"
    "traincal_balanced"
)


def _features() -> FrozenFulltrackAppearanceFeatures:
    spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES[FAMILY]
    rows, candidates = 4, 2
    names = tuple(
        f"{profile}__uniform_top4_mean_ncc" for profile in spec.profile_names
    )
    return FrozenFulltrackAppearanceFeatures(
        paths=(Path("/tmp/fulltrack-summary-overlay-fixture.npz"),),
        query_ids=np.asarray(["train0.png", "train1.png", "val0.png", "val1.png"]),
        split_names=np.asarray(["train", "train", "validation", "validation"]),
        source_row_indices=np.arange(rows, dtype=np.int64),
        xy=np.zeros((rows, 2), dtype=np.float32),
        candidate_track_ids=np.asarray(
            [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.int64
        ),
        candidate_probabilities=np.asarray([[0.6, 0.2]] * rows, dtype=np.float32),
        null_probabilities=np.full((rows,), 0.2, dtype=np.float32),
        candidate_summary_features=np.zeros((rows, candidates, len(names)), dtype=np.float32),
        candidate_summary_feature_valid=np.ones(
            (rows, candidates, len(names)), dtype=bool
        ),
        feature_names=names,
        profile_names=spec.profile_names,
        artifact_metadata=({},),
        compatibility={},
    )


def _family_contract() -> dict[str, object]:
    spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES[FAMILY]
    return {
        "architecture": SUMMARY_TOP4_BALANCED_ARCHITECTURE,
        "candidate_evidence_transform": (
            "relu_positive_relative_summary_top4_uplift_tanh_bounded_traincal_v1"
        ),
        "summary_statistic": "uniform_top4_mean_ncc",
        "profile_names": list(spec.profile_names),
        "profile_feature_names": [
            f"{profile}__uniform_top4_mean_ncc" for profile in spec.profile_names
        ],
        "training_objective": "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1",
        "rank2_hard_pair_weight": spec.rank2_hard_pair_weight,
        "coarse_top1_stability_weight": spec.coarse_top1_stability_weight,
        "residual_scale": spec.residual_scale,
        "residual_cap": spec.residual_cap,
        "missing_evidence_semantics": "common_top1_profile_missing_zero_residual_v1",
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "per_view_model": False,
    }


def test_summary_top4_pose_overlay_replaces_only_validation_rows(tmp_path) -> None:
    features = _features()
    proposals = tmp_path / "proposals.npz"
    np.savez_compressed(
        proposals,
        query_ids=features.query_ids,
        candidate_track_ids=features.candidate_track_ids,
    )
    base = tmp_path / "base_overlay.npz"
    np.savez_compressed(
        base,
        candidate_track_ids=features.candidate_track_ids,
        candidate_probabilities=features.candidate_probabilities,
        null_probabilities=features.null_probabilities,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_maplet_prior_overlay_v1",
                    "contains_ground_truth": False,
                    "contains_target_errors": False,
                    "probability_semantics": IDENTITY_PROBABILITY_SEMANTICS,
                    "proposals_sha256": file_sha256_short(proposals),
                }
            )
        ),
    )
    predicted = np.asarray(
        [[[0.6, 0.2], [0.6, 0.2], [0.1, 0.7], [0.3, 0.5]]],
        dtype=np.float32,
    )
    prediction_metadata = {
        "format": FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "training_supervision_split": "train",
        "validation_or_test_labels_used_by_fit": False,
        "prediction_frozen_before_validation_target_join": True,
        "identity_supervision_colmap_images_bin": "/tmp/fixture-images.bin",
        "identity_supervision_colmap_images_sha256": "fixture-images-hash",
        "fixed_global_top_l": 20,
        "candidate_reselection": False,
        "support_reselection": False,
        "support_view_selection": False,
        "all_observation_aggregation_preserved": True,
        "feature_granularity": SUMMARY_TOP4_FEATURE_GRANULARITY,
        "per_view_model": False,
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "image_retrieval_or_submap_used": False,
        "render": False,
        "pose_scoring": False,
        "fulltrack_compatibility": dict(features.compatibility),
        "family_architectures": {FAMILY: SUMMARY_TOP4_BALANCED_ARCHITECTURE},
        "family_evidence_contracts": {FAMILY: _family_contract()},
    }
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_ids=features.query_ids,
        split_names=features.split_names,
        source_row_indices=features.source_row_indices,
        candidate_track_ids=features.candidate_track_ids,
        family_names=np.asarray([FAMILY]),
        candidate_probabilities=predicted,
        null_probabilities=features.null_probabilities[None, :],
        candidate_residuals=np.zeros_like(predicted),
        baseline_candidate_probabilities=features.candidate_probabilities,
        baseline_null_probabilities=features.null_probabilities,
        metadata_json=np.asarray(json.dumps(prediction_metadata)),
    )
    output = tmp_path / "overlay.npz"
    result = build_frozen_fulltrack_summary_top4_validation_pose_overlay(
        features=features,
        predictions_path=predictions,
        base_prior_overlay_path=base,
        proposals_path=proposals,
        family=FAMILY,
        output_path=output,
    )
    assert result["protocol"]["per_view_s2_claimed"] is False
    with np.load(output, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        candidate = payload["candidate_probabilities"]
        null = payload["null_probabilities"]
    assert metadata["format"] == FROZEN_FULLTRACK_SUMMARY_TOP4_POSE_OVERLAY_FORMAT
    assert metadata["updated_source_row_count"] == 2
    np.testing.assert_array_equal(candidate[:2], features.candidate_probabilities[:2])
    np.testing.assert_array_equal(null[:2], features.null_probabilities[:2])
    np.testing.assert_array_equal(candidate[2:], predicted[0, 2:])
    np.testing.assert_allclose(candidate.sum(axis=1) + null, 1.0, atol=1e-6)
    overlay, loaded_metadata = _load_candidate_prior_overlay(
        output,
        proposals_path=proposals,
        proposals={"candidate_track_ids": features.candidate_track_ids},
    )
    assert loaded_metadata["format"] == FROZEN_FULLTRACK_SUMMARY_TOP4_POSE_OVERLAY_FORMAT
    np.testing.assert_array_equal(overlay["candidate_probabilities"], candidate)
