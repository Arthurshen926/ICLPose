from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    _load_candidate_prior_overlay,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_RAW_NCC_FEATURE_GRANULARITY,
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_PREDICTION_FORMAT,
    FrozenFulltrackPerViewAppearanceFeatures,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_pose_overlay import (
    FROZEN_FULLTRACK_PER_VIEW_POSE_OVERLAY_FORMAT,
    IDENTITY_PROBABILITY_SEMANTICS,
    build_frozen_fulltrack_per_view_validation_pose_overlay,
)


def _features() -> FrozenFulltrackPerViewAppearanceFeatures:
    rows, candidates = 4, 2
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/fulltrack-overlay-fixture.npz"),),
        query_ids=np.asarray(["train0.png", "train1.png", "val0.png", "val1.png"]),
        split_names=np.asarray(["train", "train", "validation", "validation"]),
        source_row_indices=np.arange(rows, dtype=np.int64),
        xy=np.zeros((rows, 2), dtype=np.float32),
        candidate_track_ids=np.asarray(
            [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.int64
        ),
        candidate_probabilities=np.asarray([[0.6, 0.2]] * rows, dtype=np.float32),
        null_probabilities=np.full((rows,), 0.2, dtype=np.float32),
        candidate_support_observation_counts=np.ones((rows, candidates), dtype=np.int64),
        edge_candidate_offsets=np.arange(rows * candidates + 1, dtype=np.int64),
        edge_geometry_rows=np.arange(rows * candidates, dtype=np.int64),
        edge_profile_scores=np.zeros((rows * candidates, 1), dtype=np.float32),
        edge_profile_valid=np.ones((rows * candidates, 1), dtype=bool),
        profile_names=("alike_center",),
        artifact_metadata=({},),
        compatibility={
            "per_view_edge_feature_semantics": (
                FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS
            )
        },
    )


def test_fulltrack_pose_overlay_replaces_only_validation_rows(tmp_path) -> None:
    features = _features()
    proposals = tmp_path / "proposals.npz"
    np.savez_compressed(
        proposals,
        query_ids=features.query_ids,
        candidate_track_ids=features.candidate_track_ids,
    )
    base = tmp_path / "base_overlay.npz"
    base_metadata = {
        "format": "candidate_maplet_prior_overlay_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "probability_semantics": IDENTITY_PROBABILITY_SEMANTICS,
        "proposals_sha256": file_sha256_short(proposals),
    }
    np.savez_compressed(
        base,
        candidate_track_ids=features.candidate_track_ids,
        candidate_probabilities=features.candidate_probabilities,
        null_probabilities=features.null_probabilities,
        metadata_json=np.asarray(json.dumps(base_metadata)),
    )
    family = "fixedprior_fulltrack_rawtop4_positive_uplift_alike"
    predicted = np.asarray(
        [
            [
                [0.6, 0.2],
                [0.6, 0.2],
                [0.1, 0.7],
                [0.3, 0.5],
            ]
        ],
        dtype=np.float32,
    )
    prediction_metadata = {
        "format": FULLTRACK_PER_VIEW_PREDICTION_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "diagnostic_only": True,
        "training_supervision_split": "train",
        "validation_or_test_labels_used_by_fit": False,
        "prediction_frozen_before_validation_target_join": True,
        "fixed_global_top_l": 20,
        "candidate_reselection": False,
        "support_reselection": False,
        "support_view_selection": False,
        "fulltrack_compatibility": dict(features.compatibility),
        "feature_granularity": FULLTRACK_PER_VIEW_RAW_NCC_FEATURE_GRANULARITY,
        "identity_supervision_colmap_images_bin": "/tmp/fixture-images.bin",
        "identity_supervision_colmap_images_sha256": "fixture-images-hash",
        "family_architectures": {
            family: FULLTRACK_PER_VIEW_FAMILIES[family].architecture
        },
        "family_edge_feature_semantics": {
            family: FULLTRACK_PER_VIEW_FAMILIES[family].edge_feature_semantics
        },
        "family_evidence_contracts": {family: {"fixture": True}},
    }
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_ids=features.query_ids,
        split_names=features.split_names,
        source_row_indices=features.source_row_indices,
        candidate_track_ids=features.candidate_track_ids,
        family_names=np.asarray([family]),
        candidate_probabilities=predicted,
        null_probabilities=features.null_probabilities[None, :],
        candidate_residuals=np.zeros_like(predicted),
        baseline_candidate_probabilities=features.candidate_probabilities,
        baseline_null_probabilities=features.null_probabilities,
        metadata_json=np.asarray(json.dumps(prediction_metadata)),
    )
    output = tmp_path / "overlay.npz"
    result = build_frozen_fulltrack_per_view_validation_pose_overlay(
        features=features,
        predictions_path=predictions,
        base_prior_overlay_path=base,
        proposals_path=proposals,
        family=family,
        output_path=output,
    )
    assert result["protocol"]["validation_rows_only"] is True
    with np.load(output, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        candidate = payload["candidate_probabilities"]
        null = payload["null_probabilities"]
    assert metadata["format"] == FROZEN_FULLTRACK_PER_VIEW_POSE_OVERLAY_FORMAT
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
    assert loaded_metadata["format"] == FROZEN_FULLTRACK_PER_VIEW_POSE_OVERLAY_FORMAT
    np.testing.assert_array_equal(overlay["candidate_probabilities"], candidate)
