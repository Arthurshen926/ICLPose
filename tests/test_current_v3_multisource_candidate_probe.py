from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from feature_extract.tools.vfm.audit_current_v3_multisource_candidate_probe import (
    audit_current_v3_multisource_candidate_probe,
)
from feature_extract.tools.vfm.fit_current_v3_multisource_candidate_probe import (
    fit_current_v3_multisource_candidate_probe,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import ColmapImageObservation
from feature_extract.vfm.localization.current_v3_candidate_probe import (
    CURRENT_V3_PREDICTION_FORMAT,
    EXACT_IDENTITY_PROBABILITY_SEMANTICS,
    align_current_v3_features_and_evidence,
    build_current_v3_validation_pose_overlay,
    load_current_v3_evidence_inference,
    load_current_v3_features,
    train_geometric_target_membership,
    train_registered_track_identity_membership_from_images,
    validation_registered_track_identity_labels_from_images,
    validation_geometric_labels,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
)


def _write_current_v3_fixture(tmp_path):
    evidence = tmp_path / "evidence.npz"
    rows = np.asarray([10, 20, 30, 40], dtype=np.int64)
    splits = np.asarray(["train", "train", "validation", "test"])
    query_ids = np.asarray(["train0.png", "train1.png", "val.png", "test.png"])
    tracks = np.asarray([[101, 102], [103, 104], [105, 106], [107, 108]], dtype=np.int64)
    banks = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    priors = np.asarray(
        [[0.6, 0.2], [0.6, 0.2], [0.6, 0.2], [0.6, 0.2]], dtype=np.float32
    )
    residuals = np.asarray(
        [[1.0, 30.0], [30.0, 1.0], [30.0, 1.0], [30.0, 1.0]], dtype=np.float32
    )
    np.savez_compressed(
        evidence,
        selected_rows=rows,
        query_ids=query_ids,
        query_xy=np.asarray([[10.0, 10.0]] * 4, dtype=np.float32),
        split_names=splits,
        candidate_valid=np.ones((4, 2), dtype=bool),
        candidate_source_columns=np.asarray(
            [[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64
        ),
        candidate_track_ids=tracks,
        candidate_bank_rows=banks,
        candidate_prior_probabilities=priors,
        unknown_probability=np.asarray([0.2] * 4, dtype=np.float32),
        candidate_target_gt_residuals_px=residuals,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_evidence_v3",
                    "candidate_probability_semantics": "factorized_top_l_availability_times_conditional_identity",
                    "pose_used_for_selection": False,
                    "image_retrieval": False,
                    "render": False,
                    "maplet_support_index_sha256": "maplet-test",
                    "proposals_sha256": "proposal-test",
                }
            )
        ),
    )
    layout = tmp_path / "layout.npz"
    np.savez_compressed(
        layout,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "candidate_evidence_sha256": file_sha256_short(evidence),
                }
            )
        ),
    )
    feature_rows = np.asarray([0, 1, 2], dtype=np.int64)
    values = np.zeros(
        (3, 2, 1, len(MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    # The correct train/validation candidate has visual value one.  This is a
    # deterministic fixture for checking that train-only fitting can replace a
    # wrong frozen validation top-1 candidate.
    values[0, 0, 0] = 1.0
    values[1, 1, 0] = 1.0
    values[2, 1, 0] = 1.0
    features = tmp_path / "features.npz"
    np.savez_compressed(
        features,
        source_row_indices=rows[feature_rows],
        query_ids=query_ids[feature_rows],
        split_names=splits[feature_rows],
        xy=np.asarray([[10.0, 10.0]] * 3, dtype=np.float32),
        candidate_track_ids=tracks[feature_rows],
        candidate_canonical_rows=banks[feature_rows],
        candidate_features=values,
        candidate_view_valid=np.ones((3, 2, 1), dtype=bool),
        feature_names=np.asarray(MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
                    "contains_ground_truth": False,
                    "contains_target_errors": False,
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "whole_image_summary_or_global_used": False,
                    "render": False,
                    "is_complete_frozen_layout": True,
                    "source_frozen_layout": str(layout),
                    "source_frozen_layout_sha256": file_sha256_short(layout),
                    "maplet_support_index_sha256": "maplet-test",
                    "proposals_sha256": "proposal-test",
                }
            )
        ),
    )
    return features, evidence


def _registered_image(name: str, *, point3d_ids: list[int]) -> ColmapImageObservation:
    count = len(point3d_ids)
    return ColmapImageObservation(
        image_id=count + 1,
        image_name=name,
        camera_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.zeros((3,), dtype=np.float64),
        xys=np.asarray([[10.0, 10.0]] * count, dtype=np.float64),
        point3d_ids=np.asarray(point3d_ids, dtype=np.int64),
    )


def test_current_v3_registered_identity_targets_are_train_only_and_exact(tmp_path) -> None:
    features_path, _evidence_path = _write_current_v3_fixture(tmp_path)
    base = load_current_v3_features(features_path)
    # Add a third train token with no nearby SfM observation.  It must not be
    # converted into a null target simply because SfM is sparse there.
    features = replace(
        base,
        source_rows=np.asarray([10, 20, 25, 30], dtype=np.int64),
        query_ids=np.asarray(["train0.png", "train1.png", "train2.png", "val.png"]),
        split_names=np.asarray(["train", "train", "train", "validation"]),
        xy=np.asarray([[10.0, 10.0]] * 4, dtype=np.float32),
        candidate_tracks=np.asarray(
            [[101, 102], [103, 104], [103, 104], [105, 106]], dtype=np.int64
        ),
        candidate_canonical_rows=np.asarray(
            [[1, 2], [3, 4], [3, 4], [5, 6]], dtype=np.int64
        ),
        candidate_features=np.zeros((4, 2, 1, 1), dtype=np.float32),
        candidate_view_valid=np.ones((4, 2, 1), dtype=bool),
        feature_names=("fixture",),
    )
    images = {
        "train0.png": _registered_image("train0.png", point3d_ids=[101]),
        # The registered identity is absent from top-L, so this is an explicit
        # null example rather than an arbitrary candidate positive.
        "train1.png": _registered_image("train1.png", point3d_ids=[999]),
        "train2.png": ColmapImageObservation(
            image_id=3,
            image_name="train2.png",
            camera_id=1,
            qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            tvec=np.zeros((3,), dtype=np.float64),
            xys=np.asarray([[100.0, 100.0]], dtype=np.float64),
            point3d_ids=np.asarray([103], dtype=np.int64),
        ),
        "val.png": _registered_image("val.png", point3d_ids=[106]),
    }
    train_rows, membership, audit = train_registered_track_identity_membership_from_images(
        features, images_by_name=images, identity_radius_px=2.0
    )
    assert train_rows.tolist() == [0, 1]
    assert membership.tolist() == [[True, False, False], [False, False, True]]
    assert audit["unsupervised_train_row_count"] == 1
    assert audit["validation_or_test_target_used"] is False

    validation_mask, labels, validation_audit = (
        validation_registered_track_identity_labels_from_images(
            features, images_by_name=images, identity_radius_px=2.0
        )
    )
    assert validation_mask.tolist() == [False, False, False, True]
    assert labels[3].tolist() == [False, True]
    assert validation_audit["prediction_frozen_before_validation_target_join"] is True


def test_current_v3_train_and_validation_target_boundaries(tmp_path) -> None:
    features_path, evidence_path = _write_current_v3_fixture(tmp_path)
    features = load_current_v3_features(features_path)
    evidence = load_current_v3_evidence_inference(evidence_path)
    aligned = align_current_v3_features_and_evidence(features, evidence)

    train_rows, membership, audit = train_geometric_target_membership(
        evidence_path, aligned, threshold_px=5.0
    )
    assert train_rows.tolist() == [0, 1]
    assert membership.tolist() == [[True, False, False], [False, True, False]]
    assert audit["validation_or_test_target_used"] is False

    validation_mask, labels = validation_geometric_labels(
        evidence_path, aligned, threshold_px=5.0
    )
    assert validation_mask.tolist() == [False, False, True]
    assert labels[2].tolist() == [False, True]


def test_current_v3_fit_and_validation_audit_are_target_separated(tmp_path) -> None:
    features_path, evidence_path = _write_current_v3_fixture(tmp_path)
    output_dir = tmp_path / "fit"
    family = "multisource_landmark_region_candidate_specific_appearance_only"
    summary = fit_current_v3_multisource_candidate_probe(
        features_path=features_path,
        candidate_evidence_path=evidence_path,
        output_dir=output_dir,
        families=(family,),
        geometric_positive_threshold_px=5.0,
        epochs=80,
        batch_size=2,
        learning_rate=0.1,
        seed=0,
        device="cpu",
        architecture="linear",
        hidden_dim=8,
        prior_residual=True,
    )
    prediction_path = output_dir / "predictions.npz"
    assert summary["protocol"]["validation_or_test_labels_used_by_fit"] is False
    with np.load(prediction_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        assert metadata["contains_ground_truth"] is False
        assert metadata["validation_or_test_labels_used_by_fit"] is False
        assert payload["split_names"].tolist() == ["train", "train", "validation"]

    report = audit_current_v3_multisource_candidate_probe(
        features_path=features_path,
        candidate_evidence_path=evidence_path,
        predictions_path=prediction_path,
        thresholds_px=(2.0, 5.0),
        primary_threshold_px=5.0,
    )
    block = report["families"][family]["thresholds"]["5"]
    assert report["protocol"]["test_used_for_model_selection"] is False
    assert block["baseline"]["top1_geometry_valid_rate"] == 0.0
    assert block["probe"]["top1_geometry_valid_rate"] == 1.0


def test_current_v3_pose_overlay_only_replaces_frozen_validation_rows(tmp_path) -> None:
    features_path, evidence_path = _write_current_v3_fixture(tmp_path)
    family = "multisource_landmark_region_candidate_specific_appearance_only"
    fit_dir = tmp_path / "fit"
    fit_current_v3_multisource_candidate_probe(
        features_path=features_path,
        candidate_evidence_path=evidence_path,
        output_dir=fit_dir,
        families=(family,),
        geometric_positive_threshold_px=5.0,
        epochs=80,
        batch_size=2,
        learning_rate=0.1,
        seed=0,
        device="cpu",
        architecture="linear",
        hidden_dim=8,
        prior_residual=True,
    )
    base = tmp_path / "base_overlay.npz"
    base_tracks = np.full((50, 2), -1, dtype=np.int64)
    base_probability = np.zeros((50, 2), dtype=np.float32)
    base_null = np.ones((50,), dtype=np.float32)
    source_rows = np.asarray([10, 20, 30, 40], dtype=np.int64)
    tracks = np.asarray(
        [[101, 102], [103, 104], [105, 106], [107, 108]], dtype=np.int64
    )
    base_tracks[source_rows] = tracks
    base_probability[source_rows] = np.asarray([[0.6, 0.2]] * 4, dtype=np.float32)
    base_null[source_rows] = 0.2
    np.savez_compressed(
        base,
        candidate_track_ids=base_tracks,
        candidate_probabilities=base_probability,
        null_probabilities=base_null,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_maplet_prior_overlay_v1",
                    "contains_ground_truth": False,
                    "contains_target_errors": False,
                    "proposals_sha256": "proposal-test",
                    "probability_semantics": (
                        "candidate_identity_probability_plus_explicit_null_equals_one"
                    ),
                }
            )
        ),
    )
    output = tmp_path / "validation_overlay.npz"
    result = build_current_v3_validation_pose_overlay(
        features_path=features_path,
        candidate_evidence_path=evidence_path,
        predictions_path=fit_dir / "predictions.npz",
        base_prior_overlay_path=base,
        family=family,
        output_path=output,
    )
    with np.load(fit_dir / "predictions.npz", allow_pickle=False) as payload:
        expected_candidate = np.asarray(payload["candidate_probabilities"])[0, 2]
        expected_null = float(np.asarray(payload["null_probabilities"])[0, 2])
    with np.load(output, allow_pickle=False) as payload:
        candidate = np.asarray(payload["candidate_probabilities"])
        unknown = np.asarray(payload["null_probabilities"])
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert np.allclose(candidate[30], expected_candidate)
    assert unknown[30] == expected_null
    assert np.array_equal(candidate[10], base_probability[10])
    assert unknown[10] == base_null[10]
    assert np.array_equal(candidate[40], base_probability[40])
    assert unknown[40] == base_null[40]
    assert result["protocol"]["validation_rows_only"] is True
    assert metadata["updated_split_names"] == ["validation"]
