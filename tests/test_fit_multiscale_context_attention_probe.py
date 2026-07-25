from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.fit_multiscale_context_attention_probe import (
    EXACT_IDENTITY_TRAIN_TARGET_CACHE_FORMAT,
    GEOMETRIC_TRAIN_TARGET_CACHE_FORMAT,
    _MIXED_POINTS_CANDIDATE_INPUT,
    _coarse_prior_hard_negative_ranking_loss,
    _family_profile,
    _load_exact_identity_train_target_cache,
    _load_geometric_train_target_cache,
    _load_candidate_prior_input,
    _train_geometric_targets_with_explicit_null,
    _train_identity_targets_with_explicit_null,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_INTERMEDIATE,
)
from feature_extract.vfm.localization.query_observation_identity import (
    RegisteredQueryObservationTargets,
)


def _write_mixed_points(path) -> None:
    metadata = {
        "format": MIXED_VERIFICATION_POINTS_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    np.savez_compressed(
        path,
        # Deliberately reverse NPZ order.  The fitter must use source IDs,
        # never incidental archive row order.
        source_point_ids=np.asarray([1, 0], dtype=np.int64),
        query_ids=np.asarray(["q1.png", "q0.png"]),
        split_names=np.asarray(["train", "validation"]),
        xy=np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        point_sources=np.asarray([POINT_SOURCE_ALIKE, POINT_SOURCE_RADIO_INTERMEDIATE]),
        source_detector_rows=np.asarray([7, -1], dtype=np.int64),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        candidate_bank_rows=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        candidate_track_ids=np.asarray([[101, 102], [201, 202]], dtype=np.int64),
        candidate_prototype_ids=np.zeros((2, 2), dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.5, 0.3], [0.6, 0.2]], dtype=np.float32),
        null_probabilities=np.asarray([0.2, 0.2], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_mixed_candidate_input_uses_explicit_dense_source_ids(tmp_path) -> None:
    points_path = tmp_path / "points.npz"
    _write_mixed_points(points_path)
    digest = file_sha256_short(points_path)
    source = _load_candidate_prior_input(
        contract={
            "candidate_input_kind": _MIXED_POINTS_CANDIDATE_INPUT,
            "candidate_input_lineage_sha256": digest,
            "proposals_sha256": digest,
        },
        proposals_path=None,
        base_overlay_path=None,
        verification_points_path=points_path,
    )

    assert source.kind == _MIXED_POINTS_CANDIDATE_INPUT
    # Source ID 0 was the second physical NPZ row.
    np.testing.assert_array_equal(source.candidate_track_ids[0], [201, 202])
    np.testing.assert_allclose(source.candidate_probabilities[0], [0.6, 0.2])
    np.testing.assert_allclose(source.null_probabilities, [0.2, 0.2])


def test_mixed_candidate_input_rejects_stale_contract_lineage(tmp_path) -> None:
    points_path = tmp_path / "points.npz"
    _write_mixed_points(points_path)

    with pytest.raises(ValueError, match="differ from frozen candidate lineage"):
        _load_candidate_prior_input(
            contract={
                "candidate_input_kind": _MIXED_POINTS_CANDIDATE_INPUT,
                "candidate_input_lineage_sha256": "stale",
                "proposals_sha256": "stale",
            },
            proposals_path=None,
            base_overlay_path=None,
            verification_points_path=points_path,
        )


def test_bidirectional_profile_and_hard_negative_loss_rank_the_true_track() -> None:
    families, position_encoding = _family_profile("bidirectional_absolute_v2")
    assert families == (
        "bidirectional_absolute_visual_v2",
        "bidirectional_absolute_position_control_v2",
    )
    assert position_encoding == "absolute_dual_frame_v1"
    raw_layout_families, raw_layout_position_encoding = _family_profile(
        "bidirectional_absolute_raw_layout_v4"
    )
    assert raw_layout_families == (
        "bidirectional_absolute_raw_layout_visual_v4",
        "bidirectional_absolute_raw_layout_position_control_v4",
    )
    assert raw_layout_position_encoding == "absolute_dual_frame_v1"
    dual_head_families, dual_head_position_encoding = _family_profile(
        "bidirectional_absolute_dual_head_raw_layout_v5"
    )
    assert dual_head_families == (
        "bidirectional_absolute_dual_head_raw_layout_visual_v5",
        "bidirectional_absolute_dual_head_raw_layout_position_control_v5",
    )
    assert dual_head_position_encoding == "absolute_dual_frame_v1"

    membership = torch.as_tensor([[True, False], [False, False]])
    prior = torch.as_tensor([[0.2, 0.7], [0.4, 0.4]])
    correct, active = _coarse_prior_hard_negative_ranking_loss(
        torch.as_tensor([[2.0, -1.0], [1.0, 1.0]]),
        positive_membership=membership,
        candidate_prior=prior,
        margin=0.1,
        prior_power=1.0,
        evidence_weight=1.0,
    )
    incorrect, active_incorrect = _coarse_prior_hard_negative_ranking_loss(
        torch.as_tensor([[-1.0, 2.0], [1.0, 1.0]]),
        positive_membership=membership,
        candidate_prior=prior,
        margin=0.1,
        prior_power=1.0,
        evidence_weight=1.0,
    )

    assert bool(active[0]) and not bool(active[1])
    assert torch.equal(active, active_incorrect)
    assert float(correct) < float(incorrect)


def test_explicit_null_targets_exclude_the_null_column_from_hard_negative_membership(
    tmp_path, monkeypatch
) -> None:
    import feature_extract.tools.vfm.fit_multiscale_context_attention_probe as module

    (tmp_path / "images.bin").touch()
    targets = RegisteredQueryObservationTargets(
        track_ids=np.asarray([101, 999], dtype=np.int64),
        distances_px=np.asarray([0.1, 0.2], dtype=np.float32),
        supervised=np.asarray([True, True]),
    )
    monkeypatch.setattr(
        module,
        "read_colmap_images_binary",
        lambda _path: {1: SimpleNamespace(image_name="q0")},
    )
    monkeypatch.setattr(
        module, "registered_query_observation_targets", lambda **_kwargs: targets
    )

    rows, classes, membership, audit = _train_identity_targets_with_explicit_null(
        query_ids=np.asarray(["q0", "q1"]),
        query_xy=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        candidate_tracks=np.asarray([[101, 5], [8, 7]], dtype=np.int64),
        split_names=np.asarray(["train", "train"]),
        colmap_model_dir=tmp_path,
        radius_px=2.0,
    )

    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_array_equal(classes, [0, 2])
    assert membership.shape == (2, 2)
    assert membership.tolist() == [[True, False], [False, False]]
    assert audit["explicit_null_registered_train_row_count"] == 1


def test_geometric_targets_keep_multiple_valid_candidates_and_train_only_rows(tmp_path) -> None:
    proposals_path = tmp_path / "proposals.npz"
    np.savez_compressed(
        proposals_path,
        candidate_gt_residuals_px=np.asarray(
            [
                [8.0, 8.0, np.inf],
                [1.0, 1.5, np.inf],
                [9.0, 9.0, np.inf],
                [0.5, 4.0, np.inf],
            ],
            dtype=np.float32,
        ),
    )

    rows, membership, audit = _train_geometric_targets_with_explicit_null(
        proposals_path=proposals_path,
        # The frozen layout order deliberately differs from proposal order.
        source_rows=np.asarray([3, 1, 0, 2], dtype=np.int64),
        candidate_tracks=np.asarray(
            [[10, 11, -1], [20, 21, -1], [30, 31, -1], [40, 41, -1]]
        ),
        split_names=np.asarray(["train", "train", "validation", "train"]),
        positive_threshold_px=2.0,
    )

    np.testing.assert_array_equal(rows, [0, 1, 3])
    # Source proposal row 3 has one candidate at 0.5 px; source row 1 has two
    # valid candidates; source row 2 has no valid candidate and supervises null.
    assert membership.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, False, True],
    ]
    assert audit["geometry_positive_train_row_count"] == 2
    assert audit["multi_positive_train_row_count"] == 1
    assert audit["explicit_null_train_row_count"] == 1


def test_geometric_train_target_cache_cannot_include_or_misalign_nontrain_rows(tmp_path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}\n")
    proposals_path = tmp_path / "proposals.npz"
    np.savez_compressed(proposals_path, placeholder=np.asarray([1], dtype=np.int64))
    source_rows = np.asarray([3, 1, 0], dtype=np.int64)
    tracks = np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.int64)
    split_names = np.asarray(["train", "validation", "train"])
    train_rows = np.asarray([0, 2], dtype=np.int64)
    train_source_rows = source_rows[train_rows]
    train_tracks = tracks[train_rows]
    cache_path = tmp_path / "train_targets.npz"
    metadata = {
        "format": GEOMETRIC_TRAIN_TARGET_CACHE_FORMAT,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "training_split": "train",
        "contract_sha256": file_sha256_short(contract_path),
        "frozen_layout_features_sha256": "layout-sha",
        "proposals_sha256": file_sha256_short(proposals_path),
        "geometric_positive_threshold_px": 2.0,
        "layout_row_indices_sha256": _array_sha256_short(train_rows),
        "source_row_indices_sha256": _array_sha256_short(train_source_rows),
        "candidate_track_ids_sha256": _array_sha256_short(train_tracks),
    }
    np.savez_compressed(
        cache_path,
        layout_row_indices=train_rows,
        source_row_indices=train_source_rows,
        candidate_track_ids=train_tracks,
        target_membership=np.asarray([[True, False, False], [False, False, True]]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    rows, membership, audit = _load_geometric_train_target_cache(
        target_cache_path=cache_path,
        contract_path=contract_path,
        contract={"frozen_layout_features_sha256": "layout-sha"},
        proposals_path=proposals_path,
        source_rows=source_rows,
        candidate_tracks=tracks,
        split_names=split_names,
        positive_threshold_px=2.0,
    )

    np.testing.assert_array_equal(rows, train_rows)
    assert membership.tolist() == [[True, False, False], [False, False, True]]
    assert audit["validation_or_test_labels_loaded_by_fit"] is False

    metadata["contains_validation_or_test_targets"] = True
    np.savez_compressed(
        cache_path,
        layout_row_indices=train_rows,
        source_row_indices=train_source_rows,
        candidate_track_ids=train_tracks,
        target_membership=np.asarray([[True, False, False], [False, False, True]]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="lineage or layout differs"):
        _load_geometric_train_target_cache(
            target_cache_path=cache_path,
            contract_path=contract_path,
            contract={"frozen_layout_features_sha256": "layout-sha"},
            proposals_path=proposals_path,
            source_rows=source_rows,
            candidate_tracks=tracks,
            split_names=split_names,
            positive_threshold_px=2.0,
        )


def test_exact_identity_train_target_cache_is_train_only_and_lineage_bound(tmp_path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}\n")
    source_rows = np.asarray([3, 1, 0], dtype=np.int64)
    tracks = np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.int64)
    split_names = np.asarray(["train", "validation", "train"])
    selected_rows = np.asarray([0, 2], dtype=np.int64)
    selected_source_rows = source_rows[selected_rows]
    selected_tracks = tracks[selected_rows]
    cache_path = tmp_path / "identity_train_targets.npz"
    metadata = {
        "format": EXACT_IDENTITY_TRAIN_TARGET_CACHE_FORMAT,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "training_split": "train",
        "contract_sha256": file_sha256_short(contract_path),
        "frozen_layout_features_sha256": "layout-sha",
        "registered_identity_radius_px": 2.0,
        "layout_row_indices_sha256": _array_sha256_short(selected_rows),
        "source_row_indices_sha256": _array_sha256_short(selected_source_rows),
        "candidate_track_ids_sha256": _array_sha256_short(selected_tracks),
        "target_class_semantics": "registered_exact_track_if_in_fixed_topl_else_explicit_null",
        "fit_must_not_open_colmap_identity_model": True,
    }
    np.savez_compressed(
        cache_path,
        layout_row_indices=selected_rows,
        source_row_indices=selected_source_rows,
        candidate_track_ids=selected_tracks,
        # Candidate 0 for the first registered anchor; explicit null for the
        # second when its exact track was outside fixed top-L.
        target_classes=np.asarray([0, 2], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    rows, classes, audit = _load_exact_identity_train_target_cache(
        target_cache_path=cache_path,
        contract_path=contract_path,
        contract={"frozen_layout_features_sha256": "layout-sha"},
        source_rows=source_rows,
        candidate_tracks=tracks,
        split_names=split_names,
        radius_px=2.0,
    )

    np.testing.assert_array_equal(rows, selected_rows)
    np.testing.assert_array_equal(classes, [0, 2])
    assert audit["validation_or_test_labels_loaded_by_fit"] is False

    metadata["contains_validation_or_test_targets"] = True
    np.savez_compressed(
        cache_path,
        layout_row_indices=selected_rows,
        source_row_indices=selected_source_rows,
        candidate_track_ids=selected_tracks,
        target_classes=np.asarray([0, 2], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="lineage or layout differs"):
        _load_exact_identity_train_target_cache(
            target_cache_path=cache_path,
            contract_path=contract_path,
            contract={"frozen_layout_features_sha256": "layout-sha"},
            source_rows=source_rows,
            candidate_tracks=tracks,
            split_names=split_names,
            radius_px=2.0,
        )
